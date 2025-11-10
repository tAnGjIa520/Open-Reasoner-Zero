"""
独立的评估模块 - 支持从 checkpoint 自动加载模型并执行评估
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ray
import torch
from loguru import logger
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader
from vllm import SamplingParams

from orz.ppo.tools.math_utils import is_equal, solution2answer
from orz.ppo.utils import create_vllm_engines
from playground.zero_setting_base import EvalCustomDataset

# 全局 executor，用于异步执行
_executor: Optional[ThreadPoolExecutor] = None


def get_executor(max_workers: int = 64) -> ThreadPoolExecutor:
    """获取或创建全局 executor"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


@dataclass
class EvalConfig:
    """评估配置类 - 封装所有评估所需的参数"""

    # 模型加载（核心！）
    model_path: str  # checkpoint 路径或 HF 模型名，如 "Qwen/Qwen2.5-7B" 或 "/path/to/checkpoint/iter100/policy"

    # VLLM 引擎配置
    vllm_num_engines: int = 4
    vllm_tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    max_len: int = 8192

    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.9
    generate_max_len: int = 8000
    stop: Optional[List[str]] = None

    # 评估数据
    eval_prompt_data: List[str] = field(default_factory=list)  # 评估数据文件路径列表
    eval_dataset: Optional[Any] = None  # 或直接传入数据集对象

    # 输出配置
    save_path: str = "./eval_output"
    global_step: int = 0

    # TensorBoard（可选）
    writer: Optional[Any] = None

    # 内部使用：现有的 VLLM 引擎（如果有）
    vllm_engines: Optional[List[Any]] = None


async def eval_model(
    config: EvalConfig,
    skip_engine_init: bool = False,
    tokenizer: Optional[Any] = None,
    strategy: Optional[Any] = None,
) -> Dict[str, float]:
    """
    执行模型评估 - 支持自动加载模型和创建 VLLM 引擎

    Args:
        config: EvalConfig 配置对象
        skip_engine_init: 是否跳过引擎初始化（已有 vllm_engines 时为 True）
        tokenizer: tokenizer 对象（如果需要重新加载数据集）
        strategy: 分布式策略对象（如果需要重新加载数据集）

    Returns:
        log_dict: 包含评估指标的字典
    """

    executor = get_executor()
    vllm_engines = config.vllm_engines
    created_engines = False

    try:
        # Step 1: 初始化 VLLM 引擎（如果需要）
        if not skip_engine_init and vllm_engines is None:
            logger.info(f"Creating VLLM engines from model: {config.model_path}")
            vllm_engines = create_vllm_engines(
                num_engines=config.vllm_num_engines,
                tensor_parallel_size=config.vllm_tensor_parallel_size,
                pretrain=config.model_path,  # 关键：自动加载模型
                seed=42,
                enable_prefix_caching=config.enable_prefix_caching,
                enforce_eager=config.enforce_eager,
                max_model_len=config.max_len,
                colocate_with_actor=False,  # 评估模式不需要共置
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_num_seqs=256,
            )
            created_engines = True
            logger.info(f"Successfully created {len(vllm_engines)} VLLM engines")

        if vllm_engines is None:
            raise RuntimeError("VLLM engines not initialized and skip_engine_init is True")

        # Step 2: 准备评估数据集
        if config.eval_dataset is None:
            if not config.eval_prompt_data:
                raise ValueError("Either eval_dataset or eval_prompt_data must be provided")
            if tokenizer is None or strategy is None:
                raise ValueError("tokenizer and strategy required to load dataset from eval_prompt_data")
            logger.info(f"Loading eval dataset from {len(config.eval_prompt_data)} files")
            config.eval_dataset = load_eval_dataset(
                config.eval_prompt_data, tokenizer, strategy
            )
            logger.info(f"Loaded {len(config.eval_dataset)} samples")

        dataset = config.eval_dataset

        # Step 3: 创建采样参数
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.generate_max_len,
            stop=config.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        # Step 4: 执行评估
        logger.info("Starting evaluation...")
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
        prompt_pre_llm = (len(dataset) + config.vllm_num_engines - 1) // config.vllm_num_engines

        output_for_save = []
        log_dict = defaultdict(float)

        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1]["file_name"])

            # 使用多个 VLLM 引擎并行生成
            outputs = []
            for i, llm in enumerate(vllm_engines):
                start_idx = i * prompt_pre_llm
                end_idx = (i + 1) * prompt_pre_llm
                outputs.append(
                    llm.generate.remote(prompts=prompts[start_idx:end_idx], sampling_params=sampling_params)
                )

            # 等待所有引擎完成
            outputs = await asyncio.gather(*outputs)
            outputs = sum(outputs, [])

            # 提取答案
            final_answers = []
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])
                else:
                    final_answers.append("")

            # 判断正确性并统计
            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                label = solution2answer(answer)
                prefix_response = solution2answer(final_answer)
                iscorrect = await is_equal(label, prefix_response, executor)

                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output.outputs[0].text,
                        final_answer=final_answer,
                        answer=answer,
                        iscorrect=iscorrect,
                    )
                )

                log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                log_dict[f"{file_name}/correct"] += iscorrect
                log_dict[f"{file_name}/total"] += 1

        # Step 5: 计算指标
        all_file_names: List[str] = [
            os.path.splitext(os.path.basename(file_path))[0]
            for file_path in config.eval_prompt_data
        ]

        for file_name in all_file_names:
            if log_dict[f"{file_name}/total"] > 0:
                log_dict[f"{file_name}/response_len_in_char"] = (
                    log_dict[f"{file_name}/total_response_len_in_char"] / log_dict[f"{file_name}/total"]
                )
                log_dict[f"{file_name}/accuracy"] = log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
            else:
                log_dict[f"{file_name}/response_len_in_char"] = 0.0
                log_dict[f"{file_name}/accuracy"] = 0.0

            log_dict.pop(f"{file_name}/total_response_len_in_char", None)
            log_dict.pop(f"{file_name}/correct", None)
            log_dict.pop(f"{file_name}/total", None)

        # 计算平均准确率
        if all_file_names:
            log_dict["eval_accuracy"] = sum(
                [log_dict[f"{file_name}/accuracy"] for file_name in all_file_names]
            ) / len(all_file_names)
        else:
            log_dict["eval_accuracy"] = 0.0

        # Step 6: 保存结果
        os.makedirs(config.save_path, exist_ok=True)
        dump_file_name = f"eval_output_iter{config.global_step}"
        for file_name in all_file_names:
            dump_file_name += f"_{file_name}{log_dict[f'{file_name}/accuracy']:.4f}"
        dump_file_name += ".jsonl"

        save_file_path = os.path.join(config.save_path, dump_file_name)
        with open(save_file_path, "w") as f:
            for item in output_for_save:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Evaluation results saved to {save_file_path}")

        # Step 7: 记录日志
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        if config.writer is not None:
            for k, v in log_dict.items():
                config.writer.add_scalar(f"evals/{k}", v, config.global_step)

        return dict(log_dict)

    finally:
        # 清理资源（仅清理我们创建的引擎）
        if created_engines and vllm_engines is not None:
            logger.info("Cleaning up VLLM engines...")
            cleanup_vllm_engines(vllm_engines)


def load_eval_dataset(
    eval_prompt_data: List[str],
    tokenizer: Any,
    strategy: Any,
    prompt_max_len: int = 2048,
) -> EvalCustomDataset:
    """
    从评估数据文件加载数据集

    Args:
        eval_prompt_data: 评估数据文件路径列表
        tokenizer: tokenizer 对象
        strategy: 分布式策略对象
        prompt_max_len: 最大 prompt 长度

    Returns:
        EvalCustomDataset 对象
    """
    dialogues = []
    for file_path in eval_prompt_data:
        with open(file_path, "r") as f:
            loaded_data = json.load(f)
            for loaded_data_item in loaded_data:
                # 保留不带后缀的文件名
                loaded_data_item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
            dialogues.extend(loaded_data)

    logger.info(f"Start processing {len(dialogues)} evaluation dialogues")
    dataset = EvalCustomDataset(
        dialogues,
        tokenizer,
        prompt_max_len,
        strategy,
        pretrain_mode=False,
        num_processors=1,
    )
    logger.info(f"Finished processing {len(dataset)} evaluation dialogues")
    return dataset


def cleanup_vllm_engines(vllm_engines: List[Any]) -> None:
    """
    清理 VLLM 引擎并释放资源

    Args:
        vllm_engines: VLLM 引擎列表
    """
    if not vllm_engines:
        return

    try:
        # 尝试优雅关闭
        kill_refs = []
        for engine in vllm_engines:
            try:
                kill_refs.append(ray.kill(engine))
            except Exception as e:
                logger.warning(f"Error killing engine: {e}")

        # 等待所有关闭完成
        if kill_refs:
            ray.get(kill_refs, timeout=30)

        logger.info(f"Successfully cleaned up {len(vllm_engines)} VLLM engines")
    except Exception as e:
        logger.error(f"Error cleaning up VLLM engines: {e}")
