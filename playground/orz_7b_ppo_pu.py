"""
独立评估模块 - 用于评估已训练的模型
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import ray
from loguru import logger

from orz.ppo.tools.math_utils import is_equal, solution2answer
from orz.ppo.openrlhf_deepspeed import DeepspeedStrategy
from playground.zero_setting_base import EvalCustomDataset

# Global executor for async operations
executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class EvaluatorConfig:
    """独立评估配置类"""
    # Model and tokenizer
    model_path: str  # checkpoint path or HF model name
    tokenizer_path: Optional[str] = None  # if None, use model_path

    # vLLM engine settings
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True
    gpu_memory_utilization: float = 0.3
    max_model_len: int = 8192

    # Generation settings
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    generate_max_len: int = 8000
    stop: List[str] = field(default_factory=lambda: ["User:", "Human:", "Assistant:", "</answer>"])

    # Data settings
    eval_prompt_data: List[str] = field(default_factory=lambda: [
        "data/eval_data/math500.json",
        "data/eval_data/aime2024.json",
        "data/eval_data/gpqa_diamond.json",
    ])
    prompt_max_len: int = 2048

    # Output settings
    output_dir: str = "eval_results"
    save_detailed_results: bool = True

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path


class Evaluator:
    """独立评估类，支持从checkpoint加载模型进行评估"""

    def __init__(self, config: EvaluatorConfig):
        """
        初始化评估器

        Args:
            config: EvaluatorConfig 配置对象
        """
        self.cfg = config
        self.tokenizer = None
        self.vllm_engines = []
        self.eval_dataset = None
        self.executor = executor

        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()

        logger.info(f"Initializing Evaluator with config: {config}")

        # Load components
        self._load_tokenizer()
        self._create_vllm_engines()
        self._load_eval_datasets()

        logger.info("Evaluator initialization completed")

    def _load_tokenizer(self):
        """Load tokenizer from pretrained model"""
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {self.cfg.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_path,
            trust_remote_code=True,
        )

    def _create_vllm_engines(self):
        """Create vLLM inference engines"""
        from orz.ppo.utils import create_vllm_engines

        logger.info(f"Creating {self.cfg.vllm_num_engines} vLLM engines from {self.cfg.model_path}")
        self.vllm_engines = create_vllm_engines(
            num_engines=self.cfg.vllm_num_engines,
            tensor_parallel_size=self.cfg.vllm_tensor_parallel_size,
            pretrain=self.cfg.model_path,
            seed=42,
            enable_prefix_caching=self.cfg.enable_prefix_caching,
            enforce_eager=False,
            max_model_len=self.cfg.max_model_len,
            colocate_with_actor=False,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
        )
        logger.info(f"Successfully created {len(self.vllm_engines)} vLLM engines")

    def _load_eval_datasets(self):
        """Load evaluation datasets"""
        logger.info(f"Loading evaluation datasets from {self.cfg.eval_prompt_data}")
        dialogues = []
        for file_path in self.cfg.eval_prompt_data:
            logger.info(f"Loading dataset from {file_path}")
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
                for item in loaded_data:
                    # Add file name as metadata
                    item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                dialogues.extend(loaded_data)

        logger.info(f"Loaded {len(dialogues)} samples from evaluation datasets")

        # Create strategy object for dataset processing
        strategy = DeepspeedStrategy()

        self.eval_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Processed {len(self.eval_dataset)} evaluation samples")

    async def eval(self) -> dict:
        """
        执行评估

        Returns:
            dict: 包含各数据集准确率的字典
        """
        logger.info("Starting evaluation on datasets")
        from vllm import SamplingParams
        from torch.utils.data import DataLoader

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            stop=self.cfg.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        # Create dataloader
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=len(self.eval_dataset),
            shuffle=False,
            drop_last=False,
        )

        output_for_save = []
        log_dict = defaultdict(float)

        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1]["file_name"])

            # Distribute prompts to vLLM engines
            prompt_per_engine = (len(prompts) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
            outputs = []
            for i, llm in enumerate(self.vllm_engines):
                start_idx = i * prompt_per_engine
                end_idx = min((i + 1) * prompt_per_engine, len(prompts))
                if start_idx < len(prompts):
                    outputs.append(
                        llm.generate.remote(
                            prompts=prompts[start_idx:end_idx],
                            sampling_params=sampling_params,
                        )
                    )

            # Gather outputs from all engines
            outputs = await asyncio.gather(*outputs)
            outputs = sum(outputs, [])

            # Extract final answers
            final_answers = []
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])
                else:
                    final_answers.append("")

            # Check correctness for each sample
            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                label = solution2answer(answer)
                prefix_response = solution2answer(final_answer)
                iscorrect = await is_equal(label, prefix_response, self.executor)

                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output.outputs[0].text,
                        final_answer=final_answer,
                        answer=answer,
                        iscorrect=iscorrect,
                    )
                )

                # Log metrics
                log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                log_dict[f"{file_name}/correct"] += iscorrect
                log_dict[f"{file_name}/total"] += 1

        # Calculate metrics per dataset
        all_file_names = [
            os.path.splitext(os.path.basename(file_path))[0]
            for file_path in self.cfg.eval_prompt_data
        ]

        for file_name in all_file_names:
            if log_dict[f"{file_name}/total"] > 0:
                log_dict[f"{file_name}/response_len_in_char"] = (
                    log_dict[f"{file_name}/total_response_len_in_char"]
                    / log_dict[f"{file_name}/total"]
                )
                log_dict[f"{file_name}/accuracy"] = (
                    log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
                )
                log_dict.pop(f"{file_name}/total_response_len_in_char")
                log_dict.pop(f"{file_name}/correct")
                log_dict.pop(f"{file_name}/total")

        # Calculate average accuracy
        accuracies = [log_dict[f"{fn}/accuracy"] for fn in all_file_names if f"{fn}/accuracy" in log_dict]
        if accuracies:
            log_dict["eval_accuracy"] = sum(accuracies) / len(accuracies)

        # Save results if requested
        if self.cfg.save_detailed_results:
            os.makedirs(self.cfg.output_dir, exist_ok=True)

            # Generate result filename
            dump_file_name = "eval_results"
            for file_name in all_file_names:
                if f"{file_name}/accuracy" in log_dict:
                    dump_file_name += f"_{file_name}_{log_dict[f'{file_name}/accuracy']:.4f}"
            dump_file_name += ".jsonl"

            result_path = os.path.join(self.cfg.output_dir, dump_file_name)
            logger.info(f"Saving evaluation results to {result_path}")
            with open(result_path, "w") as f:
                for item in output_for_save:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Log results
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(f"Evaluation completed: {logging_str}")

        return dict(log_dict)

    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up evaluator resources")
        self.vllm_engines.clear()
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    # Evaluation mode
    logger.info("Running in evaluation mode")

    path = "/mnt/shared-storage-user/tangjia/orz/Open-Reasoner-Zero/jericho/orz_ckpt_1gpu/orz_0p5b_ppo_jericho_1012_1gpu/iter12/policy"

    # Create evaluator config
    eval_config = EvaluatorConfig(
        model_path=path,  # TODO: replace with your checkpoint path
        tokenizer_path=path,
        vllm_num_engines=1,
        vllm_tensor_parallel_size=1,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.3,
        max_model_len=8192,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        generate_max_len=8000,
        stop=["User:", "Human:", "Assistant:", "</answer>"],
        eval_prompt_data=[
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json",
        ],
        prompt_max_len=2048,
        output_dir="eval_results",
        save_detailed_results=True,
    )

    # Create evaluator and run evaluation
    evaluator = Evaluator(eval_config)
    try:
        results = asyncio.run(evaluator.eval())
        logger.info(f"Evaluation results: {results}")
    finally:
        evaluator.cleanup()
