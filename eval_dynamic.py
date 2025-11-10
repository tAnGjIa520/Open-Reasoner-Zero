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
from datetime import datetime

import numpy as np
import torch
import ray
from loguru import logger

# 配置日志文件输出到 orz_dynamic_log 目录
log_dir = "orz_dynamic_log"
os.makedirs(log_dir, exist_ok=True)
log_date = datetime.now().strftime("%Y%m%d")
logger.add(
    os.path.join(log_dir, f"eval_{log_date}.log"),
    rotation="00:00",  # 每天午夜轮转
    retention="30 days",  # 保留30天
    level="INFO",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# 可视化库
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from transformers import AutoModelForCausalLM, AutoTokenizer
from orz.ppo.tools.math_utils import is_equal, solution2answer
from orz.ppo.deepspeed_strategy import DeepspeedStrategy
from dataset.eval_dataset import EvalCustomDataset
from extracted_get_batch_logps import analyze_learning_dynamics

# Global executor for async operations
executor = ThreadPoolExecutor(max_workers=64)


# ====================================================================
# 【多 Checkpoint 汇总和可视化函数】
# ====================================================================

def extract_iter_from_path(checkpoint_path: str) -> int:
    """
    从 checkpoint 路径中提取 iter 编号

    示例：
    "/path/to/orz_7b_ppo_jericho_1013/iter45/policy" → 45
    "/path/to/iter90/policy" → 90

    Args:
        checkpoint_path: checkpoint 的完整路径

    Returns:
        iter 编号（int），如果无法提取则返回 0
    """
    match = re.search(r'iter(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


def aggregate_multi_checkpoint_results(all_model_results: dict) -> dict:
    """
    聚合多个 checkpoint 的评估结果，按数据集、样本类型、指标组织

    输入格式：
    {
        checkpoint_path_1: {
            'dataset1/ld_correct/prob_energy': 0.45,
            'dataset1/ld_correct/A_norm': 8.34,
            'dataset1/ld_incorrect/prob_energy': 0.65,
            ...
        },
        checkpoint_path_2: {...},
        ...
    }

    输出格式：
    {
        'dataset1': {
            'correct': {
                'prob_energy': {'iters': [45, 90, ...], 'values': [0.45, 0.38, ...]},
                'A_norm': {'iters': [45, 90, ...], 'values': [8.34, 7.82, ...]},
                ...
            },
            'incorrect': {...},
            'random': {...}
        },
        'dataset2': {...},
        ...
    }

    Args:
        all_model_results: 所有 checkpoint 的评估结果字典

    Returns:
        聚合后的数据结构
    """
    aggregated = {}

    # 提取所有的 (dataset_name, sample_type, metric_name)
    for checkpoint_path, results in all_model_results.items():
        iter_num = extract_iter_from_path(checkpoint_path)

        for key, value in results.items():
            # 解析 key: "dataset_name/ld_sample_type/metric_name" 或 "dataset_name/ld_sample_type/count"
            parts = key.split('/')
            if len(parts) != 3:
                continue

            dataset_name, ld_prefix, metric_or_count = parts

            # 跳过 count 和其他非指标项
            if metric_or_count == 'count':
                continue

            # 解析样本类型：ld_correct → correct
            if not ld_prefix.startswith('ld_'):
                continue
            sample_type = ld_prefix[3:]  # 去掉 'ld_' 前缀
            metric_name = metric_or_count

            # 初始化数据结构
            if dataset_name not in aggregated:
                aggregated[dataset_name] = {}
            if sample_type not in aggregated[dataset_name]:
                aggregated[dataset_name][sample_type] = {}
            if metric_name not in aggregated[dataset_name][sample_type]:
                aggregated[dataset_name][sample_type][metric_name] = {
                    'iters': [],
                    'values': []
                }

            # 添加数据点
            aggregated[dataset_name][sample_type][metric_name]['iters'].append(iter_num)
            aggregated[dataset_name][sample_type][metric_name]['values'].append(float(value))

    # 按 iter 排序
    for dataset_name in aggregated:
        for sample_type in aggregated[dataset_name]:
            for metric_name in aggregated[dataset_name][sample_type]:
                data = aggregated[dataset_name][sample_type][metric_name]
                # 按 iter 排序
                sorted_pairs = sorted(zip(data['iters'], data['values']), key=lambda x: x[0])
                if sorted_pairs:
                    data['iters'], data['values'] = zip(*sorted_pairs)
                    data['iters'] = list(data['iters'])
                    data['values'] = list(data['values'])

    return aggregated


def visualize_multi_checkpoint_trends(aggregated_data: dict, output_dir: str = "eval_results"):
    """
    为每个数据集生成折线图对比，显示指标在不同 checkpoint 上的变化趋势

    对每个数据集：
    - 生成一个大图（2x3 网格或其他布局）
    - 包含所有 5 个核心指标
    - 每个指标子图显示 3 条折线（correct, incorrect, random）
    - x 轴为 iter 编号，y 轴为指标值

    Args:
        aggregated_data: aggregate_multi_checkpoint_results 的输出
        output_dir: 输出目录
    """
    import math

    os.makedirs(output_dir, exist_ok=True)

    # 定义指标和显示名称
    metrics = ['out_token', 'out_argmax', 'A_norm', 'prob_gap2_mean', 'prob_energy']
    metric_labels = {
        'out_token': 'True Label Log Prob',
        'out_argmax': 'Argmax Token Log Prob',
        'A_norm': 'Output Vector Norm',
        'prob_gap2_mean': 'Prediction-Label Gap',
        'prob_energy': 'Pull-up Energy',
    }

    # 样本类型配置
    sample_types = ['correct', 'incorrect', 'random']
    colors = {
        'correct': '#2ecc71',    # 绿色
        'incorrect': '#e74c3c',  # 红色
        'random': '#3498db',     # 蓝色
    }

    # 对每个数据集生成一张大图
    for dataset_name, dataset_data in aggregated_data.items():
        logger.info(f"Generating trend visualization for {dataset_name}...")

        # 检查是否有有效数据
        has_data = False
        for sample_type in sample_types:
            if sample_type in dataset_data and dataset_data[sample_type]:
                has_data = True
                break

        if not has_data:
            logger.warning(f"Dataset {dataset_name} has no valid data, skipping visualization")
            continue

        # 计算网格布局：5 个指标，选择 2x3 或 3x2
        num_metrics = len(metrics)
        ncols = 3
        nrows = math.ceil(num_metrics / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows))

        # 确保 axes 是 2D 数组
        if nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        axes = axes.flatten()

        # 绘制每个指标
        for idx, metric_name in enumerate(metrics):
            ax = axes[idx]

            # 获取该指标的数据
            has_metric_data = False
            for sample_type in sample_types:
                if (sample_type in dataset_data and
                    metric_name in dataset_data[sample_type] and
                    dataset_data[sample_type][metric_name]['values']):

                    data = dataset_data[sample_type][metric_name]
                    iters = data['iters']
                    values = data['values']

                    ax.plot(iters, values,
                           marker='o',
                           label=sample_type.capitalize(),
                           color=colors[sample_type],
                           linewidth=2.5,
                           markersize=8,
                           alpha=0.8)
                    has_metric_data = True

            # 设置子图标题和标签
            ax.set_title(f'{metric_labels[metric_name]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iter', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='best')

            # 如果没有数据，显示空白
            if not has_metric_data:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        # 隐藏多余的子图
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')

        # 设置总标题
        fig.suptitle(f'Learning Dynamics Trends - {dataset_name}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # 保存图表到 figures 子目录
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_path = os.path.join(figures_dir, f"ld_trends_{dataset_name}_{timestamp}.png")
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Trend visualization saved to: {figure_path}")

    logger.info("=" * 80)


@dataclass
class EvaluatorConfig:
    """独立评估配置类"""
    # Model and tokenizer
    model_path: str  # checkpoint path or HF model name
    tokenizer_path: Optional[str] = None  # if None, use model_path

    # Generation settings
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    generate_max_len: int = 8000
    do_sample: bool = True

    # Data settings
    eval_prompt_data: List[str] = field(default_factory=lambda: [
        "data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json",
        "data/eval_data/math500.json",
        "data/eval_data/aime2024.json",
        "data/eval_data/gpqa_diamond.json",
    ])
    prompt_max_len: int = 2048
    max_eval_samples: Optional[int] = None  # 随机抽取的样本数，None表示使用全部数据

    # Output settings
    output_dir: str = "orz_dynamic_log"
    save_detailed_results: bool = True

    # Visualization settings
    enable_visualization: bool = True  # 是否启用可视化

    # Custom analysis settings
    custom_sentence_files: Optional[List[str]] = None  # JSON文件路径列表，每个文件包含{"prompt": "...", "response": "..."}
    verbose_learning_dynamics: bool = False     # 详细输出learning dynamics分析结果

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path


class Evaluator:
    """独立评估类，支持从checkpoint加载模型进行评估"""

    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
        model_path: Optional[str] = None,
        eval_prompt_data: Optional[List[str]] = None,
        model=None,
        tokenizer=None,
        **kwargs
    ):
        """
        初始化评估器

        支持多种初始化方式：
        1. 传入 EvaluatorConfig 对象（保留原有方式）
           Evaluator(config=EvaluatorConfig(...))

        2. 传入必要参数，其他使用默认值
           Evaluator(model_path="...", eval_prompt_data=[...])

        3. 传入已加载的模型对象
           Evaluator(model=my_model, tokenizer=my_tokenizer, eval_prompt_data=[...])

        Args:
            config: EvaluatorConfig 配置对象（可选）
            model_path: 模型路径（可选，当不传 config 时使用）
            eval_prompt_data: 评估数据路径列表（可选）
            model: 已加载的 transformers 模型对象（可选）
            tokenizer: 已加载的 tokenizer 对象（可选）
            **kwargs: 其他 EvaluatorConfig 参数
        """
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()

        # 处理配置对象
        if config is not None:
            # 使用传入的 config 对象
            self.cfg = config
        else:
            # 从参数构建 config 对象
            if model_path is None and model is None:
                raise ValueError("必须指定 model_path 或 model")

            config_kwargs = {
                "model_path": model_path or "dummy_path",  # 当使用预加载模型时，可以是占位符
                "eval_prompt_data": eval_prompt_data or [
                    "data/eval_data/math500.json",
                    "data/eval_data/aime2024.json",
                    "data/eval_data/gpqa_diamond.json",
                ],
            }
            # 合并其他 kwargs
            config_kwargs.update(kwargs)
            self.cfg = EvaluatorConfig(**config_kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.eval_dataset = None
        self.executor = executor
        self._user_provided_model = model
        self._user_provided_tokenizer = tokenizer

        logger.info(f"Initializing Evaluator with config: {self.cfg}")

        # Load components
        if not self._user_provided_tokenizer:
            self._load_tokenizer()
        if not self._user_provided_model:
            self._load_model()
        self._load_eval_datasets()

        logger.info("Evaluator initialization completed")

    def _load_tokenizer(self):
        """Load tokenizer from pretrained model"""
        logger.info(f"Loading tokenizer from {self.cfg.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_path,
            trust_remote_code=True,
        )

    def _load_model(self):
        """Load transformers model"""
        logger.info(f"Loading transformers model from {self.cfg.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model loaded successfully")

    def _load_eval_datasets(self):
        """Load evaluation datasets"""
        import random
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

        # 随机抽取样本（如果指定了max_eval_samples）
        if self.cfg.max_eval_samples is not None and len(dialogues) > self.cfg.max_eval_samples:
            logger.info(f"Randomly sampling {self.cfg.max_eval_samples} samples from {len(dialogues)} total samples")
            dialogues = random.sample(dialogues, self.cfg.max_eval_samples)
            logger.info(f"Sampled {len(dialogues)} samples for evaluation")

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

    async def _analyze_custom_sentence(self):
        """
        分析用户提供的多个完整语句，自回归计算每个位置的logits

        文件格式 (JSON):
        {
            "prompt": "Solve this problem: ...",
            "response": "The answer is ..."
        }

        流程：
        1. 循环处理每个自定义句子文件
        2. 对每个文件：
           - 读取JSON文件
           - Tokenize prompt 和 response
           - 对response中每个位置 t 进行自回归前向传播
           - 拼接所有logits → (1, response_len, vocab_size)
           - 构造labels（prompt部分mask为-100）
           - 调用analyze_learning_dynamics分析
        3. 为每个文件单独输出结果
        """
        if self.cfg.custom_sentence_files is None or len(self.cfg.custom_sentence_files) == 0:
            return

        ld_custom_sentences = {}

        for file_path in self.cfg.custom_sentence_files:
            try:
                logger.info("\n" + "="*80)
                logger.info(f"【Processing: {os.path.basename(file_path)}】")
                logger.info("="*80)

                # Step 1: 加载JSON文件
                logger.info(f"Loading custom sentence from: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'prompt' not in data or 'response' not in data:
                    logger.error("JSON file must contain 'prompt' and 'response' fields")
                    continue

                prompt = data['prompt']
                response = data['response']
                logger.info(f"Prompt: {prompt[:100]}...")
                logger.info(f"Response: {response[:100]}...")

                # Step 2: Tokenize
                prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
                prompt_ids = prompt_inputs["input_ids"][0].tolist()

                response_inputs = self.tokenizer(response, return_tensors="pt", add_special_tokens=False)
                response_ids = response_inputs["input_ids"][0].tolist()

                logger.info(f"Prompt tokens: {len(prompt_ids)}")
                logger.info(f"Response tokens: {len(response_ids)}")

                # Step 3: 自回归计算每个response位置的logits
                logger.info("Computing logits via autoregressive forward pass...")
                all_logits = []

                for t in range(len(response_ids)):
                    # 构造输入：prompt + response[:t]
                    input_ids = prompt_ids + response_ids[:t]
                    input_tensor = torch.tensor([input_ids]).to(self.model.device)

                    # Forward pass，获取最后一个位置的logits（预测response[t]）
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_tensor)
                        logits_at_t = outputs.logits[0, -1, :]  # (vocab_size,)

                    all_logits.append(logits_at_t.cpu())

                logger.info(f"Computed logits for {len(all_logits)} positions")

                # Step 5: 拼接logits到完整序列位置
                full_seq_len = len(prompt_ids) + len(response_ids)
                logits_full = torch.zeros(1, full_seq_len, self.tokenizer.vocab_size)

                # 将response的logits放到对应位置（prompt部分为0）
                for t, logit in enumerate(all_logits):
                    logits_full[0, len(prompt_ids) + t, :] = logit

                # Step 6: 构造labels
                labels = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
                # Mask prompt部分
                labels[:, :len(prompt_ids)] = -100

                logger.info(f"Logits shape: {logits_full.shape}")
                logger.info(f"Labels shape: {labels.shape}")

                # Step 7: 分析Learning Dynamics
                logger.info("\n" + "="*80)
                logger.info(f"【Custom Sentence Learning Dynamics Analysis - {os.path.basename(file_path)}】")
                logger.info("="*80 + "\n")

                ld_result = analyze_learning_dynamics(
                    logits=logits_full,
                    labels=labels,
                    tokenizer=self.tokenizer,
                    verbose=self.cfg.verbose_learning_dynamics
                )

                # 提取关键指标
                ld_custom_sentences[os.path.splitext(os.path.basename(file_path))[0]] = {
                    'out_token': ld_result['per_sample']['out_token'][0],
                    'out_argmax': ld_result['per_sample']['out_argmax'][0],
                    'A_norm': float(ld_result['per_sample']['A_norm'].squeeze()),
                    'prob_gap2_mean': ld_result['per_sample']['prob_gap2_mean'][0],
                    'prob_energy': ld_result['per_sample']['prob_energy'][0],
                }

                logger.info("="*80 + "\n")

            except FileNotFoundError:
                logger.error(f"Custom sentence file not found: {file_path}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file: {e}")
            except Exception as e:
                logger.error(f"Failed to analyze custom sentence: {e}")
                import traceback
                traceback.print_exc()

        return ld_custom_sentences

    def _sample_random_tokens(self, logits, labels, token_ids, num_samples=5):
        """
        从生成的序列中随机采样 token 及其对应的 logits

        Args:
            logits: torch.FloatTensor, shape (1, seq_len, vocab_size)
            labels: torch.LongTensor, shape (1, seq_len)
            token_ids: list of generated token IDs
            num_samples: 采样的 token 数量（默认 5）

        Returns:
            sampled_data: 字典，包含采样的 token 信息
                {
                    'sample_positions': [pos1, pos2, ...],
                    'sample_tokens': [token_id1, token_id2, ...],
                    'sample_logprobs': [logprob1, logprob2, ...],
                    'sample_token_names': ['token_name1', 'token_name2', ...],
                    'sample_token_probs': [prob1, prob2, ...],
                }
        """
        import random

        if logits is None or labels is None or token_ids is None:
            return None

        seq_len = len(token_ids)

        # 有效位置数（非 -100 mask）
        valid_positions = [i for i in range(seq_len) if i < labels.shape[1] and labels[0, i] != -100]

        if len(valid_positions) == 0:
            return None

        # 确定实际采样数量
        actual_samples = min(num_samples, len(valid_positions))

        # 随机采样位置
        sampled_positions = sorted(random.sample(valid_positions, actual_samples))

        sampled_data = {
            'sample_positions': sampled_positions,
            'sample_tokens': [],
            'sample_logprobs': [],
            'sample_token_names': [],
            'sample_token_probs': [],
        }

        for pos in sampled_positions:
            token_id = int(token_ids[pos])

            # 获取完整的 logits 向量（所有 vocab）
            logit_vector = logits[0, pos, :].detach().cpu()

            # 计算该位置的 log 概率和概率
            log_probs = torch.nn.functional.log_softmax(logit_vector, dim=-1)
            probs = torch.softmax(logit_vector, dim=-1)

            # 获取该 token 的 log 概率
            token_logprob = log_probs[token_id].item()
            token_prob = probs[token_id].item()

            # 获取 token 名称
            try:
                token_name = self.tokenizer.decode([token_id]).strip()
            except:
                token_name = f"<unk_token_{token_id}>"

            sampled_data['sample_tokens'].append(token_id)
            sampled_data['sample_logprobs'].append(token_logprob)
            sampled_data['sample_token_names'].append(token_name)
            sampled_data['sample_token_probs'].append(token_prob)

        return sampled_data

    async def eval(self) -> dict:
        """执行评估"""
        logger.info("Starting evaluation on datasets (transformers mode)")
        from torch.utils.data import DataLoader

        # Create dataloader
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=len(self.eval_dataset),
            shuffle=False,
            drop_last=False,
        )

        output_for_save = []
        log_dict = defaultdict(float)

        # Learning Dynamics 分析数据收集 - 按数据集分别存储
        ld_by_dataset = {}  # {dataset_name: {'correct': {...}, 'incorrect': {...}, 'random': {...}}}

        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1]["file_name"])

            logger.info(f"Processing {len(prompts)} prompts")

            # 对每个 prompt 进行生成和分析
            for sample_idx, (prompt, answer, file_name) in enumerate(zip(prompts, answers, file_names)):
                logger.info(f"\n{'='*80}")
                logger.info(f"Sample {sample_idx + 1}/{len(prompts)}")
                logger.info(f"{'='*80}")

                # 1. Tokenize prompt
                prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt_input_ids = prompt_inputs["input_ids"].to(self.model.device)
                prompt_attention_mask = prompt_inputs["attention_mask"].to(self.model.device)

                # 2. Generate using transformers
                logger.info("Generating text with transformers...")
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=prompt_input_ids,
                        attention_mask=prompt_attention_mask,
                        max_new_tokens=self.cfg.generate_max_len,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        top_k=self.cfg.top_k if self.cfg.top_k > 0 else None,
                        do_sample=self.cfg.do_sample,
                        return_dict_in_generate=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # 3. Decode output
                generated_ids = outputs[0]
                output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                token_ids = generated_ids.tolist()

                logger.info(f"Generated {len(token_ids)} tokens")

                # 4. Forward pass to get complete logits
                logger.info("Getting complete logits...")
                with torch.no_grad():
                    model_outputs = self.model(
                        input_ids=generated_ids.unsqueeze(0),
                        attention_mask=torch.ones_like(generated_ids).unsqueeze(0),
                    )
                    logits = model_outputs.logits  # (1, seq_len, vocab_size)

                logger.info(f"Logits shape: {logits.shape}")

                # 5. Prepare labels for learning dynamics calculation
                prompt_len = prompt_input_ids.shape[1]
                labels = generated_ids.clone().unsqueeze(0)
                labels[:, :prompt_len] = -100  # mask 掉 prompt 部分
                labels = labels.long()

                # 6. Extract final answer and check correctness
                pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
                matches = re.findall(pattern, output_text)
                final_answer = matches[-1] if len(matches) > 0 else ""

                label = solution2answer(answer)
                prefix_response = solution2answer(final_answer)
                iscorrect = await is_equal(label, prefix_response, self.executor)

                logger.info(f"Final Answer: {final_answer}")
                logger.info(f"Is Correct: {iscorrect}")

                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output_text,
                        final_answer=final_answer,
                        answer=answer,
                        iscorrect=iscorrect,
                        token_ids=token_ids,
                    )
                )

                # ============================================================
                # Learning Dynamics 分析
                # ============================================================
                logger.info("Learning Dynamics Analysis")

                ld_metrics = None
                if labels is not None:
                    try:
                        # 调用分析函数
                        ld_result = analyze_learning_dynamics(
                            logits=logits,
                            labels=labels,
                            tokenizer=self.tokenizer,
                            verbose=False
                        )

                        # 提取关键指标
                        ld_metrics = {
                            'out_token': ld_result['per_sample']['out_token'][0],
                            'out_argmax': ld_result['per_sample']['out_argmax'][0],
                            'A_norm': float(ld_result['per_sample']['A_norm'].squeeze()),
                            'prob_gap2_mean': ld_result['per_sample']['prob_gap2_mean'][0],
                            'prob_energy': ld_result['per_sample']['prob_energy'][0],
                        }

                        # 初始化该数据集的 LD 数据结构（如果不存在）
                        if file_name not in ld_by_dataset:
                            ld_by_dataset[file_name] = {
                                'correct': {
                                    'out_token': [],
                                    'out_argmax': [],
                                    'A_norm': [],
                                    'prob_gap2_mean': [],
                                    'prob_energy': [],
                                },
                                'incorrect': {
                                    'out_token': [],
                                    'out_argmax': [],
                                    'A_norm': [],
                                    'prob_gap2_mean': [],
                                    'prob_energy': [],
                                },
                                'random': {
                                    'out_token': [],
                                    'out_argmax': [],
                                    'A_norm': [],
                                    'prob_gap2_mean': [],
                                    'prob_energy': [],
                                }
                            }

                        # 根据正确性分类累积数据
                        if iscorrect:
                            ld_by_dataset[file_name]['correct']['out_token'].append(ld_metrics['out_token'])
                            ld_by_dataset[file_name]['correct']['out_argmax'].append(ld_metrics['out_argmax'])
                            ld_by_dataset[file_name]['correct']['A_norm'].append(ld_metrics['A_norm'])
                            ld_by_dataset[file_name]['correct']['prob_gap2_mean'].append(ld_metrics['prob_gap2_mean'])
                            ld_by_dataset[file_name]['correct']['prob_energy'].append(ld_metrics['prob_energy'])
                        else:
                            ld_by_dataset[file_name]['incorrect']['out_token'].append(ld_metrics['out_token'])
                            ld_by_dataset[file_name]['incorrect']['out_argmax'].append(ld_metrics['out_argmax'])
                            ld_by_dataset[file_name]['incorrect']['A_norm'].append(ld_metrics['A_norm'])
                            ld_by_dataset[file_name]['incorrect']['prob_gap2_mean'].append(ld_metrics['prob_gap2_mean'])
                            ld_by_dataset[file_name]['incorrect']['prob_energy'].append(ld_metrics['prob_energy'])

                        # 添加到保存的输出
                        output_for_save[-1]['ld_metrics'] = ld_metrics

                        # ========================================================
                        # 随机采样 token 及对应的 logits
                        # ========================================================
                        sampled_tokens = self._sample_random_tokens(
                            logits=logits,
                            labels=labels,
                            token_ids=token_ids,
                            num_samples=5  # 随机采样 5 个 token
                        )

                        if sampled_tokens is not None:
                            output_for_save[-1]['sampled_tokens'] = sampled_tokens

                            # 对每个采样位置计算学习动态
                            for pos in sampled_tokens['sample_positions']:
                                # 构造单个位置的 labels（其他位置都 mask 为 -100）
                                single_pos_labels = torch.full_like(labels, -100)
                                single_pos_labels[0, pos] = labels[0, pos]

                                try:
                                    # 计算该位置的学习动态
                                    ld_result_pos = analyze_learning_dynamics(
                                        logits=logits,
                                        labels=single_pos_labels,
                                        tokenizer=self.tokenizer,
                                        verbose=False
                                    )

                                    # 累积到随机 tokens 数据（按数据集分别）
                                    ld_by_dataset[file_name]['random']['out_token'].append(ld_result_pos['per_sample']['out_token'][0])
                                    ld_by_dataset[file_name]['random']['out_argmax'].append(ld_result_pos['per_sample']['out_argmax'][0])
                                    ld_by_dataset[file_name]['random']['A_norm'].append(float(ld_result_pos['per_sample']['A_norm'].squeeze()))
                                    ld_by_dataset[file_name]['random']['prob_gap2_mean'].append(ld_result_pos['per_sample']['prob_gap2_mean'][0])
                                    ld_by_dataset[file_name]['random']['prob_energy'].append(ld_result_pos['per_sample']['prob_energy'][0])

                                except Exception as e:
                                    logger.warning(f"Failed to compute learning dynamics for sampled token at position {pos}: {e}")

                    except Exception as e:
                        logger.warning(f"Failed to compute learning dynamics: {e}")

                logger.info(f"{'='*80}\n")

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

        # ====================================================================
        # 计算 Learning Dynamics 统计 - 按数据集分别
        # ====================================================================
        for dataset_name, dataset_ld in ld_by_dataset.items():
            # 正确答案统计
            if dataset_ld['correct']['out_token']:
                log_dict[f"{dataset_name}/ld_correct/out_token"] = float(np.mean(dataset_ld['correct']['out_token']))
                log_dict[f"{dataset_name}/ld_correct/out_argmax"] = float(np.mean(dataset_ld['correct']['out_argmax']))
                log_dict[f"{dataset_name}/ld_correct/A_norm"] = float(np.mean(dataset_ld['correct']['A_norm']))
                log_dict[f"{dataset_name}/ld_correct/prob_gap2_mean"] = float(np.mean(dataset_ld['correct']['prob_gap2_mean']))
                log_dict[f"{dataset_name}/ld_correct/prob_energy"] = float(np.mean(dataset_ld['correct']['prob_energy']))
                log_dict[f"{dataset_name}/ld_correct/count"] = len(dataset_ld['correct']['out_token'])

            # 错误答案统计
            if dataset_ld['incorrect']['out_token']:
                log_dict[f"{dataset_name}/ld_incorrect/out_token"] = float(np.mean(dataset_ld['incorrect']['out_token']))
                log_dict[f"{dataset_name}/ld_incorrect/out_argmax"] = float(np.mean(dataset_ld['incorrect']['out_argmax']))
                log_dict[f"{dataset_name}/ld_incorrect/A_norm"] = float(np.mean(dataset_ld['incorrect']['A_norm']))
                log_dict[f"{dataset_name}/ld_incorrect/prob_gap2_mean"] = float(np.mean(dataset_ld['incorrect']['prob_gap2_mean']))
                log_dict[f"{dataset_name}/ld_incorrect/prob_energy"] = float(np.mean(dataset_ld['incorrect']['prob_energy']))
                log_dict[f"{dataset_name}/ld_incorrect/count"] = len(dataset_ld['incorrect']['out_token'])

            # 随机采样Token统计
            if dataset_ld['random']['out_token']:
                log_dict[f"{dataset_name}/ld_random/out_token"] = float(np.mean(dataset_ld['random']['out_token']))
                log_dict[f"{dataset_name}/ld_random/out_argmax"] = float(np.mean(dataset_ld['random']['out_argmax']))
                log_dict[f"{dataset_name}/ld_random/A_norm"] = float(np.mean(dataset_ld['random']['A_norm']))
                log_dict[f"{dataset_name}/ld_random/prob_gap2_mean"] = float(np.mean(dataset_ld['random']['prob_gap2_mean']))
                log_dict[f"{dataset_name}/ld_random/prob_energy"] = float(np.mean(dataset_ld['random']['prob_energy']))
                log_dict[f"{dataset_name}/ld_random/count"] = len(dataset_ld['random']['out_token'])

        # Output Learning Dynamics summary by dataset
        for dataset_name, dataset_ld in ld_by_dataset.items():
            if dataset_ld['correct']['out_token'] or dataset_ld['incorrect']['out_token']:
                logger.info(f"Learning Dynamics summary for {dataset_name}")

                if dataset_ld['correct']['out_token']:
                    logger.info(f"Correct answers (n={len(dataset_ld['correct']['out_token'])})")
                    logger.info(f"  Energy (prob_energy): {log_dict[f'{dataset_name}/ld_correct/prob_energy']:.4f}")
                    logger.info(f"  Gap (prob_gap2_mean): {log_dict[f'{dataset_name}/ld_correct/prob_gap2_mean']:.4f}")
                    logger.info(f"  A_norm: {log_dict[f'{dataset_name}/ld_correct/A_norm']:.4f}")
                    logger.info(f"  out_token: {log_dict[f'{dataset_name}/ld_correct/out_token']:.4f}")
                    logger.info(f"  out_argmax: {log_dict[f'{dataset_name}/ld_correct/out_argmax']:.4f}")

                if dataset_ld['incorrect']['out_token']:
                    logger.info(f"Incorrect answers (n={len(dataset_ld['incorrect']['out_token'])})")
                    logger.info(f"  Energy (prob_energy): {log_dict[f'{dataset_name}/ld_incorrect/prob_energy']:.4f}")
                    logger.info(f"  Gap (prob_gap2_mean): {log_dict[f'{dataset_name}/ld_incorrect/prob_gap2_mean']:.4f}")
                    logger.info(f"  A_norm: {log_dict[f'{dataset_name}/ld_incorrect/A_norm']:.4f}")
                    logger.info(f"  out_token: {log_dict[f'{dataset_name}/ld_incorrect/out_token']:.4f}")
                    logger.info(f"  out_argmax: {log_dict[f'{dataset_name}/ld_incorrect/out_argmax']:.4f}")

                if dataset_ld['correct']['out_token'] and dataset_ld['incorrect']['out_token']:
                    logger.info("Comparison")
                    energy_diff = log_dict[f'{dataset_name}/ld_incorrect/prob_energy'] - log_dict[f'{dataset_name}/ld_correct/prob_energy']
                    gap_diff = log_dict[f'{dataset_name}/ld_incorrect/prob_gap2_mean'] - log_dict[f'{dataset_name}/ld_correct/prob_gap2_mean']
                    logger.info(f"  Energy difference (incorrect - correct): {energy_diff:.4f}")
                    logger.info(f"  Gap difference (incorrect - correct): {gap_diff:.4f}")
                    if energy_diff > 0:
                        logger.info("  Incorrect answers need more learning energy (model less confident)")
                    else:
                        logger.info("  Correct answers need more learning energy")

                if dataset_ld['random']['out_token']:
                    logger.info(f"Random sampled tokens (n={len(dataset_ld['random']['out_token'])})")
                    logger.info(f"  Energy (prob_energy): {log_dict[f'{dataset_name}/ld_random/prob_energy']:.4f}")
                    logger.info(f"  Gap (prob_gap2_mean): {log_dict[f'{dataset_name}/ld_random/prob_gap2_mean']:.4f}")
                    logger.info(f"  A_norm: {log_dict[f'{dataset_name}/ld_random/A_norm']:.4f}")

        # Generate Learning Dynamics visualization - one chart per dataset
        if self.cfg.enable_visualization:
            for dataset_name, dataset_ld in ld_by_dataset.items():
                if dataset_ld['correct']['out_token'] or dataset_ld['incorrect']['out_token']:
                    try:
                        figure_path = self._visualize_learning_dynamics(
                            dataset_ld['correct'],
                            dataset_ld['incorrect'],
                            dataset_ld['random'],
                            dataset_name=dataset_name
                        )
                        if figure_path:
                            logger.info(f"Visualization generated for {dataset_name}: {figure_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate {dataset_name} visualization: {e}")

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
                    # 递归转换所有 torch 和 numpy 类型为 Python 原生类型
                    def convert_to_serializable(obj):
                        if isinstance(obj, dict):
                            return {k: convert_to_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_serializable(v) for v in obj]
                        elif isinstance(obj, torch.Tensor):
                            return obj.detach().cpu().tolist()
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, (np.integer, np.floating)):
                            return float(obj)
                        elif isinstance(obj, bool):
                            return bool(obj)
                        elif isinstance(obj, (int, float, str)):
                            return obj
                        else:
                            # Handle torch scalar types
                            try:
                                return float(obj)
                            except (TypeError, ValueError):
                                return str(obj)

                    serializable_item = convert_to_serializable(item)
                    f.write(json.dumps(serializable_item, ensure_ascii=False) + "\n")

        # Log results
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(f"Evaluation completed: {logging_str}")

        # ====================================================================
        # 自定义语句分析（如果提供）
        # ====================================================================
        if self.cfg.custom_sentence_files is not None and len(self.cfg.custom_sentence_files) > 0:
            try:
                logger.info("\n" + "="*80)
                logger.info("【Custom Sentence Analysis】")
                logger.info("="*80)
                ld_custom_sentences = await self._analyze_custom_sentence()

                # 统计自定义句子的学习动态指标
                if ld_custom_sentences:
                    for file_name, ld_metrics in ld_custom_sentences.items():
                        log_dict[f"custom_sentence_{file_name}/out_token"] = ld_metrics['out_token']
                        log_dict[f"custom_sentence_{file_name}/out_argmax"] = ld_metrics['out_argmax']
                        log_dict[f"custom_sentence_{file_name}/A_norm"] = ld_metrics['A_norm']
                        log_dict[f"custom_sentence_{file_name}/prob_gap2_mean"] = ld_metrics['prob_gap2_mean']
                        log_dict[f"custom_sentence_{file_name}/prob_energy"] = ld_metrics['prob_energy']

                    # 输出自定义句子统计总结
                    logger.info("\n" + "="*80)
                    logger.info("【Custom Sentence Learning Dynamics Summary】")
                    logger.info("="*80 + "\n")
                    for file_name, ld_metrics in ld_custom_sentences.items():
                        logger.info(f"\n【{file_name}】")
                        logger.info(f"  out_token: {ld_metrics['out_token']:.4f}")
                        logger.info(f"  out_argmax: {ld_metrics['out_argmax']:.4f}")
                        logger.info(f"  A_norm: {ld_metrics['A_norm']:.4f}")
                        logger.info(f"  prob_gap2_mean: {ld_metrics['prob_gap2_mean']:.4f}")
                        logger.info(f"  prob_energy: {ld_metrics['prob_energy']:.4f}")
                    logger.info("\n" + "="*80 + "\n")

            except Exception as e:
                logger.warning(f"Failed to analyze custom sentence: {e}")
                import traceback
                traceback.print_exc()

        return dict(log_dict)

    def _sample_random_tokens(self, logits, labels, token_ids, num_samples=5):
        """
        从生成的序列中随机采样 token 及其对应的 logits

        Args:
            logits: torch.FloatTensor, shape (1, seq_len, vocab_size)
            labels: torch.LongTensor, shape (1, seq_len)
            token_ids: list of generated token IDs
            num_samples: 采样的 token 数量（默认 5）

        Returns:
            sampled_data: 字典，包含采样的 token 信息
                {
                    'sample_positions': [pos1, pos2, ...],
                    'sample_tokens': [token_id1, token_id2, ...],
                    'sample_logits': [logits_vector_1, logits_vector_2, ...],  # 每个都是完整的 vocab_size 维度
                    'sample_token_names': ['token_name1', 'token_name2', ...],
                }
        """
        import random

        if logits is None or labels is None or token_ids is None:
            return None

        seq_len = len(token_ids)

        # 有效位置数（非 -100 mask）
        valid_positions = [i for i in range(seq_len) if i < labels.shape[1] and labels[0, i] != -100]

        if len(valid_positions) == 0:
            return None

        # 确定实际采样数量
        actual_samples = min(num_samples, len(valid_positions))

        # 随机采样位置
        sampled_positions = sorted(random.sample(valid_positions, actual_samples))

        sampled_data = {
            'sample_positions': sampled_positions,
            'sample_tokens': [],
            'sample_logits': [],
            'sample_logprobs': [],  # log 概率
            'sample_token_names': [],
            'sample_token_probs': [],  # 实际概率
        }

        for pos in sampled_positions:
            token_id = int(token_ids[pos])

            # 获取完整的 logits 向量（所有 vocab）
            logit_vector = logits[0, pos, :].detach().cpu()

            # 计算该位置的 log 概率和概率
            log_probs = torch.nn.functional.log_softmax(logit_vector, dim=-1)
            probs = torch.softmax(logit_vector, dim=-1)

            # 获取该 token 的 log 概率
            token_logprob = log_probs[token_id].item()
            token_prob = probs[token_id].item()

            # 获取 token 名称
            try:
                token_name = self.tokenizer.decode([token_id]).strip()
            except:
                token_name = f"<unk_token_{token_id}>"

            sampled_data['sample_tokens'].append(token_id)
            sampled_data['sample_logits'].append(logit_vector.numpy().tolist())  # 完整 logits 向量
            sampled_data['sample_logprobs'].append(token_logprob)
            sampled_data['sample_token_names'].append(token_name)
            sampled_data['sample_token_probs'].append(token_prob)

        return sampled_data

    def _visualize_learning_dynamics(self, ld_correct_samples, ld_incorrect_samples, ld_random_tokens=None, dataset_name="All"):
        """
        Generate comparison bar chart for learning dynamics metrics

        Args:
            ld_correct_samples: 正确答案的学习动态数据
            ld_incorrect_samples: 错误答案的学习动态数据
            ld_random_tokens: 随机采样token的学习动态数据（可选）
            dataset_name: 数据集名称，用于图表标题和文件名
        """
        # Check if there is data
        if not ld_correct_samples['out_token'] and not ld_incorrect_samples['out_token']:
            logger.warning(f"No learning dynamics data to visualize for {dataset_name}")
            return None

        # 5 key metrics
        metrics = ['out_token', 'out_argmax', 'A_norm', 'prob_gap2_mean', 'prob_energy']
        metric_labels = {
            'out_token': 'True Label\nLog Probability',
            'out_argmax': 'Argmax Token\nLog Probability',
            'A_norm': 'Output Vector\nNorm',
            'prob_gap2_mean': 'Prediction-Label\nGap',
            'prob_energy': 'Pull-up Energy\n(Correction Strength)',
        }

        # Calculate mean and std for each metric
        correct_means = []
        correct_stds = []
        incorrect_means = []
        incorrect_stds = []
        random_means = []
        random_stds = []

        for metric in metrics:
            # Correct answers
            if ld_correct_samples[metric]:
                correct_means.append(float(np.mean(ld_correct_samples[metric])))
                correct_stds.append(float(np.std(ld_correct_samples[metric])))
            else:
                correct_means.append(0)
                correct_stds.append(0)

            # Incorrect answers
            if ld_incorrect_samples[metric]:
                incorrect_means.append(float(np.mean(ld_incorrect_samples[metric])))
                incorrect_stds.append(float(np.std(ld_incorrect_samples[metric])))
            else:
                incorrect_means.append(0)
                incorrect_stds.append(0)

            # Random tokens（可选）
            if ld_random_tokens and ld_random_tokens[metric]:
                random_means.append(float(np.mean(ld_random_tokens[metric])))
                random_stds.append(float(np.std(ld_random_tokens[metric])))
            else:
                random_means.append(0)
                random_stds.append(0)

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))

        x = np.arange(len(metrics))

        # 调整宽度以支持 3 组数据
        if ld_random_tokens and ld_random_tokens['out_token']:
            width = 0.25  # 3 组柱子
            bars1 = ax.bar(x - width, correct_means, width, label='Correct Answers',
                           color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5,
                           yerr=correct_stds, capsize=5, error_kw={'linewidth': 1.5})
            bars2 = ax.bar(x, incorrect_means, width, label='Incorrect Answers',
                           color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5,
                           yerr=incorrect_stds, capsize=5, error_kw={'linewidth': 1.5})
            bars3 = ax.bar(x + width, random_means, width, label='Random Tokens',
                           color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5,
                           yerr=random_stds, capsize=5, error_kw={'linewidth': 1.5})

            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

            add_value_labels(bars1)
            add_value_labels(bars2)
            add_value_labels(bars3)
        else:
            # 只有 2 组数据
            width = 0.35
            bars1 = ax.bar(x - width/2, correct_means, width, label='Correct Answers',
                           color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5,
                           yerr=correct_stds, capsize=5, error_kw={'linewidth': 1.5})
            bars2 = ax.bar(x + width/2, incorrect_means, width, label='Incorrect Answers',
                           color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5,
                           yerr=incorrect_stds, capsize=5, error_kw={'linewidth': 1.5})

            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

            add_value_labels(bars1)
            add_value_labels(bars2)

        # Set axes
        ax.set_xlabel('Learning Dynamics Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Learning Dynamics Comparison - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=11)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add sample count info
        correct_count = len(ld_correct_samples['out_token']) if ld_correct_samples['out_token'] else 0
        incorrect_count = len(ld_incorrect_samples['out_token']) if ld_incorrect_samples['out_token'] else 0
        random_count = len(ld_random_tokens['out_token']) if ld_random_tokens and ld_random_tokens['out_token'] else 0

        if random_count > 0:
            info_text = f'Correct Samples: {correct_count}  |  Incorrect Samples: {incorrect_count}  |  Random Tokens: {random_count}'
        else:
            info_text = f'Correct Samples: {correct_count}  |  Incorrect Samples: {incorrect_count}'

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, style='italic')

        # Tight layout
        plt.tight_layout(rect=[0, 0.03, 1, 1])

        # Save figure with dataset name to figures subdirectory
        figures_dir = os.path.join(self.cfg.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_path = os.path.join(figures_dir, f"ld_comparison_{dataset_name}_{timestamp}.png")
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Learning Dynamics visualization saved to: {figure_path}")
        return figure_path


    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up evaluator resources")
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints with learning dynamics analysis")
    parser.add_argument("--max_eval_samples", type=int, default=500,
                        help="Max number of samples to randomly sample for evaluation (default: 500, use -1 for all)")
    parser.add_argument("--checkpoint_paths", nargs="+", default=[
        "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter0/policy",
        "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter45/policy",
        "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter90/policy",
        "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter180/policy",
    ], help="Checkpoint paths to evaluate (default: iter0, iter45, iter90, iter180)")
    parser.add_argument("--output_dir", type=str, default="orz_dynamic_log",
                        help="Output directory for results (default: orz_dynamic_log)")
    parser.add_argument("--generate_max_len", type=int, default=8000,
                        help="Maximum generation length (default: 8000)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling parameter (default: -1, disabled)")

    args = parser.parse_args()

    logger.info("Running in evaluation mode (transformers)")
    logger.info(f"Command line arguments: {args}")

    # Handle max_eval_samples: -1 means use all samples
    max_eval_samples = args.max_eval_samples if args.max_eval_samples > 0 else None

    checkpoint_path_list = args.checkpoint_paths
    all_model_results = {}

    for checkpoint_path in checkpoint_path_list:
        logger.info(f"\n{'='*80}")
        logger.info(f"【Processing: {checkpoint_path}】")
        logger.info(f"{'='*80}\n")

        eval_config = EvaluatorConfig(
            model_path=checkpoint_path,
            tokenizer_path=checkpoint_path,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            generate_max_len=args.generate_max_len,
            eval_prompt_data=[
                "data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json",
                "data/eval_data/math500.json",
                "data/eval_data/aime2024.json",
                "data/eval_data/gpqa_diamond.json",
            ],
            prompt_max_len=2048,
            output_dir=args.output_dir,
            save_detailed_results=True,
            max_eval_samples=max_eval_samples,  # 从命令行参数传入
        )
        evaluator = Evaluator(eval_config)

        try:
            results = asyncio.run(evaluator.eval())
            logger.info(f"Evaluation results for {checkpoint_path}: {results}")
            all_model_results[checkpoint_path] = results
        finally:
            evaluator.cleanup()

    # ====================================================================
    # 【多 Checkpoint 趋势分析和可视化】
    # ====================================================================
    logger.info("Multi-Checkpoint Learning Dynamics Trend Analysis")

    logger.info("Aggregating results from multiple checkpoints...")
    aggregated_data = aggregate_multi_checkpoint_results(all_model_results)

    logger.info("Aggregated data statistics:")
    for dataset_name, dataset_data in aggregated_data.items():
        logger.info(f"  Dataset: {dataset_name}")
        for sample_type in ['correct', 'incorrect', 'random']:
            if sample_type in dataset_data and dataset_data[sample_type]:
                num_metrics = len(dataset_data[sample_type])
                logger.info(f"    {sample_type}: {num_metrics} metrics")

    logger.info("Generating multi-checkpoint trends visualization...")
    visualize_multi_checkpoint_trends(aggregated_data, output_dir="orz_dynamic_log")

    logger.info("All evaluations and visualizations completed!")
