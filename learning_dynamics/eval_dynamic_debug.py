"""
快速调试版本 - 使用 transformers 模型生成和分析 Learning Dynamics
(替代 vLLM，直接获取完整 vocab_size 的 logits)
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

# 配置日志文件输出到 orz_dynamic_debug_log 目录
log_dir = "orz_dynamic_debug_log"
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
from .extracted_get_batch_logps import analyze_learning_dynamics

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
    import re
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


def visualize_multi_checkpoint_trends(aggregated_data: dict, output_dir: str = "orz_dynamic_debug_log"):
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


@dataclass
class EvaluatorConfig:
    """JSONL 分析专用配置类"""
    # Model and tokenizer
    model_path: str
    tokenizer_path: Optional[str] = None

    # JSONL 文件路径和分析设置
    jsonl_file: str = ""  # JSONL 文件路径，包含已生成的模型输出
    num_correct_samples: int = 10   # 分析的正确样本数
    num_incorrect_samples: int = 10 # 分析的错误样本数

    # Output settings
    output_dir: str = "orz_dynamic_debug_log"

    # Visualization settings
    enable_visualization: bool = True  # 是否启用多 checkpoint 趋势可视化

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path


class Evaluator:
    """快速调试版评估器 - 使用 transformers"""

    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
        model_path: Optional[str] = None,
        model=None,
        tokenizer=None,
        **kwargs
    ):
        """初始化 JSONL 分析评估器"""
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()

        # 处理配置对象
        if config is not None:
            self.cfg = config
        else:
            if model_path is None and model is None:
                raise ValueError("必须指定 model_path 或 model")

            config_kwargs = {
                "model_path": model_path or "dummy_path",
            }
            config_kwargs.update(kwargs)
            self.cfg = EvaluatorConfig(**config_kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.executor = executor
        self._user_provided_model = model
        self._user_provided_tokenizer = tokenizer

        logger.info(f"Initializing JSONL Analyzer with config: {self.cfg}")

        # Load model and tokenizer only
        if not self._user_provided_tokenizer:
            self._load_tokenizer()
        if not self._user_provided_model:
            self._load_model()

        logger.info("JSONL Analyzer initialization completed")

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
        """
        import random

        if logits is None or labels is None or token_ids is None:
            return None

        seq_len = len(token_ids)
        valid_positions = [i for i in range(seq_len) if i < labels.shape[1] and labels[0, i] != -100]

        if len(valid_positions) == 0:
            return None

        actual_samples = min(num_samples, len(valid_positions))
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
            logit_vector = logits[0, pos, :].detach().cpu()
            log_probs = torch.nn.functional.log_softmax(logit_vector, dim=-1)
            probs = torch.softmax(logit_vector, dim=-1)
            token_logprob = log_probs[token_id].item()
            token_prob = probs[token_id].item()

            try:
                token_name = self.tokenizer.decode([token_id]).strip()
            except:
                token_name = f"<unk_token_{token_id}>"

            sampled_data['sample_tokens'].append(token_id)
            sampled_data['sample_logprobs'].append(token_logprob)
            sampled_data['sample_token_names'].append(token_name)
            sampled_data['sample_token_probs'].append(token_prob)

        return sampled_data

    async def _analyze_custom_jsonl(self):
        """
        分析JSONL格式的已生成评估输出

        JSONL格式（每行一个JSON对象）：
        {
            "prompt": "完整的prompt文本",
            "output": "模型生成的完整输出",
            "final_answer": "从输出中提取的答案",
            "answer": "ground truth答案",
            "iscorrect": true/false
        }

        流程：
        1. 读取JSONL文件
        2. 根据iscorrect字段分类为正确和错误样本
        3. 收集指定数量的正确和错误样本
        4. 对每个样本：
           - Tokenize prompt 和 output
           - 对output中每个位置 t 进行自回归前向传播
           - 拼接所有logits → (1, output_len, vocab_size)
           - 构造labels（prompt部分mask为-100）
           - 调用analyze_learning_dynamics分析
        5. 为正确/错误样本分别汇总结果
        """
        if self.cfg.jsonl_file is None or self.cfg.jsonl_file == "":
            return None

        ld_jsonl_results = {
            'correct': [],
            'incorrect': []
        }

        try:
            logger.info(f"Loading JSONL file: {self.cfg.jsonl_file}")

            # Step 1: 读取JSONL文件，分类样本
            correct_samples = []
            incorrect_samples = []

            with open(self.cfg.jsonl_file, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if line.strip() == '':
                        continue

                    try:
                        data = json.loads(line)

                        # 检查必要字段
                        if 'prompt' not in data or 'output' not in data:
                            logger.warning(f"Line {line_idx}: Missing 'prompt' or 'output' field, skipping")
                            continue

                        # 检查是否提供了correctness信息
                        is_correct = data.get('iscorrect', None)

                        if is_correct is True:
                            correct_samples.append(data)
                        elif is_correct is False:
                            incorrect_samples.append(data)
                        else:
                            logger.warning(f"Line {line_idx}: 'iscorrect' field not found or not boolean, skipping")
                            continue

                        # 检查是否已收集足够的样本
                        if (len(correct_samples) >= self.cfg.num_correct_samples and
                            len(incorrect_samples) >= self.cfg.num_incorrect_samples):
                            break

                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_idx}: Failed to parse JSON: {e}")
                        continue

            logger.info(f"Loaded {len(correct_samples)} correct samples, {len(incorrect_samples)} incorrect samples")

            # Step 2: 处理正确样本
            if len(correct_samples) > 0:
                logger.info(f"Analyzing {min(len(correct_samples), self.cfg.num_correct_samples)} correct samples...")
                for sample_idx, sample in enumerate(correct_samples[:self.cfg.num_correct_samples]):
                    try:
                        ld_metrics = self._process_jsonl_sample(sample, f"correct_{sample_idx}")
                        if ld_metrics:
                            ld_jsonl_results['correct'].append(ld_metrics)
                    except Exception as e:
                        logger.error(f"Failed to analyze correct sample {sample_idx}: {e}")
                        continue

            # Step 3: 处理错误样本
            if len(incorrect_samples) > 0:
                logger.info(f"Analyzing {min(len(incorrect_samples), self.cfg.num_incorrect_samples)} incorrect samples...")
                for sample_idx, sample in enumerate(incorrect_samples[:self.cfg.num_incorrect_samples]):
                    try:
                        ld_metrics = self._process_jsonl_sample(sample, f"incorrect_{sample_idx}")
                        if ld_metrics:
                            ld_jsonl_results['incorrect'].append(ld_metrics)
                    except Exception as e:
                        logger.error(f"Failed to analyze incorrect sample {sample_idx}: {e}")
                        continue

            logger.info("="*80 + "\n")
            return ld_jsonl_results

        except FileNotFoundError:
            logger.error(f"JSONL file not found: {self.cfg.custom_sentence_jsonl_file}")
            return None
        except Exception as e:
            logger.error(f"Failed to analyze JSONL file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_jsonl_sample(self, sample: dict, sample_id: str) -> Optional[dict]:
        """
        处理单个JSONL样本的Learning Dynamics分析

        Args:
            sample: JSONL中的单个样本
            sample_id: 样本ID（用于日志）

        Returns:
            LD指标字典，或None如果处理失败
        """
        try:
            prompt = sample['prompt']
            output = sample['output']

            logger.info(f"Processing sample: {sample_id}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Output: {output[:100]}...")

            # Step 1: Tokenize
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_ids = prompt_inputs["input_ids"][0].tolist()

            output_inputs = self.tokenizer(output, return_tensors="pt", add_special_tokens=False)
            output_ids = output_inputs["input_ids"][0].tolist()

            logger.info(f"Prompt tokens: {len(prompt_ids)}, Output tokens: {len(output_ids)}")

            # Step 2: 自回归计算每个output位置的logits
            logger.info("Computing logits via autoregressive forward pass...")
            all_logits = []

            for t in range(len(output_ids)):
                # 构造输入：prompt + output[:t]
                input_ids = prompt_ids + output_ids[:t]
                input_tensor = torch.tensor([input_ids]).to(self.model.device)

                # Forward pass，获取最后一个位置的logits（预测output[t]）
                with torch.no_grad():
                    outputs = self.model(input_ids=input_tensor)
                    logits_at_t = outputs.logits[0, -1, :]  # (vocab_size,)

                all_logits.append(logits_at_t.cpu())

            logger.info(f"Computed logits for {len(all_logits)} positions")

            # Step 3: 拼接logits到完整序列位置
            full_seq_len = len(prompt_ids) + len(output_ids)
            # 使用实际 logits 的 vocab_size，而不是 tokenizer 报告的 vocab_size
            # 因为模型可能经过了 token embedding resize 等操作
            actual_vocab_size = all_logits[0].shape[0] if len(all_logits) > 0 else self.tokenizer.vocab_size
            logits_full = torch.zeros(1, full_seq_len, actual_vocab_size)

            # 将output的logits放到对应位置（prompt部分为0）
            for t, logit in enumerate(all_logits):
                logits_full[0, len(prompt_ids) + t, :] = logit

            # Step 4: 构造labels
            labels = torch.tensor([prompt_ids + output_ids], dtype=torch.long)
            # Mask prompt部分
            labels[:, :len(prompt_ids)] = -100

            logger.info(f"Logits shape: {logits_full.shape}, Labels shape: {labels.shape}")

            # Step 5: Analyze Learning Dynamics
            ld_result = analyze_learning_dynamics(
                logits=logits_full,
                labels=labels,
                tokenizer=self.tokenizer,
                verbose=False
            )

            # 提取关键指标
            metrics = {
                'sample_id': sample_id,
                'prompt_len': len(prompt_ids),
                'output_len': len(output_ids),
                'out_token': float(ld_result['per_sample']['out_token'][0]),
                'out_argmax': float(ld_result['per_sample']['out_argmax'][0]),
                'A_norm': float(ld_result['per_sample']['A_norm'].squeeze()),
                'prob_gap2_mean': float(ld_result['per_sample']['prob_gap2_mean'][0]),
                'prob_energy': float(ld_result['per_sample']['prob_energy'][0]),
            }

            logger.info(f"✓ Completed analysis for {sample_id}")
            return metrics

        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def eval(self) -> dict:
        """JSONL 文件分析评估"""
        log_dict = defaultdict(float)

        logger.info("="*80)
        logger.info("📄 Analyzing JSONL file")
        logger.info(f"File: {self.cfg.jsonl_file}")
        logger.info(f"Correct samples: {self.cfg.num_correct_samples}, Incorrect samples: {self.cfg.num_incorrect_samples}")
        logger.info("="*80)

        # 分析JSONL文件
        ld_jsonl_results = await self._analyze_custom_jsonl()

        # 处理和汇总结果 - 调整格式以适配多checkpoint汇总
        if ld_jsonl_results:
            # 为每个样本生成 checkpoint/dataset/type/metric 格式的 key
            # 这样可以被 aggregate_multi_checkpoint_results 正确处理
            if ld_jsonl_results['correct']:
                for metric in ['out_token', 'out_argmax', 'A_norm', 'prob_gap2_mean', 'prob_energy']:
                    values = [s[metric] for s in ld_jsonl_results['correct']]
                    if values:
                        log_dict[f"jsonl_dataset/ld_correct/{metric}"] = values
                        log_dict[f"jsonl_dataset_correct_{metric}_mean"] = float(np.mean(values))
                        log_dict[f"jsonl_dataset_correct_{metric}_std"] = float(np.std(values))

            if ld_jsonl_results['incorrect']:
                for metric in ['out_token', 'out_argmax', 'A_norm', 'prob_gap2_mean', 'prob_energy']:
                    values = [s[metric] for s in ld_jsonl_results['incorrect']]
                    if values:
                        log_dict[f"jsonl_dataset/ld_incorrect/{metric}"] = values
                        log_dict[f"jsonl_dataset_incorrect_{metric}_mean"] = float(np.mean(values))
                        log_dict[f"jsonl_dataset_incorrect_{metric}_std"] = float(np.std(values))

            # 输出统计总结
            logger.info("\n" + "="*80)
            logger.info("【JSONL Analysis Summary】")
            logger.info("="*80 + "\n")

            if ld_jsonl_results['correct']:
                logger.info(f"\n【Correct Samples (n={len(ld_jsonl_results['correct'])})】")
                logger.info(f"  out_token: {float(np.mean([s['out_token'] for s in ld_jsonl_results['correct']])):.4f}")
                logger.info(f"  out_argmax: {float(np.mean([s['out_argmax'] for s in ld_jsonl_results['correct']])):.4f}")
                logger.info(f"  A_norm: {float(np.mean([s['A_norm'] for s in ld_jsonl_results['correct']])):.4f}")
                logger.info(f"  prob_gap2_mean: {float(np.mean([s['prob_gap2_mean'] for s in ld_jsonl_results['correct']])):.4f}")
                logger.info(f"  prob_energy: {float(np.mean([s['prob_energy'] for s in ld_jsonl_results['correct']])):.4f}")

            if ld_jsonl_results['incorrect']:
                logger.info(f"\n【Incorrect Samples (n={len(ld_jsonl_results['incorrect'])})】")
                logger.info(f"  out_token: {float(np.mean([s['out_token'] for s in ld_jsonl_results['incorrect']])):.4f}")
                logger.info(f"  out_argmax: {float(np.mean([s['out_argmax'] for s in ld_jsonl_results['incorrect']])):.4f}")
                logger.info(f"  A_norm: {float(np.mean([s['A_norm'] for s in ld_jsonl_results['incorrect']])):.4f}")
                logger.info(f"  prob_gap2_mean: {float(np.mean([s['prob_gap2_mean'] for s in ld_jsonl_results['incorrect']])):.4f}")
                logger.info(f"  prob_energy: {float(np.mean([s['prob_energy'] for s in ld_jsonl_results['incorrect']])):.4f}")
            logger.info("\n" + "="*80 + "\n")

        return dict(log_dict)

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

        metrics = ['out_token', 'out_argmax', 'A_norm', 'prob_gap2_mean', 'prob_energy']
        metric_labels = {
            'out_token': 'True Label\nLog Probability',
            'out_argmax': 'Argmax Token\nLog Probability',
            'A_norm': 'Output Vector\nNorm',
            'prob_gap2_mean': 'Prediction-Label\nGap',
            'prob_energy': 'Pull-up Energy\n(Correction Strength)',
        }

        correct_means = []
        correct_stds = []
        incorrect_means = []
        incorrect_stds = []
        random_means = []
        random_stds = []

        for metric in metrics:
            if ld_correct_samples[metric]:
                correct_means.append(float(np.mean(ld_correct_samples[metric])))
                correct_stds.append(float(np.std(ld_correct_samples[metric])))
            else:
                correct_means.append(0)
                correct_stds.append(0)

            if ld_incorrect_samples[metric]:
                incorrect_means.append(float(np.mean(ld_incorrect_samples[metric])))
                incorrect_stds.append(float(np.std(ld_incorrect_samples[metric])))
            else:
                incorrect_means.append(0)
                incorrect_stds.append(0)

            # 随机 tokens 数据（可选）
            if ld_random_tokens and ld_random_tokens[metric]:
                random_means.append(float(np.mean(ld_random_tokens[metric])))
                random_stds.append(float(np.std(ld_random_tokens[metric])))
            else:
                random_means.append(0)
                random_stds.append(0)

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

        ax.set_xlabel('Learning Dynamics Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Learning Dynamics Comparison - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=11)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        correct_count = len(ld_correct_samples['out_token']) if ld_correct_samples['out_token'] else 0
        incorrect_count = len(ld_incorrect_samples['out_token']) if ld_incorrect_samples['out_token'] else 0
        random_count = len(ld_random_tokens['out_token']) if ld_random_tokens and ld_random_tokens['out_token'] else 0

        if random_count > 0:
            info_text = f'Correct Samples: {correct_count}  |  Incorrect Samples: {incorrect_count}  |  Random Tokens: {random_count}'
        else:
            info_text = f'Correct Samples: {correct_count}  |  Incorrect Samples: {incorrect_count}'

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 1])

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
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    import argparse

    # Debug evaluation mode - 快速测试（使用 transformers）
    logger.info("Running in DEBUG evaluation mode (TRANSFORMERS)")
    logger.info("This is a fast debug version with minimal samples")
    logger.info("Using transformers for generation (not vLLM)")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="JSONL Learning Dynamics Analyzer")
    parser.add_argument("--checkpoint_paths", nargs="+", default=None, help="Checkpoint paths to evaluate")
    parser.add_argument("--model_path", type=str, default=None, help="Model path (alternative to checkpoint_paths)")
    parser.add_argument("--jsonl_file", type=str, required=True, help="JSONL file path containing pre-generated evaluation outputs")
    parser.add_argument("--num_correct_samples", type=int, default=10, help="Number of correct samples to analyze")
    parser.add_argument("--num_incorrect_samples", type=int, default=10, help="Number of incorrect samples to analyze")
    parser.add_argument("--output_dir", type=str, default="eval_results_debug", help="Output directory for results")
    parser.add_argument("--log_dir", type=str, default="orz_dynamic_debug_log", help="Log directory name")

    args = parser.parse_args()

    # Reconfigure logger with user-provided log_dir
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_date = datetime.now().strftime("%Y%m%d")
    # Remove existing handlers and add new one with the custom log_dir
    logger.remove()  # Remove all existing handlers
    logger.add(
        os.path.join(log_dir, f"eval_{log_date}.log"),
        rotation="00:00",  # 每天午夜轮转
        retention="30 days",  # 保留30天
        level="INFO",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.info(f"Logger reconfigured with log_dir: {log_dir}")

    # Determine checkpoint paths
    if args.checkpoint_paths:
        checkpoint_path_list = args.checkpoint_paths
    elif args.model_path:
        checkpoint_path_list = [args.model_path]
    else:
        checkpoint_path_list = [
            "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter0/policy",
            "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter45/policy",
            "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter90/policy",
            "/mnt/shared-storage-user/tangjia/orz_7b_ppo_jericho_1013/iter180/policy",
            # 可以添加更多 checkpoint 路径
        ]

    all_model_results = {}

    for checkpoint_path in checkpoint_path_list:
        logger.info(f"\n{'='*80}")
        logger.info(f"【Processing: {checkpoint_path}】")
        logger.info(f"{'='*80}\n")

        eval_config = EvaluatorConfig(
            model_path=checkpoint_path,
            tokenizer_path=checkpoint_path,
            jsonl_file=args.jsonl_file,
            num_correct_samples=args.num_correct_samples,
            num_incorrect_samples=args.num_incorrect_samples,
            output_dir=args.output_dir,
            enable_visualization=True,
        )
        evaluator = Evaluator(eval_config)

        try:
            results = asyncio.run(evaluator.eval())
            logger.info(f"Debug evaluation results for {checkpoint_path}: {results}")
            all_model_results[checkpoint_path] = results
        finally:
            evaluator.cleanup()

    # 汇总所有结果
    logger.info(f"\n{'='*80}")
    logger.info(f"【All Model Results Summary】")
    logger.info(f"{'='*80}\n")
    for checkpoint_path, results in all_model_results.items():
        logger.info(f"\n【{checkpoint_path}】")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")

    logger.info("Multi-Checkpoint Learning Dynamics Trend Analysis")

    # Aggregate results
    logger.info("Aggregating results from multiple checkpoints...")
    aggregated_data = aggregate_multi_checkpoint_results(all_model_results)

    # Output summary statistics
    logger.info("Aggregated data statistics:")
    for dataset_name, dataset_data in aggregated_data.items():
        logger.info(f"  Dataset: {dataset_name}")
        for sample_type in ['correct', 'incorrect', 'random']:
            if sample_type in dataset_data and dataset_data[sample_type]:
                num_metrics = len(dataset_data[sample_type])
                logger.info(f"    {sample_type}: {num_metrics} metrics")

    # Generate trend visualization
    logger.info("Generating multi-checkpoint trends visualization...")
    visualize_multi_checkpoint_trends(aggregated_data, output_dir="orz_dynamic_debug_log")

    logger.info("All evaluations and visualizations completed!")


