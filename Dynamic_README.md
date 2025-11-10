# ORZ Dynamic - 学习动态评估系统

ORZ Dynamic 是一个用于评估大型语言模型（LLM）学习动态的独立模块。该系统通过分析模型生成过程中的内部激活和概率分布，提供深度的学习动态洞察，包括模型确定性、能量流动和预测偏差等关键指标。

## 功能特性

- 🔬 **学习动态分析** - 计算5大核心指标：能量、差距、A_norm等
- 📊 **多维度评估** - 按数据集、答案正确性进行分层统计
- 📈 **趋势可视化** - 支持多checkpoint的趋势分析和对比
- ⚡ **轻量级调试** - 提供快速调试版本用于快速迭代
- 🎯 **自定义分析** - 支持对任意句子的自回归分析

## 项目结构

```
orz-dynamic/
├── eval_dynamic.py                    # 完整评估系统
├── eval_dynamic_debug.py              # 快速调试版本
├── extracted_get_batch_logps.py       # 核心分析引擎
├── requirements.txt                   # 依赖列表
├── README.md                          # 本文件
├── checkpoints/                       # 模型checkpoints（软链接）
├── data/                              # 评估数据集
│   └── eval_data/
│       ├── math500.json               # Math500数据集
│       ├── aime2024.json              # AIME2024数据集
│       ├── gpqa_diamond.json          # GPQA Diamond数据集
│       └── eval_jericho_*.json        # Jericho评估数据
├── dataset/                           # 数据集处理模块
│   ├── __init__.py
│   └── eval_dataset.py
├── orz/                               # 核心库
│   ├── ppo/
│   │   ├── dataset.py
│   │   ├── deepspeed_strategy.py
│   │   ├── vllm_utils.py
│   │   └── tools/
│   │       ├── __init__.py
│   │       └── math_utils.py
│   └── exp_engine/
│       ├── accelerators/
│       │   └── inference/
│       │       ├── vllm_engine.py
│       │       └── vllm_worker_wrap.py
│       └── parallels/
│           └── orz_distributed_c10d.py
└── orz_dynamic_log/                   # 日志和结果输出（自动创建）
    ├── eval_*.log
    ├── eval_results_*.jsonl
    └── figures/
        ├── ld_comparison_*.png
        └── ld_trends_*.png
```

## 三大核心模块

### 1. **eval_dynamic.py** - 完整评估系统

完整的学习动态评估引擎，用于生产环境。

**主要功能：**
- 对多个checkpoint进行完整评估
- 支持Transformers模型加载和生成
- 详细的学习动态分析
- 多数据集的结果聚合和可视化
- 支持自定义句子分析
- 生成多checkpoint的趋势图表

**使用：**
```bash
python eval_dynamic.py \
  --checkpoint_paths \
    /path/to/iter0/policy \
    /path/to/iter45/policy \
  --max_eval_samples 500 \
  --output_dir orz_dynamic_log
```

### 2. **eval_dynamic_debug.py** - 快速调试版本

轻量级评估系统，用于快速迭代和测试。

**主要特点：**
- 仅处理少量样本（默认2个）
- 较短的生成长度（256而非8000）
- 详细的学习动态输出
- API与eval_dynamic.py兼容

**使用：**
```bash
python eval_dynamic_debug.py
```

### 3. **extracted_get_batch_logps.py** - 核心分析引擎

学习动态计算的基础函数库。

**核心函数：**
- `_get_batch_logps()`: 计算6大指标（out_token, out_argmax, A_norm等）
- `analyze_learning_dynamics()`: 高层分析接口，支持多样本类型对比

## 学习动态指标详解

### 5个核心指标

| 指标 | 英文名 | 含义 | 解读 |
|------|--------|------|------|
| 能量 | prob_energy | 1-标签概率均值 | 越小表示模型越确定 |
| 差距 | prob_gap2_mean | 预测与真实标签的L2距离 | 越小表示预测越准确 |
| A_norm | A_norm | 输出激活的Frobenius范数 | 越大表示样本影响力越大 |
| 真实log概率 | out_token | 所有真实token的log概率和 | 越接近0表示模型越有信心 |
| 最可能log概率 | out_argmax | 最可能token的log概率和 | 衡量模型的预测倾向 |

### 计算原理

```
核心分解:
L(x,y) = ||A_o||_F * (1 - p_y) + other_terms

其中:
- ||A_o||_F: 输出激活的Frobenius范数
- (1 - p_y): 真实标签的缺失概率（拉力能量）
- 此分解将损失分解为"影响力"与"不确定性"
```

## 快速开始

### 基础使用

```python
import asyncio
from eval_dynamic import Evaluator, EvaluatorConfig

# 配置
config = EvaluatorConfig(
    model_path="/path/to/checkpoint",
    eval_prompt_data=["data/eval_data/math500.json"],
    max_eval_samples=100,
    enable_visualization=True
)

# 运行
evaluator = Evaluator(config=config)
results = asyncio.run(evaluator.eval())
evaluator.cleanup()

# 查看结果
print(f"正确率: {results.get('math500/accuracy', 0):.2%}")
```

### 多Checkpoint趋势分析

```python
# 在eval_dynamic.py中修改checkpoint_paths
checkpoint_path_list = [
    "/path/to/iter0/policy",
    "/path/to/iter45/policy",
    "/path/to/iter90/policy",
]

# 运行会自动：
# 1. 评估每个checkpoint
# 2. 聚合结果
# 3. 生成趋势图表（对比3类样本的5个指标）
```

### 自定义句子分析

```python
# 创建custom_sentence.json:
{
  "prompt": "What is 2+2?",
  "response": "The answer is 4."
}

# 配置
config = EvaluatorConfig(
    model_path="/path/to/checkpoint",
    custom_sentence_files=["custom_sentence.json"],
    verbose_learning_dynamics=True
)

evaluator = Evaluator(config=config)
results = asyncio.run(evaluator.eval())
# 结果中包含自定义句子的LD指标
```

## 数据格式

### 输入数据格式

```json
[
  {
    "prompt": "问题描述...",
    "answer": "\\boxed{答案}"
  }
]
```

### 输出格式（JSONL）

```json
{
  "prompt": "问题描述...",
  "output": "模型生成的文本...",
  "final_answer": "\\boxed{提取的答案}",
  "answer": "\\boxed{真实答案}",
  "iscorrect": true,
  "ld_metrics": {
    "out_token": -142.5,
    "out_argmax": -115.3,
    "A_norm": 12.1,
    "prob_gap2_mean": 0.54,
    "prob_energy": 0.72
  }
}
```

## 可视化输出

### 对比柱状图（ld_comparison_*.png）
- X轴：5个指标
- Y轴：指标值
- 3组柱子：正确/错误/随机采样
- 包含误差棒（标准差）

### 趋势折线图（ld_trends_*.png）
- X轴：iteration编号
- Y轴：指标值
- 3条线：正确/错误/随机采样
- 为每个数据集生成2x3网格

## 配置参数

```python
@dataclass
class EvaluatorConfig:
    # 模型配置
    model_path: str                      # 模型路径
    tokenizer_path: Optional[str] = None # tokenizer路径

    # 生成参数
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    generate_max_len: int = 8000

    # 数据配置
    eval_prompt_data: List[str] = [...]  # 数据集列表
    prompt_max_len: int = 2048
    max_eval_samples: Optional[int] = None

    # 输出配置
    output_dir: str = "orz_dynamic_log"
    save_detailed_results: bool = True

    # 可视化
    enable_visualization: bool = True
    custom_sentence_files: Optional[List[str]] = None
    verbose_learning_dynamics: bool = False
```

## 系统要求

```
Python >= 3.8
torch >= 1.10
transformers >= 4.20
numpy
matplotlib
loguru
ray
```

**推荐：**
- CUDA >= 11.0（GPU推理）
- 足够的GPU显存（7B模型需要~16GB）
- 足够的磁盘空间（日志和结果）

## 故障排除

### 显存不足
```python
# 解决方案：
config = EvaluatorConfig(
    model_path=...,
    max_eval_samples=50,    # 减少样本数
    generate_max_len=2000,  # 减少生成长度
)
```

### 数值不稳定
```
函数已包含4层clamping防止溢出：
- Layer 1: prob_norm_sum.clamp(max=200)
- Layer 2: prob_norm2_mean.clamp(max=0.3)
- Layer 3: A_norm_inner.clamp(max=50000)
- Layer 4: 检测和警告
```

### 内存泄漏
```python
# 确保调用cleanup()
evaluator.cleanup()

# 循环评估时定期清理
torch.cuda.empty_cache()
gc.collect()
```

## 文件说明

| 文件 | 功能 |
|------|------|
| eval_dynamic.py | 完整评估系统 |
| eval_dynamic_debug.py | 快速调试版本 |
| extracted_get_batch_logps.py | 核心分析引擎 |
| dataset/eval_dataset.py | 数据集加载和处理 |
| orz/ppo/deepspeed_strategy.py | 策略配置 |
| orz/ppo/tools/math_utils.py | 数学验证工具 |

## 许可证

