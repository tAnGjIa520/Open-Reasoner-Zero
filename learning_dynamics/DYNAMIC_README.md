# ORZ Dynamic - 学习动态评估系统

ORZ Dynamic 是一个用于评估大型语言模型（LLM）学习动态的独立模块。该系统通过分析模型生成过程中的内部激活和概率分布，提供深度的学习动态洞察，包括模型确定性、能量流动和预测偏差等关键指标。

## 参考资源

本项目基于以下研究工作：

- **论文：** [Understanding LLM Learning Dynamics](https://arxiv.org/pdf/2407.10490)
- **原始仓库：** [Learning_dynamics_LLM](https://github.com/Joshua-Ren/Learning_dynamics_LLM)


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

| 指标 | 英文名 | 计算公式 | 含义 | 解读 |
|------|--------|--------|------|------|
| 能量 | prob_energy | $\frac{1}{M}\sum_{t=1}^{M} (1 - p_{\text{label},t})$ | 真实标签的缺失概率均值 | 越小表示模型越确定 |
| 差距 | prob_gap2_mean | $\frac{1}{M}\sum_{t=1}^{M} \|p_t - e_t\|_2$ | 预测分布与真实标签的L2距离 | 越小表示预测越准确 |
| A_norm | A_norm | $\|p\|_F = \sqrt{\sum_{t=1}^{M}\sum_{v=1}^{V} p_{t,v}^2}$ | 输出概率分布的Frobenius范数 | 越大表示样本影响力越大 |
| 真实log概率 | out_token | $\sum_{t=1}^{M} \log(p_{\text{label},t})$ | 所有真实token的log概率和 | 越接近0表示模型越有信心 |
| 最可能log概率 | out_argmax | $\sum_{t=1}^{M} \log(p_{\text{argmax},t})$ | 最可能token的log概率和 | 衡量模型的预测倾向 |

**符号说明：**
- $M$ = 非padding的token数量
- $V$ = 词汇表大小
- $t$ = 时间步（序列位置）索引，$t \in [1, M]$
- $v$ = 词汇索引，$v \in [1, V]$
- $p = \text{softmax}(\text{logits})$，预测的概率分布，形状为 $(M, V)$
- $p_{t,v}$ = 在时间步 $t$ 处词汇 $v$ 的概率
- $p_{\text{label},t}$ = 真实标签在时间步 $t$ 的概率，即 $p_t[\text{label\_id}_t]$
- $p_{\text{argmax},t}$ = 在时间步 $t$ 概率最高的token的概率，即 $\max_v p_{t,v}$
- $e_t = \text{one\_hot}(\text{label}_t)$，真实标签在时间步 $t$ 的one-hot向量
- $\text{loss\_mask}$ = $(labels \neq -100)$ 的布尔掩码，用于过滤padding位置

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

## 数据集构建方法

评估系统支持多种数据集，每个数据集都包含三类样本用于对比分析学习动态：

### MATH500 数据集

从MATH500数据集测试集中随机抽取500个问题，作为OOD（Out-of-Distribution）探测集。

**难点**：没有标准答案，需要模型自己生成正确和错误答案进行对比。

**三类样本**：
- **y_correct_math500**: 模型自己回答的正确答案（n条）
- **y_wrong_math500**: 模型自己回答的错误答案（n条）
- **y_random_math500**: 与回答长度相似的随机Token序列，用于测试模型对无意义内容的抑制能力

### AIME2024 数据集

美国数学邀请赛2024年试卷数据集。

**三类样本**：
- **y_correct_AIME2024**: 模型自己回答的正确答案（n条）
- **y_wrong_AIME2024**: 模型自己回答的错误答案（n条）
- **y_random_AIME2024**: 与回答长度相似的随机Token序列

### AIME2025 数据集

美国数学邀请赛2025年试卷数据集。

**三类样本**：
- **y_correct_AIME2025**: 模型自己回答的正确答案（n条）
- **y_wrong_AIME2025**: 模型自己回答的错误答案（n条）
- **y_random_AIME2025**: 与回答长度相似的随机Token序列

### GPQA Diamond 数据集

高难度通用问答数据集。

**三类样本**：
- **y_correct_GPQD**: 模型自己回答的正确答案（n条）
- **y_wrong_GPQD**: 模型自己回答的错误答案（n条）
- **y_random_GPQD**: 与回答长度相似的随机Token序列

### Jericho 数据集

序列决策任务数据集，基于强化学习中的奖励（reward）进行样本选择。

**构建方法**：
- 生成若干组回答序列
- 根据环境反馈计算每组回答的reward

**两类样本**（注：无random类型）：
- **y_correct_jericho**: Reward最高的n个回答（正确/最优决策路径）
- **y_wrong_jericho**: Reward最低的n个回答（错误/次优决策路径）


### 强化学习中对 label 当中 CoT 缺失的难点

在数学推理任务中计算Learning Dynamics时，存在两种不同的标签范围选择方式，它们在RL场景中产生不同的信号：

#### 方式一：完整响应序列（包含CoT推理过程）

**定义**：
- **Label范围**：prompt + CoT推理过程 + 最终答案（如 "\\boxed{42}"）
- **样本标记**：根据最终答案是否正确，将整个生成序列标记为y_correct或y_wrong
- **LD计算**：对整个序列计算5个核心指标

**特点**：
- ✅ 可以分析推理过程中的学习动态演变
- ✅ 能反映"思维链"在解题过程中的不确定性
- ❌ 不同的CoT推理路径可能导致相同答案，难以精确区分
- ❌ 计算量较大（序列更长）

#### 方式二：仅最终答案（CoT缺失）

**定义**：
- **Label范围**：仅最终答案token部分（如 "\\boxed{42}"），不包含推理过程
- **样本标记**：给定prompt，直接计算预测最终答案的概率
- **LD计算**：仅在答案部分计算LD指标

**特点**：
- ✅ 聚焦于最终结果的确定性（result-focused）
- ✅ 计算高效，指标更精炼
- ❌ **丢失推理过程的动态信息** - 无法分析"如何推出答案"
- ❌ **Label信息部分缺失** - reward信号仅来自最终结果，中间步骤无直接监督

#### 与论文方法的对比

原论文 [[Understanding LLM Learning Dynamics](https://arxiv.org/pdf/2407.10490)] 中的分析基于：
- **监督学习场景**：完整的标签序列（包括所有中间步骤）
- **完整的真值信号**：每一步都有明确的ground truth

而在RL强化学习场景中：
- **奖励函数仅反馈最终结果**：环境只告诉模型"答案对还是错"
- **中间步骤无显式监督**：CoT过程的正确性无法从reward中直接获得
- **这导致label的"部分缺失"问题**：LD指标难以解释推理过程的贡献

#### 实践选择建议

| 目标 | 推荐方式 | 原因 |
|------|--------|------|
| 分析**整体推理能力** | 方式一（完整序列） | 能捕捉推理过程的不确定性 |
| 分析**答案生成确定性** | 方式二（仅答案） | 更接近RL的实际奖励信号 |
| 诊断**推理错误来源** | 方式一（完整序列） | 可分析CoT和答案的分离度 |
| 计算**最终结果可靠度** | 方式二（仅答案） | 精炼指标，易于解释 |

**特点**：
- 评估指标直接反映模型在长序列决策中的学习动态
- Reward分数的差异可用于评估模型对决策路径的区分能力

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

## 文件说明

| 文件路径 | 功能 |
|--------|------|
| `eval_dynamic.py` | 完整的学习动态评估引擎（生产环境） |
| `eval_dynamic_debug.py` | 轻量级快速调试版本 |
| `extracted_get_batch_logps.py` | 核心分析引擎，计算学习动态指标 |
| `../dataset/eval_dataset.py` | 数据集加载和处理模块 |
| `../orz/ppo/dataset.py` | PPO训练数据集处理 |
| `../orz/ppo/deepspeed_strategy.py` | DeepSpeed分布式训练策略配置 |
| `../orz/ppo/tools/math_utils.py` | 数学答案验证工具（MATH、GSM8K等） |
| `../orz/exp_engine/` | 实验执行引擎（推理加速、分布式） |

## 许可证

内部使用

---

**最后更新：2024-11-10**
