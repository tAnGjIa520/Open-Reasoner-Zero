# Preference Utils 偏好学习工具库 - 使用说明

这个工具库提供了用于偏好学习（DPO/IPO）和语言模型学习动态分析的核心函数。主要包含三个核心函数，用于计算偏好损失、批次指标分析等。

## 核心函数概览

| 函数名 | 主要用途 | 输入维度 | 输出内容 |
|--------|---------|---------|---------|
| `preference_loss` | DPO/IPO偏好学习损失计算 | (batch_size,) 标量 | 损失值 + 奖励值 |
| `simple_dynamic_analysis_for_batch` | 学习动态分析（6个指标） | (B,L,V) logits矩阵 | 对数概率 + 6个指标 |
| `detail_dynamic_analysis_for_batch` | 通用分析工具（9个指标） | (B,L,V) logits矩阵 | 对数概率 + 9个指标 |

## 核心函数详解

### 1. `preference_loss` - DPO/IPO偏好损失计算

#### 功能描述
计算Direct Preference Optimization (DPO) 或 Identity Preference Optimization (IPO) 的偏好损失函数，这是训练语言模型遵循人类偏好的核心算法。

#### 输入参数格式
```python
preference_loss(
    policy_chosen_logps,      # torch.FloatTensor, 维度: (batch_size,)
    policy_rejected_logps,    # torch.FloatTensor, 维度: (batch_size,)
    reference_chosen_logps,   # torch.FloatTensor, 维度: (batch_size,)
    reference_rejected_logps, # torch.FloatTensor, 维度: (batch_size,)
    beta=0.1,                # float, 温度参数，通常0.1-0.5
    label_smoothing=0.0,     # float, 标签平滑，0.0-0.1
    ipo=False,               # bool, 是否使用IPO损失
    reference_free=False     # bool, 是否忽略参考模型
)
```

#### 输出格式
```python
losses, chosen_rewards, rejected_rewards = preference_loss(...)
# losses: (batch_size,) - 每个样本的损失值
# chosen_rewards: (batch_size,) - 选中回答的奖励分数
# rejected_rewards: (batch_size,) - 拒绝回答的奖励分数
```

#### 使用示例
```python
import torch
from preference_utils import preference_loss

# 准备数据 (batch_size=4)
policy_chosen = torch.tensor([-2.1, -1.8, -2.5, -1.9])
policy_rejected = torch.tensor([-3.2, -2.9, -3.1, -2.8])
ref_chosen = torch.tensor([-2.0, -1.9, -2.4, -2.0])
ref_rejected = torch.tensor([-3.0, -2.8, -3.0, -2.7])

# 计算DPO损失
losses, chosen_rewards, rejected_rewards = preference_loss(
    policy_chosen, policy_rejected, ref_chosen, ref_rejected,
    beta=0.1, label_smoothing=0.0
)

print(f"损失: {losses}")
print(f"选中奖励: {chosen_rewards}")
print(f"拒绝奖励: {rejected_rewards}")
```

---

### 2. `simple_dynamic_analysis_for_batch` - 批次对数概率及详细指标计算

#### 功能描述
根据模型logits计算目标标签的对数概率，并计算多种学习动态相关的详细指标，用于分析模型训练过程。

#### 输入参数格式
```python
simple_dynamic_analysis_for_batch(
    logits,                  # torch.FloatTensor, 维度: (batch_size, seq_len, vocab_size)
    labels,                  # torch.LongTensor, 维度: (batch_size, seq_len)
    average_log_prob=False   # bool, 是否返回平均值而非累加和
)
```

#### 重要说明：-100 掩码机制
- **labels中的-100值会被自动忽略**，这些位置不参与损失计算
- 通常用于忽略padding token、prompt部分等不需要监督的位置
- 函数内部会自动处理-100值，将其替换为dummy值0，但通过mask忽略

#### 输出格式
```python
log_probs, additional_metrics = simple_dynamic_analysis_for_batch(...)

# log_probs: (batch_size,) - 目标标签的对数概率
# additional_metrics: 包含6个元素的元组
#   [0] argmax_logps: (batch_size,) - 最大概率token的对数概率
#   [1] except_argmax_logps: (batch_size,) - 除最大概率外的对数概率
#   [2] A_norm: (batch_size,) - 概率向量范数指标
#   [3] prob_gap2_mean: (batch_size,) - 概率分布与真实标签的L2距离
#   [4] prob_energy: (batch_size,) - 概率"拉升"能量指标
#   [5] labels_argmax: (batch_size, seq_len-1) - 每个位置的最大概率token
```

#### 使用示例
```python
import torch
from preference_utils import simple_dynamic_analysis_for_batch

# 模拟数据
batch_size, seq_len, vocab_size = 2, 5, 1000
logits = torch.randn(batch_size, seq_len, vocab_size)

# 创建标签，-100表示忽略的位置
labels = torch.tensor([
    [1, 234, 567, -100, -100],  # 前3个token有效
    [89, 345, 678, 910, -100]   # 前4个token有效
])

# 计算对数概率和指标
log_probs, metrics = simple_dynamic_analysis_for_batch(logits, labels, average_log_prob=True)

argmax_logps, except_argmax, A_norm, gap_norm, energy, argmax_tokens = metrics

print(f"平均对数概率: {log_probs}")
print(f"概率能量: {energy}")
print(f"概率范数: {A_norm}")
```

---

### 3. `detail_dynamic_analysis_for_batch` - 通用token概率和指标计算

#### 功能描述
这是一个高度通用的函数，提供了最全面的学习动态指标计算，包括熵、置信度等高级指标，适用于深入的模型分析。

#### 输入参数格式
```python
detail_dynamic_analysis_for_batch(
    logits,                      # torch.FloatTensor, (batch_size, seq_len, vocab_size)
    labels,                      # torch.LongTensor, (batch_size, seq_len)
    ignore_index=-100,           # int, 忽略的标签值
    average_log_prob=False,      # bool, 返回平均值还是累加和
    compute_detailed_metrics=True, # bool, 是否计算详细指标
    shift_labels=True            # bool, 是否处理标签位移(用于生成任务)
)
```

#### 特殊参数说明
- **ignore_index=-100**: 与上面函数一致，-100位置会被忽略
- **shift_labels=True**: 适用于GPT等自回归生成模型，自动处理标签位移（预测下一个token）
- **shift_labels=False**: 适用于BERT等填空式模型或分类任务，不进行标签位移
- **compute_detailed_metrics**: 控制是否计算详细指标，False时只返回对数概率，节省计算资源

#### 输出格式
```python
# 简单模式 (compute_detailed_metrics=False)
log_probs = detail_dynamic_analysis_for_batch(logits, labels, compute_detailed_metrics=False)

# 详细模式 (compute_detailed_metrics=True)
log_probs, metrics = detail_dynamic_analysis_for_batch(logits, labels)

# metrics字典包含:
metrics = {
    'argmax_logps': torch.FloatTensor,           # (batch_size,) - 最大概率token对数概率
    'except_argmax_logps': torch.FloatTensor,    # (batch_size,) - 除最大概率外的对数概率
    'prob_vector_norm': torch.FloatTensor,       # (batch_size,) - 概率向量L2范数
    'prob_label_distance': torch.FloatTensor,    # (batch_size,) - 概率与标签距离
    'prob_energy': torch.FloatTensor,            # (batch_size,) - 概率拉升能量
    'max_prob_tokens': torch.LongTensor,         # (batch_size, seq_len) - 最大概率token
    'effective_token_count': torch.FloatTensor,  # (batch_size,) - 有效token数量
    'entropy': torch.FloatTensor,                # (batch_size,) - 平均熵
    'confidence': torch.FloatTensor,             # (batch_size,) - 平均置信度
}
```

#### 使用示例
```python
from preference_utils import detail_dynamic_analysis_for_batch
import torch

# 模拟数据
logits = torch.randn(2, 10, 5000)  # 2个样本，10个token，5000词汇
labels = torch.randint(0, 5000, (2, 10))
labels[:, :3] = -100  # 前3个位置忽略 (prompt部分)

# 基础用法：只要对数概率
log_probs = detail_dynamic_analysis_for_batch(
    logits, labels, compute_detailed_metrics=False
)

# 详细分析：所有指标
log_probs, metrics = detail_dynamic_analysis_for_batch(
    logits, labels,
    average_log_prob=True,  # 返回平均值
    compute_detailed_metrics=True
)

print(f"平均对数概率: {log_probs}")
print(f"平均熵: {metrics['entropy']}")
print(f"平均置信度: {metrics['confidence']}")
print(f"有效token数: {metrics['effective_token_count']}")

# 用于BERT等非生成模型
log_probs, metrics = detail_dynamic_analysis_for_batch(
    logits, labels,
    shift_labels=False,  # 不进行标签位移
    compute_detailed_metrics=True
)
```

---

## 重要使用注意事项

### 1. 掩码机制 (-100)
- **所有函数都使用-100作为掩码值**
- -100位置的token不参与损失计算和指标统计
- 通常用于：
  - Padding token
  - Prompt部分（不需要模型生成的部分）
  - 特殊token（如system message）

### 2. 数据格式要求
- 所有tensor必须在同一设备上（CPU或GPU）
- batch中的序列长度可以不同，函数内部会自动padding
- 确保attention_mask正确设置，0表示padding位置

### 3. 内存优化建议
- 对于大模型，建议使用较小的batch_size
- 评估时可以使用`torch.no_grad()`减少内存使用
- `compute_detailed_metrics=False`可以节省计算资源

### 4. 指标解释
- **entropy**: 越高表示模型越不确定
- **confidence**: 越高表示模型越自信
- **prob_energy**: 衡量模型对正确答案的"拉升"程度
- **prob_vector_norm**: 概率分布的集中程度

### 5. 函数对比与选择指南

#### simple_dynamic_analysis_for_batch vs detail_dynamic_analysis_for_batch

| 特性 | simple_dynamic_analysis_for_batch | detail_dynamic_analysis_for_batch |
|------|----------------|--------------------------------|
| **设计目的** | 专门的学习动态分析 | 通用的token概率计算器 |
| **指标数量** | 6个指标 | 9个指标 |
| **返回格式** | 元组格式 | 字典格式（更易使用） |
| **配置选项** | 较少 | 丰富（支持不同模型类型） |
| **独有指标** | except_argmax_logps, max_prob_tokens | entropy, confidence, effective_token_count |
| **推荐使用** | 兼容老代码 | 新项目首选 |

#### 选择建议
- **新项目**: 推荐使用 `detail_dynamic_analysis_for_batch`
- **学习动态分析**: 两者都可以，新项目推荐后者
- **快速计算**: `detail_dynamic_analysis_for_batch` 设置 `compute_detailed_metrics=False`
- **DPO训练**: 必须使用 `preference_loss`

### 6. 常见用法模式

```python
# 模式1: DPO训练完整流程
# 步骤1: 计算策略模型和参考模型的对数概率
policy_chosen_logps = detail_dynamic_analysis_for_batch(
    policy_logits_chosen, chosen_labels, compute_detailed_metrics=False
)
policy_rejected_logps = detail_dynamic_analysis_for_batch(
    policy_logits_rejected, rejected_labels, compute_detailed_metrics=False
)
reference_chosen_logps = detail_dynamic_analysis_for_batch(
    reference_logits_chosen, chosen_labels, compute_detailed_metrics=False
)
reference_rejected_logps = detail_dynamic_analysis_for_batch(
    reference_logits_rejected, rejected_labels, compute_detailed_metrics=False
)

# 步骤2: 计算DPO损失
losses, chosen_rewards, rejected_rewards = preference_loss(
    policy_chosen_logps, policy_rejected_logps,
    reference_chosen_logps, reference_rejected_logps,
    beta=0.1
)
loss = losses.mean()
loss.backward()

# 模式2: 深度模型分析
with torch.no_grad():
    log_probs, metrics = detail_dynamic_analysis_for_batch(
        model_logits, target_labels,
        compute_detailed_metrics=True,
        average_log_prob=True
    )
    print(f"模型熵: {metrics['entropy'].mean():.4f}")
    print(f"置信度: {metrics['confidence'].mean():.4f}")
    print(f"概率能量: {metrics['prob_energy'].mean():.4f}")

# 模式3: 快速对数概率计算
log_probs = detail_dynamic_analysis_for_batch(
    logits, labels, compute_detailed_metrics=False
)

# 模式4: 不同模型类型的适配
# GPT等生成模型
gpt_logps, gpt_metrics = detail_dynamic_analysis_for_batch(
    gpt_logits, labels, shift_labels=True  # 默认值
)

# BERT等填空模型
bert_logps, bert_metrics = detail_dynamic_analysis_for_batch(
    bert_logits, labels, shift_labels=False
)
```

### 7. 测试和验证

本库提供了完整的测试demo，运行以下命令即可验证所有函数：

```bash
cd /path/to/your/project
python src/preference_utils.py
```

这将运行内置的 `demo_test()` 函数，展示所有三个核心函数的使用示例和输出结果。

这个工具库为深度分析语言模型的学习动态提供了全面的支持，可以帮助研究者理解模型训练过程中的概率分布变化、收敛特性等重要信息。