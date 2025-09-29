"""
Utility functions for preference learning and batch metrics computation.
These functions are extracted to be reusable across different libraries and projects.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Tuple, Optional


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    计算DPO/IPO偏好损失函数

    功能：基于策略模型和参考模型的对数概率计算偏好学习损失

    输入参数：
        policy_chosen_logps: 策略模型对于选中回答的对数概率，维度：(batch_size,)
        policy_rejected_logps: 策略模型对于拒绝回答的对数概率，维度：(batch_size,)
        reference_chosen_logps: 参考模型对于选中回答的对数概率，维度：(batch_size,)
        reference_rejected_logps: 参考模型对于拒绝回答的对数概率，维度：(batch_size,)
        beta: DPO损失的温度参数，通常在0.1到0.5之间，beta趋于0时忽略参考模型
        label_smoothing: 标签平滑参数，假设偏好有噪声的保守性参数，默认0.0
        ipo: 是否使用IPO损失而不是DPO损失，默认False
        reference_free: 是否忽略提供的参考模型，使用隐式等概率参考模型，默认False

    输出：
        返回三元组 (losses, chosen_rewards, rejected_rewards)
        - losses: 每个样本的DPO损失，维度：(batch_size,)
        - chosen_rewards: 选中回答的奖励值，维度：(batch_size,)
        - rejected_rewards: 拒绝回答的奖励值，维度：(batch_size,)

    用法示例：
        losses, chosen_rewards, rejected_rewards = preference_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            beta=0.1
        )
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def simple_dynamic_analysis_for_batch(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False
) -> Tuple[torch.FloatTensor, Tuple]:
    """
    计算给定logits下标签的对数概率及详细指标

    功能：根据模型logits计算目标标签的对数概率，并计算多种学习动态相关指标

    输入参数：
        logits: 模型未归一化的logits输出，维度：(batch_size, sequence_length, vocab_size)
        labels: 目标标签序列，-100的token会被忽略，维度：(batch_size, sequence_length)
        average_log_prob: 是否返回平均对数概率，False时返回累加和，默认False

    输出：
        返回二元组 (log_probs, additional_metrics)
        - log_probs: 对数概率值，维度：(batch_size,)
        - additional_metrics: 额外指标元组，包含：
          * argmax_logps: 最大概率token的对数概率，维度：(batch_size,)
          * except_argmax_logps: 除最大概率外的对数概率，维度：(batch_size,)
          * A_norm: 概率向量的范数指标，维度：(batch_size,)
          * prob_gap2_mean: 概率分布与真实标签的L2距离，维度：(batch_size,)
          * prob_energy: 概率能量指标，维度：(batch_size,)
          * labels_argmax: 每个位置的最大概率token，维度：(batch_size, sequence_length-1)

    用法示例：
        log_probs, metrics = simple_dynamic_analysis_for_batch(model_logits, target_labels)
        argmax_logps, except_argmax, A_norm, gap_norm, energy, argmax_tokens = metrics
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    logprob_logits = logits.log_softmax(-1)
    V = logprob_logits.shape[-1]
    per_token_logps = torch.gather(logprob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # --------- Observe the argmax for each token
    labels_argmax = torch.argmax(logits, dim=-1)  # [B, M], argmax p(y*|chi_u^+/-)
    per_token_logps_argmax = torch.gather(logprob_logits, dim=2, index=labels_argmax.unsqueeze(2)).squeeze(2)

    # ------ 2024-11-15 get all other metrics, e.g., expect_argmax, |A_o|_F, |p-e|_2
    prob_logits = logits.softmax(-1) # prob version of logits, [B, M, V], easy to get underflow, take care!!!
    # --------- expect_argmax, should be [B, M]
    per_token_prob_argmax = torch.exp(per_token_logps_argmax) #torch.gather(prob_logits, dim=2, index=labels_argmax.unsqueeze(2)).squeeze(2) #[B, M]
    per_token_prob_exceptargmax =  torch.ones_like(per_token_prob_argmax)* loss_mask - per_token_prob_argmax* loss_mask #[B, M]
    per_token_logp_exceptargmax = torch.log(per_token_prob_exceptargmax + 1e-100)
    # --------- |A_o|_F, should be [B, 1]
    prob_norm = torch.linalg.vector_norm(prob_logits, ord=2, dim=-1) # [B, M, V] -> [B, M], doing the same thing with previous line
    prob_norm = prob_norm * loss_mask # [B, M], all other dims are zeros
    prob_norm2_mean = torch.square(prob_norm.sum(-1) / loss_mask.sum(-1)) # [B, M] -> [B, 1]
    A_norm = torch.sqrt(V*prob_norm2_mean + (V-2)*torch.ones_like(prob_norm2_mean))  #[B, 1], align with the shape of all other metrics
    # ---------- |pi-e|_2, or
    e_oht = torch.nn.functional.one_hot(labels, num_classes=V) # [B, M, V]
    prob_gap_norm = torch.linalg.vector_norm(prob_logits - e_oht, ord=2, dim=-1) # [B, M, V] -> [B, M]
    prob_gap_norm = prob_gap_norm * loss_mask
    prob_gap2_mean = prob_gap_norm.sum(-1) / loss_mask.sum(-1)
    # --------- (p_label - 1), only the pull-up energy
    prob_label = torch.gather(prob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    prob_label_gap = torch.ones_like(prob_label) - prob_label # [B,M]
    prob_energy = (prob_label_gap*loss_mask).sum(-1) / loss_mask.sum(-1)

    out_token  = (per_token_logps * loss_mask).sum(-1)  #[B, 1]
    out_argmax = (per_token_logps_argmax * loss_mask).sum(-1)
    out_except_argmax = (per_token_logp_exceptargmax * loss_mask).sum(-1)

    if average_log_prob:
        return out_token / loss_mask.sum(-1), (out_argmax / loss_mask.sum(-1), out_except_argmax / loss_mask.sum(-1), A_norm, prob_gap2_mean, prob_energy, labels_argmax)
    else:
        return out_token, (out_argmax, out_except_argmax, A_norm, prob_gap2_mean, prob_energy, labels_argmax)


def detail_dynamic_analysis_for_batch(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    ignore_index: int = -100,
    average_log_prob: bool = False,
    compute_detailed_metrics: bool = True,
    shift_labels: bool = True
) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]]:
    """
    计算token对数概率及详细的学习动态指标（通用版本）

    功能：这是一个高度通用的函数，用于计算语言模型的对数概率和各种学习动态指标，
          可用于分析模型的训练过程、概率分布特性等。

    输入参数：
        logits: 模型未归一化的logits输出，维度：(batch_size, sequence_length, vocab_size)
        labels: 目标标签序列，维度：(batch_size, sequence_length)
        ignore_index: 忽略的标签值（如padding token），默认-100
        average_log_prob: 是否返回平均对数概率而非累加和，默认False
        compute_detailed_metrics: 是否计算详细指标（会增加计算开销），默认True
        shift_labels: 是否自动处理标签位移（用于自回归生成），默认True

    输出：
        如果compute_detailed_metrics=False：
            - log_probs: 对数概率值，维度：(batch_size,)
        如果compute_detailed_metrics=True：
            - log_probs: 对数概率值，维度：(batch_size,)
            - metrics: 详细指标字典，包含：
                * argmax_logps: 最大概率token的对数概率，维度：(batch_size,)
                * except_argmax_logps: 除最大概率外的对数概率，维度：(batch_size,)
                * prob_vector_norm: 概率向量的L2范数，维度：(batch_size,)
                * prob_label_distance: 概率分布与真实标签的L2距离，维度：(batch_size,)
                * prob_energy: 概率"拉升"能量（1-p_label的均值），维度：(batch_size,)
                * max_prob_tokens: 每个位置的最大概率token ID，维度：(batch_size, effective_seq_len)
                * effective_token_count: 每个样本的有效token数量，维度：(batch_size,)
                * entropy: 每个样本的平均熵，维度：(batch_size,)
                * confidence: 每个样本的平均置信度（最大概率），维度：(batch_size,)

    用法示例：
        # 基本用法：只计算对数概率
        log_probs = detail_dynamic_analysis_for_batch(
            logits, labels, compute_detailed_metrics=False
        )

        # 详细分析：计算所有指标
        log_probs, metrics = detail_dynamic_analysis_for_batch(
            logits, labels, compute_detailed_metrics=True
        )
        print(f"平均熵: {metrics['entropy']}")
        print(f"平均置信度: {metrics['confidence']}")

        # 用于BERT等非生成模型（不需要位移）
        log_probs, metrics = detail_dynamic_analysis_for_batch(
            logits, labels, shift_labels=False
        )
    """
    assert logits.shape[:-1] == labels.shape, f"Shape mismatch: logits {logits.shape[:-1]} vs labels {labels.shape}"

    # 处理标签位移（用于自回归生成任务）
    if shift_labels:
        # 对于生成任务，预测下一个token，所以labels向前位移一位
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

    # 创建mask，忽略指定的index
    loss_mask = (labels != ignore_index)
    effective_seq_len = labels.shape[1]
    vocab_size = logits.shape[-1]

    # 将忽略的标签替换为0（dummy值），后面会用mask忽略
    masked_labels = labels.clone()
    masked_labels[labels == ignore_index] = 0

    # 计算对数概率和概率
    log_probs_all = logits.log_softmax(dim=-1)  # [B, M, V]
    probs_all = logits.softmax(dim=-1)  # [B, M, V]

    # 获取目标标签的对数概率
    per_token_logps = torch.gather(
        log_probs_all, dim=2, index=masked_labels.unsqueeze(2)
    ).squeeze(2)  # [B, M]

    # 计算累加/平均对数概率
    masked_logps = per_token_logps * loss_mask
    token_counts = loss_mask.sum(dim=-1)  # [B]

    if average_log_prob:
        # 避免除零
        log_probs = masked_logps.sum(dim=-1) / torch.clamp(token_counts, min=1)
    else:
        log_probs = masked_logps.sum(dim=-1)

    if not compute_detailed_metrics:
        return log_probs

    # ========== 计算详细指标 ==========
    metrics = {}

    # 1. 最大概率token相关指标
    max_prob_tokens = torch.argmax(logits, dim=-1)  # [B, M]
    per_token_logps_argmax = torch.gather(
        log_probs_all, dim=2, index=max_prob_tokens.unsqueeze(2)
    ).squeeze(2)  # [B, M]

    masked_argmax_logps = per_token_logps_argmax * loss_mask
    argmax_logps = masked_argmax_logps.sum(dim=-1) / torch.clamp(token_counts, min=1) if average_log_prob else masked_argmax_logps.sum(dim=-1)

    # 2. 除最大概率外的对数概率（这个指标有点特殊，保持原始计算方式）
    per_token_prob_argmax = torch.exp(per_token_logps_argmax)  # [B, M]
    per_token_prob_except_argmax = torch.ones_like(per_token_prob_argmax) * loss_mask - per_token_prob_argmax * loss_mask
    per_token_logp_except_argmax = torch.log(per_token_prob_except_argmax + 1e-12)  # 增加数值稳定性

    masked_except_argmax = per_token_logp_except_argmax * loss_mask
    except_argmax_logps = masked_except_argmax.sum(dim=-1) / torch.clamp(token_counts, min=1) if average_log_prob else masked_except_argmax.sum(dim=-1)

    # 3. 概率向量范数
    prob_norms_per_token = torch.linalg.vector_norm(probs_all, ord=2, dim=-1)  # [B, M]
    masked_prob_norms = prob_norms_per_token * loss_mask
    prob_norm_mean = masked_prob_norms.sum(dim=-1) / torch.clamp(token_counts, min=1)
    # 归一化的范数（考虑词汇表大小）
    prob_vector_norm = torch.sqrt(vocab_size * prob_norm_mean.square() + (vocab_size - 2))

    # 4. 概率分布与真实标签的L2距离
    labels_onehot = torch.nn.functional.one_hot(masked_labels, num_classes=vocab_size).float()  # [B, M, V]
    prob_label_diff = probs_all - labels_onehot  # [B, M, V]
    prob_gap_norms = torch.linalg.vector_norm(prob_label_diff, ord=2, dim=-1)  # [B, M]
    masked_gap_norms = prob_gap_norms * loss_mask
    prob_label_distance = masked_gap_norms.sum(dim=-1) / torch.clamp(token_counts, min=1)

    # 5. 概率"拉升"能量：衡量模型对正确标签的信心
    per_token_target_probs = torch.gather(
        probs_all, dim=2, index=masked_labels.unsqueeze(2)
    ).squeeze(2)  # [B, M]
    per_token_energy = (1.0 - per_token_target_probs) * loss_mask  # [B, M]
    prob_energy = per_token_energy.sum(dim=-1) / torch.clamp(token_counts, min=1)

    # 6. 熵（衡量不确定性）
    per_token_entropy = -(probs_all * log_probs_all).sum(dim=-1)  # [B, M]
    masked_entropy = per_token_entropy * loss_mask
    entropy = masked_entropy.sum(dim=-1) / torch.clamp(token_counts, min=1)

    # 7. 置信度（最大概率的平均值）
    max_probs_per_token = torch.max(probs_all, dim=-1)[0]  # [B, M]
    masked_max_probs = max_probs_per_token * loss_mask
    confidence = masked_max_probs.sum(dim=-1) / torch.clamp(token_counts, min=1)

    # 组装指标字典
    metrics = {
        'argmax_logps': argmax_logps,
        'except_argmax_logps': except_argmax_logps,
        'prob_vector_norm': prob_vector_norm,
        'prob_label_distance': prob_label_distance,
        'prob_energy': prob_energy,
        'max_prob_tokens': max_prob_tokens,
        'effective_token_count': token_counts.float(),
        'entropy': entropy,
        'confidence': confidence,
    }

    return log_probs, metrics


def demo_test():
    """测试所有函数的demo"""
    print("=" * 60)
    print("preference_utils.py 函数测试演示")
    print("=" * 60)

    # Demo 1: preference_loss 函数
    print("\n1. 测试 preference_loss 函数:")
    print("-" * 40)

    batch_size = 3
    # 生成模拟数据
    policy_chosen_logps = torch.tensor([-1.2, -1.5, -0.8])
    policy_rejected_logps = torch.tensor([-2.1, -2.3, -1.9])
    reference_chosen_logps = torch.tensor([-1.4, -1.6, -1.0])
    reference_rejected_logps = torch.tensor([-2.0, -2.2, -1.8])

    print(f"策略模型选中回答logps: {policy_chosen_logps}")
    print(f"策略模型拒绝回答logps: {policy_rejected_logps}")

    # 测试DPO损失
    losses, chosen_rewards, rejected_rewards = preference_loss(
        policy_chosen_logps, policy_rejected_logps,
        reference_chosen_logps, reference_rejected_logps,
        beta=0.1
    )
    print(f"DPO损失: {losses}")
    print(f"选中奖励: {chosen_rewards}")
    print(f"拒绝奖励: {rejected_rewards}")
    print(f"平均损失: {losses.mean():.4f}")

    # Demo 2: simple_dynamic_analysis_for_batch 函数
    print("\n2. 测试 simple_dynamic_analysis_for_batch 函数:")
    print("-" * 40)

    batch_size = 2
    seq_len = 5
    vocab_size = 100

    # 生成随机logits和标签
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, 0] = -100  # 忽略第一个token

    print(f"输入形状 - logits: {logits.shape}, labels: {labels.shape}")

    log_probs, metrics = simple_dynamic_analysis_for_batch(logits, labels)
    argmax_logps, except_argmax, A_norm, gap_norm, energy, argmax_tokens = metrics

    print(f"对数概率: {log_probs}")
    print(f"argmax对数概率: {argmax_logps}")
    print(f"概率向量范数: {A_norm}")
    print(f"概率gap范数: {gap_norm}")
    print(f"概率能量: {energy}")

    # Demo 3: detail_dynamic_analysis_for_batch 函数
    print("\n3. 测试 detail_dynamic_analysis_for_batch 函数:")
    print("-" * 40)

    # 基本用法
    log_probs_basic = detail_dynamic_analysis_for_batch(
        logits, labels, compute_detailed_metrics=False
    )
    print(f"基本对数概率: {log_probs_basic}")

    # 详细指标
    log_probs, detailed_metrics = detail_dynamic_analysis_for_batch(
        logits, labels, compute_detailed_metrics=True
    )
    print(f"详细对数概率: {log_probs}")
    print(f"熵: {detailed_metrics['entropy']}")
    print(f"置信度: {detailed_metrics['confidence']}")
    print(f"有效token数: {detailed_metrics['effective_token_count']}")

    # Demo 4: 实际应用场景
    print("\n4. DPO训练场景模拟:")
    print("-" * 40)

    # 模拟chosen和rejected回答
    chosen_logits = torch.randn(2, 6, 50) * 1.2
    rejected_logits = torch.randn(2, 6, 50) * 1.0
    ref_chosen_logits = torch.randn(2, 6, 50)
    ref_rejected_logits = torch.randn(2, 6, 50)

    chosen_labels = torch.randint(0, 50, (2, 6))
    rejected_labels = torch.randint(0, 50, (2, 6))
    chosen_labels[:, 0] = -100
    rejected_labels[:, 0] = -100

    # 计算对数概率
    policy_chosen = detail_dynamic_analysis_for_batch(chosen_logits, chosen_labels, compute_detailed_metrics=False)
    policy_rejected = detail_dynamic_analysis_for_batch(rejected_logits, rejected_labels, compute_detailed_metrics=False)
    ref_chosen = detail_dynamic_analysis_for_batch(ref_chosen_logits, chosen_labels, compute_detailed_metrics=False)
    ref_rejected = detail_dynamic_analysis_for_batch(ref_rejected_logits, rejected_labels, compute_detailed_metrics=False)

    # 计算DPO损失
    final_losses, final_chosen_rewards, final_rejected_rewards = preference_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )

    print(f"最终DPO损失: {final_losses}")
    print(f"奖励差值: {final_chosen_rewards - final_rejected_rewards}")
    print(f"平均损失: {final_losses.mean():.4f}")

    print("\n" + "=" * 60)
    print("所有测试完成!")


if __name__ == "__main__":
    # 运行测试demo
    demo_test()