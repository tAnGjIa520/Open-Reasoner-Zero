"""
提取的 _get_batch_logps 函数 - 用于计算 Learning Dynamics 的核心函数

这个文件包含了从 trainers.py 提取的 _get_batch_logps 函数，
以及对其输入输出格式的详细分析。
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Any
import numpy as np
import warnings


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    用于计算learning dynamics的核心函数。计算每个样本的日志概率，以及多个学习动态指标。

    ================================================================================
    【输入格式详细说明】
    ================================================================================

    Args:
        logits: 模型的未归一化逻辑值。形状: (batch_size, sequence_length, vocab_size)

                具体示例：
                - batch_size: 通常是1用于fine_evaluation，或多个样本用于training
                - sequence_length: 实际生成的token数量，例如 256, 512, 1024等
                - vocab_size: 词汇表大小，取决于使用的tokenizer
                  例如：Qwen使用的tokenizer的vocab_size通常是128256

                数据类型：torch.FloatTensor (float32)
                设备：应该在GPU上 (cuda)

                示例shape: (1, 256, 128256)
                含义：1个批次，256个token位置，128256个词汇表大小

        labels: 用于计算日志概率的标签（真实token ID）。
                值为-100的token会被忽略（这是padding mask）。
                形状: (batch_size, sequence_length)

                具体说明：
                - 包含真实的token ID (0 到 vocab_size-1)
                - -100 表示应该被忽略的位置（通常是padding位置）
                - 实际上，这个函数内部会做移位处理：
                  labels = labels[:, 1:].clone()  # 去掉第一个token
                  logits = logits[:, :-1, :]      # 去掉最后一个logit
                  这是因为在LLM中，我们预测的是下一个token

                数据类型：torch.LongTensor (int64)
                设备：应该在GPU上 (cuda)，与logits一致

                示例shape: (1, 256)
                示例值: [101, 32, 456, -100, -100, ...]

                【重要】loss_mask = (labels != -100)
                - 只有不是-100的位置会被用于计算损失
                - 如果某个样本中有很多-100，那么实际计算的token数会很少

        average_log_prob: 布尔值，是否计算平均日志概率
                - True: 返回平均日志概率（除以非mask token的数量）
                - False: 返回求和的日志概率

                在这个实现中，即使是True，返回的也是单个样本的平均值，
                而不是整个批次的平均值。

    ================================================================================
    【输出格式详细说明】
    ================================================================================

    Returns (when average_log_prob=False):
        返回值是一个元组：(out_token, (out_argmax, out_except_argmax, A_norm, prob_gap2_mean, prob_energy, labels_argmax))

        out_token: 形状 (batch_size,) - 每个样本的总日志概率
                  示例: tensor([-156.5, -142.3, ...])
                  含义：对于这个样本，所有真实标签token的log概率之和
                  通常是负数（因为概率小于1，log为负）

        元组中的其他返回值：

            out_argmax: (batch_size,) - 每个位置最可能token的总日志概率
                       示例: tensor([-120.3, -115.7, ...])
                       含义：对于这个样本，每个位置概率最高的token的log概率之和
                       通常比out_token更大（接近0）

            out_except_argmax: (batch_size,) - 非argmax token的总日志概率
                               示例: tensor([-350.2, -320.5, ...])
                               含义：所有非最可能token的log概率之和
                               通常更加负数

            A_norm: (batch_size,) - 输出向量范数 |A_o|_F，用于learning dynamics
                   形状实际上是 (batch_size, 1) 或会squeeze到 (batch_size,)
                   示例: tensor([12.5, 11.3, ...])
                   含义：softmax输出的二范数（Frobenius norm）
                   这是learning dynamics分解中的关键量

            prob_gap2_mean: (batch_size,) - 预测概率与标签的差距 |p-e|_2
                           示例: tensor([0.5, 0.6, ...])
                           含义：对于每个位置，计算 ||p_logits - one_hot(label)||_2
                           然后求和并除以非mask位置的数量
                           这反映了模型对真实标签的"确信度"偏离

            prob_energy: (batch_size,) - 标签概率缺口 (1 - p_label)，"拉力能量"
                        示例: tensor([0.8, 0.7, ...])
                        含义：对于真实标签，计算 (1 - p_label)，即"缺失的概率"
                        这代表了模型需要"拉起"真实标签的"能量"
                        值越大说明模型越不确定

            labels_argmax: (batch_size, sequence_length) - 每个位置概率最高的token ID
                          示例: tensor([[101, 32, 456, ...], [...]])
                          含义：对于每个位置，argmax的token ID
                          用于分析模型最可能预测了什么

    Returns (when average_log_prob=True):
        返回值是一个元组：(out_token/norm, (out_argmax/norm, ...))
        所有的值都被除以了非mask位置的数量，得到平均值

        示例：
        out_token / loss_mask.sum(-1)
        → 如果有256个有效token，[-156.5] / 256 ≈ [-0.61]

    ================================================================================
    【实际调用示例】
    ================================================================================

    假设我们有一个mini-batch：

    batch_size = 1
    seq_len = 256
    vocab_size = 128256

    # 模型的输出（未归一化的logits）
    logits = torch.randn(1, 256, 128256, device='cuda')

    # 真实的token标签
    labels = torch.randint(0, 128256, (1, 256), device='cuda')
    labels[0, 200:] = -100  # 将最后56个位置标记为padding

    # 调用函数
    out_token, (out_argmax, out_except_argmax, A_norm, prob_gap2_mean, prob_energy, labels_argmax) = \
        _get_batch_logps(logits, labels, average_log_prob=False)

    # 结果：
    # out_token: tensor([-142.5], device='cuda')        - 总log概率
    # out_argmax: tensor([-115.3], device='cuda')       - argmax log概率
    # out_except_argmax: tensor([-325.8], device='cuda') - 非argmax log概率
    # A_norm: tensor([12.1], device='cuda')             - 输出范数
    # prob_gap2_mean: tensor([0.54], device='cuda')     - 概率差距
    # prob_energy: tensor([0.72], device='cuda')        - 能量（缺失概率）
    # labels_argmax: tensor([[101, 32, 456, ...]])      - 最可能token的ID

    ================================================================================
    【关键计算步骤】
    ================================================================================

    1. 掩码处理：
       loss_mask = (labels != -100)  # 标记哪些位置不是padding

    2. 概率计算：
       logprob_logits = logits.log_softmax(-1)  # 计算log概率分布
       prob_logits = logits.softmax(-1)         # 计算概率分布

    3. 真实标签的概率：
       per_token_logps = torch.gather(logprob_logits, dim=2, index=labels)
       # 为每个位置取出真实标签的log概率

    4. Argmax的概率：
       labels_argmax = torch.argmax(logits, dim=-1)  # 找最可能的token
       per_token_logps_argmax = torch.gather(logprob_logits, dim=2, index=labels_argmax)

    5. Learning Dynamics 指标：
       A_norm = ||logits.softmax(-1)||_F  # 输出向量的范数
       prob_gap = ||softmax(logits) - one_hot(label)||_2  # 预测与真实的差距
       energy = 1 - p_label  # 需要"拉动"真实标签的能量

    6. 求和（只在非mask位置）：
       out_token = (per_token_logps * loss_mask).sum(-1)
       # 只对有效的token位置求和

    """
    assert logits.shape[:-1] == labels.shape

    # 移位处理（标准的LLM预测方式：用前面的token预测后面的token）
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
    #breakpoint()
    # ------ 2024-11-15 get all other metrics, e.g., expect_argmax, |A_o|_F, |p-e|_2
    prob_logits = logits.softmax(-1) # prob version of logits, [B, M, V], easy to get underflow, take care!!!
        # --------- expect_argmax, should be [B, M]
    per_token_prob_argmax = torch.exp(per_token_logps_argmax) #torch.gather(prob_logits, dim=2, index=labels_argmax.unsqueeze(2)).squeeze(2) #[B, M]
    per_token_prob_exceptargmax =  torch.ones_like(per_token_prob_argmax)* loss_mask - per_token_prob_argmax* loss_mask #[B, M]
    per_token_logp_exceptargmax = torch.log(per_token_prob_exceptargmax + 1e-100)
        # --------- |A_o|_F, should be [B] (Frobenius norm of probability distribution)
    # Calculate Frobenius norm: sqrt(sum of all squared elements)
    A_norm = torch.sqrt((prob_logits ** 2).sum(dim=(1, 2)))  # [B, M, V] -> [B]

        # ---------- |pi-e|_2, or
    #breakpoint()
    e_oht = torch.nn.functional.one_hot(labels, num_classes=V) # [B, M, V]
    prob_gap_norm = torch.linalg.vector_norm(prob_logits - e_oht, ord=2, dim=-1) # [B, M, V] -> [B, M]
    prob_gap_norm = prob_gap_norm * loss_mask
    prob_gap2_mean = prob_gap_norm.sum(-1) / loss_mask.sum(-1)
        # --------- (p_label - 1), only the pull-up energy
    prob_label = torch.gather(prob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    prob_label_gap = torch.ones_like(prob_label) - prob_label # [B,M]
    prob_energy = (prob_label_gap*loss_mask).sum(-1) / loss_mask.sum(-1)
    #breakpoint()

    out_token  = (per_token_logps * loss_mask).sum(-1)  #[B, 1]
    out_argmax = (per_token_logps_argmax * loss_mask).sum(-1)
    out_except_argmax = (per_token_logp_exceptargmax * loss_mask).sum(-1)

    if average_log_prob:
        return out_token / loss_mask.sum(-1), (out_argmax / loss_mask.sum(-1), out_except_argmax / loss_mask.sum(-1), A_norm, prob_gap2_mean, prob_energy, labels_argmax)
    else:
        return out_token, (out_argmax, out_except_argmax, A_norm, prob_gap2_mean, prob_energy, labels_argmax)


def analyze_learning_dynamics(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    sample_types: Dict[str, torch.LongTensor] = None,
    tokenizer = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """分析模型的学习动态（Learning Dynamics）

    这个函数提取了 trainers.py 中评估阶段的核心逻辑，
    用于详细分析模型在不同样本类型上的学习动态。

    ================================================================================
    【功能说明】
    ================================================================================

    对于给定的输入，计算 6 个关键的 learning dynamics 指标：
    1. out_token: 真实标签的总log概率
    2. out_argmax: 最可能token的总log概率
    3. A_norm: 输出向量范数（学习影响力）
    4. prob_gap2_mean: 预测与真实标签的差距
    5. prob_energy: "拉力能量"（需要改正的强度）
    6. labels_argmax: 每个位置最可能的token ID

    ================================================================================
    【参数说明】
    ================================================================================

    Args:
        logits: 模型的输出 logits，形状 (batch_size, seq_len, vocab_size)
                - 通常从 model(input_ids, attention_mask).logits 获取

        labels: 真实标签，形状 (batch_size, seq_len)
                - 包含真实token ID 或 -100（padding mask）

        sample_types: 字典，包含多种样本类型的 logits 和 labels
                     例如：{
                         'chosen': {'logits': ..., 'labels': ...},
                         'rejected': {'logits': ..., 'labels': ...},
                         'irr_train': {'logits': ..., 'labels': ...},
                     }
                     如果提供，会对每种样本类型分别计算指标

        tokenizer: tokenizer 对象，用于解码token（可选）
                  如果提供，可以显示 token 级别的分析

        verbose: 是否打印详细的分析结果

    ================================================================================
    【返回值说明】
    ================================================================================

    返回一个字典，包含以下信息：

    {
        'summary': {
            'out_token': float,              # 真实标签log概率
            'out_argmax': float,             # 最可能token log概率
            'A_norm': float,                 # 输出向量范数
            'prob_gap2_mean': float,         # 预测-标签差距
            'prob_energy': float,            # 拉力能量
            'valid_tokens': int,             # 有效token数
            'argmax_match_rate': float,      # argmax与真实标签匹配率
        },
        'per_sample': {
            'out_token': ndarray,            # 每个样本的值
            'out_argmax': ndarray,
            'A_norm': ndarray,
            'prob_gap2_mean': ndarray,
            'prob_energy': ndarray,
        },
        'sample_types': {  # 如果提供了 sample_types
            'chosen': {...},
            'rejected': {...},
            ...
        }
    }

    ================================================================================
    【使用示例】
    ================================================================================

    # 单个样本类型的分析
    logits = model(input_ids, attention_mask).logits
    labels = batch['labels']

    ld_analysis = analyze_learning_dynamics(
        logits=logits,
        labels=labels,
        tokenizer=tokenizer,
        verbose=True
    )

    print(f"平均能量: {ld_analysis['summary']['prob_energy']:.4f}")
    print(f"平均差距: {ld_analysis['summary']['prob_gap2_mean']:.4f}")

    # 多个样本类型的分析
    sample_types = {
        'chosen': {'logits': chosen_logits, 'labels': chosen_labels},
        'rejected': {'logits': rejected_logits, 'labels': rejected_labels},
    }

    ld_analysis = analyze_learning_dynamics(
        logits=logits,
        labels=labels,
        sample_types=sample_types,
        verbose=True
    )

    # 比较不同样本类型的学习动态
    chosen_energy = ld_analysis['sample_types']['chosen']['summary']['prob_energy']
    rejected_energy = ld_analysis['sample_types']['rejected']['summary']['prob_energy']
    print(f"chosen 能量: {chosen_energy:.4f}, rejected 能量: {rejected_energy:.4f}")
    """

    # 计算主要的学习动态指标
    out_token, (out_argmax, out_except_argmax, A_norm, prob_gap2_mean, prob_energy, labels_argmax) = \
        _get_batch_logps(logits, labels, average_log_prob=False)

    # 计算 argmax 与真实标签的匹配率
    labels_shifted = labels[:, 1:].clone()
    labels_shifted[labels_shifted == -100] = -1
    logits_shifted = logits[:, :-1, :]
    valid_mask = (labels_shifted != -100)
    argmax_match = (labels_argmax == labels_shifted).float()
    argmax_match_rate = (argmax_match * valid_mask).sum() / valid_mask.sum()

    valid_tokens = valid_mask.sum().item()

    # 转换为 numpy，便于分析
    summary = {
        'out_token': out_token.mean().item(),
        'out_argmax': out_argmax.mean().item(),
        'A_norm': A_norm.mean().item(),
        'prob_gap2_mean': prob_gap2_mean.mean().item(),
        'prob_energy': prob_energy.mean().item(),
        'valid_tokens': valid_tokens,
        'argmax_match_rate': argmax_match_rate.item(),
    }

    per_sample = {
        'out_token': out_token.detach().cpu().numpy(),
        'out_argmax': out_argmax.detach().cpu().numpy(),
        'A_norm': A_norm.squeeze().detach().cpu().numpy(),
        'prob_gap2_mean': prob_gap2_mean.detach().cpu().numpy(),
        'prob_energy': prob_energy.detach().cpu().numpy(),
    }

    result = {
        'summary': summary,
        'per_sample': per_sample,
    }

    # 如果提供了多个样本类型，分别计算
    if sample_types is not None:
        sample_types_analysis = {}
        for sample_type_name, sample_data in sample_types.items():
            type_logits = sample_data.get('logits')
            type_labels = sample_data.get('labels')

            if type_logits is not None and type_labels is not None:
                type_result = analyze_learning_dynamics(
                    logits=type_logits,
                    labels=type_labels,
                    sample_types=None,  # 不递归嵌套
                    tokenizer=tokenizer,
                    verbose=False
                )
                sample_types_analysis[sample_type_name] = type_result

        result['sample_types'] = sample_types_analysis

    # 打印详细结果
    if verbose:
        print("\n" + "=" * 80)
        print("【Learning Dynamics 分析结果】")
        print("=" * 80)
        print(f"\n【摘要统计】")
        print(f"  有效token数: {summary['valid_tokens']}")
        print(f"  真实标签log概率: {summary['out_token']:.4f}")
        print(f"  最可能token log概率: {summary['out_argmax']:.4f}")
        print(f"  输出向量范数 (A_norm): {summary['A_norm']:.4f}")
        print(f"  预测-标签差距 (|p-e|_2): {summary['prob_gap2_mean']:.4f}")
        print(f"  拉力能量 (1-p_label): {summary['prob_energy']:.4f}")
        print(f"  argmax匹配率: {summary['argmax_match_rate']:.2%}")

        # 解释
        print(f"\n【指标解释】")
        print(f"  • out_token 越接近0，说明模型对真实标签越有信心")
        print(f"  • A_norm 越大，说明该样本对模型的影响力越大")
        print(f"  • prob_gap2_mean 越小，说明预测与真实标签越接近")
        print(f"  • prob_energy 越小，说明模型越确定")
        print(f"  • argmax_match_rate 高说明模型的最可能预测与真实标签一致")

        # 分析多个样本类型
        if 'sample_types' in result and result['sample_types']:
            print(f"\n【多样本类型对比】")
            for sample_type_name, type_analysis in result['sample_types'].items():
                type_summary = type_analysis['summary']
                print(f"\n  {sample_type_name}:")
                print(f"    - 能量: {type_summary['prob_energy']:.4f}")
                print(f"    - 差距: {type_summary['prob_gap2_mean']:.4f}")
                print(f"    - A_norm: {type_summary['A_norm']:.4f}")
                print(f"    - 匹配率: {type_summary['argmax_match_rate']:.2%}")

        print("\n" + "=" * 80 + "\n")

    return result


if __name__ == "__main__":
    """
    测试代码：演示如何使用 _get_batch_logps 和 analyze_learning_dynamics 函数
    """

    print("\n" + "=" * 80)
    print("【第一部分】低阶 _get_batch_logps 函数测试")
    print("=" * 80)

    # 设置参数
    batch_size = 2
    seq_len = 128
    vocab_size = 50257  # GPT-2的词汇表大小

    # 创建示例数据
    logits = torch.randn(batch_size, seq_len, vocab_size, device='cpu')
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cpu')

    # 添加一些padding标记
    labels[0, 100:] = -100
    labels[1, 80:] = -100

    print("\n输入形状：")
    print(f"  logits: {logits.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  有效token数: {(labels != -100).sum().item()}")

    # 调用 _get_batch_logps
    out_token, (out_argmax, out_except_argmax, A_norm, prob_gap2_mean, prob_energy, labels_argmax) = \
        _get_batch_logps(logits, labels, average_log_prob=False)

    print("\n输出形状和值：")
    print(f"  out_token: {out_token.shape} = {out_token}")
    print(f"  out_argmax: {out_argmax.shape} = {out_argmax}")
    print(f"  out_except_argmax: {out_except_argmax.shape} = {out_except_argmax}")
    print(f"  A_norm: {A_norm.shape} = {A_norm}")
    print(f"  prob_gap2_mean: {prob_gap2_mean.shape} = {prob_gap2_mean}")
    print(f"  prob_energy: {prob_energy.shape} = {prob_energy}")
    print(f"  labels_argmax: {labels_argmax.shape}")

    # 测试平均模式
    print("\n测试 average_log_prob=True:")
    out_token_avg, (out_argmax_avg, out_except_argmax_avg, A_norm_avg, prob_gap2_mean_avg, prob_energy_avg, labels_argmax_avg) = \
        _get_batch_logps(logits, labels, average_log_prob=True)

    print(f"  out_token_avg: {out_token_avg}")
    print(f"  验证: {out_token} / {(labels != -100).sum().item()} ≈ {out_token / (labels != -100).sum().item()}")

    print("\n" + "=" * 80)
    print("【第二部分】高阶 analyze_learning_dynamics 函数测试")
    print("=" * 80)

    # 使用 analyze_learning_dynamics
    ld_analysis = analyze_learning_dynamics(
        logits=logits,
        labels=labels,
        verbose=True
    )

    print("\n【返回值详解】")
    print(f"summary keys: {list(ld_analysis['summary'].keys())}")
    print(f"per_sample keys: {list(ld_analysis['per_sample'].keys())}")

    print("\n" + "=" * 80)
    print("【第三部分】多样本类型对比分析")
    print("=" * 80)

    # 创建多种样本类型
    sample_types = {
        'chosen': {
            'logits': torch.randn(batch_size, seq_len, vocab_size),
            'labels': torch.randint(0, vocab_size, (batch_size, seq_len))
        },
        'rejected': {
            'logits': torch.randn(batch_size, seq_len, vocab_size),
            'labels': torch.randint(0, vocab_size, (batch_size, seq_len))
        },
    }

    # 为 rejected 类型添加 padding
    sample_types['chosen']['labels'][0, 100:] = -100
    sample_types['rejected']['labels'][0, 80:] = -100

    # 分析多个样本类型
    ld_analysis_multi = analyze_learning_dynamics(
        logits=logits,
        labels=labels,
        sample_types=sample_types,
        verbose=True
    )

    print("\n【访问结果示例】")
    print(f"Main - 能量: {ld_analysis_multi['summary']['prob_energy']:.4f}")
    print(f"Chosen - 能量: {ld_analysis_multi['sample_types']['chosen']['summary']['prob_energy']:.4f}")
    print(f"Rejected - 能量: {ld_analysis_multi['sample_types']['rejected']['summary']['prob_energy']:.4f}")

    chosen_energy = ld_analysis_multi['sample_types']['chosen']['summary']['prob_energy']
    rejected_energy = ld_analysis_multi['sample_types']['rejected']['summary']['prob_energy']
    energy_diff = chosen_energy - rejected_energy
    print(f"\n能量差异 (chosen - rejected): {energy_diff:.4f}")
    if energy_diff > 0:
        print(f"  → rejected样本需要更多的学习能量")
    else:
        print(f"  → chosen样本需要更多的学习能量")
