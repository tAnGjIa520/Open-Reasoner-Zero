"""
Qwen2.5-7B base model + ppo (Single GPU version)

running command in 1 node with 1 GPU:
directly run `python -m playground.orz_7b_ppo_jericho_1013_1gpu` is fine
"""


import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from omegaconf.listconfig import ListConfig

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExpConfig
from playground.orz_7b_ppo_jericho_1013 import PPOExp

file_name = f"{os.path.splitext(os.path.basename(__file__))[0]}"

executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    total_num_nodes: int = 1  # 单GPU模式

    # resource related settings
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = True  # 开启内存优化

    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = "/mnt/shared-storage-user/tangjia/orz/Open-Reasoner-Zero/model/models--Qwen--Qwen2.5-7B/snapshots/e25af2efae60472008fbeaf5fb7c4274a87f78d4"  # 使用7B模型
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"jericho_his10_orz_20251013_ckpt_1gpu/{file_name}"
    save_path: str = f"jericho_his10_orz_20251013_ckpt_1gpu/{file_name}"
    tensorboard_log_dir: str = f"jericho_his10_orz_20251013_logs_1gpu/{file_name}"

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    prompt_data: ListConfig = ListConfig(
        [
            "data/jericho_dataset_his10_4games_1.8k_20251013_instruct.json",
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [
            "data/eval_data/eval_jericho_dataset_his10_4games_1.8k_20251013_instruct.json",
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json",
        ]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # ppo related settings
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 2048
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    num_episodes: int = 20
    rollout_batch_size: int = 32  # 单GPU降低批大小
    n_samples_per_prompt: int = 16  # 单GPU降低采样数
    micro_rollout_batch_size: int = 2

    policy_update_steps: int = 1
    critic_update_steps: int = 12
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = 0.0

    use_kl_loss: bool = False
    use_kl_estimator_k3: bool = True

    enable_eval: bool = True
    eval_interval: int = 15

    # generate related settings
    packing_max_len: int = 16384
    generate_max_len: int = 8000
    max_len: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])

    # grpo related settings
    use_grpo: bool = False

    gpu_memory_utilization: float = 0.8  # 单GPU可适当提高内存利用率

    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0


if __name__ == "__main__":
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())
