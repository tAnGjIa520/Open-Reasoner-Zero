import asyncio
import json
import math
import os
import random
from functools import partial
from heapq import heapify, heappop, heappush
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union

import ray
import torch
from loguru import logger
from omegaconf.dictconfig import DictConfig
from ray.util.placement_group import PlacementGroup, placement_group
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from orz.ppo.actors import PPORayActorGroup
from orz.ppo.replay_buffer import Experience, NaiveReplayBuffer
from orz.ppo.utils import ORZDeepspeedStrategy as DeepspeedStrategy
from orz.ppo.utils import (
    Timer,
    compute_approx_kl,
    compute_reward,
    get_advantages_and_returns,
    masked_mean,
    normalize_advantages,
)


class RayPPOTrainer:
    """
    基于Ray的PPO训练器类

    用于在分布式环境中训练大语言模型的PPO（Proximal Policy Optimization）算法实现。
    支持多种模型的并置策略（colocate）和分布式训练。
    """
    def __init__(
        self,
        cfg: DictConfig,
        strategy: DeepspeedStrategy,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        vllm_engines=None,
        colocate_pg: Optional[PlacementGroup] = None,
    ):
        """
        初始化PPO训练器

        这是PPO训练器的构造函数，负责初始化所有必要的组件，包括：
        - 配置参数和策略
        - 数据集和数据加载器
        - TensorBoard日志记录
        - 经验回放缓冲区
        - VLLM推理引擎（如果提供）
        - Ray placement group（用于资源调度）

        Args:
            cfg (DictConfig): 训练配置对象，包含所有超参数和设置
            strategy (DeepspeedStrategy): DeepSpeed分布式训练策略
            tokenizer: 分词器，用于文本token化
            train_dataset: 训练数据集，包含prompt-response对
            eval_dataset (optional): 评估数据集，用于模型评估
            vllm_engines (optional): VLLM推理引擎列表，用于高效文本生成
            colocate_pg (optional): Ray placement group，用于并置模式下的资源管理

        Raises:
            ValueError: 当配置参数不一致时
            RuntimeError: 当资源分配失败时
        """
        # 保存配置和基础组件
        self.cfg = cfg
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.vllm_engines = vllm_engines
        self.prompts_dataloader = self.build_dataloader(train_dataset)
        self.colocate_pg = colocate_pg

        # 初始化TensorBoard日志记录器
        self.writer = SummaryWriter(log_dir=self.cfg.tensorboard_log_dir)

        # 初始化经验回放缓冲区
        self.replay_buffer = NaiveReplayBuffer(
            sample_batch_size=self.cfg.micro_train_batch_size,
            limit=0,
            cpu_offload=True,
            packing_samples=True,
        )

    def __del__(self):
        """
        析构函数，清理资源

        在对象被销毁时自动调用，负责：
        - 关闭TensorBoard writer
        - 释放相关资源

        Note:
            这个方法会在Python垃圾回收时自动调用
        """
        self.writer.close()

    async def eval(self):
        """
        模型评估函数

        这是一个抽象方法，需要在子类或用户实验代码中实现具体的评估逻辑。
        评估通常包括：
        - 在验证集上生成回答
        - 计算评估指标（BLEU、ROUGE、准确率等）
        - 记录评估结果到TensorBoard

        该方法会在训练过程中定期调用（根据eval_interval配置）。

        Raises:
            NotImplementedError: 当子类未实现此方法时

        Example:
            async def eval(self):
                # 生成验证样本
                results = await self.generate_on_eval_dataset()
                # 计算指标
                metrics = self.compute_metrics(results)
                # 记录日志
                self.log_eval_metrics(metrics)
        """
        raise NotImplementedError("Eval function should be implemented in user's exp")

    # 会在启动脚本里调用 1
    async def train(self): # note: 主循环
        """
        PPO主要训练循环

        这是PPO训练的核心方法，执行完整的强化学习训练流程。该方法实现了标准的PPO算法，
        包括经验收集、模型更新、评估等步骤。

        训练流程：
        1. 模型初始化和权重同步
           - 初始化VLLM引擎actor组
           - 将策略模型权重同步到VLLM引擎
           - 处理并置模式下的内存管理

        2. 主训练循环（按episode和iteration）
           - 从数据集中采样prompts
           - 通过VLLM引擎生成responses
           - 收集经验数据（Experience对象）
           - 计算优势函数和回报
           - 训练策略模型和价值模型
           - 更新权重到VLLM引擎

        3. 定期操作
           - 模型评估（根据eval_interval）
           - 模型保存（根据save_interval）
           - 参考模型更新（根据update_ref_every_epoch）
           - TensorBoard日志记录

        内存管理特性：
        - 支持colocate_all模式，通过CPU卸载节省GPU内存
        - 自动处理GPU缓存清理
        - 支持模型的动态加载/卸载

        并行化：
        - 支持多GPU训练
        - VLLM引擎并行推理
        - 策略和价值模型可并行或串行训练

        异常处理：
        - 处理空数据情况
        - 自动跳过无效的训练步骤
        - 保证训练的鲁棒性

        Returns:
            None

        Raises:
            RuntimeError: 当模型初始化失败时
            ValueError: 当配置参数不一致时
            torch.cuda.OutOfMemoryError: 当GPU内存不足时

        Note:
            - 该方法是异步的，支持高效的并发操作
            - 训练状态会定期保存，支持断点恢复
            - 所有的训练指标都会记录到TensorBoard
        """
        # 1. 创建rank0策略模型和vllm引擎组，然后将权重广播到vllm引擎
        if self.cfg.colocate_all:
            # 如果启用了并置模式，先将策略模型和VLLM引擎加载到GPU
            await self.policy_model.backload_to_gpu()
            await self._backload_vllm_engines()

        # 初始化VLLM引擎actor组
        await self.policy_model.async_run_method("_init_vllm_engines_actor_group", self.vllm_engines)
        logger.info("Create vllm engine gourps done.")

        # 同步策略模型权重到VLLM引擎，vllm提供一个壳子，推理的时候用
        async with Timer("Sync actor weights to vllm engines"):
            await self._sync_policy_weights_to_vllm()

        if self.cfg.colocate_all:
            # 并置模式下，将策略模型卸载到CPU以节省GPU内存
            async with Timer("Offload policy model to cpu"):
                await self.policy_model.offload_to_cpu()

        # 2. 主要训练循环
        consumed_samples = 0
        # 计算每个episode需要的rollout次数
        num_rollouts_per_episodes = (
            self.num_update_steps_per_episodes  # 每个episode的更新步数
            * self.cfg.train_batch_size         # 训练批次大小
            // self.cfg.max_epochs               # 最大epoch数（对同一批经验重复训练的轮数）
            // self.cfg.rollout_batch_size       # rollout批次大小（每次rollout处理的prompt数）
            // self.cfg.n_samples_per_prompt     # 每个prompt生成的样本数
        )

        # 计算全局步数和起始episode，为了恢复断点
        self.global_step = consumed_samples // self.cfg.rollout_batch_size
        start_episode = consumed_samples // self.cfg.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * self.cfg.rollout_batch_size)

        # 开始主要训练循环
        for episode in range(start_episode, self.cfg.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()), desc=f"Episode [{episode + 1}/{self.cfg.num_episodes}]"
            )
            for iter, rand_prompts in enumerate(self.prompts_dataloader):

                # 1. 如果启用了评估，在指定间隔进行评估
                if self.cfg.enable_eval and (
                    self.global_step % self.cfg.eval_interval == 0 or iter == len(self.prompts_dataloader) - 1
                ):
                    await self.eval()

                # 3. 生成经验数据，计算优势和回报
                await self.make_experience(rand_prompts) ## note:make_experience

                # 检查是否有足够的数据
                if len(self.replay_buffer) <= 0:
                    if self.cfg.colocate_all:
                        # 跳过本次训练，但需要传输权重
                        await self.policy_model.backload_to_gpu()
                        await self._backload_vllm_engines()
                        await self._sync_policy_weights_to_vllm()
                        await self.policy_model.offload_to_cpu()
                    continue

                # 如果启用了优势标准化，对回放缓冲区中的优势进行标准化
                if self.cfg.advantage_normalize:
                    self.replay_buffer = normalize_advantages(self.replay_buffer)

                # 将回放缓冲区序列化保存为jsonl格式
                async with Timer("Dumping replay buffer"):
                    all_replay_buffer_save_path = os.path.join(self.cfg.save_path, "dumped_replay_buffer")
                    os.makedirs(all_replay_buffer_save_path, exist_ok=True)
                    dump_path = os.path.join(all_replay_buffer_save_path, f"iter{self.global_step}_replay_buffer.jsonl")
                    with open(dump_path, "a") as f:
                        logger.info(f"dumping replay buffer to {dump_path}")
                        for item in self.replay_buffer:
                            f.write(json.dumps(item.to_json()) + "\n")

                # 计算策略和价值模型的数据并行节点数
                num_policy_dp_nodes = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
                num_critic_dp_nodes = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node

                # 根据数据并行节点数分割回放缓冲区
                policy_buffers = self.replay_buffer.split_to_n_batches(num_policy_dp_nodes)
                if num_policy_dp_nodes != num_critic_dp_nodes:
                    critic_buffers = self.replay_buffer.split_to_n_batches(num_critic_dp_nodes)
                else:
                    critic_buffers = policy_buffers

                # 4. 训练策略/价值模型
                if self.cfg.colocate_all:
                    # 并置模式：串行训练模型以节省GPU内存
                    if self.critic_model is not None:
                        async with Timer("Critic model training"):
                            await self.critic_model.backload_to_gpu()
                            await self.ppo_local_train_critic(critic_buffers, self.global_step)
                            await self.critic_model.offload_to_cpu()
                    async with Timer("Actor model training"):
                        await self.policy_model.backload_to_gpu()
                        status = await self.ppo_local_train_policy(policy_buffers, self.global_step)
                        await self.policy_model.offload_to_cpu()

                else:
                    # 非并置模式：并行训练模型
                    if self.critic_model is not None:
                        async with Timer("Actor and Critic model training"):
                            status = await asyncio.gather(
                                self.ppo_local_train_policy(policy_buffers, self.global_step),
                                self.ppo_local_train_critic(critic_buffers, self.global_step),
                            )
                            # 清理GPU缓存
                            await asyncio.gather(
                                self.policy_model.async_run_method("empty_cache"),
                                self.critic_model.async_run_method("empty_cache"),
                            )
                            status = status[0]
                    else:
                        async with Timer("Actor model training"):
                            status = await self.ppo_local_train_policy(policy_buffers, self.global_step)
                            await self.policy_model.async_run_method("empty_cache")

                # 清空回放缓冲区
                self.replay_buffer.clear()

                # 5. 记录日志和保存模型
                logger.info(status)
                pbar.update()
                # 记录episode信息到TensorBoard
                self.writer.add_scalar("episode_idx", episode, self.global_step)
                self.global_step += 1

                # 按设定间隔保存模型
                if self.global_step % self.cfg.save_interval == 0:
                    await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                    if self.critic_model is not None:
                        await self.critic_model.async_save_model(self.tokenizer, self.global_step)
                    logger.info("Successfully save model weights, training continue.")

            # 如果配置为每个epoch更新参考模型
            if self.cfg.update_ref_every_epoch:
                await self.policy_model.backload_to_gpu()
                await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                await self.policy_model.offload_to_cpu()
                # 使用当前策略模型的权重更新参考模型
                await asyncio.gather(
                    *self.ref_model.async_init_model_from_pretrained(
                        self.strategy, os.path.join(self.cfg.save_path, f"iter{self.global_step}", "policy")
                    )
                )
                logger.info("Successfully update ref model with policy model, training continue.")

        # 训练完成后保存最终模型
        await self.policy_model.async_save_model(self.tokenizer, self.cfg.num_episodes * len(self.prompts_dataloader))
        logger.info("Successfully save model weights, training done.")

    @torch.no_grad()
    async def make_experience(self, all_inputs: Union[Tuple[str, dict], List[Tuple[str, dict]]], **generate_kwargs):
        """
        生成PPO训练所需的经验数据

        这是PPO算法中的关键方法，负责收集训练所需的经验数据。该方法执行完整的经验收集流程：
        从原始prompts到最终的Experience对象，包含所有PPO训练所需的信息。

        主要步骤：
        1. 数据预处理
           - 扩展prompts（每个prompt生成n_samples_per_prompt个样本）
           - 随机打乱数据顺序
           - 提取额外信息（如任务类型、难度等）

        2. 并行文本生成
           - 将prompts分配给多个VLLM引擎
           - 并行生成responses
           - 支持自定义生成参数（温度、top_p等）

        3. 奖励计算（可选）
           - 如果启用自定义奖励函数，计算额外奖励
           - 支持多种奖励模型

        4. 序列打包
           - 将prompt+response打包成固定长度的张量
           - 创建attention masks标识不同序列边界
           - 记录action数量和序列长度信息

        5. 模型推理
           - 策略模型：计算action log probabilities
           - 参考模型：计算baseline log probabilities
           - 价值模型：计算state values
           - 奖励模型：计算reward scores

        6. 经验对象创建
           - 计算KL散度和其他指标
           - 打包所有信息为Experience对象
           - 计算优势函数和回报

        7. 数据存储
           - 将经验数据添加到replay buffer
           - 记录统计信息到TensorBoard
           - 保存样本到文件（可选）

        Args:
            all_inputs (Union[Tuple[str, dict], List[Tuple[str, dict]]]):
                输入数据列表，每个元素包含：
                - [0]: prompt字符串
                - [1]: 额外信息字典（metadata）
            **generate_kwargs: 生成参数，如：
                - temperature: 采样温度
                - top_p: nucleus采样参数
                - max_new_tokens: 最大生成长度
                - min_new_tokens: 最小生成长度

        Returns:
            None: 经验数据直接添加到self.replay_buffer

        Side Effects:
            - 更新self.replay_buffer
            - 记录指标到TensorBoard
            - 可能触发GPU内存管理操作

        Raises:
            AssertionError: 当生成数据数量不匹配时
            RuntimeError: 当模型推理失败时
            torch.cuda.OutOfMemoryError: 当GPU内存不足时

        Note:
            - 使用@torch.no_grad()装饰器确保推理时不计算梯度
            - 支持colocate模式下的动态内存管理
            - 所有张量操作都考虑了内存效率
        """
        experiences = []
        # 根据每个提示生成多个样本
        all_prompts = sum([[prompt[0]] * self.cfg.n_samples_per_prompt for prompt in all_inputs], [])
        all_extras = sum([[prompt[1]] * self.cfg.n_samples_per_prompt for prompt in all_inputs], [])

        # 将所有提示和额外信息一起随机打乱
        indices = list(range(len(all_prompts)))
        rng = random.Random(42)
        rng.shuffle(indices)
        all_prompts = [all_prompts[i] for i in indices]
        all_extras = [all_extras[i] for i in indices]

        # 1. 生成序列并进行推理，计算价值、对数概率、奖励、KL散度
        # 1.1 通过VLLM引擎生成序列
        outputs = []
        num_vllm_dp_gruops = len(self.vllm_engines)

        async with Timer("Generate sequences via vllm engines"):
            dp_prompt_size = (len(all_prompts) + num_vllm_dp_gruops - 1) // num_vllm_dp_gruops # 每个VLLM数据并行组需要处理的prompts数量，使用的是向上取整的数学技巧。 
            dp_tasks = []
            for dp_rank in range(num_vllm_dp_gruops):
                dp_inputs = all_prompts[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                dp_extras = all_extras[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                # handle last batch has no enough data
                if len(dp_inputs) <= 0:
                    continue
                gen_func = self._get_generate_function(dp_rank)
                dp_tasks.append(self.generate_vllm(gen_func, dp_inputs, extras=dp_extras, **generate_kwargs))

            logger.info("start generation")
            local_responses = await asyncio.gather(*dp_tasks) # 开始收集experience
            outputs.extend(sum(local_responses, []))
            logger.info("generate local rollout batch done")

            # offload vllm engines when colocate all models
            if self.cfg.colocate_all:
                async with Timer("Offload vllm engines to cpu"):
                    await self._offload_vllm_engines()

        # skip when data is not enough
        if len(outputs) <= 0:
            return

        assert len(all_prompts) == len(outputs), "generate objects number must be equal to all inputs number"

        # 1.2 calculate custom rewards if has custom reward function
        if self.cfg.use_compute_reward_fn:
            async with Timer("Calculate custom rewards"):
                dp_tasks = []
                reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
                all_prompts, outputs, custom_rewards = await reward_fn(all_prompts, outputs, all_extras)
                assert len(all_prompts) == len(
                    outputs
                ), "generate objects number after custom reward function must be equal to all inputs number"
        else:
            all_prompts, outputs, custom_rewards = all_prompts, outputs, None

        # empty data
        if len(all_prompts) == 0:
            return

        # 1.3 packing samples
        async with Timer("Packing samples"):
            (
                ret_sequences,
                ret_attention_masks,
                ret_num_actions, # 每个打包批次中，每个对话的响应token数量，response的数量
                ret_packed_seq_lens, # 每个打包批次中，每个对话总的token 的数量
                ret_custom_rewards,# 每个回答产生的reward
            ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                all_prompts, outputs, custom_rewards, self.cfg.packing_max_len
            )
            action_masks = None
        # 1.4 inference and calculate values, log probs, rewards, kl divergence
        async with Timer("Inference and calculate values, log probs, rewards, kl divergence"):
            experiences = await self.inference_and_calculates( # 计算loss log 等等
                ret_sequences,
                ret_attention_masks,
                action_masks,
                ret_num_actions,
                ret_packed_seq_lens,
                ret_custom_rewards,
            )

            logger.info(f"experiences size: {len(experiences)}")

        # 2. visualization generated results example
        vis = self._detokenize(experiences[0].sequences[0][: int(experiences[0].info["total_length"].flatten()[0])])
        self.writer.add_text("generated_sequences", vis, self.global_step)
        self.writer.flush()

        # 3. calculate advantages and returns / along with tensorboard logging
        avg_rewards = 0
        avg_kl = 0
        avg_kl_max = 0
        avg_response_length = 0
        avg_orm_score = 0
        avg_custom_rewards = 0
        avg_advantages = 0
        avg_advantages_abs = 0

        async with Timer("Calculate advantages and returns"):
            adv_tasks = []
            for experience in experiences:
                adv_tasks.append(self._calc_advantages_and_returns(experience))

            for tsk in asyncio.as_completed(adv_tasks):
                experience, metrics = await tsk
                avg_rewards += metrics["avg_rewards"]
                avg_kl += metrics["avg_kl"]
                avg_kl_max += metrics["avg_kl_max"]
                avg_response_length += metrics["avg_response_length"]
                avg_orm_score += metrics["avg_orm_score"]
                avg_custom_rewards += metrics["avg_custom_rewards"]
                avg_advantages += metrics["avg_advantages"]
                avg_advantages_abs += metrics["avg_advantages_abs"]
                self.replay_buffer.append(experience)

        # 4. tensorboard logging
        logger.info(
            f"avg_raw_rewards: {avg_rewards / len(experiences)}, avg_kl: {avg_kl / len(experiences)}, avg_response_length: {avg_response_length / len(experiences)}, avg_orm_score: {avg_orm_score / len(experiences)}, avg_custom_rewards: {avg_custom_rewards / len(experiences)}"
        )
        self.writer.add_scalar("avg_raw_rewards", avg_rewards / len(experiences), self.global_step)
        self.writer.add_scalar("avg_kl", avg_kl / len(experiences), self.global_step)
        self.writer.add_scalar("avg_kl_max", avg_kl_max / len(experiences), self.global_step)
        self.writer.add_scalar("avg_response_length", avg_response_length / len(experiences), self.global_step)
        self.writer.add_scalar("avg_orm_score", avg_orm_score / len(experiences), self.global_step)
        self.writer.add_scalar("avg_custom_rewards", avg_custom_rewards / len(experiences), self.global_step)
        self.writer.add_scalar("avg_raw_advantages", avg_advantages / len(experiences), self.global_step)
        self.writer.add_scalar("avg_raw_advantages_abs", avg_advantages_abs / len(experiences), self.global_step)
        self.writer.flush()

    @torch.no_grad()
    async def inference_and_calculates(
        self,
        sequences_all: List[torch.Tensor],
        attention_mask_all: List[torch.Tensor],
        action_mask_all: Optional[List[torch.Tensor]],
        num_actions_all: Optional[List[int]],
        packed_seq_lens_all: Optional[List[int]],
        custom_rewards_all: Optional[List[torch.Tensor]],
    ):
        """
        多模型并行推理和关键指标计算

        这是PPO训练中的核心推理方法，负责对打包的序列数据进行多模型推理，
        计算PPO算法所需的所有关键指标。该方法通过精心设计的并行化和内存管理，
        实现高效的大规模模型推理。

        推理流程：
        1. 并行计算配置
           - 确定各模型的数据并行组数量
           - 创建微批次推理函数
           - 配置内存管理策略

        2. 价值模型推理（Critic）
           - 计算状态价值estimates
           - 支持colocate模式下的动态加载/卸载
           - 处理不同数据并行配置

        3. 参考模型推理（Reference Policy）
           - 计算baseline action log probabilities
           - 用于KL散度计算
           - 支持actor-ref colocate优化

        4. 奖励模型推理（Reward Model）
           - 计算sequence-level奖励分数
           - 支持多个奖励模型
           - 可选的自定义奖励计算

        5. 策略模型推理（Current Policy）
           - 计算当前策略的action log probabilities
           - 用于策略梯度计算
           - 支持动态内存管理

        6. 关键指标计算
           - KL散度：衡量策略偏移
           - 响应长度统计
           - 奖励信号处理

        7. Experience对象构造
           - 打包所有推理结果
           - 添加元数据信息
           - 准备用于优势函数计算

        内存优化特性：
        - 支持多种colocate模式
        - 自动GPU缓存管理
        - 微批次处理避免OOM
        - 异步并行执行

        Args:
            sequences_all (List[torch.Tensor]): 打包的序列张量列表
                每个张量包含多个prompt+response序列
            attention_mask_all (List[torch.Tensor]): 注意力掩码列表
                标识不同序列的边界和有效token
            action_mask_all (Optional[List[torch.Tensor]]): 动作掩码列表
                标识哪些token是可学习的actions（通常为None）
            num_actions_all (Optional[List[int]]): 每个序列的action数量
                用于区分prompt和response部分
            packed_seq_lens_all (Optional[List[int]]): 打包序列的长度信息
                每个打包批次中各序列的实际长度
            custom_rewards_all (Optional[List[torch.Tensor]]): 自定义奖励
                来自外部奖励函数的额外奖励信号

        Returns:
            List[Experience]: 经验数据对象列表，每个包含：
                - sequences: 序列数据
                - action_log_probs: 策略模型的action对数概率
                - base_log_probs: 参考模型的对数概率
                - values: 价值模型的状态价值估计
                - attention_mask: 注意力掩码
                - kl: KL散度
                - info: 包含各种元数据的字典

        Raises:
            RuntimeError: 当模型推理失败时
            torch.cuda.OutOfMemoryError: 当GPU内存不足时
            AssertionError: 当数据维度不匹配时

        Note:
            - 所有模型推理都在torch.no_grad()上下文中执行
            - 支持不同的colocate策略以优化内存使用
            - 自动处理数据并行和模型并行
            - 推理结果会移动到CPU以节省GPU内存
        """
        num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
        num_critic_dp_groups = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
        num_ref_dp_groups = self.cfg.ref_num_nodes * self.cfg.ref_num_gpus_per_node
        num_reward_dp_groups = self.cfg.reward_num_nodes * self.cfg.reward_num_gpus_per_node

        async def micro_infer_model(num_dps, model_type, sequences, num_actions, attention_mask, packed_seq_lens):
            dp_iterator = self._split_dp_batch(
                (sequences, num_actions, attention_mask, packed_seq_lens),
                num_dps,
            )
            dp_tasks = []
            for dp_rank, (
                micro_sequences,
                micro_num_actions,
                micro_attention_mask,
                micro_packed_seq_lens,
            ) in enumerate(dp_iterator):
                model = self._get_dp_group_models(dp_rank, model_type)

                async def forward_fn(
                    local_model, fwd_sequences, fwd_num_actions, fwd_attention_mask, fwd_packed_seq_lens
                ):
                    return await local_model.forward.remote(
                        sequences=fwd_sequences,
                        num_actions=fwd_num_actions,
                        attention_mask=fwd_attention_mask,
                        packed_seq_lens=fwd_packed_seq_lens,
                    )

                dp_tasks.append(
                    self._split_and_run_micro_batch(
                        partial(forward_fn, model),
                        (micro_sequences, micro_num_actions, micro_attention_mask, micro_packed_seq_lens),
                        self.cfg.micro_forward_batch_size,
                    )
                )
            results = await asyncio.gather(*dp_tasks)
            results = sum(results, [])
            return results

        if action_mask_all is not None:
            num_actions_all = action_mask_all.size(1)

        # calculate critic values
        if self.cfg.colocate_all and self.critic_model is not None:
            await self.critic_model.backload_to_gpu()

        if self.critic_model is not None:
            value_ref = micro_infer_model(
                num_critic_dp_groups,
                "critic_model",
                sequences_all,
                num_actions_all,
                attention_mask_all,
                packed_seq_lens_all,
            )
            values = None
            if self.cfg.colocate_all:
                values = await value_ref
                await self.critic_model.offload_to_cpu()

        # calculate ref log probs
        base_action_log_probs_ref = micro_infer_model(
            num_ref_dp_groups, "ref_model", sequences_all, num_actions_all, attention_mask_all, packed_seq_lens_all
        )
        base_log_probs = None

        # handle colocate critic and reward model
        if self.cfg.colocate_critic_reward and not self.cfg.colocate_all and self.critic_model is not None:
            values = await value_ref
            await self.critic_model.async_run_method("empty_cache")

        # handle colocate actor and ref model
        if self.cfg.colocate_actor_ref or self.cfg.colocate_all:
            base_log_probs = await base_action_log_probs_ref
            await self.ref_model.async_run_method("empty_cache")

        # calculate rewards
        reward_refs = []
        if self.cfg.use_orm_score and self.reward_model:
            reward_refs.append(
                micro_infer_model(
                    num_reward_dp_groups,
                    "reward_model",
                    sequences_all,
                    num_actions_all,
                    attention_mask_all,
                    packed_seq_lens_all,
                )
            )

        if self.cfg.colocate_all:
            rewards = await asyncio.gather(*reward_refs)

        # calculate action log probs
        if self.cfg.colocate_all:
            await self.policy_model.backload_to_gpu()

        action_log_probs_ref = micro_infer_model(
            num_policy_dp_groups,
            "policy_model",
            sequences_all,
            num_actions_all,
            attention_mask_all,
            packed_seq_lens_all,
        )
        action_log_probs = None
        if self.cfg.colocate_all:
            action_log_probs = await action_log_probs_ref
            await self.policy_model.offload_to_cpu()

        # wait all models done
        # if not colocate_actor_ref, then need to gather base_log_probs
        # if not colocate_critic_reward and self.critic_model is not None, then need to gather value
        # reward_refs is always handled at last
        if not self.cfg.colocate_all:
            if not self.cfg.colocate_actor_ref:
                if not self.cfg.colocate_critic_reward and self.critic_model is not None:
                    results = await asyncio.gather(
                        value_ref, base_action_log_probs_ref, action_log_probs_ref, *reward_refs
                    )
                    values, base_log_probs, action_log_probs, rewards = results[0], results[1], results[2], results[3:]
                else:
                    results = await asyncio.gather(base_action_log_probs_ref, action_log_probs_ref, *reward_refs)
                    base_log_probs, action_log_probs, rewards = results[0], results[1], results[2:]
            else:
                if not self.cfg.colocate_critic_reward and self.critic_model is not None:
                    results = await asyncio.gather(value_ref, action_log_probs_ref, *reward_refs)
                    values, action_log_probs, rewards = results[0], results[1], results[2:]
                else:
                    results = await asyncio.gather(action_log_probs_ref, *reward_refs)
                    action_log_probs, rewards = results[0], results[1:]

        r = torch.stack(rewards).sum(dim=0) if len(rewards) > 0 else None
        if not self.cfg.colocate_all:
            empty_cache_tasks = [
                self.policy_model.async_run_method("empty_cache"),
                self.ref_model.async_run_method("empty_cache"),
            ]
            if self.critic_model:
                empty_cache_tasks.append(self.critic_model.async_run_method("empty_cache"))
            if self.reward_model:
                empty_cache_tasks.extend([rm.async_run_method("empty_cache") for rm in self.reward_model])
            await asyncio.gather(*empty_cache_tasks)

        # 6. calculate kl divergence

        experiences = []
        if self.critic_model is not None:
            values = values[: len(sequences_all)]
        base_log_probs = base_log_probs[: len(sequences_all)]
        action_log_probs = action_log_probs[: len(sequences_all)]
        if r is not None:
            r = r[: len(sequences_all)]
        for i in range(len(action_log_probs)):
            response_length = torch.Tensor(num_actions_all[i]).unsqueeze(0)
            total_length = torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0)
            kl = compute_approx_kl(
                action_log_probs[i],
                base_log_probs[i],
                action_mask=None,
                use_kl_estimator_k3=self.cfg.use_kl_estimator_k3,
                use_abs_kl=self.cfg.use_abs_kl,
            )
            kl_max = torch.max(kl.abs(), dim=-1)[0]
            kl_mean = masked_mean(kl, None, dim=-1)
            if r is not None:
                local_reward = r[i]
            else:
                local_reward = None
            info = {
                "kl": kl_mean,
                "kl_max": kl_max,
                "reward": local_reward,
                "custom_rewards": custom_rewards_all[i] if custom_rewards_all is not None else None,
                "response_length": response_length,
                "total_length": total_length,
                "num_actions": num_actions_all[i],
            }
            experiences.append(
                Experience(
                    sequences_all[i],           # 序列数据
                    action_log_probs[i],        # 动作对数概率
                    base_log_probs[i],          # 基准对数概率
                    values[i] if self.critic_model is not None else None,  # 价值估计
                    None,                       # advantages（稍后计算）
                    None,                       # returns（稍后计算）
                    attention_mask_all[i],      # 注意力掩码
                    None,                       # action_mask
                    response_length,            # 响应长度
                    torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0),  # 序列总长度
                    info,                       # 额外信息字典
                    kl,                         # KL散度
                )
            )
        return experiences

    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: Optional[List[dict]] = None,
        **kwargs,
    ) -> List[str | Any]:
        """
        使用VLLM引擎进行高效文本生成

        这是一个封装方法，用于通过VLLM引擎生成高质量的文本响应。
        VLLM是一个高性能的大语言模型推理引擎，专门优化了吞吐量和延迟。

        功能特性：
        - 高吞吐量并行生成
        - 灵活的采样参数配置
        - 自动批处理优化
        - 内存高效的推理

        采样策略支持：
        - 温度采样（temperature）
        - 核采样（nucleus sampling, top_p）
        - Top-k采样
        - 长度控制（min_tokens, max_tokens）

        Args:
            gen_func (Callable): VLLM生成函数
                由_get_generate_function返回的异步生成函数
                该函数封装了特定VLLM引擎的生成逻辑
            prompts (List[str]): 输入提示列表
                需要生成回应的文本提示
            extras (Optional[List[dict]]): 额外元数据信息
                与prompts对应的额外信息，目前未直接使用但保留接口
            **kwargs: 生成控制参数，包括：
                - temperature (float): 采样温度，默认1.0
                - top_p (float): nucleus采样参数，默认1.0
                - top_k (int): top-k采样参数，默认-1（不限制）
                - max_new_tokens (int): 最大生成token数，默认1024
                - min_new_tokens (int): 最小生成token数，默认1
                - skip_special_tokens (bool): 是否跳过特殊token，默认False

        Returns:
            List[str]: 生成的回应文本列表
                与输入prompts一一对应的生成结果

        Raises:
            RuntimeError: 当VLLM引擎生成失败时
            ValueError: 当采样参数无效时
            torch.cuda.OutOfMemoryError: 当GPU内存不足时

        Example:
            prompts = ["What is AI?", "Explain machine learning"]
            responses = await self.generate_vllm(
                gen_func=self._get_generate_function(0),
                prompts=prompts,
                temperature=0.8,
                max_new_tokens=512
            )

        Note:
            - 使用@torch.no_grad()确保推理时不计算梯度
            - SamplingParams会根据kwargs自动配置
            - 生成过程是异步的，支持高效并发
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        responses, _ = await gen_func(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        return responses

    def build_dataloader(self, dataset):
        """
        构建训练数据加载器并计算训练步数

        这个方法负责创建PyTorch DataLoader用于批量加载训练数据，
        同时计算PPO训练所需的关键参数，如每个episode的更新步数和总训练步数。

        计算逻辑：
        - num_update_steps_per_episodes: 每个episode需要的模型更新步数
        - max_steps: 整个训练过程的总步数

        数据加载特性：
        - 随机打乱数据顺序
        - 多进程数据加载（8个worker）
        - 自定义collate_fn处理批数据
        - 灵活的批大小配置

        Args:
            dataset: 训练数据集对象
                应该包含prompt-response对或相关训练数据
                需要实现__len__和__getitem__方法
                需要提供collate_fn方法处理批数据

        Returns:
            DataLoader: 配置好的PyTorch数据加载器
                - batch_size: 由cfg.rollout_batch_size控制
                - shuffle: True，随机打乱数据
                - num_workers: 8，多进程加载
                - collate_fn: 使用dataset.collate_fn

        Side Effects:
            设置以下实例属性：
            - self.num_update_steps_per_episodes: 每episode更新步数
            - self._max_steps: 总训练步数

        计算公式：
            num_update_steps_per_episodes = (
                len(dataset) * n_samples_per_prompt
                // train_batch_size * max_epochs
            )
            max_steps = ceil(num_episodes * num_update_steps_per_episodes)

        Note:
            - 这些计算对于PPO训练的学习率调度和进度跟踪很重要
            - max_steps可用于配置学习率scheduler
            - 数据加载器的设计考虑了内存效率和训练速度
        """
        # 准备数据加载器
        prompts_dataloader = DataLoader(
            dataset, batch_size=self.cfg.rollout_batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=8
        )
        # 计算每个episode的更新步数
        self.num_update_steps_per_episodes = (
            len(dataset) * self.cfg.n_samples_per_prompt // self.cfg.train_batch_size * self.cfg.max_epochs
        )
        # 计算总的最大步数
        max_steps = math.ceil(self.cfg.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps
        return prompts_dataloader

    # 会在启动脚本里调用 1
    async def build_models(self, PolicyRayActor, CriticRayActor, RefRayActor, RewardRayActor=None):
        """
        构建和初始化分布式PPO模型架构

        这是PPO训练器的核心初始化方法，负责创建所有必要的模型组件并配置它们的分布式部署。
        该方法实现了复杂的模型并置（colocate）策略，以最大化GPU内存利用效率。

        模型组件：
        1. Policy Model（策略模型）：当前训练的主模型
        2. Critic Model（价值模型）：估计状态价值的辅助模型
        3. Reference Model（参考模型）：固定的基准策略，用于KL约束
        4. Reward Model（奖励模型）：评估生成质量的打分模型

        并置策略：
        1. colocate_all: 所有模型共享GPU，动态切换
           - 节省GPU内存，适合大模型
           - 需要动态加载/卸载模型
           - 要求所有模型配置一致

        2. colocate_actor_ref: 策略和参考模型共享GPU
           - 平衡内存使用和推理效率
           - 适合中等规模部署

        3. colocate_critic_reward: 价值和奖励模型共享GPU
           - 优化推理模型的内存使用
           - 减少模型间通信开销

        4. 独立部署: 每个模型使用独立GPU
           - 最大并行度和推理速度
           - 需要更多GPU资源

        Ray Placement Groups：
        - 管理多GPU资源分配
        - 确保模型部署在合适的设备上
        - 支持跨节点部署

        模型初始化：
        - 从预训练检查点加载权重
        - 配置分布式训练参数
        - 设置特殊token（如pad_token_id）
        - 根据并置模式选择性卸载到CPU

        Args:
            PolicyRayActor (Type[BasePPORole]): 策略模型的Ray Actor类
                负责策略网络的训练和推理
            CriticRayActor (Type[BasePPORole]): 价值模型的Ray Actor类
                负责状态价值估计
            RefRayActor (Type[BasePPORole]): 参考模型的Ray Actor类
                提供稳定的基准策略
            RewardRayActor (Optional[Type[BasePPORole]]): 奖励模型的Ray Actor类
                用于生成质量评估，可选

        Returns:
            None

        Side Effects:
            设置以下实例属性：
            - self.policy_model: 策略模型actor组
            - self.critic_model: 价值模型actor组
            - self.ref_model: 参考模型actor组
            - self.reward_model: 奖励模型actor组列表

        Raises:
            AssertionError: 当并置模式下的配置不一致时
            RuntimeError: 当模型初始化失败时
            ValueError: 当Actor类型不正确时

        Example:
            await trainer.build_models(
                PolicyRayActor=MyPolicyActor,
                CriticRayActor=MyCriticActor,
                RefRayActor=MyRefActor,
                RewardRayActor=MyRewardActor
            )

        Note:
            - 该方法必须在调用train()之前执行
            - 支持多种GPU内存优化策略
            - 可以处理多个奖励模型的并行部署
            - 模型权重会从指定的预训练路径加载
        """
        cfg = self.cfg
        pg = None

        if cfg.colocate_all:
            assert (
                cfg.actor_num_nodes == cfg.critic_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.critic_num_gpus_per_node
                and cfg.actor_num_nodes == cfg.ref_num_nodes 
                and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                and cfg.actor_num_gpus_per_node == 1
                and cfg.actor_num_nodes == cfg.vllm_num_engines
            ), "num_nodes and num_gpus_per_node must be the same when colocate all models and each actor has only one gpu."
            pg = self.colocate_pg #todo: pg 啥意思

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node, # todo:每个Actor节点GPU数 难道一个actor 可以部署在多个gpu上？
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.2,
            )
            ref_model = PPORayActorGroup(
                cfg.ref_num_nodes,
                cfg.ref_num_gpus_per_node,
                RefRayActor,
                pg=pg,
                num_gpus_per_actor=0.2,
            )
            if cfg.critic_pretrain:
                critic_model = PPORayActorGroup(
                    cfg.critic_num_nodes,
                    cfg.critic_num_gpus_per_node,
                    CriticRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                )
            else:
                critic_model = None

            # multiple reward models
            if RewardRayActor is not None and cfg.reward_pretrain:
                reward_pretrains = cfg.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            cfg.reward_num_nodes,
                            cfg.reward_num_gpus_per_node,
                            RewardRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.2,
                        )
                    )
            else:
                reward_models = None

        else:
            if cfg.colocate_actor_ref:
                assert (
                    cfg.actor_num_nodes == cfg.ref_num_nodes
                    and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

                bundles = [
                    {"GPU": cfg.actor_num_gpus_per_node, "CPU": cfg.actor_num_gpus_per_node}
                    for _ in range(cfg.actor_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
            )
            ref_model = PPORayActorGroup(
                cfg.ref_num_nodes,
                cfg.ref_num_gpus_per_node,
                RefRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )

            # if colocated, create placement group for critic and reward model explicitly.
            pg = None
            if cfg.colocate_critic_reward:
                assert (
                    cfg.critic_num_nodes == cfg.reward_num_nodes
                    and cfg.critic_num_gpus_per_node == cfg.reward_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

                bundles = [
                    {"GPU": cfg.critic_num_gpus_per_node, "CPU": cfg.critic_num_gpus_per_node}
                    for _ in range(cfg.critic_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            if cfg.critic_pretrain:
                critic_model = PPORayActorGroup(
                    cfg.critic_num_nodes,
                    cfg.critic_num_gpus_per_node,
                    CriticRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.75 if pg else 1,
                )
            else:
                critic_model = None

            # multiple reward models
            if RewardRayActor is not None and cfg.reward_pretrain:
                reward_pretrains = cfg.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            cfg.reward_num_nodes,
                            cfg.reward_num_gpus_per_node,
                            RewardRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.25 if pg else 1,
                        )
                    )
            else:
                reward_models = None

        if not cfg.colocate_all:
            refs = []
            refs.extend(ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            refs.extend(policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            if cfg.critic_pretrain:
                refs.extend(critic_model.async_init_model_from_pretrained(self.strategy, cfg.critic_pretrain))
            if cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    refs.extend(reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))
            await asyncio.gather(*refs)
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
        else:
            await asyncio.gather(*ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain)) # note:加载ref模型参数
            await asyncio.gather(*policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain)) #  
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id) # 设置策略模型的填充token ID
            await policy_model.offload_to_cpu() # note：为啥要这么干
            if cfg.critic_pretrain:
                await asyncio.gather(*critic_model.async_init_model_from_pretrained(self.strategy, cfg.critic_pretrain))
                await critic_model.offload_to_cpu()
            if cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    await asyncio.gather(*reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))

        self.policy_model = policy_model
        self.critic_model = critic_model
        self.ref_model = ref_model
        self.reward_model = reward_models

        logger.info("init policy/ref/critic/reward models done")

        # 设置模型实例属性，供其他方法使用
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.ref_model = ref_model
        self.reward_model = reward_models

    async def ppo_local_train_policy(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int):
        """
        PPO策略模型的本地训练过程

        这个方法负责执行策略模型（Actor）的训练，是PPO算法的核心组成部分。
        它处理策略梯度计算、权重更新以及与VLLM引擎的权重同步。

        训练流程：
        1. 检查训练条件
           - 验证global_steps是否超过freezing_actor_steps
           - 在早期训练阶段可能冻结actor训练

        2. 策略模型训练
           - 调用分布式策略模型的训练方法
           - 使用经验数据计算策略梯度
           - 应用PPO的clipped objective损失
           - 更新模型参数

        3. 训练指标记录
           - 记录PPO clip ratio（裁剪比例）
           - 记录策略更新步数
           - 记录策略熵值
           - 写入TensorBoard日志

        4. 内存管理
           - 清理GPU缓存
           - 处理colocate模式下的内存优化

        5. 权重同步
           - 在colocate_all模式下重新加载VLLM引擎
           - 将更新后的策略权重广播到所有VLLM引擎
           - 确保推理使用最新的策略参数

        PPO特性：
        - 使用clipped surrogate objective防止过大的策略更新
        - 支持多epoch训练同一批经验数据
        - 包含KL散度约束（可选）
        - 自动梯度裁剪和学习率调度

        Args:
            replay_buffers (List[NaiveReplayBuffer]): 经验回放缓冲区列表
                每个缓冲区对应一个数据并行组
                包含收集的experience数据用于训练
            global_steps (int): 全局训练步数
                用于学习率调度和日志记录
                决定是否执行训练（相对于freezing_actor_steps）

        Returns:
            Optional[Dict[str, float]]: 训练状态字典，包含：
                - policy_loss: 策略损失值
                - clip_ratio: PPO裁剪比例
                - entropy: 策略熵值
                - policy_update_steps: 实际更新步数
                如果未达到训练条件则返回None

        Raises:
            RuntimeError: 当分布式训练失败时
            torch.cuda.OutOfMemoryError: 当GPU内存不足时

        Side Effects:
            - 更新策略模型参数
            - 同步权重到VLLM引擎
            - 写入TensorBoard指标
            - 可能触发GPU内存管理操作

        Note:
            - 该方法只在global_steps > freezing_actor_steps时执行实际训练
            - 权重同步是必需的，确保VLLM推理使用最新策略
            - 支持异步执行以提高训练效率
        """
        if global_steps > self.cfg.freezing_actor_steps:
            async with Timer("Policy model training"):
                status = await self.policy_model.async_ppo_train(global_steps, replay_buffers)
            self.writer.add_scalar("ppo_clip_count", status[0]["clip_ratio"], global_steps)
            self.writer.add_scalar("policy_update_steps", status[0]["policy_update_steps"], global_steps)
            self.writer.add_scalar("policy_entropy", status[0]["entropy"], global_steps)
            await self.policy_model.async_run_method("empty_cache")
        if self.cfg.colocate_all:
            async with Timer("Backload vllm engines to gpu"):
                await self._backload_vllm_engines()
        async with Timer("Broadcast actor weights to vllm engines"):
            await self._sync_policy_weights_to_vllm()

        if global_steps > self.cfg.freezing_actor_steps:
            return status[0]

    async def ppo_local_train_critic(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int):
        """
        PPO价值模型（Critic）的本地训练过程

        这个方法负责训练价值函数网络，用于估计状态价值以计算优势函数。
        价值模型的准确性直接影响PPO算法的样本效率和训练稳定性。

        训练目标：
        - 让价值函数V(s)准确预测实际回报（returns）
        - 最小化价值预测误差
        - 为优势函数计算提供可靠基准

        损失函数：
        - 使用PPO式的clipped value loss
        - 防止价值函数更新过于激进
        - 平衡训练稳定性和收敛速度

        训练流程：
        1. 分布式训练调用
           - 将经验数据分发到各个价值模型副本
           - 并行计算价值损失和梯度
           - 同步梯度并更新参数

        2. 指标记录
           - 记录价值损失（critic_loss）
           - 记录价值模型更新步数
           - 写入TensorBoard进行监控

        Args:
            replay_buffers (List[NaiveReplayBuffer]): 经验回放缓冲区列表
                每个缓冲区对应一个数据并行组
                包含states、returns和old_values用于训练
            global_steps (int): 全局训练步数
                用于学习率调度和日志记录索引

        Returns:
            Dict[str, float]: 训练状态字典，包含：
                - critic_loss: 价值函数损失值
                - critic_update_steps: 价值模型更新步数
                - values: 平均预测价值
                - critic_lr: 价值模型学习率

        Raises:
            RuntimeError: 当分布式训练失败时
            torch.cuda.OutOfMemoryError: 当GPU内存不足时

        Side Effects:
            - 更新价值模型参数
            - 写入TensorBoard指标
            - 可能触发GPU内存管理操作

        Note:
            - 价值模型通常比策略模型更容易训练
            - 支持与策略模型独立的学习率和优化器配置
            - 价值函数的质量直接影响优势函数的估计精度
        """
        async with Timer("Critic model training"):
            status = await self.critic_model.async_ppo_train(global_steps, replay_buffers)
        if critic_loss := status[0].get("critic_loss", None):
            self.writer.add_scalar("critic_loss", critic_loss, global_steps)
            self.writer.add_scalar("critic_update_steps", status[0]["critic_update_steps"], global_steps)
        return status[0]

    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """
        自定义奖励函数接口

        这是一个抽象接口，允许用户实现特定于任务的奖励计算逻辑。
        该函数在经验收集过程中被调用，用于计算除标准奖励模型之外的额外奖励信号。

        应用场景：
        - 任务特定的评估指标（如代码正确性、数学准确性）
        - 多目标优化（结合多个奖励信号）
        - 在线奖励计算（基于实时反馈）
        - 规则基础的奖励（长度惩罚、格式要求等）

        设计模式：
        - 可以修改prompts和outputs（数据清洗、过滤等）
        - 返回额外的奖励张量
        - 支持异步计算密集型操作
        - 可以利用提供的reward_model_fn

        Args:
            prompts (List[str]): 输入提示列表
                原始的用户查询或任务描述
            outputs (List[Any]): 模型生成的输出列表
                通常是字符串，但也可能包含其他结构化数据
            extras (List[dict]): 额外元数据列表
                包含任务类型、难度级别等辅助信息
            reward_model_fn (Callable): 标准奖励模型函数
                可选使用的预训练奖励模型
                接受(prompts, outputs)返回奖励张量

        Returns:
            Tuple[List[str], List[str], List[torch.Tensor]]: 包含：
                - 处理后的prompts（可能经过过滤或修改）
                - 处理后的outputs（可能经过清洗或标准化）
                - 自定义奖励张量列表（每个输出对应一个奖励值）

        Raises:
            NotImplementedError: 当子类未实现此方法时
            ValueError: 当输入数据格式不正确时
            RuntimeError: 当奖励计算失败时

        Example:
            async def custom_reward_fn(self, prompts, outputs, extras, reward_model_fn):
                # 计算代码正确性奖励
                code_rewards = []
                for prompt, output in zip(prompts, outputs):
                    if self.is_code_task(prompt):
                        reward = self.evaluate_code_correctness(output)
                        code_rewards.append(torch.tensor(reward))
                    else:
                        # 使用标准奖励模型
                        reward = await reward_model_fn([prompt], [output])
                        code_rewards.append(reward)
                return prompts, outputs, code_rewards

        Note:
            - 这是一个模板方法，需要在具体实现中override
            - 自定义奖励会与标准奖励模型输出组合使用
            - 支持复杂的异步计算逻辑
        """
        raise NotImplementedError("custom reward function is not supported yet")

    @torch.no_grad()
    async def _calc_advantages_and_returns(self, experience: Experience):
        """
        计算PPO训练所需的优势函数和回报值

        这是PPO算法中的关键步骤，负责将原始奖励信号转换为训练所需的目标值。
        该方法实现了GAE（Generalized Advantage Estimation）算法来减少方差并提高样本效率。

        计算流程：
        1. 奖励计算
           - 结合多种奖励信号（ORM奖励、自定义奖励、KL惩罚）
           - 应用奖励裁剪避免极值
           - 计算总奖励 = 原始奖励 + KL惩罚 + 自定义奖励

        2. 优势和回报计算
           - 使用GAE算法计算优势函数 A(s,a)
           - 计算目标回报值 returns = advantages + values
           - 考虑gamma（折扣因子）和lambda（GAE参数）

        3. 统计指标计算
           - 平均奖励、KL散度、响应长度
           - ORM分数、自定义奖励等
           - 优势函数的统计特性

        4. 数据处理
           - 清理临时信息减少内存占用
           - 将数据移动到CPU节省GPU内存
           - 为replay buffer准备格式化数据

        GAE算法：
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

        奖励组合：
        total_reward = base_reward + kl_coef * kl_divergence + custom_rewards

        Args:
            experience (Experience): 单个经验对象，包含：
                - values: 价值模型预测的状态价值
                - info: 包含各种奖励和元数据的字典
                - kl: KL散度张量
                - action_mask: 动作掩码（可选）

        Returns:
            Tuple[Experience, dict]: 包含：
                - experience: 更新后的经验对象，新增：
                  * advantages: 计算得到的优势函数值
                  * returns: 计算得到的目标回报值
                - metrics: 统计指标字典，包含：
                  * avg_rewards: 平均奖励
                  * avg_kl: 平均KL散度
                  * avg_kl_max: 最大KL散度
                  * avg_response_length: 平均响应长度
                  * avg_orm_score: 平均ORM分数
                  * avg_custom_rewards: 平均自定义奖励
                  * avg_advantages: 平均优势值
                  * avg_advantages_abs: 平均优势绝对值

        Side Effects:
            - 修改传入的experience对象
            - 从experience.info中删除临时数据
            - 将experience数据移动到CPU

        Raises:
            RuntimeError: 当奖励计算失败时
            ValueError: 当数据维度不匹配时

        Note:
            - 使用@torch.no_grad()确保不计算梯度
            - GAE参数通过cfg.gamma和cfg.lambd配置
            - 支持打包序列的处理
            - 所有计算都考虑了action_mask
        """
        num_actions = experience.info["num_actions"]
        reward = await compute_reward.remote(
            experience.info["reward"],
            self.cfg.init_kl_coef,
            experience.kl,
            custom_rewards=experience.info["custom_rewards"],
            action_mask=experience.action_mask,
            num_actions=num_actions,
            reward_clip_range=self.cfg.reward_clip_range,
            use_kl_loss=self.cfg.use_kl_loss,
        )
        experience.advantages, experience.returns = await get_advantages_and_returns.remote(
            experience.values,
            reward,
            experience.action_mask,
            num_actions,
            self.cfg.gamma,
            self.cfg.lambd,
            packing=True,
        )
        return_sums = reward.sum(dim=-1)
        return_sums /= len(num_actions)
        experience.info["return"] = return_sums
        experience.kl = None

        avg_rewards = return_sums.mean().item()
        avg_kl = experience.info["kl"].mean().item()
        avg_kl_max = experience.info["kl_max"].mean().item()

        avg_response_length = experience.info["response_length"].mean().item()
        if experience.info["reward"] is not None:
            avg_orm_score = experience.info["reward"].mean().item()
        else:
            avg_orm_score = 0

        if experience.info["custom_rewards"] is not None:

            def func(x):
                return [r.sum() for r in x]

            avg_custom_rewards = torch.stack(func(experience.info["custom_rewards"])).mean().item()
            # experience.info["avg_custom_rewards"] = torch.stack(func(experience.info["custom_rewards"]))
        else:
            avg_custom_rewards = 0

        del experience.info["num_actions"]
        del experience.info["custom_rewards"]
        del experience.info["reward"]
        del experience.info["kl_max"]
        experience.to_device("cpu")

        # for replay buffer split batch
        num_packed_samples = len(num_actions)
        return_sums /= num_packed_samples
        experience.info["response_length"] = torch.Tensor(experience.info["response_length"]).mean().unsqueeze(0)
        experience.info["total_length"] = torch.Tensor(experience.info["total_length"]).mean().unsqueeze(0)

        metrics = {
            "avg_rewards": avg_rewards,
            "avg_kl": avg_kl,
            "avg_kl_max": avg_kl_max,
            "avg_response_length": avg_response_length,
            "avg_orm_score": avg_orm_score,
            "avg_custom_rewards": avg_custom_rewards,
            "avg_advantages": experience.advantages.mean().item(),
            "avg_advantages_abs": experience.advantages.abs().mean().item(),
        }

        return experience, metrics

    def _convert_prompts_outputs_to_batch_tensors(self, prompts: List[str], outputs: List[str]):
        """
        将提示和输出转换为批次张量（传统填充模式）

        这是一个传统的批处理方法，使用填充（padding）方式将不同长度的序列
        对齐到相同长度。与打包模式相比，这种方法更简单但内存效率较低。

        处理流程：
        1. 文本token化
           - 分别对prompts和outputs进行token化
           - 应用长度限制（prompt_max_len, generate_max_len）
           - 记录每个序列的实际长度

        2. 长度对齐
           - 计算批次中的最大prompt长度和最大response长度
           - 对shorter序列进行填充
           - prompts使用左填充（left padding）
           - responses使用右填充（right padding）

        3. 序列拼接
           - 将prompt和response拼接成完整序列
           - 格式：[PAD][PAD][prompt_tokens][response_tokens][PAD][PAD]
           - 确保序列长度一致

        4. 掩码生成
           - 创建attention_mask标识有效token
           - 创建action_mask标识可学习的action tokens
           - 处理EOS token的特殊逻辑

        数据格式示例：
        ```
        | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        | token token token token token | token token [EOS] [PAD] |
        | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        |<---------- prompt ----------->|<-------- answer -------->
        ```

        Args:
            prompts (List[str]): 输入提示文本列表
                将被token化并左填充到相同长度
            outputs (List[str]): 目标输出文本列表
                将被token化并右填充到相同长度

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 包含：
                - sequences: 拼接后的序列张量 [batch_size, max_seq_len]
                - attention_mask: 注意力掩码 [batch_size, max_seq_len]
                - action_mask: 动作掩码 [batch_size, max_action_len]

        Note:
            - 这个方法目前未在打包模式中使用
            - 相比打包模式，内存利用率较低但实现更简单
            - 适合序列长度相对均匀的数据集
            - 支持EOS token的自动添加和处理
        """
        # This function is used when not packing samples
        # concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        prompt_token_lens, response_token_lens = [], []
        inputs_token_ids, outputs_token_ids = [], []
        for prompt, output in zip(prompts, outputs):
            input_token_ids = self._tokenize(prompt, self.cfg.prompt_max_len, padding=False)["input_ids"]
            response_token_ids = self._tokenize(output, self.cfg.generate_max_len, padding=False)["input_ids"]

            inputs_token_ids.append(input_token_ids)
            outputs_token_ids.append(response_token_ids)

            prompt_token_len = len(input_token_ids)
            response_token_len = len(response_token_ids)
            prompt_token_lens.append(prompt_token_len)
            response_token_lens.append(response_token_len)

            max_input_len = max(max_input_len, prompt_token_len)
            max_output_len = max(max_output_len, response_token_len)

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for i, prompt in enumerate(prompts):
            # left padding input
            input_len = prompt_token_lens[i]
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(inputs_token_ids[i])

            # right padding output
            output_len = response_token_lens[i]
            output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (max_output_len - output_len)

            # replace last token with eos_token_id if it is not eos_token_id, keep the total length of output_ids
            # output_ids[output_len - 1] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)

        sequences, attention_mask, action_mask = self._process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences, attention_mask, action_mask

    def _convert_prompts_outputs_to_batch_tensors_packing(
        self, prompts: List[str], outputs: List[str], custom_rewards: Optional[List[torch.Tensor]], packing_max_len: int
    ):
        """
        将提示和输出转换为高效的打包批次张量

        这是一种内存高效的批处理方法，将多个不同长度的对话序列打包到
        固定长度的张量中，显著提高GPU内存利用率和训练效率。

        打包原理：
        - 将多个短序列连续放置在同一个张量中
        - 使用attention_mask区分不同序列的边界
        - 避免传统padding造成的内存浪费
        - 特别适合长度差异较大的数据集

        打包策略：
        1. 贪心打包：按顺序尝试将序列添加到当前张量
        2. 长度检查：确保不超过packing_max_len限制
        3. 动态分割：当无法容纳时创建新的打包张量
        4. 边界标记：用attention_mask标识序列边界

        内存优势：
        - 减少50-80%的内存浪费（相比传统padding）
        - 提高2-3倍的有效token密度
        - 支持更大的有效batch size

        序列格式：
        ```
        张量: [seq1_token1, seq1_token2, seq2_token1, seq2_token2, seq2_token3, seq3_token1, ...]
        掩码: [     1     ,     1     ,     2     ,     2     ,     2     ,     3     , ...]
        ```

        处理流程：
        1. 序列预处理
           - token化prompts和outputs
           - 拼接为完整的prompt+response序列
           - 记录每个序列的长度信息

        2. 打包算法
           - 初始化空的打包张量
           - 逐个尝试添加序列
           - 处理三种情况：能放下、正好填满、放不下

        3. 元数据管理
           - 记录每个序列的action数量（response tokens）
           - 记录每个序列的总长度
           - 管理自定义奖励的对应关系

        4. 张量生成
           - 创建最终的打包张量
           - 生成对应的attention masks
           - 保持所有元数据的一致性

        Args:
            prompts (List[str]): 输入提示文本列表
                将与对应的outputs拼接成完整对话
            outputs (List[str]): 输出回应文本列表
                模型生成的回应文本
            custom_rewards (Optional[List[torch.Tensor]]): 自定义奖励列表
                与prompts/outputs一一对应的额外奖励信号
            packing_max_len (int): 单个打包张量的最大长度
                控制内存使用和计算效率的平衡点

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[List[int]],
                  List[List[int]], Optional[List[List[torch.Tensor]]]]: 包含：
                - ret_sequences: 打包后的序列张量列表
                - ret_attention_masks: 对应的注意力掩码列表
                - ret_num_actions: 每个打包批次中各序列的action数量
                - ret_packed_seq_lens: 每个打包批次中各序列的总长度
                - ret_custom_rewards: 每个打包批次中各序列的自定义奖励

        Raises:
            AssertionError: 当prompts和outputs长度不匹配时
            ValueError: 当单个序列超过packing_max_len时

        Note:
            - 这是当前推荐的批处理方法
            - attention_mask使用整数标识不同序列（1, 2, 3, ...）
            - 支持任意数量的序列打包
            - 自动处理边界情况和内存优化
        """
        ret_sequences = []
        ret_attention_masks = []
        ret_num_actions = []
        ret_packed_seq_lens = []
        if custom_rewards is not None:
            ret_custom_rewards = []
        else:
            ret_custom_rewards = None

        assert (
            len(prompts) == len(outputs) and len(prompts) > 0
        ), "prompts and outputs must have the same length and length must be greater than 0"

        def _new_instance():
            out_sequence = torch.full((packing_max_len,), torch.tensor(self.tokenizer.pad_token_id), dtype=torch.long)
            out_attention_mask = torch.zeros((packing_max_len,), dtype=torch.int)
            out_num_actions = []
            out_packed_seq_lens = []
            rewards = [] if custom_rewards else None
            seq_offset = 0
            seq_index = 0
            return (
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                seq_offset,
                seq_index,
            )

        def _accumulate(
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
            sequence,
            attention_mask,
            num_action,
            total_len,
            custom_rewards,
            i,
        ):
            out_sequence[seq_offset : seq_offset + total_len] = torch.tensor(sequence)
            out_attention_mask[seq_offset : seq_offset + total_len] = seq_index + 1
            out_num_actions.append(num_action)
            out_packed_seq_lens.append(total_len)
            if custom_rewards:
                rewards.append(custom_rewards[i])
            return seq_offset + total_len, seq_index + 1

        sequences = []
        attention_masks = []
        num_actions = []
        total_lens = []

        input_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
        response_token_ids = self._tokenize(outputs, self.cfg.generate_max_len, padding=False)["input_ids"]

        for input_ids, response_ids in zip(input_token_ids, response_token_ids):
            sequences.append(input_ids + response_ids)
            attention_masks.append(torch.ones((len(input_ids) + len(response_ids),), dtype=torch.float32))
            num_actions.append(len(response_ids))
            total_lens.append(len(input_ids) + len(response_ids))

        # make packed sequences
        (
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
        ) = _new_instance()
        for i, (sequence, attention_mask, num_action, total_len) in enumerate(
            zip(sequences, attention_masks, num_actions, total_lens)
        ):
            if seq_offset + total_len < packing_max_len:
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )
            elif seq_offset + total_len == packing_max_len:
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )
                valid_size = out_attention_mask.nonzero().size(0)
                ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                ret_num_actions.append(out_num_actions)
                ret_packed_seq_lens.append(out_packed_seq_lens)
                if custom_rewards:
                    ret_custom_rewards.append(rewards)
                (
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                ) = _new_instance()
            elif seq_offset + total_len > packing_max_len:
                if seq_offset > 0:
                    valid_size = out_attention_mask.nonzero().size(0)
                    ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                    ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                    ret_num_actions.append(out_num_actions)
                    ret_packed_seq_lens.append(out_packed_seq_lens)
                    if custom_rewards:
                        ret_custom_rewards.append(rewards)
                    (
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                    ) = _new_instance()
                    seq_offset, seq_index = _accumulate(
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                        sequence,
                        attention_mask,
                        num_action,
                        total_len,
                        custom_rewards,
                        i,
                    )

        if seq_offset > 0:
            valid_size = out_attention_mask.nonzero().size(0)
            ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
            ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
            ret_num_actions.append(out_num_actions)
            ret_packed_seq_lens.append(out_packed_seq_lens)
            if custom_rewards:
                ret_custom_rewards.append(rewards)

        return ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards

    def _get_dp_group_models(self, dp_rank: int, model_type: str = ""):
        """
        获取指定数据并行组的模型actor

        在分布式训练中，模型通常被复制到多个GPU上进行数据并行。
        这个方法根据数据并行rank获取对应的模型actor句柄。

        Args:
            dp_rank (int): 数据并行rank，范围[0, num_dp_groups)
                指定要获取的数据并行组索引
            model_type (str): 模型类型名称，如：
                - "policy_model": 策略模型
                - "critic_model": 价值模型
                - "ref_model": 参考模型
                - "reward_model": 奖励模型

        Returns:
            Ray Actor: 对应的模型actor句柄
                可用于远程调用模型的方法

        Note:
            - 奖励模型特殊处理：使用第一个奖励模型（支持多奖励模型）
            - 所有模型都通过PPORayActorGroup管理多个actor
        """
        model = getattr(self, model_type)
        if model_type == "reward_model":
            model = model[0]
        return model._actor_handlers[dp_rank]

    def _split_dp_batch(self, batch, num_dp, drop_last=False):
        """
        将批次数据分割为多个数据并行组

        这是分布式训练的核心工具函数，负责将大批次数据均匀分配给
        多个数据并行进程，确保每个GPU获得相同大小的子批次。

        分割策略：
        - 计算每个数据并行组的子批次大小
        - 支持上向取整以避免数据丢失
        - 必要时进行数据填充以保持批次大小一致
        - 支持多种数据类型（张量、列表、None）

        填充机制：
        - 当数据无法整除时，随机采样已有数据进行填充
        - 保持数据分布的统计特性
        - 避免引入偏差

        Args:
            batch (Tuple): 批次数据元组，可包含：
                - torch.Tensor: 张量数据
                - List: 列表数据
                - None: 空数据（保持结构）
            num_dp (int): 数据并行组数量
                决定数据分割的份数
            drop_last (bool): 是否丢弃不完整的最后批次
                False: 填充数据保持完整性（推荐）
                True: 丢弃多余数据

        Yields:
            Tuple: 每次迭代返回一个子批次
                结构与输入batch相同，但大小为原来的1/num_dp

        Example:
            batch = (sequences, attention_masks, num_actions)
            for sub_batch in self._split_dp_batch(batch, 4):
                sub_sequences, sub_masks, sub_actions = sub_batch
                # 每个sub_batch包含原始数据的1/4

        Note:
            - 这是生成器函数，按需产生子批次
            - 支持混合数据类型的批次
            - 填充使用随机采样避免数据偏差
        """
        # Convert batch tuple to list of lists, handling None values
        batch_lists = []
        batch_size = None
        for item in batch:
            if item is not None:
                if batch_size is None:
                    batch_size = len(item)
                batch_lists.append(item)
            else:
                batch_lists.append(None)

        if drop_last:
            dp_size = batch_size // num_dp
        else:
            dp_size = (batch_size + num_dp - 1) // num_dp
        valid_size = dp_size * num_dp

        if not drop_last:
            padding_index = None
            for i in range(len(batch_lists)):
                if batch_lists[i] is not None and (
                    isinstance(batch_lists[i], torch.Tensor) or isinstance(batch_lists[i], list)
                ):
                    padding_size = valid_size - len(batch_lists[i])
                    if padding_size > 0:
                        if padding_index is None:
                            if padding_size > len(batch_lists[i]):
                                padding_index = random.choices(range(len(batch_lists[i])), k=padding_size)
                            else:
                                padding_index = random.sample(range(len(batch_lists[i])), padding_size)
                        if isinstance(batch_lists[i], torch.Tensor):
                            batch_lists[i] = torch.cat([batch_lists[i], batch_lists[i][padding_index]], dim=0)
                        elif isinstance(batch_lists[i], list):
                            batch_lists[i] = batch_lists[i] + [batch_lists[i][j] for j in padding_index]

        for i in range(num_dp):
            # Extract micro batch for each input list
            micro_batch = []
            for batch_list in batch_lists:
                if batch_list is None:
                    micro_batch.append(None)
                elif isinstance(batch_list, torch.Tensor) or isinstance(batch_list, list):
                    micro_batch.append(batch_list[i * dp_size : (i + 1) * dp_size])
                else:
                    micro_batch.append(batch_list)
            yield tuple(micro_batch)

    def _split_dp_batch_dynamic_balance(self, batch, num_dp, balanced_values):
        """
        基于负载均衡的动态批次分割

        这是一种高级的批次分割方法，根据计算负载（如序列长度）
        进行智能分配，确保各个数据并行组的计算时间大致相等。

        负载均衡原理：
        - 使用贪心算法将数据分配给当前负载最轻的进程
        - 考虑序列长度、计算复杂度等因素
        - 最小化最大负载，提高整体训练效率

        应用场景：
        - 序列长度差异很大的数据集
        - 计算复杂度不均匀的任务
        - 需要最小化训练时间的场景

        Args:
            batch (Tuple): 批次数据，与_split_dp_batch格式相同
            num_dp (int): 数据并行组数量
            balanced_values (List[float]): 负载权重列表
                与batch中每个样本对应的计算权重
                如序列长度、token数量等

        Yields:
            List[List]: 每个数据并行组的数据列表
                按负载均衡分配的结果

        Note:
            - 使用堆数据结构实现高效的负载均衡
            - 适合对训练时间敏感的场景
            - 比简单分割有更好的负载分布
        """
        batch = list(batch)
        assert len(batch) == len(balanced_values), "batch and balanced_values must have the same length"
        results = self._split_weighted_objects(zip(balanced_values, batch), num_dp)
        # re organize to the original format
        for i in range(num_dp):
            ret = [[] for _ in range(len(results[i][0]))]
            for sample in results[i]:
                for j, v in enumerate(sample):
                    ret[j].append(v)
            yield ret

    def _split_weighted_objects(self, items, n):
        """
        将带权重的对象分配到n个组中实现负载均衡

        使用贪心算法和最小堆数据结构，将带权重的对象尽可能均匀地
        分配到n个组中，最小化各组之间的负载差异。

        算法特点：
        - 贪心策略：优先级别将重对象分配给当前负载最轻的组
        - 最小堆：高效维护各组的当前负载
        - 时间复杂度：O(m log n)，其中m是对象数量

        分配策略：
        1. 按权重降序排列对象（重对象优先）
        2. 维护各组当前总权重的最小堆
        3. 逐个将对象分配给负载最轻的组
        4. 更新该组的总负载

        Args:
            items (Iterable[Tuple[float, Any]]): 带权重的对象列表
                每个元素为(weight, object)的元组
            n (int): 目标分组数量

        Returns:
            List[List[Any]]: n个组的对象列表
                每个组包含分配给它的对象列表

        Example:
            items = [(10, 'A'), (8, 'B'), (6, 'C'), (4, 'D')]
            result = self._split_weighted_objects(items, 2)
            # 可能返回: [['A', 'D'], ['B', 'C']]（负载为14和14）

        Note:
            - 这是近似算法，不保证全局最优但效果很好
            - 特别适合序列长度等连续权重值
            - 被_split_dp_batch_dynamic_balance内部使用
        """
        result = [[] for _ in range(n)]

        heap = [(0, i) for i in range(n)]
        heapify(heap)

        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

        for weight, obj in sorted_items:
            current_sum, index = heappop(heap)
            result[index].append(obj)
            heappush(heap, (current_sum + weight, index))

        return result

    async def _split_and_run_micro_batch(self, async_fn, batch_args, micro_size):
        """
        将大批次分割为微批次并异步执行

        这是一个重要的内存管理工具，将大批次数据分割为更小的微批次，
        避免GPU内存溢出，同时保持计算效率。

        应用场景：
        - 大模型推理时的内存限制
        - 避免CUDA OOM错误
        - 在内存和速度之间找平衡

        处理流程：
        1. 计算微批次数量和边界
        2. 逐个创建微批次参数
        3. 异步调用处理函数
        4. 收集所有结果

        参数处理：
        - 自动识别不同数据类型（张量、列表、标量）
        - 保持参数结构的完整性
        - 处理None值和边界情况

        Args:
            async_fn (Callable): 异步处理函数
                接受微批次参数并返回结果
            batch_args (Tuple): 批次参数元组
                需要分割的数据参数
            micro_size (int): 微批次大小
                每个微批次的样本数量

        Returns:
            List: 所有微批次结果的列表
                按处理顺序排列的结果

        Raises:
            RuntimeError: 当微批次处理失败时
            torch.cuda.OutOfMemoryError: 当微批次仍然太大时

        Example:
            results = await self._split_and_run_micro_batch(
                model.forward,
                (large_sequences, large_masks),
                micro_size=8
            )

        Note:
            - 这是异步函数，支持并发执行
            - 微批次大小需要根据GPU内存调整
            - 结果顺序与输入顺序保持一致
        """
        # Ensure batch_args is a sequence of lists with equal length
        batch_size = len(batch_args[0])
        results = []
        # Process in micro batches
        for i in range(0, batch_size, micro_size):
            # Take slice i:i+micro_size from each argument
            micro_batch_args = []
            for arg in batch_args:
                if arg is not None:
                    if not isinstance(arg, torch.Tensor) and not isinstance(arg, list):
                        micro_batch_args.append(arg)
                    elif micro_size > 1 or isinstance(arg, torch.Tensor):
                        micro_batch_args.append(arg[i : i + micro_size])
                    else:
                        micro_batch_args.append(arg[i])
                else:
                    micro_batch_args.append(None)
            results.append(await async_fn(*micro_batch_args))
        return results

    def _get_generate_function(self, dp_rank: int):
        """
        创建特定数据并行rank的文本生成函数

        这个方法为每个数据并行进程创建一个专用的生成函数，
        确保负载均衡和避免资源冲突。每个函数绑定到特定的VLLM引擎。

        设计模式：
        - 工厂方法：动态创建绑定特定引擎的生成函数
        - 负载均衡：使用取模运算分配引擎
        - 封装复杂性：隐藏token化和后处理细节

        生成流程：
        1. token化输入prompts
        2. 调用VLLM引擎生成
        3. 提取生成文本和元数据
        4. 处理概率信息（可选）

        Args:
            dp_rank (int): 数据并行rank
                用于选择对应的VLLM引擎

        Returns:
            Callable: 异步生成函数，接受：
                - prompts (List[str]): 输入提示列表
                - truncate_prompt (bool): 是否截断prompt
                - **kwargs: 其他生成参数
                返回：
                - responses (List[str]): 生成的文本列表
                - finish_reasons (List[str]): 结束原因列表
                - prompt_logprobs (List, optional): prompt概率信息

        Example:
            gen_func = self._get_generate_function(0)
            responses, reasons = await gen_func(
                prompts=["Hello", "How are you?"],
                temperature=0.8
            )

        Note:
            - 每个dp_rank对应一个固定的VLLM引擎
            - 支持prompt截断和概率输出
            - 返回的函数可以重复使用
        """
        llm = self.vllm_engines[dp_rank % len(self.vllm_engines)]

        async def generate(prompts: List[str], truncate_prompt=True, **kwargs):
            if truncate_prompt:
                prompt_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
            else:
                prompt_token_ids = self._tokenize(prompts, padding=False)["input_ids"]
            outputs = await llm.generate.remote(prompt_token_ids=prompt_token_ids, **kwargs)
            responses = []
            prompt_logprobs = []
            finish_reasons = []
            for i, prompt in enumerate(prompts):
                content = outputs[i].outputs[0].text
                finish_reasons.append(outputs[i].outputs[0].finish_reason)
                responses.append(content)
                if outputs[i].prompt_logprobs:
                    prompt_logprobs.append(outputs[i].prompt_logprobs)
            if len(prompt_logprobs) > 0:
                return (
                    responses,
                    finish_reasons,
                    prompt_logprobs,
                )
            else:
                return responses, finish_reasons

        return generate

    def _process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        """
        处理和规范化生成的序列

        这个方法对生成的token序列进行后处理，确保格式正确和一致性。
        主要处理EOS token的位置、attention mask的生成和action mask的创建。

        处理步骤：
        1. 生成基础attention mask（排除EOS和PAD）
        2. 定位并修正EOS token位置
        3. 处理特殊情况（如prompt中的EOS token）
        4. 创建action mask用于RL训练

        特殊处理：
        - Llama3和Qwen2模型在prompt中可能包含EOS token
        - 确保每个序列只有一个有效的EOS token
        - 正确设置attention boundaries

        Args:
            sequences (torch.Tensor): 生成的序列张量 [batch_size, seq_len]
            input_len (int): 输入prompt的长度
            eos_token_id (int): 结束token的ID
            pad_token_id (int): 填充token的ID

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - sequences: 处理后的序列张量
                - attention_mask: 注意力掩码
                - action_mask: 动作掩码（用于RL训练）

        Note:
            - 这个方法主要用于非打包模式
            - action_mask标识哪些位置可以用于策略学习
            - 处理了各种边界情况和特殊token
        """
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def _tokenize(self, texts, max_length=99999999, padding=True, device=None):
        """
        文本token化的统一接口

        提供灵活的文本token化功能，支持多种配置选项。
        这是所有文本处理的入口点，确保token化的一致性。

        特性：
        - 支持单个文本和文本列表
        - 可选的长度截断和填充
        - 灵活的设备分配
        - 统一的特殊token处理

        两种模式：
        1. padding=True: 返回对齐的张量，适合批处理
        2. padding=False: 返回列表，适合变长处理

        Args:
            texts (Union[str, List[str]]): 输入文本或文本列表
            max_length (int): 最大token长度，默认不限制
            padding (bool): 是否进行填充对齐
                True: 返回张量格式
                False: 返回列表格式
            device (Optional[torch.device]): 目标设备
                仅在padding=True时有效

        Returns:
            Union[Dict[str, torch.Tensor], Dict[str, List]]:
                padding=True时返回张量字典：
                - input_ids: token ID张量
                - attention_mask: 注意力掩码张量
                padding=False时返回列表字典：
                - input_ids: token ID列表的列表

        Example:
            # 批处理模式
            batch = self._tokenize(["Hello", "World"], padding=True)
            sequences = batch['input_ids']  # [2, max_len]

            # 变长模式
            lists = self._tokenize(["Hello", "World"], padding=False)
            token_lists = lists['input_ids']  # [[1, 2, 3], [4, 5]]

        Note:
            - 自动处理特殊token（不添加BOS/EOS）
            - 支持截断以控制序列长度
            - device参数只在padding=True时生效
        """
        """
        文本分词

        Args:
            texts: 输入文本或文本列表
            max_length: 最大长度
            padding: 是否进行填充
            device: 目标设备

        Returns:
            分词结果
        """
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _detokenize(self, token_ids):
        """
        将token ID序列转换回文本

        这是token化的逆过程，用于将模型的token输出转换为可读文本。
        主要用于日志记录、调试和结果展示。

        Args:
            token_ids (Union[torch.Tensor, List[int]]): token ID序列
                可以是张量或整数列表

        Returns:
            str: 解码后的文本字符串
                保留特殊token用于调试

        Example:
            text = self._detokenize([1, 15043, 2188, 2])
            # 返回: "<s>Hello world</s>"

        Note:
            - skip_special_tokens=False保留特殊token
            - 用于TensorBoard可视化和调试
            - 支持多种输入格式的自动转换
        """
        """
        将token ID转换回文本

        Args:
            token_ids: token ID序列

        Returns:
            str: 解码后的文本
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _warp_custom_reward_model_fn(self):
        """
        创建自定义奖励模型的包装函数

        这个方法创建一个异步函数，用于调用奖励模型对prompt-response对进行评分。
        该函数处理数据预处理、分布式推理和结果聚合的完整流程。

        功能特性：
        - 支持分布式奖励模型推理
        - 自动数据打包和分割
        - 处理微批次以避免内存溢出
        - 结果聚合和格式化

        处理流程：
        1. 将prompt-response对转换为打包张量
        2. 分割数据到各个数据并行组
        3. 并行调用奖励模型推理
        4. 聚合结果并返回奖励分数

        Returns:
            Optional[Callable]: 奖励模型包装函数或None
                如果有奖励模型，返回异步函数：
                - 输入: (prompts: List[str], outputs: List[str])
                - 输出: torch.Tensor，奖励分数张量
                如果没有奖励模型，返回None

        Example:
            reward_fn = self._warp_custom_reward_model_fn()
            if reward_fn:
                scores = await reward_fn(prompts, responses)
                # scores: [batch_size] 奖励分数

        Note:
            - 只有在self.reward_model存在时才创建包装函数
            - 支持多个奖励模型（目前使用第一个）
            - 自动处理数据并行和批处理
            - 结果tensor会自动去除填充部分
        """
        if self.reward_model:
            # TODO: support multiple reward models]
            num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node

            async def warpped_reward_model_fn(prompts: List[str], outputs: List[str]):
                (
                    sequences,
                    attention_mask,
                    _,
                    packed_seq_lens,
                    _,
                ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                    prompts, outputs, None, self.cfg.packing_max_len
                )
                split_iterator = self._split_dp_batch(
                    (sequences, attention_mask, packed_seq_lens), num_policy_dp_groups
                )
                dp_tasks = []

                async def _rm_run(rm, seq, mask, lens):
                    return await rm.forward.remote(seq, mask, packed_seq_lens=lens)

                for dp_rank, args in enumerate(split_iterator):
                    rm = self._get_dp_group_models(dp_rank, "reward_model")
                    dp_tasks.append(
                        self._split_and_run_micro_batch(
                            partial(_rm_run, rm),
                            args,
                            self.cfg.micro_forward_batch_size,
                        )
                    )
                outputs = await asyncio.gather(*dp_tasks)
                outputs = sum(outputs, [])  # gather dp
                outputs = outputs[: len(sequences)]  # drop padding
                outputs = torch.hstack(outputs)

                assert outputs.size(0) == len(prompts), "reward outputs number must be equal to prompts number"
                return outputs

            return warpped_reward_model_fn
        else:
            return None

    async def _offload_vllm_engines(self):
        """
        将VLLM引擎卸载到CPU内存

        在colocate_all模式下，这个方法用于释放GPU内存以便其他模型使用。
        所有VLLM引擎会被并行卸载以提高效率。

        应用场景：
        - colocate_all模式下的内存管理
        - 在策略模型训练前释放GPU内存
        - 避免GPU内存不足错误

        Returns:
            None

        Note:
            - 使用asyncio.gather实现并行卸载
            - 卸载后VLLM引擎暂时无法进行推理
            - 需要调用_backload_vllm_engines重新加载
        """
        offload_tasks = []
        for engine in self.vllm_engines:
            offload_tasks.append(engine.offload_to_cpu.remote())
        await asyncio.gather(*offload_tasks)

    async def _backload_vllm_engines(self):
        """
        将VLLM引擎从CPU重新加载到GPU内存

        这与_offload_vllm_engines相对应，用于恢复VLLM引擎的推理能力。
        通常在需要进行文本生成之前调用。

        应用场景：
        - colocate_all模式下的内存管理
        - 在文本生成前恢复VLLM功能
        - 内存优化的动态加载

        Returns:
            None

        Note:
            - 使用asyncio.gather实现并行加载
            - 加载后VLLM引擎恢复正常推理功能
            - 与_offload_vllm_engines配对使用
        """
        backload_tasks = []
        for engine in self.vllm_engines:
            backload_tasks.append(engine.backload_to_gpu.remote())
        await asyncio.gather(*backload_tasks)

    async def _sync_policy_weights_to_vllm(self):
        """
        同步策略模型权重到VLLM引擎

        这是PPO训练中的关键步骤，确保VLLM引擎使用最新的策略模型权重进行推理。
        在每次策略模型更新后都需要调用此方法。

        同步机制：
        1. colocate_all模式：使用CUDA IPC进行高效内存共享
        2. 非colocate模式：通过网络广播权重参数

        技术特性：
        - CUDA IPC：零拷贝内存共享，极高效率
        - 网络广播：支持跨节点权重同步
        - 异步执行：不阻塞训练流程

        应用时机：
        - 策略模型训练完成后
        - 训练开始前的初始同步
        - 模型检查点恢复后

        Returns:
            None

        Raises:
            RuntimeError: 当权重同步失败时

        Note:
            - 这是PPO算法正确性的关键保证
            - 不同的并置模式使用不同的同步策略
            - 同步完成后VLLM引擎立即可用新权重
        """
        """同步策略模型权重到VLLM引擎"""
        if self.cfg.colocate_all:
            await self.policy_model.async_run_method("_broadcast_to_vllm_cudaipc", self.vllm_engines)
        else:
            await self.policy_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)
