"""
简单脚本：加载 Qwen2.5-7B 模型并保存为 iter0
用于 orz_7b_ppo_jericho_1013_1gpu 实验

运行命令:
python save_iter0_jericho.py
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置参数
MODEL_PATH = "/mnt/shared-storage-user/tangjia/orz/Open-Reasoner-Zero/model/models--Qwen--Qwen2.5-7B/snapshots/e25af2efae60472008fbeaf5fb7c4274a87f78d4"
SAVE_BASE_PATH = "jericho_his10_orz_20251013_ckpt_1gpu/orz_7b_ppo_jericho_1013_1gpu"

def main():
    print("="*80)
    print("开始保存 iter0 模型")
    print("="*80)

    # 1. 加载模型
    print(f"\n[1/4] 加载模型: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    print("✅ 模型加载完成")

    # 2. 加载 tokenizer
    print(f"\n[2/4] 加载 tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    print("✅ Tokenizer 加载完成")

    # 3. 保存 policy 模型
    policy_save_path = os.path.join(SAVE_BASE_PATH, "iter0", "policy")
    print(f"\n[3/4] 保存 policy 模型到: {policy_save_path}")
    os.makedirs(policy_save_path, exist_ok=True)
    model.save_pretrained(policy_save_path)
    tokenizer.save_pretrained(policy_save_path)
    print("✅ Policy 模型保存完成")

    # 4. 保存 critic 模型（与 policy 相同）
    critic_save_path = os.path.join(SAVE_BASE_PATH, "iter0", "critic")
    print(f"\n[4/4] 保存 critic 模型到: {critic_save_path}")
    os.makedirs(critic_save_path, exist_ok=True)
    model.save_pretrained(critic_save_path)
    tokenizer.save_pretrained(critic_save_path)
    print("✅ Critic 模型保存完成")

    print("\n" + "="*80)
    print("🎉 所有模型已成功保存到 iter0！")
    print("="*80)
    print(f"\nPolicy 模型位置: {policy_save_path}")
    print(f"Critic 模型位置: {critic_save_path}")
    print("\n可以开始训练了！")

if __name__ == "__main__":
    main()
