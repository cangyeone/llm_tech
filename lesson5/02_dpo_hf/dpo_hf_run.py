# -*- coding: utf-8 -*-
"""
多卡 DPO 训练脚本 (基于 TRL + Accelerate)
====================================================
- 支持单机多 GPU 并行
- 自动分布式同步梯度
- 可选 AMP / ZeRO 优化
- 适配 Qwen3 系列模型（attn_implementation='eager'）

运行方式:
torchrun --nproc_per_node=4 train_dpo_trl_multigpu.py
或
accelerate launch train_dpo_trl_multigpu.py
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
import torch
import os

# =====================================================
# 1️⃣ 环境准备
# =====================================================
os.environ.setdefault("PYTORCH_USE_FLASH_ATTENTION", "0")  # 防止 FA2 冲突
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =====================================================
# 2️⃣ 加载数据 (示例：仅 0.1%)
# =====================================================
dataset = load_dataset("lvwerra/stack-exchange-paired", split="train", data_dir="data/finetune")
dataset = dataset.select(range(int(len(dataset) * 0.001)))

# 数据格式：
# question, response_j (chosen), response_k (rejected)

# =====================================================
# 3️⃣ 加载 tokenizer
# =====================================================
model_name = "Qwen/Qwen3-0.6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =====================================================
# 4️⃣ 构建 DPO Trainer (TRL)
# =====================================================
# DPO 原理：max E[ log π_θ(y⁺|x) − log π_θ(y⁻|x) ]，以参考模型为对照
#           β 控制偏好强度，越大越接近直接对比 KL。
# TRL 的 DPOTrainer 自动处理数据并行、梯度同步等。

# 初始化 policy/reference 模型
policy_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
    trust_remote_code=True,
)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
    trust_remote_code=True,
)
ref_model.requires_grad_(False)

# =====================================================
# 5️⃣ 配置 DPO 超参数（多卡自动同步）
# =====================================================
dpo_config = DPOConfig(
    beta=0.1,                  # DPO 强度系数
    learning_rate=5e-6,
    max_length=512,
    max_prompt_length=256,
    max_target_length=256,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # 小显存场景
    remove_unused_columns=False,
    logging_steps=10,
    save_strategy="epoch",
    output_dir="outputs/dpo_qwen_trl",
    report_to="none",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    bf16=torch.cuda.is_bf16_supported(),
)

# =====================================================
# 6️⃣ 启动 DPOTrainer
# =====================================================
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=TrainingArguments(
        output_dir=dpo_config.output_dir,
        learning_rate=dpo_config.learning_rate,
        per_device_train_batch_size=dpo_config.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_config.gradient_accumulation_steps,
        num_train_epochs=1,
        bf16=dpo_config.bf16,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    ),
    beta=dpo_config.beta,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_length=dpo_config.max_length,
    max_prompt_length=dpo_config.max_prompt_length,
    max_target_length=dpo_config.max_target_length,
)

# =====================================================
# 7️⃣ 启动训练（多卡自动并行）
# =====================================================
print("🚀 Starting multi-GPU DPO training ...")
trainer.train()
print("✅ Training complete.")

# =====================================================
# 8️⃣ 保存模型
# =====================================================
output_dir = dpo_config.output_dir
print(f"✅ Saving final model to {output_dir}")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
