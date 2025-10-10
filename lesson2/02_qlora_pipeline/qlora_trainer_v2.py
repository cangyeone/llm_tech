"""LLaMA/Qwen QLoRA 微调流程脚手架（含 NF4、分页优化器、梯度检查点）

功能：
- 加载偏好或指令数据集（JSONL）
- bitsandbytes 4-bit（NF4）量化加载
- peft/LoRA 注入并结合 Trainer 训练
- 分页优化器（paged_adamw_32bit）
- 梯度检查点（gradient checkpointing）
- k-bit 训练准备（prepare_model_for_kbit_training）

用法示例：
python train_qlora.py --data data.jsonl --model Qwen/Qwen3-4b --steps 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ——— 配置 ———
@dataclass
class QLoRAConfig:
    data_path: Path
    model_name: str = "Qwen/Qwen3-4b"
    output_dir: Path = Path("./outputs/llama_qlora")
    batch_size: int = 1
    gradient_accumulation: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 100
    bf16: bool = True  # 若显卡支持 BF16，建议开启
    fp16: bool = False  # 若不支持 BF16，可改为 True


def parse_args() -> QLoRAConfig:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning (NF4 + Paged Optim + Grad-CKPT)")
    parser.add_argument("--data", type=Path, default="outputs/examples_translation_en_zh.jsonl", help="指令 JSONL 数据路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4b")
    parser.add_argument("--output", type=Path, default=Path("./outputs/llama_qlora"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=16)
    parser.add_argument("--bf16", action="store_true", help="启用 BF16 训练（若硬件支持）")
    parser.add_argument("--fp16", action="store_true", help="启用 FP16 训练（若 BF16 不可用）")
    args = parser.parse_args()
    return QLoRAConfig(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        learning_rate=args.lr,
        max_steps=args.steps,
        batch_size=args.batch,
        gradient_accumulation=args.grad_acc,
        bf16=args.bf16,
        fp16=args.fp16,
    )


# ——— 数据处理 ———
def read_jsonl(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def build_dataset(records: List[Dict[str, str]]) -> Dataset:
    return Dataset.from_list(records)


def format_sample(sample: Dict[str, str]) -> str:
    return (
        f"### 指令:\n{sample.get('instruction', '')}\n"
        f"### 输入:\n{sample.get('input', '')}\n"
        f"### 回答:\n{sample.get('output', '')}"
    )


def tokenize(dataset: Dataset, tokenizer) -> Dataset:
    return dataset.map(
        lambda sample: tokenizer(
            format_sample(sample),
            max_length=2048,
            truncation=True,
            padding="max_length",
        ),
        remove_columns=dataset.column_names,
    )


def main() -> None:
    config = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True  # 提速不影响数值稳定性
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = read_jsonl(config.data_path)
    dataset = build_dataset(records)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # 有些模型（如 Qwen）需要 trust_remote_code 才能正确加载
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # —— NF4 量化配置（bitsandbytes 4-bit）——
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,       # 二次量化，进一步降低显存
        bnb_4bit_quant_type="nf4",            # NF4 量化
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )

    # 加载 4bit 基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # —— k-bit 训练准备：冻结 norm/嵌入、启用 gradient_checkpointing 前的必要设置 —— 
    # prepare_model_for_kbit_training 会：
    # - 将部分层转为 fp32（如 LayerNorm）保证数值稳定
    # - 处理梯度设置，便于在低精度下训练 LoRA
    base_model = prepare_model_for_kbit_training(base_model)

    # —— LoRA 配置 ——（按需增减 target_modules，例如 Qwen 常见是 q/k/v/o 或再加 up/down/gate）
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # —— 梯度检查点（Gradient Checkpointing）——
    # 注意：开启后需禁用 use_cache，否则 Trainer 会报错/无效
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    tokenized_train = tokenize(dataset["train"], tokenizer)
    tokenized_eval = tokenize(dataset["test"], tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # —— TrainingArguments 里启用分页优化器与混合精度 —— 
    # optim="paged_adamw_32bit"：bitsandbytes 分页 AdamW（优化器状态放 CPU，按需分页到 GPU）
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        report_to=[],
        optim="paged_adamw_32bit",              # ← 分页优化器
        bf16=config.bf16,                        # 若支持 BF16，优先使用
        fp16=(config.fp16 and not config.bf16),  # 否则用 FP16
        gradient_checkpointing=True,             # 与上面的 model.gradient_checkpointing_enable() 一致
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))


if __name__ == "__main__":
    main()
