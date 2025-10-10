# -*- coding: utf-8 -*-
"""
ZeRO-3 完整最小示例：
- 支持纯全参微调 或 LoRA/QLoRA（--use_lora/--lora_r/--lora_8bit）
- 支持 ZeRO-3 + 可选 CPU Offload（由 deepspeed config 控制）
- 兼容 'gpt2'（自动把 pad_token = eos_token）
- 文本数据：优先读取 train.txt / val.txt；若不存在则使用内置应急小语料
- 可选梯度检查点（--grad_ckpt）降低显存
启动方式：见文末 deepspeed 命令
"""

import os
import argparse
import math
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

# 可选 LoRA
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_dataset(train_path: str, val_path: str, fallback: bool = True):
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_lines = read_lines(train_path)
        val_lines = read_lines(val_path)
    elif fallback:
        # 内置应急小语料（仅演示用，建议换成你的真实数据）
        train_lines = [
            "DeepSpeed ZeRO-3 enables training large language models by partitioning parameters.",
            "This is a tiny demo dataset. Replace it with your own train.txt.",
            "Causal language modeling trains the model to predict the next token.",
        ] * 200
        val_lines = [
            "ZeRO-3 partitions optimizer states, gradients, and parameters across GPUs.",
            "Validation sentences help track overfitting."
        ] * 10
    else:
        raise FileNotFoundError("train.txt / val.txt 不存在且未开启fallback。")

    return Dataset.from_dict({"text": train_lines}), Dataset.from_dict({"text": val_lines})


def tokenize_function(examples: Dict[str, List[str]], tokenizer, block_size: int = 512):
    # 拼接后切块：简单实用的 CLM 预处理
    text = "\n\n".join(examples["text"])
    toks = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    # 切块
    blocks = []
    for i in range(0, len(toks) - block_size + 1, block_size):
        blocks.append(toks[i:i+block_size])
    return {"input_ids": blocks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--train_file", type=str, default="train.txt")
    parser.add_argument("--val_file", type=str, default="val.txt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default="ds_zero3.json")  # 指向 ds 配置
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true", help="开启梯度检查点节省显存")

    # LoRA / QLoRA 选项
    parser.add_argument("--use_lora", action="store_true", help="启用 LoRA（需安装 peft）")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="c_attn,c_proj", help="逗号分隔，匹配名字包含的线性层")
    parser.add_argument("--lora_8bit", action="store_true", help="QLoRA: 8-bit 权重加载 + LoRA（需 bitsandbytes 可用）")

    args = parser.parse_args()
    set_seed(args.seed)

    # ====== Tokenizer ======
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # GPT 系列通常没有 pad_token

    # ====== Model ======
    if args.lora_8bit:
        # QLoRA 路径：8bit 权重 + LoRA，显存极省
        # 需要 bitsandbytes；若环境无 bnb，可切换成 4bit（load_in_4bit=True）
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto"
        )
        if PEFT_AVAILABLE:
            model = prepare_model_for_kbit_training(model)
        else:
            raise RuntimeError("想用 QLoRA 但未安装 peft。请 pip install peft")
    else:
        # 常规全精/混合精度加载
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # LoRA（不论是否 8bit，都可以套用）
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("未检测到 peft，请 pip install peft")
        target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ====== Data ======
    train_ds_raw, val_ds_raw = build_dataset(args.train_file, args.val_file, fallback=True)

    # map 需要 batched=False，传字典
    train_tok = train_ds_raw.map(
        lambda examples: tokenize_function(examples, tokenizer, args.block_size),
        batched=False, remove_columns=["text"]
    )
    val_tok = val_ds_raw.map(
        lambda examples: tokenize_function(examples, tokenizer, args.block_size),
        batched=False, remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ====== Training Args ======
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        deepspeed=args.deepspeed,  # <- 挂载 ZeRO-3 配置
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,  # ZeRO/DP 下推荐关掉
        report_to="none"
    )

    # ====== Trainer ======
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    # 简单评估：困惑度
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
    print("Eval metrics:", metrics)

    # 保存
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
