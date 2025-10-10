# -*- coding: utf-8 -*-
"""
Transformers 8卡训练最小示例
- 默认用 gpt2 做 Causal LM 微调
- 可直接用 torchrun 进行单机 8 卡 DDP
- 若加 --deepspeed ds_zero3.json 则启用 ZeRO-3
- 若没有 train.txt/val.txt，会自动用内置小语料演示
"""

import os
import math
from typing import List, Dict
import argparse

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def build_dataset(train_path: str, val_path: str, fallback: bool = True):
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_lines = read_lines(train_path)
        val_lines = read_lines(val_path)
    elif fallback:
        train_lines = [
            "This is a tiny demo dataset for 8-GPU training with Transformers.",
            "Replace it with your own train.txt for real experiments.",
            "Causal language modeling predicts the next token."
        ] * 300
        val_lines = [
            "Validation sentences help track generalization.",
            "This is a tiny validation set."
        ] * 20
    else:
        raise FileNotFoundError("train.txt/val.txt 不存在，且 fallback 未开启。")

    return Dataset.from_dict({"text": train_lines}), Dataset.from_dict({"text": val_lines})

def tokenize_blockwise(texts: List[str], tokenizer, block_size: int):
    text = "\n\n".join(texts)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks = [ids[i:i+block_size] for i in range(0, len(ids) - block_size + 1, block_size)]
    return {"input_ids": chunks}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--train_file", type=str, default="train.txt")
    parser.add_argument("--val_file", type=str, default="val.txt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--block_size", type=int, default=512)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_train_epochs", type=int, default=1)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # 可选：启用 ZeRO-3（传给 Trainer 的 TrainingArguments.deepspeed）
    parser.add_argument("--deepspeed", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # 构建数据
    train_raw, val_raw = build_dataset(args.train_file, args.val_file, fallback=True)
    train_tok = train_raw.map(
        lambda ex: tokenize_blockwise(ex["text"], tokenizer, args.block_size),
        batched=False, remove_columns=["text"]
    )
    val_tok = val_raw.map(
        lambda ex: tokenize_blockwise(ex["text"], tokenizer, args.block_size),
        batched=False, remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,

        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,

        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,              # ← 传入 ds_zero3.json 即启用 ZeRO-3
        ddp_find_unused_parameters=False,      # DDP/ZeRO 下建议关
        dataloader_pin_memory=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
    print("Eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
