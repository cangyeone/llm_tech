"""LLaMA-7B QLoRA 微调流程脚手架。

功能：
- 加载偏好或指令数据集
- 使用 bitsandbytes 进行 4bit 量化加载
- 配置 `peft` LoRA 并结合 `Trainer` 训练

该脚本可直接用于课堂演示 QLoRA 的完整流程。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class QLoRAConfig:
    data_path: Path
    model_name: str = "Qwen/Qwen3-4b"
    output_dir: Path = Path("./outputs/llama_qlora")
    batch_size: int = 1
    gradient_accumulation: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 100


def parse_args() -> QLoRAConfig:
    parser = argparse.ArgumentParser(description="LLaMA-7B QLoRA 微调")
    parser.add_argument("--data", type=Path, default="outputs/examples_translation_en_zh.jsonl", help="指令 JSONL 数据路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4b")
    parser.add_argument("--output", type=Path, default=Path("./outputs/llama_qlora"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=16)
    args = parser.parse_args()
    return QLoRAConfig(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        learning_rate=args.lr,
        max_steps=args.steps,
        batch_size=args.batch,
        gradient_accumulation=args.grad_acc,
    )


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
    records = read_jsonl(config.data_path)
    dataset = build_dataset(records)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    tokenized_train = tokenize(dataset["train"], tokenizer)
    tokenized_eval = tokenize(dataset["test"], tokenizer)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
        report_to=[],
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


if __name__ == "__main__":
    main()
