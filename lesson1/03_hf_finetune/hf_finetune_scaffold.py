"""基于 Hugging Face Trainer 的微调脚手架。

该脚本演示如何封装一个 Trainer，用于课堂快速搭建监督微调实验：
- 加载 JSONL 格式的 instruction 数据
- 自动切分训练/验证集
- 提供命令行参数以调整学习率、批大小等超参数

运行示例：
```bash
python hf_finetune_scaffold.py --data ./examples.jsonl --model Qwen/Qwen1.5-0.5B
```
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class CliConfig:
    """命令行参数配置。"""

    data_path: Path
    model_name: str
    output_dir: Path
    learning_rate: float
    batch_size: int
    num_train_epochs: int


def parse_args() -> CliConfig:
    parser = argparse.ArgumentParser(description="Hugging Face 微调脚手架")
    parser.add_argument("--data", type=Path, default="outputs/examples_translation_en_zh.jsonl", help="JSONL 数据文件路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4b", help="模型名称，例如 Qwen/Qwen3-4b")
    parser.add_argument("--output", type=Path, default=Path("./outputs/hf_trainer"))
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--batch", type=int, default=2, help="每卡 batch size")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    args = parser.parse_args()
    return CliConfig(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        learning_rate=args.lr,
        batch_size=args.batch,
        num_train_epochs=args.epochs,
    )


def load_dataset(path: Path) -> Dataset:
    """读取 JSONL 数据并转换为 `datasets.Dataset` 对象。"""

    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_example(example: Dict[str, str]) -> str:
    """拼接 instruction 与 input，形成模型训练文本。"""

    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    return f"### 指令:\n{instruction}\n### 输入:\n{input_text}\n### 回答:\n{output}"


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    return dataset.map(
        lambda sample: tokenizer(
            format_example(sample),
            truncation=True,
            max_length=1024,
            padding="max_length",
        ),
        remove_columns=dataset.column_names,
    )


def main() -> None:
    config = parse_args()
    dataset = load_dataset(config.data_path)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = tokenize_dataset(dataset["train"], tokenizer)
    tokenized_eval = tokenize_dataset(dataset["test"], tokenizer)

    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("验证集指标：", metrics)


if __name__ == "__main__":
    main()
