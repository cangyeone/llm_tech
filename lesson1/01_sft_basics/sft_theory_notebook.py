"""SFT 与指令微调基础示例。

该脚本通过最小化的伪数据演示监督微调（SFT）与指令微调（Instruction Tuning）的核心差异。
重点步骤：
1. 构建包含 `instruction`、`input`、`output` 字段的教学数据。
2. 使用 Hugging Face `AutoModelForCausalLM` 与 `Trainer` 进行一次 epoch 的微调。
3. 输出训练损失并对比不同数据格式对模型学习效果的影响。

脚本仅依赖 CPU 即可运行，若需要在 GPU 上演示可增加 `accelerate launch` 命令。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@dataclass
class FinetuneConfig:
    """微调超参数配置。"""

    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    output_dir: Path = Path("./outputs/sft_basics")
    max_steps: int = 30
    learning_rate: float = 5e-5
    batch_size: int = 2
    warmup_steps: int = 3
    instruction_mode: bool = True


EXAMPLE_PAIRS: List[Dict[str, str]] = [
    {
        "instruction": "请概括下面的段落",
        "input": "大模型在监督微调阶段会借助高质量的人类标注数据，通过最小化交叉熵损失来学习回答格式。",
        "output": "监督微调依赖人工标注数据，通过交叉熵训练模型生成符合预期的回答。",
    },
    {
        "instruction": "将下列要点改写为项目计划",
        "input": "调研数据集、清洗脏数据、训练奖励模型",
        "output": "项目计划：1）完成数据调研；2）执行数据清洗；3）训练奖励模型并评估。",
    },
]


def build_dataset(config: FinetuneConfig) -> Dataset:
    """根据配置生成 SFT 或指令微调数据集。"""

    if config.instruction_mode:
        LOGGER.info("使用 instruction-tuning 数据格式")
        texts = [
            f"指令：{pair['instruction']}\n输入：{pair['input']}\n回答：{pair['output']}"
            for pair in EXAMPLE_PAIRS
        ]
    else:
        LOGGER.info("使用传统 SFT 文本拼接格式")
        texts = [f"问题：{pair['input']}\n回答：{pair['output']}" for pair in EXAMPLE_PAIRS]

    return Dataset.from_dict({"text": texts})


def tokenize(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """对文本执行分词，截断到 512 token。"""

    return dataset.map(
        lambda sample: tokenizer(
            sample["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        ),
        batched=True,
        remove_columns=["text"],
    )


def run_training(config: FinetuneConfig) -> Dict[str, float]:
    """执行一次最小化监督微调流程。"""

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = build_dataset(config)
    tokenized_ds = tokenize(raw_dataset, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        logging_steps=5,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    LOGGER.info("开始训练：instruction_mode=%s", config.instruction_mode)
    trainer.train()
    metrics = trainer.evaluate(tokenized_ds)
    LOGGER.info("训练完成，损失指标：%s", metrics)
    return metrics


if __name__ == "__main__":
    config = FinetuneConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 先运行指令微调格式
    instruction_metrics = run_training(config)

    # 再运行传统 SFT 格式做对比
    config.instruction_mode = False
    sft_metrics = run_training(config)

    LOGGER.info(
        "两种格式的损失对比：instruction=%.4f, sft=%.4f",
        instruction_metrics["eval_loss"],
        sft_metrics["eval_loss"],
    )
