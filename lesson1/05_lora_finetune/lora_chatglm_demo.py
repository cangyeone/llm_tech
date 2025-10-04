"""面向 ChatGLM-6B 的 LoRA 微调示例。

脚本展示：
1. 如何加载 ChatGLM-6B INT4 权重并冻结原始参数。
2. 使用 `peft` 配置 LoRA 超参数（r、alpha、dropout）。
3. 基于小规模指令数据执行微调，并保存 LoRA 适配器权重。

实际运行需具备至少 24GB 显存，本示例默认以伪造数据演示流程。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@dataclass
class LoRAFinetuneConfig:
    model_name: str = "THUDM/chatglm3-6b"
    output_dir: Path = Path("./outputs/chatglm_lora")
    r: int = 8
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 1e-4
    batch_size: int = 2
    steps: int = 50


def build_dummy_dataset() -> Dataset:
    """构造简化的指令数据集。"""

    samples: List[dict] = []
    for i in range(20):
        samples.append(
            {
                "instruction": "请回答一个关于模型微调的常见问题。",
                "input": f"第 {i} 个问题：LoRA 为什么可以降低显存开销？",
                "output": "因为 LoRA 只训练低秩矩阵，显著减少参数量并允许冻结原始模型。",
            }
        )
    return Dataset.from_list(samples)


def format_sample(sample: dict) -> str:
    return (
        f"问：{sample['instruction']}\n"
        f"补充信息：{sample['input']}\n"
        f"答：{sample['output']}"
    )


def tokenize(dataset: Dataset, tokenizer) -> Dataset:
    return dataset.map(
        lambda sample: tokenizer(
            format_sample(sample),
            max_length=768,
            truncation=True,
            padding="max_length",
        ),
        remove_columns=dataset.column_names,
    )


def main() -> None:
    config = LoRAFinetuneConfig()
    dataset = build_dummy_dataset()
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        load_in_4bit=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=["query_key_value", "dense"],
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    tokenized_train = tokenize(dataset["train"], tokenizer)
    tokenized_eval = tokenize(dataset["test"], tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        max_steps=config.steps,
        learning_rate=config.learning_rate,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=20,
        save_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    trainer.train()
    trainer.save_model(str(config.output_dir))
    LOGGER.info("LoRA 适配器已保存至 %s", config.output_dir)


if __name__ == "__main__":
    main()
