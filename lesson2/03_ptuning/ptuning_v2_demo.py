"""P-Tuning v2 可学习提示词示例。

展示内容：
1. 构建前缀向量并注入到 transformer block 输入。
2. 使用 `peft` PromptEncoder 实现可学习提示词。
3. 演示训练与推理流程的差异化调用方式。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import Dataset
from peft import PromptEncoderConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@dataclass
class PTuningConfig:
    model_name: str = "Qwen/Qwen1.5-1.8B-Chat"
    prompt_length: int = 32
    task_type: str = "CAUSAL_LM"
    output_dir: str = "./outputs/ptuning"


def build_dataset() -> Dataset:
    samples = [
        {
            "instruction": "请写一条客服问候语",
            "input": "背景：用户首次进入商城",
            "output": "您好，欢迎来到我们的商城，有任何问题随时咨询我。",
        }
        for _ in range(20)
    ]
    return Dataset.from_list(samples)


def format_sample(sample: Dict[str, str]) -> str:
    return f"指令：{sample['instruction']}\n输入：{sample['input']}\n回答：{sample['output']}"


def tokenize(dataset: Dataset, tokenizer) -> Dataset:
    return dataset.map(
        lambda sample: tokenizer(
            format_sample(sample),
            max_length=512,
            truncation=True,
            padding="max_length",
        ),
        remove_columns=dataset.column_names,
    )


def main() -> None:
    cfg = PTuningConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    prompt_config = PromptEncoderConfig(
        task_type=cfg.task_type,
        num_virtual_tokens=cfg.prompt_length,
        token_dim=base_model.config.hidden_size,
        encoder_hidden_size=base_model.config.hidden_size // 2,
    )

    model = get_peft_model(base_model, prompt_config)
    model.print_trainable_parameters()

    dataset = build_dataset().train_test_split(test_size=0.1, seed=42)
    tokenized_train = tokenize(dataset["train"], tokenizer)
    tokenized_eval = tokenize(dataset["test"], tokenizer)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=3e-4,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    trainer.train()

    # 推理阶段：注入可学习提示向量
    prompt_text = "请写一句积极的评价"
    inputs = tokenizer(prompt_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=32)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
