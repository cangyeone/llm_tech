"""垂直领域（文档摘要）微调案例。

流程包括：
1. 加载行业文档摘要数据
2. 配置 Trainer 微调模型
3. 通过示例 prompt 展示生成效果
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments


@dataclass
class DomainConfig:
    model_name: str = "BAAI/bge-small-zh-v1.5"
    output_dir: Path = Path("./outputs/domain_summary")


def build_dataset() -> Dataset:
    docs: List[dict] = []
    for i in range(30):
        docs.append(
            {
                "instruction": "请对以下行业报告做摘要",
                "input": f"第{i}份报告：介绍了金融合规审计流程以及关键风险点。",
                "output": "报告重点包括审计流程数字化、风险预警体系建设与合规培训。",
            }
        )
    return Dataset.from_list(docs)


def format_sample(sample: dict) -> dict:
    return {
        "input_ids": sample["instruction"] + "\n" + sample["input"],
        "labels": sample["output"],
    }


def tokenize(dataset: Dataset, tokenizer) -> Dataset:
    return dataset.map(
        lambda sample: {
            **tokenizer(sample["instruction"] + "\n" + sample["input"], truncation=True, max_length=512),
            "labels": tokenizer(sample["output"], truncation=True, max_length=256)["input_ids"],
        },
        remove_columns=dataset.column_names,
    )


def main() -> None:
    cfg = DomainConfig()
    dataset = build_dataset().train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    tokenized_train = tokenize(dataset["train"], tokenizer)
    tokenized_eval = tokenize(dataset["test"], tokenizer)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=3e-4,
        evaluation_strategy="steps",
        eval_steps=20,
        save_steps=50,
        predict_with_generate=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    trainer.train()

    prompt = "请总结以下报告：本季度客服部门引入知识库后，首次响应率提升。"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
