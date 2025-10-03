"""课程实验 3：基于人类反馈的奖励模型训练

本脚本演示如何读取成对偏好数据 (prompt, chosen, rejected)，
对 Qwen3 模型进行奖励头微调，并在训练过程中评估 Kendall Tau 指标。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import evaluate
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.8B-Instruct"
DEFAULT_DATASET = "lvwerra/stack-exchange-paired"


@dataclass
class ScriptArguments:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET
    subset: str | None = None
    output_dir: str = "./outputs/reward_model"
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_batch_size: int = 2
    max_length: int = 512


def load_preference_dataset(args: ScriptArguments) -> Dataset:
    if Path(args.dataset_name).exists():
        dataset = Dataset.load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name, args.subset, split="train[:1%]")
    return dataset.select_columns(["question", "response_j", "response_k", "quality_j", "quality_k"])


def preprocess(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _encode(batch):
        prompt = [f"问题: {q}\n回答:" for q in batch["question"]]
        chosen = batch["response_j"]
        rejected = batch["response_k"]

        chosen_enc = tokenizer(prompt, chosen, truncation=True, padding="max_length", max_length=max_length)
        rejected_enc = tokenizer(prompt, rejected, truncation=True, padding="max_length", max_length=max_length)

        stacked_input_ids = np.stack([chosen_enc["input_ids"], rejected_enc["input_ids"]], axis=1)
        stacked_attention = np.stack([chosen_enc["attention_mask"], rejected_enc["attention_mask"]], axis=1)
        return {"input_ids": stacked_input_ids, "attention_mask": stacked_attention}

    return dataset.map(_encode, batched=True, remove_columns=dataset.column_names)


def make_trainer(args: ScriptArguments, tokenized: Dataset, tokenizer: AutoTokenizer) -> Trainer:
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    kendall = evaluate.load("kendalltau")

    def compute_metrics(eval_pred):
        logits = torch.tensor(eval_pred.predictions).view(-1, 2)
        chosen, rejected = logits[:, 0], logits[:, 1]
        score_diff = (chosen - rejected).detach().cpu().numpy()
        labels = np.ones_like(score_diff)
        tau = kendall.compute(predictions=score_diff, references=labels)["kendalltau"]
        win_rate = (score_diff > 0).mean().item()
        return {"kendall_tau": tau, "win_rate": win_rate}

    def compute_loss(model, inputs, return_outputs=False):
        input_ids = torch.tensor(inputs["input_ids"])
        attention_mask = torch.tensor(inputs["attention_mask"])
        outputs = model(
            input_ids=input_ids.view(-1, input_ids.size(-1)),
            attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
        )
        logits = outputs.logits.view(-1, 2)
        loss = -torch.nn.functional.logsigmoid(logits[:, 0] - logits[:, 1]).mean()
        return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="no",
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized.select(range(min(200, len(tokenized)))),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        compute_loss_func=compute_loss,
    )


def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="奖励模型训练脚本")
    parser.add_argument("--model", dest="model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", dest="dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--output", dest="output_dir", type=str, default="./outputs/reward_model")
    parser.add_argument("--lr", dest="learning_rate", type=float, default=5e-6)
    parser.add_argument("--epochs", dest="num_train_epochs", type=int, default=1)
    parser.add_argument("--batch", dest="per_device_batch_size", type=int, default=2)
    parser.add_argument("--max-length", dest="max_length", type=int, default=512)
    parsed = parser.parse_args()
    return ScriptArguments(**vars(parsed))


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_preference_dataset(args)
    tokenized = preprocess(dataset, tokenizer, args.max_length)
    trainer = make_trainer(args, tokenized, tokenizer)
    trainer.train()
