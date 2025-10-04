"""课程实验 2：RLHF 流程串联演示

该脚本分为三部分：
1. 构造偏好数据集与分词器；
2. 奖励模型微调；
3. 使用 TRL 的 PPOTrainer 对 Qwen3 进行 RLHF 微调。

在实际教学中，可以通过 --dry-run 快速查看流程，也可以传入真实数据路径执行完整训练。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.8B-Instruct"


def build_synthetic_preference_dataset(num_samples: int = 100) -> Dataset:
    prompts, chosen, rejected = [], [], []
    for idx in range(num_samples):
        question = f"请解释 Transformer 中的注意力机制。案例编号 {idx}."
        good_answer = (
            "注意力机制通过计算查询与键的相关性, 对值进行加权求和来聚合上下文信息, "
            "从而捕获长距离依赖。"
        )
        bad_answer = "注意力只是简单的矩阵相乘, 与上下文无关。"
        prompts.append(question)
        chosen.append(good_answer)
        rejected.append(bad_answer)
    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


@dataclass
class RewardModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 512
    learning_rate: float = 5e-6
    batch_size: int = 2
    num_epochs: int = 1


@dataclass
class RLHFConfig:
    model_name: str = DEFAULT_MODEL_NAME
    learning_rate: float = 1e-6
    batch_size: int = 2
    mini_batch_size: int = 1
    max_prompt_length: int = 512
    ppo_epochs: int = 1


def prepare_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def pairwise_reward_loss(logits: torch.Tensor) -> torch.Tensor:
    chosen, rejected = logits[:, 0], logits[:, 1]
    return -torch.nn.functional.logsigmoid(chosen - rejected).mean()


def train_reward_model(dataset: Dataset, cfg: RewardModelConfig) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = prepare_tokenizer(cfg.model_name)

    def preprocess(batch: dict) -> dict:
        chosen_enc = tokenizer(batch["prompt"], batch["chosen"], truncation=True, padding="max_length", max_length=cfg.max_length, return_tensors="pt")
        rejected_enc = tokenizer(batch["prompt"], batch["rejected"], truncation=True, padding="max_length", max_length=cfg.max_length, return_tensors="pt")
        return {
            "input_ids": torch.stack([chosen_enc.input_ids, rejected_enc.input_ids], dim=1),
            "attention_mask": torch.stack([chosen_enc.attention_mask, rejected_enc.attention_mask], dim=1),
        }

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    def compute_loss(model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        outputs = model(
            input_ids=input_ids.view(-1, input_ids.size(-1)),
            attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
        )
        logits = outputs.logits.view(-1, 2)
        loss = pairwise_reward_loss(logits)
        return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir="./outputs/reward_model",
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        compute_loss_func=compute_loss,
    )
    trainer.train()
    return model, tokenizer


def build_ppo_trainer(model_name: str, tokenizer: AutoTokenizer, dataset: Dataset, cfg: RLHFConfig, reward_model: AutoModelForSequenceClassification) -> PPOTrainer:
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    policy.resize_token_embeddings(len(tokenizer))

    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    def collator(samples: List[dict]) -> dict:
        texts = [s["prompt"] for s in samples]
        tokenized = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_prompt_length, return_tensors="pt")
        return {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask, "prompts": texts}

    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=cfg.learning_rate,
        mini_batch_size=cfg.mini_batch_size,
        batch_size=cfg.batch_size,
        ppo_epochs=cfg.ppo_epochs,
        log_with="tensorboard",
    )

    def reward_fn(samples: Iterable[str], responses: Iterable[str]) -> torch.Tensor:
        batch_text = tokenizer(list(samples), list(responses), padding=True, truncation=True, max_length=cfg.max_prompt_length, return_tensors="pt").to(policy.device)
        with torch.no_grad():
            rewards = reward_model(
                input_ids=batch_text.input_ids,
                attention_mask=batch_text.attention_mask,
            ).logits.squeeze(-1)
        return rewards

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        reward_fn=reward_fn,
    )
    return ppo_trainer


def run_rlhf(cfg: RLHFConfig, num_samples: int, dry_run: bool = False) -> None:
    raw_dataset = build_synthetic_preference_dataset(num_samples=num_samples)
    reward_dataset = raw_dataset.train_test_split(test_size=0.1)["train"]

    reward_model, tokenizer = train_reward_model(reward_dataset, RewardModelConfig(model_name=cfg.model_name))

    if dry_run:
        print("完成奖励模型训练 (dry run 模式不执行 PPO)")
        return

    prompts = Dataset.from_dict({"prompt": raw_dataset["prompt"]})
    ppo_trainer = build_ppo_trainer(cfg.model_name, tokenizer, prompts, cfg, reward_model)

    for _ in range(cfg.ppo_epochs):
        stats = ppo_trainer.step()
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    save_dir = Path("./outputs/ppo_policy")
    save_dir.mkdir(parents=True, exist_ok=True)
    ppo_trainer.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLHF 流程演示")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = RLHFConfig(model_name=args.model, ppo_epochs=args.ppo_epochs)
    run_rlhf(config, num_samples=args.num_samples, dry_run=args.dry_run)
