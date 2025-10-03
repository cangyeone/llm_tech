"""课程实验 1：Qwen 对齐流程解析

本脚本以 Qwen3 指令模型为例，串联数据准备、奖励模型训练、
策略微调与上线前的验证流程，帮助学员理解生产级对齐步骤。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

DEFAULT_MODEL = "Qwen/Qwen3-1.8B-Instruct"
DEFAULT_DATASET = "Anthropic/hh-rlhf"


@dataclass
class WorkflowArguments:
    """流程参数，用于快速调整实验规模。"""

    model_name: str = DEFAULT_MODEL
    dataset_name: str = DEFAULT_DATASET
    sample_percent: str = "train[:0.5%]"  # 小样本方便课堂演示
    learning_rate: float = 1e-5
    ppo_steps: int = 8
    max_prompt_length: int = 512


def load_preference_data(dataset_name: str, sample_percent: str) -> DatasetDict:
    """加载偏好数据集，确保包含 prompt、chosen、rejected 三列。"""

    dataset = load_dataset(dataset_name)
    train_split = dataset["train"].shuffle(seed=42).select(range(max(1, int(len(dataset["train"]) * 0.01))))
    # 将采样比例解析为 datasets 的切片格式
    sampled = load_dataset(dataset_name, split=sample_percent)
    sampled = sampled.rename_columns({"chosen": "chosen", "rejected": "rejected"})
    return DatasetDict({"train": sampled, "debug": train_split})


def train_reward_model(base_model: str, dataset: DatasetDict) -> AutoModelForCausalLMWithValueHead:
    """基于偏好数据训练价值头，用于 RLHF 中的奖励评估。"""

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # 这里只展示如何迭代数据并计算简单分数，完整奖励模型训练请参考 Lesson 4。
    for sample in dataset["train"].select(range(2)):
        prompt = sample["prompt"]
        chosen = tokenizer(prompt + sample["chosen"], return_tensors="pt", truncation=True)
        with torch.no_grad():
            score = model(chosen.input_ids).value.mean().item()
            print(f"奖励模型示例分数：{score:.4f}")

    return model


def run_ppo_alignment(policy_name: str, reward_model: AutoModelForCausalLMWithValueHead, dataset: DatasetDict, args: WorkflowArguments) -> None:
    """运行简化的 PPO 对齐循环。"""

    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    config = PPOConfig(
        learning_rate=args.learning_rate,
        log_with=None,
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=1,
        target_kl=0.1,
    )

    trainer = PPOTrainer(
        config,
        policy,
        ref_policy,
        tokenizer,
        dataset["train"],
    )

    prompts: List[str] = dataset["train"]["prompt"][: args.ppo_steps]
    for step, prompt in enumerate(prompts):
        query_tensors = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_length)
        response = trainer.generate(query_tensors["input_ids"], max_new_tokens=64)
        # 利用奖励模型为生成结果打分
        with torch.no_grad():
            reward = reward_model(response).value.mean().detach()
        stats = trainer.step(query_tensors["input_ids"], response, reward)
        print(f"PPO 步骤 {step}: KL={stats['objective/kl']:.4f}, reward={reward.item():.4f}")

    trainer.model.save_pretrained("outputs/qwen_ppo_aligned")
    tokenizer.save_pretrained("outputs/qwen_ppo_aligned")


def parse_args() -> WorkflowArguments:
    """命令行参数解析，统一入口。"""

    parser = argparse.ArgumentParser(description="Qwen 对齐流程解析")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--sample", type=str, default="train[:0.5%]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ppo-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parsed = parser.parse_args()
    return WorkflowArguments(
        model_name=parsed.model,
        dataset_name=parsed.dataset,
        sample_percent=parsed.sample,
        learning_rate=parsed.lr,
        ppo_steps=parsed.ppo_steps,
        max_prompt_length=parsed.max_length,
    )


def main() -> None:
    args = parse_args()
    dataset = load_preference_data(args.dataset_name, args.sample_percent)
    reward_model = train_reward_model(args.model_name, dataset)
    run_ppo_alignment(args.model_name, reward_model, dataset, args)


if __name__ == "__main__":
    main()
