"""课程实验 4：DPO 数学推导与代码实现

该脚本实现了 Direct Preference Optimization (DPO) 训练流程，
并在注释中给出关键公式，便于教学推导。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.8B-Instruct"
DEFAULT_DATASET = "anthropic/hh-rlhf"


def build_dataset(dataset_name: str, split: str = "train[:1%]") -> Dataset:
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.rename_columns({"chosen": "chosen", "rejected": "rejected"})
    return dataset.select_columns(["prompt", "chosen", "rejected"])


@dataclass
class DPOArguments:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET
    beta: float = 0.1  # 对应公式中的温度系数
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    max_length: int = 512


class DPOMathNotes:
    r"""辅助打印数学公式的类。

    DPO 的目标函数：

    .. math::
        \mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(x, y^+, y^-)} \left[ \log \sigma\left( \beta \left(\log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x)\right) \right) \right]

    其中 \(\beta\) 控制对 KL 项的软约束强度。
    """

    @staticmethod
    def summary(beta: float) -> str:
        return (
            "DPO 优化目标: 最大化正样本与负样本对数似然差距。"\
            f" 当前 beta={beta}, 温度越大对偏好差异的放大越强。"
        )


def train_dpo(args: DPOArguments) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(args.dataset_name)

    def preprocess(batch: Dict[str, str]) -> Dict[str, str]:
        return {
            "prompt": batch["prompt"],
            "chosen": batch["chosen"],
            "rejected": batch["rejected"],
        }

    dataset = dataset.map(preprocess)

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    config = DPOConfig(
        beta=args.beta,
        learning_rate=args.learning_rate,
        max_prompt_length=args.max_length,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        remove_unused_columns=False,
    )

    print(DPOMathNotes.summary(args.beta))

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_policy,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("./outputs/dpo_policy")
    tokenizer.save_pretrained("./outputs/dpo_policy")


def parse_args() -> DPOArguments:
    parser = argparse.ArgumentParser(description="DPO 训练脚本")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parsed = parser.parse_args()
    return DPOArguments(
        model_name=parsed.model,
        dataset_name=parsed.dataset,
        beta=parsed.beta,
        learning_rate=parsed.lr,
        num_train_epochs=parsed.epochs,
        per_device_train_batch_size=parsed.batch,
        max_length=parsed.max_length,
    )


if __name__ == "__main__":
    args = parse_args()
    train_dpo(args)
