"""课程实验 2：Hugging Face DPO 实操

该脚本展示如何在 Hugging Face 生态中快速搭建 DPO 训练，
包含数据下载、模型加载、Trainer 配置以及 Hub 上传的关键步骤。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

DEFAULT_MODEL = "Qwen/Qwen3-1.8B-Instruct"
DEFAULT_DATASET = "Anthropic/hh-rlhf"


@dataclass
class HFArguments:
    """Hugging Face 相关参数，统一管理账号与训练设置。"""

    model_name: str = DEFAULT_MODEL
    dataset_name: str = DEFAULT_DATASET
    repo_id: str | None = None
    beta: float = 0.1
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    max_length: int = 512
    push_to_hub: bool = False


def check_login() -> None:
    """检查是否已登录 Hugging Face，如果没有则提示用户。"""

    token = HfFolder.get_token()
    if token is None:
        print("未检测到 Hugging Face 令牌，若需推送模型请先执行 `huggingface-cli login`。")


def build_dataset(dataset_name: str, split: str = "train[:1%]"):
    """加载 DPO 所需数据，并保证列名统一。"""

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.rename_columns({"chosen": "chosen", "rejected": "rejected"})
    return dataset.select_columns(["prompt", "chosen", "rejected"])


def run_training(args: HFArguments) -> None:
    """封装 DPO 训练逻辑，方便命令行调用。"""

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = args.model_name
    ref_policy = args.model_name

    config = DPOConfig(
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        max_prompt_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        logging_steps=5,
        remove_unused_columns=False,
        output_dir="outputs/qwen_dpo_hf",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.repo_id,
    )

    dataset = build_dataset(args.dataset_name)
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_policy,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    check_login()
    trainer.train()
    trainer.save_model("outputs/qwen_dpo_hf")
    tokenizer.save_pretrained("outputs/qwen_dpo_hf")

    if args.push_to_hub and args.repo_id:
        print(f"准备推送模型到 Hub 仓库：{args.repo_id}")
        api = HfApi()
        api.create_repo(args.repo_id, exist_ok=True)
        trainer.push_to_hub()


def parse_args() -> HFArguments:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Hugging Face DPO 实操")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--push-to-hub", action="store_true")
    parsed = parser.parse_args()
    return HFArguments(
        model_name=parsed.model,
        dataset_name=parsed.dataset,
        repo_id=parsed.repo_id,
        beta=parsed.beta,
        learning_rate=parsed.lr,
        num_train_epochs=parsed.epochs,
        per_device_train_batch_size=parsed.batch,
        max_length=parsed.max_length,
        push_to_hub=parsed.push_to_hub,
    )


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
