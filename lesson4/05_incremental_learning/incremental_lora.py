"""课程实验 5：增量学习与灾难性遗忘缓解

该脚本演示如何对 Qwen3 模型使用 LoRA 进行阶段式增量学习，并结合回放样本与正则化项缓解灾难性遗忘。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.8B-Instruct"
DEFAULT_DOMAIN_A = "squad"
DEFAULT_DOMAIN_B = "gsm8k"


@dataclass
class IncrementalArgs:
    model_name: str = DEFAULT_MODEL_NAME
    domain_a: str = DEFAULT_DOMAIN_A
    domain_b: str = DEFAULT_DOMAIN_B
    lora_rank: int = 16
    replay_ratio: float = 0.2
    kl_coeff: float = 0.05
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    max_length: int = 512


def load_qa_dataset(name: str, split: str = "train[:1%]") -> Dataset:
    dataset = load_dataset(name, split=split)
    if "question" not in dataset.column_names:
        question_key = next(k for k in dataset.column_names if "question" in k)
    else:
        question_key = "question"
    answer_key = "answer" if "answer" in dataset.column_names else dataset.column_names[-1]
    return dataset.map(
        lambda ex: {
            "prompt": f"Q: {ex[question_key]}\nA:",
            "response": ex[answer_key][0] if isinstance(ex[answer_key], list) else ex[answer_key],
        }
    ).select_columns(["prompt", "response"])


def tokenize(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: Dict[str, List[str]]):
        enc = tokenizer(batch["prompt"], batch["response"], truncation=True, padding="max_length", max_length=max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def make_lora_model(model_name: str, rank: int) -> AutoModelForCausalLM:
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    lora_cfg = LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05)
    return get_peft_model(base, lora_cfg)


def kl_reg_loss(new_logits: torch.Tensor, old_logits: torch.Tensor, coeff: float) -> torch.Tensor:
    new_log_probs = torch.log_softmax(new_logits, dim=-1)
    old_log_probs = torch.log_softmax(old_logits, dim=-1)
    kl = torch.nn.functional.kl_div(new_log_probs, old_log_probs, reduction="batchmean", log_target=True)
    return coeff * kl


def train_stage(model, tokenizer, dataset: Dataset, args: IncrementalArgs, ref_model=None) -> None:
    tokenized = tokenize(dataset, tokenizer, args.max_length)

    def compute_loss(model, inputs, return_outputs=False):
        outputs = model(**{k: torch.tensor(v).to(model.device) for k, v in inputs.items()})
        loss = outputs.loss
        if ref_model is not None:
            with torch.no_grad():
                ref_outputs = ref_model(**{k: torch.tensor(v).to(ref_model.device) for k, v in inputs.items()})
            loss = loss + kl_reg_loss(outputs.logits, ref_outputs.logits, args.kl_coeff)
        return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir="./outputs/incremental",
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        compute_loss_func=compute_loss,
    )
    trainer.train()


def incremental_learning(args: IncrementalArgs) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = make_lora_model(args.model_name, args.lora_rank)

    domain_a = load_qa_dataset(args.domain_a)
    print(f"第一阶段 (领域 {args.domain_a}) 样本数: {len(domain_a)}")
    train_stage(model, tokenizer, domain_a, args)

    domain_b = load_qa_dataset(args.domain_b)
    print(f"第二阶段 (领域 {args.domain_b}) 样本数: {len(domain_b)}")
    replay_size = int(len(domain_a) * args.replay_ratio)
    replay_subset = domain_a.shuffle(seed=42).select(range(replay_size))
    combined = Dataset.from_dict(
        {
            "prompt": list(domain_b["prompt"]) + list(replay_subset["prompt"]),
            "response": list(domain_b["response"]) + list(replay_subset["response"]),
        }
    )

    frozen_ref = make_lora_model(args.model_name, args.lora_rank)
    frozen_ref.load_state_dict(model.state_dict())
    frozen_ref.eval()
    for param in frozen_ref.parameters():
        param.requires_grad_(False)

    train_stage(model, tokenizer, combined, args, ref_model=frozen_ref)
    model.save_pretrained("./outputs/incremental_lora")
    tokenizer.save_pretrained("./outputs/incremental_lora")


def parse_args() -> IncrementalArgs:
    parser = argparse.ArgumentParser(description="增量学习 LoRA 脚本")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--domain-a", type=str, default=DEFAULT_DOMAIN_A)
    parser.add_argument("--domain-b", type=str, default=DEFAULT_DOMAIN_B)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--replay", type=float, default=0.2)
    parser.add_argument("--kl", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parsed = parser.parse_args()
    return IncrementalArgs(
        model_name=parsed.model,
        domain_a=parsed.domain_a,
        domain_b=parsed.domain_b,
        lora_rank=parsed.rank,
        replay_ratio=parsed.replay,
        kl_coeff=parsed.kl,
        learning_rate=parsed.lr,
        num_train_epochs=parsed.epochs,
        per_device_batch_size=parsed.batch,
        max_length=parsed.max_length,
    )


if __name__ == "__main__":
    args = parse_args()
    incremental_learning(args)
