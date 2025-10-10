# -*- coding: utf-8 -*-
"""
最小可运行的 RLHF（基于 TRL）：
1) 构造/加载偏好数据 → 训练奖励模型（pairwise，HF Trainer）
2) 使用 TRL 的 PPOTrainer（policy + value head + 冻结的 ref + 自适应 KL 控制）
- 兼容 MPS：不使用 pin_memory；不强制 fp16/bf16
- Qwen 建议 attn_implementation="eager"
- 默认模型较小，按需改 DEFAULT_MODEL
- KL 惩罚由 PPOTrainer 内部处理（kl_penalty='kl'），外部 reward 只传 RM 分数
Usage:
python rlhf_trl_min.py --model Qwen/Qwen3-0.6B --num-samples 200 --rm-epochs 1 --ppo-epochs 1 --dry-run
"""

from __future__ import annotations
import os, json, argparse, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# === TRL ===
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
)

# ====================== 基础设置 ======================
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
os.environ.setdefault("PYTORCH_USE_FLASH_ATTENTION", "0")  # 保险起见，禁 FA2

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ====================== 偏好数据 & RM 训练 ======================
def build_synthetic_preference_dataset(n: int = 100) -> Dataset:
    prompts, chosen, rejected = [], [], []
    for i in range(n):
        q = f"请用两三句话解释 Transformer 的注意力机制。编号{i}。"
        good = "注意力通过计算查询与键的相似度，对值进行加权求和以聚合上下文信息，从而建模长距离依赖。"
        bad = "注意力就是把所有词相加，没有权重，与上下文无关。"
        prompts.append(q); chosen.append(good); rejected.append(bad)
    return Dataset.from_dict({"prompt":prompts,"chosen":chosen,"rejected":rejected})

def prepare_tokenizer(name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok

def pairwise_loss(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B*2] or [B,2]
    l = logits.view(-1, 2)
    chosen, rejected = l[:, 0], l[:, 1]
    return -F.logsigmoid(chosen - rejected).mean()

class PairwiseTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        ids  = inputs["input_ids"]              # [B,2,L]
        mask = inputs["attention_mask"].to(torch.bool)
        out = model(
            input_ids=ids.view(-1, ids.size(-1)),          # [B*2, L]
            attention_mask=mask.view(-1, mask.size(-1)),   # [B*2, L]
        )
        loss = pairwise_loss(out.logits.squeeze(-1))
        return (loss, out) if return_outputs else loss

@dataclass
class RewardCfg:
    model_name: str = DEFAULT_MODEL
    max_len: int = 512
    lr: float = 5e-6
    bs: int = 2
    epochs: int = 1

def _safe_max_ctx(model_name: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return getattr(cfg, "max_position_embeddings", 2048)

def train_reward_model(ds: Dataset, cfg: RewardCfg, device: torch.device) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, int]:
    tok = prepare_tokenizer(cfg.model_name)
    cfg.max_len = min(cfg.max_len, _safe_max_ctx(cfg.model_name))

    def preprocess(batch):
        ch = tok(batch["prompt"], batch["chosen"], truncation=True, padding="max_length",
                 max_length=cfg.max_len, return_tensors="pt")
        rj = tok(batch["prompt"], batch["rejected"], truncation=True, padding="max_length",
                 max_length=cfg.max_len, return_tensors="pt")
        ids = torch.stack([ch.input_ids, rj.input_ids], dim=1)  # [B,2,L]
        am  = torch.stack([ch.attention_mask.bool(), rj.attention_mask.bool()], dim=1)
        return {"input_ids": ids, "attention_mask": am}

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    rm_cfg = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    rm_cfg.num_labels = 1
    rm_cfg.problem_type = "regression"

    rm = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=rm_cfg,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    rm.to(device)

    # pad_token_id 贯通设置
    rm.config.pad_token_id = tok.pad_token_id
    if hasattr(rm, "base_model") and hasattr(rm.base_model, "config"):
        rm.base_model.config.pad_token_id = tok.pad_token_id

    args = TrainingArguments(
        output_dir="outputs/reward_model",
        per_device_train_batch_size=cfg.bs,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
        dataloader_pin_memory=False,  # MPS 友好
        fp16=torch.cuda.is_available(),  # 仅 CUDA 才开启
    )

    trainer = PairwiseTrainer(model=rm, args=args, train_dataset=tokenized)
    trainer.train()

    return rm.eval(), tok, cfg.max_len

# ====================== PPO (基于 TRL) ======================
@dataclass
class RunCfg:
    model: str = DEFAULT_MODEL
    num_samples: int = 100
    rm_epochs: int = 1
    ppo_epochs: int = 1
    batch_size: int = 2
    mini_batch_size: int = 1
    lr: float = 1e-6
    gen_max_new: int = 64
    seed: int = 42
    dry_run: bool = False

def build_prompts_dataset(n: int) -> Dataset:
    prompts = [f"请简述注意力机制的核心思想，并举一个简单示例。编号{i}。" for i in range(n)]
    return Dataset.from_dict({"prompt": prompts})

@torch.no_grad()
def compute_rm_scores(
    rm: AutoModelForSequenceClassification,
    tok: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    max_len: int,
    device: torch.device,
    batch_size: int = 8,
) -> List[float]:
    scores: List[float] = []
    for s in range(0, len(prompts), batch_size):
        ps = prompts[s:s+batch_size]
        rs = responses[s:s+batch_size]
        enc = tok(ps, rs, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        enc["attention_mask"] = enc["attention_mask"].to(torch.bool)
        out = rm(**enc).logits.squeeze(-1)  # [B]
        scores.extend(out.detach().cpu().tolist())
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--rm-epochs", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gen-max-new", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = RunCfg(
        model=args.model,
        num_samples=args.num_samples,
        rm_epochs=args.rm_epochs,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        gen_max_new=args.gen_max_new,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    seed_everything(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # 1) 偏好数据 & 训练 RM
    raw_pref = build_synthetic_preference_dataset(cfg.num_samples)
    reward_train = raw_pref.train_test_split(test_size=0.1, seed=cfg.seed)["train"]
    rm, rm_tok, rm_max_len = train_reward_model(reward_train, RewardCfg(model_name=cfg.model, epochs=cfg.rm_epochs), device)

    if cfg.dry_run:
        print("✅ 奖励模型训练完成（dry-run，未执行 PPO）。")
        return

    # 2) PPO（策略 + 参考 + 值头）—— TRL
    # 统一 tokenizer（用 RM 训练时的 tok）
    tok = rm_tok

    # Qwen 建议 eager 注意力；AutoModelForCausalLMWithValueHead 里通过 base_model kwargs 传入更稳妥
    base_kwargs = dict(attn_implementation="eager", trust_remote_code=True)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.model, **base_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model, **base_kwargs)

    # pad_token_id 贯通设置
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    policy.config.pad_token_id = tok.pad_token_id
    if hasattr(policy, "pretrained_model") and hasattr(policy.pretrained_model, "config"):
        policy.pretrained_model.config.pad_token_id = tok.pad_token_id
    ref_model.config.pad_token_id = tok.pad_token_id

    policy.to(device); ref_model.to(device)

    # TRL 配置（KL 惩罚由内部处理：kl_penalty='kl'）
    ppo_config = PPOConfig(
        model_name=cfg.model,
        learning_rate=cfg.lr,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        optimize_cuda_cache=False,
        ppo_epochs=cfg.ppo_epochs,
        target_kl=0.1,             # 自适应 KL 目标
        kl_penalty="kl",           # 让 TRL 处理 KL
        init_kl_coef=0.02,         # 初始 KL 系数（自适应）
        seed=cfg.seed,
        accelerator_kwargs={"log_with": None},
    )

    # 组装 TRL 的 Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tok,
        # 注意：DataLoader 内部 pin_memory 由 accelerate 控制；这里不强制
    )

    # 3) 准备 “环境” prompts
    prompts_ds = build_prompts_dataset(min(256, cfg.num_samples))
    prompts = prompts_ds["prompt"]

    # 4) 训练循环：生成 → RM 打分 → PPO 更新
    gen_kwargs = dict(
        max_new_tokens=cfg.gen_max_new,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    steps = 0
    for epoch in range(cfg.ppo_epochs):
        for i in range(0, len(prompts), cfg.batch_size):
            batch_prompts = prompts[i : i + cfg.batch_size]

            # a) 生成 response（policy）
            query_tensors = [tok(p, return_tensors="pt").input_ids.squeeze(0).to(device) for p in batch_prompts]
            response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)

            # b) 解码文本
            responses = [tok.decode(r[len(q):], skip_special_tokens=True)  # 只取 response 段
                         for q, r in zip(query_tensors, response_tensors)]

            # c) 计算 RM 分数（序列级奖励）
            rewards = compute_rm_scores(
                rm=rm,
                tok=tok,
                prompts=batch_prompts,
                responses=responses,
                max_len=rm_max_len,
                device=device,
            )
            # 转为张量（每条一个标量奖励）
            rewards_tensors = [torch.tensor(r, dtype=torch.float32, device=device) for r in rewards]

            # d) TRL 的 PPO 更新（内部会加 KL 惩罚）
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensors)
            steps += 1

            # 打印关键统计
            log = {
                "epoch": epoch,
                "step": steps,
                "rm_reward_mean": float(np.mean(rewards)),
                "ppo/kl_coef": float(ppo_trainer.kl_ctl.value),
            }
            log.update({k: float(v) if isinstance(v, (int, float)) else v for k, v in stats.items() if isinstance(v, (int, float))})
            print(json.dumps(log, ensure_ascii=False))

    # 5) 保存策略（包含 value head）
    out = Path("outputs/ppo_policy_trl"); out.mkdir(parents=True, exist_ok=True)
    ppo_trainer.model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print("✅ TRL PPO policy saved to:", out)

if __name__ == "__main__":
    main()
