# -*- coding: utf-8 -*-
"""
最小可运行的 RLHF（不用 TRL）：
1) 构造/加载偏好数据 → 训练奖励模型（pairwise）
2) 纯 PyTorch 写 PPO（policy+value head + 冻结的 ref + KL 正则）
- 兼容 MPS：不使用 pin_memory；注意 attention_mask.bool()
- 不手动传 position_ids；Qwen 建议 attn_implementation="eager"
- 默认模型较小，按需改 DEFAULT_MODEL
"""

from __future__ import annotations
import os, json, math, random, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
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


class PairwiseTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,  # 新增：兼容新版
    ):
        ids  = inputs["input_ids"]              # [B,2,L]
        mask = inputs["attention_mask"].to(torch.bool)

        out = model(
            input_ids=ids.view(-1, ids.size(-1)),          # [B*2, L]
            attention_mask=mask.view(-1, mask.size(-1)),   # [B*2, L]
        )
        loss = pairwise_loss(out.logits)                   # pairwise: -log σ(Δ)
        return (loss, out) if return_outputs else loss


# ====================== 基础设置 ======================
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"  # 可改为 Qwen/Qwen3-1.8B 等
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

@dataclass
class RewardCfg:
    model_name: str = DEFAULT_MODEL
    max_len: int = 512
    lr: float = 5e-6
    bs: int = 2
    epochs: int = 1

def prepare_tokenizer(name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    print(tok.pad_token)
    return tok


def pairwise_loss(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B*2, 1] or [B,2,1]
    l = logits.view(-1, 2)
    chosen, rejected = l[:,0], l[:,1]
    return -torch.nn.functional.logsigmoid(chosen - rejected).mean()

def _rope_full_dim_config(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # 令 rotary_dim = head_dim，禁用 partial/scaling，避免 RoPE 维度错配
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    setattr(cfg, "rotary_dim", head_dim)
    if hasattr(cfg, "partial_rotary_factor"): cfg.partial_rotary_factor = 1.0
    if hasattr(cfg, "rope_scaling"): cfg.rope_scaling = None
    if hasattr(cfg, "use_dynamic_ntk"): cfg.use_dynamic_ntk = False
    if hasattr(cfg, "rope_theta_1"): cfg.rope_theta_1 = None
    if hasattr(cfg, "num_key_value_heads") and cfg.num_key_value_heads is None:
        cfg.num_key_value_heads = cfg.num_attention_heads
    return cfg

def train_reward_model(ds: Dataset, cfg: RewardCfg, device: torch.device) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tok = prepare_tokenizer(cfg.model_name)
    if tok.pad_token is None:
        # 对于 Qwen，直接把 pad 设为 eos（不新增词表，不需要 resize embeddings）
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id 
    tok.padding_side = "right"
    safe_max = getattr(AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True),
                    "max_position_embeddings", 2048)
    cfg.max_len = min(cfg.max_len, safe_max)
    def preprocess(batch):
        ch = tok(batch["prompt"], batch["chosen"], truncation=True, padding="max_length",
                 max_length=cfg.max_len, return_tensors="pt")
        rj = tok(batch["prompt"], batch["rejected"], truncation=True, padding="max_length",
                 max_length=cfg.max_len, return_tensors="pt")
        ids = torch.stack([ch.input_ids, rj.input_ids], dim=1)  # [B,2,L]
        am  = torch.stack([ch.attention_mask.bool(),                   # ✅ bool
                       rj.attention_mask.bool()], dim=1)

        return {"input_ids": ids, "attention_mask": am}

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    cfg_over = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    cfg_over.num_labels = 1
    cfg_over.problem_type = "regression"

    rm = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=cfg_over,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)

    # ——关键：把 pad_token_id 写到一切地方——
    rm.config.pad_token_id = tok.pad_token_id
    if hasattr(rm, "base_model") and hasattr(rm.base_model, "config"):
        rm.base_model.config.pad_token_id = tok.pad_token_id
    if hasattr(rm, "generation_config") and rm.generation_config is not None:
        rm.generation_config.pad_token_id = tok.pad_token_id

    rm.to(device)

    def compute_loss(model, inputs, return_outputs=False):
        ids = inputs.pop("input_ids")                   # [B,2,L]
        mask = inputs.pop("attention_mask").to(torch.bool)

        out = model(
            input_ids=ids.view(-1, ids.size(-1)).to(device),
            attention_mask=mask.view(-1, mask.size(-1)).to(device),
        )
        loss = pairwise_loss(out.logits)
        return (loss, out) if return_outputs else loss

    args = TrainingArguments(
        output_dir="outputs/reward_model",
        per_device_train_batch_size=cfg.bs,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
        dataloader_pin_memory=False,  # MPS 友好
    )

    trainer = PairwiseTrainer(
    model=rm,
    args=args,
    train_dataset=tokenized,
)

    trainer.train()
    return rm, tok

# ====================== 纯 PyTorch PPO ======================
class CausalLMWithValueHead(nn.Module):
    def __init__(self, base: AutoModelForCausalLM):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.value_head.weight, std=1e-2)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        h = out.hidden_states[-1]           # [B, L, H]
        values = self.value_head(h).squeeze(-1)  # [B, L]
        return out.logits, values

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)

    def resize_token_embeddings(self, n):
        return self.base.resize_token_embeddings(n)

    @property
    def config(self): return self.base.config

def tensor_bool(x): 
    return x.to(torch.bool) if x is not None else None

def gather_logprobs(logits, labels):
    logp = F.log_softmax(logits, dim=-1)      # [B, L, V]
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, L]

def masked_mean(t, mask):
    return (t * mask).sum() / (mask.sum().clamp_min(1))

@dataclass
class PPOCfg:
    model_name: str = DEFAULT_MODEL
    lr: float = 1e-6
    batch_size: int = 2
    ppo_epochs: int = 1
    mini_batch_size: int = 1
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    kl_beta: float = 0.02
    gen_max_new_tokens: int = 64
    max_prompt_len: int = 256

def load_trained_reward_model(rm_path: str, device: torch.device):
    """
    从 rm_path 加载:
    - tokenizer: AutoTokenizer
    - rm: AutoModelForSequenceClassification(num_labels=1)
    """
    tok = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    rm = AutoModelForSequenceClassification.from_pretrained(
        rm_path,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    rm.eval()
    return rm, tok

def build_policy_and_ref(cfg: PPOCfg, tok: AutoTokenizer, device: torch.device):
    policy_base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        config=_rope_full_dim_config(cfg.model_name),
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    policy = CausalLMWithValueHead(policy_base)
    policy.resize_token_embeddings(len(tok))

    ref = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        config=_rope_full_dim_config(cfg.model_name),
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)
    return policy, ref, optim

@torch.no_grad()
def generate_responses(policy: CausalLMWithValueHead, tok: AutoTokenizer, prompts: List[str], device: torch.device, cfg: PPOCfg):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_prompt_len)
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(torch.bool).to(device)
    gen = policy.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=cfg.gen_max_new_tokens,
        do_sample=True, top_k=50, top_p=0.95,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    return input_ids, attn_mask, gen  # gen = [B, L_prompt + L_resp]

@torch.no_grad()
def compute_reward_with_rm(
    rm: AutoModelForSequenceClassification,
    tok: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    enc = tok(prompts, responses, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    enc["attention_mask"] = enc["attention_mask"].to(torch.bool)

    scores = rm(**enc).logits.squeeze(-1)  # [B]
    return scores

def ppo_step(
    policy: CausalLMWithValueHead,
    ref: AutoModelForCausalLM,
    optim: torch.optim.Optimizer,
    tok: AutoTokenizer,
    rm: AutoModelForSequenceClassification,
    prompts: List[str],
    cfg: PPOCfg,
    device: torch.device,
):
    # 1) 生成
    input_ids, _, full_ids = generate_responses(policy, tok, prompts, device, cfg)
    B, L_prompt = input_ids.shape
    L_full = full_ids.size(1)
    full_mask = torch.ones_like(full_ids, dtype=torch.bool, device=device)
    resp_mask = torch.zeros_like(full_ids, dtype=torch.bool); resp_mask[:, L_prompt:] = True

    # 2) 前向
    logits_pi, values = policy(full_ids, attention_mask=full_mask)          # [B,L,V], [B,L]
    with torch.no_grad():
        logits_ref = ref(full_ids, attention_mask=full_mask).logits         # [B,L,V]

    # 3) labels、logprob、KL
    labels = full_ids.clone()
    labels[:, :-1] = full_ids[:, 1:]
    labels[:, -1] = tok.eos_token_id

    logp_pi = gather_logprobs(logits_pi, labels)     # [B, L]
    logp_ref = gather_logprobs(logits_ref, labels)   # [B, L]
    kl_per_tok = (logp_pi - logp_ref)
    kl_mean = masked_mean(kl_per_tok.abs(), resp_mask)

    # 4) 奖励（序列级） - KL 惩罚
    with torch.no_grad():
        texts_resp = tok.batch_decode(full_ids[:, L_prompt:], skip_special_tokens=True)
        scores = compute_reward_with_rm(rm, tok, prompts, texts_resp, max_len=cfg.max_prompt_len, device=device)  # [B]
        resp_lens = resp_mask.sum(dim=1).clamp_min(1)  # [B]
        kl_penalty = cfg.kl_beta * (kl_per_tok * resp_mask).sum(dim=1) / resp_lens
        final_rewards = scores - kl_penalty  # [B]

    # 5) 构造 advantages/returns（把序列级奖励均匀抹到 response tokens）
    advantages = torch.zeros_like(logp_pi)
    returns = torch.zeros_like(values)
    for b in range(B):
        m = resp_mask[b]
        if m.any():
            adv_b = (final_rewards[b] - values[b, m].detach().mean())
            advantages[b, m] = adv_b
            returns[b, m] = final_rewards[b]

    # 6) on-policy：old_logp 即当前 logp
    old_logp = logp_pi.detach()

    # 7) PPO 更新
    idx = torch.arange(B, device=device)
    total_pi, total_v, total_ent = 0.0, 0.0, 0.0
    steps = 0
    for _ in range(cfg.ppo_epochs):
        perm = idx[torch.randperm(B)]
        for s in range(0, B, cfg.mini_batch_size):
            mb = perm[s: s + cfg.mini_batch_size]
            mb_mask = resp_mask[mb]
            if mb_mask.sum() == 0: continue

            mb_full = full_ids[mb]
            mb_labels = labels[mb]
            mb_oldlogp = old_logp[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]

            mb_logits, mb_values = policy(mb_full, attention_mask=torch.ones_like(mb_full, dtype=torch.bool))
            mb_logp = gather_logprobs(mb_logits, mb_labels)

            ratio = torch.exp((mb_logp - mb_oldlogp))
            ratio = torch.where(mb_mask, ratio, torch.ones_like(ratio))           # 非响应段置 1
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1-cfg.clip_range, 1+cfg.clip_range) * mb_adv
            policy_loss = -masked_mean(torch.min(unclipped, clipped), mb_mask)

            value_loss = masked_mean((mb_values - mb_ret) ** 2, mb_mask)

            probs = F.softmax(mb_logits, dim=-1)
            ent = masked_mean(-(probs * torch.log(probs + 1e-12)).sum(-1), mb_mask)

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

            total_pi += float(policy_loss.item())
            total_v  += float(value_loss.item())
            total_ent+= float(ent.item())
            steps += 1

    stats = {
        "mean_reward": float(final_rewards.mean().item()),
        "kl_mean": float(kl_mean.item()),
        "policy_loss": total_pi / max(1, steps),
        "value_loss": total_v  / max(1, steps),
        "entropy": total_ent   / max(1, steps),
    }
    return stats

# ====================== 主流程 ======================
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
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # 1) 偏好数据 & RM
    raw = build_synthetic_preference_dataset(args.num_samples)
    reward_train = raw.train_test_split(test_size=0.1, seed=args.seed)["train"]
    rm, tok = train_reward_model(reward_train, RewardCfg(model_name=args.model, epochs=args.rm_epochs), device)

    if args.dry_run:
        print("✅ 奖励模型训练完成（dry-run，未执行 PPO）。")
        return

    # 2) PPO
    cfg = PPOCfg(
        model_name=args.model,
        lr=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        gen_max_new_tokens=args.gen_max_new,
        kl_beta=args.kl_beta,
        max_prompt_len=256,
    )
    policy, ref, optim = build_policy_and_ref(cfg, tok, device)

    prompts = raw["prompt"][:256]  # 小子集演示
    for epoch in range(args.ppo_epochs):
        for i in range(0, len(prompts), cfg.batch_size):
            batch_prompts = prompts[i:i+cfg.batch_size]
            stats = ppo_step(policy, ref, optim, tok, rm, batch_prompts, cfg, device)
            print(json.dumps({"epoch": epoch, "step": i//cfg.batch_size, **stats}, ensure_ascii=False))

    out = Path("outputs/ppo_policy"); out.mkdir(parents=True, exist_ok=True)
    # 保存策略（仅 base）与 value_head
    policy.base.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    torch.save(policy.value_head.state_dict(), out / "value_head.pt")
    print("✅ PPO policy saved to:", out)

if __name__ == "__main__":
    main()
