# -*- coding: utf-8 -*-
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
)

# ====================== 基础设置 ======================
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
os.environ.setdefault("PYTORCH_USE_FLASH_ATTENTION", "0")
torch.set_float32_matmul_precision("high")

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ====================== 偏好数据（小合成） ======================
def build_synthetic_preference_dataset(n: int = 100) -> Dataset:
    prompts, chosen, rejected = [], [], []
    for i in range(n):
        q = f"请用两三句话解释 Transformer 的注意力机制。编号{i}。"
        good = "注意力通过计算查询与键的相似度，对值进行加权求和以聚合上下文信息，从而建模长距离依赖。"
        bad = "注意力就是把所有词相加，没有权重，与上下文无关。"
        prompts.append(q); chosen.append(good); rejected.append(bad)
    return Dataset.from_dict({"prompt":prompts,"chosen":chosen,"rejected":rejected})

# ====================== RM 加载（用你已训练好的） ======================
def load_trained_reward_model(rm_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    rm = AutoModelForSequenceClassification.from_pretrained(
        rm_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device).eval()
    return rm, tok

# ====================== PPO 组件 ======================
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
    temperature: float = 0.9
    top_p: float = 0.95
    grad_clip: float = 1.0
    use_grad_ckpt: bool = False
    amp_dtype: str = "bf16"  # one of ["bf16","fp16","none"]

def _rope_full_dim_config(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    setattr(cfg, "rotary_dim", head_dim)
    if hasattr(cfg, "partial_rotary_factor"): cfg.partial_rotary_factor = 1.0
    if hasattr(cfg, "rope_scaling"): cfg.rope_scaling = None
    if hasattr(cfg, "use_dynamic_ntk"): cfg.use_dynamic_ntk = False
    if hasattr(cfg, "rope_theta_1"): cfg.rope_theta_1 = None
    if hasattr(cfg, "num_key_value_heads") and cfg.num_key_value_heads is None:
        cfg.num_key_value_heads = cfg.num_attention_heads
    return cfg

def build_policy_and_ref(cfg: PPOCfg, tok: AutoTokenizer, device: torch.device):
    base_cfg = _rope_full_dim_config(cfg.model_name)
    policy_base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, config=base_cfg, attn_implementation="eager", trust_remote_code=True
    ).to(device)

    if cfg.use_grad_ckpt and hasattr(policy_base, "gradient_checkpointing_enable"):
        policy_base.gradient_checkpointing_enable()
        policy_base.config.use_cache = False

    policy = CausalLMWithValueHead(policy_base)
    policy.resize_token_embeddings(len(tok))

    # ✅ 关键：把整个包装器（含 value_head）搬到同一 device & dtype
    policy = policy.to(device)
    # （可选）再确保 value_head 的 dtype 与 lm_head 一致
    try:
        policy.value_head.to(dtype=policy_base.lm_head.weight.dtype, device=device)
    except Exception:
        policy.value_head.to(device=device)  # 兜底

    ref = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, config=base_cfg, attn_implementation="eager", trust_remote_code=True
    ).to(device)
    ref.resize_token_embeddings(len(tok))
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
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=0,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    return input_ids, attn_mask, gen  # gen: [B, L_prompt + L_resp]（已右侧 pad）

@torch.no_grad()
def compute_reward_with_rm(
    rm, tok, prompts, responses, max_len: int, device: torch.device
) -> torch.Tensor:
    """
    兼容两种RM：
    1) 分类头（AutoModelForSequenceClassification，期望 num_labels=1）
    2) LM+ValueHead（返回 out.logits, values）
    """
    templated = [f"问题: {p}\n回答:" for p in prompts]
    enc = tok(templated, responses, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    enc["attention_mask"] = enc["attention_mask"].to(torch.bool)

    # 路径1：分类头 RM
    try:
        logits = rm(**enc).logits  # 期望 [B,1] 或 [B,2]
        if logits.ndim == 2 and logits.size(-1) == 1:
            return logits.squeeze(-1)  # [B]
        elif logits.ndim == 2 and logits.size(-1) == 2:
            # 若真的是2列，这里选择第0列作为分数；也可改成差值 logits[:,0]-logits[:,1]
            return logits[:, 0]
        else:
            raise RuntimeError(f"Unexpected classification RM logits shape: {tuple(logits.shape)}")
    except Exception:
        pass  # 可能 rm 不是分类模型，而是 LM+ValueHead

    # 路径2：LM+ValueHead RM（形如 forward 返回 {"values": [B,L]} 或 (logits, values)）
    out = rm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    # 兼容两种返回形式：字典 or tuple
    values = out["values"] if isinstance(out, dict) and "values" in out else out[1]
    # 最后一个非PAD位置
    last_idx = enc["attention_mask"].sum(dim=-1) - 1   # [B]
    scores = values.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # [B]
    return scores


def autocast_context(amp_dtype: str):
    if amp_dtype == "bf16" and (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if amp_dtype == "fp16" and torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=torch.float16)
    # MPS/CPU 或关闭 AMP
    class _Dummy:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    return _Dummy()

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
    full_mask = full_ids.ne(tok.pad_token_id)
    resp_mask = torch.zeros_like(full_ids, dtype=torch.bool)
    resp_mask[:, L_prompt:] = full_ids[:, L_prompt:].ne(tok.pad_token_id)

    # ⚠️ 2) 第一次整批前向：只用于构造 old_logp / KL / advantages baseline
    #    —— 完全关闭梯度，避免后面 backward 触发“二次反传”
    with torch.no_grad():
        logits_pi_nom, values_nom = policy(full_ids, attention_mask=full_mask)   # [B,L,V], [B,L]
        logits_ref = ref(full_ids, attention_mask=full_mask).logits              # [B,L,V]

        # labels、logprob、KL
        labels = full_ids.clone()
        labels[:, :-1] = full_ids[:, 1:]
        labels[:, -1] = tok.eos_token_id

        logp_pi  = gather_logprobs(logits_pi_nom,  labels)  # [B, L]
        logp_ref = gather_logprobs(logits_ref,     labels)  # [B, L]
        kl_per_tok = (logp_pi - logp_ref)
        kl_mean = masked_mean(kl_per_tok.abs(), resp_mask)

        # 奖励（序列级） - KL 惩罚
        texts_resp = tok.batch_decode(full_ids[:, L_prompt:], skip_special_tokens=True)
        scores = compute_reward_with_rm(rm, tok, prompts, texts_resp,
                                        max_len=cfg.max_prompt_len, device=device)  # [B]
        resp_lens = resp_mask.sum(dim=1).clamp_min(1)
        kl_penalty = cfg.kl_beta * (kl_per_tok * resp_mask).sum(dim=1) / resp_lens  # [B]
        final_rewards = scores - kl_penalty                                         # [B]

        # 构造 advantages/returns（把序列级奖励抹到 response tokens）
        advantages = torch.zeros_like(logp_pi)   # [B, L]
        returns    = torch.zeros_like(values_nom)# [B, L]
        for b in range(B):
            m = resp_mask[b]
            if m.any():
                adv_scalar = (final_rewards[b] - values_nom[b, m].mean())
                advantages[b, m] = adv_scalar.expand_as(values_nom[b, m])
                returns[b, m]    = final_rewards[b].expand_as(values_nom[b, m])

        # on-policy：old_logp 用“无梯度”的 logp
        old_logp = logp_pi.clone()  # [B, L]，已无梯度

    # 3) PPO 更新（真正的梯度只来自这里的小批重算）
    idx = torch.arange(B, device=device)
    total_pi = total_v = total_ent = 0.0
    steps = 0
    for _ in range(cfg.ppo_epochs):
        perm = idx[torch.randperm(B)]
        for s in range(0, B, cfg.mini_batch_size):
            mb = perm[s: s + cfg.mini_batch_size]
            mb_mask   = resp_mask[mb]
            if mb_mask.sum() == 0: 
                continue

            mb_full   = full_ids[mb]
            mb_labels = labels[mb]
            # ✅ detach 这些“目标量”，避免把图串回去
            mb_oldlogp= old_logp[mb].detach()
            mb_adv    = advantages[mb].detach()
            mb_ret    = returns[mb].detach()

            mb_logits, mb_values = policy(mb_full, attention_mask=mb_full.ne(tok.pad_token_id))
            mb_logp = gather_logprobs(mb_logits, mb_labels)

            ratio     = torch.exp(mb_logp - mb_oldlogp)
            ratio     = torch.where(mb_mask, ratio, torch.ones_like(ratio))
            unclipped = ratio * mb_adv
            clipped   = torch.clamp(ratio, 1-cfg.clip_range, 1+cfg.clip_range) * mb_adv
            policy_loss = -masked_mean(torch.min(unclipped, clipped), mb_mask)

            value_loss  = masked_mean((mb_values - mb_ret) ** 2, mb_mask)

            probs = F.softmax(mb_logits, dim=-1)
            ent   = masked_mean(-(probs * torch.log(probs + 1e-12)).sum(-1), mb_mask)

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
        "value_loss":  total_v  / max(1, steps),
        "entropy":     total_ent/ max(1, steps),
    }
    return stats


# ====================== 主流程 ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gen-max-new", type=int, default=64)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rm-path", type=str, default="outputs/reward_model/checkpoint-10",
                        help="已训练奖励模型目录（AutoModelForSequenceClassification+tokenizer）")
    parser.add_argument("--rm-max-len", type=int, default=512)
    parser.add_argument("--amp", type=str, default="bf16", choices=["bf16","fp16","none"])
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # 1) 加载已训练 RM
    assert os.path.isdir(args.rm_path), f"--rm-path 不存在: {args.rm_path}"
    rm, tok = load_trained_reward_model(args.rm_path, device)

    # 2) PPO 配置
    cfg = PPOCfg(
        model_name=args.model,
        lr=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        gen_max_new_tokens=args.gen_max_new,
        kl_beta=args.kl_beta,
        max_prompt_len=256,
        temperature=args.temperature,
        top_p=args.top_p,
        use_grad_ckpt=args.grad_ckpt,
        amp_dtype=args.amp,
    )
    policy, ref, optim = build_policy_and_ref(cfg, tok, device)

    if args.dry_run:
        print("✅ 已加载 RM 与策略/参考模型（dry-run 不执行 PPO）")
        return

    # 3) 准备一批 prompt（演示用）
    raw = build_synthetic_preference_dataset(args.num_samples)
    prompts = raw["prompt"][:max(1, args.batch_size * 4)]

    # 4) PPO 训练循环
    for epoch in range(args.ppo_epochs):
        for i in range(0, len(prompts), cfg.batch_size):
            batch_prompts = prompts[i:i+cfg.batch_size]
            stats = ppo_step(policy, ref, optim, tok, rm, batch_prompts, cfg, device)
            print(json.dumps({"epoch": epoch, "step": i//cfg.batch_size, **stats}, ensure_ascii=False))

    # 5) 保存策略（基座 + value_head）
    out = Path("outputs/ppo_policy"); out.mkdir(parents=True, exist_ok=True)
    policy.base.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    torch.save(policy.value_head.state_dict(), out / "value_head.pt")
    print("✅ PPO policy saved to:", out)

if __name__ == "__main__":
    main()
