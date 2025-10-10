# -*- coding: utf-8 -*-
"""
教程：对齐后生成结果的多样性评估（distinct-n & self-BLEU）
- 多次采样生成 -> 计算 distinct-1/2/3 和 self-BLEU
- 通过 temperature / top-k / top-p 调参观察模式坍塌风险
- 直接运行，无需传参

Run:
  pip install transformers torch
  python diversity_eval.py
"""

import re
from collections import Counter
from itertools import chain
from typing import List, Tuple
import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================== 可改参数 ==================
MODEL_NAME = "Qwen/Qwen3-0.6b"
PROMPT = "请用3-5句话解释什么是RLHF，以及它为何对客服机器人重要。"
SAMPLES = 20
MAX_NEW = 128
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50
# ============================================

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def simple_tok(s: str) -> List[str]:
    s = s.strip().lower()
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+|[^\s\w]", s, flags=re.IGNORECASE)

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def distinct_n(texts: List[str], n: int) -> float:
    all_ngrams = list(chain.from_iterable(ngrams(simple_tok(t), n) for t in texts))
    if not all_ngrams: return 0.0
    return len(set(all_ngrams)) / max(1, len(all_ngrams))

def bleu_score(hyp: List[str], refs: List[List[str]], max_n: int = 4) -> float:
    def mod_precision(h, rs, n):
        h_ngr = Counter(ngrams(h, n))
        if not h_ngr: return 0.0
        max_ref = Counter()
        for r in rs:
            max_ref |= Counter(ngrams(r, n))
        clipped = sum(min(c, max_ref[g]) for g, c in h_ngr.items())
        total = sum(h_ngr.values())
        return (clipped + 1.0) / (total + 1.0)
    hlen = len(hyp)
    rlen = min((len(r) for r in refs), default=1)
    bp = 1.0 if hlen > rlen else math.exp(1 - rlen / max(1, hlen))
    logs = []
    for n in range(1, max_n + 1):
        p = mod_precision(hyp, refs, n)
        logs.append(math.log(max(p, 1e-12)))
    return float(bp * math.exp(sum(logs) / max_n))

def self_bleu(texts: List[str], max_n: int = 4) -> float:
    toks = [simple_tok(t) for t in texts]
    if len(toks) <= 1: return 0.0
    scores = []
    for i in range(len(toks)):
        hyp = toks[i]
        refs = toks[:i] + toks[i+1:]
        scores.append(bleu_score(hyp, refs, max_n=max_n))
    return sum(scores) / len(scores)

def generate_samples(model_name: str, prompt: str, samples: int, max_new: int,
                     temp: float, top_p: float, top_k: int) -> List[str]:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, attn_implementation="eager"
    ).to(get_device()).eval()

    outs = []
    for _ in range(samples):
        enc = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=True,
            temperature=max(1e-3, temp), top_p=top_p, top_k=top_k,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        m = re.search(r"回答[:：]\s*(.*)", text, flags=re.S)
        outs.append((m.group(1) if m else text).strip())
    return outs

def main():
    device = get_device()
    print(f"Device: {device} | Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")
    print(f"Sampling: temp={TEMPERATURE} top_p={TOP_P} top_k={TOP_K} samples={SAMPLES} max_new={MAX_NEW}")

    texts = generate_samples(MODEL_NAME, PROMPT, SAMPLES, MAX_NEW, TEMPERATURE, TOP_P, TOP_K)

    d1 = distinct_n(texts, 1)
    d2 = distinct_n(texts, 2)
    d3 = distinct_n(texts, 3)
    sbleu = self_bleu(texts, max_n=4)

    print("\n=== Diversity Metrics ===")
    print(f"distinct-1: {d1:.4f}")
    print(f"distinct-2: {d2:.4f}")
    print(f"distinct-3: {d3:.4f}")
    print(f"self-BLEU : {sbleu:.4f}   (越低越分散，越高越相似)")

    tips = []
    if sbleu > 0.6:
        tips.append("self-BLEU 较高 → 输出相似：尝试增大 temperature 或增大 top_p/减小 top_k。")
    if d2 < 0.15:
        tips.append("distinct-2 较低 → 短语级多样性不足：尝试提高 temperature 或放宽 top-p。")
    if not tips:
        tips.append("多样性尚可。可在保持质量前提下微调 temperature/top-p 寻找最佳点。")

    print("\n=== Heuristic Tips ===")
    for t in tips: print("•", t)

    print("\n=== Samples (first 3) ===")
    for i, t in enumerate(texts[:3], 1):
        print(f"[{i}] {t[:400].replace('\\n', ' ')}")

if __name__ == "__main__":
    main()
