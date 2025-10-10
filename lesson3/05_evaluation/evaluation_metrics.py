# -*- coding: utf-8 -*-
"""
eval_ppl_and_human_table.py

功能：
1) 计算语言模型困惑度（PPL），支持长文本滑窗、GPU 优先、pad_token 处理。
2) 生成带“实例题目”的人工评估表 CSV（不依赖 pandas）。

用法示例：
python eval_ppl_and_human_table.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --out_csv human_eval_template.csv \
  --texts_file texts.txt

可选：提供 --prompts_file prompts.txt（每行一个题目），会写入 CSV 的 prompt 列。
"""

from __future__ import annotations
import os
import math
import csv
import argparse
from dataclasses import dataclass
from typing import List, Iterable, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================
# 配置对象
# =============================
@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen3-4b"
    # 若未提供 --texts_file，则使用该默认文本作为 PPL 样例
    texts: List[str] = None
    # 滑窗评估参数
    max_length: int = 1024     # 单窗口 token 上限
    stride: int = 768          # 窗口滑动步长（重叠 = max_length - stride）
    device: Optional[str] = None  # "cuda" / "mps" / "cpu"；默认自动选择
    dtype: Optional[str] = None    # "fp16"/"bf16"/"fp32"；默认自动

    def __post_init__(self):
        if self.texts is None:
            self.texts = [
                "请简述 ZeRO-3 在分布式训练中的参数/梯度/优化器状态分片与通信流程，并给出优缺点。",
                "请把这段用户投诉改写为礼貌且结构化的客服回复：‘快递一直没到，耽误我事了’。",
                "解释困惑度（Perplexity, PPL）的定义、与交叉熵的关系，并给一个数值例子。"
            ]


# =============================
# 工具函数
# =============================
def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_torch_dtype(dtype: Optional[str]):
    if dtype is None:
        return None
    dtype = dtype.lower()
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    return None


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


# =============================
# PPL 计算（滑窗）
# =============================
@torch.no_grad()
def compute_text_nll_sliding(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_length: int = 1024,
    stride: int = 768,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    对单条长文本用滑窗方式计算总 NLL（负对数似然）。返回该文本的 token 加权和的 NLL。
    """
    # 编码整条文本
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    n_tokens = input_ids.size(1)

    # 长度不超过 max_length：直接一次性计算
    if n_tokens <= max_length:
        out = model(input_ids=input_ids, labels=input_ids)
        # loss 是平均 NLL（按 token），乘 token 数得到总 NLL
        return out.loss.item() * n_tokens

    # 滑窗评估：保持因果训练的一致性
    total_nll = 0.0
    start = 0
    while start < n_tokens:
        end = min(start + max_length, n_tokens)
        window_ids = input_ids[:, start:end]

        # 在窗口内计算 NLL
        out = model(input_ids=window_ids, labels=window_ids)
        window_len = end - start
        total_nll += out.loss.item() * window_len

        if end == n_tokens:
            break
        start = end - (max_length - stride)  # 重叠部分确保上下文连贯
        if start < 0:
            start = 0

    return total_nll


@torch.no_grad()
def compute_ppl(
    texts: Iterable[str],
    model_name: str,
    max_length: int = 1024,
    stride: int = 768,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> float:
    """
    计算一组文本的平均困惑度（按 token 加权）。
    """
    device = auto_device() if device is None else torch.device(device)
    dtype_torch = to_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_torch,
        device_map=None  # 手动 .to(device)，避免自动分布式冲突
    ).to(device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        # 先 token 化拿到长度，避免重复 encode
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n_tok = len(ids)
        if n_tok == 0:
            continue
        nll = compute_text_nll_sliding(
            text, tokenizer, model, max_length=max_length, stride=stride, device=device
        )
        total_nll += nll
        total_tokens += n_tok

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_nll)
    return ppl


# =============================
# 人工评估 CSV（含实例题目）
# =============================
def build_human_eval_csv(
    out_csv_path: str,
    prompts: List[str],
    aspects: List[str] = ("帮助性", "事实性", "风格/礼貌", "一致性"),
    encoding: str = "utf-8-sig",
) -> None:
    """
    生成人工评估表格 CSV：含实例题目、多维度打分列。
    """
    fieldnames = ["id", "prompt", "参考答案(可选)", "模型回答", "总体评分(1-5)", "备注"]
    for a in aspects:
        fieldnames.append(f"{a}(1-5)")

    with open(out_csv_path, "w", encoding=encoding, newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for i, p in enumerate(prompts, start=1):
            row = {
                "id": i,
                "prompt": p,
                "参考答案(可选)": "",
                "模型回答": "",
                "总体评分(1-5)": "",
                "备注": "",
            }
            for a in aspects:
                row[f"{a}(1-5)"] = ""
            wr.writerow(row)


# =============================
# 主流程
# =============================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6b")
    ap.add_argument("--texts_file", type=str, default=None, help="用于 PPL 的文本文件（每行一条）。不提供则用内置样例。")
    ap.add_argument("--prompts_file", type=str, default=None, help="人工评估的题目文件（每行一个题目）。不提供则用内置样例。")
    ap.add_argument("--out_csv", type=str, default="human_eval_template.csv")

    ap.add_argument("--max_length", type=int, default=1024, help="PPL 滑窗的单窗口 token 上限")
    ap.add_argument("--stride", type=int, default=768, help="PPL 滑窗步长（重叠 = max_length - stride）")

    ap.add_argument("--device", type=str, default=None, help='"cuda" / "mps" / "cpu"，默认自动')
    ap.add_argument("--dtype", type=str, default=None, help='"fp16"/"bf16"/"fp32"，默认自动')

    return ap.parse_args()


def main():
    args = parse_args()

    # 文本 & 题目
    cfg = EvalConfig(model_name=args.model_name)
    texts = read_lines(args.texts_file) if args.texts_file and os.path.exists(args.texts_file) else cfg.texts

    default_prompts = [
        "请简述 ZeRO-3 的参数/梯度/优化器状态分片与通信流程，并给出优缺点。",
        "把这段用户投诉改写为礼貌且结构化的客服回复：‘快递一直没到，耽误我事了’。",
        "解释困惑度(PPL)的定义、与交叉熵的关系，并给一个数值例子。",
        "请将以下远震自动检测结果总结为 3 句话发给非专业用户（此处可粘贴结果）。",
    ]
    prompts = read_lines(args.prompts_file) if args.prompts_file and os.path.exists(args.prompts_file) else default_prompts

    # 生成人工评估 CSV
    build_human_eval_csv(args.out_csv, prompts)
    print(f"[OK] 人工评估 CSV 已生成：{os.path.abspath(args.out_csv)}  （可直接填写）")

    # 计算 PPL
    try:
        ppl = compute_ppl(
            texts,
            model_name=args.model_name,
            max_length=args.max_length,
            stride=args.stride,
            device=args.device,
            dtype=args.dtype,
        )
        print(f"[OK] 困惑度：{ppl:.4f}")
    except Exception as e:
        print(f"[WARN] 计算 PPL 失败：{e}\n可能是模型未下载或显存不足。你可以先仅生成 CSV，再换更小模型或减小 max_length。")


if __name__ == "__main__":
    main()
