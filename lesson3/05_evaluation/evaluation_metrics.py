"""模型评估指标脚本。

计算困惑度（PPL）并生成人工评估表格模板。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    texts: List[str] = (
        "微调后的模型在客服场景下的回复应更加礼貌且结构化。",
        "请描述分布式训练中 ZeRO-3 的优势。",
    )


def compute_ppl(config: EvalConfig) -> float:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in config.texts:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].shape[-1]
            total_tokens += inputs["input_ids"].shape[-1]
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def build_human_eval_template() -> pd.DataFrame:
    records = [
        {"id": i, "prompt": f"案例 {i}", "模型回答": "", "评分(1-5)": "", "备注": ""}
        for i in range(1, 11)
    ]
    return pd.DataFrame(records)


def main() -> None:
    cfg = EvalConfig()
    template = build_human_eval_template()
    template_path = Path("./human_eval_template.csv")
    template.to_csv(template_path, index=False, encoding="utf-8-sig")
    print(f"人工评估表格已导出：{template_path}")

    try:
        ppl = compute_ppl(cfg)
        print(f"困惑度：{ppl:.2f}")
    except OSError:
        print("未下载模型，跳过困惑度计算。")


if __name__ == "__main__":
    main()
