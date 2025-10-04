"""课程实验 6：多样性评估与调试

用于检查对齐后模型生成结果的多样性，通过 distinct-n、
互信息等指标帮助定位模式坍塌问题。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-1.8B-Instruct"
EVAL_PROMPTS = [
    "请介绍公司最新的会员福利。",
    "请回答客户关于退货流程的问题。",
    "请用简洁语言说明客服系统升级的好处。",
]


@dataclass
class DiversityArguments:
    """多样性调试参数。"""

    model_name: str = DEFAULT_MODEL
    sample_size: int = 3
    max_new_tokens: int = 64
    temperature: float = 0.8


def sample_responses(args: DiversityArguments) -> List[str]:
    """从模型中采样多个回答。"""

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    responses: List[str] = []
    for prompt in EVAL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_k=50,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.sample_size,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(decoded)
    return responses


def distinct_n(responses: List[str], n: int) -> float:
    """计算 distinct-n 指标。"""

    total_ngrams = 0
    unique_ngrams = set()
    for text in responses:
        tokens = text.split()
        if len(tokens) < n:
            continue
        total_ngrams += len(tokens) - n + 1
        for i in range(len(tokens) - n + 1):
            unique_ngrams.add(tuple(tokens[i : i + n]))
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams


def self_bleu(responses: List[str]) -> float:
    """计算平均自 BLEU，用于衡量回答之间的相似性。"""

    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    scores = []
    for i, hyp in enumerate(responses):
        refs = responses[:i] + responses[i + 1 :]
        if not refs:
            continue
        score = bleu.corpus_score([hyp], [refs]).score
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def analyze_diversity(responses: List[str]) -> Dict[str, float]:
    """综合 distinct 与自 BLEU 指标，输出调试建议。"""

    metrics = {
        "distinct_1": distinct_n(responses, 1),
        "distinct_2": distinct_n(responses, 2),
        "self_bleu": self_bleu(responses),
    }
    return metrics


def parse_args() -> DiversityArguments:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="多样性评估")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parsed = parser.parse_args()
    return DiversityArguments(
        model_name=parsed.model,
        sample_size=parsed.samples,
        max_new_tokens=parsed.max_new,
        temperature=parsed.temperature,
    )


def main() -> None:
    args = parse_args()
    responses = sample_responses(args)
    metrics = analyze_diversity(responses)
    print("多样性指标：")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")
    if metrics["self_bleu"] > 80:
        print("警告：自 BLEU 过高，可能存在模式坍塌。建议提高温度或引入 Top-p 采样。")


if __name__ == "__main__":
    main()
