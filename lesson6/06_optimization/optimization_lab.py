"""课程实验 6：检索召回与生成准确性优化

本脚本提供评估检索召回率、生成答案准确性的指标计算示例，
并给出实验设计建议，帮助学员持续改进 RAG 系统表现。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class EvalSample:
    """单条评估样本，包含问题、标准答案与检索文档。"""

    question: str
    reference: str
    retrieved_docs: List[str]
    generated_answer: str


def load_eval_samples(path: Path) -> List[EvalSample]:
    """从 JSONL 或 TSV 等格式加载评估样本。"""

    samples: List[EvalSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            raise ValueError("评估数据需包含问题、标准答案、检索文档、模型回答四列")
        question, reference, docs_raw, answer = parts[:4]
        retrieved_docs = docs_raw.split("<sep>")
        samples.append(EvalSample(question, reference, retrieved_docs, answer))
    return samples


def recall_at_k(sample: EvalSample, keywords: Iterable[str], k: int) -> float:
    """根据关键词判断检索片段是否覆盖关键信息。"""

    target = set(keywords)
    top_docs = sample.retrieved_docs[:k]
    hits = 0
    for doc in top_docs:
        if any(word in doc for word in target):
            hits += 1
    return hits / max(1, len(target))


def semantic_similarity(pred: str, reference: str) -> float:
    """简化版语义相似度，使用字符 n-gram Jaccard 近似。"""

    def char_shingles(text: str, n: int = 3) -> set[str]:
        return {text[i : i + n] for i in range(max(1, len(text) - n + 1))}

    pred_set = char_shingles(pred)
    ref_set = char_shingles(reference)
    if not pred_set or not ref_set:
        return 0.0
    return len(pred_set & ref_set) / len(pred_set | ref_set)


def evaluate(samples: List[EvalSample], recall_keywords: Dict[str, List[str]], k: int) -> Tuple[float, float]:
    """计算平均召回率与平均语义相似度。"""

    recalls: List[float] = []
    similarities: List[float] = []
    for sample in samples:
        keywords = recall_keywords.get(sample.question, [])
        if keywords:
            recalls.append(recall_at_k(sample, keywords, k))
        similarities.append(semantic_similarity(sample.generated_answer, sample.reference))
    recall_score = float(np.mean(recalls)) if recalls else 0.0
    sim_score = float(np.mean(similarities)) if similarities else 0.0
    return recall_score, sim_score


def optimization_recommendations() -> List[str]:
    """返回提高系统性能的策略建议。"""

    return [
        "采集真实用户问题，构建多样化测试集，避免过拟合课堂示例。",
        "对召回失败样本进行错误分析，定位分块粒度、向量质量或索引参数问题。",
        "结合 rerank 模型（如 cross-encoder）提升最终答案的相关性。",
        "引入答案事实性检测（如 GPT-4 评审）降低幻觉风险。",
        "通过 A/B 测试评估改动影响，并记录在 MLflow/W&B 中。",
    ]


def main(eval_path: Path, k: int) -> None:
    """执行评估并输出优化建议。"""

    samples = load_eval_samples(eval_path)
    recall_keywords = {sample.question: sample.reference.split()[:3] for sample in samples}
    recall_score, sim_score = evaluate(samples, recall_keywords, k=k)

    print("=== 指标结果 ===")
    print(f"Recall@{k} 平均值：{recall_score:.3f}")
    print(f"语义相似度平均值：{sim_score:.3f}")

    print("\n=== 优化建议 ===")
    for idx, tip in enumerate(optimization_recommendations(), start=1):
        print(f"{idx}. {tip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检索与生成效果评估实验")
    parser.add_argument("eval_path", type=Path, help="评估数据路径（制表符分隔四列）")
    parser.add_argument("--k", type=int, default=3, help="计算召回率的检索深度")
    args = parser.parse_args()

    main(eval_path=args.eval_path, k=args.k)
