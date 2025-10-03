"""课程实验 6：对齐数据集构建与质量分析

脚本提供了对齐数据集的清洗、打分与可视化流程，帮助学生理解高质量偏好数据的重要性。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATASET = "Dahoas/rm-static"


@dataclass
class CurationArgs:
    dataset_name: str = DEFAULT_DATASET
    split: str = "train[:1%]"
    output_csv: str | None = "./outputs/curated_pairs.csv"
    min_length: int = 32
    max_length: int = 2048
    similarity_threshold: float = 0.2


def load_pairs(args: CurationArgs) -> Dataset:
    if Path(args.dataset_name).exists():
        dataset = Dataset.load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name, split=args.split)
    columns = {
        "prompt": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    dataset = dataset.rename_columns({k: v for k, v in columns.items() if k in dataset.column_names})
    return dataset.select_columns(["prompt", "chosen", "rejected"])


def filter_by_length(dataset: Dataset, args: CurationArgs) -> Dataset:
    def _filter(example):
        lengths = [len(example["prompt"]), len(example["chosen"]), len(example["rejected"])]
        return all(args.min_length <= l <= args.max_length for l in lengths)

    return dataset.filter(_filter)


def compute_similarity_flags(dataset: Dataset, threshold: float) -> List[bool]:
    vectorizer = TfidfVectorizer(max_features=4096)
    merged = [p + "\n" + c + "\n" + r for p, c, r in zip(dataset["prompt"], dataset["chosen"], dataset["rejected"])]
    tfidf = vectorizer.fit_transform(merged)
    sims = cosine_similarity(tfidf)
    avg_sim = sims.mean(axis=1)
    return (avg_sim > threshold).tolist()


def visualize_distribution(dataset: Dataset) -> None:
    lengths = [len(text) for text in dataset["prompt"]]
    plt.hist(lengths, bins=30, color="skyblue")
    plt.title("Prompt 长度分布")
    plt.xlabel("字符数")
    plt.ylabel("数量")
    plt.tight_layout()
    plt.show()


def curate(args: CurationArgs) -> pd.DataFrame:
    dataset = load_pairs(args)
    dataset = filter_by_length(dataset, args)
    flags = compute_similarity_flags(dataset, args.similarity_threshold)
    df = dataset.to_pandas()
    df["is_redundant"] = flags
    quality_counts = Counter(flags)
    print("疑似重复/低多样性样本数:", quality_counts)

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"已导出清洗结果到 {args.output_csv}")

    try:
        visualize_distribution(dataset)
    except Exception as exc:  # noqa: BLE001 - 可视化失败不阻断流程
        print(f"可视化失败: {exc}")
    return df


def parse_args() -> CurationArgs:
    parser = argparse.ArgumentParser(description="对齐数据集构建与质量分析")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default="train[:1%]")
    parser.add_argument("--output", type=str, default="./outputs/curated_pairs.csv")
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--sim-threshold", type=float, default=0.2)
    parsed = parser.parse_args()
    return CurationArgs(
        dataset_name=parsed.dataset,
        split=parsed.split,
        output_csv=parsed.output,
        min_length=parsed.min_length,
        max_length=parsed.max_length,
        similarity_threshold=parsed.sim_threshold,
    )


if __name__ == "__main__":
    args = parse_args()
    curate(args)
