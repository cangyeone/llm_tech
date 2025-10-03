"""训练日志分析与性能调优示例。

解析 DeepSpeed/Accelerate 生成的日志，提取如下指标：
- 每步损失
- 每秒样本数（throughput）
- GPU 利用率估计

并通过 matplotlib 可视化趋势。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_logs(path: Path) -> pd.DataFrame:
    records: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(records)


def plot_metrics(df: pd.DataFrame) -> None:
    if df.empty:
        print("日志为空，请检查路径。")
        return

    df["step"] = range(1, len(df) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(df["step"], df["loss"], label="loss")
    axes[0].set_title("训练损失趋势")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")

    if "throughput" in df:
        axes[1].plot(df["step"], df["throughput"], color="orange", label="throughput")
        axes[1].set_title("每秒样本数")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("samples/s")

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    print("图表已保存为 training_metrics.png")


if __name__ == "__main__":
    log_path = Path("./sample_log.jsonl")
    log_path.write_text(
        "\n".join(
            json.dumps({"loss": 2.0 - i * 0.01, "throughput": 100 + i * 2})
            for i in range(50)
        ),
        encoding="utf-8",
    )
    df = load_logs(log_path)
    plot_metrics(df)
