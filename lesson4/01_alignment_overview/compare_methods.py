"""课程实验 1：对齐技术概览

本脚本以表格和雷达图的形式比较 RLHF、DPO、KTO 三种常见对齐方法。
运行后将输出 Markdown 表格，并可选生成 matplotlib 雷达图帮助课堂讲解。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AlignmentMethod:
    name: str
    optimization: str
    data_need: str
    pros: str
    cons: str
    score_profile: Dict[str, float]


METHODS: List[AlignmentMethod] = [
    AlignmentMethod(
        name="RLHF",
        optimization="强化学习 (PPO)",
        data_need="偏好对 + 参考模型 + 奖励模型",
        pros="稳定的生成质量控制、支持在线探索",
        cons="训练链路长、对算力与数据要求高",
        score_profile={
            "实现复杂度": 2.0,
            "样本效率": 3.0,
            "对齐效果": 4.5,
            "算力需求": 4.0,
            "稳定性": 3.5,
        },
    ),
    AlignmentMethod(
        name="DPO",
        optimization="直接最优策略 (无强化学习)",
        data_need="偏好对 + 参考模型",
        pros="无需奖励模型与 RL，易于复现",
        cons="依赖高质量偏好数据，难以做在线探索",
        score_profile={
            "实现复杂度": 4.0,
            "样本效率": 3.5,
            "对齐效果": 4.0,
            "算力需求": 2.5,
            "稳定性": 4.0,
        },
    ),
    AlignmentMethod(
        name="KTO",
        optimization="KTO 损失 (一阶方法)",
        data_need="多粒度反馈 (好/坏/无偏好)",
        pros="兼容偏好缺失场景，可统一对齐目标",
        cons="社区资料较少，调参经验欠缺",
        score_profile={
            "实现复杂度": 3.5,
            "样本效率": 4.0,
            "对齐效果": 3.8,
            "算力需求": 3.0,
            "稳定性": 3.5,
        },
    ),
]


def to_markdown(methods: List[AlignmentMethod]) -> str:
    header = "| 方法 | 优化过程 | 数据需求 | 优点 | 缺点 |\n|---|---|---|---|---|"
    rows = [
        f"| {m.name} | {m.optimization} | {m.data_need} | {m.pros} | {m.cons} |"
        for m in methods
    ]
    return "\n".join([header, *rows])


def plot_radar(methods: List[AlignmentMethod]) -> None:
    labels = list(next(iter(methods)).score_profile.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 6))

    for method in methods:
        scores = list(method.score_profile.values())
        scores += scores[:1]
        ax.plot(angles, scores, label=method.name)
        ax.fill(angles, scores, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=10)
    ax.set_title("RLHF / DPO / KTO 能力对比", fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    plt.show()


def main() -> None:
    print("\n=== 对齐方法对比表 ===\n")
    print(to_markdown(METHODS))
    try:
        plot_radar(METHODS)
    except Exception as err:  # noqa: BLE001 - 可视化失败不影响教学流程
        print(f"跳过雷达图绘制: {err}")


if __name__ == "__main__":
    main()
