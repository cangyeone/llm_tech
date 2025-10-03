"""LoRA/QLoRA/P-Tuning 场景选择助手。

根据硬件、数据规模与上线需求，自动推荐适合的微调方案。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Hardware = Literal["cpu", "single_gpu", "multi_gpu"]
DataScale = Literal["small", "medium", "large"]
LatencyRequirement = Literal["strict", "normal"]


@dataclass
class Scenario:
    hardware: Hardware
    data_scale: DataScale
    latency: LatencyRequirement
    description: str


def recommend(scenario: Scenario) -> str:
    """根据场景参数给出推荐方案。"""

    if scenario.hardware == "cpu":
        return "推荐 LoRA 或 P-Tuning，优先选择轻量模型并结合量化部署。"
    if scenario.hardware == "single_gpu":
        if scenario.data_scale == "large":
            return "推荐 QLoRA，利用 4bit 加载与梯度检查点应对大规模数据。"
        return "LoRA 即可满足需求，结合梯度累积提升吞吐。"
    if scenario.hardware == "multi_gpu":
        if scenario.latency == "strict":
            return "QLoRA + 分布式推理，或蒸馏后上线轻量模型。"
        return "LoRA/QLoRA 均可，根据数据规模选择是否量化。"
    return "请提供完整场景信息。"


if __name__ == "__main__":
    case = Scenario("single_gpu", "large", "normal", "7B 模型+单卡 A100")
    print(recommend(case))
