"""LoRA 参数注入与秩分解可视化脚本。

通过随机矩阵模拟 Transformer 权重，演示 LoRA 的低秩分解思想，并输出：
1. LoRA 权重构造过程（W = W0 + BA）。
2. 不同秩 r 对重建误差的影响。
3. 线性层插入 LoRA 适配器的伪代码。

脚本依赖 `matplotlib` 绘制误差曲线，可在 Jupyter 或命令行执行。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@dataclass
class LoraConfig:
    hidden_size: int = 512
    rank_candidates: List[int] = (2, 4, 8, 16, 32)
    alpha: float = 16.0


def simulate_lora(config: LoraConfig) -> None:
    """构造随机权重并计算低秩近似误差。"""

    base_weight = torch.randn(config.hidden_size, config.hidden_size)
    LOGGER.info("原始权重范数：%.4f", base_weight.norm().item())

    errors = []
    for rank in config.rank_candidates:
        a = torch.randn(config.hidden_size, rank)
        b = torch.randn(rank, config.hidden_size)
        lora_weight = (config.alpha / rank) * a @ b
        approx = base_weight + lora_weight
        error = torch.norm(base_weight - approx).item()
        errors.append(error)
        LOGGER.info("rank=%d -> 重建误差=%.4f", rank, error)

    plt.figure(figsize=(6, 4))
    plt.plot(config.rank_candidates, errors, marker="o")
    plt.xlabel("LoRA 秩 r")
    plt.ylabel("Frobenius 误差")
    plt.title("LoRA 秩对重建误差的影响")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lora_rank_error.png")
    LOGGER.info("误差曲线已保存为 lora_rank_error.png")


def pseudo_code() -> None:
    """打印在 Transformer Linear 层插入 LoRA 的伪代码。"""

    code = """
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lora_a = nn.Parameter(torch.zeros(r, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        # 原始线性变换
        base = F.linear(x, self.weight)
        # LoRA 低秩增量
        lora_update = F.linear(x, self.lora_a.T)
        lora_update = F.linear(lora_update, self.lora_b.T) * self.scaling
        return base + lora_update
    """
    print(code)


if __name__ == "__main__":
    cfg = LoraConfig()
    simulate_lora(cfg)
    pseudo_code()
