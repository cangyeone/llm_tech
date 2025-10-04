"""QLoRA 技术详解脚本。

内容包含：
- int4 量化流程模拟（NF4）
- 分页存储（paged optimizers）的显存统计
- 梯度检查点对显存的影响估算

脚本旨在通过数值演示帮助理解 QLoRA 的三大关键组件。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@dataclass
class QLoRAStats:
    hidden_size: int = 4096
    num_layers: int = 32
    seq_length: int = 1024
    vocab_size: int = 32000

    def parameter_count(self) -> int:
        return 12 * self.hidden_size * self.hidden_size * self.num_layers


def nf4_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """模拟 NF4 量化，将 float16 映射到 16 个离散值。"""

    qmin, qmax = -8, 7
    scale = tensor.abs().max() / qmax
    quantized = torch.clamp((tensor / scale).round(), qmin, qmax)
    LOGGER.info("NF4 量化：scale=%.5f, 范围=[%d, %d]", scale.item(), qmin, qmax)
    return quantized * scale


def estimate_paged_memory(stats: QLoRAStats) -> Dict[str, float]:
    """估算分页优化器带来的显存节省。"""

    param_bytes_fp16 = stats.parameter_count() * 2
    param_bytes_int4 = stats.parameter_count() // 2
    optimizer_state = param_bytes_int4 * 2  # 两倍用于动量与二阶矩
    total = param_bytes_int4 + optimizer_state

    LOGGER.info(
        "分页优化器：FP16=%.2fGB, QLoRA=%.2fGB",
        param_bytes_fp16 / 1024 ** 3,
        total / 1024 ** 3,
    )
    return {
        "fp16_gb": param_bytes_fp16 / 1024 ** 3,
        "qlora_gb": total / 1024 ** 3,
    }


def gradient_checkpointing_saving(stats: QLoRAStats) -> float:
    """估算梯度检查点节省的激活显存。"""

    activations = stats.hidden_size * stats.seq_length * stats.num_layers * 2
    saved = activations * 0.5
    LOGGER.info("梯度检查点预计节省激活显存 %.2f GB", saved / 1024 ** 3)
    return saved / 1024 ** 3


if __name__ == "__main__":
    stats = QLoRAStats()
    dummy = torch.randn(1024)
    _ = nf4_quantize(dummy)
    estimate_paged_memory(stats)
    gradient_checkpointing_saving(stats)
