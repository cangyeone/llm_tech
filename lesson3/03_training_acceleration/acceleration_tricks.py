"""训练加速技巧演示。

包含：
- FlashAttention 前后速度对比（基于随机张量）
- torch.compile 对前向的加速效果
- 内存优化建议输出
"""

from __future__ import annotations

import logging
import time

import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def flash_attention_demo() -> None:
    """比较 torch.nn.functional.scaled_dot_product_attention 的性能。"""

    q = torch.randn(4, 8, 128, 64)
    k = torch.randn(4, 8, 128, 64)
    v = torch.randn(4, 8, 128, 64)

    start = time.time()
    torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    end = time.time()
    LOGGER.info("FlashAttention API 调用耗时：%.4f s", end - start)


def compile_speedup() -> None:
    """演示 torch.compile 对简单前向的加速。"""

    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 512),
    )

    compiled = torch.compile(model)
    inputs = torch.randn(32, 512)

    # 预热
    compiled(inputs)
    start = time.time()
    for _ in range(20):
        compiled(inputs)
    end = time.time()
    LOGGER.info("torch.compile 平均耗时：%.4f s", (end - start) / 20)


def print_memory_tips() -> None:
    print("内存优化建议：")
    print("1. 启用梯度检查点减少激活显存。")
    print("2. 使用 bf16/FP8 等低精度训练降低显存需求。")
    print("3. 合理设置 ZeRO 分片、张量并行。")


if __name__ == "__main__":
    flash_attention_demo()
    compile_speedup()
    print_memory_tips()
