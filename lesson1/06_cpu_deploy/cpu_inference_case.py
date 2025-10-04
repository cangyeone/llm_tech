"""轻量模型 CPU 部署案例分析。

脚本功能：
1. 对比 0.5B 与 3B 规模模型在 CPU 上的推理延迟（基于随机输入模拟）。
2. 计算每秒 token 数，辅助说明轻量模型的优势。
3. 输出部署建议，例如使用 `torch.compile` 或者 INT8 量化。

> 注意：为了在教学环境快速执行，脚本默认使用 `AutoModelForCausalLM.from_pretrained` 的 `from_config` 模式生成随机权重，而不下载真实模型。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@dataclass
class DeployCandidate:
    name: str
    model_id: str
    hidden_size: int
    num_layers: int


@dataclass
class BenchmarkResult:
    name: str
    latency_ms: float
    tokens_per_second: float


CANDIDATES: List[DeployCandidate] = [
    DeployCandidate("轻量 0.5B", "Qwen/Qwen1.5-0.5B", hidden_size=1024, num_layers=24),
    DeployCandidate("中型 3B", "Qwen/Qwen1.5-3B", hidden_size=2048, num_layers=36),
]


def build_dummy_model(candidate: DeployCandidate) -> AutoModelForCausalLM:
    """基于配置生成随机初始化模型，用于模拟推理时间。"""

    config = AutoConfig.from_pretrained(
        "gpt2",
        hidden_size=candidate.hidden_size,
        num_hidden_layers=candidate.num_layers,
        num_attention_heads=max(1, candidate.hidden_size // 64),
        vocab_size=32000,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def benchmark(candidate: DeployCandidate, seq_length: int = 128, steps: int = 5) -> BenchmarkResult:
    model = build_dummy_model(candidate)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, seq_length))

    with torch.no_grad():
        start = time.time()
        for _ in range(steps):
            _ = model(dummy_input)
        end = time.time()

    latency = (end - start) * 1000 / steps
    tokens_per_second = (seq_length * steps) / (end - start)
    LOGGER.info("模型 %s 平均延迟 %.2f ms", candidate.name, latency)
    return BenchmarkResult(candidate.name, latency, tokens_per_second)


def main() -> None:
    results = [benchmark(candidate) for candidate in CANDIDATES]

    print("CPU 推理对比（随机权重模拟）")
    for result in results:
        print(f"{result.name}: 延迟={result.latency_ms:.2f}ms, Token/s={result.tokens_per_second:.2f}")

    print("部署建议：")
    print("1. 对轻量模型可直接使用 CPU 部署，并启用 torch.compile 或 ONNX Runtime 加速。")
    print("2. 对 3B 模型建议结合 INT8/INT4 量化或分层加载减少内存。")
    print("3. 使用批处理（batching）与流式生成提升整体吞吐。")


if __name__ == "__main__":
    main()
