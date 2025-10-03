"""课程实验 6：检索与生成性能压测

本脚本通过模拟检索延迟与生成耗时，演示如何统计平均值、分位数与
瓶颈提示，帮助学员理解性能调优的关注点。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BenchmarkConfig:
    """压测配置：包括查询数量、检索与生成延迟范围。"""

    queries: int
    retrieval_ms: List[int]
    generation_ms: List[int]


@dataclass
class BenchmarkResult:
    """压测结果汇总，用于输出统计指标。"""

    retrieval_times: List[float]
    generation_times: List[float]

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            "retrieval": _statistics(self.retrieval_times),
            "generation": _statistics(self.generation_times),
        }


def _statistics(samples: List[float]) -> Dict[str, float]:
    """计算平均值、P95 与最大值。"""

    if not samples:
        return {"avg": 0.0, "p95": 0.0, "max": 0.0}
    avg = statistics.fmean(samples)
    sorted_samples = sorted(samples)
    index_p95 = int(0.95 * (len(sorted_samples) - 1))
    return {
        "avg": avg,
        "p95": sorted_samples[index_p95],
        "max": max(sorted_samples),
    }


def simulate_latency(range_ms: List[int]) -> float:
    """在给定区间内随机选择延迟，并使用 sleep 模拟耗时。"""

    latency = random.uniform(range_ms[0], range_ms[1]) / 1000
    time.sleep(latency / 10)  # 缩短真实等待时间，便于课堂演示
    return latency


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """执行压测，分别记录检索与生成的延迟。"""

    retrieval_times: List[float] = []
    generation_times: List[float] = []
    for _ in range(config.queries):
        retrieval_times.append(simulate_latency(config.retrieval_ms))
        generation_times.append(simulate_latency(config.generation_ms))
    return BenchmarkResult(retrieval_times=retrieval_times, generation_times=generation_times)


def diagnose(result: BenchmarkResult) -> str:
    """根据结果判断主要瓶颈。"""

    summary = result.summary()
    retr = summary["retrieval"]
    gen = summary["generation"]
    if retr["avg"] > gen["avg"] * 1.2:
        return "建议优化检索阶段，例如引入缓存或更高效的索引结构。"
    if gen["avg"] > retr["avg"] * 1.2:
        return "建议优化生成阶段，可通过模型量化或并行推理降耗。"
    return "检索与生成耗时较为均衡，可综合优化或增加监控精度。"


def parse_args() -> argparse.Namespace:
    """命令行参数：配置压测规模与延迟区间。"""

    parser = argparse.ArgumentParser(description="RAG 性能压测演示")
    parser.add_argument("--queries", type=int, default=20, help="压测查询次数")
    parser.add_argument("--retrieval", nargs=2, type=int, default=[80, 150], help="检索延迟范围（毫秒）")
    parser.add_argument("--generation", nargs=2, type=int, default=[300, 600], help="生成延迟范围（毫秒）")
    return parser.parse_args()


def main() -> None:
    """脚本入口：运行压测并输出统计信息。"""

    args = parse_args()
    config = BenchmarkConfig(queries=args.queries, retrieval_ms=args.retrieval, generation_ms=args.generation)
    result = run_benchmark(config)
    summary = result.summary()
    print("压测结果：", json.dumps(summary, ensure_ascii=False, indent=2))
    print("瓶颈诊断：", diagnose(result))


if __name__ == "__main__":
    main()
