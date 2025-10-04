"""课程实验 1：知识库技术选型对比

本脚本通过模拟检索工作负载，对比 FAISS、Chroma、Elasticsearch 三类
向量检索方案的性能、部署成本与运维要点，帮助学员建立选型思路。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class BackendProfile:
    """记录向量检索引擎的评估指标。"""

    name: str
    build_latency: float
    query_latency: float
    memory_mb: float
    feature_notes: List[str]


def simulate_vector_data(num_vectors: int, dim: int, seed: int = 42) -> np.ndarray:
    """生成用于基准测试的随机向量。"""

    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(num_vectors, dim)).astype(np.float32)


def benchmark_backend(name: str, vectors: np.ndarray, num_queries: int = 20) -> BackendProfile:
    """模拟构建与查询过程，计算平均延迟与内存占用。"""

    start_time = time.perf_counter()
    # 这里用矩阵复制模拟建库成本，真实场景需替换为具体 SDK 调用
    index_copy = vectors.copy()
    build_latency = time.perf_counter() - start_time

    query_start = time.perf_counter()
    rng = np.random.default_rng(0)
    for _ in range(num_queries):
        query = rng.standard_normal(size=vectors.shape[1]).astype(np.float32)
        # 使用点积模拟相似度计算
        _ = index_copy @ query
    query_latency = (time.perf_counter() - query_start) / max(1, num_queries)

    memory_mb = index_copy.nbytes / 1024 / 1024
    feature_notes = summarize_backend_features(name)

    return BackendProfile(
        name=name,
        build_latency=build_latency,
        query_latency=query_latency,
        memory_mb=memory_mb,
        feature_notes=feature_notes,
    )


def summarize_backend_features(name: str) -> List[str]:
    """根据引擎名称给出课堂讨论要点。"""

    mapping: Dict[str, List[str]] = {
        "FAISS": [
            "单机部署简单，GPU 支持优秀",
            "适合离线构建+批量更新场景",
            "需自行实现高可用与监控",
        ],
        "Chroma": [
            "内置元数据管理与 REST API",
            "适合快速原型和轻量应用",
            "社区生态活跃，插件丰富",
        ],
        "Elasticsearch": [
            "与全文检索无缝整合，支持 BM25 + 向量融合",
            "可横向扩展，但资源成本较高",
            "对部署与调优能力要求高",
        ],
    }
    return mapping.get(name, ["待补充的自定义引擎信息"])


def render_report(profiles: List[BackendProfile]) -> str:
    """生成表格形式的评估报告文本。"""

    lines = [
        "检索引擎 | 建库耗时(s) | 平均查询耗时(s) | 估算内存占用(MB) | 课堂提示",
        "---|---|---|---|---",
    ]
    for profile in profiles:
        note = "；".join(profile.feature_notes)
        line = f"{profile.name} | {profile.build_latency:.4f} | {profile.query_latency:.6f} | {profile.memory_mb:.1f} | {note}"
        lines.append(line)
    return "\n".join(lines)


def main(num_vectors: int, dim: int) -> None:
    """综合执行基准测试与报告输出。"""

    vectors = simulate_vector_data(num_vectors=num_vectors, dim=dim)
    backends = ["FAISS", "Chroma", "Elasticsearch"]
    profiles = [benchmark_backend(name, vectors) for name in backends]
    report = render_report(profiles)

    print("\n=== 知识库技术选型参考 ===\n")
    print(report)
    print("\n建议结合业务场景进一步验证：数据量级、实时性、预算等因素。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模拟知识库选型评估")
    parser.add_argument("--num_vectors", type=int, default=2000, help="模拟向量数量")
    parser.add_argument("--dim", type=int, default=768, help="向量维度")
    args = parser.parse_args()

    main(num_vectors=args.num_vectors, dim=args.dim)
