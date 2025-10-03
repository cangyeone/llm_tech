"""课程实验 2：数据预处理与向量化流程

本脚本展示如何对原始文档进行分块、清洗，并使用 Sentence-BERT
生成文本向量，同时提供质量监控指标，确保知识库数据可控可靠。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class PreprocessConfig:
    """数据预处理的核心超参数。"""

    chunk_size: int = 300
    chunk_overlap: int = 50
    min_token: int = 20
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


def read_corpus(path: Path) -> List[str]:
    """读取文本语料，默认按行拆分。"""

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def chunk_documents(lines: Iterable[str], config: PreprocessConfig) -> List[str]:
    """按照固定窗口滑动切分文本，以兼顾上下文与召回率。"""

    chunks: List[str] = []
    buffer: List[str] = []
    total_chars = 0
    for line in lines:
        buffer.append(line)
        total_chars += len(line)
        if total_chars >= config.chunk_size:
            chunk = "\n".join(buffer)
            if len(chunk) >= config.min_token:
                chunks.append(chunk)
            # 处理重叠窗口，保留末尾若干字符作为上下文
            overlap_chars = config.chunk_overlap
            joined = "\n".join(buffer)
            buffer = [joined[-overlap_chars:]] if overlap_chars > 0 else []
            total_chars = sum(len(item) for item in buffer)
    # 收尾
    if buffer and len("\n".join(buffer)) >= config.min_token:
        chunks.append("\n".join(buffer))
    return chunks


def generate_embeddings(chunks: List[str], config: PreprocessConfig) -> np.ndarray:
    """调用 Sentence-BERT 模型生成文本向量。"""

    model = SentenceTransformer(config.model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def quality_report(chunks: List[str], embeddings: np.ndarray) -> str:
    """输出分块长度、向量范数等指标，辅助课堂讨论。"""

    lengths = np.array([len(chunk) for chunk in chunks], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    report = [
        "=== 数据质量报告 ===",
        f"分块数量：{len(chunks)}",
        f"平均长度：{lengths.mean():.1f} 字符 (± {lengths.std():.1f})",
        f"向量维度：{embeddings.shape[1]}",
        f"平均向量范数：{norms.mean():.4f} (± {norms.std():.4f})",
        "建议检查异常短/长片段，避免影响召回率。",
    ]
    return "\n".join(report)


def save_embeddings(chunks: List[str], embeddings: np.ndarray, output_dir: Path) -> None:
    """保存分块文本与对应向量，便于后续构建向量库。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "chunks.txt"
    vector_path = output_dir / "embeddings.npy"

    with text_path.open("w", encoding="utf-8") as text_file:
        for chunk in chunks:
            text_file.write(chunk.replace("\n", " ") + "\n")

    np.save(vector_path, embeddings)

    print(f"已保存分块文本：{text_path}")
    print(f"已保存向量矩阵：{vector_path}")


def main(input_path: Path, output_dir: Path, config: PreprocessConfig) -> None:
    """串联文档读取、分块、向量化与报告输出。"""

    lines = read_corpus(input_path)
    chunks = chunk_documents(lines, config)
    embeddings = generate_embeddings(chunks, config)
    save_embeddings(chunks, embeddings, output_dir)
    print(quality_report(chunks, embeddings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 文档预处理与向量化")
    parser.add_argument("input", type=Path, help="原始语料路径（UTF-8 文本）")
    parser.add_argument("output", type=Path, help="输出目录")
    parser.add_argument("--chunk_size", type=int, default=300, help="分块字符数")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="分块重叠字符数")
    parser.add_argument("--min_token", type=int, default=20, help="过滤短片段的最小字符数")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="用于生成向量的 Sentence-BERT 模型",
    )
    args = parser.parse_args()

    config = PreprocessConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_token=args.min_token,
        model_name=args.model_name,
    )
    main(input_path=args.input, output_dir=args.output, config=config)
