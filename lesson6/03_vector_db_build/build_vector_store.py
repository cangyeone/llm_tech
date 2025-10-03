"""课程实验 3：构建向量库与语义检索

本脚本演示如何加载预处理得到的文本片段与向量，构建 FAISS 索引，
并提供简单的语义检索接口，帮助学员理解检索阶段的关键步骤。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


@dataclass
class VectorStore:
    """封装 FAISS 索引与原始文档的映射关系。"""

    index: faiss.IndexFlatIP
    documents: List[str]
    embeddings: np.ndarray

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """执行向量检索并返回文档与得分。"""

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append((doc, float(score)))
        return results


def load_preprocessed_data(text_path: Path, embedding_path: Path) -> Tuple[List[str], np.ndarray]:
    """加载已保存的文本分块与向量矩阵。"""

    with text_path.open("r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]
    embeddings = np.load(embedding_path)
    return documents, embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """基于内积相似度构建向量索引。"""

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def encode_query(query: str, embedding_matrix: np.ndarray) -> np.ndarray:
    """课堂示例：使用文档均值模拟查询向量，实际使用需接入编码模型。"""

    # 这里采用简单的随机噪声+均值作为示范，课堂上可替换为 Sentence-BERT/Qwen 编码
    mean_vector = embedding_matrix.mean(axis=0)
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    noise = rng.standard_normal(size=mean_vector.shape).astype(np.float32) * 0.01
    return mean_vector + noise


def create_vector_store(text_path: Path, embedding_path: Path) -> VectorStore:
    """整合加载、构建索引与包装操作。"""

    documents, embeddings = load_preprocessed_data(text_path, embedding_path)
    index = build_faiss_index(embeddings.copy())
    return VectorStore(index=index, documents=documents, embeddings=embeddings)


def interactive_demo(store: VectorStore, top_k: int) -> None:
    """提供命令行交互式查询体验。"""

    print("输入问题即可获取相似文档，输入 exit 退出。")
    while True:
        query = input("用户问题> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("结束演示。")
            break
        query_vec = encode_query(query, store.embeddings)
        faiss.normalize_L2(query_vec.reshape(1, -1))
        results = store.search(query_vec, top_k=top_k)
        print("\nTop-{} 检索结果：".format(top_k))
        for rank, (doc, score) in enumerate(results, start=1):
            print(f"[{rank}] 相似度 {score:.3f}: {doc[:120]}")
        print()


def main(text_path: Path, embedding_path: Path, top_k: int, interactive: bool) -> None:
    """构建向量库并按需触发互动式检索。"""

    store = create_vector_store(text_path, embedding_path)
    print(f"已加载 {len(store.documents)} 条文档片段，向量维度 {store.embeddings.shape[1]}")
    if interactive:
        interactive_demo(store, top_k)
    else:
        query_vec = encode_query("RAG 架构介绍", store.embeddings)
        faiss.normalize_L2(query_vec.reshape(1, -1))
        results = store.search(query_vec, top_k=top_k)
        print("示例查询结果：")
        for rank, (doc, score) in enumerate(results, start=1):
            print(f"[{rank}] 相似度 {score:.3f}: {doc[:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 FAISS 向量库并执行语义检索")
    parser.add_argument("text", type=Path, help="分块文本路径")
    parser.add_argument("embeddings", type=Path, help="向量矩阵路径")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--interactive", action="store_true", help="是否开启命令行交互")
    args = parser.parse_args()

    main(text_path=args.text, embedding_path=args.embeddings, top_k=args.top_k, interactive=args.interactive)
