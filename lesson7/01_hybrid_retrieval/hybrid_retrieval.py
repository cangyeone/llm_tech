"""课程实验 1：混合检索与重排序

本脚本展示如何同时利用 BM25 与向量检索，对查询结果进行得分融合，
并通过一个简化的重排序模型提升最终的排序质量。所有注释与打印信息
均为中文，方便课堂演示与自学。
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


TokenizerOutput = List[str]


def tokenize(text: str) -> TokenizerOutput:
    """基础分词：按非字母数字字符拆分，演示用。"""

    tokens: List[str] = []
    word = []
    for char in text.lower():
        if char.isalnum():
            word.append(char)
        elif word:
            tokens.append("".join(word))
            word = []
    if word:
        tokens.append("".join(word))
    return tokens


@dataclass
class Document:
    """封装文档内容与向量表示。"""

    doc_id: str
    text: str
    embedding: np.ndarray


class BM25Index:
    """自实现的 BM25 倒排索引，便于课堂解释公式。"""

    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_len: Dict[str, int] = {}
        self.avg_len: float = 0.0
        self.index: Dict[str, Dict[str, int]] = defaultdict(dict)

        total_len = 0
        for doc in documents:
            tokens = tokenize(doc.text)
            self.doc_len[doc.doc_id] = len(tokens)
            total_len += len(tokens)
            freq = Counter(tokens)
            for term, count in freq.items():
                self.index[term][doc.doc_id] = count
                self.doc_freq[term] += 1
        self.num_docs = len(documents)
        self.avg_len = total_len / max(self.num_docs, 1)

    def score(self, query: str) -> Dict[str, float]:
        """返回每个文档的 BM25 得分。"""

        scores: Dict[str, float] = defaultdict(float)
        query_terms = tokenize(query)
        for term in query_terms:
            posting = self.index.get(term)
            if not posting:
                continue
            df = self.doc_freq[term]
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in posting.items():
                doc_len = self.doc_len[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
                scores[doc_id] += idf * numerator / denominator
        return scores


def encode_embedding(text: str, dim: int = 384) -> np.ndarray:
    """模拟向量编码：使用哈希种子生成稳定的随机向量。"""

    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vector = rng.standard_normal(dim).astype(np.float32)
    vector /= np.linalg.norm(vector) + 1e-6
    return vector


def cosine_similarity(query_vec: np.ndarray, doc_vecs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """计算查询向量与文档向量的余弦相似度。"""

    scores: Dict[str, float] = {}
    for doc_id, vec in doc_vecs.items():
        score = float(np.dot(query_vec, vec))
        scores[doc_id] = score
    return scores


def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    """最小-最大归一化，避免某一得分主导融合。"""

    if not score_dict:
        return {}
    values = np.array(list(score_dict.values()), dtype=np.float32)
    min_v = float(values.min())
    max_v = float(values.max())
    if math.isclose(min_v, max_v):
        return {k: 1.0 for k in score_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}


@dataclass
class HybridRetriever:
    """混合检索器：融合 BM25 与向量检索结果，并执行重排序。"""

    bm25: BM25Index
    embeddings: Dict[str, np.ndarray]
    weight_bm25: float = 0.5
    weight_vector: float = 0.5

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, float, float]]:
        """返回融合后的得分，以及各自来源得分。"""

        bm25_scores = self.bm25.score(query)
        query_vec = encode_embedding(query)
        vector_scores = cosine_similarity(query_vec, self.embeddings)

        bm25_norm = normalize_scores(bm25_scores)
        vector_norm = normalize_scores(vector_scores)

        combined: Dict[str, float] = defaultdict(float)
        for doc_id, score in bm25_norm.items():
            combined[doc_id] += self.weight_bm25 * score
        for doc_id, score in vector_norm.items():
            combined[doc_id] += self.weight_vector * score

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[: top_k * 2]:
            results.append((doc_id, score, bm25_norm.get(doc_id, 0.0), vector_norm.get(doc_id, 0.0)))
        reranked = self.rerank(query, results, top_k)
        return reranked

    def rerank(
        self, query: str, candidates: Sequence[Tuple[str, float, float, float]], top_k: int
    ) -> List[Tuple[str, float, float, float]]:
        """简化的重排序：使用查询-文档向量余弦相似度作为额外特征。"""

        query_vec = encode_embedding(query)
        reranked = []
        for doc_id, fused, bm25_score, vector_score in candidates:
            doc_vec = self.embeddings[doc_id]
            rerank_score = float(np.dot(query_vec, doc_vec))
            rerank_weight = 1 / (1 + math.exp(-4 * rerank_score))
            final_score = 0.6 * fused + 0.4 * rerank_weight
            reranked.append((doc_id, final_score, bm25_score, vector_score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


def prepare_corpus(raw_texts: Sequence[str]) -> Tuple[List[Document], BM25Index, Dict[str, np.ndarray]]:
    """构造示例语料与索引。"""

    documents: List[Document] = []
    embeddings: Dict[str, np.ndarray] = {}
    for idx, text in enumerate(raw_texts):
        doc_id = f"doc_{idx:03d}"
        embedding = encode_embedding(text)
        documents.append(Document(doc_id=doc_id, text=text, embedding=embedding))
        embeddings[doc_id] = embedding
    bm25 = BM25Index(documents)
    return documents, bm25, embeddings


def demo(weight_bm25: float, weight_vector: float, top_k: int) -> None:
    """脚本主流程：执行混合检索并打印分数构成。"""

    raw_texts = [
        "Qwen3 是阿里云推出的多语言大模型，支持对齐微调与推理部署。",
        "BM25 是经典的基于词项统计的检索算法，适合精确匹配场景。",
        "向量检索能够捕捉语义相似度，常与 RAG 架构结合使用。",
        "混合检索结合 BM25 与向量方法，可兼顾精确性与召回率。",
        "企业客服知识库需要定期更新，保证回答准确可靠。",
    ]
    _, bm25, embeddings = prepare_corpus(raw_texts)
    retriever = HybridRetriever(
        bm25=bm25,
        embeddings=embeddings,
        weight_bm25=weight_bm25,
        weight_vector=weight_vector,
    )

    query = "如何在客服场景提升检索效果"
    results = retriever.retrieve(query, top_k=top_k)
    print("查询：", query)
    for rank, (doc_id, final_score, bm25_score, vector_score) in enumerate(results, start=1):
        print(
            f"[{rank}] 文档 {doc_id} | 融合得分 {final_score:.3f} | BM25 {bm25_score:.3f} | 向量 {vector_score:.3f}"
        )


def parse_args() -> argparse.Namespace:
    """提供命令行参数，支持自定义权重。"""

    parser = argparse.ArgumentParser(description="混合检索融合演示")
    parser.add_argument("--bm25_weight", type=float, default=0.6, help="BM25 得分权重")
    parser.add_argument("--vector_weight", type=float, default=0.4, help="向量得分权重")
    parser.add_argument("--top_k", type=int, default=3, help="返回结果数量")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(weight_bm25=args.bm25_weight, weight_vector=args.vector_weight, top_k=args.top_k)
