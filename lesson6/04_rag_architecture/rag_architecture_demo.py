"""课程实验 4：RAG 架构详解与推理示例

本脚本梳理检索增强生成（RAG）系统的核心组件，包含检索（Retriever）、
阅读器（Reader/Qwen）、排序（Ranker）与反馈监控模块，并给出端到端的
示例流程，帮助学员理解模块间的协同关系。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RagComponent:
    """描述 RAG 子组件的职责与接口。"""

    name: str
    description: str
    key_metrics: List[str]


class SimpleRetriever:
    """基于 FAISS 的向量检索器。"""

    def __init__(self, index: faiss.IndexFlatIP, documents: List[str]) -> None:
        self.index = index
        self.documents = documents

    def retrieve(self, query_vec: np.ndarray, top_k: int) -> List[str]:
        scores, indices = self.index.search(query_vec.reshape(1, -1), top_k)
        return [self.documents[idx] for idx in indices[0] if idx != -1]


class SimpleRanker:
    """示例排序器，通过关键词命中率进行再排序。"""

    def rerank(self, query: str, candidates: List[str]) -> List[str]:
        keywords = {token for token in query.split() if len(token) > 1}
        scored = []
        for text in candidates:
            hit = sum(1 for word in keywords if word in text)
            scored.append((hit, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored]


class QwenReader:
    """调用 Qwen3 模型生成答案。"""

    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt) :].strip()


def build_components() -> List[RagComponent]:
    """总结 RAG 关键组件，便于课堂讨论。"""

    return [
        RagComponent(
            name="Retriever",
            description="使用向量相似度从知识库中检索候选上下文。",
            key_metrics=["召回率", "平均延迟", "覆盖率"],
        ),
        RagComponent(
            name="Ranker",
            description="根据语义相关度/关键词等对候选上下文进行重新排序。",
            key_metrics=["NDCG", "命中率", "人工反馈评分"],
        ),
        RagComponent(
            name="Reader (Qwen)",
            description="融合用户问题与检索上下文，生成最终回答。",
            key_metrics=["BLEU/ROUGE", "Hallucination Rate", "用户满意度"],
        ),
        RagComponent(
            name="Monitor",
            description="采集日志、指标与反馈，支撑持续优化与 A/B 测试。",
            key_metrics=["响应延迟", "错误率", "反馈采样量"],
        ),
    ]


def summarize_components(components: List[RagComponent]) -> str:
    """以 Markdown 表格输出组件说明。"""

    header = "模块 | 主要职责 | 关键指标"
    sep = "---|---|---"
    rows = [f"{comp.name} | {comp.description} | {', '.join(comp.key_metrics)}" for comp in components]
    return "\n".join([header, sep, *rows])


def load_vector_index(text_path: Path, embedding_path: Path) -> tuple[faiss.IndexFlatIP, List[str]]:
    """加载 FAISS 向量索引。"""

    documents = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    embeddings = np.load(embedding_path).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, documents


def rag_pipeline(query: str, retriever: SimpleRetriever, ranker: SimpleRanker, reader: QwenReader, top_k: int) -> str:
    """RAG 推理流程：检索 -> 排序 -> 生成。"""

    query_vec = retriever.index.reconstruct_n(0, 1)[0].copy()
    query_vec = query_vec + np.random.standard_normal(size=query_vec.shape).astype(np.float32) * 0.01
    faiss.normalize_L2(query_vec.reshape(1, -1))
    candidates = retriever.retrieve(query_vec, top_k=top_k)
    reranked = ranker.rerank(query, candidates)

    context = "\n".join(reranked)
    prompt = (
        "你是一个专业的知识库问答助手，请根据以下上下文回答用户问题。\n"
        f"上下文：\n{context}\n\n用户问题：{query}\n回答："
    )
    answer = reader.generate(prompt)
    return answer


def main(model_name: str, text_path: Path, embedding_path: Path, top_k: int) -> None:
    """打印组件说明并执行一次示例推理。"""

    components = build_components()
    print("=== RAG 架构组成 ===")
    print(summarize_components(components))

    index, documents = load_vector_index(text_path, embedding_path)
    retriever = SimpleRetriever(index=index, documents=documents)
    ranker = SimpleRanker()
    reader = QwenReader(model_name=model_name)

    query = "如何解释 RAG 系统中的检索与生成协同?"
    answer = rag_pipeline(query, retriever, ranker, reader, top_k=top_k)

    print("\n=== 示例问答 ===")
    print(f"问题：{query}")
    print(f"回答：{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 架构解析与 Qwen 推理示例")
    parser.add_argument("text", type=Path, help="知识库分块文本路径")
    parser.add_argument("embeddings", type=Path, help="知识库向量矩阵路径")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.8B-Instruct", help="Qwen3 模型名称")
    parser.add_argument("--top_k", type=int, default=3, help="检索候选数量")
    args = parser.parse_args()

    main(model_name=args.model_name, text_path=args.text, embedding_path=args.embeddings, top_k=args.top_k)
