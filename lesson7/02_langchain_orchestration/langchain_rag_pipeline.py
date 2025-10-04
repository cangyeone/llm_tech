"""课程实验 2：基于 LangChain 思路的 RAG 流程编排

本脚本不直接依赖 LangChain，而是以相同理念搭建一个轻量级调度器，
演示链式调用、状态跟踪、指标记录等关键概念，帮助学员理解真实项目中
如何组织检索与生成模块。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StageContext:
    """保存链路执行过程中的上下文状态。"""

    query: str
    retrieved_docs: List[str] = field(default_factory=list)
    generation: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStage:
    """模拟 LangChain 中的链式组件。"""

    name: str
    handler: Callable[[StageContext], StageContext]

    def run(self, context: StageContext) -> StageContext:
        print(f"执行节点：{self.name}")
        context = self.handler(context)
        return context


@dataclass
class RAGPipeline:
    """组合多个 Stage，实现可配置的 RAG 调度流程。"""

    stages: List[PipelineStage]

    def run(self, query: str) -> StageContext:
        context = StageContext(query=query)
        for stage in self.stages:
            context = stage.run(context)
        return context


def retrieval_stage(top_k: int) -> PipelineStage:
    """检索节点：从知识库中返回与问题相关的文档片段。"""

    knowledge_base = [
        "RAG 架构通过检索相关文档增强大模型生成能力。",
        "LangChain 提供模块化组件，方便搭建检索与生成流水线。",
        "企业知识库需要权限控制与监控，确保信息合规。",
        "混合检索结合 BM25 与向量方法，可提升召回率。",
    ]

    def handler(context: StageContext) -> StageContext:
        print(f"检索 {top_k} 条文档片段……")
        # 课堂示例：直接选取前 top_k 条文档
        context.retrieved_docs = knowledge_base[:top_k]
        context.metrics.setdefault("retrieval", {})
        context.metrics["retrieval"].update({"top_k": top_k, "doc_count": len(context.retrieved_docs)})
        return context

    return PipelineStage(name="retrieval", handler=handler)


def rerank_stage() -> PipelineStage:
    """重排序节点：根据简单打分调整顺序，模拟 LangChain 中的链式调用。"""

    def handler(context: StageContext) -> StageContext:
        if not context.retrieved_docs:
            return context
        scored = []
        for doc in context.retrieved_docs:
            score = 1.0 if "企业" in doc else 0.8
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        context.retrieved_docs = [doc for doc, _ in scored]
        context.metrics.setdefault("rerank", {})
        context.metrics["rerank"].update({"method": "keyword_boost", "scores": [s for _, s in scored]})
        return context

    return PipelineStage(name="rerank", handler=handler)


def generation_stage(model_name: str) -> PipelineStage:
    """生成节点：将检索到的文档组合成回答。"""

    def handler(context: StageContext) -> StageContext:
        prompt = "\n".join(context.retrieved_docs)
        answer = f"【{model_name}】根据检索内容生成：{prompt[:100]}…"
        context.generation = answer
        context.metrics.setdefault("generation", {})
        context.metrics["generation"].update({"model": model_name, "prompt_length": len(prompt)})
        return context

    return PipelineStage(name="generation", handler=handler)


def audit_stage(output_dir: Path) -> PipelineStage:
    """审计节点：将链路执行信息写入日志，便于课堂讲解监控要点。"""

    output_dir.mkdir(parents=True, exist_ok=True)

    def handler(context: StageContext) -> StageContext:
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": context.query,
            "retrieved_docs": context.retrieved_docs,
            "generation": context.generation,
            "metrics": context.metrics,
        }
        log_path = output_dir / f"rag_audit_{int(datetime.utcnow().timestamp())}.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"审计日志已写入：{log_path}")
        return context

    return PipelineStage(name="audit", handler=handler)


def build_pipeline(top_k: int, model_name: str, log_dir: Path) -> RAGPipeline:
    """根据课堂参数构建 RAG 调度流程。"""

    stages = [
        retrieval_stage(top_k=top_k),
        rerank_stage(),
        generation_stage(model_name=model_name),
        audit_stage(output_dir=log_dir),
    ]
    return RAGPipeline(stages=stages)


def parse_args() -> argparse.Namespace:
    """命令行参数：控制检索数量、模型名称与日志目录。"""

    parser = argparse.ArgumentParser(description="LangChain 风格 RAG 流程编排演示")
    parser.add_argument("query", type=str, help="用户查询")
    parser.add_argument("--top_k", type=int, default=3, help="检索文档数量")
    parser.add_argument("--model", type=str, default="Qwen3-7B-Chat", help="生成模型名称")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="审计日志目录")
    return parser.parse_args()


def main() -> None:
    """脚本入口：执行 RAG 管线并打印结果。"""

    args = parse_args()
    pipeline = build_pipeline(top_k=args.top_k, model_name=args.model, log_dir=args.log_dir)
    context = pipeline.run(query=args.query)
    print("最终回答：", context.generation)
    print("指标记录：", json.dumps(context.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
