"""课程实验 5：微调模型与 RAG 的混合推理

本脚本演示如何将微调后的 Qwen 模型输出与检索增强生成结果结合，
实现更稳健的答案生成与兜底策略。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RetrievalResult:
    """检索结果项，包含文档内容与得分。"""

    content: str
    score: float


class FineTunedModel:
    """模拟加载好的微调模型。"""

    def __init__(self, model_name: str, temperature: float = 0.3) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """根据 prompt 生成回答，演示采用模板生成。"""

        return f"[{self.model_name}] 生成回答（温度 {self.temperature}）：{prompt[:120]}…"


class HybridInferenceEngine:
    """混合推理引擎：结合检索信息与微调模型输出。"""

    def __init__(self, model: FineTunedModel, fusion_threshold: float = 0.6) -> None:
        self.model = model
        self.fusion_threshold = fusion_threshold

    def build_prompt(self, question: str, retrievals: List[RetrievalResult]) -> str:
        """拼接检索结果，构造带上下文的提示词。"""

        context = "\n".join(f"- {item.content}" for item in retrievals)
        prompt = (
            "你是一名企业知识库客服助手，请使用以下检索内容回答用户问题。\n"
            f"用户问题：{question}\n"
            f"检索结果：\n{context}\n"
        )
        return prompt

    def fuse(self, question: str, retrievals: List[RetrievalResult]) -> Tuple[str, Dict[str, float]]:
        """融合策略：计算检索得分均值，决定是否启用检索增强。"""

        if not retrievals:
            pure_prompt = f"请直接回答：{question}"
            response = self.model.generate(pure_prompt)
            return response, {"retrieval_strength": 0.0, "mode": "model_only"}

        avg_score = sum(item.score for item in retrievals) / len(retrievals)
        metrics = {"retrieval_strength": avg_score}
        if avg_score >= self.fusion_threshold:
            prompt = self.build_prompt(question, retrievals)
            response = self.model.generate(prompt)
            metrics["mode"] = "rag_fusion"
        else:
            fallback_prompt = f"检索结果噪声较高，请直接回答：{question}"
            response = self.model.generate(fallback_prompt)
            metrics["mode"] = "model_fallback"
        return response, metrics


def mock_retrieval(question: str) -> List[RetrievalResult]:
    """课堂示例：根据问题关键词返回模拟检索结果。"""

    knowledge = {
        "发票": ["发票申请需提供纳税人识别号与开票抬头信息。", "财务系统审核通过后将在 2 个工作日内寄出。"],
        "登录": ["请检查账号是否锁定，可通过企业邮箱找回密码。"],
    }
    results: List[RetrievalResult] = []
    for keyword, docs in knowledge.items():
        if keyword in question:
            for rank, doc in enumerate(docs):
                score = 0.9 - rank * 0.1
                results.append(RetrievalResult(content=doc, score=score))
    return results


def parse_args() -> argparse.Namespace:
    """命令行参数：配置模型、阈值与问题。"""

    parser = argparse.ArgumentParser(description="混合推理策略演示")
    parser.add_argument("question", type=str, help="用户问题")
    parser.add_argument("--model", type=str, default="Qwen3-7B-Chat-LoRA", help="微调模型名称")
    parser.add_argument("--temperature", type=float, default=0.3, help="生成温度")
    parser.add_argument("--threshold", type=float, default=0.6, help="检索融合阈值")
    return parser.parse_args()


def main() -> None:
    """脚本入口：执行混合推理并输出指标。"""

    args = parse_args()
    model = FineTunedModel(model_name=args.model, temperature=args.temperature)
    engine = HybridInferenceEngine(model=model, fusion_threshold=args.threshold)
    retrievals = mock_retrieval(args.question)
    response, metrics = engine.fuse(args.question, retrievals)
    print("生成回答：", response)
    print("融合指标：", metrics)


if __name__ == "__main__":
    main()
