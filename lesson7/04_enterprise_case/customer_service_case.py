"""课程实验 4：企业客服知识库应用案例

本脚本通过模拟企业内部客服知识库的搭建与调用流程，展示数据采集、
知识整理、对话策略与效果评估等关键步骤，帮助学员理解落地方案。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class KnowledgeArticle:
    """知识库文章：包含标题、正文与标签。"""

    article_id: str
    title: str
    content: str
    tags: List[str]


@dataclass
class ConversationTurn:
    """客服对话中的单轮问答。"""

    role: str
    text: str
    source_article: Optional[str] = None


class CustomerServicePlaybook:
    """封装客服知识库调取与对话策略。"""

    def __init__(self, articles: Dict[str, KnowledgeArticle]) -> None:
        self.articles = articles

    def retrieve(self, query: str, top_k: int = 2) -> List[KnowledgeArticle]:
        """演示检索：基于标签与关键词匹配筛选文章。"""

        candidates: List[KnowledgeArticle] = []
        for article in self.articles.values():
            if any(tag in query for tag in article.tags):
                candidates.append(article)
        if not candidates:
            candidates = list(self.articles.values())
        return candidates[:top_k]

    def generate_reply(self, query: str) -> ConversationTurn:
        """组合检索结果生成客服回复。"""

        articles = self.retrieve(query)
        if not articles:
            return ConversationTurn(role="agent", text="抱歉，暂时无法找到相关答案。请稍后重试。")
        summary = "；".join(article.content[:60] for article in articles)
        return ConversationTurn(
            role="agent",
            text=f"根据知识库信息：{summary}。如需更多帮助，可转人工服务。",
            source_article=articles[0].article_id,
        )


def load_articles(path: Optional[Path] = None) -> Dict[str, KnowledgeArticle]:
    """从 JSON 加载知识库，若未提供则使用内置示例。"""

    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "faq_001": {
                "title": "账号登录异常排查",
                "content": "请确认网络连接稳定，若多次失败可尝试重置密码或联系管理员。",
                "tags": ["登录", "账号", "密码"],
            },
            "faq_002": {
                "title": "发票开具流程",
                "content": "登录财务系统提交申请，填写开票信息并选择邮寄或电子发票。",
                "tags": ["发票", "财务"],
            },
            "faq_003": {
                "title": "客服机器人转人工策略",
                "content": "若用户连续两次反馈无帮助，系统应自动转接人工客服处理。",
                "tags": ["客服", "转人工"],
            },
        }
    articles = {
        key: KnowledgeArticle(
            article_id=key,
            title=value["title"],
            content=value["content"],
            tags=value.get("tags", []),
        )
        for key, value in data.items()
    }
    return articles


def simulate_dialogue(playbook: CustomerServicePlaybook, queries: List[str]) -> List[ConversationTurn]:
    """构造多轮对话，记录每轮引用的知识。"""

    conversation: List[ConversationTurn] = []
    for query in queries:
        conversation.append(ConversationTurn(role="user", text=query))
        response = playbook.generate_reply(query)
        conversation.append(response)
    return conversation


def evaluate_conversation(conversation: List[ConversationTurn]) -> Dict[str, float]:
    """根据模拟对话计算关键指标。"""

    total_questions = sum(1 for turn in conversation if turn.role == "user")
    resolved = sum(1 for turn in conversation if turn.role == "agent" and turn.source_article)
    escalation = sum(1 for turn in conversation if turn.role == "agent" and "转人工" in turn.text)
    return {
        "一次解决率": resolved / max(total_questions, 1),
        "转人工率": escalation / max(total_questions, 1),
    }


def parse_args() -> argparse.Namespace:
    """命令行参数：指定知识库文件与待模拟的问题列表。"""

    parser = argparse.ArgumentParser(description="企业客服知识库案例演示")
    parser.add_argument("--kb", type=Path, default=None, help="知识库 JSON 文件，可选")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=["无法登录系统怎么办？", "如何开具增值税专用发票？"],
        help="模拟的用户提问列表",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口：加载知识库、模拟对话并输出评估指标。"""

    args = parse_args()
    articles = load_articles(args.kb)
    playbook = CustomerServicePlaybook(articles)
    conversation = simulate_dialogue(playbook, args.queries)
    for turn in conversation:
        if turn.role == "user":
            print(f"用户：{turn.text}")
        else:
            source = f"（引用 {turn.source_article}）" if turn.source_article else ""
            print(f"客服：{turn.text}{source}")
    metrics = evaluate_conversation(conversation)
    print("评估指标：", json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
