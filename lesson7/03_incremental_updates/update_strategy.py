"""课程实验 3：知识库增量更新与重训练策略

本脚本提供检测新增/变更文档、增量写入向量库、定期触发重训练的示例，
帮助学员理解企业知识库在持续运营中的维护要点。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class DocumentRecord:
    """记录文档内容与嵌入向量，便于演示增量写入。"""

    doc_id: str
    text: str
    embedding: List[float]
    updated_at: str


@dataclass
class UpdatePlan:
    """总结一次更新所需执行的操作。"""

    new_docs: List[DocumentRecord] = field(default_factory=list)
    updated_docs: List[DocumentRecord] = field(default_factory=list)
    removed_doc_ids: List[str] = field(default_factory=list)
    need_retrain: bool = False


def encode(text: str, dim: int = 128) -> List[float]:
    """模拟编码器：返回归一化的随机向量。"""

    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.standard_normal(dim)
    vec /= np.linalg.norm(vec) + 1e-6
    return vec.astype(float).tolist()


def detect_changes(old_index: Dict[str, DocumentRecord], new_corpus: Dict[str, str]) -> UpdatePlan:
    """对比旧索引与新语料，生成更新计划。"""

    plan = UpdatePlan()
    for doc_id, text in new_corpus.items():
        if doc_id not in old_index:
            plan.new_docs.append(
                DocumentRecord(doc_id=doc_id, text=text, embedding=encode(text), updated_at=datetime.utcnow().isoformat())
            )
        elif old_index[doc_id].text != text:
            plan.updated_docs.append(
                DocumentRecord(doc_id=doc_id, text=text, embedding=encode(text), updated_at=datetime.utcnow().isoformat())
            )
    for doc_id in old_index:
        if doc_id not in new_corpus:
            plan.removed_doc_ids.append(doc_id)
    # 简化策略：若更新比例超过 30%，建议重训练
    total = max(len(new_corpus), 1)
    change_ratio = (len(plan.new_docs) + len(plan.updated_docs)) / total
    plan.need_retrain = change_ratio >= 0.3
    return plan


def apply_plan(index_path: Path, plan: UpdatePlan) -> None:
    """将更新计划应用到本地 JSON 索引，模拟增量写入与删除。"""

    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        data = {}

    for record in plan.new_docs + plan.updated_docs:
        data[record.doc_id] = {
            "text": record.text,
            "embedding": record.embedding,
            "updated_at": record.updated_at,
        }
    for doc_id in plan.removed_doc_ids:
        data.pop(doc_id, None)

    index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def simulate_retraining(plan: UpdatePlan, log_path: Path) -> None:
    """根据计划决定是否启动重训练，并记录审计日志。"""

    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "need_retrain": plan.need_retrain,
        "new_docs": [doc.doc_id for doc in plan.new_docs],
        "updated_docs": [doc.doc_id for doc in plan.updated_docs],
        "removed_docs": plan.removed_doc_ids,
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    if plan.need_retrain:
        print("触发模型重训练：建议重新训练检索器或重排序模型。")
    else:
        print("无需立即重训练，可累计更多变更后再执行。")


def load_existing_index(index_path: Path) -> Dict[str, DocumentRecord]:
    """加载历史索引，便于对比。"""

    if not index_path.exists():
        return {}
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        doc_id: DocumentRecord(
            doc_id=doc_id,
            text=item["text"],
            embedding=item["embedding"],
            updated_at=item.get("updated_at", datetime.utcnow().isoformat()),
        )
        for doc_id, item in raw.items()
    }


def parse_args() -> argparse.Namespace:
    """命令行参数：指定旧索引、最新语料与输出路径。"""

    parser = argparse.ArgumentParser(description="知识库增量更新策略演示")
    parser.add_argument("index", type=Path, help="现有索引 JSON 路径")
    parser.add_argument("corpus", type=Path, help="最新语料 JSON，键为 doc_id，值为文本")
    parser.add_argument("--log", type=Path, default=Path("update_log.json"), help="更新日志输出路径")
    return parser.parse_args()


def main() -> None:
    """脚本入口：检测变更并应用更新计划。"""

    args = parse_args()
    old_index = load_existing_index(args.index)
    new_corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    plan = detect_changes(old_index, new_corpus)
    apply_plan(args.index, plan)
    simulate_retraining(plan, args.log)
    print("新增文档：", [doc.doc_id for doc in plan.new_docs])
    print("更新文档：", [doc.doc_id for doc in plan.updated_docs])
    print("删除文档：", plan.removed_doc_ids)


if __name__ == "__main__":
    main()
