"""指令数据预处理与 Prompt 模板设计示例。

流程包括：
1. 扫描本地 markdown/文本文件并合并为原始语料。
2. 使用简单的清洗规则（去除多余空白、合并换行）。
3. 基于模板生成 instruction-tuning 格式的数据并导出为 JSONL。

> 示例使用少量虚拟文档，实际使用时请修改 `DATA_DIR` 指向真实数据目录。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass
class PreprocessConfig:
    """数据预处理配置。"""

    data_dir: Path = Path("./sample_docs")
    output_path: Path = Path("./outputs/preprocessed.jsonl")
    template: str = "你是资深助教，请根据材料生成摘要。输入：{context}"
    min_length: int = 20


def iter_documents(data_dir: Path) -> Iterable[str]:
    """遍历目录中的文本文件并返回内容。"""

    for path in data_dir.rglob("*.txt"):
        yield path.read_text(encoding="utf-8")
    for path in data_dir.rglob("*.md"):
        yield path.read_text(encoding="utf-8")


def clean_text(text: str) -> str:
    """对文本执行基础清洗：去除多余空格与重复换行。"""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\n\s*)+", "\n", text)
    return text.strip()


def build_examples(config: PreprocessConfig) -> List[dict]:
    """生成 instruction-tuning 训练样本。"""

    examples: List[dict] = []
    for doc in iter_documents(config.data_dir):
        cleaned = clean_text(doc)
        if len(cleaned) < config.min_length:
            continue
        prompt = config.template.format(context=cleaned[:400])
        examples.append(
            {
                "instruction": "请阅读输入并生成摘要。",
                "input": prompt,
                "output": cleaned[:200],
            }
        )
    return examples


def export_jsonl(examples: List[dict], output_path: Path) -> None:
    """将样本写入 JSONL 文件，便于后续微调。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for item in examples:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_dataframe(examples: List[dict]) -> pd.DataFrame:
    """使用 pandas 构建可视化数据表。"""

    return pd.DataFrame(examples)


if __name__ == "__main__":
    config = PreprocessConfig()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = config.data_dir / "demo.txt"
    sample_path.write_text(
        "大模型微调的第一步是准备数据，需要把零散文档整理为结构化文本，并设计统一的 Prompt 模板。",
        encoding="utf-8",
    )

    examples = build_examples(config)
    export_jsonl(examples, config.output_path)

    df = build_dataframe(examples)
    print("预处理样本数量：", len(df))
    print(df.head())
