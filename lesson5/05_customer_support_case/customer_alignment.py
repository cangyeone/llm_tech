"""课程实验 5：客服场景对齐案例

通过构造客服问答数据集，结合偏好排序与响应模板约束，
展示如何针对行业场景进行指令对齐与质量评估。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-1.8B-Instruct"

CUSTOMER_PROMPTS = [
    {
        "query": "客户咨询：我收到了错误的账单金额，可以帮我核实吗？",
        "ideal": "您好，感谢您的反馈，我会为您核对账单，请提供订单号。",
        "reject": "这不是我的问题，请自己再看看。",
    },
    {
        "query": "客户咨询：快递迟迟未到，该怎么办？",
        "ideal": "非常抱歉给您带来不便，我可以立即联系快递查询运输状态。",
        "reject": "晚到就等着，系统也没办法。",
    },
]


@dataclass
class CaseArguments:
    """案例分析参数。"""

    model_name: str = DEFAULT_MODEL
    output_dir: str = "outputs/customer_case"
    test_size: float = 0.2


def build_case_dataset(test_size: float) -> DatasetDict:
    """构造客服偏好数据集，包含优选与不佳回答。"""

    df = pd.DataFrame(CUSTOMER_PROMPTS)
    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=42)
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "eval": Dataset.from_pandas(eval_df.reset_index(drop=True)),
        }
    )
    return dataset


def score_responses(model_name: str, dataset: DatasetDict) -> List[Dict[str, float | str]]:
    """利用模型生成客服回复，并按模板评分。"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    results: List[Dict[str, float | str]] = []
    for record in dataset["eval"]:
        prompt = f"请作为金牌客服回答客户的问题：{record['query']}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=64)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        coverage = int("感谢" in answer and "抱歉" in answer)
        politeness = int("请" in answer or "麻烦" in answer)
        score = 0.6 * coverage + 0.4 * politeness
        results.append(
            {
                "query": record["query"],
                "answer": answer,
                "ideal": record["ideal"],
                "score": score,
            }
        )
    return results


def export_results(results: List[Dict[str, float | str]], output_dir: str) -> None:
    """导出案例分析结果。"""

    df = pd.DataFrame(results)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "customer_alignment.csv", index=False, encoding="utf-8")
    print(f"客服案例得分已保存至 {output_path}")


def parse_args() -> CaseArguments:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="客服场景对齐案例")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str, default="outputs/customer_case")
    parser.add_argument("--test-size", type=float, default=0.2)
    parsed = parser.parse_args()
    return CaseArguments(
        model_name=parsed.model,
        output_dir=parsed.output,
        test_size=parsed.test_size,
    )


def main() -> None:
    args = parse_args()
    dataset = build_case_dataset(args.test_size)
    results = score_responses(args.model_name, dataset)
    export_results(results, args.output_dir)


if __name__ == "__main__":
    main()
