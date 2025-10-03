"""课程实验 4：模型版本管理

演示如何在对齐实验中结合 MLflow 与 Weights & Biases 记录模型、
参数与指标，支持离线缓存和本地追踪，便于团队协作审计。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict

import mlflow
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-1.8B-Instruct"


@dataclass
class TrackingArguments:
    """版本管理相关参数。"""

    model_name: str = DEFAULT_MODEL
    experiment_name: str = "qwen-alignment"
    mlflow_uri: str = "file:./mlruns"
    wandb_mode: str = "offline"
    wandb_project: str = "qwen-alignment"
    sample_prompt: str = "请为企业客服写一段友好开场白。"


@contextmanager
def mlflow_run_context(experiment_name: str):
    """MLflow 上下文管理器，确保自动结束。"""

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        yield run


def init_wandb(project: str, mode: str) -> None:
    """初始化 WandB，支持离线模式。"""

    wandb.init(project=project, mode=mode, config={"course": "lesson5", "topic": "versioning"})


def sample_generation(model_name: str, prompt: str) -> Dict[str, str]:
    """生成示例回答，同时记录模型参数量。"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    params = sum(p.numel() for p in model.parameters())
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer, "params": params}


def log_metrics(result: Dict[str, str | int | float]) -> None:
    """同时向 MLflow 与 WandB 写入指标。"""

    mlflow.log_metrics({"response_length": len(result["answer"])})
    mlflow.log_param("model_params", result["params"])
    wandb.log({"response_length": len(result["answer"]), "model_params": result["params"]})


def parse_args() -> TrackingArguments:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="模型版本管理实验")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--experiment", type=str, default="qwen-alignment")
    parser.add_argument("--mlflow-uri", type=str, default="file:./mlruns")
    parser.add_argument("--wandb-mode", type=str, default="offline")
    parser.add_argument("--wandb-project", type=str, default="qwen-alignment")
    parser.add_argument("--prompt", type=str, default="请为企业客服写一段友好开场白。")
    parsed = parser.parse_args()
    return TrackingArguments(
        model_name=parsed.model,
        experiment_name=parsed.experiment,
        mlflow_uri=parsed.mlflow_uri,
        wandb_mode=parsed.wandb_mode,
        wandb_project=parsed.wandb_project,
        sample_prompt=parsed.prompt,
    )


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow_run_context(args.experiment_name):
        init_wandb(args.wandb_project, args.wandb_mode)
        result = sample_generation(args.model_name, args.sample_prompt)
        log_metrics(result)
        wandb.summary["sample_answer"] = result["answer"]
        wandb.finish()
        print("版本追踪完成，可在本地 UI 查看记录。")


if __name__ == "__main__":
    main()
