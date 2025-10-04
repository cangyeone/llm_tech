"""课程实验 5：RAGFlow 本地化部署与测试

本脚本整理 RAGFlow 在本地 GPU/CPU 环境的部署步骤，包含 Docker 启动、
配置文件准备、健康检查与问答测试示例，帮助学员快速完成端到端验证。
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DeploymentStep:
    """描述部署环节的命令与注意事项。"""

    title: str
    command: str
    notes: List[str]


def build_steps(config_dir: Path, model_name: str) -> List[DeploymentStep]:
    """生成 RAGFlow 部署流程。"""

    env_file = config_dir / "ragflow.env"
    compose_file = config_dir / "docker-compose.yaml"
    return [
        DeploymentStep(
            title="准备环境变量",
            command=f"cat > {env_file} <<'EOF'\nRAGFLOW_MODEL={model_name}\nEMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2\nEOF",
            notes=[
                "根据实际硬件替换模型名称，CPU 环境可改为 Mini 版本。",
                "确保 `.env` 文件被 docker compose 读取。",
            ],
        ),
        DeploymentStep(
            title="编写 Docker Compose",
            command=(
                f"cat > {compose_file} <<'EOF'\n"
                "version: '3.9'\n"
                "services:\n"
                "  ragflow:\n"
                "    image: ragflow/ragflow:latest\n"
                "    env_file:\n"
                f"      - {env_file}\n"
                "    ports:\n"
                "      - '8000:8000'\n"
                "    volumes:\n"
                "      - ./data:/ragflow/data\n"
                "EOF"
            ),
            notes=[
                "如需 GPU 支持，可在 services.ragflow 下添加 `deploy.resources`.",
                "默认暴露 8000 端口，课堂可根据冲突修改。",
            ],
        ),
        DeploymentStep(
            title="启动服务",
            command=f"docker compose -f {compose_file} up -d",
            notes=[
                "首次拉取镜像可能耗时较长，请预留时间。",
                "启动后可通过 `docker compose ps` 查看状态。",
            ],
        ),
        DeploymentStep(
            title="健康检查",
            command="curl -s http://localhost:8000/health",
            notes=[
                "返回 200 状态代表服务启动成功。",
                "若失败，请通过 `docker compose logs` 查看错误日志。",
            ],
        ),
    ]


def run_command(command: str) -> subprocess.CompletedProcess[str]:
    """执行部署命令并输出结果。"""

    print(f"\n> {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


def demo_query(query: str) -> None:
    """展示如何调用 RAGFlow API 进行问答。"""

    curl_cmd = (
        "curl -s -X POST http://localhost:8000/api/v1/rag/query "
        "-H 'Content-Type: application/json' "
        f"-d '{{\"query\": \"{query}\", \"top_k\": 3}}'"
    )
    print("示例问答请求：")
    print(curl_cmd)
    print("执行后将返回包含检索片段与模型回答的 JSON 响应。")


def main(config_dir: Path, model_name: str, dry_run: bool) -> None:
    """生成部署步骤并按需执行。"""

    steps = build_steps(config_dir=config_dir, model_name=model_name)
    if dry_run:
        print("=== RAGFlow 部署流程（Dry Run） ===")
        for step in steps:
            print(f"\n[{step.title}]\n命令：{step.command}\n提示：{'; '.join(step.notes)}")
    else:
        for step in steps:
            print(f"\n=== {step.title} ===")
            print("注意事项：")
            for note in step.notes:
                print(f"- {note}")
            result = run_command(step.command)
            if result.returncode != 0:
                print("命令执行失败，请根据输出排查后重试。")
                return
    demo_query("如何在本地部署 RAGFlow？")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGFlow 本地部署流程辅助脚本")
    parser.add_argument("config_dir", type=Path, help="用于存放配置文件的目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.8B-Instruct", help="生成模型名称")
    parser.add_argument("--dry_run", action="store_true", help="仅打印命令，不实际执行")
    args = parser.parse_args()

    main(config_dir=args.config_dir, model_name=args.model_name, dry_run=args.dry_run)
