
# 课程实验 5：RAGFlow 本地化部署与测试（教学版文档）

本教程配套下方教学脚本，手把手带你完成 **RAGFlow** 在本地 **CPU/GPU** 环境的快速部署、健康检查与问答 API 自测。文档同时给出脚本中每个函数/数据结构的**详细说明**、**可修改项**与**常见问题排查**。

> 适配脚本：`lesson6/05_ragflow_deployment/ragflow_demo.py`（你提供的教学代码）

---

## 目录
- [课程实验 5：RAGFlow 本地化部署与测试（教学版文档）](#课程实验-5ragflow-本地化部署与测试教学版文档)
  - [目录](#目录)
  - [目标与产出](#目标与产出)
  - [环境准备](#环境准备)
  - [快速开始](#快速开始)
  - [脚本用法（命令行）](#脚本用法命令行)
  - [关键文件结构](#关键文件结构)
  - [核心流程与函数说明](#核心流程与函数说明)
    - [`DeploymentStep` 数据类](#deploymentstep-数据类)
    - [`build_steps(config_dir: Path, model_name: str) -> List[DeploymentStep]`](#build_stepsconfig_dir-path-model_name-str---listdeploymentstep)
    - [`run_command(command: str) -> subprocess.CompletedProcess`](#run_commandcommand-str---subprocesscompletedprocess)
    - [`demo_query(query: str) -> None`](#demo_queryquery-str---none)
    - [`main(config_dir: Path, model_name: str, dry_run: bool) -> None`](#mainconfig_dir-path-model_name-str-dry_run-bool---none)
  - [Docker Compose 配置详解](#docker-compose-配置详解)
  - [健康检查与 API 测试](#健康检查与-api-测试)
  - [GPU 支持（可选）](#gpu-支持可选)
  - [常见问题（FAQ）与排障](#常见问题faq与排障)
  - [安全与持久化建议](#安全与持久化建议)
  - [扩展与作业建议](#扩展与作业建议)
  - [许可证与参考](#许可证与参考)

---

## 目标与产出

完成本实验后，你将能够：

1. 在本地通过 **Docker Compose** 启动 RAGFlow 服务（适配 CPU 或 GPU）。  
2. 生成并管理配置文件（`.env` 与 `docker-compose.yaml`）。  
3. 通过 **健康检查**确认服务已启动，并用 **HTTP API** 发起问答请求。  
4. 明确 **部署脚本的结构与每个函数的职责**，可在作业中自由二次开发。

**产出**：
- `config_dir/` 下生成：
  - `ragflow.env`：核心环境变量（模型名、嵌入模型等）。
  - `docker-compose.yaml`：服务编排文件。
- 本地运行中的 RAGFlow 实例：默认监听 `http://localhost:8000`。

---

## 环境准备

- 操作系统：Linux / macOS / Windows（需 WSL2）
- 依赖：
  - Docker ≥ 20.x  
  - Docker Compose（v2 内置于 `docker` 命令，或独立 `docker-compose` 命令）  
  - `curl`（用于健康检查与 API 测试）
- 可选（GPU 场景）：已安装 **NVIDIA 驱动** 与 **nvidia-container-toolkit**。

---

## 快速开始

```bash
# 1) 选择一个目录存放配置
mkdir -p ~/ragflow-local && cd ~/ragflow-local

# 2) 运行教学脚本（仅打印命令，不执行）
python lesson6/05_ragflow_deployment/ragflow_demo.py . --dry_run

# 3) 确认无误后，真正执行部署
python lesson6/05_ragflow_deployment/ragflow_demo.py . --model_name Qwen/Qwen3-1.8B-Instruct
# 生成 ragflow.env / docker-compose.yaml 并 docker compose up -d

# 4) 健康检查
curl -s http://localhost:8000/health

# 5) 发起问答示例
curl -s -X POST http://localhost:8000/api/v1/rag/query   -H 'Content-Type: application/json'   -d '{"query":"如何在本地部署 RAGFlow？","top_k":3}'
```

---

## 脚本用法（命令行）

```bash
python ragflow_deploy_helper.py CONFIG_DIR [--model_name Qwen/Qwen3-1.8B-Instruct] [--dry_run]
```

- `CONFIG_DIR`：**必填**。用于存放配置文件的目录（可为相对或绝对路径）。
- `--model_name`：可选。生成式模型名称，默认 `Qwen/Qwen3-1.8B-Instruct`。CPU 可换更小模型。
- `--dry_run`：可选。仅打印将要执行的命令与注意事项，不真正执行（用于课堂演示/预检查）。

**示例**：

```bash
python ragflow_deploy_helper.py ./cfg --model_name Qwen/Qwen3-0.6B-Instruct --dry_run
```

---

## 关键文件结构

脚本会在 `CONFIG_DIR` 目录下生成：

```
CONFIG_DIR/
├─ ragflow.env              # 环境变量（模型名、嵌入模型等）
└─ docker-compose.yaml      # Docker Compose 编排文件（端口、卷、镜像等）
```

> 服务运行期间，容器中的数据目录默认映射到本地 `./data`。

---

## 核心流程与函数说明

### `DeploymentStep` 数据类

```python
@dataclass
class DeploymentStep:
    title: str           # 步骤标题
    command: str         # 在 Shell 中执行的命令（可多行 heredoc）
    notes: List[str]     # 注意事项/提示
```
用于承载**每个部署环节**的可执行命令与注意事项。主流程会按序执行这些步骤。

---

### `build_steps(config_dir: Path, model_name: str) -> List[DeploymentStep]`

**职责**：根据给定的配置目录与模型名，生成**完整的部署步骤列表**，包含：

1. **准备环境变量**：生成 `ragflow.env`，设置：
   - `RAGFLOW_MODEL`：生成式模型名称（默认课堂提供的 Qwen 模型）。  
   - `EMBEDDING_MODEL`：默认 `sentence-transformers/all-MiniLM-L6-v2`。

2. **编写 Docker Compose**：生成 `docker-compose.yaml`，定义：
   - 服务镜像：`ragflow/ragflow:latest`  
   - 端口映射：`8000:8000`  
   - 环境文件：加载上面生成的 `ragflow.env`  
   - 持久化卷：`./data:/ragflow/data`

3. **启动服务**：执行 `docker compose up -d`。

4. **健康检查**：`curl http://localhost:8000/health`。

**可改项**：
- 端口冲突时，修改 `ports:` 中的宿主机端口，如 `18000:8000`。  
- 更换 `EMBEDDING_MODEL` 或增补更多 RAGFlow 所需变量。

---

### `run_command(command: str) -> subprocess.CompletedProcess`

**职责**：在 Shell 中执行单步命令，打印标准输出/错误输出，并返回结果。主流程会根据 `returncode` 判断是否继续。

**异常建议**：若执行失败（`returncode != 0`），优先查看：
- Docker 守护进程是否运行（`docker info`）。  
- Compose 版本与语法（`docker compose version`）。

---

### `demo_query(query: str) -> None`

**职责**：打印一个**问答 API 请求示例**（`curl`），帮助快速验证端到端链路是否通。默认路由：

```
POST /api/v1/rag/query
Content-Type: application/json
Body: {"query":"...","top_k":3}
```

> 返回应包含**检索片段**与**模型回答**。具体字段以 RAGFlow 版本为准。

---

### `main(config_dir: Path, model_name: str, dry_run: bool) -> None`

**职责**：整合流程：
1. 调用 `build_steps` 生成步骤列表。  
2. `--dry_run` 时仅打印命令与提示；否则逐步执行，失败即停止。  
3. 最后展示 `demo_query` 示例。

---

## Docker Compose 配置详解

脚本生成的 `docker-compose.yaml`（教学默认版）：

```yaml
version: '3.9'
services:
  ragflow:
    image: ragflow/ragflow:latest
    env_file:
      - ./ragflow.env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/ragflow/data
```

**说明**：
- `env_file`：将 `.env` 内容注入容器环境。  
- `ports`：左侧是宿主机端口，右侧是容器端口。  
- `volumes`：持久化检索库、上传材料等。

---

## 健康检查与 API 测试

1. **健康检查**：
   ```bash
   curl -i http://localhost:8000/health
   ```
   - 期望 `HTTP/1.1 200 OK`。

2. **问答 API**：
   ```bash
   curl -s -X POST http://localhost:8000/api/v1/rag/query      -H 'Content-Type: application/json'      -d '{"query":"如何在本地部署 RAGFlow？","top_k":3}'
   ```

> 若存在反向代理/Nginx，也可将 8000 暴露至 80/443 并配置 HTTPS。

---

## GPU 支持（可选）

如果主机为 **NVIDIA GPU**：
1. 安装 `nvidia-container-toolkit` 并验证 `docker run --gpus all nvidia/cuda:12.2.0-base nvidia-smi` 可用。  
2. 在 Compose 中的 `ragflow` 服务下添加：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - capabilities: ["gpu"]
```

或使用 **Compose v2** 的 `runtime`/`device_requests` 方案（依 Docker 版本而定）。

> 不同 RAGFlow 版本 GPU 开关方式可能不同，实际以官方镜像说明为准。

---

## 常见问题（FAQ）与排障

**Q1：`docker compose up -d` 失败或拉取镜像很慢？**  
A：检查网络与镜像源；可使用镜像加速或预先 `docker pull ragflow/ragflow:latest`。

**Q2：`curl /health` 超时/非 200？**  
A：`docker compose ps` 观察容器状态；用 `docker compose logs -f` 查看详细日志，关注端口占用/模型下载失败等。

**Q3：端口 8000 被占用？**  
A：修改 Compose 的 `ports: - "18000:8000"` 并重新 `up -d`。

**Q4：需要自定义模型与嵌入？**  
A：编辑 `ragflow.env`：
```
RAGFLOW_MODEL=Qwen/Qwen3-0.6B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```
重启服务：`docker compose up -d`。

**Q5：Windows 下如何运行？**  
A：请使用 **WSL2** + Docker Desktop，并在 WSL2 的 Linux 发行版中执行上述命令。

---

## 安全与持久化建议

- **敏感配置**放入 `.env`（如私有仓库 Token、向量库连接串），避免硬编码进 Compose。  
- 挂载 `./data` 到独立磁盘或云盘，便于迁移与备份。  
- 暴露外网时务必加 **反向代理 + HTTPS**，并配置访问控制/鉴权。

---

## 扩展与作业建议

- **多容器拆分**：将检索服务、向量库（如 Milvus/PGVector）、前端 UI 拆分为独立服务。  
- **评测脚本**：编写自动化脚本，批量验证同一问题在不同配置（模型、向量维度、召回阈值）下的效果差异。  
- **CI/CD**：将 Compose 与 `.env` 纳入仓库（排除敏感项），在实验环境中一键部署。

---

## 许可证与参考

- 本文档与示例脚本仅用于**教学与研究**。  
- 参考：RAGFlow 官方镜像与 README（以其最新说明为准）。

> 如需将本文档纳入课程资料，可直接拷贝或按需删改条目。

