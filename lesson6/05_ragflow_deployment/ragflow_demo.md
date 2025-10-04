# 教程：RAGFlow 本地化部署与健康检查

## 学习目标
- 理解 RAGFlow 部署流程及所需配置文件。
- 学会使用 Docker Compose 启动服务并进行健康检查。
- 掌握调用 RAGFlow API 进行文档问答测试的方法。

## 背景原理
RAGFlow 将检索、嵌入、生成等组件封装为容器化服务。部署流程通常包含：
1. 准备环境变量与模型配置。
2. 编写 `docker-compose.yaml`，描述服务端口、卷挂载等信息。
3. 启动服务并执行健康检查，确保 API 可用。
4. 通过 REST 接口发送查询，验证检索与生成链路。

## 代码结构解析
- `DeploymentStep`：封装每个步骤的标题、命令与注意事项。
- `build_steps`：根据配置目录与模型名称生成四个关键步骤（环境变量、Compose 文件、启动、健康检查）。
- `run_command`：执行命令并输出日志，便于排错。
- `demo_query`：展示调用 `/api/v1/rag/query` 的示例 cURL 请求。
- `main`：支持 `--dry_run` 仅打印命令，或真实执行部署流程。

## 实践步骤
1. 创建配置目录，例如 `deploy/ragflow`，并运行脚本：
   ```bash
   python ragflow_demo.py deploy/ragflow --dry_run
   ```
   检查生成的命令与提示。
2. 移除 `--dry_run` 执行实际部署，关注日志输出是否有错误。
3. 运行健康检查命令 `curl http://localhost:8000/health`，确认服务状态为 200。
4. 使用 `demo_query` 输出的 cURL 请求测试问答效果，可结合 Postman 或前端页面。

## 拓展问题
- 若需 GPU 推理，应如何在 Compose 文件中配置 `deploy.resources`？
- 如何将 RAGFlow 与企业内部认证系统集成，保护知识库安全？
- 部署完成后，哪些监控指标（QPS、延迟、错误率）需要接入告警系统？
