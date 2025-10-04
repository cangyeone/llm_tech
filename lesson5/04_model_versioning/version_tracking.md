# 教程：MLflow 与 Weights & Biases 的模型版本管理

## 学习目标
- 理解如何同时使用 MLflow 与 WandB 记录对齐实验指标。
- 掌握实验上下文、参数、指标与示例输出的记录方式。
- 学会配置离线模式与本地存储，便于课堂环境演示。

## 背景原理
模型版本管理涉及实验元数据、指标与可复现性。MLflow 负责追踪参数、指标、模型 artifact；WandB 适合可视化与协作。二者联合使用可形成如下流程：
1. 设置实验上下文并启动 run。
2. 记录模型参数量、响应长度等指标。
3. 将示例输出保存到仪表板，便于回顾。

## 代码结构解析
- `TrackingArguments`：配置模型名称、实验名、MLflow URI、WandB 模式等。
- `mlflow_run_context`：上下文管理器，自动开启/结束 MLflow run。
- `init_wandb`：初始化 WandB 项目，支持 `offline` 模式。
- `sample_generation`：调用模型生成示例回答，同时统计参数量。
- `log_metrics`：向两个系统写入指标与参数，并将回答存入 WandB `summary`。

## 实践步骤
1. 在本地启动 MLflow UI：`mlflow ui --backend-store-uri file:./mlruns`。
2. 若需要在线同步，将 `wandb_mode` 设置为 `online` 并执行 `wandb login`。
3. 运行脚本后，在 MLflow 与 WandB 仪表板查看记录的指标与示例回答。
4. 将脚本嵌入训练流程，每次迭代自动记录版本信息。

## 拓展问题
- 如何在 MLflow 中记录模型权重（`mlflow.pytorch.log_model`）以支持一键部署？
- WandB 支持自定义表格与可视化，能否展示训练损失、偏差检测结果等多维信息？
- 若实验较多，应如何定义命名规范与标签方便检索？
