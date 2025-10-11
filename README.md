# LLM Tech 课程实验总览

本仓库汇集了「LLM Tech」系列课程的全部实验脚本与配套资料，按照课程节次划分为七大模块，覆盖监督微调、参数高效微调、分布式训练、模型对齐、落地监控以及 RAG 检索增强应用等核心主题。每个子目录都提供结构化的代码示例与中文注释，便于在课堂上演示或课后复现。

## 课程结构与学习要点

### Lesson 1 · 监督与指令微调基础
- **学习目标**：理解监督微调（SFT）的流程、指令数据构建方法，以及 LoRA 在轻量模型上的应用与部署策略。
- **实践路径**：
  - 通过 `01_sft_basics/sft_theory_notebook.py` 梳理 SFT 与指令微调的机制差异与最小化示例。
  - 利用 `02_data_preprocess/data_pipeline.py` 完成文档转写、Prompt 模板构建与数据清洗，掌握数据工程步骤。
  - 在 `03_hf_finetune/hf_finetune_scaffold.py` 中搭建 Hugging Face Trainer 脚手架，熟悉训练参数与日志配置。
  - 借助 `04_lora_theory/lora_parameter_viz.py` 与 `05_lora_finetune/lora_chatglm_demo.py` 观察 LoRA 秩分解、注入位置及 ChatGLM 微调流程。
  - 通过 `06_cpu_deploy/cpu_inference_case.py` 评估轻量模型在 CPU 环境的吞吐表现，理解部署取舍。

### Lesson 2 · QLoRA、P-Tuning 与模型压缩
- **学习目标**：掌握 QLoRA 的量化思路、端到端微调流程，理解 P-Tuning v2 的提示学习范式，并熟悉多种模型压缩与部署策略的取舍。
- **实践路径**：
  - 在 `01_qlora_intro/qlora_theory.py` 中解析页面量化、PagedOptimizers 与梯度检查点，夯实低比特训练理论。
  - 结合 `02_qlora_pipeline/qlora_trainer.py` 复现 LLaMA-7B QLoRA 微调，熟悉训练脚手架与配置。
  - 通过 `03_ptuning/ptuning_v2_demo.py` 构建可学习提示词，体验 P-Tuning v2 的 Prompt 学习范式。
  - 借助 `04_model_compression/compression_lab.py` 对比剪枝、蒸馏、低比特量化等压缩策略的精度与成本。
  - 使用 `05_deepspeek_quant/deepspeek_deploy.py` 实践 DeepSpeed + bitsandbytes 的量化部署。
  - 通过 `06_method_selection/method_selector.py` 梳理 LoRA、QLoRA、P-Tuning 的选型建议。

### Lesson 3 · 分布式训练与性能调优（需要多GPU）
- **学习目标**：构建对 ZeRO-3、混合精度与大规模分布式训练的整体认知，掌握训练加速技巧、日志诊断与评估指标体系。
- **实践路径**：
  - 使用 `01_distributed_training/zero3_configurator.py` 一键生成 ZeRO-3、混合精度与 Offload 配置，理解显存规划方案。
  - 在 `02_8gpu_config/ds_config_builder.py` 中阅读 8 卡 671B-DS 的参考配置，学习参数切分与通信策略。
  - 通过 `03_training_acceleration/acceleration_tricks.py` 练习 FlashAttention、梯度检查点等显存优化技巧。
  - 借助 `04_log_analysis/training_log_inspector.py` 分析训练日志并可视化吞吐、梯度与损失曲线。
  - 在 `05_evaluation/evaluation_metrics.py` 中生成困惑度、人工评估表格，搭建评估体系。
  - 结合 `06_vertical_case/domain_finetune_case.py` 探索垂直场景分布式微调的配置与落地经验。

### Lesson 4 · 大模型对齐与增量学习
- **学习目标**：系统梳理 RLHF、DPO、KTO 等主流对齐技术，掌握奖励模型训练、增量学习与对齐数据构建方法。
- **实践路径**：
  - 通过 `01_alignment_overview/compare_methods.py` 对比 RLHF、DPO、KTO 的机制与适用场景。
  - 借助 `02_rlhf_pipeline/rlhf_pipeline.py` 串联奖励模型训练与 PPO 微调，复现实战对齐流程。
  - 在 `03_reward_model_training/train_reward_model.py` 中演练偏好数据处理与奖励模型评估。
  - 使用 `04_dpo/dpo_train.py` 复现 DPO 训练并理解其目标函数推导。
  - 结合 `05_incremental_learning/incremental_lora.py` 探索增量 LoRA、模型合并与灾难性遗忘缓解策略。
  - 通过 `06_dataset_curation/dataset_curation.py` 构建与审查对齐数据集质量。

### Lesson 5 · 对齐系统落地与监控
- **学习目标**：了解在真实业务中部署对齐模型的流程，包括平台化 DPO、偏差检测、安全合规与模型版本管理。
- **实践路径**：
  - 使用 `01_qwen_workflow/qwen_alignment_workflow.py` 打通 Qwen 对齐从数据接入到上线的端到端流程。
  - 通过 `02_dpo_hf/dpo_hf_run.py` 在 Hugging Face 平台复现 DPO 训练，掌握云端资源配置。
  - 借助 `03_bias_detection/bias_scan.py` 构建偏差检测与安全审查流水线。
  - 在 `04_model_versioning/version_tracking.py` 中结合 MLflow 与 W&B 管理模型版本与指标。
  - 以 `05_customer_support_case/customer_alignment.py` 拆解客服场景的提示策略与反馈闭环。
  - 通过 `06_diversity_debug/diversity_evaluation.py` 调试回答多样性，优化用户体验。

### Lesson 6 · 知识库与 RAG 实战（05课程部分需要进一步测试）
- **学习目标**：搭建检索增强生成（RAG）系统，掌握向量库选型、文档预处理、架构搭建、本地化部署与效果优化流程。
- **实践路径**：
  - 在 `01_vector_store_selection/knowledge_base_selection.py` 中对比 FAISS、Chroma、Elasticsearch 的性能与生态。
  - 借助 `02_data_preprocessing/data_preprocess.py` 完成文档分块、Sentence-BERT 向量化与数据质量监控。
  - 使用 `03_vector_db_build/build_vector_store.py` 构建 FAISS 向量库，支持批量写入与增量更新。
  - 通过 `04_rag_architecture/rag_architecture_demo.py` 解析检索、生成、重排序组件协作，并提供 Qwen3 推理示例。
  - 在 `05_ragflow_deployment/ragflow_demo.py` 中学习 RAGFlow 的本地部署与调试方法。
  - 结合 `06_optimization/optimization_lab.py` 建立检索召回率与生成准确性指标监控，实现持续优化。

### Lesson 7 · 知识库优化与混合推理
- **学习目标**：面向企业级知识库场景，掌握混合检索、流程编排、增量更新、混合推理与性能压测的最佳实践。
- **实践路径**：
  - 通过 `01_hybrid_retrieval/hybrid_retrieval.py` 构建 BM25 + 向量混合检索并评估重排序策略。
  - 在 `02_langchain_orchestration/langchain_rag_pipeline.py` 中编排 LangChain RAG 流程，模拟链路调度与监控指标。
  - 使用 `03_incremental_updates/update_strategy.py` 设计增量索引、批量重建与审计日志方案。
  - 阅读 `04_enterprise_case/customer_service_case.py` 拆解企业客服知识库案例，学习数据管线与对话策略。
  - 借助 `05_hybrid_inference/hybrid_inference.py` 探索对齐模型与 RAG 检索的协同推理与故障回退逻辑。
  - 通过 `06_performance_testing/performance_benchmark.py` 执行检索延迟与生成速度压测，对比多配置表现。

## 使用建议
- 建议先阅读各课 `README`，根据自身算力与业务场景选择合适的示例脚本与模型规模。
- 大部分脚本支持通过命令行参数或配置文件调整模型名称、批大小、向量库连接等关键参数，可灵活适配不同环境。
- 若需扩展到生产环境，可结合 Lesson 4-7 的对齐与 RAG 工具链，搭建端到端的企业级问答与知识服务体系。
- 大模型可以使用modelscope下载模型。
