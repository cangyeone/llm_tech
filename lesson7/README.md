# 第七课：知识库优化与混合推理

本课聚焦企业级知识库在客服等业务场景的进阶应用，涵盖混合检索、流程编排、增量更新、案例拆解、混合推理与性能调优。所有示例脚本均提供中文注释，方便课堂讲解与自学复现。

## 实验目录

1. `01_hybrid_retrieval/hybrid_retrieval.py`
   - 演示 BM25+向量检索融合与重排序模型的调用方式，支持参数化实验。
2. `02_langchain_orchestration/langchain_rag_pipeline.py`
   - 解析基于 LangChain 架构的 RAG 流程编排，模拟链路调度与监控指标。
3. `03_incremental_updates/update_strategy.py`
   - 展示增量索引、批量重建与定期重训练策略，提供变更检测与审计日志示例。
4. `04_enterprise_case/customer_service_case.py`
   - 基于企业内部知识库的客服案例分析，涵盖数据管线、对话策略与指标评估。
5. `05_hybrid_inference/hybrid_inference.py`
   - 演示将微调后的 Qwen 模型与 RAG 检索结果组合，实现决策逻辑与故障回退。
6. `06_performance_testing/performance_benchmark.py`
   - 提供检索延迟与生成速度的压测脚本，支持多配置对比与可视化输出。

## 使用提示

- 脚本默认使用内存数据或模拟组件，方便在课堂环境快速演示，可按需替换为真实服务。
- 涉及外部依赖（如 LangChain、Elasticsearch）的位置均在注释中标注，集成时请安装对应库。
- 建议结合 Lesson 4-6 的数据与模型产出，逐步搭建完整的企业知识库对齐与问答体系。
