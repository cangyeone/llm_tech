# 第六课：知识库与 RAG 实战

本课聚焦检索增强生成（RAG）体系的关键模块，从向量数据库选型、文档预处理到本地化部署与效果评估，帮助学员搭建端到端知识库问答系统。所有脚本的注释均为中文，便于课堂讲解与自学复现。

## 实验目录

1. `01_vector_store_selection/knowledge_base_selection.py`
   - 比较 FAISS、Chroma、Elasticsearch 的检索性能、资源成本与生态特点。
2. `02_data_preprocessing/data_preprocess.py`
   - 演示文档分块、Sentence-BERT 向量化与质量监控流程。
3. `03_vector_db_build/build_vector_store.py`
   - 构建 FAISS 向量库并实现基础语义检索接口，支持批量写入与增量更新。
4. `04_rag_architecture/rag_architecture_demo.py`
   - 解析 RAG 架构中检索、生成、排序组件的协作，并给出 Qwen3 推理示例。
5. `05_ragflow_deployment/ragflow_demo.py`
   - 说明如何本地化部署 RAGFlow，包含环境准备、配置模板与测试脚本。
6. `06_optimization/optimization_lab.py`
   - 提供检索召回率与生成准确性调优的指标监控与实验建议。

## 使用提示

- 建议先在小规模数据集上验证流程，再扩展至生产数据。
- 对接第三方服务（如 Elasticsearch、RAGFlow）时，可根据课堂环境修改连接参数。
- 所有脚本均预留 `main` 函数或命令行接口，方便集成到自动化实验管线。
