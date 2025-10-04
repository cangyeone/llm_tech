# 教程：RAG 架构详解与 Qwen 推理示例

## 学习目标
- 了解 RAG 系统中检索、排序、生成、监控等核心组件。
- 掌握简单的检索-排序-生成流水线实现方式。
- 使用 Qwen 模型结合知识库上下文生成问答示例。

## 背景原理
RAG（Retrieval-Augmented Generation）通过以下步骤提升问答准确性：
1. **Retriever**：基于向量相似度召回候选文档，关注召回率与延迟。
2. **Ranker**：对候选文档重新排序，提升相关性与精确率。
3. **Reader**：将用户问题与上下文拼接输入大模型，生成答案。
4. **Monitor**：记录日志、指标与反馈，用于持续优化与报警。

## 代码结构解析
- `RagComponent` / `build_components`：总结各组件职责与关键指标，输出 Markdown 表格。
- `SimpleRetriever`：封装 FAISS 索引的检索逻辑。
- `SimpleRanker`：用关键词命中率演示再排序机制，可替换为语义 ranker。
- `QwenReader`：调用 Qwen 模型生成回答，支持最大生成长度配置。
- `rag_pipeline`：串联检索、排序、生成流程，输出最终答案。

## 实践步骤
1. 准备 Lesson 6 生成的文本与向量文件，确保 FAISS 索引可用。
2. 运行脚本：
   ```bash
   python rag_architecture_demo.py outputs/preprocess/chunks.txt outputs/preprocess/embeddings.npy --top_k 3
   ```
3. 查看控制台输出的组件表格与示例问答，理解各环节作用。
4. 替换 `SimpleRanker` 与 `query_vec` 逻辑，接入实际的 ranker 与编码器。

## 拓展问题
- 如何在检索阶段加入 BM25 结果并进行融合排序？
- Qwen 生成时如何使用提示模板控制回答格式与引用？
- Monitor 模块应记录哪些指标（如命中率、人工反馈）以支撑 A/B 测试？
