# 教程：LangChain 风格的 RAG 流程编排

## 学习目标
- 理解 RAG 管线中检索、重排序、生成、审计等阶段的串联方式。
- 学会使用 StageContext 保存链路状态并在阶段间传递。
- 掌握记录指标与生成审计日志的方法，支撑可观测性建设。

## 背景原理
LangChain 提供链式组件（Chain）与执行器（Runnable）组织复杂流程。本脚本用纯 Python 模拟这一概念：
1. **retrieval**：根据 `top_k` 返回文档片段。
2. **rerank**：按简单规则重新排序，模拟关键字加权。
3. **generation**：调用模型生成回答（示例中为模板字符串）。
4. **audit**：写入日志，记录上下文与指标。

## 代码结构解析
- `StageContext`：存储 query、检索结果、生成内容与指标。
- `PipelineStage`：封装节点名称与处理函数，提供统一 `run` 接口。
- `RAGPipeline`：按顺序执行多个 stage，形成可配置的流程。
- `build_pipeline`：根据参数组装四个阶段，便于课堂演示扩展。
- `main`：解析命令行，运行管线并打印最终回答与指标。

## 实践步骤
1. 执行示例命令：
   ```bash
   python langchain_rag_pipeline.py "企业如何搭建RAG系统？" --top_k 2 --model Qwen3-7B-Chat
   ```
2. 查看控制台输出的阶段执行顺序、最终回答与指标。
3. 打开 `logs` 目录，检查生成的 `rag_audit_*.json`，讨论日志字段设计。
4. 将 `retrieval_stage` 替换为真实检索接口（如 FAISS/Chroma），把 `generation_stage` 改为调用 Qwen API。

## 拓展问题
- 如何引入条件分支，例如根据召回数量决定是否补充检索？
- 审计日志中还应记录哪些信息（响应时间、用户 ID）以满足合规要求？
- 可否结合 LangChain 的 `RunnableParallel` 实现并行检索与重排序？
