# 教程：知识库向量引擎选型评估

## 学习目标
- 比较 FAISS、Chroma、Elasticsearch 等常见向量引擎的优缺点。
- 通过模拟数据评估建库时间、查询延迟与内存占用。
- 掌握选型时需关注的业务因素：实时性、运维成本、生态支持。

## 背景原理
向量检索系统需在构建速度、查询效率与资源占用之间权衡。内存占用与向量数量、维度呈线性关系：
\[
\text{Memory} \approx N \times D \times 4\text{ bytes}.
\]
不同引擎在索引结构、分片方式、增量更新能力上存在差异，需要结合业务需求评估。

## 代码结构解析
- `BackendProfile`：记录建库耗时、查询延迟、内存占用与特色说明。
- `simulate_vector_data`：生成随机向量，模拟真实嵌入数据。
- `benchmark_backend`：模拟建库和查询流程，输出性能指标。
- `summarize_backend_features`：总结各引擎的课堂讨论要点。
- `render_report`：将评估结果转为 Markdown 表格，方便展示。

## 实践步骤
1. 根据业务规模调整 `--num_vectors` 与 `--dim`，评估内存需求。
2. 运行脚本查看输出表格，并与实际 SDK 测试结果对照。
3. 结合课堂讨论，引导学生根据延迟、预算、部署难度选择合适方案。
4. 可扩展 `backends` 列表，加入 Milvus、Weaviate 等云原生引擎。

## 拓展问题
- 如何在 Elasticsearch 中实现向量检索与 BM25 混合排序？
- FAISS 支持多种索引类型（IVF、HNSW、PQ），应如何根据数据规模选择？
- 若需高可用架构，Chroma/FAISS 是否需要自建副本机制或借助外部服务？
