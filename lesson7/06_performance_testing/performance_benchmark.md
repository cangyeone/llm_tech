
# 教学实验：RAG 问答系统的性能压测与可视化（检索 + 生成时延）

本教学脚本以一个**最小可运行**的 RAG（Retrieval-Augmented Generation）原型为例，演示**端到端时延**的构成与分析方法：
- 使用 **Sentence-BERT** 做语义检索（Retriever）
- 使用 **生成模型（以 Qwen 为例）** 做回答生成（Generator）
- 对**检索时延**、**生成时延**与**总时延**进行统计（平均、P95、最大）
- 绘制**时延分布直方图**，帮助识别长尾

> 该脚本关注“**如何量化**”与“**如何观察**”，便于课堂演示和作业扩展。

---

## 目录
- [教学实验：RAG 问答系统的性能压测与可视化（检索 + 生成时延）](#教学实验rag-问答系统的性能压测与可视化检索--生成时延)
  - [目录](#目录)
  - [环境与依赖](#环境与依赖)
  - [快速开始](#快速开始)
  - [脚本结构与数据流](#脚本结构与数据流)
  - [核心函数说明](#核心函数说明)
    - [`preprocess(text)`](#preprocesstext)
    - [`retrieve_answer(query, knowledge_base)`](#retrieve_answerquery-knowledge_base)
    - [`generate_answer(query, context)`](#generate_answerquery-context)
    - [`performance_test(queries, knowledge_base, threshold=0.7)`](#performance_testqueries-knowledge_base-threshold07)
    - [`calculate_metrics(times)`](#calculate_metricstimes)
    - [`plot_performance(times, title="Latency Distribution")`](#plot_performancetimes-titlelatency-distribution)
  - [运行示例与输出解读](#运行示例与输出解读)
  - [教学要点与常见坑](#教学要点与常见坑)
  - [性能优化清单](#性能优化清单)
  - [扩展作业建议](#扩展作业建议)
  - [许可证](#许可证)

---

## 环境与依赖

- Python 3.8+
- 依赖库：
```bash
pip install sentence-transformers transformers scikit-learn numpy matplotlib
```

> **模型建议**：
> - 句向量模型：`paraphrase-MiniLM-L6-v2`（轻量、下载快）。
> - 生成模型：示例使用 `Qwen/Qwen3-0.6b`；实际可替换为任何 HuggingFace 上可用的 Causal LM（A100/GPU 更快）。

---

## 快速开始

1. 将代码保存为 `lesson7/06_performance_testing/performance_benchmark.py`。  
2. 直接运行：
   ```bash
   python lesson7/06_performance_testing/performance_benchmark.py
   ```
3. 观察控制台打印的**检索/生成/总时延**的 `Avg / P95 / Max`；随后弹出**总时延直方图**。

---

## 脚本结构与数据流

```
Query 集合 ──> 检索（Sentence-BERT）计时 ──┐
                                         │
                                         ├─> 生成（Qwen）计时 ──> 汇总统计（Avg/P95/Max）
                                         │
知识库（标准问句->答案） ───────────────┘              └─> 直方图可视化
```

- **知识库**：简单的“问句→答案”字典。  
- **检索阶段**：将用户 Query 与知识库问句编码为向量，计算余弦相似度，取 Top-1 作为上下文。  
- **生成阶段**：将 `query + retrieved_answer` 拼接输入生成模型，得到自然语言回答。  
- **压测**：对一组 Query 逐一执行“检索+生成”，统计每阶段耗时与总耗时。

---

## 核心函数说明

### `preprocess(text)`
**作用**：示例中的简单文本清洗（小写、去标点、分词）。  
**签名**：`preprocess(text: str) -> List[str]`  
**说明**：本脚本未在检索流程中使用该函数（保留用于扩展 BM25/规则匹配）。

---

### `retrieve_answer(query, knowledge_base)`
**作用**：在知识库中检索最相似问句，返回其答案与相似度。  
**签名**：`retrieve_answer(query: str, knowledge_base: Dict[str, str]) -> Tuple[str, float]`  
**流程**：
1. 用 Sentence-BERT 对**知识库问句**与**用户查询**分别编码；
2. 计算余弦相似度；
3. 取相似度最高的问句对应的答案；
4. 返回 `(best_match_answer, similarity_score)`。

> **注意**：示例每次都重新编码知识库问句，真实系统应**缓存/预编索引**。

---

### `generate_answer(query, context)`
**作用**：调用生成式模型（如 Qwen）基于 Query+Context 生成答案。  
**签名**：`generate_answer(query: str, context: str) -> str`  
**要点**：
- 使用 `tokenizer(..., max_length=512, truncation=True)` 控制上限；
- 使用 `model.generate(..., max_length=150)` 控制输出长度；
- 可扩展采样参数（`temperature`、`top_p`）。

---

### `performance_test(queries, knowledge_base, threshold=0.7)`
**作用**：对多条 Query 逐条执行“检索 + 生成”，并逐条记录各阶段耗时。  
**签名**：
```python
performance_test(
    queries: List[str],
    knowledge_base: Dict[str, str],
    threshold: float = 0.7   # 预留阈值位，当前示例未启用切换逻辑
) -> Tuple[List[float], List[float], List[float]]
```
**返回**：`(retrieval_times, generation_times, total_times)` 三个等长列表。

> **提示**：若要加入“相似度阈值直返/转生成”的逻辑，可在此函数中按阈值分支，并对不同分支分别计时。

---

### `calculate_metrics(times)`
**作用**：计算时延向量的统计指标。  
**签名**：`calculate_metrics(times: List[float]) -> Tuple[float, float, float]`  
**返回**：`(avg_time, p95_time, max_time)`。

- **Avg**：平均时延  
- **P95**：第 95 百分位时延，衡量长尾  
- **Max**：本轮压测中的最大时延

---

### `plot_performance(times, title="Latency Distribution")`
**作用**：以直方图展示时延分布。  
**签名**：`plot_performance(times: List[float], title: str) -> None`  
**说明**：课堂演示可替换为箱线图/核密度曲线，以展示长尾更清晰。

---

## 运行示例与输出解读

示例 Query：
```text
What is RAG?
How does RAG improve text generation?
What is the goal of RAG?
Explain the RAG system in detail.
How can I optimize RAG for performance?
```

可能的控制台输出（示意）：
```
Retrieval Latency - Avg: 0.0123, P95: 0.0201, Max: 0.0304
Generation Latency - Avg: 0.4321, P95: 0.6799, Max: 0.9110
Total Latency - Avg: 0.4444, P95: 0.6950, Max: 0.9314
```
- **检索**耗时远低于**生成**，总时延主要受生成阶段影响；
- 关注 **P95/Max** 以识别长尾瓶颈（如首次加载、显存换页、系统抖动）。

---

## 教学要点与常见坑

1. **向量缓存**：示例每次编码知识库问句，真实系统应**一次编码、多次复用**或使用**向量库**（FAISS/Milvus）。  
2. **IO 与冷启动**：首次调用生成模型可能更慢；可在服务启动时做**warmup**。  
3. **显存/内存**：`max_length` 过大或 batch 过大可能导致 OOM；关注 GPU/CPU 内存曲线。  
4. **随机性**：生成过程默认贪心；若启用采样，时延可能随参数波动。  
5. **可重复性**：压测时固定随机种子、清理缓存、独占硬件，结果更可比。  
6. **绘图环境**：无显示环境（服务器）下，使用 `matplotlib` 非交互后端保存到文件（例如 `Agg`）。

---

## 性能优化清单

- **检索部分**
  - 预编码知识库问句 → 向量缓存或 FAISS 检索
  - 向量归一化、批量化编码
- **生成部分**
  - 使用 **更小的模型/量化**（如 4-bit/8-bit）
  - 采用 **显存高效**选项（如 `attn_implementation="eager"`/Flash-Attn 视硬件）
  - 限制 `max_length` 与解码步数（Beam/Top-p 等）
- **系统层面**
  - 复用会话、启用 `torch.compile`（PyTorch 2.x）
  - 服务化部署 + 并发队列 + 限流
  - 端到端 A/B 验证 QoS（P95/P99）

---

## 扩展作业建议

1. **分支策略**：按相似度阈值决定“直返/生成/拒答”，分别统计三类请求的 P95。  
2. **可视化增强**：绘制三段时延的 **箱线图** 与 **CDF 曲线**。  
3. **混合检索**：引入 BM25，与向量相似度做权重融合后再进入生成。  
4. **向量库接入**：把知识库问句写入 FAISS/Milvus，观察检索时延变化。  
5. **Batch 压测**：把 Queries 扩展为 1k/10k，比较不同批量与并发下的时延曲线。

---

## 许可证

本教学文档与示例代码仅用于**教学/研究**用途。请遵循所用模型与数据集的许可条款。
