
# 课程实验：RAG 系统简易评估（Recall@K 与语义相似度）— 教学版文档

本教学脚本演示如何用 **Sentence-BERT** 与简单规则对一个 RAG（Retrieval-Augmented Generation）系统进行**最小可运行**的离线评估：
- 构造一个**小型评估集**（查询/检索片段/生成答案）
- 计算 **Recall@K**（召回率）
- 计算 **语义相似度**（余弦相似度，基于 Sentence-BERT）
- 给出可操作的**优化建议清单**

> 适合课堂上快速跑通指标计算与结果解读，帮助学生建立“检索-生成-评估”的端到端意识。

---

## 目录
- [课程实验：RAG 系统简易评估（Recall@K 与语义相似度）— 教学版文档](#课程实验rag-系统简易评估recallk-与语义相似度-教学版文档)
  - [目录](#目录)
  - [运行环境与依赖](#运行环境与依赖)
  - [快速开始](#快速开始)
  - [评估数据结构](#评估数据结构)
  - [核心函数与用法说明](#核心函数与用法说明)
    - [`recall_at_k(retrieved, relevant, k=3)`](#recall_at_kretrieved-relevant-k3)
    - [`semantic_similarity(answer_1, answer_2)`](#semantic_similarityanswer_1-answer_2)
    - [`evaluate_model()`](#evaluate_model)
    - [`output_optimization_suggestions()`](#output_optimization_suggestions)
  - [脚本主流程](#脚本主流程)
  - [示例输出](#示例输出)
  - [教学提示与常见坑](#教学提示与常见坑)
  - [扩展建议（课后作业）](#扩展建议课后作业)
  - [许可证](#许可证)

---

## 运行环境与依赖

- Python 3.8+
- 依赖库：
  - `sentence-transformers`（Sentence-BERT 编码）
  - `scikit-learn`（余弦相似度）
  - `numpy`

安装：
```bash
pip install sentence-transformers scikit-learn numpy
```

> **模型路径说明**：代码默认 `SentenceTransformer('./Qwen/paraphrase')`。若本地无该目录或模型，建议改用公开可下载的模型名（如 `sentence-transformers/all-MiniLM-L6-v2`）。

---

## 快速开始

将脚本保存为 `lesson6/06_optimization/optimization_lab.py`，然后直接运行：
```bash
python lesson6/06_optimization/optimization_lab.py
```
你将看到每条样本的 **Recall@K** 与 **语义相似度**，以及一份“优化建议”清单。

---

## 评估数据结构

脚本内置 `evaluation_data`（教学样例），每个样本包含：

```python
{
  "query": "问题文本",
  "retrieved": ["检索片段1", "检索片段2", ...],
  "generated_answer": "生成答案（用于示例）"
}
```

- **query**：用户查询。
- **retrieved**：RAG 检索阶段返回的候选文本片段（按相关性排序）。
- **generated_answer**：生成模型的回答（教学示例中也被用于“相关项”参考）。

> 实际项目应提供**独立的标注答案或相关片段列表**，这里为了演示最小流程，简化为“用生成答案当作相关项”。

---

## 核心函数与用法说明

### `recall_at_k(retrieved, relevant, k=3)`

**作用**：在前 `k` 个检索结果中，有多少属于“相关集合”（`relevant`）。

**参数**：
- `retrieved: List[str]`：按相关性排序的检索结果列表。
- `relevant: List[str]`：被认为是“相关”的文本集合。
- `k: int`：只看前 `k` 条检索结果。

**返回**：`float`，范围 `[0, 1]`。值越大，说明前 `k` 的检索命中越多。

**教学提示**：
- 本脚本用 `generated_answer` 作为 `relevant` 的唯一元素（非常简化，仅为演示）。
- 真实评估中，`relevant` 应该是**人工标注**或**来源标注**的**文档片段集合**。

---

### `semantic_similarity(answer_1, answer_2)`

**作用**：用 Sentence-BERT 对两段文本编码并计算**余弦相似度**。

**参数**：
- `answer_1: str`：答案/文本 1
- `answer_2: str`：答案/文本 2

**返回**：`float`，范围通常在 `[0, 1]`。越接近 1 表示越相似。

**实现要点**：
- 使用 `SentenceTransformer.encode` 得到两个向量
- 用 `sklearn.metrics.pairwise.cosine_similarity` 计算余弦相似度

**教学提示**：
- 选择合适的句向量模型很关键（通用 vs. 领域适配）。
- 相似度阈值应结合业务分布设定，并配合人工抽查。

---

### `evaluate_model()`

**作用**：对 `evaluation_data` 中的每个样本：
1. 以 `generated_answer` 作为**相关集合**（教学简化）。
2. 计算并打印 `Recall@2`。
3. 计算并打印生成答案与“参考答案”的语义相似度（教学中用同一文本，得到 1.0）。

**实战建议**：
- 将 `relevant` 替换为**标注的相关文档集合**，例如 `relevant = ground_truth_passages[data["query"]]`。
- 将“参考答案”换成**独立的真实答案**（gold answer）。

---

### `output_optimization_suggestions()`

**作用**：打印**可操作优化建议**，覆盖检索/生成/排序三阶段：
1. 增大召回数量（`top_k`）
2. 尝试更强的检索器（BM25 / Dense）
3. 领域数据微调生成模型
4. 加入**重排序（re-ranking）**
5. 调整生成参数（`max_length`、`temperature` 等）

**教学提示**：引导学生将指标结果与策略调整关联起来，形成闭环。

---

## 脚本主流程

```python
def main():
    print("Evaluating RAG system:")
    evaluate_model()                 # 计算/打印 Recall@K 与语义相似度
    output_optimization_suggestions()# 打印优化建议
```
直接运行脚本即可执行上述两步。

---

## 示例输出

> 由于教学脚本中“参考答案 = 生成答案”，语义相似度会接近 1。Recall@K 则取决于 `retrieved` 中是否包含该文本。

```text
Evaluating RAG system:
Recall@2 for query 'What is RAG?': 0.00
Semantic similarity for query 'What is RAG?': 1.00
==================================================
Recall@2 for query 'How does RAG improve text generation?': 0.00
Semantic similarity for query 'How does RAG improve text generation?': 1.00
==================================================
Recall@2 for query 'What is the goal of RAG?': 0.00
Semantic similarity for query 'What is the goal of RAG?': 1.00
==================================================
Optimization Suggestions:
1. Increase the number of retrieved documents to ensure more relevant information is captured.
2. Use advanced retrieval techniques (e.g., BM25 or Dense Retrieval) to improve retrieval precision.
3. Fine-tune the generative model using task-specific data to improve answer accuracy.
4. Consider adding re-ranking techniques to reorder retrieved documents for better contextual relevance.
5. Experiment with different generation parameters (e.g., `max_length`, `temperature`, etc.) to optimize answer generation.
```

---

## 教学提示与常见坑

1. **不要用生成答案当作相关文档**（这里只是演示）：
   - 正确做法：准备 `relevant_passages`（人工标注或来源可追溯）。

2. **相似度模型的选择影响很大**：
   - 领域异构时（法律/医疗/财税），请考虑领域微调或使用更强的跨编码器 re-ranker。

3. **Recall 与生成质量并非等价**：
   - Recall@K 高并不保证答案正确；需要**答案质量指标**（如 F1、ROUGE、Exact Match、FactScore 等）配合。

4. **评估集规模**：
   - 建议至少上百条查询；课堂演示用 3 条仅为流程展示。

5. **编码性能**：
   - 批量编码可用 `SentenceTransformer.encode(..., batch_size=...)`。

---

## 扩展建议（课后作业）

- **A. 引入真实相关集**：为每个 query 标注 1~N 个 ground-truth passages，替换本脚本的简化逻辑。  
- **B. 加入生成质量评估**：
  - 自动：ROUGE / BLEU / BERTScore / QAFactEval / Faithfulness 检测
  - 人工：结构化打分表（信息覆盖、事实一致性、语气风格）
- **C. 加入重排序**：对 `retrieved` 进行 re-ranking（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）。  
- **D. 指标看板**：将指标写入 CSV，并用 Pandas + 可视化展示每次实验对比。  
- **E. 误差分析**：输出 Top-K 未命中的 case，打印最近邻差异，做错误归因。

---

## 许可证

本教学文档与示例代码仅用于**教学/研究用途**。请遵守所用模型与数据的相应许可协议。
