
# 教学实验：混合检索（BM25 + Sentence-BERT 向量检索）

本脚本演示了如何通过**稀疏检索（BM25）**与**稠密检索（Sentence-BERT 向量相似度）**融合，实现一个简单的**混合检索系统（Hybrid Retrieval）**。

它通过示例数据演示：
- 文本预处理（小写化 + 去除标点 + 分词）
- 使用 BM25 计算稀疏检索得分
- 使用 Sentence-BERT 计算向量检索得分
- 将两种得分归一化并加权融合
- 输出重排序后的文档与得分
- 计算 Recall@K 与 Precision@K
- 给出优化与改进建议

---

## 目录
- [教学实验：混合检索（BM25 + Sentence-BERT 向量检索）](#教学实验混合检索bm25--sentence-bert-向量检索)
  - [目录](#目录)
  - [运行环境与依赖](#运行环境与依赖)
  - [快速上手](#快速上手)
  - [脚本总体结构](#脚本总体结构)
  - [函数说明](#函数说明)
    - [`preprocess(text)`](#preprocesstext)
    - [`recall_at_k(retrieved, relevant, k=3)`](#recall_at_kretrieved-relevant-k3)
    - [`precision_at_k(retrieved, relevant, k=3)`](#precision_at_kretrieved-relevant-k3)
  - [主要步骤讲解](#主要步骤讲解)
    - [1. 语料准备](#1-语料准备)
    - [2. 预处理](#2-预处理)
    - [3. BM25 检索](#3-bm25-检索)
    - [4. Sentence-BERT 向量检索](#4-sentence-bert-向量检索)
    - [5. 得分归一化与加权融合](#5-得分归一化与加权融合)
    - [6. 结果重排序](#6-结果重排序)
    - [7. 评估指标](#7-评估指标)
  - [输出示例](#输出示例)
  - [优化与扩展建议](#优化与扩展建议)
  - [教学要点总结](#教学要点总结)

---

## 运行环境与依赖

**Python 版本要求：**
- Python ≥ 3.8

**所需库：**
```bash
pip install sentence-transformers rank-bm25 scikit-learn numpy
```

---

## 快速上手

保存脚本为 `lesson7/01_hybrid_retrieval/hybrid_retrieval.py`，然后直接运行：

```bash
python lesson7/01_hybrid_retrieval/hybrid_retrieval.py
```

系统会输出混合检索后的排序结果、各文档的得分，以及召回率与准确率指标。

---

## 脚本总体结构

1. **准备语料库**：定义一个小型 RAG 主题文档集。  
2. **文本预处理**：统一小写、去除标点、分词。  
3. **BM25 检索**：基于词频与逆文档频率计算相关性。  
4. **Sentence-BERT 向量检索**：基于句向量计算余弦相似度。  
5. **得分融合**：将 BM25 和向量得分归一化后按权重合成。  
6. **结果排序**：输出最终的综合排序文档。  
7. **评估指标**：计算 Recall@K、Precision@K。  
8. **输出优化建议**：提供改进方向。

---

## 函数说明

### `preprocess(text)`

**功能：**
- 将输入文本转换为小写。
- 去除所有标点符号。
- 按空格分词，返回单词列表。

**输入参数：**
- `text (str)`：待处理的文本字符串。

**输出：**
- `List[str]`：清洗后的词列表。

**示例：**
```python
preprocess("RAG Improves Generation!") 
# 输出: ["rag", "improves", "generation"]
```

---

### `recall_at_k(retrieved, relevant, k=3)`

**功能：**
计算前 K 个检索结果中，命中的相关文档比例。

**输入参数：**
- `retrieved (List[str])`：检索到的文档列表。
- `relevant (List[str])`：人工标注的相关文档。
- `k (int)`：评估前 K 条结果，默认 3。

**返回：**
- `float`：Recall@K 值。

**计算公式：**
$$
Recall@K = \frac{|retrieved_{topK} \cap relevant|}{\min(K, |relevant|)}
$$

---

### `precision_at_k(retrieved, relevant, k=3)`

**功能：**
计算前 K 个检索结果中，真正相关文档的比例。

**输入参数：**
- `retrieved (List[str])`：检索到的文档列表。
- `relevant (List[str])`：人工标注的相关文档。
- `k (int)`：评估前 K 条结果。

**返回：**
- `float`：Precision@K 值。

**计算公式：**
$$
Precision@K = \frac{|retrieved_{topK} \cap relevant|}{K}
$$

---

## 主要步骤讲解

### 1. 语料准备
使用五段关于 RAG（Retrieval-Augmented Generation）的描述文本构建语料库。

### 2. 预处理
统一大小写、去除标点、按空格分词，确保 BM25 的输入一致性。

### 3. BM25 检索
基于 `rank_bm25` 库实现，原理：
$$
BM25(q, D) = \sum_{t \in q} IDF(t) \cdot \frac{f(t, D) (k_1 + 1)}{f(t, D) + k_1(1 - b + b \cdot \frac{|D|}{avgdl})}
$$

### 4. Sentence-BERT 向量检索
使用 `SentenceTransformer` 将每个文档和查询编码为向量，通过 **余弦相似度** 评估语义接近度。

### 5. 得分归一化与加权融合
为保证 BM25 和向量得分可比性，使用 `MinMaxScaler` 将其映射至 `[0, 1]`，再按权重加权求和：
$$
score_{final} = w_{bm25} * score_{bm25} + w_{vector} * score_{vector}
$$
本示例中，$w_{bm25}=w_{vector}=0.5$。

### 6. 结果重排序
根据融合得分从高到低排序，并输出每个文档的：
- 原始 BM25 得分
- 语义相似度
- 最终融合得分

### 7. 评估指标
计算 Recall@3 与 Precision@3，帮助理解系统在小规模数据上的表现。

---

## 输出示例

```text
Sorted documents by hybrid scores:
Rank 1: Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.
BM25 score: 4.6502, Cosine similarity: 0.8123, Final score: 0.8931
==================================================
Rank 2: By combining pre-trained generative models with retrieval mechanisms, RAG enables the generation of more relevant and informative responses.
BM25 score: 3.8210, Cosine similarity: 0.7834, Final score: 0.8610
==================================================
Recall@3: 1.00
Precision@3: 0.67

Optimization Suggestions:
1. Increase the number of retrieved documents to ensure more relevant information is captured.
2. Use advanced retrieval techniques (e.g., BM25 or Dense Retrieval) to improve retrieval precision.
3. Fine-tune the generative model using task-specific data to improve answer accuracy.
4. Consider adding re-ranking techniques to reorder retrieved documents for better contextual relevance.
5. Experiment with different generation parameters (e.g., `max_length`, `temperature`, etc.) to optimize answer generation.
```

---

## 优化与扩展建议

1. **动态权重调整**  
   - 可用验证集自动学习 BM25 与向量检索的最优融合权重。  
2. **召回扩展**  
   - 增加语料规模或添加外部知识库。  
3. **重排序（Re-ranking）**  
   - 使用 Cross-Encoder 模型对前 K 条候选结果进行语义重排序。  
4. **评估指标扩展**  
   - 引入 MAP、MRR、nDCG 等更丰富的排序指标。  
5. **语义检索优化**  
   - 替换更强的嵌入模型，如 `all-mpnet-base-v2` 或 `text-embedding-3-large`。

---

## 教学要点总结

- **BM25**：基于词频的稀疏检索；适合关键词匹配。  
- **Sentence-BERT**：捕获语义关系的稠密检索。  
- **混合策略**：结合二者优点——兼顾**精确词匹配**与**语义理解**。  
- **评估指标**：Recall 与 Precision 可用于衡量召回率与准确率。  
- **工程实践**：在真实 RAG 系统中，可使用该方法提升检索阶段的召回与排序效果。

---

**许可证声明：**  
本文档与配套代码仅用于教学与研究，不得用于商业用途。
