
# RAG 文档检索与生成教程

本教程展示了如何使用 **RAG (Retrieval-Augmented Generation)** 模型中的文档检索和文本生成。通过 **FAISS** 对文档进行检索，并利用 **Qwen 模型** 生成基于检索结果的答案。以下是详细的函数说明与使用教程。

---

## 目录
- [RAG 文档检索与生成教程](#rag-文档检索与生成教程)
  - [目录](#目录)
  - [功能概览](#功能概览)
  - [环境与依赖](#环境与依赖)
    - [所需库：](#所需库)
    - [安装依赖：](#安装依赖)
  - [快速开始](#快速开始)
    - [步骤 1：安装所需的库](#步骤-1安装所需的库)
    - [步骤 2：运行脚本](#步骤-2运行脚本)
    - [步骤 3：查看输出结果](#步骤-3查看输出结果)
  - [命令行参数](#命令行参数)
  - [代码结构与功能说明](#代码结构与功能说明)
    - [加载模型](#加载模型)
    - [FAISS 相似度检索](#faiss-相似度检索)
    - [文本生成](#文本生成)
  - [输出与结果分析](#输出与结果分析)
    - [示例输出：](#示例输出)
  - [常见问题（FAQ）](#常见问题faq)
  - [扩展建议](#扩展建议)
  - [许可证](#许可证)

---

## 功能概览

本脚本实现了以下功能：

- **加载 Sentence-BERT 模型**：用于将查询转换为句子嵌入。
- **加载 Qwen 模型**：用于文本生成。
- **FAISS 相似度检索**：基于 **FAISS** 索引进行文档检索，找到最相似的文档块。
- **文本生成**：根据检索到的文档块生成与查询相关的回答。

---

## 环境与依赖

### 所需库：
- Python ≥ 3.7
- `torch`
- `faiss`
- `sentence-transformers`
- `transformers`
- `numpy`

### 安装依赖：
```bash
pip install torch faiss sentence-transformers transformers numpy
```

---

## 快速开始

### 步骤 1：安装所需的库
```bash
pip install torch faiss sentence-transformers transformers numpy
```

### 步骤 2：运行脚本
```bash
python lesson6/04_rag_architecture/rag_architecture_demo.py
```

### 步骤 3：查看输出结果
脚本会自动加载模型，加载 FAISS 索引，执行检索操作，并基于检索到的文档块生成答案。最终的答案将被打印出来。

---

## 命令行参数

本脚本没有命令行参数，所有配置项都在代码中定义。以下是关键配置项：

| 参数               | 默认值                | 说明 |
|--------------------|-----------------------|------|
| `MODEL_PATH`       | `"./Qwen/paraphrase"`  | Sentence-BERT 模型路径 |
| `FAISS_INDEX_FILE` | `"faiss_index.incremental.index"` | FAISS 索引文件路径 |

---

## 代码结构与功能说明

### 加载模型

- **`load_sbert_model()`**  
  加载 **Sentence-BERT** 模型，该模型用于生成查询和文档块的嵌入向量。

- **`load_qwen_model()`**  
  加载 **Qwen** 生成模型和其对应的 **tokenizer**，用于文本生成。

### FAISS 相似度检索

- **`search_with_faiss(index, query_embedding, k=3)`**  
  使用 **FAISS** 对查询嵌入进行相似度检索，返回最相似的 **k** 个文档块及其相似度。

- **`load_faiss_index(index_file="faiss_index.incremental.index")`**  
  从指定路径加载 **FAISS** 索引。

### 文本生成

- **`generate_answer(query, context, tokenizer, model)`**  
  使用 **Qwen** 模型和其对应的 **tokenizer** 生成与查询相关的回答。通过将文档块拼接成上下文来增强生成的相关性。

---

## 输出与结果分析

脚本执行后，会输出以下内容：

- **FAISS 检索的最相似文档块**：显示与查询最相似的 **k** 个文档块及其相似度。
- **生成的答案**：基于检索到的文档块生成的回答。

### 示例输出：
```text
Top 3 most similar chunks:
Rank 1:
Chunk: Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.
Distance: 0.1234
==================================================
Rank 2:
Chunk: RAG combines the strengths of retrieval-based and generation-based methods, making it versatile for a wide range of NLP tasks.
Distance: 0.1567
==================================================
Rank 3:
Chunk: The primary goal of RAG is to leverage large-scale pre-trained models while simultaneously allowing for the incorporation of up-to-date, task-specific knowledge from external sources.
Distance: 0.1789
==================================================
Generated Answer: RAG (Retrieval-Augmented Generation) 是一种模型架构，旨在提高文本生成任务的性能。通过结合预训练的生成模型和检索机制，RAG 使得生成的回答更加相关和信息丰富。
```

---

## 常见问题（FAQ）

1. **问题：如何调整检索的相似度数量？**
   - **解决方案**：可以通过修改 `k` 参数来调整每次检索返回的最相似文档块数量。

2. **问题：如何选择不同的 Sentence-BERT 模型？**
   - **解决方案**：可以修改 `load_sbert_model()` 函数中的模型路径，加载不同的 **Sentence-BERT** 模型。

3. **问题：如何更改 Qwen 模型的配置？**
   - **解决方案**：可以修改 `load_qwen_model()` 函数中的模型路径，使用不同的 **Qwen** 模型。

---

## 扩展建议

- **多样性评估**：可以通过增加 **distinct-n** 或 **self-BLEU** 等指标，进一步评估文档的多样性。
- **自定义检索算法**：可以扩展脚本，使用其他检索算法如 **BM25** 来替代 **FAISS**。
- **生成模型优化**：可以结合 **RLHF** 或 **DPO** 等方法对生成模型进行微调，提升生成质量。

---

## 许可证

此脚本仅用于教学和研究目的，使用时请遵守所用数据集和模型的许可协议。
