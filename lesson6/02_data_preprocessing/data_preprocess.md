
# RAG 文档分块与 Sentence-BERT 向量化教程

本教程展示了如何进行 **RAG (Retrieval-Augmented Generation)** 模型中的文档分块操作，并使用 **Sentence-BERT** 生成文档分块的句子嵌入向量。此流程帮助学生理解如何将长文本切分为较小的块并为每个块计算嵌入，从而评估文档分块和向量范数对召回质量的影响。

---

## 目录
- [RAG 文档分块与 Sentence-BERT 向量化教程](#rag-文档分块与-sentence-bert-向量化教程)
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
    - [文档分块](#文档分块)
    - [Sentence-BERT 嵌入计算](#sentence-bert-嵌入计算)
  - [输出与结果分析](#输出与结果分析)
    - [示例输出：](#示例输出)
  - [常见问题（FAQ）](#常见问题faq)
  - [扩展建议](#扩展建议)
  - [许可证](#许可证)

---

## 功能概览

该脚本实现了以下功能：

- **文档分块**：根据窗口大小和重叠度将长文本切分为较小的文本块。
- **Sentence-BERT 向量化**：为每个分块生成句子的嵌入向量，并保存为文件。
- **向量范数计算**：计算每个分块嵌入的 L2 范数（向量的长度），用于评估文本的多样性。

---

## 环境与依赖

### 所需库：
- Python ≥ 3.7
- `torch`
- `sentence-transformers`
- `matplotlib`
- `numpy`
- `pandas`
- `sklearn`

### 安装依赖：
```bash
pip install torch sentence-transformers matplotlib numpy pandas scikit-learn
```

---

## 快速开始

### 步骤 1：安装所需的库
```bash
pip install torch sentence-transformers matplotlib numpy pandas scikit-learn
```

### 步骤 2：运行脚本
```bash
python lesson6/02_data_preprocessing/data_preprocess.py
```

### 步骤 3：查看输出结果
脚本会自动分块文档，并为每个分块计算 **Sentence-BERT** 嵌入向量。训练完成后，嵌入向量会被保存为 `.npy` 文件，并打印每个分块的 **L2 范数**，以评估其质量。

---

## 命令行参数

本脚本没有命令行参数，所有的配置项都在代码中定义。你可以修改以下参数：

| 参数               | 默认值                | 说明 |
|--------------------|-----------------------|------|
| `WINDOW_SIZE`      | `100`                 | 每个分块的最大单词数 |
| `OVERLAP_SIZE`     | `50`                  | 分块之间的重叠单词数 |
| `MIN_CHUNK_LENGTH` | `20`                  | 最小块大小（以单词为单位） |
| `MODEL_PATH`       | `"./Qwen/paraphrase"` | Sentence-BERT 模型的路径 |
| `OUTPUT_FILE`      | `"outputs/document_embeddings.npy"` | 嵌入向量保存路径 |

---

## 代码结构与功能说明

### 文档分块

- **`chunk_document(text: str, window_size=100, overlap_size=50, min_length=20)`**  
  该函数将文档按指定的 **窗口大小** 和 **重叠大小** 切分为多个文本块。每个文本块的长度不得小于 **min_length**，并通过空格对文本进行分词。

### Sentence-BERT 嵌入计算

- **`load_model()`**  
  加载 **Sentence-BERT** 模型，该模型用于生成文本块的句子嵌入。

- **`compute_and_save_embeddings(chunks, model)`**  
  使用 **Sentence-BERT** 为每个文档块计算句子嵌入向量，并将嵌入向量保存为 `.npy` 文件。函数还计算每个嵌入的 L2 范数，并打印出每个分块的范数。

---

## 输出与结果分析

脚本执行后，生成的嵌入向量将保存为一个 **`.npy`** 文件，包含每个文档块的 **Sentence-BERT** 向量。同时，脚本还会计算并打印每个分块的 **L2 范数**，用于评估向量的质量。

### 示例输出：
```text
Embeddings saved to outputs/document_embeddings.npy
Vector norms (L2 norm) calculated for 100 chunks.
Chunk 1 L2 norm: 3.235
Chunk 2 L2 norm: 2.981
...
Average L2 norm of embeddings: 3.120
```

---

## 常见问题（FAQ）

1. **问题：如何调整分块的窗口大小和重叠度？**
   - **解决方案**：可以通过修改 `WINDOW_SIZE` 和 `OVERLAP_SIZE` 来调整分块的窗口大小和重叠度。

2. **问题：如何更改使用的 Sentence-BERT 模型？**
   - **解决方案**：可以通过修改 `load_model()` 中的路径来指定不同的 Sentence-BERT 模型。

3. **问题：如何改变向量范数计算的方式？**
   - **解决方案**：如果需要计算其他类型的范数（如 L1 范数），可以修改 `compute_and_save_embeddings()` 函数中的范数计算方式。

---

## 扩展建议

- **多样性评估**：可以通过 **distinct-n** 或 **self-BLEU** 等指标来进一步评估分块后的文档的多样性。
- **文档生成**：可以结合 **RAG** 模型进行文档生成，并利用这些分块的数据进行训练。
- **优化嵌入计算**：通过批量计算句子嵌入，优化计算速度和内存占用。

---

## 许可证

此脚本仅用于教学和研究目的，使用时请遵守所用数据集和模型的许可协议。
