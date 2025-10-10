# 教程：RAG 文档分块与 Sentence-BERT 向量化

## 学习目标
- 掌握文档分块（chunking）的设计原则：窗口大小、重叠度、最小长度。
- 学会调用 Sentence-BERT 生成文本嵌入并保存到文件。
- 理解向量范数、分块长度等质量指标在检索召回中的作用。

## 背景原理
分块策略直接影响检索召回率。给定窗口大小 $w$ 与重叠 $o$，有效步长为 $w-o$。合理的重叠有助于保持上下文连续性。Sentence-BERT 将文本映射到高维语义空间，利用余弦相似度进行检索。

## 代码结构解析
- `PreprocessConfig`：定义分块参数与嵌入模型。
- `chunk_documents`：按字符数滑动窗口切分文本，并保留重叠部分。
- `generate_embeddings`：加载 `SentenceTransformer` 模型生成向量。
- `quality_report`：统计分块数量、平均长度、向量范数等指标。
- `save_embeddings`：将分块文本与向量分别存储，供后续构建向量库。

## 实践步骤
1. 准备原始语料 `corpus.txt`，每段内容占一行。
2. 运行脚本：
   ```bash
   python data_preprocess.py corpus.txt outputs/preprocess --chunk_size 400 --chunk_overlap 80
   ```
3. 检查输出目录中的 `chunks.txt` 与 `embeddings.npy`，并阅读质量报告。
4. 根据报告调整窗口大小与模型选择，优化召回与生成准确性。

## 拓展问题
- 如何处理 PDF/Word 文档？可在预处理前加入 OCR 或段落抽取模块。
- 对于多语言语料，可否使用多语 Sentence-BERT 模型提升召回？
- 如何基于向量范数检测异常样本，防止噪声影响检索质量？
