# -*- coding: utf-8 -*-
"""
教程：RAG 文档分块与 Sentence-BERT 向量化
- 文档分块：按窗口大小与重叠度切分长文本
- 使用 Sentence-BERT 生成句子嵌入并保存为文件
- 评估分块与向量范数对召回质量的影响

Usage:
    python rag_document_chunking.py
"""

import os
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize

# ===== 1. 文档分块参数 =====
WINDOW_SIZE = 100  # 每个分块的最大单词数
OVERLAP_SIZE = 50  # 分块之间的重叠单词数
MIN_CHUNK_LENGTH = 20  # 最小块大小（以单词为单位）

# ===== 2. 加载 Sentence-BERT 模型 =====
def load_model():
    model = SentenceTransformer('./Qwen/paraphrase')  # 选择一个小型的句子模型
    return model

# ===== 3. 文档分块函数（包括重叠） =====
def chunk_document(text: str, window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE, min_length=MIN_CHUNK_LENGTH):
    # 按空格分词
    words = text.split()
    chunks = []
    for i in range(0, len(words), window_size - overlap_size):
        chunk = words[i:i + window_size]
        if len(chunk) >= min_length:
            chunks.append(" ".join(chunk))
    return chunks

# ===== 4. 计算嵌入向量并保存 =====
def compute_and_save_embeddings(chunks, model):
    # 使用 Sentence-BERT 计算每个分块的嵌入向量
    embeddings = model.encode(chunks)
    
    # 保存为 Numpy 数组
    embeddings_file = "outputs/document_embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"Embeddings saved to {embeddings_file}")
    
    # 向量范数计算（L2 norm）
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Vector norms (L2 norm) calculated for {len(norms)} chunks.")
    

    for i in range(len(norms)):  # 使用 norms 的长度来遍历
        print(f"Chunk {i+1} L2 norm: {norms[i]:.3f}")
    
    # 返回嵌入和范数
    return embeddings, norms



# ===== 5. 示例文档 =====
def example_document():
    text = """
    Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.
    By combining pre-trained generative models with retrieval mechanisms, RAG enables the generation of more relevant and informative responses.
    The model retrieves documents or passages from a knowledge base during the generation process, providing a way to incorporate external information
    dynamically into the generation process. The goal of RAG is to leverage large-scale pre-trained models while simultaneously allowing for
    the incorporation of up-to-date, task-specific knowledge from external sources.
    """
    return text

# ===== 6. 主函数 =====
def main():
    # 1) 加载模型
    model = load_model()
    
    # 2) 示例文档分块
    doc = example_document()
    chunks = chunk_document(doc)
    
    # 打印分块信息
    print(f"Document has been chunked into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:5]):  # 显示前5个分块
        print(f"Chunk {i+1}: {chunk[:60]}...")  # 截取部分显示前60字符
    
    # 3) 计算文本嵌入并保存
    embeddings, norms = compute_and_save_embeddings(chunks, model)
    
    # 4) 打印一些质量指标
    avg_norm = np.mean(norms)
    print(f"Average L2 norm of embeddings: {avg_norm:.3f}")

    # 可选：返回前5个分块及其对应的嵌入向量
    for i in range(2):
        print(f"Chunk {i+1} L2 norm: {norms[i]:.3f}")

if __name__ == "__main__":
    main()
