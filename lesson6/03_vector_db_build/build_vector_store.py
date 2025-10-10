import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ===== 1. 加载 Sentence-BERT 模型 =====
def load_model():
    model = SentenceTransformer('./Qwen/paraphrase')  # 选择一个小型的句子模型
    return model

# ===== 2. 文档分块函数（包括重叠） =====
def chunk_document(text: str, window_size=100, overlap_size=50, min_length=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), window_size - overlap_size):
        chunk = words[i:i + window_size]
        if len(chunk) >= min_length:
            chunks.append(" ".join(chunk))
    return chunks

# ===== 3. 使用 FAISS 存储向量 =====
def build_faiss_index(embeddings: np.ndarray, dim: int):
    # 创建 FAISS 索引
    index = faiss.IndexFlatL2(dim)  # L2 距离（欧几里得距离）
    
    # 将嵌入向量添加到索引中
    index.add(embeddings)
    return index

# ===== 4. 基于 FAISS 的相似度检索 =====
def search_with_faiss(index, query_embedding, k=5):
    # 使用 FAISS 进行最近邻检索
    D, I = index.search(query_embedding, k)  # D是距离，I是索引
    return D, I

# ===== 5. 示例文档（多条） =====
def example_document():
    # 添加十多条示例文档
    text_list = [
        "Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.",
        "By combining pre-trained generative models with retrieval mechanisms, RAG enables the generation of more relevant and informative responses.",
        "The model retrieves documents or passages from a knowledge base during the generation process, providing a way to incorporate external information dynamically into the generation process.",
        "RAG helps to solve issues with traditional text generation models by bringing in contextual information at generation time.",
        "The primary goal of RAG is to leverage large-scale pre-trained models while simultaneously allowing for the incorporation of up-to-date, task-specific knowledge from external sources.",
        "A key feature of RAG is its ability to retrieve and generate content in real-time, making it suitable for applications requiring fast adaptation to new data.",
        "To improve the relevance of generated content, RAG models fetch data from an external corpus before performing the generation task.",
        "RAG can be used in dialogue systems, recommendation systems, and knowledge extraction tasks, among others.",
        "The success of RAG depends on the quality of the retrieval process and how effectively the external information is integrated into the model.",
        "One challenge with RAG is ensuring that the retrieval mechanism is fast and accurate enough for real-time applications.",
        "RAG combines the strengths of retrieval-based and generation-based methods, making it versatile for a wide range of NLP tasks.",
        "The architecture of RAG includes both a retriever component that fetches relevant documents and a generator component that uses these documents to produce responses.",
        "RAG has been shown to outperform traditional generative models on certain benchmark tasks by incorporating more domain-specific knowledge."
    ]
    return text_list

# ===== 6. 主函数 =====
def main():
    # 1) 加载 Sentence-BERT 模型
    model = load_model()

    # 2) 示例文档分块
    texts = example_document()
    all_chunks = []
    for doc in texts:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    # 打印分块信息
    print(f"Document has been chunked into {len(all_chunks)} chunks.")
    for i, chunk in enumerate(all_chunks[:5]):  # 显示前5个分块
        print(f"Chunk {i+1}: {chunk[:60]}...")  # 截取部分显示前60字符

    # 3) 计算文本嵌入
    embeddings = model.encode(all_chunks)

    # 将嵌入向量保存为 numpy 数组
    embeddings = np.array(embeddings)

    # ===== 4) 使用 FAISS 创建索引 =====
    index = build_faiss_index(embeddings, dim=embeddings.shape[1])

    # 5) 假设我们有一个查询
    query = "What is RAG?"
    query_embedding = model.encode([query])  # 将查询转换为向量

    # ===== 6) 使用 FAISS 检索相似块 =====
    D, I = search_with_faiss(index, query_embedding, k=3)

    # 输出检索结果
    print("\nTop 3 most similar chunks:")
    for i in range(len(I[0])):
        print(f"Rank {i+1}:")
        print(f"Chunk: {all_chunks[I[0][i]]}")
        print(f"Distance: {D[0][i]:.4f}")
        print("=" * 50)

if __name__ == "__main__":
    main()
