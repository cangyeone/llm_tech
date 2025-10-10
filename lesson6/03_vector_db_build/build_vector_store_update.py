import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ===== 1. 加载 Sentence-BERT 模型 =====
def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 选择一个小型的句子模型
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
def build_faiss_index(dim: int):
    # 创建 FAISS 索引
    index = faiss.IndexFlatL2(dim)  # L2 距离（欧几里得距离）
    return index

# ===== 4. 基于 FAISS 的相似度检索 =====
def search_with_faiss(index, query_embedding, k=5):
    # 使用 FAISS 进行最近邻检索
    D, I = index.search(query_embedding, k)  # D是距离，I是索引
    return D, I

# ===== 5. 保存和加载 FAISS 索引 =====
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)  # 将索引保存到文件

def load_faiss_index(file_path):
    return faiss.read_index(file_path)  # 从文件加载索引

# ===== 6. 示例文档（多条） =====
def example_document():
    # 添加一些示例文档
    text_list = [
        "Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.",
        "By combining pre-trained generative models with retrieval mechanisms, RAG enables the generation of more relevant and informative responses.",
        "The model retrieves documents or passages from a knowledge base during the generation process, providing a way to incorporate external information dynamically into the generation process.",
        "RAG helps to solve issues with traditional text generation models by bringing in contextual information at generation time.",
        "The primary goal of RAG is to leverage large-scale pre-trained models while simultaneously allowing for the incorporation of up-to-date, task-specific knowledge from external sources."
    ]
    return text_list

# ===== 7. 主函数 =====
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
    index = build_faiss_index(dim=embeddings.shape[1])

    # 逐条增加嵌入并保存索引
    for i in range(len(embeddings)):
        # 增量添加每个嵌入向量
        index.add(np.expand_dims(embeddings[i], axis=0))  # 逐条添加
        print(f"Added chunk {i+1} to the index.")

        # 每次添加后保存索引
        save_faiss_index(index, 'faiss_index.incremental.index')
        print(f"FAISS index saved after adding chunk {i+1}.")

    # 5) 假设我们有两个查询
    queries = ["What is RAG?", "How does RAG improve text generation?"]
    query_embeddings = model.encode(queries)  # 将查询转换为向量

    # ===== 6) 使用 FAISS 检索相似块 =====
    for i, query_embedding in enumerate(query_embeddings):
        print(f"\nQuery {i+1}: {queries[i]}")
        D, I = search_with_faiss(index, np.expand_dims(query_embedding, axis=0), k=3)

        # 输出检索结果
        print(f"Top 3 most similar chunks to query {i+1}:")
        for j in range(len(I[0])):
            print(f"Rank {j+1}:")
            print(f"Chunk: {all_chunks[I[0][j]]}")
            print(f"Distance: {D[0][j]:.4f}")
            print("=" * 50)

if __name__ == "__main__":
    main()
