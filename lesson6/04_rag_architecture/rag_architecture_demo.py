import os
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 1. 加载 Sentence-BERT 模型 =====
def load_sbert_model():
    model = SentenceTransformer('./Qwen/paraphrase')  # 你可以换成自己的模型
    return model

# ===== 2. 加载 Qwen 模型 =====
def load_qwen_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6b")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6b")
    return tokenizer, model

# ===== 3. 基于 FAISS 检索 =====
def search_with_faiss(index, query_embedding, k=3):
    # 使用 FAISS 进行最近邻检索
    D, I = index.search(query_embedding, k)  # D是距离，I是索引
    return D, I

# ===== 4. 加载 FAISS 索引 =====
def load_faiss_index(index_file="faiss_index.incremental.index"):
    # 从文件加载 FAISS 索引
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        print("FAISS index loaded.")
        return index
    else:
        print(f"FAISS index file {index_file} not found.")
        return None

# ===== 5. 文本生成函数 =====
def generate_answer(query, context, tokenizer, model):
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ===== 6. 主函数 =====
def main():
    # 1) 加载模型
    #sbert_model = load_sbert_model()  # Sentence-BERT 模型
    #print("SBERT model loaded.")
    tokenizer, qwen_model = load_qwen_model()  # Qwen 模型
    print("MODEL Loaded")
    # 2) 加载 FAISS 索引
    index = load_faiss_index("faiss_index.incremental.index")  # 从本地加载 FAISS 索引
    print("FISS index loaded.")
    if not index:
        return

    # 3) 假设我们有一个查询
    query = "What is RAG?"
    query_embedding = sbert_model.encode([query])  # 将查询转换为嵌入

    # 4) 使用 FAISS 检索相似的文档块
    D, I = search_with_faiss(index, np.expand_dims(query_embedding, axis=0), k=3)

    # 5) 输出检索结果（最相似的块）
    print("\nTop 3 most similar chunks:")
    chunks = []  # 用于存储相似文档
    for i in range(len(I[0])):
        print(f"Rank {i+1}:")
        # 假设你有存储在某个地方的文档块
        chunk = f"Dummy chunk {I[0][i]}"  # 这里需要加载实际的文档块
        chunks.append(chunk)
        print(f"Chunk: {chunk}")
        print(f"Distance: {D[0][i]:.4f}")
        print("=" * 50)

    # 6) 将最相似的文档块作为上下文生成答案
    context = " ".join(chunks[:2])  # 选取前2个块作为上下文
    answer = generate_answer(query, context, tokenizer, qwen_model)

    # 7) 输出生成的答案
    print(f"Generated Answer: {answer}")

if __name__ == "__main__":
    main()
