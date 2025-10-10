import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== 1. 初始化模型 =====
# 加载 Sentence-BERT 模型用于检索
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 加载生成模型（例如 Qwen3 或其他）
model_name = "Qwen/Qwen3-0.6b"  # 使用一个示例模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ===== 2. 模拟的知识库 =====
knowledge_base = {
    "What is RAG?": "Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.",
    "How does RAG improve text generation?": "RAG combines pre-trained generative models with retrieval mechanisms to improve text generation.",
    "What is the goal of RAG?": "The goal of RAG is to leverage pre-trained models while allowing for the incorporation of task-specific knowledge from external sources."
}

# ===== 3. 文本预处理函数 =====
def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

# ===== 4. 检索阶段 =====
def retrieve_answer(query, knowledge_base):
    # 将知识库问题编码为向量
    corpus_embeddings = sbert_model.encode(list(knowledge_base.keys()))
    
    # 对用户查询进行编码
    query_embedding = sbert_model.encode([query])
    
    # 计算余弦相似度
    cosine_similarities = cosine_similarity(query_embedding, corpus_embeddings)
    
    # 找到相似度最高的文档
    best_match_index = cosine_similarities.argmax()
    best_match_question = list(knowledge_base.keys())[best_match_index]
    best_match_answer = knowledge_base[best_match_question]
    
    return best_match_answer, cosine_similarities[0][best_match_index]

# ===== 5. 生成阶段 =====
def generate_answer(query, context):
    inputs = tokenizer(query + " " + context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ===== 6. 性能压测：模拟多轮问答 =====
def performance_test(queries, knowledge_base, threshold=0.7):
    retrieval_times = []
    generation_times = []
    total_times = []

    for query in queries:
        # 1. 检索阶段
        start_time = time.time()
        retrieved_answer, similarity_score = retrieve_answer(query, knowledge_base)
        retrieval_time = time.time() - start_time
        
        # 2. 生成阶段
        start_time = time.time()
        generated_answer = generate_answer(query, retrieved_answer)
        generation_time = time.time() - start_time

        # 3. 记录每个阶段的时间
        retrieval_times.append(retrieval_time)
        generation_times.append(generation_time)
        total_times.append(retrieval_time + generation_time)
    
    return retrieval_times, generation_times, total_times

# ===== 7. 计算统计指标 =====
def calculate_metrics(times):
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    max_time = np.max(times)
    
    return avg_time, p95_time, max_time

# ===== 8. 绘制性能分布图 =====
def plot_performance(times, title="Latency Distribution"):
    plt.hist(times, bins=30, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.show()

# ===== 9. 执行性能压测 =====
if __name__ == "__main__":
    # 模拟查询数据集
    queries = [
        "What is RAG?",
        "How does RAG improve text generation?",
        "What is the goal of RAG?",
        "Explain the RAG system in detail.",
        "How can I optimize RAG for performance?"
    ]
    
    # 性能压测
    retrieval_times, generation_times, total_times = performance_test(queries, knowledge_base)

    # 计算统计指标
    avg_retrieval_time, p95_retrieval_time, max_retrieval_time = calculate_metrics(retrieval_times)
    avg_generation_time, p95_generation_time, max_generation_time = calculate_metrics(generation_times)
    avg_total_time, p95_total_time, max_total_time = calculate_metrics(total_times)

    # 输出统计指标
    print(f"Retrieval Latency - Avg: {avg_retrieval_time:.4f}, P95: {p95_retrieval_time:.4f}, Max: {max_retrieval_time:.4f}")
    print(f"Generation Latency - Avg: {avg_generation_time:.4f}, P95: {p95_generation_time:.4f}, Max: {max_generation_time:.4f}")
    print(f"Total Latency - Avg: {avg_total_time:.4f}, P95: {p95_total_time:.4f}, Max: {max_total_time:.4f}")
    
    # 绘制性能分布图
    plot_performance(total_times, "Total Latency Distribution")
