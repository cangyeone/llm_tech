import numpy as np
import string
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ===== 1. 准备语料库 =====
corpus = [
    "Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.",
    "By combining pre-trained generative models with retrieval mechanisms, RAG enables the generation of more relevant and informative responses.",
    "The model retrieves documents or passages from a knowledge base during the generation process, providing a way to incorporate external information dynamically into the generation process.",
    "RAG helps to solve issues with traditional text generation models by bringing in contextual information at generation time.",
    "The primary goal of RAG is to leverage large-scale pre-trained models while simultaneously allowing for the incorporation of up-to-date, task-specific knowledge from external sources."
]

# ===== 2. 文本预处理 =====
def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

corpus_tokenized = [preprocess(doc) for doc in corpus]

# ===== 3. 初始化 BM25 检索 =====
bm25 = BM25Okapi(corpus_tokenized)

# ===== 4. 加载 Sentence-BERT 模型 =====
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ===== 5. 输入查询 =====
query = "What is RAG?"

# 计算 BM25 得分
query_tokenized = preprocess(query)
bm25_scores = bm25.get_scores(query_tokenized)

# 计算 Sentence-BERT 向量
corpus_embeddings = sbert_model.encode(corpus)
query_embedding = sbert_model.encode([query])

# 计算向量检索的余弦相似度
cosine_similarities = cosine_similarity(query_embedding, corpus_embeddings)

# ===== 6. 得分归一化 =====
scaler = MinMaxScaler()

# 归一化 BM25 和向量检索得分
bm25_scores_normalized = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
cosine_similarities_normalized = scaler.fit_transform(cosine_similarities[0].reshape(-1, 1)).flatten()

# ===== 7. 权重融合 =====
bm25_weight = 0.5
vector_weight = 0.5

# 计算加权融合得分
final_scores = bm25_weight * bm25_scores_normalized + vector_weight * cosine_similarities_normalized

# ===== 8. 重排序 =====
sorted_indices = np.argsort(final_scores)[::-1]  # 从大到小排序

# 显示排序后的文档
print("Sorted documents by hybrid scores:")
for i in sorted_indices:
    print(f"Rank {i+1}: {corpus[i]}")
    print(f"BM25 score: {bm25_scores[i]:.4f}, Cosine similarity: {cosine_similarities[0][i]:.4f}, Final score: {final_scores[i]:.4f}")
    print("=" * 50)

# ===== 9. 评估 Recall@K 和 Precision@K =====
def recall_at_k(retrieved, relevant, k=3):
    retrieved_top_k = retrieved[:k]
    return len(set(retrieved_top_k) & set(relevant)) / min(k, len(relevant))

def precision_at_k(retrieved, relevant, k=3):
    retrieved_top_k = retrieved[:k]
    return len(set(retrieved_top_k) & set(relevant)) / k

# 示例评估
relevant_docs = [
    "Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.",
    "By combining pre-trained generative models with retrieval mechanisms, RAG enables the generation of more relevant and informative responses."
]

recall_k = recall_at_k(corpus, relevant_docs, k=3)
precision_k = precision_at_k(corpus, relevant_docs, k=3)

print(f"Recall@3: {recall_k:.2f}")
print(f"Precision@3: {precision_k:.2f}")

# ===== 10. 输出优化建议 =====
print("\nOptimization Suggestions:")
print("1. Increase the number of retrieved documents to ensure more relevant information is captured.")
print("2. Use advanced retrieval techniques (e.g., BM25 or Dense Retrieval) to improve retrieval precision.")
print("3. Fine-tune the generative model using task-specific data to improve answer accuracy.")
print("4. Consider adding re-ranking techniques to reorder retrieved documents for better contextual relevance.")
print("5. Experiment with different generation parameters (e.g., `max_length`, `temperature`, etc.) to optimize answer generation.")
