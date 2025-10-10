import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 创建评估数据集
evaluation_data = [
    {
        "query": "What is RAG?",
        "retrieved": [
            "Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.",
            "By combining pre-trained generative models with retrieval mechanisms, RAG enables the generation of more relevant and informative responses."
        ],
        "generated_answer": "Retrieval-Augmented Generation (RAG) improves text generation tasks by using external documents for more accurate responses."
    },
    {
        "query": "How does RAG improve text generation?",
        "retrieved": [
            "The model retrieves documents or passages from a knowledge base during the generation process, providing a way to incorporate external information dynamically into the generation process.",
            "RAG helps to solve issues with traditional text generation models by bringing in contextual information at generation time."
        ],
        "generated_answer": "RAG improves text generation by dynamically incorporating relevant external information into the process."
    },
    {
        "query": "What is the goal of RAG?",
        "retrieved": [
            "The primary goal of RAG is to leverage large-scale pre-trained models while simultaneously allowing for the incorporation of up-to-date, task-specific knowledge from external sources.",
            "RAG is designed to generate more accurate and informative responses by incorporating external documents dynamically into the text generation process."
        ],
        "generated_answer": "The goal of RAG is to use external documents for improved and more accurate text generation responses."
    }
]

# 2. 加载 Sentence-BERT 模型
sbert_model = SentenceTransformer('./Qwen/paraphrase')

# 3. 计算 Recall@K
def recall_at_k(retrieved, relevant, k=3):
    """
    计算 Recall@K
    :param retrieved: 检索到的文档片段列表
    :param relevant: 相关文档片段
    :param k: 检索结果中前 K 个片段
    :return: Recall@K 值
    """
    retrieved_top_k = retrieved[:k]
    return len(set(retrieved_top_k) & set(relevant)) / min(k, len(relevant))

# 4. 计算语义相似度（Cosine Similarity）
def semantic_similarity(answer_1, answer_2):
    """
    计算两段文本之间的余弦相似度
    :param answer_1: 第一个答案文本
    :param answer_2: 第二个答案文本
    :return: 余弦相似度值
    """
    embeddings = sbert_model.encode([answer_1, answer_2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return similarity[0][0]

# 5. 评估 Recall@K 和语义相似度
def evaluate_model():
    for data in evaluation_data:
        relevant = [data["generated_answer"]]  # 假设答案是相关文档
        retrieved = data["retrieved"]
        recall_k = recall_at_k(retrieved, relevant, k=2)  # 计算 Recall@2
        print(f"Recall@2 for query '{data['query']}': {recall_k:.2f}")
        
        generated_answer = data["generated_answer"]
        relevant_answer = data["generated_answer"]  # 假设生成答案与真实答案相同
        similarity = semantic_similarity(generated_answer, relevant_answer)
        print(f"Semantic similarity for query '{data['query']}': {similarity:.2f}")
        print("=" * 50)

# 6. 输出优化建议
def output_optimization_suggestions():
    print("Optimization Suggestions:")
    print("1. Increase the number of retrieved documents to ensure more relevant information is captured.")
    print("2. Use advanced retrieval techniques (e.g., BM25 or Dense Retrieval) to improve retrieval precision.")
    print("3. Fine-tune the generative model using task-specific data to improve answer accuracy.")
    print("4. Consider adding re-ranking techniques to reorder retrieved documents for better contextual relevance.")
    print("5. Experiment with different generation parameters (e.g., `max_length`, `temperature`, etc.) to optimize answer generation.")

# 主函数
def main():
    print("Evaluating RAG system:")
    evaluate_model()
    output_optimization_suggestions()

if __name__ == "__main__":
    main()
