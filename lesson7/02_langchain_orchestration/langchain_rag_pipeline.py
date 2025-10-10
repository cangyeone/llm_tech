import numpy as np
import string
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import logging
import json
import time

# ===== 1. 初始化 Logger 用于审计 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 2. 创建示例语料库 =====
corpus = [
    "Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.",
    "By combining pre-trained generative models with retrieval mechanisms, RAG enables the generation of more relevant and informative responses.",
    "The model retrieves documents or passages from a knowledge base during the generation process, providing a way to incorporate external information dynamically into the generation process.",
    "RAG helps to solve issues with traditional text generation models by bringing in contextual information at generation time.",
    "The primary goal of RAG is to leverage large-scale pre-trained models while simultaneously allowing for the incorporation of up-to-date, task-specific knowledge from external sources."
]

# ===== 3. 文本预处理函数 =====
def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

# ===== 4. StageContext 用于保存状态 =====
import json
import numpy as np
import logging

# 初始化 Logger 用于审计
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# StageContext 类用于保存每个阶段的状态
class StageContext:
    def __init__(self):
        self.state = {}

    def save_state(self, key, value):
        self.state[key] = value

    def get_state(self, key):
        return self.state.get(key)

    def log_state(self):
        # 自定义序列化函数，将 ndarray 转换为列表
        def json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # 将 ndarray 转为普通列表
            raise TypeError(f"Type {obj.__class__.__name__} not serializable")

        # 使用自定义序列化函数
        try:
            logger.info(f"Current State: {json.dumps(self.state, default=json_serializable, indent=2)}")
        except Exception as e:
            logger.error(f"Error during logging state: {e}")



# ===== 5. 任务处理阶段：检索 =====
def retrieval_stage(query, corpus, context):
    logger.info("Starting retrieval stage...")
    # BM25 检索
    corpus_tokenized = [preprocess(doc) for doc in corpus]
    bm25 = BM25Okapi(corpus_tokenized)
    query_tokenized = preprocess(query)
    bm25_scores = bm25.get_scores(query_tokenized)

    # 存储检索结果到 StageContext
    context.save_state('retrieval_scores', bm25_scores)
    context.save_state('retrieved_documents', [corpus[i] for i in np.argsort(bm25_scores)[::-1]])

    # 返回检索结果
    return bm25_scores, [corpus[i] for i in np.argsort(bm25_scores)[::-1]]

# ===== 6. 任务处理阶段：重排序 =====
def reranking_stage(query, retrieved_docs, context):
    logger.info("Starting reranking stage...")
    # 使用 Sentence-BERT 计算相似度进行重排序
    sbert_model = SentenceTransformer('./Qwen/paraphrase')
    query_embedding = sbert_model.encode([query])
    retrieved_embeddings = sbert_model.encode(retrieved_docs)

    # 计算余弦相似度
    cosine_similarities = cosine_similarity(query_embedding, retrieved_embeddings)
    context.save_state('reranked_scores', cosine_similarities.flatten())

    # 重排序文档
    reranked_docs = [retrieved_docs[i] for i in np.argsort(cosine_similarities.flatten())[::-1]]
    return reranked_docs, cosine_similarities.flatten()

# ===== 7. 任务处理阶段：生成答案 =====
def generation_stage(query, reranked_docs, context):
    logger.info("Starting generation stage...")
    # 这里简单模拟生成答案的过程
    context_text = " ".join(reranked_docs[:3])  # 选取前3个文档作为上下文
    answer = f"Generated answer based on context: {context_text[:100]}..."  # 简单生成的答案
    context.save_state('generated_answer', answer)
    return answer

# ===== 8. 任务处理阶段：审计 =====
def audit_stage(context):
    logger.info("Starting audit stage...")
    # 审计各阶段的状态并记录日志
    context.log_state()
    return "Audit completed."

# ===== 9. 流程串联 =====
def main():
    # 1. 创建 StageContext
    context = StageContext()

    # 2. 用户查询
    query = "What is RAG?"

    # 3. 检索阶段
    bm25_scores, retrieved_docs = retrieval_stage(query, corpus, context)

    # 4. 重排序阶段
    reranked_docs, reranked_scores = reranking_stage(query, retrieved_docs, context)

    # 5. 生成答案阶段
    generated_answer = generation_stage(query, reranked_docs, context)

    # 6. 审计阶段
    audit_result = audit_stage(context)

    # 输出最终答案
    logger.info(f"Final Answer: {generated_answer}")
    logger.info(f"Audit Result: {audit_result}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
