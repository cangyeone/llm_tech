import numpy as np
import string
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ===== 1. 初始化模型 =====
# 加载 Sentence-BERT 模型用于检索
sbert_model = SentenceTransformer('./Qwen/paraphrase')

# 假设我们有以下简单的知识库
knowledge_base = {
    "What is RAG?": "Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.",
    "How does RAG improve text generation?": "RAG combines pre-trained generative models with retrieval mechanisms to improve text generation.",
    "What is the goal of RAG?": "The goal of RAG is to leverage pre-trained models while allowing for the incorporation of task-specific knowledge from external sources."
}

# 加载生成模型（例如 Qwen3 或其他）
model_name = "Qwen/Qwen3-0.6b"  # 使用一个示例模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ===== 2. 文本预处理函数 =====
def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

# ===== 3. 检索阶段 =====
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

# ===== 4. 生成阶段 =====
def generate_answer(query, context):
    inputs = tokenizer(query + " " + context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ===== 5. 混合推理引擎：动态切换策略 =====
def hybrid_inference(query, knowledge_base, threshold=0.7):
    # 1. 检索答案
    retrieved_answer, similarity_score = retrieve_answer(query, knowledge_base)
    
    print(f"Retrieved Answer: {retrieved_answer}")
    print(f"Cosine Similarity Score: {similarity_score:.2f}")
    
    # 2. 根据检索得分决定是否使用生成模型
    if similarity_score >= threshold:
        print("High retrieval score. Using retrieved answer directly.")
        return retrieved_answer  # 直接返回检索结果
    else:
        print("Low retrieval score. Using generative model.")
        # 使用生成模型生成回答（假设我们将检索答案作为上下文）
        context = retrieved_answer
        generated_answer = generate_answer(query, context)
        return generated_answer

# ===== 6. 测试多轮对话 =====
def run_conversation():
    print("Customer Service Chatbot (Type 'exit' to end)\n")
    while True:
        query = input("User: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Ending conversation.")
            break
        # 执行混合推理引擎
        response = hybrid_inference(query, knowledge_base)
        print(f"Agent: {response}\n")

# 启动对话
run_conversation()
