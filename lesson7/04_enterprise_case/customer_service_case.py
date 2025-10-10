from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 客服知识库
knowledge_base = {
    "What is the return policy?": "Our return policy allows customers to return items within 30 days of purchase.",
    "How can I track my order?": "You can track your order by visiting our order tracking page and entering your order ID.",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and bank transfers.",
    "How do I reset my password?": "To reset your password, click on the 'Forgot Password' link on the login page."
}

# 加载 Qwen3 模型
model = SentenceTransformer('Qwen/Qwen3-0.6b')

# 用户查询
def user_query(query):
    return query.lower()

# 检索匹配答案
def retrieve_answer(query, knowledge_base):
    # 对所有知识库中的问题和答案进行编码
    knowledge_queries = list(knowledge_base.keys())
    knowledge_answers = list(knowledge_base.values())

    # 计算用户查询与知识库中问题的语义相似度
    query_embedding = model.encode([query])
    knowledge_embeddings = model.encode(knowledge_queries)

    # 计算余弦相似度
    cosine_similarities = cosine_similarity(query_embedding, knowledge_embeddings)
    
    # 返回相似度最高的答案
    best_match_index = cosine_similarities.argmax()
    return knowledge_answers[best_match_index], knowledge_queries[best_match_index]

# 模拟对话
def chat_with_customer():
    conversation_history = []
    while True:
        query = input("User: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Customer service session ended.")
            break
        answer, matched_question = retrieve_answer(query, knowledge_base)
        conversation_history.append({
            'user_query': query,
            'matched_question': matched_question,
            'response': answer
        })
        print(f"Agent: {answer}")

    return conversation_history

# 启动客服对话
conversation_history = chat_with_customer()

# 计算一次解决率和转人工率
def calculate_metrics(conversation_history, threshold=0.8):
    total_queries = len(conversation_history)
    solved_queries = 0

    for conversation in conversation_history:
        query = conversation['user_query']
        response = conversation['response']
        
        # 如果返回的答案匹配知识库，并且相似度较高，认为是一次解决
        matched_question = conversation['matched_question']
        if matched_question and response:
            solved_queries += 1  # 假设匹配的回答是有效的

    once_solved_rate = solved_queries / total_queries if total_queries > 0 else 0
    return once_solved_rate

# 计算并输出一次解决率
once_solved_rate = calculate_metrics(conversation_history)
print(f"Once Solved Rate: {once_solved_rate:.2f}")

# 输出优化建议
def output_optimization_suggestions(once_solved_rate):
    print("\nOptimization Suggestions:")
    if once_solved_rate < 0.8:
        print("1. Increase the knowledge base with more frequently asked questions.")
        print("2. Improve the model's fine-tuning with more domain-specific data.")
        print("3. Implement multi-turn conversation handling to better capture user intent.")
    else:
        print("The system is performing well. Continue monitoring and improve as needed.")

# 输出优化建议
output_optimization_suggestions(once_solved_rate)
