from sentence_transformers import SentenceTransformer

model_name = 'paraphrase-MiniLM-L6-v2'  # 你想下载的模型名称
model = SentenceTransformer(model_name)

# 将模型保存到本地指定目录
model.save('./Qwen/paraphrase')