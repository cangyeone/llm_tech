
# 教学实验：基于 Sentence-BERT + Qwen 生成模型的混合问答系统（Hybrid RAG Engine）

本教学代码演示了一个最小化的 **RAG (Retrieval-Augmented Generation)** 混合问答系统实现，结合了两类推理机制：

1. **检索模型（Retriever）** —— 使用 Sentence-BERT 计算语义相似度，从知识库中找到最相关的答案；  
2. **生成模型（Generator）** —— 当检索结果相似度较低时，调用生成式语言模型（如 Qwen3）动态生成回答。  

该示例帮助学习者理解 RAG 架构中 **检索 + 生成动态协同机制** 的原理，并通过简单代码展示完整流程。

---

## 📘 功能概览

- 使用 **Sentence-BERT** 提取文本语义向量；  
- 使用 **余弦相似度 (cosine similarity)** 计算用户问题与知识库问句的相似度；  
- 当检索相似度高于阈值时直接返回知识库答案；  
- 否则调用 **Qwen3 生成模型** 根据检索上下文生成答案；  
- 支持多轮交互式会话；  
- 可扩展为混合 RAG 引擎的教学示例（Dense + Generative pipeline）。

---

## 🧩 环境依赖

运行前请确保安装以下依赖：

```bash
pip install sentence-transformers transformers scikit-learn numpy
```

### 说明
- **SentenceTransformer**：用于句向量编码；
- **Transformers (HuggingFace)**：用于加载生成式模型（如 Qwen）；
- **scikit-learn**：用于余弦相似度计算与归一化。

> 💡 若无 GPU，代码可在 CPU 上运行，但生成速度较慢。

---

## 🚀 快速开始

1. **保存脚本** 为 `lesson7/05_hybrid_inference/hybrid_inference.py`
2. **运行命令**
   ```bash
   python lesson7/05_hybrid_inference/hybrid_inference.py
   ```
3. **交互演示**
   ```text
   Customer Service Chatbot (Type 'exit' to end)

   User: What is RAG?
   Retrieved Answer: Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.
   Cosine Similarity Score: 0.89
   High retrieval score. Using retrieved answer directly.
   Agent: Retrieval-Augmented Generation (RAG) is a model architecture designed to improve the performance of text generation tasks.
   ```

---

## 🧠 脚本结构概览

```text
1. 初始化模型（Sentence-BERT + Qwen）
2. 文本预处理
3. 检索阶段：基于余弦相似度计算相似问句
4. 生成阶段：调用 Qwen 模型生成自然语言回答
5. 混合推理引擎：根据相似度阈值决定策略
6. 多轮对话：循环交互式问答
```

---

## 🧩 函数与模块说明

### 1️⃣ `preprocess(text)` — 文本预处理函数

**功能**：  
- 小写化文本；  
- 去除标点符号；  
- 分词处理。

**代码：**
```python
def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()
```

**用途**：保证输入格式一致，减少句向量计算误差。

---

### 2️⃣ `retrieve_answer(query, knowledge_base)` — 检索阶段

**功能**：  
基于 Sentence-BERT 模型计算用户查询与知识库问题的语义相似度。

**关键步骤**：
1. 将知识库问句编码为向量；  
2. 将用户问题编码为向量；  
3. 计算两者间的余弦相似度；  
4. 返回最相似问句对应的答案及相似度分数。

**核心公式**：  
$$
\text{cosine\_similarity}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}
$$

**返回**：  
- 最佳匹配答案 `best_match_answer`  
- 相似度分数 `similarity_score`

---

### 3️⃣ `generate_answer(query, context)` — 生成阶段

**功能**：  
当检索得分较低时，调用 **Qwen3** 等生成模型进行自然语言生成。

**代码逻辑**：
```python
inputs = tokenizer(query + " " + context, return_tensors="pt", truncation=True, padding=True, max_length=512)
outputs = model.generate(inputs['input_ids'], max_length=150)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**说明**：  
- 将检索到的答案作为“上下文 (context)”输入，指导生成模型输出更相关的回答；  
- 可扩展使用 **RAG / FiD / ChatGLM-RAG** 等结构实现多文档上下文生成。

---

### 4️⃣ `hybrid_inference(query, knowledge_base, threshold=0.7)` — 混合推理引擎

**功能**：  
融合“检索 + 生成”两种策略，实现动态决策。

**逻辑：**
1. 先执行 `retrieve_answer()` 获取相似度分数；  
2. 若相似度 ≥ 阈值（默认 0.7），直接返回检索答案；  
3. 否则，进入生成模式，通过语言模型生成新答案。

**伪代码：**
```python
if similarity_score >= threshold:
    return retrieved_answer
else:
    return generate_answer(query, retrieved_answer)
```

**用途**：模拟实际客服 / QA 系统中常用的 **两阶段决策机制**。

---

### 5️⃣ `run_conversation()` — 多轮对话管理

**功能**：  
模拟客服对话循环。  
用户可多次输入问题，程序根据阈值动态选择“检索或生成”策略。

**交互逻辑**：
- 输入 `exit` / `quit` / `bye` 结束对话；  
- 每次查询打印匹配结果与生成结果；  
- 输出最终客服回答。

**返回**：  
无返回值，但打印完整对话。

---

## 📊 工作流程总结

| 阶段 | 模块 | 核心任务 | 结果 |
|------|-------|-----------|------|
| 1 | Sentence-BERT | 将文本编码为向量 | 获取句子嵌入 |
| 2 | cosine_similarity | 计算查询与知识问句相似度 | 得到最相似问题 |
| 3 | Hybrid Engine | 阈值判断是否生成 | 决定输出方式 |
| 4 | Qwen 模型 | 自然语言生成 | 动态生成回答 |

---

## 🔍 优化建议

1. **改进检索效果**：  
   - 使用 **BM25 + 向量融合**（Hybrid Retrieval）。  
   - 采用更强的句向量模型（如 `text-embedding-3-large` 或 `bge-m3`）。

2. **提升生成质量**：  
   - 对 Qwen 模型进行 **指令微调 (SFT)** 或 **RLHF 调优**。  
   - 控制生成参数，如 `temperature`、`top_p`。

3. **加入置信度控制**：  
   - 若相似度低于阈值，可提示 “我不太确定，请您进一步说明问题”。

4. **多轮上下文追踪**：  
   - 记录用户历史问题，将上下文拼接进 `generate_answer()` 输入。

5. **性能优化**：  
   - 使用 GPU 加速或批量向量化编码；  
   - 对知识库向量进行缓存，避免重复计算。

---

## 💡 扩展作业建议

| 作业主题 | 内容 |
|-----------|-------|
| 作业1 | 将知识库存入 FAISS / Milvus 实现高效检索 |
| 作业2 | 增加中文知识库并切换多语言模型 |
| 作业3 | 结合 Flask / Gradio 构建 Web 聊天界面 |
| 作业4 | 实现 Top-K 检索结果融合生成（RAG Pipeline） |
| 作业5 | 加入日志系统与相似度可视化模块 |

---

## 📜 许可证

本教学脚本及文档仅用于**教学与研究目的**。  
涉及的开源模型与数据应遵守各自的许可证条款。
