# 教程：RAG 检索召回率与生成准确性优化

## 学习目标
- 构建包含问题、检索片段与回答的评估数据集。
- 计算简化版 Recall@K 与语义相似度，用于衡量检索与生成效果。
- 根据评估结果输出优化建议，指导系统迭代。

## 背景原理
- **Recall@K**：检索阶段的指标，衡量关键知识是否在前 K 个候选中被覆盖。
- **语义相似度**：这里通过字符 n-gram Jaccard 近似，实际可替换为嵌入余弦或 BLEU/ROUGE。
通过同时关注检索与生成两端指标，可以定位问题来源（检索失败或生成幻觉）。

## 代码结构解析
- `EvalSample` 与 `load_eval_samples`：读取 TSV 数据，解析问题、标准答案、检索文档、生成结果。
- `recall_at_k`：判断检索片段是否包含关键词，模拟召回覆盖率。
- `semantic_similarity`：计算回答与参考答案的字符级相似度。
- `evaluate`：汇总平均 Recall@K 与平均相似度。
- `optimization_recommendations`：提供改进策略列表。

## 实践步骤
1. 准备评估数据，格式：`问题\t标准答案\t文档1<sep>文档2\t模型回答`。
2. 运行脚本：
   ```bash
   python optimization_lab.py eval_samples.tsv --k 5
   ```
3. 解读输出的 Recall@K 与语义相似度，识别系统瓶颈。
4. 根据推荐策略执行迭代，例如调整分块、加入 rerank 模型或引入事实性检测。

## 拓展问题
- 如何将 `semantic_similarity` 替换为基于嵌入或 LLM 评审的更精确指标？
- Recall@K 可否结合多跳检索（Multi-hop）扩展到复杂问题？
- 在 A/B 测试中，如何设计显著性检验确保指标提升可信？
