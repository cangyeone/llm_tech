
# 教程文档：对齐后生成结果的多样性评估（distinct-n & self-BLEU）

本教学脚本演示如何对语言模型的生成结果进行**多样性评估（Diversity Evaluation）**，
通过 **distinct-n** 与 **self-BLEU** 指标，量化模型输出的多样性与模式坍塌风险。

---

## 一、功能概述

- 多次采样生成模型回答（使用随机采样策略）  
- 计算文本多样性指标：
  - **distinct-1/2/3**：统计 n-gram 去重比率
  - **self-BLEU**：衡量不同生成样本之间的相似度
- 调整采样参数（temperature、top-k、top-p）观察模式坍塌趋势
- 自动输出评估结果与调参建议

运行方式：
```bash
pip install transformers torch
python diversity_eval.py
```

---

## 二、核心流程

1. 初始化模型与采样参数  
2. 多次生成样本文本  
3. 计算 distinct-n 与 self-BLEU  
4. 输出结果、分析提示与示例文本

---

## 三、主要函数说明

### 1️⃣ `get_device()`
**功能**：自动检测可用计算设备。  
**返回值**：`"cuda"`、`"mps"` 或 `"cpu"`。

---

### 2️⃣ `simple_tok(s)`
**功能**：对文本进行简单分词，兼容中英文与符号。  
**返回值**：分词后的 token 列表。  
**说明**：用于 distinct-n 与 BLEU 计算的统一分词规则。

---

### 3️⃣ `ngrams(tokens, n)`
**功能**：生成 n-gram 序列列表。  
**输入**：token 列表与 n 值。  
**输出**：长度为 n 的 token 元组列表。

---

### 4️⃣ `distinct_n(texts, n)`
**功能**：计算 distinct-n 指标。  
**原理**：
$$
\text{distinct-n} = \frac{\text{unique n-grams}}{\text{total n-grams}}
$$  
**意义**：值越大说明生成文本越多样化。

---

### 5️⃣ `bleu_score(hyp, refs, max_n=4)`
**功能**：计算单个样本的 BLEU 分数。  
**实现**：基于 n-gram 匹配与加平滑。  
**返回值**：一个 0~1 的浮点值，越高表示越相似。

---

### 6️⃣ `self_bleu(texts, max_n=4)`
**功能**：计算所有生成样本的平均相似度。  
**公式**：
$$
\text{self-BLEU} = \frac{1}{N}\sum_{i=1}^{N}\text{BLEU}(t_i, T\setminus t_i)
$$  
**意义**：值越低表示输出差异越大，越多样化。

---

### 7️⃣ `generate_samples(...)`
**功能**：使用语言模型多次采样生成回答。  
**输入参数**：
- `model_name`：模型名称，如 `"Qwen/Qwen3-0.6b"`  
- `prompt`：输入提示语  
- `samples`：采样次数  
- `max_new`：最大生成长度  
- `temp`：temperature，越高越随机  
- `top_p`：nucleus sampling 参数  
- `top_k`：top-k 采样阈值  

**输出**：生成文本列表。

---

### 8️⃣ `main()`
**功能**：主执行逻辑。  
**流程**：
1. 打印运行配置；  
2. 调用模型生成多样化回答；  
3. 计算 distinct-1/2/3 与 self-BLEU；  
4. 输出评估指标与启发式调参建议；  
5. 展示前 3 条示例文本。

---

## 四、输出指标说明

| 指标 | 含义 | 理想趋势 |
|------|------|-----------|
| distinct-1 | 不重复的 unigram 比率 | 越高越好 |
| distinct-2 | 不重复的 bigram 比率 | 越高越好 |
| distinct-3 | 不重复的 trigram 比率 | 越高越好 |
| self-BLEU | 样本间相似度 | 越低越好 |

---

## 五、运行示例

```bash
Device: cuda | Model: Qwen/Qwen3-0.6b
Prompt: 请用3-5句话解释什么是RLHF，以及它为何对客服机器人重要。
Sampling: temp=0.8 top_p=0.95 top_k=50 samples=20 max_new=128

=== Diversity Metrics ===
distinct-1: 0.5123
distinct-2: 0.2781
distinct-3: 0.1610
self-BLEU : 0.4875   (越低越分散，越高越相似)

=== Heuristic Tips ===
• 多样性尚可。可在保持质量前提下微调 temperature/top-p 寻找最佳点。

=== Samples (first 3) ===
[1] RLHF 是一种将人类反馈引入训练过程的方法，用于优化模型生成的回复质量与价值观一致性。
[2] RLHF 是强化学习在人机对齐中的一种实践，它通过人类反馈...
```

---

## 六、教学拓展

- 调整 temperature、top-p、top-k 观察多样性变化趋势；  
- 对比不同模型（如 Qwen vs ChatGLM）的 self-BLEU；  
- 可加入“语义相似度”评估指标补充判定；  
- 在 RLHF 教学中，用于分析模型模式坍塌（mode collapse）现象。

---

## 七、目录结构

```
diversity_eval.py
```

---

## 八、许可证

本代码与文档仅供教学与科研使用，禁止未经授权的商业用途。  
保留原始署名与版权说明。
