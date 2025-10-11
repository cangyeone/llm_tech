
# 教程文档：客服场景对齐案例分析（最小可运行版）

本教学脚本展示了如何在客服场景中模拟**人类偏好建模（Human Preference Modeling）**过程，
通过“优选回答（chosen）”与“拒绝回答（rejected）”示例，计算并比较回复的覆盖度与礼貌性，
从而形成一种可解释的偏好对齐评分机制。

---

## 一、功能概述

- 构造一个包含客服对话的偏好数据集；
- 为候选回复计算两个维度的得分：
  - **覆盖度（Coverage）**：回答内容是否涵盖用户诉求；
  - **礼貌性（Politeness）**：回答是否体现客服礼貌与同理心；
- 计算总分：`score = 0.6 * coverage + 0.4 * politeness`；
- 导出结果到 CSV 文件以供人工回顾；
- 可选接入 **Qwen3 模型** 自动生成额外候选回复（默认关闭）。

运行方式：
```bash
python lesson5/05_customer_support_case/customer_alignment.py
```

---

## 二、核心流程

1. 构建数据集（包含 chosen / rejected 样本）
2. 计算每条回复的礼貌性与覆盖度得分
3. 综合评分与备注生成
4. 导出为 CSV 文件
5. 生成控制台摘要报告

---

## 三、主要函数说明

### 1️⃣ `maybe_load_model()`
**功能**：懒加载 Qwen3 模型，用于自动生成候选回复。  
**返回值**：`(tokenizer, model)` 或 `(None, None)`。  
**说明**：若 `USE_MODEL=False`，此函数不会加载模型。

---

### 2️⃣ `model_generate(tok, model, prompt)`
**功能**：调用语言模型生成客服回答。  
**参数**：
- `tok`: Tokenizer 对象  
- `model`: 加载的模型对象  
- `prompt`: 输入提示文本  

**输出**：返回生成的回答字符串（截取“回答:”后内容）。

---

### 3️⃣ `politeness_score(text)`
**功能**：计算礼貌性得分。  
**原理**：
- 统计出现的正向礼貌词（如“您好”“感谢”“请”）与负向词（如“滚”“自己”“不行”）；  
- 通过公式计算：  
  $$
  politeness = \text{clamp}(0.5 + 0.15 \times pos - 0.25 \times neg)
  $$  
**返回值**：范围 `[0, 1]`。

---

### 4️⃣ `coverage_score(query, tags, reply)`
**功能**：计算回答内容与查询标签的覆盖度。  
**方法**：统计回复中命中的关键词比例，并对“可执行动作”词加奖励。  
**公式**：  
  $$
  coverage = \text{clamp}\Big(\frac{\text{hits}}{N} + 0.1\times I_\text{action}\Big)
  $$  
**返回值**：范围 `[0, 1]`。

---

### 5️⃣ `total_score(coverage, politeness)`
**功能**：根据两维指标综合计算总得分：  
  $$
  score = 0.6 \times coverage + 0.4 \times politeness
  $$

---

### 6️⃣ `annotate(example)`
**功能**：对单个客服样本进行评分与备注生成。  
**逻辑**：
- 调用 `coverage_score` 与 `politeness_score`；  
- 生成备注：
  - 覆盖不足 / 礼貌性弱 / 覆盖充分 / 礼貌到位 / 表现良好。

---

### 7️⃣ `build_dataset()`
**功能**：构建全量数据集。  
**说明**：
- 包含三条工单，每条工单有一条 chosen 和一条 rejected 回复；  
- 若开启模型生成选项（`USE_MODEL=True`），会额外加入模型生成候选。

---

### 8️⃣ `export_csv(rows, out_path)`
**功能**：将评分结果导出为 CSV 文件。  
**输出路径**：默认 `outputs/cx_alignment_review.csv`。  
**输出字段**：
`ticket_id`, `candidate_type`, `score`, `coverage`, `politeness`, `customer_query`, `reply`, `context_tags`, `notes`。

---

### 9️⃣ `print_brief_report(rows, top_k=3)`
**功能**：在控制台打印简要报告。  
**内容**：
- 样本总数与平均分；  
- 得分最高与最低的若干条记录。

---

### 🔟 `main()`
**功能**：执行完整流程。  
**步骤**：
1. 调用 `build_dataset()` 生成数据；  
2. 执行评分并导出 CSV；  
3. 打印报告。

---

## 四、运行示例

执行：
```bash
python cx_alignment_tutorial.py
```

输出示例：
```
=== 总览 ===
样本数：6；平均总分：0.742

=== Top 表现 ===
[chosen] T001  score=0.930  cov=0.900  pol=0.900  ▶ 您好～我已为您查询到该订单今天已出库...
[chosen] T002  score=0.890  cov=0.850  pol=0.850  ▶ 非常抱歉给您带来不便。我可以协助您退货...

=== 需改进 ===
[rejected] T003  score=0.250  cov=0.300  pol=0.200  ▶ 不行，不能改。...
✅ 结果已导出：outputs/cx_alignment_review.csv
👉 可用 Excel/Numbers 打开，或导入业务看板进行回顾。
```

---

## 五、教学扩展

- 修改权重比例（0.6 / 0.4）观察结果变化；  
- 添加更多正负样本，模拟不同客服风格；  
- 结合大模型生成更多候选回复，训练奖励模型；  
- 将 CSV 数据接入可视化工具（如 Power BI、Tableau）。

---

## 六、目录结构（建议）

```
cx_alignment_tutorial.py
outputs/
 └── cx_alignment_review.csv
```

---

## 七、许可证

本代码与文档仅供教学与科研使用，不得用于商业用途。  
建议保留原始署名与说明。

