
# 课程实验：知识库增量更新与微调触发策略（教学版文档）

本教学脚本通过**极简模拟**演示了在 RAG/问答系统中，围绕“文档更新 ↔ 索引维护 ↔ 模型微调触发”的基本流程与关键决策点。你将学到：

- 如何在**知识库**中进行**新增 / 更新 / 删除**（CRUD）并维护“最后更新时间”。  
- 如何用**增量策略**更新索引，而不是每次全量重建。  
- 如何用**阈值策略**判断是否需要触发**检索器/生成器微调**（Retraining Policy）。  
- 如何将这些步骤落地到工程实践（日志、调度、指标面板）。

> 脚本定位为“**最小可运行教学版**”，便于课堂讲解与作业延展，非生产可直接复用代码。

---

## 目录
- [课程实验：知识库增量更新与微调触发策略（教学版文档）](#课程实验知识库增量更新与微调触发策略教学版文档)
  - [目录](#目录)
  - [运行环境与依赖](#运行环境与依赖)
  - [快速开始](#快速开始)
  - [脚本结构总览](#脚本结构总览)
  - [核心类与函数说明](#核心类与函数说明)
    - [`KnowledgeBase`](#knowledgebase)
      - [属性](#属性)
      - [方法](#方法)
    - [`RetrainingPolicy`](#retrainingpolicy)
      - [初始化参数](#初始化参数)
      - [方法](#方法-1)
    - [主流程（增量更新 + 触发判定）](#主流程增量更新--触发判定)
  - [使用示例与输出](#使用示例与输出)
    - [示例输入（节选）](#示例输入节选)
    - [运行后可能输出](#运行后可能输出)
  - [设计要点与工程化建议](#设计要点与工程化建议)
  - [扩展示例：如何接入真实检索与训练](#扩展示例如何接入真实检索与训练)
  - [常见问题（FAQ）](#常见问题faq)
  - [教学作业建议](#教学作业建议)
  - [许可证](#许可证)

---

## 运行环境与依赖

- Python 3.8+  
- 本脚本使用标准库（`datetime`、`logging` 等），**无第三方硬依赖**。  
- 若要延展到真实索引/检索（如 FAISS、BM25、向量数据库），请安装相应库：  
  ```bash
  pip install faiss-cpu sentence-transformers rank-bm25
  ```

---

## 快速开始

将原脚本保存为 `lesson7/03_incremental_updates/update_strategy.pyy`，直接运行：
```bash
python lesson7/03_incremental_updates/update_strategy.py
```
你将看到：
1. 知识库索引在**新增 / 更新 / 删除**后的当前状态；  
2. 基于**更新占比**的**重训练触发**打印结果。

---

## 脚本结构总览

```python
# —— 场景说明（多行注释）：微调的意义与步骤 ——

# 1) 模拟的原始文档集合（documents）
# 2) 三类变化：new_documents / updated_documents / deleted_documents

class KnowledgeBase:
    add_document() / update_document() / delete_document() / display_index()

# 3) 将三类变化应用到 KnowledgeBase（增量更新）

class RetrainingPolicy:
    check_for_retraining(total_documents, updated_documents)

# 4) 执行触发判定，打印策略结论
```

---

## 核心类与函数说明

### `KnowledgeBase`

> 用于模拟知识库与其“**倒排索引/向量索引**”的最小形态。每个文档在 `self.index` 中以 `doc_id` 为键，存储 `content` 与 `last_modified`。

#### 属性
- `index: Dict[int, Dict]`：文档 ID → `{content, last_modified}` 的字典。

#### 方法
- `add_document(doc_id: int, content: str) -> None`  
  新增文档到索引。`last_modified` 设为 **当前时间**（模拟“入库时间”）。

- `update_document(doc_id: int, content: str) -> None`  
  若文档存在，更新内容并刷新 `last_modified`。若不存在，在工程实践中建议：  
  - 选择“**新增**”，或  
  - 记录**异常**交由数据治理。

- `delete_document(doc_id: int) -> None`  
  从索引中移除该文档。注意：向量库/搜索引擎通常需要**同步删除**对应条目。

- `display_index() -> None`  
  将当前索引打印到终端：包含 `Doc ID / Content / Last Modified`。

**注意**：示例中 `last_modified` 使用 `datetime.now()`，真实系统应保留**来源侧更新时间**（如对象存储的 `mtime`）、**入库时间**与**索引重建时间**等多种时间戳，以便追溯。

---

### `RetrainingPolicy`

> 根据“**文档更新比例**”判断是否需要触发**重训练/微调**。此策略可以快速让团队形成共识：**更新多 → 更可能影响检索召回与生成质量 → 值得重训**。

#### 初始化参数
- `threshold: float = 0.1`  
  触发阈值，**更新文档数 / 文档总数** 若 **> `threshold`**，判定为**需要触发**。

#### 方法
- `check_for_retraining(total_documents: int, updated_documents: Dict|List) -> None`  
  - 计算：`updated_ratio = len(updated_documents) / total_documents`  
  - 打印更新比例；对比阈值给出结论：  
    - `updated_ratio > threshold` → `"Triggering retraining..."`  
    - 否则 → `"No retraining needed."`

**建议阈值**：
- 小型库：`5%~10%`；  
- 高频更新场景（新闻、行情）：可更低阈值或引入**滚动窗口**（近 24h/7d 更新比）。

---

### 主流程（增量更新 + 触发判定）

1. **初始化**：构造 `KnowledgeBase()`。  
2. **应用变化集**：
   - `new_documents` → `add_document()`  
   - `updated_documents` → `update_document()`  
   - `deleted_documents` → `delete_document()`  
3. **展示索引**：`display_index()`  
4. **判定微调**：`RetrainingPolicy().check_for_retraining()`

> 在真实项目中，这些步骤通常由**定时任务**（如 Airflow/Cron）或**消息队列事件**（如 Kafka）触发。

---

## 使用示例与输出

### 示例输入（节选）
```python
new_documents = {4: {"content": "Document 4 about search models.", "last_modified": datetime(2023, 8, 10)}}
updated_documents = {1: {"content": "Updated Document 1 about RAG and retrieval.", "last_modified": datetime(2023, 8, 5)}}
deleted_documents = [2]
```

### 运行后可能输出
```text
Doc ID: 4, Content: Document 4 about search models., Last Modified: 2025-10-11 14:23:11.102938

Updated Ratio: 0.33
Triggering retraining due to high document updates!
```

解释：
- 最终索引里只展示**还在库中的**文档（示例只新增了 4 号）。  
- 更新比例 = `len(updated_documents)/len(documents) = 1/3 ≈ 0.33`，超过默认阈值 `0.1`，因此触发微调。

---

## 设计要点与工程化建议

1. **时间戳与版本**  
   - 存储**来源更新时间 / 入库时间 / 索引更新时间**，帮助你判断**增量索引构建**与**回滚**。

2. **增量索引**  
   - 对新增/更新/删除分别维护**待处理队列**：  
     - 新增：插入向量与倒排条目  
     - 更新：删除旧条目 + 插入新条目  
     - 删除：移除索引 & 关联元数据  
   - 定期执行**一致性校验**（漏删/漏更）。

3. **触发策略**  
   - 维度不仅限于“**更新占比**”：
     - **检索指标**（Recall@K、MRR、nDCG）持续下滑  
     - **生成质量**指标（Faithfulness、ROUGE、人工评分）下降  
     - **业务触发**（重大发布、法规变化）  
   - 支持**软触发**（先 A/B）与**硬触发**（强制全量重训）。

4. **日志与可观测性**  
   - `logging` 打印应包含：变更量、失败条目数、耗时、索引大小、模型版本号。  
   - 将关键信息写入**监控系统**（Prometheus/Grafana）或**实验追踪**（MLflow/W&B）。

5. **资源与成本**  
   - 为重训练任务设置**配额与预算**，避免频繁大规模重训。  
   - 结合**迁移学习**、**增量训练**、**蒸馏**降低成本。

---

## 扩展示例：如何接入真实检索与训练

以下为将教学脚本“落地化”的思路片段：

- **接入向量索引（FAISS/Milvus/PGVector）**
  ```python
  import faiss
  # 在 add_document/update_document 中：编码文本 -> upsert 向量
  ```

- **接入稠密编码器（Sentence-BERT/Embedding API）**
  ```python
  from sentence_transformers import SentenceTransformer
  encoder = SentenceTransformer('all-MiniLM-L6-v2')
  vec = encoder.encode([content])
  ```

- **触发训练流水线**
  ```bash
  # 触发 MLflow Job / Airflow DAG / SageMaker Pipeline
  mlflow run . -P task=retrain_retriever -P data_version=2025-10-11
  ```

- **最小评测闭环**
  - 取固定查询集，重训后对比 **Recall@K / nDCG / EM / Faithfulness**。  
  - 指标无提升 → 回滚模型/索引版本。

---

## 常见问题（FAQ）

**Q1：为什么只更新了索引，没有看到“旧文档”？**  
A：示例只演示了对 `new/updated/deleted` 的**即时处理**和**当前态**打印。若需全量视图，请先把 `documents` 批量 `add_document` 进 `KnowledgeBase`，再应用增量变化。

**Q2：阈值如何选？**  
A：结合历史数据回放（replay）与业务容忍度调参：过低会频繁重训，过高可能错过关键知识更新。

**Q3：能否按“字段级”更新？**  
A：可。建议在元数据层记录字段差异，并据此决定是否需要重新编码/重建索引。

---

## 教学作业建议

- **作业 A：接入真实向量库**（FAISS/Milvus），完成 `add/update/delete` 的 upsert 逻辑与一致性校验。  
- **作业 B：评测闭环**：加入固定 Query 集合与指标统计（Recall@K、nDCG），输出 CSV + 可视化。  
- **作业 C：训练触发策略对比**：实现“更新占比阈值 vs. 指标劣化阈值 vs. 混合策略”的优劣对比实验。  
- **作业 D：灰度与回滚**：设计上线流程，包含 Canary 流量、回滚条件、版本标记。

---

## 许可证

本教学文档与示例代码仅用于**教学/研究**。如需在生产中使用，请结合贵司合规与隐私政策完善安全与审计措施。
