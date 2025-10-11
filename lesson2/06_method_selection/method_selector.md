# LoRA / QLoRA / P‑Tuning 场景选择助手：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`finetune_strategy_selector.py`（基于你提供的教学代码整理）  
> 运行环境：Python ≥ 3.9  
> 目标：根据**硬件**、**数据规模**与**上线延迟要求**自动给出 **LoRA / QLoRA / P‑Tuning** 的微调方案建议。脚本面向教学，便于课程中快速讲解“该用哪种参数高效化技术”。

---

## 目录
- [LoRA / QLoRA / P‑Tuning 场景选择助手：使用说明与函数文档](#lora--qlora--ptuning-场景选择助手使用说明与函数文档)
  - [目录](#目录)
  - [一、功能概述](#一功能概述)
  - [二、快速开始](#二快速开始)
    - [1) 保存脚本](#1-保存脚本)
    - [2) 运行最小示例](#2-运行最小示例)
    - [3) 在你的工程中调用](#3-在你的工程中调用)
  - [三、类型与数据结构](#三类型与数据结构)
    - [枚举类型](#枚举类型)
    - [`Scenario`](#scenario)
  - [四、核心函数](#四核心函数)
    - [`recommend(scenario: Scenario) -> str`](#recommendscenario-scenario---str)
  - [五、决策逻辑：一页纸备忘](#五决策逻辑一页纸备忘)
  - [六、使用示例](#六使用示例)
    - [示例 1：CPU 环境](#示例-1cpu-环境)
    - [示例 2：单卡 + 大规模数据](#示例-2单卡--大规模数据)
    - [示例 3：多卡 + 严苛延迟](#示例-3多卡--严苛延迟)
  - [七、与工程实践的对应关系](#七与工程实践的对应关系)
  - [八、扩展与定制](#八扩展与定制)
  - [九、常见问题（FAQ）](#九常见问题faq)
  - [十、许可证](#十许可证)

---

## 一、功能概述

该脚本提供一个**场景选择助手**，输入三类关键信息：
- **硬件**（CPU / 单卡 / 多卡）  
- **数据规模**（small / medium / large）  
- **上线延迟要求**（strict / normal）  

输出一句**人类可读的推荐**，告诉你更适合 LoRA、QLoRA 还是 P‑Tuning，以及需要配合的工程手段（量化、梯度检查点、分布式等）。

> 注：逻辑基于教学经验进行简化，帮助建立“何时用什么”的直觉；不是完整的生产策略引擎。

---

## 二、快速开始

### 1) 保存脚本
将教学代码保存为 `finetune_strategy_selector.py`。

### 2) 运行最小示例
```bash
python finetune_strategy_selector.py
```
默认会打印：
```text
QLoRA 即可满足需求，利用 4bit 加载与梯度检查点应对大规模数据。
```

### 3) 在你的工程中调用
```python
from finetune_strategy_selector import Scenario, recommend

case = Scenario(hardware="single_gpu", data_scale="large", latency="normal", description="7B + A100 80G")
print(recommend(case))
```

---

## 三、类型与数据结构

### 枚举类型
```python
Hardware = Literal["cpu", "single_gpu", "multi_gpu"]
DataScale = Literal["small", "medium", "large"]
LatencyRequirement = Literal["strict", "normal"]
```
- **Hardware**：训练/推理可用的计算资源类型；  
- **DataScale**：训练数据的量级（教学分档）；  
- **LatencyRequirement**：上线延迟要求，“strict” 表示**严格**的响应时延（对 TPS/吞吐/冷启动敏感）。

### `Scenario`

```python
@dataclass
class Scenario:
    hardware: Hardware
    data_scale: DataScale
    latency: LatencyRequirement
    description: str
```
- **hardware** / **data_scale** / **latency**：决策的三要素；  
- **description**：补充说明（例如“7B + 单卡 A100 80G / 线上低延迟”），用于日志/审计。

---

## 四、核心函数

### `recommend(scenario: Scenario) -> str`
**功能**：根据场景返回简明建议。

**当前决策规则（自上而下匹配）：**
```python
if hardware == "cpu":
    "推荐 LoRA 或 P-Tuning，优先选择轻量模型并结合量化部署。"

elif hardware == "single_gpu":
    if data_scale == "large":
        "推荐 QLoRA，利用 4bit 加载与梯度检查点应对大规模数据。"
    else:
        "LoRA 即可满足需求，结合梯度累积提升吞吐。"

elif hardware == "multi_gpu":
    if latency == "strict":
        "QLoRA + 分布式推理，或蒸馏后上线轻量模型。"
    else:
        "LoRA/QLoRA 均可，根据数据规模选择是否量化。"

else:
    "请提供完整场景信息。"
```
**返回值**：`str`，人类可读的推荐语。

**设计初衷**：
- **CPU**：训练/微调资源最紧张，倾向于**最小增量**的 P‑Tuning 或 LoRA，并配合**量化部署**（8/4bit）；  
- **单卡 + 大数据**：为了**降低显存**、容纳更长上下文与更大批量，采用 **QLoRA（4bit + LoRA）** 并结合梯度检查点；  
- **多卡**：资源充裕，但**上线延迟严格**时，优先考虑 QLoRA + 分布式推理（或将 LoRA/QLoRA 结果**蒸馏**到轻量模型上线）。

---

## 五、决策逻辑：一页纸备忘

| 场景 | 推荐 | 附加建议 |
|---|---|---|
| CPU-only | **LoRA / P‑Tuning** | 选小模型；**8/4bit 量化**部署；可先离线蒸馏 |
| 单卡 + 小/中数据 | **LoRA** | **梯度累积** 提升有效 batch；适度 **prompt 模板统一** |
| 单卡 + 大数据 | **QLoRA** | **4bit + 梯度检查点**；paged optimizer；数据清洗与对齐 |
| 多卡 + 严苛延迟 | **QLoRA + 分布式推理** 或 **蒸馏上线** | 打开 KV cache；批处理/并行；必要时**蒸馏到小模型** |
| 多卡 + 正常延迟 | **LoRA / QLoRA** | 按数据规模选择量化与否；混合精度；评估吞吐/成本 |

> **术语小抄**：  
> **LoRA**（低秩增量）→ 训练少量参数；  
> **QLoRA**（4bit + LoRA）→ 显存更省；  
> **P‑Tuning v2**（可学习前缀）→ 训练参数更少，适合快速对齐与低资源场景。

---

## 六、使用示例

### 示例 1：CPU 环境
```python
case = Scenario("cpu", "small", "normal", "消费级 CPU")
print(recommend(case))
# → 推荐 LoRA 或 P-Tuning，优先选择轻量模型并结合量化部署。
```

### 示例 2：单卡 + 大规模数据
```python
case = Scenario("single_gpu", "large", "normal", "单卡 A100 80G，数据>100k")
print(recommend(case))
# → 推荐 QLoRA，利用 4bit 加载与梯度检查点应对大规模数据。
```

### 示例 3：多卡 + 严苛延迟
```python
case = Scenario("multi_gpu", "medium", "strict", "线上低延迟 SLA")
print(recommend(case))
# → QLoRA + 分布式推理，或蒸馏后上线轻量模型。
```

---

## 七、与工程实践的对应关系

- **LoRA**：优先在注意力的 `q/k/v/o`、MLP 的 `up/down/gate` 投影注入；结合 **bf16/fp16**；上线可选择 **merge & unload**。  
- **QLoRA**：通过 `bitsandbytes` **4bit（NF4）** 加载 + `prepare_model_for_kbit_training`；训练用 **paged optimizer** 与 **checkpointing**。  
- **P‑Tuning v2**：`PromptEncoder` 在各层 Attention 注入 prefix K/V；参数更少，微调更快，适合轻量场景。  
- **低延迟上线**：KV Cache、批处理、张量并行/流水并行、垂直/水平扩展；必要时将 LoRA/QLoRA 结果**蒸馏**到小模型。

---

## 八、扩展与定制

1. **更多维度**：加入 `sequence_length`、`budget`、`latency_target_ms`、`model_size_b` 等字段；  
2. **规则→评分**：从 if-else 过渡为**加权评分**或**决策树**；输出 Top‑K 方案与理由；  
3. **解释器**：返回结构化 JSON（包含推荐、加分项、风险项），便于前端渲染或日志审计；  
4. **数据驱动**：对历史实验记录做拟合（学习型策略），动态更新推荐权重；  
5. **一键脚手架**：按推荐方案生成 **训练/推理** 命令与配置（LoRA/QLoRA/P‑Tuning 模板）。

---

## 九、常见问题（FAQ）

1. **为什么只有一句推荐？**  
   - 教学示例聚焦“建立直觉”。若要工程落地，建议改为**结构化多维输出**。

2. **如何处理模型大小差异（7B/13B/70B）？**  
   - 可在 `Scenario` 中加入 `model_size_b` 字段，并在规则中显式考虑显存与吞吐的量纲。

3. **线上延迟与召回/质量怎么权衡？**  
   - 可添加“质量优先/成本优先”开关，或在推荐中给出“蒸馏上线 vs 原模型上线”的分支。

4. **能否同时返回部署建议？**  
   - 是的。比如：LoRA → FP16 合并权重部署；QLoRA → 4bit 推理 + KV cache；多卡时增加张量并行/流水并行建议。

---

## 十、许可证

本脚本与文档用于教学用途。请遵循你项目的总体许可证以及上游依赖和权重的许可证要求。
