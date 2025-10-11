# 模型压缩实验对比：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson2/04_model_compression/compression_lab.py`（基于你提供的教学代码整理）  
> 运行环境：Python ≥ 3.9；`torch`（CPU 可运行，GPU 可加速）  
> 目标：用**随机权重+最小实现**演示三类压缩技术的直觉对比：**结构化剪枝（注意力头）**、**知识蒸馏**、**低比特量化（INT8）**。

---

## 目录
- [模型压缩实验对比：使用说明与函数文档](#模型压缩实验对比使用说明与函数文档)
  - [目录](#目录)
  - [一、概述](#一概述)
  - [二、快速开始](#二快速开始)
    - [依赖安装](#依赖安装)
    - [保存并运行](#保存并运行)
  - [三、脚本结构与流程](#三脚本结构与流程)
  - [四、类与函数文档](#四类与函数文档)
    - [`ToyTransformer(nn.Module)`](#toytransformernnmodule)
    - [`DistilledStudent(nn.Module)`](#distilledstudentnnmodule)
    - [`structured_pruning(model: ToyTransformer, keep_heads: int = 4) -> None`](#structured_pruningmodel-toytransformer-keep_heads-int--4---none)
    - [`knowledge_distillation(teacher: ToyTransformer, student: DistilledStudent) -> float`](#knowledge_distillationteacher-toytransformer-student-distilledstudent---float)
    - [`simulate_int8_quantization(tensor: torch.Tensor) -> torch.Tensor`](#simulate_int8_quantizationtensor-torchtensor---torchtensor)
    - [`main() -> None`](#main---none)
  - [五、数学与工程直觉](#五数学与工程直觉)
    - [结构化剪枝（注意力头）](#结构化剪枝注意力头)
    - [知识蒸馏](#知识蒸馏)
    - [INT8 量化](#int8-量化)
  - [六、运行示例与典型输出](#六运行示例与典型输出)
  - [七、常见问题排查（FAQ）](#七常见问题排查faq)
  - [八、扩展建议](#八扩展建议)
  - [九、许可证](#九许可证)

---

## 一、概述

该教学脚本**不依赖真实数据**，通过小型模块与随机输入，直观展示：

1. **结构化剪枝**：按**注意力头**粒度屏蔽权重，降低算力/显存占用；  
2. **知识蒸馏**：训练**学生模型**去拟合**教师模型**的隐藏表示（用 MSE 作为教学示例）；  
3. **低比特量化**：将 `float` 张量映射到 **INT8** 的离散值并反量化，观察量化误差。

这些技术在真实工程中常被**组合使用**（如“蒸馏 + 量化 + 轻量结构改造”），以在保持精度的前提下降低延迟与成本。

---

## 二、快速开始

### 依赖安装
```bash
pip install torch
```

### 保存并运行
将代码保存为 `lesson2/04_model_compression/compression_lab.py` 并执行：
```bash
python lesson2/04_model_compression/compression_lab.py
```
脚本会：剪枝注意力头 → 运行蒸馏循环（50步）→ 模拟 INT8 量化 → 打印蒸馏损失。

---

## 三、脚本结构与流程

1. **定义模型**：`ToyTransformer`（单层 MHA + FFN）与 `DistilledStudent`（线性层 + tanh）。  
2. **结构化剪枝**：`structured_pruning` 对 `MultiheadAttention.in_proj_weight` 做**掩码**保留前 `keep_heads` 个头。  
3. **蒸馏训练**：`knowledge_distillation` 用**随机输入**驱动教师与学生，最小化 `MSE(student(x), teacher(x))`。  
4. **量化模拟**：`simulate_int8_quantization` 对随机向量做线性缩放 + 四舍五入 + 截断，计算反量化误差。  
5. **主程序**：按顺序调用上述步骤，并输出 `{"distill_loss": ...}`。

---

## 四、类与函数文档

### `ToyTransformer(nn.Module)`

```python
@dataclass
class ToyTransformer(nn.Module):
    hidden_size: int = 256
    num_heads: int = 8

    def __post_init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(self.hidden_size, self.num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.ffn(attn_out)
```
- **作用**：一个极简的 Transformer block（仅 1 层），包含 **多头注意力** 与 **前馈网络**。  
- **输入/输出**：
  - 输入 `x` 形状 `[B, T, H]`；
  - 输出与输入同形状 `[B, T, H]`。  
- **注意**：用于教学直觉，不含残差、归一化等完整组件。

---

### `DistilledStudent(nn.Module)`

```python
class DistilledStudent(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))
```
- **作用**：学生模型（结构更简单），用来拟合教师输出；  
- **设计**：单线性层 + `tanh` 激活；仍保持输入/输出维度一致，便于与教师对齐。

---

### `structured_pruning(model: ToyTransformer, keep_heads: int = 4) -> None`

**功能**：按**头粒度**屏蔽 `MultiheadAttention` 的 `in_proj_weight`，仅保留前 `keep_heads` 个头的投影权重。

**实现要点**：
```python
head_dim = model.hidden_size // model.num_heads
mask = torch.zeros_like(model.attn.in_proj_weight)
for i in range(keep_heads):
    start, end = i * head_dim, (i + 1) * head_dim
    mask[start:end, :] = 1.0
model.attn.in_proj_weight.data *= mask
```
- 通过与 `mask` 相乘实现**硬剪枝**；  
- **提示**：真实工程多会**重建投影矩阵**并同步处理 `out_proj`/偏置、做**微调恢复**；这里只做**快速可视化**。

---

### `knowledge_distillation(teacher: ToyTransformer, student: DistilledStudent) -> float`

**功能**：用**均方误差（MSE）**模拟蒸馏训练，让学生拟合教师的**隐藏表示**。

**流程**：
1. 随机采样小批输入 `inputs ~ N(0, I)`（形状 `[2, 16, H]`）；  
2. 冻结教师，计算 `teacher_output = teacher(inputs)`；  
3. 前向学生，得 `student_output`；  
4. `loss = MSE(student_output, teacher_output)`；  
5. `AdamW` 更新学生参数；循环 50 次并返回最终损失。

**返回**：`float`，最后一个 step 的 `loss.item()`。

> **扩展**：真实蒸馏可采用 KL 散度对**logits**蒸馏、对齐**中间层**、或引入**多任务损失**（如 CE + KD）。

---

### `simulate_int8_quantization(tensor: torch.Tensor) -> torch.Tensor`

**功能**：模拟 INT8 量化 → 反量化，并打印 `||x - dequant(x)||_2` 作为误差指标。

**实现**：
```python
qmin, qmax = -128, 127
scale = tensor.abs().max() / qmax
quantized = torch.clamp((tensor / scale).round(), qmin, qmax)
dequant = quantized * scale
```
- **线性缩放**：用最大绝对值做对称量化的 `scale`；  
- **四舍五入 + 截断**：得到离散整值；  
- **反量化**：乘回 `scale`。

**返回**：反量化后的张量（`torch.Tensor`）。

> **提示**：真实 INT8 会考虑**分组/通道级**量化、零点、统计量估计等；这里仅做教学直觉。

---

### `main() -> None`

**流程**：
1. `ToyTransformer()` → `structured_pruning(keep_heads=4)`；  
2. `teacher = ToyTransformer()`、`student = DistilledStudent(H)` → `knowledge_distillation(...)`；  
3. `simulate_int8_quantization(torch.randn(1024))`；  
4. 打印 `{"distill_loss": ...}`。

---

## 五、数学与工程直觉

### 结构化剪枝（注意力头）

- 将注意力投影矩阵按**头**划分：每个头占 `head_dim = H / num_heads`。  
- 剪枝等价于对某些头的权重置零/删除，减少计算：
  $$
  Q = XW_Q,\; K = XW_K,\; V = XW_V\quad \Rightarrow\quad
  Q^{(head)}_{pruned} = 0
  $$
- **影响**：延迟/吞吐提升；精度可能下降，通常需要**微调恢复**。

### 知识蒸馏

- 让学生去逼近教师的**输出分布/隐藏表示**：
  $$
  \mathcal{L}_{\text{KD}} = \| f_s(X) - f_t(X) \|_2^2
  $$
- 真实 KD 常用**软目标**（`softmax(z/T)`）与 CE/KL 损失；可叠加**硬标签**监督。

### INT8 量化

- 将实数张量映射到 \([-128,127]\) 的**离散整值**并记录 `scale`：
  $$
  \hat{x} = \mathrm{clip}(\mathrm{round}(x / s), q_{\min}, q_{\max})\cdot s
  $$
- 量化误差越小，说明该张量的动态范围与量化方案更匹配（可通过**分组/对称/非对称**方案改进）。

---

## 六、运行示例与典型输出

```bash
$ python compression_tutorial.py
[INFO] 结构化剪枝：保留 4/8 个注意力头
[INFO] 蒸馏损失：0.1234
[INFO] INT8 量化误差：12.3456
{'distill_loss': 0.1234}
```
> 实际数字因随机权重而异。你可多次运行观察波动。

---

## 七、常见问题排查（FAQ）

1. **蒸馏 loss 波动大/不下降？**  
   - 增大 batch 或步数；降低学习率；把学生结构稍微加宽；固定随机种子以提高可重复性。

2. **剪枝后模型发散？**  
   - 目前只是**掩码置零**，真实工程应**重建矩阵/通道**并做**微调恢复**；必要时只剪掉**贡献小**的头。

3. **量化误差太大？**  
   - 尝试**分组量化**或**统计更鲁棒的 scale**（如基于分位数而非 max）。

4. **如何迁移到真实模型？**  
   - 使用 `transformers` 的 `LlamaAttention`/`QwenAttention` 等模块做**head 剪枝**；  
   - 蒸馏使用真实语料与**教师 logits**；  
   - 量化采用 `bitsandbytes` 或 `torch.int8` 动态量化 API。

---

## 八、扩展建议

- **剪枝敏感度评估**：基于梯度/重要性评分（如 L0/L1、Taylor 展开）选择要剪的头；  
- **蒸馏损失组合**：隐藏层/注意力图/输出 logits 多路蒸馏；  
- **训练后量化 (PTQ) vs 量化感知训练 (QAT)**：比较两者对精度的影响；  
- **端到端基准**：补充速度与显存对比曲线，形成工程可决策的 trade-off。

---

## 九、许可证

本文档与脚本仅用于教学示例。请遵循你项目的总体许可证与上游依赖（PyTorch、Transformers 等）的许可证条款。
