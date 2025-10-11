# QLoRA 技术详解脚本：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`qlora_explainer.py`（基于你提供的教学代码整理）  
> 运行环境：Python ≥ 3.9；`torch`（CPU 可运行，演示型脚本）  
> 目标：以**可复现的数值演示**帮助理解 QLoRA 的三大关键组件：**NF4 量化**、**分页优化器（paged optimizers）**、**梯度检查点（gradient checkpointing）** 对显存与计算的影响。

---

## 目录
- [QLoRA 技术详解脚本：使用说明与函数文档](#qlora-技术详解脚本使用说明与函数文档)
  - [目录](#目录)
  - [一、脚本概览](#一脚本概览)
  - [二、快速开始](#二快速开始)
    - [1) 安装依赖](#1-安装依赖)
    - [2) 运行脚本](#2-运行脚本)
  - [三、核心类与函数文档](#三核心类与函数文档)
    - [`QLoRAStats`](#qlorastats)
    - [`nf4_quantize(tensor: torch.Tensor) -> torch.Tensor`](#nf4_quantizetensor-torchtensor---torchtensor)
    - [`estimate_paged_memory(stats: QLoRAStats) -> Dict[str, float]`](#estimate_paged_memorystats-qlorastats---dictstr-float)
    - [`gradient_checkpointing_saving(stats: QLoRAStats) -> float`](#gradient_checkpointing_savingstats-qlorastats---float)
  - [四、数学与工程直觉](#四数学与工程直觉)
    - [NF4（Normalized Float 4-bit）量化直观说明](#nf4normalized-float-4-bit量化直观说明)
    - [分页优化器（Paged Optimizers）为何省显存](#分页优化器paged-optimizers为何省显存)
    - [梯度检查点对激活显存的影响](#梯度检查点对激活显存的影响)
  - [五、运行示例与输出](#五运行示例与输出)
  - [六、可选扩展与改进方向](#六可选扩展与改进方向)
  - [七、常见问题排查（FAQ）](#七常见问题排查faq)
  - [八、许可证](#八许可证)

---

## 一、脚本概览

该教学脚本**不依赖 GPU**，通过随机张量与参数规模估算，演示 QLoRA 的三大组件：

1. **NF4 量化**：将 `float16` 张量映射到 **4-bit** 的 16 个离散值（本脚本为**简化版模拟**，便于直观理解缩放与离散化过程）；  
2. **分页优化器**：给出 FP16 与 QLoRA 场景下**参数与优化器状态**的显存估算；  
3. **梯度检查点**：对激活显存按层估算，展示 checkpointing 的节省量级。

> **注意**：本脚本旨在教学可视化与数量级感知，未严格复刻生产实现（如 NF4 的码本、分组尺度、自适应比例、bitsandbytes 的具体分页策略等）。

---

## 二、快速开始

### 1) 安装依赖
```bash
pip install torch
```

### 2) 运行脚本
```bash
python lesson2/01_qlora_intro/qlora_theory.py
```
你将看到日志输出：NF4 的 scale、FP16 vs QLoRA 的显存估算、以及梯度检查点带来的激活显存节省。

---

## 三、核心类与函数文档

### `QLoRAStats`

```python
@dataclass
class QLoRAStats:
    hidden_size: int = 4096
    num_layers: int = 32
    seq_length: int = 1024
    vocab_size: int = 32000

    def parameter_count(self) -> int:
        return 12 * self.hidden_size * self.hidden_size * self.num_layers
```
**作用**：收纳用于**粗略参数规模**与**显存估算**的超参数。

- **hidden_size**：Transformer 模型的隐藏维度 \(H\)。  
- **num_layers**：Transformer 层数 \(L\)。  
- **seq_length**：训练/推理时上下文长度 \(T\)（用于激活规模估算）。  
- **vocab_size**：词表大小（本脚本未用到，可扩展词嵌入/LM head 的估算）。  
- **`parameter_count()`**：返回**近似**参数规模：  
  $$
  \text{params} \approx 12 \cdot H^2 \cdot L
  $$
  用于 MHA + MLP 的数量级估算（忽略常数与偏置/LayerNorm/Embeddings 等），便于感知**大模型的参数随 \(H^2 L\)** 增长。

---

### `nf4_quantize(tensor: torch.Tensor) -> torch.Tensor`

**作用**：**简化模拟** NF4 量化，将输入张量缩放到离散值范围并反量化回 `float` 以便直观比较。

**实现**：
```python
qmin, qmax = -8, 7
scale = tensor.abs().max() / qmax
quantized = torch.clamp((tensor / scale).round(), qmin, qmax)
return quantized * scale
```
- 先用 **最大绝对值** 估计 `scale`；  
- 将张量除以 `scale` 后**取整并截断**到 \([-8, 7]\)；  
- 再乘回 `scale` 得到反量化张量。

**日志**：打印 `scale` 与离散范围。

> ⚠️ **与真实 NF4 的差异**：  
> - 真正的 NF4 是**归一化浮点**码本（非线性间隔），通常配合**分组量化（group-wise）**与更细的动态范围估计；  
> - 本实现仅为**线性缩放 + 4bit 整数量化**的教学近似，便于理解“缩放—离散—反量化”的流程。

---

### `estimate_paged_memory(stats: QLoRAStats) -> Dict[str, float]`

**作用**：估算**分页优化器**下的显存占用对比。

**实现要点**：
```python
param_bytes_fp16 = stats.parameter_count() * 2     # FP16 参数存储
param_bytes_int4 = stats.parameter_count() // 2    # INT4 参数存储（4bit = 0.5 byte）
optimizer_state = param_bytes_int4 * 2             # 假设动量+二阶矩各占 1x 参数大小（示意）
total = param_bytes_int4 + optimizer_state
```
返回：
```python
{"fp16_gb": ..., "qlora_gb": ...}
```
> 这反映了 QLoRA 中**参数为 4bit 常驻显存**，优化器状态按分页/惰性策略**仅在需要时驻留**，从而降低常驻占用。此处为教学估算，真实实现依赖 bitsandbytes/优化器细节与分布式策略。

---

### `gradient_checkpointing_saving(stats: QLoRAStats) -> float`

**作用**：估算**梯度检查点**对**激活显存**的节省。

**实现要点**：
```python
activations = stats.hidden_size * stats.seq_length * stats.num_layers * 2
saved = activations * 0.5
return saved / 1024 ** 3  # 转 GB
```
- 用 \(2 \times H \times T \times L\) 作为**数量级**估算激活元素数（双向/多张量近似）；  
- 假设 checkpointing 能节省约 **50%** 的激活占用（教学近似）；  
- 返回 GB 单位的节省量。

> 真实节省比例依赖**实现方式**（逐层 checkpoint、选择性保存、张量重计算策略等）、**精度**（fp16/bf16）、以及**批大小**、**序列长度**、**并行策略**。

---

## 四、数学与工程直觉

### NF4（Normalized Float 4-bit）量化直观说明

- **目标**：在几乎不损失精度的前提下，把 **权重/激活** 从 16-bit 或 8-bit **压缩到 4-bit**，显著降低显存带宽与容量占用；  
- **核心**：对每个小分组（group）或张量片段进行**归一化**，将其值域映射到**更合理的 4-bit 浮点码本**；  
- **益处**：较线性整数量化更好地适应长尾分布，减小量化误差；  
- **本脚本**：以线性缩放的“4-bit 整数近似”替代 NF4 码本，仅用于**演示缩放/离散/反量化**的流程与数量级。

### 分页优化器（Paged Optimizers）为何省显存

- **背景**：标准优化器（Adam 等）会为每个参数维护**一阶/二阶动量**，其显存占用往往接近或超过**参数本身**；  
- **Paged 思想**：当参数以 **4-bit 常驻**时，**优化器状态**不必**全量常驻**显存，可按需在**活跃页**上加载/计算/回写，从而降低**训练期间的峰值显存**；  
- **直觉**：把优化器状态视为“缓存在页里的临时工作区”，避免把所有状态都长期塞在显存里。

### 梯度检查点对激活显存的影响

- **常规训练**：前向时保留每层激活，反向直接使用，**显存占用与层数、序列长正相关**；  
- **Checkpointing**：只保存部分关键中间结果，在反向时**重计算**未保存的激活，**以计算换显存**；  
- **量级**：根据策略不同，可节省 30%~80% 的激活显存。脚本用 50% 作为**教学近似**。

---

## 五、运行示例与输出

入口（已在脚本底部给出）：
```python
if __name__ == "__main__":
    stats = QLoRAStats()
    dummy = torch.randn(1024)
    _ = nf4_quantize(dummy)
    estimate_paged_memory(stats)
    gradient_checkpointing_saving(stats)
```
**典型日志（示例）**：
```
[NF4] scale=0.12345, range=[-8, 7]
[分页优化器] FP16=XX.XXGB, QLoRA=YY.YYGB
[Checkpoint] 预计节省激活显存 ZZ.ZZ GB
```
你可以修改 `QLoRAStats(hidden_size, num_layers, seq_length)` 观察不同规模下的变化。

---

## 六、可选扩展与改进方向

1. **更真实的 NF4**：
   - 引入**码本查找**与**分组量化**；
   - 比较线性 4-bit 与 NF4 在 MSE/MAE 上的误差差异。

2. **更细的显存账本**：
   - 拆分参数、梯度、优化器状态（m/v）、临时缓冲区的单独占用；
   - 考虑 `optimizer_state_sharding`、ZeRO/DeepSpeed 的分布式策略。

3. **激活显存细分**：
   - 分解注意力、MLP、残差、归一化等子模块的激活规模；
   - 对比不同 checkpoint 粒度（每层/块）下的节省比例。

4. **可视化**：
   - 画出 FP16/INT8/INT4 的显存对比柱状图；
   - 绘制不同 \(H,L,T\) 下的等高线图以展示规模效应。

---

## 七、常见问题排查（FAQ）

1. **为什么量化后还是 `float` 张量？**  
   - 脚本中演示的是**量化-反量化（dequantize）**后的张量，便于比较与后续计算；真正的 4-bit 存储会以**紧凑比特表示**，需要自定义内核或专用库（如 bitsandbytes）。

2. **分页优化器的数字看起来太理想/粗糙？**  
   - 是的。本脚本是**教学估算**。真实数值因实现与硬件而异，尤其是**数据并行/张量并行**、**混合精度**、**参数切分**等策略。

3. **梯度检查点节省比例固定 50% 合理吗？**  
   - 仅为**演示默认**。推荐把该比例设为配置项，或在实际模型中统计峰值显存。

---

## 八、许可证

该文档与脚本用于教学演示。请遵循你项目的整体许可证以及上游依赖（PyTorch、Transformers、bitsandbytes 等）的许可证条款。
