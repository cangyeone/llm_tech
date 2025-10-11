# 训练加速技巧演示：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson3/03_training_acceleration/acceleration_tricks.py`（本文档基于你提供的教学代码撰写）  
> 运行环境：Python ≥ 3.9；PyTorch ≥ 2.1（建议 2.2+）  
> 硬件支持：CPU 可跑演示；**GPU（CUDA）推荐**以实际感知 FlashAttention 与 `torch.compile` 的加速效果。

---

## 目录
- [训练加速技巧演示：使用说明与函数文档](#训练加速技巧演示使用说明与函数文档)
  - [目录](#目录)
  - [一、脚本概览](#一脚本概览)
  - [二、快速开始](#二快速开始)
    - [依赖安装](#依赖安装)
    - [运行脚本](#运行脚本)
  - [三、核心函数文档](#三核心函数文档)
    - [`flash_attention_demo()`](#flash_attention_demo)
    - [`compile_speedup()`](#compile_speedup)
    - [`print_memory_tips()`](#print_memory_tips)
  - [四、FlashAttention 基本原理与开关](#四flashattention-基本原理与开关)
  - [五、`torch.compile` 加速要点](#五torchcompile-加速要点)
  - [六、基准测试建议（如何测得更稳）](#六基准测试建议如何测得更稳)
  - [七、常见问题排查（FAQ）](#七常见问题排查faq)
  - [八、扩展练习](#八扩展练习)
  - [九、许可证](#九许可证)

---

## 一、脚本概览

该教学脚本包含三部分：

1. **FlashAttention API 演示**：调用 `torch.nn.functional.scaled_dot_product_attention`（SDPA）测量一次前向的耗时；  
2. **`torch.compile` 演示**：用一个三层 MLP 对比编译前后前向吞吐；  
3. **内存优化建议**：打印训练时常见的显存优化选项（梯度检查点、低精度、ZeRO/并行策略）。

> 注：脚本使用**随机张量**并非真实模型，只为教学展示**调用方式与量级直觉**。要获得更贴近实战的数据，请替换为你任务中的模型与 batch/seq_len。

---

## 二、快速开始

### 依赖安装
```bash
pip install "torch>=2.2"  # 按你的 CUDA 版本到 PyTorch 官网选择正确指令
```

### 运行脚本
```bash
python lesson3/03_training_acceleration/acceleration_tricks.py
```
期望输出（示例，数值因环境不同而变化）：
```
[INFO] FlashAttention API 调用耗时：0.0021 s
[INFO] torch.compile 平均耗时：0.0008 s
内存优化建议：
1. 启用梯度检查点减少激活显存。
2. 使用 bf16/FP8 等低精度训练降低显存需求。
3. 合理设置 ZeRO 分片、张量并行。
```

---

## 三、核心函数文档

### `flash_attention_demo()`
**作用**：演示并测量 `scaled_dot_product_attention` 的一次前向耗时。

**代码要点**：
```python
q = torch.randn(4, 8, 128, 64)
k = torch.randn(4, 8, 128, 64)
v = torch.randn(4, 8, 128, 64)

torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```
- 张量形状 `[B, H, T, Dh] = [4, 8, 128, 64]`，代表 **4** 个样本、**8** 头、序列长 **128**、每头维度 **64**；  
- 以 `is_causal=True` 启用因果掩码（自回归）。

**注意事项**：
- **是否真的启用 FlashAttention** 取决于**设备/精度/后端开关**。通常需要：CUDA + 合适的数据类型（`float16/bfloat16`）+ 后端允许；CPU 上会退化为 math/mem‑efficient 后端，速度提升不明显。  
- 可通过上下文切换不同后端：
  ```python
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
      ...  # 强制 FlashAttention
  with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
      ...  # 仅 math，便于做对比
  ```

---

### `compile_speedup()`
**作用**：演示 `torch.compile` 对简单前向的平均耗时提升。

**代码要点**：
```python
model = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.GELU(),
    torch.nn.Linear(1024, 512),
)
compiled = torch.compile(model)  # 默认后端 torch.compile(..., backend="inductor")
inputs = torch.randn(32, 512)
# 预热一次
compiled(inputs)
# 计时 20 次，取平均
```
**提示**：
- 首次调用包含**编译开销**，因此脚本先做**预热**；  
- 对更复杂的模型，`torch.compile` 往往更能体现收益（融合算子、减少框架开销）。

---

### `print_memory_tips()`
**作用**：打印常见的**显存优化**建议，便于课堂快速回顾：
1) 梯度检查点（以**计算换显存**）；  
2) 低精度训练（`bf16/FP8` 等，硬件/库需支持）；  
3) ZeRO 分片、张量/流水并行等**分布式**手段。

---

## 四、FlashAttention 基本原理与开关

- **原理**（简述）：将注意力的 `QK^T` 与 `softmax`/`V` 乘法**块化**到共享内存与寄存器，**避免显式形成 T×T 的注意力矩阵**，从而降低显存与访存、提升吞吐。  
- **PyTorch SDPA** 会根据环境选择后端：`flash`、`mem_efficient`、`math`。  
- **如何确认后端？**  
  - 在 PyTorch 2.2+，可设置/打印 `torch.backends.cuda.sdp_kernel.is_flash_enabled()` 等；  
  - 或在不同上下文中手动对比耗时（见第 3 节的上下文字段）。
- **常见触发条件**：
  - 设备：CUDA；精度：`float16`/`bfloat16` 更可能走 `flash`；  
  - 张量形状满足 kernel 要求（head_dim、对齐等）。

---

## 五、`torch.compile` 加速要点

- **默认后端**：`backend="inductor"`，对常见张量算子和 Transformer 模块有较好覆盖；  
- **收益场景**：前向/反向包含大量 Python 框架开销、可融合的逐元素/点积算子时；  
- **冷启动**：首次编译会慢，**训练/长推理**场景收益更明显；  
- **可调参数**（进阶）：
  ```python
  torch.compile(model, mode="max-autotune")  # 更激进的调优（新版本支持）
  # 或 backend="aot_eager"/"eager" 做兼容性回退
  ```
- **与 DDP/AMP**：一般兼容，可在 DDP/AMP 环境使用（仍建议先小规模验证正确性）。

---

## 六、基准测试建议（如何测得更稳）

1. **固定随机种子与设备**：`torch.manual_seed(0)`；将张量放到 `cuda()` 并使用 `half()/bfloat16()`；  
2. **CUDA 同步计时**：在测量段前后加入 `torch.cuda.synchronize()`：
   ```python
   torch.cuda.synchronize(); t0 = time.time()
   ...  # 前向
   torch.cuda.synchronize(); t1 = time.time()
   ```
3. **多次循环取中位数/均值**；排除冷启动编译时间；  
4. **扩大问题规模**：更长 `T`/更大 `B`/更多头数能更清晰地体现差异；  
5. **对照组**：使用 `sdp_kernel` 上下文强制 `math` vs `flash`。

---

## 七、常见问题排查（FAQ）

1. **“我看不到 FlashAttention 的提升？”**  
   - 可能跑在 CPU 或后端未触发 `flash`；请确保 CUDA 可用、使用半精度、并检查 `sdp_kernel` 设置。小规模（`T=128`）差异也可能不明显。

2. **`torch.compile` 首次很慢？**  
   - 正常。首次包含编译与缓存构建。用于**持续训练/长推理**时才能摊薄编译成本。

3. **`torch.compile` 报图捕获/不支持算子？**  
   - 回退到 `backend="aot_eager"` 或设置 `fullgraph=False`；逐步缩小模型定位不支持的子图。

4. **如何在 Transformer Block 上做真实测试？**  
   - 用 `transformers` 的 `LlamaAttention`/`QwenAttention` 或自实现 Block；对比启用/禁用 KV cache、FlashAttention 的吞吐与显存。

5. **为什么脚本只测前向？**  
   - 教学目的简单可复现。训练场景应同时测**前向+反向**并记录步耗时、吞吐与显存峰值。

---

## 八、扩展练习

- **练习 A**：把 `q/k/v` 放到 `cuda()`，并加上 `with sdp_kernel(...):` 对比 **flash vs math** 的耗时。  
- **练习 B**：把 `compile_speedup` 改为**前向+反向**（加损失与 `backward()`），比较 `torch.compile` 的训练加速。  
- **练习 C**：将 MLP 换为 **Transformer Block**，测不同 `seq_len` 与 `batch` 下的吞吐曲线。  
- **练习 D**：将“内存优化建议”补充为**可执行清单**（示例超参数、ZeRO 配置、KV cache 开关等）。

---

## 九、许可证

本文档与脚本用于教学演示；请遵循你项目的总体许可证与上游依赖（PyTorch 等）的许可证条款。
