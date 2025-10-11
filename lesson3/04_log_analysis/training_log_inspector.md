# 训练日志分析与性能调优示例：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson3/04_log_analysis/training_log_inspector.py`（本文档基于你提供的教学代码撰写）  
> 运行环境：Python ≥ 3.9；`pandas`、`matplotlib`（CPU 即可）  
> 目标：演示如何解析 DeepSpeed / Accelerate 等训练框架输出的 **JSONL 日志**，提取**每步损失**、**吞吐（samples/s）**、并**可视化趋势**。文末附带**可扩展的 GPU 利用率估计思路**。

---

## 目录
- [训练日志分析与性能调优示例：使用说明与函数文档](#训练日志分析与性能调优示例使用说明与函数文档)
  - [目录](#目录)
  - [一、脚本概览](#一脚本概览)
  - [二、快速开始](#二快速开始)
    - [依赖安装](#依赖安装)
    - [运行脚本](#运行脚本)
    - [期望输出](#期望输出)
  - [三、函数文档](#三函数文档)
    - [`load_logs(path: Path) -> pd.DataFrame`](#load_logspath-path---pddataframe)
    - [`plot_metrics(df: pd.DataFrame) -> None`](#plot_metricsdf-pddataframe---none)
  - [四、日志字段与数据格式](#四日志字段与数据格式)
  - [五、可视化细节与定制](#五可视化细节与定制)
  - [六、GPU 利用率估计：实现思路（扩展）](#六gpu-利用率估计实现思路扩展)
    - [方案 A：日志侧采样（推荐）](#方案-a日志侧采样推荐)
    - [方案 B：基于吞吐与步耗时的**间接估计**](#方案-b基于吞吐与步耗时的间接估计)
  - [七、性能调优小抄](#七性能调优小抄)
  - [八、常见问题（FAQ）](#八常见问题faq)
  - [九、许可证](#九许可证)

---

## 一、脚本概览

该教学脚本包含两个核心步骤：

1. **读取日志**：从 **JSONL** 文本逐行解析为 Python 字典，并汇总为 `pandas.DataFrame`；  
2. **绘制曲线**：绘制**训练损失**与**吞吐（samples/s）** 两张趋势图，保存为 `training_metrics.png`。

> 说明：代码主体未直接计算 GPU 利用率；你可以在生成日志阶段引入 `gpu_util` 字段，或依据**吞吐/步耗时**做间接估计，详见第六节。

---

## 二、快速开始

### 依赖安装
```bash
pip install pandas matplotlib
```

### 运行脚本
将代码保存为 `lesson3/04_log_analysis/training_log_inspector.py`，执行：
```bash
python lesson3/04_log_analysis/training_log_inspector.py
```
脚本会创建一个演示日志 `sample_log.jsonl`，随后读取并绘制图表。

### 期望输出
- 终端打印：`图表已保存为 training_metrics.png`  
- 生成文件：
  - `sample_log.jsonl`（50 行演示数据）  
  - `training_metrics.png`（损失 & 吞吐趋势）

---

## 三、函数文档

### `load_logs(path: Path) -> pd.DataFrame`

**作用**：读取并解析 JSONL 日志为 `DataFrame`。

**实现要点**：
```python
records = []
for line in path.read_text(encoding="utf-8").splitlines():
    try:
        records.append(json.loads(line))
    except json.JSONDecodeError:
        continue
return pd.DataFrame(records)
```
- **容错**：对无法解析的行**跳过**（方便处理混杂日志）；  
- **返回**：包含日志字段（如 `loss`、`throughput`）的 `DataFrame`。

**输入参数**：
- `path`：日志文件路径，建议为 **JSONL**（一行一个 JSON 对象）。

**返回值**：
- `pd.DataFrame`：每行对应一步/一次日志记录。

---

### `plot_metrics(df: pd.DataFrame) -> None`

**作用**：将损失与吞吐绘制为两行子图，并保存到 `training_metrics.png`。

**实现要点**：
```python
if df.empty: print("日志为空"); return
df["step"] = range(1, len(df) + 1)
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes[0].plot(df["step"], df["loss"], label="loss")
axes[1].plot(df["step"], df["throughput"], color="orange", label="throughput")
plt.tight_layout(); plt.savefig("training_metrics.png")
```
- 自动生成 `step` 序号作为横轴；  
- 若缺少 `throughput` 列，只绘制**损失**子图（脚本当前只在有 `throughput` 时绘制第二张图）。

**输入参数**：
- `df`：来自 `load_logs` 的 `DataFrame`。

**副作用**：
- 在当前工作目录保存 **`training_metrics.png`**。

---

## 四、日志字段与数据格式

**推荐最小字段集合**（每步一行，JSONL）：
```json
{"loss": 2.345, "throughput": 512.3, "step_time_s": 0.124, "lr": 1.2e-5}
```
- `loss`：训练损失（标量）；  
- `throughput`：每秒样本数（samples/s）；  
- `step_time_s`：单步耗时（秒），可用于交叉校验吞吐；  
- `lr`：学习率（可选，用于排查训练调度）。

**DeepSpeed / Accelerate** 的日志常见位置：
- DeepSpeed：自定义 `logger`/`TrainingArguments` 或 `ds_report.json` / 推断日志；  
- Accelerate：`--logging_dir` 下的事件/指标。**建议**在训练脚本中显式 `print(json.dumps(...))` 生成一份**干净的 JSONL**。

---

## 五、可视化细节与定制

- **移动平均**：平滑抖动（示例）：
  ```python
  df["loss_ma"] = df["loss"].rolling(window=20, min_periods=1).mean()
  axes[0].plot(df["step"], df["loss_ma"], label="loss_ma", linestyle="--")
  ```
- **双轴绘图**：在同一子图中叠加 `loss` 与 `lr`（第二 y 轴）；  
- **异常标记**：对 `throughput` 的异常点（低于 P10）做红点标注，快速发现**数据加载瓶颈**或**通信抖动**。

---

## 六、GPU 利用率估计：实现思路（扩展）

> 代码主体未直接实现 GPU 利用率估计，这里给出两种可落地方案：

### 方案 A：日志侧采样（推荐）
在训练脚本中定期采样 `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`：
```python
import subprocess, time, json
util = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
).decode().strip().splitlines()
gpu_util = list(map(int, util))  # 多卡时为多行
print(json.dumps({"loss": loss, "throughput": th, "gpu_util": gpu_util}))
```
随后在本分析脚本中：
```python
if "gpu_util" in df.columns:
    # 展示均值或各卡箱线图
    df["gpu_util_mean"] = df["gpu_util"].apply(lambda xs: sum(xs)/len(xs) if isinstance(xs, list) else xs)
```

### 方案 B：基于吞吐与步耗时的**间接估计**
- 理想情况下 `throughput ≈ global_batch / step_time_s`；若实际吞吐显著低于**计算上限**（由 FLOPs/带宽估算），说明 GPU 可能**未被充分利用**；  
- 可引入**数据加载时间占比**（通过 `torch.profiler` 或自定义计时）来定位瓶颈。

> 对课堂演示，建议直接采用**方案 A**，简单、稳妥。

---

## 七、性能调优小抄

- **数据侧**：
  - 增大 `num_workers` / `prefetch_factor`；确保 **pin_memory=True**；  
  - 采用 **packing** 或 **大块样本**，减少 Python 端开销。

- **计算侧**：
  - 打开 **AMP（bf16/fp16）**、**FlashAttention**、**cudnn.benchmark**；  
  - 合理使用 **gradient_accumulation** 提升有效 batch。

- **通信侧（多卡）**：
  - 检查 **DDP Bucket** 大小、启用 **NCCL P2P**、减少梯度同步频率（梯度累积）；  
  - DeepSpeed ZeRO‑3 时关注 **offload** 对 PCIe 带宽的影响。

- **系统侧**：
  - 固定 **CPU/GPU 频点政策**（服务器默认通常即可）；  
  - 监控 **I/O** 与 **网络** 状况（尤其远程数据集）。

---

## 八、常见问题（FAQ）

1. **日志不是 JSONL，怎么用？**  
   - 可在训练阶段同步输出一份**结构化 JSONL**；或在本脚本中加入**正则提取**后再组装为字典。

2. **图片空白/只有一张图？**  
   - 若 DataFrame 为空或缺少 `throughput` 列，脚本会只画损失；请检查字段名称与大小写。

3. **如何把图表显示到屏幕？**  
   - 在 `plt.savefig(...)` 前添加 `plt.show()`（需 GUI 支持），或在 Jupyter 中直接显示。

4. **能画更多指标吗？**  
   - 当然。把 `lr/step_time_s/gpu_util_mean/memory_gb` 等列添加到 DataFrame，再拓展绘图函数即可。

---

## 九、许可证

本文档与脚本用于教学演示；请遵循你项目的总体许可证与上游依赖（Pandas、Matplotlib 等）的许可证条款。
