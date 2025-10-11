# ZeRO‑3 与混合精度配置生成器：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson3/01_distributed_training/zero3_configurator.py`（基于你提供的教学代码整理）  
> 运行环境：Python ≥ 3.9（本脚本仅生成 JSON，不依赖 GPU）  
> 目标：根据 **micro batch** 与 **梯度累积** 等输入，自动生成 **DeepSpeed ZeRO‑3** 与 **混合精度（bf16/fp16）** 的配置文件。可用于课堂演示如何将**显存预算**映射到 **train_batch_size** 与 **ZeRO 选项**。

---

## 目录
- [ZeRO‑3 与混合精度配置生成器：使用说明与函数文档](#zero3-与混合精度配置生成器使用说明与函数文档)
  - [目录](#目录)
  - [一、功能概览](#一功能概览)
  - [二、快速开始](#二快速开始)
    - [运行脚本](#运行脚本)
    - [在训练脚本中使用](#在训练脚本中使用)
  - [三、核心数据类与函数](#三核心数据类与函数)
    - [`ZeroConfig`](#zeroconfig)
    - [`build_config(cfg: ZeroConfig) -> dict`](#build_configcfg-zeroconfig---dict)
    - [`save_config(config: dict, path: Path) -> None`](#save_configconfig-dict-path-path---none)
  - [四、生成的 DeepSpeed 配置字段说明](#四生成的-deepspeed-配置字段说明)
  - [五、命令行与集成示例](#五命令行与集成示例)
    - [1) 与 HuggingFace `Trainer` 集成](#1-与-huggingface-trainer-集成)
    - [2) 自写训练循环（PyTorch）](#2-自写训练循环pytorch)
    - [3) 开启 CPU Offload](#3-开启-cpu-offload)
  - [六、典型配方与显存直觉](#六典型配方与显存直觉)
  - [七、扩展建议](#七扩展建议)
  - [八、常见问题（FAQ）](#八常见问题faq)
  - [九、许可证](#九许可证)

---

## 一、功能概览

本教学脚本根据输入的 **micro_batch_size** 与 **gradient_accumulation**，生成一份可直接用于 DeepSpeed 的 **ZeRO‑3** 配置：

- **ZeRO Stage 3 参数分片**：跨进程分片模型参数/梯度/优化器状态；  
- **混合精度**：通过 `bf16.enabled` 开关控制（亦可扩展出 `fp16` 字段）；  
- **（可选）参数离线/CPU Offload**：将参数迁移至 CPU，降低 GPU 常驻；  
- **通信与内存优化**：`overlap_comm`、`contiguous_gradients` 等安全默认项。

---

## 二、快速开始

### 运行脚本
将代码保存为 `lesson3/01_distributed_training/zero3_configurator.py`：
```bash
python lesson3/01_distributed_training/zero3_configurator.py
```
默认会生成 `./ds_zero3.json` 并打印：
```
DeepSpeed 配置已保存至 ds_zero3.json
```

### 在训练脚本中使用
```bash
deepspeed --num_gpus=4 train.py --deepspeed ds_zero3.json
```
> `train_batch_size` = `micro_batch_size × gradient_accumulation`（总是**全局**训练 batch，不含数据并行世界大小；若你的 `train.py` 内部还会乘以 DP 世界大小，请确保语义一致，避免重复放大）。

---

## 三、核心数据类与函数

### `ZeroConfig`
```python
@dataclass
class ZeroConfig:
    micro_batch_size: int
    gradient_accumulation: int
    stage: int = 3
    offload: bool = False
    bf16: bool = True
    output_path: Path = Path("./ds_zero3.json")
```
- **micro_batch_size**：每个设备上的**单次前向**批大小；  
- **gradient_accumulation**：梯度累积步数（等效扩大有效 batch）；  
- **stage**：ZeRO 阶段（默认 3）；  
- **offload**：是否开启**参数 CPU 离线**；  
- **bf16**：是否启用 `bf16.enabled`；（若你的 GPU 不支持 bf16，可设置为 `False` 并在扩展中加上 `fp16` 字段）；  
- **output_path**：JSON 输出路径。

### `build_config(cfg: ZeroConfig) -> dict`
根据 `ZeroConfig` 组装 DeepSpeed 配置字典：
```python
return {
  "train_batch_size": cfg.micro_batch_size * cfg.gradient_accumulation,
  "zero_optimization": {
    "stage": cfg.stage,
    "offload_param": {"device": "cpu" if cfg.offload else "none", "pin_memory": cfg.offload},
    "overlap_comm": True,
    "contiguous_gradients": True,
  },
  "bf16": {"enabled": cfg.bf16},
  "gradient_accumulation_steps": cfg.gradient_accumulation,
  "steps_per_print": 50,
  "gradient_clipping": 1.0,
}
```
**返回**：DeepSpeed 配置 dict。

### `save_config(config: dict, path: Path) -> None`
将配置字典以 `indent=2` 写入 JSON 文件。

---

## 四、生成的 DeepSpeed 配置字段说明

| 字段 | 含义 | 备注 |
|---|---|---|
| `train_batch_size` | 全局训练批大小（不含 DP world size 时，请与训练脚本保持一致） | 由 `micro_batch_size × gradient_accumulation` 计算 |
| `zero_optimization.stage` | ZeRO 阶段 | `3` 表示参数/优化器状态/梯度均分片 |
| `zero_optimization.offload_param.device` | 参数离线设备 | `"cpu"` 或 `"none"` |
| `zero_optimization.offload_param.pin_memory` | 是否锁页内存 | `True` 时提升 CPU↔GPU 传输效率 |
| `zero_optimization.overlap_comm` | 通信与计算重叠 | 典型优化开关 |
| `zero_optimization.contiguous_gradients` | 连续梯度内存 | 减少碎片化 |
| `bf16.enabled` | 是否启用 bf16 | 若硬件不支持，可设为 `False` 并考虑 `fp16` |
| `gradient_accumulation_steps` | 梯度累积步数 | 与 `train_batch_size` 一致 |
| `steps_per_print` | 日志间隔 | 50 步打印一次 |
| `gradient_clipping` | 梯度裁剪阈值 | 默认 1.0 |

> **关于梯度检查点（checkpointing）**：本配置生成器未直接包含该字段。实践中通常在**训练脚本**或**模型**侧开启（例如 HF `TrainingArguments(gradient_checkpointing=True)` 或手动 `model.gradient_checkpointing_enable()`），与 ZeRO‑3 **兼容**。

---

## 五、命令行与集成示例

### 1) 与 HuggingFace `Trainer` 集成
```bash
deepspeed --num_gpus=8 train_hf.py \
  --deepspeed ds_zero3.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 True \
  --gradient_checkpointing True
```
- `Trainer` 的 `gradient_accumulation_steps` 应与 JSON 中一致，或让其中一个作为**单一事实来源**。

### 2) 自写训练循环（PyTorch）
```python
import json, deepspeed
from torch.utils.data import DataLoader

cfg = json.load(open("ds_zero3.json"))
engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=cfg)
for step, batch in enumerate(DataLoader(dataset, batch_size=cfg["train_batch_size"])):
    loss = model(**batch)
    engine.backward(loss)
    engine.step()
```

### 3) 开启 CPU Offload
```python
config = ZeroConfig(micro_batch_size=2, gradient_accumulation=32, offload=True)
ds_cfg = build_config(config); save_config(ds_cfg, config.output_path)
```
适合显存特别紧张的场景（速度会受 PCIe 影响）。

---

## 六、典型配方与显存直觉

- **小显存单卡**：`micro_batch_size=1, grad_acc=32, stage=3, offload=True, bf16=True`  
  - 通过 ZeRO‑3 分片 + CPU offload + 梯度累积撑起有效 batch。

- **多卡 8×A100**：`micro_batch_size=2, grad_acc=8, stage=3, offload=False, bf16=True`  
  - ZeRO‑3 主要为**跨进程**节省优化器/梯度；少用 offload 以提升速度。

- **长序列训练**：适度提高 `grad_acc` 并在训练脚本中开启 **gradient checkpointing**，以**计算换显存**。

> **经验法则**：优先 `bf16`（数值稳定/吞吐更好）；若不支持再考虑 `fp16` 与 `loss_scale`。

---

## 七、扩展建议

1. **支持 `fp16` 配置**：在生成器里加 `"fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 12}`。  
2. **加入 `zero3` 细粒度项**：如 `stage3_param_persistence_threshold`、`stage3_max_live_parameters`、`stage3_prefetch_bucket_size` 等。  
3. **自动探测 GPU 显存**：基于 `torch.cuda.mem_get_info()` 粗估可容纳的 `micro_batch_size` 与是否需要 `offload`。  
4. **一键多档输出**：同时生成多份 JSON（小/中/大三档），便于快速 A/B。  
5. **校验一致性**：对 `train.py` 的 CLI/环境做一致性检查，防止 `train_batch_size` 被重复放大。

---

## 八、常见问题（FAQ）

1. **`bf16` 不生效或数值异常**  
   - 确认 GPU 支持 bf16（如 A100/H100）；否则设 `bf16=False` 并使用 `fp16`。

2. **显存仍不足**  
   - 提高 `gradient_accumulation`；在训练脚本开启 **gradient checkpointing**；必要时启用 `offload=True`。

3. **吞吐下降明显**  
   - 关闭 `offload`；检查网络带宽与 PCIe；适当降低 `grad_acc`；确认未重复累积（HF 与 DS 配置一致）。

4. **ZeRO‑3 与 DDP/TP/PP 组合**  
   - ZeRO 与 DDP（数据并行）是互补关系；TP/PP 需要在训练脚本中额外配置，超出本生成器范围。

---

## 九、许可证

本文档与脚本用于教学演示；请遵循你项目的总体许可证与上游依赖（DeepSpeed、PyTorch 等）的许可证条款。
