# 教程：ZeRO-3 分布式训练配置生成器

## 学习目标
- 理解 ZeRO Stage 3 参数分片的核心配置项。
- 掌握如何根据 micro batch 与梯度累积计算全局 batch size。
- 学会在配置文件中开启 CPU offload 与混合精度选项。

## 背景原理
ZeRO Stage 3 将模型参数、梯度与优化器状态分片到不同 GPU，从而将内存复杂度从 \(\mathcal{O}(N)\) 降低到 \(\mathcal{O}(N/k)\)，其中 \(k\) 为设备数。全局批量计算公式：
\[
\text{Batch}_{\text{global}} = \text{Batch}_{\text{micro}} \times \text{Acc}_{\text{grad}} \times \text{Devices}.
\]

## 代码结构解析
- `ZeroConfig`：封装 micro batch、梯度累积、ZeRO stage、offload、bf16 等参数。
- `build_config`：生成符合 DeepSpeed 要求的字典，包含 `zero_optimization`、`bf16`、`gradient_clipping` 等字段。
- `save_config`：将配置保存为 JSON，方便与训练脚本结合。

## 使用指南
1. 根据硬件情况修改 `ZeroConfig` 的 `micro_batch_size` 与 `gradient_accumulation`。
2. 若 GPU 显存不足，可将 `offload` 设为 `True`，启用 CPU offload。
3. 运行脚本后得到 `ds_zero3.json`，在训练命令中使用：
   ```bash
   deepspeed --num_gpus=8 train.py --deepspeed ds_zero3.json
   ```
4. 可进一步扩展 `build_config`，加入 `activation_checkpointing`、`optimizer` 等字段。

## 讨论问题
- 在多节点场景下，是否需要额外配置 `zero_allow_untested_optimizer` 等参数？
- 当模型对 bf16 不稳定时，如何 fallback 到 fp16 并加入损失缩放？
- 如何根据监控数据自动调整 `gradient_accumulation_steps` 以匹配吞吐目标？
