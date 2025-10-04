# 教程：LLaMA-7B 的 QLoRA 微调流水线

## 学习目标
- 熟悉 QLoRA 训练脚手架的命令行配置方式。
- 掌握 BitsAndBytes INT4 量化、LoRA 适配器与 `Trainer` 的协同工作流程。
- 理解长上下文训练时的分词策略与梯度累积设置。

## 背景原理
QLoRA 在基础模型上采用 INT4 量化并冻结原权重，仅训练 LoRA 增量 \(\Delta W\)。在梯度更新时，参数驻留在 4 bit 格式，通过 `bnb_4bit_compute_dtype` 指定的浮点精度执行反向传播。梯度累积实现有效批量：
\[
\text{Batch}_{\text{effective}} = \text{Batch}_{\text{device}} \times \text{Acc}_{\text{grad}} \times \text{Devices}.
\]

## 代码结构解析
- `QLoRAConfig` 与 `parse_args`：接收数据路径、模型名称、学习率、步数等参数。
- `read_jsonl`、`build_dataset`：加载 JSONL 指令对并转为 `Dataset`。
- `format_sample` / `tokenize`：构建 prompt 模板，限制最大长度 2048。
- `BitsAndBytesConfig`：启用 4bit 量化、双重量化（double quant）与 NF4 格式。
- `LoraConfig`：针对注意力模块 `q_proj`、`k_proj`、`v_proj`、`o_proj` 插入 LoRA。
- `TrainingArguments`：配置梯度累积、评估频率、保存策略等。

## 使用步骤
1. 准备指令数据后执行：
   ```bash
   python qlora_trainer.py --data train.jsonl --model meta-llama/Llama-2-7b-chat-hf --steps 200 --batch 1 --grad_acc 16
   ```
2. 根据显存调整 `--batch` 与 `--grad_acc`，保证有效批量满足训练需求。
3. 在多 GPU 环境中可结合 `accelerate launch` 启动脚本，并保留 `device_map="auto"`。
4. 训练结束后，在 `output_dir` 下得到 LoRA 适配器，可在推理代码中通过 `PeftModel` 加载。

## 拓展建议
- 引入验证指标记录并上传至 Weights & Biases，便于课后复盘。
- 当数据集较大时，可改用 `num_train_epochs` 与 `max_steps` 组合控制训练长度。
- 推理阶段可调用 `model.merge_and_unload()` 将 LoRA 权重合并回基础模型，减少部署依赖。
