
# DPO 策略训练与 LoRA 集成（KL 约束版）

本脚本演示了如何在 **Qwen3** 模型上使用 **LoRA** 进行 **DPO 策略训练**。该训练通过对比 **chosen** 和 **rejected** 回答之间的对数几率差异，优化 **Policy Model**，并集成了 **KL 散度** 约束以增加模型的多样性。该脚本基于 **PEFT**（参数高效微调）框架，只有 LoRA 参数是可训练的。

---

## 目录
- [DPO 策略训练与 LoRA 集成（KL 约束版）](#dpo-策略训练与-lora-集成kl-约束版)
  - [目录](#目录)
  - [功能概览](#功能概览)
  - [环境与依赖](#环境与依赖)
    - [所需库：](#所需库)
    - [安装依赖：](#安装依赖)
  - [快速开始](#快速开始)
    - [步骤 1：安装所需的库](#步骤-1安装所需的库)
    - [步骤 2：运行脚本](#步骤-2运行脚本)
    - [步骤 3：查看训练结果](#步骤-3查看训练结果)
  - [命令行参数](#命令行参数)
  - [代码结构与功能说明](#代码结构与功能说明)
    - [数据加载与预处理](#数据加载与预处理)
    - [模型与训练设置](#模型与训练设置)
    - [DPO 损失与训练](#dpo-损失与训练)
    - [KL 散度损失与约束](#kl-散度损失与约束)
  - [输出与结果分析](#输出与结果分析)
    - [示例输出：](#示例输出)
  - [常见问题（FAQ）](#常见问题faq)
  - [扩展建议](#扩展建议)
  - [许可证](#许可证)

---

## 功能概览

- **数据加载与预处理**：从 **lvwerra/stack-exchange-paired** 数据集中加载训练数据，转换为适合 **Qwen3** 模型输入的格式。
- **LoRA 集成**：使用 **PEFT** 框架对 **Qwen3** 模型应用 **LoRA** 配置，优化策略模型。
- **DPO 策略训练**：通过 **DPO 损失** 来优化 **chosen** 和 **rejected** 回答之间的对比。
- **KL 散度约束**：对 **chosen** 回答与 **reference model** 的 **KL 散度** 进行约束，帮助提升模型多样性。
- **训练与评估**：训练过程中输出 **DPO 损失** 和 **KL 损失**，并定期保存训练模型。

---

## 环境与依赖

### 所需库：
- Python ≥ 3.7
- PyTorch ≥ 1.10
- Hugging Face Transformers
- PEFT（用于 LoRA 集成）
- Datasets（用于加载数据集）

### 安装依赖：
```bash
pip install transformers datasets torch numpy peft
```

---

## 快速开始

### 步骤 1：安装所需的库
```bash
pip install transformers datasets torch numpy peft
```

### 步骤 2：运行脚本
```bash
python lesson4/05_incremental_learning/dpo_with_lora_kl.py
```

### 步骤 3：查看训练结果
训练过程中，脚本将自动训练策略模型，并输出每个训练步骤的损失值。训练完成后，模型将被保存在指定目录。

---

## 命令行参数

本脚本没有命令行参数，所有的配置项都在代码中定义。你可以修改以下参数：

| 参数              | 默认值                   | 说明 |
|-------------------|--------------------------|------|
| `MODEL_NAME`      | `"Qwen/Qwen3-0.6b"`       | 预训练模型名称或路径 |
| `DATASET_NAME`    | `"lvwerra/stack-exchange-paired"` | 数据集名称 |
| `OUTPUT_DIR`      | `"./outputs/reward_model"` | 输出模型的目录 |
| `LEARNING_RATE`   | `5e-6`                    | 学习率 |
| `EPOCHS`          | `3`                       | 训练轮数 |
| `BATCH_SIZE`      | `8`                       | 每批次样本数 |
| `MAX_LENGTH`      | `512`                     | 最大序列长度 |
| `FRACTION`        | `0.01`                    | 使用数据的比例（用于快速实验） |

---

## 代码结构与功能说明

### 数据加载与预处理

- **`load_preference_dataset(args: ScriptArguments)`**  
  该函数加载 **lvwerra/stack-exchange-paired** 数据集，并将其拆分为训练集和测试集。每条数据包括 `prompt`（问题）、`chosen`（优选回答）、`rejected`（拒绝回答）。

- **`preprocess(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int)`**  
  使用 tokenizer 对数据集进行处理，将每条数据转换为 `(B, 2, L)` 的格式，其中 `B` 是批大小，`2` 表示每个样本包含两个候选答案（优选和拒绝），`L` 是序列长度。

### 模型与训练设置

- **`CausalLMWithValueHead`**  
  该类基于 **Qwen3** 预训练模型，添加了一个 **value head**，用于训练过程中的奖励计算。`forward()` 方法返回每个 token 的值，而不是传统的 logits。

- **`apply_lora(model)`**  
  使用 LoRA 配置包装 **Qwen3** 模型，应用 **PEFT** 框架，仅优化 **LoRA 权重**。

- **`dpo_loss_policy()`**  
  该函数计算 DPO 损失，用于优化 **chosen** 和 **rejected** 之间的对比。通过计算对数几率差异，使用 **logsigmoid** 计算最终损失。

### DPO 损失与训练

- **`train_dpo_policy(dataset, policy_model, ref_model, optimizer, epochs=3, batch_size=8)`**  
  该函数执行 **DPO 策略模型** 的训练。训练过程中，模型通过 **DPO 损失** 来更新权重，并定期输出损失值。

- **`compute_loss()`**  
  该函数计算每个 batch 的损失，基于 `chosen` 和 `rejected` 回答的对数几率差异。

- **`compute_metrics()`**  
  该函数用于计算模型的评估指标，如 **win_rate**（表示 **chosen** 更好的比例）和 **Kendall's Tau**（用于评估预测排名的一致性）。

### KL 散度损失与约束

- **`kl_divergence_loss(policy_logits, ref_logits, attention_mask_shifted)`**  
  该函数计算 **Policy Model** 和 **Reference Model** 之间的 **KL 散度**，并应用 **attention mask** 来只计算非 padding token 的损失。

---

## 输出与结果分析

训练过程中，脚本会输出每个 epoch 的损失值，并最终保存训练好的 **Policy Model**。模型和训练日志将保存在指定目录。

### 示例输出：
```text
Epoch 1, Step 1/100, DPO Loss: 0.7235, KL Loss: 0.3421, Total Loss: 1.0656
Epoch 1, Step 2/100, DPO Loss: 0.7102, KL Loss: 0.3210, Total Loss: 1.0312
...
Epoch 3 DPO Loss (Avg): 0.6142
Epoch 3 KL Loss (Avg): 0.3021
```

训练结束后，模型将被保存在 `output_dir` 指定的目录，并可用于后续推理任务。

---

## 常见问题（FAQ）

1. **问题：训练数据集加载失败**
   - **解决方案**：检查 `dataset_name` 参数是否正确，并确保数据集已下载。

2. **问题：模型无法加载**
   - **解决方案**：确保 `MODEL_NAME` 参数正确，并且支持从 Hugging Face 下载模型。

3. **问题：训练过程损失无法收敛**
   - **解决方案**：尝试调整学习率，增加训练轮数，或者使用更大的批大小。

---

## 扩展建议

- **自动化超参数搜索**：集成 **Optuna** 或 **Hyperopt** 进行自动化超参数优化。
- **多任务学习**：将该模型与其他任务（如对话生成）结合，使用多任务学习进一步提升性能。
- **评估与部署**：可以将训练好的模型应用于不同的数据集，评估其泛化能力，并部署到线上服务。

---

## 许可证

此脚本仅用于教学和研究目的。请遵守所用数据集和模型的许可协议。
