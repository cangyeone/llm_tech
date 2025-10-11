
# 奖励模型训练脚本（基于 Qwen3）

本脚本演示了如何训练一个奖励模型，基于 **Qwen3** 预训练模型，在 **stack-exchange-paired** 数据集上进行微调，并使用 **pairwise 损失函数** 来训练模型。支持简单的 **train/test 拆分**，并集成了 **自定义 Trainer** 来处理 pairwise 损失。

---

## 目录
- [奖励模型训练脚本（基于 Qwen3）](#奖励模型训练脚本基于-qwen3)
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
    - [数据加载与处理](#数据加载与处理)
    - [模型与训练设置](#模型与训练设置)
    - [损失函数与评估](#损失函数与评估)
  - [输出与结果分析](#输出与结果分析)
  - [常见问题（FAQ）](#常见问题faq)
  - [扩展建议](#扩展建议)
  - [许可证](#许可证)

---

## 功能概览

- **数据加载与预处理**：加载并处理 **stack-exchange-paired** 数据集，生成包括 `prompt`、`chosen` 和 `rejected` 字段的数据集。
- **奖励模型训练**：基于 **Qwen3** 预训练模型进行微调，训练奖励模型（**pairwise 损失函数**）。
- **自定义 Trainer**：通过继承 `Trainer`，实现了自定义的损失函数，能够计算和优化模型的评分差异。
- **数据集划分**：将训练集按 80/20 的比例拆分为训练集和测试集。
- **评估与日志**：在训练过程中，通过 **MLflow** 和 **WandB** 记录训练日志和超参数。

---

## 环境与依赖

### 所需库：
- Python ≥ 3.7
- PyTorch ≥ 1.10
- Hugging Face Transformers
- Datasets（用于加载数据集）

### 安装依赖：
```bash
pip install transformers datasets evaluate torch numpy
```

---

## 快速开始

### 步骤 1：安装所需的库
```bash
pip install transformers datasets evaluate torch numpy
```

### 步骤 2：运行脚本
```bash
python lesson4/03_reward_model_training/train_reward_model.py
```

### 步骤 3：查看训练结果
脚本将自动训练奖励模型，并生成训练日志。最终模型将保存到指定的 `output_dir` 目录。

---

## 命令行参数

| 参数               | 默认值                       | 说明 |
|--------------------|------------------------------|------|
| `--model`           | `Qwen/Qwen3-0.6b`             | 预训练模型名称或路径 |
| `--dataset`         | `lvwerra/stack-exchange-paired` | 数据集名称 |
| `--output`          | `./outputs/reward_model`      | 输出模型的目录 |
| `--lr`              | `5e-6`                        | 学习率 |
| `--epochs`          | `1`                           | 训练轮数 |
| `--batch`           | `2`                           | 每个设备的批大小 |
| `--max-length`      | `512`                         | 最大序列长度 |
| `--frac`            | `0.01`                        | 训练数据的使用比例（默认使用 1% 数据） |

---

## 代码结构与功能说明

### 数据加载与处理

- **`load_preference_dataset(args: ScriptArguments)`**  
  该函数加载 **stack-exchange-paired** 数据集，并从中拆分出训练集和测试集。数据集将包含 `prompt`（问题）、`chosen`（优选回答）、`rejected`（拒绝回答）三个字段。

- **`preprocess(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int)`**  
  该函数使用 tokenizer 对数据集进行处理，构造 `(B, 2, L)` 形式的输入，其中 `B` 是批大小，`2` 是因为每个样本有两个候选（优选与拒绝），`L` 是序列长度。

### 模型与训练设置

- **`CausalLMWithValueHead`**  
  这是一个自定义的模型类，用于为 **Qwen3** 模型添加一个 **value head**，用于训练过程中的奖励计算。`forward()` 方法只返回序列每个 token 的值，而不是标准的 logits。

- **`make_trainer()`**  
  该函数配置了训练所需的 `Trainer`，并定制了损失计算方式。`PairwiseRewardTrainer` 继承了 `Trainer`，并实现了自定义的 `compute_loss()` 方法，该方法计算 `chosen` 和 `rejected` 之间的差异。

### 损失函数与评估

- **`compute_loss()`**  
  该函数计算 `chosen` 和 `rejected` 之间的 pairwise 损失，并返回损失值用于模型训练。

- **`compute_metrics(eval_pred)`**  
  该函数计算模型的评估指标，包括 **win_rate**（表示 `chosen` 更好回答的比例）和 **Kendall's Tau**（用于评估预测排名的一致性）。

---

## 输出与结果分析

训练过程中，模型的 **损失** 和 **评估指标** 将被记录到日志中。最终的模型将被保存到指定目录，且可以在 **MLflow** 中进行查看和管理。

- **训练日志**：每 100 步记录一次损失和评估指标（如 win_rate 和 Kendall's Tau）。
- **模型保存**：模型和 tokenizer 将被保存到 `output_dir` 目录。

---

## 常见问题（FAQ）

1. **问题：训练时未加载数据集**
   - **解决方案**：确保 `dataset_name` 参数正确设置，并检查数据集是否已下载。

2. **问题：模型无法加载**
   - **解决方案**：确保模型路径或名称正确，且支持下载。如果使用的是 Hugging Face 模型，需要正确配置 API 密钥。

3. **问题：损失值无法收敛**
   - **解决方案**：尝试调整学习率，增加训练轮数，或者使用更大的批大小。

---

## 扩展建议

- **增加超参数搜索**：可以集成 **Optuna** 或 **Hyperopt** 进行自动化的超参数优化。
- **多任务学习**：将奖励模型训练与其他任务（如对话生成）结合，使用多任务学习进一步提高模型性能。
- **模型评估**：可以将模型应用于不同的数据集，评估其泛化能力和鲁棒性。

---

## 许可证

此脚本仅用于教学和研究目的，使用时请遵守所用数据集和模型的许可协议。
