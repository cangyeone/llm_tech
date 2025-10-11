
# 多卡 DPO 训练脚本 (基于 TRL + Accelerate)

该脚本演示了如何使用 **Distributed Proximal Optimization (DPO)** 训练模型，并且支持 **多 GPU 并行**。脚本通过 TRL 和 Hugging Face Accelerate 实现了梯度同步、AMP 和 ZeRO 优化，并适配了 **Qwen3 系列模型**。训练过程中，DPO 以对比奖励为基础，通过最大化 **log likelihood** 来优化模型生成的响应。以下文档将详细介绍函数说明、使用方法以及训练过程。

---

## 目录
- [多卡 DPO 训练脚本 (基于 TRL + Accelerate)](#多卡-dpo-训练脚本-基于-trl--accelerate)
  - [目录](#目录)
  - [功能概览](#功能概览)
  - [环境与依赖](#环境与依赖)
  - [快速开始](#快速开始)
  - [命令行参数](#命令行参数)
  - [代码结构与函数说明](#代码结构与函数说明)
    - [环境准备与数据加载](#环境准备与数据加载)
    - [Tokenizer与模型加载](#tokenizer与模型加载)
    - [DPO Trainer 配置](#dpo-trainer-配置)
    - [训练过程](#训练过程)
    - [模型保存](#模型保存)
  - [训练流程说明（端到端）](#训练流程说明端到端)
  - [日志与输出](#日志与输出)
  - [常见问题（FAQ）](#常见问题faq)
  - [扩展建议](#扩展建议)
  - [许可证](#许可证)

---

## 功能概览

- **多卡并行训练**：支持单机多 GPU 配置，使用 Hugging Face Accelerate 自动同步梯度。
- **DPO（分布式近端优化）**：对比奖励训练，最大化生成样本的对数似然，基于参考模型。
- **ZeRO 优化**：自动处理显存优化，以支持大模型训练。
- **AMP 支持**：自动启用 **AMP（混合精度训练）**，进一步加速训练。

---

## 环境与依赖

- Python ≥ 3.9
- PyTorch（建议 ≥ 2.1；支持 CUDA 或 MPS）
- Hugging Face
  - `transformers`
  - `datasets`
- TRL（`pip install trl`）
- Accelerate（`pip install accelerate`）

**安装示例：**
```bash
pip install "torch>=2.1" -i https://pypi.org/simple
pip install transformers datasets trl accelerate
```

---

## 快速开始

1. **安装依赖**
    ```bash
    pip install "torch>=2.1" transformers datasets trl accelerate
    ```

2. **运行训练脚本**
    ```bash
    # 使用 torchrun 启动多 GPU 训练
    torchrun --nproc_per_node=4 lesson5/02_dpo_hf/dpo_hf_run.py
    ```

    或者使用 **Accelerate** 启动训练：
    ```bash
    accelerate launch lesson5/02_dpo_hf/dpo_hf_run.py
    ```

---

## 命令行参数

| 参数 | 类型 | 默认 | 说明 |
|---|---:|---:|---|
| `--model` | str | `Qwen/Qwen3-0.6B` | 预训练模型路径或名称 |
| `--num-samples` | int | 1000 | 训练样本数量 |
| `--epochs` | int | 1 | 训练轮数 |
| `--batch-size` | int | 2 | 批大小 |
| `--gradient-accumulation-steps` | int | 4 | 梯度累积步数 |
| `--learning-rate` | float | 5e-6 | 学习率 |
| `--beta` | float | 0.1 | DPO 强度系数，控制对比奖励的偏好强度 |
| `--output-dir` | str | `outputs/dpo_qwen_trl` | 模型保存路径 |

---

## 代码结构与函数说明

### 环境准备与数据加载

- **`os.environ.setdefault("PYTORCH_USE_FLASH_ATTENTION", "0")`**  
  禁用 Flash-Attn2 以防止与其他操作发生冲突，适用于多种 GPU 配置。

- **`dataset = load_dataset("lvwerra/stack-exchange-paired", split="train")`**  
  加载示例数据集（`lvwerra/stack-exchange-paired`），并选取其中 0.1% 数据进行训练。

- **数据格式**：
    ```json
    {
      "question": "请简述 Transformer 模型的核心思想。",
      "response_j": "Transformer 通过注意力机制处理长距离依赖。",
      "response_k": "Transformer 仅基于 RNN 模型进行处理。"
    }
    ```
    `response_j` 为正确答案（chosen），`response_k` 为错误答案（rejected）。

### Tokenizer与模型加载

- **`tokenizer = AutoTokenizer.from_pretrained(model_name)`**  
  加载 Hugging Face 提供的 tokenizer，适配预训练模型。确保 `pad_token` 设置正确（与 `eos_token` 对齐）。

- **模型加载**：
    - `policy_model`: 用于生成策略的模型。
    - `ref_model`: 参考模型，冻结参数用于对比。

### DPO Trainer 配置

- **`DPOConfig`**：  
  配置 DPO（分布式近端优化）的超参数，包括 `beta`（控制奖励强度）、`learning_rate`、`gradient_accumulation_steps`（梯度累积）、`max_length`（最大序列长度）等。

- **`DPOTrainer`**：
  - 将 `policy_model` 和 `ref_model` 传入 `DPOTrainer`，并设定训练参数。
  - **DPO 公式**：最大化目标是生成样本的对数似然差异：  
    \[
    \max_	heta \mathbb{E} \left[ \log \pi_	heta(y^+|x) - \log \pi_	heta(y^-|x) 
ight]
    \]
    其中 `y^+` 是正确答案，`y^-` 是错误答案。

### 训练过程

- **`trainer.train()`**  
  启动 DPO 训练，自动进行多卡并行训练，支持梯度同步和 AMP（自动混合精度）。

- **训练日志**：每 10 步记录一次，输出包括 DPO 强度系数 `beta`、训练损失、PPO 步骤统计等。

### 模型保存

- **保存最终模型**：
    ```python
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    ```

---

## 训练流程说明（端到端）

```text
训练数据 (question, response_j, response_k)
        └─> DPO：最大化对数似然差异 (log π_θ(y⁺|x) − log π_θ(y⁻|x))
prompts  ──> 策略模型生成回答 y
               └─ RM 计算奖励 r(x,y)
               └─ DPO 训练：更新策略模型以最大化奖励差异
保存策略模型及分词器
```

---

## 日志与输出

训练日志记录了每个 epoch 的训练过程，包括每步的 **DPO 强度系数**（`beta`）、**生成奖励**（`rm_reward_mean`）以及每个 PPO 步骤的梯度更新情况。所有模型输出会保存到 `output_dir` 中。

---

## 常见问题（FAQ）

1. **如何使用更大的数据集？**
   - 只需替换 `load_dataset()` 中的数据集路径，或者修改 `num-samples` 参数以调整样本量。

2. **如何配置不同的 GPU？**
   - 使用 `torchrun` 或 `accelerate` 启动多 GPU 训练，自动根据机器配置分配。

3. **如何优化内存使用？**
   - 增加 `gradient_accumulation_steps`，减少每次计算的批大小，从而降低显存压力。

4. **如何调整 DPO 强度？**
   - 调整 `beta` 参数，较大值使得对比损失更接近 KL 距离，较小值则增强对生成质量的优化。

---

## 扩展建议

- **增加自定义奖励机制**：除了使用 `DPO`，可以设计基于规则或外部评估模型的奖励函数。
- **多模态扩展**：将文本数据与其他模态（如图像、音频）结合，以扩展模型能力。
- **大模型训练**：在更强大的硬件（如 TPU、多个节点）上扩展模型规模。

---

## 许可证

本脚本仅用于教学示范，所使用的模型与数据集请遵循各自的许可协议。
