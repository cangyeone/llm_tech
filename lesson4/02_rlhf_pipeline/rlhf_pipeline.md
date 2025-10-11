# 最小可运行的 RLHF 示例：无需 TRL

## 目录
- [最小可运行的 RLHF 示例：无需 TRL](#最小可运行的-rlhf-示例无需-trl)
  - [目录](#目录)
  - [功能概览](#功能概览)
  - [快速开始](#快速开始)
    - [依赖安装](#依赖安装)
    - [运行命令](#运行命令)
  - [数据与模型配置](#数据与模型配置)
  - [训练过程](#训练过程)
    - [偏好数据生成](#偏好数据生成)
    - [奖励模型训练](#奖励模型训练)
    - [PPO 训练](#ppo-训练)
  - [函数文档](#函数文档)
    - [`PairwiseTrainer`](#pairwisetrainer)
    - [`train_reward_model`](#train_reward_model)
    - [`CausalLMWithValueHead`](#causallmwithvaluehead)
    - [`ppo_step`](#ppo_step)
  - [常见问题排查](#常见问题排查)
  - [扩展与优化建议](#扩展与优化建议)
  - [许可证](#许可证)

---

## 功能概览

该脚本实现了一个最小可运行的强化学习与人类反馈（RLHF）训练框架，完全基于 PyTorch，并没有使用 TRL。训练过程分为两部分：

1. **构建和加载偏好数据**：生成一组“好”与“不好”示例，作为奖励模型的训练数据。  
2. **奖励模型训练与 PPO**：训练奖励模型并使用 **Proximal Policy Optimization (PPO)** 进行策略优化，优化目标是通过强化学习调整策略。

该框架适用于小型模型（如 Qwen3-0.6B），并兼容 **MPS**（苹果芯片）平台。

---

## 快速开始

### 依赖安装

```bash
pip install transformers datasets torch
```

对于 **MPS** 环境，确保使用 PyTorch 适配的版本。

### 运行命令

运行 RLHF 示例（使用 **Qwen/Qwen3** 模型）：

```bash
python lesson4/02_rlhf_pipeline/rlhf_no_trl.py --num-samples 100 --ppo-epochs 1 --lr 1e-6 --kl-beta 0.02
```

- `--num-samples`: 训练用的偏好数据集样本数量。
- `--ppo-epochs`: PPO 训练轮数。
- `--lr`: 学习率。
- `--kl-beta`: KL 正则化系数。

---

## 数据与模型配置

默认使用 `Qwen/Qwen3-0.6B` 作为基模型，偏好数据集由 `build_synthetic_preference_dataset()` 自动生成。数据集包含两种回答：**好** 和 **差** 的示例，用于训练奖励模型。

如果要使用不同的模型或数据，修改如下配置项：

```python
DEFAULT_MODEL = "Qwen/Qwen3-1.8B"  # 修改为需要的模型
```

---

## 训练过程

### 偏好数据生成

`build_synthetic_preference_dataset()` 方法生成合成数据集，包含 `prompt`（问题）、`chosen`（好回答）与 `rejected`（差回答）。该数据集用于训练奖励模型。

```python
prompts.append(q)
chosen.append(good)
rejected.append(bad)
```

### 奖励模型训练

奖励模型基于 **Pairwise Loss** 进行训练，即通过对比两个回答（好与差），计算出一个相对得分。

训练使用 `train_reward_model()` 函数，返回训练好的奖励模型与分词器。

```python
rm, tok = train_reward_model(reward_train, RewardCfg(model_name=args.model, epochs=args.rm_epochs), device)
```

### PPO 训练

在策略优化阶段，首先生成样本响应，然后计算策略损失、价值损失，并进行优化。PPO 使用 KL 正则化来防止过大的策略更新。

```python
stats = ppo_step(policy, ref, optim, tok, rm, batch_prompts, cfg, device)
```

---

## 函数文档

### `PairwiseTrainer`

继承自 `Trainer`，重写了 `compute_loss` 方法来计算 pairwise loss（奖励模型的损失）。

```python
class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids  = inputs["input_ids"]
        mask = inputs["attention_mask"].to(torch.bool)

        out = model(input_ids=ids.view(-1, ids.size(-1)), attention_mask=mask.view(-1, mask.size(-1)))
        loss = pairwise_loss(out.logits)
        return (loss, out) if return_outputs else loss
```

### `train_reward_model`

该函数用于训练奖励模型，生成的奖励模型将用于 PPO 策略优化。

```python
def train_reward_model(ds: Dataset, cfg: RewardCfg, device: torch.device):
    tok = prepare_tokenizer(cfg.model_name)
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    rm = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=cfg_over)
    trainer = PairwiseTrainer(model=rm, args=args, train_dataset=tokenized)
    trainer.train()
    return rm, tok
```

### `CausalLMWithValueHead`

该类为基础 **Causal LM** 模型添加了价值头（value head），用于在生成时评估每个 token 的值。

```python
class CausalLMWithValueHead(nn.Module):
    def __init__(self, base: AutoModelForCausalLM):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)
```

### `ppo_step`

该函数执行 PPO 的一个步骤，包括生成响应、计算损失（策略损失、价值损失、KL 正则化）并进行反向传播。

```python
def ppo_step(policy, ref, optim, tok, rm, prompts, cfg, device):
    # 生成、计算损失、更新
    return stats
```

---

## 常见问题排查

1. **模型加载失败：**  
   - 检查是否正确设置了 `model_name`，并确保模型可下载。  
   - 对于 **Qwen** 模型，确保 `attn_implementation="eager"`。

2. **显存不足：**  
   - 降低 `batch_size` 或增加 `gradient_accumulation_steps`。  
   - 开启 **gradient checkpointing** 以节省显存。

3. **训练不收敛：**  
   - 调整 `lr`、`ppo_epochs`，尝试增加或减少 `kl_beta`。

---

## 扩展与优化建议

1. **LoRA/QLoRA**：在此基础上结合 **LoRA** 或 **QLoRA**（低秩适应）方法，进行模型微调，减少计算资源消耗。  
2. **评估**：为模型输出添加更全面的评估，包括人类评分、自动评估（如 BLEU、ROUGE 等）。  
3. **多模态支持**：结合视觉、语音等多模态数据进行训练，增强模型的多任务学习能力。

---

## 许可证

此代码用于教育和研究目的，任何使用均需遵循相应的开源许可证和使用政策。

