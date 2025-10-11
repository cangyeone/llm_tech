
# `lesson5/01_qwen_workflow/qwen_alignment_workflow.py` — 教学版最小可运行 RLHF（基于 TRL）说明文档

> 本文档面向**教学演示**和**快速上手**，对脚本中的函数/类进行详细说明，并提供使用方法、运行示例、常见问题与扩展建议。脚本演示了 **偏好数据 → 奖励模型（RM）训练 → TRL PPO 对齐** 的完整最小闭环。

---

## 目录
- [`lesson5/01_qwen_workflow/qwen_alignment_workflow.py` — 教学版最小可运行 RLHF（基于 TRL）说明文档](#lesson501_qwen_workflowqwen_alignment_workflowpy--教学版最小可运行-rlhf基于-trl说明文档)
  - [目录](#目录)
  - [功能概览](#功能概览)
  - [环境与依赖](#环境与依赖)
  - [快速开始](#快速开始)
  - [命令行参数](#命令行参数)
  - [数据格式](#数据格式)
  - [代码结构与函数说明](#代码结构与函数说明)
    - [全局默认与基础工具](#全局默认与基础工具)
    - [偏好数据与奖励模型（RM）训练](#偏好数据与奖励模型rm训练)
    - [PPO（基于 TRL）训练主流程](#ppo基于-trl训练主流程)
  - [训练流程说明（端到端）](#训练流程说明端到端)
  - [数学与实现要点](#数学与实现要点)
    - [奖励模型（RM）](#奖励模型rm)
    - [PPO with KL 控制（TRL）](#ppo-with-kl-控制trl)
    - [实现细节](#实现细节)
  - [日志与输出](#日志与输出)
  - [常见问题（FAQ）](#常见问题faq)
  - [扩展建议](#扩展建议)
  - [许可证](#许可证)
    - [附：关键 API 速查](#附关键-api-速查)

---

## 功能概览

- **构造/加载偏好数据**（prompt, chosen, rejected 三元组；本脚本内置合成数据用于演示）。  
- **训练奖励模型（Reward Model, RM）**：
  - 使用 `AutoModelForSequenceClassification`，`num_labels=1`，以**成对比较（pairwise）**损失近似 Bradley–Terry / logistic 目标：
    \[ \mathcal{L}_{\text{pair}} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}}) \]
- **基于 TRL 的 PPO**：
  - `AutoModelForCausalLMWithValueHead` + 冻结的 `ref_model`；
  - **自适应 KL 控制**（`kl_penalty='kl'`，`target_kl` 与 `init_kl_coef` 由 TRL 管理）；
  - **外部奖励仅传 RM 分数**，KL 惩罚由 TRL 内部计算加入。

> 兼容 **CUDA / Apple MPS / CPU**（教学优先：不强制 fp16/bf16，不用 `pin_memory`）。

---

## 环境与依赖

- Python ≥ 3.9
- PyTorch（建议 >= 2.1；支持 CUDA 或 MPS 可选）
- Hugging Face:
  - `transformers`
  - `datasets`
- TRL（`pip install trl`）

**安装示例：**
```bash
pip install "torch>=2.1" -i https://pypi.org/simple
pip install transformers datasets trl accelerate
```

> **Qwen 建议**：设置 `attn_implementation="eager"`；并禁用 Flash-Attn2（脚本中通过 `PYTORCH_USE_FLASH_ATTENTION=0` 兜底）。

---

## 快速开始

```bash
# 1) 最小干跑（仅训练 RM，以验证环境）
python lesson5/01_qwen_workflow/qwen_alignment_workflow.py --model Qwen/Qwen3-0.6B --num-samples 200 --rm-epochs 1 --dry-run

# 2) 端到端（RM + PPO；建议先小步运行）
python lesson5/01_qwen_workflow/qwen_alignment_workflow.py \
  --model Qwen/Qwen3-0.6B \
  --num-samples 200 \
  --rm-epochs 1 \
  --ppo-epochs 1 \
  --batch-size 2 \
  --mini-batch-size 1 \
  --gen-max-new 64
```

若使用 **CUDA**：自动启用 fp16；**MPS**：不使用 `pin_memory`。

---

## 命令行参数

| 参数 | 类型 | 默认 | 说明 |
|---|---:|---:|---|
| `--model` | str | `Qwen/Qwen3-0.6B` | 策略/RM 的基座模型名或路径（需支持 CausalLM / SeqCls） |
| `--num-samples` | int | 100 | 合成偏好数据/训练样本数量（教学演示用） |
| `--rm-epochs` | int | 1 | RM 训练轮数 |
| `--ppo-epochs` | int | 1 | PPO 外层 epoch 次数 |
| `--batch-size` | int | 2 | PPO 外层 batch（一次生成与更新的 prompt 数） |
| `--mini-batch-size` | int | 1 | PPO 内部小批次（梯度更新每步的最小单元） |
| `--lr` | float | 1e-6 | PPO 策略与价值头的学习率 |
| `--gen-max-new` | int | 64 | 每次生成的新 token 上限 |
| `--seed` | int | 42 | 随机种子 |
| `--dry-run` | flag | False | 仅训练 RM，不做 PPO |

---

## 数据格式

**偏好数据（pairwise）：**
```json
{
  "prompt": "问题文本",
  "chosen": "较优回答",
  "rejected": "较差回答"
}
```
脚本内置 `build_synthetic_preference_dataset(n)` 构造教学用玩具数据；实际应用中请替换为真实偏好数据集。

**PPO 环节的 prompts：**  
使用 `build_prompts_dataset(n)` 仅提供一条条 `prompt`，策略生成 `response` 后由 RM 打分。

---

## 代码结构与函数说明

### 全局默认与基础工具

- **`DEFAULT_MODEL = "Qwen/Qwen3-0.6B"`**  
  默认基座模型，可改为其它兼容 CausalLM / SeqCls 的小模型以便教学演示。

- **`seed_everything(seed: int)`**  
  统一设定 Python / NumPy / PyTorch 随机种子（含多 GPU）。

- **`get_device() -> torch.device`**  
  优先返回 `cuda`，其次 `mps`，否则 `cpu`。

- **`prepare_tokenizer(name: str) -> AutoTokenizer`**  
  - 使用 `trust_remote_code=True` 加载分词器；
  - 若无 `pad_token` 则回退为 `eos_token`；
  - `padding_side="right"`，便于生成。

- **`_safe_max_ctx(model_name: str) -> int`**  
  查询模型最大位置长度 `max_position_embeddings`，若缺失回退 `2048`。用于限制 RM 编码最大长度。

---

### 偏好数据与奖励模型（RM）训练

- **`build_synthetic_preference_dataset(n: int) -> datasets.Dataset`**  
  返回包含 `prompt/chosen/rejected` 的教学数据集。

- **`pairwise_loss(logits: torch.Tensor) -> torch.Tensor`**  
  - 输入形状：`[B*2]` 或 `[B,2]`，其中第 0 列/前半为 `chosen` 的标量分数，第 1 列/后半为 `rejected` 的标量分数；
  - 实现：`-logsigmoid(chosen - rejected)` 的 batch 均值；
  - 用途：RM 训练目标（越大表示 RM 更偏好 chosen）。

- **`class PairwiseTrainer(Trainer)`**  
  - 继承自 HF `Trainer`，覆写 `compute_loss`：
    1. 期望输入 `input_ids/attention_mask` 形状为 `[B,2,L]`；
    2. 展平为 `[B*2, L]` 送入 `AutoModelForSequenceClassification`；
    3. 取 `logits.squeeze(-1)` 并喂入 `pairwise_loss`；
  - 返回：标量训练损失。

- **`@dataclass RewardCfg`**  
  | 字段 | 含义 | 默认 |
  |---|---|---:|
  | `model_name` | RM 基座 | `DEFAULT_MODEL` |
  | `max_len` | RM 编码最大长度（自动截断到安全上限） | 512 |
  | `lr` | RM 学习率 | 5e-6 |
  | `bs` | RM 训练批大小 | 2 |
  | `epochs` | RM 训练轮数 | 1 |

- **`train_reward_model(ds, cfg, device) -> (rm, tok, max_len)`**  
  - **预处理**：将每条 `(prompt, chosen/rejected)` 分别编码并堆叠为 `[B,2,L]`；
  - **构建 RM**：
    - `AutoModelForSequenceClassification`，`num_labels=1`，`problem_type="regression"`；
    - `attn_implementation="eager"`，`trust_remote_code=True`；
    - 统一设置 `pad_token_id` 到 RM 及其 `base_model.config`；
  - **训练**：`TrainingArguments`（MPS 友好：`dataloader_pin_memory=False`；CUDA 自动 `fp16=True`）；
  - **返回**：已 `eval()` 的 RM、对应 tokenizer、RM 的最大编码长度。

---

### PPO（基于 TRL）训练主流程

- **`@dataclass RunCfg`**  
  整体运行的超参集合（见“命令行参数”）。

- **`build_prompts_dataset(n) -> Dataset`**  
  构造仅含 `prompt` 的数据集，用于 PPO 生成。

- **`compute_rm_scores(rm, tok, prompts, responses, max_len, device, batch_size=8) -> List[float]`**  
  - 将 `(prompt, response)` 对拼接编码，喂入 RM，得到每条样本的**标量分数**；
  - 输出：Python `list[float]`（每个条目一个奖励分数）。

- **`main()`（入口）**  
  1. 解析命令行，构造 `RunCfg`；设种子与设备；  
  2. **RM 训练**：用合成偏好数据 `train_reward_model(...)`；如 `--dry-run`，到此结束；  
  3. **PPO 组装**：
     - `AutoModelForCausalLMWithValueHead.from_pretrained` 作为策略；
     - `AutoModelForCausalLM.from_pretrained` 作为参考模型 `ref_model`（冻结）；
     - **统一 pad_token_id**，`attn_implementation="eager"`；
  4. **TRL 配置**：`PPOConfig(kl_penalty="kl", target_kl=0.1, init_kl_coef=0.02, ...)`；  
  5. **循环**（外层 `ppo_epochs` × 内层按 batch 划分 prompts）：
     - 生成 `response_tensors = ppo_trainer.generate(...)`；
     - 解码 `responses`（只取 `response` 段）；
     - 计算 `rewards = compute_rm_scores(...)`；
     - 转为张量列表后 `ppo_trainer.step(query_tensors, response_tensors, rewards_tensors)`；
     - 记录/打印 `rm_reward_mean`, `ppo/kl_coef`, `loss/policy`, `loss/value` 等关键统计；
  6. **保存**：`outputs/ppo_policy_trl/`（策略 + 价值头 + tokenizer）。

---

## 训练流程说明（端到端）

```text
偏好数据 (prompt, chosen, rejected)
        └─> 奖励模型 RM（SeqCls, num_labels=1）
               └─ pairwise 损失训练得到 r(x,y)
prompts  ──> 策略 πθ 生成回答 y
               └─ RM 打分 r(x,y)
               └─ TRL 内部计算 KL(πθ || π_ref)
               └─ PPO: 最大化 r(x,y) - β·KL(πθ||π_ref) 的期望
保存策略（含 value head）
```

> **要点**：外部只需提供 **RM 分数** 作为奖励；**KL** 由 TRL 内部根据 `target_kl`、`init_kl_coef` 自适应调节系数。

---

## 数学与实现要点

### 奖励模型（RM）
- **目标**：让 `r(prompt, chosen) > r(prompt, rejected)`。  
- **损失**：
  \[ \mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \sigma(r_i^+ - r_i^-) \]

### PPO with KL 控制（TRL）
- TRL 将 **KL 惩罚**加入每条样本的 advantage 中，近似优化：
  \[ \max_\theta \mathbb{E}[ r(x,y) - \beta \cdot \mathrm{KL}(\pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)) ] \]
- `β`（KL 系数）由 **`kl_ctl`** 自适应调整，使得观测到的 KL 接近 `target_kl`。

### 实现细节
- **pad/eos 对齐**：对 tokenizer、policy、ref、RM 的 `pad_token_id` 做统一设置；
- **Qwen**：建议 `attn_implementation="eager"` 以兼容更广的环境；
- **MPS**：避免 `pin_memory=True` 带来的数据传输卡顿；
- **CUDA**：可开启 `fp16=True`。

---

## 日志与输出

- **训练日志**：标准输出打印 JSON 行，包含：
  - `epoch`, `step`, `rm_reward_mean`, `ppo/kl_coef`, `loss/policy`, `loss/value` 等。
- **检查点**：
  - RM：`outputs/reward_model/`（由 HF Trainer 管理，默认只保留 1 份）。
  - 策略（含 value head）与分词器：`outputs/ppo_policy_trl/`。

**推理示例（加载已保存策略）：**
```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
tok = AutoTokenizer.from_pretrained("outputs/ppo_policy_trl")
model = AutoModelForCausalLMWithValueHead.from_pretrained("outputs/ppo_policy_trl")
inp = tok("请简述注意力机制的核心思想。", return_tensors="pt")
out_ids = model.generate(**inp, max_new_tokens=128)
print(tok.decode(out_ids[0], skip_special_tokens=True))
```

---

## 常见问题（FAQ）

1. **`TypeError: ... num_labels` 相关**  
   - 请确保 **RM 使用 `AutoModelForSequenceClassification`**，并在 `AutoConfig` 中设置 `num_labels=1`、`problem_type="regression"`。脚本已内置处理。

2. **`pad_token` 缺失**  
   - 脚本会自动用 `eos_token` 作为 `pad_token`；也会把 `pad_token_id` 写入 policy/ref/RM 的 `config`。

3. **MPS 速度/卡顿**  
   - 避免 `pin_memory`；批大小小一些；`attn_implementation="eager"`。

4. **显存不足（OOM）**  
   - 降低 `gen_max_new`、`batch_size/mini_batch_size`；换更小模型；或在 CUDA 上使用更高版本 PyTorch。

5. **生成文本质量低**  
   - 增加 PPO 迭代、提高 `num_samples`、训练更强 RM、从真实偏好数据迁移。

---

## 扩展建议

- **接入真实偏好数据集**（如 `hh-rlhf`、`ultraFeedback` 等，需遵循许可）。
- **替换基座模型**（更大/更强的 Qwen/Llama 等），并加入 LoRA/QLoRA 以节省显存。
- **自定义奖励**：将 RM 分数与其它启发式或规则（长度、事实性惩罚）线性组合。
- **评测**：加入自动化评测（BLEU、ROUGE、GPT 评审、人工打分）。
- **可视化**：接入 `wandb` 或 `tensorboard`。

---

## 许可证

本教学脚本与文档示例用于学习交流。实际应用请遵循所用模型与数据集的**原始许可协议**。

---

### 附：关键 API 速查

- **RM**：`AutoModelForSequenceClassification`（`num_labels=1`）  
- **策略**：`AutoModelForCausalLMWithValueHead.from_pretrained(...)`  
- **参考**：`AutoModelForCausalLM.from_pretrained(...)`  
- **TRL**：`PPOTrainer(...).step(queries, responses, rewards)`；`generate(...)`；`kl_ctl.value`  

> ✅ 到此即可完成教学演示的最小 RLHF 流程。
