# Transformers 8 卡训练最小示例：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson3/02_8gpu_config/dpp_torch_demo.py`（本文档基于你提供的教学代码撰写）  
> 运行环境：Python ≥ 3.9；NVIDIA GPU（单机多卡 8×）  
> 关键依赖：`transformers`、`datasets`、`torch`、（可选）`deepspeed`  
> 场景：用 **GPT‑2** 做最小化 **Causal LM** 微调，支持 **torchrun (DDP)** 与 **DeepSpeed ZeRO‑3**。当缺少 `train.txt/val.txt` 时使用内置微型语料自动演示。

---

## 目录
- [Transformers 8 卡训练最小示例：使用说明与函数文档](#transformers-8-卡训练最小示例使用说明与函数文档)
  - [目录](#目录)
  - [一、功能概览](#一功能概览)
  - [二、快速开始](#二快速开始)
    - [依赖安装](#依赖安装)
    - [准备数据](#准备数据)
    - [一键运行（DDP / DeepSpeed）](#一键运行ddp--deepspeed)
  - [三、整体流程与数据流](#三整体流程与数据流)
  - [四、命令行参数](#四命令行参数)
  - [五、函数与关键逻辑文档](#五函数与关键逻辑文档)
    - [`read_lines(path: str) -> List[str]`](#read_linespath-str---liststr)
    - [`build_dataset(train_path: str, val_path: str, fallback: bool=True)`](#build_datasettrain_path-str-val_path-str-fallback-booltrue)
    - [`tokenize_blockwise(texts: List[str], tokenizer, block_size: int) -> Dict[str, List[List[int]]]`](#tokenize_blockwisetexts-liststr-tokenizer-block_size-int---dictstr-listlistint)
    - [`main()`](#main)
  - [六、训练设置与常见配方](#六训练设置与常见配方)
  - [七、评估指标与输出](#七评估指标与输出)
  - [八、常见问题排查（FAQ）](#八常见问题排查faq)
  - [九、扩展建议](#九扩展建议)
  - [十、许可证](#十许可证)

---

## 一、功能概览

- **DDP 多卡训练**：使用 `torchrun` 单机 8 卡直接启动；  
- **ZeRO‑3**（可选）：传入 `--deepspeed ds_zero3.json` 即可启用 DeepSpeed ZeRO‑3；  
- **数据兜底**：若 `train.txt/val.txt` 不存在，自动切换到内置小语料（便于课堂演示）；  
- **块式分词**：将长文本按 `block_size` 切块，适合自回归 LM 训练；  
- **混合精度**：`--fp16/--bf16` 二选一（按硬件支持）；  
- **梯度检查点**：`--grad_ckpt` 可省显存；  
- **困惑度**：评估时计算 `perplexity = exp(eval_loss)`，快速感知训练收敛。

---

## 二、快速开始

### 依赖安装
```bash
pip install "transformers>=4.41" "datasets>=2.19.0"
# 依据 CUDA 版本安装 PyTorch（示例为 CUDA 12.1）
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# 可选：DeepSpeed（若使用 ZeRO-3）
pip install deepspeed
```

### 准备数据
- 将一行一句的训练文本写入 `train.txt`；验证文本写入 `val.txt`。  
- 若文件缺失，脚本会自动使用内置微型语料：**只用于演示**，请在真实训练中替换。

### 一键运行（DDP / DeepSpeed）

**DDP（单机 8 卡）：**
```bash
torchrun --nproc_per_node=8 lesson3/02_8gpu_config/dpp_torch_demo.py \
  --model_name gpt2 \
  --output_dir outputs_gpt2_ddp \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --bf16   # 或 --fp16
```

**DeepSpeed ZeRO‑3（单机 8 卡）：**
```bash
torchrun --nproc_per_node=8 lesson3/02_8gpu_config/dpp_torch_demo.py \
  --model_name gpt2 \
  --output_dir outputs_gpt2_ds \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --bf16 \
  --deepspeed ds_zero3.json
```
> 建议搭配前文的 **ZeRO‑3 配置生成器**（`ds_zero3.json`）。

**常用环境变量：**
```bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 如遇 NCCL 超时，可适度加大或设置网络接口
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
# IB 环境常用
# export NCCL_SOCKET_IFNAME=ib0
```

---

## 三、整体流程与数据流

1. **解析参数/设定随机种子**；  
2. **加载分词器与模型**（若 `--grad_ckpt` 则启用 `gradient_checkpointing_enable()`）；  
3. **读取/兜底构建数据集** → **块式分词** → **LM 数据整理**（collator 自动生成 `labels`）；  
4. **配置 `TrainingArguments`**（DDP/DeepSpeed、精度、日志/评估/保存间隔等）；  
5. **启动 `Trainer`** 训练与评估；  
6. **输出 perplexity**、保存权重与分词器。

---

## 四、命令行参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--model_name` | 基座模型名或路径（HF Hub） | `gpt2` |
| `--train_file` / `--val_file` | 训练/验证文本路径（每行一句） | `train.txt` / `val.txt` |
| `--output_dir` | 输出目录 | `outputs` |
| `--block_size` | 切块长度（token 级） | `512` |
| `--per_device_train_batch_size` | 每卡训练 batch | `1` |
| `--per_device_eval_batch_size` | 每卡评估 batch | `1` |
| `--gradient_accumulation_steps` | 梯度累积步数 | `8` |
| `--learning_rate` | 学习率 | `2e-5` |
| `--weight_decay` | 权重衰减 | `0.0` |
| `--warmup_ratio` | 预热比例 | `0.03` |
| `--num_train_epochs` | 训练轮数 | `1` |
| `--fp16` / `--bf16` | 混合精度 | `False` |
| `--grad_ckpt` | 开启梯度检查点 | `False` |
| `--seed` | 随机种子 | `42` |
| `--deepspeed` | DeepSpeed 配置文件路径 | `None` |

> **有效全局 batch**（单机 8 卡）：  
> `global_batch = per_device_train_batch_size × gradient_accumulation_steps × 8`。

---

## 五、函数与关键逻辑文档

### `read_lines(path: str) -> List[str]`
**功能**：按行读取文本文件，去除空行与首尾空白。  
**返回**：`List[str]`。

### `build_dataset(train_path: str, val_path: str, fallback: bool=True)`
**功能**：优先从 `train.txt/val.txt` 读取；若不存在且 `fallback=True`，返回内置演示语料。  
**返回**：`(train_ds, val_ds)`，二者都是 `datasets.Dataset`，列为 `{"text": ...}`。

### `tokenize_blockwise(texts: List[str], tokenizer, block_size: int) -> Dict[str, List[List[int]]]`
**功能**：将多条文本合并为一个长串，分词后**按 `block_size` 切块**：
```python
text = "\n\n".join(texts)
ids = tokenizer(text, add_special_tokens=False)["input_ids"]
chunks = [ids[i:i+block_size] for i in range(0, len(ids) - block_size + 1, block_size)]
return {"input_ids": chunks}
```
**返回**：字典，键为 `"input_ids"`，值为若干个长度为 `block_size` 的 token 列表。  
**注意**：这是**最小实现**。生产中可考虑：
- 丢弃过短的尾块或做 **packing**；  
- 使用 **sliding window** 或 **多文档边界处理**；  
- 对长序列任务提高 `block_size` 并打开 `grad_ckpt`。

### `main()`
- 解析 CLI → 设种子；  
- 加载 tokenizer（若无 `pad_token` 则置为 `eos_token`）；  
- 加载模型，必要时启用 `gradient_checkpointing_enable()`；  
- 调用 `build_dataset`、`map(tokenize_blockwise)` 得到 `train_tok/val_tok`；  
- 构建 `DataCollatorForLanguageModeling(mlm=False)`；  
- 设定 `TrainingArguments`：
  - `deepspeed=args.deepspeed` → 一键启用 ZeRO‑3；
  - `ddp_find_unused_parameters=False`（DDP/ZeRO 建议关闭）；
  - `fp16/bf16`、日志/评估/保存间隔等。  
- 启动 `Trainer.train()`、`Trainer.evaluate()`；计算 perplexity；保存模型与分词器。

---

## 六、训练设置与常见配方

- **显存吃紧**：`--per_device_train_batch_size 1 --gradient_accumulation_steps 32 --grad_ckpt --bf16`；必要时配合 ZeRO‑3。  
- **更长上下文**：增大 `--block_size`，并打开 `--grad_ckpt`；学习率可略降。  
- **更稳训练**：适度增大 `warmup_ratio`（如 `0.1`），或使用 `adamw_torch`/`cosine` 日程（需在 `TrainingArguments` 中显式指定）。  
- **评估间隔**：目前 `eval_steps=100`；小数据可调小以更频繁观察收敛。

---

## 七、评估指标与输出

- 训练结束后打印：
  ```text
  Eval metrics: {'eval_loss': ..., 'perplexity': ...}
  ```
- `perplexity = exp(eval_loss)`；若 `eval_loss` 过大会设为 `inf`。  
- 输出目录包含 `pytorch_model.bin`（或保存分片）、`config.json`、`tokenizer.json` 等。

---

## 八、常见问题排查（FAQ）

1. **`CUDA error: out of memory`**  
   - 降低 `per_device_train_batch_size`、提高 `gradient_accumulation_steps`、开启 `--grad_ckpt`；或使用 ZeRO‑3。

2. **`find_unused_parameters` 警告/报错**  
   - 已设置 `ddp_find_unused_parameters=False`；如仍报错，检查自定义模型是否存在**未参与反向**的分支。

3. **收敛缓慢或无效**  
   - 内置演示语料过小，仅用于跑通流程；真实训练请使用你的语料并延长训练时间。

4. **DeepSpeed 配置冲突**  
   - 确保 `ds_zero3.json` 的 `train_batch_size` 与 CLI 中的参数**语义一致**，避免重复放大；如用 HF `Trainer`，建议将“单一事实来源”固定在一端。

5. **NCCL 相关报错（超时/网络）**  
   - 设置/检查 `NCCL_SOCKET_IFNAME`；确保多卡能互联；适度增大 `NCCL_TIMEOUT`。

---

## 九、扩展建议

- **数据打包（packing）**：将多句拼接成接近满长的样本，提高利用率；  
- **学习率/调度器**：暴露 `--lr_scheduler_type`、`--weight_decay` 等更多 CLI；  
- **更换模型**：替换为 LLaMA/Qwen 指令版并采用相应聊天模板；  
- **监控与可视化**：打开 `report_to="wandb"` 或 `tensorboard`。

---

## 十、许可证

本文档与脚本用于教学演示；请遵循你项目的总体许可证以及上游依赖（Transformers、Datasets、PyTorch、DeepSpeed 等）的许可证条款。
