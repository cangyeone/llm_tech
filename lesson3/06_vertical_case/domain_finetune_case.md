# FinQA Qwen3 Chat 训练脚本：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson3/06_vertical_case/domain_finetune_case.py`（本文档基于你提供的教学代码撰写）  
> 运行环境：Python ≥ 3.9；`transformers`、`datasets`、`torch`；可选 `deepspeed`（ZeRO‑3）  
> 场景：在 **FinQA/财报问答** 数据上，以 **Qwen/Qwen3‑X** 为基座进行 **Causal LM** 的 **SFT（只在答复段计损失）** 训练，支持单机 8 卡 **DDP** 或 **DeepSpeed ZeRO‑3**。

---

## 目录
- [FinQA Qwen3 Chat 训练脚本：使用说明与函数文档](#finqa-qwen3-chat-训练脚本使用说明与函数文档)
  - [目录](#目录)
  - [一、功能概览](#一功能概览)
  - [二、快速开始](#二快速开始)
    - [依赖安装](#依赖安装)
    - [启动命令（DDP/ZeRO‑3）](#启动命令ddpzero3)
    - [常用环境变量](#常用环境变量)
  - [三、数据准备与格式](#三数据准备与格式)
  - [四、整体流程与关键逻辑](#四整体流程与关键逻辑)
  - [五、命令行参数](#五命令行参数)
  - [六、函数文档](#六函数文档)
    - [`_join_lines(x: Any, sep=" ") -> str`](#_join_linesx-any-sep----str)
    - [`_pick_table(ex: Dict[str, Any]) -> Any`](#_pick_tableex-dictstr-any---any)
    - [`_norm_table(table: Any, max_rows=50, max_cols=16) -> List[List[str]]`](#_norm_tabletable-any-max_rows50-max_cols16---listliststr)
    - [`_table_to_text(rows: List[List[str]]) -> str`](#_table_to_textrows-listliststr---str)
    - [`_extract_qa(ex: Dict[str, Any]) -> (str, str)`](#_extract_qaex-dictstr-any---str-str)
    - [`build_example_from_raw(ex: Dict[str, Any]) -> Dict[str, str]`](#build_example_from_rawex-dictstr-any---dictstr-str)
    - [`load_your_qa_json_as_dataset(json_path: str) -> Dataset`](#load_your_qa_json_as_datasetjson_path-str---dataset)
    - [`load_splits(root: str) -> DatasetDict`](#load_splitsroot-str---datasetdict)
    - [`tokenize_qwen_chat_batch(batch, tokenizer, block_size: int)`](#tokenize_qwen_chat_batchbatch-tokenizer-block_size-int)
    - [`main()`](#main)
  - [七、训练与评估要点](#七训练与评估要点)
  - [八、常见问题排查（FAQ）](#八常见问题排查faq)
  - [九、扩展建议](#九扩展建议)
  - [十、许可证](#十许可证)

---

## 一、功能概览

- **数据解析**：从 `train/dev/test` JSON 中提取 **文本 + 表格 + 问答**，拼成聊天消息：`system/user/assistant`。  
- **模板对齐**：使用 `tokenizer.apply_chat_template` 生成输入；**只在答案区间计损失**（将 prompt 段 `labels` 置为 `-100`）。  
- **多卡训练**：原生 **torchrun (DDP)**；或传 `--deepspeed ds_zero3.json` 开启 **ZeRO‑3**。  
- **混合精度/梯度检查点**：`--bf16/--fp16` 与 `--grad_ckpt`。  
- **评估指标**：打印 `eval_loss` 与 `perplexity`（困惑度）。

---

## 二、快速开始

### 依赖安装
```bash
pip install "transformers>=4.41" "datasets>=2.19.0"
# 依据 CUDA 版本从官网选择对应 PyTorch 安装指令
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# 可选：DeepSpeed（若启用 ZeRO-3）
pip install deepspeed
```

### 启动命令（DDP/ZeRO‑3）

**DDP 训练（单机 8 卡）**
```bash
export NCCL_SOCKET_IFNAME=eth0     # 换成你的网卡
export NCCL_DEBUG=INFO

torchrun --standalone --nproc_per_node=8 lesson3/06_vertical_case/domain_finetune_case.py \
  --data_dir /path/to/your/json_dir \
  --model_name Qwen/Qwen3-1.8B \
  --output_dir outputs_qwen3_v2 \
  --block_size 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 2 \
  --bf16 --grad_ckpt
```

**ZeRO‑3（DeepSpeed）**
```bash
deepspeed --num_gpus=8 lesson3/06_vertical_case/domain_finetune_case.py \
  --data_dir /path/to/your/json_dir \
  --model_name Qwen/Qwen3-1.8B \
  --deepspeed ds_zero3.json \
  --output_dir outputs_qwen3_v2_zero3 \
  --block_size 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --num_train_epochs 2 \
  --bf16 --grad_ckpt
```

> 建议配合你的 `ds_zero3.json`（ZeRO‑3 配置）使用；确保 CLI 与 JSON 的 batch 语义一致。

### 常用环境变量
```bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
# IB 环境常用
# export NCCL_SOCKET_IFNAME=ib0
```

---

## 三、数据准备与格式

目录 `--data_dir` 支持以下命名（存在哪个就读取哪个）：
- `train.json` / `dev.json` / `test.json`；  
- 或同义名：`valid.json`、`validation.json`。

**单个 JSON 文件**应是**样本列表**，每条样本可包含以下字段（取其一/多）：
- `pre_text` / `post_text`：上下文文本，字符串或字符串列表；  
- `table`（或 `table_ori`）：表格，支持 list[list]/list[dict]/dict，脚本会规整并截断；  
- `qa`：若存在，优先从其中读取：`qa.question` 与 `qa.exe_ans`；  
- 或直接使用顶层 `question` / `answer`。

**最终拼装的 user 内容**：
```
[PRE_TEXT]
...
[TABLE]
a | b | c
1 | 2 | 3
...
[POST_TEXT]
...
Question: 你的问题
```

---

## 四、整体流程与关键逻辑

1. **读取数据**：`load_splits` → 各 split（train/validation/test）转为 `(user, assistant)`。  
2. **模板编码**：`tokenize_qwen_chat_batch`  
   - 使用 `apply_chat_template` 构建 **prompt**（system+user+`<|assistant|>`）与 **full**（含答案）；  
   - 计算 `prompt_len`，将 `labels[:prompt_len]` 置为 `-100`，**只在答案 tokens 计损失**。  
3. **训练**：`Trainer` + `DataCollatorForLanguageModeling(mlm=False)`；  
4. **评估**：打印 `eval_loss` 与 `perplexity = exp(eval_loss)`；  
5. **保存**：模型与分词器到 `--output_dir`。

---

## 五、命令行参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--data_dir` | 包含 `train/dev/test` JSON 的目录 | `sample_docs/FinQA/dataset` |
| `--model_name` | HF 模型名或本地路径 | `Qwen/Qwen3-0.6B` |
| `--output_dir` | 输出目录 | `outputs_qwen3_v2` |
| `--block_size` | 最大序列长度（超出截断） | `2048` |
| `--per_device_train_batch_size` | 每卡训练 batch | `1` |
| `--per_device_eval_batch_size` | 每卡评估 batch | `1` |
| `--gradient_accumulation_steps` | 梯度累积步数 | `16` |
| `--learning_rate` | 学习率 | `2e-5` |
| `--weight_decay` | 权重衰减 | `0.0` |
| `--warmup_ratio` | 学习率 warmup 比例 | `0.03` |
| `--num_train_epochs` | 训练轮数 | `2` |
| `--fp16` / `--bf16` | 混合精度 | `False` |
| `--grad_ckpt` | 启用梯度检查点 | `False` |
| `--seed` | 随机种子 | `42` |
| `--deepspeed` | DS 配置文件路径（启用 ZeRO‑3） | `None` |
| `--eval_steps` | 评估间隔（步） | `200` |
| `--save_steps` | 保存间隔（步） | `500` |
| `--log_steps` | 日志间隔（步） | `20` |

> **有效全局 batch（单机 8 卡）**：`global_batch = per_device_train_batch_size × gradient_accumulation_steps × 8`。

---

## 六、函数文档

### `_join_lines(x: Any, sep=" ") -> str`
将 `list[str] / str / None` 统一为字符串并清洗空白。

### `_pick_table(ex: Dict[str, Any]) -> Any`
优先返回 `ex["table"]`，否则 `ex["table_ori"]`。

### `_norm_table(table: Any, max_rows=50, max_cols=16) -> List[List[str]]`
将任意结构的表**规整**为二维字符串数组，并按行/列**截断**，避免过长。

### `_table_to_text(rows: List[List[str]]) -> str`
将二维表用 ` | ` 连接为多行文本，便于注入到上下文。

### `_extract_qa(ex: Dict[str, Any]) -> (str, str)`
兼容两种标注：`qa.question/exe_ans` 或顶层 `question/answer`；缺失时回退为空串。

### `build_example_from_raw(ex: Dict[str, Any]) -> Dict[str, str]`
拼装 **system/user/assistant** 所需字段：
- system：固定 `SYS_PROMPT`（财报问答助手）；  
- user：包含 `[PRE_TEXT]/[TABLE]/[POST_TEXT]` 与 `Question:`；  
- assistant：答案，缺失时置 `"N/A"`。

### `load_your_qa_json_as_dataset(json_path: str) -> Dataset`
读取**单个** JSON 文件（样本列表）并转为 `Dataset({"user","assistant"})`。

### `load_splits(root: str) -> DatasetDict`
在 `root` 下尝试读取 train/dev/test（或 valid/validation）并返回 `DatasetDict`。若均不存在报错。

### `tokenize_qwen_chat_batch(batch, tokenizer, block_size: int)`
- 生成 `prompt_ids` 与 `full_ids`（分别为不含/含答案）；  
- 截断到 `block_size`；将 `labels[:prompt_len]=-100`，只训练答案段；  
- 返回 `{"input_ids": List[List[int]], "labels": List[List[int]]}`。

### `main()`
- 解析参数/设种子 → `load_splits`；  
- 准备 tokenizer & model（若 `--grad_ckpt` 则开启 `gradient_checkpointing_enable()`）；  
- 通过 `map` 进行批量分词与掩码；  
- 组装 `TrainingArguments`（DDP/ZeRO、精度、日志/评估/保存）；  
- `Trainer.train/evaluate`，打印 `perplexity`，保存模型与分词器。

---

## 七、训练与评估要点

- **模板一致性**：不同 Qwen 版本的聊天模板可能不同，`apply_chat_template` 能自动适配；如自定义模板，请保持 **assistant 段在最后** 且掩码正确。  
- **只对答案计损失**：`labels[:prompt_len] = -100`；这是 SFT 常规做法。  
- **长上下文/表格**：`block_size=2048` 已较长，若 GPU 吃紧可降低，或打开 `--grad_ckpt`/增加 `gradient_accumulation_steps`。  
- **ZeRO‑3 配合**：`--deepspeed ds_zero3.json` 时，确保 **CLI 与 DS JSON 的 batch 定义不冲突**。  
- **评估指标**：`perplexity = exp(eval_loss)`，数值越低越好；对生成类任务还应补充**任务级指标**与**人工评估**。

---

## 八、常见问题排查（FAQ）

1. **`CUDA out of memory`**  
   - 降低 `per_device_train_batch_size`、提高 `gradient_accumulation_steps`、开启 `--grad_ckpt`；或改用 ZeRO‑3。

2. **模板报错/分词器缺少 pad_token**  
   - 已在代码中兜底将 `pad_token = eos_token`；如仍报错，检查所用 Qwen 模型的模板支持情况。

3. **DeepSpeed 配置不生效/冲突**  
   - 检查 `ds_zero3.json` 与 CLI 是否重复/冲突放大 batch；确保版本兼容（建议 `deepspeed >= 0.10`）。

4. **表格过长/混乱**  
   - 通过 `_norm_table` 的 `max_rows/max_cols` 控制规模；或在数据清洗阶段先行规整。

5. **验证集为 None**  
   - 若目录中不存在 dev/valid/validation，将无法进行评估；可仅训练后再单独评测。

---

## 九、扩展建议

- **指令增强**：将 `SYS_PROMPT` 参数化（CLI/环境变量）以适配多任务。  
- **更灵活的掩码**：仅对 assistant 的“答案段”计损失，忽略思考/工具调用等特殊 token。  
- **数据打包**：引入 **packing** / **sliding window** 提高 token 利用率。  
- **指标与日志**：接入 `wandb`/`tensorboard`；输出 `samples/s、step_time_s、gpu_util` JSONL 以便性能分析。  
- **混合专家/LoRA**：在此脚手架上注入 LoRA/QLoRA 或切换 MoE 结构做扩展实验。

---

## 十、许可证

本文档与脚本用于教学演示；请遵循你项目的总体许可证以及上游依赖（Transformers、Datasets、PyTorch、DeepSpeed 等）的许可证条款。
