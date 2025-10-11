# LLaMA/Qwen QLoRA 微调脚手架（NF4 + Paged Optim + Grad Checkpointing）：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson2/02_qlora_pipeline/qlora_trainer.py`（本文档基于你提供的教学代码撰写）  
> 运行环境：Python ≥ 3.9（建议 3.10+）；Linux/Windows（GPU 推荐）  
> 关键依赖：`transformers`、`peft`、`datasets`、`torch`、`bitsandbytes`（4-bit/NF4）、`accelerate`（可选）  
> 支持模型：具备 chat 模板或标准 Causal LM 头的 LLaMA / Qwen 系列（或兼容结构）

---

## 目录
- [LLaMA/Qwen QLoRA 微调脚手架（NF4 + Paged Optim + Grad Checkpointing）：使用说明与函数文档](#llamaqwen-qlora-微调脚手架nf4--paged-optim--grad-checkpointing使用说明与函数文档)
  - [目录](#目录)
  - [一、脚本功能概览](#一脚本功能概览)
  - [二、快速开始](#二快速开始)
    - [依赖安装](#依赖安装)
    - [准备数据（JSONL）](#准备数据jsonl)
    - [一键运行](#一键运行)
  - [三、核心配置与命令行参数](#三核心配置与命令行参数)
  - [四、数据处理与分词流水线](#四数据处理与分词流水线)
    - [`read_jsonl(path: Path) -> List[Dict[str, str]]`](#read_jsonlpath-path---listdictstr-str)
    - [`build_dataset(records) -> Dataset`](#build_datasetrecords---dataset)
    - [`format_sample(sample) -> str`](#format_samplesample---str)
    - [`tokenize(dataset, tokenizer) -> Dataset`](#tokenizedataset-tokenizer---dataset)
  - [五、模型加载与 QLoRA 关键步骤](#五模型加载与-qlora-关键步骤)
    - [BitsAndBytes NF4 量化](#bitsandbytes-nf4-量化)
    - [`prepare_model_for_kbit_training`](#prepare_model_for_kbit_training)
    - [LoRA 注入（peft）](#lora-注入peft)
    - [梯度检查点](#梯度检查点)
    - [分页优化器（`paged_adamw_32bit`）](#分页优化器paged_adamw_32bit)
  - [六、训练器与重要训练参数](#六训练器与重要训练参数)
  - [七、训练产物与推理加载](#七训练产物与推理加载)
  - [八、常见问题排查（FAQ）](#八常见问题排查faq)
  - [九、扩展建议](#九扩展建议)
  - [十、许可证](#十许可证)

---

## 一、脚本功能概览

该脚手架演示 **QLoRA** 的完整微调流程：

1. **加载偏好/指令数据（JSONL）** 到 `datasets.Dataset`；  
2. **4-bit NF4 量化** 加载基础模型（`bitsandbytes`）；  
3. **k-bit 训练准备**（`prepare_model_for_kbit_training`）以保证低精度训练的数值稳定；  
4. **LoRA 注入**（`peft`）并通过 `Trainer` 训练；  
5. **分页优化器** `optim="paged_adamw_32bit"`，显著降低显存常驻；  
6. **梯度检查点**（`gradient_checkpointing=True`），以计算换显存；  
7. 保存 **LoRA 适配器** 与 **分词器** 到 `output_dir`。

---

## 二、快速开始

### 依赖安装
```bash
pip install "transformers>=4.41" "peft>=0.11.0" "datasets>=2.19.0" "accelerate>=0.31.0"
# 按你的 CUDA 版本安装 torch（示例：CUDA 12.x）
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 4-bit / NF4
pip install bitsandbytes
```

> `bitsandbytes` 需要 NVIDIA GPU（多数情况下计算能力 >= 7.0 更稳）。若 CPU 或不支持 4bit，需改为 8bit/16bit 加载并关闭相关选项。

### 准备数据（JSONL）

每行一个对象，至少包含以下字段：
```json
{"instruction": "解释 LoRA 的核心思想", "input": "", "output": "LoRA 通过低秩增量..."}
{"instruction": "把要点改写为项目计划", "input": "调研/清洗/训练RM", "output": "计划：1) 调研..."}
```

> **提示**：若你的任务是“纯对话”或“仅输入→输出”，保证 `format_sample` 能拼出合理的教学 Prompt。

### 一键运行
```bash
python lesson2/02_qlora_pipeline/qlora_trainer.py \
  --data data.jsonl \
  --model Qwen/Qwen3-4b \
  --steps 200 \
  --batch 1 --grad_acc 16 \
  --bf16   # 如果硬件支持 BF16
```
训练产物默认写入 `./outputs/llama_qlora`。

---

## 三、核心配置与命令行参数

脚本使用 `@dataclass QLoRAConfig` 与 `argparse` 组合：
```python
@dataclass
class QLoRAConfig:
    data_path: Path
    model_name: str = "Qwen/Qwen3-4b"
    output_dir: Path = Path("./outputs/llama_qlora")
    batch_size: int = 1
    gradient_accumulation: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 100
    bf16: bool = True
    fp16: bool = False
```
命令行参数：
- `--data`：JSONL 数据路径；  
- `--model`：基础模型（HF Hub 名称或本地路径）；  
- `--output`：输出目录；  
- `--lr`：学习率；  
- `--steps`：最大训练步数；  
- `--batch`：单卡 batch；  
- `--grad_acc`：梯度累积步数（扩大有效 batch）；  
- `--bf16` / `--fp16`：混合精度选项（优先 BF16，若不支持再用 FP16）。

---

## 四、数据处理与分词流水线

### `read_jsonl(path: Path) -> List[Dict[str, str]]`
- **功能**：读取 JSONL 数据到 Python 列表。  
- **返回**：元素为 `{"instruction","input","output"}` 的字典。

### `build_dataset(records) -> Dataset`
- **功能**：将 Python 列表封装成 `datasets.Dataset` 便于后续 `map`。

### `format_sample(sample) -> str`
- **功能**：将一条样本拼接为训练文本：
  ```
  ### 指令:
  {instruction}
  ### 输入:
  {input}
  ### 回答:
  {output}
  ```
- **提示**：生产中应使用**统一的对话模板**（含 role 标记），并根据需要只对 `回答` 段计算 loss（见扩展建议）。

### `tokenize(dataset, tokenizer) -> Dataset`
- **功能**：对拼接后的文本进行分词，统一 `max_length=2048` 且 `padding="max_length"`，移除原始列。  
- **注意**：如果只想对 `output` 计算损失，请自定义 `data_collator` 以在 `labels` 中把非答案 token 置为 `-100`。

---

## 五、模型加载与 QLoRA 关键步骤

### BitsAndBytes NF4 量化

脚本中：
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```
- **NF4**：归一化 4-bit 量化码本；  
- **double_quant**：二次量化进一步压缩；  
- **compute_dtype**：在更高精度（bf16/fp16）上进行计算。

### `prepare_model_for_kbit_training`

```python
base_model = prepare_model_for_kbit_training(base_model)
```
- 典型行为：
  - 将部分层（如 LayerNorm）置为 `fp32` 以稳定训练；  
  - 调整 `requires_grad`；  
  - 配合后续 LoRA 仅训练低秩增量。

### LoRA 注入（peft）

```python
lora_config = LoraConfig(
    r=64, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```
- **target_modules**：按模型实现选择；Qwen/LLaMA 常见为 `q/k/v/o`，也可加入 `gate_proj/up_proj/down_proj`。  
- **r/alpha/dropout**：LoRA 三要素，控制表达能力与正则强度。

### 梯度检查点
```python
model.gradient_checkpointing_enable()
model.config.use_cache = False
```
- **作用**：以计算换显存，尤其在长序列或大 batch 时节省显存峰值；  
- **注意**：禁用 `use_cache` 以避免与 checkpoint 冲突。

### 分页优化器（`paged_adamw_32bit`）
在 `TrainingArguments` 中：
```python
optim="paged_adamw_32bit"
```
- bitsandbytes 的分页优化器将优化器状态驻留在 CPU，按需分页到 GPU，降低显存常驻。

---

## 六、训练器与重要训练参数

```python
training_args = TrainingArguments(
    output_dir=str(config.output_dir),
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation,
    learning_rate=config.learning_rate,
    max_steps=config.max_steps,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    report_to=[],
    optim="paged_adamw_32bit",
    bf16=config.bf16,
    fp16=(config.fp16 and not config.bf16),
    gradient_checkpointing=True,
    dataloader_num_workers=2,
)
```
- **梯度累积**：在小显存场景扩大有效 batch；  
- **eval/save 间隔**：教学示例设置较短步数便于观测；  
- **混合精度**：优先 BF16，再考虑 FP16。

---

## 七、训练产物与推理加载

脚本保存 **LoRA 适配器** 与 **分词器**：
```python
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
```
**推理加载（示例）**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4b",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"),
    device_map="auto",
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "./outputs/llama_qlora")
model.eval()

prompt = "请解释低秩适配（LoRA）的直观含义"
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=True))
```
> 如需**合并权重**以便部署（非 4bit 形态），可参考 `peft` 的 `merge_and_unload` 或社区脚手架工具；需要充足显存/磁盘。

---

## 八、常见问题排查（FAQ）

1. **`bitsandbytes` 安装失败或 4bit 报错**  
   - 检查 GPU/驱动/CUDA 与 `torch` 的匹配；必要时切到 8bit 或关闭量化。

2. **显存 OOM**  
   - 减小 `batch`/`max_length`，增大 `grad_acc`，或调低 LoRA `r`；确保开启 `gradient_checkpointing` 与分页优化器。

3. **Loss 不稳定/不下降**  
   - 降低学习率；提升样本质量与模板一致性；增大 `lora_dropout`；扩大训练步数。

4. **生成质量一般**  
   - 只对 `答案段` 计算 loss（自定义 collator）；为模型提供明确的 system 指令与角色模板；开展更长时间训练。

5. **`target_modules` 不匹配**  
   - 打印 `named_modules()`，确认投影层命名（如 `q_proj/k_proj/v_proj/o_proj`、`gate_proj`/`up_proj`/`down_proj`）。

---

## 九、扩展建议

- **仅对答案打标签**：自定义 `DataCollator` 制作 `labels`，将 prompt 与“指令/输入”对应的 token 置 `-100`。  
- **SFT → DPO/ORPO**：基于本脚手架产出的 SFT 模型继续做偏好优化。  
- **多模板/多任务**：在 `format_sample` 中引入模板多样性与角色标注。  
- **分布式并行**：结合 `accelerate` 或 `deepspeed`，并使用 ZeRO 优化进一步节省显存。  
- **评估与对齐**：加入自动化验证集指标（如长度惩罚、覆盖率、BLEU/ROUGE，或任务自定义打分）。

---

## 十、许可证

本脚本与文档用于教学演示；请遵循你项目的总体许可证以及上游模型与依赖（Transformers、PEFT、bitsandbytes、PyTorch 等）的许可证条款。

