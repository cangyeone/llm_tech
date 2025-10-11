# P-Tuning v2 可学习提示词示例（修订版）：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson2/03_ptuning/ptuning_v2_demo.py`（本文档基于你提供的教学代码整理）  
> 运行环境：Python ≥ 3.9；`transformers>=4.40`、`peft>=0.10`、`torch`（CPU/MPS/GPU 皆可；GPU 推荐）  
> 目标：通过最小教学工程，演示 **P‑Tuning v2**（可学习前缀）如何与 HF `Trainer` 配合完成 Causal LM 的高效训练与推理。

---

## 目录
- [P-Tuning v2 可学习提示词示例（修订版）：使用说明与函数文档](#p-tuning-v2-可学习提示词示例修订版使用说明与函数文档)
  - [目录](#目录)
  - [一、概述](#一概述)
  - [二、快速开始](#二快速开始)
    - [依赖安装](#依赖安装)
    - [保存并运行](#保存并运行)
  - [三、整体流程](#三整体流程)
  - [四、配置与数据流水线](#四配置与数据流水线)
    - [`PTuningConfig`](#ptuningconfig)
    - [`build_dataset()`](#build_dataset)
    - [`format_sample(sample)`](#format_samplesample)
    - [`tokenize(dataset, tokenizer)`](#tokenizedataset-tokenizer)
  - [五、模型与 P‑Tuning v2 关键步骤](#五模型与-ptuning-v2-关键步骤)
    - [分词器与基座模型加载](#分词器与基座模型加载)
    - [`PromptEncoderConfig` 与前缀注入](#promptencoderconfig-与前缀注入)
    - [训练参数与 Trainer](#训练参数与-trainer)
  - [六、推理阶段（自动注入虚拟 tokens）](#六推理阶段自动注入虚拟-tokens)
  - [七、常见问题排查（FAQ）](#七常见问题排查faq)
  - [八、扩展建议](#八扩展建议)
  - [九、函数与类的逐条文档](#九函数与类的逐条文档)
    - [`PTuningConfig`（数据类）](#ptuningconfig数据类)
    - [`build_dataset() -> datasets.Dataset`](#build_dataset---datasetsdataset)
    - [`format_sample(sample: Dict[str,str]) -> str`](#format_samplesample-dictstrstr---str)
    - [`tokenize(dataset, tokenizer) -> datasets.Dataset`](#tokenizedataset-tokenizer---datasetsdataset)
    - [`main() -> None`](#main---none)
  - [十、许可证](#十许可证)

---

## 一、概述

该教学脚本展示：

1. 以 **`peft.PromptEncoder`（P‑Tuning v2）** 在**每层 Attention** 注入可学习的 **prefix K/V**；  
2. 使用 **HF `Trainer` + `DataCollatorForLanguageModeling`** 进行自回归训练；  
3. 在**推理阶段**，P‑Tuning v2 会**自动在前向中注入虚拟 tokens**，无需手工拼接前缀文本。

> **何时用 P‑Tuning v2？**  
> 当你希望**冻结或基本冻结**基座模型，仅通过**小量可学习参数**快速对齐一个任务（如摘要、改写、客服话术等），且希望**显存占用极小**时，P‑Tuning v2 是极佳选择。

---

## 二、快速开始

### 依赖安装
```bash
pip install "transformers>=4.40" "peft>=0.10.0" torch datasets
```

### 保存并运行
将代码保存为 `lesson2/03_ptuning/ptuning_v2_demo.py` 并执行：
```bash
python lesson2/03_ptuning/ptuning_v2_demo.py
```
脚本会训练 1 个 epoch（200 条样本的演示集），训练结束后保存模型与分词器，并给出一段推理生成示例。

---

## 三、整体流程

1. **配置加载**：`PTuningConfig` 定义训练超参与路径；  
2. **Tokenizer/Model**：加载分词器与 Causal LM（如 Qwen 小模型）；  
3. **P‑Tuning v2**：构建 `PromptEncoderConfig`，通过 `get_peft_model` 注入可学习前缀；  
4. **数据管线**：构建/切分/分词数据集；  
5. **Trainer**：使用 `Trainer` 进行最小训练；  
6. **保存&推理**：保存权重与分词器；演示在推理时自动注入前缀。

---

## 四、配置与数据流水线

### `PTuningConfig`

```python
@dataclass
class PTuningConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    prompt_length: int = 32
    task_type: str = "CAUSAL_LM"
    output_dir: str = "./outputs/ptuning"
    bf16: bool = False
    fp16: bool = False
    epochs: int = 1
    lr: float = 3e-4
    per_device_train_batch_size: int = 2
    logging_steps: int = 5
```
- **model_name**：基座模型名称或本地路径（如 Qwen/LLaMA 指令版）；  
- **prompt_length**：**虚拟 token 数**（前缀长度）；  
- **task_type**：固定 `"CAUSAL_LM"`；  
- **bf16/fp16**：混合精度（GPU 推荐 `bf16`，MPS/CPU 设为 `False`）；  
- **epochs/lr/batch/logging_steps**：常规训练超参；  
- **output_dir**：输出目录。

### `build_dataset()`
- **功能**：生成 200 条伪造客服语料（`instruction/input/output`）。  
- **用途**：教学演示可跑通；实际项目请替换为真实任务数据。

### `format_sample(sample)`
- **功能**：将一条样本拼接为简单的指令模板：
  ```text
  指令：{instruction}
  输入：{input}
  回答：{output}
  ```
- **提示**：生产环境建议采用**统一/稳定**的对话模板（可包含 role 标记等）。

### `tokenize(dataset, tokenizer)`
- **功能**：把拼接后的文本编码为定长序列（`max_length=512`，`padding=max_length`）。  
- **返回**：移除原始列后，保留 `input_ids/attention_mask` 等张量列的 `Dataset`。

---

## 五、模型与 P‑Tuning v2 关键步骤

### 分词器与基座模型加载
```python
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if (cfg.bf16 and torch.cuda.is_available()) else None
base_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True
)
# pad id 兜底
if getattr(base_model.config, "pad_token_id", None) is None:
    base_model.config.pad_token_id = tokenizer.pad_token_id
```
- **`trust_remote_code=True`**：部分模型（Qwen/GLM）需要该参数；  
- **混合精度**：在 GPU 上可启用 `bf16`，其余平台保持默认。

### `PromptEncoderConfig` 与前缀注入
```python
prompt_config = PromptEncoderConfig(
    task_type=cfg.task_type,
    num_virtual_tokens=cfg.prompt_length,
    encoder_hidden_size=base_model.config.hidden_size // 2,
    # encoder_type="MLP"（默认）或 "LSTM"
)
model = get_peft_model(base_model, prompt_config)
model.print_trainable_parameters()
```
- **核心思想**：在每层 Attention 的 K/V 前注入 **`num_virtual_tokens`** 个可学习前缀；  
- **encoder_hidden_size**：PromptEncoder 内部的小 MLP/LSTM 的中间维度（常取 `hidden_size/2` 或同量级）。  
- **参数规模**：远小于 LoRA/SFT，全程只更新 PromptEncoder 的参数。

### 训练参数与 Trainer
```python
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    num_train_epochs=cfg.epochs,
    learning_rate=cfg.lr,
    logging_steps=cfg.logging_steps,
    save_strategy="epoch",
    gradient_checkpointing=False,
)
trainer = Trainer(
    model=model, args=training_args, data_collator=collator,
    train_dataset=tokenized_train, eval_dataset=tokenized_eval,
)
trainer.train()
trainer.save_model(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
```
- **collator**：为 Causal LM 自动构建 `labels`；  
- **gradient_checkpointing**：P‑Tuning v2 的显存占用已很低，一般无需开启；确实需要时可安全开启以进一步节省显存。

---

## 六、推理阶段（自动注入虚拟 tokens）

```python
model.eval()
prompt_text = "请写一句积极的评价：这家店的服务如何？"
inputs = tokenizer(prompt_text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
- **无需手动拼接虚拟 token**。`peft` 在前向内部会将**可学习前缀**注入到各层 Attention 的 K/V 上。  
- 可搭配 `temperature/top_p` 调整采样风格；部署时关闭采样（贪心/beam search）以稳定输出。

---

## 七、常见问题排查（FAQ）

1. **训练无效/不收敛**  
   - 提升数据质量与模板一致性；适当增大 `prompt_length`、训练轮次或调小学习率。

2. **生成内容和任务无关**  
   - 增强 system/instruction 约束；数据中提供更明确的任务描述与正例。

3. **显存/内存不足**  
   - 减小 `max_length` 与 batch；在 GPU 上使用 `bf16/fp16`；或切换更小基座模型。

4. **不同模型支持差异**  
   - 某些模型需 `trust_remote_code=True`；如遇特殊聊天模板/特殊 tokenizer，按模型仓库文档做适配。

5. **只想对“回答段”计 loss**  
   - 当前用 `DataCollatorForLanguageModeling` 会对全序列建标签；如需仅对 `回答` 计算损失，请自定义 collator，将非答案 token 的 label 置为 `-100`。

---

## 八、扩展建议

- **混合提示学习**：将 P‑Tuning v2 与 LoRA 结合（Prompt+LoRA），在极小参数下追求更高上限；  
- **更复杂的 PromptEncoder**：切换 `encoder_type="LSTM"`，在长前缀或序列相关性强时或有提升；  
- **多任务模板**：构造多模板/多指令风格，提升泛化；  
- **评估指标**：引入任务相关的自动指标（如 ROUGE/BLEU/准确率）和人工评估指南。

---

## 九、函数与类的逐条文档

### `PTuningConfig`（数据类）
- **用途**：集中管理训练/模型/输出的关键超参；便于快速改动与追踪实验。

### `build_dataset() -> datasets.Dataset`
- **用途**：返回演示用 `Dataset`；可替换为你的真实数据读取函数。

### `format_sample(sample: Dict[str,str]) -> str`
- **用途**：把字典样本转为统一的训练文本模板；保持一致性。

### `tokenize(dataset, tokenizer) -> datasets.Dataset`
- **用途**：将文本转为 token，并固定到统一长度；便于小批量教学演示。

### `main() -> None`
- **用途**：端到端地连接 **加载 → 注入前缀 → 数据 → 训练 → 保存 → 推理** 的各步骤。

---

## 十、许可证

本文档与脚本用于教学演示；请遵循你项目的总体许可证及相关依赖（Transformers、PEFT、PyTorch 等）的许可证条款。
