# 面向 ChatGLM-6B 的 LoRA 微调示例：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson1/05_lora_finetune/lora_chatglm_demo.py`（基于你提供的教学代码整理）  
> 运行环境：Python ≥ 3.9；NVIDIA GPU（推荐，支持 INT4 推理/微调）  
> 关键依赖：`transformers`、`peft`、`datasets`、`torch`、`bitsandbytes`（用于 4bit 量化）  
> 目标：以最小教学流程演示 **ChatGLM-6B** 类模型的 **LoRA** 微调（含 INT4 加载、LoRA 注入、训练与保存）。

---

## 目录
- [面向 ChatGLM-6B 的 LoRA 微调示例：使用说明与函数文档](#面向-chatglm-6b-的-lora-微调示例使用说明与函数文档)
  - [目录](#目录)
  - [一、重要说明与定位](#一重要说明与定位)
  - [二、快速开始](#二快速开始)
    - [1) 安装依赖](#1-安装依赖)
    - [2) 保存脚本并执行](#2-保存脚本并执行)
  - [三、脚本总览与执行流程](#三脚本总览与执行流程)
  - [四、类与函数文档](#四类与函数文档)
    - [`LoRAFinetuneConfig`](#lorafinetuneconfig)
    - [`build_dummy_dataset() -> datasets.Dataset`](#build_dummy_dataset---datasetsdataset)
    - [`format_sample(sample: dict) -> str`](#format_samplesample-dict---str)
    - [`tokenize(dataset: Dataset, tokenizer) -> Dataset`](#tokenizedataset-dataset-tokenizer---dataset)
    - [`main() -> None`](#main---none)
  - [五、LoRA/PEFT 关键参数说明](#五lorapeft-关键参数说明)
  - [六、将示例切换为 ChatGLM-6B](#六将示例切换为-chatglm-6b)
  - [七、训练产物与推理使用](#七训练产物与推理使用)
  - [八、常见问题排查（FAQ）](#八常见问题排查faq)
  - [九、扩展建议](#九扩展建议)
  - [十、参考伪代码与模块选择](#十参考伪代码与模块选择)
  - [许可证](#许可证)

---

## 一、重要说明与定位

- **脚本定位**：这是一个 **教学/演示** 脚本，默认用**伪造的小样本指令数据**跑通 LoRA 微调流程与日志管线，便于理解端到端步骤。  
- **显存需求**：注释中提示 *“至少 24GB 显存”*，这是针对 **ChatGLM-6B INT4** 的典型建议。实际显存需求取决于：LoRA 注入的模块数量、batch size、序列长度、优化器状态等。  
- **当前默认模型**：`LoRAFinetuneConfig.model_name = "Qwen/Qwen3-0.6b"`（体量更小，便于教学）。若要严格 **对齐 ChatGLM-6B**，请见下面的[将示例切换为 ChatGLM-6B](#六将示例切换为-chatglm-6b)。

---

## 二、快速开始

### 1) 安装依赖
```bash
pip install "transformers>=4.41" "peft>=0.11.0" "datasets>=2.19.0" "accelerate>=0.31.0"
# CUDA 环境：
pip install --upgrade torch  # 按你的 CUDA 版本安装官方匹配 whl
# INT4 量化：
pip install bitsandbytes
```

> `bitsandbytes` 需要英伟达 GPU（计算能力 >= 7.0 更稳），在 macOS/CPU 环境通常不可用。无 GPU 时请将 `load_in_4bit=True` 改为 `False`。

### 2) 保存脚本并执行
```bash
python lesson1/05_lora_finetune/lora_chatglm_demo.py
```
默认流程：构建伪造数据 → INT4 加载基础模型（如可行）→ 注入 LoRA → 训练若干步 → 保存 LoRA 适配器到 `./outputs/chatglm_lora/`。

---

## 三、脚本总览与执行流程

1. **构造数据**：`build_dummy_dataset()` 生成 20 条 *instruction/input/output* 示例；  
2. **切分数据**：`train_test_split(test_size=0.1)`；  
3. **分词器**：`AutoTokenizer.from_pretrained(..., trust_remote_code=True)`；设置 `pad_token = eos_token`；  
4. **加载模型**：`AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)`（需要 GPU + bitsandbytes）；  
5. **配置 LoRA**：`LoraConfig(r, lora_alpha, lora_dropout, target_modules=...)`；  
6. **注入 LoRA**：`get_peft_model(base_model, peft_config)`；  
7. **Trainer**：配置 `TrainingArguments` 与 `Trainer`，执行训练与评估；  
8. **保存**：`trainer.save_model(output_dir)` 仅保存 **LoRA 适配器**。

---

## 四、类与函数文档

### `LoRAFinetuneConfig`

```python
@dataclass
class LoRAFinetuneConfig:
    model_name: str = "Qwen/Qwen3-0.6b"
    output_dir: Path = Path("./outputs/chatglm_lora")
    r: int = 8
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 1e-4
    batch_size: int = 2
    steps: int = 50
```
**字段说明**
- **model_name**：基础 Causal LM 的名称或本地路径。教学默认较小模型；切换 ChatGLM-6B 见下文。  
- **output_dir**：保存 LoRA 适配器的目录。  
- **r**（rank）：LoRA 的秩，控制低秩增量 \(BA\) 的瓶颈维度，越大表达能力越强、参数越多。  
- **alpha**：LoRA 缩放系数（实际缩放为 `alpha / r`）。  
- **dropout**：LoRA 分支的 dropout 概率，规避过拟合。  
- **learning_rate**：训练学习率。  
- **batch_size**：`per_device_train_batch_size`。  
- **steps**：最大训练步数 `max_steps`。

---

### `build_dummy_dataset() -> datasets.Dataset`
**功能**：构造 20 条简单指令样本，字段为 `instruction/input/output`。  
**返回**：`Dataset`（`datasets` 库对象）。  
**用途**：教学可跑通；实际项目请替换为你的真实指令数据。

---

### `format_sample(sample: dict) -> str`
**功能**：把一条样本格式化为单段 Prompt：
```
问：{instruction}
补充信息：{input}
答：{output}
```
**用途**：便于直接做自回归训练（把“答：{output}”作为目标序列）。

---

### `tokenize(dataset: Dataset, tokenizer) -> Dataset`
**功能**：对每条格式化后的文本调用 `tokenizer(...)` 进行编码。  
**关键参数**：`max_length=768`、`truncation=True`、`padding="max_length"`。  
**返回**：移除原始列后，保留 `input_ids/attention_mask` 等张量列的数据集。

> 在因果 LM 训练中，若要精细控制 label 掩码（仅对 `output` 部分计算 loss），可改用自定义 `collator` 以设置 `labels` 中对应 `input` 的 token 为 `-100`。

---

### `main() -> None`
**核心流程**
1. 构建并切分数据；  
2. 加载分词器并设置 `pad_token`；  
3. **加载基础模型**（可尝试 `load_in_4bit=True`）：
   ```python
   base_model = AutoModelForCausalLM.from_pretrained(
       config.model_name,
       trust_remote_code=True,
       load_in_4bit=True,
       device_map="auto" if torch.cuda.is_available() else None,
   )
   ```
4. **构造 LoRA 配置并注入**：
   ```python
   peft_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       r=config.r,
       lora_alpha=config.alpha,
       lora_dropout=config.dropout,
       target_modules=["query_key_value", "dense"],
   )
   model = get_peft_model(base_model, peft_config)
   model.print_trainable_parameters()
   ```
5. **构造 `TrainingArguments` 与 `Trainer`**，开启训练与评估；  
6. **保存 LoRA 适配器**：`trainer.save_model(config.output_dir)`。

**副作用**：在 `output_dir` 下写入 LoRA 适配器权重与配置。

---

## 五、LoRA/PEFT 关键参数说明

- **`r`（rank）**：低秩瓶颈维度，典型取值 4/8/16/32。`r` 越大，训练参数量与显存/算力开销越大。  
- **`lora_alpha`**：缩放系数，实际生效为 `alpha / r`；用于控制 LoRA 分支更新幅度。  
- **`lora_dropout`**：LoRA 分支 dropout，训练时生效，推理时关闭。  
- **`target_modules`**：决定在哪些子层注入 LoRA。  
  - ChatGLM/GLM 类常见投影层：`query_key_value`、`dense`、或 `dense_h_to_4h` / `dense_4h_to_h`（不同版本命名略有差异）。  
  - 你可以**打印模块名**来确认：
    ```python
    for n, _ in base_model.named_modules():
        if any(k in n for k in ["query", "key", "value", "dense"]):
            print(n)
    ```
- **冻结与可训练参数**：`get_peft_model` 会**冻结**基础模型参数，仅训练 LoRA 分支。

---

## 六、将示例切换为 ChatGLM-6B

将配置中的 `model_name` 改为对应仓库（示例）：
```python
model_name = "THUDM/chatglm-6b"        # 或 "THUDM/chatglm3-6b"
```
**注意事项**
1. **信任远程代码**：`trust_remote_code=True` 必须（GLM 系列自定义模型类与分词器）。  
2. **INT4 加载**：`load_in_4bit=True` 需要 `bitsandbytes` 与 GPU；若报错，请先验证 GPU/CUDA/驱动是否匹配。  
3. **`target_modules` 适配**：不同 ChatGLM 版本的线性层命名可能不同，建议先打印模块名再选择注入层。常见可尝试：  
   - `["query_key_value", "dense"]`  
   - 或（按版本）`["dense_h_to_4h", "dense_4h_to_h"]`  
4. **分词器**：ChatGLM 分词器通常内置 `eos_token`，将 `pad_token = eos_token` 可避免 padding 相关报错。

---

## 七、训练产物与推理使用

- **保存内容**：`trainer.save_model(output_dir)` 会保存 **LoRA 适配器**（而非基础权重）。  
- **推理加载**（示例）：
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from peft import PeftModel

  base = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_4bit=True, device_map="auto")
  tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  model = PeftModel.from_pretrained(base, "./outputs/chatglm_lora")
  model.eval()
  ```
- **合并权重（可选）**：部分场景可将 LoRA 合并回基座权重以便部署（需足够显存/磁盘；注意区分不同工具链的合并方法）。

---

## 八、常见问题排查（FAQ）

1. **`bitsandbytes`/4bit 报错或 CPU 环境**  
   - 在 CPU 或不支持的 GPU 上，关闭 4bit：`load_in_4bit=False`；或切换更小模型。

2. **显存溢出（OOM）**  
   - 降低 `batch_size`/`max_length`/`r`；提高 `gradient_accumulation_steps`；关闭评估或降低 `eval_steps`。

3. **训练 Loss 不下降或过拟合**  
   - 增加样本量、引入更规范的模板（system/user/assistant）、调小学习率或增大 `lora_dropout`。

4. **`target_modules` 不生效或命名不一致**  
   - 打印模块名后再选择；必要时用正则匹配或传入更具体的层名列表。

5. **INT4 下训练速度慢**  
   - 4bit 更省显存但速度可能受限；可权衡切换 8bit 或半精度（fp16/bf16）以提升吞吐。

---

## 九、扩展建议

- **自定义 `data_collator`**：仅对 `output` 段计算损失（把非标签位置的 `labels` 置 `-100`）；  
- **混合精度/AMP**：`fp16=True` 或 `bf16=True`；  
- **模块精细注入**：只在注意力 Q/K/V 或 MLP 投影注入 LoRA，以平衡质量与开销；  
- **学习率搜索**：使用更稳定的优化器与调度器（如 Adafactor、Cosine）；  
- **对齐真实数据**：将示例数据替换为你的高质量指令数据（JSONL / SFT 模式）。

---

## 十、参考伪代码与模块选择

**LoRA 注入线性层（示意）**：
```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, dropout=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # 冻结
        self.lora_a = nn.Parameter(torch.zeros(r, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        # 初始化略

    def forward(self, x):
        base = F.linear(x, self.weight)                # 冻结主干
        lora = F.linear(x, self.lora_a.T)              # x @ A^T
        lora = self.dropout(F.linear(lora, self.lora_b.T)) * self.scaling  # (xA^T)B^T
        return base + lora
```

**如何挑选 `target_modules`**（打印并筛选）：
```python
for name, module in base_model.named_modules():
    if any(k in name for k in ["query", "key", "value", "dense"]):
        print(name)
```

---

## 许可证

脚本面向教学用途。请同时遵循你项目的整体许可证以及上游模型与依赖库（ChatGLM/Qwen、PEFT、Transformers、BitsAndBytes、PyTorch）的许可证要求。
