# SFT 与指令微调（Instruction Tuning）教学脚本使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`sft_basics.py`（你提供的教学代码）  
> 运行环境：仅 CPU 可运行；如需 GPU 演示可配合 `accelerate launch`。  
> 面向人群：教学/入门演示，帮助理解 **SFT** 与 **Instruction Tuning** 的数据格式差异及其对训练收敛的影响。

---

## 目录
- [SFT 与指令微调（Instruction Tuning）教学脚本使用说明与函数文档](#sft-与指令微调instruction-tuning教学脚本使用说明与函数文档)
  - [目录](#目录)
  - [快速开始](#快速开始)
    - [1) 安装依赖](#1-安装依赖)
    - [2) 保存脚本](#2-保存脚本)
    - [3) 一键运行](#3-一键运行)
  - [脚本总体流程](#脚本总体流程)
  - [函数与类文档](#函数与类文档)
    - [`FinetuneConfig`](#finetuneconfig)
    - [`EXAMPLE_PAIRS`](#example_pairs)
    - [`build_dataset`](#build_dataset)
    - [`tokenize`](#tokenize)
    - [`run_training`](#run_training)
    - [入口主程序](#入口主程序)
  - [数学原理与损失函数](#数学原理与损失函数)
    - [自回归语言模型目标](#自回归语言模型目标)
    - [交叉熵损失与标签屏蔽](#交叉熵损失与标签屏蔽)
    - [SFT vs Instruction Tuning 的本质区别](#sft-vs-instruction-tuning-的本质区别)
  - [运行与复现实验](#运行与复现实验)
    - [CPU 直接运行](#cpu-直接运行)
    - [GPU/多卡（可选）](#gpu多卡可选)
    - [输出与日志](#输出与日志)
  - [常见问题排查](#常见问题排查)
  - [扩展建议](#扩展建议)
  - [许可证](#许可证)

---

## 快速开始

### 1) 安装依赖
```bash
pip install "transformers>=4.41" datasets torch
# 可选：支持 GPU/分布式
pip install accelerate
```

> 建议 Python ≥ 3.9，`torch` 版本与本机 CUDA/MPS 环境匹配。

### 2) 保存脚本
将你提供的代码保存为 `lesson1/01_sft_basics/sft_theory_notebook.py`（文件名可自定）。

### 3) 一键运行
```bash
python lesson1/01_sft_basics/sft_theory_notebook.py
```
脚本会：
1. 先以 **Instruction Tuning** 数据格式训练与评估一次；
2. 再切换为 **传统 SFT** 文本拼接格式训练与评估一次；
3. 在日志中打印二者的 `eval_loss` 对比。

---

## 脚本总体流程

1. **构造数据集**：根据 `FinetuneConfig.instruction_mode` 决定采用 *指令格式* 还是 *SFT 拼接格式*。
2. **分词与张量化**：调用 `tokenize()` 将文本批量编码为定长序列。
3. **加载模型**：`AutoModelForCausalLM.from_pretrained(config.model_name)`。
4. **训练器**：配置 `TrainingArguments` 与 `Trainer`，执行一次最小化的训练与评估。
5. **对比指标**：输出两种格式的 `eval_loss`，演示格式差异对学习效果的影响。

---

## 函数与类文档

### `FinetuneConfig`

```python
@dataclass
class FinetuneConfig:
    model_name: str = "Qwen/Qwen3-4b"
    output_dir: Path = Path("./outputs/sft_basics")
    max_steps: int = 30
    learning_rate: float = 5e-5
    batch_size: int = 2
    warmup_steps: int = 3
    instruction_mode: bool = True
```
**说明**：微调超参数与运行选项。

- **`model_name`**：Hugging Face 上的 Causal LM 模型名称或本地路径。示例：`"Qwen/Qwen3-4b"`。
- **`output_dir`**：训练输出目录（权重、日志、事件文件等）。
- **`max_steps`**：训练最大 step 数；教学演示设置较小值便于快速结束。
- **`learning_rate`**：AdamW 学习率。
- **`batch_size`**：`per_device_train_batch_size`。
- **`warmup_steps`**：学习率预热步数，缓解初期不稳定。
- **`instruction_mode`**：
  - `True`：采用 **Instruction Tuning** 的 *指令+输入+回答* 三段式格式；
  - `False`：采用 **传统 SFT** 的 *问答拼接* 两段式格式。

---

### `EXAMPLE_PAIRS`

```python
EXAMPLE_PAIRS: List[Dict[str, str]] = [
    {"instruction": "...", "input": "...", "output": "..."},
    ...
]
```
**说明**：少量伪造的教学数据，用于可复现实验。你可以自由添加更多条目。

- 字段含义：
  - **`instruction`**：任务指令（如“请概括下面的段落”）；
  - **`input`**：原始输入内容；
  - **`output`**：理想答案。

---

### `build_dataset`

```python
def build_dataset(config: FinetuneConfig) -> Dataset:
    """根据配置生成 SFT 或指令微调数据集。"""
```
**功能**：根据 `instruction_mode` 生成对应格式的纯文本数据集（`datasets.Dataset`）。

- 当 `instruction_mode=True`（Instruction Tuning）：每条样本会格式化为：
  ```text
  指令：{instruction}
  输入：{input}
  回答：{output}
  ```
- 当 `instruction_mode=False`（传统 SFT）：每条样本会格式化为：
  ```text
  问题：{input}
  回答：{output}
  ```

**返回值**：`Dataset`，单列 `text`。

**注意**：该函数**不涉及**模板中的特殊标记（如 `<|system|>`、`<|user|>`、`<|assistant|>`）。教学演示力求最小化；生产中建议引入**一致的对话模板**与**role 标记**以提升泛化。

---

### `tokenize`

```python
def tokenize(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """对文本执行分词，截断到 512 token。"""
```
**功能**：将 `text` 列批量编码为 token 序列，固定 `max_length=512` 并 `padding="max_length"`。

- **输入**：
  - `dataset`：包含 `text` 列的数据集；
  - `tokenizer`：与 `model_name` 匹配的分词器。
- **处理**：
  - 截断：超长序列将被**截断**至 512 token；
  - 填充：不足 512 的序列将被右侧**填充**至定长；
  - 列移除：去掉原始 `text` 列，只保留 `input_ids`、`attention_mask` 等。
- **返回值**：张量化后的 `Dataset`。

**实现要点**：
- 主程序中若 `tokenizer.pad_token_id is None`，会将 `pad_token` 对齐为 `eos_token`，避免 `DataCollatorForLanguageModeling` 在无 `pad_token` 时出错。

---

### `run_training`

```python
def run_training(config: FinetuneConfig) -> Dict[str, float]:
    """执行一次最小化监督微调流程。"""
```
**功能**：完成一次从**建库 → 分词 → 加载模型 → 训练 → 评估**的端到端流程。

- **关键步骤**：
  1. 加载分词器并保证 `pad_token` 就绪；
  2. 基于配置构造数据集并分词；
  3. 加载 `AutoModelForCausalLM`；
  4. 构造 `DataCollatorForLanguageModeling(tokenizer, mlm=False)`：
     - `mlm=False` 意味着**因果语言模型**目标（不是 BERT 式 MLM）；
     - collator 会将 **padding 位置的标签置为 `-100`**，从而在损失计算中**屏蔽**这些位置；
  5. 设置 `TrainingArguments`（小步数、适合教学）；
  6. 初始化 `Trainer` 并调用 `train()`；
  7. 用 `trainer.evaluate()` 返回 `metrics`（如 `eval_loss`）。

- **返回值**：`Dict[str, float]`，典型字段：
  - `eval_loss`：评估集/训练集上的平均交叉熵（如果未显式区分验证集，则为对训练数据的 `evaluate` 结果）。

---

### 入口主程序

```python
if __name__ == "__main__":
    config = FinetuneConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 先运行指令微调格式
    instruction_metrics = run_training(config)

    # 再运行传统 SFT 格式做对比
    config.instruction_mode = False
    sft_metrics = run_training(config)

    LOGGER.info(
        "两种格式的损失对比：instruction=%.4f, sft=%.4f",
        instruction_metrics["eval_loss"],
        sft_metrics["eval_loss"],
    )
```
**说明**：固定顺序跑两次，便于直观对比不同数据格式的 `eval_loss`。

---

## 数学原理与损失函数

### 自回归语言模型目标

对因果语言模型（Causal LM），给定输入 token 序列 \(x_{1:T}\)，模型学习条件分布
$$
p_\theta(x_{1:T})=\prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}).
$$

训练最小化**负对数似然（NLL）**，等价于逐位置交叉熵损失之和：
$$
\mathcal{L}_{\text{NLL}}(\theta)
= - \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).
$$

### 交叉熵损失与标签屏蔽

由于采用定长 padding，需在损失中**忽略**填充位置：若将填充位置的标签设为 \(-100\)，则 PyTorch 的 `CrossEntropyLoss` 会自动**跳过**这些位置：
$$
\mathcal{L}(\theta)
= - \sum_{t \in \mathcal{V}} \log p_\theta(x_t \mid x_{<t}),\quad
\mathcal{V}=\{t \mid \text{label}_t \neq -100\}.
$$

这正是 `DataCollatorForLanguageModeling(mlm=False)` 在 Causal LM 训练中的默认行为。

### SFT vs Instruction Tuning 的本质区别

- **SFT（两段式）**：仅将 *输入/问题* 与 *输出/答案* 简单拼接，学习到的是**从输入到输出的映射**：
  $$
  \text{prompt} = \text{"问题："} + x,\quad
  \text{target} = y.
  $$

- **Instruction Tuning（三段式）**：显式提供**任务指令** `instruction`，模型不仅学习输入→输出的映射，还学习**如何遵循指令**：
  $$
  \text{prompt}=\text{"指令："}+i+\text{"输入："}+x+\text{"回答："},\quad
  \text{target} = y.
  $$

经验上，**一致、清晰、结构化的指令模板**通常能带来更好的泛化，尤其在多任务、多领域的小数据教学场景中。

---

## 运行与复现实验

### CPU 直接运行
```bash
python sft_basics.py
```
- 运行结束后在 `outputs/sft_basics/` 下可看到训练产物与日志；
- 终端将打印 Instruction vs SFT 的 `eval_loss` 对比。

### GPU/多卡（可选）
```bash
accelerate launch --num_processes 1 sft_basics.py
# 多卡示例（按需）
accelerate launch --num_processes 4 sft_basics.py
```
> 使用前请先 `accelerate config` 完成环境检测与配置。

### 输出与日志
- 日志采用 `logging`，默认级别 `INFO`；
- 关键日志：数据格式选择、训练开始/结束、`eval_loss`。

---

## 常见问题排查

1. **`pad_token_id is None` 报错或 Loss 异常**
   - 处理：主程序已自动将 `pad_token = eos_token`。若仍异常，检查分词器与模型是否完全匹配。

2. **显存不足 / 内存不足**
   - 处理：减小 `batch_size`、`max_length`、`max_steps`；或切换更小的 `model_name`。

3. **`DataCollatorForLanguageModeling` 行为不符合预期**
   - 确认 `mlm=False`（必须）。若希望更细粒度的 label 掩码，可自定义 collator。

4. **收敛过快/无差异**
   - 该脚本是**最小教学版**，`max_steps` 很小、样本很少，差异可能不明显。可：
     - 增加 `EXAMPLE_PAIRS` 数量；
     - 提高 `max_steps`；
     - 引入更规范的指令模板或系统提示。

5. **评估集与训练集未区分**
   - 该脚本用训练数据直接调用 `evaluate()` 仅为演示；实际项目应切分 `train/valid`。

---

## 扩展建议

- **引入模板**：加入角色标记（如 `<|system|> / <|user|> / <|assistant|>`）与统一 Prompt 模板。
- **加入验证集**：`train_test_split` 划分验证集，监控早停与泛化。
- **LoRA/QLoRA**：以低资源微调更多参数高效模型。
- **指令多样性**：扩展不同任务类型（改写、摘要、分类、结构化抽取等）。
- **评测指标**：除 `eval_loss` 外，增加任务相关指标（BLEU/ROUGE/准确率等）。

---

## 许可证

若无特别声明，沿用上游模型与数据集的原始许可证；示例脚本本身可按教学用途自由修改与再分发（请在文档中保留来源与致谢）。
