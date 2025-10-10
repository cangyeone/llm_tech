# 教程：SFT 与指令微调理论实践

## 学习目标
- 理解监督微调（Supervised Fine-Tuning, SFT）与指令微调（Instruction Tuning）的数据格式差异。
- 学会使用 Hugging Face `Trainer` 在小规模示例上复现两种格式的训练流程。
- 掌握交叉熵损失在语言模型微调中的作用，并能分析损失曲线差异。

## 背景原理
SFT 的目标是最小化模型输出分布与标注答案之间的交叉熵损失：

$$
\mathcal{L}_{\text{SFT}} = - \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x),
$$

其中 $x$ 为输入文本，$y_t$ 为第 $t$ 个目标 token。指令微调在输入序列中显式引入 `instruction` 字段，通过模板化拼接让模型学习对齐的指令遵循行为。本脚本以 Qwen 系列模型为例，演示两种格式对损失的影响。

## 代码结构解析
- `FinetuneConfig`：使用 `@dataclass` 定义的超参数集合，包含模型名称、学习率、步数等配置。
- `EXAMPLE_PAIRS`：用于教学的最小标注数据集，覆盖摘要与计划编写两类任务。
- `build_dataset`：根据 `instruction_mode` 参数决定是否拼接指令信息，输出 `datasets.Dataset` 对象。
- `tokenize`：调用 `AutoTokenizer` 进行分词、截断与填充，保障样本长度一致。
- `run_training`：构建模型、数据整理器与 `Trainer`，执行训练与评估，最终比较两种模式的损失。

## 实验步骤
1. 安装依赖：`pip install transformers datasets accelerate`。
2. 修改 `FinetuneConfig` 中的 `model_name` 为本地可用的 Qwen 或其他中文模型。
3. 运行脚本后将依次完成指令格式与传统 SFT 格式训练，并在日志中输出损失对比。
4. 对比 `instruction_metrics` 与 `sft_metrics` 的 `eval_loss`，分析哪种格式更适合特定任务。

## 思考题
- 如果将 `instruction` 与 `input` 合并为单一字段，模型在处理多任务时是否会出现混淆？
- 当样本数量增大时，应如何调整 `max_steps` 与 `warmup_steps`？
- 如何在 GPU 环境下使用 `accelerate launch` 提升训练速度？
