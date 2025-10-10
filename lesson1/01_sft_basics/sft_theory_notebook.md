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
- `FinetuneConfig`：使用 `@dataclass` 定义的超参数集合，包含模型名称、学习率、步数等配置，并提供 `from_instruction()` 辅助方法快速切换训练模式。
- `EXAMPLE_PAIRS`：用于教学的最小标注数据集，覆盖摘要与计划编写两类任务，是后续构建 `datasets.Dataset` 的核心样本来源。
- `build_dataset`：根据 `instruction_mode` 参数决定是否拼接指令信息，输出结构为 `{"instruction", "input", "output"}` 或 `{"prompt", "response"}` 的数据集。
- `tokenize`：调用 `AutoTokenizer` 进行分词、截断与填充，保障样本长度一致，并生成 `labels` 以匹配语言模型的自回归训练目标。
- `run_training`：构建模型、数据整理器与 `Trainer`，执行训练与评估，最终比较两种模式的损失。

## 使用指南
1. **环境准备**：确保安装 `transformers`、`datasets`、`accelerate` 和 `peft` 等依赖，可执行
   ```bash
   pip install -U transformers datasets accelerate peft
   ```
2. **配置模型**：在脚本顶部的 `FinetuneConfig` 中，将 `model_name` 修改为本地可用的基座模型，如 `Qwen/Qwen1.5-0.5B`。
3. **选择训练模式**：设置 `instruction_mode=True` 以启用指令微调模板，或设为 `False` 以执行传统 SFT。
4. **运行脚本**：执行
   ```bash
   python lesson1/01_sft_basics/sft_theory_notebook.py
   ```
   程序会先后运行指令模式与纯 SFT 模式，并在控制台打印 `training_args.output_dir` 中保存的日志路径。
5. **分析结果**：在控制台或保存的 `trainer_state.json` 中对比 `instruction_metrics` 与 `sft_metrics` 的 `eval_loss` 与 `perplexity`，判断模板化对对齐效果的影响。

## 数学原理详解
在自回归语言模型中，训练目标是最大化条件概率 $p_\theta(y \mid x)$，等价于最小化交叉熵损失：

$$
\mathcal{L}_{\text{CE}}(\theta) = - \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x).
$$

梯度通过反向传播更新参数：

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{CE}}(\theta),
$$

其中学习率 $\eta$ 由 `FinetuneConfig.learning_rate` 控制。指令模式下，通过模板化将输入组织为

$$
\text{prompt} = \text{Instruction} \oplus \text{Input} \oplus \text{Response},
$$

从而在注意力机制中显式地为模型提供任务描述。若采用传统 SFT，仅拼接 `Input` 与 `Response`，则概率建模退化为 $p_\theta(y \mid x)$ 的标准形式。训练中 `Trainer` 会在每个批次上累积损失并根据优化器（如 AdamW）规则更新参数，AdamW 的一阶、二阶动量分别按照

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}_t)^2,
\end{aligned}
$$

并使用权重衰减控制参数范数。上述过程在指令与非指令模式下共用，只是 `build_dataset` 决定了输入序列的拼接方式。

## 实验步骤
1. 选定指令或非指令模式并运行脚本，观察日志中的 `loss` 曲线变化。
2. 打开输出目录中的 `events.out.tfevents` 或 `trainer_state.json` 文件，可视化训练与验证损失，验证交叉熵最小化趋势。
3. 修改 `max_steps`、`warmup_steps` 或 `per_device_train_batch_size`，重复实验验证学习率调度与批大小对收敛速度的影响。

## 思考题
- 如果将 `instruction` 与 `input` 合并为单一字段，模型在处理多任务时是否会出现混淆？
- 当样本数量增大时，应如何调整 `max_steps` 与 `warmup_steps`？
- 如何在 GPU 环境下使用 `accelerate launch` 提升训练速度？
- 能否通过在 `EXAMPLE_PAIRS` 中加入负面案例来模拟对抗式指令，从而观察损失曲线的变化？
