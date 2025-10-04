# 教程：Hugging Face Trainer 微调脚手架

## 学习目标
- 了解如何使用命令行参数快速配置微调实验。
- 掌握将 JSONL 指令数据集加载为 `datasets.Dataset` 并完成训练/验证切分。
- 熟悉 `Trainer` 的关键参数，如学习率、评估步数与数据整理器配置。

## 背景原理
监督微调的训练目标仍是最小化交叉熵损失。通过 `Trainer` 封装的优化器与调度器，我们可以在固定迭代次数内更新参数：
\[
\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \mathcal{L}_{\text{SFT}}(\theta^{(t)}),
\]
其中 \(\eta\) 为学习率。脚手架为课程提供统一入口，保证实验可重复。

## 代码结构解析
- `CliConfig` 与 `parse_args`：解析命令行输入，支持自定义数据路径、模型、输出目录及超参数。
- `load_dataset`：读取 JSONL 文件并转为 `Dataset`，方便进一步处理。
- `format_example`：将 instruction/input/output 拼接成模型可学习的 prompt。
- `tokenize_dataset`：统一调用 `tokenizer`，设置最大长度与填充策略。
- `main`：完成数据切分、模型加载、训练参数配置、训练与评估全过程。

## 使用说明
1. 准备 JSONL 指令数据，字段需包含 `instruction`、`input`、`output`。
2. 执行示例命令：
   ```bash
   python hf_finetune_scaffold.py --data data.jsonl --model Qwen/Qwen1.5-0.5B --lr 1e-5 --epochs 3
   ```
3. 查看日志中训练/验证损失，必要时修改 `eval_steps` 或 `batch` 以适配硬件。
4. 如果要在多卡环境运行，可改用 `accelerate launch` 或 `torchrun` 包裹脚本。

## 拓展思考
- 如何将 `TrainingArguments` 中的 `lr_scheduler_type` 调整为余弦衰减以获得更平滑的收敛？
- 可以将数据增强逻辑（如同义句替换）融合到 `tokenize_dataset` 之前的映射流程中吗？
- 若要保存最佳模型权重，应开启 `save_strategy="steps"` 并结合 `load_best_model_at_end`，如何选择合适的 `save_steps`？
