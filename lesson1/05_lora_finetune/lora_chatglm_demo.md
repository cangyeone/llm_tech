# 教程：ChatGLM-6B 的 LoRA 微调实践

## 学习目标
- 掌握在 `peft` 框架下为 ChatGLM 系列模型配置 LoRA 适配器的步骤。
- 理解 LoRA 训练参数（秩 $r$、缩放 $\alpha$、dropout）的含义及取值建议。
- 学会在 INT4 量化权重上继续执行低秩微调并保存增量权重。

## 背景原理
LoRA 将原始权重冻结，仅更新增量部分 $\Delta W = B A$。在训练中，损失对可训练参数的梯度为：
$$
\frac{\partial \mathcal{L}}{\partial A} = B^\top \frac{\partial \mathcal{L}}{\partial \Delta W}, \qquad
\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial \Delta W} A^\top.
$$
这种分解显著降低参数规模，适合对大模型进行快速定制。

## 代码结构解析
- `LoRAFinetuneConfig`：集中定义模型名称、LoRA 超参数、训练步数等。
- `build_dummy_dataset`：构造演示用的小型问答数据集，真实场景可替换为课程自制数据。
- `format_sample` 与 `tokenize`：将指令对拼接成单轮问答，并统一分词长度。
- `main`：
  - 加载 ChatGLM INT4 权重并指定 `trust_remote_code`。
  - 设置 `LoraConfig` 目标模块 `query_key_value`、`dense` 等注意力层。
  - 通过 `Trainer` 执行训练、评估与模型保存。

## 操作步骤
1. 安装依赖：`pip install transformers datasets peft bitsandbytes`。
2. 准备真实指令数据并替换 `build_dummy_dataset` 的逻辑。
3. 根据显存情况调节 `batch_size` 与 `gradient_accumulation_steps`，确保 `max_steps` 足够收敛。
4. 训练完成后，在推理脚本中加载基础模型 + LoRA 适配器进行增量推断。

## 进阶问题
- 如果显存不足，可将 `load_in_4bit` 改为 `load_in_8bit`，但需要调节学习率保持稳定。
- 如何使用 `model.save_pretrained` 与 `peft` 的 `merge_and_unload` 将 LoRA 权重合并回原模型？
- 若要针对多轮对话做对齐，需要如何扩展 `format_sample` 的对话模板？
