# 教程：P-Tuning v2 可学习提示词实践

## 学习目标
- 理解 P-Tuning v2 通过可学习虚拟 token 调整输入分布的原理。
- 掌握 `peft` 中 `PromptEncoderConfig` 的关键参数配置方法。
- 熟悉训练阶段与推理阶段调用方式的差异。

## 背景原理
P-Tuning v2 在输入嵌入前插入一段可训练的虚拟提示向量 $P \in \mathbb{R}^{m \times d}$。对于输入 token 表示 $X$，模型实际接收的是 $[P; X]$。训练目标为：
$$
\min_{P} \mathcal{L}(f_{\theta}(P, X), Y),
$$
其中模型参数 $\theta$ 可选择固定或局部调整。通过学习虚拟 token，可以在保持主模型冻结的情况下完成任务特化。

## 代码结构解析
- `PTuningConfig`：设置模型名称、虚拟 token 长度、任务类型等。
- `build_dataset`：构造客服问候语的示例数据集，便于课堂演示。
- `format_sample` / `tokenize`：统一输入格式并完成分词。
- `PromptEncoderConfig`：指定虚拟 token 数量、隐层维度以及编码器大小。
- `Trainer` 流程：执行一次 epoch 的训练并输出推理示例。

## 实操步骤
1. 安装依赖：`pip install transformers datasets peft`。
2. 可将 `build_dataset` 替换为真实指令数据，确保字段齐全。
3. 调节 `prompt_length` 观察虚拟 token 长度对任务效果的影响。
4. 推理阶段直接调用 `model.generate`，无需额外传入提示向量，`peft` 会自动注入。

## 进一步思考
- 如果想在推理阶段关闭提示，可以调用 `model.disable_adapter()`，验证输出差异。
- 是否需要对虚拟 token 进行正则化（例如 L2 惩罚）以防止过拟合？
- 与 LoRA 相比，P-Tuning v2 的可训练参数更少，但能否兼容并行使用提升效果？
