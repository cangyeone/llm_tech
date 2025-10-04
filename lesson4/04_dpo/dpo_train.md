# 教程：DPO（Direct Preference Optimization）训练实操

## 学习目标
- 理解 DPO 的损失函数及温度参数 \(\beta\) 的作用。
- 掌握使用 TRL 库的 `DPOTrainer` 对偏好数据执行训练的流程。
- 学会构建参考模型（reference policy）以维持与原模型的 KL 约束。

## 背景原理
DPO 通过对比正样本与负样本的对数似然差距实现对齐，其核心目标为：
\[
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[ \log \sigma\left( \beta \left(\log \pi_{\theta}(y^+|x) - \log \pi_{\theta}(y^-|x)\right) - \left(\log \pi_{\text{ref}}(y^+|x) - \log \pi_{\text{ref}}(y^-|x)\right) \right) \right].
\]
其中 \(\beta\) 调节偏好差异的放大程度，参考模型 \(\pi_{\text{ref}}\) 约束训练后的策略不要偏离原模型太远。

## 代码结构解析
- `build_dataset`：加载并筛选偏好数据，只保留 prompt、chosen、rejected 字段。
- `DPOArguments`：封装模型名称、学习率、batch size 等超参数。
- `DPOMathNotes`：提供公式说明，运行时打印 \(\beta\) 的解释。
- `train_dpo`：
  - 初始化策略模型与参考模型。
  - 创建 `DPOConfig`，设置最大长度、batch、epoch 等。
  - 调用 `DPOTrainer` 执行训练并保存模型。
- `parse_args`：解析命令行参数，支持自定义数据集与温度。

## 实操步骤
1. 将 `dataset_name` 替换为自建偏好数据集，格式需包含 `prompt`、`chosen`、`rejected`。
2. 根据任务难度调节 `beta`：数值越大越强调偏好差异，但过大可能导致过拟合。
3. 运行脚本并关注日志中的损失变化，必要时增加 `num_train_epochs`。
4. 训练完成后，在推理脚本中加载 `./outputs/dpo_policy` 测试生成质量。

## 拓展思考
- 能否将 DPO 与 LoRA/QLoRA 结合，降低训练成本？
- 当偏好数据覆盖度有限时，是否需要混合 SFT 数据以保持多样性？
- 如何监控 KL 距离，防止策略过度偏离参考模型？
