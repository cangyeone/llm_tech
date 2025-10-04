# 教程：奖励模型训练与指标评估

## 学习目标
- 掌握基于偏好对的数据集训练奖励模型的步骤。
- 理解 Kendall Tau 与胜率（win rate）用于评估奖励模型排序能力。
- 学会自定义 `Trainer` 的损失函数，以成对比较形式优化模型。

## 背景原理
奖励模型旨在学习偏好函数 $r_{\theta}(x, y)$，对同一 prompt 的两个回答进行排序。训练目标常采用 pairwise log-sigmoid 损失：

$$
\mathcal{L} = -\log \sigma\big(r_{\theta}(x, y^+) - r_{\theta}(x, y^-)\big).
$$

评估时可以使用 Kendall Tau 相关系数衡量预测排序与真实偏好的一致性。

## 代码结构解析
- `ScriptArguments`：封装命令行参数，包括模型、数据集、学习率等。
- `load_preference_dataset`：从 Hugging Face Hub 或本地磁盘加载偏好数据。
- `preprocess`：为每条样本生成 `input_ids` 与 `attention_mask`，堆叠为二维张量。
- `make_trainer`：
  - 初始化奖励模型。
  - 使用 `compute_metrics` 计算 Kendall Tau 与胜率。
  - 通过 `compute_loss` 实现成对损失。
- `main`：整合加载、预处理、训练流程。

## 实践步骤
1. 准备偏好数据集（如 `response_j`/`response_k` 字段），必要时进行清洗。
2. 根据显存调整 `per_device_batch_size` 与 `max_length`。
3. 运行脚本后查看日志中的 `kendall_tau` 与 `win_rate`，判断奖励模型是否具备区分能力。
4. 可将训练好的权重用于 RLHF 的奖励函数或单独的质量评估工具。

## 进阶问题
- 如何对奖励模型进行正则化（如 L2、Dropout），防止过拟合？
- 当偏好数据存在噪声时，可否引入温度参数或加权策略缓解？
- 是否可以扩展 `compute_metrics`，加入 AUC 或命中率等指标？
