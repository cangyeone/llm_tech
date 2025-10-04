# 教程：RLHF 奖励模型训练与 PPO 微调流程

## 学习目标
- 理解 RLHF 的完整链路：偏好数据构造 → 奖励模型训练 → PPO 策略优化。
- 掌握奖励模型的成对损失函数与 PPO 中的策略更新配置。
- 学会在教学中使用 `--dry-run` 快速演示流程，或加载真实数据执行全量训练。

## 背景原理
1. **奖励模型训练**：基于偏好对 $(x, y^+, y^-)$，优化目标为：

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r_{\theta}(x, y^+) - r_{\theta}(x, y^-)).
$$

2. **PPO 策略更新**：在 KL 约束下最大化奖励：

$$
\max_{\pi} \mathbb{E}_{y \sim \pi}[r(y)] - \beta \mathrm{KL}(\pi \Vert \pi_{\text{ref}}).
$$

通过价值头估计优势函数，迭代更新策略。

## 代码结构解析
- `build_synthetic_preference_dataset`：生成教学用偏好对。
- `RewardModelConfig` / `train_reward_model`：训练 `AutoModelForSequenceClassification`，使用成对损失。
- `build_ppo_trainer`：初始化带价值头的策略模型、参考模型与奖励函数。
- `run_rlhf`：串联奖励模型训练与 PPO 环节，支持 `dry_run` 模式只训练奖励模型。
- `parse_args`：提供命令行配置，便于课堂演示。

## 实践步骤
1. 准备真实偏好数据替换合成数据，并调整 `max_length`、`batch_size` 等超参数。
2. 先运行 `--dry-run` 验证奖励模型训练是否正常，再移除该参数执行完整 PPO。
3. 观察 `ppo_trainer.step()` 输出的 KL、奖励、优势等指标，用于评估训练稳定性。
4. 训练完成后，保存策略与 tokenizer，在推理脚本中加载 `./outputs/ppo_policy` 进行对话测试。

## 深入讨论
- 奖励模型是否需要正则化或裁剪，避免极端奖励导致 PPO 不稳定？
- 如何在奖励函数中融合多维信号（安全性、礼貌性）？
- 当偏好数据稀缺时，可否结合 SFT 预训练的 logits 作为先验，减缓过拟合？
