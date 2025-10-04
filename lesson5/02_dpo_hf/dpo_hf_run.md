# 教程：在 Hugging Face 上运行 DPO 并推送模型

## 学习目标
- 掌握使用 TRL 的 `DPOTrainer` 在 Hugging Face 生态中训练模型的流程。
- 理解账号登录、仓库创建与模型推送的注意事项。
- 学会通过命令行参数灵活控制训练与上传行为。

## 背景原理
DPO 损失通过比较正负回答的对数似然差来最大化偏好一致性。将 DPO 与 Hugging Face Hub 结合，可以实现：
1. 使用公开数据集快速启动训练。
2. 训练完成后直接将权重推送到远程仓库，便于协作与部署。

## 代码结构解析
- `HFArguments`：集中管理模型、数据集、Hub 仓库 ID、超参数与 `push_to_hub` 开关。
- `check_login`：检测本地是否存储 Hugging Face 令牌，提示用户登录。
- `build_dataset`：加载偏好数据并统一列名为 `prompt`、`chosen`、`rejected`。
- `run_training`：
  - 初始化 tokenizer、策略模型与参考模型。
  - 构建 `DPOConfig`，指定输出目录与 Hub 推送配置。
  - 调用 `DPOTrainer` 训练并保存结果，如需上传则调用 `trainer.push_to_hub()`。
- `parse_args` / `main`：提供命令行入口。

## 实践步骤
1. 如需推送到 Hub，先执行 `huggingface-cli login` 获取令牌。
2. 运行示例：
   ```bash
   python dpo_hf_run.py --model Qwen/Qwen3-1.8B-Instruct --dataset Anthropic/hh-rlhf --beta 0.05 --push-to-hub --repo-id yourname/qwen-dpo-demo
   ```
3. 训练完成后，检查 `outputs/qwen_dpo_hf` 文件夹与 Hugging Face 仓库是否同步更新。
4. 在课程中展示日志中的损失变化、`beta` 影响以及推送流程注意事项。

## 拓展问题
- 如何结合 `gradient_accumulation_steps` 与 `per_device_train_batch_size` 适配不同显存？
- 若想上传 LoRA 适配器，应如何在 `push_to_hub` 中指定 `peft` 模型？
- 能否通过 Git LFS 存储额外评估指标或训练日志，提升团队协作效率？
