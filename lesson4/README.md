# Lesson 4 实验：大模型对齐与增量学习

本目录包含六个与课程内容对应的实验脚本，均以 Qwen3 系列模型为主线，覆盖 RLHF、DPO、KTO 等核心对齐技术以及增量学习与数据构建方法。每个小节都提供了可以在 GPU 环境中运行的示例代码，并附带关键步骤说明，方便在课堂或自学场景中复现。

## 环境依赖

- Python 3.10+
- [Transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [trl](https://github.com/huggingface/trl)
- [peft](https://github.com/huggingface/peft)
- [accelerate](https://github.com/huggingface/accelerate)
- [scikit-learn](https://scikit-learn.org/)（用于数据分析）

可以使用如下命令安装：

```bash
pip install -U "transformers>=4.39" datasets "trl>=0.8" peft accelerate scikit-learn
```

> **提示**：示例脚本默认使用 Hugging Face Hub 上的 `Qwen/Qwen3-1.8B-Instruct` 轻量模型，便于在单卡环境中演示。若有更强算力，可替换为更大的 Qwen3 权重。

## 目录总览

1. `01_alignment_overview/compare_methods.py` — RLHF、DPO、KTO 机制对比与可视化。
2. `02_rlhf_pipeline/rlhf_pipeline.py` — RLHF 流程串联脚本（奖励模型训练 + PPO 微调）。
3. `03_reward_model_training/train_reward_model.py` — 基于偏好数据的奖励模型微调实操。
4. `04_dpo/dpo_train.py` — 无需强化学习的 DPO 训练脚本与数学公式推导注释。
5. `05_incremental_learning/incremental_lora.py` — 增量学习、灾难性遗忘缓解策略示例。
6. `06_dataset_curation/dataset_curation.py` — 对齐数据集构建与质量分析工具。

在运行脚本前，请根据自身数据路径和资源情况调整超参数。每个文件均附带注释，帮助理解实验逻辑。
