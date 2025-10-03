# Lesson 5 实验：对齐系统落地与监控

本课聚焦于将 Qwen3 模型部署到真实业务中的对齐策略，涵盖 Hugging Face 平台上的 DPO 实践、
偏差检测、安全合规审查，以及模型版本管理与客服场景优化等内容。所有脚本默认使用中文注释，
便于在课堂或自学时快速理解实现思路。

## 环境依赖

- Python 3.10+
- transformers、datasets、trl、peft
- evaluate、scikit-learn、pandas
- mlflow、wandb（可选，脚本提供离线模式）

建议执行：

```bash
pip install -U "transformers>=4.39" datasets "trl>=0.8" peft accelerate evaluate scikit-learn pandas mlflow wandb
```

## 目录总览

1. `01_qwen_workflow/qwen_alignment_workflow.py` — Qwen 对齐流程全景演练。
2. `02_dpo_hf/dpo_hf_run.py` — 直接在 Hugging Face 上复现 DPO 训练。
3. `03_bias_detection/bias_scan.py` — 对齐结果的偏差检测与安全审查。
4. `04_model_versioning/version_tracking.py` — 使用 MLflow 与 W&B 管理模型版本。
5. `05_customer_support_case/customer_alignment.py` — 客服场景对齐策略案例分析。
6. `06_diversity_debug/diversity_evaluation.py` — 对齐后回答多样性评估与调试。

运行脚本前，请按照自身环境调整路径、账号与超参数，确保遵循企业数据合规要求。
