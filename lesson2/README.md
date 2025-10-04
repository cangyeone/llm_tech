# Lesson 2 实验：QLoRA、P-Tuning 与模型压缩

本目录聚焦第二课的六个主题，涵盖 QLoRA 理论、实操流程、P-Tuning v2、模型压缩以及部署策略选择。脚本均配有中文注释，便于理解量化、提示学习与压缩技术的核心思想。

## 环境依赖

- Python 3.10+
- transformers
- datasets
- peft
- bitsandbytes（QLoRA 与低比特量化）
- accelerate
- deepspeed（部分脚本示例需用）

安装参考命令：

```bash
pip install -U transformers datasets peft bitsandbytes accelerate deepspeed
```

> **提示**：QLoRA 示例默认使用 `meta-llama/Llama-2-7b-chat-hf`，请提前在 Hugging Face 完成协议授权。

## 目录总览

1. `01_qlora_intro/qlora_theory.py` — 解析 QLoRA 的量化、分页存储与梯度检查点。
2. `02_qlora_pipeline/qlora_trainer.py` — LLaMA-7B 的 QLoRA 微调流程脚手架。
3. `03_ptuning/ptuning_v2_demo.py` — P-Tuning v2 可学习提示词实现。
4. `04_model_compression/compression_lab.py` — 剪枝、蒸馏与低比特量化实验对比。
5. `05_deepspeek_quant/deepspeek_deploy.py` — 基于 DeepSpeed + bitsandbytes 的量化部署。
6. `06_method_selection/method_selector.py` — LoRA/QLoRA/P-Tuning 场景选择助手。

如需在多 GPU 环境运行，请结合 `accelerate` 或 `deepspeed` 启动命令，并根据显存情况调整 batch size 与梯度累积参数。
