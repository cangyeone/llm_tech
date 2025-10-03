# Lesson 1 实验：监督与指令微调基础

本目录围绕课程第一课的六个核心主题，提供配套的实验脚本与说明，聚焦监督微调（SFT）、指令数据构建、LoRA 机制以及轻量模型在 CPU 端的部署示例。所有脚本均包含中文注释，便于课堂演示与自学复现。

## 环境依赖

- Python 3.10+
- [Transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [peft](https://github.com/huggingface/peft)
- [accelerate](https://github.com/huggingface/accelerate)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)（可选，用于低比特训练）
- [sentencepiece](https://github.com/google/sentencepiece)（ChatGLM 等模型依赖）

可以使用如下命令安装：

```bash
pip install -U transformers datasets peft accelerate bitsandbytes sentencepiece
```

> **提示**：脚本示例默认使用 Hugging Face Hub 上的轻量开源模型（如 `THUDM/chatglm3-6b` 或 `Qwen/Qwen2-1.5B-Instruct`），请根据实际算力调整模型名称与批量大小。

## 目录总览

1. `01_sft_basics/sft_theory_notebook.py` — 解析 SFT 与指令微调的理论差异，并给出最小化训练示例。
2. `02_data_preprocess/data_pipeline.py` — 将文档批量转文本、构建 Prompt 模板与清洗数据。
3. `03_hf_finetune/hf_finetune_scaffold.py` — 基于 Hugging Face Trainer 的微调脚手架。
4. `04_lora_theory/lora_parameter_viz.py` — 展示 LoRA 秩分解、参数注入位置与可视化。
5. `05_lora_finetune/lora_chatglm_demo.py` — 面向 ChatGLM-6B 的 LoRA 微调示例。
6. `06_cpu_deploy/cpu_inference_case.py` — 对比轻量模型在 CPU 上的吞吐表现与部署建议。

运行脚本前，请先在 `config` 或脚本顶部调整数据路径、模型名称与硬件参数。每个脚本均附带日志输出或可视化，帮助理解对应知识点。
