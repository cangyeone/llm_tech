# Lesson 3 实验：分布式训练与性能调优

第三课聚焦大模型分布式训练、训练加速与评估。此目录提供六个脚本，覆盖 ZeRO-3、混合精度、FlashAttention、日志分析、评估指标以及垂直领域案例。

## 环境依赖

- Python 3.10+
- torch>=2.1
- deepspeed
- accelerate
- transformers
- flash-attn（可选，加速注意力）
- pandas / matplotlib（日志分析）

## 目录总览

1. `01_distributed_training/zero3_configurator.py` — 生成 ZeRO-3 与混合精度配置。
2. `02_8gpu_config/ds_config_builder.py` — 8 卡 671B-DS 分布式训练配置样例。
3. `03_training_acceleration/acceleration_tricks.py` — FlashAttention 与内存优化技巧演示。
4. `04_log_analysis/training_log_inspector.py` — 日志解析与性能调优指标可视化。
5. `05_evaluation/evaluation_metrics.py` — 困惑度与人工评估表格生成。
6. `06_vertical_case/domain_finetune_case.py` — 文档摘要场景的垂直领域微调案例。

根据硬件资源适当调整配置文件中的批大小与并行策略。部分脚本使用伪造数据用于课堂演示。
