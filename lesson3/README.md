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

## 基础教程 
```bash 
# 纯 GPU ZeRO-3
deepspeed --num_gpus=8 train.py \
  --model_name gpt2 \
  --deepspeed ds_zero3.json \
  --output_dir outputs_zero3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --fp16 \
  --grad_ckpt
```
```bash
# ZeRO-3 + CPU Offload（更省显存）
deepspeed --num_gpus=8 train.py \
  --model_name gpt2 \
  --deepspeed ds_zero3_offload.json \
  --output_dir outputs_zero3_offload \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --fp16 \
  --grad_ckpt
``` 


主节点执行：
```bash
export MASTER_ADDR=10.0.0.10   # 主节点 IP/主机名
export MASTER_PORT=29500
export NNODES=2
export GPUS_PER_NODE=8
export NODE_RANK=0
deepspeed --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  --num_nodes $NNODES --num_gpus $GPUS_PER_NODE \
  train.py --deepspeed ds_zero3.json \
  --model_name gpt2 \
  --output_dir /shared/outputs_zero3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --fp16 --grad_ckpt
```

从节点执行：
```bash
export MASTER_ADDR=10.0.0.10   # 主节点 IP/主机名
export MASTER_PORT=29500
export NNODES=2
export GPUS_PER_NODE=8
export NODE_RANK=1
deepspeed --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  --num_nodes $NNODES --num_gpus $GPUS_PER_NODE \
  train.py --deepspeed ds_zero3.json \
  --model_name gpt2 \
  --output_dir /shared/outputs_zero3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --fp16 --grad_ckpt
```


或者直接执行
```bash 
deepspeed --hostfile=hostfile \
  --launcher=openssh \
  train.py \
  --deepspeed ds_zero3.json \
  --model_name gpt2 \
  --output_dir /shared/outputs_zero3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --fp16 --grad_ckpt
``` 

```hostfile 
# 主机名
nodeA slots=8
nodeB slots=8

# FQDN
gpu01.cluster.local slots=8
gpu02.cluster.local slots=8

# 直接用 IP
10.0.0.11 slots=8
10.0.0.12 slots=8

# 指定用户
user@10.0.0.11 slots=8
user@gpu02.cluster.local slots=8
```