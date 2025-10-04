# 教程：8 卡分布式 DeepSpeed 配置说明

## 学习目标
- 理解流水线并行与张量并行的组合方式。
- 掌握 ZeRO-3 与并行策略同时启用时的配置要点。
- 学会生成适用于 8 卡训练的 DeepSpeed JSON 文件。

## 背景原理
当模型规模超过单卡显存时，可以采用 \(\text{TP}=4\) 与 \(\text{PP}=2\) 的组合。总设备数为 \(\text{TP} \times \text{PP} = 8\)。ZeRO-3 进一步在每个并行组内切分参数，激活检查点减少前向存储需求。

## 代码结构解析
- `CONFIG` 字典：包含微批量大小、梯度累积、bf16、ZeRO 设置、并行参数等。
- `pipeline.parallel_size` 与 `tensor.parallel_size`：分别指定流水线与张量并行度。
- `activation_checkpointing`：开启激活分区，实现显存节省。
- `__main__`：将配置写入 `ds_config_8gpu.json`，供训练脚本引用。

## 使用步骤
1. 修改 `train_micro_batch_size_per_gpu` 与 `gradient_accumulation_steps` 以匹配实际显存。
2. 根据模型层数重新划分流水线阶段，确保负载均衡。
3. 运行脚本生成配置文件，在训练命令中添加 `--deepspeed ds_config_8gpu.json`。
4. 可进一步扩展配置，加入 `wall_clock_breakdown`、`optimizer`、`scheduler` 等字段。

## 深度思考
- 在多节点场景下，如何为每个节点指定本地并行度与通信拓扑？
- 流水线并行可能带来 bubble 时间，如何通过增加 micro batch 缓解？
- 若模型包含 MoE 结构，还需额外配置专家并行（EP），如何与 TP/PP 协同？
