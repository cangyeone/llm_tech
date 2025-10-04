# 教程：LoRA 增量学习与灾难性遗忘缓解

## 学习目标
- 掌握跨领域增量训练 LoRA 适配器的流程。
- 理解重放样本（replay）与 KL 约束在缓解灾难性遗忘中的作用。
- 学会使用自定义损失在第二阶段训练时对旧模型输出进行正则。

## 背景原理
当模型先后接触不同领域数据时，易出现灾难性遗忘。常见缓解策略包括：
1. **样本重放**：从旧领域抽取部分样本与新数据混合训练。
2. **知识蒸馏/KL 正则**：保持新模型输出与旧模型一致，损失形式：
$$
\mathcal{L}_{\text{KL}} = \lambda \cdot \mathrm{KL}\big(\pi_{\text{new}}(y|x) \Vert \pi_{\text{old}}(y|x)\big).
$$
## 代码结构解析
- `IncrementalArgs`：配置模型名称、领域数据集、LoRA 秩、重放比例等。
- `load_qa_dataset`：从公开数据集中抽样并格式化为问答对。
- `make_lora_model`：加载基础模型并注入 LoRA 适配器。
- `train_stage`：定义自定义损失，若提供 `ref_model` 则加入 KL 正则。
- `incremental_learning`：
  - 第一阶段在领域 A 上训练。
  - 第二阶段加载领域 B，并混合重放样本。
  - 冻结复制的参考模型参与 KL 约束。

## 实践步骤
1. 替换 `domain_a`、`domain_b` 为真实任务（如客服问答 → 财经问答）。
2. 根据数据差异调整 `replay_ratio` 与 `kl_coeff`，平衡新旧知识。
3. 运行脚本后，保存的 `./outputs/incremental_lora` 包含增量适配器，可在推理时加载。
4. 通过对比增量前后的旧领域性能，验证灾难性遗忘是否缓解。

## 深入讨论
- 如果旧领域数据保密，是否可以使用旧模型生成伪样本作为重放数据？
- KL 系数过大可能抑制新知识学习，应如何动态调整？
- 是否可以结合 Elastic Weight Consolidation (EWC) 等正则进一步提升稳定性？
