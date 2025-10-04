# 教程：LoRA 秩分解与参数注入可视化

## 学习目标
- 理解 LoRA（Low-Rank Adaptation）的核心思想和数学表达式。
- 学会通过随机矩阵模拟不同秩 \(r\) 对重建误差的影响。
- 掌握如何在 Transformer 线性层插入 LoRA 适配器的伪代码实现。

## 背景原理
LoRA 将原始权重 \(W_0 \in \mathbb{R}^{d \times k}\) 进行低秩增量调整：
\[
W = W_0 + \Delta W, \quad \Delta W = B A, \quad A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r},
\]
并通过缩放因子 \(\alpha\) 控制更新幅度。通过限制秩 \(r \ll \min(d, k)\)，LoRA 可显著降低可训练参数量，同时保持表达能力。

## 代码结构解析
- `LoraConfig`：配置隐层维度、候选秩 `rank_candidates` 与缩放参数 `alpha`。
- `simulate_lora`：随机生成基线权重，并计算不同秩下的 Frobenius 范数误差，最后输出并保存曲线图。
- `pseudo_code`：打印 LoRA 在线性层中的典型插入伪代码，帮助理解前向传播流程。

## 实验流程
1. 安装 `matplotlib`：`pip install matplotlib`。
2. 运行脚本后将在控制台看到每个 rank 的重建误差，并在当前目录生成 `lora_rank_error.png`。
3. 调整 `rank_candidates` 观察秩越大误差越小但参数量越高的权衡关系。
4. 阅读伪代码片段，结合课程 PPT 对照 LoRA 注入位置。

## 深入讨论
- 尝试将 `simulate_lora` 中的随机权重替换为真实模型层权重，对比误差随秩变化的趋势。
- 结合缩放 \(\alpha\) 与学习率设计，分析 LoRA 更新幅度如何影响训练稳定性。
- 如果要应用到注意力模块的 Query、Value 投影，该脚本需要做哪些参数维度上的调整？
