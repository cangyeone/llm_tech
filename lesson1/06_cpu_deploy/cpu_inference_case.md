# 教程：轻量模型的 CPU 推理优势分析

## 学习目标
- 理解不同规模模型在 CPU 上的推理延迟差异。
- 学会通过模拟实验估算 tokens/s 指标，评估部署可行性。
- 掌握常见的 CPU 加速策略，如 `torch.compile`、ONNX Runtime 与低比特量化。

## 背景原理
推理延迟近似与模型参数量、层数和算力成正比。对于序列长度 $L$ 与隐藏维度 $d$，单层注意力复杂度约为 $\mathcal{O}(L^2 d)$。轻量模型通过降低 $d$ 与层数，使得 CPU 场景中延迟显著降低。Tokens per second 指标计算为：
$$
\text{TPS} = \frac{L \times N_{\text{steps}}}{T_{\text{total}}},
$$
其中 $T_{\text{total}}$ 为总耗时。

## 代码结构解析
- `DeployCandidate` 与 `CANDIDATES`：定义比较的模型规格（0.5B 与 3B）。
- `build_dummy_model`：基于 `AutoConfig` 构造随机权重，用于模拟结构复杂度。
- `benchmark`：在 CPU 上重复执行前向计算，统计平均延迟与吞吐。
- `main`：遍历候选模型，打印对比结果并给出部署建议。

## 实践指南
1. 若要评估真实模型，可将 `build_dummy_model` 替换为 `from_pretrained`，并确保安装所需权重。
2. 可调节 `seq_length` 和 `steps` 观察不同上下文长度下的性能变化。
3. 对于 3B 以上模型，推荐结合 `torch.int8` 或 `bitsandbytes` 量化后再执行对比。
4. 将输出的建议作为课程讨论的起点，扩展到多线程、批量生成等优化策略。

## 思考题
- 如果在同一台服务器上部署多个轻量模型，如何调度线程和内存以避免争用？
- `torch.compile` 在 CPU 上的收益是否稳定？可以尝试与 ONNX Runtime Benchmark 对比。
- 如何设计自动化脚本定期测试 TPS，追踪模型升级或参数修改对性能的影响？
