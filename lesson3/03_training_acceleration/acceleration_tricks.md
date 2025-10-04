# 教程：训练加速技巧与内存优化

## 学习目标
- 理解 FlashAttention API 在自注意力计算上的加速原理。
- 学会使用 `torch.compile` 对前向网络进行图优化。
- 掌握常见的内存优化策略，为大模型训练提供参考。

## 背景原理
1. **FlashAttention**：通过块状计算与读写优化，将注意力复杂度从 \(\mathcal{O}(L^2)\) 的显存开销降至 \(\mathcal{O}(L)\)，并利用 GPU 共享内存实现高效访问。
2. **`torch.compile`**：PyTorch 2.x 的动态图编译器，利用 AOTAutograd、Inductor 等技术生成优化后的计算图，减少 Python 调度开销。
3. **内存优化**：梯度检查点、低精度训练、参数分片等策略可以显著降低显存占用。

## 代码结构解析
- `flash_attention_demo`：构造随机 Q/K/V 张量，通过 `scaled_dot_product_attention` 测试耗时。
- `compile_speedup`：创建简单前馈网络，使用 `torch.compile` 编译并测量平均耗时。
- `print_memory_tips`：汇总内存优化建议，便于课堂讨论。

## 实践步骤
1. 确保 PyTorch>=2.0 并支持 FlashAttention API。
2. 运行脚本观察日志输出，记录 FlashAttention 与 `torch.compile` 的性能。
3. 尝试在实际模型中替换自注意力实现，并启用 `torch.compile` 验证收益。
4. 根据实验情况，选择合适的内存优化策略组合。

## 讨论与拓展
- 对长序列任务，FlashAttention 如何与 KV Cache 协同？
- `torch.compile` 在训练阶段需关注哪些限制（如不支持的算子）？
- 是否可以将梯度检查点与 ZeRO 分片结合，以进一步减少内存占用？
