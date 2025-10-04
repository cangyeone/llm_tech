# 教程：训练日志解析与可视化

## 学习目标
- 熟悉 DeepSpeed/Accelerate 训练日志的 JSONL 结构。
- 学会提取损失、吞吐等关键指标并绘制趋势图。
- 掌握利用 `pandas` 与 `matplotlib` 构建轻量分析工具的方法。

## 背景原理
训练日志通常以逐步记录的形式存储，包含损失 $\mathcal{L}$、吞吐 $\text{samples/s}$ 等字段。通过可视化可以快速判断训练是否收敛：
- 损失曲线应整体下降，波动可反映学习率设置。
- 吞吐曲线可帮助定位 I/O 或通信瓶颈。

## 代码结构解析
- `load_logs`：逐行解析 JSONL 日志，构建 `DataFrame`。
- `plot_metrics`：绘制损失与吞吐趋势，并保存图像 `training_metrics.png`。
- `__main__`：生成示例日志，便于课堂无依赖演示。

## 实践步骤
1. 将训练脚本产生的日志复制到本地，例如 `output/train_log.jsonl`。
2. 修改 `log_path` 指向真实路径并运行脚本。
3. 查看生成的 `training_metrics.png`，讨论损失收敛与性能变化。
4. 可在 `plot_metrics` 中新增 GPU 利用率、显存占用等曲线。

## 进阶问题
- 如何结合 `seaborn` 或 Plotly 创建交互式仪表板？
- 训练日志包含分布式 rank 信息时，应如何聚合？
- 是否可以设置报警阈值，当吞吐降至某个水平以下时触发通知？
