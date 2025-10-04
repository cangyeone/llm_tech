# 教程：对齐后生成结果的多样性评估

## 学习目标
- 通过多次采样分析模型回答的多样性与模式坍塌风险。
- 掌握 distinct-n 与 self-BLEU 指标的计算方法。
- 根据指标结果调整采样策略（温度、top-k/top-p）。

## 背景原理
- **distinct-n**：衡量输出中不同 n-gram 的比例，数值越高表示多样性越好。
- **self-BLEU**：将每个回答视作假设，其他回答作为参考，得分越低表示回答差异越大。若 self-BLEU 过高，说明输出趋同。

## 代码结构解析
- `DiversityArguments`：配置模型名称、采样数量、最大生成长度与温度。
- `sample_responses`：针对多个任务提示采样多条回答，支持批量生成。
- `distinct_n` / `self_bleu`：分别计算多样性指标，返回浮点数。
- `analyze_diversity`：整合指标并给出调试建议。
- `main`：解析参数、执行评估并打印结果，当 self-BLEU>80 发出警告。

## 实践步骤
1. 根据场景调整 `EVAL_PROMPTS`，覆盖常见任务类型。
2. 运行脚本：
   ```bash
   python diversity_evaluation.py --model Qwen/Qwen3-1.8B-Instruct --samples 5 --temperature 0.9
   ```
3. 解读输出：
   - `distinct_1`、`distinct_2` 低说明重复 token/短语较多，可提高温度或引入惩罚。
   - `self_bleu` 高说明回答相似，尝试 top-p 采样或增加对齐数据多样性。
4. 记录指标随训练阶段的变化，构建多样性监控面板。

## 拓展问题
- 能否引入 `MAUVE`、`n-gram entropy` 等指标获得更全面的多样性评估？
- 如何在对齐训练中加入熵奖励或 diversity loss 直接优化多样性？
- 对于多轮对话，需如何定义上下文级别的多样性指标？
