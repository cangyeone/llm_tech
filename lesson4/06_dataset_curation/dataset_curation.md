# 教程：对齐数据集清洗与质量分析

## 学习目标
- 学会对偏好数据集进行长度过滤、重复检测与多样性分析。
- 掌握 TF-IDF + 余弦相似度用于发现冗余样本的方法。
- 理解如何导出清洗结果并可视化长度分布。

## 背景原理
高质量对齐数据集应满足长度适中、语义多样、偏好信号清晰等条件。通过 TF-IDF 表示文本并计算余弦相似度，可以识别高度相似的样本：

$$
\text{sim}(x_i, x_j) = \frac{\langle \mathbf{t}_i, \mathbf{t}_j \rangle}{\|\mathbf{t}_i\|\, \|\mathbf{t}_j\|}.
$$

若平均相似度超过阈值，说明样本可能重复或缺乏多样性。

## 代码结构解析
- `CurationArgs`：配置数据集名称、长度阈值、相似度阈值、输出路径。
- `load_pairs`：从 Hugging Face Hub 或本地加载偏好数据，统一字段。
- `filter_by_length`：过滤过短或过长的 prompt/回答，保证可训练性。
- `compute_similarity_flags`：使用 `TfidfVectorizer` 计算平均相似度并打标。
- `visualize_distribution`：绘制 prompt 长度直方图，直观展示分布。
- `curate`：串联上述步骤，导出 CSV 并打印统计信息。

## 实践步骤
1. 根据课程需求设置 `--dataset` 与 `--split`，可在本地缓存后重复使用。
2. 调整 `--min-length`、`--max-length` 适配不同任务；适度提高 `--sim-threshold` 减少误报。
3. 运行脚本后在 `outputs/curated_pairs.csv` 查看筛选结果，并结合标记 `is_redundant` 做人工复核。
4. 使用 `visualize_distribution` 生成的图表与学员讨论数据分布是否合理。

## 拓展问题
- 是否可以用嵌入模型（如 bge, sentence-BERT）替换 TF-IDF 提升语义检测能力？
- 对于极端短文本，是否需要定制模板或合成补充信息？
- 如何将清洗流程自动化嵌入数据标注平台，实现持续监控？
