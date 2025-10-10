# 教程：指令数据预处理与 Prompt 模板设计

## 学习目标
- 熟悉文本数据清洗的常见步骤，包括去噪、长度过滤与模板化转换。
- 掌握如何利用 `dataclasses` 与 `pandas` 构建可复用的预处理流水线。
- 学会将清洗后的样本导出为 JSONL 以适配指令微调。

## 背景原理
在指令微调中，我们通常将原始文档 $d$ 通过预处理函数 $f(\cdot)$ 映射为结构化样本 $(\text{instruction}, \text{input}, \text{output})$。整个流程可抽象为：
$$
(\text{instruction}, \text{input}, \text{output}) = f\big( \text{clean}(d), \text{template} \big),
$$
其中 `clean` 函数负责消除噪声，`template` 用于构建上下文提示，最终生成模型可学习的对齐对话对。合理的模板能提升模型在下游任务上的泛化能力。

## 代码结构解析
- `PreprocessConfig`：集中管理数据目录、输出路径、模板、最小长度等参数，方便在不同场景复用。
- `iter_documents`：递归读取目录下的 `.txt` 与 `.md` 文件，形成原始语料迭代器。
- `clean_text`：通过正则表达式消除多余空白与重复换行，保证上下文整洁。
- `build_examples`：将清洗后的内容裁剪至合理长度，并填充模板生成 instruction-tuning 样本。
- `export_jsonl`：将样本逐行写入 JSONL 文件，是 Hugging Face `datasets` 可直接加载的格式。
- `build_dataframe`：用 `pandas` 快速查看样本分布，为数据检查提供便利。

## 实践指南
1. 将真实语料复制到 `sample_docs` 目录，或在配置中重定向 `data_dir`。
2. 根据任务类型修改 `template` 与 `instruction` 内容，例如问答、总结、信息抽取等。
3. 运行脚本后检查控制台输出的样本数量与 `df.head()` 结果，确保标签字段完整。
4. 将生成的 `preprocessed.jsonl` 导入微调脚本，配合 Tokenizer 进一步处理。

## 进阶思考
- 若想实现多任务混合，可在 `examples` 中加入 `task_type` 字段，训练时使用条件模板。
- 可以结合 SentencePiece 或正则分句对长文档进行分块，再拼接模板生成更多样本。
- 如何将 `min_length` 与 `max_length` 动态化，使得短内容能够通过补充背景信息满足长度要求？
