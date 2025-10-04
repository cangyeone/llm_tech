# 教程：构建 FAISS 文档向量库与检索演示

## 学习目标
- 将预处理后的文本片段与嵌入加载到 FAISS 索引中。
- 理解向量归一化、内积检索的实现细节。
- 通过命令行交互演示语义检索结果，验证 RAG 数据管线。

## 背景原理
FAISS 支持多种相似度度量，其中内积（IP）在向量归一化后等价于余弦相似度：
\[
\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|\, \|\mathbf{d}\|}.
\]
将查询和文档向量归一化后，内积越大表示语义越相近。

## 代码结构解析
- `VectorStore` 数据类：封装索引、文档列表与嵌入矩阵，提供 `search` 方法。
- `load_preprocessed_data`：加载 `chunks.txt` 与 `embeddings.npy`。
- `build_faiss_index`：创建 `IndexFlatIP`，并对文档向量执行 L2 归一化。
- `encode_query`：示例性地将查询映射到向量空间，可替换为真实编码模型。
- `interactive_demo`：支持命令行互动查询，输出 top-k 文档。

## 实践步骤
1. 确保已运行 Lesson 6 的预处理脚本，生成文本与向量文件。
2. 构建向量库并进行检索：
   ```bash
   python build_vector_store.py outputs/preprocess/chunks.txt outputs/preprocess/embeddings.npy --top_k 3
   ```
3. 如需体验交互模式，加入 `--interactive`，可实时输入问题。
4. 分析检索结果是否符合预期，必要时调整分块或嵌入模型。

## 拓展问题
- 如何将 FAISS 替换为 Annoy、HNSWlib 等索引结构？
- 对于百万级数据，应如何使用 IVF/PQ 等压缩索引节省内存？
- 若要返回元数据（如文档 ID、标题），可在 `VectorStore` 中增加映射表。
