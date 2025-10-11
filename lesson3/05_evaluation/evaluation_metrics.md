# 计算 PPL 与人工评估表生成：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`lesson3/05_evaluation/evaluation_metrics.py`（本文档基于你提供的教学代码撰写）  
> 运行环境：Python ≥ 3.9；`transformers`、`torch`（GPU/MPS 优先，但 CPU 也可运行）  
> 目标：**一键完成两件事**  
> 1) 计算语言模型在一组文本上的**困惑度（Perplexity, PPL）**（支持**长文本滑窗**、自动设备与 dtype、`pad_token` 兜底）；  
> 2) **生成人工评估 CSV**（不依赖 pandas），内置多维度评分列与“实例题目”。

---

## 目录
- [计算 PPL 与人工评估表生成：使用说明与函数文档](#计算-ppl-与人工评估表生成使用说明与函数文档)
  - [目录](#目录)
  - [一、快速开始](#一快速开始)
    - [安装依赖](#安装依赖)
    - [最小用法](#最小用法)
  - [二、整体流程](#二整体流程)
  - [三、命令行参数](#三命令行参数)
  - [四、函数与数据结构文档](#四函数与数据结构文档)
    - [`EvalConfig`（dataclass）](#evalconfigdataclass)
    - [`auto_device() -> torch.device`](#auto_device---torchdevice)
    - [`to_torch_dtype(dtype: Optional[str]) -> Optional[torch.dtype]`](#to_torch_dtypedtype-optionalstr---optionaltorchdtype)
    - [`read_lines(path: str) -> List[str]`](#read_linespath-str---liststr)
    - [`compute_text_nll_sliding(...) -> float`](#compute_text_nll_sliding---float)
    - [`compute_ppl(texts, model_name, max_length, stride, device, dtype) -> float`](#compute_ppltexts-model_name-max_length-stride-device-dtype---float)
    - [`build_human_eval_csv(out_csv_path, prompts, aspects=("帮助性","事实性","风格/礼貌","一致性"), encoding="utf-8-sig")`](#build_human_eval_csvout_csv_path-prompts-aspects帮助性事实性风格礼貌一致性-encodingutf-8-sig)
    - [`parse_args()` \& `main()`](#parse_args--main)
  - [五、PPL 定义与滑窗评估细节](#五ppl-定义与滑窗评估细节)
  - [六、CSV 字段说明与评审指引](#六csv-字段说明与评审指引)
  - [七、常见问题（FAQ）](#七常见问题faq)
  - [八、扩展建议](#八扩展建议)
  - [九、许可证](#九许可证)

---

## 一、快速开始

### 安装依赖
```bash
pip install torch transformers
```

### 最小用法
```bash
python lesson3/05_evaluation/evaluation_metrics.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --out_csv human_eval_template.csv \
  --texts_file texts.txt
```
可选：传入 `--prompts_file prompts.txt`（每行一个题目），将写入 CSV 的 `prompt` 列。

运行后你将得到：
- `human_eval_template.csv`：待人工填写的评估表；  
- 终端打印困惑度：`[OK] 困惑度：xx.xxxx`。

> 若未提供 `texts.txt`，脚本会用 `EvalConfig` 中的**内置样例文本**做演示。

---

## 二、整体流程

1. **解析参数** → 准备文本与题目（若用户未提供则使用内置示例）；  
2. **生成 CSV**（包含多维度评分列）；  
3. **加载 Tokenizer & Model**（自动设备/精度） → **滑窗评估** → 打印 **PPL**。

---

## 三、命令行参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--model_name` | HF Hub 模型名 / 本地路径 | `Qwen/Qwen3-0.6b` |
| `--texts_file` | PPL 文本（每行一条） | `None`（用内置示例） |
| `--prompts_file` | 人工评估题目（每行一条） | `None`（用内置示例） |
| `--out_csv` | 输出 CSV 文件路径 | `human_eval_template.csv` |
| `--max_length` | 滑窗窗口 token 上限 | `1024` |
| `--stride` | 滑窗步长（重叠 = `max_length - stride`） | `768` |
| `--device` | `"cuda"` / `"mps"` / `"cpu"`（默认自动） | `None` |
| `--dtype` | `"fp16"` / `"bf16"` / `"fp32"`（默认自动） | `None` |

**建议**：GPU 上优先 `bf16`；若不支持，再选 `fp16`。

---

## 四、函数与数据结构文档

### `EvalConfig`（dataclass）
```python
@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen3-4b"
    texts: List[str] = None  # 未提供文本时的演示样例
    max_length: int = 1024
    stride: int = 768
    device: Optional[str] = None
    dtype: Optional[str] = None
```
- **用途**：封装默认模型与滑窗参数、样例文本；  
- **`__post_init__`**：当 `texts is None` 时注入 3 条教学示例。

---

### `auto_device() -> torch.device`
- **用途**：自动优先返回 `cuda`，否则 `mps`，否则 `cpu`。  
- **注意**：在某些 macOS 环境，`mps` 可用但显存较小；遇 OOM 可改 `--dtype fp32` 或降低 `max_length`。

---

### `to_torch_dtype(dtype: Optional[str]) -> Optional[torch.dtype]`
- **用途**：将 `"fp16"/"bf16"/"fp32"` 映射到 PyTorch dtype，其他输入返回 `None`。  
- **配合**：传给 `AutoModelForCausalLM.from_pretrained(..., torch_dtype=...)`。

---

### `read_lines(path: str) -> List[str]`
- **用途**：读取文件的非空行；去除末尾换行符与空白。

---

### `compute_text_nll_sliding(...) -> float`
```python
@torch.no_grad()
def compute_text_nll_sliding(text, tokenizer, model, max_length=1024, stride=768, device=torch.device("cpu")) -> float:
    """对单条长文本用滑窗方式计算总 NLL（负对数似然）。"""
```
**实现要点**：
1. `tokenizer(text, add_special_tokens=False)` 获得 `input_ids`；  
2. 若 `n_tokens <= max_length`：一次性计算 `out.loss`；  
3. 否则按窗口 `[start:end]` 滑动：
   - `window_ids = input_ids[:, start:end]`；
   - `out = model(input_ids=window_ids, labels=window_ids)`；
   - `total_nll += out.loss * window_len`；
   - `start = end - (max_length - stride)`（**保持重叠**，确保上下文连贯）。

**返回**：该文本**所有 token** 的总 NLL（非平均）。

> 细节：此实现**未跨窗携带隐藏态**，而是按窗口**自回归对齐**（`labels=inputs`），适合**统一评估**；对长上下文模型可进一步实现**跨窗缓存**以减少偏差与开销。

---

### `compute_ppl(texts, model_name, max_length, stride, device, dtype) -> float`
- **用途**：对一组文本计算**加权平均 PPL**：
  1) 自动选择设备与 dtype；  
  2) 加载 tokenizer & model（若 tokenizer 无 `pad_token`，兜底设为 `eos_token`）；  
  3) 遍历文本：先取长度 `n_tok`，再调用 `compute_text_nll_sliding` 获得 `nll`；  
  4) 累加 `total_nll` 与 `total_tokens`；  
  5) `avg_nll = total_nll / max(total_tokens, 1)`，`ppl = exp(avg_nll)`。

- **返回**：`float`，整体 PPL。

> **稳健性**：当文本为空（`n_tok == 0`）直接跳过，避免除零。

---

### `build_human_eval_csv(out_csv_path, prompts, aspects=("帮助性","事实性","风格/礼貌","一致性"), encoding="utf-8-sig")`
- **用途**：生成人工评估模板 CSV（**不依赖 pandas**）。  
- **字段**：`id, prompt, 参考答案(可选), 模型回答, 总体评分(1-5), 备注`，以及每个 `aspects` 的单项评分列（`(1-5)`）。  
- **编码**：默认 `utf-8-sig`，便于 Excel 直接打开不乱码。

---

### `parse_args()` & `main()`
- **`parse_args`**：定义并解析 CLI；  
- **`main` 流程**：
  1. 收集文本/题目（若文件不存在则使用内置默认）；  
  2. 调用 `build_human_eval_csv` 输出评估表；  
  3. 调用 `compute_ppl` 打印困惑度；  
  4. 异常捕获：若模型下载失败/显存不足，则仅提示 CSV 已生成并给出降配建议。

---

## 五、PPL 定义与滑窗评估细节

**困惑度（Perplexity）** 定义：
\[
\mathrm{PPL} = \exp\!\left(\frac{1}{N}\sum_{t=1}^{N} -\log p(x_t \mid x_{<t})\right)
\]
其中 \(N\) 为 token 总数。

**为何滑窗？**  
当文本长度 > `max_length` 时，无法一次性送入模型。滑窗把长文本切成多个**重叠窗口**，每个窗口在**自回归掩码**下计算平均 NLL，再按窗口 token 数**加权求和**得到总 NLL。最终对**全体 token** 取平均并指数化，得到全体 PPL。

**常见取值建议**：
- `max_length ∈ [512, 4096]` 视模型上下文与显存而定；  
- `stride` 通常略小于 `max_length`（例如 `1024/768`）以保证上下文连续与计算效率。

---

## 六、CSV 字段说明与评审指引

- **prompt**：题目/指令文本（来自 `--prompts_file` 或内置示例）；  
- **参考答案(可选)**：若有黄金答案可填入，便于对照；  
- **模型回答**：评审员粘贴模型输出；  
- **总体评分(1-5)**：总体主观评价；  
- **各维度评分(1-5)**：
  - **帮助性**：是否解决了问题、是否可用；  
  - **事实性**：是否客观准确、无编造；  
  - **风格/礼貌**：语气是否得体、是否符合场景；  
  - **一致性**：回答是否与输入约束/上下文一致。

> **建议**：明确评分 Rubric（例如 1=差，3=可用但有缺陷，5=优秀），提高多评审一致性。

---

## 七、常见问题（FAQ）

1. **PPL 计算慢/显存不足？**  
   - 降低 `--max_length`、提高 `--stride`（减少重叠）、切换小一点的模型；在 GPU 上使用 `--dtype bf16/fp16`。

2. **为什么要 `pad_token = eos_token`？**  
   - 部分因果语言模型未设置 `pad_token`，这里做**安全兜底**以避免分词器/`DataCollator` 出现错误。

3. **滑窗是否会引入偏差？**  
   - 会有轻微近似（窗口内独立计算），但在教学/对比中足够直观。追求更精确可复用**跨窗隐藏态**或采用**流式评估**。

4. **CSV 打开乱码？**  
   - 默认 `utf-8-sig` 以兼容 Excel；若仍乱码，可改为 `gbk`（`encoding="gbk"`）。

5. **文本里有空行怎么办？**  
   - `read_lines` 会过滤空行；如需保留，请改函数逻辑或换用 JSON/CSV 数据格式。

---

## 八、扩展建议

- **更精细的度量**：输出分文本 PPL、分段 PPL 分布（箱线图）；  
- **批量评估**：加入 `batch_size` 与 `torch.no_grad()` 的批处理编码；  
- **指令/对话模板**：对 Chat 模型按模板（system/user/assistant）构造 `labels` 只在答案段计损失；  
- **CSV 增列**：如“是否使用检索”“是否违规”“评审员ID”等，方便后续统计与质量追踪；  
- **导出 JSON**：除 CSV 外，额外保存结构化 JSON 便于程序读取。

---

## 九、许可证

本文档与脚本用于教学演示；请遵循你项目的整体许可证与上游依赖（Transformers、PyTorch 等）的许可证条款。
