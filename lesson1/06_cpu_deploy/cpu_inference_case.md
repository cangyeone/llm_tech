# 多轮对话 + KV 缓存 + 终端 Markdown 渲染：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`chat_kv_rich.py`（基于你提供的教学代码整理）  
> 运行环境：Python ≥ 3.9；`transformers>=4.41`、`torch`、`rich`  
> 目标读者：希望在命令行中实现**多轮聊天**、**KV 缓存复用（DynamicCache）**、**Markdown 渲染**的教学/演示用户。

---

## 目录
- [多轮对话 + KV 缓存 + 终端 Markdown 渲染：使用说明与函数文档](#多轮对话--kv-缓存--终端-markdown-渲染使用说明与函数文档)
  - [目录](#目录)
  - [一、功能总览](#一功能总览)
  - [二、快速开始](#二快速开始)
    - [依赖安装](#依赖安装)
    - [运行脚本](#运行脚本)
  - [三、核心流程与数据流](#三核心流程与数据流)
  - [四、配置项说明](#四配置项说明)
  - [五、函数文档](#五函数文档)
    - [`build_input_ids_from_messages(msgs) -> torch.LongTensor`](#build_input_ids_from_messagesmsgs---torchlongtensor)
    - [`prefill_or_update_cache(new_input_ids) -> torch.FloatTensor`](#prefill_or_update_cachenew_input_ids---torchfloattensor)
    - [`generate_with_kv(last_logits, max_tokens=512, eos_id=None) -> torch.LongTensor`](#generate_with_kvlast_logits-max_tokens512-eos_idnone---torchlongtensor)
    - [`split_think(text: str) -> tuple[list[str], str]`](#split_thinktext-str---tupleliststr-str)
    - [`render_markdown(md_text: str, title: str | None = None) -> None`](#render_markdownmd_text-str-title-str--none--none---none)
  - [六、主循环逻辑](#六主循环逻辑)
  - [七、KV 缓存与前缀对齐要点](#七kv-缓存与前缀对齐要点)
  - [八、采样与贪心解码](#八采样与贪心解码)
  - [九、显示/隐藏 `<think>` 思考过程](#九显示隐藏-think-思考过程)
  - [十、常见问题与排错](#十常见问题与排错)
  - [十一、扩展建议](#十一扩展建议)
  - [十二、许可证](#十二许可证)

---

## 一、功能总览

该脚本在命令行实现：

1. **多轮对话**：维护 `messages = [{role, content}, ...]` 历史，使用 `tokenizer.apply_chat_template` 构造对话 Prompt；  
2. **KV 缓存（DynamicCache）**：通过**前缀对齐**，仅对新增 token 进行增量 `prefill`，显著降低多轮推理的、重复前向的开销；  
3. **终端 Markdown 渲染**：使用 `rich` 在终端以 Markdown 方式展示“最终答案”；  
4. **可选显示 `<think>`**：如模型在输出中含 `<think>...</think>` 段，可选择是否在终端显示该段（默认 **不显示**）。

---

## 二、快速开始

### 依赖安装
```bash
pip install "transformers>=4.41" torch rich
```

### 运行脚本
```bash
python lesson1/06_cpu_deploy/cpu_inference_case.py
```
随后即可在终端进行多轮对话：输入内容并回车，输入 `q`/`quit`/`exit` 退出。

> **GPU**：脚本使用 `device_map="auto"` 自动选择设备；若仅有 CPU，也能运行但速度较慢。

---

## 三、核心流程与数据流

1. **初始化**：加载 `AutoTokenizer` 与 `AutoModelForCausalLM`；设置 `pad_token_id`；  
2. **构造 messages**：含 `system` 和首条 `user`；  
3. **输入 → tokens**：`build_input_ids_from_messages()` 使用聊天模板构造输入 `input_ids`；  
4. **KV 复用**：`prefill_or_update_cache()` 对比新旧 `input_ids` 前缀，一致则仅增量 `prefill`；  
5. **逐 token 生成**：`generate_with_kv()` 使用**同一** `kv_cache` 连续喂入单个 token，得到下一个 `logits`；  
6. **解码与渲染**：解码为文本，抽出 `<think>` 外部内容，用 `rich` Markdown 渲染；  
7. **写回历史**：将 assistant 回复写回 `messages`，并更新 `cached_input_ids`。

---

## 四、配置项说明

脚本顶部“基本配置”：
```python
model_name = "Qwen/Qwen3-4b"  # 模型名称/本地路径（需支持 chat_template）
max_new_tokens = 2048         # 单轮最大生成 token 数
use_sampling = False          # True 采样；False 贪心
temperature = 0.7             # 采样温度
top_p = 0.9                   # nucleus sampling 概率阈值
show_think = False            # 是否显示 <think> 段落
```
> **注意**：当前 `generate_with_kv` 并未显式使用 `temperature/top_p`，若需采样请按[八、采样与贪心解码](#八采样与贪心解码)中的示例改造。

---

## 五、函数文档

### `build_input_ids_from_messages(msgs) -> torch.LongTensor`
**作用**：将形如
```python
[{"role": "system", "content": "..."},
 {"role": "user", "content": "..."},
 {"role": "assistant", "content": "..."}]
```
的 `messages` 通过 `tokenizer.apply_chat_template(..., add_generation_prompt=True)` 转为字符串 Prompt，再分词得到 `input_ids`（形状 `[1, L]`），并移动到模型设备。

**参数**
- `msgs: list[dict]`：对话历史。

**返回**
- `input_ids: torch.LongTensor`：模型输入 token 序列（批大小为 1）。

**要点**
- 需要模型/分词器提供兼容的 `chat_template`；
- `add_generation_prompt=True` 会自动添加 assistant 起始标记，以便模型续写。

---

### `prefill_or_update_cache(new_input_ids) -> torch.FloatTensor`
**作用**：**前缀对齐** + **增量 prefill**。维护全局 `cached_input_ids` 与 `kv_cache`。

**逻辑**
1. **首次/无缓存**：新建 `DynamicCache()`，对 `new_input_ids` 做一次全量前向；  
2. **命中前缀**：若 `new_input_ids` 以 `cached_input_ids` 为前缀，仅对 `delta` 部分做前向，并更新缓存；  
3. **未命中**：重新构造 `DynamicCache()`，全量前向。

**返回**
- `last_logits: torch.FloatTensor`：对 `new_input_ids` 最后一个位置的 `logits`（形状 `[1, vocab]`）。

**要点**
- 使用 `use_cache=True, past_key_values=kv_cache` 明确复用同一缓存；  
- **必须**在每次增量前向后更新 `cached_input_ids`。

---

### `generate_with_kv(last_logits, max_tokens=512, eos_id=None) -> torch.LongTensor`
**作用**：在**不清空** `kv_cache` 的前提下，显式逐 token 解码。每步：从 `cur_logits` 选出 `next_id`，再把 `next_id` 作为 `input_ids` 输入模型，获取新一步的 `logits`。

**参数**
- `last_logits`：上一步输出的 `logits`；
- `max_tokens`：最多生成 token 数；
- `eos_id`：若下一个 token 等于 `eos_id` 则提前停止。

**返回**
- `generated_ids: torch.LongTensor`，形状 `[1, T_gen]`。若无生成则返回空张量。

**要点**
- 函数内使用全局 `kv_cache` 继续滚动窗口前向；  
- 当前实现的**采样/贪心**切换由 `use_sampling` 控制（简化版）；如需 `temperature/top_p`，见下文示例。

---

### `split_think(text: str) -> tuple[list[str], str]`
**作用**：把模型输出中 `<think>...</think>` 段落提取为 `insides` 列表，并返回**去掉** `<think>` 后的正文 `outside`。

**返回**
- `insides: list[str]`：所有 `<think>` 片段（按出现顺序）；  
- `outside: str`：移除 `<think>` 段后的文本。

---

### `render_markdown(md_text: str, title: str | None = None) -> None`
**作用**：用 `rich` 在终端渲染 Markdown 文本。可选显示标题分隔线。

**要点**
- `console.rule` 绘制标题分割线；`Markdown(md_text)` 渲染正文；  
- 终端需支持 ANSI 控制序列（多数现代终端均支持）。

---

## 六、主循环逻辑

```python
console.rule("[bold green]进入多轮对话（输入 q 退出）")
while True:
    user_text = console.input("[bold]你：[/bold]").strip()
    if user_text.lower() in {"q", "quit", "exit"}: break
    if not user_text: continue

    messages.append({"role": "user", "content": user_text})

    input_ids = build_input_ids_from_messages(messages)
    last_logits = prefill_or_update_cache(input_ids)
    gen_ids = generate_with_kv(last_logits, max_tokens=max_new_tokens, eos_id=tokenizer.eos_token_id)

    full_ids = torch.cat([input_ids, gen_ids], dim=1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)

    think_spans, answer = split_think(text)
    final_answer = answer.split("assistant")[-1].strip()

    if show_think and think_spans:
        console.print(Panel.fit(think_spans[-1].strip(), title="思考过程 <think>", border_style="yellow"))

    render_markdown(final_answer, title="助手回复（Markdown 渲染）")

    messages.append({"role": "assistant", "content": final_answer})
    cached_input_ids = torch.cat([input_ids, gen_ids], dim=1)
```

**关键点**
- **写回历史**：一定要把 `assistant` 的 `final_answer` 写回 `messages`，下一轮才有正确前缀；  
- **缓存更新**：将 `cached_input_ids` 更新为 `input_ids + gen_ids`；  
- **安全默认**：`show_think=False`，避免在终端显示任何 `<think>` 内容。

---

## 七、KV 缓存与前缀对齐要点

- **为什么要前缀对齐？**  
  多轮对话时，新一轮输入通常是**旧输入的前缀 + 新增内容**。若不复用 KV，每轮都要对**整段历史**做一次前向，计算量随历史长度增长。通过对齐前缀并仅对**新增 token** 前向，可将当轮复杂度近似降到与**新增长度成正比**。

- **DynamicCache 的使用**：
  - 首次创建：`kv_cache = DynamicCache()`；
  - 每次前向都传同一个 `past_key_values=kv_cache`；
  - 模型将自动在缓存尾部**追加**新的 K/V。

- **什么时候重建缓存？**  
  - 若用户修改了较早的历史导致**前缀不再一致**；  
  - 若切换了系统提示或模板，需全量 `prefill`。

---

## 八、采样与贪心解码

当前实现的采样是：
```python
if use_sampling:
    probs = torch.softmax(cur_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
else:
    next_id = torch.argmax(cur_logits, dim=-1, keepdim=True)
```
若要加入 **温度 / top-p**，可替换为：
```python
def sample_with_temperature_top_p(logits, temperature=0.7, top_p=0.9):
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    # top-p 截断
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > top_p
    # 保证至少取一个
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, num_samples=1)
    next_id = sorted_idx.gather(-1, choice)
    return next_id
```
在 `generate_with_kv` 中用该函数替换采样分支。

---

## 九、显示/隐藏 `<think>` 思考过程

- 默认 `show_think = False`，仅展示**答案部分**；  
- 若设为 `True`，则用 `split_think` 把 `<think>...</think>` 中的**最后一段**打印到一个 `Panel` 中；  
- 教学建议：解释“思考过程”与“最终答案”的区分，但避免泄露敏感/冗长内容。

---

## 十、常见问题与排错

1. **`apply_chat_template` 报错**  
   - 该函数依赖模型/分词器自带的 `chat_template`。若模型不支持，请改为自己拼接 Prompt，或换用支持聊天模板的模型（如 Qwen 系列、Llama 系列的指令版等）。

2. **控制台乱码或颜色异常**  
   - 升级终端或 `rich`；Windows 可使用 Windows Terminal/PowerShell 7。

3. **生成重复/卡住**  
   - 适度降低 `max_new_tokens`；或使用采样并设置 `temperature/top_p`；检查是否缺少 `eos_token_id`。

4. **首次轮次过慢**  
   - 首轮为**全量 prefill**，之后轮次将显著加速。

5. **显存不足**  
   - 减小上下文长度（历史轮次）、换更小模型、或采用 8bit/4bit 加载（需相应依赖与支持）。

---

## 十一、扩展建议

- **自定义聊天模板**：在 `system/user/assistant` 之间加入明确分隔符、角色注释；  
- **仅对“答案”打标签**：自定义 `stopping criteria` 或正则裁剪，避免把提示词和多余文本带入下一轮；  
- **流式输出**：把逐 token 结果实时打印到终端（逐字符/逐词刷新）；  
- **对话持久化**：将 `messages` 与 `cached_input_ids` 序列化保存，实现会话恢复；  
- **安全/审计**：对 `<think>` 内容长度/敏感词做限制或不显示策略。

---

## 十二、许可证

该文档与脚本用于教学演示。请遵循你项目的总体许可证以及上游依赖（Transformers、PyTorch、Rich、模型权重）的许可证条款。
