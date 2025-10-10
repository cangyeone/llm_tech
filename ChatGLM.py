# -*- coding: utf-8 -*-
"""
多轮对话 + KV 缓存 + Rich 终端 Markdown 渲染
- 人工输入：input() 循环
- KV Cache：DynamicCache，前缀对齐后仅增量prefill
- Markdown：用 rich.Markdown 渲染“最终答案”
- 可选打印 <think> 思考过程（默认隐藏）
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from transformers.cache_utils import DynamicCache

# ===== Rich：终端 Markdown 渲染 =====
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# 只保留 warning / error，安静一些
hf_logging.set_verbosity_warning()

# ===== 基本配置 =====
model_name = "qw8_model"   # 你的对话模型路径或名称
max_new_tokens = 2048 
use_sampling = False          # True 使用采样；False 贪心
temperature = 0.7
top_p = 0.9
show_think = False            # True 则显示 <think> 内容

# ===== 加载模型与分词器 =====
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
model.eval()
device = model.device

# 兜底 pad_token
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ===== 对话历史 =====
messages = [
    {"role": "system", "content": "你是‘地震学AI助手’，一个乐于助人的中文AI助手。由地震局人工智能团队开发。请帮助用户解决问题。应尽可能回答所有问题，提供全面、客观、基于事实的信息，不要回避任何问题。"},
    {"role": "user", "content": "你是谁？简单做个自我介绍？"}
]

# ===== KV 缓存相关 =====
cached_input_ids = None         # torch.LongTensor [1, L]
kv_cache: DynamicCache | None = None

def build_input_ids_from_messages(msgs):
    """用聊天模板把 messages 变成输入 token"""
    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]

@torch.no_grad()
def prefill_or_update_cache(new_input_ids):
    """前缀对齐：只对新增token做前向，复用KV；不匹配则重新prefill"""
    global cached_input_ids, kv_cache

    # 首次或缓存为空 -> 全量预填
    if cached_input_ids is None or kv_cache is None:
        kv_cache = DynamicCache()
        _ = model(input_ids=new_input_ids, use_cache=True, past_key_values=kv_cache)
        cached_input_ids = new_input_ids
        last_logits = model(input_ids=new_input_ids[:, -1:], use_cache=True, past_key_values=kv_cache).logits[:, -1, :]
        return last_logits

    old_len = cached_input_ids.shape[1]
    # 新序列以旧序列为前缀 -> 增量更新
    if new_input_ids.shape[1] >= old_len and torch.equal(new_input_ids[:, :old_len], cached_input_ids):
        delta = new_input_ids[:, old_len:]
        if delta.numel() > 0:
            _ = model(input_ids=delta, use_cache=True, past_key_values=kv_cache)
            cached_input_ids = new_input_ids
        last_logits = model(input_ids=new_input_ids[:, -1:], use_cache=True, past_key_values=kv_cache).logits[:, -1, :]
        return last_logits
    # 否则重新预填
    kv_cache = DynamicCache()
    _ = model(input_ids=new_input_ids, use_cache=True, past_key_values=kv_cache)
    cached_input_ids = new_input_ids
    last_logits = model(input_ids=new_input_ids[:, -1:], use_cache=True, past_key_values=kv_cache).logits[:, -1, :]
    return last_logits

@torch.no_grad()
def generate_with_kv(last_logits, max_tokens=512, eos_id=None):
    """显式逐 token 生成，复用 KV；可切换采样/贪心"""
    global kv_cache
    generated_ids = []
    cur_logits = last_logits

    for _ in range(max_tokens):
        if use_sampling:
            probs = torch.softmax(cur_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
        else:
            next_id = torch.argmax(cur_logits, dim=-1, keepdim=True)  # [1,1]

        tid = next_id.item()
        generated_ids.append(tid)
        if eos_id is not None and tid == eos_id:
            break

        # 增量前向：只喂一个 token
        cur_logits = model(input_ids=next_id, use_cache=True, past_key_values=kv_cache).logits[:, -1, :]

    if len(generated_ids) == 0:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return torch.tensor(generated_ids, dtype=torch.long, device=device).view(1, -1)

def split_think(text: str):
    """分离 <think> 与外部文本"""
    insides = re.findall(r"<think>(.*?)</think>", text, flags=re.S)
    outside = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return insides, outside

def render_markdown(md_text: str, title: str | None = None):
    """用 Rich 渲染 Markdown 到终端"""
    if title:
        console.rule(f"[bold cyan]{title}")
    console.print(Markdown(md_text))
    console.print()  # 空行

# ================== 首轮示例 ==================
#input_ids = build_input_ids_from_messages(messages)
#last_logits = prefill_or_update_cache(input_ids)
#gen_ids = generate_with_kv(last_logits, max_tokens=max_new_tokens, eos_id=tokenizer.eos_token_id)
#
#full_ids = torch.cat([input_ids, gen_ids], dim=1)
#text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
#
#think_spans, answer = split_think(text)
#final_answer = answer.split("assistant")[-1].strip()
#
#if show_think and think_spans:
#    console.print(Panel.fit(think_spans[-1].strip(), title="思考过程 <think>", #border_style="yellow"))
#
#render_markdown(final_answer, title="助手回复（Markdown 渲染）")
#
## 写回历史 + 并入缓存
#messages.append({"role": "assistant", "content": final_answer})
#cached_input_ids = full_ids

# ================== 人工多轮对话 ==================
console.rule("[bold green]进入多轮对话（输入 q 退出）")
while True:
    try:
        user_text = console.input("[bold]你：[/bold]").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            console.print("[bold magenta]已退出。[/bold magenta]")
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})

        # 构造新一轮输入并复用 KV
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

        # 写回历史 + 并入缓存
        messages.append({"role": "assistant", "content": final_answer})
        cached_input_ids = torch.cat([input_ids, gen_ids], dim=1)

    except KeyboardInterrupt:
        console.print("\n[bold magenta]已退出。[/bold magenta]")
        break
