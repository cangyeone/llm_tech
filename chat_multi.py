# -*- coding: utf-8 -*-
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from transformers.cache_utils import DynamicCache

# 只保留 warning / error，安静一些
hf_logging.set_verbosity_warning()

# ===== 基本配置 =====
model_name = "qw1.7_model"  # 你的对话模型目录或名称
max_new_tokens = 512
use_sampling = False         # True 则使用采样（temperature/top_p），False 为贪心
temperature = 0.7
top_p = 0.9

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

# 某些模型没有 pad_token，统一兜底为 eos
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ===== 对话历史 =====
messages = [
    {"role": "system", "content": "你是‘地震学AI助手’，一个乐于助人的中文AI助手。由地震局人工智能团队开发。请帮助用户解决问题。"},
]

# ===== KV 缓存相关变量 =====
# 缓存的是“到目前为止，已经跑过 forward 的输入 token”
cached_input_ids = None     # shape: [1, L]
kv_cache: DynamicCache | None = None

# ===== 工具函数：构建当前 prompt 的 token =====
def build_input_ids_from_messages(msgs):
    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True  # 让模型继续生成 assistant 的回答
    )
    return tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]

# ===== 工具函数：把新增 token 过一遍模型，更新 KV；必要时重新预填 =====
@torch.no_grad()
def prefill_or_update_cache(new_input_ids):
    global cached_input_ids, kv_cache

    if cached_input_ids is None or kv_cache is None:
        # 第一次或缓存失效：对全量输入做 prefill
        kv_cache = DynamicCache()
        _ = model(input_ids=new_input_ids, use_cache=True, past_key_values=kv_cache)
        cached_input_ids = new_input_ids
        # 返回当前最后一个位置的 logits，用于解码起点（可选）
        with torch.no_grad():
            last_logits = model(input_ids=new_input_ids[:, -1:], use_cache=True, past_key_values=kv_cache).logits[:, -1, :]
        return last_logits

    # 否则尝试前缀对齐：新输入是否以旧输入为前缀？
    old_len = cached_input_ids.shape[1]
    if new_input_ids.shape[1] >= old_len and torch.equal(new_input_ids[:, :old_len], cached_input_ids):
        # 只把“新增部分”过一遍，更新 KV
        delta = new_input_ids[:, old_len:]
        if delta.numel() > 0:
            _ = model(input_ids=delta, use_cache=True, past_key_values=kv_cache)
            cached_input_ids = new_input_ids
        # 返回当前位置的 logits
        with torch.no_grad():
            last_logits = model(input_ids=new_input_ids[:, -1:], use_cache=True, past_key_values=kv_cache).logits[:, -1, :]
        return last_logits
    else:
        # 前缀不匹配（例如系统 prompt 改了）：重新预填
        kv_cache = DynamicCache()
        _ = model(input_ids=new_input_ids, use_cache=True, past_key_values=kv_cache)
        cached_input_ids = new_input_ids
        with torch.no_grad():
            last_logits = model(input_ids=new_input_ids[:, -1:], use_cache=True, past_key_values=kv_cache).logits[:, -1, :]
        return last_logits

# ===== 逐步生成（显式使用 KV 缓存） =====
@torch.no_grad()
def generate_with_kv(last_logits, max_tokens=512, eos_id=None):
    global kv_cache

    generated_ids = []
    cur_logits = last_logits  # 起点：prefill 后的最后 logits

    for _ in range(max_tokens):
        if use_sampling:
            probs = torch.softmax(cur_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
        else:
            next_id = torch.argmax(cur_logits, dim=-1, keepdim=True)  # [1, 1]

        token_id = next_id.item()
        generated_ids.append(token_id)

        if eos_id is not None and token_id == eos_id:
            break

        # 将新 token 继续前向，更新 KV，并得到下一个位置的 logits
        cur_logits = model(input_ids=next_id, use_cache=True, past_key_values=kv_cache).logits[:, -1, :]

    if len(generated_ids) == 0:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return torch.tensor(generated_ids, dtype=torch.long, device=device).view(1, -1)

# ===== 思考/最终答案分离 =====
def strip_think(text: str):
    insides = re.findall(r"<think>(.*?)</think>", text, flags=re.S)
    outside = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    outside = re.sub(r"</think>", "", outside, flags=re.S).strip()
    return insides, outside

# ======== 首轮对话：演示 ========
#input_ids = build_input_ids_from_messages(messages)
#last_logits = prefill_or_update_cache(input_ids)
#gen_ids = generate_with_kv(last_logits, max_tokens=max_new_tokens, eos_id=tokenizer.eos_token_id)
#
#full_ids = torch.cat([input_ids, gen_ids], dim=1)
#text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
#
#think_spans, answer = strip_think(text)
#print("\n==== 首轮助手回复 ====")
#if think_spans:
#    print("【思考过程】", think_spans[0].strip())
#print("【最终答案】", answer.split("assistant")[-1].strip())
#
## 把“最终答案”写回对话历史（非常重要：保证下一轮模板一致）
#messages.append({"role": "assistant", "content": answer.split("assistant")[-1].strip()})
#
## 更新缓存：把本轮生成也视作已缓存的前缀
#cached_input_ids = full_ids  # 让下一轮能直接在这个基础上增量

# ======== 人工交互循环 ========
print("\n进入多轮对话（输入 q 退出）")
while True:
    try:
        user_text = input("\n你：").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break
        if not user_text:
            continue

        # 追加用户消息
        messages.append({"role": "user", "content": user_text})

        # 构造新一轮输入，并复用 KV（仅增量部分过一遍）
        input_ids = build_input_ids_from_messages(messages)
        last_logits = prefill_or_update_cache(input_ids)

        # 生成
        gen_ids = generate_with_kv(last_logits, max_tokens=max_new_tokens, eos_id=tokenizer.eos_token_id)

        # 拼接得到整段文本，并解析 think/outside
        full_ids = torch.cat([input_ids, gen_ids], dim=1)
        text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
        think_spans, answer = strip_think(text)

        print("\n助手：")
        # 如需隐藏思考过程，注释掉下一行即可
        if think_spans:
            print("【思考过程】", think_spans[-1].strip())
        final_answer = answer.split("assistant")[-1].strip()
        print(final_answer)

        # 写回历史，确保下一轮模板一致；并把生成的 token 并入缓存前缀
        messages.append({"role": "assistant", "content": final_answer})
        cached_input_ids = torch.cat([input_ids, gen_ids], dim=1)

    except KeyboardInterrupt:
        print("\n已退出。")
        break
