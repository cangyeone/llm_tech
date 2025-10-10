# -*- coding: utf-8 -*-
"""
教程：客服场景对齐案例分析（最小可运行版）
- 构造含优选/拒绝回答的偏好数据集
- 为候选回复打分：coverage & politeness
- 总分公式：score = 0.6*coverage + 0.4*politeness
- 导出 CSV 供业务回顾
- 可选：接入 Qwen3 生成额外候选（默认关闭，离线可跑）

Run:
  python cx_alignment_tutorial.py
"""

import os, csv, math, re
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

# ===== 可选：是否调用预训练模型（如 Qwen/Qwen3-0.6b）生成回复 =====
USE_MODEL = False       # 默认 False：完全离线、无需依赖
MODEL_NAME = "Qwen/Qwen3-0.6b"
MAX_NEW_TOKENS = 128

# ===== 如果启用模型，懒加载 transformers =====
def maybe_load_model():
    if not USE_MODEL:
        return None, None
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, attn_implementation="eager"
    )
    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    return tok, model

def model_generate(tok, model, prompt: str) -> str:
    import torch, torch.nn.functional as F
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_p=0.9, temperature=0.7,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    # 仅截取“回答：”之后的部分（若有）
    m = re.search(r"回答[:：](.*)", text, flags=re.S)
    return (m.group(1).strip() if m else text.strip())[:600]

# ===== 数据建模 =====
@dataclass
class Example:
    ticket_id: str
    customer_query: str
    context_tags: List[str]          # 用于覆盖度的主题词（人工/模板）
    candidate_type: str              # "chosen" | "rejected" | "generated"
    reply: str
    coverage: float = 0.0            # [0,1]
    politeness: float = 0.0          # [0,1]
    score: float = 0.0               # 0.6*coverage + 0.4*politeness
    notes: str = ""                  # 评语/原因

# ===== 示例偏好数据（3条工单，每条含 chosen/rejected）=====
PREF_DATA: List[Tuple[str, str, List[str], str, str]] = [
    (
        "T001",
        "我昨天下单的耳机什么时候发货？可以帮我查下物流吗？",
        ["发货", "物流", "订单", "耳机", "查询"],
        # chosen
        "您好～我已为您查询到该订单今天已出库，预计24小时内揽收。"
        "这里是物流单号：SF123456，可在顺丰官网/小程序实时跟踪。"
        "若 24 小时内仍未更新，请随时回复我为您催单。",
        # rejected
        "自己去查快递，我们也没办法。"
    ),
    (
        "T002",
        "我想申请退货，耳机有电流声。需要怎么操作？",
        ["退货", "售后", "流程", "电流声", "耳机"],
        "非常抱歉给您带来不便。我可以协助您退货："
        "请在“我的订单-申请售后”提交退货，原因选择“性能故障”。"
        "系统会生成上门取件单，快递费由我们承担。",
        "这个问题不是我管的，你自己找售后。"
    ),
    (
        "T003",
        "我买错颜色了，想改成黑色版，还能改吗？",
        ["改色", "变更", "订单", "黑色", "客服"],
        "理解您的需求～若尚未发货可直接为您改为黑色。"
        "我这边先提交变更申请，同时给仓库留言优先处理；"
        "若已发货也没关系，收到后支持换货。",
        "不行，不能改。"
    ),
]

# ===== 简易礼貌性/覆盖度评分器（规则探针版）=====
POLITE_POS = [
    "您好", "抱歉", "非常抱歉", "请", "感谢", "辛苦", "为您", "理解", "协助", "麻烦",
    "很抱歉", "建议", "可以", "将为您", "已为您", "如有需要", "欢迎随时"
]
POLITE_NEG = ["自己", "不管", "不行", "没办法", "麻烦死", "你们", "闭嘴", "滚", "差评"]

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def politeness_score(text: str) -> float:
    t = text.strip()
    pos = sum(t.count(w) for w in POLITE_POS)
    neg = sum(t.count(w) for w in POLITE_NEG)
    raw = 0.5 + 0.15*pos - 0.25*neg
    return clamp01(raw)

def coverage_score(query: str, tags: List[str], reply: str) -> float:
    """
    粗略覆盖度：回复中命中多少上下文标签（去重），并奖励提供“可执行动作”的提示词。
    """
    r = reply.lower()
    hits = 0
    for k in set(tags):
        if k.lower() in r:
            hits += 1
    base = hits / max(1, len(set(tags)))
    # 简易“可执行动作”词奖励
    action_bonus = 0.1 if any(w in r for w in ["查询", "单号", "申请", "提交", "更改", "上门", "退货", "换货"]) else 0.0
    return clamp01(base + action_bonus)

def total_score(coverage: float, politeness: float) -> float:
    return clamp01(0.6*coverage + 0.4*politeness)

def annotate(example: Example) -> Example:
    example.coverage   = coverage_score(example.customer_query, example.context_tags, example.reply)
    example.politeness = politeness_score(example.reply)
    example.score      = total_score(example.coverage, example.politeness)
    # 生成简单评语
    notes = []
    if example.coverage < 0.5: notes.append("覆盖不足")
    if example.politeness < 0.5: notes.append("礼貌性弱")
    if example.coverage >= 0.7: notes.append("覆盖充分")
    if example.politeness >= 0.7: notes.append("礼貌到位")
    example.notes = "；".join(notes) if notes else "表现良好"
    return example

def build_dataset() -> List[Example]:
    rows: List[Example] = []
    for (tid, query, tags, pos, neg) in PREF_DATA:
        rows.append(annotate(Example(tid, query, tags, "chosen",   pos)))
        rows.append(annotate(Example(tid, query, tags, "rejected", neg)))
    # 可选：模型生成额外候选
    if USE_MODEL:
        tok, model = maybe_load_model()
        if tok is not None:
            for (tid, query, tags, _, _) in PREF_DATA:
                prompt = f"问题：{query}\n请用礼貌、同理心、且可执行的步骤进行回复。回答："
                gen = model_generate(tok, model, prompt)
                rows.append(annotate(Example(tid, query, tags, "generated", gen)))
    return rows

def export_csv(rows: List[Example], out_path="outputs/cx_alignment_review.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = ["ticket_id","candidate_type","score","coverage","politeness","customer_query","reply","context_tags","notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([
                r.ticket_id, r.candidate_type, f"{r.score:.3f}",
                f"{r.coverage:.3f}", f"{r.politeness:.3f}",
                r.customer_query, r.reply, "|".join(r.context_tags), r.notes
            ])
    return out_path

def print_brief_report(rows: List[Example], top_k=3):
    print("\n=== 总览 ===")
    avg = sum(r.score for r in rows)/len(rows)
    print(f"样本数：{len(rows)}；平均总分：{avg:.3f}")
    print("\n=== Top 表现 ===")
    for r in sorted(rows, key=lambda x: x.score, reverse=True)[:top_k]:
        print(f"[{r.candidate_type}] {r.ticket_id}  score={r.score:.3f}  cov={r.coverage:.3f}  pol={r.politeness:.3f}")
        print("  ▶", r.reply[:120].replace("\n"," "), "...")
    print("\n=== 需改进 ===")
    for r in sorted(rows, key=lambda x: x.score)[:top_k]:
        print(f"[{r.candidate_type}] {r.ticket_id}  score={r.score:.3f}  cov={r.coverage:.3f}  pol={r.politeness:.3f}")
        print("  ▶", r.reply[:120].replace("\n"," "), "...")

def main():
    rows = build_dataset()
    out_csv = export_csv(rows)
    print_brief_report(rows)
    print(f"\n✅ 结果已导出：{out_csv}")
    print("👉 可用 Excel/Numbers 打开；或导入到数据看板供业务回顾。")

if __name__ == "__main__":
    main()
