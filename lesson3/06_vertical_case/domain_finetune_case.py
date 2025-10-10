# -*- coding: utf-8 -*-
"""
DDP训练
export NCCL_SOCKET_IFNAME=eth0     # 换成你的网卡
export NCCL_DEBUG=INFO

torchrun --standalone --nproc_per_node=8 train_finqa_qwen3_chat_v2.py \
  --data_dir /path/to/your/json_dir \
  --model_name Qwen/Qwen3-1.8B \
  --output_dir outputs_qwen3_v2 \
  --block_size 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 2 \
  --bf16 --grad_ckpt

ZeRO训练
deepspeed --num_gpus=8 train_finqa_qwen3_chat_v2.py \
  --data_dir /path/to/your/json_dir \
  --model_name Qwen/Qwen3-1.8B \
  --deepspeed ds_zero3.json \
  --output_dir outputs_qwen3_v2_zero3 \
  --block_size 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --num_train_epochs 2 \
  --bf16 --grad_ckpt
"""

import os, math, argparse, json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

SYS_PROMPT = (
    "You are a financial analysis assistant. "
    "Given the context from a financial report (text and tables) and a question, "
    "produce a concise numeric or textual answer."
)

def _join_lines(x: Any, sep: str = " ") -> str:
    """把 list[str] / str / None 统一成字符串"""
    if x is None:
        return ""
    if isinstance(x, list):
        return sep.join([s.strip() for s in x if isinstance(s, str)]).strip()
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

def _pick_table(ex: Dict[str, Any]) -> Any:
    """优先取清洗后的 table，否则回退到 table_ori"""
    if ex.get("table") is not None:
        return ex["table"]
    return ex.get("table_ori")

def _norm_table(table: Any, max_rows=50, max_cols=16) -> List[List[str]]:
    """把任意结构的表规整为 list[list[str]] 并截断规模"""
    if table is None:
        return []
    rows = []
    if isinstance(table, list):
        for r in table[:max_rows]:
            if isinstance(r, list):
                rows.append([str(c) if c is not None else "" for c in r[:max_cols]])
            elif isinstance(r, dict):
                rows.append([str(v) if v is not None else "" for v in list(r.values())[:max_cols]])
            else:
                rows.append([str(r)])
    elif isinstance(table, dict):
        rows.append([f"{k}: {v}" for k, v in list(table.items())[:max_cols]])
    else:
        rows.append([str(table)])
    return rows

def _table_to_text(rows: List[List[str]]) -> str:
    return "\n".join(" | ".join(r) for r in rows)

def _extract_qa(ex: Dict[str, Any]) -> (str, str):
    """同时兼容两种标注：
       1) ex['qa']['question'] / ex['qa']['exe_ans']
       2) ex['question'] / ex['answer']"""
    q = ""
    a = ""
    qa = ex.get("qa")
    if isinstance(qa, dict):
        q = _join_lines(qa.get("question"))
        a = _join_lines(qa.get("exe_ans"))
    if not q:
        q = _join_lines(ex.get("question"))
    if not a:
        a = _join_lines(ex.get("answer"))
    return q, a

def build_example_from_raw(ex: Dict[str, Any]) -> Dict[str, str]:
    pre_text  = _join_lines(ex.get("pre_text"))
    post_text = _join_lines(ex.get("post_text"))
    table_txt = _table_to_text(_norm_table(_pick_table(ex)))

    q, ans = _extract_qa(ex)
    if not ans:
        ans = "N/A"

    context_parts = []
    if pre_text:  context_parts.append(f"[PRE_TEXT]\n{pre_text}")
    if table_txt: context_parts.append(f"[TABLE]\n{table_txt}")
    if post_text: context_parts.append(f"[POST_TEXT]\n{post_text}")
    context = "\n\n".join(context_parts).strip()

    user_msg = f"Context:\n{context}\n\nQuestion: {q}"
    return {"user": user_msg, "assistant": ans}

def load_your_qa_json_as_dataset(json_path: str) -> Dataset:
    """读你的单个 JSON 文件（列表样本），并转换为 (user, assistant)"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    records = []
    for ex in data:
        pa = build_example_from_raw(ex)
        records.append(pa)
    return Dataset.from_list(records)

def load_splits(root: str) -> DatasetDict:
    """支持以下命名（存在哪个就加载哪个）：
       - train.json / dev.json / test.json
       - 或 your_file_train.json / your_file_dev.json 等（手动改名也行）"""
    d = Path(root)
    candidates = {
        "train": ["train.json"],
        "validation": ["dev.json", "valid.json", "validation.json"],
        "test": ["test.json"],
    }
    out = {}
    for split, names in candidates.items():
        for nm in names:
            p = d / nm
            if p.exists():
                out[split] = load_your_qa_json_as_dataset(str(p))
                break
    if not out:
        raise FileNotFoundError(f"No JSON found under {d} (expected train/dev/test).")
    return DatasetDict(out)

def tokenize_qwen_chat_batch(batch, tokenizer, block_size: int):
    input_ids_list, labels_list = [], []
    for u, a in zip(batch["user"], batch["assistant"]):
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]
        # prompt（到 assistant 开始）
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        # full（含答案）
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_tensors=None
        )
        if len(full_ids) > block_size:
            full_ids = full_ids[:block_size]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = full_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        input_ids_list.append(full_ids)
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "labels": labels_list}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="sample_docs/FinQA/dataset", help="包含 train/dev/test JSON 的目录")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output_dir", type=str, default="outputs_qwen3_v2")
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deepspeed", type=str, default=None)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--log_steps", type=int, default=20)
    args = ap.parse_args()
    set_seed(args.seed)

    ds = load_splits(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    def _tok(batch):
        return tokenize_qwen_chat_batch(batch, tokenizer, args.block_size)

    keep = ["user", "assistant"]
    ds_tok = DatasetDict()
    for sp in ds.keys():
        ds_tok[sp] = ds[sp].remove_columns([c for c in ds[sp].column_names if c not in keep]).map(
            _tok, batched=True, remove_columns=keep
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok.get("train"),
        eval_dataset=ds_tok.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
    print("Eval metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
