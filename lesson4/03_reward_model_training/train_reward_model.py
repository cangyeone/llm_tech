from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import evaluate
import numpy as np
import torch
from datasets import Dataset, load_dataset, get_dataset_split_names
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import torch.nn as nn 
# --- 常量定义 ---
DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6b" 
DEFAULT_DATASET = "lvwerra/stack-exchange-paired"


@dataclass
class ScriptArguments:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET
    subset: str | None = None
    output_dir: str = "./outputs/reward_model"
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_batch_size: int = 2
    max_length: int = 512
    fraction: float = 0.01 # 默认只取 1% 数据进行快速实验


# --- 数据加载和处理 ---

def load_preference_dataset(args: ScriptArguments) -> tuple[Dataset, Dataset]:
    name   = args.dataset_name
    frac   = getattr(args, "fraction", 0.01)
    frac_str = "" if frac >= 1.0 else f"[:1%]"

    # 🚀 修正 1：指定数据子目录 (stack-exchange-paired 数据集需要这个)
    ds_train = load_dataset(name, data_dir='data/finetune', split=f"train{frac_str}")
    # 注意：这个数据集默认只有 train split，我们从 train split 拆分出评估集
    
    # 简单的 train/test 拆分 (80/20)
    ds_split = ds_train.train_test_split(test_size=0.2, seed=42)
    ds_train = ds_split["train"]
    ds_test  = ds_split["test"]

    # 统一成 {prompt, chosen, rejected}
    def to_pref(ex):
        # lvwerra/stack-exchange-paired 的字段
        
        # 1. 确认字段存在：你应该检查数据集的原始字段名。
        # 根据数据集设计，response_j 应该对应 chosen (更好的回答)，
        # response_k 应该对应 rejected (较差的回答)。
        
        # 确保字段存在
        if "question" not in ex or "response_j" not in ex or "response_k" not in ex:
            # 如果你看到了这个错误，说明加载的数据集结构不正确
            raise KeyError("加载的数据集缺少 'question', 'response_j', 或 'response_k' 字段。")
            
        prompt = ex["question"]
        chosen = ex["response_j"]
        rejected = ex["response_k"]
        
        
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


    ds_train = ds_train.map(to_pref, remove_columns=ds_train.column_names)
    ds_test  = ds_test.map(to_pref,  remove_columns=ds_test.column_names)
    return ds_train, ds_test


# ==== 1) 预处理：串接 prompt+answer，返回 (B,2,L) ====
def preprocess(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _encode(batch):
        prompts  = [f"问题：{q}\n回答：" for q in batch["prompt"]]
        chosens  = batch["chosen"]
        rejecteds= batch["rejected"]

        seq_chosen  = [p + c for p, c in zip(prompts, chosens)]
        seq_rejected= [p + r for p, r in zip(prompts, rejecteds)]

        enc_c = tokenizer(seq_chosen,  truncation=True, padding="max_length", max_length=max_length)
        enc_r = tokenizer(seq_rejected,truncation=True, padding="max_length", max_length=max_length)

        input_ids      = np.stack([enc_c["input_ids"],      enc_r["input_ids"]], axis=1)   # (B,2,L)
        attention_mask = np.stack([enc_c["attention_mask"], enc_r["attention_mask"]], axis=1)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return dataset.map(_encode, batched=True, remove_columns=dataset.column_names)


# --- 损失函数和评估指标 ---

def compute_loss(self, model, inputs, return_outputs=False):
    # ⚠️ 1. 确保只提取 input_ids 和 attention_mask 
    # 并且将它们转移到正确的设备 (如果模型不在 CPU 上)
    device = model.device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # ⚠️ 2. 检查 attention_mask 的 Dtype: 
    # Qwen3 模型可能期望 attention_mask 是 LongTensor (0和1)
    # 确保它不是浮点数，且不是你意外传入的 input_ids。
    if attention_mask.dtype != torch.long:
        attention_mask = attention_mask.long()

    # 将 (batch_size, 2, seq_len) reshape 为 (batch_size * 2, seq_len) 
    outputs = model(
        input_ids=input_ids.view(-1, input_ids.size(-1)),
        attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
        # 移除所有其他可能的参数，如 token_type_ids, labels 等，除非你明确需要
    )
    
    # 3. 损失计算
    logits = outputs.logits.view(-1, 2) 
    loss = -torch.nn.functional.logsigmoid(logits[:, 0] - logits[:, 1]).mean()
    
    return (loss, outputs) if return_outputs else loss


# ==== 4) metrics：用 (chosen - rejected) 作为预测分差 ====
import evaluate, numpy as np, torch

def compute_metrics(eval_pred):
    # 来自 Trainer 的 predictions 可能是 None，这里从 inputs 里重算更稳妥。
    # 为简洁，这里假设我们在 evaluation 时复用 compute_loss 的逻辑：
    # 直接让 Trainer 返回 predictions=scores_diff（见下方 make_trainer）。
    diff = np.array(eval_pred.predictions)  # shape: (N,)
    win_rate = float((diff > 0).mean())
    kendall = evaluate.load("kendalltau")
    tau = kendall.compute(predictions=diff, references=np.ones_like(diff), variant="b")["kendalltau"]
    return {"win_rate": win_rate, "kendall_tau": tau}


def _rope_full_dim_config(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # 令 rotary_dim = head_dim，禁用 partial/scaling，避免 RoPE 维度错配
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    setattr(cfg, "rotary_dim", head_dim)
    if hasattr(cfg, "partial_rotary_factor"): cfg.partial_rotary_factor = 1.0
    if hasattr(cfg, "rope_scaling"): cfg.rope_scaling = None
    if hasattr(cfg, "use_dynamic_ntk"): cfg.use_dynamic_ntk = False
    if hasattr(cfg, "rope_theta_1"): cfg.rope_theta_1 = None
    if hasattr(cfg, "num_key_value_heads") and cfg.num_key_value_heads is None:
        cfg.num_key_value_heads = cfg.num_attention_heads
    return cfg

# ==== 2) LM + Value Head：仅返回 values（序列每个位置的值），不返回 LM logits ====
class CausalLMWithValueHead(nn.Module):
    def __init__(self, base: AutoModelForCausalLM):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.value_head.weight, std=1e-2)
        nn.init.zeros_(self.value_head.bias)

    @property
    def config(self): 
        return self.base.config

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        h = out.hidden_states[-1]             # [B, L, H]
        values = self.value_head(h).squeeze(-1)  # [B, L]
        return {"values": values}             # 只暴露序列每个 token 的 value

# --- Trainer 封装 ---
# ==== 3) 自定义 Trainer：实现 pairwise 损失 ====
class PairwiseRewardTrainer(Trainer):
    # 兼容新版本多的参数
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # (B, 2, L) —— 已经在正确设备上了
        input_ids = inputs["input_ids"]              # tensor, device 已正确
        attention_mask = inputs["attention_mask"]    # tensor, device 已正确
        if attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        B, P, L = input_ids.shape  # P=2
        input_ids = input_ids.view(B * P, L)
        attention_mask = attention_mask.view(B * P, L)

        out = model(input_ids=input_ids, attention_mask=attention_mask)  # {"values": [B*P, L]}
        values = out["values"]

        # 用最后一个非 PAD 位置的 value 作为序列分数
        last_idx = attention_mask.sum(dim=-1) - 1            # (B*P,)
        seq_score = values.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # (B*P,)

        scores = seq_score.view(B, 2)    # (B, 2) -> [chosen, rejected]
        chosen, rejected = scores[:, 0], scores[:, 1]

        loss = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
        return (loss, out) if return_outputs else loss



import os 
# ==== 5) 组装 Trainer：去掉 RoPE 改写，开启评测 ====
def make_trainer(args: ScriptArguments, tokenized_train: Dataset, tokenized_test: Dataset, tokenizer: AutoTokenizer) -> Trainer:
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = CausalLMWithValueHead(base)
    model.base.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        eval_steps=100,
        save_strategy="steps",         # ✅ 按步保存
        save_steps=10,   
        save_safetensors=False, 
        remove_unused_columns=False,
    )

    # 让 Trainer 在评测时输出 predictions=score_diff，便于 compute_metrics
    class EvalPairwiseTrainer(PairwiseRewardTrainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            with torch.no_grad():
                device = model.device
                input_ids = torch.as_tensor(inputs["input_ids"], device=device)
                attention_mask = torch.as_tensor(inputs["attention_mask"], device=device).long()
                B, P, L = input_ids.shape
                input_ids = input_ids.view(B*P, L)
                attention_mask = attention_mask.view(B*P, L)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                values = out["values"]
                last_idx = attention_mask.sum(dim=-1) - 1
                seq_score = values.gather(1, last_idx.unsqueeze(1)).squeeze(1)   # (B*P,)
                scores = seq_score.view(B, 2)
                chosen, rejected = scores[:, 0], scores[:, 1]
                diff = (chosen - rejected).detach().float().cpu().numpy()        # (B,)
                loss = -torch.nn.functional.logsigmoid(chosen - rejected).mean().detach().cpu()
            return (loss, diff, None)
        def _save(self, output_dir: str, state_dict=None):
            os.makedirs(output_dir, exist_ok=True)
            m = self.model
            # 若是我们包装的 CausalLMWithValueHead
            if hasattr(m, "base") and hasattr(m, "value_head"):
                # 1) 保存基座：HF 自己处理 tied weights
                m.base.save_pretrained(output_dir)
                # 2) 保存 value_head
                vh_path = os.path.join(output_dir, "value_head.pt")
                torch.save(m.value_head.state_dict(), vh_path)
                # 3) 保存 tokenizer（若可用）
                if getattr(self, "tokenizer", None) is not None:
                    self.tokenizer.save_pretrained(output_dir)
                # 4) 额外写个小标记，方便加载时识别
                with open(os.path.join(output_dir, "value_head_config.json"), "w") as f:
                    f.write("{}")
            else:
                # 回退到父类默认保存
                super()._save(output_dir, state_dict=state_dict)
    return EvalPairwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


# --- 主程序执行 ---

def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="奖励模型训练脚本")
    parser.add_argument("--model", dest="model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", dest="dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", dest="output_dir", type=str, default="./outputs/reward_model")
    parser.add_argument("--lr", dest="learning_rate", type=float, default=5e-6)
    parser.add_argument("--epochs", dest="num_train_epochs", type=int, default=1)
    parser.add_argument("--batch", dest="per_device_batch_size", type=int, default=2)
    parser.add_argument("--max-length", dest="max_length", type=int, default=512)
    parser.add_argument("--frac", dest="fraction", type=float, default=0.001) # 新增 fraction 参数
    parsed = parser.parse_args()
    return ScriptArguments(**vars(parsed))


if __name__ == "__main__":
    args = parse_args()
    
    # 确保安装了需要的库
    print("🚀 检查依赖：请确保已安装 transformers, datasets, evaluate, numpy, torch")
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
    
    # 2. 加载数据集
    print(f"🧩 正在加载数据集: {args.dataset_name} (比例: {args.fraction})")
    ds_train, ds_test = load_preference_dataset(args)
    print(f"📊 训练集大小: {len(ds_train)}, 测试集大小: {len(ds_test)}")
    
    # 3. 数据预处理
    print("📝 正在进行数据预处理和 Tokenization...")
    tokenized_train = preprocess(ds_train, tokenizer, args.max_length)
    tokenized_test  = preprocess(ds_test, tokenizer, args.max_length)
    
    # 4. 准备 Trainer
    print(f"🧠 正在准备 Trainer, 使用模型: {args.model_name}")
    trainer = make_trainer(args, tokenized_train, tokenized_test, tokenizer)
    
    # 5. 开始训练
    print("🔥 开始训练奖励模型...")
    trainer.train()
    print("✅ 训练完成。")