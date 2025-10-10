# -*- coding: utf-8 -*-
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

# ===== 你的加载函数（原样复用） =====
class CausalLMWithValueHead(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.value_head.weight, std=1e-2)
        nn.init.zeros_(self.value_head.bias)
    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=output_hidden_states, use_cache=False)
        h = out.hidden_states[-1]              # [B, L, H]
        values = self.value_head(h).squeeze(-1)  # [B, L]
        return {"values": values}

def load_policy_with_value_head(path, device="cuda"):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    model = CausalLMWithValueHead(base)
    vh_path = os.path.join(path, "value_head.pt")
    if os.path.exists(vh_path):
        state = torch.load(vh_path, map_location="cpu")
        model.value_head.load_state_dict(state)
    model.to(device).eval()
    return model, tok

# ===== 打分函数（与训练前处理一致）=====
@torch.no_grad()
def score_pair(tokenizer, model, prompt: str, chosen: str, rejected: str, max_length: int = 512, device=None):
    device = device or next(model.parameters()).device
    full_prompt = f"问题: {prompt}\n回答:"  # 和训练时完全一致的前缀

    seq_c = full_prompt + chosen
    seq_r = full_prompt + rejected

    enc_c = tokenizer(seq_c, truncation=True, padding="max_length",
                      max_length=max_length, return_tensors="pt")
    enc_r = tokenizer(seq_r, truncation=True, padding="max_length",
                      max_length=max_length, return_tensors="pt")

    input_ids      = torch.cat([enc_c["input_ids"], enc_r["input_ids"]], dim=0).to(device)  # [2, L]
    attention_mask = torch.cat([enc_c["attention_mask"], enc_r["attention_mask"]], dim=0).to(device)
    if attention_mask.dtype != torch.long:
        attention_mask = attention_mask.long()

    out = model(input_ids=input_ids, attention_mask=attention_mask)  # {"values": [2, L]}
    values = out["values"]                                           # [2, L]

    last_idx = attention_mask.sum(dim=-1) - 1                        # [2]
    scores = values.gather(1, last_idx.unsqueeze(1)).squeeze(1)      # [2]
    return scores[0].item(), scores[1].item(), (scores[0]-scores[1]).item()

# ===== 演示用法 =====
if __name__ == "__main__":
    # 1) 指向保存目录（包含 base 的权重与 value_head.pt）
    CKPT_DIR = "./outputs/reward_model/checkpoint-10"   # 或根目录 ./outputs/reward_model

    # 2) 加载
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_policy_with_value_head(CKPT_DIR, device=device)

    # 3) 单对测试
    prompt = "如何用Python实现一个快速排序？"
    better = "使用递归选择一个枢轴，将列表分为小于与大于枢轴的两部分，对两部分递归快排并合并结果。"
    worse  = "可以用很多 if-else 来交换元素，但复杂度不明确，效率不高。"

    c, r, d = score_pair(tok, model, prompt, better, worse, max_length=512, device=device)
    print(f"得分 (Chosen):   {c:.4f}")
    print(f"得分 (Rejected): {r:.4f}")
    print(f"偏好差异:        {d:.4f}  (越大越好)")
