import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) 基础 Multi-Head Attention
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, H, T, Hd)
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return x  # (B, H, T, Hd)

    def forward(
        self,
        x_q: torch.Tensor,           # (B, Tq, D)
        x_kv: Optional[torch.Tensor] = None,  # (B, Tk, D) or None => self-attn
        attn_mask: Optional[torch.Tensor] = None,  # (B, 1, Tq, Tk) or (1, 1, Tq, Tk)
    ) -> torch.Tensor:
        if x_kv is None:
            x_kv = x_q

        Q = self._shape(self.q_proj(x_q))   # (B, H, Tq, Hd)
        K = self._shape(self.k_proj(x_kv))  # (B, H, Tk, Hd)
        V = self._shape(self.v_proj(x_kv))  # (B, H, Tk, Hd)

        # scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,Tq,Tk)

        if attn_mask is not None:
            scores = scores + attn_mask  # mask应为非常小的负值填充(-inf)

        attn = torch.softmax(scores, dim=-1)  # (B,H,Tq,Tk)
        out = torch.matmul(attn, V)           # (B,H,Tq,Hd)

        out = out.transpose(1, 2).contiguous().view(x_q.size(0), x_q.size(1), self.d_model)  # (B,Tq,D)
        out = self.o_proj(out)  # (B,Tq,D)
        return out


# --------------------------------------------
# 2) P-Tuning v2 风格：前缀编码器 -> K/V 前缀
# --------------------------------------------
class PrefixEncoder(nn.Module):
    """
    将 num_virtual_tokens 个“虚拟 token”映射为每层使用的 K/V 前缀。
    这里用一个共享的可学习 embedding + MLP 产生 2*D（K 和 V）的表征。
    """
    def __init__(self, num_virtual_tokens: int, d_model: int, hidden: int):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.d_model = d_model

        self.prefix_emb = nn.Embedding(num_virtual_tokens, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * d_model),  # 输出拼在一起：[..., D(K) + D(V)]
        )

        # 初始化：较小尺度更稳定
        nn.init.normal_(self.prefix_emb.weight, mean=0.0, std=0.02)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        返回 shape: (B, num_virtual_tokens, 2*D)
        """
        # (num_virtual_tokens, D)
        base = self.prefix_emb.weight  # 共享，不随 batch 变
        # (1, Tvp, D) -> (B, Tvp, D)
        base = base.unsqueeze(0).expand(batch_size, -1, -1)
        # 通过小 MLP 产生 K/V 混合表征
        kv = self.mlp(base)  # (B, Tvp, 2D)
        return kv


# ---------------------------------------------------------
# 3) 将 P-Tuning 前缀注入 MultiHeadAttention 的 K/V 序列
# ---------------------------------------------------------
@dataclass
class PTuningConfig:
    num_virtual_tokens: int = 16
    encoder_hidden: int = 512   # 前缀MLP隐层
    causal: bool = True         # 是否使用因果掩码（decoder self-attn 常用）


class PTuningMHA(nn.Module):
    """
    封装基础 MHA，在每次前向时为 K/V 拼接 prefix（不改 Q）。
    """
    def __init__(self, d_model: int, n_heads: int, cfg: PTuningConfig):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.cfg = cfg
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.prefix_encoder = PrefixEncoder(cfg.num_virtual_tokens, d_model, cfg.encoder_hidden)

    @staticmethod
    def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
        # (B, T, D) -> (B, H, T, Hd)
        B, T, D = x.shape
        Hd = D // n_heads
        return x.view(B, T, n_heads, Hd).permute(0, 2, 1, 3).contiguous()

    def _build_attn_mask(
        self,
        B: int,
        Tq: int,
        Tk_orig: int,
        Tvp: int,
        device: torch.device,
        causal: bool,
        attn_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        将原始 attn_mask（若有）在 keys 维度左侧 pad 上 Tvp（前缀长度）个 0，
        并根据 causal 再构造或合并掩码。返回 shape: (B,1,Tq,Tk_new)
        """
        Tk_new = Tvp + Tk_orig

        # 先构造因果掩码（查询看到 prefix+过往，不看到未来）
        causal_mask = None
        if causal:
            # (Tq, Tk_new)，上三角为 -inf，下三角含对 prefix 全可见
            causal_mask = torch.full((Tq, Tk_new), fill_value=0.0, device=device)
            # 对真实 token 部分做因果：query 的第 i 个只能看到 <= i 的真实 token
            # 真实 token 在 keys 的 index 从 Tvp..(Tvp+Tk_orig-1)
            q_ids = torch.arange(Tq, device=device).unsqueeze(1)              # (Tq,1)
            k_ids = torch.arange(Tk_new, device=device).unsqueeze(0)          # (1,Tk_new)
            # prefix 区域(0..Tvp-1)全部可见，无需mask；
            # 对真实 token 区域(Tvp..end)施加 i<k -> mask
            mask_real = (k_ids >= (Tvp + q_ids))  # 未来的真实 token
            causal_mask[mask_real] = float("-inf")
            # 注意：prefix 区域 (k<Tvp) 永远为 0（可见）

        # 处理传入的 attn_mask（例如 padding mask），扩展 keys 维
        if attn_mask is not None:
            # 期望 (B,1,Tq,Tk_orig) 或 broadcastable
            # 在 keys 维度左侧 pad Tvp 个 0（prefix 全可见）
            pad = (Tvp, 0)  # 针对最后一维 Tk
            expanded = F.pad(attn_mask, pad=pad, mode="constant", value=0.0)  # (B,1,Tq,Tk_new)
        else:
            expanded = None

        if causal_mask is None and expanded is None:
            return None

        # 统一成 (B,1,Tq,Tk_new)
        if causal_mask is not None:
            causal_mask = causal_mask.view(1, 1, Tq, Tk_new).expand(B, 1, Tq, Tk_new)
        if expanded is None:
            return causal_mask
        if causal_mask is None:
            return expanded

        return causal_mask + expanded  # 两者相加：有 -inf 会生效

    def forward(
        self,
        x_q: torch.Tensor,                 # (B, Tq, D)
        x_kv: Optional[torch.Tensor] = None,  # (B, Tk, D) or None
        attn_mask: Optional[torch.Tensor] = None,  # (B,1,Tq,Tk)
    ) -> torch.Tensor:
        if x_kv is None:
            x_kv = x_q
        B, Tq, D = x_q.shape
        Tk = x_kv.shape[1]
        device = x_q.device

        # 1) 基座的 K/V（尚未分头）
        K_base_in = x_kv  # (B,Tk,D)
        V_base_in = x_kv  # (B,Tk,D)

        # 2) 生成前缀 K/V（先得到 (B,Tvp,2D)）
        kv_prefix = self.prefix_encoder(batch_size=B).to(device)  # (B,Tvp,2D)
        Tvp = kv_prefix.shape[1]
        Kp, Vp = torch.split(kv_prefix, self.d_model, dim=-1)     # (B,Tvp,D), (B,Tvp,D)

        # 3) 拼接到 keys/vals 的时间维度左侧（prefix 在序列最前）
        K_cat = torch.cat([Kp, K_base_in], dim=1)  # (B, Tvp+Tk, D)
        V_cat = torch.cat([Vp, V_base_in], dim=1)  # (B, Tvp+Tk, D)

        # 4) 构造/合并注意力掩码（支持因果与外部mask）
        attn_mask_exp = self._build_attn_mask(
            B=B, Tq=Tq, Tk_orig=Tk, Tvp=Tvp, device=device, causal=self.cfg.causal, attn_mask=attn_mask
        )  # (B,1,Tq,Tvp+Tk) or None

        # 5) 调用基础 MHA（它内部会完成投影与拆头）
        #    注意：这里将 x_q 作为查询，K_cat/V_cat 作为键值输入
        out = self.mha(x_q=x_q, x_kv=torch.stack([K_cat, V_cat], dim=0), attn_mask=attn_mask_exp)
        # 上面 MultiHeadAttention 期望 x_kv 是 (B,T,D)，而我们传的是 K/V 叠在一起…
        # 为了保持 MultiHeadAttention 简洁，不去修改它，这里我们拆开调用一次 K 和 V 的 proj。
        # ——> 简化做法：把 K_cat/V_cat 直接塞回去不合适，我们下面重写一次 forward 以对齐：
        return self._forward_with_prefix(x_q, K_cat, V_cat, attn_mask_exp)

    def _forward_with_prefix(
        self,
        x_q: torch.Tensor,   # (B,Tq,D)
        K_cat: torch.Tensor, # (B,Tk_new,D)
        V_cat: torch.Tensor, # (B,Tk_new,D)
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # 复用 mha 的投影，但分别对 K_cat/V_cat 做 k_proj/v_proj
        Q = self.mha._shape(self.mha.q_proj(x_q))         # (B,H,Tq,Hd)
        K = self.mha._shape(self.mha.k_proj(K_cat))       # (B,H,Tk_new,Hd)
        V = self.mha._shape(self.mha.v_proj(V_cat))       # (B,H,Tk_new,Hd)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,Tq,Tk_new)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B,H,Tq,Hd)
        out = out.transpose(1, 2).contiguous().view(x_q.size(0), x_q.size(1), self.d_model)
        out = self.mha.o_proj(out)
        return out


# -------------------------------
# 4) 简单的 Block + 训练/推理示例
# -------------------------------
class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pfx_cfg: PTuningConfig):
        super().__init__()
        self.attn = PTuningMHA(d_model, n_heads, pfx_cfg)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        x = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + h
        h = x
        x = self.ffn(self.ln2(x))
        x = x + h
        return x


def make_causal_mask(B: int, T: int, device: torch.device) -> torch.Tensor:
    # (1,1,T,T) 下三角=0, 上三角=-inf
    mask = torch.full((T, T), 0.0, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask[mask > 0] = float("-inf")
    return mask.view(1, 1, T, T).expand(B, 1, T, T)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, D, H = 4, 32, 128, 8
    pfx_cfg = PTuningConfig(num_virtual_tokens=12, encoder_hidden=256, causal=True)

    block = TinyBlock(D, H, pfx_cfg).to(device)

    # 冻结主干，只训练前缀（P-Tuning）
    for n, p in block.named_parameters():
        p.requires_grad = ("prefix_encoder" in n)

    # 验证只剩前缀在训练
    trainable = [n for n, p in block.named_parameters() if p.requires_grad]
    print(f"Trainable params count: {sum(p.numel() for n,p in block.named_parameters() if p.requires_grad)}")
    print("Trainables:", trainable)

    # 伪数据
    x = torch.randn(B, T, D, device=device)
    y = torch.randn(B, T, D, device=device)

    # 因果掩码（注意 PTuningMHA 内部会自动扩展以包含前缀）
    attn_mask = make_causal_mask(B, T, device)

    # 简单训练几步
    opt = torch.optim.AdamW([p for p in block.parameters() if p.requires_grad], lr=1e-3)
    block.train()
    for step in range(20):
        opt.zero_grad()
        out = block(x, attn_mask=attn_mask)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        if (step + 1) % 5 == 0:
            print(f"step {step+1:02d} | loss {loss.item():.4f}")

    # 推理：同一路径，prefix 会被自动注入（无需额外代码）
    block.eval()
    with torch.no_grad():
        out = block(x, attn_mask=attn_mask)
        print("inference output shape:", out.shape)
