from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel

# =======================================================
# 0. PEFT 配置与 LoRA 包装函数
# =======================================================

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"] # Qwen模型通常针对 q/v 投影层

def apply_lora(model):
    """应用 LoRA 配置到基础模型上"""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 使用 get_peft_model 包装模型，只有 LoRA 参数可训练
    return get_peft_model(model, lora_config)

# 移除了 custom_collate_fn 的定义和引用 (依赖 set_format 解决问题)

# 1. 数据加载（只使用 1% 的数据）
dataset = load_dataset('lvwerra/stack-exchange-paired', split='train', data_dir='data/finetune')

# 使用 1% 的数据
dataset = dataset.select(range(int(len(dataset) * 0.001)))

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

# 3. 预处理函数 (保持不变)
def preprocess(dataset, tokenizer, max_length=512):
    def _encode(batch):
        prompts  = [f"问题：{q}\n回答：" for q in batch["question"]]
        chosens  = batch["response_j"]
        rejecteds= batch["response_k"]

        seq_chosen  = [p + c for p, c in zip(prompts, chosens)]
        seq_rejected= [p + r for p, r in zip(prompts, rejecteds)]

        enc_c = tokenizer(seq_chosen, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        enc_r = tokenizer(seq_rejected, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

        input_ids = torch.stack([enc_c["input_ids"], enc_r["input_ids"]], axis=1)
        attention_mask = torch.stack([enc_c["attention_mask"], enc_r["attention_mask"]], axis=1)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return dataset.map(_encode, batched=True, remove_columns=dataset.column_names)

# 数据预处理
tokenized = preprocess(dataset, tokenizer)

# 强制数据集输出 PyTorch 张量格式
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask']) 

# =======================================================
# 4. 模型初始化 (集成 LoRA)
# =======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Policy Model 并应用 LoRA
base_policy_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6b")
policy_model = apply_lora(base_policy_model).to(device) # LoRA 模型
policy_model.print_trainable_parameters() # 打印可训练参数数量

# Reference Model: 保持为全参数模型，冻结参数
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6b").to(device)
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

# =======================================================
# 5. DPO 核心辅助函数与损失函数 (新增 KL 损失)
# =======================================================

# 5. 定义获取对数几率的辅助函数 (保持不变)
def get_log_probs(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    log_probs = log_probs * shift_attention_mask
    
    return log_probs.sum(dim=-1)
# 定义获取 Logits 和序列对数几率的辅助函数
def get_model_outputs(model, input_ids, attention_mask, labels):
    """返回 Logits（用于 KL）和序列总对数几率（用于 DPO）。"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 计算序列总对数几率 (用于 DPO 损失)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    seq_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    seq_log_probs = seq_log_probs * shift_attention_mask
    
    # 返回 Logits (用于 KL) 和 序列总对数几率 (用于 DPO)
    return shift_logits, seq_log_probs.sum(dim=-1)
# 6. 定义 DPO 损失函数 (保持不变)
def dpo_loss_policy(
    policy_chosen_log_probs, policy_rejected_log_probs,
    ref_chosen_log_probs, ref_rejected_log_probs,
    beta=0.1
):
    log_ratio_chosen = policy_chosen_log_probs - ref_chosen_log_probs
    log_ratio_rejected = policy_rejected_log_probs - ref_rejected_log_probs
    ratio_diff = log_ratio_chosen - log_ratio_rejected
    
    loss = -F.logsigmoid(beta * ratio_diff).mean()
    
    return loss

# 6b. 定义 KL 散度约束损失
def kl_divergence_loss(policy_logits, ref_logits, attention_mask_shifted):
    """
    计算 Policy Logits 和 Reference Logits 之间的 KL 散度。
    Logits 形状应为 (Batch * Sequence_Length, Vocab_Size)
    """
    # 1. 计算 Log Softmax
    # target (ref) 必须是 log 概率
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    # input (policy) 必须是 log 概率
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    
    # 2. KL 散度计算 (KL(Policy || Reference))
    # reduction="none" 允许我们使用 attention mask
    kl_div_all = F.kl_div(
        policy_log_probs, # input (log P)
        ref_log_probs,    # target (log Q), log_target=True
        reduction="none", 
        log_target=True
    )
    
    # 3. 对词汇表维度求和得到每个 token 的 KL 散度
    kl_per_token = kl_div_all.sum(dim=-1)

    # 4. 应用 mask 并求平均 (只计算非 padding token 的 KL)
    # 将 mask 展平以匹配 kl_per_token 的形状 (B * L)
    kl_per_token_masked = kl_per_token * attention_mask_shifted.flatten()
    
    # 5. 对所有非 padding token 求平均
    sum_kl = kl_per_token_masked.sum()
    num_tokens = attention_mask_shifted.sum()
    
    # 避免除以零
    kl_loss = sum_kl / num_tokens if num_tokens > 0 else 0.0
    
    return kl_loss
# =======================================================
# 7. DPO 训练函数 (集成 KL 约束)
# =======================================================

# 新增 KL 约束超参数
KL_LAMBDA = 0.05 # KL 惩罚的权重，需要根据实际效果调整

def train_dpo_policy(dataset, policy_model, ref_model, optimizer, epochs=3, batch_size=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    policy_model.train()
    ref_model.eval()

    print(f"Starting DPO training on device: {policy_model.device}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        kl_epoch_loss = 0
        
        for i, batch in enumerate(dataloader):
            
            input_ids = batch["input_ids"].to(policy_model.device)
            attention_mask = batch["attention_mask"].to(policy_model.device)

            # DPO 损失只需 Chosen/Rejected，我们使用 Chosen 的数据来计算 KL
            input_ids_c, input_ids_r = input_ids[:, 0, :], input_ids[:, 1, :]
            mask_c, mask_r = attention_mask[:, 0, :], attention_mask[:, 1, :]
            # --- 1. Policy & Reference Model 输出 (Logits 和 Log Probs) ---
            # KL 损失通常只在 chosen 序列上计算
            policy_chosen_logits, policy_chosen_log_probs = get_model_outputs(policy_model, input_ids_c, mask_c, labels=input_ids_c)
            policy_rejected_logits, policy_rejected_log_probs = get_model_outputs(policy_model, input_ids_r, mask_r, labels=input_ids_r)
            
            with torch.no_grad():
                ref_chosen_logits, ref_chosen_log_probs = get_model_outputs(ref_model, input_ids_c, mask_c, labels=input_ids_c)
                # DPO 损失需要 rejected 的 Log Probs，但 KL 只需要 chosen 的 logits
                _, ref_rejected_log_probs = get_model_outputs(ref_model, input_ids_r, mask_r, labels=input_ids_r)
            
            # --- 2. DPO 损失 ---
            dpo_loss = dpo_loss_policy(
                policy_chosen_log_probs, policy_rejected_log_probs,
                ref_chosen_log_probs, ref_rejected_log_probs
            )
            
            # --- 3. KL 约束损失 ---
            # 准备 mask: 截断并展平 chosen 序列的 attention mask (与 logits 匹配)
            mask_c_shifted = mask_c[:, 1:].contiguous()
            
            # 重塑 Logits 以匹配 KL 损失函数的输入要求 (B * L, V)
            B, L, V = policy_chosen_logits.shape
            
            kl_loss = kl_divergence_loss(
                policy_chosen_logits.view(-1, V), 
                ref_chosen_logits.view(-1, V),
                mask_c_shifted
            )

         
            # --- 5. 总损失 ---
            total_loss = dpo_loss + KL_LAMBDA * kl_loss
            
            epoch_loss += dpo_loss.item()
            kl_epoch_loss += kl_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if (i + 1) % 1 == 0:
                print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, DPO Loss: {dpo_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

        print(f"\nEpoch {epoch+1} DPO Loss (Avg): {epoch_loss / len(dataloader):.4f}")
        print(f"Epoch {epoch+1} KL Loss (Avg): {kl_epoch_loss / len(dataloader):.4f}\n")

# 8. 初始化优化器 (只优化 Policy Model 的可训练参数)
# 注意：当使用 PEFT 时，优化器会自动检测并只优化 LoRA 权重。
optimizer = AdamW(policy_model.parameters(), lr=5e-6)

# 9. 开始训练
train_dpo_policy(tokenized, policy_model, ref_model, optimizer, epochs=3)

# 10. 保存模型 (保存 LoRA 适配器)
output_dir = "outputs/dpo_lora_kl_model"
print(f"Saving LoRA model to {output_dir}")
# 保存 LoRA 权重，而不是整个 Policy Model
policy_model.save_pretrained(output_dir) 
tokenizer.save_pretrained(output_dir)