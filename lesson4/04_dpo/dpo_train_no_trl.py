from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 移除了 custom_collate_fn 的定义

# 1. 数据加载（只使用 1% 的数据）
dataset = load_dataset('lvwerra/stack-exchange-paired', split='train', data_dir='data/finetune')

# 使用 0.1% 的数据
dataset = dataset.select(range(int(len(dataset) * 0.001)))

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6b")
# 设置填充 token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

# 3. 预处理函数：将每一条数据构造成符合输入格式
def preprocess(dataset, tokenizer, max_length=512):
    def _encode(batch):
        prompts  = [f"问题：{q}\n回答：" for q in batch["question"]]
        chosens  = batch["response_j"]
        rejecteds= batch["response_k"]

        seq_chosen  = [p + c for p, c in zip(prompts, chosens)]
        seq_rejected= [p + r for p, r in zip(prompts, rejecteds)]

        # 编码 chosen 序列
        enc_c = tokenizer(seq_chosen, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        # 编码 rejected 序列
        enc_r = tokenizer(seq_rejected, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

        # Stack 输入: (B, 2, L) -> 2代表 chosen 和 rejected
        input_ids = torch.stack([enc_c["input_ids"], enc_r["input_ids"]], axis=1)
        attention_mask = torch.stack([enc_c["attention_mask"], enc_r["attention_mask"]], axis=1)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return dataset.map(_encode, batched=True, remove_columns=dataset.column_names)

# 数据预处理
tokenized = preprocess(dataset, tokenizer)

# =======================================================
# !!! 关键修复步骤: 强制数据集输出 PyTorch 张量格式
# =======================================================
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask']) 

# 4. 初始化 Policy Model 和 Reference Model
device = "cuda" if torch.cuda.is_available() else "cpu"

policy_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6b").to(device)
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6b").to(device)
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

# 5. 定义获取对数几率的辅助函数
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

# 6. 定义 DPO 损失函数 (Policy 版本)
def dpo_loss_policy(
    policy_chosen_log_probs,
    policy_rejected_log_probs,
    ref_chosen_log_probs,
    ref_rejected_log_probs,
    beta=0.1
):
    log_ratio_chosen = policy_chosen_log_probs - ref_chosen_log_probs
    log_ratio_rejected = policy_rejected_log_probs - ref_rejected_log_probs
    
    ratio_diff = log_ratio_chosen - log_ratio_rejected
    
    loss = -F.logsigmoid(beta * ratio_diff).mean()
    
    return loss

# 7. DPO 训练函数
def train_dpo_policy(dataset, policy_model, ref_model, optimizer, epochs=3, batch_size=8):
    # !!! 关键修改: 移除了 collate_fn 参数
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    policy_model.train()
    ref_model.eval()

    print(f"Starting DPO training on device: {policy_model.device}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            # 由于 set_format('torch')，batch['input_ids'] 现在是 Tensor
            input_ids = batch["input_ids"].to(policy_model.device)
            attention_mask = batch["attention_mask"].to(policy_model.device)

            input_ids_c, input_ids_r = input_ids[:, 0, :], input_ids[:, 1, :]
            mask_c, mask_r = attention_mask[:, 0, :], attention_mask[:, 1, :]

            # Policy Log Probs
            policy_chosen_log_probs   = get_log_probs(policy_model, input_ids_c, mask_c, labels=input_ids_c)
            policy_rejected_log_probs = get_log_probs(policy_model, input_ids_r, mask_r, labels=input_ids_r)
            
            # Reference Log Probs (No grad)
            with torch.no_grad():
                ref_chosen_log_probs   = get_log_probs(ref_model, input_ids_c, mask_c, labels=input_ids_c)
                ref_rejected_log_probs = get_log_probs(ref_model, input_ids_r, mask_r, labels=input_ids_r)
            
            # 计算 DPO 损失
            loss = dpo_loss_policy(
                policy_chosen_log_probs, policy_rejected_log_probs,
                ref_chosen_log_probs, ref_rejected_log_probs
            )
            
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 1 == 0:
                print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        print(f"\nEpoch {epoch+1} DPO Loss (Avg): {epoch_loss / len(dataloader):.4f}\n")

# 8. 初始化优化器 (只优化 Policy Model 的参数)
optimizer = AdamW(policy_model.parameters(), lr=5e-6)

# 9. 开始训练
train_dpo_policy(tokenized, policy_model, ref_model, optimizer, epochs=3)

# 10. 保存模型 (只保存 Policy Model)
output_dir = "outputs/dpo_policy_model"
print(f"Saving model to {output_dir}")
policy_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)