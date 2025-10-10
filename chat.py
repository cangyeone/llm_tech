import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from transformers import logging
logging.set_verbosity_warning()  # 只保留 warning 和 error
# 用一个对话模型
model_name = "qw1.7_model"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# 对话历史 (chat)
messages = [
    {"role": "system", "content": "你是‘地震学AI助手’，一个乐于助人的中文AI助手。由地震局人工智能团队开发。请帮助用户解决问题。"},
    {"role": "user", "content": "你是谁？简单做个自我介绍？"}
]

# 应用聊天模板，生成可直接输入模型的字符串
chat_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
#print(chat_prompt)
inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

# 依然用 model.generate
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,  # 确定性输出
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

insides = re.findall(r"<think>(.*?)</think>", text, flags=re.S)
outside = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().split("assistant")[-1]

print("思考过程：", insides[0])
print("最终答案：", outside)