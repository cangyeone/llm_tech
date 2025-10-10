"""P-Tuning v2 可学习提示词示例（修订版）。

展示内容：
1) 使用 peft.PromptEncoder(P-Tuning v2) 在每层 Attention 注入可学习前缀。
2) HF Trainer + DataCollatorForLanguageModeling 进行 Causal LM 训练。
3) 演示推理阶段的调用（自动注入虚拟 tokens）。

环境：Transformers >= 4.40，peft >= 0.10（或相近）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import Dataset
from peft import PromptEncoderConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@dataclass
class PTuningConfig:
    model_name: str = "Qwen/Qwen3-0.6B"     # 注意大小写：官方模型多为 0.6B/1.8B 等格式
    prompt_length: int = 32
    task_type: str = "CAUSAL_LM"
    output_dir: str = "./outputs/ptuning"
    bf16: bool = False                      # CUDA+bf16 可开；mac/MPS 保持 False
    fp16: bool = False
    epochs: int = 1
    lr: float = 3e-4
    per_device_train_batch_size: int = 2
    logging_steps: int = 5


def build_dataset() -> Dataset:
    samples = [
        {
            "instruction": "请写一条客服问候语",
            "input": "背景：用户首次进入商城",
            "output": "您好，欢迎来到我们的商城，有任何问题随时咨询我。",
        }
        for _ in range(200)
    ]
    return Dataset.from_list(samples)


def format_sample(sample: Dict[str, str]) -> str:
    # 简单的指令模板；真实场景可替换为更完善的对话模板
    return f"指令：{sample['instruction']}\n输入：{sample['input']}\n回答：{sample['output']}"


def tokenize(dataset: Dataset, tokenizer) -> Dataset:
    return dataset.map(
        lambda sample: tokenizer(
            format_sample(sample),
            max_length=512,
            truncation=True,
            padding="max_length",
        ),
        remove_columns=dataset.column_names,
    )


def main() -> None:
    cfg = PTuningConfig()

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Base model ===
    # 统一 dtype & device；mac/MPS 推荐不设 fp16/bf16
    dtype = torch.bfloat16 if (cfg.bf16 and torch.cuda.is_available()) else None
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    # pad id
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # === P-Tuning v2 配置 ===
    # PromptEncoder 会为每层 Attention 生成可学习的 prefix K/V
    prompt_config = PromptEncoderConfig(
        task_type=cfg.task_type,
        num_virtual_tokens=cfg.prompt_length,     # prefix 长度
        encoder_hidden_size=base_model.config.hidden_size // 2,  # 生成前缀的小MLP隐层
        # encoder_type 可选 'MLP'/'LSTM'（新版本默认 MLP），如需 LSTM: encoder_type="LSTM"
    )
    model = get_peft_model(base_model, prompt_config)
    model.print_trainable_parameters()

    # === 数据 ===
    dataset = build_dataset().train_test_split(test_size=0.1, seed=42)
    tokenized_train = tokenize(dataset["train"], tokenizer)
    tokenized_eval = tokenize(dataset["test"], tokenizer)

    # 让 collator 为 CausalLM 生成 labels
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === 训练参数 ===
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        gradient_checkpointing=False,  # 如需省显存可开启；P-Tuning v2 本身占用已很低
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # === 推理 ===
    model.eval()
    prompt_text = "请写一句积极的评价：这家店的服务如何？"
    inputs = tokenizer(prompt_text, return_tensors="pt")
    # 放到同一设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
