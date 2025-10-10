"""
export NCCL_SOCKET_IFNAME=eth0
torchrun --standalone --nproc_per_node=8 hf_ddp_minimal.py \
  --model_name gpt2 \
  --output_dir outputs_hf_ddp \
  --block_size 256 \
  --per_device_bs 1 \
  --grad_accum 32 \
  --epochs 1 \
  --fp16

"""

# hf_ddp_minimal.py
import math, argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
)

SMALL_CORPUS = [
    "Deep learning enables powerful language models.",
    "Distributed training with DDP scales across GPUs.",
    "ZeRO-3 partitions optimizer states, gradients and parameters.",
    "Transformers fine-tuning can be done with Trainer."
] * 200  # 800 条样本，演示足够

def build_ds(tokenizer, block_size=256):
    # 拼接后切块
    ids = tokenizer("\n\n".join(SMALL_CORPUS), add_special_tokens=False)["input_ids"]
    chunks = [ids[i:i+block_size] for i in range(0, len(ids)-block_size+1, block_size)]
    return Dataset.from_dict({"input_ids": chunks})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--output_dir", default="outputs_hf_ddp")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = build_ds(tok, args.block_size)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    tr_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=1,
        fp16=args.fp16, bf16=args.bf16,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=tr_args,
        train_dataset=ds.select(range(int(0.9*len(ds)))),
        eval_dataset=ds.select(range(int(0.9*len(ds)), len(ds))),
        data_collator=collator, tokenizer=tok
    )

    trainer.train()
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
    print(metrics)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
