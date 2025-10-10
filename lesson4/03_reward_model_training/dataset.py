from datasets import load_dataset
import random
import textwrap

# 加载数据集
dataset = load_dataset('lvwerra/stack-exchange-paired', split='train', data_dir='data/finetune')

print("列名：", dataset.column_names)
print("样本总数：", len(dataset))

def pretty_print_example(ex, idx=None, width=100, max_lines=6):
    def trunc(s):
        s = s.replace("\n", " ")
        s = textwrap.shorten(s, width=width, placeholder=" …")
        return s
    head = f"\n===== 示例 {idx if idx is not None else ''} ====="
    q   = trunc(ex.get("question", ""))
    cj  = trunc(ex.get("response_j", ""))
    rk  = trunc(ex.get("response_k", ""))
    print(head)
    print(f"[Question]\n{q}")
    print(f"[Chosen  (response_j)]\n{cj}")
    print(f"[Rejected(response_k)]\n{rk}")

# 随机抽 3 条样本
k = 3
for i in random.sample(range(len(dataset)), k=min(k, len(dataset))):
    pretty_print_example(dataset[i], idx=i)

# —— 如果你想看前 5 条并在 DataFrame 里快速浏览（可选）——
try:
    import pandas as pd
    cols = [c for c in ["question", "response_j", "response_k"] if c in dataset.column_names]
    df = dataset.select(range(min(5, len(dataset)))).to_pandas()[cols]
    print("\nDataFrame 预览（前 5 条，已自动截断列宽）：")
    with pd.option_context('display.max_colwidth', 120):
        print(df)
except Exception as e:
    print("（可选 DataFrame 预览失败，可能缺少 pandas，忽略即可）", e)
