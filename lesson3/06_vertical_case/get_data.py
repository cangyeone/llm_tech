"""
pip install -U datasets
"""

from datasets import load_dataset

# 有些 loader 需要执行自定义脚本，建议加 trust_remote_code=True
ds = load_dataset("ibm-research/finqa", trust_remote_code=True)

print(ds)                 # 查看包含的 split
print(ds["train"][0].keys())  # 字段结构
ex = ds["train"][0]
print(ex["filing_type"], ex["id"])
print(ex["qa"]["question"])
print(ex["qa"]["exe_ans"])
