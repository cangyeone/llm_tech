"""8 卡 671B-DS 分布式训练配置样例。

生成包含流水线并行、张量并行与 ZeRO 优化的 DeepSpeed 配置文件。
"""

from __future__ import annotations

import json
from pathlib import Path


CONFIG = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,
    "steps_per_print": 20,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "reduce_scatter": True,
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "pipeline": {"parallel_size": 2},
    "tensor": {"parallel_size": 4},
    "activation_checkpointing": {"partition_activations": True},
}


if __name__ == "__main__":
    path = Path("./ds_config_8gpu.json")
    path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    print(f"配置文件已保存至 {path}")
