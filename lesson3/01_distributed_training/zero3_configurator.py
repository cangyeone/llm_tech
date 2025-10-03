"""ZeRO-3 与混合精度配置生成器。

根据显存与 batch size 自动生成 DeepSpeed 配置文件，支持：
- ZeRO Stage 3 参数分片
- 混合精度（fp16/bf16）
- 梯度检查点
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ZeroConfig:
    micro_batch_size: int
    gradient_accumulation: int
    stage: int = 3
    offload: bool = False
    bf16: bool = True
    output_path: Path = Path("./ds_zero3.json")


def build_config(cfg: ZeroConfig) -> dict:
    return {
        "train_batch_size": cfg.micro_batch_size * cfg.gradient_accumulation,
        "zero_optimization": {
            "stage": cfg.stage,
            "offload_param": {
                "device": "cpu" if cfg.offload else "none",
                "pin_memory": cfg.offload,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": cfg.bf16},
        "gradient_accumulation_steps": cfg.gradient_accumulation,
        "steps_per_print": 50,
        "gradient_clipping": 1.0,
    }


def save_config(config: dict, path: Path) -> None:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    config = ZeroConfig(micro_batch_size=4, gradient_accumulation=8)
    ds_config = build_config(config)
    save_config(ds_config, config.output_path)
    print(f"DeepSpeed 配置已保存至 {config.output_path}")
