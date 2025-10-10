# -*- coding: utf-8 -*-
"""
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
# 浏览器打开 http://127.0.0.1:5000

运行后在 ./wandb_offline/ 看到离线目录（可拷贝、打包）。
若有线上 W&B 项目，切到在线：
export WANDB_MODE=online
wandb login   # 输入你的 API Key
# 再次运行脚本即可自动同步
"""
import os, time, random
import torch, torch.nn as nn, torch.optim as optim
import mlflow
import mlflow.pytorch
import wandb

# ===== 推荐：代码内兜底离线配置（也可用环境变量） =====
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DIR", "./wandb_offline")

# ----- 伪数据与小模型（演示用）-----
class TinyNet(nn.Module):
    def __init__(self, in_dim=10, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim))
    def forward(self, x): return self.net(x)

def fake_batch(bs=32, in_dim=10, num_classes=2):
    x = torch.randn(bs, in_dim)
    y = torch.randint(0, num_classes, (bs,))
    return x, y

# ====== 超参 ======
params = dict(
    project="alignment-demo",
    run_name=f"tiny_dpo_probe_{int(time.time())}",
    lr=5e-4,
    epochs=3,
    batch_size=32,
    in_dim=10,
    num_classes=2,
    note="示例：同时把指标记到 MLflow 与 W&B；并将模型注册为版本",
)

# ====== 1) 初始化 W&B 与 MLflow ======
wandb.init(project=params["project"], name=params["run_name"], config=params, mode=os.getenv("WANDB_MODE", "offline"))
mlflow.set_experiment(params["project"])

with mlflow.start_run(run_name=params["run_name"]) as run:
    run_id = run.info.run_id

    # 记录超参
    mlflow.log_params({k: v for k, v in params.items() if k not in ["project", "run_name", "note"]})
    wandb.config.update(params)

    # ====== 2) 构建模型与优化器 ======
    model = TinyNet(in_dim=params["in_dim"], out_dim=params["num_classes"])
    opt = optim.AdamW(model.parameters(), lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # ====== 3) 训练 & 同步记录指标 ======
    global_step = 0
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        steps = 50  # 演示：每个 epoch 跑 50 个 step

        for _ in range(steps):
            x, y = fake_batch(params["batch_size"], params["in_dim"], params["num_classes"])
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(); loss.backward(); opt.step()

            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == y).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc  += acc
            global_step += 1

            # —— 每步日志（可选，演示细粒度记录）——
            mlflow.log_metrics({"train/loss": loss.item(), "train/acc": acc}, step=global_step)
            wandb.log({"train/loss": loss.item(), "train/acc": acc, "step": global_step})

        # —— 每个 epoch 的汇总指标 —— 
        avg_loss = epoch_loss / steps
        avg_acc  = epoch_acc  / steps
        mlflow.log_metrics({"epoch/loss": avg_loss, "epoch/acc": avg_acc}, step=epoch)
        wandb.log({"epoch/loss": avg_loss, "epoch/acc": avg_acc, "epoch": epoch})

        # —— 记录一个“示例输出”到两边（可换成文本/图片/表格）——
        sample_x, _ = fake_batch(1, params["in_dim"], params["num_classes"])
        sample_logits = model(sample_x)[0].tolist()
        sample_note = {"example/logits": sample_logits, "epoch": epoch}
        mlflow.log_dict(sample_note, f"examples/epoch_{epoch}.json")
        wandb.log(sample_note)

    # ====== 4) 保存与登记模型 ======
    # 保存到当前 run 的 artifacts
    mlflow.pytorch.log_model(model, artifact_path="model")
    # 注册为“模型版本”：需要后端使用 DB（如上设置 sqlite:///mlruns.db）
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = "tiny_alignment_model"

    print(f"Registering model from {model_uri} ...")
    mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    print("Registered:", mv.name, "version:", mv.version)

    # 可选：打个“阶段”标签（Staging/Production）
    # from mlflow.tracking import MlflowClient
    # client = MlflowClient()
    # client.transition_model_version_stage(name=registered_model_name, version=mv.version, stage="Staging")

    # 记录一些 run-level 文本/标签
    mlflow.set_tag("note", params["note"])
    wandb.summary["final/acc"] = avg_acc
    print("Done. Run ID:", run_id)
