
# 教学实验：同时使用 MLflow 与 Weights & Biases（W&B）进行训练追踪与模型注册

本教学脚本展示**如何在同一次训练中，既把指标/参数写入 MLflow，又同步到 W&B**，并演示**模型保存与注册（Model Registry）**的最小化流程。脚本使用一个玩具全连接网络 `TinyNet` 和伪造数据，便于课堂快速跑通 **端到端实验追踪 → 指标可视化 → 模型制品与版本管理**。

---

## 目录
- [教学实验：同时使用 MLflow 与 Weights \& Biases（W\&B）进行训练追踪与模型注册](#教学实验同时使用-mlflow-与-weights--biaseswb进行训练追踪与模型注册)
  - [目录](#目录)
  - [环境准备](#环境准备)
  - [一键运行](#一键运行)
  - [功能概览](#功能概览)
  - [脚本结构与函数说明](#脚本结构与函数说明)
    - [`TinyNet`](#tinynet)
    - [`fake_batch`](#fake_batch)
    - [训练主循环（指标同步）](#训练主循环指标同步)
    - [模型保存与注册（MLflow Model Registry）](#模型保存与注册mlflow-model-registry)
  - [可视化与产物查看](#可视化与产物查看)
    - [MLflow](#mlflow)
    - [W\&B](#wb)
  - [关键超参与可改项](#关键超参与可改项)
  - [常见问题与排错](#常见问题与排错)
  - [课堂练习与扩展](#课堂练习与扩展)
  - [许可证](#许可证)

---

## 环境准备

1. 安装依赖：
   ```bash
   pip install torch mlflow wandb
   ```

2. **（可选）配置 W&B 在线模式**  
   - 离线默认：脚本内已设置
     ```bash
     export WANDB_MODE=offline
     ```
     运行后会在本地生成 `./wandb_offline/` 目录。
   - 切换到在线：
     ```bash
     export WANDB_MODE=online
     wandb login   # 按提示粘贴 API Key
     ```

3. **MLflow UI（本地 SQLite 后端）**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
   ```
   打开浏览器访问：`http://127.0.0.1:5000`

> 注：脚本内部已设置默认环境变量  
> ```python
> os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
> os.environ.setdefault("WANDB_MODE", "offline")
> os.environ.setdefault("WANDB_DIR", "./wandb_offline")
> ```
> 你也可以在外部通过环境变量覆盖。

---

## 一键运行

将脚本保存为 `lesson5/04_model_versioning/version_tracking.py`，直接执行：
```bash
python lesson5/04_model_versioning/version_tracking.py
```

运行结束后你将看到：
- 控制台打印训练进度、模型注册信息（Model URI 与版本号）；
- `mlruns.db`（SQLite）与 `./mlruns/` 目录（包含 artifacts）；
- `./wandb_offline/`（或在线模式下同步到你的 W&B 项目）。

---

## 功能概览

- **双通道日志**：把训练过程中的**参数**（params）与**指标**（metrics）同时写入 *MLflow* 与 *W&B*。
- **细粒度与汇总**：每步（step）与每个 epoch 的汇总指标均有记录。
- **示例产物**：每个 epoch 额外记录一份“示例输出”JSON（`examples/epoch_*.json`）。
- **模型制品**：以 `mlflow.pytorch.log_model` 写入当前 run 的 artifacts。
- **模型注册**：把当前 run 的模型注册为 `tiny_alignment_model` 的一个新版本（需要可写的后端存储，如 SQLite）。

---

## 脚本结构与函数说明

### `TinyNet`

```python
class TinyNet(nn.Module):
    def __init__(self, in_dim=10, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim))
    def forward(self, x): return self.net(x)
```
- **作用**：一个两层 MLP，用于分类演示。
- **输入/输出**：输入形状 `[batch_size, in_dim]`，输出维度为 `out_dim` 的 logits。

### `fake_batch`

```python
def fake_batch(bs=32, in_dim=10, num_classes=2):
    x = torch.randn(bs, in_dim)
    y = torch.randint(0, num_classes, (bs,))
    return x, y
```
- **作用**：生成随机特征与标签，便于不依赖外部数据即可运行。
- **注意**：用于教学演示，不代表真实训练数据分布。

### 训练主循环（指标同步）

- **优化器**：`AdamW(lr=params["lr"])`
- **损失函数**：`CrossEntropyLoss()`
- **记录频率**：
  - **每步**：`train/loss`, `train/acc`
  - **每个 epoch**：`epoch/loss`, `epoch/acc`
- **双端同步**：
  - MLflow：`mlflow.log_params(...)`, `mlflow.log_metrics(...)`
  - W&B：`wandb.log(...)`（并把 `params` 注入 `wandb.config`）
- **示例产物**：
  ```python
  sample_note = {"example/logits": sample_logits, "epoch": epoch}
  mlflow.log_dict(sample_note, f"examples/epoch_{epoch}.json")
  wandb.log(sample_note)
  ```

### 模型保存与注册（MLflow Model Registry）

1. **保存到当前 run 的 artifacts**：
   ```python
   mlflow.pytorch.log_model(model, artifact_path="model")
   ```
2. **拼接 Model URI**：
   ```python
   model_uri = f"runs:/{run_id}/model"
   ```
3. **注册为模型版本**：
   ```python
   registered_model_name = "tiny_alignment_model"
   mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
   ```
4. **（可选）阶段切换**：设置为 `Staging/Production` 等（需要 `MlflowClient`）。

> **前提**：`MLFLOW_TRACKING_URI` 指向**可写的后端存储**（本示例使用 SQLite）。无后端存储时无法使用 Model Registry。

---

## 可视化与产物查看

### MLflow
- 启动 UI：`mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000`
- 在 `Experiments` 里选中 `alignment-demo` 项目：
  - 比较不同 run 的 loss/acc 曲线；
  - 在 `Artifacts` 里查看 `model/` 与 `examples/epoch_*.json`；
  - 在 `Models`（左侧）查看 `tiny_alignment_model` 的版本与阶段。

### W&B
- 离线：检查 `./wandb_offline/` 目录（可打包共享）。
- 在线：登录后在 W&B 项目页查看 run 概览、历史曲线与系统信息。

---

## 关键超参与可改项

脚本集中在 `params` 字典：
```python
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
```
- `project`：MLflow 实验名 / W&B 项目名；
- `run_name`：一次实验的唯一名称；
- `lr/epochs/batch_size`：训练超参；
- `in_dim/num_classes`：`TinyNet` 输入/类别数；
- `note`：会写入 MLflow 的 tag 与 W&B 的 summary。

**建议扩展**：
- 支持 CLI 参数（`argparse`）覆盖 `params`；
- 增加验证集与 `eval_*` 指标；
- 记录混淆矩阵、模型大小、推理延迟等。

---

## 常见问题与排错

1. **W&B 离线/在线切换不起作用？**  
   - 优先级：**环境变量 > 代码默认值**。请确认是否在同一 Shell 会话中导出变量并重新运行脚本。

2. **Model Registry 报错（数据库/权限）**  
   - 确认 `MLFLOW_TRACKING_URI` 指向可写的后端（如 `sqlite:///mlruns.db`）。
   - 若远程服务器，需配置文件系统权限与端口映射。

3. **没有看到 artifacts？**  
   - 检查 run 是否成功结束；
   - 在 MLflow UI -> 具体 run 下的 `Artifacts` 面板查看。

4. **如何把 W&B 离线目录迁移到其他机器查看？**  
   - 打包 `./wandb_offline/` 并复制到目标机器，或改为在线模式重新跑一次同步。

5. **想要同时记录学习率、梯度范数、系统显存等？**  
   - 可在训练循环中增加自定义度量并 `mlflow.log_metrics` / `wandb.log`。

---

## 课堂练习与扩展

- **练习 A**：把 `TinyNet` 换成你自己的模型，加入验证集与早停（Early Stopping）。  
- **练习 B**：引入 `mlflow.register_model` 之后，练习用 `MlflowClient` 切换 `Staging/Production`。  
- **练习 C**：在 W&B 上增加表格/图像日志（如样本可视化）。  
- **练习 D**：把 run 的关键指标导出为 CSV，并在 Notebook 中画对比图。  
- **练习 E**：用 Docker 打包环境，确保在不同机器上可复现。

---

## 许可证

本教学脚本与文档仅用于**教学/研究**。使用 MLflow、W&B 与相关依赖时，请遵守各自的开源许可与服务条款。
