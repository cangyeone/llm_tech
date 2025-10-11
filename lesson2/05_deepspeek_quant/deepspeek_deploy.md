# DeepSpeed + bitsandbytes 量化部署示例：使用说明与函数文档

> 版本：v1.0  
> 适用脚本：`ds_bnb_infer.py`（基于你提供的教学代码整理）  
> 运行环境：**GPU**（NVIDIA）  
> 关键依赖：`deepspeed>=0.10`、`transformers`、`torch`、`bitsandbytes`（可选，用于 8/4bit 量化）  
> 目标：给出 **DeepSpeed Inference** 与（可选）**bitsandbytes 量化**的最小教学脚手架，演示初始化、并行配置与吞吐/显存统计。

---

## 目录
- [DeepSpeed + bitsandbytes 量化部署示例：使用说明与函数文档](#deepspeed--bitsandbytes-量化部署示例使用说明与函数文档)
  - [目录](#目录)
  - [一、脚本概览](#一脚本概览)
  - [二、安装与环境准备](#二安装与环境准备)
  - [三、快速开始](#三快速开始)
  - [四、配置项：`DeployConfig`](#四配置项deployconfig)
  - [五、函数文档](#五函数文档)
    - [`init_engine(config)`](#init_engineconfig)
    - [`generate(engine, tokenizer, prompt, config)`](#generateengine-tokenizer-prompt-config)
  - [六、并行策略与常见配置](#六并行策略与常见配置)
  - [七、与 bitsandbytes 的集成要点（实战建议）](#七与-bitsandbytes-的集成要点实战建议)
  - [八、性能度量与显存统计](#八性能度量与显存统计)
  - [九、常见问题排查（FAQ）](#九常见问题排查faq)
  - [十、扩展建议](#十扩展建议)
  - [十一、许可证](#十一许可证)

---

## 一、脚本概览

该教学脚本演示 **DeepSpeed Inference** 推理引擎的初始化与调用流程，并给出 **流水线并行（PP）**、**张量并行（TP）** 的配置字段。示例中：
1. **初始化引擎**：`deepspeed.init_inference`（内含 kernel injection）。  
2. **生成回答**：封装 `generate(...)`，返回文本、**延迟**与**吞吐**（token/s）。  
3. **并行字段**：展示 `tp_size` / `pp_size` 的使用位置（示例默认 1）。

> 说明：示例中的 `init_engine` 使用了**伪代码风格**（`model_or_module=config.model_name`）。在**实战**中请参考[与 bitsandbytes 的集成要点](#七与-bitsandbytes-的集成要点实战建议)给出的更稳妥写法。

---

## 二、安装与环境准备

```bash
# 建议先安装与 CUDA 匹配的 PyTorch
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 基础依赖
pip install transformers deepspeed

# 可选：8/4bit 量化（需要支持的 NVIDIA GPU）
pip install bitsandbytes
```

> ⚠️ **CUDA/驱动匹配**：确保 `nvidia-smi`、`torch.cuda.is_available()` 正常；`deepspeed`/`bitsandbytes` 通常要求较新的 CUDA 与 GPU 架构（建议计算能力 ≥ 7.0）。

---

## 三、快速开始

保存脚本为 `ds_bnb_infer.py`：
```bash
python ds_bnb_infer.py
```
若环境未安装 `deepspeed`，脚本会以日志形式**跳过执行**。在多卡环境中，你可以：
```bash
deepspeed --num_gpus=2 ds_bnb_infer.py
```
并将 `DeployConfig(tp_size=2)` 以代码/环境变量方式生效（见下文）。

---

## 四、配置项：`DeployConfig`

```python
@dataclass
class DeployConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    tp_size: int = 1     # 张量并行（Model Parallel, mp_size）
    pp_size: int = 1     # 流水线并行（Pipeline Parallel）— 示例未直接用到
    max_new_tokens: int = 64
```
- **model_name**：HF Hub 模型或本地权重路径。  
- **tp_size**：张量并行度（DeepSpeed `mp_size`）。通常与 `--num_gpus` 对齐。  
- **pp_size**：流水线并行度（教学字段；若采用 PP，应使用 DeepSpeed 推理/训练的对应入口与切分策略）。  
- **max_new_tokens**：单次生成 token 上限，用于吞吐计算。

---

## 五、函数文档

### `init_engine(config)`
**作用**：初始化 DeepSpeed 推理引擎与分词器。

**伪代码（与示例一致）**
```python
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
engine = deepspeed.init_inference(
    model_or_module=config.model_name,
    mp_size=config.tp_size,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    max_tokens=config.max_new_tokens,
)
```
**返回**：`(engine, tokenizer)`。

**要点与注意**
- 真实工程中 **`model_or_module` 推荐传入“已加载的模型对象”**（见第七节实战建议）；  
- `replace_with_kernel_inject=True` 会尝试注入高性能 kernel；不同模型/版本兼容性略有差异；  
- `mp_size` 为 **张量并行**的世界大小，需与 `deepspeed --num_gpus` 或分布式环境一致。

---

### `generate(engine, tokenizer, prompt, config)`
**作用**：运行生成并统计时延/吞吐。

**流程**
1. `inputs = tokenizer(prompt, return_tensors="pt").to(engine.module.device)`  
2. 调用 `engine.generate(..., max_new_tokens=...)`  
3. 统计 `latency = end - start`；`throughput = max_new_tokens / latency`  
4. 返回 `{"text", "latency", "throughput"}`

**可替换/增强**
- 将 `max_new_tokens` 替换为**真实输出长度**参与吞吐计算：`throughput = generated_tokens / latency`；  
- 在多轮/批量请求下，统计 **tokens/s/gpu** 与 **tokens/s/cluster**。

---

## 六、并行策略与常见配置

- **张量并行（TP / mp_size）**：把单层内部矩阵按列/行切分到多卡并行计算。启动示例：
  ```bash
  deepspeed --num_gpus=4 ds_bnb_infer.py
  # 代码中将 DeployConfig.tp_size = 4
  ```

- **流水线并行（PP / pp_size）**：按层级把模型切分到不同 GPU，按 micro-batch 流水推进。推理侧通常**不常用**纯 PP，更多见于训练；如需 PP，应结合 DeepSpeed 的专门入口与图切分。

- **TP + PP 混合**：大模型/多节点场景下使用；需要**明确切分策略与拓扑**，超出本教学脚本范围。

---

## 七、与 bitsandbytes 的集成要点（实战建议）

**推荐实战写法**：先用 `transformers` + `bitsandbytes` **加载量化模型对象**，再传入 `deepspeed.init_inference`：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                # 或 load_in_4bit=True
    bnb_4bit_quant_type="nf4",        # 仅 4bit 时有效
    bnb_4bit_use_double_quant=True,   # 4bit 二次量化
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",                # 或 'cpu'，由 DeepSpeed 接管
    trust_remote_code=True,
)

engine = deepspeed.init_inference(
    model_or_module=base_model,       # ← 传入模型对象更稳妥
    mp_size=tp_size,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)
```
**提示**
- 某些情况下 `device_map="auto"` 与 DeepSpeed 接管会重复迁移设备；若冲突，先在 CPU 加载再交给 DeepSpeed。  
- 4bit 推理对 kernel 兼容性较敏感；若遇到不支持，可退回 8bit 或 FP16。

---

## 八、性能度量与显存统计

在推理前后添加简单的**显存/吞吐**统计：
```python
def gpu_mem_gb():
    if not torch.cuda.is_available():
        return {"free_gb": None, "total_gb": None}
    free, total = torch.cuda.mem_get_info()
    return {"free_gb": free / 1024**3, "total_gb": total / 1024**3}

before = gpu_mem_gb()
t0 = time.time()
outputs = engine.generate(**inputs, max_new_tokens=cfg.max_new_tokens)
t1 = time.time()
after = gpu_mem_gb()

print({
    "latency_s": t1 - t0,
    "throughput_tok_s": (outputs.shape[-1] - inputs["input_ids"].shape[-1]) / max(t1 - t0, 1e-6),
    "mem_before_gb": before, "mem_after_gb": after,
})
```
- **批量**请求时，可记录 **平均/95 分位延迟**；  
- 若使用多卡，建议统计 **每卡显存峰值** 与 **集群吞吐**。

---

## 九、常见问题排查（FAQ）

1. **`deepspeed.init_inference` 报错**  
   - 确保版本 ≥ 0.10；核注入（`replace_with_kernel_inject=True`）与模型实现可能不兼容，尝试关闭或升级。

2. **bitsandbytes 不可用**  
   - 检查 GPU 架构与 CUDA；在容器/服务器上确认 `LD_LIBRARY_PATH`，或退回 8bit/FP16。

3. **吞吐低/延迟高**  
   - 增大 `max_new_tokens` 或开启批处理（多 prompt 合并）；关闭采样使用贪心/beam；确认 `mp_size` 与 GPU 数对应。

4. **显存不如预期**  
   - 4bit/8bit 推理能显著降显存，但 kernel 兼容性/图优化会影响常驻；尝试 8bit 或 FP16 做对照。

5. **Llama-2 权限问题**  
   - HF 上游仓库可能需要**许可/访问 token**；确保已在环境中配置 `HF_TOKEN`。

---

## 十、扩展建议

- **KV Cache**：启用/验证 KV 缓存（生成长文本时提升吞吐）。  
- **TensorRT / vLLM**：与其他推理后端的吞吐/延迟/显存对比。  
- **多节点**：结合 DeepSpeed Launch/Accelerate 做多机推理拓展。  
- **服务化**：封装为 FastAPI/gRPC，集成队列与批处理调度。

---

## 十一、许可证

本文档与脚本用于教学演示；请遵循你项目的整体许可证与上游依赖（DeepSpeed、Transformers、bitsandbytes、PyTorch 等）的许可证要求。
