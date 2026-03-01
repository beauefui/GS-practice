# GS-Practice: 从 Gemma Scope 学习 SAE

一个动手学习项目，基于 Google DeepMind 的 [Gemma Scope](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) 项目，理解并实现 **稀疏自编码器 (Sparse Autoencoders, SAEs)**。

## 🎯 项目目标

本项目是一个个人学习练习，旨在：

1. **理解 JumpReLU SAE 架构** — 学习稀疏自编码器如何将神经网络的激活值分解为可解释的特征
2. **从零实现 SAE** — 参考 [Gemma Scope 2 Tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r)，逐步构建 SAE 核心组件
3. **在 Gemma 模型激活值上训练 SAE** — 实践完整流程：激活值提取 → SAE 训练 → 评估
4. **探索可解释性技术** — 特征可视化、模型引导 (Steering)、重建质量指标

## 📚 背景知识

### 什么是稀疏自编码器 (SAE)？

SAE 是一种无监督模型，用于将神经网络的内部激活值分解为一组**稀疏的、过完备的可解释特征**。核心思想：

- **编码器 (Encoder)**：将模型激活值（维度 `d_model`）映射到更高维的潜空间（维度 `d_sae`，其中 `d_sae >> d_model`）
- **稀疏性 (Sparsity)**：对于任意给定输入，只有少量潜在特征被激活
- **解码器 (Decoder)**：从稀疏的潜在表示重建原始激活值

### 什么是 JumpReLU？

Gemma Scope 使用 **JumpReLU** 激活函数替代标准 ReLU。JumpReLU 为每个特征引入一个可学习的**阈值** — 低于阈值的预激活值被置零。优势包括：

- 更好地控制稀疏性（直接优化 L0）
- 在相同稀疏度下获得更高的重建保真度
- 比 TopK 或 Gated SAE 方案提供更清晰的特征分离

## 🏗️ 项目结构

```
GS-practice/
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── .gitignore                 # Git 忽略规则
├── src/                       # 核心源代码
│   ├── __init__.py
│   ├── model.py               # JumpReLU SAE 模型定义 (encode/decode/forward)
│   ├── hooks.py               # 通过 forward hooks 提取模型激活值
│   ├── metrics.py             # 评估指标 (L0, FVU, MSE, Dead Features)
│   ├── train.py               # 损失函数 + 训练循环 + 激活值收集
│   └── utils.py               # 工具函数 (模型加载/权重下载/checkpoint)
├── scripts/                   # 入口脚本
│   ├── train_sae.py           # 训练入口 (CLI, 支持 --smoke-test)
│   └── eval_sae.py            # 评估入口 (指标报告 + Top 特征分析)
├── configs/                   # 训练配置文件
│   └── default.yaml           # 默认超参配置
├── sae/                       # 保存的 SAE 权重和 checkpoint
└── model/                     # 缓存的基座模型文件
```

## 🔬 学习路线

- [ ] **阶段 1：SAE 推理** — 加载预训练的 Gemma Scope SAE 权重，运行推理
- [ ] **阶段 2：激活值提取** — Hook 进 Gemma 模型各层，提取激活值
- [ ] **阶段 3：SAE 训练** — 实现 JumpReLU 损失函数的训练循环（重建 + 稀疏性）
- [ ] **阶段 4：评估** — 计算 L0、FVU 和 Delta Loss 指标
- [ ] **阶段 5：可解释性实验** — 可视化 top-activating 特征，实验 Steering

## 🔧 前置条件

- Python 3.10+
- CUDA 12.x（A800 GPU 训练）
- ~16GB+ GPU 显存（用于 Gemma 模型 + SAE）
- HuggingFace Token（下载 Gemma 模型需要）

## 🚀 完整使用流程

### Step 0：环境搭建

```bash
git clone https://github.com/beauefui/GS-practice.git
cd GS-practice
conda create -n gs python=3.10 -y
conda activate gs
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 1：下载权重

```bash
# 将 <YOUR_HF_TOKEN> 替换为你的 token
# 下载 Gemma 模型 + Google 预训练 SAE 权重
python scripts/download_weights.py --token <YOUR_HF_TOKEN>
```

**得到：** `model/gemma-3-1b-pt/` (Gemma 基座模型) 和 `sae/gemma-scope-2-1b-pt/` (Google 预训练 SAE 权重)

### Step 2：Smoke Test（验证环境）

```bash
python scripts/train_sae.py --smoke-test
python scripts/eval_sae.py --smoke-test
```

**得到：** 使用随机数据跑几步训练和评估，确认环境无问题

---

### 🅰️ 主路线：使用 Google 预训练 SAE 评估（推荐）

> 这是与 [Colab 教程](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r) 对齐的用法。
> 直接加载 Google 花大量算力训练好的 SAE 权重，对 Gemma 模型进行分析。

```bash
# 直接评估 Google 预训练 SAE
CUDA_VISIBLE_DEVICES=0 python scripts/eval_sae.py --pretrained
```

**过程：**
1. 加载 Gemma 模型 + Google 预训练 SAE 权重 (`sae/gemma-scope-2-1b-pt/`)
2. 提取激活值 → 通过 SAE 编码/解码 → 计算评估指标

**得到：**
- 终端打印评估报告（L0 稀疏度、FVU 重建质量、Top-10 活跃特征）
- 自动生成 `reports/report_<时间戳>.md` 和 `.json` 报告文件
- 预期效果：**L0 ≈ 70, FVU ≈ 2-3%**（与 Colab 教程一致）

---

### 🅱️ 可选路线：从零训练 SAE（学习用）

> 这条路线是为了**理解 SAE 训练过程**，效果远不如 Google 预训练版本，
> 但对学习 SAE 的工作原理非常有帮助。

**训练：**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sae.py --config configs/default.yaml
```

**评估自训练的 checkpoint：**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_sae.py --checkpoint sae/checkpoints/checkpoint_final.pt
```

**调参（编辑 `configs/default.yaml`）：**

```yaml
model:
  hook_layer: 22        # 要 hook 的层 (0-25)
sae:
  d_sae: 16384          # SAE 宽度
training:
  num_steps: 20000      # 训练步数
  sparsity_coeff: 0.01  # 稀疏性强度 (越大越稀疏)
  lr: 1e-4              # 学习率
```

## 🔄 切换不同的 Gemma 模型和 SAE

### 可用的模型和对应的 Gemma Scope

每个 Gemma 基座模型都有对应的 Gemma Scope SAE 权重（`-pt` = 预训练版，`-it` = 指令微调版）：

| Gemma 基座模型 | 对应 Gemma Scope | 层数 | d_model | 显存需求 |
|---------------|-----------------|------|---------|---------|
| `google/gemma-3-270m-pt` | `google/gemma-scope-2-270m-pt` | 18 | 1536 | ~2 GB |
| `google/gemma-3-1b-pt` ← **当前** | `google/gemma-scope-2-1b-pt` | 26 | 1152 | ~4 GB |
| `google/gemma-3-4b-pt` | `google/gemma-scope-2-4b-pt` | 34 | 2560 | ~10 GB |
| `google/gemma-3-12b-pt` | `google/gemma-scope-2-12b-pt` | 48 | 3840 | ~28 GB |
| `google/gemma-3-27b-pt` | `google/gemma-scope-2-27b-pt` | 62 | 4608 | ~60 GB |

> 把 `-pt` 换成 `-it` 即可使用指令微调版本（如 `gemma-3-4b-it` + `gemma-scope-2-4b-it`）

### 切换步骤

**以切换到 4B 模型为例：**

#### 1. 修改下载脚本 `scripts/download_weights.py`

```python
# 改 repo_id 和 local_dir
snapshot_download(
    repo_id="google/gemma-3-4b-pt",          # ← 改这里
    local_dir="model/gemma-3-4b-pt",          # ← 改这里
    token=args.token,
)

snapshot_download(
    repo_id="google/gemma-scope-2-4b-pt",     # ← 改这里
    local_dir="sae/gemma-scope-2-4b-pt",      # ← 改这里
    allow_patterns=["resid_post/layer_20_width_65k_l0_medium/*"],  # ← 改层号
    token=args.token,
)
```

#### 2. 修改配置文件 `configs/default.yaml`

```yaml
model:
  name: "model/gemma-3-4b-pt"     # ← 改模型路径
  hook_layer: 20                   # ← 改层号 (通常选中间偏后的层)

pretrained_sae:
  repo_id: "google/gemma-scope-2-4b-pt"    # ← 改 scope 仓库
  local_dir: "sae/gemma-scope-2-4b-pt"     # ← 改本地路径
  layer: 20                                 # ← 和 hook_layer 一致
  width: "65k"
  l0: "medium"
```

#### 3. 重新下载并评估

```bash
python scripts/download_weights.py --token <YOUR_HF_TOKEN>
CUDA_VISIBLE_DEVICES=0 python scripts/eval_sae.py --pretrained
```

### SAE 变体选择

每个层下有不同宽度和稀疏度的 SAE 可选，在 `allow_patterns` 和 `configs/default.yaml` 中修改：

| 参数 | 可选值 | 说明 |
|------|-------|------|
| `width` | `16k`, `65k`, `262k`, `1m` | 特征数量，越大越细粒度 |
| `l0` | `small`, `medium`, `big` | 目标稀疏度，small=更稀疏 |

> 例如 `layer_15_width_262k_l0_small` 表示第 15 层、262k 特征、高稀疏度

## 🦙 拓展：从 Gemma Scope 到 Llama Scope

[Llama Scope](https://github.com/OpenMOSS/Language-Model-SAEs) 是 OpenMOSS 团队为 **Llama-3.1-8B** 训练的 SAE 套件，提供了所有层和子层的 256 个 TopK SAE。

### Gemma Scope vs Llama Scope 核心区别

| | Gemma Scope | Llama Scope |
|---|---|---|
| **基座模型** | Gemma 3 (270M ~ 27B) | Llama 3.1 8B |
| **SAE 架构** | JumpReLU (可学习阈值) | **TopK** (固定选 top-k 个特征) |
| **权重来源** | `google/gemma-scope-2-*` | `fnlp/Llama-Scope` |
| **特征数量** | 16k / 65k / 262k / 1m | **32k (8x)** / 128k (32x) |
| **命名规则** | `layer_22_width_65k_l0_medium` | `L22R-8x` (层号+位置+倍率) |

#### Llama Scope 命名规则

`L[层号][位置]-[倍率]x`：

| 位置代码 | 含义 | 对应 Gemma Scope |
|---------|------|-----------------|
| `R` | post-MLP **R**esidual stream | `resid_post` |
| `A` | **A**ttention output | `attn_output` |
| `M` | **M**LP output | `mlp_output` |

> 例如 `L15R-8x` = 第 15 层的 Residual stream，8x 扩展 (32k 特征)
> ⚠️ `32x` 的 SAE (128k 特征) 死特征较多，**推荐用 `8x` (32k 特征)**

---

### 方法 A：使用 `lm-saes` 框架（推荐）

> 这是 Llama Scope 官方推荐的方式，适合深入研究。

#### Step 1：创建新环境并安装

```bash
# 建议新建一个 conda 环境，避免与 Gemma Scope 依赖冲突
conda create -n llama-scope python=3.10 -y
conda activate llama-scope

# 安装 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装 lm-saes 框架
pip install lm-saes==2.0.0b16
```

#### Step 2：下载 Llama 模型和 SAE 权重

```python
# 新建一个脚本: scripts/download_llama_scope.py
from huggingface_hub import snapshot_download

# 1. 下载 Llama 3.1 8B 基座模型 (约 16GB)
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B",
    local_dir="model/Llama-3.1-8B",
    token="<YOUR_HF_TOKEN>",
)

# 2. 下载 Llama Scope SAE 权重 (只下需要的一个)
snapshot_download(
    repo_id="fnlp/Llama-Scope",
    allow_patterns=["L15R-8x/*"],   # 第15层 Residual 8x, 按需修改
    local_dir="sae/llama-scope",
    token="<YOUR_HF_TOKEN>",
)
```

```bash
python scripts/download_llama_scope.py
```

#### Step 3：使用 lm-saes 加载和评估

```python
# 新建一个脚本: scripts/eval_llama_scope.py
from lm_saes import SparseAutoEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载 Llama 模型
model = AutoModelForCausalLM.from_pretrained(
    "model/Llama-3.1-8B",
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("model/Llama-3.1-8B")

# 加载 Llama Scope SAE
sae = SparseAutoEncoder.from_pretrained("sae/llama-scope/L15R-8x")
sae = sae.to("cuda")

print(f"SAE 加载完成: {sae}")
print(f"  d_model: {sae.d_model}")
print(f"  d_sae:   {sae.d_sae}")

# 提取激活值并通过 SAE 编码
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # 第 15 层的 hidden states (0-indexed, +1 因为包含 embedding 层)
    activations = outputs.hidden_states[16].float()  # (1, seq_len, d_model)

    # 通过 SAE 编码/解码
    acts = activations.reshape(-1, activations.shape[-1])  # (seq_len, d_model)
    encoded = sae.encode(acts)
    decoded = sae.decode(encoded)

    # 计算指标
    l0 = (encoded > 0).float().sum(dim=-1).mean().item()
    fvu = ((acts - decoded).pow(2).sum() / acts.pow(2).sum()).item()

print(f"\n评估结果:")
print(f"  L0 (稀疏度): {l0:.1f}")
print(f"  FVU (重建误差): {fvu:.4f}")
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_llama_scope.py
```

---

### 方法 B：改造我们的代码（学习用）

> 如果你想用本项目的代码框架来加载 Llama Scope，需要做以下修改。

#### Step 1. `src/model.py` — 新增 TopKSAE 类

在 `JumpReLUSAE` 类下面添加一个新的 SAE 类：

```python
class TopKSAE(nn.Module):
    """TopK SAE (Llama Scope 使用的架构)

    与 JumpReLU 的区别:
      - JumpReLU: 每个特征有可学习阈值, 低于阈值的置零
      - TopK: 固定选前 k 个最大的特征, 其余置零 (k 是超参数, 不可学)
    """
    def __init__(self, d_model: int, d_sae: int, k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        # TopK: 只保留前 k 个最大值
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts
```

#### Step 2. `scripts/download_weights.py` — 添加 Llama 下载

在 `main()` 中添加 Llama 的下载逻辑（或新建脚本）：

```python
# 下载 Llama 3.1 8B
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B",
    local_dir="model/Llama-3.1-8B",
    token=args.token,
)

# 下载 Llama Scope SAE
snapshot_download(
    repo_id="fnlp/Llama-Scope",
    allow_patterns=["L15R-8x/*"],
    local_dir="sae/llama-scope",
    token=args.token,
)
```

#### Step 3. `src/utils.py` — 添加 Llama Scope 权重加载函数

```python
def load_llama_scope_weights(
    local_dir: str = "sae/llama-scope",
    sae_name: str = "L15R-8x",
) -> dict:
    """加载 Llama Scope SAE 权重"""
    from safetensors.torch import load_file
    path = Path(local_dir) / sae_name / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"权重文件不存在: {path}")
    params = load_file(str(path))
    print(f"[Llama Scope] 加载完成: {sae_name}")
    for k, v in params.items():
        print(f"  {k}: {v.shape}")
    return params
```

#### Step 4. `src/hooks.py` — 不需要改

Llama 和 Gemma 结构相同，都是 `model.model.layers[i]`，hooks 代码**完全通用**。

#### Step 5. `configs/default.yaml` — 改配置

```yaml
model:
  name: "model/Llama-3.1-8B"
  hook_layer: 15
  dtype: "bfloat16"

pretrained_sae:
  local_dir: "sae/llama-scope"
  sae_name: "L15R-8x"
```

---

### 对照参考

| 来源 | 链接 |
|------|------|
| **Llama Scope 论文** | [arxiv.org/abs/2410.20526](https://arxiv.org/abs/2410.20526) |
| **训练框架** | [github.com/OpenMOSS/Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) |
| **预训练权重** | [huggingface.co/fnlp/Llama-Scope](https://huggingface.co/fnlp/Llama-Scope) |


## 📦 主要依赖

| 包名 | 用途 |
|------|------|
| `torch` | 深度学习框架 |
| `transformers` | 从 HuggingFace 加载 Gemma 模型 |
| `huggingface_hub` | 下载 SAE 权重 |
| `safetensors` | 高效的权重序列化格式 |
| `einops` | 张量运算 |
| `pyyaml` | 配置文件解析 |
| `datasets` | 加载训练数据集 |
| `wandb` | 实验追踪（可选） |

## 📖 参考资料

- **Gemma Scope 博客**: [deepmind.google/gemma-scope](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/)
- **Gemma Scope 2 Tutorial (Colab)**: [colab.research.google.com](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r)
- **JumpReLU 论文**: [Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders](https://arxiv.org/abs/2407.14435)
- **Gemma Scope 权重 (HuggingFace)**: [google/gemma-scope-2b-pt-res](https://huggingface.co/google/gemma-scope-2b-pt-res)
- **SAELens (训练库)**: [github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)

## ⚖️ 许可

本项目仅用于个人学习和研究目的。
