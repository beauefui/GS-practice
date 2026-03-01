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
| **权重来源** | Google 官方 HuggingFace | `fnlp/Llama-Scope` (OpenMOSS) |
| **特征数量** | 16k / 65k / 262k / 1m | **32k (8x)** / 128k (32x) |
| **框架** | 自定义代码即可 | 推荐使用 `lm-saes` 框架 |
| **命名规则** | `layer_22_width_65k_l0_medium` | `L22R-8x` (层号+位置+倍率) |

### Llama Scope 命名规则

`L[层号][位置]-[倍率]x`，例如：

| 名称 | 含义 |
|------|------|
| `L15R-8x` | 第 15 层，post-MLP **R**esidual stream，8x 扩展 (32k 特征) |
| `L15A-8x` | 第 15 层，**A**ttention output，8x 扩展 |
| `L15M-8x` | 第 15 层，**M**LP output，8x 扩展 |
| `L15R-32x` | 第 15 层，Residual，32x 扩展 (128k 特征，不推荐，死特征多) |

### 使用方式

Llama Scope 推荐使用官方的 `lm-saes` 框架，而不是我们的自定义代码：

```bash
# 安装 lm-saes 框架
pip install lm-saes==2.0.0b16
```

基本用法参考 [lm-saes examples](https://github.com/OpenMOSS/Language-Model-SAEs/tree/main/examples)。

### 如果要用我们的代码加载 Llama Scope 权重

**需要修改的代码：**

#### 1. `src/model.py` — 添加 TopK 激活函数

```python
# Llama Scope 使用 TopK 而非 JumpReLU
# TopK: 只保留前 k 个最大的激活值，其余置零
def topk_activation(pre_acts, k=64):
    topk_vals, topk_idx = pre_acts.topk(k, dim=-1)
    acts = torch.zeros_like(pre_acts)
    acts.scatter_(-1, topk_idx, topk_vals)
    return acts
```

#### 2. `src/utils.py` — 修改权重加载路径

```python
# Llama Scope 权重下载
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="fnlp/Llama-Scope",
    allow_patterns=["L15R-8x/*"],  # 按需选择层和位置
    local_dir="sae/llama-scope",
)
```

#### 3. `src/hooks.py` — 层访问路径不变

Llama 和 Gemma 模型结构类似，都是 `model.model.layers[i]`，**hooks 代码不需要改**。

### 对照参考

| 来源 | 链接 |
|------|------|
| **Llama Scope 论文** | [Llama Scope: Extracting Millions of Features from Llama-3.1-8B](https://arxiv.org/abs/2410.20526) |
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
