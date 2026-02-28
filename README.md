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

### Step 1：下载模型权重

```bash
# 下载 Gemma 3 1B 基座模型 (~2GB)
huggingface-cli download google/gemma-3-1b-pt \
    --local-dir model/gemma-3-1b-pt --token <YOUR_HF_TOKEN>

# 下载 Gemma Scope SAE 权重 (只下需要的层)
huggingface-cli download google/gemma-scope-2-1b-pt \
    --include "resid_post/layer_22/width_65k_l0_medium/*" \
    --local-dir sae/gemma-scope-2-1b-pt --token <YOUR_HF_TOKEN>
```

**得到：** `model/gemma-3-1b-pt/` 和 `sae/gemma-scope-2-1b-pt/` 目录下有模型权重文件

### Step 2：Smoke Test（验证代码能跑）

```bash
python scripts/train_sae.py --smoke-test
python scripts/eval_sae.py --smoke-test
```

**得到：** 使用随机数据跑几步训练和评估，确认环境无问题。会看到 loss 下降 + 一份评估报告

### Step 3：正式训练 SAE

```bash
python scripts/train_sae.py --config configs/default.yaml
```

**过程：**
1. 加载 Gemma 3 1B 模型 → 提取第 22 层的激活值
2. 释放 Gemma 显存 → 在激活值上训练 JumpReLU SAE（50000 步）
3. 终端实时打印 `loss / L0 / FVU`，每 5000 步自动保存 checkpoint

**得到：** `sae/checkpoints/checkpoint_step_5000.pt`, `..._10000.pt`, ..., `checkpoint_final.pt`

### Step 4：评估训练结果

```bash
python scripts/eval_sae.py --checkpoint sae/checkpoints/checkpoint_final.pt
```

**得到：**
- 终端打印评估报告（L0 稀疏度、FVU 重建质量、Top-10 活跃特征）
- 自动生成 `sae/checkpoints/report_<时间戳>.md` 和 `.json` 文件

### 调参

编辑 `configs/default.yaml` 修改超参数：

```yaml
model:
  hook_layer: 22        # 要 hook 的层 (0-25)
sae:
  d_sae: 16384          # SAE 宽度
training:
  num_steps: 50000      # 训练步数
  sparsity_coeff: 1e-3  # 稀疏性强度 (越大越稀疏)
  lr: 3e-4              # 学习率
```

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
