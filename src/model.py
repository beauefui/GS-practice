"""
JumpReLU Sparse Autoencoder (SAE) 模型定义

参考: Gemma Scope 2 Tutorial
论文: Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs (arXiv:2407.14435)

核心思想:
  SAE 将模型的激活值 x (维度 d_model) 编码到一个更高维的稀疏潜空间 (维度 d_sae),
  然后从稀疏表示重建原始激活值。JumpReLU 通过可学习的阈值来控制稀疏性。

架构:
  Encoder:  f(x) = JumpReLU(W_enc @ x + b_enc)
  Decoder:  x̂ = W_dec @ f(x) + b_dec

  其中 JumpReLU(z) = z * (z > threshold), threshold 是可学习的参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JumpReLUSAE(nn.Module):
    """
    JumpReLU 稀疏自编码器

    与标准 ReLU SAE 的区别:
    - 标准 ReLU: f(z) = max(0, z)
    - JumpReLU: f(z) = z * (z > threshold)

    JumpReLU 引入可学习的 per-feature 阈值, 低于阈值的激活被完全置零。
    这让模型可以更精确地控制稀疏性 (L0), 而不是依赖 L1 正则化间接实现。

    参数:
        d_model (int): 输入激活值的维度 (即基座模型的隐藏层维度)
        d_sae (int): SAE 潜空间维度 (通常是 d_model 的 8-16 倍)
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()

        # ---- 可学习参数 ----
        # 编码器权重: 将 d_model 维的激活值映射到 d_sae 维的潜空间
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        # 编码器偏置
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # 解码器权重: 将 d_sae 维的稀疏表示映射回 d_model 维
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        # 解码器偏置
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # JumpReLU 的可学习阈值: 每个特征有独立的阈值
        self.threshold = nn.Parameter(torch.zeros(d_sae))

        # 保存维度信息
        self.d_model = d_model
        self.d_sae = d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码: 将模型激活值映射到稀疏潜空间

        计算过程:
          1. 线性变换: pre_acts = x @ W_enc + b_enc
          2. JumpReLU:  acts = ReLU(pre_acts) * (pre_acts > threshold)

        Args:
            x: 输入激活值, shape (..., d_model)
               例如 (batch_size, seq_len, d_model)

        Returns:
            acts: 稀疏潜在表示, shape (..., d_sae)
                  大部分元素为 0 (稀疏性由 threshold 控制)
        """
        # 线性投影到潜空间
        pre_acts = x @ self.W_enc + self.b_enc  # (..., d_sae)

        # JumpReLU 激活:
        #   1. mask: 标记哪些特征的预激活值超过了阈值
        #   2. 只保留超过阈值的特征, 其余置零
        mask = (pre_acts > self.threshold)  # (..., d_sae), bool
        acts = mask * F.relu(pre_acts)      # (..., d_sae)

        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        解码: 从稀疏潜在表示重建原始激活值

        计算: x̂ = acts @ W_dec + b_dec

        Args:
            acts: 稀疏潜在表示, shape (..., d_sae)

        Returns:
            recon: 重建的激活值, shape (..., d_model)
        """
        return acts @ self.W_dec + self.b_dec  # (..., d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        完整前向传播: encode → decode

        Args:
            x: 输入激活值, shape (..., d_model)

        Returns:
            recon: 重建的激活值, shape (..., d_model)
            acts:  稀疏潜在表示, shape (..., d_sae)
                   (返回 acts 是为了在训练时计算稀疏性损失)
        """
        acts = self.encode(x)      # (..., d_sae)
        recon = self.decode(acts)   # (..., d_model)
        return recon, acts

    @classmethod
    def from_pretrained(cls, params: dict[str, torch.Tensor]) -> "JumpReLUSAE":
        """
        从预训练权重字典构建 SAE 实例

        Gemma Scope 的权重文件 (safetensors) 加载后是一个 dict, 键名为:
          - "w_enc"    : shape (d_model, d_sae)
          - "b_enc"    : shape (d_sae,)
          - "w_dec"    : shape (d_sae, d_model)
          - "b_dec"    : shape (d_model,)
          - "threshold": shape (d_sae,)

        用法:
            from safetensors.torch import load_file
            params = load_file("path/to/params.safetensors")
            sae = JumpReLUSAE.from_pretrained(params)

        Args:
            params: 权重字典, 由 safetensors.torch.load_file() 返回

        Returns:
            加载了预训练权重的 JumpReLUSAE 实例
        """
        # 从权重形状推断维度
        d_model, d_sae = params["w_enc"].shape

        # 创建模型实例
        sae = cls(d_model=d_model, d_sae=d_sae)

        # 加载权重 (注意: Gemma Scope 用小写 w_enc, 我们的参数名用 W_enc)
        sae.W_enc.data = params["w_enc"]
        sae.b_enc.data = params["b_enc"]
        sae.W_dec.data = params["w_dec"]
        sae.b_dec.data = params["b_dec"]
        sae.threshold.data = params["threshold"]

        return sae

    def __repr__(self) -> str:
        return (
            f"JumpReLUSAE(\n"
            f"  d_model={self.d_model},\n"
            f"  d_sae={self.d_sae},\n"
            f"  expansion_factor={self.d_sae / self.d_model:.1f}x,\n"
            f"  total_params={sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )
