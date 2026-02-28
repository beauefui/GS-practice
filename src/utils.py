"""
工具函数

提供模型加载、权重管理、设备选择等通用功能。
将这些功能从训练/评估逻辑中分离出来, 保持各模块职责单一。
"""

import os
import json
import random
import torch
import numpy as np
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """
    设置全局随机种子, 确保实验可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    自动选择最佳计算设备

    优先级: CUDA GPU > CPU

    Returns:
        torch.device 对象
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] 显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[Device] 使用 CPU (无可用 GPU)")
    return device


def load_gemma_model(
    model_name: str = "google/gemma-3-1b-pt",
    device: torch.device | str = "auto",
    dtype: torch.dtype = torch.float32,
):
    """
    加载 Gemma 基座模型和 tokenizer

    注意:
      - 使用 base model (pt), 不用 instruction-tuned (it)
      - SAE 是在 base model 的激活值上训练的
      - 需要先登录 HuggingFace: huggingface-cli login

    Args:
        model_name: HuggingFace 模型名称
        device: 设备 ("auto" 让 transformers 自动分配)
        dtype: 精度 (float32 / bfloat16)

    Returns:
        (model, tokenizer) 元组
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Model] 加载模型: {model_name}")
    print(f"[Model] 精度: {dtype}")

    # 加载模型, 关闭梯度 (我们只需要提取激活值, 不训练基座模型)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    print(f"[Model] 加载完成: {num_layers} layers, d_model={d_model}")

    return model, tokenizer


def load_sae_weights(
    repo_id: str = "google/gemma-scope-2-1b-pt",
    layer: int = 22,
    width: str = "65k",
    l0: str = "medium",
    hook_point: str = "resid_post",
) -> dict[str, torch.Tensor]:
    """
    从 HuggingFace 下载 Gemma Scope 预训练 SAE 权重

    Gemma Scope 的文件组织结构:
      google/gemma-scope-2-1b-pt/
      ├── resid_post/          # residual stream SAEs
      │   └── layer_{L}/
      │       └── width_{W}_l0_{L0}/
      │           └── params.safetensors
      └── resid_post_all/      # all-layer SAEs
          └── ...

    Args:
        repo_id: HuggingFace 仓库 ID
        layer: 层索引
        width: SAE 宽度 ("16k" / "65k" / "262k" / "1m")
        l0: 目标稀疏度 ("small" / "medium" / "big")
        hook_point: hook 点位 ("resid_post" / "resid_post_all")

    Returns:
        权重字典 (w_enc, b_enc, w_dec, b_dec, threshold)
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    filename = f"{hook_point}/layer_{layer}/width_{width}_l0_{l0}/params.safetensors"
    print(f"[SAE Weights] 下载: {repo_id}/{filename}")

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    params = load_file(path)

    print(f"[SAE Weights] 加载完成:")
    for k, v in params.items():
        print(f"  {k}: {v.shape}")

    return params


def save_checkpoint(
    sae,
    optimizer,
    step: int,
    metrics: dict,
    save_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """
    保存训练 checkpoint

    保存内容: SAE 权重 + 优化器状态 + 训练进度 + 指标

    Args:
        sae: JumpReLUSAE 模型
        optimizer: 优化器
        step: 当前训练步数
        metrics: 当前指标字典
        save_dir: 保存目录
        filename: 文件名 (默认: checkpoint_step_{step}.pt)

    Returns:
        保存的文件路径
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_step_{step}.pt"

    save_path = save_dir / filename

    checkpoint = {
        "step": step,
        "sae_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "sae_config": {
            "d_model": sae.d_model,
            "d_sae": sae.d_sae,
        },
    }

    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] 已保存: {save_path}")
    return save_path


def load_checkpoint(
    path: str | Path,
    sae=None,
    optimizer=None,
    device: torch.device | str = "cpu",
) -> dict:
    """
    加载训练 checkpoint

    Args:
        path: checkpoint 文件路径
        sae: 可选, 传入则自动加载权重
        optimizer: 可选, 传入则自动加载优化器状态
        device: 加载到哪个设备

    Returns:
        checkpoint 字典
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    print(f"[Checkpoint] 加载: {path}, step={checkpoint['step']}")

    if sae is not None:
        sae.load_state_dict(checkpoint["sae_state_dict"])
        print(f"[Checkpoint] SAE 权重已加载")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[Checkpoint] 优化器状态已加载")

    return checkpoint
