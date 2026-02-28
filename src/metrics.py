"""
SAE 评估指标

评估一个 SAE 好不好, 主要看两个维度的 trade-off:
  1. 重建质量: SAE 重建的激活值与原始激活值有多接近?
  2. 稀疏性: SAE 的潜在表示有多稀疏?

理想的 SAE: 用尽量少的特征 (高稀疏性) 尽量准确地重建原始激活值 (高重建质量)。

核心指标:
  - L0:  平均有多少个特征被激活 (越小越稀疏)
  - FVU: 未解释方差比例 (越小重建越好, 0 = 完美重建)
  - MSE: 均方误差 (与 FVU 相关, 但没有归一化)
  - Dead Features: 从未激活的特征占比 (越少越好, 死特征浪费容量)
"""

import torch


def compute_l0(latent_acts: torch.Tensor) -> float:
    """
    计算 L0 稀疏度 — 平均每个样本有多少个特征被激活

    L0 = mean(count_nonzero(acts))

    例如: d_sae=16384, L0=50 表示平均只有 50/16384 ≈ 0.3% 的特征是活跃的

    Args:
        latent_acts: SAE 编码后的稀疏表示, shape (..., d_sae)

    Returns:
        L0 值 (float), 即平均非零特征数
    """
    # 沿最后一维 (d_sae) 统计非零元素个数
    # 先展平前面所有维度, 变成 (num_samples, d_sae)
    flat = latent_acts.reshape(-1, latent_acts.shape[-1])
    # 每个样本的非零特征数
    l0_per_sample = (flat != 0).float().sum(dim=-1)  # (num_samples,)
    return l0_per_sample.mean().item()


def compute_fvu(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    计算 FVU (Fraction of Variance Unexplained) — 未解释方差比例

    FVU = Var(original - reconstructed) / Var(original)

    FVU 的含义:
      - FVU = 0:   完美重建
      - FVU = 1:   重建效果和直接用均值预测一样差
      - FVU > 1:   重建比均值预测还差 (说明 SAE 出了问题)
      - FVU = 0.05: 解释了 95% 的方差 (通常认为不错)

    与 MSE 的区别: FVU 被原始数据的方差归一化了, 所以跨层/跨模型可比

    Args:
        original:      原始激活值, shape (..., d_model)
        reconstructed: 重建的激活值, shape (..., d_model)

    Returns:
        FVU 值 (float)
    """
    # 展平为 (num_samples, d_model)
    orig_flat = original.reshape(-1, original.shape[-1]).float()
    recon_flat = reconstructed.reshape(-1, reconstructed.shape[-1]).float()

    # 重建误差的方差
    residual = orig_flat - recon_flat
    unexplained_var = residual.var(dim=0).sum()  # 沿样本维度计算方差, 再求和

    # 原始数据的方差
    total_var = orig_flat.var(dim=0).sum()

    # 避免除零
    if total_var == 0:
        return 0.0

    return (unexplained_var / total_var).item()


def compute_mse(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    计算 MSE (Mean Squared Error) — 均方误差

    MSE = mean((original - reconstructed)^2)

    Args:
        original:      原始激活值, shape (..., d_model)
        reconstructed: 重建的激活值, shape (..., d_model)

    Returns:
        MSE 值 (float)
    """
    return (original - reconstructed).pow(2).mean().item()


def compute_dead_features(
    latent_acts: torch.Tensor,
    threshold: float = 0.0,
) -> dict[str, float]:
    """
    统计 "死特征" — 在整个 batch 中从未激活的特征

    死特征意味着 SAE 的部分容量被浪费了。训练中如果死特征过多,
    可能需要调整学习率、threshold 初始化, 或使用 resampling 策略。

    Args:
        latent_acts: SAE 编码后的稀疏表示, shape (..., d_sae)
        threshold: 判断 "激活" 的最小值 (默认 0, 即 >0 就算激活)

    Returns:
        dict 包含:
          - "dead_ratio": 死特征比例 (0~1)
          - "dead_count": 死特征数量
          - "total_features": 总特征数
    """
    flat = latent_acts.reshape(-1, latent_acts.shape[-1])
    # 每个特征在所有样本中是否至少激活过一次
    ever_activated = (flat > threshold).any(dim=0)  # (d_sae,), bool
    total = ever_activated.numel()
    alive = ever_activated.sum().item()
    dead = total - alive

    return {
        "dead_ratio": dead / total,
        "dead_count": int(dead),
        "total_features": total,
    }


def compute_all_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    latent_acts: torch.Tensor,
) -> dict[str, float]:
    """
    一次性计算所有核心指标

    用法:
        recon, acts = sae(activations)
        metrics = compute_all_metrics(activations, recon, acts)
        # metrics = {"l0": 47.2, "fvu": 0.03, "mse": 0.0012, "dead_ratio": 0.15, ...}

    Args:
        original:      原始激活值
        reconstructed: SAE 重建的激活值
        latent_acts:   SAE 的稀疏潜在表示

    Returns:
        包含所有指标的字典
    """
    dead_info = compute_dead_features(latent_acts)
    return {
        "l0": compute_l0(latent_acts),
        "fvu": compute_fvu(original, reconstructed),
        "mse": compute_mse(original, reconstructed),
        "dead_ratio": dead_info["dead_ratio"],
        "dead_count": dead_info["dead_count"],
        "total_features": dead_info["total_features"],
    }
