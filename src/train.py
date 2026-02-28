"""
SAE 训练循环

训练流程:
  1. 从基座模型 (Gemma) 提取指定层的激活值
  2. 在激活值上训练 SAE (不训练基座模型)
  3. 损失函数 = 重建损失 + 稀疏性惩罚

损失函数设计:
  L = MSE(x, x̂) + λ * L_sparsity

  - MSE(x, x̂): 重建误差, 让 SAE 学会准确重建
  - L_sparsity: 稀疏性约束, 鼓励 SAE 只用少量特征

  对于 JumpReLU SAE, 稀疏性主要由 threshold 参数控制,
  但训练时仍可加入辅助稀疏性损失来引导学习。

关于 Straight-Through Estimator (STE):
  JumpReLU 的阈值操作 (z > threshold) 产生阶跃函数, 梯度为 0。
  STE 技巧: 前向传播用真实阶跃函数, 反向传播假装梯度为 1,
  让梯度能 "穿过" 不可微的操作。在我们的简化实现中,
  PyTorch 的 autograd 会自动处理 mask * relu(z) 的梯度。
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from src.model import JumpReLUSAE
from src.metrics import compute_l0, compute_fvu, compute_mse, compute_dead_features


@dataclass
class TrainConfig:
    """
    训练超参数配置

    Attributes:
        lr: 学习率
        num_steps: 总训练步数
        batch_size: 每步用多少个 token 的激活值
        sparsity_coeff: 稀疏性惩罚系数 λ
        log_every: 每隔多少步打印日志
        checkpoint_every: 每隔多少步保存 checkpoint
        checkpoint_dir: checkpoint 保存目录
    """
    lr: float = 3e-4
    num_steps: int = 50000
    batch_size: int = 4096
    sparsity_coeff: float = 1e-3
    log_every: int = 100
    checkpoint_every: int = 5000
    checkpoint_dir: str = "sae"


def sae_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    latent_acts: torch.Tensor,
    sparsity_coeff: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算 SAE 训练损失

    Loss = MSE(x, x̂) + λ * L1(acts)

    其中:
      - MSE(x, x̂) = mean((x - x̂)²)  → 重建质量
      - L1(acts) = mean(|acts|)         → 稀疏性惩罚 (鼓励激活值小且少)

    为什么用 L1 做稀疏惩罚?
      L0 (非零个数) 不可微, 无法直接优化。
      L1 是 L0 的凸松弛, 梯度存在且鼓励稀疏。
      JumpReLU 的 threshold 提供了额外的硬稀疏性。

    Args:
        original:       原始激活值 (batch, d_model)
        reconstructed:  重建激活值 (batch, d_model)
        latent_acts:    稀疏潜在表示 (batch, d_sae)
        sparsity_coeff: 稀疏性惩罚系数 λ

    Returns:
        (loss, loss_dict):
          - loss: 总损失标量 (用于反向传播)
          - loss_dict: 各项损失的数值 (用于日志)
    """
    # 重建损失
    recon_loss = (original - reconstructed).pow(2).mean()

    # 稀疏性损失 (L1 范数)
    sparsity_loss = latent_acts.abs().mean()

    # 总损失
    total_loss = recon_loss + sparsity_coeff * sparsity_loss

    return total_loss, {
        "total_loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
    }


def make_activation_dataloader(
    activations: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    将提取好的激活值张量包装成 DataLoader

    训练 SAE 的流程是:
      Phase 1: 跑一遍基座模型, 提取大量激活值, 存为张量
      Phase 2: 在这些激活值上训练 SAE (多次遍历)

    这样做的好处: 避免每个 training step 都要跑一遍基座模型

    Args:
        activations: 激活值张量, shape (num_tokens, d_model)
        batch_size: 每个 batch 的 token 数
        shuffle: 是否打乱顺序

    Returns:
        DataLoader, 每个 batch 的 shape: (batch_size, d_model)
    """
    dataset = torch.utils.data.TensorDataset(activations)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,  # 丢弃最后不完整的 batch
    )


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    hook_point: str = "residual",
    max_seq_len: int = 256,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    从基座模型中提取激活值

    对每段文本:
      1. tokenize
      2. 前向传播 (无梯度)
      3. 通过 hook 捕获指定层的激活值
      4. 收集所有 token 的激活值

    Args:
        model: Gemma 基座模型
        tokenizer: tokenizer
        texts: 输入文本列表
        layer_idx: 要提取的层索引
        hook_point: hook 点位
        max_seq_len: 最大序列长度
        device: 设备

    Returns:
        activations: shape (total_tokens, d_model)
    """
    from src.hooks import ActivationCache

    model.eval()
    all_acts = []

    for i, text in enumerate(texts):
        # tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
        ).to(device)

        # 通过 hook 捕获激活值
        with ActivationCache(model, layer_idx=layer_idx, hook_point=hook_point) as cache:
            model(**inputs)
            acts = cache.get()  # (1, seq_len, d_model)

        # 展平 batch 和 seq 维度 → (seq_len, d_model)
        acts = acts.reshape(-1, acts.shape[-1])
        all_acts.append(acts.cpu())

        if (i + 1) % 100 == 0:
            print(f"[Collect] 已处理 {i + 1}/{len(texts)} 段文本")

    # 合并所有激活值
    all_acts = torch.cat(all_acts, dim=0)
    print(f"[Collect] 收集完成: {all_acts.shape[0]} tokens, d_model={all_acts.shape[1]}")
    return all_acts


def train_sae(
    sae: JumpReLUSAE,
    activations: torch.Tensor,
    config: TrainConfig,
    device: torch.device | str = "cpu",
) -> dict[str, list]:
    """
    训练 SAE 的主循环

    流程:
      1. 将激活值包装成 DataLoader
      2. 循环训练: 取 batch → 前向传播 → 计算损失 → 反向传播 → 更新参数
      3. 定期记录指标和保存 checkpoint

    Args:
        sae: JumpReLUSAE 模型
        activations: 预先提取的激活值, shape (num_tokens, d_model)
        config: 训练配置
        device: 训练设备

    Returns:
        history: 训练历史 {"step": [...], "total_loss": [...], "l0": [...], ...}
    """
    sae = sae.to(device)
    sae.train()

    # 优化器: Adam, SAE 训练的标准选择
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)

    # 数据加载
    dataloader = make_activation_dataloader(activations, batch_size=config.batch_size)

    # 训练历史记录
    history = {
        "step": [],
        "total_loss": [],
        "recon_loss": [],
        "sparsity_loss": [],
        "l0": [],
        "fvu": [],
    }

    step = 0
    print(f"[Train] 开始训练: {config.num_steps} steps, batch_size={config.batch_size}")
    print(f"[Train] SAE: d_model={sae.d_model}, d_sae={sae.d_sae}")
    print(f"[Train] 激活值: {activations.shape[0]} tokens")
    print(f"[Train] 稀疏性系数 λ = {config.sparsity_coeff}")
    print("-" * 60)

    while step < config.num_steps:
        for (batch,) in dataloader:
            if step >= config.num_steps:
                break

            batch = batch.to(device)  # (batch_size, d_model)

            # ---- 前向传播 ----
            recon, acts = sae(batch)

            # ---- 计算损失 ----
            loss, loss_dict = sae_loss(
                original=batch,
                reconstructed=recon,
                latent_acts=acts,
                sparsity_coeff=config.sparsity_coeff,
            )

            # ---- 反向传播 ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- 日志 ----
            if step % config.log_every == 0:
                with torch.no_grad():
                    l0 = compute_l0(acts)
                    fvu = compute_fvu(batch, recon)

                history["step"].append(step)
                history["total_loss"].append(loss_dict["total_loss"])
                history["recon_loss"].append(loss_dict["recon_loss"])
                history["sparsity_loss"].append(loss_dict["sparsity_loss"])
                history["l0"].append(l0)
                history["fvu"].append(fvu)

                print(
                    f"[Step {step:>6d}] "
                    f"loss={loss_dict['total_loss']:.4f} "
                    f"recon={loss_dict['recon_loss']:.4f} "
                    f"sparse={loss_dict['sparsity_loss']:.4f} "
                    f"L0={l0:.1f} "
                    f"FVU={fvu:.4f}"
                )

            # ---- Checkpoint ----
            if step > 0 and step % config.checkpoint_every == 0:
                from src.utils import save_checkpoint
                save_checkpoint(
                    sae=sae,
                    optimizer=optimizer,
                    step=step,
                    metrics={"l0": l0, "fvu": fvu, **loss_dict},
                    save_dir=config.checkpoint_dir,
                )

            step += 1

    print("-" * 60)
    print(f"[Train] 训练完成! 共 {step} steps")

    # 保存最终 checkpoint
    from src.utils import save_checkpoint
    with torch.no_grad():
        final_recon, final_acts = sae(batch)
        final_l0 = compute_l0(final_acts)
        final_fvu = compute_fvu(batch, final_recon)

    save_checkpoint(
        sae=sae,
        optimizer=optimizer,
        step=step,
        metrics={"l0": final_l0, "fvu": final_fvu},
        save_dir=config.checkpoint_dir,
        filename="checkpoint_final.pt",
    )

    return history
