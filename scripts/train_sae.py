"""
SAE 训练入口脚本

用法:
    python scripts/train_sae.py --config configs/default.yaml

    # 快速测试 (小规模, CPU)
    python scripts/train_sae.py --config configs/default.yaml --smoke-test
"""

import argparse
import sys
import yaml
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import JumpReLUSAE
from src.train import TrainConfig, collect_activations, train_sae
from src.utils import set_seed, get_device, load_gemma_model


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"[Config] 已加载配置: {config_path}")
    return config


def main():
    # ---- 解析命令行参数 ----
    parser = argparse.ArgumentParser(description="训练 JumpReLU SAE")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="快速测试模式: 用随机数据跑几步, 验证流程是否正常",
    )
    args = parser.parse_args()

    # ---- 加载配置 ----
    config = load_config(args.config)

    # ---- 设置随机种子 ----
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    # ---- 获取设备 ----
    device = get_device()

    # ---- Smoke Test 模式: 用随机数据快速验证 ----
    if args.smoke_test:
        print("\n" + "=" * 60)
        print("[SMOKE TEST] 模式 - 使用随机数据快速验证")
        print("=" * 60 + "\n")

        import torch

        d_model = 64
        d_sae = 512
        num_tokens = 2000

        sae = JumpReLUSAE(d_model=d_model, d_sae=d_sae)
        # 随机初始化权重
        torch.nn.init.xavier_uniform_(sae.W_enc.data)
        torch.nn.init.xavier_uniform_(sae.W_dec.data)
        sae.threshold.data = torch.ones(d_sae) * 0.1

        fake_activations = torch.randn(num_tokens, d_model)

        train_config = TrainConfig(
            lr=1e-3,
            num_steps=20,
            batch_size=256,
            sparsity_coeff=1e-3,
            log_every=5,
            checkpoint_every=9999,
            checkpoint_dir=str(PROJECT_ROOT / "sae"),
        )

        history = train_sae(sae, fake_activations, train_config, device=device)

        # 验证 loss 趋势
        if len(history["total_loss"]) >= 2:
            first = history["total_loss"][0]
            last = history["total_loss"][-1]
            if last < first:
                print(f"\n[PASS] Smoke test 通过! Loss 下降了: {first:.4f} -> {last:.4f}")
            else:
                print(f"\n[WARN] Loss 没有明显下降: {first:.4f} -> {last:.4f}")

        return

    # ---- 正式训练模式 ----
    print("\n" + "=" * 60)
    print("[TRAIN] 正式训练模式")
    print("=" * 60 + "\n")

    import torch

    # 1. 加载基座模型
    model_name = config["model"]["name"]
    dtype_str = config["model"].get("dtype", "float32")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    model, tokenizer = load_gemma_model(model_name, device="auto", dtype=dtype)

    # 2. 收集激活值
    print("\n[Phase 1] 收集激活值...")
    hook_layer = config["model"]["hook_layer"]
    hook_point = config["model"]["hook_point"]
    max_seq_len = config["data"]["max_seq_len"]
    num_texts = config["data"]["num_texts"]

    # 加载数据集
    from datasets import load_dataset
    dataset = load_dataset(
        config["data"]["dataset"],
        split="train",
        streaming=True,
    )
    texts = []
    for i, sample in enumerate(dataset):
        if i >= num_texts:
            break
        texts.append(sample["text"])
    print(f"[Data] 已加载 {len(texts)} 段文本")

    # 提取激活值
    activations = collect_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=hook_layer,
        hook_point=hook_point,
        max_seq_len=max_seq_len,
        device=device,
    )

    # 释放基座模型显存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("[Memory] 已释放基座模型")

    # 3. 初始化 SAE
    d_model = activations.shape[-1]
    d_sae = config["sae"]["d_sae"]
    sae = JumpReLUSAE(d_model=d_model, d_sae=d_sae)
    # Xavier 初始化
    torch.nn.init.xavier_uniform_(sae.W_enc.data)
    torch.nn.init.xavier_uniform_(sae.W_dec.data)
    sae.threshold.data = torch.ones(d_sae) * 0.01  # 初始阈值设小, 让更多特征激活
    print(f"[SAE] 初始化完成: {sae}")

    # 4. 训练
    train_config = TrainConfig(
        lr=config["training"]["lr"],
        num_steps=config["training"]["num_steps"],
        batch_size=config["training"]["batch_size"],
        sparsity_coeff=config["training"]["sparsity_coeff"],
        log_every=config["training"]["log_every"],
        checkpoint_every=config["training"]["checkpoint_every"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
    )

    history = train_sae(sae, activations.float(), train_config, device=device)

    print("\n[DONE] 训练完成!")
    print(f"最终 Loss: {history['total_loss'][-1]:.4f}")
    print(f"最终 L0: {history['l0'][-1]:.1f}")
    print(f"最终 FVU: {history['fvu'][-1]:.4f}")


if __name__ == "__main__":
    main()
