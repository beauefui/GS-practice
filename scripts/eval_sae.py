"""
SAE 评估和可视化脚本

用法:
    # 评估训练好的 SAE
    python scripts/eval_sae.py --checkpoint sae/checkpoint_final.pt

    # 评估 Gemma Scope 预训练 SAE
    python scripts/eval_sae.py --pretrained --config configs/default.yaml

    # Smoke test (随机数据)
    python scripts/eval_sae.py --smoke-test
"""

import argparse
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import JumpReLUSAE
from src.metrics import compute_all_metrics
from src.utils import get_device, load_checkpoint


def evaluate_on_activations(
    sae: JumpReLUSAE,
    activations: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    """
    在一组激活值上评估 SAE

    返回聚合的 metrics
    """
    sae.eval()
    sae = sae.to(device)

    all_l0, all_fvu, all_mse = [], [], []

    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i + batch_size].to(device)
            recon, acts = sae(batch)
            metrics = compute_all_metrics(batch, recon, acts)
            all_l0.append(metrics["l0"])
            all_fvu.append(metrics["fvu"])
            all_mse.append(metrics["mse"])

    return {
        "l0": np.mean(all_l0),
        "fvu": np.mean(all_fvu),
        "mse": np.mean(all_mse),
    }


def find_top_activating_features(
    sae: JumpReLUSAE,
    activations: torch.Tensor,
    device: torch.device,
    top_k: int = 10,
) -> dict:
    """
    找出最频繁激活的特征

    Args:
        sae: SAE 模型
        activations: 激活值, shape (num_tokens, d_model)
        top_k: 返回前 k 个特征

    Returns:
        dict 包含:
          - feature_indices: top-k 特征的索引
          - activation_freqs: 对应的激活频率
          - mean_activations: 对应的平均激活强度
    """
    sae.eval()
    sae = sae.to(device)

    with torch.no_grad():
        acts = sae.encode(activations.to(device))  # (num_tokens, d_sae)

    # 每个特征的激活频率 (多少比例的 token 激活了这个特征)
    activation_freqs = (acts > 0).float().mean(dim=0).cpu()  # (d_sae,)

    # 每个特征的平均激活强度 (只看激活了的 token)
    act_sum = acts.sum(dim=0).cpu()
    act_count = (acts > 0).float().sum(dim=0).cpu()
    mean_activations = torch.where(
        act_count > 0,
        act_sum / act_count,
        torch.zeros_like(act_sum),
    )

    # 找 top-k
    top_indices = activation_freqs.argsort(descending=True)[:top_k]

    return {
        "feature_indices": top_indices.tolist(),
        "activation_freqs": activation_freqs[top_indices].tolist(),
        "mean_activations": mean_activations[top_indices].tolist(),
    }


def print_evaluation_report(
    metrics: dict,
    top_features: dict | None = None,
):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("[REPORT] SAE 评估报告")
    print("=" * 60)

    print(f"\n{'指标':<20} {'值':<15} {'解读'}")
    print("-" * 60)

    l0 = metrics["l0"]
    fvu = metrics["fvu"]
    mse = metrics["mse"]

    print(f"{'L0 (稀疏度)':<20} {l0:<15.1f} 平均激活特征数")
    print(f"{'FVU (重建质量)':<20} {fvu:<15.4f} {'[GOOD]' if fvu < 0.1 else '[OK]' if fvu < 0.5 else '[BAD]'}")
    print(f"{'MSE (均方误差)':<20} {mse:<15.6f}")

    if top_features:
        print(f"\n{'Top-10 最活跃特征':}")
        print("-" * 60)
        print(f"{'排名':<6} {'特征ID':<10} {'激活频率':<12} {'平均强度'}")
        for i, (idx, freq, strength) in enumerate(zip(
            top_features["feature_indices"],
            top_features["activation_freqs"],
            top_features["mean_activations"],
        )):
            print(f"#{i+1:<5} {idx:<10} {freq:<12.4f} {strength:.4f}")

    print("\n" + "=" * 60)


def save_report(
    metrics: dict,
    top_features: dict | None = None,
    save_dir: str | Path = "reports",
    source: str = "unknown",
):
    """
    保存评估报告到文件 (Markdown + JSON)

    Args:
        metrics: 评估指标字典
        top_features: top-k 特征信息
        save_dir: 保存目录
        source: 来源说明 (如 checkpoint 路径或 "pretrained")
    """
    import json
    from datetime import datetime

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 保存 JSON (方便程序读取) ----
    json_path = save_dir / f"report_{timestamp}.json"
    report_data = {
        "timestamp": timestamp,
        "source": source,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    if top_features:
        report_data["top_features"] = top_features

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # ---- 保存 Markdown (方便人阅读) ----
    md_path = save_dir / f"report_{timestamp}.md"

    l0 = metrics["l0"]
    fvu = metrics["fvu"]
    mse = metrics["mse"]
    fvu_label = "GOOD" if fvu < 0.1 else "OK" if fvu < 0.5 else "BAD"

    lines = [
        f"# SAE Evaluation Report",
        f"",
        f"- **Time**: {timestamp}",
        f"- **Source**: {source}",
        f"",
        f"## Metrics",
        f"",
        f"| Metric | Value | Status |",
        f"|--------|-------|--------|",
        f"| L0 (Sparsity) | {l0:.1f} | - |",
        f"| FVU (Reconstruction) | {fvu:.4f} | {fvu_label} |",
        f"| MSE | {mse:.6f} | - |",
    ]

    if top_features:
        lines += [
            f"",
            f"## Top-10 Features",
            f"",
            f"| Rank | Feature ID | Activation Freq | Mean Strength |",
            f"|------|-----------|-----------------|---------------|",
        ]
        for i, (idx, freq, strength) in enumerate(zip(
            top_features["feature_indices"],
            top_features["activation_freqs"],
            top_features["mean_activations"],
        )):
            lines.append(f"| #{i+1} | {idx} | {freq:.4f} | {strength:.4f} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[Report] Markdown: {md_path}")
    print(f"[Report] JSON:     {json_path}")


def main():
    parser = argparse.ArgumentParser(description="评估 SAE")
    parser.add_argument("--checkpoint", type=str, help="训练好的 checkpoint 路径")
    parser.add_argument("--pretrained", action="store_true", help="评估 Gemma Scope 预训练 SAE")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件")
    parser.add_argument("--smoke-test", action="store_true", help="快速测试模式")
    args = parser.parse_args()

    device = get_device()

    # ---- Smoke Test ----
    if args.smoke_test:
        print("\n[SMOKE TEST] 模式\n")

        d_model, d_sae = 64, 512
        sae = JumpReLUSAE(d_model=d_model, d_sae=d_sae)
        torch.nn.init.xavier_uniform_(sae.W_enc.data)
        torch.nn.init.xavier_uniform_(sae.W_dec.data)
        sae.threshold.data = torch.ones(d_sae) * 0.1

        fake_acts = torch.randn(500, d_model)
        metrics = evaluate_on_activations(sae, fake_acts, device)
        top_features = find_top_activating_features(sae, fake_acts, device)
        print_evaluation_report(metrics, top_features)
        save_report(metrics, top_features, save_dir="reports", source="smoke-test")
        print("\n[PASS] Smoke test 通过!")
        return

    # ---- 加载 checkpoint 评估 ----
    if args.checkpoint:
        print(f"\n[LOAD] 加载 checkpoint: {args.checkpoint}\n")
        ckpt = load_checkpoint(args.checkpoint, device=device)
        sae_config = ckpt["sae_config"]
        sae = JumpReLUSAE(d_model=sae_config["d_model"], d_sae=sae_config["d_sae"])
        load_checkpoint(args.checkpoint, sae=sae, device=device)

        # 需要激活值来评估 — 用随机数据做演示
        print("[WARN] 使用随机激活值进行评估 (正式评估请加载真实数据)")
        fake_acts = torch.randn(1000, sae_config["d_model"])
        metrics = evaluate_on_activations(sae, fake_acts, device)
        top_features = find_top_activating_features(sae, fake_acts, device)
        print_evaluation_report(metrics, top_features)
        save_report(metrics, top_features, save_dir="reports", source=args.checkpoint)
        return

    # ---- 加载预训练 SAE 评估 ----
    if args.pretrained:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        from src.utils import load_sae_weights
        params = load_sae_weights(
            repo_id=config["pretrained_sae"]["repo_id"],
            layer=config["pretrained_sae"]["layer"],
            width=config["pretrained_sae"]["width"],
            l0=config["pretrained_sae"]["l0"],
            local_dir=config["pretrained_sae"].get("local_dir"),
        )
        sae = JumpReLUSAE.from_pretrained(params)
        print(f"\n[OK] 加载预训练 SAE: {sae}\n")

        # 加载 Gemma 模型, 提取真实激活值进行评估
        from src.utils import load_gemma_model
        from src.train import collect_activations

        model_name = config["model"]["name"]
        hook_layer = config["model"]["hook_layer"]
        hook_point = config["model"].get("hook_point", "residual")
        dtype_str = config["model"].get("dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

        print(f"\n[Phase 1] 加载 Gemma 模型: {model_name}")
        model, tokenizer = load_gemma_model(model_name, device="auto", dtype=dtype)

        # 收集少量激活值用于评估 (不需要太多)
        print(f"[Phase 2] 收集第 {hook_layer} 层激活值...")
        eval_texts = ["The quick brown fox jumps over the lazy dog.",
                       "Machine learning is a subset of artificial intelligence.",
                       "Neural networks consist of layers of interconnected nodes.",
                       "Sparse autoencoders decompose activations into interpretable features.",
                       "The transformer architecture revolutionized natural language processing."] * 20
        activations = collect_activations(
            model, tokenizer, eval_texts,
            layer_idx=hook_layer, hook_point=hook_point,
            max_seq_len=128, device=device,
        ).float()

        # 释放 Gemma 显存
        del model
        torch.cuda.empty_cache()
        print("[Phase 2] Gemma 模型已释放\n")

        # 评估
        print("[Phase 3] 评估预训练 SAE...")
        metrics = evaluate_on_activations(sae, activations, device)
        top_features = find_top_activating_features(sae, activations, device)
        print_evaluation_report(metrics, top_features)
        save_report(metrics, top_features, save_dir="reports", source="pretrained-gemma-scope")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
