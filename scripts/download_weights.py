"""
下载模型权重脚本

用法:
    python scripts/download_weights.py --token hf_你的token
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="下载模型权重")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace token")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    # 1. 下载 Gemma 3 1B 基座模型
    print("=" * 60)
    print("[1/2] Gemma 3 1B ...")
    print("=" * 60)
    snapshot_download(
        repo_id="google/gemma-3-1b-pt",
        local_dir="model/gemma-3-1b-pt",
        token=args.token,
    )
    print("[OK] model/gemma-3-1b-pt/\n")

    # 2. 下载 Gemma Scope SAE 权重 (只下需要的)
    print("=" * 60)
    print("[2/2] Gemma Scope SAE ...")
    print("=" * 60)
    snapshot_download(
        repo_id="google/gemma-scope-2-1b-pt",
        local_dir="sae/gemma-scope-2-1b-pt",
        allow_patterns=["resid_post/layer_22_width_65k_l0_medium/*"],
        token=args.token,
    )
    print("[OK] sae/gemma-scope-2-1b-pt/\n")

    print("[DONE]")


if __name__ == "__main__":
    main()
