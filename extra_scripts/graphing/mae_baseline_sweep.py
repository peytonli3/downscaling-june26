#!/usr/bin/env python3
"""
Per-pixel MAE diagnostic for the two baselines fed into the Swin model:
  - EnsCGP posterior mean   (enscgp_posterior.npy, channels 0-1)
  - ERA5 bicubic            (era5_uv_2ch_bicubic.npy)
  - Weighted sums:          alpha * EnsCGP + (1-alpha) * bicubic

Reports mean-over-samples MAE for u, v, and combined (u+v)/2 at each alpha.

Usage:
    python mae_baseline_sweep.py
    python mae_baseline_sweep.py --n_samples 256 --split test
    python mae_baseline_sweep.py --alphas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from new_enscgp_swin import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--splits_path", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                        help="EnsCGP weights to sweep (0=pure bicubic, 1=pure EnsCGP)")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = args.data_dir or Path(paths["data_dir"])
    splits_path = args.splits_path or Path(paths["splits_path"])

    wrf = np.load(data_dir / "wrf_uv.npy", mmap_mode="r")
    posterior = np.load(data_dir / "enscgp_posterior.npy", mmap_mode="r")
    bicubic = np.load(data_dir / "era5_uv_2ch_bicubic.npy", mmap_mode="r")

    splits = np.load(splits_path)
    pool = splits[f"{args.split}_idx"]
    n = min(args.n_samples, len(pool))
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(pool, size=n, replace=False))
    print(f"Split: {args.split}  |  Samples: {n} of {len(pool)}  |  Seed: {args.seed}\n")

    wrf_uv = np.array(wrf[idx, :2], dtype=np.float32)          # (N, 2, H, W)
    post_uv = np.array(posterior[idx, :2], dtype=np.float32)   # (N, 2, H, W)
    bic_uv = np.array(bicubic[idx, :2], dtype=np.float32)      # (N, 2, H, W)

    # Header
    col_w = 12
    header = f"{'Source':<28}  {'MAE u':>{col_w}}  {'MAE v':>{col_w}}  {'MAE (u+v)/2':>{col_w}}"
    print(header)
    print("-" * len(header))

    def report(label: str, pred: np.ndarray) -> None:
        mae_u = float(np.abs(pred[:, 0] - wrf_uv[:, 0]).mean())
        mae_v = float(np.abs(pred[:, 1] - wrf_uv[:, 1]).mean())
        mae_avg = (mae_u + mae_v) / 2.0
        print(f"{label:<28}  {mae_u:>{col_w}.5f}  {mae_v:>{col_w}.5f}  {mae_avg:>{col_w}.5f}")

    report("EnsCGP posterior mean (α=1)", post_uv)
    report("ERA5 bicubic (α=0)", bic_uv)

    alphas = sorted(set(args.alphas) - {0.0, 1.0})
    if alphas:
        print()
        for alpha in alphas:
            blend = alpha * post_uv + (1.0 - alpha) * bic_uv
            report(f"α·EnsCGP + (1-α)·bicubic  α={alpha:.2f}", blend)


if __name__ == "__main__":
    main()
