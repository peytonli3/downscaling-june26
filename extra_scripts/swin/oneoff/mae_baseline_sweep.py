#!/usr/bin/env python3
"""
Per-pixel MAE diagnostic for the two baselines fed into the Swin model:
  - EnsCGP posterior mean   (enscgp_posterior.npy, channels 0-1)
  - ERA5 bicubic            (era5_uv_2ch_bicubic.npy)
  - Weighted sums:          alpha * EnsCGP + (1-alpha) * bicubic

Reports mean-over-samples MAE for u, v, and combined (u+v)/2 at each alpha, on all pixels
and on each wind-speed stratum requested by --top-pcts (the windiest N% of each sample's
pixels by WIND SPEED -- quantile_metrics.top_speed_mask, the same stratification
eval_model_scorecard.py and the training logs use). The strata are what make this table
comparable with the model's own numbers: a baseline that wins on bulk MAE by being smooth
tends to lose badly where the wind is actually strong, and only the stratified column shows
it.

These are the BASELINE rows of the README results table; the model's own row comes from
eval_model_scorecard.py, which stratifies identically.

Usage:
    python mae_baseline_sweep.py
    python mae_baseline_sweep.py --n_samples 256 --split test
    python mae_baseline_sweep.py --top-pcts 10 5 1
    python mae_baseline_sweep.py --alphas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import resolve as resolve_path  # noqa: E402

from new_enscgp_swin import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from quantile_metrics import stratified_error, top_speed_frac_to_pct  # noqa: E402

DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_TOP_PCTS = [10.0, 5.0]  # matches eval_model_scorecard.DEFAULT_TOP_PCTS


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
    parser.add_argument("--top-pcts", type=float, nargs="*", default=DEFAULT_TOP_PCTS,
                        metavar="PCT",
                        help="Also report MAE on the windiest PCT%% of each sample's pixels. "
                             "Pass none for all-pixel MAE only.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = args.data_dir or resolve_path(paths["data_dir"])
    splits_path = args.splits_path or resolve_path(paths["splits_path"])

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

    # One column group per stratum: all pixels, then each --top-pcts entry.
    strata = ["all"] + [f"top{top_speed_frac_to_pct(p / 100.0)}" for p in args.top_pcts]
    col_w = 12
    groups = "".join(f"  {'MAE u (' + s + ')':>{col_w}}  {'MAE v (' + s + ')':>{col_w}}"
                     f"  {'MAE avg (' + s + ')':>{col_w}}" for s in strata)
    header = f"{'Source':<28}{groups}"
    print(f"Strata: 'all' = every pixel; 'topN' = the windiest N% of EACH sample's pixels "
          f"(by wind speed).\n")
    print(header)
    print("-" * len(header))

    def report(label: str, pred: np.ndarray) -> None:
        # per_component -> [u, v] per stratum, in one pass over the masks.
        per_comp = stratified_error(pred, wrf_uv, top_fracs=[p / 100.0 for p in args.top_pcts],
                                    metric="l1", per_component=True)
        cells = ""
        for s in strata:
            mae_u, mae_v = float(per_comp[s][0]), float(per_comp[s][1])
            cells += (f"  {mae_u:>{col_w}.5f}  {mae_v:>{col_w}.5f}"
                      f"  {(mae_u + mae_v) / 2.0:>{col_w}.5f}")
        print(f"{label:<28}{cells}")

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
