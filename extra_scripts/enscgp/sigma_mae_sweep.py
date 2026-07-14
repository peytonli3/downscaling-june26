"""Mean-absolute-error vs sigma for the EnsCGP posterior mean.

Checks whether enlarging sigma (needed to fix the posterior's drastically
overconfident spread -- see tune_enscgp_sigma.py) trades away mean accuracy: as
sigma grows, the ERA5 observation is trusted less, so mean_post reverts toward the
prior (k=36 nearest-neighbor analog) mean. If that prior mean is itself a worse
predictor of WRF truth than a properly ERA5-conditioned posterior mean, increasing
sigma should show up as MAE(mean_post, truth) rising toward MAE(prior_mean, truth)
-- the prior-only, sigma->infinity limit, also reported as a reference line.

Reuses precompute_priors_and_obs from tune_enscgp_sigma.py: (mean, A, y) per sample
is sigma-independent, computed once and reused for every candidate sigma.

Usage:
    python sigma_mae_sweep.py --split val --n_samples 150 --sweep 2.77 10 30 60 92.34 150 250 445
"""
import argparse
from pathlib import Path

import numpy as np

from enscgp_train import build_r_inv, enscgp, load_era5, load_hr, load_neighbors, load_observation_operator
from tune_enscgp_sigma import DEFAULT_DATA_DIR, precompute_priors_and_obs


def mae_at_sigma(sigma: float, cached: list, H_valid, n_valid: int, truth_uv: np.ndarray, hw: int = 200) -> dict:
    R_inv = build_r_inv(n_valid, sigma)
    n = hw * hw
    abs_err_u, abs_err_v = [], []
    for (mean, A, y), truth in zip(cached, truth_uv):
        mean_post, _ = enscgp(mean, A, H_valid, R_inv, y)
        abs_err_u.append(np.abs(truth[0].ravel() - mean_post[:n]))
        abs_err_v.append(np.abs(truth[1].ravel() - mean_post[n:]))
    mae_u = float(np.mean(np.concatenate(abs_err_u)))
    mae_v = float(np.mean(np.concatenate(abs_err_v)))
    return {"sigma": sigma, "mae_u": mae_u, "mae_v": mae_v, "mae_combined": 0.5 * (mae_u + mae_v)}


def prior_only_mae(cached: list, truth_uv: np.ndarray, hw: int = 200) -> dict:
    """sigma -> infinity limit: mean_post == prior mean (no conditioning at all)."""
    n = hw * hw
    abs_err_u, abs_err_v = [], []
    for (mean, A, y), truth in zip(cached, truth_uv):
        abs_err_u.append(np.abs(truth[0].ravel() - mean[:n]))
        abs_err_v.append(np.abs(truth[1].ravel() - mean[n:]))
    mae_u = float(np.mean(np.concatenate(abs_err_u)))
    mae_v = float(np.mean(np.concatenate(abs_err_v)))
    return {"mae_u": mae_u, "mae_v": mae_v, "mae_combined": 0.5 * (mae_u + mae_v)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--sweep", type=float, nargs="+", default=[2.77, 10, 30, 60, 92.34, 150, 250, 445])
    args = parser.parse_args()

    data_dir = args.data_dir
    neighbors = load_neighbors(data_dir / "neighbor_train_only.npy")
    wrf = load_hr(data_dir / "wrf_uv.npy")
    era5 = load_era5(data_dir / "era5_uv_2ch_native34.npy")
    H_valid, valid = load_observation_operator(data_dir / "coarsening_operator_H.npy")
    n_valid = int(valid.sum())

    splits = np.load(data_dir / "splits_70_15_15" / "split_indices.npz")
    rng = np.random.default_rng(0)
    split_idx = splits[f"{args.split}_idx"]
    sample_indices = rng.choice(split_idx, size=min(args.n_samples, len(split_idx)), replace=False)
    print(f"Using {len(sample_indices)} samples from split '{args.split}'")

    truth_uv = np.asarray(wrf[sample_indices], dtype=np.float64)  # (S,2,200,200)

    print("Precomputing priors and observations (sigma-independent)...")
    cached = precompute_priors_and_obs(sample_indices, neighbors, wrf, era5, valid)

    prior_mae = prior_only_mae(cached, truth_uv)
    print("\nPrior-only MAE (sigma -> infinity limit, no ERA5 conditioning at all):")
    print(f"  u={prior_mae['mae_u']:.4f}  v={prior_mae['mae_v']:.4f}  combined={prior_mae['mae_combined']:.4f}")

    print(f"\n{'sigma':>10}  {'mae_u':>8}  {'mae_v':>8}  {'mae_combined':>13}")
    for sigma in args.sweep:
        r = mae_at_sigma(sigma, cached, H_valid, n_valid, truth_uv)
        print(f"{r['sigma']:>10.3f}  {r['mae_u']:>8.4f}  {r['mae_v']:>8.4f}  {r['mae_combined']:>13.4f}")


if __name__ == "__main__":
    main()
