"""Spread-error rank correlation vs sigma, and a check of decoupling sigma for the
EnsCGP posterior MEAN vs sigma for the posterior SPREAD.

Part 1: Spearman rank correlation between posterior sigma_pred (Fortin-corrected std
from A_post) and realized |error| against WRF truth, pooled over held-out pixels, at
a range of sigma. A SSR close to 1.0 means the spread is right on average, but says
nothing about whether the spread tracks error pixel-to-pixel; rank correlation is
that separate, complementary check -- if it's still decent at the MAE-optimal sigma
(~10-30, see sigma_mae_sweep.py), the spread is still doing *some* useful ranking
work there even though its overall scale (SSR) is off.

Part 2: if so, there's no mathematical reason the posterior MEAN and posterior
SPREAD have to come from the same conditioning pass. enscgp() takes R_inv (i.e.
sigma) and returns both mean_post and A_post from one linear-Gaussian update -- but
nothing stops running it twice over the same precomputed (mean, A, y) at two
different sigmas and keeping mean_post from one, A_post from the other. This
evaluates exactly that: mean from a MAE-tuned sigma_mean, spread (Cholesky/variance)
from an SSR-tuned sigma_spread, and reports whether the combination gets close to
the best of both (sigma_mean's MAE, sigma_spread's SSR) rather than the compromise
either single sigma forces.

Reuses precompute_priors_and_obs from tune_enscgp_sigma.py (sigma-independent,
computed once).

Usage:
    python sigma_decouple_check.py --split val --n_samples 150 \
        --correlation_sweep 10 20 30 60 92.34 \
        --decouple_pairs 20,92.34 10,92.34 30,92.34
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from enscgp_train import build_r_inv, enscgp, load_era5, load_hr, load_neighbors, load_observation_operator
from tune_enscgp_sigma import DEFAULT_DATA_DIR, precompute_priors_and_obs
from variance_recalibration import fortin_factor, spread_skill_ratio


def _posterior_arrays(sigma: float, cached: list, H_valid, n_valid: int, hw: int = 200):
    """mean_post, sigma_pred_u, sigma_pred_v (Fortin-corrected) per sample, at one sigma."""
    R_inv = build_r_inv(n_valid, sigma)
    n = hw * hw
    k = cached[0][1].shape[1]
    fac = np.sqrt(fortin_factor(k=k))

    means, sp_us, sp_vs = [], [], []
    for mean, A, y in cached:
        mean_post, A_post = enscgp(mean, A, H_valid, R_inv, y)
        means.append(mean_post)
        sp_us.append(fac * np.sqrt(np.sum(A_post[:n] * A_post[:n], axis=1)))
        sp_vs.append(fac * np.sqrt(np.sum(A_post[n:] * A_post[n:], axis=1)))
    return means, sp_us, sp_vs


def spread_error_rank_correlation(sigma: float, cached: list, H_valid, n_valid: int, truth_uv: np.ndarray,
                                   hw: int = 200, max_points: int = 300_000, seed: int = 0) -> dict:
    """Spearman correlation between sigma_pred and |error|, pooled over held-out pixels
    (subsampled to max_points per component for speed -- exact Spearman on tens of
    millions of points is unnecessary and slow; this estimate is already precise enough)."""
    n = hw * hw
    means, sp_us, sp_vs = _posterior_arrays(sigma, cached, H_valid, n_valid, hw)

    err_u_list, err_v_list = [], []
    for mean_post, truth in zip(means, truth_uv):
        err_u_list.append(np.abs(truth[0].ravel() - mean_post[:n]))
        err_v_list.append(np.abs(truth[1].ravel() - mean_post[n:]))

    sp_u, sp_v = np.concatenate(sp_us), np.concatenate(sp_vs)
    err_u, err_v = np.concatenate(err_u_list), np.concatenate(err_v_list)
    sp_combined = np.concatenate([sp_u, sp_v])
    err_combined = np.concatenate([err_u, err_v])

    rng = np.random.default_rng(seed)

    def _corr(sp, err):
        if len(sp) > max_points:
            idx = rng.choice(len(sp), size=max_points, replace=False)
            sp, err = sp[idx], err[idx]
        rho, _ = spearmanr(sp, err)
        return float(rho)

    return {
        "sigma": sigma,
        "rho_u": _corr(sp_u, err_u),
        "rho_v": _corr(sp_v, err_v),
        "rho_combined": _corr(sp_combined, err_combined),
    }


def evaluate_decoupled(sigma_mean: float, sigma_spread: float, cached: list, H_valid, n_valid: int,
                        truth_uv: np.ndarray, extreme_percentile: float = 95.0, hw: int = 200) -> dict:
    """Posterior MEAN from a conditioning pass at sigma_mean; posterior SPREAD (A_post,
    hence the Cholesky/variance) from a SEPARATE pass at sigma_spread, over the same
    precomputed (mean, A, y). Reports combined MAE and truth-stratified bulk/extreme SSR.
    """
    n = hw * hw
    means, _, _ = _posterior_arrays(sigma_mean, cached, H_valid, n_valid, hw)
    _, sp_us, sp_vs = _posterior_arrays(sigma_spread, cached, H_valid, n_valid, hw)

    abs_err_u, abs_err_v = [], []
    err_u_list, err_v_list, truth_u_list, truth_v_list = [], [], [], []
    for mean_post, sp_u, sp_v, truth in zip(means, sp_us, sp_vs, truth_uv):
        mean_u, mean_v = mean_post[:n], mean_post[n:]
        abs_err_u.append(np.abs(truth[0].ravel() - mean_u))
        abs_err_v.append(np.abs(truth[1].ravel() - mean_v))
        err_u_list.append(truth[0].ravel() - mean_u)
        err_v_list.append(truth[1].ravel() - mean_v)
        truth_u_list.append(truth[0].ravel())
        truth_v_list.append(truth[1].ravel())

    mae_u = float(np.mean(np.concatenate(abs_err_u)))
    mae_v = float(np.mean(np.concatenate(abs_err_v)))

    sp_u, sp_v = np.concatenate(sp_us), np.concatenate(sp_vs)
    err_u, err_v = np.concatenate(err_u_list), np.concatenate(err_v_list)
    truth_u, truth_v = np.concatenate(truth_u_list), np.concatenate(truth_v_list)

    extreme_u = np.abs(truth_u) >= np.percentile(np.abs(truth_u), extreme_percentile)
    extreme_v = np.abs(truth_v) >= np.percentile(np.abs(truth_v), extreme_percentile)

    bulk_ssr = spread_skill_ratio(
        np.concatenate([sp_u[~extreme_u], sp_v[~extreme_v]]), np.concatenate([err_u[~extreme_u], err_v[~extreme_v]])
    )
    extreme_ssr = spread_skill_ratio(
        np.concatenate([sp_u[extreme_u], sp_v[extreme_v]]), np.concatenate([err_u[extreme_u], err_v[extreme_v]])
    )

    return {
        "sigma_mean": sigma_mean, "sigma_spread": sigma_spread,
        "mae_u": mae_u, "mae_v": mae_v, "mae_combined": 0.5 * (mae_u + mae_v),
        "bulk_ssr": bulk_ssr, "extreme_ssr": extreme_ssr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--correlation_sweep", type=float, nargs="+", default=[10.0, 20.0, 30.0, 60.0, 92.34])
    parser.add_argument("--decouple_pairs", type=str, nargs="+", default=["20,92.34", "10,92.34", "30,92.34"],
                         help="sigma_mean,sigma_spread pairs to evaluate, e.g. 20,92.34")
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

    truth_uv = np.asarray(wrf[sample_indices], dtype=np.float64)

    print("Precomputing priors and observations (sigma-independent)...")
    cached = precompute_priors_and_obs(sample_indices, neighbors, wrf, era5, valid)

    print("\n--- Part 1: spread-error Spearman rank correlation vs sigma ---")
    print(f"{'sigma':>10}  {'rho_u':>8}  {'rho_v':>8}  {'rho_combined':>13}")
    for sigma in args.correlation_sweep:
        r = spread_error_rank_correlation(sigma, cached, H_valid, n_valid, truth_uv)
        print(f"{r['sigma']:>10.3f}  {r['rho_u']:>8.4f}  {r['rho_v']:>8.4f}  {r['rho_combined']:>13.4f}")

    print("\n--- Part 2: decoupled sigma_mean / sigma_spread ---")
    print(f"{'sigma_mean':>10}  {'sigma_spread':>12}  {'mae_combined':>13}  {'bulk_ssr':>10}  {'extreme_ssr':>12}")
    for pair in args.decouple_pairs:
        sigma_mean_str, sigma_spread_str = pair.split(",")
        r = evaluate_decoupled(float(sigma_mean_str), float(sigma_spread_str), cached, H_valid, n_valid, truth_uv)
        print(f"{r['sigma_mean']:>10.3f}  {r['sigma_spread']:>12.3f}  {r['mae_combined']:>13.4f}  "
              f"{r['bulk_ssr']:>10.4f}  {r['extreme_ssr']:>12.4f}")


if __name__ == "__main__":
    main()
