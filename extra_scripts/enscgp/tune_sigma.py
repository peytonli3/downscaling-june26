"""Choosing the EnsCGP ERA5 observation-noise sigma, from three angles.

Context shared by every mode below
---------------------------------
sigma=2.77 (enscgp_train.py's default) is not arbitrary -- it is
sqrt(Var(H @ WRF - ERA5)), the genuine empirically-measured representativeness /
retrieval error between coarsened WRF and ERA5 (see data_prep/compare_WRF_coarse_ERA5.py,
which reports Var=7.65, sqrt=2.766). It checks out as a real physical
observation-noise estimate, not a bug.

But the resulting POSTERIOR (data/enscgp_posterior.npy, built with that sigma) is
drastically overconfident: bulk and extreme spread-skill ratio both ~0.13, not the
~1.01/~0.70 profile originally assumed -- that figure turned out to describe the
*unconditioned* k=36 neighbor ensemble (see neighbor_spread_skill.py), a different
and much better-behaved quantity. The likely mechanism: the neighbor ensemble's
disagreement, as SEEN THROUGH the coarsening operator H, is much smaller than its
disagreement at full WRF resolution (neighbors agree strongly at the synoptic
scale even where they differ a lot pixel-to-pixel) -- so the ERA5 observation looks
far more informative to the linear-Gaussian update than it should, and the
posterior over-collapses even at a physically-correct sigma.

These modes therefore treat sigma as an effective tuning parameter compensating for
that structural mismatch (a form of covariance inflation, standard practice for
underdispersive ensembles in DA). That is distinct from the measured ERA5
representativeness error, and distinct from variance_recalibration.py's job, which
fixes any *residual* tail-specific shape mis-calibration left after this global
correction.

Modes
-----
ssr        Bisect for the sigma whose posterior has bulk spread-skill ratio == 1.0
           (or just report SSR at given sigmas with --sweep). Tunes for calibration.
mae        MAE of the posterior mean vs sigma. Tunes for accuracy -- and shows what
           calibration costs, since as sigma grows the mean reverts toward the
           unconditioned prior mean (reported as the sigma -> infinity reference).
decouple   Part 1: Spearman correlation between predicted spread and realized
           |error| -- SSR ~ 1 says the spread is right ON AVERAGE, this says whether
           it tracks error pixel-to-pixel. Part 2: nothing requires the posterior
           MEAN and SPREAD to come from the same conditioning pass; this runs
           enscgp() twice over the same precomputed (mean, A, y) and keeps the mean
           from a MAE-tuned sigma and the spread from an SSR-tuned one, to see
           whether that beats the compromise a single sigma forces.

Considered and rejected: analytic GCV
------------------------------------
The research group's Ens-CGP implementation ships `select_regularization_gcv`, which
picks the regularization lambda (= sigma_obs^2) analytically from an ensemble's own
geometry -- no bisection, no re-running enscgp(), no ERA5 observation needed. A
`tune_lambda_gcv.py` here tried it; that script was removed (2026-08-08) together with
the un-redistributable dependency it needed. Why it does not apply, recorded so nobody
re-derives it:

- Its `beta = U.T @ dY` step needs the SVD input and the regression target in the SAME
  pixel space -- ken_enscgp_model.py's built-in H=identity assumption (its
  build_training_matrices applies one shared nanmask to both X and Y). This repo
  conditions with a genuine HR->LR coarsening operator, so U from SVD(H @ A) is 2178
  rows (observation space) and cannot multiply A's 80000 rows (full HR space). Trying
  it directly raises a shape mismatch.
- It CAN be forced to run by using era5_uv_2ch_bicubic.npy as the X side (ERA5 already
  resampled onto the same 200x200 grid as WRF, index-aligned with wrf_uv.npy), so X and
  Y share one grid. But that answers a question about the bicubic-vs-WRF regression,
  not about the H-coarsened conditioning the pipeline actually performs.
- Even where it runs, GCV minimizes prediction error, which makes it an alternative to
  sigma_mean (accurate posterior mean), never to sigma_spread -- it will not target
  SSR == 1 the way `ssr` above does, and SSR is the calibration problem that motivated
  tuning sigma in the first place.

Every mode precomputes (mean, A, y) per sample once -- only R_inv depends on sigma --
and reuses it across all candidate sigmas. No disk writes.

Usage:
    python tune_sigma.py ssr --split val --n_samples 150
    python tune_sigma.py ssr --sweep 2.77 10 30 60 100 200
    python tune_sigma.py mae --sweep 2.77 10 30 60 92.34 150 250 445
    python tune_sigma.py decouple --correlation_sweep 10 20 30 60 92.34 \
        --decouple_pairs 20,92.34 10,92.34 30,92.34
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import DATA_DIR as DEFAULT_DATA_DIR  # noqa: E402
from enscgp_train import (  # noqa: E402
    build_r_inv, enscgp, load_era5, load_hr, load_neighbors, load_observation,
    load_observation_operator, prior,
)
from variance_recalibration import fortin_factor, spread_skill_ratio  # noqa: E402

HW = 200
N_PIX = HW * HW


# --------------------------------------------------------------------------------------
# Shared setup: identical for all three modes
# --------------------------------------------------------------------------------------
def precompute_priors_and_obs(sample_indices, neighbors, wrf, era5, valid) -> list:
    """(mean, A, y) per sample -- everything that does NOT depend on sigma."""
    cached = []
    for idx in sample_indices:
        mean, A = prior(int(idx), neighbors, wrf)
        y = load_observation(era5, int(idx), valid)
        cached.append((mean, A, y))
    return cached


def load_inputs(args):
    """Load arrays, draw the held-out sample subset, and precompute the sigma-independent part.

    Returns (cached, H_valid, n_valid, truth_uv).
    """
    d = args.data_dir
    neighbors = load_neighbors(d / "neighbor_train_only.npy")
    wrf = load_hr(d / "wrf_uv.npy")
    era5 = load_era5(d / "era5_uv_2ch_native34.npy")
    H_valid, valid = load_observation_operator(d / "coarsening_operator_H.npy")
    n_valid = int(valid.sum())

    splits = np.load(d / "splits_70_15_15" / "split_indices.npz")
    split_idx = splits[f"{args.split}_idx"]
    rng = np.random.default_rng(0)
    sample_indices = rng.choice(split_idx, size=min(args.n_samples, len(split_idx)), replace=False)
    print(f"Using {len(sample_indices)} samples from split '{args.split}'")

    truth_uv = np.asarray(wrf[sample_indices], dtype=np.float64)  # (S, 2, 200, 200)
    print("Precomputing priors and observations (sigma-independent)...")
    cached = precompute_priors_and_obs(sample_indices, neighbors, wrf, era5, valid)
    return cached, H_valid, n_valid, truth_uv


def posterior_arrays(sigma: float, cached: list, H_valid, n_valid: int):
    """mean_post, and Fortin-corrected per-pixel spread for u and v, at one sigma."""
    R_inv = build_r_inv(n_valid, sigma)
    fac = np.sqrt(fortin_factor(k=cached[0][1].shape[1]))
    means, sp_us, sp_vs = [], [], []
    for mean, A, y in cached:
        mean_post, A_post = enscgp(mean, A, H_valid, R_inv, y)
        means.append(mean_post)
        sp_us.append(fac * np.sqrt(np.sum(A_post[:N_PIX] * A_post[:N_PIX], axis=1)))
        sp_vs.append(fac * np.sqrt(np.sum(A_post[N_PIX:] * A_post[N_PIX:], axis=1)))
    return means, sp_us, sp_vs


def _errors_and_truth(means, truth_uv):
    """Flattened signed error and truth per component, pooled over samples."""
    err_u, err_v, tru_u, tru_v = [], [], [], []
    for mean_post, truth in zip(means, truth_uv):
        err_u.append(truth[0].ravel() - mean_post[:N_PIX])
        err_v.append(truth[1].ravel() - mean_post[N_PIX:])
        tru_u.append(truth[0].ravel())
        tru_v.append(truth[1].ravel())
    return (np.concatenate(err_u), np.concatenate(err_v),
            np.concatenate(tru_u), np.concatenate(tru_v))


def _extreme_masks(truth_u, truth_v, pct: float):
    """Truth-based, per-channel extreme masks.

    Matches neighbor_spread_skill.py --pct exactly, so numbers are directly
    comparable to that script's. (This is a diagnostic stratification, not the
    deployed rule -- variance_recalibration.py cannot use truth at apply time.)
    """
    return (np.abs(truth_u) >= np.percentile(np.abs(truth_u), pct),
            np.abs(truth_v) >= np.percentile(np.abs(truth_v), pct))


# --------------------------------------------------------------------------------------
# ssr
# --------------------------------------------------------------------------------------
def evaluate_sigma(sigma, cached, H_valid, n_valid, truth_uv, extreme_percentile=95.0) -> dict:
    """Per-channel and combined bulk/extreme spread-skill ratio at one sigma."""
    means, sp_us, sp_vs = posterior_arrays(sigma, cached, H_valid, n_valid)
    sp_u, sp_v = np.concatenate(sp_us), np.concatenate(sp_vs)
    err_u, err_v, truth_u, truth_v = _errors_and_truth(means, truth_uv)
    ext_u, ext_v = _extreme_masks(truth_u, truth_v, extreme_percentile)

    return {
        "sigma": sigma,
        "u_bulk_ssr": spread_skill_ratio(sp_u[~ext_u], err_u[~ext_u]),
        "u_extreme_ssr": spread_skill_ratio(sp_u[ext_u], err_u[ext_u]),
        "v_bulk_ssr": spread_skill_ratio(sp_v[~ext_v], err_v[~ext_v]),
        "v_extreme_ssr": spread_skill_ratio(sp_v[ext_v], err_v[ext_v]),
        "bulk_ssr": spread_skill_ratio(np.concatenate([sp_u[~ext_u], sp_v[~ext_v]]),
                                       np.concatenate([err_u[~ext_u], err_v[~ext_v]])),
        "extreme_ssr": spread_skill_ratio(np.concatenate([sp_u[ext_u], sp_v[ext_v]]),
                                          np.concatenate([err_u[ext_u], err_v[ext_v]])),
    }


def find_sigma_for_bulk_ssr(cached, H_valid, n_valid, truth_uv, lo, hi,
                            target=1.0, tol=1e-3, max_iter=40) -> float:
    """Bisect for the sigma where bulk SSR == target.

    Assumes bulk_ssr(sigma) increases monotonically (weaker conditioning -> posterior
    spread closer to the well-calibrated unconditioned ensemble); the bracket check
    below is what actually verifies that assumption holds over [lo, hi].
    """
    f_lo = evaluate_sigma(lo, cached, H_valid, n_valid, truth_uv)["bulk_ssr"] - target
    f_hi = evaluate_sigma(hi, cached, H_valid, n_valid, truth_uv)["bulk_ssr"] - target
    print(f"Bracket check: bulk_ssr({lo})={f_lo + target:.4f}, bulk_ssr({hi})={f_hi + target:.4f}")
    if f_lo > 0 or f_hi < 0:
        raise ValueError(
            f"Bracket [lo={lo}, hi={hi}] does not contain the root: bulk_ssr(lo)-1={f_lo:.4f}, "
            f"bulk_ssr(hi)-1={f_hi:.4f}. Widen --lo/--hi."
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = evaluate_sigma(mid, cached, H_valid, n_valid, truth_uv)["bulk_ssr"] - target
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def run_ssr(args) -> None:
    cached, H_valid, n_valid, truth_uv = load_inputs(args)

    if args.sweep is not None:
        print(f"\n{'sigma':>10}  {'bulk_ssr':>10}  {'extreme_ssr':>12}  {'u_bulk':>8}  "
              f"{'u_extreme':>10}  {'v_bulk':>8}  {'v_extreme':>10}")
        for sigma in args.sweep:
            r = evaluate_sigma(sigma, cached, H_valid, n_valid, truth_uv)
            print(f"{r['sigma']:>10.3f}  {r['bulk_ssr']:>10.4f}  {r['extreme_ssr']:>12.4f}  "
                  f"{r['u_bulk_ssr']:>8.4f}  {r['u_extreme_ssr']:>10.4f}  "
                  f"{r['v_bulk_ssr']:>8.4f}  {r['v_extreme_ssr']:>10.4f}")
        return

    print(f"\nBisecting for bulk SSR == 1.0 in sigma in [{args.lo}, {args.hi}]...")
    sigma_star = find_sigma_for_bulk_ssr(cached, H_valid, n_valid, truth_uv, args.lo, args.hi)
    r = evaluate_sigma(sigma_star, cached, H_valid, n_valid, truth_uv)
    print(f"\nsigma* = {sigma_star:.4f}")
    print(f"  bulk SSR    = {r['bulk_ssr']:.4f}  (u={r['u_bulk_ssr']:.4f}, v={r['v_bulk_ssr']:.4f})")
    print(f"  extreme SSR = {r['extreme_ssr']:.4f}  (u={r['u_extreme_ssr']:.4f}, v={r['v_extreme_ssr']:.4f})")


# --------------------------------------------------------------------------------------
# mae
# --------------------------------------------------------------------------------------
def _mae_from_means(means, truth_uv) -> dict:
    err_u, err_v, _, _ = _errors_and_truth(means, truth_uv)
    mae_u, mae_v = float(np.mean(np.abs(err_u))), float(np.mean(np.abs(err_v)))
    return {"mae_u": mae_u, "mae_v": mae_v, "mae_combined": 0.5 * (mae_u + mae_v)}


def run_mae(args) -> None:
    cached, H_valid, n_valid, truth_uv = load_inputs(args)

    prior_means = [mean for mean, _, _ in cached]
    prior_mae = _mae_from_means(prior_means, truth_uv)
    print("\nPrior-only MAE (sigma -> infinity limit, no ERA5 conditioning at all):")
    print(f"  u={prior_mae['mae_u']:.4f}  v={prior_mae['mae_v']:.4f}  "
          f"combined={prior_mae['mae_combined']:.4f}")

    print(f"\n{'sigma':>10}  {'mae_u':>8}  {'mae_v':>8}  {'mae_combined':>13}")
    for sigma in args.sweep:
        means, _, _ = posterior_arrays(sigma, cached, H_valid, n_valid)
        r = _mae_from_means(means, truth_uv)
        print(f"{sigma:>10.3f}  {r['mae_u']:>8.4f}  {r['mae_v']:>8.4f}  {r['mae_combined']:>13.4f}")


# --------------------------------------------------------------------------------------
# decouple
# --------------------------------------------------------------------------------------
def spread_error_rank_correlation(sigma, cached, H_valid, n_valid, truth_uv,
                                  max_points=300_000, seed=0) -> dict:
    """Spearman correlation between predicted spread and |error|, pooled over pixels.

    Subsampled to max_points per component: exact Spearman over tens of millions of
    points is slow and unnecessary, the estimate is already precise at this size.
    """
    means, sp_us, sp_vs = posterior_arrays(sigma, cached, H_valid, n_valid)
    err_u, err_v, _, _ = _errors_and_truth(means, truth_uv)
    err_u, err_v = np.abs(err_u), np.abs(err_v)
    sp_u, sp_v = np.concatenate(sp_us), np.concatenate(sp_vs)
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
        "rho_combined": _corr(np.concatenate([sp_u, sp_v]), np.concatenate([err_u, err_v])),
    }


def evaluate_decoupled(sigma_mean, sigma_spread, cached, H_valid, n_valid, truth_uv,
                       extreme_percentile=95.0) -> dict:
    """MEAN from a conditioning pass at sigma_mean, SPREAD from a separate pass at sigma_spread."""
    means, _, _ = posterior_arrays(sigma_mean, cached, H_valid, n_valid)
    _, sp_us, sp_vs = posterior_arrays(sigma_spread, cached, H_valid, n_valid)

    err_u, err_v, truth_u, truth_v = _errors_and_truth(means, truth_uv)
    sp_u, sp_v = np.concatenate(sp_us), np.concatenate(sp_vs)
    ext_u, ext_v = _extreme_masks(truth_u, truth_v, extreme_percentile)

    mae_u, mae_v = float(np.mean(np.abs(err_u))), float(np.mean(np.abs(err_v)))
    return {
        "sigma_mean": sigma_mean, "sigma_spread": sigma_spread,
        "mae_u": mae_u, "mae_v": mae_v, "mae_combined": 0.5 * (mae_u + mae_v),
        "bulk_ssr": spread_skill_ratio(np.concatenate([sp_u[~ext_u], sp_v[~ext_v]]),
                                       np.concatenate([err_u[~ext_u], err_v[~ext_v]])),
        "extreme_ssr": spread_skill_ratio(np.concatenate([sp_u[ext_u], sp_v[ext_v]]),
                                          np.concatenate([err_u[ext_u], err_v[ext_v]])),
    }


def run_decouple(args) -> None:
    cached, H_valid, n_valid, truth_uv = load_inputs(args)

    print("\n--- Part 1: spread-error Spearman rank correlation vs sigma ---")
    print(f"{'sigma':>10}  {'rho_u':>8}  {'rho_v':>8}  {'rho_combined':>13}")
    for sigma in args.correlation_sweep:
        r = spread_error_rank_correlation(sigma, cached, H_valid, n_valid, truth_uv)
        print(f"{r['sigma']:>10.3f}  {r['rho_u']:>8.4f}  {r['rho_v']:>8.4f}  {r['rho_combined']:>13.4f}")

    print("\n--- Part 2: decoupled sigma_mean / sigma_spread ---")
    print(f"{'sigma_mean':>10}  {'sigma_spread':>12}  {'mae_combined':>13}  {'bulk_ssr':>10}  {'extreme_ssr':>12}")
    for pair in args.decouple_pairs:
        s_mean, s_spread = (float(v) for v in pair.split(","))
        r = evaluate_decoupled(s_mean, s_spread, cached, H_valid, n_valid, truth_uv)
        print(f"{r['sigma_mean']:>10.3f}  {r['sigma_spread']:>12.3f}  {r['mae_combined']:>13.4f}  "
              f"{r['bulk_ssr']:>10.4f}  {r['extreme_ssr']:>12.4f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def shared(q):
        q.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
        q.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
        q.add_argument("--n_samples", type=int, default=150)
        return q

    q = shared(sub.add_parser("ssr", help="Tune/report posterior spread-skill ratio"))
    q.add_argument("--sweep", type=float, nargs="*", default=None,
                   help="Report SSR at these sigmas instead of bisecting")
    q.add_argument("--lo", type=float, default=2.77, help="Bisection lower bound (known too-tight)")
    q.add_argument("--hi", type=float, default=200.0, help="Bisection upper bound (should be too-loose)")
    q.set_defaults(func=run_ssr)

    q = shared(sub.add_parser("mae", help="Posterior-mean MAE vs sigma"))
    q.add_argument("--sweep", type=float, nargs="+",
                   default=[2.77, 10, 30, 60, 92.34, 150, 250, 445])
    q.set_defaults(func=run_mae)

    q = shared(sub.add_parser("decouple", help="Spread-error rank correlation, and split sigmas"))
    q.add_argument("--correlation_sweep", type=float, nargs="+", default=[10.0, 20.0, 30.0, 60.0, 92.34])
    q.add_argument("--decouple_pairs", type=str, nargs="+", default=["20,92.34", "10,92.34", "30,92.34"],
                   help="sigma_mean,sigma_spread pairs to evaluate, e.g. 20,92.34")
    q.set_defaults(func=run_decouple)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
