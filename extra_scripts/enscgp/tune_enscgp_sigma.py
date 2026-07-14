"""Tune the EnsCGP ERA5 observation-noise sigma so the POSTERIOR's bulk spread-skill
ratio lands near 1.0.

Context: sigma=2.77 (enscgp_train.py's default) is not arbitrary -- it's
sqrt(Var(H @ WRF - ERA5)), the genuine empirically-measured representativeness/
retrieval error between coarsened WRF and ERA5 (see extra_scripts/
compare_WRF_coarse_ERA5.py, which reports Var=7.65, sqrt=2.766). It checks out as a
real physical observation-noise estimate, not a bug.

But the resulting POSTERIOR (data/enscgp_posterior.npy, built with this sigma) is
drastically overconfident overall: bulk and extreme spread-skill ratio both ~0.13,
not the ~1.01/~0.70 profile originally assumed (that figure turned out to describe
the *unconditioned* k=36 nearest-neighbor ensemble, see
extra_scripts/eval_neighbor_mean_spread_skill.py -- a different, much better-behaved
quantity). The likely mechanism: the neighbor ensemble's disagreement, as SEEN
THROUGH the coarsening operator H, is apparently much smaller than its disagreement
at full WRF resolution (the neighbors agree strongly at the synoptic/smoothed scale
even though they differ a lot pixel-to-pixel) -- so the ERA5 observation looks far
more informative to the linear-Gaussian EnsCGP update than it should, and the
posterior over-collapses even at a physically-correct sigma.

This script treats sigma as an effective tuning parameter that compensates for that
structural mismatch (a form of ensemble covariance inflation, standard practice for
underdispersive ensembles in DA) -- distinct from the empirically-measured ERA5
representativeness error, and distinct from variance_recalibration.py's job, which
is to fix any *residual* tail-specific shape mis-calibration left over after this
global/bulk correction.

Re-runs EnsCGP conditioning fresh (no disk writes; prior/observation precomputed
once and reused across all candidate sigmas, since only R_inv depends on sigma)
over a held-out sample subset, and bisects for the sigma whose resulting posterior
has bulk spread-skill ratio == 1.0, reporting what the extreme-stratum SSR comes out
to there.

Usage:
    python tune_enscgp_sigma.py --split val --n_samples 150
    python tune_enscgp_sigma.py --split val --n_samples 150 --sweep 2.77 10 30 60 100 200
"""
import argparse
from pathlib import Path

import numpy as np

from enscgp_train import (
    build_r_inv, enscgp, load_era5, load_hr, load_neighbors, load_observation,
    load_observation_operator, prior,
)
from variance_recalibration import fortin_factor, spread_skill_ratio

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def precompute_priors_and_obs(sample_indices, neighbors, wrf, era5, valid):
    """(mean, A, y) per sample -- everything that does NOT depend on sigma, computed
    once and reused for every candidate sigma in the sweep."""
    cached = []
    for idx in sample_indices:
        mean, A = prior(int(idx), neighbors, wrf)
        y = load_observation(era5, int(idx), valid)
        cached.append((mean, A, y))
    return cached


def evaluate_sigma(sigma: float, cached: list, H_valid, n_valid: int, truth_uv: np.ndarray,
                    extreme_percentile: float = 95.0, hw: int = 200) -> dict:
    """Re-run EnsCGP conditioning at this sigma for every precomputed (mean, A, y), and
    return per-channel (u, v) bulk/extreme/overall spread-skill ratio, Fortin-corrected.
    `truth_uv`: (S, 2, hw, hw) ground truth, same order as `cached`/sample_indices.

    Extreme stratification is TRUTH-based and per-channel -- |truth_c| >= the channel's
    own `extreme_percentile`-th percentile -- matching extra_scripts/
    eval_neighbor_mean_spread_skill_extreme.py exactly, so results are directly
    comparable to that script's numbers. (Not what variance_recalibration.py's
    operational maps use at apply time, since truth is unknown then -- this is a
    diagnostic comparison, not the deployed stratification rule.)
    """
    R_inv = build_r_inv(n_valid, sigma)
    n = hw * hw
    k = cached[0][1].shape[1]
    fac = np.sqrt(fortin_factor(k=k))

    sp_u_list, sp_v_list, err_u_list, err_v_list, truth_u_list, truth_v_list = [], [], [], [], [], []
    for (mean, A, y), truth in zip(cached, truth_uv):
        mean_post, A_post = enscgp(mean, A, H_valid, R_inv, y)
        mean_u, mean_v = mean_post[:n], mean_post[n:]
        A_u, A_v = A_post[:n], A_post[n:]

        sp_u_list.append(np.sqrt(np.sum(A_u * A_u, axis=1)))
        sp_v_list.append(np.sqrt(np.sum(A_v * A_v, axis=1)))
        err_u_list.append(truth[0].ravel() - mean_u)
        err_v_list.append(truth[1].ravel() - mean_v)
        truth_u_list.append(truth[0].ravel())
        truth_v_list.append(truth[1].ravel())

    sp_u, sp_v = fac * np.concatenate(sp_u_list), fac * np.concatenate(sp_v_list)
    err_u, err_v = np.concatenate(err_u_list), np.concatenate(err_v_list)
    truth_u, truth_v = np.concatenate(truth_u_list), np.concatenate(truth_v_list)

    extreme_u = np.abs(truth_u) >= np.percentile(np.abs(truth_u), extreme_percentile)
    extreme_v = np.abs(truth_v) >= np.percentile(np.abs(truth_v), extreme_percentile)

    return {
        "sigma": sigma,
        "u_bulk_ssr": spread_skill_ratio(sp_u[~extreme_u], err_u[~extreme_u]),
        "u_extreme_ssr": spread_skill_ratio(sp_u[extreme_u], err_u[extreme_u]),
        "v_bulk_ssr": spread_skill_ratio(sp_v[~extreme_v], err_v[~extreme_v]),
        "v_extreme_ssr": spread_skill_ratio(sp_v[extreme_v], err_v[extreme_v]),
        "bulk_ssr": spread_skill_ratio(
            np.concatenate([sp_u[~extreme_u], sp_v[~extreme_v]]), np.concatenate([err_u[~extreme_u], err_v[~extreme_v]])
        ),
        "extreme_ssr": spread_skill_ratio(
            np.concatenate([sp_u[extreme_u], sp_v[extreme_v]]), np.concatenate([err_u[extreme_u], err_v[extreme_v]])
        ),
    }


def find_sigma_for_bulk_ssr_one(cached, H_valid, n_valid, truth_uv, lo: float, hi: float,
                                 target: float = 1.0, tol: float = 1e-3, max_iter: int = 40) -> float:
    """Bisect for the sigma where bulk SSR == target. Assumes bulk_ssr(sigma) is monotonic
    increasing in sigma (weaker conditioning -> posterior spread closer to the
    well-calibrated, unconditioned ensemble), which is checked by the bracket assertion below."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--sweep", type=float, nargs="*", default=None,
                         help="If given, just report SSR at these sigma values (no bisection)")
    parser.add_argument("--lo", type=float, default=2.77, help="Bisection lower bound (known too-tight)")
    parser.add_argument("--hi", type=float, default=200.0, help="Bisection upper bound (should be too-loose)")
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

    if args.sweep is not None:
        print(f"\n{'sigma':>10}  {'bulk_ssr':>10}  {'extreme_ssr':>12}  {'u_bulk':>8}  {'u_extreme':>10}  {'v_bulk':>8}  {'v_extreme':>10}")
        for sigma in args.sweep:
            r = evaluate_sigma(sigma, cached, H_valid, n_valid, truth_uv)
            print(f"{r['sigma']:>10.3f}  {r['bulk_ssr']:>10.4f}  {r['extreme_ssr']:>12.4f}  "
                  f"{r['u_bulk_ssr']:>8.4f}  {r['u_extreme_ssr']:>10.4f}  {r['v_bulk_ssr']:>8.4f}  {r['v_extreme_ssr']:>10.4f}")
        return

    print(f"\nBisecting for bulk SSR == 1.0 in sigma in [{args.lo}, {args.hi}]...")
    sigma_star = find_sigma_for_bulk_ssr_one(cached, H_valid, n_valid, truth_uv, args.lo, args.hi)
    r = evaluate_sigma(sigma_star, cached, H_valid, n_valid, truth_uv)
    print(f"\nsigma* = {sigma_star:.4f}")
    print(f"  bulk SSR    = {r['bulk_ssr']:.4f}  (u={r['u_bulk_ssr']:.4f}, v={r['v_bulk_ssr']:.4f})")
    print(f"  extreme SSR = {r['extreme_ssr']:.4f}  (u={r['u_extreme_ssr']:.4f}, v={r['v_extreme_ssr']:.4f})")


if __name__ == "__main__":
    main()
