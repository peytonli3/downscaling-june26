"""eval_neighbor_mean_spread_skill_extreme.py

Like `eval_neighbor_mean_spread_skill.py`, but restricted to "extreme"
grid points (default: top 5% of pixels by |HR value|, i.e. --pct 95), and
additionally reports the verification rank histogram (Talagrand diagram)
for those pixels.

Extremity is defined per channel: a (sample, pixel) location for channel
c counts as extreme if |hr value| there is >= the channel's `pct`-th
percentile of |hr value|, computed over all evaluated samples. The
"combined" row pools whichever locations are extreme for their own
channel together.

Spread/skill follow Fortin et al. (2014): for each extreme location, the
k nearest neighbors' HR values are treated as ensemble members, their
mean is the prediction, and:

    skill  = RMSE(ensemble_mean, truth)
    spread = sqrt( (k+1)/k * mean(unbiased variance across the k members) )
    ratio  = spread / skill

Rank histogram: for each extreme location, find the rank of the true HR
value among itself and its k neighbor HR values (k+1 possible ranks). A
flat histogram indicates a well-calibrated ensemble; U-shaped means the
ensemble is too narrow (truth often falls outside the neighbor spread,
i.e. underdispersive); dome-shaped means it's too wide (overdispersive).

Example:
    python eval_neighbor_mean_spread_skill_extreme.py \
        --neighbors /home/peytonli/26.6_wind/data/neighbors.npy \
        --hr_npy /home/peytonli/26.6_wind/data/wrf_uv.npy --k 36 --pct 95

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chisquare


def load_hr(hr_path: Path) -> np.ndarray:
    arr = np.load(hr_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array (N, C, H, W), got {arr.shape} from {hr_path}")
    return arr


def extreme_thresholds(hr: np.ndarray, n_queries: int, pct: float) -> np.ndarray:
    """Per-channel threshold on |value| marking the top (100-pct)% of evaluated pixels."""
    C = hr.shape[1]
    thresholds = np.empty(C, dtype=np.float64)
    for c in range(C):
        channel_vals = np.abs(np.asarray(hr[:n_queries, c], dtype=np.float64))
        thresholds[c] = np.percentile(channel_vals, pct)
    return thresholds


def accumulate(
    hr: np.ndarray, neighbors: np.ndarray, k: int, n_queries: int, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate per-channel extreme-pixel error/variance sums and rank histograms.

    Returns (sum_sq_err, sum_var, n_extreme, rank_hist): the first three have
    shape (C,), rank_hist has shape (C, k+1) with rank_hist[c, r] counting
    extreme locations in channel c where the truth ranked r+1 among itself
    and its k neighbors (1 = smallest).
    """
    C = hr.shape[1]
    sum_sq_err = np.zeros(C, dtype=np.float64)
    sum_var = np.zeros(C, dtype=np.float64)
    n_extreme = np.zeros(C, dtype=np.int64)
    rank_hist = np.zeros((C, k + 1), dtype=np.int64)

    for i in range(n_queries):
        x = np.asarray(hr[i], dtype=np.float64)  # (C, H, W)
        members = np.asarray(hr[neighbors[i, :k]], dtype=np.float64)  # (k, C, H, W)
        ens_mean = members.mean(axis=0)
        sq_err = (ens_mean - x) ** 2
        var = members.var(axis=0, ddof=1)
        rank = (members < x[None, ...]).sum(axis=0) + 1  # (C, H, W), in [1, k+1]

        for c in range(C):
            mask = np.abs(x[c]) >= thresholds[c]
            sum_sq_err[c] += sq_err[c][mask].sum()
            sum_var[c] += var[c][mask].sum()
            n_extreme[c] += int(mask.sum())
            rank_hist[c] += np.bincount(rank[c][mask], minlength=k + 2)[1 : k + 2]

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{n_queries} samples")

    return sum_sq_err, sum_var, n_extreme, rank_hist


def describe_flatness(hist: np.ndarray) -> str:
    """Chi-square goodness-of-fit against a uniform rank histogram, plus a shape call."""
    nbins = len(hist)
    total = hist.sum()
    expected = np.full(nbins, total / nbins)
    chi2, p = chisquare(hist, f_exp=expected)

    third = max(1, nbins // 3)
    low = hist[:third].sum() / third
    mid = hist[third : nbins - third].sum() / max(1, nbins - 2 * third)
    high = hist[nbins - third :].sum() / third
    edge_avg = (low + high) / 2

    if edge_avg > 1.1 * mid:
        shape = "U-shaped (underdispersive: truth often falls outside the neighbor spread)"
    elif mid > 1.1 * edge_avg:
        shape = "dome-shaped (overdispersive: truth often falls near the middle of the neighbor spread)"
    else:
        shape = "relatively flat (well-calibrated)"

    flat_verdict = "consistent with flat/uniform" if p > 0.05 else "NOT flat (deviates significantly from uniform)"
    return f"chi2={chi2:.1f} (df={nbins - 1}), p={p:.4g} -> {flat_verdict}; shape looks {shape}"


def print_ascii_histogram(hist: np.ndarray, width: int = 40) -> None:
    max_count = hist.max()
    for r, count in enumerate(hist, start=1):
        bar_len = int(round(width * count / max_count)) if max_count > 0 else 0
        print(f"  rank {r:>3}: {count:>8d} {'#' * bar_len}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Spread-skill ratio and rank histogram for k-neighbor mean predictions, restricted to extreme pixels"
    )
    p.add_argument("--neighbors", type=Path, default=Path("/home/peytonli/26.6_wind/data/neighbors.npy"))
    p.add_argument("--hr_npy", type=Path, default=Path("/home/peytonli/26.6_wind/data/wrf_uv.npy"))
    p.add_argument("--k", type=int, default=36, help="Number of nearest neighbors to use as the ensemble")
    p.add_argument("--pct", type=float, default=95.0, help="Percentile of |HR value| defining 'extreme' (95 = top 5%)")
    p.add_argument("--subset", type=int, default=0, help="If >0, evaluate only first SUBSET samples")
    p.add_argument(
        "--channel_names", type=str, nargs="*", default=["u", "v"], help="Names for each channel, for display"
    )
    p.add_argument("--show_histogram", action="store_true", help="Print an ASCII rank histogram per channel/combined")
    args = p.parse_args(argv)

    if not args.neighbors.exists():
        raise FileNotFoundError(f"Neighbors file not found: {args.neighbors}")
    neighbors = np.load(args.neighbors)
    if neighbors.ndim != 2:
        raise ValueError("Neighbors array must be 2D (N, k)")
    if neighbors.shape[1] < args.k:
        raise ValueError(f"Neighbors file only has k={neighbors.shape[1]} columns, need >= {args.k}")

    if not args.hr_npy.exists():
        raise FileNotFoundError(f"HR file not found: {args.hr_npy}")

    print(f"Loading HR data from {args.hr_npy}")
    hr = load_hr(args.hr_npy)

    N = min(hr.shape[0], neighbors.shape[0])
    if args.subset and args.subset > 0:
        N = min(N, args.subset)

    C = hr.shape[1]
    channel_names = args.channel_names if len(args.channel_names) == C else [f"ch{c}" for c in range(C)]

    print(f"Computing per-channel |value| thresholds for the top {100 - args.pct:.1f}% of pixels...")
    thresholds = extreme_thresholds(hr, N, args.pct)
    for c in range(C):
        print(f"  {channel_names[c]}: threshold = {thresholds[c]:.6f}")

    print(f"Computing spread-skill ratio and rank histograms for k={args.k} over {N} samples (extreme pixels only)...")
    sum_sq_err, sum_var, n_extreme, rank_hist = accumulate(hr, neighbors, args.k, N, thresholds)

    correction = (args.k + 1) / args.k

    skill = np.sqrt(sum_sq_err / n_extreme)
    spread = np.sqrt(correction * sum_var / n_extreme)
    ratio = spread / skill

    print(f"\n{'channel':>8}  {'n_extreme':>10}  {'skill (RMSE)':>14}  {'spread':>14}  {'spread/skill':>14}")
    for c in range(C):
        print(f"{channel_names[c]:>8}  {n_extreme[c]:>10d}  {skill[c]:>14.6f}  {spread[c]:>14.6f}  {ratio[c]:>14.6f}")

    total_sq_err, total_var, total_n = sum_sq_err.sum(), sum_var.sum(), n_extreme.sum()
    skill_combined = np.sqrt(total_sq_err / total_n)
    spread_combined = np.sqrt(correction * total_var / total_n)
    ratio_combined = spread_combined / skill_combined
    print(f"{'combined':>8}  {total_n:>10d}  {skill_combined:>14.6f}  {spread_combined:>14.6f}  {ratio_combined:>14.6f}")

    print("\nRank histogram flatness (extreme pixels only):")
    for c in range(C):
        print(f"  {channel_names[c]}: {describe_flatness(rank_hist[c])}")
        if args.show_histogram:
            print_ascii_histogram(rank_hist[c])
    combined_hist = rank_hist.sum(axis=0)
    print(f"  combined: {describe_flatness(combined_hist)}")
    if args.show_histogram:
        print_ascii_histogram(combined_hist)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
