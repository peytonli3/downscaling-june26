"""Is the analog ensemble's spread an honest estimate of its own error?

Treats each sample's k nearest neighbors' WRF fields as ensemble members, their
mean as the prediction, and asks whether the spread of those members matches the
error of that mean. Following Fortin et al. (2014, "Why Should Ensemble Spread
Match the RMSE of the Ensemble Mean?"), pooled over samples and grid points:

    skill  = RMSE(ensemble_mean, truth)
    spread = sqrt( (k+1)/k * mean(unbiased variance across the k members) )
    ratio  = spread / skill

The (k+1)/k correction accounts for finite ensemble size; for a perfectly
calibrated ensemble (truth statistically exchangeable with a member) ratio == 1.
Reported per channel (u, v) and combined.

Also reports a RANK HISTOGRAM: where the truth falls among its k neighbors at
each grid point. A flat histogram means calibrated; U-shaped means the truth
lands outside the neighbor spread too often (underdispersive); dome-shaped means
the spread is too wide.

`--pct P` restricts every statistic to pixels whose |value| is above the P-th
percentile of that channel. Calibration in the bulk says nothing about
calibration in the tail, and the tail is what the downstream model is being
asked to get right -- so `--pct 95` is the interesting run, not the default one.

Usage (paths default to data/, see scripts/paths.py):
    python neighbor_spread_skill.py --k 36
    python neighbor_spread_skill.py --k 36 --pct 95 --show_histogram
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import chisquare

from _common import (
    add_data_args, channel_names, load_field, load_neighbors, n_queries, progress,
)


def extreme_thresholds(hr: np.ndarray, n: int, pct: float) -> np.ndarray:
    """Per-channel |value| threshold marking the top (100-pct)% of evaluated pixels."""
    return np.array([
        np.percentile(np.abs(np.asarray(hr[:n, c], dtype=np.float64)), pct)
        for c in range(hr.shape[1])
    ], dtype=np.float64)


def accumulate(hr: np.ndarray, neighbors: np.ndarray, k: int, n: int,
               thresholds: np.ndarray | None):
    """Per-channel sums of squared error, member variance, pixel counts, and rank histogram.

    `thresholds=None` pools every pixel; otherwise only pixels at or above the
    per-channel threshold contribute. Returns (sum_sq_err, sum_var, n_px, rank_hist)
    with shapes (C,), (C,), (C,), (C, k+1).
    """
    C = hr.shape[1]
    sum_sq_err = np.zeros(C, dtype=np.float64)
    sum_var = np.zeros(C, dtype=np.float64)
    n_px = np.zeros(C, dtype=np.int64)
    rank_hist = np.zeros((C, k + 1), dtype=np.int64)

    for i in range(n):
        x = np.asarray(hr[i], dtype=np.float64)                       # (C, H, W)
        members = np.asarray(hr[neighbors[i, :k]], dtype=np.float64)  # (k, C, H, W)
        sq_err = (members.mean(axis=0) - x) ** 2
        var = members.var(axis=0, ddof=1)
        rank = (members < x[None, ...]).sum(axis=0) + 1               # (C, H, W) in [1, k+1]

        for c in range(C):
            mask = slice(None) if thresholds is None else (np.abs(x[c]) >= thresholds[c])
            sum_sq_err[c] += sq_err[c][mask].sum()
            sum_var[c] += var[c][mask].sum()
            sel = rank[c][mask]
            n_px[c] += sel.size
            rank_hist[c] += np.bincount(sel.ravel(), minlength=k + 2)[1:k + 2]
        progress(i, n)

    return sum_sq_err, sum_var, n_px, rank_hist


def describe_flatness(hist: np.ndarray) -> str:
    """Chi-square goodness-of-fit against a uniform rank histogram, plus a shape call."""
    nbins = len(hist)
    total = hist.sum()
    chi2, p = chisquare(hist, f_exp=np.full(nbins, total / nbins))

    third = max(1, nbins // 3)
    low = hist[:third].sum() / third
    mid = hist[third:nbins - third].sum() / max(1, nbins - 2 * third)
    high = hist[nbins - third:].sum() / third
    edge_avg = (low + high) / 2

    if edge_avg > 1.1 * mid:
        shape = "U-shaped (underdispersive: truth often falls outside the neighbor spread)"
    elif mid > 1.1 * edge_avg:
        shape = "dome-shaped (overdispersive: truth often falls near the middle of the spread)"
    else:
        shape = "relatively flat (well-calibrated)"

    verdict = "consistent with flat/uniform" if p > 0.05 else "NOT flat (deviates significantly from uniform)"
    return f"chi2={chi2:.1f} (df={nbins - 1}), p={p:.4g} -> {verdict}; shape looks {shape}"


def print_ascii_histogram(hist: np.ndarray, width: int = 40) -> None:
    max_count = hist.max()
    for r, count in enumerate(hist, start=1):
        bar = "#" * (int(round(width * count / max_count)) if max_count > 0 else 0)
        print(f"  rank {r:>3}: {count:>8d} {bar}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--k", type=int, default=36, help="Neighbors treated as the ensemble")
    p.add_argument("--pct", type=float, default=None,
                   help="Restrict to pixels above this percentile of |value| (e.g. 95 = top 5%%). "
                        "Omit to pool every pixel.")
    p.add_argument("--channel_names", type=str, nargs="*", default=["u", "v"])
    p.add_argument("--show_histogram", action="store_true", help="Print the ASCII rank histogram")
    args = p.parse_args(argv)

    neighbors = load_neighbors(args.neighbors, min_k=args.k)
    hr = load_field(args.hr_npy, "HR")
    n = n_queries(args.subset, hr, neighbors)
    C = hr.shape[1]
    names = channel_names(args.channel_names, C)

    thresholds = None
    if args.pct is not None:
        print(f"Per-channel |value| thresholds for the top {100 - args.pct:.1f}% of pixels...")
        thresholds = extreme_thresholds(hr, n, args.pct)
        for c in range(C):
            print(f"  {names[c]}: threshold = {thresholds[c]:.6f}")

    scope = "extreme pixels only" if thresholds is not None else "all pixels"
    print(f"Spread-skill for k={args.k} over {n} samples ({scope})...")
    sum_sq_err, sum_var, n_px, rank_hist = accumulate(hr, neighbors, args.k, n, thresholds)

    correction = (args.k + 1) / args.k
    skill = np.sqrt(sum_sq_err / n_px)
    spread = np.sqrt(correction * sum_var / n_px)

    print(f"\n{'channel':>8}  {'n_pixels':>12}  {'skill (RMSE)':>14}  {'spread':>14}  {'spread/skill':>14}")
    for c in range(C):
        print(f"{names[c]:>8}  {n_px[c]:>12d}  {skill[c]:>14.6f}  {spread[c]:>14.6f}  "
              f"{spread[c] / skill[c]:>14.6f}")

    total_n = n_px.sum()
    skill_all = np.sqrt(sum_sq_err.sum() / total_n)
    spread_all = np.sqrt(correction * sum_var.sum() / total_n)
    print(f"{'combined':>8}  {total_n:>12d}  {skill_all:>14.6f}  {spread_all:>14.6f}  "
          f"{spread_all / skill_all:>14.6f}")

    print(f"\nRank histogram flatness ({scope}):")
    for c in range(C):
        print(f"  {names[c]}: {describe_flatness(rank_hist[c])}")
        if args.show_histogram:
            print_ascii_histogram(rank_hist[c])
    combined = rank_hist.sum(axis=0)
    print(f"  combined: {describe_flatness(combined)}")
    if args.show_histogram:
        print_ascii_histogram(combined)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
