"""eval_neighbor_mean_spread_skill.py

Spread-skill ratio for k=36 nearest-neighbor mean predictions of HR/WRF
images, treating each sample's k closest neighbors (from a precomputed
`neighbors.npy`, see `scripts/nearest_neighbors.py`) as an ensemble whose
mean is the "prediction" for that sample.

Following Fortin et al. (2014, "Why Should Ensemble Spread Match the RMSE
of the Ensemble Mean?"), for each grid point we treat the k neighbor HR
values as ensemble members and compute, pooled over all samples and grid
points:

    skill   = RMSE(ensemble_mean, truth)
    spread  = sqrt( (k+1)/k * mean(unbiased variance across the k members) )
    ratio   = spread / skill

The (k+1)/k correction accounts for the finite ensemble size; under a
perfectly calibrated ensemble (truth statistically exchangeable with a
member), ratio == 1. Reported per-channel (u, v) and combined across both.

Example:
    python eval_neighbor_mean_spread_skill.py \
        --neighbors /home/peytonli/26.6_wind/data/neighbors.npy \
        --hr_npy /home/peytonli/26.6_wind/data/wrf_uv.npy --k 36

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_hr(hr_path: Path) -> np.ndarray:
    arr = np.load(hr_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array (N, C, H, W), got {arr.shape} from {hr_path}")
    return arr


def spread_skill_sums(
    hr: np.ndarray, neighbors: np.ndarray, k: int, n_queries: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulate per-channel sums of squared error and unbiased variance.

    For each of the first `n_queries` samples, the ensemble is its k nearest
    neighbors' HR fields (looked up in the full `hr` array, since neighbor
    indices may point anywhere in the dataset); the ensemble mean is the
    prediction. Returns (sum_sq_err, sum_var, n_per_channel) where the first
    two have shape (C,) and n_per_channel = n_queries * H * W is the number
    of grid points pooled into each channel's mean.
    """
    C, H, W = hr.shape[1], hr.shape[2], hr.shape[3]
    sum_sq_err = np.zeros(C, dtype=np.float64)
    sum_var = np.zeros(C, dtype=np.float64)
    for i in range(n_queries):
        x = np.asarray(hr[i], dtype=np.float64)
        members = np.asarray(hr[neighbors[i, :k]], dtype=np.float64)
        ens_mean = members.mean(axis=0)
        sq_err = (ens_mean - x) ** 2
        var = members.var(axis=0, ddof=1)
        sum_sq_err += sq_err.sum(axis=(1, 2))
        sum_var += var.sum(axis=(1, 2))
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{n_queries} samples")
    return sum_sq_err, sum_var, n_queries * H * W


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Spread-skill ratio for k-neighbor mean predictions of HR images")
    p.add_argument("--neighbors", type=Path, default=Path("/home/peytonli/26.6_wind/data/neighbors.npy"))
    p.add_argument("--hr_npy", type=Path, default=Path("/home/peytonli/26.6_wind/data/wrf_uv.npy"))
    p.add_argument("--k", type=int, default=36, help="Number of nearest neighbors to use as the ensemble")
    p.add_argument("--subset", type=int, default=0, help="If >0, evaluate only first SUBSET samples")
    p.add_argument(
        "--channel_names", type=str, nargs="*", default=["u", "v"], help="Names for each channel, for display"
    )
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

    print(f"Computing spread-skill ratio for k={args.k} neighbor-mean predictions over {N} samples...")
    sum_sq_err, sum_var, n_per_channel = spread_skill_sums(hr, neighbors, args.k, N)

    correction = (args.k + 1) / args.k

    mean_sq_err = sum_sq_err / n_per_channel
    mean_var = sum_var / n_per_channel
    skill = np.sqrt(mean_sq_err)
    spread = np.sqrt(correction * mean_var)
    ratio = spread / skill

    print(f"\n{'channel':>8}  {'skill (RMSE)':>14}  {'spread':>14}  {'spread/skill':>14}")
    for c in range(C):
        print(f"{channel_names[c]:>8}  {skill[c]:>14.6f}  {spread[c]:>14.6f}  {ratio[c]:>14.6f}")

    total_sq_err = sum_sq_err.sum()
    total_var = sum_var.sum()
    n_combined = n_per_channel * C
    skill_combined = np.sqrt(total_sq_err / n_combined)
    spread_combined = np.sqrt(correction * total_var / n_combined)
    ratio_combined = spread_combined / skill_combined
    print(f"{'combined':>8}  {skill_combined:>14.6f}  {spread_combined:>14.6f}  {ratio_combined:>14.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
