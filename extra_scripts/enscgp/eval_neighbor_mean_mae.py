"""eval_neighbor_mean_mae.py

For each sample, average its top-k nearest neighbors (from a precomputed
`neighbors.npy`, see `scripts/nearest_neighbors.py`) into a single
"neighbor-mean" image, then compute the MAE between that mean and the
sample itself. Reports the average and standard deviation of these
per-sample errors across the dataset, for both ERA5 and HR/WRF.

Example:
    python eval_neighbor_mean_mae.py \
        --neighbors /home/peytonli/26.6_wind/data/neighbors.npy \
        --top_k 60

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array (N, C, H, W), got {arr.shape} from {path}")
    return arr


def neighbor_mean_mae(data: np.ndarray, neighbors: np.ndarray, top_k: int) -> np.ndarray:
    """MAE between each sample and the pixel-wise mean of its top_k neighbors."""
    N = neighbors.shape[0]
    mae = np.empty(N, dtype=np.float64)
    for i in range(N):
        neighbor_imgs = np.asarray(data[neighbors[i, :top_k]], dtype=np.float64)
        mean_img = neighbor_imgs.mean(axis=0)
        mae[i] = np.mean(np.abs(np.asarray(data[i], dtype=np.float64) - mean_img))
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{N} samples")
    return mae


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MAE between each sample and the mean of its top-k nearest neighbors")
    p.add_argument("--neighbors", type=Path, default=Path("/home/peytonli/26.6_wind/data/neighbors.npy"))
    p.add_argument("--era_npy", type=Path, default=Path("/home/peytonli/26.6_wind/data/era5_uv_2ch.npy"))
    p.add_argument("--hr_npy", type=Path, default=Path("/home/peytonli/26.6_wind/data/wrf_uv.npy"))
    p.add_argument("--top_k", type=int, default=60, help="Number of nearest neighbors to average")
    p.add_argument("--subset", type=int, default=0, help="If >0, evaluate only first SUBSET samples")
    args = p.parse_args(argv)

    if not args.neighbors.exists():
        raise FileNotFoundError(f"Neighbors file not found: {args.neighbors}")
    neighbors = np.load(args.neighbors)
    if neighbors.ndim != 2:
        raise ValueError("Neighbors array must be 2D (N, k)")
    if neighbors.shape[1] < args.top_k:
        raise ValueError(f"Neighbors file only has k={neighbors.shape[1]} columns, need >= {args.top_k}")

    print(f"Loading ERA5 from {args.era_npy}")
    era = load_npy(args.era_npy)
    print(f"Loading HR from {args.hr_npy}")
    hr = load_npy(args.hr_npy)

    N = min(era.shape[0], hr.shape[0], neighbors.shape[0])
    if args.subset and args.subset > 0:
        N = min(N, args.subset)
    neighbors = neighbors[:N]

    print(f"Computing MAE to top-{args.top_k} neighbor mean for {N} samples...")
    era_mae = neighbor_mean_mae(era, neighbors, args.top_k)
    hr_mae = neighbor_mean_mae(hr, neighbors, args.top_k)

    print(f"ERA5: mean MAE = {era_mae.mean():.6f} +/- {era_mae.std():.6f}")
    print(f"HR:   mean MAE = {hr_mae.mean():.6f} +/- {hr_mae.std():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
