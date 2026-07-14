"""Evaluate average MAE to k-nearest neighbors.

Loads a `neighbors.npy` file of shape (N, k) with neighbor indices, an ERA5
`.npy` array of shape (N, C, H, W), and a matching HR/WRF `.npy` array of the
same length. Computes mean absolute error between each image and its r-th
nearest neighbor for r=1..k, and prints the mean and standard deviation of
the MAE over all images for each neighbor rank for both ERA5 and HR images.

Example:
    python eval_neighbors_mae.py --neighbors /home/peytonli/26.6_wind/neighbors.npy \
        --lr_npy /home/peytonli/26.3_wind/SWIN/preprocessed/era5_uv_2ch.npy \
        --hr_npy /home/peytonli/26.3_wind/SWIN/preprocessed/wrf_uv.npy --subset 200

"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def find_era_default() -> Path | None:
    candidates = [
        Path("/home/peytonli/26.6_wind/data/era5_uv_2ch.npy"),
        Path("/home/peytonli/26.3_wind/SWIN/preprocessed/era5_uv_2ch.npy"),
        Path("/home/peytonli/26.3_wind/SWIN/preprocessed/era5_up_uv.npy"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_era(era_path: Path, subset: int | None = None) -> np.ndarray:
    # Use mmap_mode to avoid loading the entire array into memory.
    arr = np.load(era_path, mmap_mode="r")
    if arr.ndim == 2:
        # flattened (N, features)
        return arr
    if arr.ndim == 4:
        # return the mmap array itself; caller can index subset while other indices
        # (neighbors) may reference the full dataset.
        return arr
    raise ValueError(f"Unsupported ERA array shape: {arr.shape}")


def load_hr(hr_path: Path) -> np.ndarray:
    """Load HR/WRF array with memory mapping."""
    arr = np.load(hr_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Unsupported HR array shape: {arr.shape}")
    return arr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--neighbors", type=Path, default=Path("/home/peytonli/26.6_wind/data/neighbors.npy"))
    p.add_argument("--lr_npy", type=Path, default=Path("/home/peytonli/26.6_wind/data/era5_uv_2ch.npy"))
    p.add_argument("--hr_npy", type=Path, default=Path("/home/peytonli/26.6_wind/data/wrf_uv.npy"))
    p.add_argument("--subset", type=int, default=0, help="If >0, evaluate only first SUBSET samples")
    args = p.parse_args(argv)

    if not args.neighbors.exists():
        raise FileNotFoundError(f"Neighbors file not found: {args.neighbors}")
    neighbors = np.load(args.neighbors)
    if neighbors.ndim != 2:
        raise ValueError("Neighbors array must be 2D (N, k)")
    N, k = neighbors.shape

    era_path = args.lr_npy
    if era_path is None:
        era_path = find_era_default()
        if era_path is None:
            raise FileNotFoundError("No ERA5 .npy found; provide --era_npy")
    else:
        era_path = Path(era_path)
        if not era_path.exists():
            raise FileNotFoundError(f"ERA file not found: {era_path}")

    if not args.hr_npy.exists():
        raise FileNotFoundError(f"HR file not found: {args.hr_npy}")

    print(f"Loading ERA data from {era_path}")
    era = load_era(era_path, subset=args.subset if args.subset > 0 else None)
    print(f"Loading HR data from {args.hr_npy}")
    hr = load_hr(args.hr_npy)

    # If era is flattened (N, F) we can compute MAE directly; otherwise (N,C,H,W)
    if era.ndim == 2:
        # flattened features
        if args.subset and args.subset > 0:
            N = min(N, args.subset, hr.shape[0])
        else:
            N = min(N, hr.shape[0])
        era_mae = np.empty((N, k), dtype=np.float64)
        hr_mae = np.empty((N, k), dtype=np.float64)
        for i in range(N):
            x = era[i]
            hx = hr[i]
            for r in range(k):
                j = int(neighbors[i, r])
                y = era[j]
                hy = hr[j]
                era_mae[i, r] = np.mean(np.abs(x - y))
                hr_mae[i, r] = np.mean(np.abs(hx - hy))

    else:
        # era shape (N, C, H, W)
        N = min(era.shape[0], neighbors.shape[0], hr.shape[0])
        if args.subset and args.subset > 0:
            N = min(N, args.subset)

        era_mae = np.empty((N, k), dtype=np.float64)
        hr_mae = np.empty((N, k), dtype=np.float64)
        for r in range(k):
            for i in range(N):
                j = int(neighbors[i, r])
                x = era[i]
                y = era[j]
                hx = hr[i]
                hy = hr[j]
                era_mae[i, r] = np.mean(np.abs(x - y))
                hr_mae[i, r] = np.mean(np.abs(hx - hy))

    # Print mean +/- std MAE per rank
    mae_mean, mae_std = era_mae.mean(axis=0), era_mae.std(axis=0)
    hr_mae_mean, hr_mae_std = hr_mae.mean(axis=0), hr_mae.std(axis=0)
    for r in range(k):
        print(
            f"Rank {r+1}: ERA5 MAE = {mae_mean[r]:.6f} +/- {mae_std[r]:.6f} | "
            f"HR MAE = {hr_mae_mean[r]:.6f} +/- {hr_mae_std[r]:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
