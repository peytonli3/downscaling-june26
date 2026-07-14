"""eval_neighbors_mae_sweep_k.py

Pick 200 samples, one from each of 200 distinct storm events. For each k
swept from 10 to 60, average each sample's k closest neighbors (from a
precomputed `neighbors.npy`, see `scripts/nearest_neighbors.py`) into a
single "neighbor-mean" image (as in `eval_neighbor_mean_mae.py`), then
compute the MAE between that mean and the sample itself. Reports the mean
and standard deviation of these per-sample errors across the 200 samples,
for both ERA5 and HR/WRF, at each k.

Event ids come from `data/sample_event_ids.csv` (sample_index -> event_id),
which is index-aligned with `era5_uv_2ch.npy` / `wrf_uv.npy`.

Example:
    python eval_neighbors_mae_sweep_k.py

"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array (N, C, H, W), got {arr.shape} from {path}")
    return arr


def load_event_ids(csv_path: Path) -> np.ndarray:
    """Load the sample_index -> event_id mapping written by split_temporal_dataset.py's sibling script."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    event_id = np.empty(len(rows), dtype=np.int64)
    for row in rows:
        event_id[int(row["sample_index"])] = int(row["event_id"])
    return event_id


def pick_one_sample_per_event(event_id: np.ndarray, n_events: int, seed: int) -> np.ndarray:
    """Pick `n_events` distinct events and return one sample index from each."""
    rows_by_event: dict[int, list[int]] = {}
    for idx, eid in enumerate(event_id):
        rows_by_event.setdefault(int(eid), []).append(idx)

    rng = random.Random(seed)
    chosen_events = rng.sample(sorted(rows_by_event), n_events)
    return np.array([rows_by_event[eid][0] for eid in chosen_events], dtype=np.int64)


def mae_to_neighbor_mean_by_k(
    data: np.ndarray, sample_idx: np.ndarray, neighbors: np.ndarray, k_min: int, k_max: int
) -> np.ndarray:
    """MAE between each sample and the pixel-wise mean of its k closest neighbors, for k in [k_min, k_max].

    Builds the mean image incrementally (running sum / k) so each k only costs one extra
    neighbor read rather than re-averaging from scratch. Shape (len(sample_idx), k_max - k_min + 1).
    """
    mae = np.empty((len(sample_idx), k_max - k_min + 1), dtype=np.float64)
    for row, i in enumerate(sample_idx):
        x = np.asarray(data[i], dtype=np.float64)
        running_sum = np.zeros_like(x)
        for r in range(k_max):
            j = int(neighbors[i, r])
            running_sum += np.asarray(data[j], dtype=np.float64)
            k = r + 1
            if k >= k_min:
                mean_img = running_sum / k
                mae[row, k - k_min] = np.mean(np.abs(x - mean_img))
    return mae


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Sweep k and report MAE between each sample and the mean of its k nearest neighbors, "
        "for 200 samples from 200 distinct events"
    )
    p.add_argument("--event_csv", type=Path, default=Path("/home/peytonli/26.6_wind/data/sample_event_ids.csv"))
    p.add_argument("--neighbors", type=Path, default=Path("/home/peytonli/26.6_wind/data/neighbors.npy"))
    p.add_argument("--era_npy", type=Path, default=Path("/home/peytonli/26.6_wind/data/era5_uv_2ch_bicubic.npy"))
    p.add_argument("--hr_npy", type=Path, default=Path("/home/peytonli/26.6_wind/data/wrf_uv.npy"))
    p.add_argument("--n_samples", type=int, default=200, help="Number of distinct events/samples to evaluate")
    p.add_argument("--k_min", type=int, default=10)
    p.add_argument("--k_max", type=int, default=60)
    p.add_argument("--seed", type=int, default=42, help="Random seed used to pick events/samples")
    args = p.parse_args(argv)

    if not args.neighbors.exists():
        raise FileNotFoundError(f"Neighbors file not found: {args.neighbors}")
    neighbors = np.load(args.neighbors)
    if neighbors.ndim != 2:
        raise ValueError("Neighbors array must be 2D (N, k)")
    if neighbors.shape[1] < args.k_max:
        raise ValueError(f"Neighbors file only has k={neighbors.shape[1]} columns, need >= {args.k_max}")

    print(f"Loading ERA5 from {args.era_npy}")
    era = load_npy(args.era_npy)
    print(f"Loading HR from {args.hr_npy}")
    hr = load_npy(args.hr_npy)

    print(f"Loading event ids from {args.event_csv}")
    event_id = load_event_ids(args.event_csv)

    sample_idx = pick_one_sample_per_event(event_id, args.n_samples, args.seed)
    print(f"Picked {len(sample_idx)} samples from {len(sample_idx)} distinct events")

    print(f"Computing MAE to neighbor mean for k={args.k_min}..{args.k_max}...")
    era_mae = mae_to_neighbor_mean_by_k(era, sample_idx, neighbors, args.k_min, args.k_max)
    hr_mae = mae_to_neighbor_mean_by_k(hr, sample_idx, neighbors, args.k_min, args.k_max)

    print(f"{'k':>4}  {'ERA5 mean':>12}  {'ERA5 std':>12}  {'HR mean':>12}  {'HR std':>12}")
    for k in range(args.k_min, args.k_max + 1):
        idx = k - args.k_min
        era_col, hr_col = era_mae[:, idx], hr_mae[:, idx]
        print(f"{k:>4}  {era_col.mean():>12.6f}  {era_col.std():>12.6f}  {hr_col.mean():>12.6f}  {hr_col.std():>12.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
