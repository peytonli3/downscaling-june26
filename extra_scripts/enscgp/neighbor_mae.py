"""How well do a sample's nearest analogs reconstruct it? (MAE, four framings)

The analog ensemble is the EnsCGP prior: sample i's prediction is built from its
nearest neighbors in ERA5 space. These four modes bound what that prior can do
before any conditioning, on both ERA5 and WRF fields.

Modes
-----
rank      MAE between each sample and its r-th nearest neighbor, r = 1..k
          individually. Shows how fast similarity decays with rank.
mean      MAE between each sample and the mean of its top-k neighbors. One k.
sweep-k   `mean`, swept over k, on one sample from each of N distinct events.
          This is the mode that picks k -- the others describe, this one decides.
baseline  MAE between random samples and ALL other samples (no neighbor
          structure at all). The floor the other three must beat to show the
          analog ensemble is doing anything.

Usage (paths default to data/, see scripts/paths.py):
    python neighbor_mae.py rank --subset 200
    python neighbor_mae.py mean --top_k 60
    python neighbor_mae.py sweep-k --k_min 10 --k_max 60
    python neighbor_mae.py baseline --n_samples 50
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from _common import (
    add_data_args, load_event_ids, load_field, load_neighbors,
    n_queries, pick_one_sample_per_event, progress,
)


# --------------------------------------------------------------------------------------
# rank: MAE to the r-th nearest neighbor, for each r
# --------------------------------------------------------------------------------------
def mae_by_rank(data: np.ndarray, neighbors: np.ndarray, n: int, k: int) -> np.ndarray:
    """(n, k) MAE between each sample and its r-th nearest neighbor."""
    mae = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        x = np.asarray(data[i], dtype=np.float64)
        for r in range(k):
            y = np.asarray(data[int(neighbors[i, r])], dtype=np.float64)
            mae[i, r] = np.mean(np.abs(x - y))
        progress(i, n)
    return mae


def run_rank(args) -> None:
    neighbors = load_neighbors(args.neighbors)
    era = load_field(args.era_npy, "ERA5")
    hr = load_field(args.hr_npy, "HR")
    n = n_queries(args.subset, era, hr, neighbors)
    k = neighbors.shape[1]

    print(f"MAE to each of the {k} nearest neighbors, over {n} samples...")
    era_mae = mae_by_rank(era, neighbors, n, k)
    hr_mae = mae_by_rank(hr, neighbors, n, k)

    print(f"\n{'rank':>5}  {'ERA5 mean':>12}  {'ERA5 std':>12}  {'HR mean':>12}  {'HR std':>12}")
    for r in range(k):
        print(f"{r + 1:>5}  {era_mae[:, r].mean():>12.6f}  {era_mae[:, r].std():>12.6f}  "
              f"{hr_mae[:, r].mean():>12.6f}  {hr_mae[:, r].std():>12.6f}")


# --------------------------------------------------------------------------------------
# mean: MAE to the top-k neighbor mean
# --------------------------------------------------------------------------------------
def mae_to_neighbor_mean(data: np.ndarray, neighbors: np.ndarray, n: int, top_k: int) -> np.ndarray:
    """(n,) MAE between each sample and the pixel-wise mean of its top_k neighbors."""
    mae = np.empty(n, dtype=np.float64)
    for i in range(n):
        members = np.asarray(data[neighbors[i, :top_k]], dtype=np.float64)
        mae[i] = np.mean(np.abs(np.asarray(data[i], dtype=np.float64) - members.mean(axis=0)))
        progress(i, n)
    return mae


def run_mean(args) -> None:
    neighbors = load_neighbors(args.neighbors, min_k=args.top_k)
    era = load_field(args.era_npy, "ERA5")
    hr = load_field(args.hr_npy, "HR")
    n = n_queries(args.subset, era, hr, neighbors)

    print(f"MAE to the top-{args.top_k} neighbor mean, over {n} samples...")
    era_mae = mae_to_neighbor_mean(era, neighbors, n, args.top_k)
    hr_mae = mae_to_neighbor_mean(hr, neighbors, n, args.top_k)
    print(f"ERA5: mean MAE = {era_mae.mean():.6f} +/- {era_mae.std():.6f}")
    print(f"HR:   mean MAE = {hr_mae.mean():.6f} +/- {hr_mae.std():.6f}")


# --------------------------------------------------------------------------------------
# sweep-k: `mean` as a function of k, on one sample per event
# --------------------------------------------------------------------------------------
def mae_to_neighbor_mean_by_k(
    data: np.ndarray, sample_idx: np.ndarray, neighbors: np.ndarray, k_min: int, k_max: int
) -> np.ndarray:
    """(len(sample_idx), k_max-k_min+1) MAE to the k-neighbor mean, for each k.

    The mean is accumulated incrementally (running sum / k), so each additional k
    costs one neighbor read instead of re-averaging the whole set from scratch.
    """
    mae = np.empty((len(sample_idx), k_max - k_min + 1), dtype=np.float64)
    for row, i in enumerate(sample_idx):
        x = np.asarray(data[i], dtype=np.float64)
        running_sum = np.zeros_like(x)
        for r in range(k_max):
            running_sum += np.asarray(data[int(neighbors[i, r])], dtype=np.float64)
            k = r + 1
            if k >= k_min:
                mae[row, k - k_min] = np.mean(np.abs(x - running_sum / k))
    return mae


def run_sweep_k(args) -> None:
    neighbors = load_neighbors(args.neighbors, min_k=args.k_max)
    era = load_field(args.era_npy, "ERA5")
    hr = load_field(args.hr_npy, "HR")
    event_id = load_event_ids(args.event_csv)

    sample_idx = pick_one_sample_per_event(event_id, args.n_samples, args.seed)
    print(f"Picked {len(sample_idx)} samples from {len(sample_idx)} distinct events")
    print(f"Sweeping k = {args.k_min}..{args.k_max}...")
    era_mae = mae_to_neighbor_mean_by_k(era, sample_idx, neighbors, args.k_min, args.k_max)
    hr_mae = mae_to_neighbor_mean_by_k(hr, sample_idx, neighbors, args.k_min, args.k_max)

    print(f"\n{'k':>4}  {'ERA5 mean':>12}  {'ERA5 std':>12}  {'HR mean':>12}  {'HR std':>12}")
    for k in range(args.k_min, args.k_max + 1):
        c = k - args.k_min
        print(f"{k:>4}  {era_mae[:, c].mean():>12.6f}  {era_mae[:, c].std():>12.6f}  "
              f"{hr_mae[:, c].mean():>12.6f}  {hr_mae[:, c].std():>12.6f}")


# --------------------------------------------------------------------------------------
# baseline: all-pairs MAE, ignoring neighbor structure entirely
# --------------------------------------------------------------------------------------
def run_baseline(args) -> None:
    hr = load_field(args.hr_npy, "HR")
    n_total = hr.shape[0]

    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(n_total), min(args.n_samples, n_total))
    print(f"All-pairs MAE for {len(sample_indices)} random samples against all {n_total}...")

    # Chunked over the full dataset: the difference array for one query against every
    # sample at once would be several GB, and this runs off a memory-mapped file.
    maes = []
    for q, i in enumerate(sample_indices):
        x = np.asarray(hr[i], dtype=np.float32)
        total = 0.0
        for start in range(0, n_total, args.chunk_size):
            block = np.asarray(hr[start:start + args.chunk_size], dtype=np.float32)
            total += np.abs(block - x[None, ...]).mean(axis=(1, 2, 3)).sum()
        maes.append((total - 0.0) / (n_total - 1))  # the self term contributes exactly 0
        progress(q, len(sample_indices), every=10)
    maes = np.asarray(maes)
    print(f"Unconditional baseline MAE = {maes.mean():.6f} +/- {maes.std():.6f}")
    print("(any neighbor-based mode above must beat this to be doing real work)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    q = sub.add_parser("rank", help="MAE to the r-th nearest neighbor, for each r")
    add_data_args(q, era=True)
    q.set_defaults(func=run_rank)

    q = sub.add_parser("mean", help="MAE to the top-k neighbor mean")
    add_data_args(q, era=True)
    q.add_argument("--top_k", type=int, default=60)
    q.set_defaults(func=run_mean)

    q = sub.add_parser("sweep-k", help="MAE to the k-neighbor mean, swept over k")
    add_data_args(q, era=True, events=True)
    q.add_argument("--n_samples", type=int, default=200, help="Distinct events to draw one sample from each")
    q.add_argument("--k_min", type=int, default=10)
    q.add_argument("--k_max", type=int, default=60)
    q.add_argument("--seed", type=int, default=42)
    q.set_defaults(func=run_sweep_k)

    q = sub.add_parser("baseline", help="All-pairs MAE with no neighbor structure")
    add_data_args(q)
    q.add_argument("--n_samples", type=int, default=50)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--chunk_size", type=int, default=256)
    q.set_defaults(func=run_baseline)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
