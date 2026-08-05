"""Shared loaders and CLI plumbing for the analog-ensemble (neighbor) diagnostics.

Not an entry point -- import it. The scripts in this directory all operate on the
same three inputs (a `neighbors.npy` index table, an ERA5 field array, and the WRF
truth array), and every one of them used to re-implement the loading and validation
of those. That lives here now.

Conventions shared by every caller:
  * field arrays are 4D `(N, C, H, W)`, memory-mapped -- these files are tens of GB
    and no diagnostic needs more than a few samples resident at a time;
  * `neighbors[i, r]` is the index of sample i's r-th nearest neighbor (r=0 nearest),
    and may point anywhere in the dataset, so neighbor lookups always index the FULL
    array even when the query loop is restricted to a subset.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import DATA_DIR  # noqa: E402  (re-exported for callers)


def load_field(path: Path, what: str = "field") -> np.ndarray:
    """Memory-map a 4D (N, C, H, W) field array."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} file not found: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array (N, C, H, W), got {arr.shape} from {path}")
    return arr


def load_neighbors(path: Path, min_k: int | None = None) -> np.ndarray:
    """Load and validate the (N, k) neighbor index table."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Neighbors file not found: {path}")
    neighbors = np.load(path)
    if neighbors.ndim != 2:
        raise ValueError(f"Neighbors array must be 2D (N, k), got {neighbors.shape}")
    if min_k is not None and neighbors.shape[1] < min_k:
        raise ValueError(f"Neighbors file only has k={neighbors.shape[1]} columns, need >= {min_k}")
    return neighbors


def load_event_ids(csv_path: Path) -> np.ndarray:
    """sample_index -> event_id, from the CSV written by data_prep/split_temporal_dataset.py."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Event id CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    event_id = np.empty(len(rows), dtype=np.int64)
    for row in rows:
        event_id[int(row["sample_index"])] = int(row["event_id"])
    return event_id


def pick_one_sample_per_event(event_id: np.ndarray, n_events: int, seed: int) -> np.ndarray:
    """One sample index from each of `n_events` randomly chosen distinct events.

    Samples within an event are near-duplicate consecutive hours, so drawing more
    than one from the same event would understate variance across the selection.
    """
    rows_by_event: dict[int, list[int]] = {}
    for idx, eid in enumerate(event_id):
        rows_by_event.setdefault(int(eid), []).append(idx)
    rng = random.Random(seed)
    chosen = rng.sample(sorted(rows_by_event), n_events)
    return np.array([rows_by_event[eid][0] for eid in chosen], dtype=np.int64)


def n_queries(subset: int, *arrays: np.ndarray) -> int:
    """How many leading samples to iterate: the shortest input, capped by --subset."""
    n = min(a.shape[0] for a in arrays)
    return min(n, subset) if subset and subset > 0 else n


def channel_names(requested: list[str], n_channels: int) -> list[str]:
    """Display names per channel, falling back to ch0..chN if the count doesn't match."""
    return requested if len(requested) == n_channels else [f"ch{c}" for c in range(n_channels)]


def add_data_args(p: argparse.ArgumentParser, *, era: bool = False, events: bool = False) -> None:
    """Attach the input-path flags shared across these diagnostics."""
    p.add_argument("--neighbors", type=Path, default=DATA_DIR / "neighbors.npy")
    p.add_argument("--hr_npy", type=Path, default=DATA_DIR / "wrf_uv.npy")
    if era:
        # The 200x200 bicubic product, index-aligned with wrf_uv.npy so the two are
        # directly comparable pixel-for-pixel. (The pre-reorg default was a plain
        # `era5_uv_2ch.npy` that no longer exists in data/ -- see data/README.md.)
        p.add_argument("--era_npy", type=Path, default=DATA_DIR / "era5_uv_2ch_bicubic.npy")
    if events:
        p.add_argument("--event_csv", type=Path, default=DATA_DIR / "sample_event_ids.csv")
    p.add_argument("--subset", type=int, default=0, help="If >0, evaluate only the first SUBSET samples")


def progress(i: int, total: int, every: int = 1000) -> None:
    if (i + 1) % every == 0:
        print(f"Processed {i + 1}/{total} samples")
