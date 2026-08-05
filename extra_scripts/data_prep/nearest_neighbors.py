"""nearest_neighbors.py

Compute k-nearest neighbors among ERA5 low-resolution samples (L1 / Manhattan).

By default the script will try to load the original `wndata.mat` ERA5 dataset
(`era5ens`) which contains flattened 2*34*34 = 2312 features per sample. If a
MAT file is not provided or not found, you can pass an ERA5 NumPy file instead
(`--era_npy`) but note that that may be at higher resolution and slower to
process. The output is a compressed NPZ with a single array `neighbors` of
shape (N, k) containing neighbor indices for each sample (excluding self).

Neighbors are always restricted to samples from a *different* storm event
than the query sample (event ids are recovered from `wndata.mat`, the same
way `extra_scripts/split_temporal_dataset.py` does it). Consecutive hours
within the same event are highly autocorrelated, so same-event matches would
otherwise dominate the nearest-neighbor list without being informative.

Pass `--split train` to restrict the *candidate* pool to one split (train/
val/test), as assigned in `data/sample_event_ids.csv` (the same file produced
alongside `extra_scripts/split_temporal_dataset.py`'s output). Every sample
is still queried for neighbors regardless of its own split -- only the pool
of candidates it can match against is restricted. Output indices are always
global, i.e. valid row indices into the original full-length ERA5/WRF
arrays (`--era_npy` / `--mat_path`), exactly as in the unrestricted case.

Example:
    python nearest_neighbors.py --k 10 --output neighbors_k10.npz

    python nearest_neighbors.py --split train --k 10 --output neighbors_vs_train_k10.npy

"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - fallback if sklearn isn't installed
    NearestNeighbors = None

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import DATA_DIR, WNDATA_MAT  # noqa: E402
from split_temporal_dataset import load_event_metadata  # noqa: E402  (same directory)


def load_era_from_mat(mat_path: str) -> np.ndarray:
    """Load flattened ERA5 samples from a MAT file (h5py-backed)."""
    with h5py.File(mat_path, "r") as f:
        if "era5ens" not in f:
            raise KeyError("'era5ens' dataset not found in mat file")
        era = f["era5ens"][...].astype(np.float32)
    # era shape is (N, 2312) = (N, 2*34*34) as in the original dataset
    return era


def load_era_from_npy(npy_path: str) -> np.ndarray:
    """Load ERA5 from a preprocessed .npy file and flatten spatial dims.

    Accepts arrays of shape (N, C, H, W) and returns (N, C*H*W).
    """
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Unsupported ERA5 npy shape: {arr.shape}")
    N, C, H, W = arr.shape
    flat = arr.reshape(N, C * H * W)
    return flat.astype(np.float32)


def load_split_assignment(csv_path: Path, n_samples: int) -> np.ndarray:
    """Load per-sample split labels from `data/sample_event_ids.csv`.

    Expects columns `sample_index, event_id, split, timestamp` (see
    `extra_scripts/split_temporal_dataset.py`), with `sample_index` matching
    the row order of the ERA5 array. Returns an array of shape (n_samples,)
    of split labels ("train"/"val"/"test"), indexed by sample_index.
    """
    split = np.full(n_samples, None, dtype=object)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            split[int(row["sample_index"])] = row["split"]
    if any(s is None for s in split):
        raise ValueError(f"{csv_path} does not cover all {n_samples} samples (sample_index out of range or missing rows)")
    return split


def compute_neighbors(
    data: np.ndarray,
    event_id: np.ndarray,
    k: int,
    n_jobs: int | None = None,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute k nearest neighbors (Manhattan) for each row in data, excluding
    the query sample itself and any candidate from the same event.

    Every row of `data` is queried, regardless of `candidate_mask`. If
    `candidate_mask` is given, only rows where it is True are eligible to be
    returned as neighbors (e.g. restrict matches to the training split while
    still finding neighbors for every sample). Returned indices are always
    global, i.e. row positions in the original `data`/`event_id` arrays.

    Returns an array shape (N, k) of neighbor indices.
    """
    N = data.shape[0]
    chunk = 256

    candidate_idx_map = np.arange(N) if candidate_mask is None else np.nonzero(candidate_mask)[0]
    candidate_data = data if candidate_mask is None else data[candidate_idx_map]
    candidate_event_id = event_id if candidate_mask is None else event_id[candidate_idx_map]
    Nc = candidate_data.shape[0]

    # Same-event samples (highly autocorrelated, often the closest matches)
    # must be filtered out before keeping the top k, so query for more than
    # k+1 candidates: enough to guarantee k cross-event neighbors even in the
    # worst case where every other member of an event outranks all of them.
    max_event_size = int(np.unique(candidate_event_id, return_counts=True)[1].max())
    query_k = min(Nc, k + max_event_size)

    def cross_event_topk(candidate_pos: np.ndarray, candidate_dist: np.ndarray, row: int) -> np.ndarray:
        order = np.argsort(candidate_dist)
        sel = candidate_pos[order]
        global_sel = candidate_idx_map[sel]
        keep = global_sel[(global_sel != row) & (candidate_event_id[sel] != event_id[row])]
        assert len(keep) >= k, (
            f"Sample {row}: only found {len(keep)} cross-event neighbors within top {query_k} candidates"
        )
        return keep[:k]

    # Use sklearn's NearestNeighbors when available, but query in chunks
    if NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=query_k, metric="manhattan", algorithm="brute", n_jobs=n_jobs)
        nn.fit(candidate_data)
        neighbors = np.empty((N, k), dtype=np.int64)
        processed = 0
        for i in range(0, N, chunk):
            j = min(N, i + chunk)
            # Query neighbors for this block
            dists, inds = nn.kneighbors(data[i:j], n_neighbors=query_k)
            for r, row in enumerate(range(i, j)):
                neighbors[row] = cross_event_topk(inds[r], dists[r], row)
            processed = j
            # print progress for each 100 samples
            for p in range(((processed - chunk) // 100 + 1) * 100, processed + 1, 100):
                if p <= N:
                    print(f"Processed {p}/{N} samples")
        # final flush if not a multiple of 100
        if processed % 100 != 0:
            print(f"Processed {processed}/{N} samples")
        return neighbors

    # Fallback: naive pairwise distance in chunks to avoid huge memory use
    from scipy.spatial.distance import cdist

    neighbors = np.empty((N, k), dtype=np.int64)
    processed = 0
    for i in range(0, N, chunk):
        j = min(N, i + chunk)
        # compute pairwise cityblock distances between block and all candidates
        d = cdist(data[i:j], candidate_data, metric="cityblock")
        # partition to the closest query_k candidates per row
        idx = np.argpartition(d, kth=query_k - 1, axis=1)[:, :query_k]
        for r in range(d.shape[0]):
            row = i + r
            neighbors[row] = cross_event_topk(idx[r], d[r, idx[r]], row)
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{N} samples")

    # final flush if not a multiple of 100
    if processed % 100 != 0:
        print(f"Processed {processed}/{N} samples")

    return neighbors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute k-nearest neighbors among ERA5 samples (L1 distance)")
    p.add_argument("--mat_path", type=str, default=str(WNDATA_MAT))
    p.add_argument("--era_npy", type=str, default=str(DATA_DIR / "era5_uv_2ch.npy"), help="ERA5 npy option")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--output", type=str, default=str(DATA_DIR / "neighbors.npy"))
    p.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for sklearn (use -1 for all)")
    p.add_argument("--subset", type=int, default=0, help="If >0, compute neighbors only for first SUBSET samples (for testing)")
    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="If set, restrict the candidate pool to this split (e.g. 'train'); every sample is still "
        "queried for neighbors regardless of its own split. Split assignment comes from --event_ids_csv",
    )
    p.add_argument(
        "--event_ids_csv",
        type=str,
        default=str(DATA_DIR / "sample_event_ids.csv"),
        help="CSV with sample_index,event_id,split,timestamp columns, used to filter by --split",
    )
    args = p.parse_args(argv)

    era = None
    if args.era_npy and Path(args.era_npy).exists():
        print(f"Loading ERA5 from NPZ/NPY: {args.era_npy}")
        era = load_era_from_npy(args.era_npy)
    elif args.mat_path and Path(args.mat_path).exists():
        print(f"Loading ERA5 from MAT: {args.mat_path}")
        era = load_era_from_mat(args.mat_path)
    else:
        raise FileNotFoundError("No valid ERA5 source found. Provide --mat_path or --era_npy")

    mat_path = Path(args.mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"--mat_path is required to recover event ids for filtering: {mat_path} not found")
    print(f"Loading event ids from {mat_path}")
    event_id, _ = load_event_metadata(mat_path, era.shape[0])

    split_label = None
    if args.split:
        csv_path = Path(args.event_ids_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"--event_ids_csv not found: {csv_path}")
        split_label = load_split_assignment(csv_path, era.shape[0])

    if args.subset and 0 < args.subset < era.shape[0]:
        print(f"Using subset: first {args.subset} samples")
        era = era[: args.subset]
        event_id = event_id[: args.subset]
        if split_label is not None:
            split_label = split_label[: args.subset]

    candidate_mask = None
    if split_label is not None:
        candidate_mask = split_label == args.split
        if not candidate_mask.any():
            raise ValueError(f"No samples found for split='{args.split}' in {args.event_ids_csv}")
        print(
            f"Restricting candidate pool to split='{args.split}': "
            f"{int(candidate_mask.sum())}/{len(candidate_mask)} samples (queries remain unrestricted)"
        )

    print(f"Data shape: {era.shape}, dtype={era.dtype}")
    print(f"Computing {args.k} nearest neighbors (Manhattan), excluding same-event candidates...")

    neighbors = compute_neighbors(era, event_id, args.k, n_jobs=args.n_jobs, candidate_mask=candidate_mask)

    out_path = Path(args.output)
    # Force .npy suffix for consistency
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as a plain .npy array for fastest load/save
    np.save(out_path, neighbors)
    print(f"Saved neighbors to {out_path} shape={neighbors.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
