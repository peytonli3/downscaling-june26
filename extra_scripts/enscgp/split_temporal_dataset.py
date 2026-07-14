"""Split ERA5 and WRF `.npy` datasets into train/val/test sets by whole event.

Each sample is one hourly snapshot belonging to a storm "track" (event) in
wndata.mat. Consecutive hours within the same event are highly
autocorrelated, so a per-sample split (random or interleaved) leaks
information between splits. This script instead holds out whole events:
every hour of a given event lands in exactly one of train/val/test.

Events are processed in chronological order (by each event's first
timestamp) and assigned via proportional allocation: each event goes to
whichever split is currently furthest below its target share of samples
seen so far. This converges to ~70/15/15 by sample count while keeping
held-out events spread across the full time range, rather than clustered
at the end.

Outputs:
    - era5_train.npy / era5_val.npy / era5_test.npy
    - wrf_train.npy / wrf_val.npy / wrf_test.npy
    - split_indices.npz (train_idx, val_idx, test_idx, train_events, val_events, test_events)

Example:
    python split_temporal_dataset.py \
        --era_npy /home/peytonli/26.6_wind/data/era5_uv_2ch.npy \
        --wrf_npy /home/peytonli/26.6_wind/data/wrf_uv.npy \
        --output_dir /home/peytonli/26.6_wind/data/splits_70_15_15

"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

DEFAULT_MAT_PATH = "/net/momo/data/projects/downscaling/datasource/wndata.mat"


def find_default_path(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected a 4D array (N, C, H, W), got {arr.shape} from {path}")
    return arr


def load_event_metadata(mat_path: Path, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover per-sample event id and timestamp from the wndata.mat ERA5 table.

    wndata.mat stores ERA5 as a MATLAB `table` (v7.3/HDF5), which h5py,
    scipy, and pymatreader cannot decode structurally (table internals are
    MCOS-serialized). The numeric table columns are still present as plain
    arrays in the file's `#refs#` group, so each column is identified by
    its value signature (range/integrality) rather than by name, since
    table column order isn't otherwise addressable outside MATLAB.
    """
    with h5py.File(mat_path, "r") as f:
        candidates: dict[str, np.ndarray] = {}
        for key, obj in f["#refs#"].items():
            if isinstance(obj, h5py.Dataset) and obj.dtype == np.float64 and obj.size == n_samples:
                candidates[key] = obj[()].reshape(-1)

    def is_integral(arr: np.ndarray) -> bool:
        return bool(np.allclose(arr, np.round(arr)))

    year = month = day = hour = None
    used: set[str] = set()
    for key, arr in candidates.items():
        if not is_integral(arr):
            continue
        lo, hi = arr.min(), arr.max()
        if year is None and 1900 <= lo and hi <= 2100 and hi - lo > 1:
            year, used = arr, used | {key}
        elif month is None and lo >= 1 and hi <= 12 and hi > 9:
            month, used = arr, used | {key}
        elif day is None and lo >= 1 and hi <= 31 and hi > 12:
            day, used = arr, used | {key}
        elif hour is None and lo == 0 and hi <= 23 and hi > 12:
            hour, used = arr, used | {key}

    if any(col is None for col in (year, month, day, hour)):
        raise RuntimeError(f"Could not locate year/month/day/hour columns in {mat_path}")

    track_candidates = [
        arr
        for key, arr in candidates.items()
        if key not in used and is_integral(arr) and 1 < len(np.unique(arr)) < n_samples
    ]
    if not track_candidates:
        raise RuntimeError(f"Could not locate a track/event id column in {mat_path}")
    event_id = track_candidates[0].astype(np.int64)

    timestamp = np.array(
        [
            np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(h):02d}", "h")
            for y, m, d, h in zip(year, month, day, hour)
        ]
    )
    return event_id, timestamp


def build_event_split_indices(
    event_id: np.ndarray,
    timestamp: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, np.ndarray]:
    """Assign whole events to train/val/test, targeting val/test ~= given fractions of samples."""
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1:
        raise ValueError("Invalid split fractions: require 0 <= val/test and val+test < 1")
    target = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}

    rows_by_event: dict[int, list[int]] = {}
    for idx, eid in enumerate(event_id):
        rows_by_event.setdefault(int(eid), []).append(idx)

    event_start = {eid: timestamp[rows].min() for eid, rows in rows_by_event.items()}
    ordered_events = sorted(rows_by_event.keys(), key=lambda eid: event_start[eid])

    counts = {"train": 0, "val": 0, "test": 0}
    split_rows: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    total_assigned = 0
    for eid in ordered_events:
        rows = rows_by_event[eid]
        projected_total = total_assigned + len(rows)
        deficits = {name: target[name] * projected_total - counts[name] for name in target}
        chosen = max(deficits, key=deficits.get)
        split_rows[chosen].extend(rows)
        counts[chosen] += len(rows)
        total_assigned = projected_total

    split_events: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for eid in ordered_events:
        rows = rows_by_event[eid]
        for name, idx_list in split_rows.items():
            if rows[0] in idx_list:
                split_events[name].append(eid)
                break

    return {
        "indices": {name: np.array(sorted(idx_list), dtype=np.int64) for name, idx_list in split_rows.items()},
        "events": {name: np.array(sorted(ev_list), dtype=np.int64) for name, ev_list in split_events.items()},
    }


def save_split_npy(source: np.ndarray, indices: np.ndarray, output_path: Path, chunk_size: int = 256) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_shape = (len(indices),) + source.shape[1:]
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=source.dtype, shape=out_shape)

    write_pos = 0
    for start in range(0, len(indices), chunk_size):
        stop = min(len(indices), start + chunk_size)
        chunk_indices = indices[start:stop]
        out[write_pos : write_pos + len(chunk_indices)] = np.asarray(source[chunk_indices])
        write_pos += len(chunk_indices)

    out.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split ERA5 and WRF arrays into train/val/test by whole event")
    parser.add_argument(
        "--era_npy",
        type=Path,
        default=None,
        help="Path to era5_uv_2ch.npy (default: auto-detect local copies)",
    )
    parser.add_argument(
        "--wrf_npy",
        type=Path,
        default=None,
        help="Path to wrf_uv.npy (default: auto-detect local copies)",
    )
    parser.add_argument(
        "--mat_path",
        type=Path,
        default=Path(DEFAULT_MAT_PATH),
        help="Path to wndata.mat, used to recover per-sample event id and timestamp",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/peytonli/26.6_wind/data/splits_70_15_15"),
        help="Directory to write split arrays and indices",
    )
    parser.add_argument("--val_frac", type=float, default=0.15, help="Target fraction of samples held out for val")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Target fraction of samples held out for test")
    parser.add_argument("--chunk_size", type=int, default=256, help="Chunk size used when writing split arrays")
    args = parser.parse_args(argv)

    era_path = args.era_npy or find_default_path(
        [
            "/home/peytonli/26.6_wind/data/era5_uv_2ch.npy",
            "/home/peytonli/26.3_wind/SWIN/preprocessed/era5_uv_2ch.npy",
        ]
    )
    wrf_path = args.wrf_npy or find_default_path(
        [
            "/home/peytonli/26.6_wind/data/wrf_uv.npy",
            "/home/peytonli/26.3_wind/SWIN/preprocessed/wrf_uv.npy",
        ]
    )

    if era_path is None:
        raise FileNotFoundError("Could not find era5_uv_2ch.npy; pass --era_npy")
    if wrf_path is None:
        raise FileNotFoundError("Could not find wrf_uv.npy; pass --wrf_npy")

    print(f"Loading ERA5 from {era_path}")
    era = load_npy(era_path)
    print(f"Loading WRF from {wrf_path}")
    wrf = load_npy(wrf_path)

    if era.shape[0] != wrf.shape[0]:
        raise ValueError(f"ERA5 and WRF must have the same number of samples: {era.shape[0]} vs {wrf.shape[0]}")

    n_samples = era.shape[0]

    print(f"Loading event metadata from {args.mat_path}")
    event_id, timestamp = load_event_metadata(args.mat_path, n_samples)

    result = build_event_split_indices(event_id, timestamp, val_frac=args.val_frac, test_frac=args.test_frac)
    splits, split_events = result["indices"], result["events"]
    counts = {name: len(idx) for name, idx in splits.items()}

    print(f"Dataset size: {n_samples} samples across {len(np.unique(event_id))} events")
    for name in ("train", "val", "test"):
        idx = splits[name]
        ts = timestamp[idx]
        print(
            f"  {name}: {counts[name]} samples ({100 * counts[name] / n_samples:.1f}%), "
            f"{len(split_events[name])} events, {ts.min()} to {ts.max()}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "split_indices.npz",
        train_idx=splits["train"],
        val_idx=splits["val"],
        test_idx=splits["test"],
        train_events=split_events["train"],
        val_events=split_events["val"],
        test_events=split_events["test"],
    )

    for split_name, indices in splits.items():
        print(f"Writing {split_name} split with {len(indices)} samples")
        save_split_npy(era, indices, args.output_dir / f"era5_{split_name}.npy", chunk_size=args.chunk_size)
        save_split_npy(wrf, indices, args.output_dir / f"wrf_{split_name}.npy", chunk_size=args.chunk_size)

    print(f"Done. Split files written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
