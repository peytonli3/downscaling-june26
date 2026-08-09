"""Storm-event membership for the sample index -- the unit every aggregate must respect.

Not an entry point -- import it. Samples in this dataset are consecutive HOURS of a storm,
so two samples from one event are near-duplicates, not independent draws. Any statistic that
pools them as if they were independent understates its own variance and over-weights long
events. The repo's convention is therefore to aggregate at the EVENT level: reduce within an
event first, then take an unweighted mean over events.

This module only answers "which samples belong to which event, in which split". The reducing
is the caller's business, because the correct reduction differs: a bias map averages the
FIELDS within an event, while MAE/RMSE must average the METRIC (mean|a-b| is not
|mean a - mean b|).

Deliberately imports nothing but stdlib/numpy and `paths`, so it is safe to import from
`extra_scripts/swin/oneoff/_v6_common.py`, which pins an old model class on `sys.path` and
must not risk pulling the current one in behind it.
"""
from __future__ import annotations

import csv
from pathlib import Path

SAMPLE_EVENT_CSV = "sample_event_ids.csv"


def load_event_split_map(data_dir: Path) -> dict[str, dict[int, list[int]]]:
    """split -> {event_id: [sample_index, ...]}, from data/sample_event_ids.csv.

    Sample indices index directly into the field arrays (wrf_uv.npy et al.), so a caller can
    slice with them without any further mapping.
    """
    path = Path(data_dir) / SAMPLE_EVENT_CSV
    if not path.exists():
        raise FileNotFoundError(f"Event id CSV not found: {path}")
    out: dict[str, dict[int, list[int]]] = {"train": {}, "val": {}, "test": {}}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["split"]].setdefault(int(row["event_id"]), []).append(int(row["sample_index"]))
    return out
