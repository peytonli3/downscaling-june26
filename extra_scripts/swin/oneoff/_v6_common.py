"""
Shared plumbing for the q50 coverage-bias diagnostic (Parts A-D) against
runs/0714/checkpoints/best.pth.

0714 predates several architecture changes (EnsCGP-sigma-seeded, gated offset head; see
CHANGELOG "0714" vs later versions), so its checkpoint must be interpreted with 0714's OWN
model class, not the current one -- loading it with the current class either fails strict
`load_state_dict` (missing/unexpected keys) or, worse, could silently misinterpret weights if
shapes happened to match. `_v6_0714_arch/new_enscgp_swin.py` is an exact copy of
`git show v6-0714:scripts/new_enscgp_swin.py`, kept alongside these diagnostic scripts so they
are reproducibly rerunnable without depending on git state at run time.
`network_swin2sr.py` / `terrain_encoder.py` are unchanged since v6-0714 (verified via
`git diff v6-0714`), so those are imported from the current scripts/ directory.

Central, load-bearing statistical convention used by every Part: aggregate at the EVENT level,
never at the pixel/sample level. Samples within an event are near-duplicates in time (same
storm, consecutive hours), so treating them as independent draws understates variance and can
manufacture spurious "significant" structure. `compute_event_bias` returns one bias map per
EVENT (already averaged over that event's samples); every downstream mean/se/bootstrap in
Parts A-D operates on the event axis of that array, not on raw samples or pixels.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
CUR_SCRIPTS = REPO / "scripts"
# Sits next to this module, and must move with it -- hence __file__-relative here
# rather than anchored on REPO.
OLD_ARCH_DIR = Path(__file__).resolve().parent / "_v6_0714_arch"

sys.path.insert(0, str(CUR_SCRIPTS))  # provisional, just to reach paths.py; re-ordered below

from paths import DATA_DIR, RUNS_DIR  # noqa: E402

CHECKPOINT = RUNS_DIR / "0714/checkpoints/best.pth"
SPLITS_PATH = DATA_DIR / "splits_70_15_15/split_indices.npz"
OUTPUT_DIR = RUNS_DIR / "0714/figures/bias_diagnostic"

if not (OLD_ARCH_DIR / "new_enscgp_swin.py").is_file():
    raise RuntimeError(
        f"pinned v6-0714 model class not found at {OLD_ARCH_DIR}. Without it the import "
        "below would silently bind the CURRENT ProbabilisticSwin2SR and misinterpret the "
        "0714 checkpoint's weights. Restore it with:\n"
        f"  git show v6-0714:scripts/new_enscgp_swin.py > {OLD_ARCH_DIR}/new_enscgp_swin.py"
    )

for _p in (str(OLD_ARCH_DIR), str(CUR_SCRIPTS)):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(CUR_SCRIPTS))   # network_swin2sr.py, terrain_encoder.py
sys.path.insert(0, str(OLD_ARCH_DIR))  # new_enscgp_swin.py -- OLD (v6-0714) class, must win

from new_enscgp_swin import ProbabilisticSwin2SR, build_model  # noqa: E402  (the OLD class)
from terrain_encoder import load_terrain_input  # noqa: E402

# Fail loudly if the CURRENT class won the import race anyway (e.g. a caller already
# imported new_enscgp_swin from scripts/ before importing this module).
_bound = Path(sys.modules["new_enscgp_swin"].__file__).resolve().parent
if _bound != OLD_ARCH_DIR:
    raise RuntimeError(
        f"new_enscgp_swin resolved to {_bound}, expected the pinned v6-0714 copy at "
        f"{OLD_ARCH_DIR}. Import _v6_common before anything that pulls in "
        "the current model class."
    )

COMPONENTS = ("u", "v")

# Exact model config the 0714 run used (confirmed from its own training log).
CONFIG = {
    "seed": 0,
    "model": {
        "img_size": 200, "embed_dim": 96, "depths": [4, 4, 4], "num_heads": [6, 6, 6],
        "window_size": 8, "mlp_ratio": 4.0, "residual_base": "bicubic",
        "residual_gate_init": 0.1, "init_spread_scale": 1.28,
        "drop_rate": 0.05, "attn_drop_rate": 0.05, "drop_path_rate": 0.1, "head_dropout": 0.15,
    },
}

# 3-quantile CRPS decomposition, re-exported from scripts/quantile_metrics.py so these
# diagnostics score the 0714 checkpoint with the same functional the run itself optimized.
# Safe to import despite the pinned-OLD-architecture juggling above: quantile_metrics
# imports nothing from this repo (only numpy/torch), so it cannot drag the CURRENT model
# class in behind it -- which is the specific hazard the sys.path ordering guards against.
from quantile_metrics import (  # noqa: E402,F401  (re-exported for the diagnostics; see CLAUDE.md "quantile_metrics")
    CRPS_TAUS, CRPS_WEIGHTS, crps_3q, pinball,
)


def load_model(device):
    model = build_model(CONFIG).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_terrain(device):
    return load_terrain_input(DATA_DIR).unsqueeze(0).to(device)


def load_event_split_map() -> dict[str, dict[int, list[int]]]:
    """split -> {event_id: [sample_index, ...]}, from data/sample_event_ids.csv."""
    m = {"train": {}, "val": {}, "test": {}}
    with open(DATA_DIR / "sample_event_ids.csv") as f:
        for row in csv.DictReader(f):
            m[row["split"]].setdefault(int(row["event_id"]), []).append(int(row["sample_index"]))
    return m


def load_land_mask() -> np.ndarray:
    return np.load(DATA_DIR / "land_mask_hires.npz")["wrf"].astype(bool)  # (200,200)


_ELEV_CACHE: np.ndarray | None = None


def load_elevation_200() -> np.ndarray:
    """5x average-pool the native 1000x1000 elevation channel down to the 200x200 model grid
    (same convention the model's own terrain encoder uses to get onto the WRF grid)."""
    global _ELEV_CACHE
    if _ELEV_CACHE is None:
        elev = np.asarray(np.load(DATA_DIR / "topography_features.npy", mmap_mode="r")[0], dtype=np.float64)
        _ELEV_CACHE = elev.reshape(200, 5, 200, 5).mean(axis=(1, 3))
    return _ELEV_CACHE


def coastline_distance(land_mask: np.ndarray) -> np.ndarray:
    """Unsigned per-pixel distance (in grid cells) to the nearest opposite-type pixel."""
    from scipy import ndimage
    return ndimage.distance_transform_edt(land_mask) + ndimage.distance_transform_edt(~land_mask)


def bh_fdr_mask(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR mask at level q. pvals: any shape; returns bool of the same shape."""
    shape = pvals.shape
    flat = pvals.ravel()
    m = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresh
    if not below.any():
        return np.zeros(shape, dtype=bool)
    cutoff = ranked[np.max(np.where(below)[0])]
    return (flat <= cutoff).reshape(shape)


def select_event_ids(split_map, split: str, max_events: int | None = None, seed: int = 0) -> np.ndarray:
    """Sorted event ids for `split`, optionally subsampled to `max_events`."""
    event_ids = np.array(sorted(split_map[split].keys()))
    if max_events is not None and max_events < len(event_ids):
        rng = np.random.default_rng(seed)
        event_ids = np.sort(rng.choice(event_ids, size=max_events, replace=False))
    return event_ids


@torch.no_grad()
def iter_event_batches(model, device, terrain_raw, split_map, split: str, batch_size: int = 16,
                       max_events: int | None = None, seed: int = 0):
    """Run `model` over a split, event by event, yielding (ei, event_id, pred, truth).

    `ei` is the event's position in `select_event_ids(...)` (so a caller can write straight
    into a preallocated (E, ...) array), `pred` is the model's raw output and `truth` the
    matching WRF batch, both torch float32 on `device`. One event may span several yields;
    the yields for an event are contiguous.

    The raw output is handed back UNSLICED on purpose -- the v6 and v7 architectures put the
    quantiles in different channels, and the two callers of this generator load different
    ones (see eval_model_scorecard.py), so slicing is the caller's business.

    This is the one place the "load three mmaps, walk a split's events in index order,
    forward in batches" loop lives; compute_event_bias, compute_event_predictions, and
    eval_model_scorecard's event_metrics all used to carry their own copy of it.
    """
    posterior = np.load(DATA_DIR / "enscgp_posterior.npy", mmap_mode="r")
    bicubic = np.load(DATA_DIR / "era5_uv_2ch_bicubic.npy", mmap_mode="r")
    wrf = np.load(DATA_DIR / "wrf_uv.npy", mmap_mode="r")

    events = split_map[split]
    for ei, eid in enumerate(select_event_ids(split_map, split, max_events, seed)):
        idx = np.array(sorted(events[int(eid)]))
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            post_b = torch.from_numpy(np.array(posterior[b], dtype=np.float32, copy=True)).to(device)
            bic_b = torch.from_numpy(np.array(bicubic[b], dtype=np.float32, copy=True)).to(device)
            truth_b = torch.from_numpy(np.array(wrf[b], dtype=np.float32, copy=True)).to(device)
            yield ei, int(eid), model(post_b, bic_b, terrain_raw), truth_b


def _per_event_mean(model, device, terrain_raw, split_map, split: str, slices, batch_size: int,
                    max_events: int | None, seed: int):
    """Mean over each event's samples of `slices` applied to the prediction, plus the truth.

    Returns (event_ids (E,), [ (E,2,H,W) float64 per slice ], truth (E,2,H,W) float64).
    Averaging WITHIN an event before any across-event statistic is the load-bearing part:
    it collapses each event's correlated, near-duplicate samples into one independent unit
    (see the module docstring).
    """
    event_ids = select_event_ids(split_map, split, max_events, seed)
    E, H, W = len(event_ids), 200, 200
    sums = [np.zeros((E, 2, H, W), dtype=np.float64) for _ in slices]
    truth_sum = np.zeros((E, 2, H, W), dtype=np.float64)
    counts = np.zeros(E, dtype=np.float64)

    for ei, _eid, pred, truth in iter_event_batches(model, device, terrain_raw, split_map, split,
                                                    batch_size, max_events, seed):
        pred_np = pred.cpu().numpy()
        for acc, sl in zip(sums, slices):
            acc[ei] += pred_np[:, sl].astype(np.float64).sum(axis=0)
        truth_sum[ei] += truth.cpu().numpy().astype(np.float64).sum(axis=0)
        counts[ei] += pred_np.shape[0]

    n = counts.reshape(-1, 1, 1, 1)
    return event_ids, [acc / n for acc in sums], truth_sum / n


def compute_event_bias(model, device, terrain_raw, split_map, split: str, batch_size: int = 16,
                        max_events: int | None = None, seed: int = 0):
    """Per-event mean signed q50 bias (pred - truth), per component.

    Returns (event_ids: (E,) int, event_bias: (E,2,H,W) float64).
    """
    event_ids, (q50,), truth = _per_event_mean(
        model, device, terrain_raw, split_map, split,
        (ProbabilisticSwin2SR.Q50_SLICE,), batch_size, max_events, seed)
    return event_ids, q50 - truth


def compute_event_predictions(model, device, terrain_raw, split_map, split: str, batch_size: int = 16,
                               max_events: int | None = None, seed: int = 0):
    """Like compute_event_bias, but returns per-event-averaged q10/q50/q90 AND truth (each
    (E,2,H,W)) rather than just the bias -- needed by Part B for coverage/CRPS after correction,
    which need the full quantile triple, not only q50's bias."""
    event_ids, (q10, q50, q90), truth = _per_event_mean(
        model, device, terrain_raw, split_map, split,
        (ProbabilisticSwin2SR.Q10_SLICE, ProbabilisticSwin2SR.Q50_SLICE,
         ProbabilisticSwin2SR.Q90_SLICE), batch_size, max_events, seed)
    return event_ids, q10, q50, q90, truth


def event_stats(event_bias: np.ndarray):
    """event_bias: (E,2,H,W) -> M (2,H,W) mean-over-events, se (2,H,W) standard error over
    events (ddof=1), n_events."""
    n = event_bias.shape[0]
    M = event_bias.mean(axis=0)
    se = event_bias.std(axis=0, ddof=1) / np.sqrt(n)
    return M, se, n


def plot_diverging(ax, data: np.ndarray, land_mask: np.ndarray, title: str, pct: float = 99.0,
                    vmax: float | None = None):
    """vmax: if given, used directly (e.g. to share one color scale across several panels);
    otherwise computed per-call as the pct-th percentile of |data|."""
    if vmax is None:
        vmax = float(np.percentile(np.abs(data), pct))
    vmax = max(vmax, 1e-8)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    ax.contour(land_mask.astype(float), levels=[0.5], colors="black", linewidths=0.7)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im
