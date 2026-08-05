"""Shared evaluation harness for the Swin quantile diagnostics.

Not an entry point -- import it. Every script in this directory does the same four things
before it can say anything interesting (resolve a config, build the model and load a
checkpoint into it, memory-map the aligned field arrays, pick a sample subset from a
split), and each one used to carry its own copy. That lives here now, in `setup()`.

The intended shape of a diagnostic in this directory::

    p = add_eval_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--my_own_flag", ...)          # only what is genuinely yours
    args = p.parse_args()

    ev = setup(args)                               # config -> model -> arrays -> device
    idx = choose_indices(split_pool(ev.splits_path, args.split), ev.arrays.n_total,
                         args.n_samples, args.seed, args.sample_indices)
    pred = predict(ev, idx)                        # (len(idx), 6, H, W), batched
    ...
    fig.savefig(ev.figure_path("my_diagnostic.png"))

Anything that does NOT go through `setup()` is re-deriving the checkpoint/array/config
plumbing by hand, and will silently drift from the rest of the directory the next time the
checkpoint format or an array filename changes.

Checkpoint format is the dict written by `train_new_enscgp_swin.py`'s `save_checkpoint`:
`{epoch, best_val_*, model_state_dict, optimizer_state_dict}`. It stores no config, so the
architecture is rebuilt from `--config` -- which must match the run that produced the
checkpoint, or `load_state_dict` fails (loudly, and `assert_quantile_output` below catches
the one case where it wouldn't).

Requires a 0714+ QUANTILE checkpoint (6-channel output). A pre-0714 Gaussian/Cholesky
checkpoint fails the channel guard rather than silently producing nonsense.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import RUNS_DIR, figures_dir, resolve as resolve_path  # noqa: E402  (re-exported)
from new_enscgp_swin import (  # noqa: E402
    DEFAULT_CONFIG_PATH, ProbabilisticSwin2SR, build_model, load_config,
)
# The pinball/CRPS definitions are shared with the training loss itself -- see
# scripts/quantile_metrics.py. Re-exported here so a diagnostic needs one import.
from quantile_metrics import (  # noqa: E402  (re-exported)
    CRPS_TAUS, CRPS_WEIGHTS, crps_3q, crps_from_pinball, pinball,
)
from terrain_encoder import load_terrain_input  # noqa: E402

COMPONENTS = ("u", "v")

# The index-aligned field arrays every diagnostic draws from, and the name each is reached
# by on EvalArrays. Adding a product here is the only thing needed to expose it everywhere.
ARRAY_FILES = {
    "posterior": "enscgp_posterior.npy",       # EnsCGP first guess fed to the model
    "bicubic": "era5_uv_2ch_bicubic.npy",      # bicubic baseline fed to the model
    "wrf": "wrf_uv.npy",                       # ground truth
    "era5": "era5_uv_2ch_native34.npy",        # native 34x34 ERA5, for LR panels
}
LAND_MASK_FILE = "land_mask_hires.npz"


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u ** 2 + v ** 2)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
class EvalArrays:
    """The index-aligned field arrays (ARRAY_FILES), memory-mapped.

    `n_total` is the shortest of them -- the largest index any of these diagnostics may
    legally reference. Each array is reachable as an attribute (`.wrf`, `.posterior`, ...).

    `overrides` maps an ARRAY_FILES key (or "land_mask") to an explicit path, for the
    `--wrf_path`-style flags; a None value means "use the default under data_dir", so CLI
    defaults can be passed straight through.
    """

    def __init__(self, data_dir: Path, **overrides):
        unknown = set(overrides) - set(ARRAY_FILES) - {"land_mask"}
        if unknown:
            raise ValueError(f"unknown array override(s) {sorted(unknown)}; "
                             f"expected any of {sorted(ARRAY_FILES)} or 'land_mask'")
        self.data_dir = Path(data_dir)
        self.paths = {name: Path(overrides.get(name) or self.data_dir / filename)
                      for name, filename in ARRAY_FILES.items()}
        self.land_mask_path = Path(overrides.get("land_mask") or self.data_dir / LAND_MASK_FILE)
        for name, path in self.paths.items():
            setattr(self, name, np.load(path, mmap_mode="r"))
        self.n_total = min(getattr(self, name).shape[0] for name in ARRAY_FILES)

    def land_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """(era34, wrf) boolean land masks as float, from land_mask_hires.npz."""
        m = np.load(self.land_mask_path)
        return m["era34"].astype(np.float64), m["wrf"].astype(np.float64)

    def wrf_land_mask(self) -> np.ndarray:
        """Just the 200x200 WRF-grid mask -- what the contour overlays actually use."""
        return self.land_masks()[1]


def resolve_paths(args, config: dict) -> tuple[Path, Path, Path, Path]:
    """(data_dir, log_dir, splits_path, checkpoint_path), CLI overriding config."""
    paths = config["paths"]
    data_dir = getattr(args, "data_dir", None) or resolve_path(paths["data_dir"])
    log_dir = resolve_path(paths["log_dir"])
    splits_path = getattr(args, "splits_path", None) or resolve_path(paths["splits_path"])
    checkpoint = getattr(args, "checkpoint", None) or (log_dir / "checkpoints" / "best.pth")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return Path(data_dir), Path(log_dir), Path(splits_path), Path(checkpoint)


def load_model(config: dict, checkpoint: Path, device) -> tuple[torch.nn.Module, dict]:
    """Build the architecture from `config`, load `checkpoint`'s weights into it, and hand
    back the raw checkpoint dict too (every caller reports its epoch / best_val_loss)."""
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_terrain(data_dir: Path, device) -> torch.Tensor:
    """(1, 4, 1000, 1000) terrain tensor, batch-broadcast by the model."""
    return load_terrain_input(data_dir).unsqueeze(0).to(device)


def split_pool(splits_path: Path, split: str) -> np.ndarray:
    return np.load(splits_path)[f"{split}_idx"]


def _parse_sample_indices(sample_indices: str, n_total: int) -> np.ndarray:
    """Parse and bounds-check an explicit `--sample_indices` string. Shared by both
    selection policies below -- the policies differ in their DEFAULT, never in how an
    explicit list is read."""
    idx = np.array([int(x.strip()) for x in sample_indices.split(",") if x.strip() != ""], dtype=int)
    if np.any(idx < 0) or np.any(idx >= n_total):
        raise ValueError(f"sample_indices must be in [0, {n_total - 1}]")
    return np.unique(idx)


def choose_indices(pool: np.ndarray, n_total: int, n_samples: int, seed: int,
                   sample_indices: str | None) -> np.ndarray:
    """Sample indices to evaluate: explicit `--sample_indices` wins, else a seeded draw from `pool`."""
    if sample_indices:
        return _parse_sample_indices(sample_indices, n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool, size=min(n_samples, len(pool)), replace=False)
    idx.sort()
    return idx


def choose_indices_all(pool: np.ndarray, n_total: int, max_samples: int | None, seed: int,
                       sample_indices: str | None) -> np.ndarray:
    """Like `choose_indices`, but defaults to the WHOLE pool rather than a small draw.

    Calibration statistics want every held-out sample (estimation variance from a
    handful is large enough to matter); the panel plots want a handful. Those are
    different policies, not one policy with a flag.
    """
    if sample_indices:
        return _parse_sample_indices(sample_indices, n_total)
    idx = np.sort(np.asarray(pool, dtype=int))
    if max_samples is not None and max_samples < len(idx):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    return idx


# --------------------------------------------------------------------------------------
# Forward pass
# --------------------------------------------------------------------------------------
def assert_quantile_output(pred: np.ndarray, checkpoint: Path | None = None) -> None:
    if pred.shape[1] != ProbabilisticSwin2SR.OUT_CHANNELS:
        raise ValueError(
            f"Expected a {ProbabilisticSwin2SR.OUT_CHANNELS}-channel quantile model output, got "
            f"{pred.shape[1]}. Is {checkpoint} a pre-0714 Gaussian/Cholesky checkpoint?"
        )


def iter_predictions(ev: "EvalSetup", idx: np.ndarray, batch_size: int = 8):
    """Yield (chunk_idx, pred) per batch, pred being (len(chunk), 6, H, W) numpy.

    Streaming, so a whole-split diagnostic never materializes every prediction at once
    (a full test split of 6-channel 200x200 float32 is ~1 GB). Use `predict` instead when
    the sample count is small enough to hold.
    """
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            post = torch.from_numpy(np.array(ev.arrays.posterior[chunk], dtype=np.float32, copy=True)).to(ev.device)
            bic = torch.from_numpy(np.array(ev.arrays.bicubic[chunk], dtype=np.float32, copy=True)).to(ev.device)
            pred = ev.model(post, bic, ev.terrain_raw).cpu().numpy()
            assert_quantile_output(pred, ev.checkpoint)
            yield chunk, pred


def predict(ev: "EvalSetup", idx: np.ndarray, batch_size: int = 8) -> np.ndarray:
    """(len(idx), 6, 200, 200) predictions [q10_u, q10_v, q50_u, q50_v, q90_u, q90_v].

    Batched, so the sample count is bounded by disk rather than GPU memory.
    """
    return np.concatenate([pred for _chunk, pred in iter_predictions(ev, idx, batch_size)], axis=0)


def split_quantiles(pred: np.ndarray):
    """(q10, q50, q90), each (B, 2, H, W), from the 6-channel model output.

    Always take the central field through this (or Q50_SLICE) rather than by raw index:
    channels 0-1 are q10 under the quantile head, so `pred[:, :2]` would silently hand back
    the lower envelope where the old Gaussian head put the mean.
    """
    return (pred[:, ProbabilisticSwin2SR.Q10_SLICE],
            pred[:, ProbabilisticSwin2SR.Q50_SLICE],
            pred[:, ProbabilisticSwin2SR.Q90_SLICE])


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def plot_panel(ax, data: np.ndarray, cmap: str, vmin: float, vmax: float, title: str,
               cbar_label: str, lsm: np.ndarray | None = None,
               uv: tuple[np.ndarray, np.ndarray] | None = None, quiver_skip: int = 1):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    if lsm is not None:
        ax.contour(lsm, levels=[0.5], colors="black", linewidths=0.8)
    if uv is not None:
        u, v = uv
        skip = max(1, quiver_skip)
        ys, xs = np.arange(0, u.shape[0], skip), np.arange(0, u.shape[1], skip)
        X, Y = np.meshgrid(xs, ys)
        # V is negated: imshow(origin="upper") inverts the y-axis, so +v (northward)
        # must point toward decreasing row index to still point up the page.
        ax.quiver(X, Y, u[::skip, ::skip], -v[::skip, ::skip], color="black", scale_units="xy")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, label=cbar_label)


def sym_max(*arrays: np.ndarray, floor: float = 1e-12) -> float:
    """Symmetric colour limit: the largest |value| across all inputs."""
    return max(floor, max(float(np.max(np.abs(a))) for a in arrays))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def add_eval_args(p: argparse.ArgumentParser, *, samples: str | None = "draw",
                  arrays: bool = True, n_samples_default: int = 4) -> argparse.ArgumentParser:
    """Attach the flags every diagnostic in this directory accepts.

    `samples` picks the selection policy's flags, matching the two functions above:
      "draw" -> --n_samples/--seed/--sample_indices  (use with choose_indices)
      "all"  -> --max_samples/--seed/--sample_indices (use with choose_indices_all)
      None   -> no selection flags (the script evaluates something fixed)
    `arrays` adds the per-array path overrides consumed by `setup()`.
    """
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                   help="new_enscgp_swin_config.json (architecture + default paths)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Defaults to <log_dir>/checkpoints/best.pth from --config")
    p.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --config")
    p.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --config")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                   help="Sample pool for default selection")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    if arrays:
        for name, filename in ARRAY_FILES.items():
            p.add_argument(f"--{name}_path", type=Path, default=None,
                           help=f"Defaults to <data_dir>/{filename}")
        p.add_argument("--hires_land_mask_path", type=Path, default=None,
                       help=f"Defaults to <data_dir>/{LAND_MASK_FILE}")
    if samples == "draw":
        p.add_argument("--n_samples", type=int, default=n_samples_default,
                       help="Number of random samples to evaluate")
    elif samples == "all":
        p.add_argument("--max_samples", type=int, default=None,
                       help="Cap the split size (random subset) for a quick look; default is the whole split")
    elif samples is not None:
        raise ValueError(f"samples must be 'draw', 'all', or None; got {samples!r}")
    if samples is not None:
        p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
        p.add_argument("--sample_indices", type=str, default=None,
                       help="Comma-separated raw sample indices, overriding the --split draw")
    return p


@dataclass
class EvalSetup:
    """Everything a diagnostic needs from the config/checkpoint, built once by `setup()`."""
    config: dict
    data_dir: Path
    log_dir: Path
    splits_path: Path
    checkpoint: Path
    device: torch.device
    model: torch.nn.Module
    ckpt: dict
    terrain_raw: torch.Tensor
    arrays: EvalArrays

    def figure_path(self, name: str) -> Path:
        """<log_dir>/figures/<name> -- the one output convention (see paths.figures_dir)."""
        return figures_dir(self.log_dir) / name

    def describe_checkpoint(self) -> str:
        return (f"{self.checkpoint} (epoch {self.ckpt.get('epoch')}, "
                f"best_val_loss {self.ckpt.get('best_val_loss')})")


def setup(args) -> EvalSetup:
    """The whole preamble every diagnostic shares, in one call.

    Reads the `--*_path` overrides from `args` when they are present (see `add_eval_args`),
    so a script that exposes them and one that does not both work unchanged.
    """
    config = load_config(args.config)
    data_dir, log_dir, splits_path, checkpoint = resolve_paths(args, config)
    device = torch.device(args.device)
    model, ckpt = load_model(config, checkpoint, device)
    terrain_raw = load_terrain(data_dir, device)
    overrides = {name: getattr(args, f"{name}_path", None) for name in ARRAY_FILES}
    overrides["land_mask"] = getattr(args, "hires_land_mask_path", None)
    arrays = EvalArrays(data_dir, **overrides)
    return EvalSetup(config=config, data_dir=data_dir, log_dir=log_dir, splits_path=splits_path,
                     checkpoint=checkpoint, device=device, model=model, ckpt=ckpt,
                     terrain_raw=terrain_raw, arrays=arrays)
