"""Shared evaluation harness for the Swin quantile diagnostics.

Not an entry point -- import it. Every script in this directory does the same four
things before it can say anything interesting (resolve a config, build the model and
load a checkpoint into it, memory-map the four aligned field arrays, pick a sample
subset from a split), and each one used to carry its own copy. That lives here now.

Checkpoint format is the dict written by `train_new_enscgp_swin.py`'s
`save_checkpoint`: `{epoch, best_val_*, model_state_dict, optimizer_state_dict}`.
It stores no config, so the architecture is rebuilt from `--config` -- which must
match the run that produced the checkpoint, or `load_state_dict` fails (loudly, and
`assert_quantile_output` below catches the one case where it wouldn't).

Requires a 0714+ QUANTILE checkpoint (6-channel output). A pre-0714
Gaussian/Cholesky checkpoint fails the channel guard rather than silently
producing nonsense.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import RUNS_DIR, resolve as resolve_path  # noqa: E402  (re-exported)
from new_enscgp_swin import (  # noqa: E402
    DEFAULT_CONFIG_PATH, ProbabilisticSwin2SR, build_model, load_config,
)
from terrain_encoder import load_terrain_input  # noqa: E402

COMPONENTS = ("u", "v")

# Quantile levels the model predicts, with midpoint-rule integration weights over
# tau in [0, 1] (boundaries at 0.3 / 0.7) for the CRPS quantile decomposition.
# A coarse 3-point rule: good for relative comparison, not an absolute CRPS.
CRPS_TAUS = (0.1, 0.5, 0.9)
CRPS_WEIGHTS = (0.3, 0.4, 0.3)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u ** 2 + v ** 2)


def pinball(q: np.ndarray, truth: np.ndarray, tau: float) -> np.ndarray:
    """Per-pixel pinball (tilted-L1) loss at quantile level tau."""
    err = truth - q
    return np.maximum(tau * err, (tau - 1.0) * err)


def crps_from_quantiles(q_by_tau: tuple[np.ndarray, ...], truth: np.ndarray) -> np.ndarray:
    """Per-pixel CRPS via the quantile decomposition on CRPS_TAUS.

    `q_by_tau` is aligned with CRPS_TAUS and must be MARGINAL quantiles of the same
    scalar quantity as `truth` (here: one wind component).
    """
    total = np.zeros_like(truth, dtype=np.float64)
    for tau, w, q in zip(CRPS_TAUS, CRPS_WEIGHTS, q_by_tau):
        total += w * pinball(q, truth, tau)
    return 2.0 * total


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
class EvalArrays:
    """The four index-aligned field arrays, memory-mapped.

    `n_total` is the shortest of them -- the largest index any of these diagnostics
    may legally reference.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.posterior = np.load(data_dir / "enscgp_posterior.npy", mmap_mode="r")
        self.bicubic = np.load(data_dir / "era5_uv_2ch_bicubic.npy", mmap_mode="r")
        self.wrf = np.load(data_dir / "wrf_uv.npy", mmap_mode="r")
        self.era5 = np.load(data_dir / "era5_uv_2ch_native34.npy", mmap_mode="r")
        self.n_total = min(a.shape[0] for a in
                           (self.posterior, self.bicubic, self.wrf, self.era5))

    def land_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """(era34, wrf) boolean land masks as float, from land_mask_hires.npz."""
        m = np.load(self.data_dir / "land_mask_hires.npz")
        return m["era34"].astype(np.float64), m["wrf"].astype(np.float64)


def resolve_paths(args, config: dict) -> tuple[Path, Path, Path, Path]:
    """(data_dir, log_dir, splits_path, checkpoint_path), CLI overriding config."""
    paths = config["paths"]
    data_dir = args.data_dir or resolve_path(paths["data_dir"])
    log_dir = resolve_path(paths["log_dir"])
    splits_path = getattr(args, "splits_path", None) or resolve_path(paths["splits_path"])
    checkpoint = args.checkpoint or (log_dir / "checkpoints" / "best.pth")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return Path(data_dir), Path(log_dir), Path(splits_path), Path(checkpoint)


def load_model(config: dict, checkpoint: Path, device) -> torch.nn.Module:
    """Build the architecture from `config` and load `checkpoint`'s weights into it."""
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_terrain(data_dir: Path, device) -> torch.Tensor:
    """(1, 4, 1000, 1000) terrain tensor, batch-broadcast by the model."""
    return load_terrain_input(data_dir).unsqueeze(0).to(device)


def split_pool(splits_path: Path, split: str) -> np.ndarray:
    return np.load(splits_path)[f"{split}_idx"]


def choose_indices(pool: np.ndarray, n_total: int, n_samples: int, seed: int,
                   sample_indices: str | None) -> np.ndarray:
    """Sample indices to evaluate: explicit `--sample_indices` wins, else a seeded draw from `pool`."""
    if sample_indices:
        idx = np.array([int(x.strip()) for x in sample_indices.split(",") if x.strip() != ""], dtype=int)
        if np.any(idx < 0) or np.any(idx >= n_total):
            raise ValueError(f"sample_indices must be in [0, {n_total - 1}]")
        return np.unique(idx)
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
        idx = np.array([int(x.strip()) for x in sample_indices.split(",") if x.strip() != ""], dtype=int)
        if np.any(idx < 0) or np.any(idx >= n_total):
            raise ValueError(f"sample_indices must be in [0, {n_total - 1}]")
        return np.unique(idx)
    idx = np.sort(np.asarray(pool, dtype=int))
    if max_samples is not None and max_samples < len(idx):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    return idx


# --------------------------------------------------------------------------------------
# Forward pass
# --------------------------------------------------------------------------------------
def assert_quantile_output(pred: np.ndarray, checkpoint: Path) -> None:
    if pred.shape[1] != ProbabilisticSwin2SR.OUT_CHANNELS:
        raise ValueError(
            f"Expected a {ProbabilisticSwin2SR.OUT_CHANNELS}-channel quantile model output, got "
            f"{pred.shape[1]}. Is {checkpoint} a pre-0714 Gaussian/Cholesky checkpoint?"
        )


def predict(model, arrays: EvalArrays, idx: np.ndarray, terrain_raw, device,
            batch_size: int = 8, checkpoint: Path | None = None) -> np.ndarray:
    """(len(idx), 6, 200, 200) predictions [q10_u, q10_v, q50_u, q50_v, q90_u, q90_v].

    Batched so the sample count is bounded by disk, not GPU memory.
    """
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            post = torch.from_numpy(np.array(arrays.posterior[chunk], dtype=np.float32)).to(device)
            bic = torch.from_numpy(np.array(arrays.bicubic[chunk], dtype=np.float32)).to(device)
            out.append(model(post, bic, terrain_raw).cpu().numpy())
    pred = np.concatenate(out, axis=0)
    if checkpoint is not None:
        assert_quantile_output(pred, checkpoint)
    return pred


def split_quantiles(pred: np.ndarray):
    """(q10, q50, q90), each (B, 2, H, W), from the 6-channel model output."""
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
def add_eval_args(p: argparse.ArgumentParser, *, samples: bool = True) -> argparse.ArgumentParser:
    """Attach the flags every diagnostic in this directory accepts."""
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Defaults to <log_dir>/checkpoints/best.pth from --config")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --config")
    p.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --config")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    if samples:
        p.add_argument("--n_samples", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--sample_indices", type=str, default=None,
                       help="Comma-separated indices, overriding --n_samples/--seed")
    return p


def setup(args):
    """Config -> (config, data_dir, log_dir, splits_path, checkpoint, device, model, terrain, arrays).

    The whole preamble every diagnostic shares, in one call.
    """
    config = load_config(args.config)
    data_dir, log_dir, splits_path, checkpoint = resolve_paths(args, config)
    device = torch.device(args.device)
    model = load_model(config, checkpoint, device)
    terrain_raw = load_terrain(data_dir, device)
    arrays = EvalArrays(data_dir)
    return dict(config=config, data_dir=data_dir, log_dir=log_dir, splits_path=splits_path,
                checkpoint=checkpoint, device=device, model=model,
                terrain_raw=terrain_raw, arrays=arrays)
