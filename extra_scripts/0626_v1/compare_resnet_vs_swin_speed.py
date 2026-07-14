#!/usr/bin/env python3
"""
Compare predicted wind-speed magnitude between a trained ResNetRefiner checkpoint
(scripts/resnet_refiner.py) and a trained ProbabilisticSwin2SR checkpoint
(scripts/new_enscgp_swin.py) on the same randomly chosen samples.

Both models consume the identical EnsCGP first-guess posterior (data/enscgp_posterior.npy)
and are evaluated against the same WRF ground truth (data/wrf_uv.npy), so any difference
between their outputs is attributable to the two architectures/training runs, not to
different inputs.

For each sample, plots 4 panels:
1) WRF HR speed (ground truth) -- for visual reference only, not part of the diff
2) ResNet posterior mean speed (sqrt(mu_u^2 + mu_v^2) from the ResNet checkpoint)
3) Swin posterior mean speed (same, from the Swin checkpoint)
4) Speed difference: ResNet - Swin -- positive (red) where ResNet predicts a higher
   speed than Swin at that pixel, negative (blue) where Swin predicts higher.

Panels 1-3 share one color scale per row (so absolute speed is visually comparable
across the row); panel 4 uses a separate diverging scale, by default shared across all
chosen samples (computed once from the max |diff| over every sample, via --diff_clip to
override) so different samples' diff panels are also comparable to each other, not just
within a row. Mirrors the panel/display conventions of
extra_scripts/graphing/resnet/eval_resnet_refiner_checkpoint.py (land/sea contour
overlay, wind-direction quiver arrows, WRF-grid panels flipped along axis 0 for display).

Sample source: by default, sample indices are drawn from the held-out test split
(data/splits_70_15_15/split_indices.npz, test_idx); --sample_indices bypasses split
filtering with explicit raw indices, as in the single-model eval scripts.

Usage:
    python compare_resnet_vs_swin_speed.py
    python compare_resnet_vs_swin_speed.py --n_samples 8 --seed 7
    python compare_resnet_vs_swin_speed.py --sample_indices 12,4081,6500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from resnet_refiner import (  # noqa: E402
    DEFAULT_CONFIG_PATH as RESNET_DEFAULT_CONFIG_PATH,
    build_model as build_resnet_model,
    load_config as load_resnet_config,
)
from new_enscgp_swin import (  # noqa: E402
    DEFAULT_CONFIG_PATH as SWIN_DEFAULT_CONFIG_PATH,
    build_model as build_swin_model,
    load_config as load_swin_config,
)
from terrain_encoder import load_terrain_input  # noqa: E402


def choose_indices(pool: np.ndarray, n_total: int, n_samples: int, seed: int, sample_indices: str | None) -> np.ndarray:
    if sample_indices:
        idx = np.array([int(x.strip()) for x in sample_indices.split(",") if x.strip() != ""], dtype=int)
        if np.any(idx < 0) or np.any(idx >= n_total):
            raise ValueError(f"sample_indices must be in [0, {n_total - 1}]")
        return np.unique(idx)

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(pool))
    idx = rng.choice(pool, size=n, replace=False)
    idx.sort()
    return idx


def speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u ** 2 + v ** 2)


def plot_panel(
    ax, data: np.ndarray, cmap: str, vmin: float, vmax: float, title: str, cbar_label: str,
    lsm: np.ndarray | None = None, uv: tuple[np.ndarray, np.ndarray] | None = None, quiver_skip: int = 1,
):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    if lsm is not None:
        ax.contour(lsm, levels=[0.5], colors="black", linewidths=0.8)
    if uv is not None:
        u, v = uv
        skip = max(1, quiver_skip)
        ys, xs = np.arange(0, u.shape[0], skip), np.arange(0, u.shape[1], skip)
        X, Y = np.meshgrid(xs, ys)
        # V is negated: imshow(origin="upper") inverts the y-axis, so +v (northward) must point
        # toward decreasing row index to still point up the page.
        ax.quiver(X, Y, u[::skip, ::skip], -v[::skip, ::skip], color="black", scale_units="xy")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, label=cbar_label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resnet_config", type=Path, default=RESNET_DEFAULT_CONFIG_PATH)
    parser.add_argument("--resnet_checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth from --resnet_config")
    parser.add_argument("--swin_config", type=Path, default=SWIN_DEFAULT_CONFIG_PATH)
    parser.add_argument("--swin_checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth from --swin_config")
    parser.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --resnet_config (shared by both models)")
    parser.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --resnet_config")
    parser.add_argument("--wrf_path", type=Path, default=None, help="Defaults to <data_dir>/wrf_uv.npy")
    parser.add_argument("--posterior_path", type=Path, default=None, help="Defaults to <data_dir>/enscgp_posterior.npy")
    parser.add_argument("--hires_land_mask_path", type=Path, default=None, help="Defaults to <data_dir>/land_mask_hires.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Sample pool for default random selection")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--sample_indices", type=str, default=None,
        help="Comma-separated raw sample indices (overrides --split/--n_samples/--seed), e.g. 0,25,100",
    )
    parser.add_argument("--output", type=Path, default=Path("/home/peytonli/26.6_wind/inference_results/resnet_vs_swin_speed_diff.png"))
    parser.add_argument("--quiver_skip", type=int, default=10, help="Arrow subsampling on the 200x200 panels")
    parser.add_argument("--diff_clip", type=float, default=None, help="Fixed +/- color-scale bound for the diff panel; default is max|diff| over all chosen samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    resnet_config = load_resnet_config(args.resnet_config)
    swin_config = load_swin_config(args.swin_config)
    resnet_paths, swin_paths = resnet_config["paths"], swin_config["paths"]

    data_dir = args.data_dir or Path(resnet_paths["data_dir"])
    splits_path = args.splits_path or Path(resnet_paths["splits_path"])
    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    posterior_path = args.posterior_path or (data_dir / "enscgp_posterior.npy")
    hires_land_mask_path = args.hires_land_mask_path or (data_dir / "land_mask_hires.npz")
    resnet_checkpoint_path = args.resnet_checkpoint or (Path(resnet_paths["log_dir"]) / "checkpoints" / "best.pth")
    swin_checkpoint_path = args.swin_checkpoint or (Path(swin_paths["log_dir"]) / "checkpoints" / "best.pth")

    if not resnet_checkpoint_path.exists():
        raise FileNotFoundError(f"ResNet checkpoint not found: {resnet_checkpoint_path}")
    if not swin_checkpoint_path.exists():
        raise FileNotFoundError(f"Swin checkpoint not found: {swin_checkpoint_path}")

    device = torch.device(args.device)

    resnet_model = build_resnet_model(resnet_config).to(device)
    resnet_ckpt = torch.load(resnet_checkpoint_path, map_location=device)
    resnet_model.load_state_dict(resnet_ckpt["model_state_dict"])
    resnet_model.eval()
    resnet_terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device)

    swin_model = build_swin_model(swin_config).to(device)
    swin_ckpt = torch.load(swin_checkpoint_path, map_location=device)
    swin_model.load_state_dict(swin_ckpt["model_state_dict"])
    swin_model.eval()
    swin_terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device)

    wrf = np.load(wrf_path, mmap_mode="r")
    posterior = np.load(posterior_path, mmap_mode="r")
    n_total = min(wrf.shape[0], posterior.shape[0])

    splits = np.load(splits_path)
    pool = splits[f"{args.split}_idx"]
    idx = choose_indices(pool, n_total, args.n_samples, args.seed, args.sample_indices)

    hires_masks = np.load(hires_land_mask_path)
    lsm_wrf = hires_masks["wrf"].astype(np.float64)

    posterior_batch = torch.from_numpy(np.array(posterior[idx], dtype=np.float32, copy=True)).to(device)
    with torch.no_grad():
        resnet_pred = resnet_model(
            posterior_batch, posterior_batch[:, :2], posterior_batch[:, 2:5], resnet_terrain_raw
        ).cpu().numpy()  # (B, 5, 200, 200): mu_u, mu_v, L11, L21, L22
        swin_pred = swin_model(posterior_batch, swin_terrain_raw).cpu().numpy()

    diffs = []  # per-sample (200, 200) ResNet-speed-minus-Swin-speed, for the shared diff color scale and summary stats
    per_sample = []
    for row, i in enumerate(idx):
        wrf_u, wrf_v = np.asarray(wrf[i, 0]), np.asarray(wrf[i, 1])
        resnet_mu_u, resnet_mu_v = resnet_pred[row, 0], resnet_pred[row, 1]
        swin_mu_u, swin_mu_v = swin_pred[row, 0], swin_pred[row, 1]

        wrf_speed = speed(wrf_u, wrf_v)
        resnet_speed = speed(resnet_mu_u, resnet_mu_v)
        swin_speed = speed(swin_mu_u, swin_mu_v)
        diff = resnet_speed - swin_speed

        diffs.append(diff)
        per_sample.append((wrf_u, wrf_v, resnet_mu_u, resnet_mu_v, swin_mu_u, swin_mu_v,
                            wrf_speed, resnet_speed, swin_speed, diff))

    diff_vmax = args.diff_clip if args.diff_clip is not None else max(float(np.abs(d).max()) for d in diffs)
    diff_vmax = max(diff_vmax, 1e-6)

    col_titles = ["WRF HR speed (ground truth)", "ResNet posterior mean speed", "Swin posterior mean speed", "Diff: ResNet - Swin speed"]
    cbar_labels = ["m/s", "m/s", "m/s", "m/s"]

    n_rows, n_cols = len(idx), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows), squeeze=False)

    for row, i in enumerate(idx):
        (wrf_u, wrf_v, resnet_mu_u, resnet_mu_v, swin_mu_u, swin_mu_v,
         wrf_speed, resnet_speed, swin_speed, diff) = per_sample[row]

        speed_vmin = float(min(wrf_speed.min(), resnet_speed.min(), swin_speed.min()))
        speed_vmax = float(max(wrf_speed.max(), resnet_speed.max(), swin_speed.max()))

        panels = [
            (wrf_speed, "jet", speed_vmin, speed_vmax, (wrf_u, wrf_v)),
            (resnet_speed, "jet", speed_vmin, speed_vmax, (resnet_mu_u, resnet_mu_v)),
            (swin_speed, "jet", speed_vmin, speed_vmax, (swin_mu_u, swin_mu_v)),
            (diff, "RdBu_r", -diff_vmax, diff_vmax, None),
        ]
        for col, (data, cmap, vmin, vmax, uv) in enumerate(panels):
            title = f"{col_titles[col]} | idx={i}" if col == 0 else col_titles[col]
            plot_panel(axes[row, col], data, cmap, vmin, vmax, title, cbar_labels[col],
                       lsm=lsm_wrf, uv=uv, quiver_skip=args.quiver_skip)

        mean_abs, max_abs = float(np.abs(diff).mean()), float(np.abs(diff).max())
        print(f"idx={i}: mean|ResNet-Swin speed diff|={mean_abs:.4f} m/s, max|diff|={max_abs:.4f} m/s")

    all_diffs = np.concatenate([d.ravel() for d in diffs])
    print(f"Pooled over {len(idx)} samples: mean diff={all_diffs.mean():.4f} m/s "
          f"(positive = ResNet predicts higher speed), mean|diff|={np.abs(all_diffs).mean():.4f} m/s, "
          f"RMS diff={np.sqrt((all_diffs ** 2).mean()):.4f} m/s")

    fig.suptitle(
        f"ResNet vs Swin predicted speed (split={args.split}, "
        f"resnet_ckpt={resnet_checkpoint_path.name}, swin_ckpt={swin_checkpoint_path.name})",
        fontsize=14, y=1.0,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"ResNet checkpoint: {resnet_checkpoint_path} (epoch {resnet_ckpt.get('epoch')}, best_val_loss {resnet_ckpt.get('best_val_loss')})")
    print(f"Swin checkpoint: {swin_checkpoint_path} (epoch {swin_ckpt.get('epoch')}, best_val_loss {swin_ckpt.get('best_val_loss')})")
    print(f"Split: {args.split}")
    print(f"Samples used: {idx.tolist()}")


if __name__ == "__main__":
    main()
