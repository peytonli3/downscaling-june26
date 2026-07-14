"""
Evaluate a trained ResNetRefiner checkpoint (scripts/resnet_refiner.py) against
ERA5 (LR) and WRF (HR) ground truth.

For each sample, plots 7 panels:
1) ERA5 LR wind speed (native 34x34)
2) WRF HR wind speed (200x200, ground truth)
3) EnsCGP posterior mean wind speed (200x200) -- the first-guess ResNet starts from
4) ResNet posterior mean wind speed (200x200) -- model's refined mu_u/mu_v
5) ResNet posterior var(u)
6) ResNet posterior var(v)
7) Speed z-score: (ResNet predicted speed - WRF truth speed) / sigma_speed, i.e. how many
   predicted standard deviations the speed prediction is off truth at each pixel.

Mirrors the panel layout/style of plot_enscgp_results.py (random sample selection,
shared color scales within a row, land/sea contour overlay, wind-direction quiver arrows,
the same north-up-ERA5/flipped-WRF display convention -- see that script's docstring for
why the WRF-grid panels are flipped along axis 0 for display); panel 3 (EnsCGP) is read
directly from that same data/enscgp_posterior.npy with no model involved, so it shows
exactly what's fed to the model below. Unlike plot_enscgp_results.py, this script also runs
real model inference: it loads resnet_refiner_config.json, builds a ResNetRefiner, loads
--checkpoint's model_state_dict (the format saved by train_resnet_refiner.py's
save_checkpoint: model/optimizer/scheduler state + epoch + best_val_loss -- no embedded
config, hence --config), and feeds it the EnsCGP posterior first-guess
(data/enscgp_posterior.npy) -- sliced into first_guess_mu/first_guess_chol per
ResNetRefiner.forward()'s explicit-residual-args signature -- plus the static terrain
input, batched over all chosen sample indices in one forward pass. This is the same
evaluation this project runs against the Swin refiner (see
extra_scripts/graphing/swin/eval_new_enscgp_swin_checkpoint.py), so the two models'
checkpoints are directly comparable panel-for-panel.

Sample source: by default, sample indices are drawn from the held-out test split
(data/splits_70_15_15/split_indices.npz, test_idx) rather than the unrestricted pool
plot_enscgp_results.py uses -- this script is meant to evaluate a trained checkpoint, so
the model should not have seen these samples during training. --split train/val/test
switches the pool; --sample_indices (raw indices into era5/wrf/posterior, as in
plot_enscgp_results.py) bypasses split filtering entirely as an explicit override.

Speed z-score (panel 7): the model gives a per-pixel bivariate Gaussian over (u, v) (mean
mu, covariance Sigma = L L^T from the predicted Cholesky factor), not a distribution over
speed = sqrt(u^2+v^2) directly. sigma_speed is the delta-method (first-order Taylor)
approximation:
    Var(speed) ~= g_u^2 * var(u) + g_v^2 * var(v) + 2 * g_u * g_v * cov(u, v),
    g_u = mu_u / speed, g_v = mu_v / speed   (gradient of sqrt(u^2+v^2) at the mean)
This is exact only in the limit sigma << speed; it degrades when predicted speed is near
zero (the gradient is singular there), so speed in the denominator of g_u/g_v is clamped
to a small epsilon. The z-score itself, (pred_speed - truth_speed) / sigma_speed, is
plotted on a FIXED [-3, 3] color scale (standard calibration-plot convention, comparable
across samples/figures) rather than per-sample dynamic scaling; values beyond +/-3 sigma
saturate the colormap. Positive z = ResNet over-predicted speed relative to its own claimed
uncertainty; negative = under-predicted.

Usage:
    python eval_resnet_refiner_checkpoint.py --checkpoint /home/peytonli/26.6_wind/logs/resnet_refiner/checkpoints/best.pth
    python eval_resnet_refiner_checkpoint.py --checkpoint .../best.pth --split test --n_samples 6 --seed 42
    python eval_resnet_refiner_checkpoint.py --checkpoint .../best.pth --sample_indices 12,4081,6500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from resnet_refiner import DEFAULT_CONFIG_PATH, build_model, load_config  # noqa: E402
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


def speed_zscore(mu_u: np.ndarray, mu_v: np.ndarray, var_u: np.ndarray, var_v: np.ndarray, cov_uv: np.ndarray,
                  truth_speed: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(pred_speed - truth_speed) / sigma_speed, sigma_speed via the delta-method linearization
    of speed=sqrt(u^2+v^2) at the predicted mean (see module docstring)."""
    pred_speed = speed(mu_u, mu_v)
    s_safe = np.maximum(pred_speed, eps)
    g_u, g_v = mu_u / s_safe, mu_v / s_safe
    var_speed = g_u ** 2 * var_u + g_v ** 2 * var_v + 2 * g_u * g_v * cov_uv
    sigma_speed = np.sqrt(np.clip(var_speed, 0.0, None))
    return (pred_speed - truth_speed) / np.maximum(sigma_speed, eps)


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
        # toward decreasing row index to still point up the page (see module docstring).
        ax.quiver(X, Y, u[::skip, ::skip], -v[::skip, ::skip], color="black", scale_units="xy")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, label=cbar_label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="resnet_refiner_config.json (architecture + default paths)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth from --config")
    parser.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --config")
    parser.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --config")
    parser.add_argument("--era5_path", type=Path, default=None, help="Defaults to <data_dir>/era5_uv_2ch_native34.npy")
    parser.add_argument("--wrf_path", type=Path, default=None, help="Defaults to <data_dir>/wrf_uv.npy")
    parser.add_argument("--posterior_path", type=Path, default=None, help="Defaults to <data_dir>/enscgp_posterior.npy (EnsCGP first guess fed to the model)")
    parser.add_argument("--hires_land_mask_path", type=Path, default=None, help="Defaults to <data_dir>/land_mask_hires.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Sample pool for default random selection")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--sample_indices", type=str, default=None,
        help="Comma-separated raw sample indices (overrides --split/--n_samples/--seed), e.g. 0,25,100",
    )
    parser.add_argument("--output", type=Path, default=Path("/home/peytonli/26.6_wind/inference_results/resnet_checkpoint_panels.png"), help="Output PNG path")
    parser.add_argument("--quiver_skip_lr", type=int, default=2, help="Arrow subsampling on the 34x34 ERA5 panel")
    parser.add_argument("--quiver_skip_hr", type=int, default=10, help="Arrow subsampling on the 200x200 WRF/ResNet panels")
    parser.add_argument("--z_clip", type=float, default=3.0, help="Fixed +/- color-scale bound for the speed z-score panel")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = args.data_dir or Path(paths["data_dir"])
    checkpoint_path = args.checkpoint or (Path(paths["log_dir"]) / "checkpoints" / "best.pth")
    splits_path = args.splits_path or Path(paths["splits_path"])
    era5_path = args.era5_path or (data_dir / "era5_uv_2ch_native34.npy")
    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    posterior_path = args.posterior_path or (data_dir / "enscgp_posterior.npy")
    hires_land_mask_path = args.hires_land_mask_path or (data_dir / "land_mask_hires.npz")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device)

    era5 = np.load(era5_path, mmap_mode="r")
    wrf = np.load(wrf_path, mmap_mode="r")
    posterior = np.load(posterior_path, mmap_mode="r")
    n_total = min(era5.shape[0], wrf.shape[0], posterior.shape[0])

    splits = np.load(splits_path)
    pool = splits[f"{args.split}_idx"]
    idx = choose_indices(pool, n_total, args.n_samples, args.seed, args.sample_indices)

    hires_masks = np.load(hires_land_mask_path)
    lsm_era34 = hires_masks["era34"].astype(np.float64)
    lsm_wrf = hires_masks["wrf"].astype(np.float64)

    posterior_batch = torch.from_numpy(np.array(posterior[idx], dtype=np.float32, copy=True)).to(device)
    with torch.no_grad():
        pred_batch = model(
            posterior_batch, posterior_batch[:, :2], posterior_batch[:, 2:5], terrain_raw
        ).cpu().numpy()  # (B, 5, 200, 200): mu_u, mu_v, L11, L21, L22

    col_titles = [
        "ERA5 LR speed (34x34)",
        "WRF HR speed (ground truth)",
        "EnsCGP posterior mean speed",
        "ResNet posterior mean speed",
        "ResNet var(u)",
        "ResNet var(v)",
        "Speed z-score: (pred-truth)/sigma",
    ]
    cbar_labels = ["m/s", "m/s", "m/s", "m/s", "(m/s)^2", "(m/s)^2", "sigma"]

    n_rows, n_cols = len(idx), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for row, i in enumerate(idx):
        era5_u, era5_v = np.asarray(era5[i, 0]), np.asarray(era5[i, 1])
        wrf_u, wrf_v = np.asarray(wrf[i, 0]), np.asarray(wrf[i, 1])
        ens_u, ens_v = np.asarray(posterior[i, 0]), np.asarray(posterior[i, 1])  # EnsCGP first guess fed to the model
        mu_u, mu_v, L11, L21, L22 = pred_batch[row]

        era5_speed = speed(era5_u, era5_v)
        wrf_speed = speed(wrf_u, wrf_v)
        ens_speed = speed(ens_u, ens_v)
        pred_speed = speed(mu_u, mu_v)

        var_u = L11 ** 2
        var_v = L21 ** 2 + L22 ** 2
        cov_uv = L11 * L21
        z = speed_zscore(mu_u, mu_v, var_u, var_v, cov_uv, wrf_speed)

        speed_vmin = float(min(era5_speed.min(), wrf_speed.min(), ens_speed.min(), pred_speed.min()))
        speed_vmax = float(max(era5_speed.max(), wrf_speed.max(), ens_speed.max(), pred_speed.max()))
        var_vmax = max(float(max(var_u.max(), var_v.max())), 1e-12)

        panels = [
            (era5_speed, "jet", speed_vmin, speed_vmax, lsm_era34, (era5_u, era5_v), args.quiver_skip_lr),
            (wrf_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (wrf_u, wrf_v), args.quiver_skip_hr),
            (ens_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (ens_u, ens_v), args.quiver_skip_hr),
            (pred_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (mu_u, mu_v), args.quiver_skip_hr),
            (var_u, "magma", 0.0, var_vmax, lsm_wrf, None, 1),
            (var_v, "magma", 0.0, var_vmax, lsm_wrf, None, 1),
            (z, "RdBu_r", -args.z_clip, args.z_clip, lsm_wrf, None, 1),
        ]
        for col, (data, cmap, vmin, vmax, lsm, uv, qskip) in enumerate(panels):
            title = f"{col_titles[col]} | idx={i}" if col == 0 else col_titles[col]
            plot_panel(axes[row, col], data, cmap, vmin, vmax, title, cbar_labels[col], lsm=lsm, uv=uv, quiver_skip=qskip)

    fig.suptitle(f"ResNet checkpoint eval vs ERA5 / WRF (split={args.split}, ckpt={checkpoint_path.name})", fontsize=14, y=1.0)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"Checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')})")
    print(f"Split: {args.split}")
    print(f"Samples used: {idx.tolist()}")


if __name__ == "__main__":
    main()
