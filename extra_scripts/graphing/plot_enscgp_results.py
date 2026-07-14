"""
Visualize EnsCGP posterior results against ERA5 (LR) and WRF (HR) ground truth.

For each randomly chosen sample, plots 6 panels:
1) ERA5 LR wind speed (native 34x34)
2) WRF HR wind speed (200x200, ground truth)
3) EnsCGP posterior mean wind speed (200x200)
4) Posterior var(u)
5) Posterior var(v)
6) Posterior cov(u, v)

References the structure of 26.3_wind/SWIN/evaluation/plot_wind_quiver_swin_checkpoint.py (random
sample selection, shared color scales, one figure with a row per sample), but runs no model
inference -- everything plotted here is either raw data or scripts/enscgp_train.py's precomputed
data/enscgp_posterior.npy. Posterior var(u)/var(v)/cov(u,v) are reconstructed from that file's
[u, v, L11, L21, L22] channels (L = lower-Cholesky factor of the 2x2 per-pixel covariance):
var(u) = L11^2, cov(u, v) = L11*L21, var(v) = L21^2 + L22^2.

Display orientation: all data files (era5_uv_2ch_native34.npy, wrf_uv.npy,
enscgp_posterior.npy) are stored north-up (row 0 = north), so all panels render correctly
under plain imshow(origin="upper") without any display flips.

The land/sea mask overlay is rasterized straight from Natural Earth 10m vector coastlines (see
scripts/build_hires_land_mask.py), not the 0.25-deg ERA5 LSM raster used as a model feature in
26.3_wind -- that raster has only ~32x32 native cells spanning this domain, far coarser than the
200x200 WRF/posterior grid, which is why it looked blocky here.

Wind-direction arrows and the "jet" speed colormap adapt the convention of
26.3_wind/SWIN/evaluation/plot_wind_quiver_swin_checkpoint.py (which itself draws on
26.3_wind/SWIN/infer_wind_swin2sr.py's inference pipeline). That script passes real lon/lat to
quiver under a non-inverted cartopy axis, so +v (northward) already points "up" with no sign
games. Here we instead plot on raw pixel-index axes via imshow(origin="upper"), whose y-axis is
inverted (row 0 at the top) -- so on these axes, a quiver arrow needs V_quiver = -v_physical
(confirmed empirically) for +v (northward) to still point up the page; U_quiver = +u_physical
needs no such flip since the x-axis is not inverted.

Usage:
    python plot_enscgp_results.py --n_samples 6 --seed 42 --output enscgp_panels.png
    python plot_enscgp_results.py --sample_indices 0,1000,3000 --output enscgp_panels.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def choose_indices(n_total: int, n_samples: int, seed: int, sample_indices: str | None) -> np.ndarray:
    if sample_indices:
        idx = np.array([int(x.strip()) for x in sample_indices.split(",") if x.strip() != ""], dtype=int)
        if np.any(idx < 0) or np.any(idx >= n_total):
            raise ValueError(f"sample_indices must be in [0, {n_total - 1}]")
        return np.unique(idx)

    rng = np.random.default_rng(seed)
    n = min(n_samples, n_total)
    idx = rng.choice(n_total, size=n, replace=False)
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
        # toward decreasing row index to still point up the page (see module docstring).
        ax.quiver(X, Y, u[::skip, ::skip], -v[::skip, ::skip], color="black", scale_units="xy")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, label=cbar_label)


def main() -> None:
    data_dir = Path("/home/peytonli/26.6_wind/data")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era5_path", type=Path, default=data_dir / "era5_uv_2ch_native34.npy")
    parser.add_argument("--wrf_path", type=Path, default=data_dir / "wrf_uv.npy")
    parser.add_argument("--posterior_path", type=Path, default=data_dir / "enscgp_posterior.npy")
    parser.add_argument(
        "--hires_land_mask_path", type=Path, default=data_dir / "land_mask_hires.npz",
        help="Land mask built by scripts/build_hires_land_mask.py: keys 'wrf' (200,200), 'era34' (34,34)",
    )
    parser.add_argument("--n_samples", type=int, default=6, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--sample_indices", type=str, default=None,
        help="Comma-separated sample indices (overrides --n_samples/--seed), e.g. 0,25,100",
    )
    parser.add_argument("--output", type=Path, default=Path("/home/peytonli/26.6_wind/inference_results/enscgp_panels.png"), help="Output PNG path")
    parser.add_argument("--quiver_skip_lr", type=int, default=2, help="Arrow subsampling on the 34x34 ERA5 panel")
    parser.add_argument("--quiver_skip_hr", type=int, default=10, help="Arrow subsampling on the 200x200 WRF/posterior panels")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    era5 = np.load(args.era5_path, mmap_mode="r")
    wrf = np.load(args.wrf_path, mmap_mode="r")
    posterior = np.load(args.posterior_path, mmap_mode="r")

    n_total = min(era5.shape[0], wrf.shape[0], posterior.shape[0])
    idx = choose_indices(n_total, args.n_samples, args.seed, args.sample_indices)

    # Hi-res land masks, each in their own grid's native row order (era34: row 0 = north;
    # wrf: row 0 = south -- see scripts/build_hires_land_mask.py, which reads the same raw
    # lat arrays used everywhere else in this module).
    hires_masks = np.load(args.hires_land_mask_path)
    lsm_era34 = hires_masks["era34"].astype(np.float64)
    lsm_wrf = hires_masks["wrf"].astype(np.float64)

    col_titles = [
        "ERA5 LR speed (34x34)",
        "WRF HR speed (ground truth)",
        "EnsCGP posterior mean speed",
        "Posterior var(u)",
        "Posterior var(v)",
        "Posterior cov(u, v)",
    ]
    cbar_labels = ["m/s", "m/s", "m/s", "(m/s)^2", "(m/s)^2", "(m/s)^2"]

    n_rows, n_cols = len(idx), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)

    for row, i in enumerate(idx):
        era5_speed = speed(era5[i, 0], era5[i, 1])
        wrf_speed = speed(wrf[i, 0], wrf[i, 1])

        post = np.asarray(posterior[i])  # (5, 200, 200): u, v, L11, L21, L22
        post_speed = speed(post[0], post[1])
        L11, L21, L22 = post[2], post[3], post[4]
        var_u = L11 ** 2
        var_v = L21 ** 2 + L22 ** 2
        cov_uv = L11 * L21

        speed_vmin = float(min(era5_speed.min(), wrf_speed.min(), post_speed.min()))
        speed_vmax = float(max(era5_speed.max(), wrf_speed.max(), post_speed.max()))
        var_vmax = max(float(max(var_u.max(), var_v.max())), 1e-12)
        cov_vmax = max(float(np.abs(cov_uv).max()), 1e-12)

        era5_u, era5_v = era5[i, 0], era5[i, 1]
        wrf_u, wrf_v = wrf[i, 0], wrf[i, 1]
        post_u, post_v = post[0], post[1]

        panels = [
            (era5_speed, "jet", speed_vmin, speed_vmax, lsm_era34, (era5_u, era5_v), args.quiver_skip_lr),
            (wrf_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (wrf_u, wrf_v), args.quiver_skip_hr),
            (post_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (post_u, post_v), args.quiver_skip_hr),
            (var_u, "magma", 0.0, var_vmax, lsm_wrf, None, 1),
            (var_v, "magma", 0.0, var_vmax, lsm_wrf, None, 1),
            (cov_uv, "RdBu_r", -cov_vmax, cov_vmax, lsm_wrf, None, 1),
        ]
        for col, (data, cmap, vmin, vmax, lsm, uv, qskip) in enumerate(panels):
            title = f"{col_titles[col]} | idx={i}" if col == 0 else col_titles[col]
            plot_panel(axes[row, col], data, cmap, vmin, vmax, title, cbar_labels[col], lsm=lsm, uv=uv, quiver_skip=qskip)

    fig.suptitle("EnsCGP posterior vs ERA5 / WRF", fontsize=14, y=1.0)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"Samples used: {idx.tolist()}")


if __name__ == "__main__":
    main()
