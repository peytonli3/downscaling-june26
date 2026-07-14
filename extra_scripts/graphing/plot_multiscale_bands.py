"""
Visualize the Laplacian-pyramid band decomposition (scripts/multiscale_loss.py) applied to
real WRF high-resolution wind fields, using the n_levels/base_sigma from
new_enscgp_swin_config.json's "multiscale" section -- the exact same decomposition
multiscale_loss.MultiscaleLoss uses during training.

For each randomly chosen sample, plots n_levels+1 panels:
1) WRF HR wind speed (200x200, full field, ground truth)
2..n_levels+1) Each Laplacian band's magnitude sqrt(band_u^2 + band_v^2), ordered
   coarse -> fine (matching the left-to-right convention of
   extra_scripts/band_error_diagnostics.py's plots), labeled with its approximate physical
   scale range in km (via the WRF grid's anisotropic spacing -- DY_KM/DX_KM below, same
   constants/derivation as band_error_diagnostics.py and
   extra_scripts/graphing/swin/compare_eigenspectra_enscgp_swin.py).

Each band panel is normalized to ITS OWN min/max (not a shared scale), so its spatial
structure is visible regardless of how much energy is at that scale -- bands span ~2 orders
of magnitude in RMS power between the coarsest and finest level (see
band_error_diagnostics.py's eigenspectrum-equivalent finding), so a shared scale would
render every band but the coarsest as a blank/uniform panel.

References the structure of plot_enscgp_results.py in this same directory (random sample
selection, one figure with a row per sample, land/sea contour overlay; plot_panel/
choose_indices below are near-identical) -- but visualizes a decomposition of the ground
truth itself: runs no model, uses no EnsCGP/Swin output.

Sanity check (printed, not plotted): asserts the bands sum back to the original WRF field
exactly for every sample plotted -- the same assertion multiscale_loss.py's self-test makes
on synthetic data, confirming the decomposition is lossless on real data too.

Display orientation: wrf_uv.npy renders south-up under plain imshow(origin="upper") (see
plot_enscgp_results.py's docstring); flipped along axis 0 below for display consistency,
matching that script.

Usage:
    python plot_multiscale_bands.py --n_samples 4 --seed 42
    python plot_multiscale_bands.py --sample_indices 0,1000,3000 --n_levels 6 --base_sigma 1.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPTS_DIR = "/home/peytonli/26.6_wind/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from new_enscgp_swin import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from multiscale_loss import band_sigmas, laplacian_bands  # noqa: E402

# WRF native 200x200 grid spacing (km/pixel) -- same source/derivation as
# band_error_diagnostics.py and compare_eigenspectra_enscgp_swin.py.
DY_KM = 4.44780
DX_KM = 3.48764
PIXEL_KM = float(np.sqrt(DX_KM * DY_KM))  # isotropic-equivalent pixel size for scale labels


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


def band_scale_labels(n_levels: int, base_sigma: float) -> list[str]:
    """Approximate physical scale range (km) each band's content sits in, finest -> coarsest
    (matching laplacian_bands' own return order)."""
    sig = band_sigmas(n_levels, base_sigma)
    labels = []
    for j in range(n_levels):
        lo_km = sig[j] * PIXEL_KM
        if j < n_levels - 1:
            hi_km = sig[j + 1] * PIXEL_KM
            labels.append(f"{lo_km:.0f}-{hi_km:.0f} km")
        else:
            labels.append(f">{lo_km:.0f} km (residual)")
    return labels


def plot_panel(ax, data: np.ndarray, cmap: str, vmin: float, vmax: float, title: str,
                cbar_label: str, lsm: np.ndarray | None = None) -> None:
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    if lsm is not None:
        ax.contour(lsm, levels=[0.5], colors="black", linewidths=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, label=cbar_label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                         help="new_enscgp_swin_config.json (for paths.data_dir and the multiscale n_levels/base_sigma)")
    parser.add_argument("--wrf_path", type=Path, default=None, help="Defaults to <data_dir>/wrf_uv.npy")
    parser.add_argument("--hires_land_mask_path", type=Path, default=None,
                         help="Defaults to <data_dir>/land_mask_hires.npz")
    parser.add_argument("--n_levels", type=int, default=None, help="Defaults to config's training.multiscale.n_levels")
    parser.add_argument("--base_sigma", type=float, default=None, help="Defaults to config's training.multiscale.base_sigma")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--sample_indices", type=str, default=None,
        help="Comma-separated sample indices (overrides --n_samples/--seed), e.g. 0,25,100",
    )
    parser.add_argument("--output", type=Path,
                         default=Path("/home/peytonli/26.6_wind/inference_results/multiscale_bands_panels.png"),
                         help="Output PNG path")
    parser.add_argument("--quiver_skip", type=int, default=10, help="Arrow subsampling on the full-field panel")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(config["paths"]["data_dir"])
    ms_cfg = config["training"]["multiscale"]
    n_levels = args.n_levels or ms_cfg["n_levels"]
    base_sigma = args.base_sigma if args.base_sigma is not None else ms_cfg["base_sigma"]

    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    hires_land_mask_path = args.hires_land_mask_path or (data_dir / "land_mask_hires.npz")

    wrf = np.load(wrf_path, mmap_mode="r")
    idx = choose_indices(wrf.shape[0], args.n_samples, args.seed, args.sample_indices)

    hires_masks = np.load(hires_land_mask_path)
    lsm_wrf = hires_masks["wrf"].astype(np.float64)

    labels = band_scale_labels(n_levels, base_sigma)  # finest -> coarsest
    display_order = list(range(n_levels))[::-1]       # coarse (left) -> fine (right), house convention
    col_titles = ["WRF speed (full field)"] + [labels[b] for b in display_order]

    n_rows, n_cols = len(idx), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)

    max_recon_diff = 0.0
    with torch.no_grad():
        for row, i in enumerate(idx):
            field = torch.from_numpy(np.array(wrf[i, :2], dtype=np.float32)).unsqueeze(0)  # (1, 2, H, W)
            bands = laplacian_bands(field, n_levels, base_sigma)  # finest -> coarsest

            recon_diff = (sum(bands) - field).abs().max().item()
            max_recon_diff = max(max_recon_diff, recon_diff)

            full_u, full_v = field[0, 0].numpy(), field[0, 1].numpy()
            full_speed = speed(full_u, full_v)
            band_mags = [speed(b[0, 0].numpy(), b[0, 1].numpy()) for b in bands]

            panels = [
                (full_speed, "jet", 0.0, float(full_speed.max()), "m/s",
                 (full_u, full_v)),
            ]
            for b in display_order:
                mag = band_mags[b]
                panels.append((mag, "magma", 0.0, max(float(mag.max()), 1e-12), "band magnitude (m/s)", None))

            for col, (data, cmap, vmin, vmax, cbar_label, uv) in enumerate(panels):
                title = f"{col_titles[col]} | idx={i}" if col == 0 else col_titles[col]
                plot_panel(axes[row, col], data, cmap, vmin, vmax, title, cbar_label, lsm=lsm_wrf)
                if uv is not None:
                    u, v = uv
                    skip = max(1, args.quiver_skip)
                    ys, xs = np.arange(0, u.shape[0], skip), np.arange(0, u.shape[1], skip)
                    X, Y = np.meshgrid(xs, ys)
                    # V negated: imshow(origin="upper") inverts the y-axis (see
                    # plot_enscgp_results.py's docstring for the same convention).
                    axes[row, col].quiver(X, Y, u[::skip, ::skip], -v[::skip, ::skip],
                                          color="black", scale_units="xy")

    assert max_recon_diff < 1e-3, f"Laplacian bands do not sum to the field, max abs diff {max_recon_diff}"

    fig.suptitle(
        f"WRF wind field Laplacian-pyramid band decomposition (n_levels={n_levels}, base_sigma={base_sigma}px)",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"Samples used: {idx.tolist()}")
    print(f"n_levels={n_levels}, base_sigma={base_sigma}px; bands sum back to the original field "
          f"(max abs diff {max_recon_diff:.2e} across all samples) OK")


if __name__ == "__main__":
    main()
