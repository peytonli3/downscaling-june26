"""
Visualize SWIN (ProbabilisticSwin2SR) prediction error decomposed by spatial-frequency
band, for a few held-out samples. u and v are shown SEPARATELY (not combined into a
magnitude) -- each sample gets two rows, "u" then "v", 7 panels each (14 per sample):

  1) SWIN posterior-mean <component> (200x200) -- the result
  2) error <component> total: SWIN-minus-WRF, signed
  3..7) the SAME error split into Laplacian-pyramid bands, coarse -> fine: each panel
       shows that band's signed error for THIS component, labeled with its Gaussian-sigma
       scale range in km.

(The WRF ground-truth field itself isn't plotted here -- see
eval_new_enscgp_swin_checkpoint.py / plot_enscgp_results.py for truth-vs-prediction
panels; this script is specifically about the error and its scale decomposition.)

The band decomposition reuses band_error_diagnostics.py (laplacian_bands / band_sigmas /
scale_labels): a non-decimated Laplacian pyramid where the bands of the error sum back to
the full error exactly (the decomposition is linear and per-channel, so band(pred-truth)
== band(pred) - band(truth), applied to u and v independently). Per-band error energy can
span ~2 orders of magnitude (see band_error_diagnostics.py); rather than rescaling each
band panel independently (which would hide that imbalance), all band panels for a given
(sample, component) row share one symmetric color scale [-m, m], m = max |error| over that
row's bands -- so panel color is directly comparable across scales, showing which scale
actually dominates that row's error. Each band panel's title still prints its own actual
peak |error| alongside the shared scale. The scale is per-row (not shared across rows or
samples), since different samples/components can have very different overall error
magnitude. The "error total" panel keeps its own independent symmetric scale (it is the
sum of the bands, not one of "the scales" being compared).

Error is signed (SWIN - WRF, can be +/-), so error panels use a diverging colormap
(RdBu_r) centered at 0; the SWIN result panel is likewise a signed wind component, with
its own scale (RdBu_r, +/- its own max). No quiver arrows here (direction is implicit in
the u/v split, not the point of this plot) -- see eval_new_enscgp_swin_checkpoint.py /
plot_enscgp_results.py for direction-quiver speed plots.

Model loading, sample selection, and the display orientation follow
eval_new_enscgp_swin_checkpoint.py: builds a ProbabilisticSwin2SR from
new_enscgp_swin_config.json, loads --checkpoint's model_state_dict, runs
model(posterior, bicubic, terrain) for mu_u/mu_v, draws from the held-out test split by
default, overlays the land/sea contour, and flips the WRF-grid panels north-up for display
(see that script's docstring for the flip convention).

Usage:
    python plot_band_errors_swin.py --checkpoint /home/peytonli/26.6_wind/logs/checkpoints/best.pth
    python plot_band_errors_swin.py --checkpoint .../best.pth --n_samples 4 --n_levels 5 --base_sigma 2
    python plot_band_errors_swin.py --checkpoint .../best.pth --sample_indices 12,4081,6500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPTS_DIR = "/net/flood/home/peytonli/26.6_wind/scripts"
EXTRA_SCRIPTS_DIR = "/net/flood/home/peytonli/26.6_wind/extra_scripts"
for _d in (SCRIPTS_DIR, EXTRA_SCRIPTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from new_enscgp_swin import DEFAULT_CONFIG_PATH, build_model, load_config  # noqa: E402
from terrain_encoder import load_terrain_input  # noqa: E402
# Band decomposition + km scale labels, reused verbatim from the diagnostic so this plot
# and band_error_diagnostics.py decompose the field identically.
from band_error_diagnostics import laplacian_bands, scale_labels  # noqa: E402

COMPONENTS = ("u", "v")


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


def sym_max(*arrays: np.ndarray, floor: float = 1e-12) -> float:
    return max(max(float(np.abs(a).max()) for a in arrays), floor)


def plot_panel(ax, data, cmap, vmin, vmax, title, cbar_label, lsm=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    if lsm is not None:
        ax.contour(lsm, levels=[0.5], colors="black", linewidths=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, label=cbar_label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="new_enscgp_swin_config.json (architecture + default paths)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth from --config")
    parser.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --config")
    parser.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --config")
    parser.add_argument("--wrf_path", type=Path, default=None, help="Defaults to <data_dir>/wrf_uv.npy")
    parser.add_argument("--posterior_path", type=Path, default=None, help="Defaults to <data_dir>/enscgp_posterior.npy")
    parser.add_argument("--bicubic_path", type=Path, default=None, help="Defaults to <data_dir>/era5_uv_2ch_bicubic.npy")
    parser.add_argument("--hires_land_mask_path", type=Path, default=None, help="Defaults to <data_dir>/land_mask_hires.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Sample pool for default random selection")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--sample_indices", type=str, default=None,
        help="Comma-separated raw sample indices (overrides --split/--n_samples/--seed), e.g. 0,25,100",
    )
    parser.add_argument("--n_levels", type=int, default=5, help="Laplacian pyramid levels (bands)")
    parser.add_argument("--base_sigma", type=float, default=2.0,
                        help="Finest Gaussian blur sigma (px); default 2 matches the training config's multiscale bands")
    parser.add_argument("--output", type=Path,
                        default=Path("/home/peytonli/26.6_wind/inference_results/0701/swin_band_errors.png"),
                        help="Output PNG path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = args.data_dir or Path(paths["data_dir"])
    checkpoint_path = args.checkpoint or (Path(paths["log_dir"]) / "checkpoints" / "best.pth")
    splits_path = args.splits_path or Path(paths["splits_path"])
    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    posterior_path = args.posterior_path or (data_dir / "enscgp_posterior.npy")
    bicubic_path = args.bicubic_path or (data_dir / "era5_uv_2ch_bicubic.npy")
    hires_land_mask_path = args.hires_land_mask_path or (data_dir / "land_mask_hires.npz")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device)

    wrf = np.load(wrf_path, mmap_mode="r")
    posterior = np.load(posterior_path, mmap_mode="r")
    bicubic = np.load(bicubic_path, mmap_mode="r")
    n_total = min(wrf.shape[0], posterior.shape[0], bicubic.shape[0])

    splits = np.load(splits_path)
    pool = splits[f"{args.split}_idx"]
    idx = choose_indices(pool, n_total, args.n_samples, args.seed, args.sample_indices)

    hires_masks = np.load(hires_land_mask_path)
    lsm_wrf = hires_masks["wrf"].astype(np.float64)

    posterior_batch = torch.from_numpy(np.array(posterior[idx], dtype=np.float32, copy=True)).to(device)
    bicubic_batch = torch.from_numpy(np.array(bicubic[idx], dtype=np.float32, copy=True)).to(device)
    with torch.no_grad():
        pred_batch = model(posterior_batch, bicubic_batch, terrain_raw).cpu().numpy()  # (B, 5, 200, 200)

    labels = scale_labels(args.n_levels, args.base_sigma)         # finest -> coarsest
    display_order = list(range(args.n_levels))[::-1]              # coarse (left) -> fine (right)
    band_titles = [f"err {labels[b]} km" for b in display_order]
    col_titles = ["SWIN mean", "error total"] + band_titles
    n_rows, n_cols = len(idx) * len(COMPONENTS), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)

    max_recon_diff = 0.0
    for s, i in enumerate(idx):
        wrf_uv = np.asarray(wrf[i, :2], dtype=np.float64)      # (2, H, W)
        pred_uv = pred_batch[s, :2].astype(np.float64)         # (2, H, W)

        err = pred_uv - wrf_uv                                          # (2, H, W), signed
        err_bands = laplacian_bands(err, args.n_levels, args.base_sigma)  # (n_levels, 2, H, W)
        max_recon_diff = max(max_recon_diff, float(np.abs(err_bands.sum(axis=0) - err).max()))

        for c, comp in enumerate(COMPONENTS):
            row = s * len(COMPONENTS) + c
            pred_c, err_c = pred_uv[c], err[c]
            band_c = err_bands[:, c]                       # (n_levels, H, W)
            band_peaks = [float(np.abs(band_c[b]).max()) for b in range(args.n_levels)]
            # Uniform symmetric color scale across THIS row's band panels (see docstring).
            band_vmax = sym_max(*[band_c[b] for b in range(args.n_levels)])
            field_vmax = sym_max(pred_c)
            err_total_vmax = sym_max(err_c)

            panels = [
                (pred_c, "RdBu_r", -field_vmax, field_vmax, "m/s", lsm_wrf),
                (err_c, "RdBu_r", -err_total_vmax, err_total_vmax, "m/s", lsm_wrf),
            ]
            for b in display_order:
                mag = band_c[b]
                panels.append((mag, "RdBu_r", -band_vmax, band_vmax, "m/s", lsm_wrf))

            for col, (data, cmap, vmin, vmax, cbar_label, lsm) in enumerate(panels):
                base = col_titles[col]
                if col == 0:
                    title = f"{base} {comp} | idx={i}"
                elif col >= 2:
                    b = display_order[col - 2]
                    title = f"{base} {comp} (peak {band_peaks[b]:.2f}, scale {vmax:.2f})"
                else:
                    title = f"{base} {comp}"
                plot_panel(axes[row, col], data, cmap, vmin, vmax, title, cbar_label, lsm=lsm)

    assert max_recon_diff < 1e-3, f"Laplacian error bands do not sum to the error field (max abs diff {max_recon_diff})"

    fig.suptitle(
        f"SWIN prediction error by frequency band, u/v separate (split={args.split}, ckpt={checkpoint_path.name}, "
        f"n_levels={args.n_levels}, base_sigma={args.base_sigma}px)",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"Checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')})")
    print(f"Split: {args.split}; samples used: {idx.tolist()}")
    print(f"Band scale ranges (coarse -> fine, km): {[labels[b] for b in display_order]}")
    print(f"Error bands sum back to the error field (max abs diff {max_recon_diff:.2e}) OK")


if __name__ == "__main__":
    main()
