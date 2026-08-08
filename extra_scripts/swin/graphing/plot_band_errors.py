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
eval_checkpoint.py / plot_enscgp_results.py for truth-vs-prediction
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
the u/v split, not the point of this plot) -- see eval_checkpoint.py /
plot_enscgp_results.py for direction-quiver speed plots.

Model loading, sample selection, and the display orientation follow
eval_checkpoint.py: builds a ProbabilisticSwin2SR from
new_enscgp_swin_config.json, loads --checkpoint's model_state_dict, runs
model(posterior, bicubic, terrain) for mu_u/mu_v, draws from the held-out test split by
default, overlays the land/sea contour, and flips the WRF-grid panels north-up for display
(see that script's docstring for the flip convention).

Usage:
    python plot_band_errors.py --checkpoint runs/<version>/checkpoints/best.pth
    python plot_band_errors.py --checkpoint .../best.pth --n_samples 4 --n_levels 5 --base_sigma 2
    python plot_band_errors.py --checkpoint .../best.pth --sample_indices 12,4081,6500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
for _d in (REPO / "scripts", REPO / "extra_scripts" / "swin",
           REPO / "extra_scripts" / "swin" / "oneoff"):
    sys.path.insert(0, str(_d))

from paths import run_figures  # noqa: E402
from _common import (  # noqa: E402  (shared harness; also puts scripts/ on sys.path)
    add_eval_args, choose_indices, plot_panel, predict, setup, split_pool, split_quantiles,
    sym_max,
)
# Band decomposition + km scale labels, reused verbatim from the diagnostic so this plot
# and band_error_diagnostics.py decompose the field identically.
from band_error_diagnostics import laplacian_bands, scale_labels  # noqa: E402

COMPONENTS = ("u", "v")


def main() -> None:
    parser = add_eval_args(
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
    )
    parser.add_argument("--n_levels", type=int, default=5, help="Laplacian pyramid levels (bands)")
    parser.add_argument("--base_sigma", type=float, default=2.0,
                        help="Finest Gaussian blur sigma (px); default 2 matches the training config's multiscale bands")
    parser.add_argument("--output", type=Path,
                        default=run_figures("0629") / "swin_band_errors.png",
                        help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ev = setup(args)
    wrf = ev.arrays.wrf

    idx = choose_indices(split_pool(ev.splits_path, args.split), ev.arrays.n_total,
                         args.n_samples, args.seed, args.sample_indices)

    lsm_wrf = ev.arrays.wrf_land_mask()

    # q50 is the central field this plot is about ("SWIN mean" below). It MUST come out of
    # split_quantiles: under the 0714+ quantile head channels 0-1 are q10, so the
    # pre-0714 `pred[:, :2]` that used to sit here silently decomposed the LOWER ENVELOPE's
    # error and labelled it the mean.
    _q10, q50_batch, _q90 = split_quantiles(predict(ev, idx))  # q50: (B, 2, 200, 200)

    labels = scale_labels(args.n_levels, args.base_sigma)         # finest -> coarsest
    display_order = list(range(args.n_levels))[::-1]              # coarse (left) -> fine (right)
    band_titles = [f"err {labels[b]} km" for b in display_order]
    col_titles = ["SWIN mean", "error total"] + band_titles
    n_rows, n_cols = len(idx) * len(COMPONENTS), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)

    max_recon_diff = 0.0
    for s, i in enumerate(idx):
        wrf_uv = np.asarray(wrf[i, :2], dtype=np.float64)      # (2, H, W)
        pred_uv = q50_batch[s].astype(np.float64)              # (2, H, W)

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
        f"SWIN prediction error by frequency band, u/v separate (split={args.split}, ckpt={ev.checkpoint.name}, "
        f"n_levels={args.n_levels}, base_sigma={args.base_sigma}px)",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"Checkpoint: {ev.describe_checkpoint()}")
    print(f"Split: {args.split}; samples used: {idx.tolist()}")
    print(f"Band scale ranges (coarse -> fine, km): {[labels[b] for b in display_order]}")
    print(f"Error bands sum back to the error field (max abs diff {max_recon_diff:.2e}) OK")


if __name__ == "__main__":
    main()
