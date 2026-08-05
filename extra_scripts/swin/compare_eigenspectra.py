#!/usr/bin/env python3
"""
Compare eigenspectra (radial-averaged power spectral density) of four pipeline
stages for the EnsCGP -> ProbabilisticSwin2SR wind-downscaling pipeline:

1. ERA5 bicubic interpolation (naive upsampling baseline; data/era5_uv_2ch_bicubic.npy)
2. EnsCGP posterior mean (first guess fed to the model; data/enscgp_posterior.npy)
3. ProbabilisticSwin2SR model output -- the CENTRAL field (q50_u, q50_v) from a trained
   0714+ quantile checkpoint
4. WRF (200x200 ground truth; data/wrf_uv.npy)

Despite the name, this script takes no eigenvalues of any covariance: "eigenspectra" here
means the radial-averaged PSD of the u/v FIELDS. It therefore survived the 0714 switch from
the Gaussian/Cholesky head to quantiles unchanged in substance -- only which output channels
carry the central field moved. The model now emits 6 channels
[q10_u, q10_v, q50_u, q50_v, q90_u, q90_v]; the central field is the Q50 pair, which is the
direct analog of the old predicted mean (it is literally still mean_head's output, and it is
the field the structural losses train).

The q10/q90 envelopes are deliberately NOT plotted: an uncertainty envelope is not a wind
field, so its spectrum has no "should match WRF" target (the same category error the loss in
train_new_enscgp_swin.py is careful to avoid). The question this figure asks is whether the
model's central field recovers the high-wavenumber content that bicubic/EnsCGP lack.

All four already live on the same 200x200 grid (no upsampling step needed, unlike
26.3_wind/SWIN/evaluation/compare_eigenspectra_downscaled.py, which this mirrors --
radial_psd is ported from there, reworked to bin in physical km rather than pixel
bins).

The WRF grid is a regular lat/lon grid, not an isotropic map projection, so its
north-south and east-west pixel spacing differ (see DY_KM/DX_KM below) -- radial_psd
accounts for this by building the 2D frequency grid from per-axis cycles/km (via
np.fft.fftfreq(..., d=spacing_km)) rather than treating pixel-index distance as
isotropic. Plots default to wavelength (km) on the x-axis (--x_units wavenumber_per_km
for cycles/km instead).

Spectra across --n_samples are reduced to a mean +/- uncertainty band per source, with
u and v plotted as separate panels in one figure. --band controls how the band is
computed: "std" (default) takes mean +/- 1 std of log10(power) across samples then
maps back to linear power (a multiplicative band, appropriate since power spans
decades and is strictly positive); "pct" instead uses the mean plus
--pct_lower/--pct_upper percentiles directly in linear power.

Sample selection mirrors eval_checkpoint.py: by default drawn from the
held-out test split (data/splits_70_15_15/split_indices.npz); --sample_indices bypasses
split filtering with explicit raw indices.

Usage:
    python compare_eigenspectra.py --checkpoint runs/0714/checkpoints/best.pth
    python compare_eigenspectra.py --checkpoint .../best.pth --n_samples 32
    python compare_eigenspectra.py --checkpoint .../best.pth --sample_indices 12,4081,6500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch


from _common import (  # noqa: E402  (shared harness; also puts scripts/ on sys.path)
    add_eval_args, choose_indices, predict, setup, split_pool, split_quantiles,
)

SOURCES = ("ERA5 bicubic", "EnsCGP posterior", "SWIN output (q50)", "WRF (truth)")
COLORS = {"ERA5 bicubic": "purple", "EnsCGP posterior": "C0", "SWIN output (q50)": "green", "WRF (truth)": "C1"}
STYLES = {"ERA5 bicubic": "--", "EnsCGP posterior": "--", "SWIN output (q50)": "-", "WRF (truth)": "-"}
COMPONENTS = ("u", "v")

# WRF native 200x200 grid spacing (km/pixel), computed via haversine distance between
# adjacent wrflats/wrflons cell centers (/net/momo/data/projects/downscaling/datasource/
# wndata.mat -- same source and method as extra_scripts/build_hires_features.py's
# _haversine_km). The grid has uniform spacing in degrees, so it is uniform in km along
# each axis individually, but the two axes differ: at this domain's ~38-46N latitude
# band, a degree of longitude (DX_KM, east-west, axis 1) spans less physical distance
# than a degree of latitude (DY_KM, north-south, axis 0).
DY_KM = 4.44780
DX_KM = 3.48764


def radial_psd(img: np.ndarray, dx_km: float, dy_km: float, nbins: int | None = None):
    """Radial-averaged 2D power spectral density of a 2D image, binned in true physical
    cycles/km (not pixel distance) so that dx_km != dy_km is handled correctly.

    Returns (k_km, psd): k_km are radial wavenumber bin centers in cycles/km, with
    k_km[0] the lowest-frequency bin (includes the DC component -- exclude it before
    plotting on a log axis, same as the original pixel-bin version).
    """
    assert img.ndim == 2
    H, W = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    psd2d = np.abs(fshift) ** 2

    fy = np.fft.fftshift(np.fft.fftfreq(H, d=dy_km))  # cycles/km, axis 0 (north-south)
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=dx_km))  # cycles/km, axis 1 (east-west)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    k_phys = np.hypot(fx_grid, fy_grid).ravel()
    p_flat = psd2d.ravel()

    if nbins is None:
        nbins = min(H, W) // 2

    # Cap the radial range at the smaller axis's Nyquist frequency -- the largest disc
    # fully inscribed in the sampled (rectangular, anisotropic) frequency grid -- and
    # lump anything beyond it (the sparsely-sampled rectangle corners) into the last
    # bin, mirroring how the original pixel-distance version capped r_bin at nbins-1
    # instead of extending bins out to the (corner-only) diagonal.
    k_cap = min(fx.max(), fy.max())
    bin_width = k_cap / nbins
    bin_idx = np.minimum((k_phys / bin_width).astype(int), nbins - 1)

    tbin = np.bincount(bin_idx, weights=p_flat, minlength=nbins)
    nr = np.bincount(bin_idx, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        radial_mean = tbin / np.maximum(nr, 1)
    k_km = (np.arange(nbins) + 0.5) * bin_width
    return k_km, radial_mean


def x_axis_values(k_km: np.ndarray, x_units: str) -> np.ndarray:
    if x_units == "wavenumber_per_km":
        return k_km
    return 1.0 / k_km  # wavelength_km


def x_axis_label(x_units: str) -> str:
    return "Radial wavenumber (cycles/km)" if x_units == "wavenumber_per_km" else "Wavelength (km)"


def maybe_invert_xaxis(ax, x_units: str) -> None:
    # Wavelength decreases as wavenumber increases; invert so large scales (km) are on
    # the left and small scales on the right, matching the original low-k-on-the-left
    # layout instead of flipping it.
    if x_units == "wavelength_km":
        ax.invert_xaxis()


def _format_tick(x: float, _pos=None) -> str:
    """Plain-number tick label (no 10^n notation), trimmed of trailing zeros."""
    if x <= 0:
        return ""
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.4f}".rstrip("0").rstrip(".")


def style_log_xaxis(ax) -> None:
    """Label more than just whole decades on a log-scale x-axis: major ticks (with
    plain-number labels) at fractions of each decade instead of only 10^n."""
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_tick))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="major", labelsize=7, labelrotation=45)


def compute_spectra(idx: np.ndarray, bicubic, posterior, wrf, pred_batch: np.ndarray, dx_km: float, dy_km: float):
    """Return (k_km, spectra) where spectra maps component ('u'|'v') -> {source: (n_samples, L) array}.
    pred_batch is the model's CENTRAL field only, (B, 2, H, W) = [q50_u, q50_v] -- sliced in
    main() via Q50_SLICE, so comp_idx indexes u/v here (not the 6-channel raw output)."""
    spectra = {comp: {s: [] for s in SOURCES} for comp in COMPONENTS}
    k_common = None
    for row, i in enumerate(idx):
        for comp_idx, comp in enumerate(COMPONENTS):
            bicubic_field = np.asarray(bicubic[int(i), comp_idx])
            ens_field = np.asarray(posterior[int(i), comp_idx])
            wrf_field = np.asarray(wrf[int(i), comp_idx])
            pred_field = pred_batch[row, comp_idx]

            k_b, s_b = radial_psd(bicubic_field, dx_km, dy_km)
            k_e, s_e = radial_psd(ens_field, dx_km, dy_km)
            k_p, s_p = radial_psd(pred_field, dx_km, dy_km)
            k_w, s_w = radial_psd(wrf_field, dx_km, dy_km)

            L = min(len(s_b), len(s_e), len(s_p), len(s_w))
            spectra[comp]["ERA5 bicubic"].append(s_b[:L])
            spectra[comp]["EnsCGP posterior"].append(s_e[:L])
            spectra[comp]["SWIN output (q50)"].append(s_p[:L])
            spectra[comp]["WRF (truth)"].append(s_w[:L])
            k_common = k_b[:L]

    return k_common, {comp: {s: np.vstack(v) for s, v in d.items()} for comp, d in spectra.items()}


def band_std_log(values: np.ndarray, eps: float = 1e-20):
    """Mean +/- 1 std of log10(power) across samples (axis 0), mapped back to linear
    power -- a multiplicative band, appropriate since power spans decades and is
    strictly positive."""
    log_v = np.log10(np.maximum(values, eps))
    mean_log, std_log = log_v.mean(axis=0), log_v.std(axis=0)
    center = 10 ** mean_log
    lower = 10 ** (mean_log - std_log)
    upper = 10 ** (mean_log + std_log)
    return center, lower, upper


def band_percentile(values: np.ndarray, pct_lower: float, pct_upper: float):
    center = values.mean(axis=0)
    lower = np.percentile(values, pct_lower, axis=0)
    upper = np.percentile(values, pct_upper, axis=0)
    return center, lower, upper


def plot_aggregate(idx: np.ndarray, k_km: np.ndarray, spectra: dict, band: str,
                    pct_lower: float, pct_upper: float, x_units: str, output: Path,
                    zoom_km: tuple[float, float] | None = None) -> None:
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(6.5 * len(COMPONENTS), 5.5))
    x = x_axis_values(k_km[1:], x_units)
    band_label = "mean log10(power) +/- 1 std" if band == "std" else f"mean, [{pct_lower:g},{pct_upper:g}] pct"

    for ax, comp in zip(axes, COMPONENTS):
        y_lowers, y_uppers = [], []
        for source in SOURCES:
            values = spectra[comp][source][:, 1:]
            if band == "std":
                center, lower, upper = band_std_log(values)
            else:
                center, lower, upper = band_percentile(values, pct_lower, pct_upper)
            ax.loglog(x, center, label=source, color=COLORS[source], linestyle=STYLES[source], linewidth=2)
            ax.fill_between(x, lower, upper, color=COLORS[source], alpha=0.2)
            y_lowers.append(lower)
            y_uppers.append(upper)

        maybe_invert_xaxis(ax, x_units)
        style_log_xaxis(ax)
        if zoom_km is not None:
            lo, hi = zoom_km
            if x_units == "wavelength_km":
                in_range = (x >= lo) & (x <= hi)
                ax.set_xlim(hi, lo)
            else:
                in_range = (x >= 1.0 / hi) & (x <= 1.0 / lo)
                ax.set_xlim(1.0 / hi, 1.0 / lo)
            if in_range.any():
                y_min = min(arr[in_range].min() for arr in y_lowers)
                y_max = max(arr[in_range].max() for arr in y_uppers)
                log_pad = 0.15
                ax.set_ylim(y_min * 10 ** (-log_pad), y_max * 10 ** log_pad)
        ax.set_xlabel(x_axis_label(x_units))
        ax.set_ylabel("Power")
        ax.set_title(f"{comp} component")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    fig.suptitle(f"Bicubic vs EnsCGP vs SWIN vs WRF eigenspectra ({band_label})\nn_samples={len(idx)}")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved aggregate eigenspectra comparison ({len(idx)} samples) to {output}")


def main() -> None:
    parser = add_eval_args(
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
        n_samples_default=8,
    )
    parser.add_argument("--dx_km", type=float, default=DX_KM, help="East-west grid spacing (km/pixel)")
    parser.add_argument("--dy_km", type=float, default=DY_KM, help="North-south grid spacing (km/pixel)")
    parser.add_argument("--x_units", type=str, default="wavelength_km", choices=["wavelength_km", "wavenumber_per_km"])
    parser.add_argument("--band", type=str, default="std", choices=["std", "pct"], help="Uncertainty band style")
    parser.add_argument("--pct_lower", type=float, default=10.0, help="Lower percentile when --band pct")
    parser.add_argument("--pct_upper", type=float, default=90.0, help="Upper percentile when --band pct")
    parser.add_argument("--zoom_km", type=float, nargs=2, default=None, metavar=("LO_KM", "HI_KM"),
                        help="Zoom x-axis to this wavelength range in km, e.g. --zoom_km 10 70")
    parser.add_argument("--output", type=Path, default=None,
                         help="Output PNG. Defaults to <log_dir>/figures/eigenspectra_aggregate_swin.png")
    args = parser.parse_args()

    ev = setup(args)
    wrf, posterior, bicubic = ev.arrays.wrf, ev.arrays.posterior, ev.arrays.bicubic
    output_path = args.output or ev.figure_path("eigenspectra_aggregate_swin.png")

    idx = choose_indices(split_pool(ev.splits_path, args.split), ev.arrays.n_total,
                         args.n_samples, args.seed, args.sample_indices)

    # (B, 6, H, W): [q10_u, q10_v, q50_u, q50_v, q90_u, q90_v]. The CENTRAL field is the
    # analog of the old predicted mean and must be taken via split_quantiles/Q50_SLICE:
    # channels 0-1 are q10 under the quantile head, so raw index 0/1 would silently plot the
    # lower envelope's spectrum as "SWIN output".
    _q10, q50_batch, _q90 = split_quantiles(predict(ev, idx))  # q50: (B, 2, H, W)

    k_km, spectra = compute_spectra(idx, bicubic, posterior, wrf, q50_batch, args.dx_km, args.dy_km)
    plot_aggregate(idx, k_km, spectra, args.band, args.pct_lower, args.pct_upper, args.x_units, output_path,
                   zoom_km=args.zoom_km)

    print(f"Checkpoint: {ev.describe_checkpoint()}")
    print(f"Split: {args.split}")
    print(f"Samples used ({len(idx)}): {idx.tolist()}")


if __name__ == "__main__":
    main()
