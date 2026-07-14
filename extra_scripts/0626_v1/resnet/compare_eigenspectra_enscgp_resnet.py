#!/usr/bin/env python3
"""
Compare eigenspectra (radial-averaged power spectral density) of four pipeline
stages for the EnsCGP -> ResNetRefiner wind-downscaling pipeline:

1. ERA5 bicubic interpolation (naive upsampling baseline; data/era5_uv_2ch_bicubic.npy)
2. EnsCGP posterior mean (first guess fed to the model; data/enscgp_posterior.npy)
3. ResNetRefiner model output (mu_u, mu_v from a trained checkpoint)
4. WRF (200x200 ground truth; data/wrf_uv.npy)

All four already live on the same 200x200 grid (no upsampling step needed, unlike
26.3_wind/SWIN/evaluation/compare_eigenspectra_downscaled.py, which this mirrors --
radial_psd is ported from there, reworked to bin in physical km rather than pixel
bins). This is the same comparison this project runs against the Swin refiner (see
extra_scripts/graphing/swin/compare_eigenspectra_enscgp_swin.py), so the two models'
spectra are directly comparable.

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

Sample selection mirrors eval_resnet_refiner_checkpoint.py: by default drawn from the
held-out test split (data/splits_70_15_15/split_indices.npz); --sample_indices bypasses
split filtering with explicit raw indices.

Usage:
    python compare_eigenspectra_enscgp_resnet.py --checkpoint /home/peytonli/26.6_wind/logs/resnet_refiner/checkpoints/best.pth
    python compare_eigenspectra_enscgp_resnet.py --checkpoint .../best.pth --n_samples 32
    python compare_eigenspectra_enscgp_resnet.py --checkpoint .../best.pth --sample_indices 12,4081,6500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from resnet_refiner import DEFAULT_CONFIG_PATH, build_model, load_config  # noqa: E402
from terrain_encoder import load_terrain_input  # noqa: E402

SOURCES = ("ERA5 bicubic", "EnsCGP posterior", "ResNet output", "WRF (truth)")
COLORS = {"ERA5 bicubic": "purple", "EnsCGP posterior": "C0", "ResNet output": "green", "WRF (truth)": "C1"}
STYLES = {"ERA5 bicubic": "--", "EnsCGP posterior": "--", "ResNet output": "-", "WRF (truth)": "-"}
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
    ax.tick_params(axis="x", which="major", labelsize=7, labelrotation=45)


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


def compute_spectra(idx: np.ndarray, bicubic, posterior, wrf, pred_batch: np.ndarray, dx_km: float, dy_km: float):
    """Return (k_km, spectra) where spectra maps component ('u'|'v') -> {source: (n_samples, L) array}."""
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
            spectra[comp]["ResNet output"].append(s_p[:L])
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
                    pct_lower: float, pct_upper: float, x_units: str, output: Path) -> None:
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(6.5 * len(COMPONENTS), 5.5))
    x = x_axis_values(k_km[1:], x_units)
    band_label = "mean log10(power) +/- 1 std" if band == "std" else f"mean, [{pct_lower:g},{pct_upper:g}] pct"

    for ax, comp in zip(axes, COMPONENTS):
        for source in SOURCES:
            values = spectra[comp][source][:, 1:]
            if band == "std":
                center, lower, upper = band_std_log(values)
            else:
                center, lower, upper = band_percentile(values, pct_lower, pct_upper)
            ax.loglog(x, center, label=source, color=COLORS[source], linestyle=STYLES[source], linewidth=2)
            ax.fill_between(x, lower, upper, color=COLORS[source], alpha=0.2)

        maybe_invert_xaxis(ax, x_units)
        style_log_xaxis(ax)
        ax.set_xlabel(x_axis_label(x_units))
        ax.set_ylabel("Power")
        ax.set_title(f"{comp} component")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    fig.suptitle(f"Bicubic vs EnsCGP vs ResNet vs WRF eigenspectra ({band_label})\nn_samples={len(idx)}")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved aggregate eigenspectra comparison ({len(idx)} samples) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth from --config")
    parser.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --config")
    parser.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --config")
    parser.add_argument("--wrf_path", type=Path, default=None, help="Defaults to <data_dir>/wrf_uv.npy")
    parser.add_argument("--posterior_path", type=Path, default=None, help="Defaults to <data_dir>/enscgp_posterior.npy")
    parser.add_argument("--bicubic_path", type=Path, default=None, help="Defaults to <data_dir>/era5_uv_2ch_bicubic.npy")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Sample pool for default random selection")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples to draw (also the aggregate sample size)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_indices", type=str, default=None,
        help="Comma-separated raw sample indices (overrides --split/--n_samples/--seed), e.g. 0,25,100",
    )
    parser.add_argument("--dx_km", type=float, default=DX_KM, help="East-west grid spacing (km/pixel)")
    parser.add_argument("--dy_km", type=float, default=DY_KM, help="North-south grid spacing (km/pixel)")
    parser.add_argument("--x_units", type=str, default="wavelength_km", choices=["wavelength_km", "wavenumber_per_km"])
    parser.add_argument("--band", type=str, default="std", choices=["std", "pct"], help="Uncertainty band style")
    parser.add_argument("--pct_lower", type=float, default=10.0, help="Lower percentile when --band pct")
    parser.add_argument("--pct_upper", type=float, default=90.0, help="Upper percentile when --band pct")
    parser.add_argument("--output", type=Path,
                         default=Path("/home/peytonli/26.6_wind/inference_results/eigenspectra_aggregate_resnet.png"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = args.data_dir or Path(paths["data_dir"])
    checkpoint_path = args.checkpoint or (Path(paths["log_dir"]) / "checkpoints" / "best.pth")
    splits_path = args.splits_path or Path(paths["splits_path"])
    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    posterior_path = args.posterior_path or (data_dir / "enscgp_posterior.npy")
    bicubic_path = args.bicubic_path or (data_dir / "era5_uv_2ch_bicubic.npy")

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

    posterior_batch = torch.from_numpy(np.array(posterior[idx], dtype=np.float32, copy=True)).to(device)
    with torch.no_grad():
        pred_batch = model(
            posterior_batch, posterior_batch[:, :2], posterior_batch[:, 2:5], terrain_raw
        ).cpu().numpy()  # (B, 5, H, W): mu_u, mu_v, L11, L21, L22

    k_km, spectra = compute_spectra(idx, bicubic, posterior, wrf, pred_batch, args.dx_km, args.dy_km)
    plot_aggregate(idx, k_km, spectra, args.band, args.pct_lower, args.pct_upper, args.x_units, args.output)

    print(f"Checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')})")
    print(f"Split: {args.split}")
    print(f"Samples used ({len(idx)}): {idx.tolist()}")


if __name__ == "__main__":
    main()
