#!/usr/bin/env python3
"""
Compare eigenspectra (radial-averaged power spectral density) of four pipeline
stages for the EnsCGP -> ProbabilisticSwin2SR wind-downscaling pipeline,
using the 0626_v1 checkpoint (old model architecture: 37-channel conv_first,
32-channel terrain encoder, no mean_gate/chol_gate, no bicubic input).

1. ERA5 bicubic interpolation (naive upsampling baseline; data/era5_uv_2ch_bicubic.npy)
2. EnsCGP posterior mean (first guess fed to the model; data/enscgp_posterior.npy)
3. ProbabilisticSwin2SR model output (mu_u, mu_v from the 0626_v1 checkpoint)
4. WRF (200x200 ground truth; data/wrf_uv.npy)

Uses scripts/0626_v1/new_enscgp_swin.py (forward(posterior, terrain_raw) -- no bicubic)
and scripts/0626_v1/terrain_encoder_32.py (32-channel terrain encoder).

See extra_scripts/graphing/swin/compare_eigenspectra_enscgp_swin.py for the
current-model version; this copy exists because the 0626_v1 checkpoint has a
different architecture (37 vs 11 input channels, shorter terrain encoder, no gates)
that is incompatible with the current model code.

Usage:
    python compare_eigenspectra_enscgp_swin.py
    python compare_eigenspectra_enscgp_swin.py --checkpoint /home/peytonli/26.6_wind/logs/0626_v1/checkpoints/best.pth
    python compare_eigenspectra_enscgp_swin.py --n_samples 32 --zoom_km 10 70
    python compare_eigenspectra_enscgp_swin.py --sample_indices 12,4081,6500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

# Old model + terrain_encoder_32 live in scripts/0626_v1; network_swin2sr.py lives
# in the parent scripts/ dir (unchanged, shared between model versions).
_OLD_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "0626_v1"
_SCRIPTS_DIR = _OLD_SCRIPTS_DIR.parent
for _d in (str(_SCRIPTS_DIR), str(_OLD_SCRIPTS_DIR)):  # insert parent first so 0626_v1 ends up at position 0
    if _d not in sys.path:
        sys.path.insert(0, _d)

from new_enscgp_swin import DEFAULT_CONFIG_PATH, build_model, load_config  # noqa: E402
from terrain_encoder import load_terrain_input  # noqa: E402

SOURCES = ("ERA5 bicubic", "EnsCGP posterior", "SWIN v1 output", "WRF (truth)")
COLORS = {"ERA5 bicubic": "purple", "EnsCGP posterior": "C0", "SWIN v1 output": "green", "WRF (truth)": "C1"}
STYLES = {"ERA5 bicubic": "--", "EnsCGP posterior": "--", "SWIN v1 output": "-", "WRF (truth)": "-"}
COMPONENTS = ("u", "v")

# WRF native 200x200 grid spacing (km/pixel).
DY_KM = 4.44780
DX_KM = 3.48764

DEFAULT_CHECKPOINT = Path("/home/peytonli/26.6_wind/logs/0626_v1/checkpoints/best.pth")
DEFAULT_OUTPUT = Path("/home/peytonli/26.6_wind/inference_results/0626_v1/eigenspectra_aggregate_swin.png")


def radial_psd(img: np.ndarray, dx_km: float, dy_km: float, nbins: int | None = None):
    assert img.ndim == 2
    H, W = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    psd2d = np.abs(fshift) ** 2

    fy = np.fft.fftshift(np.fft.fftfreq(H, d=dy_km))
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=dx_km))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    k_phys = np.hypot(fx_grid, fy_grid).ravel()
    p_flat = psd2d.ravel()

    if nbins is None:
        nbins = min(H, W) // 2

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
    return 1.0 / k_km


def x_axis_label(x_units: str) -> str:
    return "Radial wavenumber (cycles/km)" if x_units == "wavenumber_per_km" else "Wavelength (km)"


def maybe_invert_xaxis(ax, x_units: str) -> None:
    if x_units == "wavelength_km":
        ax.invert_xaxis()


def _format_tick(x: float, _pos=None) -> str:
    if x <= 0:
        return ""
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.4f}".rstrip("0").rstrip(".")


def style_log_xaxis(ax) -> None:
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0, 7.0)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_tick))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
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
            spectra[comp]["SWIN v1 output"].append(s_p[:L])
            spectra[comp]["WRF (truth)"].append(s_w[:L])
            k_common = k_b[:L]

    return k_common, {comp: {s: np.vstack(v) for s, v in d.items()} for comp, d in spectra.items()}


def band_std_log(values: np.ndarray, eps: float = 1e-20):
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

    fig.suptitle(f"Bicubic vs EnsCGP vs SWIN v1 vs WRF eigenspectra ({band_label})\nn_samples={len(idx)}")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved aggregate eigenspectra comparison ({len(idx)} samples) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --config")
    parser.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --config")
    parser.add_argument("--wrf_path", type=Path, default=None, help="Defaults to <data_dir>/wrf_uv.npy")
    parser.add_argument("--posterior_path", type=Path, default=None, help="Defaults to <data_dir>/enscgp_posterior.npy")
    parser.add_argument("--bicubic_path", type=Path, default=None, help="Defaults to <data_dir>/era5_uv_2ch_bicubic.npy (used for ERA5 bicubic spectra only, not fed to the model)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated raw sample indices (overrides --split/--n_samples/--seed)")
    parser.add_argument("--dx_km", type=float, default=DX_KM)
    parser.add_argument("--dy_km", type=float, default=DY_KM)
    parser.add_argument("--x_units", type=str, default="wavelength_km", choices=["wavelength_km", "wavenumber_per_km"])
    parser.add_argument("--band", type=str, default="std", choices=["std", "pct"])
    parser.add_argument("--pct_lower", type=float, default=10.0)
    parser.add_argument("--pct_upper", type=float, default=90.0)
    parser.add_argument("--zoom_km", type=float, nargs=2, default=None, metavar=("LO_KM", "HI_KM"),
                        help="Zoom x-axis to this wavelength range in km, e.g. --zoom_km 10 70")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = args.data_dir or Path(paths["data_dir"])
    splits_path = args.splits_path or Path(paths["splits_path"])
    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    posterior_path = args.posterior_path or (data_dir / "enscgp_posterior.npy")
    bicubic_path = args.bicubic_path or (data_dir / "era5_uv_2ch_bicubic.npy")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
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
        pred_batch = model(posterior_batch, terrain_raw).cpu().numpy()  # (B, 5, H, W)

    k_km, spectra = compute_spectra(idx, bicubic, posterior, wrf, pred_batch, args.dx_km, args.dy_km)
    plot_aggregate(idx, k_km, spectra, args.band, args.pct_lower, args.pct_upper, args.x_units, args.output,
                   zoom_km=args.zoom_km)

    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')})")
    print(f"Split: {args.split}")
    print(f"Samples used ({len(idx)}): {idx.tolist()}")


if __name__ == "__main__":
    main()
