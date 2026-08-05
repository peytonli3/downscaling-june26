#!/usr/bin/env python3
"""
Per-band error-type diagnostic for wind downscaling: at each spatial scale, is the Swin
prediction error AMPLITUDE/POWER (right place, wrong strength) or PHASE/DISPLACEMENT
(right strength, wrong place)?

Hypothesis under test: error is phase/displacement-dominated at large scales (low freq)
and amplitude-dominated at fine scales (high freq).

Pipeline
--------
1. Predictions: load a trained ProbabilisticSwin2SR checkpoint (scripts/new_enscgp_swin.py)
   and run it on the held-out test split, exactly as eval_checkpoint.py
   does -- model(posterior, bicubic, terrain) -> mu_u/mu_v. Truth is wrf_uv.npy[:, :2].
   Both pred and truth are (2, 200, 200) [u, v] vector fields.

2. Scale decomposition: a non-decimated ("a trous") Laplacian pyramid -- blur the field
   with Gaussian sigmas doubling per level (base_sigma * 2^(k-1)) and take successive
   differences, the coarsest level being the residual low-pass. All bands stay at full
   200x200 and sum back to the original field exactly (asserted). Band 0 is the finest
   (highest freq); the last band is the coarsest (large-scale residual).

3. Per-band diagnostics on the (u, v) vector field (u and v pixels pooled together):
   - power_ratio = RMS(pred_band) / RMS(truth_band)         -- amplitude (eigenspectrum view)
   - pattern_corr = <pred,truth> / (RMS_pred RMS_truth)     -- placement (phase)
   - exact MSE split: MSE = (s_p - s_t)^2 + 2 s_p s_t (1 - corr)
                            \_ amplitude _/   \__ phase/displacement __/
     phase_fraction = phase_term / MSE   (->1 displacement-dominated, ->0 amplitude-dominated)
   - displacement: peak offset of the 2D vector cross-correlation (a systematic spatial
     shift at that scale), and whether it is SYSTEMATIC across samples (consistent
     direction -> learnable bias, "L1+pin it") or RANDOM (varies per sample -> irreducible,
     "Wasserstein-dominant").

   Note on the correlation: RMS (s) and the correlation are UNCENTERED (the field mean is
   not subtracted). This is what makes the MSE = amplitude + phase identity exact for every
   band (asserted per band). Band-pass bands are ~zero-mean so this equals the usual Pearson
   pattern correlation; only the coarsest residual band differs appreciably.

Aggregation: every diagnostic is computed per sample, then reported as mean +/- std across
the held-out samples.

Verification (always run, before the real analysis): synthetic sanity checks confirming the
diagnostics separate the two error types -- a SHIFTED copy of a field must read as high
phase_fraction + nonzero displacement, a SCALED (x0.5) copy as phase_fraction ~ 0, corr ~ 1,
zero displacement. Plus the two structural asserts (bands sum to the field; amplitude + phase
= per-band MSE).

Outputs (to runs/<version>/figures/): a CSV + printed table (rows coarse->fine) and a PNG
(phase_fraction and power_ratio vs band; pattern_corr and mean displacement vs band).

Usage:
    python band_error_diagnostics.py                       # default checkpoint, test split, 256 samples
    python band_error_diagnostics.py --n_samples -1        # all test samples
    python band_error_diagnostics.py --self_test_only      # just the synthetic verification
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import run_figures, resolve as resolve_path  # noqa: E402

# WRF native 200x200 grid spacing (km/pixel) -- same source as the eigenspectra scripts
# (extra_scripts/swin/compare_eigenspectra.py). Anisotropic: a degree
# of longitude (east-west, axis 2) spans less km than a degree of latitude (axis 1).
DY_KM = 4.44780  # north-south, axis 1
DX_KM = 3.48764  # east-west, axis 2
PIXEL_KM = float(np.sqrt(DX_KM * DY_KM))  # isotropic-equivalent pixel size for scale labels

# Historical home of this diagnostic's output; override with --output.
OUTPUT_DIR = run_figures("0627")


# --------------------------------------------------------------------------------------
# Scale decomposition
# --------------------------------------------------------------------------------------
def band_sigmas(n_levels: int, base_sigma: float) -> list[float]:
    """Cumulative Gaussian-blur sigmas (pixels) bounding each band: sig[0]=0 (the original
    field), sig[k] = base_sigma * 2^(k-1). Band j lies between sig[j] and sig[j+1] (the
    last band, the residual, is everything coarser than sig[n_levels-1])."""
    return [0.0] + [base_sigma * (2 ** (k - 1)) for k in range(1, n_levels)]


def laplacian_bands(field: np.ndarray, n_levels: int, base_sigma: float) -> np.ndarray:
    """Non-decimated Laplacian pyramid of a (C, H, W) field. Returns (n_levels, C, H, W),
    band 0 finest -> band n_levels-1 coarsest (residual). Bands sum to the field exactly."""
    f64 = field.astype(np.float64)
    sigmas = band_sigmas(n_levels, base_sigma)
    # g[k] = field blurred with cumulative sigma sig[k] (g[0] = field, sig[0]=0).
    blurred = [f64] + [
        gaussian_filter(f64, sigma=(0, s, s), mode="reflect") for s in sigmas[1:]
    ]
    bands = [blurred[j] - blurred[j + 1] for j in range(n_levels - 1)]
    bands.append(blurred[-1])  # coarsest residual low-pass
    bands = np.stack(bands, axis=0)
    assert np.allclose(bands.sum(axis=0), f64, atol=1e-6), "Laplacian bands do not sum to the field"
    return bands


def scale_labels(n_levels: int, base_sigma: float) -> list[str]:
    """Human-readable Gaussian-sigma scale range (km) bounding each band, finest -> coarsest."""
    sig = band_sigmas(n_levels, base_sigma)
    labels = []
    for j in range(n_levels):
        lo = sig[j] * PIXEL_KM
        if j < n_levels - 1:
            hi = sig[j + 1] * PIXEL_KM
            labels.append(f"{lo:.0f}-{hi:.0f}")
        else:
            labels.append(f">{lo:.0f}")
    return labels


# --------------------------------------------------------------------------------------
# Per-band diagnostics (on the pooled u/v vector field)
# --------------------------------------------------------------------------------------
def band_diagnostics(pred_band: np.ndarray, truth_band: np.ndarray, eps: float = 1e-12) -> dict:
    """Amplitude/phase diagnostics for one band's (C, H, W) pred vs truth, u/v pooled.
    Uncentered RMS + correlation so MSE = amplitude_term + phase_term is exact."""
    p = pred_band.ravel().astype(np.float64)
    t = truth_band.ravel().astype(np.float64)
    s_p = float(np.sqrt(np.mean(p * p)))
    s_t = float(np.sqrt(np.mean(t * t)))
    corr = float(np.mean(p * t) / (s_p * s_t + eps))
    mse = float(np.mean((p - t) ** 2))
    amp_term = (s_p - s_t) ** 2
    phase_term = 2.0 * s_p * s_t * (1.0 - corr)
    assert abs(amp_term + phase_term - mse) <= 1e-6 * max(1.0, mse), (
        f"MSE decomposition not exact: amp({amp_term}) + phase({phase_term}) != mse({mse})"
    )
    return {
        "power_ratio": s_p / (s_t + eps),
        "pattern_corr": corr,
        "amplitude_mse": amp_term,
        "phase_mse": phase_term,
        "total_mse": mse,
        "phase_fraction": phase_term / (amp_term + phase_term + eps),
    }


def band_displacement(pred_band: np.ndarray, truth_band: np.ndarray, max_shift: int) -> tuple[int, int]:
    """Peak offset (dx, dy) in pixels of the 2D vector cross-correlation (summed over u/v),
    searched within +/- max_shift. (dx, dy) = (0, 0) means no spatial shift at this scale."""
    C, H, W = pred_band.shape
    cc = np.zeros((H, W), dtype=np.float64)
    for c in range(C):
        P = np.fft.fft2(pred_band[c])
        T = np.fft.fft2(truth_band[c])
        cc += np.fft.ifft2(P * np.conj(T)).real
    cc = np.fft.fftshift(cc)
    cy, cx = H // 2, W // 2
    window = cc[cy - max_shift: cy + max_shift + 1, cx - max_shift: cx + max_shift + 1]
    py, px = np.unravel_index(int(np.argmax(window)), window.shape)
    return int(px - max_shift), int(py - max_shift)  # dx (axis 2), dy (axis 1)


def displacement_km(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Anisotropic displacement magnitude in km (dx along east-west DX_KM, dy north-south DY_KM)."""
    return np.sqrt((dx * DX_KM) ** 2 + (dy * DY_KM) ** 2)


# --------------------------------------------------------------------------------------
# Accumulate diagnostics over a set of (pred, truth) field pairs
# --------------------------------------------------------------------------------------
SCALAR_KEYS = ("power_ratio", "pattern_corr", "amplitude_mse", "phase_mse", "total_mse", "phase_fraction")


def accumulate(pred_fields: np.ndarray, truth_fields: np.ndarray, n_levels: int,
               base_sigma: float, max_shift: int) -> dict:
    """pred_fields/truth_fields: (S, C, H, W). Returns per-band arrays over the S samples:
    scalars -> (S, n_levels); displacement vectors -> (S, n_levels, 2) in pixels."""
    S = pred_fields.shape[0]
    scal = {k: np.zeros((S, n_levels)) for k in SCALAR_KEYS}
    disp = np.zeros((S, n_levels, 2))  # (dx, dy) pixels
    for i in range(S):
        pred_bands = laplacian_bands(pred_fields[i], n_levels, base_sigma)
        truth_bands = laplacian_bands(truth_fields[i], n_levels, base_sigma)
        for b in range(n_levels):
            d = band_diagnostics(pred_bands[b], truth_bands[b])
            for k in SCALAR_KEYS:
                scal[k][i, b] = d[k]
            disp[i, b] = band_displacement(pred_bands[b], truth_bands[b], max_shift)
    return {"scalars": scal, "disp": disp}


def summarize(acc: dict, n_levels: int, base_sigma: float, systematic_ratio: float = 0.5) -> list[dict]:
    """Reduce accumulated per-sample arrays to per-band mean/std rows, finest -> coarsest."""
    scal, disp = acc["scalars"], acc["disp"]
    labels = scale_labels(n_levels, base_sigma)
    rows = []
    for b in range(n_levels):
        row = {"band": b, "scale_range_km": labels[b]}
        for k in SCALAR_KEYS:
            row[f"{k}_mean"] = float(scal[k][:, b].mean())
            row[f"{k}_std"] = float(scal[k][:, b].std())
        dxy = disp[:, b, :]                                  # (S, 2) pixels
        sample_mag_km = displacement_km(dxy[:, 0], dxy[:, 1])
        mean_vec = dxy.mean(axis=0)                          # mean (dx, dy)
        mean_vec_mag_km = float(displacement_km(np.array([mean_vec[0]]), np.array([mean_vec[1]]))[0])
        mean_sample_mag_km = float(sample_mag_km.mean())
        ratio = mean_vec_mag_km / (mean_sample_mag_km + 1e-12)
        row["mean_displacement_km_mean"] = mean_sample_mag_km
        row["mean_displacement_km_std"] = float(sample_mag_km.std())
        row["mean_shift_vec_px"] = (float(mean_vec[0]), float(mean_vec[1]))
        row["systematic_ratio"] = ratio
        row["displacement_systematic"] = ratio > systematic_ratio
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def print_table(rows: list[dict]) -> None:
    hdr = (f"{'band':>4} {'scale_km':>10} {'power_ratio':>16} {'pattern_corr':>16} "
           f"{'amp_MSE':>10} {'phase_MSE':>10} {'phase_frac':>16} {'disp_km':>14} {'systematic':>11}")
    print("\nPer-band error decomposition (mean +/- std across samples), coarse -> fine:")
    print(hdr)
    print("-" * len(hdr))
    for row in reversed(rows):  # coarse (large scale) first
        print(
            f"{row['band']:>4} {row['scale_range_km']:>10} "
            f"{row['power_ratio_mean']:>7.3f}+/-{row['power_ratio_std']:<7.3f} "
            f"{row['pattern_corr_mean']:>7.3f}+/-{row['pattern_corr_std']:<7.3f} "
            f"{row['amplitude_mse_mean']:>10.4f} {row['phase_mse_mean']:>10.4f} "
            f"{row['phase_fraction_mean']:>7.3f}+/-{row['phase_fraction_std']:<7.3f} "
            f"{row['mean_displacement_km_mean']:>6.1f}+/-{row['mean_displacement_km_std']:<6.1f} "
            f"{('yes' if row['displacement_systematic'] else 'no'):>11}"
        )


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["band", "scale_range_km",
            "power_ratio_mean", "power_ratio_std",
            "pattern_corr_mean", "pattern_corr_std",
            "amplitude_mse_mean", "amplitude_mse_std",
            "phase_mse_mean", "phase_mse_std",
            "total_mse_mean", "total_mse_std",
            "phase_fraction_mean", "phase_fraction_std",
            "mean_displacement_km_mean", "mean_displacement_km_std",
            "mean_shift_dx_px", "mean_shift_dy_px",
            "systematic_ratio", "displacement_systematic"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in reversed(rows):  # coarse -> fine, matching the table
            r = dict(row)
            r["mean_shift_dx_px"], r["mean_shift_dy_px"] = row["mean_shift_vec_px"]
            f.write(",".join(_csv_val(r.get(c)) for c in cols) + "\n")
    print(f"Saved per-band CSV to {path}")


def _csv_val(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def plot_diagnostics(rows: list[dict], n_samples: int, path: Path) -> None:
    order = list(reversed(rows))  # coarse (left) -> fine (right)
    x = np.arange(len(order))
    xticklabels = [r["scale_range_km"] for r in order]

    def col(key_mean, key_std=None):
        m = np.array([r[key_mean] for r in order])
        s = np.array([r[key_std] for r in order]) if key_std else None
        return m, s

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # Top: phase_fraction (left) + power_ratio (right)
    pf_m, pf_s = col("phase_fraction_mean", "phase_fraction_std")
    h_pf = ax_top.errorbar(x, pf_m, yerr=pf_s, marker="o", color="C3", capsize=3, label="phase_fraction")
    ax_top.axhline(0.5, color="C3", ls=":", alpha=0.5)
    ax_top.set_ylabel("phase_fraction\n(1=displacement, 0=amplitude)", color="C3")
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.tick_params(axis="y", labelcolor="C3")

    ax_top_r = ax_top.twinx()
    pr_m, pr_s = col("power_ratio_mean", "power_ratio_std")
    h_pr = ax_top_r.errorbar(x, pr_m, yerr=pr_s, marker="s", color="C0", capsize=3, label="power_ratio")
    ax_top_r.axhline(1.0, color="C0", ls=":", alpha=0.5)
    ax_top_r.set_ylabel("power_ratio (pred RMS / truth RMS)", color="C0")
    ax_top_r.tick_params(axis="y", labelcolor="C0")
    ax_top.set_title(f"Per-band error type: phase/displacement vs amplitude  (n_samples={n_samples})")
    ax_top.legend([h_pf, h_pr], ["phase_fraction", "power_ratio"], loc="center left")

    # Bottom: pattern_corr (left) + mean displacement km (right)
    pc_m, pc_s = col("pattern_corr_mean", "pattern_corr_std")
    h_pc = ax_bot.errorbar(x, pc_m, yerr=pc_s, marker="o", color="C2", capsize=3, label="pattern_corr")
    ax_bot.set_ylabel("pattern_corr", color="C2")
    ax_bot.tick_params(axis="y", labelcolor="C2")
    ax_bot.set_ylim(-0.02, 1.02)

    ax_bot_r = ax_bot.twinx()
    dk_m, dk_s = col("mean_displacement_km_mean", "mean_displacement_km_std")
    h_dk = ax_bot_r.errorbar(x, dk_m, yerr=dk_s, marker="d", color="C1", capsize=3, label="mean_displacement")
    sys_flags = [r["displacement_systematic"] for r in order]
    for xi, dyi, sysflag in zip(x, dk_m, sys_flags):
        ax_bot_r.annotate("sys" if sysflag else "rand", (xi, dyi), textcoords="offset points",
                          xytext=(0, 6), ha="center", fontsize=7, color="C1")
    ax_bot_r.set_ylabel("mean displacement (km)", color="C1")
    ax_bot_r.tick_params(axis="y", labelcolor="C1")

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(xticklabels)
    ax_bot.set_xlabel("band Gaussian-sigma scale range (km)   [coarse / large-scale  ->  fine]")
    ax_bot.legend([h_pc, h_dk], ["pattern_corr", "mean_displacement"], loc="center right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved diagnostic plot to {path}")


# --------------------------------------------------------------------------------------
# Synthetic verification
# --------------------------------------------------------------------------------------
def run_synthetic_tests(n_levels: int, base_sigma: float, max_shift: int) -> None:
    print("Running synthetic verification of the diagnostics...")
    rng = np.random.default_rng(0)
    truth = gaussian_filter(rng.standard_normal((2, 200, 200)), sigma=(0, 3, 3), mode="reflect")

    # (a) SHIFTED copy: expect high phase_fraction + nonzero displacement matching the shift.
    shift_dy, shift_dx = 4, 7
    shifted = np.roll(truth, shift=(shift_dy, shift_dx), axis=(1, 2))
    acc_s = accumulate(shifted[None], truth[None], n_levels, base_sigma, max_shift)
    pf_shift = acc_s["scalars"]["phase_fraction"][0]
    disp_shift = acc_s["disp"][0]
    # Bands with real energy (exclude the near-empty finest band of a smooth field).
    energetic = acc_s["scalars"]["total_mse"][0] > 1e-8
    assert pf_shift[energetic].mean() > 0.8, f"shifted: expected high phase_fraction, got {pf_shift}"
    assert np.allclose(disp_shift[energetic], [shift_dx, shift_dy]), (
        f"shifted: expected displacement ({shift_dx},{shift_dy}), got {disp_shift}"
    )
    print(f"  (a) shifted copy: phase_fraction(energetic bands) mean={pf_shift[energetic].mean():.3f} "
          f"(>0.8 OK), recovered shift (dx,dy)={tuple(disp_shift[energetic][-1].astype(int))} "
          f"(true ({shift_dx},{shift_dy})) OK")

    # (b) SCALED copy (x0.5): expect phase_fraction ~ 0, corr ~ 1, zero displacement.
    scaled = 0.5 * truth
    acc_c = accumulate(scaled[None], truth[None], n_levels, base_sigma, max_shift)
    pf_scale = acc_c["scalars"]["phase_fraction"][0]
    corr_scale = acc_c["scalars"]["pattern_corr"][0]
    disp_scale = acc_c["disp"][0]
    pr_scale = acc_c["scalars"]["power_ratio"][0]
    assert pf_scale.max() < 1e-3, f"scaled: expected phase_fraction ~ 0, got {pf_scale}"
    assert corr_scale.min() > 0.999, f"scaled: expected corr ~ 1, got {corr_scale}"
    assert np.all(disp_scale == 0), f"scaled: expected zero displacement, got {disp_scale}"
    assert np.allclose(pr_scale, 0.5), f"scaled: expected power_ratio 0.5, got {pr_scale}"
    print(f"  (b) scaled x0.5 copy: phase_fraction max={pf_scale.max():.2e} (~0 OK), "
          f"corr min={corr_scale.min():.4f} (~1 OK), power_ratio={pr_scale.mean():.3f} (0.5 OK), "
          f"displacement=0 OK")
    print("Synthetic verification passed.\n")


# --------------------------------------------------------------------------------------
# Real predictions
# --------------------------------------------------------------------------------------
def generate_predictions(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the Swin checkpoint on the chosen split. Returns (pred, truth, idx),
    pred/truth shape (S, 2, 200, 200)."""
    import torch
    from new_enscgp_swin import DEFAULT_CONFIG_PATH, build_model, load_config
    from terrain_encoder import load_terrain_input

    config = load_config(args.config or DEFAULT_CONFIG_PATH)
    paths = config["paths"]
    data_dir = args.data_dir or resolve_path(paths["data_dir"])
    checkpoint = args.checkpoint or (resolve_path(paths["log_dir"]) / "checkpoints" / "best.pth")
    splits_path = args.splits_path or resolve_path(paths["splits_path"])
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device)
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device)

    wrf = np.load(data_dir / "wrf_uv.npy", mmap_mode="r")
    posterior = np.load(data_dir / "enscgp_posterior.npy", mmap_mode="r")
    bicubic = np.load(data_dir / "era5_uv_2ch_bicubic.npy", mmap_mode="r")

    pool = np.load(splits_path)[f"{args.split}_idx"]
    if args.n_samples > 0 and args.n_samples < len(pool):
        idx = np.random.default_rng(args.seed).choice(pool, size=args.n_samples, replace=False)
    else:
        idx = np.array(pool)
    idx = np.sort(idx)
    print(f"Running checkpoint {checkpoint} on {len(idx)} '{args.split}' samples "
          f"(epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')})...")

    preds = np.empty((len(idx), 2, 200, 200), dtype=np.float32)
    for start in range(0, len(idx), args.batch_size):
        batch = idx[start:start + args.batch_size]
        post_b = torch.from_numpy(np.array(posterior[batch], dtype=np.float32, copy=True)).to(device)
        bic_b = torch.from_numpy(np.array(bicubic[batch], dtype=np.float32, copy=True)).to(device)
        with torch.no_grad():
            preds[start:start + len(batch)] = model(post_b, bic_b, terrain_raw)[:, :2].cpu().numpy()
    truth = np.array(wrf[idx, :2], dtype=np.float32)
    return preds, truth, idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="new_enscgp_swin_config.json (default)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--splits_path", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=256, help="Held-out samples to aggregate over (-1 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_levels", type=int, default=5, help="Laplacian pyramid levels (bands)")
    parser.add_argument("--base_sigma", type=float, default=1.0, help="Finest Gaussian blur sigma (pixels)")
    parser.add_argument("--max_shift", type=int, default=25, help="Max +/- pixel shift searched for displacement")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="band_error_diagnostics")
    parser.add_argument("--self_test_only", action="store_true", help="Run only the synthetic verification and exit")
    args = parser.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_synthetic_tests(args.n_levels, args.base_sigma, args.max_shift)
    if args.self_test_only:
        return

    preds, truth, idx = generate_predictions(args)
    acc = accumulate(preds, truth, args.n_levels, args.base_sigma, args.max_shift)
    rows = summarize(acc, args.n_levels, args.base_sigma)

    print_table(rows)
    write_csv(rows, OUTPUT_DIR / f"{args.output_prefix}.csv")
    plot_diagnostics(rows, len(idx), OUTPUT_DIR / f"{args.output_prefix}.png")

    pf = np.array([r["phase_fraction_mean"] for r in rows])  # finest -> coarsest
    print(f"\nHypothesis check (phase_fraction, coarse -> fine): "
          f"{[f'{v:.2f}' for v in reversed(pf)]}")
    print(f"  coarsest band phase_fraction = {pf[-1]:.3f}; finest band phase_fraction = {pf[0]:.3f}")
    if pf[-1] > pf[0]:
        print("  -> consistent with the hypothesis: displacement-dominated at large scales, "
              "amplitude-dominated at fine scales.")
    else:
        print("  -> NOT consistent with the hypothesis (large-scale phase_fraction is not higher).")


if __name__ == "__main__":
    main()
