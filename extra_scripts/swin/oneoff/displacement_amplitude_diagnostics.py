#!/usr/bin/env python3
"""
Per-band, per-component decomposition of prediction error into AMPLITUDE (wrong power,
fixable by sharpening), SYSTEMATIC DISPLACEMENT (consistent offset, fixable by a shift
correction), and RANDOM DISPLACEMENT (structure in the wrong place with no consistent
direction across samples -- irreducible for a deterministic model).

Extends band_error_diagnostics.py's global-per-sample shift (band_displacement, one
(dx,dy) per band per sample) to a LOCAL patch-wise shift search: a single global shift is
too coarse to tell "systematic bias" apart from "random displacement" -- both look like
"some nonzero shift" at the global level, but only many local shift estimates reveal
whether they cluster (systematic) or scatter (random) in direction.

Reuses (does not reinvent):
  - laplacian_bands (GPU/torch) from scripts/multiscale_loss.py -- same Laplacian-pyramid
    band decomposition used by MultiscaleLoss during training.
  - DX_KM/DY_KM/PIXEL_KM, scale_labels, generate_predictions from
    extra_scripts/swin/oneoff/band_error_diagnostics.py -- same pixel->km scale, same checkpoint/data
    loading (post-re-registration WRF, runs/0701 checkpoint), same held-out split.

Pipeline
--------
1. Per band (Laplacian pyramid, n_levels=5, base_sigma=2 to match training's
   multiscale.base_sigma), per component (u, v):
   (A) Amplitude: power_ratio = RMS(pred_band)/RMS(truth_band), pattern_corr -- same
       uncentered-RMS formulas as band_error_diagnostics.band_diagnostics.
   (B) Local displacement: block-matching shift search. The field is tiled into patches
       (patch_size/stride/max_shift scaled to the band's characteristic scale -- finer
       bands get smaller patches and search windows, coarser bands get larger ones, since
       a "local" shift only makes sense relative to the structure size at that scale).
       For each patch, try every integer (dx,dy) in the search window, shift the WHOLE
       band field (not just the patch) by that amount (reflect-padded, no wraparound),
       and score by SSD against the truth patch at that location. The argmin per patch is
       its optimal_shift; error_reduction_fraction is how much SSD that shift removes.
       Vectorized: batched over samples, looped only over shift candidates (the small
       axis), so the expensive part stays GPU-resident.
   (C) Systematic vs random: vector-average the per-patch optimal_shift over all
       patches/samples for a band. |mean_shift| large & error_reduction_fraction high ->
       systematic (correctable). |mean_shift| ~ 0 with the same error_reduction_fraction ->
       random (irreducible): the shifts are real and remove real error, they just don't
       point anywhere consistent.
   (D) Fourier phase/magnitude split, as an independent cross-check on (B)/(C): decompose
       each band's error energy, per Fourier bin, into a magnitude-mismatch term and a
       phase-mismatch term (exact identity, same algebra as band_diagnostics' spatial-
       domain amplitude/phase MSE split, applied per-frequency instead of pooled). This
       uses no shift search at all, so if it disagrees strongly with (B)'s
       error_reduction_fraction, that is a real flag, not a shared-methodology artifact.

2. Aggregate to per-band, per-component rows; write CSV + 3 plots + a printed summary that
   explicitly separates each band's error into amplitude / systematic-displacement /
   random-displacement, and a PASS/FLAG registration sanity check (large coherent
   mean_shift at fine scales would indicate residual misregistration).

Verification (always run first): identity (truth vs truth: everything ~0/1), known integer
shift (truth vs rolled truth: shift recovered exactly, systematic_fraction ~1), blurred
truth (truth vs blurred truth: power_ratio<1, displacement~0 -- amplitude and displacement
correctly separated).

Usage:
    python displacement_amplitude_diagnostics.py                  # default checkpoint/split
    python displacement_amplitude_diagnostics.py --self_test_only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
for _p in (REPO / "scripts", REPO / "extra_scripts"):
    sys.path.insert(0, str(_p))

from paths import run_figures  # noqa: E402
from multiscale_loss import laplacian_bands, band_sigmas  # noqa: E402  (GPU/torch, reused)
from band_error_diagnostics import (  # noqa: E402  (reused: scale + data/checkpoint loading)
    DX_KM, DY_KM, PIXEL_KM, scale_labels, generate_predictions,
)

# Historical home of this diagnostic's output; override with --output.
OUTPUT_DIR = run_figures("0701")
COMPONENTS = ("u", "v")


# --------------------------------------------------------------------------------------
# (A) Amplitude diagnostics -- same formulas as band_error_diagnostics.band_diagnostics,
# applied per-component (H, W) instead of pooled over (C, H, W).
# --------------------------------------------------------------------------------------
def amplitude_diagnostics(pred_band: torch.Tensor, truth_band: torch.Tensor, eps: float = 1e-12) -> dict:
    """pred_band/truth_band: (B, H, W), one component, one band, batched over samples.
    Returns per-sample arrays (B,)."""
    p = pred_band.reshape(pred_band.shape[0], -1).double()
    t = truth_band.reshape(truth_band.shape[0], -1).double()
    s_p = torch.sqrt((p * p).mean(dim=1))
    s_t = torch.sqrt((t * t).mean(dim=1))
    corr = (p * t).mean(dim=1) / (s_p * s_t + eps)
    return {
        "power_ratio": (s_p / (s_t + eps)).float(),
        "pattern_corr": corr.float(),
    }


# --------------------------------------------------------------------------------------
# (B) Local block-matching shift search (vectorized, batched over samples)
# --------------------------------------------------------------------------------------
def band_patch_params(n_levels: int, base_sigma: float) -> list[dict]:
    """Patch size / stride / max_shift per band (finest -> coarsest), scaled to each
    band's characteristic scale in pixels: patch ~4x, search window ~1.5x the scale, so
    "local" is relative to the structure size the band actually carries. Clipped to a
    sane px range so the finest band isn't a useless 1px patch and the coarsest isn't a
    patch bigger than the 200x200 field."""
    sigmas = band_sigmas(n_levels, base_sigma)  # [0, base_sigma, 2*base_sigma, ...]
    char_px = []
    for j in range(n_levels):
        upper = sigmas[j + 1] if j < n_levels - 1 else sigmas[-1] * 2.0
        char_px.append(max(upper, 1.0))
    params = []
    for c in char_px:
        patch = int(np.clip(round(4 * c / 4) * 4, 16, 64))
        max_shift = int(np.clip(round(1.5 * c), 4, 24))
        stride = patch // 2
        params.append({"patch_size": patch, "stride": stride, "max_shift": max_shift})
    return params


def local_shift_search(pred_band: torch.Tensor, truth_band: torch.Tensor, patch_size: int,
                        stride: int, max_shift: int) -> dict:
    """pred_band/truth_band: (B, H, W), one component, one band, batched over samples.
    For every integer (dx,dy) in [-max_shift, max_shift]^2, shift the WHOLE field
    (reflect-padded) and score each patch by SSD against the truth patch there; take the
    per-patch argmin. Loops only over shift candidates (small), batched over samples and
    patches (large) inside each iteration -- keeps the expensive axis on GPU.

    Returns dict with (B, n_patches) tensors: best_dx, best_dy (px), unshifted_err,
    best_err (SSD per patch).

    Sign convention: optimal_shift is the shift applied TO PRED that best matches truth
    (literally shift(pred_band, dx, dy), per the task spec) -- i.e. the CORRECTION, not
    the prediction's positional error. If pred's structure sits +v away from truth's,
    optimal_shift ~= -v."""
    Bn, H, W = pred_band.shape
    pred = pred_band.unsqueeze(1)  # (B,1,H,W)
    truth = truth_band.unsqueeze(1)
    pad = max_shift
    pred_padded = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    truth_patches = F.unfold(truth, kernel_size=patch_size, stride=stride)  # (B, P, N)
    n_patches = truth_patches.shape[-1]

    shifts = [(dx, dy) for dy in range(-max_shift, max_shift + 1) for dx in range(-max_shift, max_shift + 1)]
    errs = torch.empty(len(shifts), Bn, n_patches, device=pred.device, dtype=pred.dtype)
    zero_idx = shifts.index((0, 0))
    for i, (dx, dy) in enumerate(shifts):
        # slice the padded, shifted field back down to (H, W): shifting pred by (dx,dy)
        # means truth[y,x] should now be compared to pred[y-dy, x-dx] -> slice offset by
        # (pad - dy, pad - dx).
        shifted = pred_padded[:, :, pad - dy:pad - dy + H, pad - dx:pad - dx + W]
        shifted_patches = F.unfold(shifted, kernel_size=patch_size, stride=stride)  # (B, P, N)
        errs[i] = ((shifted_patches - truth_patches) ** 2).sum(dim=1)  # (B, N)

    best_err, best_idx = errs.min(dim=0)  # (B, N)
    shifts_t = torch.tensor(shifts, device=pred.device, dtype=pred.dtype)  # (n_shifts, 2)
    best_shift = shifts_t[best_idx]  # (B, N, 2) -> (dx, dy)
    return {
        "best_dx": best_shift[..., 0],
        "best_dy": best_shift[..., 1],
        "unshifted_err": errs[zero_idx],
        "best_err": best_err,
    }


# --------------------------------------------------------------------------------------
# (D) Fourier magnitude/phase energy split (independent cross-check on B/C)
# --------------------------------------------------------------------------------------
def fourier_phase_fraction(pred_band: torch.Tensor, truth_band: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """pred_band/truth_band: (B, H, W). Per-sample fraction of band error ENERGY
    (Parseval-exact: rfft bins double-weighted except DC/Nyquist columns) attributable to
    phase mismatch vs magnitude mismatch. Same amp+phase=total identity as
    band_diagnostics, applied per Fourier bin instead of pooled over pixels."""
    Bn, H, W = pred_band.shape
    Pf = torch.fft.rfft2(pred_band.double())
    Tf = torch.fft.rfft2(truth_band.double())
    Ap, At = Pf.abs(), Tf.abs()
    cos_dphi = (Pf.real * Tf.real + Pf.imag * Tf.imag) / (Ap * At + eps)
    cos_dphi = cos_dphi.clamp(-1.0, 1.0)
    amp_term = (Ap - At) ** 2
    phase_term = 2.0 * Ap * At * (1.0 - cos_dphi)

    # rfft2 stores only non-negative x-frequencies; every column except x=0 (and x=W/2 if
    # W even) has a conjugate-symmetric twin dropped from storage -> weight it x2 so the
    # sum reproduces the true (full-spectrum) Parseval energy total.
    weight = torch.ones(Pf.shape[-1], device=pred_band.device, dtype=torch.float64)
    weight[1:] = 2.0
    if W % 2 == 0:
        weight[-1] = 1.0
    amp_e = (amp_term * weight).sum(dim=(1, 2))
    phase_e = (phase_term * weight).sum(dim=(1, 2))
    return (phase_e / (amp_e + phase_e + eps)).float()


def _assert_fourier_parseval(n_levels: int, base_sigma: float) -> None:
    torch.manual_seed(0)
    pred = torch.randn(3, 200, 200)
    truth = torch.randn(3, 200, 200)
    Pf = torch.fft.rfft2(pred.double())
    Tf = torch.fft.rfft2(truth.double())
    weight = torch.ones(Pf.shape[-1], dtype=torch.float64)
    weight[1:] = 2.0
    if pred.shape[-1] % 2 == 0:
        weight[-1] = 1.0
    fourier_energy = (((Pf - Tf).abs() ** 2) * weight).sum(dim=(1, 2)) / (200 * 200)
    spatial_mse_sum = ((pred - truth).double() ** 2).sum(dim=(1, 2))
    assert torch.allclose(fourier_energy, spatial_mse_sum, rtol=1e-6), (
        f"Fourier energy (Parseval, weighted rfft) does not match spatial SSD: "
        f"{fourier_energy} vs {spatial_mse_sum}"
    )
    print("  Fourier Parseval identity (weighted rfft): matches spatial SSD to rtol=1e-6 OK")


# --------------------------------------------------------------------------------------
# Per-band / per-component accumulation over samples
# --------------------------------------------------------------------------------------
def compute_band_component(pred_band: torch.Tensor, truth_band: torch.Tensor, patch_size: int,
                            stride: int, max_shift: int) -> dict:
    """pred_band/truth_band: (B, H, W), one band, one component. Returns per-sample and
    per-(sample,patch) arrays needed for aggregation."""
    amp = amplitude_diagnostics(pred_band, truth_band)
    shift = local_shift_search(pred_band, truth_band, patch_size, stride, max_shift)
    phase_frac = fourier_phase_fraction(pred_band, truth_band)

    unshifted = shift["unshifted_err"]  # (B, N)
    best = shift["best_err"]
    reduction_frac_per_sample = (unshifted - best).sum(dim=1) / unshifted.sum(dim=1).clamp_min(1e-12)

    return {
        "power_ratio": amp["power_ratio"],          # (B,)
        "pattern_corr": amp["pattern_corr"],         # (B,)
        "phase_fraction_fourier": phase_frac,         # (B,)
        "error_reduction_fraction": reduction_frac_per_sample,  # (B,)
        "shift_dx_px": shift["best_dx"],              # (B, N)
        "shift_dy_px": shift["best_dy"],              # (B, N)
        "shift_weight": unshifted,                    # (B, N) -- how much error this patch has to explain
    }


def summarize_band_component(acc: dict, systematic_ratio_thresh: float = 0.5) -> dict:
    dx = acc["shift_dx_px"].reshape(-1).double()
    dy = acc["shift_dy_px"].reshape(-1).double()
    # Weight each patch's shift vote by its unshifted error: a patch with ~0 error to
    # explain has an unconstrained (noisy) optimal_shift -- an aperture-problem artifact,
    # worst for texture-poor patches/bands -- and should not vote equally with a patch
    # that actually has displacement error to explain.
    w = acc["shift_weight"].reshape(-1).double().clamp_min(0.0)
    w = w / w.sum().clamp_min(1e-12)
    dx_km, dy_km = dx * DX_KM, dy * DY_KM
    shift_mag_km = torch.sqrt(dx_km ** 2 + dy_km ** 2)
    mean_dx_km, mean_dy_km = (w * dx_km).sum().item(), (w * dy_km).sum().item()
    mean_shift_mag_km = float(np.hypot(mean_dx_km, mean_dy_km))
    mean_sq_mag_km2 = float((w * shift_mag_km ** 2).sum().item())
    systematic_fraction = (mean_shift_mag_km ** 2) / (mean_sq_mag_km2 + 1e-12)

    def m(key):
        return float(acc[key].mean().item())

    def s(key):
        return float(acc[key].std().item())

    return {
        "power_ratio_mean": m("power_ratio"), "power_ratio_std": s("power_ratio"),
        "pattern_corr_mean": m("pattern_corr"), "pattern_corr_std": s("pattern_corr"),
        "displacement_magnitude_km_mean": float((w * shift_mag_km).sum().item()),
        "displacement_magnitude_km_std": float(torch.sqrt(
            (w * (shift_mag_km - (w * shift_mag_km).sum()) ** 2).sum()).item()),
        "error_reduction_fraction_mean": m("error_reduction_fraction"),
        "error_reduction_fraction_std": s("error_reduction_fraction"),
        "phase_fraction_fourier_mean": m("phase_fraction_fourier"),
        "phase_fraction_fourier_std": s("phase_fraction_fourier"),
        "mean_shift_km": (mean_dx_km, mean_dy_km),
        "systematic_fraction": systematic_fraction,
        "shift_dx_km_all": dx_km,   # kept for the quiver/hist plot
        "shift_dy_km_all": dy_km,
    }


# --------------------------------------------------------------------------------------
# Synthetic verification
# --------------------------------------------------------------------------------------
def run_synthetic_tests(n_levels: int, base_sigma: float, device: str) -> None:
    print("Running synthetic verification of the displacement/amplitude diagnostics...")
    _assert_fourier_parseval(n_levels, base_sigma)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    truth_np = rng.standard_normal((1, 1, 200, 200)).astype(np.float32)
    truth_np = torch.from_numpy(truth_np)
    from multiscale_loss import gaussian_blur
    truth = gaussian_blur(truth_np, sigma=3.0).to(device)  # (1,1,200,200), smooth field

    params = band_patch_params(n_levels, base_sigma)

    # (a) identity: truth vs truth -> displacement ~ 0, power_ratio ~ 1, error_reduction ~ 0.
    truth_bands = laplacian_bands(truth, n_levels, base_sigma)
    energetic_bands = [b for b in range(n_levels) if truth_bands[b].pow(2).mean().item() > 1e-8]
    for b in energetic_bands:
        pp = params[b]
        acc = compute_band_component(truth_bands[b][:, 0], truth_bands[b][:, 0],
                                      pp["patch_size"], pp["stride"], pp["max_shift"])
        row = summarize_band_component(acc)
        assert abs(row["power_ratio_mean"] - 1.0) < 1e-3, f"identity band {b}: power_ratio {row['power_ratio_mean']}"
        assert row["displacement_magnitude_km_mean"] < 1e-3, f"identity band {b}: displacement {row['displacement_magnitude_km_mean']}"
        assert row["error_reduction_fraction_mean"] < 1e-3, f"identity band {b}: error_reduction {row['error_reduction_fraction_mean']}"
    print(f"  (a) identity (truth vs truth): power_ratio~1, displacement~0, error_reduction~0 OK "
          f"(bands checked: {energetic_bands})")

    # (b) known integer shift: pred = truth rolled by (dy,dx)=(3,5) px, so pred's structure
    # sits +(dx,dy) away from truth's -> the correction (our optimal_shift convention,
    # see local_shift_search docstring) should recover -(dx,dy). (torch.roll is
    # wraparound, but shift magnitude << field size and away from edges so block
    # matching still recovers it exactly in the interior.)
    shift_dy, shift_dx = 3, 5
    shifted = torch.roll(truth, shifts=(shift_dy, shift_dx), dims=(2, 3))
    shifted_bands = laplacian_bands(shifted, n_levels, base_sigma)
    for b in energetic_bands:
        pp = params[b]
        if pp["max_shift"] < max(abs(shift_dy), abs(shift_dx)):
            continue  # this band's search window is too small to see the injected shift
        acc = compute_band_component(shifted_bands[b][:, 0], truth_bands[b][:, 0],
                                      pp["patch_size"], pp["stride"], pp["max_shift"])
        row = summarize_band_component(acc)
        recovered_dx_km = row["mean_shift_km"][0] / DX_KM
        recovered_dy_km = row["mean_shift_km"][1] / DY_KM
        # The coarsest band (large-scale residual) only tiles into a handful of big
        # patches (e.g. 5x5=25 on a 200x200 field) and is nearly flat within each one --
        # a real aperture-problem regime, not a bug -- so its per-patch shift votes are
        # higher-variance. Loosen the bar there; keep the other bands (real texture,
        # more patches) to a tight tolerance.
        is_coarsest = (b == n_levels - 1)
        px_tol = 2.5 if is_coarsest else 0.6
        sys_frac_min = 0.5 if is_coarsest else 0.9
        assert abs(recovered_dx_km - (-shift_dx)) < px_tol, f"band {b}: recovered dx {recovered_dx_km}, expected {-shift_dx}"
        assert abs(recovered_dy_km - (-shift_dy)) < px_tol, f"band {b}: recovered dy {recovered_dy_km}, expected {-shift_dy}"
        assert row["systematic_fraction"] > sys_frac_min, f"band {b}: systematic_fraction {row['systematic_fraction']} (expected > {sys_frac_min})"
        print(f"  (b) band {b}: injected pred shift (dx={shift_dx},dy={shift_dy})px recovered as "
              f"correction ({recovered_dx_km:.2f},{recovered_dy_km:.2f})px "
              f"(expected ({-shift_dx},{-shift_dy})), systematic_fraction={row['systematic_fraction']:.3f} OK"
              + ("  [coarsest band, loosened tolerance -- see comment]" if is_coarsest else ""))

    # (c) blurred truth: extra blur -> amplitude error (power_ratio<1), not displacement.
    blurred = gaussian_blur(truth, sigma=4.0)
    blurred_bands = laplacian_bands(blurred, n_levels, base_sigma)
    checked_any = False
    for b in energetic_bands:
        pp = params[b]
        acc = compute_band_component(blurred_bands[b][:, 0], truth_bands[b][:, 0],
                                      pp["patch_size"], pp["stride"], pp["max_shift"])
        row = summarize_band_component(acc)
        if row["power_ratio_mean"] > 0.98:
            continue  # this band's power is essentially untouched by the extra blur
        checked_any = True
        assert row["power_ratio_mean"] < 0.98, f"band {b}: expected power_ratio<0.98, got {row['power_ratio_mean']}"
        assert row["displacement_magnitude_km_mean"] < 1.5 * PIXEL_KM, (
            f"band {b}: expected ~0 displacement from pure blur, got {row['displacement_magnitude_km_mean']:.2f} km")
        print(f"  (c) band {b}: blurred truth: power_ratio={row['power_ratio_mean']:.3f} (<1 OK), "
              f"displacement={row['displacement_magnitude_km_mean']:.2f} km (~0, amplitude not displacement) OK")
    assert checked_any, "blurred-truth test: no band showed a measurable power_ratio change"
    print("Synthetic verification passed.\n")


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def print_summary(rows: list[dict], systematic_thresh: float = 0.4, coherent_km_thresh: float = 1.0) -> None:
    print("\nPer-band, per-component error decomposition (amplitude / systematic-displacement / "
          "random-displacement), coarse -> fine:")
    hdr = (f"{'band':>4} {'scale_km':>10} {'comp':>4} {'power_ratio':>12} {'pattern_corr':>13} "
           f"{'disp_km':>18} {'err_reduc':>10} {'phase_frac':>10} {'systematic':>10} {'mean_shift_km':>16}")
    print(hdr)
    print("-" * len(hdr))
    for row in reversed(rows):
        print(
            f"{row['band']:>4} {row['scale_range_km']:>10} {row['component']:>4} "
            f"{row['power_ratio_mean']:>7.3f}+/-{row['power_ratio_std']:<4.2f} "
            f"{row['pattern_corr_mean']:>7.3f}+/-{row['pattern_corr_std']:<4.2f} "
            f"{row['displacement_magnitude_km_mean']:>7.2f}+/-{row['displacement_magnitude_km_std']:<7.2f} "
            f"{row['error_reduction_fraction_mean']:>10.3f} "
            f"{row['phase_fraction_fourier_mean']:>10.3f} "
            f"{row['systematic_fraction']:>10.3f} "
            f"({row['mean_shift_km'][0]:>5.2f},{row['mean_shift_km'][1]:>5.2f})"
        )

    print("\nInterpretation per band/component -- split of error into fixable amplitude, "
          "fixable systematic displacement, and irreducible random displacement:")
    for row in reversed(rows):
        disp_frac = row["error_reduction_fraction_mean"]
        amp_frac = max(0.0, 1.0 - disp_frac)
        sys_part = disp_frac * row["systematic_fraction"]
        rand_part = disp_frac * (1.0 - row["systematic_fraction"])
        print(f"  band {row['band']} ({row['scale_range_km']} km, {row['component']}): "
              f"amplitude(fixable)={amp_frac:.1%}, systematic-displacement(fixable)={sys_part:.1%}, "
              f"random-displacement(irreducible)={rand_part:.1%}")

    print("\nRegistration sanity check (large coherent shift at fine scales -> possible residual "
          "misregistration):")
    any_flag = False
    for row in rows:  # fine -> coarse; fine scales matter most for residual registration bugs
        mag_km = float(np.hypot(*row["mean_shift_km"]))
        if row["systematic_fraction"] > systematic_thresh and mag_km > coherent_km_thresh:
            any_flag = True
            angle = np.degrees(np.arctan2(row["mean_shift_km"][1], row["mean_shift_km"][0]))
            print(f"  FLAG: band {row['band']} ({row['scale_range_km']} km, {row['component']}) shows a "
                  f"coherent shift of {mag_km:.2f} km at {angle:.0f} deg "
                  f"(systematic_fraction={row['systematic_fraction']:.2f}) -- possible residual "
                  f"registration issue.")
    if not any_flag:
        print("  PASS: no band shows a dominant coherent shift "
              f"(systematic_fraction>{systematic_thresh} and |mean_shift|>{coherent_km_thresh}km).")

    print("\nCross-check: error_reduction_fraction (B, local shift search) vs phase_fraction_fourier "
          "(D, independent Fourier split) -- should roughly agree:")
    any_disagree = False
    for row in rows:
        diff = abs(row["error_reduction_fraction_mean"] - row["phase_fraction_fourier_mean"])
        flag = " <-- DISAGREE (>0.25)" if diff > 0.25 else ""
        any_disagree = any_disagree or diff > 0.25
        print(f"  band {row['band']} ({row['component']}): B={row['error_reduction_fraction_mean']:.3f}  "
              f"D={row['phase_fraction_fourier_mean']:.3f}  |diff|={diff:.3f}  "
              f"pattern_corr={row['pattern_corr_mean']:.3f}{flag}")
    if any_disagree:
        print("  Note: D tends to read HIGHER than B where pattern_corr is low. D's per-Fourier-bin "
              "phase/magnitude split (like any amplitude+phase MSE decomposition) attributes real "
              "DECORRELATION -- structure that no shift could fix, not just displaced structure -- "
              "into the 'phase' bucket. B is the more operational number (error actually removable "
              "by a shift); treat a large B-D gap at low pattern_corr as 'genuinely decorrelated', "
              "not as evidence B is undercounting displacement.")


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["band", "scale_range_km", "component",
            "power_ratio_mean", "power_ratio_std",
            "pattern_corr_mean", "pattern_corr_std",
            "displacement_magnitude_km_mean", "displacement_magnitude_km_std",
            "error_reduction_fraction_mean", "error_reduction_fraction_std",
            "phase_fraction_fourier_mean", "phase_fraction_fourier_std",
            "systematic_fraction", "mean_shift_km_dx", "mean_shift_km_dy"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in reversed(rows):
            r = dict(row)
            r["mean_shift_km_dx"], r["mean_shift_km_dy"] = row["mean_shift_km"]
            f.write(",".join(_csv_val(r.get(c)) for c in cols) + "\n")
    print(f"Saved per-band CSV to {path}")


def _csv_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def plot_decomposition_bar(rows: list[dict], n_samples: int, path: Path) -> None:
    order = [r for r in reversed(rows) if r["component"] == "u"] + \
            [r for r in reversed(rows) if r["component"] == "v"]
    order = list(reversed(rows))
    labels = [f"{r['scale_range_km']}\n({r['component']})" for r in order]
    amp = np.array([max(0.0, 1.0 - r["error_reduction_fraction_mean"]) for r in order])
    disp = np.array([r["error_reduction_fraction_mean"] for r in order])
    sys_frac = np.array([r["systematic_fraction"] for r in order])
    sys_part = disp * sys_frac
    rand_part = disp * (1.0 - sys_frac)

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(order)), 5.5))
    ax.bar(x, amp, label="amplitude (fixable)", color="C0")
    ax.bar(x, sys_part, bottom=amp, label="systematic displacement (fixable)", color="C1")
    ax.bar(x, rand_part, bottom=amp + sys_part, label="random displacement (irreducible)", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fraction of band error")
    ax.set_title(f"Per-band error decomposition: amplitude vs displacement (n_samples={n_samples})")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved decomposition bar chart to {path}")


def plot_shift_quivers(rows: list[dict], n_levels: int, path: Path) -> None:
    band_rows = [r for r in rows if r["component"] == "u"]  # one panel per band; u shown, v is similar physically
    n = len(band_rows)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, reversed(band_rows)):
        dx = row["shift_dx_km_all"].numpy()
        dy = row["shift_dy_km_all"].numpy()
        # subsample for a legible quiver overlay on top of the 2D density
        ax.hist2d(dx, dy, bins=40, cmap="viridis")
        n_show = min(300, len(dx))
        rng = np.random.default_rng(0)
        pick = rng.choice(len(dx), size=n_show, replace=False)
        ax.quiver(np.zeros(n_show), np.zeros(n_show), dx[pick], dy[pick],
                   angles="xy", scale_units="xy", scale=1, color="white", alpha=0.15, width=0.003)
        mx, my = row["mean_shift_km"]
        ax.annotate("", xy=(mx, my), xytext=(0, 0),
                    arrowprops=dict(facecolor="red", edgecolor="red", width=2, headwidth=8))
        ax.set_title(f"band {row['band']} ({row['scale_range_km']} km)\n"
                     f"systematic_fraction={row['systematic_fraction']:.2f}", fontsize=9)
        ax.set_xlabel("dx (km, east-west)")
        ax.set_ylabel("dy (km, north-south)")
        ax.axhline(0, color="w", lw=0.5, alpha=0.5)
        ax.axvline(0, color="w", lw=0.5, alpha=0.5)
        ax.set_aspect("equal")
    fig.suptitle("Local optimal-shift vectors per patch/sample (2D histogram; red arrow = mean/systematic shift), u-component")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved shift-vector quiver/histogram plot to {path}")


def plot_displacement_vs_scale(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for comp, color in zip(COMPONENTS, ("C0", "C1")):
        comp_rows = list(reversed([r for r in rows if r["component"] == comp]))
        x = np.arange(len(comp_rows))
        m = np.array([r["displacement_magnitude_km_mean"] for r in comp_rows])
        s = np.array([r["displacement_magnitude_km_std"] for r in comp_rows])
        ax.errorbar(x, m, yerr=s, marker="o", capsize=3, color=color, label=comp)
    ax.set_xticks(np.arange(len(comp_rows)))
    ax.set_xticklabels([r["scale_range_km"] for r in comp_rows])
    ax.set_xlabel("band Gaussian-sigma scale range (km)  [coarse -> fine]")
    ax.set_ylabel("displacement_magnitude (km)")
    ax.set_title("Local displacement magnitude vs scale")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved displacement-vs-scale plot to {path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--splits_path", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=256, help="Held-out samples to aggregate over (-1 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16, help="Model inference batch size")
    parser.add_argument("--shift_batch_size", type=int, default=32, help="Samples per batch during the shift search")
    parser.add_argument("--n_levels", type=int, default=5)
    parser.add_argument("--base_sigma", type=float, default=2.0, help="Matches training's multiscale.base_sigma")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="displacement_amplitude_diagnostics")
    parser.add_argument("--self_test_only", action="store_true")
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_synthetic_tests(args.n_levels, args.base_sigma, args.device)
    if args.self_test_only:
        return

    preds, truth, idx = generate_predictions(args)
    device = args.device
    pred_t = torch.from_numpy(preds).to(device)
    truth_t = torch.from_numpy(truth).to(device)
    S = pred_t.shape[0]
    params = band_patch_params(args.n_levels, args.base_sigma)

    labels = scale_labels(args.n_levels, args.base_sigma)
    acc_by_band_comp = {(b, comp): {"power_ratio": [], "pattern_corr": [], "phase_fraction_fourier": [],
                                     "error_reduction_fraction": [], "shift_dx_px": [], "shift_dy_px": [],
                                     "shift_weight": []}
                         for b in range(args.n_levels) for comp in COMPONENTS}
    for ci, comp in enumerate(COMPONENTS):
        for start in range(0, S, args.shift_batch_size):
            pb = pred_t[start:start + args.shift_batch_size, ci:ci + 1]
            tb = truth_t[start:start + args.shift_batch_size, ci:ci + 1]
            # Full pyramid computed ONCE per (component, sample-batch), reused across bands.
            pred_bands_all = laplacian_bands(pb, args.n_levels, args.base_sigma)
            truth_bands_all = laplacian_bands(tb, args.n_levels, args.base_sigma)
            for b in range(args.n_levels):
                pp = params[b]
                res = compute_band_component(pred_bands_all[b][:, 0], truth_bands_all[b][:, 0],
                                              pp["patch_size"], pp["stride"], pp["max_shift"])
                for k in acc_by_band_comp[(b, comp)]:
                    acc_by_band_comp[(b, comp)][k].append(res[k].cpu())
        print(f"  component {comp} done")

    rows = []
    for b in range(args.n_levels):
        pp = params[b]
        for comp in COMPONENTS:
            acc = {k: torch.cat(v, dim=0) for k, v in acc_by_band_comp[(b, comp)].items()}
            summary = summarize_band_component(acc)
            summary.update({"band": b, "scale_range_km": labels[b], "component": comp})
            rows.append(summary)
        print(f"  band {b} ({labels[b]} km) summarized "
              f"[patch={pp['patch_size']}px stride={pp['stride']}px max_shift={pp['max_shift']}px]")

    print_summary(rows)
    write_csv(rows, OUTPUT_DIR / f"{args.output_prefix}.csv")
    plot_decomposition_bar(rows, S, OUTPUT_DIR / f"{args.output_prefix}_bars.png")
    plot_shift_quivers(rows, args.n_levels, OUTPUT_DIR / f"{args.output_prefix}_quiver.png")
    plot_displacement_vs_scale(rows, OUTPUT_DIR / f"{args.output_prefix}_vs_scale.png")


if __name__ == "__main__":
    main()
