"""q50 coverage-bias diagnostic for the 0714 checkpoint (Parts A-D).

A one-time investigation into where and why the v6-0714 model's q50 is biased,
kept for provenance. Parts run in order and share state ON DISK: Part A writes the
per-event bias arrays that B, C and D all read back, so `--part all` (the default)
is the only ordering guaranteed to work from a clean tree.

Central, load-bearing statistical convention used by every Part: aggregate at the
EVENT level, never at the pixel/sample level. Samples within an event are
near-duplicates in time (same storm, consecutive hours), so treating them as
independent draws understates variance and can manufacture spurious "significant"
structure. `compute_event_bias` returns one bias map per EVENT (already averaged
over that event's samples); every mean/se/bootstrap below operates on the event
axis of that array, not on raw samples or pixels.

Parts
-----
  a           static (event-averaged) signed-error maps, on the FULL val and test splits
  b           does a smoothed static-bias correction actually improve coverage?
  c           is the bias explained by terrain strata (elevation, coastline distance)?
  d           is the bias explained by the skewness of the training target?
  a_meangate  Part A rerun against the v7 0729_meangate checkpoint, for comparison

Usage:
    python bias_diagnostic.py                 # all parts, in order
    python bias_diagnostic.py --part b        # one part (A must have run before)
    python bias_diagnostic.py --part a_meangate

Diagnostic only -- does not touch the model, loss, or training loop.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.stats import skew

from _v6_common import (
    COMPONENTS, DATA_DIR, OUTPUT_DIR, RUNS_DIR, bh_fdr_mask, coastline_distance,
    compute_event_bias, compute_event_predictions, crps_3q_approx, event_stats,
    load_elevation_200, load_event_split_map, load_land_mask, load_model, load_terrain,
    pinball, plot_diverging,
)
import eval_model_scorecard as scorecard  # v7 loader used by the meangate variant

OUT_DIR_MEANGATE = RUNS_DIR / "0729_meangate/figures/bias_diagnostic"



# --------------------------------------------------------------------------------------
# Part A: static (event-averaged) signed-error maps, on the FULL val and test splits
#
# Part A of the q50 coverage-bias diagnostic: static (event-averaged) signed-error maps for
# runs/0714/checkpoints/best.pth, on the FULL val and test splits (not a subsample -- coverage
# estimation variance from the earlier 96-sample subset was large enough to matter).
# 
# Everything here is event-level: for each event, average signed bias (q50 - truth) over that
# event's samples FIRST, then compute all statistics (mean, standard error, significance,
# variance-explained, land/ocean domain means) over the resulting one-map-per-event array. This
# treats each event as one independent unit; treating individual samples/pixels as independent
# would understate variance (samples within an event are near-duplicate consecutive hours).
# 
# u and v are never pooled -- every map, statistic, and plot is per component.
# 
# Outputs (runs/0714/figures/bias_diagnostic/):
#   event_bias_val.npy, event_bias_test.npy   (E,2,H,W) per-event bias, val/test
#   event_ids_val.npy, event_ids_test.npy
#   M_val.npy, se_val.npy, M_test.npy, se_test.npy   (2,H,W) each
#   sig_mask_val.npy, sig_mask_test.npy              (2,H,W) bool, BH-FDR q=0.05
#   static_bias_maps.png     M_val / M_test, per component, diverging, coastline overlay
#   significance_masks.png   BH-FDR significant pixels, per component, val/test
# 
# Diagnostic only -- does not touch the model, loss, or training loop.
# --------------------------------------------------------------------------------------

def variance_fraction_explained(event_bias: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per component: fraction of total (event x pixel) squared deviation from the domain-mean
    bias that is explained by the static per-pixel pattern M, i.e. R^2 of the model
    event_bias[e,p] = M[p] + noise (no per-event term). High => the bias is a persistent spatial
    pattern, not event-varying noise."""
    n_comp = event_bias.shape[1]
    out = np.zeros(n_comp)
    for c in range(n_comp):
        eb = event_bias[:, c]           # (E,H,W)
        m = M[c]                         # (H,W)
        grand_mean = m.mean()
        ss_total = np.sum((eb - grand_mean) ** 2)
        ss_resid = np.sum((eb - m[None]) ** 2)
        out[c] = 1.0 - ss_resid / ss_total
    return out


def domain_mean_bias(event_bias: np.ndarray, land_mask: np.ndarray):
    """Per component, per domain (land/ocean): event-level mean and standard error of the
    domain-mean bias (per-event domain mean first, THEN mean/se across events)."""
    n_comp = event_bias.shape[1]
    out = {}
    for c in range(n_comp):
        eb = event_bias[:, c]  # (E,H,W)
        land_means = eb[:, land_mask].mean(axis=1)     # (E,)
        ocean_means = eb[:, ~land_mask].mean(axis=1)    # (E,)
        out[c] = {
            "land": (float(land_means.mean()), float(land_means.std(ddof=1) / np.sqrt(len(land_means)))),
            "ocean": (float(ocean_means.mean()), float(ocean_means.std(ddof=1) / np.sqrt(len(ocean_means)))),
        }
    return out


def significance_mask(M: np.ndarray, se: np.ndarray, n_events: int, q: float = 0.05):
    """Two-sided t-test per pixel per component (df = n_events - 1), BH-FDR at level q."""
    n_comp = M.shape[0]
    masks = np.zeros_like(M, dtype=bool)
    pvals_all = np.zeros_like(M, dtype=np.float64)
    for c in range(n_comp):
        t = M[c] / np.clip(se[c], 1e-12, None)
        p = 2.0 * stats.t.sf(np.abs(t), df=n_events - 1)
        pvals_all[c] = p
        masks[c] = bh_fdr_mask(p, q=q)
    return masks, pvals_all


def run_split(model, device, terrain_raw, split_map, split: str, out_dir):
    print(f"\n=== {split} ===")
    event_ids, event_bias = compute_event_bias(model, device, terrain_raw, split_map, split)
    print(f"  {len(event_ids)} events, {event_bias.shape} bias array")
    M, se, n = event_stats(event_bias)
    np.save(out_dir / f"event_bias_{split}.npy", event_bias)
    np.save(out_dir / f"event_ids_{split}.npy", event_ids)
    np.save(out_dir / f"M_{split}.npy", M)
    np.save(out_dir / f"se_{split}.npy", se)

    masks, pvals = significance_mask(M, se, n)
    np.save(out_dir / f"sig_mask_{split}.npy", masks)

    fve = variance_fraction_explained(event_bias, M)
    land_mask = load_land_mask()
    dmb = domain_mean_bias(event_bias, land_mask)

    print(f"  n_events={n}")
    for c, comp in enumerate(COMPONENTS):
        frac_sig = masks[c].mean()
        print(f"  [{comp}] domain-mean(|M|)={np.abs(M[c]).mean():.4f} m/s  "
              f"FVE(static)={fve[c]:.3f}  frac_sig(BH q=0.05)={frac_sig:.3f}")
        land_m, land_se = dmb[c]["land"]
        ocean_m, ocean_se = dmb[c]["ocean"]
        print(f"       land  bias: {land_m:+.4f} +/- {land_se:.4f} m/s")
        print(f"       ocean bias: {ocean_m:+.4f} +/- {ocean_se:.4f} m/s")

    return dict(event_ids=event_ids, event_bias=event_bias, M=M, se=se, n=n,
                masks=masks, pvals=pvals, fve=fve, dmb=dmb)


def run_part_a():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(device)
    terrain_raw = load_terrain(device)
    split_map = load_event_split_map()
    print(f"Checkpoint: epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')}")

    val = run_split(model, device, terrain_raw, split_map, "val", out_dir)
    test = run_split(model, device, terrain_raw, split_map, "test", out_dir)

    land_mask = load_land_mask()

    # ── static bias maps ──
    fig, axes = plt.subplots(len(COMPONENTS), 2, figsize=(9.5, 4.4 * len(COMPONENTS)), squeeze=False)
    for c, comp in enumerate(COMPONENTS):
        im0 = plot_diverging(axes[c, 0], val["M"][c], land_mask, f"{comp}: M_val (n={val['n']} events)")
        plt.colorbar(im0, ax=axes[c, 0], shrink=0.8, label="m/s (pred - truth)")
        im1 = plot_diverging(axes[c, 1], test["M"][c], land_mask, f"{comp}: M_test (n={test['n']} events)")
        plt.colorbar(im1, ax=axes[c, 1], shrink=0.8, label="m/s (pred - truth)")
    fig.suptitle("Static signed q50 bias (event-averaged, then split-averaged) -- "
                 "0714 checkpoint\nlimits = per-panel 99th percentile of |bias|", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_dir / "static_bias_maps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_dir / 'static_bias_maps.png'}")

    # ── significance masks ──
    fig, axes = plt.subplots(len(COMPONENTS), 2, figsize=(9.5, 4.4 * len(COMPONENTS)), squeeze=False)
    for c, comp in enumerate(COMPONENTS):
        for j, (split_name, d) in enumerate((("val", val), ("test", test))):
            ax = axes[c, j]
            ax.imshow(d["masks"][c], cmap="gray_r", origin="upper", vmin=0, vmax=1)
            ax.contour(land_mask.astype(float), levels=[0.5], colors="red", linewidths=0.6)
            ax.set_title(f"{comp}: {split_name} BH-FDR significant (q=0.05), "
                         f"{d['masks'][c].mean()*100:.1f}% of pixels", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Pixels where the static bias is significant at BH-FDR q=0.05\n"
                 "(two-sided t-test on event-level bias, df = n_events - 1)", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_dir / "significance_masks.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'significance_masks.png'}")

    print("\nDone with Part A.")


# --------------------------------------------------------------------------------------
# Part B: does a smoothed static-bias correction actually improve coverage?
#
# Part B (the decisive one) of the q50 coverage-bias diagnostic: does the static bias pattern
# found in Part A (M_val) actually TRANSFER to an independent split (M_test)? If it does, a
# static per-pixel correction should help; if it doesn't, Part A's structure is val-specific
# noise and no correction (nor Part C's stratification) is meaningful.
# 
# 1. Spatial correlation of M_val vs M_test (overall / land / ocean, per component), with an
#    event-bootstrap CI (resample val's events and test's events independently, recompute both
#    maps and the correlation each draw -- this measures how much of the observed correlation
#    survives event-sampling noise in EACH map's own estimate).
# 2. A correction q50_corr = q50 - lambda * G_sigma(M), with sigma (Gaussian-smoothing
#    bandwidth, pixels) and lambda (strength) selected via grid search ONLY on a 50/50
#    event-level split of val (val-A builds the candidate M, val-B's q50 MAE picks sigma/lambda)
#    -- test is never touched during selection. The final M is then re-estimated on the FULL val
#    split (more data) at the selected (sigma, lambda) and applied to test.
# 3. Two variants: band-shift (shifts q10/q50/q90 together, monotonicity trivially preserved) and
#    median-only (shifts only q50; a monotonicity-violation check is reported, not assumed away).
# 4. Before/after test metrics (MAE, RMSE, crps_3q_approx, coverage at q10/50/90 aggregate + maps,
#    extreme-stratum coverage) with PAIRED event-level bootstrap CIs on the (after - before)
#    difference.
# 
# If the selected lambda is ~0 for a component, the honest conclusion is "the correction doesn't
# help" for that component -- printed explicitly, and Part C should not be run for a component
# where this holds for both u and v.
# 
# Diagnostic only -- does not touch the model, loss, or training loop; never fits on test.
# --------------------------------------------------------------------------------------

SIGMAS = [0, 2, 4, 8, 16, 32, 64]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
LAMBDA_NEAR_ZERO = 0.05
N_BOOT = 1000


def gaussian_smooth(M2d: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return M2d
    return gaussian_filter(M2d, sigma=sigma, mode="nearest")


def spatial_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float:
    av = a[mask] if mask is not None else a.ravel()
    bv = b[mask] if mask is not None else b.ravel()
    return float(np.corrcoef(av, bv)[0, 1])


def bootstrap_corr(event_bias_val, event_bias_test, mask_dict, n_boot=N_BOOT, seed=1):
    rng = np.random.default_rng(seed)
    nV, nT = event_bias_val.shape[0], event_bias_test.shape[0]
    out = {name: {c: [] for c in range(2)} for name in mask_dict}
    for _ in range(n_boot):
        vi = rng.integers(0, nV, nV)
        ti = rng.integers(0, nT, nT)
        Mv = event_bias_val[vi].mean(axis=0)
        Mt = event_bias_test[ti].mean(axis=0)
        for name, mask in mask_dict.items():
            for c in range(2):
                out[name][c].append(spatial_corr(Mv[c], Mt[c], mask))
    return out


def select_sigma_lambda(event_bias_val, event_ids_val, q50_val, truth_val, seed=0):
    n = len(event_ids_val)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    A_idx, B_idx = perm[:half], perm[half:]
    M_A = event_bias_val[A_idx].mean(axis=0)  # (2,H,W)
    q50_B, truth_B = q50_val[B_idx], truth_val[B_idx]

    results = {}
    for c in range(2):
        best_score, best_sigma, best_lambda = np.inf, 0.0, 0.0
        grid = np.zeros((len(SIGMAS), len(LAMBDAS)))
        for si, sigma in enumerate(SIGMAS):
            Gs = gaussian_smooth(M_A[c], sigma)
            for li, lam in enumerate(LAMBDAS):
                corrected = q50_B[:, c] - lam * Gs[None]
                mae = np.abs(corrected - truth_B[:, c]).mean()
                grid[si, li] = mae
                if mae < best_score:
                    best_score, best_sigma, best_lambda = mae, sigma, lam
        baseline_mae = np.abs(q50_B[:, c] - truth_B[:, c]).mean()
        results[c] = dict(sigma=best_sigma, lam=best_lambda, valB_mae=best_score,
                           valB_mae_uncorrected=baseline_mae, grid=grid)
    return results, A_idx, B_idx


def compute_metrics(q10, q50, q90, truth):
    err = q50 - truth
    mae = np.abs(err).mean(axis=(2, 3))
    rmse = np.sqrt((err ** 2).mean(axis=(2, 3)))
    crps = crps_3q_approx(q10, q50, q90, truth).mean(axis=(2, 3))
    cov10 = (truth <= q10).mean(axis=(2, 3))
    cov50 = (truth <= q50).mean(axis=(2, 3))
    cov90 = (truth <= q90).mean(axis=(2, 3))
    thr = np.quantile(np.abs(truth), 0.9, axis=(2, 3), keepdims=True)
    ext = np.abs(truth) >= thr

    def masked_frac(cond):
        num = (cond & ext).sum(axis=(2, 3)).astype(np.float64)
        den = ext.sum(axis=(2, 3)).astype(np.float64)
        return num / np.clip(den, 1, None)

    return dict(mae=mae, rmse=rmse, crps=crps, cov10=cov10, cov50=cov50, cov90=cov90,
                cov10_ext=masked_frac(truth <= q10), cov50_ext=masked_frac(truth <= q50),
                cov90_ext=masked_frac(truth <= q90))


def paired_bootstrap_ci(before: dict, after: dict, n_boot=N_BOOT, seed=2):
    n = before["mae"].shape[0]
    rng = np.random.default_rng(seed)
    out = {}
    for key in before:
        diffs_all = after[key] - before[key]  # (E,2)
        boot = np.zeros((n_boot, 2))
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            boot[b] = diffs_all[idx].mean(axis=0)
        point = diffs_all.mean(axis=0)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        out[key] = dict(point=point, lo=lo, hi=hi)
    return out


def monotonicity_violation_frac(q10, q50_corr, q90):
    viol = (q50_corr < q10) | (q50_corr > q90)
    return viol.mean(axis=(0, 2, 3))


def run_part_b():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    land_mask = load_land_mask()

    model, ckpt = load_model(device)
    terrain_raw = load_terrain(device)
    split_map = load_event_split_map()
    print(f"Checkpoint: epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')}")

    print("\nRunning full val split (q10/q50/q90/truth per event)...")
    ev_val, q10_val, q50_val, q90_val, truth_val = compute_event_predictions(
        model, device, terrain_raw, split_map, "val")
    event_bias_val = q50_val - truth_val
    print(f"  {len(ev_val)} val events")

    print("Running full test split (q10/q50/q90/truth per event)...")
    ev_test, q10_test, q50_test, q90_test, truth_test = compute_event_predictions(
        model, device, terrain_raw, split_map, "test")
    event_bias_test = q50_test - truth_test
    print(f"  {len(ev_test)} test events")

    M_val = event_bias_val.mean(axis=0)
    M_test = event_bias_test.mean(axis=0)

    # ── 1. spatial correlation, val vs test ──
    print("\n=== Spatial correlation of M_val vs M_test ===")
    mask_dict = {"overall": None, "land": land_mask, "ocean": ~land_mask}
    boot = bootstrap_corr(event_bias_val, event_bias_test, mask_dict)
    for name, mask in mask_dict.items():
        for c, comp in enumerate(COMPONENTS):
            point = spatial_corr(M_val[c], M_test[c], mask)
            lo, hi = np.percentile(boot[name][c], [2.5, 97.5])
            print(f"  [{comp}] {name:8s}: r={point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"(n_boot={N_BOOT}, event-resampled)")

    # ── 2. select sigma/lambda on val-A/val-B ──
    print("\n=== Correction hyperparameter selection (val-A fits M, val-B picks sigma/lambda) ===")
    sel, A_idx, B_idx = select_sigma_lambda(event_bias_val, ev_val, q50_val, truth_val)
    lambdas_selected = {}
    for c, comp in enumerate(COMPONENTS):
        r = sel[c]
        lambdas_selected[c] = r["lam"]
        print(f"  [{comp}] selected sigma={r['sigma']}, lambda={r['lam']:.2f}  "
              f"val-B MAE: {r['valB_mae']:.4f} (uncorrected: {r['valB_mae_uncorrected']:.4f})")

    near_zero = all(abs(lambdas_selected[c]) < LAMBDA_NEAR_ZERO for c in range(2))
    if near_zero:
        print(f"\n  Selected lambda ~0 for both components (< {LAMBDA_NEAR_ZERO}): the "
              "correction does not help on held-out val data. Honest conclusion: the static "
              "bias pattern does not transfer usefully. STOPPING before applying any "
              "correction to test, and Part C should NOT be run.")
        np.save(out_dir / "part_b_verdict.npy", np.array([False]))
        return
    for c, comp in enumerate(COMPONENTS):
        if abs(lambdas_selected[c]) < LAMBDA_NEAR_ZERO:
            print(f"  NOTE: [{comp}] lambda ~0 -- correction does not help for this component "
                  "specifically, even though the other component transfers.")

    # ── 3. build the final correction (fit on FULL val), apply to test ──
    print("\n=== Applying correction to test (never fit on test) ===")
    shift = np.zeros((2, 200, 200))
    for c in range(2):
        Gs_full = gaussian_smooth(M_val[c], sel[c]["sigma"])
        shift[c] = sel[c]["lam"] * Gs_full

    q10_bs = q10_test - shift[None]
    q50_bs = q50_test - shift[None]
    q90_bs = q90_test - shift[None]

    q10_mo = q10_test
    q50_mo = q50_test - shift[None]
    q90_mo = q90_test
    mono_viol = monotonicity_violation_frac(q10_test, q50_mo, q90_test)
    for c, comp in enumerate(COMPONENTS):
        print(f"  [{comp}] median-only monotonicity violations: {mono_viol[c]*100:.2f}% "
              "of (event, pixel)")

    before = compute_metrics(q10_test, q50_test, q90_test, truth_test)
    after_bs = compute_metrics(q10_bs, q50_bs, q90_bs, truth_test)
    after_mo = compute_metrics(q10_mo, q50_mo, q90_mo, truth_test)

    for variant_name, after in (("band-shift", after_bs), ("median-only", after_mo)):
        print(f"\n=== Test metrics, before vs after ({variant_name}), paired event-bootstrap CI ===")
        ci = paired_bootstrap_ci(before, after)
        for key in ["mae", "rmse", "crps", "cov10", "cov50", "cov90", "cov10_ext", "cov50_ext", "cov90_ext"]:
            for c, comp in enumerate(COMPONENTS):
                b, a = before[key][:, c].mean(), after[key][:, c].mean()
                d = ci[key]
                print(f"  [{comp}] {key:9s}: before={b:.4f}  after={a:.4f}  "
                      f"Delta={d['point'][c]:+.4f} [{d['lo'][c]:+.4f}, {d['hi'][c]:+.4f}]")

    # ── save arrays ──
    np.save(out_dir / "part_b_verdict.npy", np.array([True]))
    np.save(out_dir / "correction_shift.npy", shift)
    np.save(out_dir / "M_val_full.npy", M_val)
    for c, comp in enumerate(COMPONENTS):
        np.save(out_dir / f"selected_sigma_lambda_{comp}.npy",
                np.array([sel[c]["sigma"], sel[c]["lam"]]))

    # ── coverage maps: q50 before/after (band-shift) ──
    cov50_before_map = (truth_test <= q50_test).mean(axis=0)  # (2,H,W)
    cov50_after_map = (truth_test <= q50_bs).mean(axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0), squeeze=False)
    for c, comp in enumerate(COMPONENTS):
        before_dev = cov50_before_map[c] - 0.5
        after_dev = cov50_after_map[c] - 0.5
        # Shared scale per component (before vs after) so the two panels are visually comparable.
        vmax = float(max(np.percentile(np.abs(before_dev), 99), np.percentile(np.abs(after_dev), 99)))
        im0 = plot_diverging(axes[c, 0], before_dev, land_mask,
                              f"{comp}: q50 coverage - 0.5, before", vmax=vmax)
        plt.colorbar(im0, ax=axes[c, 0], shrink=0.8)
        im1 = plot_diverging(axes[c, 1], after_dev, land_mask,
                              f"{comp}: q50 coverage - 0.5, after (band-shift)", vmax=vmax)
        plt.colorbar(im1, ax=axes[c, 1], shrink=0.8)
    fig.suptitle("q50 coverage deviation on test, before vs after static correction", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_dir / "correction_coverage_before_after.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_dir / 'correction_coverage_before_after.png'}")
    print("\nDone with Part B.")


# --------------------------------------------------------------------------------------
# Part C: is the bias explained by terrain strata (elevation, coastline distance)?
#
# Part C of the q50 coverage-bias diagnostic: stratify the static bias pattern by land/ocean,
# elevation quartiles, and coastline distance -- to characterize WHERE the bias that Part B
# showed transfers to test actually lives.
# 
# Gated on Part B: run only if Part B found the bias transfers (selected lambda not ~0 for at
# least one component). Retrieval-distance terciles (the brief's optional 4th axis) are skipped
# -- data/neighbors.npy stores only k-NN indices, not the distances themselves, so this axis is
# not cheaply recoverable from stored artifacts (confirmed by reading nearest_neighbors.py; it
# computes distances transiently and never persists them).
# 
# Every stat is event-level: per event, take the mean bias within a stratum's pixels FIRST, then
# report mean/SE across events of that per-event stratum mean. u and v never pooled.
# 
# Diagnostic only -- does not touch the model, loss, or training loop.
# --------------------------------------------------------------------------------------

def stratum_event_stats(event_bias: np.ndarray, mask: np.ndarray):
    """event_bias: (E,2,H,W); mask: (H,W) bool. Returns (mean, se) per component over the
    per-event within-mask mean bias."""
    out = []
    for c in range(event_bias.shape[1]):
        per_event = event_bias[:, c][:, mask].mean(axis=1)  # (E,)
        out.append((float(per_event.mean()), float(per_event.std(ddof=1) / np.sqrt(len(per_event)))))
    return out


def report_strata(event_bias: np.ndarray, strata: list[tuple[str, np.ndarray]], split_name: str):
    print(f"\n  -- {split_name} --")
    for name, mask in strata:
        n_px = int(mask.sum())
        stats = stratum_event_stats(event_bias, mask)
        line = "  ".join(f"{comp}={m:+.4f}+/-{se:.4f}" for comp, (m, se) in zip(COMPONENTS, stats))
        print(f"    {name:22s} (n_px={n_px:5d}): {line}")


def run_part_c():
    out_dir = OUTPUT_DIR
    verdict_path = out_dir / "part_b_verdict.npy"
    if not verdict_path.exists():
        print("Part B has not been run yet (no part_b_verdict.npy). Run Part B first.")
        return
    if not bool(np.load(verdict_path)[0]):
        print("Part B found the static bias does NOT transfer to test (selected lambda ~0 for "
              "both components). Per the diagnostic brief, Part C should not be run on noise. "
              "Stopping.")
        return

    event_bias_val = np.load(out_dir / "event_bias_val.npy")
    event_bias_test = np.load(out_dir / "event_bias_test.npy")
    land_mask = load_land_mask()
    elevation = load_elevation_200()
    coast_dist = coastline_distance(land_mask)

    print("=== Land / ocean ===")
    strata = [("land", land_mask), ("ocean", ~land_mask)]
    for split_name, eb in (("val", event_bias_val), ("test", event_bias_test)):
        report_strata(eb, strata, split_name)

    print("\n=== Elevation quartiles (land pixels only -- ocean elevation is not meaningful) ===")
    land_elev = elevation[land_mask]
    edges = np.quantile(land_elev, [0.0, 0.25, 0.5, 0.75, 1.0])
    strata = []
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        m = land_mask & (elevation >= lo) & (elevation <= hi if q == 3 else elevation < hi)
        strata.append((f"elev_q{q+1} [{lo:.0f},{hi:.0f}m]", m))
    for split_name, eb in (("val", event_bias_val), ("test", event_bias_test)):
        report_strata(eb, strata, split_name)

    print("\n=== Coastline-distance quartiles (all pixels, land + ocean near/far from coast) ===")
    edges = np.quantile(coast_dist, [0.0, 0.25, 0.5, 0.75, 1.0])
    strata = []
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        m = (coast_dist >= lo) & (coast_dist <= hi if q == 3 else coast_dist < hi)
        strata.append((f"coast_q{q+1} [{lo:.1f},{hi:.1f}px]", m))
    for split_name, eb in (("val", event_bias_val), ("test", event_bias_test)):
        report_strata(eb, strata, split_name)

    print("\n[Retrieval-distance terciles skipped: not recoverable from stored artifacts "
          "(data/neighbors.npy holds only k-NN indices; distances were computed transiently by "
          "nearest_neighbors.py and never saved).]")
    print("\nDone with Part C.")


# --------------------------------------------------------------------------------------
# Part D: is the bias explained by the skewness of the training target?
#
# Part D of the q50 coverage-bias diagnostic: discriminate two hypotheses for the static q50
# bias found in Part A/B.
# 
# H1 (loss-induced tilt): FreqBandLoss's down_extremes pixel weighting (see multiscale_loss.py,
# cdf_weight_mode="down_extremes") down-weights extreme-residual pixels in the structural loss
# that trains q50. Wherever the truth-vs-first-guess residual distribution is SKEWED, the
# pinball/L1-family minimizer under that weighting is not the conditional median -- it is tilted
# toward whichever tail is being down-weighted. So if H1 drives the bias, pixels with skewed
# training residuals should predict pixels with large bias, independent of terrain.
# 
# H2 (genuine model/terrain bias): the bias reflects the model's actual spatial error structure
# (e.g. systematic underestimation over ocean, coastal effects) with no particular connection to
# residual skewness.
# 
# Discriminating statistic: S[pixel] = skew(truth - ens_cgp_mean) over the TRAINING split only
# (never val/test -- this must be a property of the data/first-guess, not contaminated by the
# model's own predictions or by val/test statistics). Skewness is a POPULATION property of the
# residual distribution and is estimated by pooling all training samples per pixel (unlike the
# bias maps M, this needs no event-level treatment: it is not a mean+SE where within-event
# correlation would understate variance, and no significance test is computed on S itself --
# only on corr(S, M), which IS event-bootstrapped through M).
# 
# Reported: corr(S, M_val), corr(S, M_test) with event-bootstrap CI (M resampled by event, S
# fixed), plus the actual discriminating statistic -- the PARTIAL correlation of S with M after
# regressing out elevation and land-mask from both S and M (linear regression residuals). If the
# partial correlation survives (stays large), skewness predicts bias independent of terrain,
# supporting H1. If it collapses toward 0, the raw correlation was a terrain confound and H1 is
# not supported by this test.
# 
# Diagnostic only -- does not touch the model, loss, or training loop.
# --------------------------------------------------------------------------------------



def compute_skewness_map(train_idx: np.ndarray) -> np.ndarray:
    """S[c,H,W] = skew over training samples of (truth - ens_cgp_mean), per component."""
    wrf = np.load(DATA_DIR / "wrf_uv.npy", mmap_mode="r")
    posterior = np.load(DATA_DIR / "enscgp_posterior.npy", mmap_mode="r")
    truth = np.array(wrf[train_idx], dtype=np.float64, copy=True)          # (N,2,H,W)
    ens_mean = np.array(posterior[train_idx, 0:2], dtype=np.float64, copy=True)  # (N,2,H,W)
    resid = truth - ens_mean
    S = skew(resid, axis=0, bias=False)  # (2,H,W)
    return S


def regress_out(y: np.ndarray, covariates: list[np.ndarray]) -> np.ndarray:
    """y, each covariate: flat (n_pixels,). Returns residual of y after OLS on
    [intercept, *covariates]."""
    X = np.column_stack([np.ones_like(y)] + covariates)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def bootstrap_corr_S_M(S: np.ndarray, event_bias: np.ndarray, n_boot=N_BOOT, seed=3):
    """S: (2,H,W) fixed. event_bias: (E,2,H,W). Resample events, recompute M, correlate with S."""
    rng = np.random.default_rng(seed)
    n = event_bias.shape[0]
    out = {c: [] for c in range(2)}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        M = event_bias[idx].mean(axis=0)
        for c in range(2):
            out[c].append(float(np.corrcoef(S[c].ravel(), M[c].ravel())[0, 1]))
    return out


def run_part_d():
    out_dir = OUTPUT_DIR
    splits = np.load(DATA_DIR / "splits_70_15_15/split_indices.npz")
    train_idx = splits["train_idx"]
    land_mask = load_land_mask()
    elevation = load_elevation_200()

    print(f"Computing training-split-only skewness proxy over {len(train_idx)} samples...")
    S = compute_skewness_map(train_idx)
    np.save(out_dir / "skewness_proxy_S.npy", S)
    for c, comp in enumerate(COMPONENTS):
        print(f"  [{comp}] S: mean={S[c].mean():+.3f}  std={S[c].std():.3f}  "
              f"|S| land={np.abs(S[c][land_mask]).mean():.3f}  ocean={np.abs(S[c][~land_mask]).mean():.3f}")

    event_bias_val = np.load(out_dir / "event_bias_val.npy")
    event_bias_test = np.load(out_dir / "event_bias_test.npy")
    M_val = event_bias_val.mean(axis=0)
    M_test = event_bias_test.mean(axis=0)

    elev_flat = elevation.ravel()
    land_flat = land_mask.astype(np.float64).ravel()

    print("\n=== corr(S, M) -- raw, and partial (controlling for elevation + land-mask) ===")
    for split_name, event_bias, M in (("val", event_bias_val, M_val), ("test", event_bias_test, M_test)):
        boot = bootstrap_corr_S_M(S, event_bias)
        for c, comp in enumerate(COMPONENTS):
            raw_r = float(np.corrcoef(S[c].ravel(), M[c].ravel())[0, 1])
            lo, hi = np.percentile(boot[c], [2.5, 97.5])

            S_resid = regress_out(S[c].ravel(), [elev_flat, land_flat])
            M_resid = regress_out(M[c].ravel(), [elev_flat, land_flat])
            partial_r = float(np.corrcoef(S_resid, M_resid)[0, 1])

            print(f"  [{comp}] vs M_{split_name}: raw r={raw_r:+.3f} [{lo:+.3f}, {hi:+.3f}]   "
                  f"partial r (elev+land controlled)={partial_r:+.3f}")

    print("\nReading guide: if the partial correlation stays close to the raw correlation, "
          "skewness predicts bias independent of terrain (supports H1, loss-induced tilt). If "
          "the partial correlation collapses toward 0, the raw correlation was a terrain "
          "confound (does not support H1 via this test; consistent with H2 or an "
          "unidentified terrain-linked mechanism).")
    print("\nDone with Part D.")


# --------------------------------------------------------------------------------------
# Part A_MEANGATE: Part A rerun against the v7 0729_meangate checkpoint, for comparison
#
# Part A of the q50 coverage-bias diagnostic, re-run against runs/0729_meangate/checkpoints/best.pth
# (v7 architecture) instead of 0714 -- prompted by that run's quantile_coverage_maps.png visibly
# differing from 0714's.
# 
# _v6_common.py's load_model()/CHECKPOINT are hardcoded to 0714's v6 architecture
# class (a different nn.Module -- different EnsCGP-sigma seeding / offset_gate -- so 0714's
# checkpoint fails strict load_state_dict against the current v7 class and vice versa; see
# _v6_common's own docstring). Everything else in _v6_common
# (compute_event_bias, event_stats, bh_fdr_mask, load_land_mask, plot_diverging,
# load_event_split_map, load_terrain) is architecture-agnostic -- it only calls `model(...)` and
# slices Q50_SLICE, which is identical (slice(2,4)) on both classes. So this script builds the v7
# model via eval_model_scorecard.py's already-established v7 loader (MODELS["v7-meangate"],
# load_v7_class/build_and_load -- reused, not reimplemented) and otherwise reuses
# bias_diagnostic_part_a.py's analysis functions (run_split, and the plotting logic, adapted for
# one checkpoint instead of a val/test pair against a fixed 0714 baseline) UNCHANGED.
# 
# CAVEAT: runs/0729_meangate/checkpoints/best.pth is being actively overwritten by a live
# training run as of this diagnostic (started 2026-08-04 11:35, still training) -- this is a
# snapshot of whatever epoch's checkpoint happens to be on disk at run time, not a finished
# model. The script prints the checkpoint's epoch/val score so the snapshot is identifiable.
# 
# Outputs -- runs/0729_meangate/figures/bias_diagnostic/ (separate from 0714's, which this does
# NOT touch or overwrite):
#   event_bias_val.npy, event_bias_test.npy, event_ids_val.npy, event_ids_test.npy
#   M_val.npy, se_val.npy, M_test.npy, se_test.npy, sig_mask_val.npy, sig_mask_test.npy
#   static_bias_maps.png, significance_masks.png
# 
# Diagnostic only -- does not touch the model, loss, or training loop (and does not interfere
# with the concurrently-running training job; read-only checkpoint load, separate GPU device).
# --------------------------------------------------------------------------------------

def run_part_a_meangate():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # idle GPU as of this run
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, slices, ckpt = scorecard.build_and_load("v7-meangate", device)
    print(f"Checkpoint: {scorecard.MODELS['v7-meangate']['checkpoint']} "
         f"(epoch {ckpt.get('epoch')}, best_val_score {ckpt.get('best_val_score', ckpt.get('best_val_loss'))}) "
         "-- NOTE: this run was still actively training as of this diagnostic; re-run to "
         "refresh once it finishes.")

    terrain_raw = scorecard.common.load_terrain(device)
    split_map = load_event_split_map()

    val = run_split(model, device, terrain_raw, split_map, "val", OUT_DIR)
    test = run_split(model, device, terrain_raw, split_map, "test", OUT_DIR)

    land_mask = load_land_mask()

    fig, axes = plt.subplots(len(COMPONENTS), 2, figsize=(9.5, 4.4 * len(COMPONENTS)), squeeze=False)
    for c, comp in enumerate(COMPONENTS):
        im0 = plot_diverging(axes[c, 0], val["M"][c], land_mask, f"{comp}: M_val (n={val['n']} events)")
        plt.colorbar(im0, ax=axes[c, 0], shrink=0.8, label="m/s (pred - truth)")
        im1 = plot_diverging(axes[c, 1], test["M"][c], land_mask, f"{comp}: M_test (n={test['n']} events)")
        plt.colorbar(im1, ax=axes[c, 1], shrink=0.8, label="m/s (pred - truth)")
    fig.suptitle(f"Static signed q50 bias (event-averaged, then split-averaged) -- "
                f"0729_meangate checkpoint (epoch {ckpt.get('epoch')}, IN PROGRESS)\n"
                "limits = per-panel 99th percentile of |bias|", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "static_bias_maps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_DIR / 'static_bias_maps.png'}")

    fig, axes = plt.subplots(len(COMPONENTS), 2, figsize=(9.5, 4.4 * len(COMPONENTS)), squeeze=False)
    for c, comp in enumerate(COMPONENTS):
        for j, (split_name, d) in enumerate((("val", val), ("test", test))):
            ax = axes[c, j]
            ax.imshow(d["masks"][c], cmap="gray_r", origin="upper", vmin=0, vmax=1)
            ax.contour(land_mask.astype(float), levels=[0.5], colors="red", linewidths=0.6)
            ax.set_title(f"{comp}: {split_name} BH-FDR significant (q=0.05), "
                        f"{d['masks'][c].mean()*100:.1f}% of pixels", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Pixels where the static bias is significant at BH-FDR q=0.05 -- 0729_meangate\n"
                "(two-sided t-test on event-level bias, df = n_events - 1)", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "significance_masks.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'significance_masks.png'}")

    print("\nDone with Part A (0729_meangate).")


PARTS = {
    "a": run_part_a,
    "b": run_part_b,
    "c": run_part_c,
    "d": run_part_d,
    "a_meangate": run_part_a_meangate,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=[*PARTS, "all"], default="all",
                   help="'all' runs a, b, c, d in order (a_meangate is opt-in)")
    args = p.parse_args(argv)

    todo = ["a", "b", "c", "d"] if args.part == "all" else [args.part]
    for name in todo:
        print(f"\n{'=' * 72}\nPart {name.upper()}\n{'=' * 72}")
        PARTS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
