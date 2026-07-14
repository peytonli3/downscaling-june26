"""Verify the variance-conditioning change to ProbabilisticSwin2SR (new_enscgp_swin.py):

(A) --init_check: build a conditioned model FROM the baseline (unconditioned) checkpoint --
    chol_cond is zero-initialized -- and assert its output equals the unconditioned model's
    output on real samples (mean AND covariance), confirming conditioning is a no-op at init
    (so fine-tuning truly "continues from" the checkpoint).

(B) default (with --finetuned): compare the fine-tuned conditioned model against the baseline
    on the held-out split. Because only the variance head moved, the predicted MEAN must be
    unchanged (asserted) -- so |error| is the SAME field for both, and the question is purely
    whether the new sigma tracks that fixed error better. Reports, baseline vs fine-tuned:
      - aggregate z-score calibration (mean/std, |z|<=1, |z|<=2; should stay ~N(0,1)),
      - Pearson correlation between predicted sigma and |error|, over ALL pixels and over the
        "blotch" pixels (should INCREASE -- sigma now rises where the model bets wrong),
      - mean |z| over the blotch pixels (should DECREASE -- inflated sigma now covers the
        wrong-bet errors).
    Blotches are defined from the BASELINE z magnitude sqrt(z_u^2+z_v^2): per sample, the
    pixels above its --blotch_pct percentile (the current model's own wrong-bet regions), so
    both models are scored on the same regions.

Mirrors eval_zscore_calibration.py's data plumbing (forward(posterior, bicubic, terrain_raw
[, prior_spread]); test-split sampling). The conditioned model additionally needs
enscgp_prior_spread.npy + cond_feature_stats.npz (precompute_prior_spread.py).

Usage:
    python verify_variance_conditioning.py --init_check
    python verify_variance_conditioning.py --finetuned .../best_variance.pth --n_samples 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = "/home/peytonli/26.6_wind/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from new_enscgp_swin import DEFAULT_CONFIG_PATH, build_model, load_config  # noqa: E402
from terrain_encoder import load_terrain_input  # noqa: E402

EPS = 1e-6


def build_baseline(config, checkpoint_path, device):
    """Unconditioned model (variance_conditioning off) with the baseline checkpoint."""
    cfg = json.loads(json.dumps(config))
    cfg["model"]["variance_conditioning"] = False
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def build_conditioned(config, checkpoint_path, stats_path, device, strict_load):
    """Conditioned model. strict_load=False for the init check (loading the baseline
    checkpoint, chol_cond fresh zero-init); strict_load=True for a fine-tuned checkpoint."""
    cfg = json.loads(json.dumps(config))
    cfg["model"]["variance_conditioning"] = True
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict_load)
    stats = np.load(stats_path)
    model.load_cond_stats(stats["cond_mean"], stats["cond_std"])
    model.eval()
    return model


def load_arrays(data_dir):
    return (
        np.load(data_dir / "enscgp_posterior.npy", mmap_mode="r"),
        np.load(data_dir / "era5_uv_2ch_bicubic.npy", mmap_mode="r"),
        np.load(data_dir / "wrf_uv.npy", mmap_mode="r"),
        np.load(data_dir / "enscgp_prior_spread.npy", mmap_mode="r"),
    )


def batched_indices(idx, batch_size):
    for s in range(0, len(idx), batch_size):
        yield idx[s:s + batch_size]


@torch.no_grad()
def run_model(model, idx, posterior, bicubic, prior_spread, terrain_raw, device, batch_size, conditioned):
    """Return stacked predictions (len(idx), 5, H, W) -> mu_u, mu_v, L11, L21, L22."""
    out = []
    for b in batched_indices(idx, batch_size):
        post = torch.from_numpy(np.array(posterior[b], dtype=np.float32, copy=True)).to(device)
        bic = torch.from_numpy(np.array(bicubic[b], dtype=np.float32, copy=True)).to(device)
        if conditioned:
            ps = torch.from_numpy(np.array(prior_spread[b], dtype=np.float32, copy=True)).to(device)
            pred = model(post, bic, terrain_raw, ps)
        else:
            pred = model(post, bic, terrain_raw)
        out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0)


def sigma_components(pred):
    """sigma_u = L11, sigma_v = sqrt(L21^2 + L22^2) from the Cholesky channels."""
    L11, L21, L22 = pred[:, 2], pred[:, 3], pred[:, 4]
    return L11, np.sqrt(L21 ** 2 + L22 ** 2)


def pearson(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def calibration_line(tag, z):
    m, s = float(z.mean()), float(z.std())
    f1, f2 = float(np.mean(np.abs(z) <= 1)), float(np.mean(np.abs(z) <= 2))
    return f"  {tag}: mean={m:+.3f} std={s:.3f} |z|<=1 {f1:.1%} (68.3%) |z|<=2 {f2:.1%} (95.4%)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--baseline", type=Path, default=None, help="Unconditioned checkpoint; defaults to <log_dir>/checkpoints/best.pth")
    parser.add_argument("--finetuned", type=Path, default=None, help="Fine-tuned conditioned checkpoint (best_variance.pth)")
    parser.add_argument("--init_check", action="store_true",
                        help="Only verify conditioning is a no-op at init (conditioned-from-baseline == baseline)")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--stats_path", type=Path, default=None, help="Defaults to <data_dir>/cond_feature_stats.npz")
    parser.add_argument("--splits_path", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--blotch_pct", type=float, default=90.0,
                        help="Per-sample percentile of baseline |z| magnitude defining blotch pixels (default top 10%%)")
    parser.add_argument("--init_tol", type=float, default=1e-4, help="Max abs output diff allowed in --init_check")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = args.data_dir or Path(paths["data_dir"])
    baseline_ckpt = args.baseline or (Path(paths["log_dir"]) / "checkpoints" / "best.pth")
    stats_path = args.stats_path or (data_dir / "cond_feature_stats.npz")
    splits_path = args.splits_path or Path(paths["splits_path"])
    device = torch.device(args.device)

    posterior, bicubic, wrf, prior_spread = load_arrays(data_dir)
    terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device)

    splits = np.load(splits_path)
    pool = splits[f"{args.split}_idx"]
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(pool))
    idx = np.sort(rng.choice(pool, size=n, replace=False))

    # ---- (A) init no-op check ----
    if args.init_check:
        baseline = build_baseline(config, baseline_ckpt, device)
        cond = build_conditioned(config, baseline_ckpt, stats_path, device, strict_load=False)
        small = idx[: min(8, len(idx))]
        base_out = run_model(baseline, small, posterior, bicubic, prior_spread, terrain_raw, device, args.batch_size, False)
        cond_out = run_model(cond, small, posterior, bicubic, prior_spread, terrain_raw, device, args.batch_size, True)
        max_diff = float(np.abs(cond_out - base_out).max())
        print(f"[init_check] max|conditioned - baseline| over {len(small)} samples (all 5 channels): {max_diff:.3e}")
        assert max_diff < args.init_tol, (
            f"conditioning is NOT a no-op at init (max diff {max_diff:.3e} >= tol {args.init_tol}); "
            "chol_cond should be zero-initialized"
        )
        print(f"[init_check] PASS: conditioned model output == baseline within {args.init_tol:.0e}.")
        return

    if args.finetuned is None:
        parser.error("provide --finetuned <best_variance.pth>, or use --init_check")

    # ---- (B) baseline vs fine-tuned ----
    baseline = build_baseline(config, baseline_ckpt, device)
    finetuned = build_conditioned(config, args.finetuned, stats_path, device, strict_load=True)

    base_pred = run_model(baseline, idx, posterior, bicubic, prior_spread, terrain_raw, device, args.batch_size, False)
    ft_pred = run_model(finetuned, idx, posterior, bicubic, prior_spread, terrain_raw, device, args.batch_size, True)
    truth = np.asarray(wrf[idx], dtype=np.float64)  # (S, 2, H, W)

    # Mean must be unchanged (variance-head-only fine-tune).
    mean_diff = float(np.abs(ft_pred[:, :2] - base_pred[:, :2]).max())
    print(f"max|mean_finetuned - mean_baseline|: {mean_diff:.3e} (should be ~0; mean head was frozen)")
    assert mean_diff < 1e-3, f"MEAN CHANGED by {mean_diff:.3e} -- the fine-tune should only touch the variance head"

    err_u = base_pred[:, 0] - truth[:, 0]   # mean identical for both, so one error field
    err_v = base_pred[:, 1] - truth[:, 1]
    abs_err_u, abs_err_v = np.abs(err_u), np.abs(err_v)

    bs_u, bs_v = sigma_components(base_pred)
    ft_u, ft_v = sigma_components(ft_pred)

    z_base_u, z_base_v = err_u / np.maximum(bs_u, EPS), err_v / np.maximum(bs_v, EPS)
    z_ft_u, z_ft_v = err_u / np.maximum(ft_u, EPS), err_v / np.maximum(ft_v, EPS)

    # Blotch mask from baseline z magnitude, per-sample top (100 - blotch_pct)%.
    zmag_base = np.sqrt(z_base_u ** 2 + z_base_v ** 2)  # (S, H, W)
    S = zmag_base.shape[0]
    thresh = np.percentile(zmag_base.reshape(S, -1), args.blotch_pct, axis=1)  # (S,)
    blotch = zmag_base >= thresh[:, None, None]  # (S, H, W) bool

    print(f"\n=== Aggregate z-score calibration ({len(idx)} {args.split} samples, all pixels) ===")
    print("u component:")
    print(calibration_line("baseline ", z_base_u))
    print(calibration_line("finetuned", z_ft_u))
    print("v component:")
    print(calibration_line("baseline ", z_base_v))
    print(calibration_line("finetuned", z_ft_v))

    print("\n=== Pearson corr(sigma, |error|) -- higher = sigma tracks error better ===")
    for comp, ae, bsig, ftsig in [("u", abs_err_u, bs_u, ft_u), ("v", abs_err_v, bs_v, ft_v)]:
        all_b, all_f = pearson(bsig, ae), pearson(ftsig, ae)
        bl_b, bl_f = pearson(bsig[blotch], ae[blotch]), pearson(ftsig[blotch], ae[blotch])
        print(f"  [{comp}] all pixels:    baseline {all_b:+.4f} -> finetuned {all_f:+.4f}  (delta {all_f - all_b:+.4f})")
        print(f"  [{comp}] blotch pixels: baseline {bl_b:+.4f} -> finetuned {bl_f:+.4f}  (delta {bl_f - bl_b:+.4f})")

    print(f"\n=== Mean |z| on blotch pixels (top {100 - args.blotch_pct:g}% baseline |z|) -- lower = errors now covered ===")
    for comp, zb, zf in [("u", z_base_u, z_ft_u), ("v", z_base_v, z_ft_v)]:
        mb, mf = float(np.abs(zb[blotch]).mean()), float(np.abs(zf[blotch]).mean())
        print(f"  [{comp}] baseline {mb:.3f} -> finetuned {mf:.3f}  (delta {mf - mb:+.3f})")

    print(f"\nchol_gate: finetuned={finetuned.chol_gate.item():.4f} (baseline={baseline.chol_gate.item():.4f})")


if __name__ == "__main__":
    main()
