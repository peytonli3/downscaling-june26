#!/usr/bin/env python3
"""
Quantify how much pinball training actually changed the offset head, and how asymmetric
the learned up/down offsets are in practice.

Motivation: the aggregate Pin loss barely moves across a training run's logs, which could
mean either (a) the offset head isn't learning much beyond its bias-init, or (b) it
converged to a good, genuinely-learned state early and the aggregate number just doesn't
show it. Comparing against a scratch model separates these.

Two questions this answers:

1. Is pinball training doing real work, or did the offsets stay near their init?
   Compares the TRAINED model's up/down offsets against a FRESHLY INITIALIZED model built
   from the SAME config/seed (so it reproduces the exact starting point that run began
   from), run on the SAME real inputs. A fresh model's offsets are close to spatially
   UNIFORM by construction -- softplus(raw) with raw's bias fixed at
   inverse_softplus(offset_spread_init) and only small Kaiming-noise variation around it.
   A fresh model CANNOT have learned per-pixel structure. So:
     - if trained ~= fresh in both mean AND spatial variance -> pinball barely moved anything
     - if the mean shifted and/or spatial variance grew substantially -> real learning
       happened, even if the aggregate loss curve looks flat
   Coverage (empirical exceedance on real truth) is reported for both too, since a
   surprisingly-good fresh-model coverage would help explain why pinball has little left
   to fix.

2. How asymmetric are up and down in practice (trained model only)?
   Per-pixel/per-sample statistics of up vs down, the fraction of pixels where up > down,
   and correlation of the asymmetry (up - down) with wind magnitude and with the signed
   component -- does the learned band actually skew the way the original "wind is
   right-skewed" motivation for switching to quantiles predicted it would, or is the
   asymmetry small/noise-level?

A gradient-still-alive check is included too: one real forward+backward pass on the
TRAINED model, reporting offset_head's gradient norm -- near-zero would mean training has
genuinely saturated (nothing left to learn); a healthy gradient with a flat loss curve
would point at something else (e.g. LR) limiting progress instead.

Usage:
    python eval_pinball_impact.py --checkpoint runs/0729_meangate/checkpoints/best.pth
    python eval_pinball_impact.py --checkpoint .../best.pth --n_samples 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


from _common import (  # noqa: E402  (shared harness; also puts scripts/ on sys.path)
    ProbabilisticSwin2SR, add_eval_args, build_model, choose_indices, setup, split_pool,
)
from train_new_enscgp_swin import MeanAuxLosses, compute_weighted_loss  # noqa: E402


def load_batch(ev, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(posterior, bicubic, wrf) for `idx`, as one resident batch on ev.device.

    Unlike its sibling diagnostics this script does not use `_common.predict()`: it runs TWO
    models (trained and fresh) over the same inputs, so what it needs is the inputs.
    """
    return tuple(
        torch.from_numpy(np.array(arr[idx], dtype=np.float32, copy=True)).to(ev.device)
        for arr in (ev.arrays.posterior, ev.arrays.bicubic, ev.arrays.wrf)
    )


@torch.no_grad()
def run_model(model, posterior, bicubic, terrain_raw, batch_size=8):
    """Chunked inference (the full sample set may not fit in one forward pass)."""
    outs = []
    for start in range(0, posterior.shape[0], batch_size):
        out = model(posterior[start:start + batch_size], bicubic[start:start + batch_size], terrain_raw)
        outs.append(out.cpu())
    return torch.cat(outs, dim=0)


def offsets(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q10 = pred[:, ProbabilisticSwin2SR.Q10_SLICE]
    q50 = pred[:, ProbabilisticSwin2SR.Q50_SLICE]
    q90 = pred[:, ProbabilisticSwin2SR.Q90_SLICE]
    return (q90 - q50), (q50 - q10)  # up, down


def coverage(pred: torch.Tensor, truth: torch.Tensor) -> dict:
    q10 = pred[:, ProbabilisticSwin2SR.Q10_SLICE]
    q90 = pred[:, ProbabilisticSwin2SR.Q90_SLICE]
    return {
        "cov_q10": (truth <= q10).float().mean().item(),
        "cov_q90": (truth <= q90).float().mean().item(),
    }


def report_offset_stats(name: str, up: torch.Tensor, down: torch.Tensor):
    # spatial std: std across (H,W) within each sample, then averaged over samples/components
    spatial_std_up = up.std(dim=(2, 3)).mean().item()
    spatial_std_down = down.std(dim=(2, 3)).mean().item()
    print(f"  [{name}] up:   mean={up.mean().item():.4f}  std={up.std().item():.4f}  "
          f"spatial_std={spatial_std_up:.4f}")
    print(f"  [{name}] down: mean={down.mean().item():.4f}  std={down.std().item():.4f}  "
          f"spatial_std={spatial_std_down:.4f}")


def main() -> None:
    parser = add_eval_args(
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
        n_samples_default=256, arrays=False,
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None, help="Asymmetry figure. Defaults to <log_dir>/figures/pinball_impact.png")
    args = parser.parse_args()

    # setup() builds the TRAINED model (and resolves the config/paths/terrain/arrays).
    ev = setup(args)
    config, device, terrain_raw = ev.config, ev.device, ev.terrain_raw
    trained = ev.model
    output = args.output or ev.figure_path("pinball_impact.png")

    print(f"Checkpoint: {ev.checkpoint}")
    print(f"Split: {args.split}, n_samples: {args.n_samples}\n")
    print(f"Trained checkpoint: epoch {ev.ckpt.get('epoch')}, best_val_loss {ev.ckpt.get('best_val_loss')}")

    # ── build a FRESH model reproducing this run's actual starting point (same seed) ──
    torch.manual_seed(config.get("seed", 0))
    fresh = build_model(config).to(device)
    fresh.eval()

    n_params_offset = sum(p.numel() for p in trained.offset_head.parameters())
    print(f"offset_head parameters: {n_params_offset:,}\n")

    # ── real data ──
    idx = choose_indices(split_pool(ev.splits_path, args.split), ev.arrays.n_total,
                         args.n_samples, args.seed, args.sample_indices)
    posterior, bicubic, wrf = load_batch(ev, idx)

    pred_trained = run_model(trained, posterior, bicubic, terrain_raw, args.batch_size)
    pred_fresh = run_model(fresh, posterior, bicubic, terrain_raw, args.batch_size)
    wrf_cpu = wrf.cpu()

    up_t, down_t = offsets(pred_trained)
    up_f, down_f = offsets(pred_fresh)

    # ── Q1: has training actually moved the offset head? ──
    print("=" * 70)
    print("Q1: Is pinball training doing real work, or near its bias-init still?")
    print("=" * 70)
    report_offset_stats("fresh  ", up_f, down_f)
    report_offset_stats("trained", up_t, down_t)
    mean_abs_move_up = (up_t - up_f).abs().mean().item()
    mean_abs_move_down = (down_t - down_f).abs().mean().item()
    print(f"\n  Mean |trained - fresh|: up={mean_abs_move_up:.4f} m/s, down={mean_abs_move_down:.4f} m/s")
    spatial_std_ratio_up = up_t.std(dim=(2, 3)).mean().item() / max(up_f.std(dim=(2, 3)).mean().item(), 1e-6)
    spatial_std_ratio_down = down_t.std(dim=(2, 3)).mean().item() / max(down_f.std(dim=(2, 3)).mean().item(), 1e-6)
    print(f"  Spatial-structure ratio (trained/fresh spatial std): up={spatial_std_ratio_up:.2f}x, "
          f"down={spatial_std_ratio_down:.2f}x")
    print("  (ratio ~1x = no more per-pixel structure than a fresh, untrained model has; "
          "notably >1x = genuine learned spatial structure)")

    cov_fresh = coverage(pred_fresh, wrf_cpu)
    cov_trained = coverage(pred_trained, wrf_cpu)
    print(f"\n  Coverage q10/q90 -- fresh:   {cov_fresh['cov_q10']:.3f} / {cov_fresh['cov_q90']:.3f} (nominal .10/.90)")
    print(f"  Coverage q10/q90 -- trained: {cov_trained['cov_q10']:.3f} / {cov_trained['cov_q90']:.3f}")

    # ── gradient-still-alive check on the trained model ──
    trained.zero_grad()
    mean_aux = MeanAuxLosses(spectral_low_freq_cutoff=config["training"].get("spectral_low_freq_cutoff", 0.28))
    weights = {"ms": 0.0, "freq": 0.0, "l1": 0.0, "spectral": 0.0, "gradient": 0.0, "pin": 1.0}
    extreme_cfg = config["training"].get("extreme_weight")
    b = min(args.batch_size, posterior.shape[0])
    pred_grad = trained(posterior[:b], bicubic[:b], terrain_raw)
    loss_dict = compute_weighted_loss(pred_grad, wrf[:b], weights, mean_aux, extreme_cfg=extreme_cfg)
    loss_dict["total"].backward()
    offset_grad_norm = torch.cat([p.grad.flatten() for p in trained.offset_head.parameters() if p.grad is not None]).norm().item()
    print(f"\n  offset_head gradient norm (pinball only, real batch): {offset_grad_norm:.4e}")
    print("  (near-zero would mean training has saturated; a healthy norm despite a flat loss")
    print("   curve would point at something else -- e.g. LR -- limiting further progress)")

    # ── Q2: how asymmetric is the trained model's band? ──
    print("\n" + "=" * 70)
    print("Q2: How asymmetric are up/down in practice (trained model)?")
    print("=" * 70)
    diff = (up_t - down_t)  # positive = up > down (right-skewed band)
    frac_up_bigger = (diff > 0).float().mean().item()
    print(f"  mean(up - down)   = {diff.mean().item():+.4f} m/s   median = {diff.median().item():+.4f} m/s")
    print(f"  fraction of pixels with up > down: {frac_up_bigger:.3f}  (0.5 = symmetric)")
    for c, name in enumerate(("u", "v")):
        d = diff[:, c]
        print(f"  {name}: mean(up-down)={d.mean().item():+.4f}, frac(up>down)={((d>0).float().mean().item()):.3f}")

    mag = torch.sqrt(wrf_cpu[:, 0:1] ** 2 + wrf_cpu[:, 1:2] ** 2 + 1e-6).expand(-1, 2, -1, -1)
    diff_np, mag_np = diff.flatten().numpy(), mag.flatten().numpy()
    corr = np.corrcoef(diff_np, mag_np)[0, 1]
    print(f"\n  corr(up-down, |wind|): {corr:+.4f}  (positive = band skews further up in high-wind regions)")
    signed_corr_u = np.corrcoef(diff[:, 0].flatten().numpy(), wrf_cpu[:, 0].flatten().numpy())[0, 1]
    signed_corr_v = np.corrcoef(diff[:, 1].flatten().numpy(), wrf_cpu[:, 1].flatten().numpy())[0, 1]
    print(f"  corr(up-down, signed u): {signed_corr_u:+.4f}   corr(up-down, signed v): {signed_corr_v:+.4f}")

    # ── figure: asymmetry histogram + binned asymmetry vs magnitude ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(diff_np, bins=80, color="C0", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("up - down (m/s)")
    axes[0].set_ylabel("pixel count")
    axes[0].set_title(f"Asymmetry distribution (frac up>down = {frac_up_bigger:.2f})")

    n_bins = 20
    bin_edges = np.quantile(mag_np, np.linspace(0, 1, n_bins + 1))
    bin_centers, bin_means = [], []
    for i in range(n_bins):
        m = (mag_np >= bin_edges[i]) & (mag_np <= bin_edges[i + 1])
        if m.sum() > 0:
            bin_centers.append(mag_np[m].mean())
            bin_means.append(diff_np[m].mean())
    axes[1].plot(bin_centers, bin_means, marker="o", color="C3")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("|wind| (m/s)")
    axes[1].set_ylabel("mean(up - down) (m/s)")
    axes[1].set_title(f"Asymmetry vs wind magnitude (corr={corr:+.2f})")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Pinball impact / asymmetry -- {ev.checkpoint.name} (epoch {ev.ckpt.get('epoch')})")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {output}")


if __name__ == "__main__":
    main()
