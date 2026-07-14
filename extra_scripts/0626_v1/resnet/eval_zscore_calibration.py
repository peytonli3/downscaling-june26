#!/usr/bin/env python3
"""
Pool per-pixel wind z-scores -- for speed, and for u/v individually -- over many samples
and compare their distributions to the standard normal N(0, 1) the model's covariance
head implicitly claims they should follow if it is well-calibrated.

The model directly predicts a per-pixel bivariate Gaussian over (u, v): mean mu,
covariance Sigma = L L^T from the predicted Cholesky factor [L11, L21, L22], so
Var(u) = L11^2 and Var(v) = L21^2 + L22^2 exactly -- z_u = (mu_u - truth_u) / sqrt(Var(u))
and z_v = (mu_v - truth_v) / sqrt(Var(v)) are exact standardizations of that Gaussian,
not approximations.

speed = sqrt(u^2+v^2) is not itself part of that Gaussian, though, so z_speed instead
uses eval_resnet_refiner_checkpoint.py's speed_zscore: a delta-method (first-order
Taylor) approximation,
    Var(speed) ~= g_u^2 * var(u) + g_v^2 * var(v) + 2 * g_u * g_v * cov(u, v),
    g_u = mu_u / speed, g_v = mu_v / speed   (gradient of sqrt(u^2+v^2) at the mean)
z_speed = (pred_speed - truth_speed) / sigma_speed. This degrades near-zero predicted
speed (the gradient is singular there), which can show up as heavier-than-normal tails
in z_speed even for an otherwise well-calibrated model -- see that script's docstring
for detail. z_u/z_v have no such caveat.

For each of --n_samples WRF/EnsCGP-posterior pairs run through a trained ResNetRefiner
checkpoint, z_speed/z_u/z_v are computed at every one of the 200x200 pixels and pooled
(no masking) into one array per component; each pooled array is binned into its own
histogram panel and plotted against the N(0, 1) PDF, alongside the fraction of pixels
within +/-1 and +/-2 sigma versus the 68.3%/95.4% a perfectly calibrated model would
have. This is the same calibration check this project runs against the Swin refiner
(see extra_scripts/graphing/swin/eval_zscore_calibration.py), so the two models'
calibration is directly comparable.

Sample selection mirrors eval_resnet_refiner_checkpoint.py: by default drawn from the
held-out test split (data/splits_70_15_15/split_indices.npz); --sample_indices bypasses
split filtering with explicit raw indices. The model runs in --batch_size chunks rather
than one giant batch, since --n_samples is meant to be large for a stable histogram.

--top_pct restricts all three histograms to only the windiest pixels, defined PER
SAMPLE: each chosen sample gets its own (100 - top_pct)-th percentile of wind
magnitude (over that sample's 200x200 pixels), and pixels below ITS sample's threshold
are dropped before binning. This deliberately differs from variance_recalibration.py's
stratification, which pools one threshold across its whole fit set -- that script needs
a single serializable threshold to re-apply later to individual new samples at
inference time, so it stays pooled; this script always has the full chosen-sample set
in hand, so "windiest part of each individual event" is the more natural reading of
"windiest pixels" here. --magnitude_source picks which speed defines "wind magnitude":
"truth" (default, the actual WRF event intensity) or "pred" (the model's own predicted
speed, matching variance_recalibration.py's choice there).

Usage:
    python eval_zscore_calibration.py --checkpoint /home/peytonli/26.6_wind/logs/resnet_refiner/checkpoints/best.pth
    python eval_zscore_calibration.py --checkpoint .../best.pth --n_samples 300 --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from resnet_refiner import DEFAULT_CONFIG_PATH, build_model, load_config  # noqa: E402
from terrain_encoder import load_terrain_input  # noqa: E402


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


def speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u ** 2 + v ** 2)


def speed_zscore(mu_u: np.ndarray, mu_v: np.ndarray, var_u: np.ndarray, var_v: np.ndarray, cov_uv: np.ndarray,
                  truth_speed: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(pred_speed - truth_speed) / sigma_speed; ported from
    eval_resnet_refiner_checkpoint.py -- see that module's docstring for the
    delta-method derivation of sigma_speed."""
    pred_speed = speed(mu_u, mu_v)
    s_safe = np.maximum(pred_speed, eps)
    g_u, g_v = mu_u / s_safe, mu_v / s_safe
    var_speed = g_u ** 2 * var_u + g_v ** 2 * var_v + 2 * g_u * g_v * cov_uv
    sigma_speed = np.sqrt(np.clip(var_speed, 0.0, None))
    return (pred_speed - truth_speed) / np.maximum(sigma_speed, eps)


def component_zscore(mu: np.ndarray, truth: np.ndarray, var: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(pred - truth) / sigma for u or v individually. Exact standardization of the
    model's predicted Gaussian for that component (unlike speed_zscore's delta-method
    approximation), since the model predicts mu and var for u/v directly."""
    sigma = np.sqrt(np.clip(var, 0.0, None))
    return (mu - truth) / np.maximum(sigma, eps)


COMPONENTS = ("speed", "u", "v")
MAGNITUDE_KEYS = ("magnitude_truth", "magnitude_pred")


def collect_zscores(idx: np.ndarray, posterior, wrf, model, terrain_raw, device, batch_size: int) -> dict:
    """Run the model over `idx` in batches; return {"speed"/"u"/"v": flat array} with
    every per-pixel z-score pooled across all chosen samples (n_samples * 200 * 200
    values each), plus "magnitude_truth"/"magnitude_pred" (truth/predicted wind speed,
    same per-pixel order) so callers can filter the z-score arrays by wind magnitude."""
    chunks = {c: [] for c in COMPONENTS + MAGNITUDE_KEYS}
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        posterior_batch = torch.from_numpy(np.array(posterior[batch_idx], dtype=np.float32, copy=True)).to(device)
        with torch.no_grad():
            pred_batch = model(
                posterior_batch, posterior_batch[:, :2], posterior_batch[:, 2:5], terrain_raw
            ).cpu().numpy()  # (b, 5, H, W)

        for row, i in enumerate(batch_idx):
            mu_u, mu_v, L11, L21, L22 = pred_batch[row]
            wrf_u, wrf_v = np.asarray(wrf[int(i), 0]), np.asarray(wrf[int(i), 1])
            truth_speed = speed(wrf_u, wrf_v)
            pred_speed = speed(mu_u, mu_v)

            var_u = L11 ** 2
            var_v = L21 ** 2 + L22 ** 2
            cov_uv = L11 * L21

            chunks["speed"].append(speed_zscore(mu_u, mu_v, var_u, var_v, cov_uv, truth_speed).ravel())
            chunks["u"].append(component_zscore(mu_u, wrf_u, var_u).ravel())
            chunks["v"].append(component_zscore(mu_v, wrf_v, var_v).ravel())
            chunks["magnitude_truth"].append(truth_speed.ravel())
            chunks["magnitude_pred"].append(pred_speed.ravel())

        print(f"Processed {min(start + batch_size, len(idx))}/{len(idx)} samples")

    return {c: np.concatenate(v) for c, v in chunks.items()}


def filter_by_magnitude(z: dict, top_pct: float, source: str, n_samples: int) -> tuple[dict, np.ndarray, int]:
    """Keep only each sample's OWN windiest top_pct percent of pixels (z["magnitude_truth"]
    or z["magnitude_pred"], per `source`) -- a threshold computed independently per sample,
    not pooled across samples first. Relies on collect_zscores appending one contiguous,
    equal-size (H*W-pixel) block per sample, in order, so the pooled magnitude array can be
    reshaped to (n_samples, H*W) and percentiled along axis=1. Returns (filtered z-score
    dict (speed/u/v only), per-sample thresholds (n_samples,), and the number of pixels kept)."""
    if not 0.0 < top_pct < 100.0:
        raise ValueError(f"--top_pct must be in (0, 100), got {top_pct}")
    magnitude = z[f"magnitude_{source}"]
    if magnitude.size % n_samples != 0:
        raise ValueError(f"magnitude array size {magnitude.size} not divisible by n_samples {n_samples}")
    per_sample = magnitude.reshape(n_samples, -1)
    thresholds = np.percentile(per_sample, 100.0 - top_pct, axis=1)
    mask = (per_sample >= thresholds[:, None]).ravel()
    return {c: z[c][mask] for c in COMPONENTS}, thresholds, int(mask.sum())


COMPONENT_XLABELS = {
    "speed": "Speed z-score: (pred - truth) / sigma",
    "u": "u z-score: (pred - truth) / sigma",
    "v": "v z-score: (pred - truth) / sigma",
}


def plot_zscore_histograms(z: dict, n_samples: int, n_bins: int, z_clip: float, output: Path,
                            title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(6 * len(COMPONENTS), 5.5), squeeze=False)
    axes = axes[0]

    bins = np.linspace(-z_clip, z_clip, n_bins + 1)
    x = np.linspace(-z_clip, z_clip, 400)
    normal_pdf = norm.pdf(x)

    for ax, comp in zip(axes, COMPONENTS):
        z_comp = z[comp]
        ax.hist(np.clip(z_comp, -z_clip, z_clip), bins=bins, density=True, color="C0", alpha=0.6,
                edgecolor="white", linewidth=0.3, label=f"Observed (n={z_comp.size:,})")
        ax.plot(x, normal_pdf, color="black", linewidth=2, label="N(0, 1)")

        ax.set_xlabel(COMPONENT_XLABELS[comp])
        ax.set_ylabel("Density")
        ax.set_title(comp)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.6)

        mean_z, std_z = float(z_comp.mean()), float(z_comp.std())
        frac_1 = float(np.mean(np.abs(z_comp) <= 1.0))
        frac_2 = float(np.mean(np.abs(z_comp) <= 2.0))
        stats_text = (
            f"mean={mean_z:.3f}, std={std_z:.3f}\n"
            f"|z|<=1: {frac_1:.1%} (N(0,1): 68.3%)\n"
            f"|z|<=2: {frac_2:.1%} (N(0,1): 95.4%)"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va="top", ha="left",
                fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        print(f"[{comp}] pooled over {z_comp.size:,} pixels from {n_samples} samples")
        print(f"[{comp}] mean={mean_z:.4f}, std={std_z:.4f} (well-calibrated: mean~0, std~1)")
        print(f"[{comp}] |z|<=1: {frac_1:.4f} (N(0,1) expects 0.6827); |z|<=2: {frac_2:.4f} (N(0,1) expects 0.9545)")

    fig.suptitle(f"z-score distributions vs standard normal (n_samples={n_samples}){title_suffix}")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved z-score histograms to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth from --config")
    parser.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --config")
    parser.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --config")
    parser.add_argument("--wrf_path", type=Path, default=None, help="Defaults to <data_dir>/wrf_uv.npy")
    parser.add_argument("--posterior_path", type=Path, default=None, help="Defaults to <data_dir>/enscgp_posterior.npy")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Sample pool for default random selection")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to pool pixels from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_indices", type=str, default=None,
        help="Comma-separated raw sample indices (overrides --split/--n_samples/--seed), e.g. 0,25,100",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Model forward-pass batch size")
    parser.add_argument("--n_bins", type=int, default=60)
    parser.add_argument("--z_clip", type=float, default=5.0, help="Histogram +/- x-range; z-scores beyond this are clipped into the edge bins")
    parser.add_argument("--top_pct", type=float, default=None,
                         help="If set, restrict histograms to the windiest top_pct%% of pixels (e.g. 5 for the top 5%%)")
    parser.add_argument("--magnitude_source", type=str, default="truth", choices=["truth", "pred"],
                         help="Which wind speed defines magnitude for --top_pct: WRF truth (default) or the model's predicted speed")
    parser.add_argument("--output", type=Path, default=None,
                         help="Defaults to inference_results/zscore_histograms_resnet.png, or "
                              ".../zscore_histograms_resnet_top<top_pct>pct_<magnitude_source>.png if --top_pct is set")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output is None:
        if args.top_pct is None:
            args.output = Path("/home/peytonli/26.6_wind/inference_results/zscore_histograms_resnet.png")
        else:
            args.output = Path(
                f"/home/peytonli/26.6_wind/inference_results/"
                f"zscore_histograms_resnet_top{args.top_pct:g}pct_{args.magnitude_source}.png"
            )

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = args.data_dir or Path(paths["data_dir"])
    checkpoint_path = args.checkpoint or (Path(paths["log_dir"]) / "checkpoints" / "best.pth")
    splits_path = args.splits_path or Path(paths["splits_path"])
    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    posterior_path = args.posterior_path or (data_dir / "enscgp_posterior.npy")

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
    n_total = min(wrf.shape[0], posterior.shape[0])

    splits = np.load(splits_path)
    pool = splits[f"{args.split}_idx"]
    idx = choose_indices(pool, n_total, args.n_samples, args.seed, args.sample_indices)

    z = collect_zscores(idx, posterior, wrf, model, terrain_raw, device, args.batch_size)

    title_suffix = ""
    if args.top_pct is not None:
        z, thresholds, n_kept = filter_by_magnitude(z, args.top_pct, args.magnitude_source, len(idx))
        print(f"Restricting to top {args.top_pct:g}% of pixels by {args.magnitude_source} wind magnitude, "
              f"per-sample (threshold range [{thresholds.min():.3f}, {thresholds.max():.3f}] m/s, "
              f"median {np.median(thresholds):.3f} m/s): kept {n_kept:,} pixels")
        title_suffix = (f"\ntop {args.top_pct:g}% wind magnitude pixels per-sample "
                         f"({args.magnitude_source}, median threshold {np.median(thresholds):.2f} m/s)")
    else:
        z = {c: z[c] for c in COMPONENTS}

    plot_zscore_histograms(z, len(idx), args.n_bins, args.z_clip, args.output, title_suffix=title_suffix)

    print(f"Checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')})")
    print(f"Split: {args.split}")
    print(f"Samples used ({len(idx)}): {idx.tolist()}")


if __name__ == "__main__":
    main()
