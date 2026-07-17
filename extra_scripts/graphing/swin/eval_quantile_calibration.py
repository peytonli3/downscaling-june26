#!/usr/bin/env python3
"""
Calibration diagnostics for a ProbabilisticSwin2SR QUANTILE checkpoint (0714+): a PIT /
rank histogram and per-pixel spatial coverage maps, aggregated over a whole data split.

The model predicts per-pixel MARGINAL quantiles q10/q50/q90 of u and of v separately, so
calibration is assessed per COMPONENT (each quantile is a genuine marginal quantile of its
own component -- no speed / dependence assumption is involved anywhere here).

Two figures are produced.

(1) PIT / rank histogram  [<log_dir>/figures/quantile_pit_histogram.png]
    The three quantiles partition the real line into four bins; a calibrated model places
    the truth in them at the nominal rates:
        truth < q10 : 0.10      q10 <= truth < q50 : 0.40
        q50 <= truth < q90 : 0.40   truth >= q90 : 0.10
    One panel per component, bars = observed bin frequencies, black steps = the nominal
    rates. Overall bars plus an "extreme tail" series (pixels whose wind magnitude is above
    that sample's --ext_quantile percentile) show whether calibration that holds in bulk
    breaks in the damaging tail. Shape reading: outer bins too tall (U) => intervals too
    narrow / overconfident; inner bins too tall (hump) => intervals too wide. This is the
    quantile-era replacement for the retired Gaussian z-score calibration histogram.

(2) Spatial coverage maps  [<log_dir>/figures/quantile_coverage_maps.png]
    Per-pixel empirical coverage minus nominal, i.e. mean over the split of
    (truth_component <= q_k) minus {.10,.50,.90}, on a symmetric diverging scale
    (blue = below nominal, red = above). Rows = component (u, v), cols = q10/q50/q90.
    This localizes WHERE the band is miscalibrated -- e.g. whether q90 under-coverage
    concentrates on ridgelines / coastal peaks.

By default the ENTIRE split is used (this is an aggregate statistic, not a few examples);
--max_samples caps it for a quick look, --sample_indices overrides the split entirely.
Prefer --device cuda: a full test split is ~1k forward passes.

Display convention: north-up as stored (post-0701 alignment fix); origin="upper".

Requires a 0714+ quantile checkpoint (6-channel output); a pre-0714 Gaussian/Cholesky
checkpoint fails the channel-count guard.

Usage:
    python eval_quantile_calibration.py --checkpoint /home/peytonli/26.6_wind/runs/0714/checkpoints/best.pth
    python eval_quantile_calibration.py --checkpoint .../best.pth --split test --device cuda
    python eval_quantile_calibration.py --checkpoint .../best.pth --max_samples 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

SCRIPTS_DIR = "/net/flood/home/peytonli/26.6_wind/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from new_enscgp_swin import DEFAULT_CONFIG_PATH, ProbabilisticSwin2SR, build_model, load_config  # noqa: E402
from terrain_encoder import load_terrain_input  # noqa: E402

COMPONENTS = ("u", "v")
QUANTILES = (("q10", 0.10, ProbabilisticSwin2SR.Q10_SLICE),
             ("q50", 0.50, ProbabilisticSwin2SR.Q50_SLICE),
             ("q90", 0.90, ProbabilisticSwin2SR.Q90_SLICE))
# Nominal PIT bin probabilities for [ <q10, q10-q50, q50-q90, >=q90 ].
PIT_BIN_EDGES_LABELS = ("<q10", "q10–q50", "q50–q90", "≥q90")
PIT_NOMINAL = np.array([0.10, 0.40, 0.40, 0.10])


def choose_indices(pool: np.ndarray, n_total: int, max_samples: int | None, seed: int,
                   sample_indices: str | None) -> np.ndarray:
    if sample_indices:
        idx = np.array([int(x.strip()) for x in sample_indices.split(",") if x.strip() != ""], dtype=int)
        if np.any(idx < 0) or np.any(idx >= n_total):
            raise ValueError(f"sample_indices must be in [0, {n_total - 1}]")
        return np.unique(idx)
    idx = np.sort(np.asarray(pool, dtype=int))
    if max_samples is not None and max_samples < len(idx):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    return idx


class CalibrationAccumulator:
    """Streaming accumulation of coverage sums and PIT bin counts over batches, so the whole
    split never needs to be held in memory at once."""

    def __init__(self, H: int, W: int):
        self.H, self.W = H, W
        self.n_samples = 0
        # coverage sums: [quantile, component, H, W]
        self.cov_sum = np.zeros((len(QUANTILES), len(COMPONENTS), H, W), dtype=np.float64)
        # PIT bin counts: [component, 4], overall and extreme-tail-only
        self.bin_counts = np.zeros((len(COMPONENTS), 4), dtype=np.float64)
        self.bin_counts_ext = np.zeros((len(COMPONENTS), 4), dtype=np.float64)

    def update(self, pred: np.ndarray, wrf: np.ndarray, ext_quantile: float):
        # pred (B,6,H,W); wrf (B,2,H,W). Extreme mask is per-pixel (shared by u,v), from
        # wind magnitude vs this sample's ext_quantile percentile.
        B = pred.shape[0]
        self.n_samples += B
        mag = np.sqrt(wrf[:, 0] ** 2 + wrf[:, 1] ** 2)                       # (B,H,W)
        thr = np.quantile(mag.reshape(B, -1), ext_quantile, axis=1)         # (B,)
        ext = mag >= thr[:, None, None]                                     # (B,H,W)

        q = {name: pred[:, sl] for name, _nom, sl in QUANTILES}            # each (B,2,H,W)
        for qi, (name, _nom, _sl) in enumerate(QUANTILES):
            for ci in range(len(COMPONENTS)):
                self.cov_sum[qi, ci] += (wrf[:, ci] <= q[name][:, ci]).sum(axis=0)

        for ci in range(len(COMPONENTS)):
            t = wrf[:, ci]
            q10, q50, q90 = q["q10"][:, ci], q["q50"][:, ci], q["q90"][:, ci]
            bins = [t < q10, (t >= q10) & (t < q50), (t >= q50) & (t < q90), t >= q90]
            for bi, mask in enumerate(bins):
                self.bin_counts[ci, bi] += mask.sum()
                self.bin_counts_ext[ci, bi] += (mask & ext).sum()

    def coverage_maps(self) -> np.ndarray:
        return self.cov_sum / max(self.n_samples, 1)  # [quantile, component, H, W]

    def pit_fractions(self):
        overall = self.bin_counts / self.bin_counts.sum(axis=1, keepdims=True).clip(min=1)
        ext = self.bin_counts_ext / self.bin_counts_ext.sum(axis=1, keepdims=True).clip(min=1)
        return overall, ext


def plot_pit(pit_overall: np.ndarray, pit_ext: np.ndarray, ext_quantile: float, output: Path) -> None:
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(6.0 * len(COMPONENTS), 4.5), squeeze=False)
    x = np.arange(4)
    w = 0.38
    for ci, comp in enumerate(COMPONENTS):
        ax = axes[0, ci]
        ax.bar(x - w / 2, pit_overall[ci], width=w, color="C0", label="overall")
        ax.bar(x + w / 2, pit_ext[ci], width=w, color="C3", alpha=0.85,
               label=f"extreme tail (|wind| > p{int(ext_quantile * 100)})")
        # Nominal rates as a step reference.
        ax.step(np.concatenate([x - 0.5, [x[-1] + 0.5]]), np.concatenate([PIT_NOMINAL, [PIT_NOMINAL[-1]]]),
                where="post", color="black", linewidth=1.5, label="nominal")
        ax.set_xticks(x)
        ax.set_xticklabels(PIT_BIN_EDGES_LABELS)
        ax.set_ylabel("Fraction of truth pixels")
        ax.set_title(f"{comp} component")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("PIT / rank histogram — where truth falls relative to q10/q50/q90\n"
                 "(outer bins too tall = overconfident; inner bins too tall = too wide)", fontsize=12)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PIT histogram: {output}")


def plot_coverage_maps(cov_maps: np.ndarray, lsm: np.ndarray | None, spread: float, output: Path) -> None:
    n_rows, n_cols = len(COMPONENTS), len(QUANTILES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), squeeze=False)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-spread, vmax=spread)
    for ci, comp in enumerate(COMPONENTS):
        for qi, (name, nominal, _sl) in enumerate(QUANTILES):
            ax = axes[ci, qi]
            dev = cov_maps[qi, ci] - nominal  # empirical coverage minus nominal
            im = ax.imshow(dev, cmap="RdBu_r", norm=norm, origin="upper")
            if lsm is not None:
                ax.contour(lsm, levels=[0.5], colors="black", linewidths=0.7)
            ax.set_title(f"{comp}: {name} coverage − {nominal:.2f}\n"
                         f"(mean {cov_maps[qi, ci].mean():.3f})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.8)
    fig.suptitle("Per-pixel coverage minus nominal (blue = under-covers, red = over-covers)", fontsize=13)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved coverage maps: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--splits_path", type=Path, default=None)
    parser.add_argument("--wrf_path", type=Path, default=None)
    parser.add_argument("--posterior_path", type=Path, default=None)
    parser.add_argument("--bicubic_path", type=Path, default=None)
    parser.add_argument("--hires_land_mask_path", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Cap the split size (random subset) for a quick look")
    parser.add_argument("--sample_indices", type=str, default=None, help="Raw indices; overrides --split/--max_samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ext_quantile", type=float, default=0.9, help="Wind-magnitude percentile defining the 'extreme tail'")
    parser.add_argument("--cov_spread", type=float, default=0.15, help="+/- range of the coverage-deviation color scale")
    parser.add_argument("--pit_output", type=Path, default=None, help="Defaults to <log_dir>/figures/quantile_pit_histogram.png")
    parser.add_argument("--cov_output", type=Path, default=None, help="Defaults to <log_dir>/figures/quantile_coverage_maps.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    log_dir = Path(paths["log_dir"])
    data_dir = args.data_dir or Path(paths["data_dir"])
    checkpoint_path = args.checkpoint or (log_dir / "checkpoints" / "best.pth")
    splits_path = args.splits_path or Path(paths["splits_path"])
    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    posterior_path = args.posterior_path or (data_dir / "enscgp_posterior.npy")
    bicubic_path = args.bicubic_path or (data_dir / "era5_uv_2ch_bicubic.npy")
    hires_land_mask_path = args.hires_land_mask_path or (data_dir / "land_mask_hires.npz")
    pit_output = args.pit_output or (log_dir / "figures" / "quantile_pit_histogram.png")
    cov_output = args.cov_output or (log_dir / "figures" / "quantile_coverage_maps.png")

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
    idx = choose_indices(pool, n_total, args.max_samples, args.seed, args.sample_indices)

    H, W = wrf.shape[2], wrf.shape[3]
    acc = CalibrationAccumulator(H, W)
    print(f"Evaluating {len(idx)} samples from split '{args.split}' on {device} "
          f"(batch_size={args.batch_size})...")
    for start in range(0, len(idx), args.batch_size):
        batch_idx = idx[start:start + args.batch_size]
        post_b = torch.from_numpy(np.array(posterior[batch_idx], dtype=np.float32, copy=True)).to(device)
        bic_b = torch.from_numpy(np.array(bicubic[batch_idx], dtype=np.float32, copy=True)).to(device)
        with torch.no_grad():
            pred = model(post_b, bic_b, terrain_raw).cpu().numpy()
        if pred.shape[1] != ProbabilisticSwin2SR.OUT_CHANNELS:
            raise ValueError(
                f"Expected a {ProbabilisticSwin2SR.OUT_CHANNELS}-channel quantile output, got "
                f"{pred.shape[1]}. Is {checkpoint_path} a pre-0714 Gaussian/Cholesky checkpoint?"
            )
        wrf_b = np.array(wrf[batch_idx], dtype=np.float32, copy=True)
        acc.update(pred, wrf_b, args.ext_quantile)

    cov_maps = acc.coverage_maps()
    pit_overall, pit_ext = acc.pit_fractions()

    hires_masks = np.load(hires_land_mask_path)
    lsm_wrf = hires_masks["wrf"].astype(np.float64)

    plot_pit(pit_overall, pit_ext, args.ext_quantile, pit_output)
    plot_coverage_maps(cov_maps, lsm_wrf, args.cov_spread, cov_output)

    # ── printed summary ──
    print(f"\nCheckpoint: {checkpoint_path} (epoch {ckpt.get('epoch')}, best_val_loss {ckpt.get('best_val_loss')})")
    print(f"Split: {args.split} | samples: {acc.n_samples}")
    print("\nMean coverage (nominal in parentheses):")
    for qi, (name, nominal, _sl) in enumerate(QUANTILES):
        per_comp = " ".join(f"{comp}={cov_maps[qi, ci].mean():.3f}" for ci, comp in enumerate(COMPONENTS))
        print(f"  {name} ({nominal:.2f}): {per_comp}")
    print("\nPIT bin fractions [ <q10, q10–q50, q50–q90, ≥q90 ] (nominal [.10 .40 .40 .10]):")
    for ci, comp in enumerate(COMPONENTS):
        ov = " ".join(f"{v:.3f}" for v in pit_overall[ci])
        ex = " ".join(f"{v:.3f}" for v in pit_ext[ci])
        print(f"  {comp}: overall [{ov}] | extreme [{ex}]")


if __name__ == "__main__":
    main()
