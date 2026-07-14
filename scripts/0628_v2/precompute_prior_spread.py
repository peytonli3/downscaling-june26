"""Precompute the EnsCGP PRIOR (pre-conditioning) ensemble spread -- the analog-
disagreement signal that conditions the variance head in new_enscgp_swin.py (when
model.variance_conditioning is on).

For every sample i, the EnsCGP prior is the ensemble of that sample's k analog WRF
fields (data/neighbor_train_only.npy gives the k=36 train-only analog indices per
sample; their WRF fields come from data/wrf_uv.npy). enscgp_train.py:prior() forms the
anomaly matrix A = anomalies.T / sqrt(k-1) so diag(A A.T) is the per-pixel ensemble
VARIANCE (ddof=1). This script writes its square root -- the per-pixel ensemble STD,
[u, v] -- which is the placement-uncertainty feature: where the analogs disagree on
the placement of mesoscale structure, the spread is large.

Outputs (under <data_dir>):
- enscgp_prior_spread.npy : (N, 2, 200, 200) float32, per-pixel analog std [u, v] for
  every sample (written incrementally via open_memmap; ~2.2 GB, same footprint as
  wrf_uv.npy). This is the `prior_spread` input to ProbabilisticSwin2SR.forward().
- cond_feature_stats.npz  : cond_mean/cond_std (each length 4) -- per-channel
  standardization for the four variance-conditioning features, in the model's channel
  order [prior_spread_u, prior_spread_v, |grad speed(mean)|, |terrain slope|], computed
  over the TRAINING split only (so val/test are never used to set the scale). Load into
  the model with ProbabilisticSwin2SR.load_cond_stats(**np.load(...)).
  * prior_spread_u/v stats come from the array above (train rows).
  * |grad speed(mean)| stats use the EnsCGP posterior mean (enscgp_posterior.npy[:, :2])
    as a stand-in for the model mean (the mean head starts at, and stays near, the
    EnsCGP mean with residual_base="enscgp") -- same central-difference scheme as the
    model's _speed_grad_mag.
  * |terrain slope| stats come from the single static terrain map (terrain_raw[2:4],
    avg-pooled 5x), matching the model's _terrain_grad_mag.

Leakage note: neighbors are train-only, so building each sample's prior spread from its
analogs' WRF fields is safe for val/test samples too (it never touches the target's own
field) -- exactly as enscgp_train.py already does to build the posterior.

Usage:
    python precompute_prior_spread.py [--config new_enscgp_swin_config.json]
    python precompute_prior_spread.py --stats_max_samples 2000 --force
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from new_enscgp_swin import DEFAULT_CONFIG_PATH, load_config

K_EXPECTED = 36  # EnsCGP analog count (data/neighbor_train_only.npy), see enscgp_train.py
GRAD_EPS = 1e-6  # matches ProbabilisticSwin2SR.GRAD_EPS


def speed_grad_mag(mean_uv: np.ndarray) -> np.ndarray:
    """|grad(speed)| of the mean wind, (B, H, W). Central differences with edge (replicate)
    padding -- numpy mirror of ProbabilisticSwin2SR._speed_grad_mag."""
    sp = np.sqrt(mean_uv[:, 0] ** 2 + mean_uv[:, 1] ** 2 + GRAD_EPS)  # (B, H, W)
    sp_x = np.pad(sp, ((0, 0), (0, 0), (1, 1)), mode="edge")
    gx = 0.5 * (sp_x[:, :, 2:] - sp_x[:, :, :-2])
    sp_y = np.pad(sp, ((0, 0), (1, 1), (0, 0)), mode="edge")
    gy = 0.5 * (sp_y[:, 2:, :] - sp_y[:, :-2, :])
    return np.sqrt(gx ** 2 + gy ** 2 + GRAD_EPS)


def terrain_grad_mag(data_dir: Path) -> np.ndarray:
    """|terrain slope| on the 200x200 WRF grid -- numpy mirror of
    ProbabilisticSwin2SR._terrain_grad_mag (uses terrain_encoder.load_terrain_input's
    channels [2,3] = u_slope_z/v_slope_z at 1000x1000, magnitude, 5x avg-pool)."""
    from terrain_encoder import load_terrain_input
    terrain = load_terrain_input(data_dir).numpy()  # (4, 1000, 1000)
    mag = np.sqrt(terrain[2] ** 2 + terrain[3] ** 2 + GRAD_EPS)  # (1000, 1000)
    H = mag.shape[0] // 5
    # 5x5 non-overlapping average pool -> (200, 200)
    return mag[: H * 5, : H * 5].reshape(H, 5, H, 5).mean(axis=(1, 3))


def channel_mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data_dir", type=Path, default=None, help="Defaults to paths.data_dir from --config")
    parser.add_argument("--splits_path", type=Path, default=None, help="Defaults to paths.splits_path from --config")
    parser.add_argument("--neighbors_path", type=Path, default=None,
                        help="Defaults to <data_dir>/neighbor_train_only.npy (the k=36 analog indices)")
    parser.add_argument("--wrf_path", type=Path, default=None, help="Defaults to <data_dir>/wrf_uv.npy")
    parser.add_argument("--posterior_path", type=Path, default=None,
                        help="Defaults to <data_dir>/enscgp_posterior.npy (for the mean-gradient stats)")
    parser.add_argument("--stats_max_samples", type=int, default=2000,
                        help="Subsample this many training samples when computing cond_feature_stats (the full "
                             "prior_spread array is always written for ALL samples regardless)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recompute enscgp_prior_spread.npy even if it exists")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = args.data_dir or Path(config["paths"]["data_dir"])
    splits_path = args.splits_path or Path(config["paths"]["splits_path"])
    neighbors_path = args.neighbors_path or (data_dir / "neighbor_train_only.npy")
    wrf_path = args.wrf_path or (data_dir / "wrf_uv.npy")
    posterior_path = args.posterior_path or (data_dir / "enscgp_posterior.npy")
    spread_path = data_dir / "enscgp_prior_spread.npy"
    stats_path = data_dir / "cond_feature_stats.npz"

    neighbors = np.load(neighbors_path)  # (N, k) int
    wrf = np.load(wrf_path, mmap_mode="r")  # (N, 2, 200, 200)
    n_total, _, H, W = wrf.shape
    k = neighbors.shape[1]
    if neighbors.shape[0] != n_total:
        raise ValueError(f"neighbors N={neighbors.shape[0]} != wrf N={n_total}")
    if k != K_EXPECTED:
        print(f"WARNING: k={k} analogs, expected {K_EXPECTED} -- continuing with k={k}")
    print(f"N={n_total}, k={k}, grid {H}x{W}")

    # ---- prior spread (N, 2, H, W): per-pixel analog std, written incrementally ----
    if spread_path.exists() and not args.force:
        print(f"{spread_path} exists; reusing it (pass --force to recompute)")
        spread = np.load(spread_path, mmap_mode="r")
    else:
        print(f"Computing prior spread -> {spread_path}")
        spread = np.lib.format.open_memmap(spread_path, mode="w+", dtype=np.float32, shape=(n_total, 2, H, W))
        for i in range(n_total):
            nbr = neighbors[i]  # (k,)
            fields = np.asarray(wrf[nbr], dtype=np.float64)  # (k, 2, H, W)
            spread[i] = fields.std(axis=0, ddof=1).astype(np.float32)  # ddof=1 == diag(A A.T) in prior()
            if (i + 1) % 500 == 0 or i + 1 == n_total:
                print(f"  prior spread {i + 1}/{n_total}")
        spread.flush()
        spread = np.load(spread_path, mmap_mode="r")

    # ---- cond_feature_stats over a training-split subsample ----
    splits = np.load(splits_path)
    train_idx = splits["train_idx"]
    rng = np.random.default_rng(args.seed)
    if args.stats_max_samples and len(train_idx) > args.stats_max_samples:
        stat_idx = np.sort(rng.choice(train_idx, size=args.stats_max_samples, replace=False))
    else:
        stat_idx = np.sort(train_idx)
    print(f"Computing cond_feature_stats over {len(stat_idx)} training samples")

    spread_stat = np.asarray(spread[stat_idx], dtype=np.float64)  # (S, 2, H, W)
    ps_u_mean, ps_u_std = channel_mean_std(spread_stat[:, 0])
    ps_v_mean, ps_v_std = channel_mean_std(spread_stat[:, 1])

    posterior = np.load(posterior_path, mmap_mode="r")  # (N, 5, H, W)
    mean_uv = np.asarray(posterior[stat_idx, :2], dtype=np.float64)  # (S, 2, H, W)
    grad = speed_grad_mag(mean_uv)  # (S, H, W)
    grad_mean, grad_std = channel_mean_std(grad)

    tgm = terrain_grad_mag(data_dir)  # (H, W)
    terr_mean, terr_std = channel_mean_std(tgm)

    cond_mean = np.array([ps_u_mean, ps_v_mean, grad_mean, terr_mean], dtype=np.float64)
    cond_std = np.array([ps_u_std, ps_v_std, grad_std, terr_std], dtype=np.float64)
    np.savez(stats_path, cond_mean=cond_mean, cond_std=cond_std)

    print(f"Saved {spread_path} ({spread.shape}) and {stats_path}")
    print("cond_feature_stats (channel order [prior_spread_u, prior_spread_v, |grad speed(mean)|, |terrain slope|]):")
    for name, m, s in zip(
        ["prior_spread_u", "prior_spread_v", "mean_speed_grad", "terrain_slope"], cond_mean, cond_std
    ):
        print(f"  {name:16s} mean={m:.4f} std={s:.4f}")


if __name__ == "__main__":
    main()
