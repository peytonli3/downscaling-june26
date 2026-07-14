"""
EnsCGP layer (https://arxiv.org/pdf/2602.13871) for providing a background for SWIN predictions.

Input:
- Index of sample to be conditioned
- data/wrf_uv.npy: 2-channel (u/v) wind data from WRF, shape (N, 2, 200, 200) -- supplies the
  prior ensemble (via each sample's neighbors)
- data/neighbor_train_only.npy: indices of the 36 closest neighbors to each sample (different
  event, from train only), shape (N, 36)
- data/era5_uv_2ch_native34.npy: 2-channel (u/v) ERA5 observation, shape (N, 2, 34, 34), used as y
- data/coarsening_operator_H.npy: sparse HR->LR observation operator H (1156, 40000) plus a
  validity mask for LR cells not fully covered by the HR domain

Output:
- 5x200x200 array representing the posterior of the original sample given its neighbors and the
  ERA5 observation at that index: channels are [u, v, L11, L21, L22] -- the posterior mean (u, v)
  and the lower-Cholesky factor of the per-pixel 2x2 (u, v) posterior covariance.

sigma_mean vs sigma_spread: the ERA5 observation-noise sigma is decoupled between the posterior
MEAN and the posterior SPREAD, each conditioned in its own enscgp() pass over the same prior/
observation (see tune_enscgp_sigma.py / sigma_mae_sweep.py / sigma_decouple_check.py for how
these were chosen). A single sigma can't do both well: sigma~2.77 (the literal empirical
ERA5-vs-WRF representativeness error) gives the best posterior MEAN accuracy (MAE) but a
drastically overconfident posterior SPREAD (bulk+extreme SSR ~0.13); sigma~92 fixes the SPREAD's
calibration (bulk SSR ~1.0, extreme SSR ~0.70, matching the unconditioned k-neighbor-ensemble
baseline) but costs ~5% MAE relative to the small-sigma optimum. Spread-error rank correlation
stays decent (~0.40) even at small sigma, so the small-sigma SPREAD is still informative, just
miscalibrated in absolute scale -- hence: small sigma_mean for an accurate mean, larger
sigma_spread for a calibrated spread, from two passes over the same ensemble/observation.

Usage:
    python enscgp_train.py --index 5 --sigma_mean 10 --sigma_spread 92.34 [--output out.npy]
    python enscgp_train.py --all --sigma_mean 10 --sigma_spread 92.34 [--output data/enscgp_posterior.npy]
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve

from coarsening_operator import load_coarsening_operator
from variance_recalibration import StratifiedRecalibrationMaps, apply_fortin_to_cholesky, recalibrate_cholesky


def load_neighbors(neighbors_path: Path) -> np.ndarray:
    if not neighbors_path.exists():
        raise FileNotFoundError(f"Neighbors file not found: {neighbors_path}")
    neighbors = np.load(neighbors_path)
    if neighbors.ndim != 2:
        raise ValueError("Neighbors array must be 2D (N, k)")
    return neighbors

def load_hr(wrf_path: Path) -> np.ndarray:
    if not wrf_path.exists():
        raise FileNotFoundError(f"WRF file not found: {wrf_path}")
    arr = np.load(wrf_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Unsupported WRF array shape: {arr.shape}")
    return arr

def load_era5(era5_path: Path) -> np.ndarray:
    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {era5_path}")
    arr = np.load(era5_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Unsupported ERA5 array shape: {arr.shape}")
    return arr

def load_observation_operator(coarsening_path: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    """Load H (1156, 40000), mapping one HR (200x200) channel to one LR (34x34) channel, and the
    boolean mask of LR cells actually covered by the HR domain. Returns H restricted to the valid
    rows (the rest carry no information) together with the mask, which is also needed to pick out
    the matching entries of y.
    """
    if not coarsening_path.exists():
        raise FileNotFoundError(f"Coarsening operator file not found: {coarsening_path}")
    H, valid = load_coarsening_operator(str(coarsening_path))
    return H[valid], valid

def build_r_inv(n_valid: int, sigma: float) -> sp.csr_matrix:
    """R_inv = (1/sigma^2) * I over both channels' valid observations, shape (2*n_valid, 2*n_valid)."""
    return sp.identity(2 * n_valid, format="csr") / (sigma ** 2)

def load_observation(era5: np.ndarray, index: int, valid: np.ndarray) -> np.ndarray:
    """y for `index`: ERA5 u then v, raveled and restricted to the valid LR cells -- matches the
    [u; v] layout of mean/A and the row layout of H_full = block_diag([H, H]) inside enscgp().
    """
    sample = np.asarray(era5[index], dtype=np.float64)  # (2, 34, 34)
    u = sample[0].ravel()[valid]
    v = sample[1].ravel()[valid]
    return np.concatenate([u, v])

def prior(index: int, neighbors: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble prior (mean, A) for `index`, built from its neighbors' WRF fields.
    mean : (n,)    ensemble mean
    A    : (n, k)  anomaly matrix, A @ A.T = sample covariance
    """
    k = neighbors.shape[1]
    neighbor_indices = neighbors[index]  # shape (k,)
    neighbor_data = data[neighbor_indices].astype(np.float64)  # shape (k, 2, 200, 200)
    flat = neighbor_data.reshape(k, -1)  # shape (k, 2*200*200)
    mean = flat.mean(axis=0)  # shape (2*200*200,)
    anomalies = flat - mean  # shape (k, 2*200*200)
    A = anomalies.T / np.sqrt(k - 1) # shape (2*200*200,k)
    return mean, A

def enscgp(mean: np.ndarray, A: np.ndarray, H: np.ndarray, R_inv: np.ndarray, y: np.ndarray):
    """
    Ens-CGP conditioning via the low-rank (ensemble-space) form.

    mean : (n,)        prior mean, n = 2*200*200 = 80000
    A    : (n, k)      anomaly matrix, K = A @ A.T (implied, never formed)
    H    : (m, n)      sparse observation operator, m = obs dimension
    R_inv: (m, m)      inverse observation-noise covariance (sparse/diagonal)
    y    : (m,)        observation

    n = 2*200*200, k = 36, m = 34*34

    Returns:
    mean_post : (n,)     posterior mean
    A_post    : (n, k)   posterior anomaly matrix (posterior square root)
    """
    H_full = sp.block_diag([H, H])
    k = A.shape[1]

    # Project anomalies into observation space: HA, shape (m, k)
    HA = H_full @ A                          # sparse @ dense -> dense (m, k)

    # Innovation: y - H @ mean, shape (m,)
    innovation = y - H_full @ mean          # (m,)

    # --- The key low-rank trick: work in k-dimensional space ---
    # We need (H K H^T + R)^{-1} applied to things, where K = A A^T.
    # Using the matrix inversion (Woodbury) identity, this reduces to a
    # k x k system instead of an m x m one.

    # S_k = I_k + (HA)^T R_inv (HA)    -> (k, k)  small!
    HA_Rinv = HA.T @ R_inv             # (k, m)
    S_k = np.eye(k) + HA_Rinv @ HA     # (k, k)

    # --- Posterior mean ---
    # gain applied to innovation, all routed through k-dim space:
    # rhs = (HA)^T R_inv innovation     -> (k,)
    rhs = HA_Rinv @ innovation         # (k,)
    # solve S_k w = rhs                 -> (k,)
    w = solve(S_k, rhs, assume_a='pos')
    # mean_post = mean + A @ ( (HA)^T R_inv innovation  -  ... )  via the identity
    # The clean Woodbury result for the mean update:
    mean_post = mean + A @ (rhs - HA_Rinv @ HA @ w)   # see derivation note below

    # --- Posterior anomalies (square-root update) ---
    # K_post = A (I - (HA)^T (HKH^T+R)^{-1} HA) A^T  =  A T A^T
    # We need a symmetric factor: A_post = A @ X where X X^T = (I_k - M),
    # M = (HA)^T (H K H^T + R)^{-1} (HA), computed in k-space.
    M = HA_Rinv @ HA - HA_Rinv @ HA @ solve(S_k, HA_Rinv @ HA, assume_a='pos')
    # symmetric eigardecomposition of (I_k - M) to get the matrix square root
    T = np.eye(k) - M
    evals, evecs = np.linalg.eigh(T)
    evals = np.clip(evals, 0, None)               # guard tiny negatives
    X = evecs @ np.diag(np.sqrt(evals)) @ evecs.T  # (k, k), symmetric sqrt
    A_post = A @ X                                  # (n, k)

    return mean_post, A_post

def posterior_to_output(
    mean_post: np.ndarray, A_post: np.ndarray, hw: int = 200, eps: float = 1e-12,
    recalibration_maps: StratifiedRecalibrationMaps | None = None,
) -> np.ndarray:
    """Pack (mean_post, A_post) into the [u, v, L11, L21, L22] output: the posterior mean plus the
    lower-Cholesky factor of the per-pixel 2x2 (u, v) covariance implied by A_post (the posterior
    anomaly matrix, A_post @ A_post.T = posterior covariance).

    If `recalibration_maps` is given (see variance_recalibration.py), the Fortin finite-ensemble
    correction (k taken from A_post's column count) and the fitted spread-skill recalibration are
    applied to the Cholesky factor before packing the output. The mean is never touched; output is
    bit-for-bit identical to before when `recalibration_maps` is None (the default).
    """
    n = hw * hw
    u = mean_post[:n].reshape(hw, hw)
    v = mean_post[n:].reshape(hw, hw)
    A_u, A_v = A_post[:n], A_post[n:]

    var_u = np.sum(A_u * A_u, axis=1)
    var_v = np.sum(A_v * A_v, axis=1)
    cov_uv = np.sum(A_u * A_v, axis=1)

    L11 = np.sqrt(np.maximum(var_u, eps))
    L21 = cov_uv / L11
    L22 = np.sqrt(np.maximum(var_v - L21 ** 2, eps))

    if recalibration_maps is not None:
        k = A_post.shape[1]
        L11, L21, L22 = apply_fortin_to_cholesky(L11, L21, L22, k=k)
        magnitude = np.sqrt(u.ravel() ** 2 + v.ravel() ** 2)
        L11, L21, L22 = recalibrate_cholesky(L11, L21, L22, magnitude, recalibration_maps)

    return np.stack(
        [u, v, L11.reshape(hw, hw), L21.reshape(hw, hw), L22.reshape(hw, hw)], axis=0
    ).astype(np.float32)

def compute_index(
    index: int,
    neighbors: np.ndarray,
    wrf: np.ndarray,
    era5: np.ndarray,
    H_valid: sp.csr_matrix,
    valid: np.ndarray,
    R_inv_mean: sp.csr_matrix,
    R_inv_spread: sp.csr_matrix,
    recalibration_maps: StratifiedRecalibrationMaps | None = None,
) -> np.ndarray:
    """Two EnsCGP conditioning passes over the same prior/observation: the posterior MEAN
    comes from the R_inv_mean pass, the posterior SPREAD (A_post, hence the output Cholesky
    factor) from the R_inv_spread pass. See module docstring for why these are decoupled.
    """
    mean, A = prior(index, neighbors, wrf)
    y = load_observation(era5, index, valid)
    mean_post, _ = enscgp(mean, A, H_valid, R_inv_mean, y)
    _, A_post = enscgp(mean, A, H_valid, R_inv_spread, y)
    return posterior_to_output(mean_post, A_post, recalibration_maps=recalibration_maps)

def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wrf_path", type=Path, default=data_dir / "wrf_uv.npy")
    parser.add_argument("--neighbors_path", type=Path, default=data_dir / "neighbor_train_only.npy")
    parser.add_argument("--era5_path", type=Path, default=data_dir / "era5_uv_2ch_native34.npy")
    parser.add_argument("--coarsening_path", type=Path, default=data_dir / "coarsening_operator_H.npy")
    parser.add_argument(
        "--sigma_mean", type=float, default=15.0,
        help="ERA5 observation-noise std used for the posterior MEAN's conditioning pass "
             "(small = trusts ERA5 more = better MAE; see module docstring)",
    )
    parser.add_argument(
        "--sigma_spread", type=float, default=92.91,
        help="ERA5 observation-noise std used for the posterior SPREAD's conditioning pass "
             "(larger = weaker conditioning = better-calibrated spread; see module docstring)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Compute the EnsCGP posterior for a single sample index")
    group.add_argument("--all", action="store_true", help="Compute and save the EnsCGP posterior for every sample")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .npy path. --all default: data/enscgp_posterior.npy; --index: only saved if given",
    )
    parser.add_argument(
        "--recalibration_map", type=Path, default=None,
        help="Path to a fitted StratifiedRecalibrationMaps JSON (see variance_recalibration.py "
             "--fit-split/--eval-split). If given, applies the Fortin finite-ensemble correction "
             "and the fitted spread-skill recalibration to L11/L21/L22 before output; means are "
             "untouched. If omitted (default), output is identical to before -- uncalibrated.",
    )
    args = parser.parse_args()

    neighbors = load_neighbors(args.neighbors_path)
    wrf = load_hr(args.wrf_path)
    era5 = load_era5(args.era5_path)
    H_valid, valid = load_observation_operator(args.coarsening_path)
    n_valid = int(valid.sum())
    R_inv_mean = build_r_inv(n_valid, args.sigma_mean)
    R_inv_spread = build_r_inv(n_valid, args.sigma_spread)
    recalibration_maps = (
        StratifiedRecalibrationMaps.load(args.recalibration_map) if args.recalibration_map is not None else None
    )

    if args.index is not None:
        result = compute_index(
            args.index, neighbors, wrf, era5, H_valid, valid, R_inv_mean, R_inv_spread, recalibration_maps
        )
        print(f"index={args.index}: output shape={result.shape}, dtype={result.dtype}")
        if args.output is not None:
            np.save(args.output, result)
            print(f"Saved to {args.output}")
        return

    n = neighbors.shape[0]
    out_path = args.output or (data_dir / "enscgp_posterior.npy")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, 5, 200, 200))
    for i in range(n):
        out[i] = compute_index(i, neighbors, wrf, era5, H_valid, valid, R_inv_mean, R_inv_spread, recalibration_maps)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"Processed {i + 1}/{n}")
    out.flush()
    print(f"Saved EnsCGP posteriors to {out_path}, shape={out.shape}")

if __name__ == "__main__":
    main()
