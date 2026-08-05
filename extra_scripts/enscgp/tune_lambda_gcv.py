"""Select the GCV-optimal EnsCGP regularization lambda (= sigma_obs^2) analytically,
via select_regularization_gcv from the research group's Ens-CGP implementation (not
bundled here -- see WIND_KEN_ENSCGP_DIR below).

Context: tune_sigma.py finds sigma by bisecting for a target posterior
spread-skill ratio -- an empirical spread-calibration criterion, evaluated by
actually re-running EnsCGP conditioning many times. This script asks a different
question: for each sample's k=36 neighbor ensemble, what regularization would a
Generalized-Cross-Validation criterion recommend, computed analytically from the
ensemble's own geometry (no bisection, no re-running enscgp(), no ERA5 observation
needed).

select_regularization_gcv's `beta = U.T @ dY` step requires the SVD input and the
regression target to live in the SAME pixel space (this is ken_enscgp_model.py's
built-in H=identity assumption: its build_training_matrices applies one shared
nanmask to both X and Y). The real EnsCGP conditioning in this repo uses H !=
identity (a genuine HR->LR coarsening operator), so U from SVD(H @ A) (2178 rows,
observation space) can't multiply A (80000 rows, full HR space) -- confirmed by a
shape-mismatch error when tried directly.

Fix: use data/era5_uv_2ch_bicubic.npy (ERA5 already bicubic-upsampled onto the
same 200x200 HR grid as WRF, index-aligned with wrf_uv.npy) as the "coarse-but-
resampled" X side, so X and Y share one grid -- exactly ken_enscgp_model.py's own
input/target convention, applied locally to one sample's k=36-neighbor ensemble
instead of ken's whole-dataset ensemble:
    X_j = era5_bicubic[neighbor_j],  Y_j = wrf[neighbor_j]   for each of the k neighbors
    A_X, A_Y = ensemble anomalies (mean-subtracted, /sqrt(k-1))
    U, s, _ = SVD(A_X)
    lambda = select_regularization_gcv(s, U, A_Y)   # = sigma_obs^2 for that sample

Since lambda is a variance (sigma_obs^2) while every other script here works in
terms of sigma (std) -- tune_sigma.py, enscgp_train.py's --sigma_mean/
--sigma_spread, build_r_inv -- this script reports both. Note GCV minimizes
prediction error, so this is a principled alternative to sigma_mean (accurate
posterior MEAN), not a spread-calibration target like sigma_spread; it will not
target SSR==1 the way tune_sigma.py does.

Usage:
    python tune_lambda_gcv.py --split val --n_samples 150
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import DATA_DIR as DEFAULT_DATA_DIR, KEN_ENSCGP_DIR  # noqa: E402
from enscgp_train import load_hr, load_neighbors  # noqa: E402

# select_regularization_gcv comes from the research group's Ens-CGP implementation,
# which is NOT redistributed with this repository (it is not ours to publish). This
# is the only script that needs it; everything else here runs without it.
if KEN_ENSCGP_DIR is None or not (KEN_ENSCGP_DIR / "utils.py").is_file():
    raise SystemExit(
        "tune_lambda_gcv.py needs the group's Ens-CGP `utils.select_regularization_gcv`,\n"
        "which is not bundled with this repository. Point WIND_KEN_ENSCGP_DIR at a\n"
        "checkout containing utils.py:\n"
        "    WIND_KEN_ENSCGP_DIR=/path/to/ken_enscgp python tune_lambda_gcv.py ...\n"
        "Every other script in this repo runs without it."
    )
sys.path.insert(0, str(KEN_ENSCGP_DIR))

from utils import select_regularization_gcv  # noqa: E402


def local_ensemble_anomalies(
    index: int, neighbors: np.ndarray, coarse: np.ndarray, fine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build (A_X, A_Y) ensemble anomaly matrices from `index`'s k neighbors: X_j is the
    coarse-but-HR-grid-resampled proxy for neighbor j, Y_j is neighbor j's true HR field.
    Mirrors ken_enscgp_model.py's compute_ensemble_statistics, restricted to one sample's
    local k-neighbor ensemble instead of the whole dataset."""
    neighbor_idx = neighbors[index]
    k = len(neighbor_idx)
    X = np.asarray(coarse[neighbor_idx], dtype=np.float64).reshape(k, -1).T  # (n_pixels, k)
    Y = np.asarray(fine[neighbor_idx], dtype=np.float64).reshape(k, -1).T
    A_X = (X - X.mean(axis=1, keepdims=True)) / np.sqrt(k - 1)
    A_Y = (Y - Y.mean(axis=1, keepdims=True)) / np.sqrt(k - 1)
    return A_X, A_Y


def lambda_for_sample(
    index: int, neighbors: np.ndarray, coarse: np.ndarray, fine: np.ndarray,
    n_grid: int, rank_threshold: float, verbose: bool = False,
) -> dict:
    A_X, A_Y = local_ensemble_anomalies(index, neighbors, coarse, fine)
    U, s, _ = np.linalg.svd(A_X, full_matrices=False)
    lam = select_regularization_gcv(s, U, A_Y, n_grid=n_grid, rank_threshold=rank_threshold, verbose=verbose)
    return {"index": index, "lambda": lam, "sigma": float(np.sqrt(lam))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--n_grid", type=int, default=200, help="Number of log-spaced lambda candidates GCV evaluates")
    parser.add_argument("--rank_threshold", type=float, default=1e-3,
                         help="Fraction of the largest singular value below which modes are discarded as noise")
    parser.add_argument("--verbose", action="store_true", help="Print GCV diagnostics for the first sample")
    args = parser.parse_args()

    data_dir = args.data_dir
    neighbors = load_neighbors(data_dir / "neighbor_train_only.npy")
    fine = load_hr(data_dir / "wrf_uv.npy")
    coarse = load_hr(data_dir / "era5_uv_2ch_bicubic.npy")

    splits = np.load(data_dir / "splits_70_15_15" / "split_indices.npz")
    rng = np.random.default_rng(0)
    split_idx = splits[f"{args.split}_idx"]
    sample_indices = rng.choice(split_idx, size=min(args.n_samples, len(split_idx)), replace=False)
    print(f"Using {len(sample_indices)} samples from split '{args.split}'")

    results = []
    for i, idx in enumerate(sample_indices):
        r = lambda_for_sample(
            int(idx), neighbors, coarse, fine, args.n_grid, args.rank_threshold,
            verbose=args.verbose and i == 0,
        )
        results.append(r)

    lambdas = np.array([r["lambda"] for r in results])
    sigmas = np.array([r["sigma"] for r in results])

    print(f"\n{'index':>10}  {'lambda':>14}  {'sigma':>10}")
    for r in results:
        print(f"{r['index']:>10}  {r['lambda']:>14.4f}  {r['sigma']:>10.4f}")

    print(f"\nGCV-optimal lambda (=sigma_obs^2) over {len(results)} samples from split '{args.split}':")
    print(f"  lambda: mean={lambdas.mean():.4f}  median={np.median(lambdas):.4f}  std={lambdas.std():.4f}")
    print(f"  sigma:  mean={sigmas.mean():.4f}  median={np.median(sigmas):.4f}  std={sigmas.std():.4f}")
    print(f"\nRecommended sigma (median of sqrt(lambda)) = {np.median(sigmas):.4f}")


if __name__ == "__main__":
    main()
