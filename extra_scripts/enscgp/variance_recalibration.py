"""Variance recalibration for Ens-CGP posteriors (spread-skill calibration).

Statistical downscaling of wind (ERA5 -> WRF), 200x200 grid, components u, v.
The Ens-CGP posterior (enscgp_train.py) is well-calibrated in bulk (SSR ~ 1.0)
but underconfident in the high-wind-magnitude tail (SSR ~ 0.70). This module
fits a low-parameter, monotonic correction map sigma_pred -> sigma_corrected
that fixes the fixable part of that mis-scaling without touching the mean or
over-dispersing the bulk.

Pipeline (in order):
1. Fortin finite-ensemble correction (apply_fortin_to_cholesky / fortin_factor):
   a fixed, deterministic inflation of the whole posterior covariance by
   (k+1)/k (or by effective sample size N_eff if weighted), correcting the
   known low bias of a k-member sample covariance. Always applied first;
   everything below operates on the Fortin-corrected sigma_pred.
2. Binned monotonic recalibration (fit_recalibration_map / RecalibrationMap):
   per component, pixels are quantile-binned by sigma_pred, sigma_realized =
   RMS(error) is computed per bin, and an isotonic regression through those
   bin points gives a monotonic sigma_pred -> sigma_corrected map.
3. Magnitude stratification (fit_stratified_maps / StratifiedRecalibrationMaps):
   a separate map per (component, stratum), stratum in {"bulk", "extreme"}
   split by a wind-magnitude percentile (computed from the POSTERIOR mean, so
   it's available at apply time, when truth is unknown).
4. Diagnostics (spread_skill_ratio, reliability_table, evaluate_calibration,
   rank_histogram / prior_ensemble_rank_histogram): SSR and reliability
   pre/post recalibration, plus a rank histogram against the k prior analog
   members as a structural coverage-limit check that recalibration cannot fix
   (a marginal-variance rescaling can't manufacture ensemble members that
   were never sampled).

Split discipline: every fit_* function takes explicit fit-set arrays, and
every evaluate_* function takes explicit eval-set arrays. Nothing here picks
a train/val/test split internally -- the caller (see main() below for the
real-data driver) controls which event-disjoint slice is which.

Usage (fits real maps from data/enscgp_posterior.npy + data/wrf_uv.npy):
    python variance_recalibration.py --fit-split val --eval-split test
Smoke test (synthetic data, no files needed):
    python variance_recalibration.py --smoke-test
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import DATA_DIR as DEFAULT_DATA_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1: finite-ensemble (Fortin) correction
# ---------------------------------------------------------------------------

def fortin_factor(k: int | None = None, weights: np.ndarray | None = None) -> float:
    """(k+1)/k finite-ensemble variance-inflation factor. If `weights` is given,
    k is replaced by the effective sample size N_eff = (sum w)^2 / sum(w^2).
    """
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        k_eff = float((weights.sum() ** 2) / np.sum(weights ** 2))
    elif k is not None:
        k_eff = float(k)
    else:
        raise ValueError("fortin_factor requires either k or weights")
    return (k_eff + 1.0) / k_eff


def apply_fortin_to_cholesky(L11: np.ndarray, L21: np.ndarray, L22: np.ndarray,
                              k: int | None = None, weights: np.ndarray | None = None):
    """Inflate the WHOLE posterior covariance by the Fortin factor: Sigma <- factor * Sigma,
    equivalently L <- sqrt(factor) * L. Scaling the entire Cholesky factor by one scalar
    leaves the u-v correlation coefficient exactly unchanged (it cancels in the ratio
    cov/(std_u*std_v)), so this step never needs to touch the correlation structure.
    """
    scale = np.sqrt(fortin_factor(k=k, weights=weights))
    return L11 * scale, L21 * scale, L22 * scale


# ---------------------------------------------------------------------------
# Spread-skill ratio
# ---------------------------------------------------------------------------

def spread_skill_ratio(sigma_pred: np.ndarray, error: np.ndarray) -> float:
    """SSR = RMS(sigma_pred) / RMS(error), in variance space. 1=calibrated, <1
    overconfident (underdispersed), >1 underconfident (overdispersed)."""
    sigma_pred = np.asarray(sigma_pred)
    error = np.asarray(error)
    return float(np.sqrt(np.mean(sigma_pred ** 2)) / np.sqrt(np.mean(error ** 2)))


# ---------------------------------------------------------------------------
# Step 2: binned monotonic recalibration map
# ---------------------------------------------------------------------------

@dataclass
class RecalibrationMap:
    """Monotonic piecewise-linear map sigma_pred -> sigma_corrected, defined by
    increasing knot points (x=sigma_pred bin centers, y=isotonic-fit sigma_realized).
    Extrapolation beyond [x[0], x[-1]] is clamped (flat) to the nearest boundary value.
    """
    x: list
    y: list

    def __call__(self, sigma_pred: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(sigma_pred), self.x, self.y)

    def to_dict(self) -> dict:
        return {"x": list(self.x), "y": list(self.y)}

    @classmethod
    def from_dict(cls, d: dict) -> "RecalibrationMap":
        return cls(x=list(d["x"]), y=list(d["y"]))


def fit_recalibration_map(sigma_pred: np.ndarray, error: np.ndarray, n_bins: int = 18) -> RecalibrationMap:
    """Fit a monotonic sigma_pred -> sigma_corrected map from a FIT-set sample of
    per-pixel (sigma_pred, error) pairs:
      1. Quantile-bin pixels by sigma_pred (~equal count per bin).
      2. Per bin: x = median(sigma_pred), y = sigma_realized = RMS(error).
      3. Isotonic regression (monotonic increasing) through the (x, y) bin points.
    Bins with fewer than 2 points are dropped (can happen with heavily tied sigma_pred).
    """
    sigma_pred = np.asarray(sigma_pred).ravel()
    error = np.asarray(error).ravel()
    if sigma_pred.shape != error.shape:
        raise ValueError(f"sigma_pred and error must match shapes, got {sigma_pred.shape} vs {error.shape}")
    if sigma_pred.size < 2 * n_bins:
        raise ValueError(f"Need at least {2 * n_bins} pixels to fill {n_bins} bins, got {sigma_pred.size}")

    edges = np.quantile(sigma_pred, np.linspace(0.0, 1.0, n_bins + 1))
    bin_idx = np.digitize(sigma_pred, edges[1:-1], right=False)

    centers, realized = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 2:
            continue
        centers.append(float(np.median(sigma_pred[mask])))
        realized.append(float(np.sqrt(np.mean(error[mask] ** 2))))

    centers_arr = np.asarray(centers)
    realized_arr = np.asarray(realized)
    order = np.argsort(centers_arr)
    centers_arr, realized_arr = centers_arr[order], realized_arr[order]

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    fitted = iso.fit_transform(centers_arr, realized_arr)
    return RecalibrationMap(x=centers_arr.tolist(), y=fitted.tolist())


# ---------------------------------------------------------------------------
# Step 3: magnitude stratification
# ---------------------------------------------------------------------------

@dataclass
class StratifiedRecalibrationMaps:
    """Fitted recalibration maps keyed by f"{component}|{stratum}", component in
    {"u","v"}, stratum in {"bulk","extreme"} split by `magnitude_threshold` (a wind-speed
    value, the `percentile`-th percentile of the FIT set's posterior-mean magnitude).
    """
    maps: dict
    magnitude_threshold: float
    percentile: float

    def apply(self, sigma_pred: np.ndarray, magnitude: np.ndarray, component: str) -> np.ndarray:
        sigma_pred = np.asarray(sigma_pred, dtype=np.float64)
        magnitude = np.asarray(magnitude, dtype=np.float64)
        is_extreme = magnitude >= self.magnitude_threshold
        out = np.empty_like(sigma_pred)
        out[~is_extreme] = self.maps[f"{component}|bulk"](sigma_pred[~is_extreme])
        out[is_extreme] = self.maps[f"{component}|extreme"](sigma_pred[is_extreme])
        return out

    def to_dict(self) -> dict:
        return {
            "magnitude_threshold": self.magnitude_threshold,
            "percentile": self.percentile,
            "maps": {key: m.to_dict() for key, m in self.maps.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StratifiedRecalibrationMaps":
        return cls(
            maps={key: RecalibrationMap.from_dict(m) for key, m in d["maps"].items()},
            magnitude_threshold=d["magnitude_threshold"],
            percentile=d["percentile"],
        )

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StratifiedRecalibrationMaps":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def fit_stratified_maps(sigma_pred: dict, error: dict, magnitude: np.ndarray,
                         percentile: float = 95.0, n_bins: int = 18) -> StratifiedRecalibrationMaps:
    """Fit (component x stratum) recalibration maps on a FIT-set.

    sigma_pred, error: {"u": array, "v": array}, all pixels from the FIT slice, already
        Fortin-corrected (sigma_pred should be sqrt of Fortin-corrected variance).
    magnitude: posterior-mean wind speed per pixel, same flattening/order as sigma_pred/error.
    """
    magnitude = np.asarray(magnitude).ravel()
    threshold = float(np.percentile(magnitude, percentile))
    is_extreme = magnitude >= threshold

    maps = {}
    for component in ("u", "v"):
        sp_c = np.asarray(sigma_pred[component]).ravel()
        err_c = np.asarray(error[component]).ravel()
        for stratum, mask in (("bulk", ~is_extreme), ("extreme", is_extreme)):
            maps[f"{component}|{stratum}"] = fit_recalibration_map(sp_c[mask], err_c[mask], n_bins=n_bins)

    return StratifiedRecalibrationMaps(maps=maps, magnitude_threshold=threshold, percentile=percentile)


def recalibrate_cholesky(L11: np.ndarray, L21: np.ndarray, L22: np.ndarray, magnitude: np.ndarray,
                          maps: StratifiedRecalibrationMaps, eps: float = 1e-12):
    """Apply fitted recalibration maps to a per-pixel 2x2 Cholesky factor (steps 2+3 only --
    call apply_fortin_to_cholesky first and pass its output in here).

    var_u = L11^2, var_v = L21^2+L22^2, rho = L21/sqrt(var_v) (u-v correlation). The two
    marginal stds are recalibrated independently via `maps`; rho is held fixed and the
    Cholesky factor rebuilt from (std_u_new, std_v_new, rho), which is the only
    non-arbitrary way to turn two independently-rescaled marginal stds back into a valid
    joint covariance without silently changing the dependence structure. The mean is
    never touched -- only L11/L21/L22 are returned.
    """
    L11 = np.asarray(L11, dtype=np.float64)
    L21 = np.asarray(L21, dtype=np.float64)
    L22 = np.asarray(L22, dtype=np.float64)
    std_v = np.sqrt(L21 ** 2 + L22 ** 2)
    rho = L21 / np.maximum(std_v, eps)

    std_u_corrected = maps.apply(L11, magnitude, "u")
    std_v_corrected = maps.apply(std_v, magnitude, "v")

    L11_new = std_u_corrected
    L21_new = rho * std_v_corrected
    L22_new = std_v_corrected * np.sqrt(np.clip(1.0 - rho ** 2, 0.0, 1.0))
    return L11_new, L21_new, L22_new


# ---------------------------------------------------------------------------
# Step 5: diagnostics
# ---------------------------------------------------------------------------

def reliability_table(sigma_pred: np.ndarray, error: np.ndarray, n_bins: int = 18) -> list:
    """[{"sigma_pred": bin median, "sigma_realized": RMS(error) in bin, "count": n}, ...]."""
    sigma_pred = np.asarray(sigma_pred).ravel()
    error = np.asarray(error).ravel()
    edges = np.quantile(sigma_pred, np.linspace(0.0, 1.0, n_bins + 1))
    bin_idx = np.digitize(sigma_pred, edges[1:-1], right=False)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "sigma_pred": float(np.median(sigma_pred[mask])),
            "sigma_realized": float(np.sqrt(np.mean(error[mask] ** 2))),
            "count": int(mask.sum()),
        })
    return rows


def evaluate_calibration(sigma_pred: dict, error: dict, magnitude: np.ndarray,
                          maps: StratifiedRecalibrationMaps, n_bins: int = 18) -> dict:
    """Pre/post-recalibration diagnostics on an EVAL-set: overall SSR, bulk/extreme
    SSR, and reliability tables, per component. sigma_pred/error must already be
    Fortin-corrected (pre = Fortin-only baseline; post = Fortin + recalibration map).
    """
    magnitude = np.asarray(magnitude).ravel()
    is_extreme = magnitude >= maps.magnitude_threshold

    report = {
        "magnitude_threshold": maps.magnitude_threshold,
        "percentile": maps.percentile,
        "n_extreme": int(is_extreme.sum()),
        "n_bulk": int((~is_extreme).sum()),
        "components": {},
    }

    for component in ("u", "v"):
        sp_c = np.asarray(sigma_pred[component]).ravel()
        err_c = np.asarray(error[component]).ravel()
        sp_corrected = maps.apply(sp_c, magnitude, component)

        report["components"][component] = {
            "overall_ssr_pre": spread_skill_ratio(sp_c, err_c),
            "overall_ssr_post": spread_skill_ratio(sp_corrected, err_c),
            "bulk_ssr_pre": spread_skill_ratio(sp_c[~is_extreme], err_c[~is_extreme]),
            "bulk_ssr_post": spread_skill_ratio(sp_corrected[~is_extreme], err_c[~is_extreme]),
            "extreme_ssr_pre": spread_skill_ratio(sp_c[is_extreme], err_c[is_extreme]),
            "extreme_ssr_post": spread_skill_ratio(sp_corrected[is_extreme], err_c[is_extreme]),
            "reliability_pre": reliability_table(sp_c, err_c, n_bins=n_bins),
            "reliability_post": reliability_table(sp_corrected, err_c, n_bins=n_bins),
        }

    return report


def _rank_counts(truth: np.ndarray, members: np.ndarray, k: int) -> np.ndarray:
    """truth: (...,) flat; members: (..., k) same leading shape. Returns counts[0..k]."""
    ranks = np.sum(members <= truth[..., None], axis=-1).ravel()
    return np.bincount(ranks, minlength=k + 1)


def rank_histogram(truth: np.ndarray, members: np.ndarray) -> dict:
    """Verification rank histogram: for each pixel, rank of `truth` among `members` (k
    values), rank in [0, k]. rank=0: truth below every member; rank=k: truth above every
    member (the high-side pile-up relevant to the underdispersed extreme tail).
    """
    k = members.shape[-1]
    counts = _rank_counts(np.asarray(truth), np.asarray(members), k)
    fractions = counts / counts.sum()
    return {
        "k": k,
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "frac_below_all": float(fractions[0]),
        "frac_above_all": float(fractions[k]),
    }


def prior_ensemble_rank_histogram(wrf: np.ndarray, neighbors: np.ndarray, sample_indices: np.ndarray,
                                   component: int, batch_size: int = 50) -> dict:
    """Rank histogram of truth among the k PRIOR analog members (raw WRF neighbor fields --
    not the post-conditioning posterior ensemble, which isn't cached, see module docstring),
    for `component` (0=u, 1=v) over `sample_indices`. Batched over samples since materializing
    all (S, 200, 200, k) members at once does not fit in memory for realistic S, k.
    """
    k = neighbors.shape[1]
    sample_indices = np.asarray(sample_indices)
    counts = np.zeros(k + 1, dtype=np.int64)

    for start in range(0, len(sample_indices), batch_size):
        batch = sample_indices[start:start + batch_size]
        truth = np.asarray(wrf[batch, component])  # (b, 200, 200)
        neighbor_idx = neighbors[batch]  # (b, k)
        members = np.asarray(wrf[neighbor_idx, component])  # (b, k, 200, 200)
        members = np.moveaxis(members, 1, -1)  # (b, 200, 200, k)
        counts += _rank_counts(truth, members, k)

    fractions = counts / counts.sum()
    return {
        "k": k,
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "frac_below_all": float(fractions[0]),
        "frac_above_all": float(fractions[k]),
    }


# ---------------------------------------------------------------------------
# Synthetic smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    n = 200_000
    percentile, n_bins = 95.0, 18

    magnitude = rng.uniform(0.0, 20.0, size=n)
    is_extreme = magnitude >= np.percentile(magnitude, percentile)

    sigma_pred_by_component = {}
    error_by_component = {}
    for component in ("u", "v"):
        sigma_pred = rng.uniform(0.5, 3.0, size=n)
        # Known miscalibration: errors are 1.0x sigma_pred in bulk, 1.4x in the extreme
        # stratum (underdispersed/overconfident there) -- exactly the failure mode
        # described in the module docstring.
        true_std = np.where(is_extreme, 1.4 * sigma_pred, 1.0 * sigma_pred)
        error = rng.normal(0.0, 1.0, size=n) * true_std
        sigma_pred_by_component[component] = sigma_pred
        error_by_component[component] = error

    fit_mask = rng.random(n) < 0.5  # disjoint fit/eval halves
    eval_mask = ~fit_mask

    maps = fit_stratified_maps(
        sigma_pred={c: a[fit_mask] for c, a in sigma_pred_by_component.items()},
        error={c: a[fit_mask] for c, a in error_by_component.items()},
        magnitude=magnitude[fit_mask],
        percentile=percentile,
        n_bins=n_bins,
    )

    report = evaluate_calibration(
        sigma_pred={c: a[eval_mask] for c, a in sigma_pred_by_component.items()},
        error={c: a[eval_mask] for c, a in error_by_component.items()},
        magnitude=magnitude[eval_mask],
        maps=maps,
        n_bins=n_bins,
    )

    for component in ("u", "v"):
        comp = report["components"][component]
        print(
            f"[{component}] bulk SSR pre={comp['bulk_ssr_pre']:.3f} post={comp['bulk_ssr_post']:.3f} | "
            f"extreme SSR pre={comp['extreme_ssr_pre']:.3f} post={comp['extreme_ssr_post']:.3f}"
        )
        # Bulk is already near-perfectly calibrated by construction (true_std = 1.0*sigma_pred),
        # so bulk_ssr_pre is itself just sampling noise around 1.0 -- recalibration fit on a
        # separate finite FIT half adds its own noise on top. Check absolute closeness to 1.0
        # rather than requiring strict improvement over an already-near-1.0 baseline.
        assert abs(comp["bulk_ssr_post"] - 1.0) < 0.05, "bulk SSR should stay close to 1.0 after recalibration"
        assert abs(comp["extreme_ssr_post"] - 1.0) < abs(comp["extreme_ssr_pre"] - 1.0), (
            "extreme SSR should move closer to 1.0 after recalibration"
        )

    # Serialization round-trip.
    tmp_path = Path("/tmp/_variance_recalibration_smoke_maps.json")
    maps.save(tmp_path)
    loaded = StratifiedRecalibrationMaps.load(tmp_path)
    np.testing.assert_allclose(
        loaded.apply(sigma_pred_by_component["u"][:100], magnitude[:100], "u"),
        maps.apply(sigma_pred_by_component["u"][:100], magnitude[:100], "u"),
    )
    tmp_path.unlink()

    # Fortin factor: plain k, weighted-uniform should match unweighted k, and the
    # Cholesky-scaling form should leave the u-v correlation exactly unchanged.
    assert abs(fortin_factor(k=36) - 37.0 / 36.0) < 1e-9
    assert abs(fortin_factor(weights=np.ones(36)) - fortin_factor(k=36)) < 1e-9

    L11 = rng.uniform(0.5, 2.0, size=1000)
    L21 = rng.uniform(-1.0, 1.0, size=1000)
    L22 = rng.uniform(0.5, 2.0, size=1000)
    rho_before = L21 / np.sqrt(L21 ** 2 + L22 ** 2)
    L11_f, L21_f, L22_f = apply_fortin_to_cholesky(L11, L21, L22, k=36)
    rho_after_fortin = L21_f / np.sqrt(L21_f ** 2 + L22_f ** 2)
    np.testing.assert_allclose(rho_before, rho_after_fortin, atol=1e-10)

    # recalibrate_cholesky must also preserve rho, regardless of what the maps do.
    mag = rng.uniform(0.0, 20.0, size=1000)
    L11_r, L21_r, L22_r = recalibrate_cholesky(L11_f, L21_f, L22_f, mag, maps)
    rho_after_recal = L21_r / np.sqrt(L21_r ** 2 + L22_r ** 2)
    np.testing.assert_allclose(rho_before, rho_after_recal, atol=1e-8)

    print("Smoke test passed.")


# ---------------------------------------------------------------------------
# Real-data driver: fit maps from data/enscgp_posterior.npy + data/wrf_uv.npy
# ---------------------------------------------------------------------------

def _load_split_pixels(data_dir: Path, idx: np.ndarray, k: int) -> dict:
    """Per-pixel arrays (flattened over samples x H x W) for one split slice: posterior
    mean/sigma_pred (Fortin-corrected) per component, error per component, and the
    posterior-mean wind magnitude used for stratification.
    """
    posterior = np.load(data_dir / "enscgp_posterior.npy", mmap_mode="r")[idx]  # (S,5,200,200)
    wrf = np.load(data_dir / "wrf_uv.npy", mmap_mode="r")[idx]  # (S,2,200,200)

    mean_u, mean_v = posterior[:, 0], posterior[:, 1]
    L11, L21, L22 = posterior[:, 2], posterior[:, 3], posterior[:, 4]
    truth_u, truth_v = wrf[:, 0], wrf[:, 1]

    L11_f, L21_f, L22_f = apply_fortin_to_cholesky(L11, L21, L22, k=k)
    sigma_pred_u_raw = np.abs(L11).ravel()
    sigma_pred_v_raw = np.sqrt(L21 ** 2 + L22 ** 2).ravel()
    sigma_pred_u = np.abs(L11_f).ravel()
    sigma_pred_v = np.sqrt(L21_f ** 2 + L22_f ** 2).ravel()

    return {
        "sigma_pred_raw": {"u": sigma_pred_u_raw, "v": sigma_pred_v_raw},
        "sigma_pred": {"u": sigma_pred_u, "v": sigma_pred_v},
        "error": {"u": (truth_u - mean_u).ravel(), "v": (truth_v - mean_v).ravel()},
        "magnitude": np.sqrt(mean_u ** 2 + mean_v ** 2).ravel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true", help="Run the synthetic smoke test and exit")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits_path", type=Path, default=None,
                         help="Defaults to <data_dir>/splits_70_15_15/split_indices.npz")
    parser.add_argument("--neighbors_path", type=Path, default=None,
                         help="Defaults to <data_dir>/neighbor_train_only.npy (read k from its shape)")
    parser.add_argument("--fit-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--n_bins", type=int, default=18)
    parser.add_argument("--output", type=Path, default=None,
                         help="Defaults to <data_dir>/variance_recalibration_maps.json")
    parser.add_argument("--report_output", type=Path, default=None,
                         help="Defaults to <data_dir>/variance_recalibration_diagnostics.json")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
        return

    if args.fit_split == args.eval_split:
        raise ValueError("--fit-split and --eval-split must differ (no fitting and evaluating on the same pixels)")

    data_dir = args.data_dir
    splits_path = args.splits_path or (data_dir / "splits_70_15_15" / "split_indices.npz")
    neighbors_path = args.neighbors_path or (data_dir / "neighbor_train_only.npy")

    splits = np.load(splits_path)
    neighbors = np.load(neighbors_path, mmap_mode="r")
    k = neighbors.shape[1]
    print(f"k = {k} (from {neighbors_path.name}, unweighted)")

    fit_idx = splits[f"{args.fit_split}_idx"]
    eval_idx = splits[f"{args.eval_split}_idx"]
    print(f"Fit split '{args.fit_split}': {len(fit_idx)} samples; eval split '{args.eval_split}': {len(eval_idx)} samples")

    fit_data = _load_split_pixels(data_dir, fit_idx, k)
    maps = fit_stratified_maps(
        sigma_pred=fit_data["sigma_pred"], error=fit_data["error"], magnitude=fit_data["magnitude"],
        percentile=args.percentile, n_bins=args.n_bins,
    )
    print(f"Magnitude threshold (p{args.percentile}): {maps.magnitude_threshold:.3f} m/s")

    eval_data = _load_split_pixels(data_dir, eval_idx, k)
    report = evaluate_calibration(
        sigma_pred=eval_data["sigma_pred"], error=eval_data["error"], magnitude=eval_data["magnitude"],
        maps=maps, n_bins=args.n_bins,
    )

    print("\n--- Spread-skill ratio on eval split (1.0 = calibrated) ---")
    for component in ("u", "v"):
        raw_ssr = spread_skill_ratio(eval_data["sigma_pred_raw"][component], eval_data["error"][component])
        comp = report["components"][component]
        print(f"[{component}] overall: raw(no Fortin)={raw_ssr:.3f}  Fortin-only(pre)={comp['overall_ssr_pre']:.3f}  "
              f"recalibrated(post)={comp['overall_ssr_post']:.3f}")
        print(f"[{component}] bulk:    Fortin-only(pre)={comp['bulk_ssr_pre']:.3f}  "
              f"recalibrated(post)={comp['bulk_ssr_post']:.3f}  (n={report['n_bulk']:,})")
        print(f"[{component}] extreme: Fortin-only(pre)={comp['extreme_ssr_pre']:.3f}  "
              f"recalibrated(post)={comp['extreme_ssr_post']:.3f}  (n={report['n_extreme']:,})")

    print("\n--- Rank histogram vs. prior k-member analog ensemble (eval split; structural coverage limit) ---")
    rank_reports = {}
    for component_idx, component in enumerate(("u", "v")):
        wrf = np.load(data_dir / "wrf_uv.npy", mmap_mode="r")
        rh = prior_ensemble_rank_histogram(wrf, neighbors, eval_idx, component=component_idx)
        rank_reports[component] = rh
        print(f"[{component}] k={rh['k']}  frac_below_all={rh['frac_below_all']:.4f}  "
              f"frac_above_all (top-rank pile-up)={rh['frac_above_all']:.4f}")

    report["rank_histogram"] = rank_reports

    output_path = args.output or (data_dir / "variance_recalibration_maps.json")
    maps.save(output_path)
    print(f"\nSaved fitted maps to {output_path}")

    report_path = args.report_output or (data_dir / "variance_recalibration_diagnostics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved diagnostics report to {report_path}")


if __name__ == "__main__":
    main()
