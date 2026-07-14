"""Local-patch (tiled) Ensemble Conditional Gaussian Process.

Motivation
----------
For high-resolution targets where the number of valid HR pixels ``p``
exceeds the ensemble size ``N`` (e.g. exp41: p = 28 224 vs N ≈ 2 232),
the global Kalman gain

    M = K_YX (K_XX + sigma_obs^2 I)^{-1}

is rank-deficient and GCV collapses onto a heavily shrunk solution.
Following the localization idea in the EnKF literature (Hamill &
Whitaker 2001; Houtekamer & Mitchell 2001) we fit EnsCGP independently
on small overlapping HR tiles where p_tile = W*W << N, so each local
covariance is well-conditioned.

Blend
-----
Overlapping tiles are merged by a Lagrangian KKT solution of

    min_{x_i}  sum_i (x_i - m_i)^T C_i^{-1} (x_i - m_i)
    s.t.       x_i[k] = x_j[k]  for k in overlap(i,j)

with C_i approximated by the stored diagonal posterior variance
``K_post_diag`` per tile.  The closed-form stationary point is the
precision-weighted average per HR pixel:

    y_hat[k]       = sum_i w_ik m_ik / sum_i w_ik,     w_ik = 1 / sigma_ik^2
    Var(y_hat[k])  = 1 / sum_i w_ik

which is exact for diagonal C.

Edge handling
-------------
If ``(H - W) % stride != 0`` the last tile along an axis is shifted
inward so its upper-left corner sits at ``H - W`` (likewise for x).
Every HR pixel is then covered by at least one tile.

Auto-activation
---------------
This class is swapped in for :class:`EnsCGPDownscaler` by
``scripts/train.py`` whenever ``p_valid > N`` (strict), or when the
experiment config explicitly requests ``local_patches.mode: true``.

Environment
-----------
``micromamba activate downscaling``
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pathlib
import pickle

from .model import (
    EnsCGPModel,
    create_nanmask,
    train_enscgp,
)


__all__ = [
    "LocalEnsCGPModel",
    "LocalEnsCGPDownscaler",
    "tile_indices",
]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_indices(
    shape: Tuple[int, int], window: int, stride: int
) -> List[Tuple[int, int]]:
    """Return the ``(y0, x0)`` upper-left corners of a tiling of ``shape``.

    Tiles have size ``window x window``.  The last tile along each axis
    is shifted inward so the full grid is covered when
    ``(H - window) % stride != 0``.

    Args:
        shape:    ``(H, W)``.
        window:   tile side length ``W_tile``.
        stride:   step between consecutive upper-left corners.

    Returns:
        List of ``(y0, x0)`` tuples.  Each tile spans rows
        ``y0 : y0 + window`` and columns ``x0 : x0 + window``.
    """
    H, W = shape
    if window > H or window > W:
        raise ValueError(
            f"window={window} larger than grid shape={shape}"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    def _axis(n: int) -> List[int]:
        starts = list(range(0, n - window + 1, stride))
        if not starts or starts[-1] + window < n:
            starts.append(n - window)
        return starts

    ys = _axis(H)
    xs = _axis(W)
    return [(y, x) for y in ys for x in xs]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class LocalEnsCGPModel:
    """Container for a collection of per-tile :class:`EnsCGPModel`."""

    tiles: List[EnsCGPModel]                 # one per tile
    tile_corners: List[Tuple[int, int]]      # (y0, x0) per tile
    window: int
    stride: int
    nanmask_flat: np.ndarray                 # HR nanmask (flattened order='F')
    output_shape: Tuple[int, int]
    input_shape: Tuple[int, int]
    n_ensemble: int
    tag: str = ""

    # Metrics recorded at fit time (helpful for diagnostics)
    obs_noise_var_mean: float = 0.0
    n_tiles: int = 0

    # Back-compat with EnsCGPModel attribute access used by train.py
    obs_noise_var: float = 0.0               # alias of obs_noise_var_mean
    regularization: Optional[float] = None   # kept for legacy readers

    # Lazy caches for reconstructed global operator (populated on first access)
    _M_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _mX_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _mY_cache: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Reconstruct effective global M, mX, mY from tiles
    # ------------------------------------------------------------------

    def _reconstruct(self) -> None:
        """Build effective global M, mX, mY via precision-weighted tiles.

        For each global valid pixel *k*, the effective prediction is:

            y_hat[k] = sum_i w_ik (M_i x_i + mY_i) / sum_i w_ik

        where the sum runs over tiles covering *k* and
        w_ik = 1 / K_post_diag_i[k_local].  Rearranging into the form
        ``M_global @ (x_global - mX_global) + mY_global`` gives a
        precision-weighted average of per-tile operators.

        The reconstructed M is ``(p_valid, p_valid)`` with the same
        semantic as :attr:`EnsCGPModel.M`.
        """
        if self._M_cache is not None:
            return

        ny, nx = self.output_shape
        p_full = ny * nx
        global_mask = self.nanmask_flat.astype(bool)
        p = int(global_mask.sum())
        W = self.window

        # Map from global flat index → global valid index
        g_flat_to_valid = np.full(p_full, -1, dtype=np.int64)
        g_flat_to_valid[global_mask] = np.arange(p)

        # Accumulators
        M_acc = np.zeros((p, p), dtype=np.float64)
        mX_acc = np.zeros(p, dtype=np.float64)
        mY_acc = np.zeros(p, dtype=np.float64)
        w_acc = np.zeros(p, dtype=np.float64)

        var_floor = 1e-12

        for (y0, x0), tile in zip(self.tile_corners, self.tiles):
            if tile is None:
                continue

            local_mask = tile.nanmask_flat.astype(bool)
            if not local_mask.any():
                continue

            # Global flat indices for this tile (column-major, same as fit)
            rows, cols = np.meshgrid(
                np.arange(y0, y0 + W),
                np.arange(x0, x0 + W),
                indexing="ij",
            )
            g_flat = (cols.ravel(order="F") * ny + rows.ravel(order="F"))

            # Map local valid → global valid
            local_valid_flat = g_flat[local_mask]
            g_idx = g_flat_to_valid[local_valid_flat]
            assert (g_idx >= 0).all(), "tile pixel not in global mask"

            # Precision weights per local valid pixel
            var = np.maximum(tile.K_post_diag, var_floor)
            w = 1.0 / var  # (p_tile_valid,)

            # Accumulate weighted operators using outer indexing
            ix = np.ix_(g_idx, g_idx)
            M_acc[ix] += w[:, None] * tile.M
            mX_acc[g_idx] += w * tile.mX
            mY_acc[g_idx] += w * tile.mY
            w_acc[g_idx] += w

        # Normalize
        safe = w_acc > 0
        inv_w = np.zeros_like(w_acc)
        inv_w[safe] = 1.0 / w_acc[safe]

        M_acc *= inv_w[:, None]      # row-wise divide
        mX_acc *= inv_w
        mY_acc *= inv_w

        self._M_cache = M_acc.astype(np.float64)
        self._mX_cache = mX_acc.astype(np.float64)
        self._mY_cache = mY_acc.astype(np.float64)

    @property
    def M(self) -> np.ndarray:
        """Effective global transfer matrix (lazy, cached)."""
        self._reconstruct()
        return self._M_cache

    @property
    def mX(self) -> np.ndarray:
        """Effective global input mean (lazy, cached)."""
        self._reconstruct()
        return self._mX_cache

    @property
    def mY(self) -> np.ndarray:
        """Effective global output mean (lazy, cached)."""
        self._reconstruct()
        return self._mY_cache


# ---------------------------------------------------------------------------
# Downscaler
# ---------------------------------------------------------------------------

class LocalEnsCGPDownscaler:
    """Tiled EnsCGP with Lagrangian / precision-weighted blending.

    Mirrors the public API of :class:`EnsCGPDownscaler`
    (``fit`` / ``predict`` / ``sample`` / ``save``) so that
    ``scripts/train.py`` can swap implementations transparently.
    """

    def __init__(
        self,
        window: int = 4,
        stride: int = 2,
        obs_noise_var: Optional[Union[float, str]] = "gcv",
        tag: str = "",
    ):
        if stride > window:
            raise ValueError(
                f"stride={stride} > window={window} would leave gaps"
            )
        self.window = int(window)
        self.stride = int(stride)
        self.obs_noise_var = obs_noise_var
        self.tag = tag
        self.model: Optional[LocalEnsCGPModel] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        nanmask_flat: Optional[np.ndarray] = None,
    ) -> "LocalEnsCGPDownscaler":
        """Fit one EnsCGP model per ``window x window`` HR tile.

        Args:
            input_data:  ``(N, H, W)`` bicubic-upscaled LR on the HR grid.
            target_data: ``(N, H, W)`` HR target.
            nanmask_flat: optional precomputed HR nanmask (flattened
                order='F').  Defaults to ``create_nanmask(target_data[0])``.
        """
        N, H, Wgrid = input_data.shape
        if target_data.shape != input_data.shape:
            raise ValueError(
                f"input {input_data.shape} and target {target_data.shape} "
                "must share shape (input must be upscaled to HR first)"
            )

        if nanmask_flat is None:
            nanmask_flat = create_nanmask(target_data[0]).flatten(order="F")
        nanmask_2d = nanmask_flat.reshape((H, Wgrid), order="F").astype(bool)

        corners = tile_indices((H, Wgrid), self.window, self.stride)

        tiles: List[EnsCGPModel] = []
        obs_vars: List[float] = []

        W = self.window
        for y0, x0 in corners:
            y1, x1 = y0 + W, x0 + W

            mask_tile = nanmask_2d[y0:y1, x0:x1]
            if not mask_tile.any():
                # Fully masked tile (over ocean edge etc.).  Skip.
                tiles.append(None)  # type: ignore[arg-type]
                continue

            # Flatten in column-major to stay consistent with the global
            # EnsCGP path (order='F' throughout the codebase).
            mask_tile_flat = mask_tile.flatten(order="F")

            # (N, W*W)  ->  (W*W, N)  ->  keep only valid rows
            X_tile = input_data[:, y0:y1, x0:x1].reshape(N, W * W, order="F").T
            Y_tile = target_data[:, y0:y1, x0:x1].reshape(N, W * W, order="F").T
            X_tile = X_tile[mask_tile_flat]
            Y_tile = Y_tile[mask_tile_flat]

            (M, mX, mY, A_Y, K_Y_diag, K_post_diag, U_post, obs_var) = (
                train_enscgp(X_tile, Y_tile, self.obs_noise_var)
            )

            tile_model = EnsCGPModel(
                M=M,
                mX=mX,
                mY=mY,
                A_Y=A_Y,
                K_Y_diag=K_Y_diag,
                K_post_diag=K_post_diag,
                U_post=U_post,
                nanmask_flat=mask_tile_flat.astype(np.int8),
                input_shape=(W, W),
                output_shape=(W, W),
                obs_noise_var=float(obs_var),
                n_ensemble=N,
                tag=f"{self.tag}_tile_{y0}_{x0}",
            )
            tiles.append(tile_model)
            obs_vars.append(float(obs_var))

        n_fit = sum(1 for t in tiles if t is not None)
        mean_obs = float(np.mean(obs_vars)) if obs_vars else 0.0

        self.model = LocalEnsCGPModel(
            tiles=tiles,
            tile_corners=corners,
            window=W,
            stride=self.stride,
            nanmask_flat=nanmask_flat.astype(np.int8),
            output_shape=(H, Wgrid),
            input_shape=input_data.shape[1:],
            n_ensemble=N,
            tag=self.tag,
            obs_noise_var_mean=mean_obs,
            obs_noise_var=mean_obs,
            n_tiles=n_fit,
        )

        print(
            f"✓ LocalEnsCGP trained (tag={self.tag or 'unnamed'}): "
            f"{n_fit}/{len(corners)} tiles, W={W}, stride={self.stride}, "
            f"<sigma_obs^2>={mean_obs:.6f}, ensemble={N}"
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_single(
        self,
        field: np.ndarray,
        return_uncertainty: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        H, Wgrid = field.shape
        if (H, Wgrid) != self.model.output_shape:
            raise ValueError(
                f"field shape {field.shape} != expected "
                f"{self.model.output_shape}"
            )

        W = self.model.window
        # Precision-weighted accumulation on the HR grid.
        num = np.zeros((H, Wgrid), dtype=np.float64)
        den = np.zeros((H, Wgrid), dtype=np.float64)

        # Variance floor: treats a zero-variance pixel (exact fit on the
        # training ensemble) as a very-high but finite precision so that
        # it dominates the blend but never produces inf/nan.
        var_floor = 1e-12

        for (y0, x0), tile_model in zip(
            self.model.tile_corners, self.model.tiles
        ):
            if tile_model is None:
                continue

            y1, x1 = y0 + W, x0 + W
            # Extract W*W input patch in the same column-major order
            # used during fit.
            patch_flat = field[y0:y1, x0:x1].reshape(W * W, order="F")

            mask_bool = tile_model.nanmask_flat.astype(bool)
            pred_flat = np.full(W * W, np.nan, dtype=np.float64)
            std_flat = np.full(W * W, np.nan, dtype=np.float64)

            assert tile_model.K_post_diag is not None
            pred_flat[mask_bool] = (
                tile_model.M @ (patch_flat[mask_bool] - tile_model.mX)
                + tile_model.mY
            )
            var_tile = np.maximum(tile_model.K_post_diag, var_floor)
            std_flat[mask_bool] = np.sqrt(var_tile)

            pred_tile = pred_flat.reshape((W, W), order="F")
            # Build variance patch by filling the flat buffer FIRST and then
            # reshaping; an order='F' reshape of a C-contiguous array returns
            # a copy, so in-place writes through the reshape view silently
            # dropped earlier (producing all-NaN predictions).
            var_flat_full = np.full(W * W, np.nan, dtype=np.float64)
            var_flat_full[mask_bool] = var_tile
            var_tile_2d = var_flat_full.reshape((W, W), order="F")

            valid = ~np.isnan(pred_tile) & ~np.isnan(var_tile_2d)
            weight = np.zeros_like(pred_tile)
            weight[valid] = 1.0 / var_tile_2d[valid]

            num[y0:y1, x0:x1] += weight * np.where(valid, pred_tile, 0.0)
            den[y0:y1, x0:x1] += weight

        # Final blend.
        valid_mask = self.model.nanmask_flat.reshape(
            (H, Wgrid), order="F"
        ).astype(bool)
        pred = np.full((H, Wgrid), np.nan, dtype=np.float32)
        pred[valid_mask & (den > 0)] = (
            num[valid_mask & (den > 0)] / den[valid_mask & (den > 0)]
        )
        pred[pred < 0] = 0

        if return_uncertainty:
            std = np.full((H, Wgrid), np.nan, dtype=np.float32)
            std[valid_mask & (den > 0)] = np.sqrt(
                1.0 / den[valid_mask & (den > 0)]
            )
            return pred, std
        return pred

    def predict(
        self,
        input_data: np.ndarray,
        return_uncertainty: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if input_data.ndim == 2:
            return self._predict_single(input_data, return_uncertainty)

        n = input_data.shape[0]
        H, Wgrid = self.model.output_shape
        out = np.full((n, H, Wgrid), np.nan, dtype=np.float32)
        if return_uncertainty:
            std = np.full((n, H, Wgrid), np.nan, dtype=np.float32)
            for i in range(n):
                out[i], std[i] = self._predict_single(input_data[i], True)
            return out, std
        for i in range(n):
            out[i] = self._predict_single(input_data[i], False)
        return out

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(
        self,
        input_data: np.ndarray,
        n_samples: int = 10,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior samples with precision-weighted stitching.

        Samples are drawn independently per tile and then blended on
        overlaps with the same precision weights used for the mean.
        Overlap equality holds in expectation and variance; individual
        draws may differ by O(sigma) at tile boundaries (see
        ``LocalEnsCGPModel`` docstring for the exact-constraint
        alternative).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if input_data.ndim != 2:
            raise ValueError("sample() expects a single HR field (2D)")

        rng = np.random.default_rng(seed)

        H, Wgrid = input_data.shape
        W = self.model.window

        samples = np.full((n_samples, H, Wgrid), np.nan, dtype=np.float32)
        for s in range(n_samples):
            num = np.zeros((H, Wgrid), dtype=np.float64)
            den = np.zeros((H, Wgrid), dtype=np.float64)
            var_floor = 1e-12

            for (y0, x0), tm in zip(
                self.model.tile_corners, self.model.tiles
            ):
                if tm is None:
                    continue
                y1, x1 = y0 + W, x0 + W
                mask_bool = tm.nanmask_flat.astype(bool)
                assert tm.K_post_diag is not None and tm.U_post is not None

                patch_flat = input_data[y0:y1, x0:x1].reshape(
                    W * W, order="F"
                )
                m_post = tm.M @ (patch_flat[mask_bool] - tm.mX) + tm.mY

                rank = tm.U_post.shape[1] if tm.U_post.size else 0
                if rank > 0:
                    z = rng.standard_normal(rank)
                    draw = m_post + tm.U_post @ z
                else:
                    draw = m_post

                draw_full = np.full(W * W, np.nan, dtype=np.float64)
                draw_full[mask_bool] = draw
                draw_2d = draw_full.reshape((W, W), order="F")

                var_tile = np.maximum(tm.K_post_diag, var_floor)
                var_full = np.full(W * W, np.nan, dtype=np.float64)
                var_full[mask_bool] = var_tile
                var_2d = var_full.reshape((W, W), order="F")

                valid = ~np.isnan(draw_2d) & ~np.isnan(var_2d)
                w = np.zeros_like(draw_2d)
                w[valid] = 1.0 / var_2d[valid]
                num[y0:y1, x0:x1] += w * np.where(valid, draw_2d, 0.0)
                den[y0:y1, x0:x1] += w

            valid_mask = self.model.nanmask_flat.reshape(
                (H, Wgrid), order="F"
            ).astype(bool)
            out = np.full((H, Wgrid), np.nan, dtype=np.float32)
            ok = valid_mask & (den > 0)
            out[ok] = num[ok] / den[ok]
            out[out < 0] = 0
            samples[s] = out

        return samples

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: pathlib.Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✓ Saved LocalEnsCGP model to: {path}")

    @classmethod
    def load(cls, path: pathlib.Path) -> "LocalEnsCGPDownscaler":
        with open(path, "rb") as f:
            model: LocalEnsCGPModel = pickle.load(f)
        obj = cls(window=model.window, stride=model.stride, tag=model.tag)
        obj.model = model
        print(f"✓ Loaded LocalEnsCGP model from: {path}")
        return obj
