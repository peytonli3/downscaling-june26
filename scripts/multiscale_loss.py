"""Multi-scale, displacement-tolerant loss for the mean prediction (mu_u, mu_v) of
new_enscgp_swin.py's ProbabilisticSwin2SR.

Motivation: a per-band error diagnostic (extra_scripts/swin/oneoff/band_error_diagnostics.py) found
the dominant error against WRF truth is RANDOM SPATIAL DISPLACEMENT at every scale (not
amplitude), worst at 8-32 km. Pointwise losses (L1, etc.) reward smoothing instead of
correcting displacement, since shifting a sharp structure by a few pixels costs as much
pointwise loss as blurring it away. MultiscaleLoss replaces that legacy mean-loss stack
with a Laplacian-pyramid decomposition + a displacement-tolerant per-band metric.

Pipeline
--------
1. Laplacian pyramid (laplacian_bands, n_levels default 5): non-decimated, full-resolution
   bands via cascaded-from-the-original-field Gaussian blurs (sigma doubling per level:
   base_sigma * 2^(k-1)), differences between successive blurs; the coarsest level is the
   low-pass residual. Bands sum back to the field exactly (asserted in self_test below).
   Pure torch (depthwise separable conv2d, reflect-padded), differentiable, GPU-resident --
   ported from extra_scripts/swin/oneoff/band_error_diagnostics.py's numpy/scipy version.

2. Per band, per component (u, v) -- summed over both:
     band_loss = METRIC_SCALE[metric] * D_band(pred_band, truth_band) / (sigma_band[band, component] + eps)
   D_band is configurable per band via metric_per_band (default
   ["l1","sw","sw","sw","sw"], coarse -> fine): "l1" on the coarsest band (it's already
   ~correct per the diagnostic, no need for OT there), sliced Wasserstein on the active
   bands where displacement dominates.
   sigma_band = RMS of TRUTH per band per component, precomputed ONCE over a sample of the
   training set and cached to disk (see precompute_sigma_band/load_or_compute_sigma_band)
   -- never recomputed per-batch. sigma_floor keeps the (small) finest band's sigma from
   amplifying grid noise into a huge loss weight. sigma_band calibrates ACROSS BANDS (so
   each band's natural energy scale doesn't bias how much it matters).

   METRIC_SCALE ({"l1": 1.0, "sw": 2.75}, a fixed module constant, not a config knob)
   calibrates ACROSS METRIC TYPES instead: sliced-Wasserstein and L1 are computed
   differently and read at different absolute scales even on identical data (empirically
   L1 ~2-4x larger than sliced-W on this dataset's bands -- displacement-dominated errors
   genuinely cost less under sliced-W than under L1, which is the whole point, but that
   means an "sw" band's raw contribution is *systematically* smaller than an "l1" band's
   purely from the choice of metric, not from that band actually mattering less).
   METRIC_SCALE corrects that systematic bias (one global multiplier per metric, not per
   band -- sigma_band already handles per-band calibration) WITHOUT erasing the
   displacement-tolerance effect itself: 2.75 was measured as the average l1-vs-sw ratio on
   real data (via a one-off diagnostic comparing both metrics on the same pred/truth
   bands), and a genuinely-displaced (not actually wrong-amplitude) band still reads
   smaller under calibrated-sw than under l1 -- it's just no longer smaller for the wrong
   (purely-units) reason on top of that. Re-measure and update this constant if the data
   distribution shifts enough to change the ratio.

3. Sliced Wasserstein, spatial-but-displacement-tolerant (sliced_wasserstein_patches): the
   standard image "Sliced Wasserstein Distance" recipe (Karras et al., used as a GAN
   texture-similarity metric) -- extract overlapping patches (patch_size, stride), flatten
   each to a vector, project onto n_projections random directions, and for each direction
   compute the 1D Wasserstein-1 distance between the two (equal-size) sets of projected
   scalars via sort + L1 (the exact closed-form 1D OT distance between empirical
   distributions of equal size). Averaged over directions. NOT exact 2D OT, NOT Sinkhorn --
   hand-implemented (sort + random projections only, no new dependency). Patches make it
   "spatial" (local gradient/edge structure inside a patch matters) yet shift-tolerant: a
   pure translation barely changes the *set* of patch content away from boundaries, so this
   metric stays small under a shift that a pointwise L1 would score as large (verified
   below).

4. (Optional, off by default) histogram_wasserstein: 1D Wasserstein between the pred/truth
   PIXEL VALUE distributions of a band (sort the flattened band + L1) -- spatially blind
   (no patches, no position), unlike (3). Useful as an auxiliary heavy-tail term; kept
   separate from the main per-band metric.

Tuning guardrail (read before raising ms_weight in new_enscgp_swin_config.json; post-0714,
uncertainty is trained by pinball on q10/q90, not NLL -- this only concerns q50):
raise ms_weight GRADUALLY from 0. After each bump, check:
  - the eigenspectrum (compare_eigenspectra.py) should rise toward WRF, not
    away from it.
  - coverage / the PIT histogram (eval_quantile_calibration.py) shouldn't move much --
    ms_weight only supervises q50, so a shift there suggests q50 and the offset head are
    fighting (e.g. via the shared backbone), not that ms_weight itself is miscalibrated.

Self-test (no GPU/checkpoint/training data required for the synthetic parts):
    python multiscale_loss.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quantile_metrics import rank_cdf

DEFAULT_METRIC_PER_BAND = ("l1", "sw", "sw", "sw", "sw")  # coarse -> fine, len == n_levels


def to_fine_first(per_band):
    """Reverse a COARSE -> FINE per-band config into the FINEST -> COARSEST order the band
    decompositions (laplacian_bands / fft_gaussian_bands) actually produce.

    Both band losses take their per-band settings coarse -> fine, because that is how a
    human reads "coarsest band first" in a config; both iterate finest -> coarsest, because
    that is the order the filter banks emit. Every such config must cross that boundary
    exactly once, here -- the two classes used to each do it their own way (one by index
    arithmetic at call time, one by `reversed()` at construction), which is precisely the
    setup for an off-by-one that silently mislabels which band got which setting.
    """
    return tuple(reversed(tuple(per_band)))


# --------------------------------------------------------------------------------------
# Differentiable Laplacian pyramid (torch)
# --------------------------------------------------------------------------------------
def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable, reflect-padded, depthwise Gaussian blur. x: (B, C, H, W). sigma<=0 is a
    no-op (matches the original/finest pyramid level)."""
    if sigma <= 0:
        return x
    B, C, H, W = x.shape
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    radius = (k.shape[0] - 1) // 2
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, ky, groups=C)
    return x


def band_sigmas(n_levels: int, base_sigma: float) -> list[float]:
    """Cumulative blur sigma (pixels) for each pyramid level; sigmas[0]=0 (original)."""
    return [0.0] + [base_sigma * (2 ** (k - 1)) for k in range(1, n_levels)]


def laplacian_bands(field: torch.Tensor, n_levels: int, base_sigma: float) -> list[torch.Tensor]:
    """field: (B, C, H, W). Returns n_levels tensors (B, C, H, W), finest -> coarsest
    (the last is the low-pass residual). Each level's blur is applied directly to the
    ORIGINAL field at its cumulative sigma (not cascaded level-to-level), so bands sum
    back to the field exactly regardless of n_levels/base_sigma."""
    sigmas = band_sigmas(n_levels, base_sigma)
    blurred = [field] + [gaussian_blur(field, s) for s in sigmas[1:]]
    bands = [blurred[j] - blurred[j + 1] for j in range(n_levels - 1)]
    bands.append(blurred[-1])
    return bands


# --------------------------------------------------------------------------------------
# Per-band metrics
# --------------------------------------------------------------------------------------
def sliced_wasserstein_patches(pred: torch.Tensor, truth: torch.Tensor, patch_size: int = 8,
                                stride: int = 4, n_projections: int = 64) -> torch.Tensor:
    """pred/truth: (B, 1, H, W), one band/component. Patch-vector sliced-Wasserstein (see
    module docstring point 3). Differentiable through torch.sort."""
    pred_patches = F.unfold(pred, kernel_size=patch_size, stride=stride)    # (B, P, N)
    truth_patches = F.unfold(truth, kernel_size=patch_size, stride=stride)  # P = patch_size**2
    P = pred_patches.shape[1]
    pred_vecs = pred_patches.permute(0, 2, 1).reshape(-1, P)    # (B*N, P)
    truth_vecs = truth_patches.permute(0, 2, 1).reshape(-1, P)
    directions = torch.randn(P, n_projections, device=pred.device, dtype=pred.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    pred_proj = pred_vecs @ directions      # (B*N, n_projections)
    truth_proj = truth_vecs @ directions
    pred_sorted, _ = torch.sort(pred_proj, dim=0)
    truth_sorted, _ = torch.sort(truth_proj, dim=0)
    return (pred_sorted - truth_sorted).abs().mean()


def histogram_wasserstein(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """pred/truth: (B, 1, H, W). Spatially-blind 1D Wasserstein between the pooled (over
    B,H,W) pixel-value distributions: sort + L1. Same size on both sides (same B,H,W), so
    this is the exact 1D OT distance between the two empirical distributions."""
    p_sorted, _ = torch.sort(pred.reshape(-1))
    t_sorted, _ = torch.sort(truth.reshape(-1))
    return (p_sorted - t_sorted).abs().mean()


# --------------------------------------------------------------------------------------
# sigma_band: precompute once over the training set, cache to disk
# --------------------------------------------------------------------------------------
def precompute_sigma_band(wrf_path: Path, indices: np.ndarray, n_levels: int, base_sigma: float,
                           device: str = "cpu", max_samples: int | None = 2000,
                           chunk_size: int = 32, seed: int = 0) -> torch.Tensor:
    """RMS of WRF truth [u, v] per band per component, over (a subsample of) `indices`.
    Returns (n_levels, 2). Call once (e.g. from train_new_enscgp_swin.py at startup when
    ms_weight > 0) and cache via load_or_compute_sigma_band -- NOT per-batch."""
    wrf = np.load(wrf_path, mmap_mode="r")
    idx = np.asarray(indices)
    if max_samples is not None and len(idx) > max_samples:
        idx = np.random.default_rng(seed).choice(idx, size=max_samples, replace=False)
    idx = np.sort(idx)

    sum_sq = torch.zeros(n_levels, 2, dtype=torch.float64)
    count = torch.zeros(n_levels, 2, dtype=torch.float64)
    for start in range(0, len(idx), chunk_size):
        chunk = idx[start:start + chunk_size]
        field = torch.from_numpy(np.array(wrf[chunk, :2], dtype=np.float32)).to(device)  # (b,2,H,W)
        bands = laplacian_bands(field, n_levels, base_sigma)
        for b, band in enumerate(bands):
            sum_sq[b] += (band.double() ** 2).sum(dim=(0, 2, 3)).cpu()
            count[b] += band.shape[0] * band.shape[2] * band.shape[3]
    return torch.sqrt(sum_sq / count).float()  # (n_levels, 2)


def sigma_band_cache_path(data_dir: Path, n_levels: int, base_sigma: float) -> Path:
    return Path(data_dir) / f"sigma_band_cache_L{n_levels}_s{base_sigma:g}.npy"


def load_or_compute_sigma_band(data_dir: Path, wrf_path: Path, train_idx: np.ndarray, n_levels: int,
                                base_sigma: float, device: str = "cpu", max_samples: int | None = 2000,
                                force_recompute: bool = False) -> torch.Tensor:
    path = sigma_band_cache_path(data_dir, n_levels, base_sigma)
    if path.exists() and not force_recompute:
        return torch.from_numpy(np.load(path)).float()
    sigma = precompute_sigma_band(wrf_path, train_idx, n_levels, base_sigma, device, max_samples)
    np.save(path, sigma.numpy())
    return sigma


# --------------------------------------------------------------------------------------
# MultiscaleLoss
# --------------------------------------------------------------------------------------
# Global per-metric-type multiplier (NOT per-band -- sigma_band already handles per-band
# calibration; this corrects the systematic scale gap BETWEEN metric types instead). A
# fixed module constant, not a config knob: 2.75 was measured as the average l1-vs-sw
# ratio on real data -- see module docstring point 2. Re-measure and update in code if the
# data distribution shifts enough to change the ratio.
METRIC_SCALE = {"l1": 1.0, "sw": 2.75}


class _BandLoss:
    """Shared skeleton for the two per-band losses below.

    Both decompose pred/truth into `n_levels` bands, score each band per component (u, v)
    against the truth, and normalize each score by that band+component's `sigma_band` before
    summing -- only the DECOMPOSITION and the per-band SCORE differ. Subclasses supply those
    two as `_iter_bands` and `_band_component_term`; the validation, the sigma flooring, and
    the accumulation loop live here so they cannot drift apart.
    """

    def __init__(self, sigma_band: torch.Tensor, n_levels: int, base_sigma: float,
                 sigma_floor: float = 1e-3, eps: float = 1e-8):
        if sigma_band.shape != (n_levels, 2):
            raise ValueError(f"sigma_band must have shape ({n_levels}, 2), got {tuple(sigma_band.shape)}")
        self.n_levels = n_levels
        self.base_sigma = base_sigma
        self.eps = eps
        # Floored ONCE at construction: a degenerate (near-zero) band sigma would otherwise
        # turn into a huge loss weight amplifying that band's grid noise.
        self.sigma_band = sigma_band.clamp_min(sigma_floor)  # (n_levels, 2)

    def _check_per_band(self, name: str, value) -> None:
        """Validate a per-band config sequence's length (coarse -> fine, len == n_levels)."""
        if value is not None and len(value) != self.n_levels:
            raise ValueError(f"{name} must have length n_levels={self.n_levels}, got {len(value)}")

    def _iter_bands(self, pred: torch.Tensor, truth: torch.Tensor):
        """Yield (band_index, pred_band, truth_band), FINEST -> COARSEST. A subclass may skip
        a band entirely (rather than scoring it and multiplying by zero) to save its
        decomposition cost."""
        raise NotImplementedError

    def _band_component_term(self, b: int, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Scalar score for band `b`, ONE component. p/t are (B, 1, H, W). Not yet divided
        by sigma_band -- the caller does that."""
        raise NotImplementedError

    def __call__(self, pred_mean: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """pred_mean/truth: (B, 2, H, W) [u, v] -> scalar."""
        sigma = self.sigma_band.to(pred_mean.device)
        total = pred_mean.new_zeros(())
        for b, pred_band, truth_band in self._iter_bands(pred_mean, truth):
            for c in range(2):
                p = pred_band[:, c:c + 1]
                t = truth_band[:, c:c + 1]
                total = total + self._band_component_term(b, p, t) / (sigma[b, c] + self.eps)
        return total


class MultiscaleLoss(_BandLoss):
    """Callable: __call__(pred_mean, truth) -> scalar. pred_mean/truth: (B, 2, H, W) [u, v].
    See module docstring for the per-band/per-component formula. Independently
    toggleable/bisectable from the rest of the loss via ms_weight in the caller."""

    def __init__(self, sigma_band: torch.Tensor, n_levels: int = 5, base_sigma: float = 1.0,
                 metric_per_band: tuple[str, ...] = DEFAULT_METRIC_PER_BAND,
                 patch_size: int = 8, patch_stride: int = 4, n_projections: int = 64,
                 sigma_floor: float = 1e-3, eps: float = 1e-8,
                 histogram_weight: float = 0.0):
        super().__init__(sigma_band, n_levels, base_sigma, sigma_floor, eps)
        self._check_per_band("metric_per_band", metric_per_band)
        for m in metric_per_band:
            if m not in ("l1", "sw"):
                raise ValueError(f"metric_per_band entries must be 'l1' or 'sw', got {m!r}")
        # Kept coarse -> fine, as given and as logged by the caller; `_metric_fine_first` is
        # the band-order copy the loop actually indexes (see to_fine_first).
        self.metric_per_band = tuple(metric_per_band)
        self._metric_fine_first = to_fine_first(metric_per_band)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_projections = n_projections
        self.histogram_weight = histogram_weight

    def _band_metric(self, metric: str, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if metric == "l1":
            d = F.l1_loss(p, t)
        else:
            d = sliced_wasserstein_patches(p, t, self.patch_size, self.patch_stride, self.n_projections)
        return METRIC_SCALE[metric] * d

    def _iter_bands(self, pred: torch.Tensor, truth: torch.Tensor):
        pred_bands = laplacian_bands(pred, self.n_levels, self.base_sigma)    # finest -> coarsest
        truth_bands = laplacian_bands(truth, self.n_levels, self.base_sigma)
        yield from zip(range(self.n_levels), pred_bands, truth_bands)

    def _band_component_term(self, b: int, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        term = self._band_metric(self._metric_fine_first[b], p, t)
        if self.histogram_weight > 0:
            term = term + self.histogram_weight * histogram_wasserstein(p, t)
        return term


# --------------------------------------------------------------------------------------
# FFT Gaussian filter bank
# --------------------------------------------------------------------------------------
def _fft_gaussian_lp_masks(H: int, W: int, sigmas: list[float],
                             device, dtype) -> list[torch.Tensor]:
    """Low-pass Gaussian transfer functions G(σ,r) = exp(-2π²σ²r²) for each sigma,
    in rfft2 frequency layout.  sigmas[0]=0 returns all-ones (identity / no blur)."""
    fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype)
    r2 = fy[:, None] ** 2 + fx[None, :] ** 2   # (H, W//2+1)
    two_pi2 = 2.0 * math.pi ** 2
    return [
        torch.exp(-two_pi2 * (s ** 2) * r2) if s > 0 else torch.ones_like(r2)
        for s in sigmas
    ]


def fft_band_masks(H: int, W: int, n_levels: int, base_sigma: float,
                   device, dtype) -> list[torch.Tensor]:
    """Band-pass transfer functions, finest -> coarsest, as (1, 1, H, W//2+1) tensors ready
    to multiply an rfft2.

    Successive differences of the low-pass masks, with the coarsest low-pass kept whole, so
    the bands telescope back to the identity -- i.e. they sum to the field exactly (asserted
    in self_test). The single definition behind both `fft_gaussian_bands` and
    `FreqBandLoss._get_masks`; they used to build this list separately.
    """
    sigmas = band_sigmas(n_levels, base_sigma)
    lp = _fft_gaussian_lp_masks(H, W, sigmas, device, dtype)
    masks = [lp[j] - lp[j + 1] for j in range(n_levels - 1)]
    masks.append(lp[-1])
    return [m.unsqueeze(0).unsqueeze(0) for m in masks]


def fft_gaussian_bands(field: torch.Tensor, n_levels: int, base_sigma: float) -> list[torch.Tensor]:
    """FFT Gaussian filter bank with the same sigma sequence as laplacian_bands.
    field: (B, C, H, W). Returns n_levels tensors (B, C, H, W), finest -> coarsest.
    Bands telescope exactly: sum(fft_gaussian_bands(field, ...)) == field."""
    H, W = field.shape[-2], field.shape[-1]
    masks = fft_band_masks(H, W, n_levels, base_sigma, field.device, field.dtype)
    F_field = torch.fft.rfft2(field)
    return [torch.fft.irfft2(m * F_field, s=(H, W)) for m in masks]


# --------------------------------------------------------------------------------------
# FreqBandLoss
# --------------------------------------------------------------------------------------
class FreqBandLoss(_BandLoss):
    """Per-band pointwise weighted L1 loss using an FFT Gaussian filter bank.

    Bands use the same sigma sequence as laplacian_bands (band_sigmas), so sigma_band
    can be shared directly with MultiscaleLoss (same cache file).  Pixel weights come
    from the per-sample CDF of |truth_band| magnitudes; gradient does not flow through
    the weights.

    cdf_weight_mode='down_extremes' (default): weight = 1 - F(|m|), down-weights the
      largest-magnitude pixels so the loss focuses on mid-range structure.
    cdf_weight_mode='up_extremes': weight = F(|m|), opposite.

    band_weight (default None, i.e. all 1.0 -- backward-compatible no-op): optional per-band
      multiplier, COARSE -> FINE like metric_per_band (see DEFAULT_METRIC_PER_BAND), len ==
      n_levels. A 0.0 entry fully deactivates that band (skipped, not just zero-weighted --
      no irfft2/pixel-weight cost for it). Motivating case: the finest band is the raw-grid
      Laplacian residual, which on this data is dominated by unresolvable grid noise rather
      than real structure (see multiscale_loss.py's module docstring point 2) -- down-
      weighting or zeroing band_weight[-1] stops the loss from chasing that noise.

    pixel_weight_enabled (default None, i.e. all True -- backward-compatible no-op): optional
      per-band bool, COARSE -> FINE like band_weight, len == n_levels. False makes that band's
      L1 UNWEIGHTED (pw=1 for every pixel) instead of applying the down_extremes/up_extremes
      CDF weight -- independent of band_weight, which scales the whole band's contribution
      rather than changing how pixels within it are weighted. Motivating evidence
      (extra_scripts/swin/band_crossover_diagnostic.py's Part 1 weighted-vs-unweighted
      breakdown): at some bands, bicubic only beats SWIN's freq_l1 because of the down_extremes
      weighting -- the gap shrinks to noise (or reverses) once that band is scored unweighted,
      while swd/melr (which don't use this weighting at all) already favor SWIN there. That
      band's "SWIN loses" signal is a weighting artifact, not a real placement deficit -- so
      the fix is disabling the weighting AT THAT BAND specifically, not discarding the band via
      band_weight (which would also discard the real, correctly-supervising L1 gradient there).
      At other bands, the same diagnostic found the weighted-vs-unweighted gap does NOT vanish
      -- a real, unavoidable pointwise-L1 effect survives regardless of weighting -- so
      band_weight (not this) is the right lever there.

    Set freq_weight=0.0 in the training config for a zero-cost no-op (default)."""

    def __init__(self, sigma_band: torch.Tensor, n_levels: int = 5, base_sigma: float = 2.0,
                 cdf_weight_mode: str = "down_extremes", sigma_floor: float = 1e-3,
                 eps: float = 1e-8, band_weight: tuple[float, ...] | None = None,
                 pixel_weight_enabled: tuple[bool, ...] | None = None):
        super().__init__(sigma_band, n_levels, base_sigma, sigma_floor, eps)
        if cdf_weight_mode not in ("down_extremes", "up_extremes"):
            raise ValueError(f"cdf_weight_mode must be 'down_extremes' or 'up_extremes', got {cdf_weight_mode!r}")
        self._check_per_band("band_weight", band_weight)
        self._check_per_band("pixel_weight_enabled", pixel_weight_enabled)
        self.cdf_weight_mode = cdf_weight_mode
        # Both stored FINEST -> COARSEST -- the order _iter_bands yields -- from the
        # constructor's coarse -> fine input. See to_fine_first.
        self.band_weight = to_fine_first(band_weight) if band_weight is not None else (1.0,) * n_levels
        self.pixel_weight_enabled = (to_fine_first(pixel_weight_enabled) if pixel_weight_enabled is not None
                                     else (True,) * n_levels)
        self._mask_cache: dict = {}

    def _get_masks(self, H: int, W: int, device, dtype) -> list[torch.Tensor]:
        key = (H, W, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = fft_band_masks(H, W, self.n_levels, self.base_sigma, device, dtype)
        return self._mask_cache[key]

    def _pixel_weights(self, band_mag: torch.Tensor) -> torch.Tensor:
        """band_mag: (B, 1, H, W) → (B, 1, H, W) weights normalized to mean=1 per sample.

        n_leading=1: each SAMPLE is ranked against itself, so one unusually energetic sample
        cannot flatten the rest's weights."""
        cdf = rank_cdf(band_mag, n_leading=1)
        w = (1.0 - cdf) if self.cdf_weight_mode == "down_extremes" else cdf
        return w / w.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)

    def _iter_bands(self, pred: torch.Tensor, truth: torch.Tensor):
        H, W = pred.shape[-2], pred.shape[-1]
        masks = self._get_masks(H, W, pred.device, pred.dtype)
        pred_F = torch.fft.rfft2(pred)
        truth_F = torch.fft.rfft2(truth)
        for b, mask in enumerate(masks):
            if self.band_weight[b] == 0.0:
                continue  # fully deactivated: skip this band's irfft2/pixel-weight cost too
            yield (b,
                   torch.fft.irfft2(mask * pred_F, s=(H, W)),
                   torch.fft.irfft2(mask * truth_F, s=(H, W)))

    def _band_component_term(self, b: int, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        abs_err = (p - t).abs()
        if self.pixel_weight_enabled[b]:
            pw = self._pixel_weights(t.abs().detach())
            band_term = (pw * abs_err).mean()
        else:
            band_term = abs_err.mean()  # unweighted: skip the rank/CDF cost entirely
        return self.band_weight[b] * band_term


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------
def _assert_bands_sum_to_field(n_levels: int, base_sigma: float) -> None:
    field = torch.randn(2, 2, 200, 200)
    bands = laplacian_bands(field, n_levels, base_sigma)
    recon = sum(bands)
    diff = (recon - field).abs().max().item()
    assert diff < 1e-4, f"Laplacian bands do not sum to the field, max abs diff {diff}"
    print(f"  Laplacian bands sum to field: max abs diff {diff:.2e} (n_levels={n_levels}) OK")


def _assert_sliced_w_shift_tolerant() -> None:
    """A pure (wraparound) shift barely changes patch-SW (the bag of local patch content
    is ~unchanged away from boundaries) but is large under pixelwise L1 -- this is the
    displacement-tolerance property the whole multiscale_loss design rests on."""
    torch.manual_seed(0)
    field = gaussian_blur(torch.randn(1, 1, 200, 200), sigma=3.0)
    shifted = torch.roll(field, shifts=(5, 7), dims=(2, 3))
    l1 = F.l1_loss(shifted, field).item()
    sw = sliced_wasserstein_patches(shifted, field, patch_size=8, stride=4, n_projections=64).item()
    # sigma_band-style normalization scale, for a sane side-by-side comparison
    scale = field.std().item()
    assert sw < 0.15 * l1, f"shifted: expected SW << L1 (displacement-tolerant), got SW={sw:.4f} L1={l1:.4f}"
    print(f"  Shifted-field test: L1={l1:.4f} (large), patch-SW={sw:.4f} (small, "
          f"{sw / l1:.1%} of L1) -- displacement-tolerance confirmed (field std={scale:.3f})")


def _assert_histogram_w_permutation_invariant() -> None:
    """A full random permutation of pixel values leaves the value distribution unchanged,
    so histogram_wasserstein ~ 0 exactly -- the 'spatially blind' property from the module
    docstring. (patch-SW is NOT asserted here: with random isotropic projections on a
    smooth/correlated patch, most of a patch's energy lives in a low-dim subspace, so most
    random directions miss it and under-detect scrambling -- a known property of vanilla
    sliced-W, not a bug. It's reported for visibility only.)"""
    torch.manual_seed(1)
    field = gaussian_blur(torch.randn(1, 1, 200, 200), sigma=3.0)
    perm = torch.randperm(field.numel())
    permuted = field.reshape(-1)[perm].reshape(field.shape)
    hw = histogram_wasserstein(permuted, field).item()
    sw = sliced_wasserstein_patches(permuted, field, patch_size=8, stride=4, n_projections=64).item()
    assert hw < 1e-5, f"permuted: expected histogram_wasserstein ~ 0, got {hw}"
    print(f"  Permuted-field test: histogram_W={hw:.2e} (~0, spatially blind) OK "
          f"[patch-SW={sw:.4f}, informational only]")


def _assert_sigma_band_floor() -> None:
    """A degenerate (zero) sigma_band must be floored, not divide-by-zero."""
    zero_sigma = torch.zeros(5, 2)
    loss = MultiscaleLoss(sigma_band=zero_sigma, n_levels=5, sigma_floor=1e-3)
    assert torch.all(loss.sigma_band >= 1e-3), "sigma_band floor was not applied"
    print(f"  sigma_band floor: zero input -> floored to {loss.sigma_band[0, 0].item():.1e} OK")


def _assert_fft_bands_sum_to_field(n_levels: int, base_sigma: float) -> None:
    field = torch.randn(2, 2, 200, 200)
    bands = fft_gaussian_bands(field, n_levels, base_sigma)
    recon = sum(bands)
    diff = (recon - field).abs().max().item()
    assert diff < 1e-4, f"FFT bands do not sum to the field, max abs diff {diff:.2e}"
    print(f"  FFT bands sum to field: max abs diff {diff:.2e} (n_levels={n_levels}) OK")


def _assert_freq_band_pixel_weights() -> None:
    torch.manual_seed(42)
    loss_fn = FreqBandLoss(sigma_band=torch.ones(5, 2))
    band_mag = torch.randn(3, 1, 64, 64).abs()
    w = loss_fn._pixel_weights(band_mag)
    for b in range(3):
        mean_w = w[b].mean().item()
        assert abs(mean_w - 1.0) < 1e-5, f"pixel_weights mean={mean_w:.6f} (expected 1.0)"
    flat_mag = band_mag[0, 0].reshape(-1)
    flat_w = w[0, 0].reshape(-1)
    top_idx = flat_mag.topk(100).indices
    bot_idx = flat_mag.topk(100, largest=False).indices
    assert flat_w[top_idx].mean() < flat_w[bot_idx].mean(), \
        "down_extremes: top-magnitude pixels should have lower weight than bottom-magnitude"
    print("  pixel_weights: mean=1.0 per sample OK; down_extremes direction OK")


def _assert_per_band_config_alignment() -> None:
    """Per-band config is given COARSE -> FINE but consumed FINEST -> COARSEST. Both classes
    cross that boundary via to_fine_first; this pins down that they cross it exactly once
    and in the same direction, since an off-by-one here silently applies every band's
    setting to the wrong band instead of raising."""
    ms = MultiscaleLoss(sigma_band=torch.ones(5, 2), metric_per_band=("l1", "sw", "sw", "sw", "sw"))
    assert ms.metric_per_band == ("l1", "sw", "sw", "sw", "sw"), "public attr must stay coarse -> fine"
    assert ms._metric_fine_first == ("sw", "sw", "sw", "sw", "l1"), \
        f"coarsest band should get 'l1'; got {ms._metric_fine_first}"

    fb = FreqBandLoss(sigma_band=torch.ones(5, 2), band_weight=(1, 2, 3, 4, 5),
                      pixel_weight_enabled=(True, False, True, False, True))
    assert fb.band_weight == (5, 4, 3, 2, 1), f"band_weight not reversed to fine-first: {fb.band_weight}"
    assert fb.pixel_weight_enabled == (True, False, True, False, True)[::-1]

    # Behavioural: a constant offset is pure DC, so it lands ENTIRELY in the coarsest
    # (low-pass residual) band. Activating only the coarsest band must therefore see it, and
    # deactivating only the coarsest band must not.
    truth = torch.randn(2, 2, 64, 64)
    pred = truth + 3.0
    coarsest_only = FreqBandLoss(sigma_band=torch.ones(5, 2), band_weight=(1, 0, 0, 0, 0))(pred, truth)
    coarsest_off = FreqBandLoss(sigma_band=torch.ones(5, 2), band_weight=(0, 1, 1, 1, 1))(pred, truth)
    assert coarsest_only.item() > 1.0, \
        f"a DC offset must show up in the coarsest band, got {coarsest_only.item():.2e} (band order flipped?)"
    assert coarsest_off.item() < 1e-3, \
        f"a DC offset must NOT show up in the finer bands, got {coarsest_off.item():.2e} (band order flipped?)"
    print(f"  per-band config alignment: coarse->fine in, fine-first out; DC offset lands in the "
          f"coarsest band only (on {coarsest_only.item():.3f} / off {coarsest_off.item():.2e}) OK")


def _assert_freq_band_zero_loss() -> None:
    loss_fn = FreqBandLoss(sigma_band=torch.ones(5, 2))
    field = torch.randn(2, 2, 64, 64)
    loss = loss_fn(field, field).item()
    assert loss < 1e-5, f"FreqBandLoss(pred=truth) should be ~0, got {loss:.4e}"
    print(f"  FreqBandLoss(pred=truth)={loss:.2e} (~0) OK")


def _assert_freq_band_fwd_bwd() -> None:
    torch.manual_seed(0)
    loss_fn = FreqBandLoss(sigma_band=torch.ones(5, 2), n_levels=5, base_sigma=2.0)
    pred = torch.randn(2, 2, 64, 64, requires_grad=True)
    truth = torch.randn(2, 2, 64, 64)
    loss = loss_fn(pred, truth)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().max().item() > 0
    print(f"  FreqBandLoss fwd+bwd: loss={loss.item():.4f}, gradients flow OK")


def _assert_gate_fwd_bwd() -> None:
    """One real fwd+bwd against the live model: multiscale_loss on q50 + pinball on
    q10/q90, batch=4, gradients reach the backbone and BOTH heads.

    This mirrors how train_new_enscgp_swin.py actually wires the loss: the structural
    term supervises q50 only (reaching the backbone through mean_head/mean_gate), and
    pinball supervises q10/q90 only (reaching it through offset_head). Checking both
    paths in one backward is the point -- a structural-only loss would still pass
    while offset_head sat dead.

    Imports the actual model -- run from scripts/ so new_enscgp_swin's relative
    imports (network_swin2sr, terrain_encoder) resolve.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from new_enscgp_swin import ProbabilisticSwin2SR

    torch.manual_seed(0)
    model = ProbabilisticSwin2SR(img_size=200, embed_dim=24, depths=(2,), num_heads=(4,))
    model.train()
    B = 4
    posterior = torch.randn(B, 5, 200, 200)
    posterior[:, 2] = F.softplus(posterior[:, 2])  # L11 > 0
    posterior[:, 4] = F.softplus(posterior[:, 4])  # L22 > 0
    bicubic = torch.randn(B, 2, 200, 200)
    wrf = torch.randn(B, 2, 200, 200)
    terrain_raw = torch.randn(1, 4, 1000, 1000)

    sigma_band = torch.ones(5, 2)  # synthetic, just needs the right shape for this check
    ms_loss_fn = MultiscaleLoss(sigma_band=sigma_band, n_levels=5)

    pred = model(posterior, bicubic, terrain_raw)
    q10 = pred[:, ProbabilisticSwin2SR.Q10_SLICE]
    q50 = pred[:, ProbabilisticSwin2SR.Q50_SLICE]
    q90 = pred[:, ProbabilisticSwin2SR.Q90_SLICE]

    def pinball(q: torch.Tensor, tau: float) -> torch.Tensor:
        err = wrf - q
        return torch.maximum(tau * err, (tau - 1.0) * err).mean()

    ms = ms_loss_fn(q50, wrf)
    pin = pinball(q90, 0.9) + pinball(q10, 0.1)
    total = ms + pin

    model.zero_grad()
    total.backward()
    assert model.conv_first.weight.grad is not None and model.conv_first.weight.grad.abs().max().item() > 0, \
        "backbone received no gradient"
    assert model.mean_gate.grad is not None and model.mean_gate.grad.abs().item() > 0, \
        "mean_gate received no gradient (structural loss is not reaching q50)"
    offset_w = model.offset_head[-1].weight
    assert offset_w.grad is not None and offset_w.grad.abs().max().item() > 0, \
        "offset_head received no gradient (pinball is not reaching q10/q90)"
    print(f"  Fwd+bwd (batch={B}): multiscale_loss={ms.item():.4f}, pinball={pin.item():.4f}, "
          f"total={total.item():.4f}, gradients flow to backbone + mean_gate + offset_head OK")


def self_test() -> None:
    print("Laplacian pyramid:")
    _assert_bands_sum_to_field(n_levels=5, base_sigma=1.0)
    _assert_bands_sum_to_field(n_levels=3, base_sigma=2.0)
    print("Sliced-Wasserstein behavior:")
    identical = torch.randn(1, 1, 64, 64)
    sw_self = sliced_wasserstein_patches(identical, identical, patch_size=8, stride=4, n_projections=64).item()
    assert sw_self < 1e-6, f"identical fields: expected SW ~ 0, got {sw_self}"
    print(f"  Identical fields: patch-SW={sw_self:.2e} (~0) OK")
    _assert_sliced_w_shift_tolerant()
    _assert_histogram_w_permutation_invariant()
    print("sigma_band:")
    _assert_sigma_band_floor()
    print("FreqBandLoss:")
    _assert_fft_bands_sum_to_field(n_levels=5, base_sigma=2.0)
    _assert_fft_bands_sum_to_field(n_levels=3, base_sigma=1.0)
    _assert_freq_band_pixel_weights()
    _assert_per_band_config_alignment()
    _assert_freq_band_zero_loss()
    _assert_freq_band_fwd_bwd()
    print("Model integration:")
    _assert_gate_fwd_bwd()
    print("\nAll multiscale_loss self-tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    self_test()
