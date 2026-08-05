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

DEFAULT_METRIC_PER_BAND = ("l1", "sw", "sw", "sw", "sw")  # coarse -> fine, len == n_levels


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


class MultiscaleLoss:
    """Callable: __call__(pred_mean, truth) -> scalar. pred_mean/truth: (B, 2, H, W) [u, v].
    See module docstring for the per-band/per-component formula. Independently
    toggleable/bisectable from the rest of the loss via ms_weight in the caller."""

    def __init__(self, sigma_band: torch.Tensor, n_levels: int = 5, base_sigma: float = 1.0,
                 metric_per_band: tuple[str, ...] = DEFAULT_METRIC_PER_BAND,
                 patch_size: int = 8, patch_stride: int = 4, n_projections: int = 64,
                 sigma_floor: float = 1e-3, eps: float = 1e-8,
                 histogram_weight: float = 0.0):
        if len(metric_per_band) != n_levels:
            raise ValueError(f"metric_per_band must have length n_levels={n_levels}, got {len(metric_per_band)}")
        for m in metric_per_band:
            if m not in ("l1", "sw"):
                raise ValueError(f"metric_per_band entries must be 'l1' or 'sw', got {m!r}")
        if sigma_band.shape != (n_levels, 2):
            raise ValueError(f"sigma_band must have shape ({n_levels}, 2), got {tuple(sigma_band.shape)}")
        self.n_levels = n_levels
        self.base_sigma = base_sigma
        self.metric_per_band = tuple(metric_per_band)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_projections = n_projections
        self.eps = eps
        self.sigma_band = sigma_band.clamp_min(sigma_floor)  # (n_levels, 2), floored once
        self.histogram_weight = histogram_weight

    def _band_metric(self, metric: str, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if metric == "l1":
            d = F.l1_loss(p, t)
        else:
            d = sliced_wasserstein_patches(p, t, self.patch_size, self.patch_stride, self.n_projections)
        return METRIC_SCALE[metric] * d

    def __call__(self, pred_mean: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        pred_bands = laplacian_bands(pred_mean, self.n_levels, self.base_sigma)    # finest -> coarsest
        truth_bands = laplacian_bands(truth, self.n_levels, self.base_sigma)
        sigma = self.sigma_band.to(pred_mean.device)
        total = pred_mean.new_zeros(())
        for b in range(self.n_levels):
            # metric_per_band is coarse -> fine (see module docstring/DEFAULT_METRIC_PER_BAND);
            # pred_bands/sigma_band are finest -> coarsest (laplacian_bands' own order) -- index
            # from the other end to align them.
            metric = self.metric_per_band[self.n_levels - 1 - b]
            for c in range(2):
                p = pred_bands[b][:, c:c + 1]
                t = truth_bands[b][:, c:c + 1]
                d = self._band_metric(metric, p, t)
                total = total + d / (sigma[b, c] + self.eps)
                if self.histogram_weight > 0:
                    total = total + self.histogram_weight * histogram_wasserstein(p, t) / (sigma[b, c] + self.eps)
        return total


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


def fft_gaussian_bands(field: torch.Tensor, n_levels: int, base_sigma: float) -> list[torch.Tensor]:
    """FFT Gaussian filter bank with the same sigma sequence as laplacian_bands.
    field: (B, C, H, W). Returns n_levels tensors (B, C, H, W), finest -> coarsest.
    Bands telescope exactly: sum(fft_gaussian_bands(field, ...)) == field."""
    B, C, H, W = field.shape
    sigmas = band_sigmas(n_levels, base_sigma)
    lp = _fft_gaussian_lp_masks(H, W, sigmas, field.device, field.dtype)
    F_field = torch.fft.rfft2(field)
    band_masks = [lp[j] - lp[j + 1] for j in range(n_levels - 1)]
    band_masks.append(lp[-1])
    return [torch.fft.irfft2(m.unsqueeze(0).unsqueeze(0) * F_field, s=(H, W))
            for m in band_masks]


# --------------------------------------------------------------------------------------
# FreqBandLoss
# --------------------------------------------------------------------------------------
class FreqBandLoss:
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
        if sigma_band.shape != (n_levels, 2):
            raise ValueError(f"sigma_band must have shape ({n_levels}, 2), got {tuple(sigma_band.shape)}")
        if cdf_weight_mode not in ("down_extremes", "up_extremes"):
            raise ValueError(f"cdf_weight_mode must be 'down_extremes' or 'up_extremes', got {cdf_weight_mode!r}")
        if band_weight is not None and len(band_weight) != n_levels:
            raise ValueError(f"band_weight must have length n_levels={n_levels}, got {len(band_weight)}")
        if pixel_weight_enabled is not None and len(pixel_weight_enabled) != n_levels:
            raise ValueError(f"pixel_weight_enabled must have length n_levels={n_levels}, "
                             f"got {len(pixel_weight_enabled)}")
        self.n_levels = n_levels
        self.base_sigma = base_sigma
        self.cdf_weight_mode = cdf_weight_mode
        self.eps = eps
        self.sigma_band = sigma_band.clamp_min(sigma_floor)
        # Both stored FINEST -> COARSEST (band b=0 in __call__'s loop is finest -- see
        # _get_masks), the reverse of the constructor's coarse -> fine input, mirroring how
        # MultiscaleLoss.__call__ reverses metric_per_band against the same band order.
        self.band_weight = tuple(reversed(band_weight)) if band_weight is not None else (1.0,) * n_levels
        self.pixel_weight_enabled = (tuple(reversed(pixel_weight_enabled)) if pixel_weight_enabled is not None
                                     else (True,) * n_levels)
        self._mask_cache: dict = {}

    def _get_masks(self, H: int, W: int, device, dtype) -> list[torch.Tensor]:
        key = (H, W, str(device))
        if key not in self._mask_cache:
            sigmas = band_sigmas(self.n_levels, self.base_sigma)
            lp = _fft_gaussian_lp_masks(H, W, sigmas, device, dtype)
            band_masks = [lp[j] - lp[j + 1] for j in range(self.n_levels - 1)]
            band_masks.append(lp[-1])
            self._mask_cache[key] = [m.unsqueeze(0).unsqueeze(0) for m in band_masks]
        return self._mask_cache[key]

    def _pixel_weights(self, band_mag: torch.Tensor) -> torch.Tensor:
        """band_mag: (B, 1, H, W) → (B, 1, H, W) weights normalized to mean=1 per sample."""
        B, _, H, W = band_mag.shape
        flat = band_mag.reshape(B, -1)          # (B, N)
        N = flat.shape[1]
        ranks = torch.argsort(torch.argsort(flat, dim=1), dim=1).float()   # [0..N-1]
        if self.cdf_weight_mode == "down_extremes":
            w = 1.0 - ranks / (N - 1)
        else:
            w = ranks / (N - 1)
        w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-12)
        return w.reshape(B, 1, H, W)

    def __call__(self, pred_mean: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """pred_mean/truth: (B, 2, H, W) → scalar."""
        B, C, H, W = pred_mean.shape
        sigma = self.sigma_band.to(pred_mean.device)
        masks = self._get_masks(H, W, pred_mean.device, pred_mean.dtype)

        pred_F = torch.fft.rfft2(pred_mean)
        truth_F = torch.fft.rfft2(truth)

        total = pred_mean.new_zeros(())
        for b, mask in enumerate(masks):
            w = self.band_weight[b]
            if w == 0.0:
                continue  # fully deactivated: skip this band's irfft2/pixel-weight cost too
            pred_band = torch.fft.irfft2(mask * pred_F, s=(H, W))
            truth_band = torch.fft.irfft2(mask * truth_F, s=(H, W))
            weight_this_band = self.pixel_weight_enabled[b]
            for c in range(2):
                p = pred_band[:, c:c + 1]
                t = truth_band[:, c:c + 1]
                abs_err = (p - t).abs()
                if weight_this_band:
                    pw = self._pixel_weights(t.abs().detach())
                    band_term = (pw * abs_err).mean()
                else:
                    band_term = abs_err.mean()  # unweighted: skip the rank/CDF cost entirely
                total = total + w * band_term / (sigma[b, c] + self.eps)
        return total


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
    print(f"  pixel_weights: mean=1.0 per sample OK; down_extremes direction OK")


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
    """One real fwd+bwd: NLL + multiscale_loss at batch=4, gradients flow, no error.
    Imports the actual model -- run from scripts/ so new_enscgp_swin's relative imports
    (network_swin2sr, terrain_encoder) resolve."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from new_enscgp_swin import ProbabilisticSwin2SR
    from terrain_encoder import load_terrain_input

    torch.manual_seed(0)
    model = ProbabilisticSwin2SR(img_size=200, embed_dim=24, depths=(2,), num_heads=(4,))
    model.train()
    B = 4
    posterior = torch.randn(B, 5, 200, 200)
    posterior[:, 2] = F.softplus(posterior[:, 2])
    posterior[:, 4] = F.softplus(posterior[:, 4])
    bicubic = torch.randn(B, 2, 200, 200)
    wrf = torch.randn(B, 2, 200, 200)
    terrain_raw = torch.randn(1, 4, 1000, 1000)

    sigma_band = torch.ones(5, 2)  # synthetic, just needs the right shape for this check
    ms_loss_fn = MultiscaleLoss(sigma_band=sigma_band, n_levels=5)

    pred = model(posterior, bicubic, terrain_raw)
    mu, L11, L21, L22 = pred[:, :2], pred[:, 2].clamp_min(1e-6), pred[:, 3], pred[:, 4].clamp_min(1e-6)
    z1 = (wrf[:, 0] - mu[:, 0]) / L11
    z2 = (wrf[:, 1] - mu[:, 1] - L21 * z1) / L22
    nll = (torch.log(L11) + torch.log(L22) + 0.5 * (z1 ** 2 + z2 ** 2)).mean()
    ms = ms_loss_fn(mu, wrf)
    total = 0.5 * nll + 1.0 * ms

    model.zero_grad()
    total.backward()
    assert model.conv_first.weight.grad is not None and model.conv_first.weight.grad.abs().max().item() > 0
    assert model.mean_gate.grad is not None and model.mean_gate.grad.abs().item() > 0
    assert model.chol_gate.grad is not None and model.chol_gate.grad.abs().item() > 0
    print(f"  Fwd+bwd (batch={B}): NLL={nll.item():.4f}, multiscale_loss={ms.item():.4f}, "
          f"total={total.item():.4f}, gradients flow to backbone + both gates OK")


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
    _assert_freq_band_zero_loss()
    _assert_freq_band_fwd_bwd()
    print("Model integration:")
    _assert_gate_fwd_bwd()
    print("\nAll multiscale_loss self-tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    self_test()
