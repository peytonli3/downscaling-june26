"""Train ProbabilisticSwin2SR (new_enscgp_swin.py) as a residual refiner of the
EnsCGP first-guess posterior, supervised against WRF ground truth.

Per sample:
- input:  data/enscgp_posterior.npy        (N, 5, 200, 200) = [u, v, L11, L21, L22],
          the EnsCGP first guess (enscgp_train.py), and
          data/era5_uv_2ch_bicubic.npy      (N, 2, 200, 200) = [u, v], the bicubic-
          upsampled ERA5 baseline. The model is fed bicubic + the whole EnsCGP
          posterior directly, and predicts a residual on top of a configurable base
          (see new_enscgp_swin.py's residual_base).
- target: data/wrf_uv.npy                   (N, 2, 200, 200) = [u, v], WRF ground truth.
- terrain: land_sea_mask_features.npy + topography_features.npy, static across
          all samples, encoded by TerrainEncoder (terrain_encoder.py) -- a
          submodule of the model, trained jointly (not a frozen feature map).

Loss = ms_weight * multiscale_loss(q50, wrf_uv)          [see multiscale_loss.py]
     + freq_weight * freq_band_loss(q50, wrf_uv)
     + l1_weight * L1(q50, wrf_uv)
     + spectral_weight * high-freq spectral L1(q50, wrf_uv)
     + gradient_weight * divergence L1(q50, wrf_uv)
     + pin_weight * [ pinball(q90, wrf_uv, 0.9) + pinball(q10, wrf_uv, 0.1) ]
The STRUCTURAL / mean-supervision terms (ms/freq/l1/spectral/gradient) attach to q50 ONLY
-- exactly the role the old predicted mean had -- so q50 stays sharp and displacement-
tolerant. They must never be pointed at q10/q90 (an uncertainty envelope is not a wind
field; matching its spectrum/texture is a category error) nor at the quantile spread.
The uncertainty is trained SOLELY by the pinball (quantile) loss on q90/q10 -- the old
Gaussian NLL and its Cholesky are gone. q50 is deliberately NOT pinball-trained (a
pinball(0.5) term would pull it toward the blurry pointwise median and fight the
structural losses). All weights are configurable ("training" section).

Zero-weight structural terms (l1/spectral/gradient in the active config) are SKIPPED
during training -- not computed at all, not just multiplied by 0 -- so no wasted compute
(two rfft2 calls for spectral() alone) or log noise for a term that isn't influencing the
gradient. At VALIDATION time every structural term is still computed and logged (weight or
not), purely as a diagnostic -- so you can see e.g. what l1/spectral currently read even
while they're off, without paying that cost every training batch. See compute_all in
compute_weighted_loss / evaluate.

Extreme weighting: rare high-wind pixels are a tiny fraction of the field, so an
unweighted pinball under-trains them and q90 gets smoothed down off the damaging peaks.
u and v are SIGNED (a strong westward gust is exactly as extreme as an equally strong
eastward one), so extremity is judged from |target|'s per-component rank, then ROUTED to
whichever tail the sign indicates: q90's pinball is up-weighted where that component is
strongly positive, q10's (if apply_to_q10) where strongly negative -- see
extreme_pixel_weights_signed. Weight = 1 + alpha * F(|target|), so peak pixels get up to
~(1+alpha)x weight on the relevant tail only; DETACHED (no gradient through the ranking)
and mean-normalized per (sample, component) so the loss scale is invariant to alpha.
Calibration is checked post-hoc via coverage (see evaluate), NOT enforced by the loss.

multiscale_loss (scripts/multiscale_loss.py) is now the PRIMARY mean-supervision term --
a Laplacian-pyramid, displacement-tolerant (sliced-Wasserstein) loss built to replace the
legacy l1/spectral/gradient stack, which a per-band error diagnostic
(extra_scripts/band_error_diagnostics.py) found rewards smoothing instead of correcting the
dominant error mode (random spatial displacement at every scale). The active config
("new_enscgp_swin_config.json") sets l1_weight/spectral_weight/gradient_weight to 0.0 and
ms_weight > 0 -- each structural weight is still independently toggleable/zeroable (e.g. to
bisect back to the legacy stack, set ms_weight=0 and restore l1_weight=1.0 etc.). The first
time ms_weight > 0, this script precomputes and
caches multiscale_loss's per-band sigma_band normalization over a sample of the training
set (see multiscale_loss.load_or_compute_sigma_band) -- a one-time cost, not per-batch.

LR schedule: MultiStepLR, stepped every optimizer step (not per-epoch). Milestones are
given as fractions of total training steps ("lr_milestone_fractions", default
[0.5, 0.75, 0.9]) rather than raw step counts, since the actual step count depends on
batch_size/num_epochs/dataset size; LR is multiplied by "lr_gamma" (default 0.5) at
each one, so by default the LR is halved three times over a run (final LR = lr/8).

Train/val split: data/splits_70_15_15/split_indices.npz (train_idx/val_idx),
indexing directly into enscgp_posterior.npy/era5_uv_2ch_bicubic.npy/wrf_uv.npy
(all length N=6767, index-aligned with each other and with that split file).

Config: new_enscgp_swin_config.json ("paths"/"model"/"training"/"logging"
sections), same convention as 26.3_wind/SWIN/wind_swin2sr_config.json.

Usage:
    python train_new_enscgp_swin.py [--config new_enscgp_swin_config.json] [--device cuda] [--resume PATH]
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from new_enscgp_swin import DEFAULT_CONFIG_PATH, ProbabilisticSwin2SR, build_model, load_config
from terrain_encoder import load_terrain_input
from multiscale_loss import DEFAULT_METRIC_PER_BAND, METRIC_SCALE, FreqBandLoss, MultiscaleLoss, load_or_compute_sigma_band


class EnsCGPSwinDataset(Dataset):
    def __init__(self, posterior_path: Path, bicubic_path: Path, wrf_path: Path, indices: np.ndarray):
        self.posterior = np.load(posterior_path, mmap_mode="r")
        self.bicubic = np.load(bicubic_path, mmap_mode="r")
        self.wrf = np.load(wrf_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx])
        # Writable copies: memmap slices are read-only, which torch warns/errors on.
        posterior = torch.from_numpy(np.array(self.posterior[sample_idx], dtype=np.float32, copy=True))
        bicubic = torch.from_numpy(np.array(self.bicubic[sample_idx], dtype=np.float32, copy=True))
        wrf = torch.from_numpy(np.array(self.wrf[sample_idx], dtype=np.float32, copy=True))
        return posterior, bicubic, wrf


class MeanAuxLosses:
    """High-frequency spectral L1 and divergence ("gradient") L1 computed exclusively on
    q50 (the central field) vs. wrf_uv -- adapted from WeightedWindLoss in
    26.3_wind/SWIN/train_wind_swin2sr.py (its spectral_high/sparse_grad terms). These are
    STRUCTURAL terms on q50, kept alongside the primary multiscale/freq losses (all default
    to weight 0 in the active config). The old mu-pinball ("quantile") term is gone: the
    quantile heads (q10/q90) are trained by the pinball loss in compute_weighted_loss, and
    q50 is deliberately not pinball-trained. Stateless aside from a cache for the FFT
    frequency mask, which depends only on (H, W, device).
    """

    def __init__(self, spectral_low_freq_cutoff: float = 0.28):
        self.spectral_low_freq_cutoff = float(spectral_low_freq_cutoff)
        if not 0.0 < self.spectral_low_freq_cutoff < 1.0:
            raise ValueError(f"spectral_low_freq_cutoff must be in (0,1), got {self.spectral_low_freq_cutoff}")
        self._high_mask_cache: tuple | None = None  # (H, W, device) -> mask

    @staticmethod
    def _frequency_band_mask(height: int, width: int, cutoff: float, device: torch.device) -> torch.Tensor:
        y_freq = torch.fft.fftfreq(height, d=1.0, device=device)
        x_freq = torch.fft.rfftfreq(width, d=1.0, device=device)
        radius = torch.sqrt(y_freq[:, None] ** 2 + x_freq[None, :] ** 2)
        radius = radius / radius.max().clamp(min=1e-12)
        return radius > cutoff  # high-frequency mask

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not torch.any(mask):
            return torch.tensor(0.0, device=values.device, dtype=values.dtype)
        mask = mask.to(device=values.device, dtype=values.dtype)[None, None, :, :]
        return (values * mask).sum() / (mask.sum() * values.shape[0] * values.shape[1])

    def spectral(self, mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """High-frequency-band L1 of the 2D FFT difference: penalizes mu for being too
        spatially smooth relative to wrf_uv's true high-frequency content."""
        H, W = mu.shape[-2], mu.shape[-1]
        cache_key = (H, W, mu.device)
        if self._high_mask_cache is None or self._high_mask_cache[0] != cache_key:
            high_mask = self._frequency_band_mask(H, W, self.spectral_low_freq_cutoff, mu.device)
            self._high_mask_cache = (cache_key, high_mask)
        else:
            high_mask = self._high_mask_cache[1]
        mu_fft = torch.fft.rfft2(mu, dim=(-2, -1), norm="ortho")
        target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
        return self._masked_mean(torch.abs(mu_fft - target_fft), high_mask)

    @staticmethod
    def gradient(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Divergence-difference L1: penalizes mismatched du/dx + dv/dy between the
        predicted and true wind vector fields (mu/target channel 0=u, 1=v)."""
        def divergence(x: torch.Tensor) -> torch.Tensor:
            u, v = x[:, 0:1], x[:, 1:2]
            u_pad = F.pad(u, (1, 1, 0, 0), mode="replicate")
            u_x = 0.5 * (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2])
            v_pad = F.pad(v, (0, 0, 1, 1), mode="replicate")
            v_y = 0.5 * (v_pad[:, :, 2:, :] - v_pad[:, :, :-2, :])
            return u_x + v_y
        return torch.mean(torch.abs(divergence(mu) - divergence(target)))


def _rank_cdf_per_channel(x: torch.Tensor) -> torch.Tensor:
    """x: (B,C,H,W), non-negative. Empirical CDF (rank/(N-1)) computed independently within
    each (batch, channel) slice's H*W pixels -- so u and v (or any two channels) are ranked
    against their OWN distribution, not mixed together. Same shape as x."""
    B, C, H, W = x.shape
    flat = x.reshape(B, C, H * W)
    n = flat.shape[-1]
    ranks = flat.argsort(dim=-1).argsort(dim=-1).to(x.dtype)
    cdf = ranks / max(n - 1, 1)
    return cdf.reshape(B, C, H, W)


def extreme_pixel_weights_signed(target: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sign-aware extreme weighting for a SIGNED quantity (wind components u, v can be
    positive or negative -- a strong westward gust is exactly as extreme as an equally
    strong eastward one). A SINGLE weight scale is computed from |target| -- so "how extreme
    is this pixel" means the same thing regardless of direction -- then ROUTED per pixel to
    whichever tail its sign indicates:
      w = 1 + alpha * F(|target|)     F = empirical CDF within each (sample, component)
                                       independently (_rank_cdf_per_channel), mean-normalized
      w_upper = w where target >= 0, else 1 (baseline -- nothing extreme for q90 to reach there)
      w_lower = w where target <  0, else 1 (baseline -- nothing extreme for q10 to reach there)
    Returns (w_upper, w_lower), each (B,2,H,W). DETACHED (no gradient through the ranking).

    Ranking the SIGNED value's magnitude directly (rather than ranking the positive and
    negative parts separately, each renormalized on its own) matters: if a sample's wind is
    overwhelmingly one-directional -- say strongly negative u with only mild positive
    gusts -- the "most positive" pixel might be a modest +3 m/s, nowhere near as severe as
    the -30 m/s on the other side. Two independently-normalized rankings would boost that
    +3 m/s pixel toward the SAME max weight as the -30 m/s one, purely for being locally
    largest on its own (mostly unremarkable) side. Ranking |target| once means a pixel's
    weight reflects how extreme it actually is in absolute terms, comparable across both
    signs; only the ROUTING (which tail receives it) depends on sign.

    This replaces an earlier design that reused one magnitude-based weight identically for
    both q90 and q10 (w10 = w90) with no sign-awareness at all: that version pushed both
    tails equally hard at every extreme pixel regardless of which direction the extreme was
    in, wasting half the pressure on a tail that had nothing to reach for there.
    """
    with torch.no_grad():
        w = 1.0 + alpha * _rank_cdf_per_channel(target.abs())
        w = w / w.mean(dim=(2, 3), keepdim=True)  # mean-normalize per (B, C)
        is_pos = target >= 0
        ones = torch.ones_like(w)
        w_upper = torch.where(is_pos, w, ones)
        w_lower = torch.where(is_pos, ones, w)
    return w_upper, w_lower


def pinball_loss(q: torch.Tensor, target: torch.Tensor, tau: float,
                 weight: torch.Tensor | None = None) -> torch.Tensor:
    """Mean pinball (tilted-L1) loss for quantile level tau: err = target - q;
    loss = max(tau*err, (tau-1)*err) per pixel (== tau*err if target>q else (1-tau)*(q-target)).
    Fully differentiable in q. weight is (B,2,H,W) (per-component) or (B,1,H,W) (broadcasts
    over both components) if given."""
    err = target - q
    loss = torch.maximum(tau * err, (tau - 1.0) * err)  # (B, 2, H, W)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def compute_weighted_loss(pred: torch.Tensor, target: torch.Tensor, weights: dict, mean_aux: MeanAuxLosses,
                           ms_loss_fn: MultiscaleLoss | None = None,
                           freq_loss_fn: FreqBandLoss | None = None,
                           extreme_cfg: dict | None = None,
                           compute_all: bool = False) -> dict:
    """pred: (B,6,H,W) [q10_u,q10_v, q50_u,q50_v, q90_u,q90_v]; target: (B,2,H,W) [u,v].
    weights: {"ms","freq","l1","spectral","gradient","pin"} -> float.

    Structural terms (ms/freq/l1/spectral/gradient) attach to q50 ONLY (see multiscale_loss.py
    / MeanAuxLosses) -- q50 keeps the old mean's role and stays sharp. Uncertainty is trained
    only by pinball(q90, .9) + pinball(q10, .1). ms_loss_fn/freq_loss_fn are None when their
    weight is 0 (no precompute, no per-batch cost) -- see train()'s startup.

    compute_all: when False (training), a structural term whose weight is 0 is SKIPPED
    entirely (its dict entry is None, not a zero tensor) -- since weight=0 makes it a no-op
    for `total` regardless, skipping it changes no gradient, only saves compute. When True
    (validation), every structural term is computed and returned regardless of weight, as a
    diagnostic -- see evaluate().

    extreme_cfg (or None = off): {"enabled": bool, "alpha": float, "apply_to_q10": bool}.
    When enabled, up-weights q90's pinball wherever that component is strongly positive
    (and, if apply_to_q10, q10's wherever it's strongly negative) -- see
    extreme_pixel_weights_signed. u and v can each be positive or negative, so this is a
    sign-aware, per-component weighting, not a single shared magnitude-based one.

    Returns a dict with every component (all detached except "total", which carries the
    graph); a skipped structural term's value is None.
    """
    q10 = pred[:, ProbabilisticSwin2SR.Q10_SLICE]
    q50 = pred[:, ProbabilisticSwin2SR.Q50_SLICE]
    q90 = pred[:, ProbabilisticSwin2SR.Q90_SLICE]

    ms = ms_loss_fn(q50, target) if (ms_loss_fn is not None and (compute_all or weights["ms"] > 0)) else None
    freq = freq_loss_fn(q50, target) if (freq_loss_fn is not None and (compute_all or weights["freq"] > 0)) else None
    l1 = F.l1_loss(q50, target) if (compute_all or weights["l1"] > 0) else None
    spectral = mean_aux.spectral(q50, target) if (compute_all or weights["spectral"] > 0) else None
    gradient = mean_aux.gradient(q50, target) if (compute_all or weights["gradient"] > 0) else None

    w90 = w10 = None
    if extreme_cfg is not None and extreme_cfg.get("enabled", False):
        w_upper, w_lower = extreme_pixel_weights_signed(target, extreme_cfg.get("alpha", 1.0))
        w90 = w_upper
        if extreme_cfg.get("apply_to_q10", False):
            w10 = w_lower
    pin90 = pinball_loss(q90, target, 0.9, w90)
    pin10 = pinball_loss(q10, target, 0.1, w10)
    pin = pin90 + pin10

    total = weights["pin"] * pin
    for key, term in (("ms", ms), ("freq", freq), ("l1", l1), ("spectral", spectral), ("gradient", gradient)):
        if term is not None:
            total = total + weights[key] * term

    return {
        "total": total,
        "ms": ms.detach() if ms is not None else None,
        "freq": freq.detach() if freq is not None else None,
        "l1": l1.detach() if l1 is not None else None,
        "spectral": spectral.detach() if spectral is not None else None,
        "gradient": gradient.detach() if gradient is not None else None,
        "pin": pin.detach(), "pin90": pin90.detach(), "pin10": pin10.detach(),
    }


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("train_new_enscgp_swin")
    logger.info("Logging to %s", log_path)
    return logger


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_loss: float):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }, path)


# ALWAYS_KEYS are computed every call, train or val. STRUCTURAL_KEYS are only computed when
# compute_all=True (val) or their own weight is > 0 (train) -- see compute_weighted_loss.
ALWAYS_KEYS = ("total", "pin", "pin90", "pin10")
STRUCTURAL_KEYS = ("ms", "freq", "l1", "spectral", "gradient")
STRUCTURAL_LABELS = {"ms": "MS", "freq": "Freq", "l1": "L1", "spectral": "Spectral", "gradient": "Gradient"}
LOSS_COMPONENT_KEYS = ALWAYS_KEYS + STRUCTURAL_KEYS  # evaluate() always computes all of these
# Coverage = fraction of truth pixels at or below each predicted quantile (target nominal
# in parentheses). "_ext" variants restrict to the extreme tail (|wind| above its per-sample
# 90th percentile) -- the damage-relevant region where q90 must actually reach the peaks.
COVERAGE_KEYS = ("cov_q10", "cov_q50", "cov_q90", "cov_q10_ext", "cov_q50_ext", "cov_q90_ext")
NOMINAL_COVERAGE = {"cov_q10": 0.10, "cov_q50": 0.50, "cov_q90": 0.90}


@torch.no_grad()
def coverage_metrics(pred: torch.Tensor, target: torch.Tensor, ext_quantile: float = 0.9) -> dict:
    """Empirical coverage: fraction of truth (u,v) pixels <= each predicted quantile, both
    overall and within the extreme tail (|wind| above its per-sample ext_quantile). A
    calibrated model has cov_q10~=.10, cov_q50~=.50, cov_q90~=.90. Post-hoc CHECK, not a loss.
    Returns per-key (count_below, count_total) so a running total can be aggregated exactly."""
    q10 = pred[:, ProbabilisticSwin2SR.Q10_SLICE]
    q50 = pred[:, ProbabilisticSwin2SR.Q50_SLICE]
    q90 = pred[:, ProbabilisticSwin2SR.Q90_SLICE]
    below = {"cov_q10": (target <= q10), "cov_q50": (target <= q50), "cov_q90": (target <= q90)}

    mag = torch.sqrt(target[:, 0:1] ** 2 + target[:, 1:2] ** 2 + 1e-6)   # (B,1,H,W)
    thresh = torch.quantile(mag.reshape(mag.shape[0], -1), ext_quantile, dim=1)  # (B,)
    ext = (mag >= thresh.reshape(-1, 1, 1, 1)).expand_as(q50)             # (B,2,H,W)

    out = {}
    for base, mask in below.items():
        out[base] = (mask.sum().item(), mask.numel())
        out[base + "_ext"] = ((mask & ext).sum().item(), ext.sum().item())
    return out


@torch.no_grad()
def evaluate(model, loader, terrain_raw, device, weights: dict, mean_aux: MeanAuxLosses,
             ms_loss_fn: MultiscaleLoss | None = None,
             freq_loss_fn: FreqBandLoss | None = None,
             extreme_cfg: dict | None = None) -> dict:
    """Returns per-sample-averaged loss components plus empirical coverage (see
    compute_weighted_loss / coverage_metrics). Every structural term is computed
    (compute_all=True) regardless of its weight -- this is the one place they're always
    visible, even while off during training; see compute_weighted_loss."""
    model.eval()
    totals = {k: 0.0 for k in LOSS_COMPONENT_KEYS}
    cov_below = {k: 0 for k in COVERAGE_KEYS}
    cov_total = {k: 0 for k in COVERAGE_KEYS}
    n = 0
    for posterior, bicubic, wrf in loader:
        posterior = posterior.to(device, non_blocking=True)
        bicubic = bicubic.to(device, non_blocking=True)
        wrf = wrf.to(device, non_blocking=True)
        pred = model(posterior, bicubic, terrain_raw)
        loss_dict = compute_weighted_loss(pred, wrf, weights, mean_aux, ms_loss_fn, freq_loss_fn,
                                          extreme_cfg, compute_all=True)
        b = posterior.shape[0]
        for k in LOSS_COMPONENT_KEYS:
            totals[k] += loss_dict[k].item() * b
        for k, (below, tot) in coverage_metrics(pred, wrf).items():
            cov_below[k] += below
            cov_total[k] += tot
        n += b
    model.train()
    out = {k: v / n for k, v in totals.items()}
    out.update({k: (cov_below[k] / cov_total[k] if cov_total[k] else float("nan")) for k in COVERAGE_KEYS})
    return out


def train(config: dict, device_str: str, resume_path: Path | None = None,
          resume_weights_only: bool = False, max_steps: int | None = None):
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    paths = config["paths"]
    data_dir = Path(paths["data_dir"])
    log_dir = Path(paths["log_dir"])
    splits_path = Path(paths["splits_path"])

    logger = setup_logging(log_dir)
    logger.info("Config:\n%s", json.dumps(config, indent=2))

    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    splits = np.load(splits_path)
    train_idx, val_idx = splits["train_idx"], splits["val_idx"]
    logger.info("Train samples: %d, Val samples: %d", len(train_idx), len(val_idx))

    posterior_path = data_dir / "enscgp_posterior.npy"
    bicubic_path = data_dir / "era5_uv_2ch_bicubic.npy"
    wrf_path = data_dir / "wrf_uv.npy"
    train_ds = EnsCGPSwinDataset(posterior_path, bicubic_path, wrf_path, train_idx)
    val_ds = EnsCGPSwinDataset(posterior_path, bicubic_path, wrf_path, val_idx)

    t = config["training"]
    batch_size = t.get("batch_size", 4)
    num_workers = t.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = build_model(config).to(device)
    terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device)
    logger.info("Total parameters: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=t.get("learning_rate", 1e-3), weight_decay=t.get("weight_decay", 0.0),
    )

    num_epochs = t.get("num_epochs", 100)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    lr_milestone_fractions = t.get("lr_milestone_fractions", [0.5, 0.75, 0.9])
    lr_milestones = [int(f * total_steps) for f in lr_milestone_fractions]
    lr_gamma = t.get("lr_gamma", 0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    logger.info(
        "LR schedule: milestones (steps) = %s (of %d total), gamma = %.3f", lr_milestones, total_steps, lr_gamma
    )

    start_epoch, best_val_loss = 0, float("inf")
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        if resume_weights_only:
            # PARTIAL, non-strict load: transfer only tensors whose name AND shape match.
            # Resuming a 0714 (v6) checkpoint into this (v7) architecture: mean_head,
            # offset_head, backbone, and terrain_encoder all keep the same module shapes, so
            # they transfer even though mean_head/offset_head's INIT scheme changed (irrelevant
            # for a resume -- their weights are already trained, not at init values). The
            # dropped mean_gate/offset_gate scalars (v7 has no gates) are simply not loaded.
            # CAVEAT: conv_first's weight shape is ALSO unchanged (still 11 input channels),
            # so it transfers too -- but its channels 2-3 now carry raw EnsCGP u/v instead of
            # an EnsCGP-minus-bicubic residual (see new_enscgp_swin.py's module docstring).
            # That's a semantic change hiding behind an unchanged shape: a naive partial load
            # gives conv_first weights calibrated to the WRONG input at those two channels.
            # Prefer training v7 fresh from EnsCGP; only resume through this boundary if you
            # explicitly want to test it.
            model_sd = model.state_dict()
            filtered = {k: v for k, v in ckpt["model_state_dict"].items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            fresh = sorted(k for k in model_sd if k not in filtered)
            dropped = sorted(k for k in ckpt["model_state_dict"] if k not in model_sd)
            model.load_state_dict(filtered, strict=False)
            logger.info("Partial weights-only load from %s: transferred %d/%d tensors.",
                        resume_path, len(filtered), len(model_sd))
            logger.info("  Fresh (not in checkpoint): %s", fresh)
            logger.info("  Dropped (not in model): %s", dropped)
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            logger.info("Resumed from %s at epoch %d (best_val_loss=%.5f)", resume_path, start_epoch, best_val_loss)

    weights = {
        "ms": t.get("ms_weight", 0.0),
        "freq": t.get("freq_weight", 0.0),
        "l1": t.get("l1_weight", 0.0),
        "spectral": t.get("spectral_weight", 0.0),
        "gradient": t.get("gradient_weight", 0.0),
        "pin": t.get("pin_weight", 1.0),
    }
    extreme_cfg = t.get("extreme_weight", {"enabled": True, "alpha": 1.5, "apply_to_q10": False})
    mean_aux = MeanAuxLosses(spectral_low_freq_cutoff=t.get("spectral_low_freq_cutoff", 0.28))
    logger.info("Loss weights: %s", weights)
    logger.info("Extreme weighting: %s", extreme_cfg)

    # Structural terms with weight 0 are skipped entirely during TRAINING (see
    # compute_weighted_loss's compute_all) -- only the active ones are tracked/logged per
    # batch. evaluate() always computes and logs every structural term regardless.
    active_structural_keys = [k for k in STRUCTURAL_KEYS if weights[k] > 0]
    active_train_keys = list(ALWAYS_KEYS) + active_structural_keys
    logger.info("Structural terms computed during training (weight > 0): %s", active_structural_keys)

    # multiscale_loss / freq_band_loss: built UNCONDITIONALLY (not gated on weight > 0) so
    # evaluate()'s compute_all can always report them on val, even at weight 0 -- see
    # compute_weighted_loss. This costs one load_or_compute_sigma_band call each at startup
    # (a cache hit after the first run; a few minutes to precompute on a fresh n_levels/
    # base_sigma combo) -- a ONE-TIME cost, not per-batch, unlike computing the terms
    # themselves. Per-batch, compute_weighted_loss still only CALLS a term when
    # compute_all=True (val) or its own weight > 0 (train) -- see multiscale_loss.py's
    # module docstring for the ms_weight tuning guardrail.
    ms_cfg = t.get("multiscale", {})
    ms_n_levels = ms_cfg.get("n_levels", 5)
    ms_base_sigma = ms_cfg.get("base_sigma", 1.0)
    sigma_band = load_or_compute_sigma_band(
        data_dir, wrf_path, train_idx, ms_n_levels, ms_base_sigma, device=str(device),
        max_samples=ms_cfg.get("sigma_max_samples", 2000),
        force_recompute=ms_cfg.get("force_recompute_sigma_band", False),
    )
    ms_loss_fn = MultiscaleLoss(
        sigma_band=sigma_band, n_levels=ms_n_levels, base_sigma=ms_base_sigma,
        metric_per_band=tuple(ms_cfg.get("metric_per_band", DEFAULT_METRIC_PER_BAND)),
        patch_size=ms_cfg.get("patch_size", 8), patch_stride=ms_cfg.get("patch_stride", 4),
        n_projections=ms_cfg.get("n_projections", 64),
        sigma_floor=ms_cfg.get("sigma_floor", 1e-3),
        histogram_weight=ms_cfg.get("histogram_weight", 0.0),
    )
    logger.info(
        "multiscale_loss built (active during training: %s): n_levels=%d, base_sigma=%.2f, "
        "metric_per_band=%s, METRIC_SCALE=%s (fixed), sigma_band=%s",
        weights["ms"] > 0, ms_n_levels, ms_base_sigma, ms_loss_fn.metric_per_band, METRIC_SCALE, sigma_band.tolist(),
    )

    fb_cfg = t.get("freq_band", {})
    fb_n_levels = fb_cfg.get("n_levels", 5)
    fb_base_sigma = fb_cfg.get("base_sigma", 2.0)
    # Reuse the same sigma_band cache as MultiscaleLoss (same filter sigmas).
    fb_sigma_band = load_or_compute_sigma_band(
        data_dir, wrf_path, train_idx, fb_n_levels, fb_base_sigma, device=str(device),
        max_samples=fb_cfg.get("sigma_max_samples", 2000),
        force_recompute=fb_cfg.get("force_recompute_sigma_band", False),
    )
    freq_loss_fn = FreqBandLoss(
        sigma_band=fb_sigma_band, n_levels=fb_n_levels, base_sigma=fb_base_sigma,
        cdf_weight_mode=fb_cfg.get("cdf_weight_mode", "down_extremes"),
        sigma_floor=fb_cfg.get("sigma_floor", 1e-3),
    )
    logger.info(
        "freq_band_loss built (active during training: %s): n_levels=%d, base_sigma=%.2f, "
        "cdf_weight_mode=%s, sigma_band=%s",
        weights["freq"] > 0, fb_n_levels, fb_base_sigma, freq_loss_fn.cdf_weight_mode, fb_sigma_band.tolist(),
    )

    grad_clip_norm = t.get("grad_clip_norm", 1.0)
    log_every = config["logging"].get("log_every", 50)
    val_interval = t.get("val_interval", 1)
    early_stop_patience = t.get("early_stop_patience", 8)
    logger.info("Early stop patience: %s val check(s) without improvement%s",
                early_stop_patience, "" if early_stop_patience > 0 else " (disabled)")

    global_step = 0
    epochs_since_improvement = 0
    model.train()
    for epoch in range(start_epoch, num_epochs):
        running = {k: 0.0 for k in active_train_keys}
        for i, (posterior, bicubic, wrf) in enumerate(train_loader):
            posterior = posterior.to(device, non_blocking=True)
            bicubic = bicubic.to(device, non_blocking=True)
            wrf = wrf.to(device, non_blocking=True)

            pred = model(posterior, bicubic, terrain_raw)
            loss_dict = compute_weighted_loss(pred, wrf, weights, mean_aux, ms_loss_fn, freq_loss_fn, extreme_cfg)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            for k in active_train_keys:
                running[k] += loss_dict[k].item()
            global_step += 1

            if (i + 1) % log_every == 0:
                struct_str = "".join(
                    f", {STRUCTURAL_LABELS[k]} {running[k] / log_every:.5f}" for k in active_structural_keys
                )
                logger.info(
                    "Epoch %d | Batch %d/%d | Loss %.5f (Pin %.5f [q90 %.5f, q10 %.5f]%s) | LR %.2e",
                    epoch, i + 1, len(train_loader),
                    running["total"] / log_every, running["pin"] / log_every,
                    running["pin90"] / log_every, running["pin10"] / log_every,
                    struct_str,
                    scheduler.get_last_lr()[0],
                )
                running = {k: 0.0 for k in active_train_keys}

            if max_steps is not None and global_step >= max_steps:
                logger.info("Reached --max-steps=%d, stopping early.", max_steps)
                return

        if (epoch + 1) % val_interval == 0:
            # Every structural term is computed here regardless of weight (compute_all),
            # purely as a diagnostic -- see compute_weighted_loss.
            val_metrics = evaluate(model, val_loader, terrain_raw, device, weights, mean_aux,
                                   ms_loss_fn, freq_loss_fn, extreme_cfg)
            val_loss = val_metrics["total"]
            logger.info(
                "Epoch %d | Val Loss %.5f (Pin %.5f [q90 %.5f, q10 %.5f], MS %.5f, Freq %.5f, "
                "L1 %.5f, Spectral %.5f, Gradient %.5f)",
                epoch, val_loss, val_metrics["pin"], val_metrics["pin90"], val_metrics["pin10"],
                val_metrics["ms"], val_metrics["freq"],
                val_metrics["l1"], val_metrics["spectral"], val_metrics["gradient"],
            )
            logger.info(
                "Epoch %d | Coverage q10/q50/q90 = %.3f/%.3f/%.3f (nominal .10/.50/.90) | "
                "extreme tail = %.3f/%.3f/%.3f",
                epoch, val_metrics["cov_q10"], val_metrics["cov_q50"], val_metrics["cov_q90"],
                val_metrics["cov_q10_ext"], val_metrics["cov_q50_ext"], val_metrics["cov_q90_ext"],
            )

            save_checkpoint(checkpoint_dir / "last.pth", model, optimizer, scheduler, epoch, best_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, scheduler, epoch, best_val_loss)
                logger.info("New best val loss: %.5f (saved checkpoints/best.pth)", best_val_loss)
            else:
                epochs_since_improvement += 1
                logger.info("Epoch %d | Val loss did not improve (%d/%s val checks since best=%.5f)",
                            epoch, epochs_since_improvement,
                            early_stop_patience if early_stop_patience > 0 else "inf", best_val_loss)
                if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
                    logger.info(
                        "Epoch %d | Early stopping: no improvement for %d val check(s) (patience=%d).",
                        epoch, epochs_since_improvement, early_stop_patience,
                    )
                    break

    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--resume-weights-only", action="store_true",
                        help="Load only model weights from --resume; reset optimizer, scheduler, and epoch")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop after this many optimizer steps -- for smoke-testing the pipeline, not real training runs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, device_str=args.device, resume_path=args.resume,
          resume_weights_only=args.resume_weights_only, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
