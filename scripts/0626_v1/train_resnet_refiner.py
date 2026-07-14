"""Train ResNetRefiner (resnet_refiner.py) as a residual refiner of the EnsCGP
first-guess posterior, supervised against WRF ground truth -- the training
counterpart to train_new_enscgp_swin.py, for the "simple baseline" ResNet model.

The loss, dataset, and checkpointing code below is duplicated from
train_new_enscgp_swin.py rather than imported from it: both are generic (no
Swin-specific logic), but importing from a script named/scoped to the Swin model
would tie this baseline's training pipeline to the Swin one, which goes against
resnet_refiner.py's "prefer independence over sharing" design (see its docstring).

Per sample:
- input:  data/enscgp_posterior.npy  (N, 5, 200, 200) = [u, v, L11, L21, L22],
          the EnsCGP first guess (enscgp_train.py) being refined. first_guess_mu
          and first_guess_chol are sliced from this right before each forward()
          call -- ResNetRefiner.forward() takes them as explicit args rather than
          re-deriving them internally, see resnet_refiner.py for why.
- target: data/wrf_uv.npy            (N, 2, 200, 200) = [u, v], WRF ground truth.
- terrain: land_sea_mask_features.npy + topography_features.npy, static across
          all samples, encoded by TerrainEncoder (terrain_encoder.py) -- a
          submodule of the model, trained jointly (not a frozen feature map).

Loss = nll_weight * NLL(mu, Sigma=LL^T; wrf_uv)
     + l1_weight * L1(mu, wrf_uv)
     + spectral_weight * high-freq spectral L1(mu, wrf_uv)
     + gradient_weight * divergence L1(mu, wrf_uv)
     + quantile_weight * pinball(mu, wrf_uv)
Identical loss formulation to train_new_enscgp_swin.py (see that file for the
derivation of the NLL split into logdet/mahalanobis terms), so the two models are
comparable under the same training objective.

LR schedule: MultiStepLR, stepped every optimizer step (not per-epoch). Milestones
are given as fractions of total training steps ("lr_milestone_fractions", default
[0.5, 0.75, 0.9]); LR is multiplied by "lr_gamma" (default 0.5) at each one.

Train/val split: data/splits_70_15_15/split_indices.npz (train_idx/val_idx),
indexing directly into enscgp_posterior.npy/wrf_uv.npy (both length N=6767,
index-aligned with each other and with that split file) -- same split as the Swin
model's training run, so the two models are compared on identical val data.

Config: resnet_refiner_config.json ("paths"/"model"/"training"/"logging" sections),
same convention as new_enscgp_swin_config.json. Its "paths.log_dir" points at a
separate logs/resnet_refiner directory (not the Swin run's logs/), so the two
models' checkpoints and training logs don't collide.

Usage:
    python train_resnet_refiner.py [--config resnet_refiner_config.json] [--device cuda] [--resume PATH]
"""
import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from resnet_refiner import DEFAULT_CONFIG_PATH, ResNetRefiner, build_model, load_config
from terrain_encoder import load_terrain_input


class EnsCGPDataset(Dataset):
    def __init__(self, posterior_path: Path, wrf_path: Path, indices: np.ndarray):
        self.posterior = np.load(posterior_path, mmap_mode="r")
        self.wrf = np.load(wrf_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx])
        # Writable copies: memmap slices are read-only, which torch warns/errors on.
        posterior = torch.from_numpy(np.array(self.posterior[sample_idx], dtype=np.float32, copy=True))
        wrf = torch.from_numpy(np.array(self.wrf[sample_idx], dtype=np.float32, copy=True))
        return posterior, wrf


class MeanAuxLosses:
    """High-frequency spectral L1, divergence ("gradient") L1, and high-quantile pinball
    losses computed exclusively on the predicted mean (mu_u, mu_v) vs. wrf_uv -- adapted
    from WeightedWindLoss in 26.3_wind/SWIN/train_wind_swin2sr.py (its spectral_high/
    sparse_grad/quantile terms only; this project's L1 and NLL already live in
    compute_weighted_loss below, so are not duplicated here). Stateless aside from a
    cache for the FFT frequency mask, which depends only on (H, W, device).
    """

    def __init__(self, spectral_low_freq_cutoff: float = 0.28, quantiles: tuple[float, ...] = (0.95, 0.99)):
        self.spectral_low_freq_cutoff = float(spectral_low_freq_cutoff)
        if not 0.0 < self.spectral_low_freq_cutoff < 1.0:
            raise ValueError(f"spectral_low_freq_cutoff must be in (0,1), got {self.spectral_low_freq_cutoff}")
        self.quantiles = tuple(float(q) for q in quantiles)
        for q in self.quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError(f"Quantiles must be in (0,1), got {q}")
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

    def quantile(self, mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean pinball (quantile) loss over self.quantiles: asymmetrically penalizes mu
        for under-predicting the q-th quantile of the (u, v) error more than over-
        predicting it, biasing mu toward the upper tail for high q."""
        if not self.quantiles:
            return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
        err = target - mu
        losses = [torch.mean(torch.maximum((q - 1.0) * err, q * err)) for q in self.quantiles]
        return torch.mean(torch.stack(losses))


def compute_weighted_loss(pred: torch.Tensor, target: torch.Tensor, weights: dict, mean_aux: MeanAuxLosses,
                           eps: float = 1e-6) -> dict:
    """pred: (B,5,H,W) [mu_u,mu_v,L11,L21,L22]; target: (B,2,H,W) [u,v].
    weights: {"nll","l1","spectral","gradient","quantile"} -> float.

    NLL is the exact bivariate-Gaussian negative log-likelihood under Sigma = L @ L.T,
    solved via the 2x2 lower-triangular system L z = (target - mu) rather than forming
    Sigma^-1 directly. Split into its two data-dependent terms (det(L) = L11*L22 since L
    is lower-triangular; log(2*pi) is a constant, folded into `nll` only):
      logdet      = log(L11) + log(L22)  = 0.5 * log(det(Sigma))
      mahalanobis = 0.5 * (z1^2 + z2^2)  = 0.5 * (target-mu)^T Sigma^-1 (target-mu)
      nll = logdet + mahalanobis + log(2*pi)
    spectral/gradient/quantile (see MeanAuxLosses) use mu only, never the covariance.

    Returns a dict with every component (all detached except "total", which carries the
    graph for backward()).
    """
    mu = pred[:, :2]
    L11 = pred[:, 2].clamp_min(eps)
    L21 = pred[:, 3]
    L22 = pred[:, 4].clamp_min(eps)

    e1 = target[:, 0] - mu[:, 0]
    e2 = target[:, 1] - mu[:, 1]
    z1 = e1 / L11
    z2 = (e2 - L21 * z1) / L22

    logdet = (torch.log(L11) + torch.log(L22)).mean()
    mahalanobis = (0.5 * (z1 ** 2 + z2 ** 2)).mean()
    nll = logdet + mahalanobis + math.log(2 * math.pi)
    l1 = F.l1_loss(mu, target)
    spectral = mean_aux.spectral(mu, target)
    gradient = mean_aux.gradient(mu, target)
    quantile = mean_aux.quantile(mu, target)

    total = (
        weights["nll"] * nll
        + weights["l1"] * l1
        + weights["spectral"] * spectral
        + weights["gradient"] * gradient
        + weights["quantile"] * quantile
    )
    return {
        "total": total,
        "nll": nll.detach(), "l1": l1.detach(),
        "logdet": logdet.detach(), "mahalanobis": mahalanobis.detach(),
        "spectral": spectral.detach(), "gradient": gradient.detach(), "quantile": quantile.detach(),
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
    logger = logging.getLogger("train_resnet_refiner")
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


LOSS_COMPONENT_KEYS = ("total", "nll", "l1", "logdet", "mahalanobis", "spectral", "gradient", "quantile")


def _refine(model: ResNetRefiner, posterior: torch.Tensor, terrain_raw: torch.Tensor | None) -> torch.Tensor:
    """Slices first_guess_mu/first_guess_chol from posterior at the call site and
    forwards them explicitly, per ResNetRefiner.forward()'s signature."""
    return model(posterior, posterior[:, :2], posterior[:, 2:5], terrain_raw)


@torch.no_grad()
def evaluate(model, loader, terrain_raw, device, weights: dict, mean_aux: MeanAuxLosses) -> dict:
    """Returns per-sample-averaged loss components (see compute_weighted_loss)."""
    model.eval()
    totals = {k: 0.0 for k in LOSS_COMPONENT_KEYS}
    n = 0
    for posterior, wrf in loader:
        posterior, wrf = posterior.to(device, non_blocking=True), wrf.to(device, non_blocking=True)
        pred = _refine(model, posterior, terrain_raw)
        loss_dict = compute_weighted_loss(pred, wrf, weights, mean_aux)
        b = posterior.shape[0]
        for k in LOSS_COMPONENT_KEYS:
            totals[k] += loss_dict[k].item() * b
        n += b
    model.train()
    return {k: v / n for k, v in totals.items()}


def train(config: dict, device_str: str, resume_path: Path | None = None, max_steps: int | None = None):
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
    wrf_path = data_dir / "wrf_uv.npy"
    train_ds = EnsCGPDataset(posterior_path, wrf_path, train_idx)
    val_ds = EnsCGPDataset(posterior_path, wrf_path, val_idx)

    t = config["training"]
    batch_size = t.get("batch_size", 4)
    num_workers = t.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = build_model(config).to(device)
    terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device) if model.use_terrain else None
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
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        logger.info("Resumed from %s at epoch %d (best_val_loss=%.5f)", resume_path, start_epoch, best_val_loss)

    weights = {
        "nll": t.get("nll_weight", 1.0),
        "l1": t.get("l1_weight", 1.0),
        "spectral": t.get("spectral_weight", 0.0),
        "gradient": t.get("gradient_weight", 0.0),
        "quantile": t.get("quantile_weight", 0.0),
    }
    mean_aux = MeanAuxLosses(
        spectral_low_freq_cutoff=t.get("spectral_low_freq_cutoff", 0.28),
        quantiles=tuple(t.get("quantiles", [0.95, 0.99])),
    )
    logger.info("Loss weights: %s", weights)
    logger.info(
        "Mean-aux loss params: spectral_low_freq_cutoff=%.3f, quantiles=%s",
        mean_aux.spectral_low_freq_cutoff, list(mean_aux.quantiles),
    )

    grad_clip_norm = t.get("grad_clip_norm", 1.0)
    log_every = config["logging"].get("log_every", 50)
    val_interval = t.get("val_interval", 1)

    global_step = 0
    model.train()
    for epoch in range(start_epoch, num_epochs):
        running = {k: 0.0 for k in LOSS_COMPONENT_KEYS}
        for i, (posterior, wrf) in enumerate(train_loader):
            posterior, wrf = posterior.to(device, non_blocking=True), wrf.to(device, non_blocking=True)

            pred = _refine(model, posterior, terrain_raw)
            loss_dict = compute_weighted_loss(pred, wrf, weights, mean_aux)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            for k in LOSS_COMPONENT_KEYS:
                running[k] += loss_dict[k].item()
            global_step += 1

            if (i + 1) % log_every == 0:
                logger.info(
                    "Epoch %d | Batch %d/%d | Loss %.5f (NLL %.5f [logdet %.5f, mahal %.5f], L1 %.5f, "
                    "Spectral %.5f, Gradient %.5f, Quantile %.5f) | LR %.2e",
                    epoch, i + 1, len(train_loader),
                    running["total"] / log_every, running["nll"] / log_every,
                    running["logdet"] / log_every, running["mahalanobis"] / log_every,
                    running["l1"] / log_every, running["spectral"] / log_every,
                    running["gradient"] / log_every, running["quantile"] / log_every,
                    scheduler.get_last_lr()[0],
                )
                running = {k: 0.0 for k in LOSS_COMPONENT_KEYS}

            if max_steps is not None and global_step >= max_steps:
                logger.info("Reached --max-steps=%d, stopping early.", max_steps)
                return

        if (epoch + 1) % val_interval == 0:
            val_metrics = evaluate(model, val_loader, terrain_raw, device, weights, mean_aux)
            val_loss = val_metrics["total"]
            logger.info(
                "Epoch %d | Val Loss %.5f (NLL %.5f [logdet %.5f, mahal %.5f], L1 %.5f, "
                "Spectral %.5f, Gradient %.5f, Quantile %.5f)",
                epoch, val_loss, val_metrics["nll"], val_metrics["logdet"], val_metrics["mahalanobis"],
                val_metrics["l1"], val_metrics["spectral"], val_metrics["gradient"], val_metrics["quantile"],
            )

            save_checkpoint(checkpoint_dir / "last.pth", model, optimizer, scheduler, epoch, best_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, scheduler, epoch, best_val_loss)
                logger.info("New best val loss: %.5f (saved checkpoints/best.pth)", best_val_loss)

    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop after this many optimizer steps -- for smoke-testing the pipeline, not real training runs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, device_str=args.device, resume_path=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
