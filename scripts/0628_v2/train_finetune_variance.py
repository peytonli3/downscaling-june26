"""Fine-tune ONLY the variance (Cholesky) head of a trained ProbabilisticSwin2SR, with
the head newly conditioned on placement-uncertainty features (new_enscgp_swin.py's
variance_conditioning). Continues FROM an existing checkpoint -- the mean head, backbone,
terrain encoder, and loss are untouched; only chol_head / chol_cond / chol_gate train.

Why: the mean/spectrum are already good, but the model makes coherent "wrong-bet" blotches
where it misplaces mesoscale structure, and the predicted sigma does NOT inflate there. The
new conditioning feeds the variance head the EnsCGP analog disagreement (prior spread), the
mean-wind-speed gradient, and terrain slope -- signals of WHERE placement is uncertain -- so
it can learn to raise sigma on the blotches. The new conv (chol_cond) is zero-initialized,
so at load the model output is IDENTICAL to the checkpoint; fine-tuning then teaches
chol_cond to use the features. NLL stays the variance's only training signal.

Setup:
- Loads --checkpoint (default <log_dir>/checkpoints/best.pth) with strict=False: the only
  missing keys are chol_cond.* (asserted -- any other missing/unexpected key is a bug).
- Forces model.variance_conditioning on regardless of the config flag, and loads
  cond_feature_stats.npz (precompute_prior_spread.py) into the standardization buffers.
- Freezes every parameter except chol_head.* / chol_cond.* / chol_gate.
- Runs the whole model in eval() mode so the FROZEN backbone/mean/terrain are deterministic
  (no backbone dropout, frozen BN running stats); autograd still flows to the trainable
  variance params. (Consequence: head_dropout is inactive during this phase -- the tiny
  trainable param count + frozen backbone are the regularizer.)
- Loss = NLL only (mean is frozen, so multiscale/L1/etc. would be constant -- zeroed).

Inputs per sample (data_dir): enscgp_posterior.npy, era5_uv_2ch_bicubic.npy, wrf_uv.npy
(as in train_new_enscgp_swin.py) plus enscgp_prior_spread.npy (precompute_prior_spread.py).

Saves checkpoints/best_variance.pth and last_variance.pth (full model state_dict, so the
conditioned model can be reloaded for eval with variance_conditioning on).

Usage:
    python train_finetune_variance.py [--config new_enscgp_swin_config.json] \
        [--checkpoint .../best.pth] [--num_epochs 30] [--lr 1e-4] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from new_enscgp_swin import DEFAULT_CONFIG_PATH, build_model, load_config
from terrain_encoder import load_terrain_input
from train_new_enscgp_swin import (
    LOSS_COMPONENT_KEYS,
    MeanAuxLosses,
    compute_weighted_loss,
    save_checkpoint,
    setup_logging,
)

# Parameters that train (the variance head); everything else is frozen.
TRAINABLE_PREFIXES = ("chol_head.", "chol_cond.")
TRAINABLE_EXACT = ("chol_gate",)


def is_trainable(name: str) -> bool:
    return name in TRAINABLE_EXACT or any(name.startswith(p) for p in TRAINABLE_PREFIXES)


class ConditionedDataset(Dataset):
    """Returns (posterior, bicubic, wrf, prior_spread) for each index -- the EnsCGPSwinDataset
    triple plus the per-pixel analog prior spread the variance head conditions on."""

    def __init__(self, posterior_path, bicubic_path, wrf_path, prior_spread_path, indices):
        self.posterior = np.load(posterior_path, mmap_mode="r")
        self.bicubic = np.load(bicubic_path, mmap_mode="r")
        self.wrf = np.load(wrf_path, mmap_mode="r")
        self.prior_spread = np.load(prior_spread_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        posterior = torch.from_numpy(np.array(self.posterior[i], dtype=np.float32, copy=True))
        bicubic = torch.from_numpy(np.array(self.bicubic[i], dtype=np.float32, copy=True))
        wrf = torch.from_numpy(np.array(self.wrf[i], dtype=np.float32, copy=True))
        prior_spread = torch.from_numpy(np.array(self.prior_spread[i], dtype=np.float32, copy=True))
        return posterior, bicubic, wrf, prior_spread


def build_conditioned_model(config: dict, checkpoint_path: Path, stats_path: Path, device, logger):
    """Build the model with variance_conditioning forced on, load the checkpoint
    (strict=False, asserting only chol_cond.* is missing), and load the cond stats."""
    config = json.loads(json.dumps(config))  # deep copy; don't mutate caller's dict
    config["model"]["variance_conditioning"] = True
    model = build_model(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    allowed_missing = {n for n, _ in model.named_parameters() if n.startswith("chol_cond.")}
    allowed_missing |= {"cond_mean", "cond_std"}  # buffers, also absent from the old checkpoint
    unexpected_missing = set(missing) - allowed_missing
    if unexpected_missing:
        raise RuntimeError(f"Unexpected missing keys loading {checkpoint_path}: {sorted(unexpected_missing)}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in {checkpoint_path}: {sorted(unexpected)}")
    logger.info("Loaded %s (epoch %s); chol_cond/cond buffers initialized fresh (zero-init no-op).",
                checkpoint_path, ckpt.get("epoch"))

    stats = np.load(stats_path)
    model.load_cond_stats(stats["cond_mean"], stats["cond_std"])
    logger.info("Loaded cond_feature_stats: mean=%s std=%s",
                stats["cond_mean"].tolist(), stats["cond_std"].tolist())
    return model


def freeze_all_but_variance(model, logger):
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        if is_trainable(name):
            p.requires_grad_(True)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
            n_freeze += p.numel()
    logger.info("Trainable (variance head): %d params; frozen: %d params", n_train, n_freeze)
    if n_train == 0:
        raise RuntimeError("No trainable parameters matched the variance head -- check TRAINABLE_PREFIXES")


@torch.no_grad()
def evaluate(model, loader, terrain_raw, device, weights, mean_aux):
    totals = {k: 0.0 for k in LOSS_COMPONENT_KEYS}
    n = 0
    for posterior, bicubic, wrf, prior_spread in loader:
        posterior = posterior.to(device, non_blocking=True)
        bicubic = bicubic.to(device, non_blocking=True)
        wrf = wrf.to(device, non_blocking=True)
        prior_spread = prior_spread.to(device, non_blocking=True)
        pred = model(posterior, bicubic, terrain_raw, prior_spread)
        loss_dict = compute_weighted_loss(pred, wrf, weights, mean_aux, ms_loss_fn=None)
        b = posterior.shape[0]
        for k in LOSS_COMPONENT_KEYS:
            totals[k] += loss_dict[k].item() * b
        n += b
    return {k: v / n for k, v in totals.items()}


def train(config: dict, args):
    paths = config["paths"]
    data_dir = Path(args.data_dir) if args.data_dir else Path(paths["data_dir"])
    log_dir = Path(paths["log_dir"])
    splits_path = Path(paths["splits_path"])

    logger = setup_logging(log_dir)
    logger.info("Variance-head fine-tune. Config:\n%s", json.dumps(config, indent=2))

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    torch.manual_seed(config.get("seed", 0))
    np.random.seed(config.get("seed", 0))

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (log_dir / "checkpoints" / "best.pth")
    stats_path = Path(args.stats_path) if args.stats_path else (data_dir / "cond_feature_stats.npz")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"cond_feature_stats not found: {stats_path} -- run precompute_prior_spread.py")

    model = build_conditioned_model(config, checkpoint_path, stats_path, device, logger)
    freeze_all_but_variance(model, logger)
    # Whole model in eval() so the frozen backbone/mean/terrain are deterministic; autograd
    # still flows to the (unfrozen) variance params. Never switched back to train().
    model.eval()
    terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device)

    splits = np.load(splits_path)
    train_idx, val_idx = splits["train_idx"], splits["val_idx"]
    posterior_path = data_dir / "enscgp_posterior.npy"
    bicubic_path = data_dir / "era5_uv_2ch_bicubic.npy"
    wrf_path = data_dir / "wrf_uv.npy"
    prior_spread_path = data_dir / "enscgp_prior_spread.npy"
    if not prior_spread_path.exists():
        raise FileNotFoundError(f"{prior_spread_path} not found -- run precompute_prior_spread.py")

    t = config["training"]
    batch_size = args.batch_size or t.get("batch_size", 4)
    num_workers = t.get("num_workers", 2)
    train_ds = ConditionedDataset(posterior_path, bicubic_path, wrf_path, prior_spread_path, train_idx)
    val_ds = ConditionedDataset(posterior_path, bicubic_path, wrf_path, prior_spread_path, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # NLL-only: the mean is frozen, so the mean-supervision terms would be constant. mean_aux
    # is still constructed (compute_weighted_loss needs it) but its terms carry weight 0.
    weights = {"nll": t.get("nll_weight", 1.0), "ms": 0.0, "l1": 0.0,
               "spectral": 0.0, "gradient": 0.0, "quantile": 0.0}
    mean_aux = MeanAuxLosses(
        spectral_low_freq_cutoff=t.get("spectral_low_freq_cutoff", 0.28),
        quantiles=tuple(t.get("quantiles", [0.95, 0.99])),
    )
    logger.info("Loss weights (NLL-only fine-tune): %s", weights)

    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else t.get("learning_rate", 1e-4)
    weight_decay = args.weight_decay if args.weight_decay is not None else t.get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

    num_epochs = args.num_epochs or t.get("num_epochs", 30)
    grad_clip_norm = t.get("grad_clip_norm", 1.0)
    log_every = config["logging"].get("log_every", 50)
    early_stop_patience = t.get("early_stop_patience", 0)
    logger.info("LR=%.2e, weight_decay=%.2e, num_epochs=%d, batch_size=%d, early_stop_patience=%d",
                lr, weight_decay, num_epochs, batch_size, early_stop_patience)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(num_epochs):
        running = {k: 0.0 for k in LOSS_COMPONENT_KEYS}
        for i, (posterior, bicubic, wrf, prior_spread) in enumerate(train_loader):
            posterior = posterior.to(device, non_blocking=True)
            bicubic = bicubic.to(device, non_blocking=True)
            wrf = wrf.to(device, non_blocking=True)
            prior_spread = prior_spread.to(device, non_blocking=True)

            pred = model(posterior, bicubic, terrain_raw, prior_spread)
            loss_dict = compute_weighted_loss(pred, wrf, weights, mean_aux, ms_loss_fn=None)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            optimizer.step()

            for k in LOSS_COMPONENT_KEYS:
                running[k] += loss_dict[k].item()
            if (i + 1) % log_every == 0:
                logger.info(
                    "Epoch %d | Batch %d/%d | Loss %.5f (NLL %.5f [logdet %.5f, mahal %.5f]) | chol_gate %.4f",
                    epoch, i + 1, len(train_loader), running["total"] / log_every, running["nll"] / log_every,
                    running["logdet"] / log_every, running["mahalanobis"] / log_every, model.chol_gate.item(),
                )
                running = {k: 0.0 for k in LOSS_COMPONENT_KEYS}

        val_metrics = evaluate(model, val_loader, terrain_raw, device, weights, mean_aux)
        val_loss = val_metrics["total"]
        logger.info("Epoch %d | Val Loss %.5f (NLL %.5f [logdet %.5f, mahal %.5f])",
                    epoch, val_loss, val_metrics["nll"], val_metrics["logdet"], val_metrics["mahalanobis"])

        save_checkpoint(checkpoint_dir / "last_variance.pth", model, optimizer,
                        _DummyScheduler(), epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(checkpoint_dir / "best_variance.pth", model, optimizer,
                            _DummyScheduler(), epoch, best_val_loss)
            logger.info("New best val loss: %.5f (saved checkpoints/best_variance.pth)", best_val_loss)
        else:
            no_improve += 1
            if early_stop_patience and no_improve >= early_stop_patience:
                logger.info("Early stopping: %d epochs without improvement.", no_improve)
                break

    logger.info("Variance-head fine-tune complete. Best val loss: %.5f", best_val_loss)


class _DummyScheduler:
    """save_checkpoint expects a scheduler with state_dict(); this fine-tune uses none."""
    def state_dict(self):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <log_dir>/checkpoints/best.pth")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--stats_path", type=Path, default=None, help="Defaults to <data_dir>/cond_feature_stats.npz")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Defaults to training.learning_rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Defaults to training.weight_decay")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
