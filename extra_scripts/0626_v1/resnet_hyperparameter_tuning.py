"""Optuna (TPE sampler + median pruning) Bayesian search over ResNetRefiner loss
weights, scored on stratified validation CRPS.

Each trial fine-tunes from a single shared baseline checkpoint (--resume) for up to
--epochs-per-eval epochs with a trial-sampled (spectral, quantile, gradient, nll)
weight combination (l1 anchored at 1.0), reporting CRPS after every epoch so Optuna's
pruner can cut off unpromising trials early. This replaces a prior coordinate-descent
search (see git history) -- Optuna's sampler/pruner subsume the step-size/accept-k
machinery that drove, since trials are independent draws rather than local steps from
a shared "current best".

CRPS: closed-form Gaussian CRPS per pixel from the model's own predicted marginals
(sigma_u = L11, sigma_v = sqrt(L21^2 + L22^2)), summed over u+v -- the CRPS analogue of
this project's existing comp_rmse = u_rmse + v_rmse convention. Stratified into:
  bulk_crps:    mean CRPS over every validation pixel.
  extreme_crps: mean CRPS restricted to each sample's OWN windiest top-5% pixels (by
                WRF truth wind-speed magnitude) -- a per-sample threshold, not one
                threshold pooled across the whole validation set. (This deliberately
                diverges from variance_recalibration.py's pooled-threshold convention,
                which needs a single serializable threshold to re-apply at inference
                time on new, individual samples; this search has the whole val set
                available up front every time, so "windiest part of each event" is the
                more natural per-trial reading of "extreme".)
  crps_total = bulk_crps + extreme_crps -- the value Optuna minimizes.
"""
import argparse
import atexit
import copy
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path("/home/peytonli/26.6_wind/scripts")))
from resnet_refiner import ResNetRefiner, build_model
from terrain_encoder import load_terrain_input
from train_resnet_refiner import EnsCGPDataset, MeanAuxLosses, compute_weighted_loss

SEARCH_PARAMS = ("spectral", "quantile", "gradient", "nll")
EXTREME_PCT = 5.0  # "top 5% extreme", per-sample
SQRT_TWO = math.sqrt(2.0)
SQRT_PI = math.sqrt(math.pi)


def parse_dict_arg(arg_str: str) -> dict[str, float]:
    """Parse key=value,key=value string into a dictionary of floats."""
    result = {}
    for pair in arg_str.split(','):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split('=')
        result[k.strip()] = float(v.strip())
    return result


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_crps(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Closed-form CRPS of N(mu, sigma^2) evaluated at y (Gneiting & Raftery 2007):
    sigma * [z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)], z = (y-mu)/sigma."""
    sigma = sigma.clamp_min(eps)
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / SQRT_TWO))
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / SQRT_PI)


def search_bounds(start: float, factor: float, floor: float) -> tuple[float, float]:
    """Log-uniform bounds [start/factor, start*factor], floored away from 0 so a
    starting weight of 0 doesn't collapse the search range."""
    low = max(start / factor, floor)
    high = max(start * factor, floor * factor)
    return low, high


def build_components(config: dict, device: torch.device):
    paths = config["paths"]
    data_dir = Path(paths["data_dir"])
    splits_path = Path(paths["splits_path"])
    t = config["training"]

    splits = np.load(splits_path)
    train_idx, val_idx = splits["train_idx"], splits["val_idx"]

    posterior_path = data_dir / "enscgp_posterior.npy"
    wrf_path = data_dir / "wrf_uv.npy"
    train_ds = EnsCGPDataset(posterior_path, wrf_path, train_idx)
    val_ds = EnsCGPDataset(posterior_path, wrf_path, val_idx)

    batch_size = t.get("batch_size", 4)
    num_workers = t.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = build_model(config).to(device)
    terrain_raw = load_terrain_input(data_dir).unsqueeze(0).to(device) if model.use_terrain else None

    mean_aux = MeanAuxLosses(
        spectral_low_freq_cutoff=t.get("spectral_low_freq_cutoff", 0.28),
        quantiles=tuple(t.get("quantiles", [0.95, 0.99])),
    )
    return model, terrain_raw, train_loader, val_loader, mean_aux


def build_optimizer_scheduler(model: ResNetRefiner, config: dict, steps_per_epoch: int):
    t = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=t.get("learning_rate", 1e-3),
                                  weight_decay=t.get("weight_decay", 0.0))
    num_epochs = t.get("num_epochs", 100)
    total_steps = num_epochs * steps_per_epoch
    lr_milestones = [int(f * total_steps) for f in t.get("lr_milestone_fractions", [0.5, 0.75, 0.9])]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=t.get("lr_gamma", 0.5))
    return optimizer, scheduler


def snapshot_state(model: ResNetRefiner, optimizer, scheduler):
    return (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
    )


def restore_state(model: ResNetRefiner, optimizer, scheduler, snapshot):
    # model.load_state_dict copies tensor VALUES (no aliasing), so the snapshot's
    # model_state needs no defensive copy. optimizer/scheduler load_state_dict instead
    # stores tensor REFERENCES, and Adam's in-place ops (mul_, addcdiv_) would corrupt
    # the snapshot through those aliases on the next trial -- deepcopy before loading.
    model_state, opt_state, sched_state = snapshot
    model.load_state_dict(model_state)
    optimizer.load_state_dict(copy.deepcopy(opt_state))
    scheduler.load_state_dict(copy.deepcopy(sched_state))


def train_one_epoch(model: ResNetRefiner, loader, terrain_raw, device, weights, mean_aux, optimizer, scheduler,
                     grad_clip_norm: float):
    model.train()
    for posterior, wrf in loader:
        posterior = posterior.to(device, non_blocking=True)
        wrf = wrf.to(device, non_blocking=True)
        pred = model(posterior, posterior[:, :2], posterior[:, 2:5], terrain_raw)
        loss_dict = compute_weighted_loss(pred, wrf, weights, mean_aux)

        optimizer.zero_grad()
        loss_dict["total"].backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def validate_crps(model: ResNetRefiner, loader, terrain_raw, device) -> dict:
    model.eval()
    bulk_sum, bulk_n = 0.0, 0
    extreme_sum, extreme_n = 0.0, 0
    sq_err_u, sq_err_v, n_px = 0.0, 0.0, 0

    for posterior, wrf in loader:
        posterior = posterior.to(device, non_blocking=True)
        wrf = wrf.to(device, non_blocking=True)
        pred = model(posterior, posterior[:, :2], posterior[:, 2:5], terrain_raw)

        mu_u, mu_v = pred[:, 0], pred[:, 1]
        L11, L21, L22 = pred[:, 2], pred[:, 3], pred[:, 4]
        sigma_u = L11
        sigma_v = torch.sqrt(L21 ** 2 + L22 ** 2)
        true_u, true_v = wrf[:, 0], wrf[:, 1]

        pixel_crps = gaussian_crps(true_u, mu_u, sigma_u) + gaussian_crps(true_v, mu_v, sigma_v)  # (B, H, W)
        bulk_sum += pixel_crps.sum().item()
        bulk_n += pixel_crps.numel()

        b = pixel_crps.shape[0]
        speed = torch.sqrt(true_u ** 2 + true_v ** 2)
        threshold = torch.quantile(speed.reshape(b, -1), 1.0 - EXTREME_PCT / 100.0, dim=1)  # (B,), per-sample
        extreme_mask = speed >= threshold.view(b, 1, 1)
        extreme_sum += pixel_crps[extreme_mask].sum().item()
        extreme_n += int(extreme_mask.sum().item())

        sq_err_u += torch.sum((mu_u - true_u) ** 2).item()
        sq_err_v += torch.sum((mu_v - true_v) ** 2).item()
        n_px += mu_u.numel()

    model.train()
    bulk_crps = bulk_sum / bulk_n
    extreme_crps = extreme_sum / extreme_n
    return {
        "bulk_crps": bulk_crps,
        "extreme_crps": extreme_crps,
        "crps_total": bulk_crps + extreme_crps,
        "u_rmse": math.sqrt(sq_err_u / n_px),
        "v_rmse": math.sqrt(sq_err_v / n_px),
    }


def save_best_checkpoint(model: ResNetRefiner, optimizer, scheduler, config: dict, output_dir: str,
                          weights: dict, metrics: dict, epoch: int):
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "weights": weights,
        "val_crps_bulk": metrics["bulk_crps"],
        "val_crps_extreme": metrics["extreme_crps"],
        "val_crps_total": metrics["crps_total"],
    }
    torch.save(state, ckpt_dir / "search_resume_state.pth")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna (TPE + median pruning) Bayesian search for ResNetRefiner loss weights, "
                     "scored on stratified validation CRPS (bulk + per-sample top-5% extreme)."
    )
    parser.add_argument("--starting-weights", type=str, default=None,
                         help="l1=...,spectral=...,quantile=...,gradient=...,nll=... "
                              "Used to center the search ranges and as the first (enqueued) trial.")
    parser.add_argument("--epochs-per-eval", type=int, default=7,
                         help="Epochs to fine-tune per trial; also the pruning horizon.")
    parser.add_argument("--num-eval-seeds", type=int, default=3,
                         help="Random seeds averaged per trial for a stable CRPS estimate. Each seed runs the "
                              "full epochs-per-eval fine-tune independently from the same base checkpoint; "
                              "pruning checks fire after every (seed, epoch) using the running cross-seed average.")
    parser.add_argument("--num-trials", type=int, default=50, help="Additional trials to run this invocation.")
    parser.add_argument("--search-range-factor", type=float, default=50.0,
                         help="Log-uniform search bounds = [start/factor, start*factor] per weight.")
    parser.add_argument("--search-min-weight", type=float, default=1e-3,
                         help="Floor for search bounds, so a starting weight of 0 doesn't collapse the range.")
    parser.add_argument("--base-config", type=str, default="/home/peytonli/26.6_wind/scripts/resnet_refiner_config.json", help="Base config json file.")
    parser.add_argument("--output-dir", type=str, default="/home/peytonli/26.6_wind/logs/resnet_refiner",
                         help="Directory to save search results.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.base_config, "r") as f:
        config_data = json.load(f)

    loss_cfg = config_data.get("training", {})
    if not args.starting_weights:
        weights = {
            "l1": float(loss_cfg.get("l1_weight", 1.0)),
            "spectral": float(loss_cfg.get("spectral_weight", 1.0)),
            "quantile": float(loss_cfg.get("quantile_weight", 1.0)),
            "gradient": float(loss_cfg.get("gradient_weight", 1.0)),
            "nll": float(loss_cfg.get("nll_weight", 0.5)),
        }
    else:
        weights = parse_dict_arg(args.starting_weights)
    weights["l1"] = 1.0

    if "logging" not in config_data:
        config_data["logging"] = {}
    config_data["logging"]["log_dir"] = args.output_dir

    stdout_log_path = os.path.join(args.output_dir, "search_stdout.log")
    stdout_log_file = open(stdout_log_path, "a", buffering=1)
    atexit.register(stdout_log_file.close)

    def search_log(message: str):
        print(message)
        stdout_log_file.write(message + "\n")

    # Scope the scheduler to a short, fresh fine-tune horizon rather than the
    # original full training horizon -- mirrors --resume loading model WEIGHTS ONLY
    # (optimizer/scheduler are built fresh, not resumed; see main()'s build call below).
    if "training" not in config_data:
        config_data["training"] = {}
    config_data["training"]["num_epochs"] = args.epochs_per_eval

    device = torch.device(args.device)
    model, terrain_raw, train_loader, val_loader, mean_aux = build_components(config_data, device)
    optimizer, scheduler = build_optimizer_scheduler(model, config_data, steps_per_epoch=len(train_loader))

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    # else: keep build_model's own fresh init (small-random-near-zero heads).
    base_snapshot = snapshot_state(model, optimizer, scheduler)

    base_seed = config_data.get("seed", 0)
    grad_clip_norm = config_data["training"].get("grad_clip_norm", 1.0)

    search_csv_path = os.path.join(args.output_dir, "search_history.csv")
    csv_headers = [
        "trial", "state", "l1", "spectral", "quantile", "gradient", "nll",
        "seed", "epoch", "bulk_crps", "extreme_crps", "crps_total", "crps_total_std", "crps_total_sem",
        "u_rmse", "v_rmse", "num_eval_seeds", "is_best",
    ]
    history_exists = os.path.exists(search_csv_path)
    history_has_content = history_exists and os.path.getsize(search_csv_path) > 0
    if not history_exists or not history_has_content:
        with open(search_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_headers)

    def append_trial_log(trial_number: int, state: str, w: dict, seed_idx: int, epoch: int, metrics: dict,
                          is_best: bool):
        with open(search_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                trial_number, state, w.get("l1", 1.0), w["spectral"], w["quantile"], w["gradient"], w["nll"],
                seed_idx, epoch, metrics["bulk_crps"], metrics["extreme_crps"], metrics["crps_total"],
                metrics.get("crps_total_std", ""), metrics.get("crps_total_sem", ""),
                metrics["u_rmse"], metrics["v_rmse"], metrics.get("num_eval_seeds", ""), is_best,
            ])

    best_score = math.inf

    def run_trial_training(trial_weights: dict, trial: optuna.Trial, label: str) -> tuple[dict, tuple]:
        """Runs --num-eval-seeds independent fine-tunes (each up to epochs-per-eval
        epochs, reset to base_snapshot, seeded base_seed+seed_idx) and averages their
        final CRPS for a noise-reduced trial score -- the multi-seed-averaging
        counterpart of the old coordinate-descent script's evaluate(). Pruning checks
        fire after every (seed, epoch), reporting the running cross-seed average of
        whatever's completed so far (a smoothing proxy: seeds not yet started
        contribute nothing, the in-progress seed contributes its current partial
        epoch), so a bad trial can still be cut off mid-seed rather than only at
        seed boundaries. Returns (metrics averaged across seeds, seed 0's
        post-training (model, optimizer, scheduler) state -- the canonical candidate
        checkpoint if this trial becomes the new best; seed 0 rather than the best- or
        last-seed avoids cherry-picking a lucky noise draw)."""
        seed_finals: list[dict] = []
        seed0_snapshot = None

        for seed_idx in range(args.num_eval_seeds):
            restore_state(model, optimizer, scheduler, base_snapshot)
            _seed_everything(base_seed + seed_idx)

            metrics = None
            for epoch in range(args.epochs_per_eval):
                train_one_epoch(model, train_loader, terrain_raw, device, trial_weights, mean_aux,
                                 optimizer, scheduler, grad_clip_norm)
                metrics = validate_crps(model, val_loader, terrain_raw, device)
                running = [m["crps_total"] for m in seed_finals] + [metrics["crps_total"]]
                running_avg = float(np.mean(running))
                step = seed_idx * args.epochs_per_eval + epoch
                search_log(
                    f"  [{label}] seed {seed_idx + 1}/{args.num_eval_seeds} epoch {epoch + 1}/{args.epochs_per_eval} "
                    f"crps_total={metrics['crps_total']:.5f} running_avg={running_avg:.5f}"
                )
                trial.report(running_avg, step=step)
                if trial.should_prune():
                    append_trial_log(trial.number, "PRUNED", trial_weights, seed_idx, epoch, metrics, is_best=False)
                    raise optuna.TrialPruned()

            seed_finals.append(metrics)
            if seed_idx == 0:
                seed0_snapshot = snapshot_state(model, optimizer, scheduler)

        crps_vals = [m["crps_total"] for m in seed_finals]
        averaged = {
            key: float(np.mean([m[key] for m in seed_finals]))
            for key in ("bulk_crps", "extreme_crps", "crps_total", "u_rmse", "v_rmse")
        }
        averaged["crps_total_std"] = float(np.std(crps_vals, ddof=1)) if len(crps_vals) > 1 else 0.0
        averaged["crps_total_sem"] = averaged["crps_total_std"] / math.sqrt(len(crps_vals))
        averaged["num_eval_seeds"] = args.num_eval_seeds
        return averaged, seed0_snapshot

    def objective(trial: optuna.Trial) -> float:
        trial_weights = {"l1": 1.0}
        for name in SEARCH_PARAMS:
            low, high = search_bounds(weights[name], args.search_range_factor, args.search_min_weight)
            trial_weights[name] = trial.suggest_float(name, low, high, log=True)

        metrics, seed0_snapshot = run_trial_training(trial_weights, trial, label=f"trial {trial.number}")
        score = metrics["crps_total"]
        search_log(
            f"  [trial {trial.number}] crps_total={score:.5f} +/-{metrics['crps_total_std']:.5f} "
            f"(sem {metrics['crps_total_sem']:.5f}, n={metrics['num_eval_seeds']})"
        )

        nonlocal best_score
        is_best = score < best_score
        if is_best:
            best_score = score
            restore_state(model, optimizer, scheduler, seed0_snapshot)
            save_best_checkpoint(model, optimizer, scheduler, config_data, args.output_dir,
                                  trial_weights, metrics, epoch=args.epochs_per_eval)
            search_log(f"  [trial {trial.number}] NEW BEST crps_total={score:.5f} weights={trial_weights}")

        append_trial_log(trial.number, "COMPLETE", trial_weights, args.num_eval_seeds - 1, args.epochs_per_eval - 1,
                          metrics, is_best)
        return score

    search_log("=== Starting Optuna Bayesian Search (TPE + median pruning) ===")
    search_log(f"Starting weights: {weights}")

    storage = f"sqlite:///{os.path.join(args.output_dir, 'optuna_study.db')}"
    study = optuna.create_study(
        study_name="resnet_loss_weight_search",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    if len(study.trials) == 0:
        study.enqueue_trial({name: weights[name] for name in SEARCH_PARAMS})

    study.optimize(objective, n_trials=args.num_trials)

    search_log("=== Search Complete ===")
    best_trial = study.best_trial
    best_weights = {"l1": 1.0, **best_trial.params}
    search_log(f"Best trial: #{best_trial.number}")
    search_log(f"Best weights: {best_weights}")
    search_log(f"Best crps_total: {study.best_value:.5f}")


if __name__ == "__main__":
    main()
