"""
Neutral head-to-head scorecard for two quantile checkpoints on a held-out split.

WHY THIS EXISTS
The per-run `Val Loss` printed by train_new_enscgp_swin.py is NOT comparable across runs
that differ in `extreme_weight`: evaluate() passes the run's own extreme_cfg into
compute_weighted_loss, so the Pin term is extreme-weighted with that run's alpha /
apply_to_q10 / weighting implementation. v6-0714 used alpha=1.5, apply_to_q10=False and the
pre-12fc4bb split-sign weighting; v7 uses alpha=2.0, apply_to_q10=True and sign-aware
routing. Those two "Pin" columns measure different functionals, so their totals cannot be
differenced. Every metric HERE is defined by this script alone -- identical for both models,
independent of either run's loss config -- so the comparison is decidable.

WHAT IT MEASURES (per component, never pooled across u and v)
  Deterministic (q50): mae, rmse, bias
  Probabilistic:       crps (3-quantile approx), pinball_q10/q90 UNWEIGHTED, width (q90-q10)
  Calibration:         cov_q10/q50/q90 -- scored by |coverage - nominal|, not by direction
  Extreme tail:        the same metrics restricted to pixels where |wind| is at or above its
                       own per-sample 90th percentile (the exact mask
                       train_new_enscgp_swin.py::coverage_metrics uses, so *_ext here and
                       "extreme tail" in the training logs mean the same region).

AGGREGATION -- the load-bearing convention
Samples inside an event are near-duplicates (same storm, consecutive hours), so they are not
independent draws. Every number is aggregated at the EVENT level: metric summed over that
event's samples and pixels, divided by its own count -> ONE scalar per (event, component,
metric); then an unweighted mean over events. This mirrors bias_diagnostic_common's
convention, but note the difference from its compute_event_predictions(): that function
averages the FIELDS within an event, which is correct for a bias map and wrong here --
mean|q50 - y| is not |mean q50 - mean y|, and coverage of event-averaged fields is
meaningless. We aggregate the METRIC, not the fields.

RMSE is aggregated as MSE and square-rooted last (inside each bootstrap replicate), since
sqrt does not commute with the mean over events.

UNCERTAINTY
PAIRED bootstrap over events: one resample of event indices per replicate, applied to BOTH
models, so the CI is on the difference and the (large, shared) event-to-event variability
cancels. A difference is called only when its 95% CI excludes 0.

The two checkpoints need DIFFERENT model classes: 0714 predates the v7 simplification
(it has init_spread_scale / no offset_spread_init), so loading it with the current class
fails strict load_state_dict. bias_diagnostic_common already pins the v6 class from
_old_0714_arch/; the v7 class is loaded here from scripts/ under its own module name via
importlib, so the two never collide in sys.modules.

Usage:
    python eval_model_scorecard.py [--split test] [--device cuda:1] [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

# Importing this first pins sys.path (OLD_ARCH_DIR ahead of scripts/) and binds the v6 class.
import bias_diagnostic_common as common  # noqa: E402

REPO = Path("/home/peytonli/26.6_wind")
OUTPUT_DIR = REPO / "runs/0729_meangate/figures/scorecard"

# Model configs are taken from each run's own training log -- the checkpoints store only
# {epoch, best_val_loss, model_state_dict, optimizer_state_dict}, no config.
MODELS = {
    "v6-0714": {
        "checkpoint": REPO / "runs/0714/checkpoints/best.pth",
        "arch": "v6",
        "model_cfg": {
            "img_size": 200, "embed_dim": 96, "depths": [4, 4, 4], "num_heads": [6, 6, 6],
            "window_size": 8, "mlp_ratio": 4.0, "residual_base": "bicubic",
            "residual_gate_init": 0.1, "init_spread_scale": 1.28,
            "drop_rate": 0.05, "attn_drop_rate": 0.05, "drop_path_rate": 0.1,
            "head_dropout": 0.15,
        },
    },
    "v7-meangate": {
        "checkpoint": REPO / "runs/0729_meangate/checkpoints/best.pth",
        "arch": "v7",
        "model_cfg": {
            "img_size": 200, "embed_dim": 96, "depths": [4, 4, 4], "num_heads": [6, 6, 6],
            "window_size": 8, "mlp_ratio": 4.0, "residual_base": "bicubic",
            "residual_gate_init": 0.1, "offset_spread_init": 3.0,
            "drop_rate": 0.05, "attn_drop_rate": 0.05, "drop_path_rate": 0.1,
            "head_dropout": 0.15,
        },
    },
}

NOMINAL = {"cov_q10": 0.10, "cov_q50": 0.50, "cov_q90": 0.90}

# Accumulator keys. "_ext" twins are added programmatically -- see accumulate().
BASE_KEYS = ("abs_err", "sq_err", "err", "crps", "pin10", "pin90",
             "cov_q10", "cov_q50", "cov_q90", "width")

# (metric, how a DIFFERENCE is judged). "lower": B better if diff < 0. "calib": scored by
# |value - nominal|, so direction alone is meaningless. "none": diagnostic only, not scored.
METRIC_SENSE = {
    "mae": "lower", "rmse": "lower", "bias": "none", "crps": "lower",
    "pin10": "lower", "pin90": "lower", "width": "none",
    "cov_q10": "calib", "cov_q50": "calib", "cov_q90": "calib",
}


def load_v7_class():
    """Load scripts/new_enscgp_swin.py under its own module name so it cannot collide with
    the v6 module of the same basename that bias_diagnostic_common already imported."""
    path = REPO / "scripts/new_enscgp_swin.py"
    spec = importlib.util.spec_from_file_location("_arch_v7_new_enscgp_swin", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # its `from network_swin2sr import ...` resolves via scripts/
    return mod


_V7_MOD = None


def build_and_load(name: str, device):
    spec = MODELS[name]
    if spec["arch"] == "v6":
        model = common.build_model({"model": spec["model_cfg"]})
        slices = (common.ProbabilisticSwin2SR.Q10_SLICE,
                  common.ProbabilisticSwin2SR.Q50_SLICE,
                  common.ProbabilisticSwin2SR.Q90_SLICE)
    else:
        global _V7_MOD
        if _V7_MOD is None:
            _V7_MOD = load_v7_class()
        model = _V7_MOD.build_model({"model": spec["model_cfg"]})
        slices = (_V7_MOD.ProbabilisticSwin2SR.Q10_SLICE,
                  _V7_MOD.ProbabilisticSwin2SR.Q50_SLICE,
                  _V7_MOD.ProbabilisticSwin2SR.Q90_SLICE)
    model = model.to(device)
    ckpt = torch.load(spec["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])  # strict: a silent mismatch is a bug
    model.eval()
    return model, slices, ckpt


def pinball_t(q: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    err = y - q
    return torch.maximum(tau * err, (tau - 1.0) * err)


def extreme_mask(truth: torch.Tensor, ext_quantile: float = 0.9) -> torch.Tensor:
    """(B,2,H,W) bool. Exactly train_new_enscgp_swin.py::coverage_metrics' definition: pixels
    where the wind MAGNITUDE is at or above its own per-sample ext_quantile. The mask is
    shared by u and v (it marks a region of the storm, not a per-component condition); the
    metrics computed inside it are still strictly per component."""
    mag = torch.sqrt(truth[:, 0:1] ** 2 + truth[:, 1:2] ** 2 + 1e-6)
    thresh = torch.quantile(mag.reshape(mag.shape[0], -1), ext_quantile, dim=1)
    return (mag >= thresh.reshape(-1, 1, 1, 1)).expand_as(truth)


@torch.no_grad()
def event_metrics(model, slices, device, terrain_raw, split_map, split: str,
                  batch_size: int = 16) -> tuple[np.ndarray, dict]:
    """-> (event_ids (E,), {metric_key: (E,2) float64}).

    Per event: sum each metric over that event's samples and pixels, then divide by the
    matching count -> one scalar per (event, component). Counts differ between the full-grid
    and _ext metrics (and the _ext count varies per sample), so each family carries its own
    denominator rather than assuming a shared one.
    """
    q10_sl, q50_sl, q90_sl = slices
    posterior = np.load(common.DATA_DIR / "enscgp_posterior.npy", mmap_mode="r")
    bicubic = np.load(common.DATA_DIR / "era5_uv_2ch_bicubic.npy", mmap_mode="r")
    wrf = np.load(common.DATA_DIR / "wrf_uv.npy", mmap_mode="r")

    events = split_map[split]
    event_ids = np.array(sorted(events.keys()))
    keys = list(BASE_KEYS) + [k + "_ext" for k in BASE_KEYS]
    out = {k: np.zeros((len(event_ids), 2), dtype=np.float64) for k in keys}

    for ei, eid in enumerate(event_ids):
        idx = np.array(sorted(events[int(eid)]))
        acc = {k: torch.zeros(2, dtype=torch.float64, device=device) for k in keys}
        n_full = torch.zeros(2, dtype=torch.float64, device=device)
        n_ext = torch.zeros(2, dtype=torch.float64, device=device)

        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            post_b = torch.from_numpy(np.array(posterior[b], dtype=np.float32, copy=True)).to(device)
            bic_b = torch.from_numpy(np.array(bicubic[b], dtype=np.float32, copy=True)).to(device)
            y = torch.from_numpy(np.array(wrf[b], dtype=np.float32, copy=True)).to(device).double()

            pred = model(post_b, bic_b, terrain_raw)
            q10 = pred[:, q10_sl].double()
            q50 = pred[:, q50_sl].double()
            q90 = pred[:, q90_sl].double()

            p10, p50, p90 = (pinball_t(q10, y, 0.1), pinball_t(q50, y, 0.5), pinball_t(q90, y, 0.9))
            per_pixel = {
                "abs_err": (q50 - y).abs(),
                "sq_err": (q50 - y) ** 2,
                "err": q50 - y,
                # Same midpoint-rule 3-quantile CRPS as bias_diagnostic_common.crps_3q_approx
                # (weights 0.3/0.4/0.3, doubled) -- kept in torch to stay on the GPU.
                "crps": 2.0 * (0.3 * p10 + 0.4 * p50 + 0.3 * p90),
                "pin10": p10,
                "pin90": p90,
                "cov_q10": (y <= q10).double(),
                "cov_q50": (y <= q50).double(),
                "cov_q90": (y <= q90).double(),
                "width": q90 - q10,
            }
            ext = extreme_mask(y)
            for k, v in per_pixel.items():
                acc[k] += v.sum(dim=(0, 2, 3))
                acc[k + "_ext"] += (v * ext).sum(dim=(0, 2, 3))
            n_full += float(y.shape[0] * y.shape[2] * y.shape[3])
            n_ext += ext.sum(dim=(0, 2, 3)).double()

        for k in BASE_KEYS:
            out[k][ei] = (acc[k] / n_full).cpu().numpy()
            out[k + "_ext"][ei] = (acc[k + "_ext"] / n_ext.clamp_min(1.0)).cpu().numpy()

    return event_ids, out


# Metric name -> the accumulator holding its per-event values. Anything not listed is stored
# under its own name already (crps, pin10, pin90, width, cov_*).
_SRC = {"mae": "abs_err", "rmse": "sq_err", "bias": "err"}


def event_vector(per_event: dict, key: str, comp: int) -> np.ndarray:
    """The (E,) per-event values backing one metric, for one component. For rmse this is the
    per-event MSE -- the sqrt is applied AFTER averaging over events (see apply_reduction),
    because sqrt does not commute with the mean."""
    base, ext = (key[:-4], "_ext") if key.endswith("_ext") else (key, "")
    return per_event[_SRC.get(base, base) + ext][:, comp]


def apply_reduction(key: str, mean_over_events: np.ndarray | float):
    return np.sqrt(mean_over_events) if key.replace("_ext", "") == "rmse" else mean_over_events


def paired_bootstrap(a: dict, b: dict, keys: list[str], n_events: int,
                     n_boot: int, seed: int = 0) -> dict:
    """{(key, comp): (val_a, val_b, diff, lo, hi)} with a PAIRED event bootstrap: each
    replicate draws ONE set of event indices and applies it to both models, so shared
    event-to-event variability cancels out of the difference."""
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_events, size=(n_boot, n_events))
    res = {}
    for key in keys:
        for comp in (0, 1):
            xa, xb = event_vector(a, key, comp), event_vector(b, key, comp)
            va = float(apply_reduction(key, xa.mean()))
            vb = float(apply_reduction(key, xb.mean()))
            diffs = (apply_reduction(key, xb[boot_idx].mean(axis=1))
                     - apply_reduction(key, xa[boot_idx].mean(axis=1)))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            res[(key, comp)] = (va, vb, vb - va, float(lo), float(hi))
    return res


def verdict(key: str, va: float, vb: float, lo: float, hi: float, name_a: str, name_b: str) -> str:
    """A call is made only when the 95% CI on the difference excludes 0. Calibration metrics
    are judged on |value - nominal| (closer is better), not on the sign of the difference."""
    sense = METRIC_SENSE.get(key.replace("_ext", ""), "none")
    significant = (lo > 0) or (hi < 0)
    if sense == "none":
        return "--"
    if not significant:
        return "ns"
    if sense == "calib":
        nominal = NOMINAL[key.replace("_ext", "")]
        return name_b if abs(vb - nominal) < abs(va - nominal) else name_a
    return name_b if vb < va else name_a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=("test", "val", "train"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    terrain_raw = common.load_terrain(device)
    split_map = common.load_event_split_map()
    n_events = len(split_map[args.split])
    names = list(MODELS.keys())
    print(f"split={args.split}  events={n_events}  device={args.device}  n_boot={args.n_boot}")

    per_event = {}
    for name in names:
        model, slices, ckpt = build_and_load(name, device)
        gate = model.mean_gate.item() if hasattr(model, "mean_gate") else float("nan")
        print(f"  {name}: epoch {ckpt['epoch']}, val {ckpt['best_val_loss']:.5f}, mean_gate {gate:.4f}")
        eids, per_event[name] = event_metrics(model, slices, device, terrain_raw,
                                              split_map, args.split, args.batch_size)
        del model
        torch.cuda.empty_cache()

    keys = ["mae", "rmse", "bias", "crps", "pin10", "pin90", "width",
            "cov_q10", "cov_q50", "cov_q90"]
    keys = keys + [k + "_ext" for k in keys]
    res = paired_bootstrap(per_event[names[0]], per_event[names[1]], keys,
                           n_events, args.n_boot, args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"scorecard_{args.split}.csv"
    hdr = f"{'metric':<14}{'cmp':<5}{names[0]:>12}{names[1]:>14}{'diff':>11}{'95% CI':>22}  verdict"
    print("\n" + hdr)
    print("-" * len(hdr))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "component", names[0], names[1], "diff", "ci_lo", "ci_hi", "verdict"])
        for key in keys:
            for comp, cname in enumerate(common.COMPONENTS):
                va, vb, d, lo, hi = res[(key, comp)]
                v = verdict(key, va, vb, lo, hi, names[0], names[1])
                w.writerow([key, cname, f"{va:.6f}", f"{vb:.6f}", f"{d:.6f}",
                            f"{lo:.6f}", f"{hi:.6f}", v])
                print(f"{key:<14}{cname:<5}{va:>12.4f}{vb:>14.4f}{d:>11.4f}"
                      f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}  {v}")
            print()

    # Headline tally over the metrics that have a defined direction.
    scored = [k for k in keys if METRIC_SENSE.get(k.replace("_ext", ""), "none") != "none"]
    tally = {names[0]: 0, names[1]: 0, "ns": 0}
    for key in scored:
        for comp in (0, 1):
            va, vb, d, lo, hi = res[(key, comp)]
            tally[verdict(key, va, vb, lo, hi, names[0], names[1])] += 1
    total = sum(tally.values())
    print(f"Scored comparisons (metric x component): {total}")
    for k, v in tally.items():
        print(f"  {k:<14} {v}")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
