#!/usr/bin/env python3
"""
The README results table: MAE and RMSE for every candidate predictor, on all pixels and on
the windiest N% of each sample's pixels, computed ONE way over ONE sample set.

WHY THIS EXISTS
The four rows of that table used to come from two different scripts with two different
aggregation conventions and two different sample sets: the baselines from
`oneoff/mae_baseline_sweep.py` (pooled over a 128-sample draw) and the model row from
`oneoff/eval_model_scorecard.py` (aggregated over EVENTS, whole split). Numbers produced
that way are not differenceable, however carefully each is computed on its own. This script
evaluates every predictor in the SAME pass over the SAME events with the SAME masks, so the
rows can be read against each other.

PREDICTORS
  ERA5 bicubic             era5_uv_2ch_bicubic.npy -- the naive low-resolution baseline
  EnsCGP posterior mean    enscgp_posterior.npy channels 0-1 -- the first guess
  EnsCGP/bicubic blend     alpha*EnsCGP + (1-alpha)*bicubic, --alpha (default 0.5)
  This model (q50)         the checkpoint's central field, via Q50_SLICE
All four are scored against the same WRF truth, per component, never pooled across u and v.

AGGREGATION -- the load-bearing convention (see scripts/events.py)
Samples inside an event are consecutive hours of one storm and are not independent draws.
Every number here is aggregated at the EVENT level: the metric is summed over that event's
samples and pixels and divided by its own count, giving one scalar per (event, predictor,
component, metric); the reported value is an unweighted mean over events. This matches
eval_model_scorecard.py, so the model column here reproduces the model column there.

RMSE is accumulated as MSE and square-rooted LAST, after the mean over events -- sqrt does
not commute with the mean, and rooting per event then averaging would give a different (and
wrong) number.

STRATA
"all" plus one per --top-pcts entry: the windiest N% of EACH sample's pixels by wind speed
(quantile_metrics.top_speed_mask). Masks come from the WRF truth, so all four predictors are
scored on exactly the same pixels -- no predictor can define its own easy cases.

Usage:
    python eval_results_table.py --device cuda:0
    python eval_results_table.py --top-pcts 10 5 1 --alpha 0.5
    python eval_results_table.py --split val --max-events 20      # quick look
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from _common import (  # noqa: E402  (shared harness; also puts scripts/ on sys.path)
    COMPONENTS, ProbabilisticSwin2SR, add_eval_args, setup, top_speed_frac_to_pct,
    top_speed_mask,
)
from events import load_event_split_map  # noqa: E402

DEFAULT_TOP_PCTS = (10.0, 5.0)  # matches eval_model_scorecard.DEFAULT_TOP_PCTS
METRICS = ("mae", "rmse")


def predictors(posterior, bicubic, pred, alpha: float) -> dict[str, torch.Tensor]:
    """label -> (B,2,H,W) prediction, all on the same device/dtype as the inputs.

    `posterior` is the full 5-channel EnsCGP tensor; only its mean channels (0-1) are a
    prediction of the wind -- 2-4 are the Cholesky and are not comparable to a wind field.
    """
    post_uv = posterior[:, :2]
    return {
        "ERA5 bicubic": bicubic,
        "EnsCGP posterior mean": post_uv,
        f"EnsCGP/bicubic blend (a={alpha:g})": alpha * post_uv + (1.0 - alpha) * bicubic,
        "This model (q50)": pred[:, ProbabilisticSwin2SR.Q50_SLICE],
    }


@torch.no_grad()
def event_table(ev, split_map, split: str, top_pcts, alpha: float,
                batch_size: int = 16, max_events: int | None = None) -> tuple[dict, int]:
    """-> ({(label, metric, stratum): (2,) per-component value}, n_events).

    One pass over the split. For each batch every predictor is scored against the same truth
    under the same masks, so the comparison is paired at the pixel level.
    """
    strata = ["all"] + [f"top{top_speed_frac_to_pct(p / 100.0)}" for p in top_pcts]
    event_ids = np.array(sorted(split_map[split].keys()))
    if max_events is not None and max_events < len(event_ids):
        event_ids = event_ids[:max_events]
    E = len(event_ids)

    labels = list(predictors(torch.zeros(1, 5, 1, 1), torch.zeros(1, 2, 1, 1),
                             torch.zeros(1, ProbabilisticSwin2SR.OUT_CHANNELS, 1, 1), alpha))
    acc = {(lab, m, s): torch.zeros(E, 2, dtype=torch.float64, device=ev.device)
           for lab in labels for m in METRICS for s in strata}
    counts = {s: torch.zeros(E, 2, dtype=torch.float64, device=ev.device) for s in strata}

    arrays = ev.arrays
    for ei, eid in enumerate(event_ids):
        idx = np.array(sorted(split_map[split][int(eid)]))
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            post = torch.from_numpy(np.array(arrays.posterior[b], dtype=np.float32, copy=True)).to(ev.device)
            bic = torch.from_numpy(np.array(arrays.bicubic[b], dtype=np.float32, copy=True)).to(ev.device)
            y = torch.from_numpy(np.array(arrays.wrf[b], dtype=np.float32, copy=True)).to(ev.device).double()
            pred = ev.model(post, bic, ev.terrain_raw)

            masks = {"all": None}
            for pct, name in zip(top_pcts, strata[1:]):
                masks[name] = top_speed_mask(y, pct / 100.0).expand_as(y)

            for lab, p in predictors(post.double(), bic.double(), pred.double(), alpha).items():
                err = p - y
                per_pixel = {"mae": err.abs(), "rmse": err ** 2}  # rmse accumulates MSE
                for s, mask in masks.items():
                    for m in METRICS:
                        v = per_pixel[m]
                        acc[(lab, m, s)][ei] += (v if mask is None else v * mask).sum(dim=(0, 2, 3))

            for s, mask in masks.items():
                counts[s][ei] += (float(y.shape[0] * y.shape[2] * y.shape[3]) if mask is None
                                  else mask.sum(dim=(0, 2, 3)).double())

    out = {}
    for (lab, m, s), a in acc.items():
        per_event = (a / counts[s].clamp_min(1.0)).cpu().numpy()   # (E, 2)
        mean_over_events = per_event.mean(axis=0)                  # (2,)
        # sqrt LAST -- after the mean over events, never per event.
        out[(lab, m, s)] = np.sqrt(mean_over_events) if m == "rmse" else mean_over_events
    return out, E


def print_table(table: dict, labels, strata, n_events: int, split: str) -> None:
    for m in METRICS:
        print(f"\n{m.upper()} (m/s) -- split={split}, {n_events} events, event-level mean"
              f"{', sqrt applied after the mean' if m == 'rmse' else ''}")
        head = f"{'Predictor':<34}" + "".join(f"{s + ' u':>12}{s + ' v':>12}{s + ' avg':>12}"
                                              for s in strata)
        print(head)
        print("-" * len(head))
        for lab in labels:
            row = f"{lab:<34}"
            for s in strata:
                u, v = table[(lab, m, s)]
                row += f"{u:>12.4f}{v:>12.4f}{(u + v) / 2.0:>12.4f}"
            print(row)


def print_markdown(table: dict, labels, strata, n_events: int, split: str) -> None:
    """The README block, ready to paste."""
    print(f"\n\n--- markdown for README.md (split={split}, {n_events} events) ---\n")
    cols = [(m, s) for s in strata for m in METRICS]
    print("| | " + " | ".join(f"{m.upper()} {s}" for m, s in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for lab in labels:
        cells = [f"{table[(lab, m, s)].mean():.3f}" for m, s in cols]
        bold = lab.startswith("This model")
        name = f"**{lab}**" if bold else lab
        print(f"| {name} | " + " | ".join(f"**{c}**" if bold else c for c in cells) + " |")
    print("\n(u/v averaged; per-component numbers are in the CSV.)")


def main() -> None:
    parser = add_eval_args(
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
        samples=None,
    )
    parser.add_argument("--top-pcts", type=float, nargs="*", default=list(DEFAULT_TOP_PCTS),
                        metavar="PCT",
                        help="Wind-speed strata: the windiest PCT%% of each sample's pixels")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="EnsCGP weight in the blend row (0=pure bicubic, 1=pure EnsCGP)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=None,
                        help="Evaluate only the first N events -- for a quick look, not for results")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV. Defaults to <log_dir>/figures/results_table_<split>.csv")
    args = parser.parse_args()

    ev = setup(args)
    split_map = load_event_split_map(ev.data_dir)
    output = args.output or ev.figure_path(f"results_table_{args.split}.csv")

    strata = ["all"] + [f"top{top_speed_frac_to_pct(p / 100.0)}" for p in args.top_pcts]
    print(f"Checkpoint: {ev.describe_checkpoint()}")
    print(f"Split: {args.split} | strata: {', '.join(strata)} | blend alpha={args.alpha:g}")

    table, n_events = event_table(ev, split_map, args.split, args.top_pcts, args.alpha,
                                  args.batch_size, args.max_events)
    labels = sorted({lab for lab, _m, _s in table}, key=lambda l: (l.startswith("This model"), l))

    print_table(table, labels, strata, n_events, args.split)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predictor", "metric", "stratum", "component", "value"])
        for lab in labels:
            for m in METRICS:
                for s in strata:
                    for ci, cname in enumerate(COMPONENTS):
                        w.writerow([lab, m, s, cname, f"{table[(lab, m, s)][ci]:.6f}"])
    print(f"\nWrote {output}")

    print_markdown(table, labels, strata, n_events, args.split)


if __name__ == "__main__":
    main()
