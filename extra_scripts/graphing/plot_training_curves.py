#!/usr/bin/env python3
"""
Plot training curves (val loss and components) from train_new_enscgp_swin.py log files
(0714+ quantile format: pinball loss + q10/q50/q90 coverage; the pre-0714 NLL/Cholesky log
format is no longer parsed).

Auto-join: when a log says "Resumed from ... at epoch N", the script finds another
parsed log whose val records end at (or include) epoch N and stitches them into a
continuous curve -- data before epoch N comes from the predecessor, data from N onward
comes from the resumed log. Overlapping epochs use the resumed log's values. (A
weights-only / architecture-change start logs no "Resumed from" line, so it correctly
begins a fresh chain at epoch 0 rather than joining a prior run.)

Without --no_auto_join, all logs in --log_dir (or --logs) are parsed and chained
automatically. Logs that cannot be joined to any other are plotted as separate lines.

Layout (4 panels, shared x-axis), or just panel 1 with --no_components:
  1. Val total loss (+ optional train total loss with --show_train)
  2. Pinball decomposition: mean of the q90/q10 pinball (kept close to the two component
     lines rather than their sum), q90 pinball, q10 pinball
  3. Structural components (weighted): MS, Freq, L1
  4. Coverage: empirical fraction of truth <= q10/q50/q90, overall (solid) and in the
     extreme tail (dashed), with nominal .10/.50/.90 reference lines. This is the
     calibration read: q90 should sit near 0.90, q10 near 0.10.

Best-val epochs are marked with vertical dashed lines.

Usage:
    python plot_training_curves.py --logs /home/peytonli/26.6_wind/runs/0714/train_20260714_235959.log
    python plot_training_curves.py --log_dir /home/peytonli/26.6_wind/runs/0714
    python plot_training_curves.py --logs a.log b.log --no_auto_join
    python plot_training_curves.py --show_train --no_components
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from new_enscgp_swin import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

# ── regex patterns ────────────────────────────────────────────────────────────

_F = r"([\d.]+)"          # positive float group
_C = r"(nan|[\d.]+)"      # coverage value (may be nan if a stratum is empty)

_LOSS_BODY = (
    r" \(Pin " + _F + r" \[q90 " + _F + r", q10 " + _F + r"\]"
    r", MS " + _F + r", Freq " + _F + r", L1 " + _F + r", Spectral " + _F + r", Gradient " + _F + r"\)"
)

VAL_RE = re.compile(r"Epoch (\d+) \| Val Loss " + _F + _LOSS_BODY)
TRAIN_RE = re.compile(r"Epoch (\d+) \| Batch \d+/\d+ \| Loss " + _F + _LOSS_BODY + r" \| LR ([\d.e+\-]+)")
COVERAGE_RE = re.compile(
    r"Epoch (\d+) \| Coverage q10/q50/q90 = " + _C + r"/" + _C + r"/" + _C +
    r" \(nominal .10/.50/.90\) \| extreme tail = " + _C + r"/" + _C + r"/" + _C
)
RESUME_RE = re.compile(r"Resumed from (.+) at epoch (\d+) \(best_val_loss=" + _F + r"\)")
BEST_RE = re.compile(r"New best val loss: " + _F)


# ── data structures ───────────────────────────────────────────────────────────

# Loss components present in both train and val lines.
FIELDS = ("total", "pin", "q90", "q10", "ms", "freq", "l1", "spectral", "gradient")
# Coverage fields (val lines only); attached to the matching val record.
COVERAGE_FIELDS = ("cov_q10", "cov_q50", "cov_q90", "cov_q10_ext", "cov_q50_ext", "cov_q90_ext")


def _loss_fields(groups: tuple) -> dict:
    """Map the 9 captured loss groups (total, pin, q90, q10, ms, freq, l1, spectral,
    gradient) to a dict."""
    total, pin, q90, q10, ms, freq, l1, spectral, gradient = groups
    return {
        "total": float(total), "pin": float(pin), "q90": float(q90), "q10": float(q10),
        "ms": float(ms), "freq": float(freq), "l1": float(l1),
        "spectral": float(spectral), "gradient": float(gradient),
    }


def _val_rec(m: re.Match) -> dict:
    rec = {"epoch": int(m.group(1))}
    rec.update(_loss_fields(m.groups()[1:]))
    rec.update({k: None for k in COVERAGE_FIELDS})  # filled in by the following Coverage line
    return rec


def _train_rec(m: re.Match) -> dict:
    rec = {"epoch": int(m.group(1))}
    groups = m.groups()
    rec.update(_loss_fields(groups[1:-1]))
    rec["lr"] = float(groups[-1])
    return rec


class LogData:
    def __init__(self, path: Path):
        self.path = path
        self.resume_from: str | None = None
        self.resume_epoch: int | None = None
        self.val: list[dict] = []
        self.train_batches: dict[int, list[dict]] = defaultdict(list)
        self.best_epochs: list[int] = []

    def max_val_epoch(self) -> int:
        return max((r["epoch"] for r in self.val), default=-1)

    def short_name(self) -> str:
        return self.path.name


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_log(path: Path) -> LogData:
    log = LogData(path)
    with open(path) as f:
        for line in f:
            if (m := RESUME_RE.search(line)):
                log.resume_from = m.group(1).strip()
                log.resume_epoch = int(m.group(2))
            elif (m := VAL_RE.search(line)):
                log.val.append(_val_rec(m))
            elif (m := COVERAGE_RE.search(line)):
                # Logged immediately after its Val Loss line, same epoch -- attach to it.
                ep = int(m.group(1))
                if log.val and log.val[-1]["epoch"] == ep:
                    vals = [float(g) for g in m.groups()[1:]]
                    log.val[-1].update(dict(zip(COVERAGE_FIELDS, vals)))
            elif (m := TRAIN_RE.search(line)):
                r = _train_rec(m)
                log.train_batches[r["epoch"]].append(r)
            elif BEST_RE.search(line) and log.val:
                log.best_epochs.append(log.val[-1]["epoch"])
    return log


# ── auto-join ─────────────────────────────────────────────────────────────────

def _find_predecessor(log: LogData, candidates: list[LogData]) -> LogData | None:
    """For a log with resume_epoch=N, find the candidate whose max_val_epoch is
    closest to N-1 from above (i.e. >= N-1), excluding logs that themselves resume
    at or after N (would form a cycle or wrong order)."""
    N = log.resume_epoch
    eligible = [
        c for c in candidates
        if c is not log
        and c.max_val_epoch() >= N - 1
        and (c.resume_epoch is None or c.resume_epoch < N)
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda c: abs(c.max_val_epoch() - (N - 1)))


def build_chains(logs: list[LogData]) -> list[list[LogData]]:
    """Group logs into ordered chains (each list = a continuous run, earliest first).
    Logs that can't be joined appear as single-element lists."""
    used: set[int] = set()
    resuming = [l for l in logs if l.resume_epoch is not None]
    starting = [l for l in logs if l.resume_epoch is None]

    chains: list[list[LogData]] = []

    for start in starting:
        chain = [start]
        used.add(id(start))
        while True:
            tail = chain[-1]
            successor = None
            for rlog in resuming:
                if id(rlog) in used:
                    continue
                pred = _find_predecessor(rlog, [tail])
                if pred is tail:
                    if successor is None or rlog.resume_epoch > successor.resume_epoch:
                        successor = rlog
            if successor is None:
                break
            chain.append(successor)
            used.add(id(successor))
        chains.append(chain)

    for rlog in resuming:
        if id(rlog) not in used:
            chains.append([rlog])

    return chains


def merge_chain(chain: list[LogData]) -> dict:
    """Merge a chain into a single val timeline and per-epoch averaged train stats."""
    val_by_epoch: dict[int, dict] = {}
    train_by_epoch: dict[int, list[dict]] = {}
    best_epochs: list[int] = []

    for seg_idx, log in enumerate(chain):
        cutoff = chain[seg_idx + 1].resume_epoch if seg_idx + 1 < len(chain) else None
        for rec in log.val:
            ep = rec["epoch"]
            if cutoff is None or ep < cutoff:
                val_by_epoch[ep] = rec
        for ep, batches in log.train_batches.items():
            if cutoff is None or ep < cutoff:
                train_by_epoch[ep] = batches
        for ep in log.best_epochs:
            if cutoff is None or ep < cutoff:
                best_epochs.append(ep)

    epochs = sorted(val_by_epoch)
    val_records = [val_by_epoch[e] for e in epochs]

    # Average train stats per epoch
    train_avg: dict[int, dict] = {}
    for ep in sorted(train_by_epoch):
        batches = train_by_epoch[ep]
        avg = {"epoch": ep}
        for key in FIELDS:
            vals = [b[key] for b in batches if b.get(key) is not None]
            avg[key] = float(np.mean(vals)) if vals else None
        lrs = [b["lr"] for b in batches]
        avg["lr"] = float(np.mean(lrs)) if lrs else None
        train_avg[ep] = avg

    return {
        "val": val_records,
        "train": train_avg,
        "best_epochs": sorted(set(best_epochs)),
        "label": " → ".join(l.short_name() for l in chain),
        "n_logs": len(chain),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# Fixed colors per quantile level for the coverage panel (independent of the per-run color).
Q_COVERAGE = (("cov_q10", 0.10, "C0", "q10"), ("cov_q50", 0.50, "0.4", "q50"), ("cov_q90", 0.90, "C3", "q90"))


def _epochs(records: list[dict]) -> np.ndarray:
    return np.array([r["epoch"] for r in records])


def _field(records: list[dict], key: str) -> np.ndarray | None:
    vals = [r.get(key) for r in records]
    if all(v is None for v in vals):
        return None
    return np.array([v if v is not None else np.nan for v in vals])


def plot_curves(merged_runs: list[dict], show_train: bool, show_components: bool, output: Path) -> None:
    n_panels = 4 if show_components else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.3 * n_panels), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    if n_panels == 1:
        axes = [axes]

    ax_total = axes[0]
    ax_pin = axes[1] if show_components else None
    ax_struct = axes[2] if show_components else None
    ax_cov = axes[3] if show_components else None
    multi_run = len(merged_runs) > 1

    for run_idx, run in enumerate(merged_runs):
        color = COLORS[run_idx % len(COLORS)]
        val = run["val"]
        train = run["train"]
        label = run["label"]
        best = run["best_epochs"]
        epochs = _epochs(val)

        # ── panel 1: total val loss ──
        ax_total.plot(epochs, _field(val, "total"), color=color, linewidth=1.8, label=label)
        if show_train and train:
            tr_epochs = np.array(sorted(train))
            tr_total = np.array([train[e]["total"] for e in tr_epochs])
            ax_total.plot(tr_epochs, tr_total, color=color, linewidth=0.8, alpha=0.4,
                          linestyle="--", label=f"{label} (train)")
        for ep in best:
            ax_total.axvline(ep, color=color, linewidth=0.7, linestyle=":", alpha=0.7)

        if show_components:
            # ── panel 2: pinball decomposition ──
            # Aggregate line is the MEAN of the q90/q10 pinball (not their sum), so it sits
            # among the two component lines rather than at ~2x their level.
            q90_arr, q10_arr = _field(val, "q90"), _field(val, "q10")
            if q90_arr is not None and q10_arr is not None:
                ax_pin.plot(epochs, 0.5 * (q90_arr + q10_arr), color=color, linestyle="-",
                            linewidth=1.5, label=f"Pin (mean q10,q90) [{label}]")
            if q90_arr is not None:
                ax_pin.plot(epochs, q90_arr, color=color, linestyle="--", linewidth=1.5, label=f"q90 [{label}]")
            if q10_arr is not None:
                ax_pin.plot(epochs, q10_arr, color=color, linestyle=(0, (5, 2)), linewidth=1.5, label=f"q10 [{label}]")
            for ep in best:
                ax_pin.axvline(ep, color=color, linewidth=0.7, linestyle=":", alpha=0.7)

            # ── panel 3: structural components ──
            struct_styles = {
                "ms":   ("-",  1.4, "MS"),
                "freq": ("--", 1.4, "Freq"),
                "l1":   ((0, (5, 2)), 1.2, "L1"),
            }
            for key, (ls, lw, disp) in struct_styles.items():
                arr = _field(val, key)
                if arr is not None:
                    ax_struct.plot(epochs, arr, color=color, linestyle=ls, linewidth=lw, label=f"{disp} [{label}]")
            for ep in best:
                ax_struct.axvline(ep, color=color, linewidth=0.7, linestyle=":", alpha=0.7)

            # ── panel 4: coverage (colored by quantile level, not by run) ──
            prefix = f"{label}: " if multi_run else ""
            for key, _nominal, qcolor, qname in Q_COVERAGE:
                arr = _field(val, key)
                if arr is not None:
                    ax_cov.plot(epochs, arr, color=qcolor, linewidth=1.6, label=f"{prefix}{qname}")
                arr_ext = _field(val, key + "_ext")
                if arr_ext is not None:
                    ax_cov.plot(epochs, arr_ext, color=qcolor, linewidth=1.2, linestyle="--", alpha=0.8,
                                label=f"{prefix}{qname} (extreme)")

    # ── formatting ──
    ax_total.set_ylabel("Val total loss")
    ax_total.legend(fontsize=7, loc="upper right")
    ax_total.grid(True, which="both", alpha=0.3)

    if show_components:
        ax_pin.set_ylabel("Pinball (val)")
        ax_pin.legend(fontsize=6, loc="upper right", ncol=2)
        ax_pin.grid(True, which="both", alpha=0.3)

        ax_struct.set_ylabel("Structural (weighted, val)")
        ax_struct.legend(fontsize=6, loc="upper right", ncol=2)
        ax_struct.grid(True, which="both", alpha=0.3)

        for _key, nominal, qcolor, _qname in Q_COVERAGE:
            ax_cov.axhline(nominal, color=qcolor, linewidth=0.8, linestyle=":", alpha=0.5)
        ax_cov.set_ylabel("Coverage (fraction ≤ q)")
        ax_cov.set_ylim(0.0, 1.0)
        ax_cov.legend(fontsize=6, loc="center right", ncol=2)
        ax_cov.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Training curves", fontsize=13)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves: {output}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log_dir", type=Path, default=None,
                        help="Directory to scan for *.log files; defaults to paths.log_dir from --config")
    parser.add_argument("--logs", type=Path, nargs="+", default=None,
                        help="Explicit log file(s) to use; overrides --log_dir")
    parser.add_argument("--no_auto_join", action="store_true",
                        help="Disable auto-joining; each log file is plotted as a separate line")
    parser.add_argument("--show_train", action="store_true",
                        help="Overlay per-epoch averaged train loss (dashed, lighter)")
    parser.add_argument("--no_components", action="store_true",
                        help="Show only the total val-loss panel (no pinball/structural/coverage breakdowns)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path; defaults to <log_dir>/figures/training_curves.png")
    args = parser.parse_args()

    config = load_config(args.config)
    log_dir = args.log_dir or Path(config["paths"]["log_dir"])

    if args.logs:
        log_paths = sorted(args.logs)
    else:
        log_paths = sorted(log_dir.glob("*.log"))

    if not log_paths:
        print(f"No log files found in {log_dir}. Use --logs to specify files explicitly.")
        sys.exit(1)

    output = args.output or (log_dir / "figures" / "training_curves.png")

    print(f"Parsing {len(log_paths)} log file(s)...")
    logs = []
    for p in log_paths:
        log = parse_log(p)
        n_val = len(log.val)
        resume_info = f", resumes at epoch {log.resume_epoch}" if log.resume_epoch is not None else ""
        print(f"  {p.name}: {n_val} val epochs (max {log.max_val_epoch()}){resume_info}")
        if n_val > 0:
            logs.append(log)
        else:
            print(f"    (skipped — no val loss entries; is this a pre-0714 log?)")

    if not logs:
        print("No usable log data found.")
        sys.exit(1)

    if args.no_auto_join:
        runs = [merge_chain([log]) for log in logs]
    else:
        chains = build_chains(logs)
        runs = [merge_chain(chain) for chain in chains]
        for run in runs:
            print(f"  Chain: {run['label']} ({len(run['val'])} val epochs total)")

    plot_curves(runs, show_train=args.show_train, show_components=not args.no_components, output=output)


if __name__ == "__main__":
    main()
