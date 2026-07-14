#!/usr/bin/env python3
"""
Plot training curves (val loss and components) from train_new_enscgp_swin.py log files.

Auto-join: when a log says "Resumed from ... at epoch N", the script finds another
parsed log whose val records end at (or include) epoch N and stitches them into a
continuous curve -- data before epoch N comes from the predecessor, data from N onward
comes from the resumed log. Overlapping epochs (e.g. the resumed log re-runs epoch 41)
use the resumed log's values.

Without --no_auto_join, all logs in --log_dir (or --logs) are parsed and chained
automatically. Logs that cannot be joined to any other (either because they stand
alone or because their predecessor isn't in the set) are plotted as separate lines.

Layout (3 panels, same x-axis):
  1. Val total loss (+ optional train total loss with --show_train)
  2. NLL decomposition: logdet and mahalanobis distance
  3. Weighted auxiliary components: MS, L1, spectral, gradient, quantile

Best-val epochs are marked with vertical dashed lines. Use --no_components to show
only panel 1 (single-panel output).

Usage:
    python plot_training_curves.py
    python plot_training_curves.py --log_dir /home/peytonli/26.6_wind/logs
    python plot_training_curves.py --logs train_20260628_113032.log train_20260628_194244.log
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

_F = r"([\d.]+)"   # float group

VAL_RE = re.compile(
    r"Epoch (\d+) \| Val Loss " + _F +
    r" \(NLL " + _F + r" \[logdet " + _F + r", mahal " + _F + r"\]"
    r"(?:, MS " + _F + r")?"
    r"(?:, Freq " + _F + r")?"
    r", L1 " + _F + r", Spectral " + _F + r", Gradient " + _F + r", Quantile " + _F + r"\)"
)

TRAIN_RE = re.compile(
    r"Epoch (\d+) \| Batch \d+/\d+ \| Loss " + _F +
    r" \(NLL " + _F + r" \[logdet " + _F + r", mahal " + _F + r"\]"
    r"(?:, MS " + _F + r")?"
    r"(?:, Freq " + _F + r")?"
    r", L1 " + _F + r", Spectral " + _F + r", Gradient " + _F + r", Quantile " + _F + r"\)"
    r" \| LR " + r"([\d.e+\-]+)"
)

RESUME_RE = re.compile(r"Resumed from (.+) at epoch (\d+) \(best_val_loss=" + _F + r"\)")
BEST_RE = re.compile(r"New best val loss: " + _F)


# ── data structures ───────────────────────────────────────────────────────────

FIELDS = ("total", "nll", "logdet", "mahal", "ms", "freq", "l1", "spectral", "gradient", "quantile")


def _val_rec(m: re.Match) -> dict:
    epoch, total, nll, logdet, mahal, ms, freq, l1, spectral, gradient, quantile = m.groups()
    return {
        "epoch": int(epoch),
        "total": float(total), "nll": float(nll),
        "logdet": float(logdet), "mahal": float(mahal),
        "ms": float(ms) if ms is not None else None,
        "freq": float(freq) if freq is not None else None,
        "l1": float(l1), "spectral": float(spectral),
        "gradient": float(gradient), "quantile": float(quantile),
    }


def _train_rec(m: re.Match) -> dict:
    epoch, total, nll, logdet, mahal, ms, freq, l1, spectral, gradient, quantile, lr = m.groups()
    return {
        "epoch": int(epoch),
        "total": float(total), "nll": float(nll),
        "logdet": float(logdet), "mahal": float(mahal),
        "ms": float(ms) if ms is not None else None,
        "freq": float(freq) if freq is not None else None,
        "l1": float(l1), "spectral": float(spectral),
        "gradient": float(gradient), "quantile": float(quantile),
        "lr": float(lr),
    }


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


def _epochs(records: list[dict]) -> np.ndarray:
    return np.array([r["epoch"] for r in records])


def _field(records: list[dict], key: str) -> np.ndarray | None:
    vals = [r.get(key) for r in records]
    if all(v is None for v in vals):
        return None
    return np.array([v if v is not None else np.nan for v in vals])


def plot_curves(merged_runs: list[dict], show_train: bool, show_components: bool, output: Path) -> None:
    n_panels = 3 if show_components else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.5 * n_panels), sharex=True,
                             gridspec_kw={"hspace": 0.08})
    if n_panels == 1:
        axes = [axes]

    ax_total = axes[0]
    ax_nll = axes[1] if show_components else None
    ax_aux = axes[2] if show_components else None

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
            # ── panel 2: NLL decomposition ──
            logdet = _field(val, "logdet")
            mahal = _field(val, "mahal")
            if logdet is not None:
                ax_nll.plot(epochs, logdet, color=color, linewidth=1.5, label=f"logdet [{label}]")
            if mahal is not None:
                ax_nll.plot(epochs, mahal, color=color, linewidth=1.5, linestyle="--",
                            label=f"mahal [{label}]")
            for ep in best:
                ax_nll.axvline(ep, color=color, linewidth=0.7, linestyle=":", alpha=0.7)

            # ── panel 3: aux components ──
            aux_styles = {
                "ms":       ("-",  1.4, "MS"),
                "freq":     ("--", 1.4, "Freq"),
                "l1":       ((0, (5, 2)), 1.2, "L1"),
                "spectral": ("-.", 1.2, "Spectral"),
                "gradient": (":",  1.2, "Gradient"),
                "quantile": ((0, (3, 1, 1, 1)), 1.2, "Quantile"),
            }
            for key, (ls, lw, display) in aux_styles.items():
                arr = _field(val, key)
                if arr is not None:
                    ax_aux.plot(epochs, arr, color=color, linestyle=ls, linewidth=lw,
                                label=f"{display} [{label}]")
            for ep in best:
                ax_aux.axvline(ep, color=color, linewidth=0.7, linestyle=":", alpha=0.7)

    # ── formatting ──
    ax_total.set_ylabel("Val total loss")
    ax_total.legend(fontsize=7, loc="upper right")
    ax_total.grid(True, which="both", alpha=0.3)

    if show_components and ax_nll is not None and ax_aux is not None:
        ax_nll.set_ylabel("NLL components")
        ax_nll.legend(fontsize=6, loc="upper right")
        ax_nll.grid(True, which="both", alpha=0.3)

        ax_aux.set_ylabel("Aux components (weighted)")
        ax_aux.legend(fontsize=6, loc="upper right", ncol=2)
        ax_aux.grid(True, which="both", alpha=0.3)

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
                        help="Show only the total-loss panel (no NLL/aux breakdowns)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path; defaults to <log_dir>/training_curves.png")
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

    output = args.output or (log_dir / "training_curves.png")

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
            print(f"    (skipped — no val loss entries)")

    if not logs:
        print("No usable log data found.")
        sys.exit(1)

    if args.no_auto_join:
        runs = [merge_chain([log]) for log in logs]
    else:
        chains = build_chains(logs)
        runs = [merge_chain(chain) for chain in chains]
        for run in runs:
            joined = run["label"]
            print(f"  Chain: {joined} ({len(run['val'])} val epochs total)")

    plot_curves(runs, show_train=args.show_train, show_components=not args.no_components, output=output)


if __name__ == "__main__":
    main()
