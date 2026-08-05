"""
Evaluate a trained ProbabilisticSwin2SR QUANTILE checkpoint (scripts/new_enscgp_swin.py,
version 0714+) against ERA5 (LR) and WRF (HR) ground truth.

For each sample, plots 7 panels (one row per sample):
1) ERA5 LR wind speed (native 34x34)
2) EnsCGP posterior mean wind speed (200x200) -- the first guess the model starts from,
   read straight from data/enscgp_posterior.npy with no model involved
3) WRF HR wind speed (200x200, ground truth)
4) SWIN q10 field speed (200x200)  -- SCENARIO, see the warning below
5) SWIN q50 field speed (200x200)  -- the central field the structural losses train
6) SWIN q90 field speed (200x200)  -- SCENARIO, see the warning below
7) CRPS (200x200), averaged over the u and v components -- see below
Panels 1-6 share one speed color scale per row so they are directly comparable; CRPS gets
its own sequential scale. Each wind-field panel carries a direction quiver overlay.

*** What panels 4 and 6 are NOT ***
The model predicts per-pixel MARGINAL quantiles of u and of v SEPARATELY, and (unlike the
retired Cholesky head) carries no u/v correlation. Panels 4/6 show
speed(q10_u, q10_v) / speed(q90_u, q90_v): the speed OF THE COMPONENT-WISE QUANTILE FIELD --
a scenario, NOT the 10th/90th percentile of wind speed. These are different things: e.g. at
a pixel with q10_u = q10_v = -1.28, the "q10 field speed" is sqrt(1.28^2+1.28^2) = 1.81 --
a HIGH speed, not a low one. So panel 4 is not a lower speed bound and panel 6 is not an
upper one; do not read them as a speed confidence band. (Getting true speed percentiles
would require assuming a u/v dependence the model does not provide -- e.g. sampling the
marginals under independence -- which this script deliberately does not do.) Panel 5 (q50)
is unaffected by this caveat in the usual sense: it is the model's central field, exactly
the quantity the structural losses train.

CRPS: the Continuous Ranked Probability Score satisfies the quantile decomposition
    CRPS(F, y) = 2 * integral_0^1 pinball_tau(F^-1(tau), y) dtau.
With only three predicted quantiles the integral is approximated on the tau grid
{0.1, 0.5, 0.9} by the midpoint rule -- each tau represents the interval to the midpoints of
its neighbours (boundaries at 0.3 and 0.7, endpoints 0 and 1), giving weights
{0.3, 0.4, 0.3} which sum to 1:
    CRPS ~= 2 * (0.3*pinball_0.1 + 0.4*pinball_0.5 + 0.3*pinball_0.9)
It is computed per pixel on each COMPONENT against that component's WRF truth (where the
quantiles are genuinely marginal quantiles, so this is well posed -- no speed/dependence
assumption), then the u and v maps are averaged into the single panel shown. This is a
COARSE 3-point approximation: the true integrand is unrepresented in the tails beyond
q10/q90, so treat the values as a relative comparison metric (lower = better, across
pixels/samples/checkpoints), not an absolute CRPS.

Sample source: indices are drawn from the held-out test split by default
(data/splits_70_15_15/split_indices.npz, test_idx), since this script evaluates a trained
checkpoint and the model should not have seen these samples. --split train/val/test switches
the pool; --sample_indices (raw indices into era5/wrf/posterior) bypasses split filtering.

Display convention: all grids are north-up as stored (the "0701 special" alignment fix
flipped the WRF .npy files north-south in the data dir, so no display-time flip is applied
here). Panels use origin="upper"; quiver v is negated so +v (northward) points up the page.

Checkpoint format: the dict saved by train_new_enscgp_swin.py's save_checkpoint
(model/optimizer/scheduler state + epoch + best_val_loss -- no embedded config, hence
--config). Requires a 0714+ quantile checkpoint (6-channel output); a pre-0714
Gaussian/Cholesky checkpoint will fail to load (see CHANGELOG "0714").

Usage:
    python eval_checkpoint.py --checkpoint runs/0714/checkpoints/best.pth
    python eval_checkpoint.py --checkpoint .../best.pth --split test --n_samples 6 --seed 42
    python eval_checkpoint.py --checkpoint .../best.pth --sample_indices 12,4081,6500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import (  # noqa: E402  (shared harness; also puts scripts/ on sys.path)
    add_eval_args, choose_indices, crps_3q, plot_panel, predict, setup, speed, split_pool,
)


def main() -> None:
    parser = add_eval_args(
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
        n_samples_default=6,
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PNG. Defaults to <log_dir>/figures/swin_quantile_panels.png")
    parser.add_argument("--quiver_skip_lr", type=int, default=2, help="Arrow subsampling on the 34x34 ERA5 panel")
    parser.add_argument("--quiver_skip_hr", type=int, default=10, help="Arrow subsampling on the 200x200 panels")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    ev = setup(args)
    era5, wrf, posterior = ev.arrays.era5, ev.arrays.wrf, ev.arrays.posterior
    output_path = args.output or ev.figure_path("swin_quantile_panels.png")

    idx = choose_indices(split_pool(ev.splits_path, args.split), ev.arrays.n_total,
                         args.n_samples, args.seed, args.sample_indices)

    lsm_era34, lsm_wrf = ev.arrays.land_masks()

    # (B, 6, 200, 200): [q10_u, q10_v, q50_u, q50_v, q90_u, q90_v]
    pred_batch = predict(ev, idx)

    col_titles = [
        "ERA5 LR speed (34x34)",
        "EnsCGP posterior mean speed",
        "WRF HR speed (ground truth)",
        "SWIN q10 field speed (scenario, not %ile)",
        "SWIN q50 speed (central field)",
        "SWIN q90 field speed (scenario, not %ile)",
        "CRPS (mean of u,v; 3-quantile approx.)",
    ]
    cbar_labels = ["m/s", "m/s", "m/s", "m/s", "m/s", "m/s", "m/s"]

    n_rows, n_cols = len(idx), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)

    crps_means = []
    for row, i in enumerate(idx):
        era5_u, era5_v = np.asarray(era5[i, 0]), np.asarray(era5[i, 1])
        wrf_u, wrf_v = np.asarray(wrf[i, 0]), np.asarray(wrf[i, 1])
        ens_u, ens_v = np.asarray(posterior[i, 0]), np.asarray(posterior[i, 1])
        q10_u, q10_v = pred_batch[row, 0], pred_batch[row, 1]
        q50_u, q50_v = pred_batch[row, 2], pred_batch[row, 3]
        q90_u, q90_v = pred_batch[row, 4], pred_batch[row, 5]

        era5_speed = speed(era5_u, era5_v)
        wrf_speed = speed(wrf_u, wrf_v)
        ens_speed = speed(ens_u, ens_v)
        # Speed OF the component-wise quantile fields -- scenarios, not speed percentiles.
        q10_speed = speed(q10_u, q10_v)
        q50_speed = speed(q50_u, q50_v)
        q90_speed = speed(q90_u, q90_v)

        # CRPS per component (well posed: marginal quantiles vs that component's truth),
        # then averaged into a single map.
        crps_u = crps_3q(q10_u, q50_u, q90_u, wrf_u)
        crps_v = crps_3q(q10_v, q50_v, q90_v, wrf_v)
        crps = 0.5 * (crps_u + crps_v)
        crps_means.append(float(crps.mean()))

        speed_fields = (era5_speed, ens_speed, wrf_speed, q10_speed, q50_speed, q90_speed)
        speed_vmin = float(min(f.min() for f in speed_fields))
        speed_vmax = float(max(f.max() for f in speed_fields))

        panels = [
            (era5_speed, "jet", speed_vmin, speed_vmax, lsm_era34, (era5_u, era5_v), args.quiver_skip_lr),
            (ens_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (ens_u, ens_v), args.quiver_skip_hr),
            (wrf_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (wrf_u, wrf_v), args.quiver_skip_hr),
            (q10_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (q10_u, q10_v), args.quiver_skip_hr),
            (q50_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (q50_u, q50_v), args.quiver_skip_hr),
            (q90_speed, "jet", speed_vmin, speed_vmax, lsm_wrf, (q90_u, q90_v), args.quiver_skip_hr),
            (crps, "magma", 0.0, max(float(crps.max()), 1e-12), lsm_wrf, None, 1),
        ]
        for col, (data, cmap, vmin, vmax, lsm, uv, qskip) in enumerate(panels):
            title = f"{col_titles[col]} | idx={i}" if col == 0 else col_titles[col]
            plot_panel(axes[row, col], data, cmap, vmin, vmax, title, cbar_labels[col], lsm=lsm, uv=uv, quiver_skip=qskip)

    fig.suptitle(
        f"SWIN quantile checkpoint eval vs ERA5 / WRF (split={args.split}, ckpt={ev.checkpoint.name})",
        fontsize=14, y=1.0,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")
    print(f"Checkpoint: {ev.describe_checkpoint()}")
    print(f"Split: {args.split}")
    print(f"Samples used: {idx.tolist()}")
    print(f"Mean CRPS over shown samples: {np.mean(crps_means):.4f}  (per-sample: "
          f"{', '.join(f'{v:.4f}' for v in crps_means)})")


if __name__ == "__main__":
    main()
