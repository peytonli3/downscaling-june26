# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Wind downscaling: an EnsCGP (ensemble-conditional Gaussian process) posterior
first-guess, refined by a Swin2SR-based transformer, trained to predict
per-pixel quantiles (q10/q50/q90) of ERA5-to-WRF downscaled wind (u, v).
See `README.md` for the versioning workflow and `docs/CHANGELOG.md` for the full
architecture history (read the most recent entries before touching the model
or loss -- this repo's architecture changes frequently, so verify any prior
description against current code/CHANGELOG rather than trusting a stale summary).

## Environment

- Conda env: `/home/peytonli/.conda/envs/downscaling_all` (PyTorch 2.5.1+cu118).
  No repo-local venv; always invoke scripts with this env's Python.
- Shared 3-GPU host (`cuda:0/1/2`, A100 80GB each) -- other users run unrelated
  jobs on this machine. Check `nvidia-smi` before picking a device or a batch
  size; memory scales ~linearly with batch size (roughly 6.5GB/sample
  forward+backward for the current model), so a batch size that fit before can
  OOM after a model change, independent of GPU contention.

## Commands

### Self-tests
Scripts with meaningful logic end in a smoke test / self-test guarded by
`if __name__ == "__main__"`, and `tests/` runs them as a set:
```
python tests/test_selftests.py          # all of them (pytest is NOT installed; stdlib only)
python tests/test_selftests.py --fast   # skip the ones that build a real model
python scripts/new_enscgp_swin.py       # model smoke test + pinball-quantile-recovery unit test
python scripts/multiscale_loss.py       # loss self-test (band decomposition, shift-tolerance)
python scripts/quantile_metrics.py      # pinball/CRPS numpy-vs-torch agreement, rank_cdf
```
`tests/test_selftests.py` is also a valid pytest module if you install pytest. A new
self-test should be registered there -- one nothing runs is documentation, not a test.

### Train
```
cd scripts
python train_new_enscgp_swin.py [--config new_enscgp_swin_config.json] [--device cuda:N] [--resume PATH] [--resume-weights-only] [--max-steps N]
```
- `--max-steps N` runs a handful of optimizer steps then exits -- use this to
  verify a config/code change against real data before a multi-hour run.
- For a detached run: `nohup ... python train_new_enscgp_swin.py ... > out.log 2>&1 & disown`.
- `--resume-weights-only` does a partial, shape-matched (not strict) load --
  check the relevant CHANGELOG entry before resuming across a version
  boundary, since a tensor can match by shape while its channels have changed
  meaning (see conv_first note below).

### Evaluate / diagnose a checkpoint
Under `extra_scripts/swin/`: `eval_checkpoint.py` (panel figures),
`eval_quantile_calibration.py` (PIT histogram + coverage maps),
`compare_eigenspectra.py` (spectral fidelity of q50), `eval_pinball_impact.py`.
Point `--checkpoint` at `runs/<version>/checkpoints/*.pth` (it defaults to
`<log_dir>/checkpoints/best.pth` from `--config`).

They all go through `swin/_common.py`, and a new diagnostic must too:
```python
p = add_eval_args(argparse.ArgumentParser(description=__doc__))  # the shared flags
args = p.parse_args()
ev = setup(args)                       # config -> model+ckpt -> terrain -> mmapped arrays
idx = choose_indices(split_pool(ev.splits_path, args.split), ev.arrays.n_total, ...)
pred = predict(ev, idx)                # or iter_predictions(ev, idx) to stream a whole split
fig.savefig(ev.figure_path("name.png"))
```
Re-deriving that preamble by hand is how these scripts drift apart; it is also
how `plot_band_errors.py` ended up plotting `pred[:, :2]` (q10 under the
quantile head) and labelling it the mean. Take q50 via `split_quantiles(...)`
or `Q50_SLICE`, never a raw index.

Multi-mode tools take a subcommand rather than existing as separate scripts:
`enscgp/neighbor_mae.py {rank,mean,sweep-k,baseline}`,
`enscgp/tune_sigma.py {ssr,mae,decouple}`,
`swin/oneoff/bias_diagnostic.py --part {a,b,c,d,a_meangate,all}`.

### Versioning
A checkpoint-incompatible architecture change = a new git tag:
```
git commit -am "vN: <what broke compatibility>"
git tag vN-<date>
```
`git checkout <tag>` restores the exact model/train/config for that version.
`git tag -l` / `git log --oneline` show what's committed; not every
experimental change is tagged (some are plain commits) -- check both.

## Architecture

### Pipeline
ERA5 (low-res) -> EnsCGP analog-ensemble posterior (mean + Cholesky
covariance) -> `ProbabilisticSwin2SR` (Swin2SR backbone + two heads) ->
per-pixel q10/q50/q90 of (u, v).

### Model (`scripts/new_enscgp_swin.py`)
- Wraps `network_swin2sr.py`'s `Swin2SR` at upscale=1 (same-resolution
  restoration, not super-resolution).
- Input to `conv_first`: `cat([bicubic, posterior])` (7ch: bic_u, bic_v,
  enscgp_u, enscgp_v, L11, L21, L22) + `terrain_encoder(terrain_raw)` (4ch) = 11ch.
- Two heads on the shared backbone features, each with its own residual style
  -- read the module docstring for the exact current formulas, they've
  changed more than once:
  - `mean_head` -> q50 (the central field), gated onto `mean_base`. Trained
    ONLY by the structural losses, never pinball -- pinball(0.5) would pull
    q50 toward the blurry pointwise median and fight the displacement-tolerant
    structural loss.
  - `offset_head` -> q90/q10 via monotone-by-construction softplus offsets
    around q50. Trained ONLY by pinball loss. Monotonicity (q10 <= q50 <= q90)
    holds by construction, not by a penalty term.
- `terrain_encoder.py`: processes terrain at native 1000x1000 resolution
  before downsampling to the 200x200 model grid (preserves fine
  coastline/slope detail); negligible compute cost (~0.6% of a training step)
  despite ~7% of total parameters -- do not "optimize" this without a
  measured reason.

### Loss (`scripts/train_new_enscgp_swin.py` + `scripts/multiscale_loss.py`)
```
L = ms_weight   * MultiscaleLoss(q50, wrf)     # Laplacian-pyramid; sliced-Wasserstein on fine bands, L1 on coarsest
  + freq_weight * FreqBandLoss(q50, wrf)       # FFT-band-split, pointwise L1, down-extremes pixel weighting
  + pin_weight  * [pinball(q90, .9) + pinball(q10, .1)]   # sign-aware extreme weighting, see extreme_pixel_weights_signed
```
Structural terms never touch q10/q90 (an uncertainty envelope is not a wind
field; matching its spectrum/texture is a category error). Pinball never
touches q50. Zero-weight structural terms are skipped during training but
always computed on validation as a diagnostic (`compute_all` in
`compute_weighted_loss`).

### Directory conventions
- `scripts/` -- the live model/training pipeline, flat (modules import each
  other by bare name, e.g. `from new_enscgp_swin import ...`) -- do not nest
  this into packages.
- `scripts/paths.py` -- the ONLY place that knows where anything lives. No file
  in this repo may hardcode an absolute path; add a constant here instead.
  Config values go through `paths.resolve()` (absolute wins, relative is
  repo-relative). `python scripts/paths.py` prints and checks every path.
  Figure outputs go through `paths.figures_dir(log_dir)` / `paths.run_figures(version)`
  (or `EvalSetup.figure_path`) -- never a hand-written `log_dir / "figures"`.
- `scripts/quantile_metrics.py` -- the tau grid, its CRPS weights, `pinball`,
  `crps_3q`, and `rank_cdf`, shared by the training loss and every diagnostic so
  a run and the scorecard cannot score on different functionals. It imports
  nothing from this repo, which is what makes it safe to import from
  `swin/oneoff/_v6_common.py` (that module pins an OLD model class on sys.path).
- `extra_scripts/` -- organized by pipeline stage, then by role:
  `data_prep/`, `enscgp/{,graphing/}`, `swin/{,graphing/,oneoff/}`,
  `presentation/`. The recurring evals sit at `swin/` top level;
  `swin/oneoff/` holds one-time investigations kept for provenance.
- Modules starting with `_` are shared helpers, never entry points:
  `swin/_common.py` (eval harness), `enscgp/_common.py` (neighbor loaders),
  `swin/oneoff/_v6_common.py` (pins the v6-0714 class for the 0714 diagnostics).
- Scripts outside `scripts/` bootstrap by walking UP to find the repo (see
  `paths.py`'s docstring). Never replace that with a fixed `.parent` count --
  that is exactly what broke the last time files were moved.
- `data/` -- gitignored; raw-vs-derived classification and regeneration
  commands are in `data/README.md`.
- `runs/<version>/` -- everything one run produced: `train_*.log` (tracked),
  `checkpoints/` and `figures/` (gitignored, except figures' `*.csv`). This is
  the ONLY per-version output location; the old `logs/` + `inference_results/`
  split is gone.
- `../26.6_wind_archive/` -- OUTSIDE the repo, the on-disk graveyard for retired
  binaries; the corresponding code is recoverable via `archive/*` git tags.

### Known sharp edges
- `conv_first`'s weight SHAPE has stayed constant across several architecture
  versions even when the MEANING of its channels changed underneath it (e.g.
  the bicubic-residual-vs-raw-EnsCGP decomposition removed at v7). A
  shape-matching partial checkpoint load can silently transfer weights
  calibrated to the wrong input semantics without erroring.
- The EnsCGP-Cholesky-derived per-pixel prior that used to seed the
  uncertainty head's initial spread was removed in favor of a single global,
  data-measured constant -- a deliberate simplification, not an oversight;
  check CHANGELOG before assuming per-pixel seeding still exists.