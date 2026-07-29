# 26.6_wind

Wind downscaling: EnsCGP posterior + a probabilistic Swin2SR refiner, supervised
with a Laplacian-pyramid / sliced-Wasserstein multiscale loss.

## Layout

```
scripts/        Live pipeline (flat — modules import each other by bare name).
                  network_swin2sr, new_enscgp_swin, terrain_encoder, multiscale_loss,
                  coarsening_operator, enscgp_train, train_new_enscgp_swin,
                  precompute_prior_spread, train_finetune_variance, *_config.json
extra_scripts/  Eval / diagnostics / plotting, organized by function:
                  enscgp/ · graphing/ · ken_enscgp/ (vendored) + top-level diagnostics
data/           Inputs + derived arrays (gitignored). See data/README.md.
logs/           Per-version training logs + checkpoints (gitignored except *.log/*.csv).
inference_results/  Per-version figures/CSVs (gitignored except *.csv).
_archive/       On-disk graveyard for retired artifacts (gitignored).
CHANGELOG.md    Breaking (checkpoint-incompatible) architecture changes, by version.
```

Only `scripts/` and `extra_scripts/` are code and fully tracked. The heavy dirs
(`data/`, checkpoints, `*.npy`, `*.png`, optuna `*.db`) are gitignored and stay on
disk; small text logs (`*.log`, `*.csv`) are tracked as cheap history.

## Versioning

**A non-compatible version = a git tag.** When a change breaks checkpoint
compatibility, commit and tag it:

```bash
git commit -am "v6: <what broke compatibility>"
git tag v6-<date>
```

Revisit an old version with `git checkout <tag>` — it restores the exact
model/train/config. No more snapshot directories inside `scripts/`.

- Current tip is tagged `v7-0729`.
- `archive/pre-reorg-snapshots` holds the original tree, including the `0626_v1`
  and `0628_v2` snapshot dirs. Retrieve old code without checking out, e.g.:
  `git show archive/pre-reorg-snapshots:scripts/0626_v1/new_enscgp_swin.py`
- `archive/resnet-refiner` — the retired ResNet-refiner line (code); its binaries
  live in `_archive/`.
- `archive/variance-conditioning` — the retired `variance_conditioning` model path
  (removed at 0714) and its supporting scripts (removed at 0729); the data
  products it consumed live in `_archive/`.
- **v2 (0627) and v4 (0629) code is not recoverable** — it was overwritten in the
  mutable top-level `scripts/` before git existed. `CHANGELOG.md` describes the
  diffs, but there is no source snapshot. (This is exactly the gap git now closes.)

## Runs (going forward)

Training paths come from `scripts/new_enscgp_swin_config.json`
(`data_dir`, `log_dir`, `splits_path`). For the next version, point `log_dir` at
`runs/<version>/` to keep logs, checkpoints, and figures for a version together.
Existing `logs/` and `inference_results/` were left in place.
