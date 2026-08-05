# 26.6_wind

Wind downscaling: an EnsCGP (ensemble-conditional Gaussian process) posterior
first-guess, refined by a Swin2SR transformer that predicts per-pixel quantiles
(q10/q50/q90) of ERA5-to-WRF downscaled wind (u, v), supervised with a
Laplacian-pyramid / sliced-Wasserstein multiscale loss.

## Layout

```
scripts/              The live pipeline. Flat by design -- modules import each other
                      by bare name. paths.py resolves every path in the repo.
extra_scripts/
  data_prep/          Builds data/ from the upstream wndata.mat + DEM.
  enscgp/             EnsCGP stage: sigma/lambda tuning, neighbor diagnostics,
                      variance recalibration.  graphing/ for its plots.
  swin/               Swin stage. The recurring evals live at this level;
                      _common.py is the shared harness they all use.
    graphing/         Plots (training curves, band decompositions).
    oneoff/           One-time investigations, kept for provenance rather than
                      rerun each version. _v6_0714_arch/ pins the v6 model class.
  presentation/       .pptx generators (they write into runs/<version>/).
  vendor/ken_enscgp/  Third-party Ens-CGP reference implementation.
runs/<version>/       Everything a run produced:
                        train_*.log        (tracked)
                        checkpoints/*.pth  (gitignored)
                        figures/           (gitignored except *.csv)
                        config.json        (tracked, where present)
data/                 Inputs + derived arrays. Gitignored; see data/README.md.
CHANGELOG.md          Checkpoint-incompatible architecture changes, by version.
```

Only `scripts/` and `extra_scripts/` are code. Everything under `runs/` is
output, so it can be regenerated, blanket-ignored, or deleted without touching
source — which is why the presentation generators live in `extra_scripts/` even
though their `.pptx` output lands in `runs/`.

## Paths

Nothing in this repo hardcodes an absolute path. `scripts/paths.py` is the single
source: it locates the repo from its own file location and exports `REPO`,
`DATA_DIR`, `RUNS_DIR`, `SPLITS_PATH`, plus `resolve()` for config values (which
may be absolute or repo-relative). Scripts outside `scripts/` bootstrap with:

```python
REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))
```

That walks *up* rather than counting `.parent` hops, so moving a script between
subdirectories cannot silently break it.

Environment overrides, for pointing the heavy trees at other storage:

| Variable | Default |
|---|---|
| `WIND_DATA_DIR` | `<repo>/data` |
| `WIND_RUNS_DIR` | `<repo>/runs` |
| `WIND_MAT_PATH` | the upstream `wndata.mat` |
| `WIND_DEM_TIF`  | the Copernicus DEM clip |

`python scripts/paths.py` prints every resolved path and whether it exists.

## Running things

Training (see `CLAUDE.md` for the conda env and GPU etiquette):

```bash
cd scripts
python train_new_enscgp_swin.py [--device cuda:N] [--resume PATH] [--max-steps N]
```

The recurring evaluations, pointed at any run's checkpoint:

```bash
cd extra_scripts/swin
python eval_checkpoint.py            --checkpoint ../../runs/<version>/checkpoints/best.pth
python eval_quantile_calibration.py  --checkpoint ...   # PIT histogram + coverage maps
python compare_eigenspectra.py       --checkpoint ...   # spectral fidelity of q50
python eval_pinball_impact.py        --checkpoint ...
python graphing/plot_training_curves.py --log_dir ../../runs/<version>
```

Multi-mode tools take a subcommand: `neighbor_mae.py {rank,mean,sweep-k,baseline}`,
`tune_sigma.py {ssr,mae,decouple}`, `bias_diagnostic.py --part {a,b,c,d,all}`.

## Versioning

**A checkpoint-incompatible change = a git tag.**

```bash
git commit -am "vN: <what broke compatibility>"
git tag vN-<date>
```

`git checkout <tag>` restores the exact model/train/config for that version.
Not every experimental change is tagged — check `git log --oneline` too.

- Current tip is tagged `v7-0729`.
- `archive/pre-reorg-snapshots` holds the original tree including the `0626_v1`
  and `0628_v2` snapshot dirs. Read without checking out:
  `git show archive/pre-reorg-snapshots:scripts/0626_v1/new_enscgp_swin.py`
- `archive/resnet-refiner` — the retired ResNet-refiner line.
- `archive/variance-conditioning` — the retired `variance_conditioning` path.
- **v2 (0627) and v4 (0629) code is not recoverable** — it was overwritten in a
  mutable `scripts/` before git existed. `CHANGELOG.md` describes the diffs, but
  there is no source snapshot. This is the gap git now closes.

Retired binaries (old checkpoints, optuna studies, dead data products) live
outside the repo in `../26.6_wind_archive/`; the code that produced them is
recoverable from the `archive/*` tags above.
