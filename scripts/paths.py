"""Repo-relative path resolution -- the one place that knows where things live.

Nothing else in this repo should contain an absolute path. A clone works
wherever it is checked out, and the heavy directories can be moved off the
repo filesystem with an environment variable instead of a code edit.

Importing
---------
`scripts/` modules are already on sys.path together, so they just::

    from paths import DATA_DIR, RUNS_DIR

`extra_scripts/` modules are not, so they bootstrap first. Copy this block
verbatim -- do NOT replace it with a fixed number of `.parent` hops. It walks
UP from the file until it finds the repo, so it keeps working no matter how
deeply the script is nested; the previous fixed-hop version broke silently
every time a script was moved between subdirectories::

    import sys
    from pathlib import Path
    REPO = next(p for p in Path(__file__).resolve().parents
                if (p / "scripts" / "paths.py").is_file())
    sys.path.insert(0, str(REPO / "scripts"))
    from paths import DATA_DIR, RUNS_DIR  # noqa: E402

Overrides
---------
`WIND_DATA_DIR` and `WIND_RUNS_DIR` relocate `data/` and `runs/`. Config files
may give paths either absolute or repo-relative; run them through `resolve()`.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SCRIPTS_DIR = REPO / "scripts"
EXTRA_SCRIPTS_DIR = REPO / "extra_scripts"


def _env_dir(var: str, default: Path) -> Path:
    """`var` from the environment if set, else `default`."""
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


# The two heavy, gitignored trees -- overridable so a clone can point them at
# scratch/network storage without touching code or configs.
DATA_DIR = _env_dir("WIND_DATA_DIR", REPO / "data")
RUNS_DIR = _env_dir("WIND_RUNS_DIR", REPO / "runs")

SPLITS_PATH = DATA_DIR / "splits_70_15_15" / "split_indices.npz"


def run_figures(version: str) -> Path:
    """Figure/CSV output directory for a run: runs/<version>/figures/.

    Every per-version artifact lives under its run -- the old split of logs into
    a top-level `logs/` and figures into a top-level `inference_results/` is gone.
    """
    return RUNS_DIR / version / "figures"

# The upstream raw dataset (ERA5 + WRF ensembles, lat/lon grids, event table).
# Genuinely external to the repo -- everything in data/ is derived from it by the
# scripts in extra_scripts/enscgp/. Override with WIND_MAT_PATH.
WNDATA_MAT = Path(os.environ.get(
    "WIND_MAT_PATH", "/net/momo/data/projects/downscaling/datasource/wndata.mat"))

# Copernicus DEM clip, the terrain source for build_hires_features.py. Also
# external, and currently sitting in a sibling project's preprocessing output.
# Override with WIND_DEM_TIF.
COPERNICUS_DEM_TIF = Path(os.environ.get(
    "WIND_DEM_TIF", "/home/peytonli/26.3_wind/SWIN/preprocessed/copernicus_dem_clip.tif"))


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config-supplied path: absolute wins, relative is REPO-relative."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def repo_root_from(file: str | os.PathLike) -> Path:
    """Locate the repo by walking up from `file`.

    Only useful once `scripts/` is importable; the bootstrap block in this
    module's docstring is the version that works before that.
    """
    for parent in Path(file).resolve().parents:
        if (parent / "scripts" / "paths.py").is_file():
            return parent
    raise RuntimeError(f"no repo root above {file}")


if __name__ == "__main__":
    for name in ("REPO", "SCRIPTS_DIR", "EXTRA_SCRIPTS_DIR", "DATA_DIR",
                 "RUNS_DIR", "SPLITS_PATH", "WNDATA_MAT", "COPERNICUS_DEM_TIF"):
        value = globals()[name]
        print(f"{'ok ' if value.exists() else 'MISSING'}  {name:22s} {value}")
