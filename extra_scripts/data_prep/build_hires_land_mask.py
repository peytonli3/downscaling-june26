"""Build a high-resolution land/sea mask for the WRF and ERA5 grids from Natural Earth vector
coastlines, as a sharper alternative to the 0.25-deg ERA5 LSM raster used as a model feature in
26.3_wind (data/land_sea_mask_features.npy, ~32x32 native cells spanning this ~8x8 deg domain --
very coarse next to WRF's 200x200, ~4.4 km/pixel grid).

Natural Earth's 10m land polygons are rasterized directly onto each grid's exact lat/lon cell
centers via shapely.vectorized.contains, so the result is limited only by the true coastline
geometry (effectively much finer than even the WRF grid), not by a coarse intermediate raster.

Requires cartopy + shapely, which are not on the default system Python -- run this with the
project conda env (see CLAUDE.md), e.g.
    python build_hires_land_mask.py

Output: data/land_mask_hires.npz with keys "wrf" (200,200) and "era34" (34,34), boolean,
True = land. plot_enscgp_results.py loads this directly (no cartopy/shapely needed at plot time).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "paths.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))

from paths import DATA_DIR, WNDATA_MAT  # noqa: E402


def to_lon180(lon: np.ndarray) -> np.ndarray:
    return np.where(lon > 180.0, lon - 360.0, lon)


def build_hires_land_mask(lat: np.ndarray, lon180: np.ndarray, resolution: str = "10m") -> np.ndarray:
    """Boolean land mask, shape (len(lat), len(lon180)), True = land."""
    import cartopy.io.shapereader as shpreader
    import shapely.vectorized
    from shapely.geometry import box
    from shapely.ops import unary_union

    shp_path = shpreader.natural_earth(resolution=resolution, category="physical", name="land")
    reader = shpreader.Reader(shp_path)

    pad = 1.0
    bbox = box(lon180.min() - pad, lat.min() - pad, lon180.max() + pad, lat.max() + pad)
    geoms = [g for g in reader.geometries() if g.intersects(bbox)]
    land = unary_union(geoms)

    lon2d, lat2d = np.meshgrid(lon180, lat)
    return shapely.vectorized.contains(land, lon2d, lat2d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat_path", type=str, default=str(WNDATA_MAT))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "land_mask_hires.npz")
    parser.add_argument("--resolution", type=str, default="10m", choices=["10m", "50m", "110m"])
    args = parser.parse_args()

    with h5py.File(args.mat_path, "r") as f:
        wrf_lat = np.asarray(f["wrflats"][()][:, 0], dtype=np.float64)
        wrf_lon = to_lon180(np.asarray(f["wrflons"][()][0, :], dtype=np.float64))
        era_lat = np.asarray(f["eralats"][()].ravel(), dtype=np.float64)
        era_lon = to_lon180(np.asarray(f["eralons"][()].ravel(), dtype=np.float64))

    wrf_mask = build_hires_land_mask(wrf_lat, wrf_lon, args.resolution)
    era_mask = build_hires_land_mask(era_lat, era_lon, args.resolution)

    # wrf_lat is ascending (south-up); flip wrf_mask to north-up.
    # era_lat is descending (already north-up); era_mask is left as-is.
    wrf_mask = wrf_mask[::-1, :]

    print(f"WRF mask: {wrf_mask.shape}, land fraction={wrf_mask.mean():.3f}")
    print(f"ERA5 mask: {era_mask.shape}, land fraction={era_mask.mean():.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, wrf=wrf_mask, era34=era_mask)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
