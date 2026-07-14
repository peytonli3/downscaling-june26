"""Rebuild land_sea_mask_features.npy and topography_features.npy at 5x the WRF
grid resolution (200x200, ~4.4 km/pixel -> 1000x1000, ~0.88 km/pixel), replacing
the native-WRF-resolution versions copied from 26.3_wind/SWIN/preprocessed.

Land mask: rasterized directly from Natural Earth 10m coastline polygons (same
approach as build_hires_land_mask.py) -- sharp at any resolution since it's
vector-based, unlike the original land_sea_mask_features.npy which was a
bilinear interpolation of the coarse 0.25-deg ERA5 LSM raster.

Topography: re-sampled from the already-cached Copernicus/SRTM1 DEM
(26.3_wind/SWIN/preprocessed/copernicus_dem_clip.tif, ~30 m native -- no
re-download needed) onto the finer grid; slope/aspect recomputed from the
finer pixel spacing.

The fine grid nests under the WRF grid: averaging every `factor` consecutive
fine pixels along each axis reproduces the original WRF cell value, so the
fine grid covers exactly the same domain.

Requires cartopy + shapely + rasterio, which are not on the default system
Python -- run this with the `downscaling_all` conda env:
    /home/peytonli/.conda/envs/downscaling_all/bin/python3 build_hires_features.py
"""
from __future__ import annotations

import argparse
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import h5py
import numpy as np


def to_lon180(lon: np.ndarray) -> np.ndarray:
    return np.where(lon > 180.0, lon - 360.0, lon)


def fine_centers(coarse: np.ndarray, factor: int) -> np.ndarray:
    """1D fine-grid cell centers nesting under `coarse`'s cell centers: averaging
    every `factor` consecutive fine cells reproduces the original coarse value."""
    dx = float(coarse[1] - coarse[0])
    edge_min = coarse[0] - dx / 2.0
    dx_fine = dx / factor
    n_fine = len(coarse) * factor
    return edge_min + dx_fine / 2.0 + np.arange(n_fine) * dx_fine


def build_land_mask(lat: np.ndarray, lon180: np.ndarray, resolution: str = "10m") -> np.ndarray:
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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * r * atan2(sqrt(a), sqrt(1.0 - a))


def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=np.float32, copy=True)
    if not np.isnan(out).any():
        return out
    for _ in range(8):
        if not np.isnan(out).any():
            break
        for axis in (0, 1):
            out = np.where(np.isnan(out), np.roll(out, 1, axis=axis), out)
            out = np.where(np.isnan(out), np.roll(out, -1, axis=axis), out)
    if np.isnan(out).any():
        out[np.isnan(out)] = float(np.nanmedian(out))
    return out.astype(np.float32, copy=False)


def build_topography(lat: np.ndarray, lon180: np.ndarray, dem_tif_path: str) -> np.ndarray:
    """(3,H,W) topography features aligned to the fine grid: [elevation_m, slope_deg, aspect_deg]."""
    import rasterio

    lon2d, lat2d = np.meshgrid(lon180, lat)
    with rasterio.open(dem_tif_path) as src:
        coords = list(zip(lon2d.ravel().tolist(), lat2d.ravel().tolist()))
        sampled = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
        elev = sampled.reshape(lat2d.shape)
        nodata = src.nodata
        if nodata is not None:
            elev = np.where(elev == nodata, np.nan, elev)

    elev = np.where((elev > 9000.0) | (elev < -500.0), np.nan, elev)
    elev = np.where((elev == 32767.0) | (elev == -32768.0), np.nan, elev)
    elev = _fill_nan_nearest(elev)
    elev = np.maximum(elev, 0.0).astype(np.float32, copy=False)

    dy_m = _haversine_km(lat[0], lon180[0], lat[1], lon180[0]) * 1000.0
    dx_m = _haversine_km(lat[0], lon180[0], lat[0], lon180[1]) * 1000.0
    dz_dy, dz_dx = np.gradient(elev, dy_m, dx_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))).astype(np.float32)
    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))
    aspect_deg = ((aspect + 360.0) % 360.0).astype(np.float32)

    return np.stack([elev, slope_deg, aspect_deg], axis=0).astype(np.float32)


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat_path", type=str, default="/net/momo/data/projects/downscaling/datasource/wndata.mat")
    parser.add_argument(
        "--dem_tif_path", type=str,
        default="/home/peytonli/26.3_wind/SWIN/preprocessed/copernicus_dem_clip.tif",
    )
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--lsm_output", type=Path, default=data_dir / "land_sea_mask_features.npy")
    parser.add_argument("--topo_output", type=Path, default=data_dir / "topography_features.npy")
    args = parser.parse_args()

    with h5py.File(args.mat_path, "r") as f:
        wrf_lat = np.asarray(f["wrflats"][()][:, 0], dtype=np.float64)
        wrf_lon = to_lon180(np.asarray(f["wrflons"][()][0, :], dtype=np.float64))

    fine_lat = fine_centers(wrf_lat, args.factor)
    fine_lon = fine_centers(wrf_lon, args.factor)
    km_per_deg_lat = _haversine_km(fine_lat[0], fine_lon[0], fine_lat[1], fine_lon[0])
    print(f"Fine grid: {fine_lat.shape[0]}x{fine_lon.shape[0]}, ~{km_per_deg_lat * 1000:.0f} m/pixel")

    land_mask = build_land_mask(fine_lat, fine_lon)
    # fine_lat is ascending (south-up, inheriting wrf_lat order); flip to north-up.
    lsm = land_mask.astype(np.float32)[None, ::-1, :]
    args.lsm_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.lsm_output, lsm)
    print(f"Saved {args.lsm_output}: {lsm.shape}, land fraction={lsm.mean():.3f}")

    topo = build_topography(fine_lat, fine_lon, args.dem_tif_path)
    # Flip to north-up to match the land mask convention above.
    topo = topo[:, ::-1, :]
    np.save(args.topo_output, topo)
    print(f"Saved {args.topo_output}: {topo.shape}, elev range=[{topo[0].min():.1f}, {topo[0].max():.1f}] m")


if __name__ == "__main__":
    main()
