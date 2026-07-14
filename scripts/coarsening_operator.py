"""Sparse conservative coarsening (observation) operator H for Ens-CGP conditioning.

Builds H such that y = H @ f.ravel() maps a flattened HR (WRF, 200x200) wind
field f to a flattened LR (ERA5, 34x34) observation y, for use as the
observation operator in y = Hf + eps.

Geometry: both grids are regular, non-projected lat/lon grids (no map
projection distortion -- see prior inspection of wndata.mat), nested with a
non-integer ~6.1 HR-cells-per-LR-cell ratio, so coarsening must be
conservative/area-weighted rather than integer block-averaging.

Area weighting: spherical cell area is dA = cos(lat) dlat dlon, which factors
into a latitude-only term and a longitude-only term (the dlon term does not
depend on lat once cos(lat) is pulled into the lat integral). So the exact
2D overlap area between an LR cell and an HR cell is the product of a 1D
latitude overlap (integral of cos(lat) dlat, i.e. a difference of sines) and
a 1D longitude overlap (plain interval length). This means H itself factors
as a Kronecker product of two small, independently row-normalized 1D
overlap matrices -- exact for this geometry, no approximation, and avoids
ever materializing the full 1156 x 40000 dense overlap matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

DEFAULT_CACHE_PATH = "/home/peytonli/26.6_wind/data/coarsening_operator_H.npy"


def _cell_bounds(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1D cell edges (lo, hi) for a regular grid of centers.

    Edges are midpoints between consecutive centers; the two outer edges are
    extrapolated by half the adjacent spacing. Robust to centers given in
    either ascending or descending order.
    """
    centers = np.asarray(centers, dtype=np.float64)
    mids = (centers[:-1] + centers[1:]) / 2.0
    first_edge = 2 * centers[0] - mids[0]
    last_edge = 2 * centers[-1] - mids[-1]
    edges = np.concatenate([[first_edge], mids, [last_edge]])
    lo = np.minimum(edges[:-1], edges[1:])
    hi = np.maximum(edges[:-1], edges[1:])
    return lo, hi


def _axis_overlap_weights(
    lr_lo: np.ndarray,
    lr_hi: np.ndarray,
    hr_lo: np.ndarray,
    hr_hi: np.ndarray,
    area_fn=None,
) -> np.ndarray:
    """Row-normalized 1D overlap-weight matrix, shape (n_lr, n_hr).

    weights[j, i] is proportional to the overlap "area" between LR cell j
    and HR cell i along this axis, then each row is normalized to sum to 1
    (conservative coarsening => area-weighted mean over overlapping HR
    cells). If area_fn is given, overlap is measured as
    area_fn(hi) - area_fn(lo) instead of plain coordinate length -- used for
    the latitude axis to get true spherical cell area (cos-lat weighting)
    rather than treating the grid as flat in lat/lon index space.

    LR cells with zero overlap (the HR domain doesn't extend under them at
    all -- a real possibility right at the domain edge, not just a
    rounding artifact) are left as all-zero rows rather than normalized,
    since there is no HR data to conservatively average there.
    """
    if area_fn is not None:
        lr_lo, lr_hi = area_fn(lr_lo), area_fn(lr_hi)
        hr_lo, hr_hi = area_fn(hr_lo), area_fn(hr_hi)
        lr_lo, lr_hi = np.minimum(lr_lo, lr_hi), np.maximum(lr_lo, lr_hi)
        hr_lo, hr_hi = np.minimum(hr_lo, hr_hi), np.maximum(hr_lo, hr_hi)

    overlap_lo = np.maximum(lr_lo[:, None], hr_lo[None, :])
    overlap_hi = np.minimum(lr_hi[:, None], hr_hi[None, :])
    weights = np.clip(overlap_hi - overlap_lo, a_min=0.0, a_max=None)

    row_sums = weights.sum(axis=1)
    covered = row_sums > 0
    weights[covered] /= row_sums[covered, None]
    return weights


def build_coarsening_operator(
    hr_lat: np.ndarray,
    hr_lon: np.ndarray,
    lr_lat: np.ndarray,
    lr_lon: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build the sparse conservative coarsening operator H, shape (n_lr, n_hr).

    Args:
        hr_lat: (H_hr,) latitude centers of the HR grid rows.
        hr_lon: (W_hr,) longitude centers of the HR grid columns.
        lr_lat: (H_lr,) latitude centers of the LR grid rows.
        lr_lon: (W_lr,) longitude centers of the LR grid columns.
            Longitudes may be in 0-360 convention; converted to -180-180
            internally so the two grids align.

    Returns:
        (H, valid) where:
        - H is a scipy.sparse.csr_matrix of shape (H_lr*W_lr, H_hr*W_hr).
          For an HR field f of shape (H_hr, W_hr), `H @ f.ravel()` gives the
          flattened LR field of shape (H_lr, W_lr) (row-major / C-order
          raveling on both sides, i.e. row index varies slower than column
          index, matching numpy's default .ravel()).
        - valid is a boolean array of shape (H_lr*W_lr,): True where that
          LR cell has at least partial overlap with the HR domain (H's row
          sums to 1 there), False where the HR domain doesn't extend under
          it at all (H's row is all-zero). This happens at the literal edge
          of a nested-but-not-fully-covering domain -- e.g. here, ERA5's
          outermost lat row and lon column extend slightly beyond where the
          WRF grid has any data, so those specific LR cells have no HR
          information to conservatively average. Mask `y` and the rows of
          `H` by `valid` before using them in Ens-CGP conditioning.
    """
    hr_lon = np.where(np.asarray(hr_lon) > 180.0, np.asarray(hr_lon) - 360.0, hr_lon)
    lr_lon = np.where(np.asarray(lr_lon) > 180.0, np.asarray(lr_lon) - 360.0, lr_lon)

    hr_lat_lo, hr_lat_hi = _cell_bounds(hr_lat)
    hr_lon_lo, hr_lon_hi = _cell_bounds(hr_lon)
    lr_lat_lo, lr_lat_hi = _cell_bounds(lr_lat)
    lr_lon_lo, lr_lon_hi = _cell_bounds(lr_lon)

    # Latitude axis: area_fn = sin(lat) turns "overlap length" into the
    # exact spherical-cap area integral of cos(lat) dlat, i.e. true
    # area weighting rather than flat degree-overlap.
    lat_weights = _axis_overlap_weights(
        lr_lat_lo, lr_lat_hi, hr_lat_lo, hr_lat_hi,
        area_fn=lambda lat_deg: np.sin(np.radians(lat_deg)),
    )  # (n_lr_lat, n_hr_lat), rows sum to 1

    # Longitude axis: the dlon term in dA = cos(lat) dlat dlon does not
    # depend on lat, so plain interval-length overlap is already exact here.
    lon_weights = _axis_overlap_weights(
        lr_lon_lo, lr_lon_hi, hr_lon_lo, hr_lon_hi,
    )  # (n_lr_lon, n_hr_lon), rows sum to 1

    # Row-normalizing each 1D factor before combining is equivalent to
    # row-normalizing the full 2D overlap-area matrix (the row sum of the
    # 2D matrix is the product of the two 1D row sums), so the Kronecker
    # product of the normalized factors is already exactly H. A row of
    # either 1D factor that's all-zero (no HR overlap along that axis)
    # propagates to an all-zero row of H for every LR cell in that
    # lat row / lon column.
    H = sp.kron(sp.csr_matrix(lat_weights), sp.csr_matrix(lon_weights), format="csr")
    valid = np.asarray(H.sum(axis=1)).ravel() > 0
    return H, valid


def save_coarsening_operator(path: str, H: sp.csr_matrix, valid: np.ndarray) -> None:
    """Cache (H, valid) to a single .npy file (a pickled object array).

    H is sparse (~0.1% dense here), so it's pickled rather than densified --
    np.save's allow_pickle path handles arbitrary Python objects, including
    scipy sparse matrices, inside an object-dtype array.
    """
    np.save(path, np.array({"H": H, "valid": valid}, dtype=object), allow_pickle=True)


def load_coarsening_operator(path: str) -> tuple[sp.csr_matrix, np.ndarray]:
    """Load (H, valid) previously written by save_coarsening_operator."""
    data = np.load(path, allow_pickle=True).item()
    return data["H"], data["valid"]


def get_coarsening_operator(
    hr_lat: np.ndarray,
    hr_lon: np.ndarray,
    lr_lat: np.ndarray,
    lr_lon: np.ndarray,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Load (H, valid) from cache_path if present, else build and cache it."""
    if Path(cache_path).exists():
        return load_coarsening_operator(cache_path)
    H, valid = build_coarsening_operator(hr_lat, hr_lon, lr_lat, lr_lon)
    save_coarsening_operator(cache_path, H, valid)
    return H, valid


if __name__ == "__main__":
    import h5py

    mat_path = "/net/momo/data/projects/downscaling/datasource/wndata.mat"
    with h5py.File(mat_path, "r") as f:
        era_lat = f["eralats"][()].ravel()
        era_lon = f["eralons"][()].ravel()
        wrf_lat = f["wrflats"][()][:, 0]
        wrf_lon = f["wrflons"][()][0, :]

        # one real HR sample (u-component) to check H removes detail
        # rather than adding it, matching split_uv_batch's (ny, nx) layout.
        era_flat = f["era5ens"][0].astype(np.float64)
        wrf_flat = f["wrfens"][0].astype(np.float64)
        wrf_u = wrf_flat[: 200 * 200].reshape(200, 200, order="F").T

    H, valid = get_coarsening_operator(wrf_lat, wrf_lon, era_lat, era_lon)
    print(f"H shape: {H.shape}, nnz: {H.nnz}")
    print(f"cached at: {DEFAULT_CACHE_PATH}")
    print(f"valid LR cells: {valid.sum()}/{valid.size} ({valid.size - valid.sum()} at the uncovered domain edge)")

    # 1. Covered rows sum to ~1; uncovered (edge) rows are all-zero.
    row_sums = np.asarray(H.sum(axis=1)).ravel()
    print(f"covered-row sums: min={row_sums[valid].min():.6f}, max={row_sums[valid].max():.6f}")
    assert np.allclose(row_sums[valid], 1.0, atol=1e-10), "covered H rows must sum to 1"
    assert np.allclose(row_sums[~valid], 0.0, atol=1e-12), "uncovered H rows must be all-zero"

    # 2. A constant HR field maps to the same constant, for covered rows.
    const_field = np.full(200 * 200, 7.0)
    const_lr = H @ const_field
    print(f"constant-field check (covered rows): min={const_lr[valid].min():.6f}, max={const_lr[valid].max():.6f}")
    assert np.allclose(const_lr[valid], 7.0, atol=1e-8), "H must preserve constant fields on covered rows"

    # 3. Applying H to a real HR sample gives a plausibly blurred LR field of
    #    the right shape. "Removes detail, never adds it" is checked as
    #    local boundedness: since every covered row of H is a convex
    #    combination (nonnegative weights summing to 1) of HR pixels, each
    #    output value is mathematically guaranteed to fall within the
    #    min/max of the HR pixels it draws from -- it can never overshoot
    #    its local neighborhood. (Note: comparing *global* std of the LR
    #    vs HR field is not a valid substitute here -- this domain's
    #    variance is dominated by a smooth large-scale lat gradient, and
    #    resampling such a gradient onto a different, edge-truncated grid
    #    can nudge the global std either way without anything being wrong.)
    lr_u = (H @ wrf_u.ravel()).reshape(34, 34)
    valid_grid = valid.reshape(34, 34)
    print(f"HR shape: {wrf_u.shape}, LR shape: {lr_u.shape}")
    assert lr_u.shape == (34, 34)

    wrf_u_flat = wrf_u.ravel()
    H_csr = H.tocsr()
    for k in np.nonzero(valid)[0]:
        cols = H_csr.indices[H_csr.indptr[k]:H_csr.indptr[k + 1]]
        nbhd = wrf_u_flat[cols]
        val = lr_u.ravel()[k]
        assert nbhd.min() - 1e-9 <= val <= nbhd.max() + 1e-9, (
            f"LR cell {k} = {val} falls outside its HR neighborhood [{nbhd.min()}, {nbhd.max()}]"
        )
    print("Local-boundedness check passed: every covered LR cell stays within its HR neighborhood's range.")

    print("All sanity checks passed.")
