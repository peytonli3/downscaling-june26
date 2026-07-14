"""
Core Utilities for Multi-Scale Downscaling

This module provides interpolation functions (upscaling, downscaling),
data transformations, and general utility functions.

Environment: downscaling or downscaling_demo
    micromamba activate downscaling
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from skimage.transform import resize
from skimage.measure import block_reduce
from typing import Tuple, Optional, Union, List
import warnings


# ============================================================
# Bicubic Interpolation Functions
# ============================================================

def bicubic_upscale(
    data: np.ndarray,
    target_shape: Tuple[int, int],
    preserve_range: bool = True,
    anti_aliasing: bool = False
) -> np.ndarray:
    """
    Upscale data using bicubic interpolation.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 2D (H, W) or 3D (N, H, W).
    target_shape : tuple
        Target spatial shape (H_target, W_target).
    preserve_range : bool
        If True, preserve the original data range.
    anti_aliasing : bool
        If True, apply Gaussian smoothing before upscaling.
        
    Returns
    -------
    np.ndarray
        Upscaled data with shape (H_target, W_target) or (N, H_target, W_target).
        
    Examples
    --------
    >>> data_48km = np.random.rand(100, 10, 10)  # 100 samples at 48km
    >>> data_25km = bicubic_upscale(data_48km, target_shape=(20, 20))
    >>> data_25km.shape
    (100, 20, 20)
    """
    if data.ndim == 2:
        return resize(
            data, 
            target_shape, 
            order=3,  # Bicubic
            mode='reflect', 
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing
        ).astype(data.dtype)
    elif data.ndim == 3:
        result = np.empty((data.shape[0], *target_shape), dtype=data.dtype)
        for i in range(data.shape[0]):
            result[i] = resize(
                data[i], 
                target_shape, 
                order=3,
                mode='reflect', 
                preserve_range=preserve_range,
                anti_aliasing=anti_aliasing
            ).astype(data.dtype)
        return result
    elif data.ndim == 4:
        # 4D: (N, T, H, W) - batch of sequences
        result = np.empty((data.shape[0], data.shape[1], *target_shape), dtype=data.dtype)
        for i in range(data.shape[0]):
            for t in range(data.shape[1]):
                result[i, t] = resize(
                    data[i, t], 
                    target_shape, 
                    order=3,
                    mode='reflect', 
                    preserve_range=preserve_range,
                    anti_aliasing=anti_aliasing
                ).astype(data.dtype)
        return result
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D array, got {data.ndim}D")


def bicubic_downscale(
    data: np.ndarray,
    target_shape: Tuple[int, int],
    preserve_range: bool = True,
    anti_aliasing: bool = True
) -> np.ndarray:
    """
    Downscale data using bicubic interpolation with optional anti-aliasing.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 2D (H, W) or 3D (N, H, W).
    target_shape : tuple
        Target spatial shape (H_target, W_target).
    preserve_range : bool
        If True, preserve the original data range.
    anti_aliasing : bool
        If True, apply Gaussian smoothing before downscaling (recommended).
        
    Returns
    -------
    np.ndarray
        Downscaled data.
        
    Examples
    --------
    >>> data_3km = np.random.rand(100, 160, 160)  # 100 samples at 3km
    >>> data_12km = bicubic_downscale(data_3km, target_shape=(40, 40))
    >>> data_12km.shape
    (100, 40, 40)
    """
    if data.ndim == 2:
        return resize(
            data, 
            target_shape, 
            order=3,
            mode='reflect', 
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing
        ).astype(data.dtype)
    elif data.ndim == 3:
        result = np.empty((data.shape[0], *target_shape), dtype=data.dtype)
        for i in range(data.shape[0]):
            result[i] = resize(
                data[i], 
                target_shape, 
                order=3,
                mode='reflect', 
                preserve_range=preserve_range,
                anti_aliasing=anti_aliasing
            ).astype(data.dtype)
        return result
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def bilinear_interpolate(
    data: np.ndarray,
    target_shape: Tuple[int, int],
    preserve_range: bool = True
) -> np.ndarray:
    """
    Interpolate data using bilinear interpolation.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 2D (H, W) or 3D (N, H, W).
    target_shape : tuple
        Target spatial shape (H_target, W_target).
    preserve_range : bool
        If True, preserve the original data range.
        
    Returns
    -------
    np.ndarray
        Interpolated data.
    """
    if data.ndim == 2:
        return resize(
            data, 
            target_shape, 
            order=1,  # Bilinear
            mode='reflect', 
            preserve_range=preserve_range,
            anti_aliasing=False
        ).astype(data.dtype)
    elif data.ndim == 3:
        result = np.empty((data.shape[0], *target_shape), dtype=data.dtype)
        for i in range(data.shape[0]):
            result[i] = resize(
                data[i], 
                target_shape, 
                order=1,
                mode='reflect', 
                preserve_range=preserve_range,
                anti_aliasing=False
            ).astype(data.dtype)
        return result
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


# ============================================================
# Area-Weighted Upscaling (Conservative)
# ============================================================

def area_weighted_upscale(
    data: np.ndarray,
    factor: int
) -> np.ndarray:
    """
    Upscale data using area-weighted averaging (conservative method).
    
    This method preserves the total quantity (e.g., precipitation amount)
    by distributing values uniformly across upscaled pixels.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 2D (H, W) or 3D (N, H, W).
    factor : int
        Upscaling factor. Output will be factor times larger.
        
    Returns
    -------
    np.ndarray
        Upscaled data with shape (H*factor, W*factor) or (N, H*factor, W*factor).
        
    Examples
    --------
    >>> data_48km = np.random.rand(100, 10, 10)
    >>> data_12km = area_weighted_upscale(data_48km, factor=4)
    >>> data_12km.shape
    (100, 40, 40)
    """
    if data.ndim == 2:
        return np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)
    elif data.ndim == 3:
        return np.repeat(np.repeat(data, factor, axis=1), factor, axis=2)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def area_weighted_downscale(
    data: np.ndarray,
    factor: int,
    func: callable = np.nanmean
) -> np.ndarray:
    """
    Downscale data using area-weighted aggregation (conservative method).
    
    This method aggregates values using the specified function (default: mean).
    
    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 2D (H, W) or 3D (N, H, W).
    factor : int
        Downscaling factor. Output will be factor times smaller.
    func : callable
        Aggregation function (np.nanmean, np.nansum, etc.)
        
    Returns
    -------
    np.ndarray
        Downscaled data.
        
    Examples
    --------
    >>> data_3km = np.random.rand(100, 160, 160)
    >>> data_12km = area_weighted_downscale(data_3km, factor=4)
    >>> data_12km.shape
    (100, 40, 40)
    """
    if data.ndim == 2:
        return block_reduce(data, (factor, factor), func)
    elif data.ndim == 3:
        return np.stack([
            block_reduce(data[i], (factor, factor), func)
            for i in range(data.shape[0])
        ])
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


# ============================================================
# Coordinate-based Interpolation
# ============================================================

def interpolate_to_grid(
    data: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str = 'cubic'
) -> np.ndarray:
    """
    Interpolate data from source grid to target grid using coordinates.
    
    Parameters
    ----------
    data : np.ndarray
        Input data on source grid. Shape (H_src, W_src) or (N, H_src, W_src).
    source_lat, source_lon : np.ndarray
        Source grid coordinates. Shape (H_src, W_src) or 1D arrays.
    target_lat, target_lon : np.ndarray
        Target grid coordinates. Shape (H_tgt, W_tgt) or 1D arrays.
    method : str
        Interpolation method: 'linear', 'cubic', or 'nearest'.
        
    Returns
    -------
    np.ndarray
        Interpolated data on target grid.
    """
    # Ensure 1D coordinates for RegularGridInterpolator
    if source_lat.ndim == 2:
        source_lat = source_lat[:, 0]
        source_lon = source_lon[0, :]
    if target_lat.ndim == 2:
        target_lat_1d = target_lat[:, 0]
        target_lon_1d = target_lon[0, :]
    else:
        target_lat_1d = target_lat
        target_lon_1d = target_lon
        
    # Create meshgrid of target points
    target_lat_grid, target_lon_grid = np.meshgrid(target_lat_1d, target_lon_1d, indexing='ij')
    target_points = np.column_stack([target_lat_grid.ravel(), target_lon_grid.ravel()])
    
    if data.ndim == 2:
        interp = RegularGridInterpolator(
            (source_lat, source_lon), 
            data, 
            method=method,
            bounds_error=False,
            fill_value=np.nan
        )
        result = interp(target_points).reshape(target_lat_grid.shape)
        return result
    elif data.ndim == 3:
        result = np.empty((data.shape[0], *target_lat_grid.shape), dtype=data.dtype)
        for i in range(data.shape[0]):
            interp = RegularGridInterpolator(
                (source_lat, source_lon), 
                data[i], 
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            result[i] = interp(target_points).reshape(target_lat_grid.shape)
        return result
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


# ============================================================
# NaN Handling Functions
# ============================================================

def create_nanmask(array: np.ndarray) -> np.ndarray:
    """
    Create a binary mask where 1 indicates valid (non-NaN) pixels.
    
    Parameters
    ----------
    array : np.ndarray
        Input array.
        
    Returns
    -------
    np.ndarray
        Binary mask (same shape as input).
    """
    mask = np.ones_like(array, dtype=np.float32)
    mask[np.isnan(array)] = 0
    return mask


def fillnan(field: np.ndarray) -> np.ndarray:
    """
    Fill NaN values in a 2D field using linear interpolation.
    
    Parameters
    ----------
    field : np.ndarray
        2D array with potential NaN values.
        
    Returns
    -------
    np.ndarray
        Array with NaN values filled.
    """
    field_copy = field.copy().flatten()
    mask = np.isnan(field_copy)
    if mask.any() and (~mask).any():
        field_copy[mask] = np.interp(
            np.flatnonzero(mask), 
            np.flatnonzero(~mask), 
            field_copy[~mask]
        )
    return field_copy.reshape(field.shape)


def fillnan_batch(batch: np.ndarray) -> np.ndarray:
    """
    Fill NaN values for a batch of 2D fields.
    
    Parameters
    ----------
    batch : np.ndarray
        3D array (N, H, W) with potential NaN values.
        
    Returns
    -------
    np.ndarray
        Array with NaN values filled.
    """
    return np.stack([fillnan(arr) for arr in batch])


# ============================================================
# Gaussian Smoothing Functions
# ============================================================

def create_gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Create a 2D Gaussian kernel.
    
    Parameters
    ----------
    size : int
        Kernel size (will be size x size).
    sigma : float
        Standard deviation of the Gaussian.
        
    Returns
    -------
    np.ndarray
        Normalized 2D Gaussian kernel.
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def masked_convolve2d(
    array: np.ndarray, 
    kernel: np.ndarray, 
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply 2D convolution with NaN-aware masking.
    
    Parameters
    ----------
    array : np.ndarray
        Input 2D array (NaN values will be handled).
    kernel : np.ndarray
        Convolution kernel.
    mask : np.ndarray
        Valid pixel mask (1 = valid, 0 = invalid).
        
    Returns
    -------
    np.ndarray
        Convolved array.
    """
    array = np.nan_to_num(array)
    array_convolved = ndimage.convolve(array, kernel)
    mask_convolved = ndimage.convolve(mask, kernel)
    return np.divide(
        array_convolved, 
        mask_convolved, 
        out=np.full_like(array_convolved, np.nan), 
        where=mask_convolved != 0
    )


def smooth_field(
    field: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a field, handling NaN values.
    
    Parameters
    ----------
    field : np.ndarray
        Input 2D or 3D array.
    kernel_size : int
        Size of Gaussian kernel.
    sigma : float
        Standard deviation of Gaussian.
    iterations : int
        Number of smoothing iterations.
        
    Returns
    -------
    np.ndarray
        Smoothed field.
    """
    kernel = create_gaussian_kernel(kernel_size, sigma)
    
    if field.ndim == 2:
        result = field.copy()
        permmask = create_nanmask(result)
        for _ in range(iterations):
            tempmask = create_nanmask(result)
            result = masked_convolve2d(result, kernel, tempmask)
        result[permmask == 0] = np.nan
        return result
    elif field.ndim == 3:
        return np.stack([
            smooth_field(field[i], kernel_size, sigma, iterations)
            for i in range(field.shape[0])
        ])
    else:
        raise ValueError(f"Expected 2D or 3D array, got {field.ndim}D")


# ============================================================
# Scale Conversion Utilities
# ============================================================

# Standard grid sizes for each resolution (after PR box crop)
# Actual sizes verified from latlon.nc files with PR_BOX crop
# PR_BOX = {lon_w: -68.53, lon_e: -64.0, lat_s: 15.97, lat_n: 20.5}
# Native grids: 48km=60×90, 25km=120×180, 12km=240×360, 3km=925×1480
SCALE_GRID_SIZES = {
    '48km': (11, 10),    # Verified: lat in [19.09,23.64], lon in [-68.18,-64.55]
    '25km': (21, 20),    # Verified: lat in [17.45,22.45], lon in [-68.55,-63.8]
    '12km': (44, 41),    # Verified: 240×360 native → 44×41 after PR crop
    '3km': (168, 168),   # ~168×168 after PR box crop (TBD exact)
}


def get_scale_factor(source_scale: str, target_scale: str) -> float:
    """
    Calculate the approximate scale factor between two resolutions.
    
    Parameters
    ----------
    source_scale : str
        Source resolution ('48km', '25km', '12km', '3km').
    target_scale : str
        Target resolution.
        
    Returns
    -------
    float
        Approximate scale factor (target_size / source_size).
    """
    scale_values = {'48km': 48, '25km': 25, '12km': 12, '3km': 3}
    return scale_values[source_scale] / scale_values[target_scale]


def get_target_shape(source_scale: str, target_scale: str) -> Tuple[int, int]:
    """
    Get the target grid shape for interpolation.
    
    Parameters
    ----------
    source_scale : str
        Source resolution.
    target_scale : str
        Target resolution.
        
    Returns
    -------
    tuple
        Target grid shape (H, W).
    """
    return SCALE_GRID_SIZES[target_scale]


def convert_scale(
    data: np.ndarray,
    source_scale: str,
    target_scale: str,
    method: str = 'bicubic'
) -> np.ndarray:
    """
    Convert data from one resolution scale to another.
    
    Parameters
    ----------
    data : np.ndarray
        Input data at source scale.
    source_scale : str
        Source resolution ('48km', '25km', '12km', '3km').
    target_scale : str
        Target resolution.
    method : str
        Interpolation method: 'bicubic', 'bilinear', 'area_weighted'.
        
    Returns
    -------
    np.ndarray
        Data at target scale.
        
    Examples
    --------
    >>> data_3km = np.random.rand(100, 160, 160)  # 100 samples at 3km
    >>> data_12km = convert_scale(data_3km, '3km', '12km', method='bicubic')
    >>> data_12km.shape
    (100, 40, 40)
    """
    target_shape = get_target_shape(source_scale, target_scale)
    
    if method == 'bicubic':
        scale_factor = get_scale_factor(source_scale, target_scale)
        if scale_factor > 1:  # Upscaling (e.g., 48km -> 25km)
            return bicubic_upscale(data, target_shape)
        else:  # Downscaling (e.g., 3km -> 12km)
            return bicubic_downscale(data, target_shape)
    elif method == 'bilinear':
        return bilinear_interpolate(data, target_shape)
    elif method == 'area_weighted':
        # Calculate integer factor if possible
        source_shape = SCALE_GRID_SIZES[source_scale]
        if target_shape[0] % source_shape[0] == 0 and target_shape[1] % source_shape[1] == 0:
            # Perfect factor - use area_weighted_upscale
            factor_h = target_shape[0] // source_shape[0]
            factor_w = target_shape[1] // source_shape[1]
            if factor_h == factor_w:
                return area_weighted_upscale(data, factor_h)
        elif source_shape[0] % target_shape[0] == 0 and source_shape[1] % target_shape[1] == 0:
            # Perfect factor - use area_weighted_downscale
            factor_h = source_shape[0] // target_shape[0]
            factor_w = source_shape[1] // target_shape[1]
            if factor_h == factor_w:
                return area_weighted_downscale(data, factor_h)
        # Fall back to bicubic if no perfect factor
        warnings.warn(f"No perfect factor for area_weighted between {source_scale} and {target_scale}, using bicubic")
        return bicubic_upscale(data, target_shape) if get_scale_factor(source_scale, target_scale) > 1 else bicubic_downscale(data, target_shape)
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# Data Normalization Utilities
# ============================================================

def normalize_data(
    data: np.ndarray,
    method: str = 'minmax',
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    Normalize data using specified method.
    
    Parameters
    ----------
    data : np.ndarray
        Input data.
    method : str
        'minmax' or 'zscore'.
    min_val, max_val : float, optional
        Pre-computed min/max for minmax normalization.
    mean, std : float, optional
        Pre-computed mean/std for zscore normalization.
        
    Returns
    -------
    tuple
        (normalized_data, stats_dict)
    """
    if method == 'minmax':
        if min_val is None:
            min_val = np.nanmin(data)
        if max_val is None:
            max_val = np.nanmax(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        stats = {'min': min_val, 'max': max_val}
    elif method == 'zscore':
        if mean is None:
            mean = np.nanmean(data)
        if std is None:
            std = np.nanstd(data)
        normalized = (data - mean) / (std + 1e-8)
        stats = {'mean': mean, 'std': std}
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized, stats


def denormalize_data(
    data: np.ndarray,
    stats: dict,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Denormalize data using saved statistics.
    
    Parameters
    ----------
    data : np.ndarray
        Normalized data.
    stats : dict
        Statistics from normalize_data.
    method : str
        'minmax' or 'zscore'.
        
    Returns
    -------
    np.ndarray
        Denormalized data.
    """
    if method == 'minmax':
        return data * (stats['max'] - stats['min']) + stats['min']
    elif method == 'zscore':
        return data * stats['std'] + stats['mean']
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# Precipitation-specific utilities
# ============================================================

def enforce_non_negative(data: np.ndarray) -> np.ndarray:
    """
    Enforce non-negative values (clip negative to zero).
    Useful for precipitation data.
    """
    return np.clip(data, 0, None)


def log_transform(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Apply log transformation (useful for skewed precipitation distributions).
    """
    return np.log1p(data + epsilon)


def inverse_log_transform(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Inverse log transformation.
    """
    return np.expm1(data) - epsilon


# =============================================================================
# GCV / L-curve regularization selection for SVD-based regression
# =============================================================================

def select_regularization_gcv(
    s: np.ndarray,
    U: np.ndarray,
    dY: np.ndarray,
    n_grid: int = 200,
    verbose: bool = False,
    rank_threshold: float = 1e-3,
) -> float:
    """Select Tikhonov regularization λ via Generalized Cross-Validation (GCV).

    For the SVD regression M = dY V S (S² + λI)^{-1} U^T where dX = U S V^T,
    the multi-output GCV score is computed analytically from the singular values
    and the projections of dY onto the left singular vectors U.

    The GCV criterion minimizes the leave-one-out prediction error *without*
    explicitly recomputing the model for each left-out sample:

        GCV(λ) = (1/n) Σ_j ‖Y_j - Ŷ_j(λ)‖² / (1 - h(λ)/n)²

    where h(λ) = Σ_i σ_i² / (σ_i² + λ) is the effective degrees of freedom
    (trace of the hat matrix).

    **Effective-rank truncation**: only singular values above
    ``rank_threshold * s_max`` are included in the GCV optimisation.
    Modes below this threshold are treated as noise and their projections
    are added to the irreducible residual.  This prevents GCV from
    under-regularising when p ≈ n and the spectrum has a long noise tail.

    Args:
        s: Singular values from SVD of dX (or A_X), shape (r,).
        U: Left singular vectors of dX (or A_X), shape (p, r).
        dY: Target anomaly matrix, shape (p, n).  For Ens-CGP, this is A_Y.
        n_grid: Number of λ candidates to evaluate (log-spaced).
        verbose: Print diagnostic information.
        rank_threshold: Fraction of s_max below which modes are discarded.
            Default 1e-3 corresponds to a condition number of 1000.

    Returns:
        Optimal λ value.

    References:
        Golub, Heath, Wahba (1979) "Generalized cross-validation as a method
        for choosing a good ridge parameter." Technometrics 21(2):215-223.

        Hansen (1998) "Rank-Deficient and Discrete Ill-Posed Problems."
    """
    r = len(s)
    p, n = dY.shape  # p = pixels, n = samples
    s2 = s ** 2

    # ---- Effective rank truncation ----
    s_threshold = s[0] * rank_threshold
    r_eff = int(np.sum(s > s_threshold))
    r_eff = max(r_eff, 1)  # at least 1 mode

    s_eff = s[:r_eff]
    s2_eff = s_eff ** 2

    # Project dY onto U basis: beta_i = U_i^T @ dY, shape (r, n)
    beta = U.T @ dY  # (r, n)

    # Energy in retained modes vs discarded modes
    beta_eff = beta[:r_eff]      # signal modes
    beta_noise = beta[r_eff:]    # noise modes

    # Residual = energy NOT in retained span of U
    dY_norm2 = np.sum(dY ** 2)
    residual_perp = dY_norm2 - np.sum(beta ** 2)   # not in any U
    residual_perp += np.sum(beta_noise ** 2)        # noise-mode energy → residual
    residual_perp = max(residual_perp, 0.0)

    # λ search range: from s_eff_min² * 1e-4 to s_eff_max² * 1e2
    s_min2 = s2_eff[-1] if s2_eff[-1] > 0 else 1e-20
    s_max2 = s2_eff[0]
    lam_min = s_min2 * 1e-4
    lam_max = s_max2 * 1e2
    lambdas = np.logspace(np.log10(lam_min), np.log10(lam_max), n_grid)

    best_lam = lambdas[0]
    best_gcv = np.inf

    gcv_scores = np.empty(n_grid)

    for k, lam in enumerate(lambdas):
        # Filter factors: f_i = σ_i² / (σ_i² + λ)  for retained modes only
        f = s2_eff / (s2_eff + lam)

        # Residual for each retained mode i: (1 - f_i) * beta_i
        # Total residual² = Σ_i (1-f_i)² ||beta_i||² + residual_perp
        resid_coeffs = (1 - f) ** 2
        rss = np.sum(resid_coeffs[:, np.newaxis] * beta_eff ** 2) + residual_perp

        # Effective degrees of freedom = Σ f_i (retained modes only)
        dof = np.sum(f)

        # GCV denominator: (1 - dof/n)²
        denom = (1.0 - dof / n) ** 2
        if denom < 1e-15:
            gcv_scores[k] = np.inf
            continue

        gcv_scores[k] = (rss / (n * p)) / denom

        if gcv_scores[k] < best_gcv:
            best_gcv = gcv_scores[k]
            best_lam = lam

    if verbose:
        # Also compute the median heuristic for comparison
        median_lam = float(np.median(s))
        print(f"  GCV λ search: [{lam_min:.2e}, {lam_max:.2e}], {n_grid} points")
        print(f"  Effective rank: {r_eff} / {r} modes (threshold={rank_threshold})")
        print(f"  GCV optimal λ = {best_lam:.6e} (GCV score = {best_gcv:.6e})")
        print(f"  Median heuristic λ = {median_lam:.6e}")
        print(f"  Ratio GCV/median = {best_lam / median_lam:.3f}")
        dof_opt = np.sum(s2_eff / (s2_eff + best_lam))
        print(f"  Effective DOF at optimal λ: {dof_opt:.1f} / {r_eff} retained modes")

    return float(best_lam)


def select_regularization_lcurve(
    s: np.ndarray,
    U: np.ndarray,
    dY: np.ndarray,
    n_grid: int = 200,
    verbose: bool = False,
) -> float:
    """Select Tikhonov λ via the L-curve corner method.

    Finds the point of maximum curvature on the log-log plot of
    ‖residual‖ vs ‖solution norm‖ as a function of λ.

    Args:
        s, U, dY: Same as select_regularization_gcv.
        n_grid: Number of λ candidates.
        verbose: Print diagnostics.

    Returns:
        Optimal λ at the L-curve corner.
    """
    r = len(s)
    p, n = dY.shape
    s2 = s ** 2

    beta = U.T @ dY  # (r, n)
    dY_norm2 = np.sum(dY ** 2)
    beta_norm2 = np.sum(beta ** 2)
    residual_perp = dY_norm2 - beta_norm2

    s_min2 = s2[-1] if s2[-1] > 0 else 1e-20
    s_max2 = s2[0]
    lambdas = np.logspace(
        np.log10(s_min2 * 1e-4), np.log10(s_max2 * 1e2), n_grid
    )

    log_resid = np.empty(n_grid)
    log_soln = np.empty(n_grid)

    for k, lam in enumerate(lambdas):
        f = s2 / (s2 + lam)

        # Residual norm²
        rss = np.sum((1 - f) ** 2 * np.sum(beta ** 2, axis=1)) + residual_perp

        # Solution norm² (‖M‖_F²) — using SVD form:
        # M_coeff_i = σ_i / (σ_i² + λ), solution norm = Σ_i coeff_i² ||V_i^T dY||²
        # But we use f_i / σ_i as the coefficient: M has coeffs σ_i/(σ_i²+λ) = f_i/σ_i
        soln_norm2 = np.sum((f / s) ** 2 * np.sum(beta ** 2, axis=1))

        log_resid[k] = np.log(max(rss, 1e-30))
        log_soln[k] = np.log(max(soln_norm2, 1e-30))

    # Curvature of the L-curve in log-log space
    # κ = (x'y'' - y'x'') / (x'² + y'²)^{3/2}
    dx = np.gradient(log_resid)
    dy = np.gradient(log_soln)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    # Avoid edges
    curvature[:5] = -np.inf
    curvature[-5:] = -np.inf

    best_idx = np.argmax(curvature)
    best_lam = float(lambdas[best_idx])

    if verbose:
        median_lam = float(np.median(s))
        print(f"  L-curve λ search: [{lambdas[0]:.2e}, {lambdas[-1]:.2e}]")
        print(f"  L-curve optimal λ = {best_lam:.6e}")
        print(f"  Median heuristic λ = {median_lam:.6e}")
        print(f"  Ratio L-curve/median = {best_lam / median_lam:.3f}")

    return best_lam
