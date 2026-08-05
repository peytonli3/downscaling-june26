"""Ensemble Conditional Gaussian Process (Ens-CGP) model.

Implements Ens-CGP following Ravela et al. (2026), arXiv:2602.13871.

The Ens-CGP defines a conditional Gaussian law from ensemble statistics:
    Prior: f ~ N(f_bar, K),  where K = A @ A.T  (empirical ensemble covariance)
    Observation model: y = H f + eps,  eps ~ N(0, R)
    Gain: G = K H.T @ inv(H K H.T + R)
    Posterior mean: m_post = f_bar + G @ (y - H f_bar)
    Posterior covariance: K_post = K - G @ H @ K

For downscaling with H = I (identity), this simplifies to:
    G = K_YX @ inv(K_XX + R)
    M = G  (the linear downscaling operator)

The regularization R = sigma_obs^2 * I is the observation noise covariance,
which is the *only* regularization in the Ens-CGP framework.
This is equivalent to Tikhonov regularization with lambda = sigma_obs^2.

Key difference from ridge regression (cgp/model.py):
- CGP ridge: uses unscaled anomalies dX, so lambda acts on singular values of dX
- Ens-CGP: uses scaled anomalies A_X = dX / sqrt(E-1), so sigma_obs^2 acts on
  singular values of A_X.  Equivalently, lambda_cgp = sigma_obs^2 * (E-1).

Environment: downscaling
    micromamba activate downscaling
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pathlib
import pickle
import sys
from scipy import linalg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from core.utils import select_regularization_gcv, select_regularization_lcurve


@dataclass
class EnsCGPModel:
    """Container for a trained Ensemble CGP model.
    
    Unlike CGPModel, this stores posterior covariance information
    for uncertainty quantification.
    
    The regularization is the observation noise variance sigma_obs^2,
    which enters as R = sigma_obs^2 * I in the Kalman gain formula.
    """
    
    # Regression matrix (for point estimates, similar to CGP)
    M: np.ndarray
    mX: np.ndarray
    mY: np.ndarray
    
    # Ensemble statistics for posterior covariance
    A_Y: np.ndarray              # Anomaly matrix for Y (target)
    K_Y_diag: np.ndarray         # Diagonal of prior covariance (for efficiency)
    
    # Posterior covariance information (stored efficiently)
    K_post_diag: Optional[np.ndarray]  # Diagonal of posterior covariance
    U_post: Optional[np.ndarray]       # Low-rank factor: K_post ≈ U @ U.T
    
    # Metadata
    nanmask_flat: np.ndarray
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    obs_noise_var: float         # sigma_obs^2 (the ONLY regularization)
    n_ensemble: int
    tag: str = ""


def create_nanmask(array: np.ndarray) -> np.ndarray:
    """Create a binary mask where NaNs are 0 and valid pixels are 1."""
    mask = np.ones_like(array, dtype=np.float32)
    mask[np.isnan(array)] = 0
    return mask


def merge_nanmask(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    strategy: str = "intersection",
) -> np.ndarray:
    """Combine two masks with either intersection or choosing one side."""
    if mask_a.shape != mask_b.shape:
        raise ValueError("Nanmask sizes differ; grids are inconsistent.")

    if strategy == "intersection":
        merged = mask_a.astype(bool) & mask_b.astype(bool)
    elif strategy == "hist":
        merged = mask_a.astype(bool)
    elif strategy == "fut":
        merged = mask_b.astype(bool)
    else:
        raise ValueError("strategy must be intersection|hist|fut")

    return merged.astype(np.int8)


def build_training_matrices(
    input_data: np.ndarray, target_data: np.ndarray, nanmask_flat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten (n, y, x) arrays into (n_valid_pixels, n_samples) matrices."""
    n_samples, ny, nx = input_data.shape
    n_pixels = ny * nx

    X = np.reshape(input_data, (n_samples, n_pixels), order="F").T
    X = X[nanmask_flat.astype(bool)]

    Y = np.reshape(target_data, (n_samples, n_pixels), order="F").T
    Y = Y[nanmask_flat.astype(bool)]

    return X, Y


def compute_ensemble_statistics(
    X: np.ndarray, Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ensemble mean and anomaly matrices.
    
    Args:
        X: Input data matrix (n_valid_pixels, n_samples)
        Y: Target data matrix (n_valid_pixels, n_samples)
    
    Returns:
        mX: Mean of X (n_valid_pixels,)
        mY: Mean of Y (n_valid_pixels,)
        A_X: Anomaly matrix for X (n_valid_pixels, n_samples)
        A_Y: Anomaly matrix for Y (n_valid_pixels, n_samples)
        K_X: Ensemble covariance diagonal for X
        K_Y: Ensemble covariance diagonal for Y
    """
    E = X.shape[1]  # Number of ensemble members
    
    mX = np.mean(X, axis=1)
    mY = np.mean(Y, axis=1)
    
    # Anomaly matrices (scaled by sqrt(E-1) for covariance computation)
    # A @ A.T = ensemble covariance
    A_X = (X - mX[:, np.newaxis]) / np.sqrt(E - 1)
    A_Y = (Y - mY[:, np.newaxis]) / np.sqrt(E - 1)
    
    # Covariance diagonals (variance at each pixel)
    K_X_diag = np.sum(A_X**2, axis=1)
    K_Y_diag = np.sum(A_Y**2, axis=1)
    
    return mX, mY, A_X, A_Y, K_X_diag, K_Y_diag


def train_enscgp(
    X: np.ndarray,
    Y: np.ndarray,
    obs_noise_var: Optional[Union[float, str]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Train Ens-CGP model using the Kalman gain formulation.
    
    Following Ravela et al. (2026), the ensemble defines a Gaussian prior
    with covariance K = A_X @ A_X.T, and the Kalman gain is:
    
        G = K_YX @ inv(K_XX + R),  where R = sigma_obs^2 * I
    
    The observation noise variance sigma_obs^2 is the ONLY regularization.
    It enters the SVD formula as:
    
        M = A_Y @ V @ Sigma @ (Sigma^2 + sigma_obs^2 * I)^{-1} @ U.T
    
    where U @ Sigma @ V.T = SVD(A_X).
    
    Note on relationship to CGP ridge regression (cgp/model.py):
        CGP uses unscaled anomalies dX and regularization lambda.
        Ens-CGP uses scaled anomalies A_X = dX / sqrt(E-1).
        The singular values of A_X are s_i / sqrt(E-1) where s_i are from dX.
        For equivalent M: sigma_obs^2 = lambda_cgp / (E-1).
    
    Args:
        X: Input data matrix (n_valid_pixels, n_samples)
        Y: Target data matrix (n_valid_pixels, n_samples)
        obs_noise_var: Observation noise variance sigma_obs^2.  Accepts:
            - None or 'median': median heuristic (legacy default)
            - 'gcv': Generalized Cross-Validation (recommended)
            - 'lcurve': L-curve corner method
            - float: explicit sigma_obs^2 value
        verbose: Print λ selection diagnostics.
    
    Returns:
        M, mX, mY, A_Y, K_Y_diag, K_post_diag, U_post, obs_noise_var
    """
    n_pixels, E = X.shape  # n_pixels = valid pixels, E = ensemble size
    
    # Compute ensemble statistics
    mX, mY, A_X, A_Y, K_X_diag, K_Y_diag = compute_ensemble_statistics(X, Y)
    
    # SVD of anomaly matrix for efficient computation
    U_X, s_X, Vh_X = np.linalg.svd(A_X, full_matrices=False)
    V_X = Vh_X.T
    
    # --- obs_noise_var (= regularization) selection ---
    if obs_noise_var is None or obs_noise_var == 'median':
        obs_noise_var = float(
            np.median(s_X) / np.sqrt(E - 1) if len(s_X) > 0 else 1e-6
        )
        if verbose:
            print(f"  σ²_obs selection: median heuristic = {obs_noise_var:.6e}")
    elif obs_noise_var == 'gcv':
        obs_noise_var = select_regularization_gcv(s_X, U_X, A_Y, verbose=verbose)
    elif obs_noise_var == 'lcurve':
        obs_noise_var = select_regularization_lcurve(s_X, U_X, A_Y, verbose=verbose)
    elif isinstance(obs_noise_var, (int, float)):
        obs_noise_var = float(obs_noise_var)
        if verbose:
            print(f"  σ²_obs selection: manual = {obs_noise_var:.6e}")
    else:
        raise ValueError(f"Unknown obs_noise_var method: {obs_noise_var!r}. "
                         f"Use None, 'median', 'gcv', 'lcurve', or a float.")
    
    # Compute M via SVD:
    # M = A_Y @ V @ Sigma @ (Sigma^2 + sigma_obs^2 * I)^{-1} @ U.T
    # This is K_YX @ inv(K_XX + R) with R = sigma_obs^2 * I
    s_diag = np.diag(s_X)
    s_square = s_diag @ s_diag
    
    M = A_Y @ V_X @ s_diag @ np.linalg.pinv(
        s_square + obs_noise_var * np.eye(len(s_X))
    ) @ U_X.T
    
    # Compute posterior covariance diagonal for uncertainty quantification
    # K_post = K_YY - K_YX @ inv(K_XX + R) @ K_XY
    #        = A_Y @ A_Y.T - M @ A_X @ A_Y.T
    # Diagonal: K_post_diag = sum(A_Y**2, axis=1) - sum((M @ A_X) * A_Y, axis=1)
    
    MA_X = M @ A_X  # (n_pixels, E)
    K_post_diag = K_Y_diag - np.sum(MA_X * A_Y, axis=1)
    K_post_diag = np.maximum(K_post_diag, 0)  # Ensure non-negative variance
    
    # Low-rank factor for posterior covariance sampling
    # K_post = K_YY - G K_XY
    #        = A_Y A_Y^T - A_Y V Sigma^2 (Sigma^2 + sigma_obs^2 I)^{-1} V^T A_Y^T
    #        = A_Y V diag(sigma_obs^2 / (sigma_i^2 + sigma_obs^2)) V^T A_Y^T
    #        = A_post @ A_post^T
    # where A_post = A_Y V diag(sqrt(sigma_obs^2 / (sigma_i^2 + sigma_obs^2)))
    sqrt_filter = np.sqrt(obs_noise_var / (s_X**2 + obs_noise_var))
    A_post = (A_Y @ V_X) * sqrt_filter[np.newaxis, :]  # (n_pixels, E)
    
    U_post, s_post, _ = np.linalg.svd(A_post, full_matrices=False)
    # Keep only significant components
    rank_thresh = 1e-6 * s_post[0] if len(s_post) > 0 else 1e-10
    significant = s_post > rank_thresh
    U_post = U_post[:, significant] * s_post[significant]
    
    return M, mX, mY, A_Y, K_Y_diag, K_post_diag, U_post, obs_noise_var


def apply_enscgp(
    field: np.ndarray,
    M: np.ndarray,
    mX: np.ndarray,
    mY: np.ndarray,
    nanmask_flat: np.ndarray,
    return_uncertainty: bool = False,
    K_post_diag: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Apply Ens-CGP to a single field (ny, nx).
    
    Args:
        field: Input field (ny, nx)
        M: Regression matrix from training
        mX, mY: Ensemble means
        nanmask_flat: Valid pixel mask
        return_uncertainty: If True, also return uncertainty estimate
        K_post_diag: Posterior covariance diagonal for uncertainty
        
    Returns:
        pred: Predicted field (ny, nx)
        std: (optional) Standard deviation field (ny, nx) if return_uncertainty=True
    """
    ny, nx = field.shape
    field_flat = np.reshape(field, (ny * nx,), order="F")
    pred = np.full_like(field_flat, np.nan)
    
    valid_mask = nanmask_flat.astype(bool)
    pred[valid_mask] = M @ (field_flat[valid_mask] - mX) + mY
    pred[pred < 0] = 0
    
    pred_out = np.reshape(pred, (ny, nx), order="F")
    
    if return_uncertainty and K_post_diag is not None:
        std = np.full_like(field_flat, np.nan)
        std[valid_mask] = np.sqrt(K_post_diag)
        std_out = np.reshape(std, (ny, nx), order="F")
        return pred_out, std_out
    
    return pred_out


def apply_enscgp_batch(
    fields: np.ndarray,
    M: np.ndarray,
    mX: np.ndarray,
    mY: np.ndarray,
    nanmask_flat: np.ndarray,
    return_uncertainty: bool = False,
    K_post_diag: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Vectorized application across a batch of fields."""
    n_samples = fields.shape[0]
    result = np.full_like(fields, np.nan)
    
    if return_uncertainty and K_post_diag is not None:
        std_result = np.full_like(fields, np.nan)
        for i in range(n_samples):
            result[i], std_result[i] = apply_enscgp(
                fields[i], M, mX, mY, nanmask_flat,
                return_uncertainty=True, K_post_diag=K_post_diag
            )
        result[result < 0] = 0
        return result, std_result
    
    for i in range(n_samples):
        result[i] = apply_enscgp(fields[i], M, mX, mY, nanmask_flat)
    
    result[result < 0] = 0
    return result


def sample_posterior(
    field: np.ndarray,
    M: np.ndarray,
    mX: np.ndarray,
    mY: np.ndarray,
    U_post: np.ndarray,
    nanmask_flat: np.ndarray,
    n_samples: int = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample from the posterior distribution.
    
    Args:
        field: Input field (ny, nx)
        M, mX, mY: Model parameters
        U_post: Low-rank factor for posterior covariance
        nanmask_flat: Valid pixel mask
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        samples: Array of shape (n_samples, ny, nx)
    """
    if seed is not None:
        np.random.seed(seed)
    
    ny, nx = field.shape
    n_pixels = ny * nx
    valid_mask = nanmask_flat.astype(bool)
    n_valid = np.sum(valid_mask)
    
    # Get posterior mean
    field_flat = np.reshape(field, (n_pixels,), order="F")
    m_post = M @ (field_flat[valid_mask] - mX) + mY
    
    # Generate samples: sample = m_post + U_post @ z, where z ~ N(0, I)
    rank = U_post.shape[1]
    samples = np.zeros((n_samples, n_valid))
    
    for i in range(n_samples):
        z = np.random.randn(rank)
        samples[i] = m_post + U_post @ z
    
    # Map back to full grid
    result = np.full((n_samples, n_pixels), np.nan)
    result[:, valid_mask] = samples
    result[result < 0] = 0
    result = np.reshape(result, (n_samples, ny, nx), order="F")
    
    return result


def save_model(model: EnsCGPModel, path: pathlib.Path) -> None:
    """Persist an Ens-CGP model to disk."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✓ Saved Ens-CGP model to: {path}")


def load_model(path: pathlib.Path) -> EnsCGPModel:
    """Load an Ens-CGP model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    print(f"✓ Loaded Ens-CGP model from: {path}")
    return model


class EnsCGPDownscaler:
    """Wrapper around the Ens-CGP solver for train/inference.
    
    Key difference from CGPDownscaler:
    - Provides uncertainty quantification via posterior covariance
    - Can sample from the posterior distribution
    - Stores both point estimate and uncertainty information
    """
    
    def __init__(
        self,
        obs_noise_var: Optional[Union[float, str]] = None,
        tag: str = "",
    ):
        self.obs_noise_var = obs_noise_var
        self.model: Optional[EnsCGPModel] = None
        self.tag = tag
    
    def fit(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        nanmask_flat: Optional[np.ndarray] = None,
    ) -> "EnsCGPDownscaler":
        """Fit the Ens-CGP model on (N, Y, X) arrays.
        
        Args:
            input_data: Input data (N, ny, nx) - e.g., coarse resolution
            target_data: Target data (N, ny, nx) - e.g., fine resolution
            nanmask_flat: Optional pre-computed nanmask
            
        Returns:
            self for method chaining
        """
        if nanmask_flat is None:
            nanmask_flat = create_nanmask(target_data[0]).flatten(order="F")
        
        X, Y = build_training_matrices(input_data, target_data, nanmask_flat)
        
        M, mX, mY, A_Y, K_Y_diag, K_post_diag, U_post, obs_var = train_enscgp(
            X, Y, self.obs_noise_var
        )
        
        self.model = EnsCGPModel(
            M=M,
            mX=mX,
            mY=mY,
            A_Y=A_Y,
            K_Y_diag=K_Y_diag,
            K_post_diag=K_post_diag,
            U_post=U_post,
            nanmask_flat=nanmask_flat,
            input_shape=input_data.shape[1:],
            output_shape=target_data.shape[1:],
            obs_noise_var=obs_var,
            n_ensemble=input_data.shape[0],
            tag=self.tag,
        )
        
        print(
            f"✓ Ens-CGP model trained (tag={self.tag or 'unnamed'}) "
            f"with sigma_obs^2={obs_var:.6f}, "
            f"ensemble_size={input_data.shape[0]}, valid={np.sum(nanmask_flat.astype(bool))}"
        )
        
        return self
    
    def predict(
        self,
        input_data: np.ndarray,
        return_uncertainty: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply the trained Ens-CGP model.
        
        Args:
            input_data: Input data (ny, nx) or (N, ny, nx)
            return_uncertainty: If True, also return uncertainty estimate
            
        Returns:
            pred: Predicted field(s)
            std: (optional) Standard deviation field(s)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Dispatch to local-patch prediction when model is LocalEnsCGPModel
        from .local_model import LocalEnsCGPModel, LocalEnsCGPDownscaler
        if isinstance(self.model, LocalEnsCGPModel):
            _local = LocalEnsCGPDownscaler.__new__(LocalEnsCGPDownscaler)
            _local.model = self.model
            return _local.predict(input_data, return_uncertainty=return_uncertainty)
        
        if input_data.ndim == 2:
            return apply_enscgp(
                input_data,
                self.model.M,
                self.model.mX,
                self.model.mY,
                self.model.nanmask_flat,
                return_uncertainty=return_uncertainty,
                K_post_diag=self.model.K_post_diag,
            )
        
        return apply_enscgp_batch(
            input_data,
            self.model.M,
            self.model.mX,
            self.model.mY,
            self.model.nanmask_flat,
            return_uncertainty=return_uncertainty,
            K_post_diag=self.model.K_post_diag,
        )
    
    def sample(
        self,
        input_data: np.ndarray,
        n_samples: int = 10,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample from the posterior distribution.
        
        Args:
            input_data: Input field (ny, nx)
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            samples: Array of shape (n_samples, ny, nx)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if input_data.ndim != 2:
            raise ValueError("sample() only supports single field input (2D)")

        # Dispatch to local-patch sampling when model is LocalEnsCGPModel
        from .local_model import LocalEnsCGPModel, LocalEnsCGPDownscaler
        if isinstance(self.model, LocalEnsCGPModel):
            _local = LocalEnsCGPDownscaler.__new__(LocalEnsCGPDownscaler)
            _local.model = self.model
            return _local.sample(input_data, n_samples=n_samples, seed=seed)
        
        return sample_posterior(
            input_data,
            self.model.M,
            self.model.mX,
            self.model.mY,
            self.model.U_post,
            self.model.nanmask_flat,
            n_samples=n_samples,
            seed=seed,
        )
    
    def get_uncertainty(self) -> np.ndarray:
        """Get the posterior standard deviation field.
        
        Returns:
            std_field: Standard deviation at each valid pixel
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        ny, nx = self.model.output_shape
        std = np.full(ny * nx, np.nan)
        valid_mask = self.model.nanmask_flat.astype(bool)
        std[valid_mask] = np.sqrt(self.model.K_post_diag)
        
        return np.reshape(std, (ny, nx), order="F")
    
    def save(self, path: pathlib.Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        
        save_model(self.model, path)
    
    def load(self, path: pathlib.Path) -> "EnsCGPDownscaler":
        self.model = load_model(path)
        self.tag = self.model.tag
        return self
