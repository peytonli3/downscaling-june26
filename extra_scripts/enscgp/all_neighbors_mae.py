import argparse

import numpy as np
import random

def load_era_from_npy(npy_path: str) -> np.ndarray:
	"""Load ERA5 from a preprocessed .npy file and flatten spatial dims.

	Accepts arrays of shape (N, C, H, W) and returns (N, C*H*W).
	"""
	arr = np.load(npy_path, mmap_mode="r")
	if arr.ndim != 4:
		raise ValueError(f"Unsupported WRF npy shape: {arr.shape}")
	N, C, H, W = arr.shape
	flat = arr.reshape(N, C * H * W)
	return flat.astype(np.float32)

def main(argv: list[str] | None = None) -> int:
	npy_path = "/home/peytonli/26.6_wind/data/wrf_uv.npy"
	era = load_era_from_npy(npy_path)
	print(f"Loaded WRF data with shape {era.shape}")
	random.seed(42)
	sample_indices = random.sample(range(era.shape[0]), min(50, era.shape[0]))
	print("Sample indices:", sample_indices)
	maes = []
	for idx in sample_indices:
		maes.append(np.mean([np.mean(np.abs(era[idx] - era[k])) for k in range(era.shape[0]) if k != idx]))
	print("MAEs for 50 random samples:", maes)
	print("Average MAE:", np.mean(maes))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
