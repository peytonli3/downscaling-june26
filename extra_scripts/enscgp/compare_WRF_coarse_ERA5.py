import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from coarsening_operator import load_coarsening_operator  # noqa: E402


def main():
    coarsen_operator_path = "/home/peytonli/26.6_wind/data/coarsening_operator_H.npy"
    H, valid = load_coarsening_operator(coarsen_operator_path)
    print(f"Coarsening operator H shape: {H.shape}, valid LR cells: {valid.sum()}/{valid.size}")

    wrf_path = "/home/peytonli/26.6_wind/data/wrf_uv.npy"
    wrf_data = np.load(wrf_path, mmap_mode="r")
    print("WRF data shape:", wrf_data.shape)

    # Native-resolution ERA5 (34x34) -- the actual LR observation H is meant to predict, not the
    # bicubic-upsampled-to-200x200 era5_uv_2ch.npy.
    era5_path = "/home/peytonli/26.6_wind/data/era5_uv_2ch_native34.npy"
    era5_data = np.load(era5_path, mmap_mode="r")
    print("ERA5 data shape:", era5_data.shape)

    # H is (1156, 40000): a single-channel HR(200x200)->LR(34x34) operator. Apply it to each
    # (u, v) field separately by flattening the spatial dims and batching over (sample, channel).
    N, C, Hh, Ww = wrf_data.shape
    wrf_flat = wrf_data.reshape(N * C, Hh * Ww)  # (N*C, 40000), zero-copy view
    coarse_flat = H @ wrf_flat.T  # sparse (1156, 40000) @ dense (40000, N*C) -> (1156, N*C)
    WRF_coarse = np.asarray(coarse_flat).T.reshape(N, C, 34, 34)
    print("Coarsened WRF data shape:", WRF_coarse.shape)

    # Some LR edge cells aren't covered by the HR domain at all (H's row is all-zero there), so
    # they carry no information and must be excluded from the comparison.
    valid_grid = valid.reshape(34, 34)
    diff = WRF_coarse[:, :, valid_grid] - era5_data[:, :, valid_grid]
    mean_diff = np.mean(diff)
    var_diff = np.var(diff)
    print(f"Mean absolute difference between coarsened WRF and ERA5 (valid cells): {mean_diff:.4f}")
    print(f"Variance of difference between coarsened WRF and ERA5 (valid cells): {var_diff:.4f}")


if __name__ == "__main__":
    main()
