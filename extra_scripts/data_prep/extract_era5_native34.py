"""Save ERA5 u/v at native 34x34 resolution from wndata.mat (no bicubic upsampling).

Mirrors the channel-split convention used in
26.3_wind/SWIN/preprocessing/preprocess_wndata_to_npy.py (split_uv_batch),
so sample order/orientation match the existing era5_uv_2ch.npy (200x200,
bicubic-upsampled) in 26.6_wind/data.

Example (--mat_path / --output default via scripts/paths.py):
    python extract_era5_native34.py
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


def split_uv_batch(flat_batch: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Vectorized split from (B, 2*ny*nx) to (B, 2, ny, nx), matching MATLAB layout."""
    n = ny * nx
    u = flat_batch[:, :n].reshape(flat_batch.shape[0], ny, nx, order="F").transpose(0, 2, 1)
    v = flat_batch[:, n:].reshape(flat_batch.shape[0], ny, nx, order="F").transpose(0, 2, 1)
    return np.stack([u, v], axis=1).astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Extract native-resolution ERA5 u/v from wndata.mat")
    parser.add_argument("--mat_path", type=str, default=str(WNDATA_MAT))
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "era5_uv_2ch_native34.npy"))
    parser.add_argument("--era_h", type=int, default=34)
    parser.add_argument("--era_w", type=int, default=34)
    parser.add_argument("--chunk_size", type=int, default=512)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.mat_path, "r") as f:
        era_ds = f["era5ens"]
        n = int(era_ds.shape[0])
        print(f"Loaded era5ens: shape={era_ds.shape}, dtype={era_ds.dtype}")

        out = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(n, 2, args.era_h, args.era_w)
        )

        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            era_flat = era_ds[start:end].astype(np.float32)  # (B, 2312)
            out[start:end] = split_uv_batch(era_flat, args.era_h, args.era_w)
            print(f"Processed {end}/{n}", flush=True)

    print(f"Saved: {out_path} shape={out.shape}")


if __name__ == "__main__":
    main()
