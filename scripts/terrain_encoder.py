"""Small CNN encoder turning the static terrain inputs (land_sea_mask_features.npy,
topography_features.npy -- 4 channels at 1000x1000, 5x the WRF grid: land mask,
z-scored elevation, and z-scored u/v slope components) into a learned (32, 200, 200)
feature map at the WRF/model grid.

Architecture:
1. 3x3 conv, 4 -> 8, LeakyReLU      -- fine-scale extraction at native (1000x1000)
   resolution, where pixel adjacency still reflects the true coastline/slope/aspect
   geometry rather than block-quantized WRF cells.
2. PixelUnshuffle(5): (8, 1000, 1000) -> (200, 200, 200)   -- lossless spatial
   reduction to the WRF grid; all 25 sub-cells of each WRF pixel become channels.
3. LeakyReLU
4. 3x3 conv, 200 -> 64, LeakyReLU   -- reproject the folded sub-cell channels with
   spatial context, so neighboring coarse cells' sub-cell arrangements interact.
5. 3x3 conv, 64 -> 32               -- final terrain feature map.

Usage:
    python terrain_encoder.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LEAKY_SLOPE = 0.2  # matches the rest of this project's Swin2SR backbone (network_swin2sr.py)


class TerrainEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, unshuffle_factor: int = 5, stem_channels: int = 8):
        super().__init__()
        self.unshuffle_factor = unshuffle_factor
        unshuffled_channels = stem_channels * unshuffle_factor ** 2  # 8 * 25 = 200

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
        )
        self.unshuffle = nn.PixelUnshuffle(unshuffle_factor)
        self.body = nn.Sequential(
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(unshuffled_channels, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(8, 4, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.unshuffle(x)
        return self.body(x)


def _zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / a.std()


def load_terrain_input(data_dir: Path) -> torch.Tensor:
    """(4, 1000, 1000) tensor: [land_mask, elevation_z, u_slope_z, v_slope_z].

    Elevation is z-score normalized. Slope (magnitude, deg) and aspect (compass
    bearing, deg; 0=north, 90=east) are recombined into Cartesian (u, v) slope
    components -- avoiding the 359/0 degree wraparound discontinuity in raw
    aspect -- and each z-score normalized in turn.
    """
    lsm = np.load(data_dir / "land_sea_mask_features.npy")[0]  # (1000, 1000)
    topo = np.load(data_dir / "topography_features.npy")  # (3, 1000, 1000)
    elevation, slope_deg, aspect_deg = topo[0], topo[1], topo[2]

    aspect_rad = np.radians(aspect_deg)
    u_slope = slope_deg * np.sin(aspect_rad)  # eastward component
    v_slope = slope_deg * np.cos(aspect_rad)  # northward component

    channels = np.stack(
        [lsm, _zscore(elevation), _zscore(u_slope), _zscore(v_slope)], axis=0
    )
    return torch.from_numpy(channels).float()


if __name__ == "__main__":
    from paths import DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "terrain_features.npy")
    parser.add_argument("--seed", type=int, default=42, help="Encoder weights are randomly initialized (untrained); fixes them for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = load_terrain_input(DATA_DIR).unsqueeze(0)  # (1, 4, 1000, 1000)
    model = TerrainEncoder()
    with torch.no_grad():
        out = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")

    out_arr = out.squeeze(0).numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out_arr)
    print(f"Saved {args.output}: {out_arr.shape}")
