"""Two-head probabilistic Swin2SR refiner for wind downscaling (ERA5 -> WRF).

Wraps the Swin2SR backbone (network_swin2sr.py) at upscale=1 -- same-resolution
restoration, not super-resolution -- and replaces its single-conv reconstruction
tail with two parallel heads on top of the embed_dim deep-feature map (shallow
features + RSTB body + conv_after_body + long skip, all unchanged from Swin2SR).

Inputs (forward(posterior, terrain_raw)):
- posterior: (B, 5, H, W) EnsCGP first-guess posterior [u, v, L11, L21, L22]
  (e.g. data/enscgp_posterior.npy, see enscgp_train.py). The mean (first two
  channels) is also used directly as the residual base for mean_head.
- terrain_raw: (1, 4, 1000, 1000) static terrain input for TerrainEncoder (see
  terrain_encoder.py), shared by every sample -- broadcast across the batch.
  TerrainEncoder is a submodule here (not a frozen precomputed feature map),
  so its weights are trained jointly with the rest of the network.

x = cat([posterior, terrain_encoder(terrain_raw)], dim=1) feeds conv_first.

Heads:
- mean_head: 2 channels (mu_u, mu_v), added as a residual to the posterior
  mean. Its final conv's bias is zero and its weight is small random noise
  (not exact zero -- an all-zero weight would also zero the gradient flowing
  back through it, cutting the heads off from the shared backbone at step 0),
  so the model starts as an approximate pass-through of the first guess.
- chol_head: 3 channels (L11, L21, L22), the lower-triangular Cholesky factor
  of the per-pixel 2x2 (u, v) covariance, Sigma = L @ L.T. The diagonal
  (L11, L22) is passed through softplus to stay positive; L21 is
  unconstrained. Its final conv's bias is initialized so the head starts near
  a small unit, uncorrelated covariance rather than a degenerate zero one
  (weight is likewise small random noise, for the same gradient-flow reason).

forward() returns a single (B, 5, H, W) tensor ordered [mu_u, mu_v, L11, L21,
L22] -- matching the EnsCGP output convention.

Architecture and head-init hyperparameters live in a JSON config (see
new_enscgp_swin_config.json), following the same "model" section convention
as 26.3_wind/SWIN/wind_swin2sr_config.json. in_chans is not configurable: it's
derived from POSTERIOR_CHANNELS + the terrain encoder's actual output channels.

Usage:
    python new_enscgp_swin.py [--config new_enscgp_swin_config.json]
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from network_swin2sr import Swin2SR
from terrain_encoder_32 import TerrainEncoder

LEAKY_SLOPE = 0.2  # matches the rest of this project's Swin2SR backbone (terrain_encoder.py)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "new_enscgp_swin_config.json"


class ProbabilisticSwin2SR(Swin2SR):
    POSTERIOR_CHANNELS = 5  # [u, v, L11, L21, L22], see enscgp_train.py

    def __init__(self, img_size=200, embed_dim=96,
                 depths=(4, 4, 4), num_heads=(6, 6, 6), window_size=8,
                 mlp_ratio=4., head_init_std=1e-3, chol_diag_init=1.0, **kwargs):
        # Built before super().__init__() so its (fixed) output channel count can
        # feed in_chans; reassigned as a submodule below once nn.Module.__init__
        # (called inside Swin2SR.__init__) has run.
        terrain_encoder = TerrainEncoder()
        terrain_out_channels = terrain_encoder.body[-1].out_channels
        in_chans = self.POSTERIOR_CHANNELS + terrain_out_channels

        super().__init__(
            img_size=img_size, patch_size=1, in_chans=in_chans,
            embed_dim=embed_dim, depths=list(depths), num_heads=list(num_heads),
            window_size=window_size, mlp_ratio=mlp_ratio,
            upscale=1, upsampler='', **kwargs,
        )
        self.terrain_encoder = terrain_encoder
        del self.conv_last  # base class's single-head tail; replaced by the two heads below

        # Final-conv weights use small random noise rather than exact zero: an
        # all-zero weight also zeroes the gradient flowing *back through* that
        # conv (d(loss)/d(input) scales with the weight), which would cut off
        # both heads from the shared backbone (conv_first, RSTB body) at step 0.
        self.mean_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(embed_dim, 2, 3, 1, 1),
        )
        nn.init.normal_(self.mean_head[-1].weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.mean_head[-1].bias)

        self.chol_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(embed_dim, 3, 3, 1, 1),
        )
        nn.init.normal_(self.chol_head[-1].weight, mean=0.0, std=head_init_std)
        with torch.no_grad():
            # softplus(bias) ~= chol_diag_init on the Cholesky diagonal (L11, L22): a
            # small, non-degenerate initial spread. L21 (off-diagonal) starts at 0.
            inv_softplus = math.log(math.exp(chol_diag_init) - 1)
            self.chol_head[-1].bias.copy_(
                torch.tensor([inv_softplus, 0.0, inv_softplus])
            )

    def forward(self, posterior, terrain_raw):
        first_guess_mean = posterior[:, :2]

        terrain_feat = self.terrain_encoder(terrain_raw)
        if terrain_feat.shape[0] != posterior.shape[0]:
            terrain_feat = terrain_feat.expand(posterior.shape[0], -1, -1, -1)
        x = torch.cat([posterior, terrain_feat], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x)
        first_guess_mean = self.check_image_size(first_guess_mean)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_first = self.conv_first(x)
        feats = self.conv_after_body(self.forward_features(x_first)) + x_first

        mean_out = self.mean_head(feats) + first_guess_mean
        chol_raw = self.chol_head(feats)
        chol = torch.cat([
            F.softplus(chol_raw[:, 0:1]),  # L11 > 0
            chol_raw[:, 1:2],              # L21 unconstrained
            F.softplus(chol_raw[:, 2:3]),  # L22 > 0
        ], dim=1)

        out = torch.cat([mean_out, chol], dim=1)
        return out[:, :, :H, :W]


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_model(config: dict) -> ProbabilisticSwin2SR:
    m = config["model"]
    return ProbabilisticSwin2SR(
        img_size=m.get("img_size", 200),
        embed_dim=m.get("embed_dim", 96),
        depths=m.get("depths", [4, 4, 4]),
        num_heads=m.get("num_heads", [6, 6, 6]),
        window_size=m.get("window_size", 8),
        mlp_ratio=m.get("mlp_ratio", 4.0),
        head_init_std=m.get("head_init_std", 1e-3),
        chol_diag_init=m.get("chol_diag_init", 1.0),
    )


def _smoke_test(config: dict):
    torch.manual_seed(config.get("seed", 0))
    model = build_model(config)
    model.eval()

    img_size = config["model"].get("img_size", 200)
    posterior = torch.randn(2, ProbabilisticSwin2SR.POSTERIOR_CHANNELS, img_size, img_size)
    terrain_raw = torch.randn(1, 4, 1000, 1000)

    with torch.no_grad():
        out = model(posterior, terrain_raw)

    assert out.shape == (2, 5, img_size, img_size), f"unexpected output shape {out.shape}"

    mu, l11, l21, l22 = out[:, :2], out[:, 2], out[:, 3], out[:, 4]
    assert torch.all(l11 > 0), "L11 must be strictly positive"
    assert torch.all(l22 > 0), "L22 must be strictly positive"

    first_guess_mean = posterior[:, :2]
    max_abs_diff = (mu - first_guess_mean).abs().max().item()
    assert max_abs_diff < 0.1, (
        f"fresh model's mean_out should be close to first_guess_mean (near-zero-init "
        f"residual head), got max abs diff {max_abs_diff}"
    )

    # Gradient sanity check: with the final-conv weights exactly zero (instead of just
    # near-zero), d(loss)/d(input to that conv) would also be exactly zero, cutting the
    # shared backbone (conv_first, RSTB body, and the terrain encoder) off from any
    # gradient at step 0. Confirm that doesn't happen with the small-random-weight init
    # actually used.
    model.zero_grad()
    out_grad = model(posterior, terrain_raw)
    out_grad.pow(2).mean().backward()
    conv_first_grad = model.conv_first.weight.grad
    assert conv_first_grad is not None, "conv_first.weight.grad is None -- backward() did not reach the backbone"
    backbone_grad = conv_first_grad.abs().max().item()
    assert backbone_grad > 0, "conv_first received zero gradient -- backbone is disconnected from the heads"

    terrain_grad = model.terrain_encoder.stem[0].weight.grad
    assert terrain_grad is not None, "terrain_encoder received no gradient"
    assert terrain_grad.abs().max().item() > 0, "terrain_encoder received zero gradient"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"L11 range: [{l11.min().item():.4f}, {l11.max().item():.4f}]")
    print(f"L22 range: [{l22.min().item():.4f}, {l22.max().item():.4f}]")
    print(f"L21 range: [{l21.min().item():.4f}, {l21.max().item():.4f}]")
    print(f"max|mean_out - first_guess_mean| (fresh model): {max_abs_diff:.2e}")
    print(f"conv_first weight grad max (backbone receives gradient): {backbone_grad:.2e}")
    print(f"terrain_encoder weight grad max (trained jointly): {terrain_grad.abs().max().item():.2e}")
    print(f"Total parameters: {n_params:,}")
    print("Smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    _smoke_test(load_config(args.config))
