"""Flat full-resolution ResNet refiner for wind downscaling (ERA5 -> WRF) -- the
"simple baseline" companion to the Swin2SR-based refiner (new_enscgp_swin.py). Same
residual-refinement problem (Ens-CGP first guess -> WRF ground truth) and the same
in/out tensor convention, but a completely independent model: an EDSR/SRResNet-style
stack of full-resolution residual blocks (no downsampling, no attention, no U-Net),
trained and evaluated as its own model rather than a backbone swap inside the Swin file.

Compatibility with the existing pipeline (new_enscgp_swin.py / train_new_enscgp_swin.py
/ variance_recalibration.py), confirmed by reading those files before writing this one:
- Output format: forward() returns a single (B, 5, H, W) tensor ordered
  [mu_u, mu_v, L11, L21, L22] -- identical to ProbabilisticSwin2SR.forward() -- so
  compute_weighted_loss() (NLL/L1/spectral/gradient/quantile) and the recalibration
  code in variance_recalibration.py (which only index into raw [u,v,L11,L21,L22]
  arrays) work against this model's output unmodified.
- Terrain: terrain_raw is the same raw (1, 4, 1000, 1000) static input consumed by
  TerrainEncoder (terrain_encoder.py), imported and used as a submodule here too
  (trained jointly), not a frozen precomputed feature map.
- Chol head: a residual on the INPUT (first-guess) Cholesky factor, per the
  preference for "keep the first guess's uncertainty by default" (the chol analogue
  of the mean residual). See ResNetRefiner.forward() for why the residual is added
  in pre-softplus ("logit") space rather than literally `softplus(input_L + delta)`.

Inputs (forward(posterior, first_guess_mu, first_guess_chol, terrain_raw=None)):
- posterior: (B, 5, H, W) EnsCGP first-guess posterior [u, v, L11, L21, L22] (see
  enscgp_train.py), fed to the stem -- the network can read its own first guess from
  here. The residual *add* at the heads uses the next two args instead of re-slicing
  posterior, so the skip-connection wiring is explicit at the call site rather than
  implicit inside forward().
- first_guess_mu: (B, 2, H, W) [mu_u, mu_v] -- residual base for mean_head.
- first_guess_chol: (B, 3, H, W) [L11, L21, L22] -- residual base for chol_head.
- terrain_raw: (1, 4, 1000, 1000) static terrain input for TerrainEncoder, broadcast
  across the batch. Required iff use_terrain=True (the default); ignored otherwise.

Architecture (flat, full resolution throughout -- no downsampling/upsampling, no
U-Net, no norm layers, no global pooling/FC):
  stem: 3x3 conv, in_chans -> width
  -> optional terrain injection: concat TerrainEncoder features with stem features,
     1x1 conv back to width, with the terrain half of that conv's weight
     near-zero-initialized (small random noise, not exact zero -- see note in
     __init__) so terrain starts at ~0 influence
  -> num_blocks residual blocks: x -> 3x3 conv -> act -> 3x3 conv -> *residual_scale -> +x
  -> mean_head: 3x3 conv -> 2ch, near-zero init, added to first_guess_mu
  -> chol_head: 3x3 conv -> 3ch, residual on first_guess_chol (see forward())
out_chans is fixed at 5 (2 mean + 3 chol) by this head split, so it's not exposed as
a separate constructor knob the way in_chans is.

Usage:
    python resnet_refiner.py [--config resnet_refiner_config.json]
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from terrain_encoder import TerrainEncoder

LEAKY_SLOPE = 0.2  # matches the rest of this project's conventions (terrain_encoder.py)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "resnet_refiner_config.json"


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True)
    raise ValueError(f"activation must be 'relu' or 'leakyrelu', got {name!r}")


def _inverse_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """log(exp(x) - 1) in the numerically stable form x + log1p(-exp(-x)) (avoids
    ever computing exp(x), so it doesn't overflow for large x). x is clamped above
    0 first: softplus^-1 -> -inf as x -> 0+, which is correct (softplus(-inf) = 0)
    but only finite for x > 0."""
    x = x.clamp_min(eps)
    return x + torch.log1p(-torch.exp(-x))


class ResBlock(nn.Module):
    def __init__(self, width: int, activation: str, residual_scale: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, 1, 1),
            _make_activation(activation),
            nn.Conv2d(width, width, 3, 1, 1),
        )
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.residual_scale


class ResNetRefiner(nn.Module):
    POSTERIOR_CHANNELS = 5  # [u, v, L11, L21, L22], see enscgp_train.py

    def __init__(self, in_chans: int = 5, width: int = 64, num_blocks: int = 8,
                 residual_scale: float = 0.1, activation: str = "relu",
                 use_terrain: bool = True, head_init_std: float = 1e-3):
        super().__init__()
        self.use_terrain = use_terrain

        self.stem = nn.Conv2d(in_chans, width, 3, 1, 1)

        if use_terrain:
            self.terrain_encoder = TerrainEncoder()
            terrain_channels = self.terrain_encoder.body[-1].out_channels
            self.terrain_proj = nn.Conv2d(width + terrain_channels, width, 1)
            with torch.no_grad():
                # Terrain half of the 1x1 kernel starts as small random noise rather
                # than exact zero: an all-zero weight also zeroes the gradient
                # flowing back through this op into terrain_encoder (d(loss)/d(terrain_feat)
                # scales with this weight), cutting it off from training at step 0.
                # The stem half keeps its default conv init.
                nn.init.normal_(self.terrain_proj.weight[:, width:], mean=0.0, std=head_init_std)
                nn.init.zeros_(self.terrain_proj.bias)

        self.body = nn.Sequential(*[
            ResBlock(width, activation, residual_scale) for _ in range(num_blocks)
        ])

        # Final-head weights use small random noise rather than exact zero, for the
        # same gradient-flow reason as the terrain gate above (it would otherwise
        # cut mean_head/chol_head off from the shared body/stem at step 0).
        self.mean_head = nn.Conv2d(width, 2, 3, 1, 1)
        nn.init.normal_(self.mean_head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.mean_head.bias)

        self.chol_head = nn.Conv2d(width, 3, 3, 1, 1)
        nn.init.normal_(self.chol_head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.chol_head.bias)

    def forward(self, posterior: torch.Tensor, first_guess_mu: torch.Tensor,
                first_guess_chol: torch.Tensor, terrain_raw: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(posterior)

        if self.use_terrain:
            if terrain_raw is None:
                raise ValueError("use_terrain=True but terrain_raw was not provided")
            terrain_feat = self.terrain_encoder(terrain_raw)
            if terrain_feat.shape[0] != x.shape[0]:
                terrain_feat = terrain_feat.expand(x.shape[0], -1, -1, -1)
            x = self.terrain_proj(torch.cat([x, terrain_feat], dim=1))

        feats = self.body(x)

        mean_out = self.mean_head(feats) + first_guess_mu

        # Residual on the first-guess chol, added in pre-softplus ("logit") space
        # for the diagonal: softplus(input_L + delta) is NOT close to input_L at
        # delta~=0 unless input_L happens to be large (softplus is nonlinear, e.g.
        # softplus(1.0) = 1.31), so a literal additive-then-softplus residual would
        # not start near the first guess. Going through softplus^-1 first makes the
        # round trip an exact identity at delta=0 for any input_L magnitude. L21 is
        # already unconstrained, so it gets a plain additive residual.
        chol_delta = self.chol_head(feats)
        first_L11, first_L21, first_L22 = (
            first_guess_chol[:, 0:1], first_guess_chol[:, 1:2], first_guess_chol[:, 2:3]
        )
        L11 = F.softplus(_inverse_softplus(first_L11) + chol_delta[:, 0:1])
        L21 = first_L21 + chol_delta[:, 1:2]
        L22 = F.softplus(_inverse_softplus(first_L22) + chol_delta[:, 2:3])

        return torch.cat([mean_out, L11, L21, L22], dim=1)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_model(config: dict) -> ResNetRefiner:
    m = config["model"]
    return ResNetRefiner(
        in_chans=m.get("in_chans", 5),
        width=m.get("width", 64),
        num_blocks=m.get("num_blocks", 8),
        residual_scale=m.get("residual_scale", 0.1),
        activation=m.get("activation", "relu"),
        use_terrain=m.get("use_terrain", True),
        head_init_std=m.get("head_init_std", 1e-3),
    )


def _smoke_test(config: dict):
    torch.manual_seed(config.get("seed", 0))
    model = build_model(config)
    model.eval()

    # img_size is only used here to shape dummy tensors -- ResNetRefiner itself is
    # resolution-agnostic (no img_size constructor arg, unlike the Swin backbone).
    img_size = config["model"].get("img_size", 200)
    B, H, W = 2, img_size, img_size
    posterior = torch.randn(B, ResNetRefiner.POSTERIOR_CHANNELS, H, W)
    posterior[:, 2] = F.softplus(posterior[:, 2])  # L11 > 0, like a real EnsCGP posterior
    posterior[:, 4] = F.softplus(posterior[:, 4])  # L22 > 0
    first_guess_mu = posterior[:, :2].clone()
    first_guess_chol = posterior[:, 2:5].clone()
    terrain_raw = torch.randn(1, 4, 1000, 1000)

    with torch.no_grad():
        out = model(posterior, first_guess_mu, first_guess_chol, terrain_raw)

    assert out.shape == (B, 5, H, W), f"unexpected output shape {out.shape}"

    mu, L11, L21, L22 = out[:, :2], out[:, 2], out[:, 3], out[:, 4]
    assert torch.all(L11 > 0), "L11 must be strictly positive"
    assert torch.all(L22 > 0), "L22 must be strictly positive"

    mu_diff = (mu - first_guess_mu).abs().max().item()
    assert mu_diff < 0.1, f"fresh model's mean output should be close to first_guess_mu, got max abs diff {mu_diff}"

    chol_diff = (out[:, 2:5] - first_guess_chol).abs().max().item()
    assert chol_diff < 0.1, f"fresh model's chol output should be close to first_guess_chol, got max abs diff {chol_diff}"

    # Gradient sanity check (mirrors new_enscgp_swin.py): confirm the near-zero head/
    # gate weights don't cut off gradient to the shared body/stem/terrain encoder.
    model.zero_grad()
    out_grad = model(posterior, first_guess_mu, first_guess_chol, terrain_raw)
    out_grad.pow(2).mean().backward()
    stem_grad = model.stem.weight.grad.abs().max().item()
    assert stem_grad > 0, "stem received zero gradient -- backbone is disconnected from the heads"
    terrain_grad = None
    if model.use_terrain:
        terrain_grad = model.terrain_encoder.stem[0].weight.grad.abs().max().item()
        assert terrain_grad > 0, "terrain_encoder received zero gradient"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"L11 range: [{L11.min().item():.4f}, {L11.max().item():.4f}]")
    print(f"L22 range: [{L22.min().item():.4f}, {L22.max().item():.4f}]")
    print(f"max|mean_out - first_guess_mu| (fresh model): {mu_diff:.2e}")
    print(f"max|chol_out - first_guess_chol| (fresh model): {chol_diff:.2e}")
    print(f"stem weight grad max (backbone receives gradient): {stem_grad:.2e}")
    if terrain_grad is not None:
        print(f"terrain_encoder weight grad max (trained jointly): {terrain_grad:.2e}")
    print(f"Total parameters: {n_params:,}")

    # Batch-size-4 forward+backward smoke test (matches the project's default batch_size).
    model.train()
    posterior4 = torch.randn(4, ResNetRefiner.POSTERIOR_CHANNELS, H, W)
    posterior4[:, 2] = F.softplus(posterior4[:, 2])
    posterior4[:, 4] = F.softplus(posterior4[:, 4])
    out4 = model(posterior4, posterior4[:, :2], posterior4[:, 2:5], terrain_raw)
    out4.pow(2).mean().backward()
    print("Batch-size-4 forward+backward: OK")
    print("Smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    _smoke_test(load_config(args.config))
