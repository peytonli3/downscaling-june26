"""Two-head probabilistic Swin2SR refiner for wind downscaling (ERA5 -> WRF),
QUANTILE version (q10/q50/q90 per component -- replaces the Gaussian/Cholesky head).

Wraps the Swin2SR backbone (network_swin2sr.py) at upscale=1 -- same-resolution
restoration, not super-resolution -- and replaces its single-conv reconstruction
tail with two parallel heads on top of the embed_dim deep-feature map (shallow
features + RSTB body + conv_after_body + long skip, all unchanged from Swin2SR).

Wind is heavy-tailed / right-skewed, so a per-pixel Gaussian is a poor fit. Instead
of a mean + 2x2 Cholesky covariance trained by NLL, each head now predicts DIRECT,
distribution-free quantiles per component (u, v): q10, q50, q90.

Inputs (forward(posterior, bicubic, terrain_raw)):
- posterior: (B, 5, H, W) EnsCGP first-guess posterior [u, v, L11, L21, L22]
  (e.g. data/enscgp_posterior.npy, see enscgp_train.py). Fed in whole as a model
  input; there is no Cholesky OUTPUT and no NLL any more.
- bicubic: (B, 2, H, W) bicubic-upsampled ERA5 wind [u, v]
  (data/era5_uv_2ch_bicubic.npy) -- the naive low-resolution baseline. Always fed
  in as an input channel regardless of residual_base.
- terrain_raw: (1, 4, 1000, 1000) static terrain input for TerrainEncoder (see
  terrain_encoder.py), shared by every sample -- broadcast across the batch.
  TerrainEncoder is a submodule here (trained jointly with the rest of the network).

The per-pixel model input is assembled inside forward() as
  model_input = cat([bicubic, posterior], dim=1)
(7 channels: [bic_u, bic_v, enscgp_u, enscgp_v, L11, L21, L22] -- bicubic then the
whole EnsCGP posterior, in its own stored order). Earlier versions fed
[bicubic, enscgp-minus-bicubic] instead of [bicubic, enscgp] directly; since
conv_first is linear, that residual decomposition is an invertible linear
recombination of the same information and changes no function the network can
represent -- dropped as unneeded complexity (v7). This is independent of
residual_base (which only controls what the q50 head's output is added to).
x = cat([model_input, terrain_encoder(terrain_raw)], dim=1) feeds conv_first.

Heads -- two different residual styles, chosen per head's role:
- mean_head (the q50 / central head): 2 channels (q50_u, q50_v). Output =
  mean_base + mean_gate * head(feats), where mean_base is selected by
  residual_base ("enscgp" (default) -> EnsCGP posterior mean, "bicubic" ->
  bicubic baseline, "none" -> zero). RESTORED (post-0729 ablation, see
  CHANGELOG "0729_meangate"): the head's final conv is back to full-strength Kaiming-normal
  init, gated by a learnable scalar mean_gate (init 0.1) -- the v0629-v0714
  design. v7/0729 had switched this to a near-zero-init head with no gate,
  reasoning that mean_gate settling at ~0.125 after a full 0714 run (barely
  above its 0.1 init) meant the gate wasn't doing much; that run's results were
  measurably worse than 0714's across every metric, so this ablation restores
  ONLY mean_gate (not offset_head's EnsCGP-seed or the input decomposition,
  both still removed) to isolate whether this specific change was responsible.
  q50 keeps EXACTLY the role the old mean had: it is the field the structural
  losses (multi-scale Wasserstein + frequency-L1) train, so it stays sharp. It is
  NOT pinball-trained (that would pull it toward the blurry pointwise median and
  fight the displacement-tolerant structural losses).
- offset_head: 4 channels (raw_up_u, raw_up_v, raw_down_u, raw_down_v). Two
  NON-NEGATIVE offsets per component are formed by softplus (never relu -- relu has
  a dead zone where the spread can collapse to exactly zero with zero gradient;
  softplus stays positive and always trainable), so the quantiles are MONOTONIC BY
  CONSTRUCTION (q10 <= q50 <= q90, no crossing, no crossing penalty needed):
      up_offset   = softplus(raw_up)   + OFFSET_EPS
      down_offset = softplus(raw_down) + OFFSET_EPS
      q90 = q50 + up_offset
      q10 = q50 - down_offset
  The head's final conv uses standard Kaiming-normal init (full strength, so
  gradient flows normally from step 0) with its BIAS initialized to
  inverse_softplus(offset_spread_init) (default 3.0 m/s, the measured RMS of
  WRF-minus-bicubic over a training sample -- a data-grounded "typical correction
  magnitude", not a physically-precise uncertainty). So a fresh model starts with
  up/down offsets near offset_spread_init on every pixel, then softplus(raw)
  refines this per-pixel and per-side (up/down independently, since they are
  separate channels) via ordinary gradient descent. v0629-v0714 instead seeded
  this from the EnsCGP posterior's own per-pixel Cholesky (sigma_u=L11,
  sigma_v=sqrt(L21^2+L22^2)), gated in inverse-softplus space around that base;
  dropped (v7) as an unnecessary extra mechanism once the offset head is trained
  by pinball loss regardless of its starting point -- a plain bias-initialized
  softplus head is simpler and empirically not worse (see CHANGELOG "0729").

forward() returns a single (B, 6, H, W) tensor ordered [q10_u, q10_v, q50_u, q50_v,
q90_u, q90_v] (ascending quantile blocks; see Q10_SLICE / Q50_SLICE / Q90_SLICE).

Loss lives in train_new_enscgp_swin.py: structural losses on q50 (unchanged) +
pinball(q90, .9) + pinball(q10, .1), with optional extreme (high-wind) up-weighting.

Architecture / regularization hyperparameters live in a JSON config (see
new_enscgp_swin_config.json). in_chans is not configurable: it's INPUT_CHANNELS + the
terrain encoder's actual output channels.

NOTE: this is a checkpoint-incompatible change from the 0714 quantile version (see
CHANGELOG "0729" and the mean_gate-restoration entry "0729_meangate" above it):
offset_gate is gone and the offset head no longer reads the EnsCGP Cholesky as a
seed; mean_gate is back (restored post-0729). mean_head/offset_head/backbone/
terrain_encoder weights still transfer from a 0714 checkpoint by shape via
--resume-weights-only -- EXCEPT conv_first, whose weight shape also happens to be
unchanged (still 11 input channels) but whose channels 2-3 now carry raw EnsCGP u/v
instead of an EnsCGP-minus-bicubic residual: a naive partial load will transfer
conv_first's weights despite this semantic change, likely to a worse starting point
than training conv_first fresh. Prefer starting fresh from EnsCGP unless you
explicitly want to test resuming through that channel-semantics change.

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
from terrain_encoder import TerrainEncoder

LEAKY_SLOPE = 0.2  # matches the rest of this project's Swin2SR backbone (terrain_encoder.py)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "new_enscgp_swin_config.json"


class ProbabilisticSwin2SR(Swin2SR):
    POSTERIOR_CHANNELS = 5  # EnsCGP first guess [u, v, L11, L21, L22], see enscgp_train.py
    # Per-pixel model input assembled in forward(): [bic_u, bic_v, enscgp_u, enscgp_v, L11,
    # L21, L22] -- bicubic then the whole EnsCGP posterior tensor, concatenated directly.
    INPUT_CHANNELS = 7
    OUT_CHANNELS = 6  # [q10_u, q10_v, q50_u, q50_v, q90_u, q90_v]
    Q10_SLICE = slice(0, 2)
    Q50_SLICE = slice(2, 4)
    Q90_SLICE = slice(4, 6)
    RESIDUAL_BASES = ("enscgp", "bicubic", "none")
    OFFSET_EPS = 1e-3          # hard floor on each quantile offset -- spread can't underflow to 0

    def __init__(self, img_size=200, embed_dim=96,
                 depths=(4, 4, 4), num_heads=(6, 6, 6), window_size=8,
                 mlp_ratio=4., residual_base="enscgp", residual_gate_init=0.1,
                 offset_spread_init=3.0, head_dropout=0.0, **kwargs):
        # Backbone dropout knobs (drop_rate / attn_drop_rate / drop_path_rate) flow
        # through **kwargs to Swin2SR; head_dropout is consumed here. All default to
        # current behavior -- see build_model.
        if residual_base not in self.RESIDUAL_BASES:
            raise ValueError(f"residual_base must be one of {self.RESIDUAL_BASES}, got {residual_base!r}")
        # Built before super().__init__() so its (fixed) output channel count can feed
        # in_chans; reassigned as a submodule below once nn.Module.__init__ has run.
        terrain_encoder = TerrainEncoder()
        terrain_out_channels = terrain_encoder.body[-1].out_channels
        in_chans = self.INPUT_CHANNELS + terrain_out_channels

        super().__init__(
            img_size=img_size, patch_size=1, in_chans=in_chans,
            embed_dim=embed_dim, depths=list(depths), num_heads=list(num_heads),
            window_size=window_size, mlp_ratio=mlp_ratio,
            upscale=1, upsampler='', **kwargs,
        )
        self.terrain_encoder = terrain_encoder
        del self.conv_last  # base class's single-head tail; replaced by the two heads below

        self.residual_base = residual_base
        self.offset_spread_init = float(offset_spread_init)
        self.head_dropout = head_dropout
        # mean_head: full-strength (Kaiming) final-conv weight, gated by mean_gate (RESTORED,
        # see module docstring) -- output = mean_base + mean_gate*head(feats). offset_head:
        # full-strength (Kaiming) final-conv weight with a nonzero BIAS, so softplus(raw)
        # starts near offset_spread_init on every pixel, no gate. See module docstring.
        self.mean_head = self._make_head(embed_dim, 2)
        self.mean_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        offset_bias = self._inverse_softplus(self.offset_spread_init)
        self.offset_head = self._make_head(embed_dim, 4, bias_init=offset_bias)

    @staticmethod
    def _inverse_softplus(x: float) -> float:
        """Inverse of softplus (beta=1) for a plain float, used once at init time to turn
        a target spread (m/s) into the final-conv bias that produces it before any
        training: softplus(inverse_softplus(x)) == x."""
        return math.log(math.expm1(x))

    @staticmethod
    def _make_head(embed_dim: int, out_channels: int, bias_init: float = 0.0) -> nn.Sequential:
        # conv -> LeakyReLU -> conv. head_dropout (when > 0) is applied functionally in
        # _head_forward between activation and final conv, NOT as a module here, so the
        # state_dict layout is identical for every head_dropout value. Both heads use
        # full-strength Kaiming init -- mean_head relies on mean_gate (not a suppressed
        # weight) to start gentle; offset_head relies on bias_init instead of a gate.
        head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(embed_dim, out_channels, 3, 1, 1),
        )
        nn.init.kaiming_normal_(head[-1].weight, a=LEAKY_SLOPE, nonlinearity="leaky_relu")
        nn.init.constant_(head[-1].bias, bias_init)
        return head

    def _head_forward(self, head: nn.Sequential, feats: torch.Tensor) -> torch.Tensor:
        x = head[:-1](feats)  # conv + activation
        if self.head_dropout > 0:
            x = F.dropout2d(x, p=self.head_dropout, training=self.training)
        return head[-1](x)  # final conv

    def _quantile_offset(self, raw: torch.Tensor) -> torch.Tensor:
        """Non-negative, monotonic quantile offset: softplus(raw) + OFFSET_EPS, so the
        offset can never underflow to exactly zero. No external seed distribution and no
        gate -- the final conv's bias (see __init__) already starts raw near
        inverse_softplus(offset_spread_init), and ordinary Kaiming-scale gradients refine
        it per-pixel and per-side from there."""
        return F.softplus(raw) + self.OFFSET_EPS

    def forward(self, posterior, bicubic, terrain_raw):
        # Model input: bicubic baseline + the whole EnsCGP posterior, concatenated
        # directly (7ch: bic_u, bic_v, enscgp_u, enscgp_v, L11, L21, L22). Independent of
        # residual_base (which only affects the q50 OUTPUT base below).
        model_input = torch.cat([bicubic, posterior], dim=1)

        terrain_feat = self.terrain_encoder(terrain_raw)
        if terrain_feat.shape[0] != model_input.shape[0]:
            terrain_feat = terrain_feat.expand(model_input.shape[0], -1, -1, -1)
        x = torch.cat([model_input, terrain_feat], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # q50 base per residual_base (same selection the old mean used).
        if self.residual_base == "enscgp":
            mean_base = posterior[:, :2]
        elif self.residual_base == "bicubic":
            mean_base = bicubic
        else:  # "none"
            mean_base = torch.zeros_like(bicubic)
        mean_base = self.check_image_size(mean_base)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_first = self.conv_first(x)
        feats = self.conv_after_body(self.forward_features(x_first)) + x_first

        q50 = mean_base + self.mean_gate * self._head_forward(self.mean_head, feats)

        raw = self._head_forward(self.offset_head, feats)  # [up_u, up_v, down_u, down_v]
        up = self._quantile_offset(raw[:, 0:2])
        down = self._quantile_offset(raw[:, 2:4])

        q90 = q50 + up
        q10 = q50 - down
        out = torch.cat([q10, q50, q90], dim=1)  # (B, 6, Hp, Wp)
        return out[:, :, :H, :W]


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_model(config: dict) -> ProbabilisticSwin2SR:
    m = config["model"]
    # Dropout knobs default to current behavior, so omitting them is a no-op:
    # drop_rate/attn_drop_rate 0.0 and head_dropout 0.0 (off), drop_path_rate 0.1.
    return ProbabilisticSwin2SR(
        img_size=m.get("img_size", 200),
        embed_dim=m.get("embed_dim", 96),
        depths=m.get("depths", [4, 4, 4]),
        num_heads=m.get("num_heads", [6, 6, 6]),
        window_size=m.get("window_size", 8),
        mlp_ratio=m.get("mlp_ratio", 4.0),
        residual_base=m.get("residual_base", "enscgp"),
        residual_gate_init=m.get("residual_gate_init", 0.1),
        offset_spread_init=m.get("offset_spread_init", 3.0),
        head_dropout=m.get("head_dropout", 0.0),
        drop_rate=m.get("drop_rate", 0.0),
        attn_drop_rate=m.get("attn_drop_rate", 0.0),
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )


def _pinball_quantile_recovery_test():
    """Unit test: the pinball(tau) minimizer over a fixed set of truths is the empirical
    tau-quantile. Optimize a single scalar prediction against samples from a known
    distribution and confirm it converges to the true 0.9 (and 0.1) quantile -- i.e. the
    pinball loss is doing what we claim, independent of the network."""
    torch.manual_seed(0)
    y = torch.randn(200000)  # standard normal; known quantiles
    for tau, truth in ((0.9, 1.2816), (0.1, -1.2816)):
        q = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([q], lr=0.05)
        for _ in range(2000):
            opt.zero_grad()
            err = y - q
            loss = torch.maximum(tau * err, (tau - 1.0) * err).mean()
            loss.backward()
            opt.step()
        emp = torch.quantile(y, tau).item()
        got = q.item()
        assert abs(got - emp) < 0.03, f"tau={tau}: pinball minimizer {got:.4f} != empirical quantile {emp:.4f}"
        assert abs(got - truth) < 0.05, f"tau={tau}: pinball minimizer {got:.4f} != true quantile {truth:.4f}"
        print(f"  pinball tau={tau}: recovered {got:.4f} (empirical {emp:.4f}, true {truth:.4f})")
    print("Pinball quantile-recovery test passed.")


def _smoke_test(config: dict):
    torch.manual_seed(config.get("seed", 0))
    model = build_model(config)
    model.eval()

    img_size = config["model"].get("img_size", 200)
    posterior = torch.randn(2, ProbabilisticSwin2SR.POSTERIOR_CHANNELS, img_size, img_size)
    # Real EnsCGP Cholesky diagonal (L11, L22) is > 0 by construction; make the fabricated
    # test data physically valid too (softplus maps randn -> a realistic positive spread).
    posterior[:, 2] = F.softplus(posterior[:, 2])
    posterior[:, 4] = F.softplus(posterior[:, 4])
    bicubic = torch.randn(2, 2, img_size, img_size)
    terrain_raw = torch.randn(1, 4, 1000, 1000)

    with torch.no_grad():
        out = model(posterior, bicubic, terrain_raw)

    assert out.shape == (2, 6, img_size, img_size), f"unexpected output shape {out.shape}"
    q10 = out[:, ProbabilisticSwin2SR.Q10_SLICE]
    q50 = out[:, ProbabilisticSwin2SR.Q50_SLICE]
    q90 = out[:, ProbabilisticSwin2SR.Q90_SLICE]

    # Monotonicity by construction, everywhere, strictly (offsets floored by OFFSET_EPS).
    assert torch.all(q10 <= q50), "q10 <= q50 violated"
    assert torch.all(q50 <= q90), "q50 <= q90 violated"
    up = (q90 - q50)
    down = (q50 - q10)
    assert torch.all(up >= model.OFFSET_EPS - 1e-6), "up offset underflowed OFFSET_EPS"
    assert torch.all(down >= model.OFFSET_EPS - 1e-6), "down offset underflowed OFFSET_EPS"

    # Fresh model: q50 tracks its residual_base within a gate-scaled margin (mean_head is
    # full-strength Kaiming; mean_gate, not a suppressed weight, keeps a fresh model's
    # contribution small).
    base_map = {"enscgp": posterior[:, :2], "bicubic": bicubic, "none": torch.zeros_like(bicubic)}
    mean_base = base_map[model.residual_base]
    gate = model.mean_gate.item()
    mean_diff = (q50 - mean_base).abs().max().item()
    mean_bound = max(0.5, 5 * gate)
    assert mean_diff < mean_bound, (
        f"fresh q50 should track its {model.residual_base!r} base within a gate-scaled "
        f"margin (gate={gate:.3f}, bound={mean_bound:.3f}), got max abs diff {mean_diff:.4f}"
    )

    # Fresh model: offset head's bias is initialized so up/down start near
    # offset_spread_init on every pixel; Kaiming-scale weight noise adds some per-pixel
    # spread around that, so allow a looser (but still bounded) margin.
    target = model.offset_spread_init
    band_bound = 3.0
    up_diff = (up - target).abs().max().item()
    down_diff = (down - target).abs().max().item()
    assert up_diff < band_bound and down_diff < band_bound, (
        f"fresh offsets should be ~= offset_spread_init={target:.3f} (bound={band_bound}); "
        f"got up_diff={up_diff:.4f}, down_diff={down_diff:.4f}"
    )

    # Gradient sanity: both heads are full-strength (Kaiming), so gradient flows to the
    # backbone and terrain encoder strongly from step 0, and mean_gate itself must receive
    # gradient (it's what training has to grow).
    model.zero_grad()
    out_grad = model(posterior, bicubic, terrain_raw)
    out_grad.pow(2).mean().backward()
    conv_first_grad = model.conv_first.weight.grad
    assert conv_first_grad is not None and conv_first_grad.abs().max().item() > 0, \
        "conv_first received no gradient -- backbone disconnected from the heads"
    terrain_grad = model.terrain_encoder.stem[0].weight.grad
    assert terrain_grad is not None and terrain_grad.abs().max().item() > 0, "terrain_encoder received no gradient"
    mean_head_grad = model.mean_head[-1].weight.grad
    assert mean_head_grad is not None and mean_head_grad.abs().max().item() > 0, "mean_head received no gradient"
    offset_head_grad = model.offset_head[-1].weight.grad
    assert offset_head_grad is not None and offset_head_grad.abs().max().item() > 0, "offset_head received no gradient"
    assert model.mean_gate.grad is not None and model.mean_gate.grad.abs().item() > 0, "mean_gate received no gradient"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}  (ordered [q10_u,q10_v, q50_u,q50_v, q90_u,q90_v])")
    print(f"residual_base={model.residual_base!r}, mean_gate={gate:.4f}, offset_spread_init={model.offset_spread_init}")
    print(f"up  offset range: [{up.min().item():.4f}, {up.max().item():.4f}]")
    print(f"down offset range: [{down.min().item():.4f}, {down.max().item():.4f}]")
    print(f"max|q50 - {model.residual_base} base| (fresh): {mean_diff:.4f} (bound {mean_bound})")
    print(f"max|band - offset_spread_init| (fresh): up {up_diff:.4f}, down {down_diff:.4f} (bound {band_bound})")
    print(f"conv_first weight grad max: {conv_first_grad.abs().max().item():.2e}")
    print(f"mean_head final-conv weight grad max: {mean_head_grad.abs().max().item():.2e}")
    print(f"offset_head final-conv weight grad max: {offset_head_grad.abs().max().item():.2e}")
    print(f"Total parameters: {n_params:,}")
    print("Smoke test passed.")
    _pinball_quantile_recovery_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    _smoke_test(load_config(args.config))
