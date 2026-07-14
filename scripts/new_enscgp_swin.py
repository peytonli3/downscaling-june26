"""Two-head probabilistic Swin2SR refiner for wind downscaling (ERA5 -> WRF).

Wraps the Swin2SR backbone (network_swin2sr.py) at upscale=1 -- same-resolution
restoration, not super-resolution -- and replaces its single-conv reconstruction
tail with two parallel heads on top of the embed_dim deep-feature map (shallow
features + RSTB body + conv_after_body + long skip, all unchanged from Swin2SR).

Inputs (forward(posterior, bicubic, terrain_raw, prior_spread=None)):
- posterior: (B, 5, H, W) EnsCGP first-guess posterior [u, v, L11, L21, L22]
  (e.g. data/enscgp_posterior.npy, see enscgp_train.py).
- bicubic: (B, 2, H, W) bicubic-upsampled ERA5 wind [u, v]
  (data/era5_uv_2ch_bicubic.npy) -- the naive low-resolution baseline. Always fed
  in as an input channel (see model_input below) regardless of residual_base.
- terrain_raw: (1, 4, 1000, 1000) static terrain input for TerrainEncoder (see
  terrain_encoder.py), shared by every sample -- broadcast across the batch.
  TerrainEncoder is a submodule here (not a frozen precomputed feature map),
  so its weights are trained jointly with the rest of the network.
- prior_spread: (B, 2, H, W) per-pixel EnsCGP PRIOR (pre-conditioning) ensemble
  spread for [u, v] -- the analog-disagreement signal (std across each sample's k
  WRF analogs, see scripts/precompute_prior_spread.py). Only consumed when
  variance_conditioning is on (else ignored / may be None); see "Variance
  conditioning" below.

Variance conditioning (config-gated, default OFF -> exact current behavior):
when model.variance_conditioning is on, the chol_head (variance head ONLY) is
additionally conditioned on features that predict WHERE placement is uncertain, so
it can inflate sigma on the coherent "wrong-bet" blotches the mean produces. Four
extra channels are fed into the chol_head's first conv via a SEPARATE conv
(chol_cond), zero-initialized so at init it contributes nothing and the model
output is identical to the unconditioned checkpoint (then chol_cond learns during
fine-tuning). The four conditioning channels (standardized by cond_mean/cond_std
buffers, see load_cond_stats) are:
  [0,1] prior_spread u, v        -- analog disagreement (the key signal)
  [2]   |grad(speed(mean))|      -- sharp mean-wind gradients = displaceable structure
                                    (computed from the DETACHED mean, so no grad flows
                                    back into the mean head/backbone)
  [3]   |terrain slope|          -- sqrt(u_slope^2+v_slope^2) from terrain_raw[:,2:4],
                                    avg-pooled 5x to the WRF grid (static across samples)
The mean head, backbone, and loss are untouched by conditioning -- only chol_cond
(new) plus the existing chol_head/chol_gate are involved.

The per-pixel model input is assembled inside forward() as
  model_input = cat([bicubic, enscgp_mean - bicubic, L11, L21, L22], dim=1)
(7 channels: [bic_u, bic_v, res_u, res_v, L11, L21, L22]) -- i.e. the EnsCGP u/v
channels are replaced by the bicubic field and the EnsCGP-mean-minus-bicubic
residual (their sum still recovers the EnsCGP mean, so no information is lost).
This input decomposition is independent of residual_base below (which only
controls what each head's output is added to).
x = cat([model_input, terrain_encoder(terrain_raw)], dim=1) feeds conv_first.

Heads -- gated residual onto a base, not near-zero-init pass-through:
- mean_head: 2 channels (mu_u, mu_v). Output = mean_base + mean_gate * head(feats),
  where mean_base is selected by residual_base ("enscgp" (default) -> EnsCGP
  posterior mean, "bicubic" -> the bicubic baseline, "none" -> zero, i.e. predict
  the absolute mean directly). mean_gate is a learnable scalar (nn.Parameter,
  init residual_gate_init, default 0.1).
- chol_head: 3 channels (L11, L21, L22), the lower-triangular Cholesky factor of
  the per-pixel 2x2 (u, v) covariance, Sigma = L @ L.T. Output is gated onto
  EnsCGP's OWN per-pixel Cholesky (posterior[:, 2:5]) -- not configurable via
  residual_base, since EnsCGP always provides a natural starting covariance and
  "bicubic"/"none" have no covariance analog. Positivity is preserved exactly:
  the diagonal perturbation chol_gate * head(feats) is added in inverse-softplus
  (pre-activation) space around the base, then mapped back through softplus, so
  L11/L22 stay > 0 by construction however large the gated perturbation gets;
  L21 (unconstrained) is a direct gated addition.
Both heads' final conv now uses standard Kaiming-normal init (not near-zero) --
the *gate*, not a suppressed weight, is what keeps a fresh model's contribution
small ("start gentle, free to grow": small initially, but the head itself is
full-strength from step 0, so gradient flows normally and the gate alone can grow
during training).

forward() returns a single (B, 5, H, W) tensor ordered [mu_u, mu_v, L11, L21,
L22] -- matching the EnsCGP output convention.

Architecture and regularization hyperparameters live in a JSON config (see
new_enscgp_swin_config.json), following the same "model" section convention as
26.3_wind/SWIN/wind_swin2sr_config.json. Regularization knobs (all optional,
defaulting to current behavior): drop_rate / attn_drop_rate / drop_path_rate on the
Swin backbone, and head_dropout (channel dropout in the two conv heads). in_chans is
not configurable: it's derived from INPUT_CHANNELS + the terrain encoder's actual
output channels.

NOTE: this is an architecture change from the previous near-zero-init/bicubic-base
design (different state_dict shapes: mean_gate/chol_gate are new parameters) --
checkpoints trained before this change will not load with strict=True.

Usage:
    python new_enscgp_swin.py [--config new_enscgp_swin_config.json]
"""
import argparse
import json
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
    # Per-pixel model input assembled in forward(): [bic_u, bic_v, res_u, res_v, L11, L21,
    # L22], i.e. the bicubic baseline, the EnsCGP-mean-minus-bicubic residual, and the
    # EnsCGP Cholesky channels.
    INPUT_CHANNELS = 7
    RESIDUAL_BASES = ("enscgp", "bicubic", "none")
    CHOL_EPS = 1e-4  # floor on the EnsCGP base Cholesky diagonal before inverse-softplus
    COND_CHANNELS = 4  # variance-conditioning features: [prior_spread_u, prior_spread_v, |grad speed(mean)|, |terrain slope|]
    GRAD_EPS = 1e-6    # floor inside the speed / gradient-magnitude square roots

    def __init__(self, img_size=200, embed_dim=96,
                 depths=(4, 4, 4), num_heads=(6, 6, 6), window_size=8,
                 mlp_ratio=4., residual_base="enscgp", residual_gate_init=0.1,
                 head_dropout=0.0, variance_conditioning=False, **kwargs):
        # Backbone dropout knobs (drop_rate / attn_drop_rate / drop_path_rate) flow
        # through **kwargs to Swin2SR; head_dropout is consumed here (channel dropout in
        # the two conv heads). All default to current behavior -- see build_model.
        if residual_base not in self.RESIDUAL_BASES:
            raise ValueError(f"residual_base must be one of {self.RESIDUAL_BASES}, got {residual_base!r}")
        # Built before super().__init__() so its (fixed) output channel count can
        # feed in_chans; reassigned as a submodule below once nn.Module.__init__
        # (called inside Swin2SR.__init__) has run.
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
        self.head_dropout = head_dropout
        self.mean_head = self._make_head(embed_dim, 2)
        self.chol_head = self._make_head(embed_dim, 3)
        # Learnable scalars, NOT module-level dropout-style toggles: each head's output is
        # multiplied by its gate before being added to its base (see forward()). Starting
        # small (residual_gate_init) keeps the fresh model close to the base ("start
        # gentle") while the head itself is full-strength (Kaiming init, see _make_head)
        # so gradients aren't suppressed -- only the gate scalar has to grow during
        # training, not the whole head's weights from near-zero.
        self.mean_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        self.chol_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

        # Variance conditioning (config-gated, default OFF): a SEPARATE conv mapping the
        # COND_CHANNELS conditioning features -> embed_dim, added to chol_head[0]'s output
        # (see forward()). Zero-initialized (weight AND bias), so at init it is an exact
        # no-op -- the conditioned model's output equals the unconditioned checkpoint's,
        # letting us continue-from-checkpoint and let chol_cond learn during fine-tuning.
        # cond_mean/cond_std standardize the conditioning channels (default no-op; set via
        # load_cond_stats from precompute_prior_spread.py's training-set statistics).
        self.variance_conditioning = bool(variance_conditioning)
        if self.variance_conditioning:
            self.chol_cond = nn.Conv2d(self.COND_CHANNELS, embed_dim, 3, 1, 1)
            nn.init.zeros_(self.chol_cond.weight)
            nn.init.zeros_(self.chol_cond.bias)
            self.register_buffer("cond_mean", torch.zeros(1, self.COND_CHANNELS, 1, 1))
            self.register_buffer("cond_std", torch.ones(1, self.COND_CHANNELS, 1, 1))

    @staticmethod
    def _make_head(embed_dim: int, out_channels: int) -> nn.Sequential:
        # conv -> LeakyReLU -> conv. head_dropout (when > 0) is applied functionally in
        # _head_forward between the activation and the final conv, NOT as a module here,
        # so the state_dict layout is identical for every head_dropout value -- a
        # checkpoint trained at one setting loads at any other (and an un-dropped run can
        # be resumed with dropout turned on).
        head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE, inplace=True),
            nn.Conv2d(embed_dim, out_channels, 3, 1, 1),
        )
        # Standard Kaiming-normal init on the final conv (NOT near-zero): the gate (see
        # __init__) is what keeps a fresh model's contribution small, not a suppressed
        # weight -- so this head is full-strength and gradients flow normally from step 0.
        nn.init.kaiming_normal_(head[-1].weight, a=LEAKY_SLOPE, nonlinearity="leaky_relu")
        nn.init.zeros_(head[-1].bias)
        return head

    def _head_forward(self, head: nn.Sequential, feats: torch.Tensor) -> torch.Tensor:
        x = head[:-1](feats)  # conv + activation
        if self.head_dropout > 0:
            x = F.dropout2d(x, p=self.head_dropout, training=self.training)
        return head[-1](x)  # final conv

    @classmethod
    def _inverse_softplus(cls, x: torch.Tensor) -> torch.Tensor:
        """Exact inverse of F.softplus (beta=1): softplus(inverse_softplus(x)) == x.
        x is clamped to CHOL_EPS first so a (near-)zero base Cholesky entry doesn't blow
        up log(0); log1p(-exp(-x)) is the numerically stable form of log(1 - exp(-x))."""
        x = x.clamp_min(cls.CHOL_EPS)
        return x + torch.log1p(-torch.exp(-x))

    def load_cond_stats(self, cond_mean, cond_std):
        """Set the standardization buffers for the variance-conditioning features.
        cond_mean/cond_std are length-COND_CHANNELS sequences (per-channel training-set
        mean/std from precompute_prior_spread.py); reshaped to (1, C, 1, 1) for
        broadcasting over (B, C, H, W). No-op unless variance_conditioning is on."""
        if not self.variance_conditioning:
            raise RuntimeError("load_cond_stats called but variance_conditioning is off")
        mean = torch.as_tensor(cond_mean, dtype=self.cond_mean.dtype).reshape(1, self.COND_CHANNELS, 1, 1)
        std = torch.as_tensor(cond_std, dtype=self.cond_std.dtype).reshape(1, self.COND_CHANNELS, 1, 1)
        self.cond_mean.copy_(mean)
        self.cond_std.copy_(std.clamp_min(self.GRAD_EPS))

    def _speed_grad_mag(self, mean_out: torch.Tensor) -> torch.Tensor:
        """|grad(speed)| of the predicted-mean wind speed, (B,1,H,W). Central differences
        with replicate padding (same scheme as MeanAuxLosses.gradient). Caller passes the
        DETACHED mean so this conditioning feature contributes no gradient to the mean
        head/backbone."""
        sp = torch.sqrt(mean_out[:, 0:1] ** 2 + mean_out[:, 1:2] ** 2 + self.GRAD_EPS)
        sp_x = F.pad(sp, (1, 1, 0, 0), mode="replicate")
        gx = 0.5 * (sp_x[:, :, :, 2:] - sp_x[:, :, :, :-2])
        sp_y = F.pad(sp, (0, 0, 1, 1), mode="replicate")
        gy = 0.5 * (sp_y[:, :, 2:, :] - sp_y[:, :, :-2, :])
        return torch.sqrt(gx ** 2 + gy ** 2 + self.GRAD_EPS)

    def _terrain_grad_mag(self, terrain_raw: torch.Tensor) -> torch.Tensor:
        """|terrain slope| = sqrt(u_slope^2 + v_slope^2) from terrain_raw[:,2:4] (the
        u_slope_z/v_slope_z channels, see terrain_encoder.load_terrain_input), avg-pooled
        5x from the 1000x1000 terrain grid to the 200x200 WRF grid. (1,1,H,W), static
        across samples -- broadcast over the batch in forward()."""
        slope = terrain_raw[:, 2:4]
        mag = torch.sqrt(slope[:, 0:1] ** 2 + slope[:, 1:2] ** 2 + self.GRAD_EPS)
        return F.avg_pool2d(mag, kernel_size=5, stride=5)

    def _cond_features(self, prior_spread, mean_out, terrain_raw) -> torch.Tensor:
        """Assemble + standardize the COND_CHANNELS conditioning features, padded to the
        backbone's (check_image_size) resolution so they align with `feats`."""
        if prior_spread is None:
            raise ValueError("variance_conditioning is on but prior_spread was not provided to forward()")
        ps = self.check_image_size(prior_spread)                                  # (B,2,Hp,Wp)
        smg = self.check_image_size(self._speed_grad_mag(mean_out.detach()))      # (B,1,Hp,Wp)
        tgm = self._terrain_grad_mag(terrain_raw)                                 # (1,1,H,W)
        if tgm.shape[0] != ps.shape[0]:
            tgm = tgm.expand(ps.shape[0], -1, -1, -1)
        tgm = self.check_image_size(tgm)                                          # (B,1,Hp,Wp)
        cond = torch.cat([ps, smg, tgm], dim=1)                                   # (B,COND_CHANNELS,Hp,Wp)
        return (cond - self.cond_mean) / self.cond_std

    def forward(self, posterior, bicubic, terrain_raw, prior_spread=None):
        # Input decomposition: replace the EnsCGP u/v channels with the bicubic baseline
        # and the EnsCGP-mean-minus-bicubic residual; keep the EnsCGP Cholesky channels.
        # This is independent of residual_base (which only affects the OUTPUT bases below).
        residual = posterior[:, :2] - bicubic
        model_input = torch.cat([bicubic, residual, posterior[:, 2:5]], dim=1)

        terrain_feat = self.terrain_encoder(terrain_raw)
        if terrain_feat.shape[0] != model_input.shape[0]:
            terrain_feat = terrain_feat.expand(model_input.shape[0], -1, -1, -1)
        x = torch.cat([model_input, terrain_feat], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # Output bases. mean_base per residual_base; chol_base is always EnsCGP's own
        # Cholesky channels (no bicubic/none analog for a covariance -- see docstring).
        if self.residual_base == "enscgp":
            mean_base = posterior[:, :2]
        elif self.residual_base == "bicubic":
            mean_base = bicubic
        else:  # "none"
            mean_base = torch.zeros_like(bicubic)
        mean_base = self.check_image_size(mean_base)
        chol_base = self.check_image_size(posterior[:, 2:5])

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_first = self.conv_first(x)
        feats = self.conv_after_body(self.forward_features(x_first)) + x_first

        mean_residual = self._head_forward(self.mean_head, feats)
        mean_out = mean_base + self.mean_gate * mean_residual

        # chol_head, optionally conditioned: chol_cond(cond) is added to the first conv's
        # output (zero-init -> exact no-op when fresh / when conditioning is off). The rest
        # of the head (activation, functional head_dropout, final conv) is unchanged, so
        # chol_head's state_dict layout/weights are identical with or without conditioning.
        chol_pre = self.chol_head[0](feats)
        if self.variance_conditioning:
            chol_pre = chol_pre + self.chol_cond(self._cond_features(prior_spread, mean_out, terrain_raw))
        chol_pre = self.chol_head[1](chol_pre)  # LeakyReLU
        if self.head_dropout > 0:
            chol_pre = F.dropout2d(chol_pre, p=self.head_dropout, training=self.training)
        chol_residual = self.chol_head[2](chol_pre)  # final conv
        base_l11 = chol_base[:, 0:1]
        base_l21 = chol_base[:, 1:2]
        base_l22 = chol_base[:, 2:3]
        # Diagonal: gate the perturbation in inverse-softplus (pre-activation) space around
        # the base, then map back through softplus -- L11/L22 stay > 0 by construction
        # regardless of how large the gated perturbation grows (no clamping needed).
        l11 = F.softplus(self._inverse_softplus(base_l11) + self.chol_gate * chol_residual[:, 0:1])
        l22 = F.softplus(self._inverse_softplus(base_l22) + self.chol_gate * chol_residual[:, 2:3])
        l21 = base_l21 + self.chol_gate * chol_residual[:, 1:2]  # unconstrained: direct gated addition
        chol = torch.cat([l11, l21, l22], dim=1)

        out = torch.cat([mean_out, chol], dim=1)
        return out[:, :, :H, :W]


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_model(config: dict) -> ProbabilisticSwin2SR:
    m = config["model"]
    # Dropout knobs default to current behavior, so omitting them from the config is a
    # no-op: drop_rate/attn_drop_rate 0.0 and head_dropout 0.0 (off), drop_path_rate 0.1
    # (the Swin2SR default the model already trained with). Raise them to regularize.
    return ProbabilisticSwin2SR(
        img_size=m.get("img_size", 200),
        embed_dim=m.get("embed_dim", 96),
        depths=m.get("depths", [4, 4, 4]),
        num_heads=m.get("num_heads", [6, 6, 6]),
        window_size=m.get("window_size", 8),
        mlp_ratio=m.get("mlp_ratio", 4.0),
        residual_base=m.get("residual_base", "enscgp"),
        residual_gate_init=m.get("residual_gate_init", 0.1),
        head_dropout=m.get("head_dropout", 0.0),
        variance_conditioning=m.get("variance_conditioning", False),
        drop_rate=m.get("drop_rate", 0.0),
        attn_drop_rate=m.get("attn_drop_rate", 0.0),
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )


def _smoke_test(config: dict):
    torch.manual_seed(config.get("seed", 0))
    model = build_model(config)
    model.eval()

    img_size = config["model"].get("img_size", 200)
    posterior = torch.randn(2, ProbabilisticSwin2SR.POSTERIOR_CHANNELS, img_size, img_size)
    # Real EnsCGP Cholesky diagonal (L11, L22) is always > 0 by construction; make the
    # fabricated test data physically valid too (softplus maps randn -> a realistic
    # positive spread) so the chol-tracking check below isn't comparing against an
    # unphysical negative "base" that the model would have had to clamp anyway.
    posterior[:, 2] = F.softplus(posterior[:, 2])
    posterior[:, 4] = F.softplus(posterior[:, 4])
    bicubic = torch.randn(2, 2, img_size, img_size)
    terrain_raw = torch.randn(1, 4, 1000, 1000)

    with torch.no_grad():
        out = model(posterior, bicubic, terrain_raw)

    assert out.shape == (2, 5, img_size, img_size), f"unexpected output shape {out.shape}"

    mu, l11, l21, l22 = out[:, :2], out[:, 2], out[:, 3], out[:, 4]
    assert torch.all(l11 > 0), "L11 must be strictly positive"
    assert torch.all(l22 > 0), "L22 must be strictly positive"

    # Fresh model: mean_out should track its base (gated by a small residual_gate_init,
    # not a near-zero-init head -- so a moderate bound, not a tiny one). base depends on
    # residual_base; default "enscgp" -> posterior mean.
    base_map = {"enscgp": posterior[:, :2], "bicubic": bicubic, "none": torch.zeros_like(bicubic)}
    mean_base = base_map[model.residual_base]
    gate = model.mean_gate.item()
    mean_diff = (mu - mean_base).abs().max().item()
    bound = max(0.5, 5 * gate)  # gate*(head output, full-strength Kaiming-init, std~O(1))
    assert mean_diff < bound, (
        f"fresh model's mean_out should track its {model.residual_base!r} base within a "
        f"gate-scaled margin (gate={gate:.3f}, bound={bound:.3f}), got max abs diff {mean_diff}"
    )

    # Same check for chol: fresh L11/L22 should track the EnsCGP base Cholesky diagonal,
    # L21 the base off-diagonal, within a gate-scaled margin. Diagonal channels compared
    # against the CLAMPED base (matching what forward() actually inverts through softplus).
    chol_base = torch.stack([
        posterior[:, 2].clamp_min(ProbabilisticSwin2SR.CHOL_EPS),
        posterior[:, 3],
        posterior[:, 4].clamp_min(ProbabilisticSwin2SR.CHOL_EPS),
    ], dim=1)
    chol_gate = model.chol_gate.item()
    chol_diff = (torch.stack([l11, l21, l22], dim=1) - chol_base).abs().max().item()
    chol_bound = max(0.5, 5 * chol_gate)
    assert chol_diff < chol_bound, (
        f"fresh model's chol output should track the EnsCGP base within a gate-scaled "
        f"margin (chol_gate={chol_gate:.3f}, bound={chol_bound:.3f}), got max abs diff {chol_diff}"
    )

    # Gradient sanity check: both heads' final convs are now full-strength (Kaiming init,
    # not near-zero), so gradient should flow to the shared backbone strongly from step 0
    # -- and the gates themselves must receive gradient (they're the thing training has to
    # grow).
    model.zero_grad()
    out_grad = model(posterior, bicubic, terrain_raw)
    out_grad.pow(2).mean().backward()
    conv_first_grad = model.conv_first.weight.grad
    assert conv_first_grad is not None, "conv_first.weight.grad is None -- backward() did not reach the backbone"
    backbone_grad = conv_first_grad.abs().max().item()
    assert backbone_grad > 0, "conv_first received zero gradient -- backbone is disconnected from the heads"

    terrain_grad = model.terrain_encoder.stem[0].weight.grad
    assert terrain_grad is not None, "terrain_encoder received no gradient"
    assert terrain_grad.abs().max().item() > 0, "terrain_encoder received zero gradient"

    assert model.mean_gate.grad is not None and model.mean_gate.grad.abs().item() > 0, "mean_gate received zero gradient"
    assert model.chol_gate.grad is not None and model.chol_gate.grad.abs().item() > 0, "chol_gate received zero gradient"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"residual_base={model.residual_base!r}, mean_gate={gate:.4f}, chol_gate={chol_gate:.4f}")
    print(f"L11 range: [{l11.min().item():.4f}, {l11.max().item():.4f}]")
    print(f"L22 range: [{l22.min().item():.4f}, {l22.max().item():.4f}]")
    print(f"L21 range: [{l21.min().item():.4f}, {l21.max().item():.4f}]")
    print(f"max|mean_out - {model.residual_base} base| (fresh model): {mean_diff:.4f} (bound {bound:.4f})")
    print(f"max|chol_out - enscgp base| (fresh model): {chol_diff:.4f} (bound {chol_bound:.4f})")
    print(f"conv_first weight grad max (backbone receives gradient): {backbone_grad:.2e}")
    print(f"terrain_encoder weight grad max (trained jointly): {terrain_grad.abs().max().item():.2e}")
    print(f"Total parameters: {n_params:,}")
    print("Smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    _smoke_test(load_config(args.config))
