# Architecture Changelog

Breaking changes that make checkpoints **incompatible** across versions (will fail
`load_state_dict` with `strict=True`).  Non-breaking changes (config knobs, loss
hyperparameters, training details) are not listed here.

---

## Repo reorganization (2026-07-13)

Not a model change — repository structure only.

- **Now a git repo.** Versions are git tags (see `README.md`); binaries
  (`data/`, `*.pth`, `*.npy`, `*.png`, `*.db`) are gitignored and managed on disk.
- **Snapshot dirs removed.** `scripts/0626_v1` and `scripts/0628_v2` deleted;
  `0628_v2`'s two scripts folded into flat `scripts/`. Old code recoverable via
  tag `archive/pre-reorg-snapshots`.
- **ResNet-refiner line retired.** Code in tag `archive/resnet-refiner`; binary
  artifacts moved to `_archive/`.
- **Recoverability gap:** the code for **0627** and **0629** was overwritten before
  git existed and is *not* recoverable — only the diffs below describe it.
- Kept `scripts/` flat (imports are bare-name) and left `data/`/`logs/`/
  `inference_results/` in place (gitignored binaries; documented, not moved).

---


## 0701 special

**Fixed alignment issue between ERA5 and WRF** Flipped WRF .npy files in data dir north south and deleted all display scripts: All scripts currently used are north-up.
**Redid EnsCGP** Unclear as to what results are, but redid EnsCGP process with new sigma variables

## 0701  *(scripts/new_enscgp_swin.py + scripts/terrain_encoder.py)*

**Logs / checkpoints:** `logs/0701/`
**Inference results:** `inference_results/0701/`

### Changes from 0629

**Not a breaking weight change** — state_dict shapes are identical to 0629.  A 0629
checkpoint can be loaded with `--resume-weights-only` (gates and all heads are
shape-compatible), but the learned residuals will not transfer cleanly because the
model now predicts a correction relative to a different base.

- **`residual_base` changed: `"enscgp"` → `"bicubic"`.**  The mean head now adds its
  gated residual on top of the ERA5 bicubic baseline instead of the EnsCGP posterior
  mean.  The model still receives all of the same input channels (bicubic, EnsCGP
  mean expressed as bicubic + residual, EnsCGP Cholesky, terrain), so no information
  is lost — only what the head's output is anchored to changes.  Rationale: bicubic
  is a weaker, more stable base; the model must therefore learn to reproduce the
  full EnsCGP improvement rather than only a correction on top of it.
- **Inputs unchanged**: `[bic_u, bic_v, res_u, res_v, L11, L21, L22]` + 4ch terrain
  → 11ch `conv_first`.  `forward` signature unchanged.

---

## 0629  *(scripts/new_enscgp_swin.py + scripts/terrain_encoder.py)*

**Logs / checkpoints:** `logs/0629/`
**Inference results:** `inference_results/0629/` *(files were saved without the version subdir; manually moved)*

### Breaking changes from 0627

- **`mean_gate` and `chol_gate` added** — two new learnable scalar `nn.Parameter`s
  that gate each head's residual contribution onto its base.  0627 checkpoints are
  missing these keys and will error on strict load.  All `conv_first` / RSTB body /
  head weights are otherwise shape-compatible with 0627.
- **`conv_first` shape unchanged** — still 11 input channels; terrain encoder output
  channel count unchanged (4).  Only the two gate scalars are new.

### Other changes (same epoch, not necessarily breaking)

- `forward(posterior, bicubic, terrain_raw, prior_spread=None)` — `prior_spread`
  argument added for optional variance conditioning (`variance_conditioning=False`
  default keeps the model identical to 0627 when not used).  When
  `variance_conditioning=True`, also adds `chol_cond` (Conv2d), `cond_mean`, and
  `cond_std` buffers — another breaking addition, but only when that flag is on.
- **Head init changed**: final conv in each head now uses Kaiming-normal init
  (previously near-zero noise, std=1e-3).  The gate (small init via
  `residual_gate_init=0.1`) takes the role of suppressing the fresh model's output,
  not a suppressed weight — gradient flows normally from step 0.
- **Cholesky gating in inverse-softplus space**: `L11`/`L22` perturbation is added in
  `log(exp(x)-1)` space around the EnsCGP base, then mapped back through `softplus`,
  so the diagonal stays strictly positive regardless of gate magnitude.  0626/0627
  used direct `softplus(raw)` without reference to the EnsCGP base.
- `residual_base` config key added (`"enscgp"` / `"bicubic"` / `"none"`); default
  `"enscgp"`.
- Backbone dropout knobs added: `drop_rate`, `attn_drop_rate`, `drop_path_rate`,
  `head_dropout`.  Defaults match prior behaviour (0 or the Swin2SR default).
- Loss: `multiscale_loss` (Laplacian-pyramid, displacement-tolerant sliced-Wasserstein)
  replaces the L1 / spectral / gradient stack as the primary mean-supervision term.
  `ms_weight=1.0`, `l1_weight=0.0`, `spectral_weight=0.0`, `gradient_weight=0.0`.

**Parameter count:** 2,086,195 (2,086,193 from 0627 + 2 gate scalars)

---

## 0627  *(scripts/new_enscgp_swin.py + scripts/terrain_encoder.py, pre-gate version)*

**Checkpoint:** `logs/0627/best.pth`

### Breaking changes from 0626_v1

- **`conv_first` shape: 37 → 11 input channels.**  The biggest single incompatibility.
  0626_v1 fed `cat([posterior(5ch), terrain_encoder_32(32ch)])` directly into
  `conv_first`.  0627 replaces this with the bicubic-decomposed 7-channel input plus
  the rebuilt 4-channel terrain encoder.
- **Terrain encoder rebuilt, output 32 → 4 channels.**  `terrain_encoder_32.py`
  (body: Conv200→64 → Conv64→32, 2 convs) replaced by `terrain_encoder.py`
  (body: Conv200→64 → Conv64→32 → Conv32→16 → Conv16→8 → Conv8→4, 5 convs).
  All terrain encoder keys (`body.5`, `body.7`, `body.9`) are new; the first two
  body convs (`body.1`, `body.3`) are shape-incompatible with the old 64→32 final.
- **`forward` signature changed**: `forward(posterior, terrain_raw)` →
  `forward(posterior, bicubic, terrain_raw)`.  Bicubic field is now an explicit
  model input (not just a training-time baseline).
- **Input decomposition changed**: 0626_v1 passed the raw posterior 5 channels
  (u, v, L11, L21, L22) into `conv_first`.  0627 replaces u/v with the bicubic
  field plus the EnsCGP-mean-minus-bicubic residual, keeping the Cholesky channels:
  `[bic_u, bic_v, res_u, res_v, L11, L21, L22]` — 7 channels instead of 5.
- **Chol head output channels: 3 (unchanged), but interpretation changed**: the head
  now predicts a perturbation on the EnsCGP base Cholesky (not an absolute Cholesky),
  residual-added before `softplus`.

### Other changes

- Head init: still near-zero noise (std=1e-3) on the final conv; no gate scalars yet.
- Mean head: residual added directly to posterior mean (no gate); same intent as 0626.
- Loss: NLL + L1 + spectral + gradient (same legacy stack as 0626).

**Parameter count:** 2,086,193

---

## 0626_v1  *(scripts/0626_v1/new_enscgp_swin.py + scripts/0626_v1/terrain_encoder_32.py)*

**Checkpoint:** `logs/0626_v1/checkpoints/best.pth`
**Inference script:** `extra_scripts/0626_v1/swin/compare_eigenspectra_enscgp_swin.py`

Initial model version.  No prior version to compare against.

### Architecture summary

- **`forward(posterior, terrain_raw)`** — no bicubic input.
- **Input**: `cat([posterior(5ch), terrain_encoder_32(terrain_raw)(32ch)])` → `conv_first`
  — **37 input channels**.
- **Terrain encoder** (`terrain_encoder_32.py`, 32ch output):
  - stem: Conv(4→8) → LeakyReLU
  - PixelUnshuffle(5): (8, 1000×1000) → (200, 200×200)
  - body: LeakyReLU → Conv(200→64) → LeakyReLU → Conv(64→32)
- **Heads**: near-zero-init final conv (std=1e-3, zero bias); mean residual added
  directly to the posterior mean; Cholesky channels predicted absolutely (no
  EnsCGP-base tracking).
- **No gate scalars** (`mean_gate` / `chol_gate` do not exist).
- Loss: NLL + L1 + spectral + gradient.
