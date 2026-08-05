# Architecture Changelog

Breaking changes that make checkpoints **incompatible** across versions (will fail
`load_state_dict` with `strict=True`).  Non-breaking changes (config knobs, loss
hyperparameters, training details) are not listed here.

---

## 0729  *(scripts/new_enscgp_swin.py + scripts/train_new_enscgp_swin.py + scripts/multiscale_loss.py)*

**Logs / checkpoints:** `runs/0729/`

A simplification pass over the whole pipeline (audited for complexity that wasn't earning
its keep). Model-side changes are breaking; trainer/loss changes are not.

### Breaking changes from 0714

- **`mean_gate` / `offset_gate` removed.** Both were learnable scalars gating a head's
  residual onto its base (see 0629). `mean_gate` had settled at 0.125 after a full 100-epoch
  run -- barely above its 0.1 init -- indicating the vanishing-upstream-gradient problem it
  was designed to avoid wasn't costing much here in practice. Replaced with two per-head init
  schemes instead of a shared gating mechanism:
  - `mean_head`'s final conv reverts to **near-zero weight init** (std=1e-3, the pre-0629
    approach) -- a fresh model's q50 output is ~0, so q50 starts at exactly `mean_base`.
  - `offset_head`'s final conv keeps **Kaiming-normal** (full strength) but its **bias** is
    now initialized to `inverse_softplus(offset_spread_init)` (default 3.0 m/s -- the
    measured RMS of WRF-minus-bicubic over a 500-sample draw, not a physically exact
    uncertainty) instead of zero, so a fresh model's up/down offsets start near that target
    on every pixel.
- **EnsCGP-sigma offset seeding removed.** The offset head no longer reads the EnsCGP
  posterior's own Cholesky as a per-pixel seed (`sigma_u=L11, sigma_v=sqrt(L21^2+L22^2)`,
  gated in inverse-softplus space around it). `_inverse_softplus` (the per-tensor version)
  and `_enscgp_marginal_std` are removed; `_quantile_offset` is now just
  `softplus(raw) + OFFSET_EPS`, no external base. `init_spread_scale` config key replaced by
  `offset_spread_init`.
- **Input decomposition removed.** `model_input` is now `cat([bicubic, posterior])` (the
  whole 5-channel EnsCGP posterior, concatenated directly) instead of
  `cat([bicubic, enscgp_mean - bicubic, L11, L21, L22])`. Since `conv_first` is linear, the
  two forms are related by an invertible linear recombination and represent *exactly* the
  same set of functions -- the residual decomposition changed no expressivity, only added
  complexity. `INPUT_CHANNELS` stays 7 (same shape); only the *meaning* of channels 2-3
  changes (raw EnsCGP u/v, not an EnsCGP-minus-bicubic residual).
- **Net effect on `--resume-weights-only` from a 0714 checkpoint:** `mean_head`, `offset_head`,
  backbone, and `terrain_encoder` all transfer by name+shape (their module shapes are
  unchanged; a resume doesn't care about the *init* scheme, since the weights are already
  trained). `mean_gate`/`offset_gate` are dropped (no longer exist). **Caveat:** `conv_first`
  *also* transfers by shape (still 11 input channels) despite its channels 2-3 now meaning
  something different -- a naive partial load silently gives `conv_first` weights calibrated
  to the wrong input at those channels. Prefer training 0729 fresh from EnsCGP; only resume
  through this boundary if you explicitly want to test it.
- **Parameter count:** 2,087,058 (2,087,060 from 0714 minus the two removed gate scalars).

### Non-breaking (trainer / loss)

- **Early stopping actually implemented.** `early_stop_patience` existed as a config key
  since before 0714 but was never read by the trainer -- the 0714 run hit its best val loss
  at epoch 29 and trained to epoch 99 anyway (~70 wasted epochs). Now tracked as consecutive
  *val checks* without improvement (not raw epochs, so it composes correctly with
  `val_interval != 1`); training breaks out of the loop once patience is exceeded.
- **Zero-weight structural terms (l1/spectral/gradient in the active config) are skipped
  during training**, not just multiplied by 0 -- `compute_weighted_loss` takes a
  `compute_all` flag; during training a term is only computed if its own weight is > 0
  (saves real compute: `spectral()` alone is two `rfft2` calls per batch for a term that
  wasn't influencing anything). At validation, `compute_all=True` and every structural term
  is computed and logged regardless of weight, as a diagnostic -- so `ms`/`freq` also stay
  visible at their real value even if their weight were ever set to 0.
  `MultiscaleLoss`/`FreqBandLoss` are now built unconditionally at startup (a one-time,
  usually cache-hit `sigma_band` load) rather than only when their weight is > 0, so they're
  always available for that val-side reporting; only the per-batch *call* is gated.
- **`metric_scale`** (the sliced-Wasserstein-vs-L1 scale calibration in `MultiscaleLoss`) is
  now a fixed module constant (`METRIC_SCALE = {"l1": 1.0, "sw": 2.75}`) instead of a
  per-run config knob -- 2.75 was a measured constant, not something meant to vary run to
  run; removed from `new_enscgp_swin_config.json`'s `multiscale` section.
- **`multiscale_loss.py`'s tuning guardrail docstring fixed** -- it still referenced NLL and
  the (removed at 0714) z-score calibration script; updated to reference coverage/PIT
  (`eval_quantile_calibration.py`) instead.

### Removed (dead code, unrelated to the model change)

- `scripts/precompute_prior_spread.py`, `scripts/train_finetune_variance.py`, and
  `extra_scripts/graphing/verify_variance_conditioning.py` -- all downstream of
  `variance_conditioning`, a model constructor option removed entirely at 0714. Code
  recoverable via tag `archive/variance-conditioning`.
- `data/enscgp_prior_spread.npy` (2.1 GB) and `data/cond_feature_stats.npz`, the data
  products of the script above -- moved to `_archive/` on disk (not deleted).

### Investigated, left unchanged

- **Terrain encoder**: benchmarked at 0.57% of a training step's forward+backward time
  (2.4ms of 421.6ms) despite holding 6.7% of total parameters -- not a meaningful cost.
  Left as-is; the native-1000x1000-resolution processing before downsampling is believed to
  matter for fine coastline/slope detail.
- **`FreqBandLoss`** (spatial-vs-FFT decomposition mismatch with `MultiscaleLoss`'s
  `sigma_band`, flagged as a possible unification target) -- kept as a separate term; an
  informal test found measurably worse results with `freq_weight=0`. A more rigorous
  `runs/0714_freq0/` ablation (100 epochs, everything else identical to the 0714 run) is
  on disk for a fuller before/after comparison.

### Follow-up: sign-aware extreme weighting (non-breaking, post-0729)

`extreme_pixel_weights` replaced by `extreme_pixel_weights_signed`. u and v are SIGNED
quantities (a strong westward gust is exactly as extreme as an equally strong eastward
one), which the original weighting didn't account for:

- It ranked a single combined magnitude `sqrt(u^2+v^2)` and, when `apply_to_q10` was on,
  reused the *identical* weight tensor for both q90 and q10 (`w10 = w90`) -- so every
  extreme pixel pushed both tails equally hard regardless of which direction the extreme
  was actually in.
- The new version ranks `|target|` **per component**, independently for u and v (so a
  pixel extreme only in u doesn't spuriously also weight v), then **routes** that one
  shared weight scale by sign: q90's pinball is boosted where the component is positive,
  q10's where negative (if `apply_to_q10`); the *other* tail gets baseline weight (1.0) at
  that pixel, since it has nothing extreme to reach for there.
- Important subtlety caught in review: ranking the positive and negative parts
  *separately* (each independently mean-normalized) was tried first and is wrong -- on a
  one-directional sample (e.g. severe negative gusts, only mild positive values), it would
  boost the "most positive" pixel toward the same max weight as the true severe extreme,
  even though it isn't actually severe in absolute terms. Ranking `|target|` once (so
  "extreme" means the same thing on both sides) and only routing by sign fixes this.

Verified with synthetic continuous data reproducing exactly that one-directional scenario:
the true extreme gets ~max weight on its correct tail and baseline (1.0) on the other; a
merely-locally-largest-but-not-actually-severe value on the opposite sign stays near
baseline rather than being boosted.

---

## 0714  *(scripts/new_enscgp_swin.py + scripts/train_new_enscgp_swin.py)*

**Logs / checkpoints:** `runs/0714/`

### Breaking changes from 0701

Replaces the Gaussian (mean + 2×2 Cholesky) uncertainty head with **direct quantile
prediction** (q10/q50/q90 per component). Wind is heavy-tailed/right-skewed, so a
per-pixel Gaussian was a poor fit; quantiles are distribution-free and better for
extremes.

- **`chol_head` / `chol_gate` (3ch) → `offset_head` / `offset_gate` (4ch).** The head
  now predicts two non-negative offsets per component `[raw_up_u, raw_up_v, raw_down_u,
  raw_down_v]`. Old `chol_head.*` / `chol_gate` keys are gone; `offset_head.*` /
  `offset_gate` are new → strict load fails.
- **Output 5ch → 6ch**, ordered `[q10_u, q10_v, q50_u, q50_v, q90_u, q90_v]`
  (`Q10_SLICE` / `Q50_SLICE` / `Q90_SLICE`).
- **q50 = the old mean head, unchanged.** `mean_head` / `mean_gate` keep their shapes,
  role, and `residual_base` selection, so they **transfer** from a 0701 checkpoint. q50
  is what the structural losses train (stays sharp); it is *not* pinball-trained.
- **Monotonic by construction:** `up/down = softplus(inv_softplus(scale·σ_EnsCGP) +
  offset_gate·raw) + OFFSET_EPS`; `q90 = q50 + up`, `q10 = q50 − down` (⇒ q10 ≤ q50 ≤ q90,
  no crossing). `σ_u = L11`, `σ_v = √(L21²+L22²)` are the EnsCGP marginal stds; a fresh
  model emits `q50 ± 1.28σ` (the EnsCGP posterior as a symmetric 80% band).
- **New model config:** `init_spread_scale` (default 1.28 ≈ Φ⁻¹(0.9)).
- **`variance_conditioning` removed** (it conditioned the now-deleted Cholesky head):
  `chol_cond`, `cond_mean`/`cond_std` buffers, and `load_cond_stats` are gone.
- **`conv_first` unchanged** — still 7 input channels + 4 terrain = 11ch, so the
  backbone/terrain encoder transfer cleanly.

### Loss changes

- **NLL removed.** Uncertainty is trained solely by
  `pin_weight · [pinball(q90, .9) + pinball(q10, .1)]`.
- **Extreme weighting** (config `extreme_weight{enabled, alpha, apply_to_q10}`, default on
  for q90): per-pixel `1 + α·CDF(|truth|)`, detached and per-sample mean-normalized, so
  rare high-wind peaks aren't smoothed out of q90.
- Structural losses (multiscale/freq/l1/spectral/gradient) unchanged — now attach to q50.
- Config: `nll_weight`/`quantile_weight`/`quantiles` removed; `pin_weight` +
  `extreme_weight` added. **Coverage** metrics (q10/q50/q90 exceedance, overall + extreme
  tail) added to `evaluate` as a post-hoc calibration check.

### Continue-from-checkpoint

`--resume-weights-only` now does a **partial, non-strict** load: it transfers only
name+shape-matching tensors (backbone, conv_first, terrain_encoder, `mean_head`,
`mean_gate`) from a 0701 checkpoint; `offset_head`/`offset_gate` start fresh and
`chol_head`/`chol_gate` are dropped. Also supports fresh-from-EnsCGP (no `--resume`).

**Parameter count:** 2,087,060.

**Stale (not updated here):** `scripts/train_finetune_variance.py` and the Cholesky-
consuming diagnostics in `extra_scripts/` targeted the old covariance head and will not
work against a 0714 checkpoint — they need separate updates.

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
