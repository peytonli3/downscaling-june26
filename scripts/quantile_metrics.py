"""Quantile and rank primitives shared by the training loss and every diagnostic.

Not an entry point -- import it. Two families live here:

  * the quantile-forecast metrics -- the tau grid, its CRPS integration weights, pinball,
    and the 3-quantile CRPS;
  * the empirical-rank machinery those and the diagnostics both need -- `rank_cdf`, and the
    wind-speed stratification (`top_speed_mask`) used to report metrics on the windiest
    pixels only.

The first family used to be copied into `train_new_enscgp_swin.py`,
`extra_scripts/swin/_common.py`, `extra_scripts/swin/oneoff/_v6_common.py`, and
`extra_scripts/swin/oneoff/eval_model_scorecard.py`, each carrying a comment promising it
agreed with the others. They live here now, so "the scorecard scores what the run
optimized" is enforced by the import rather than by those comments.

Everything here is elementwise and dtype-preserving: no `.mean()`, no batching, no model.
Callers decide how to reduce (train averages over the batch, the diagnostics aggregate at
the event level, the panel plots keep the per-pixel map).

`pinball` and `crps_3q` accept EITHER numpy arrays or torch tensors -- the diagnostics work
in numpy on memory-mapped arrays, the loss works in torch on the GPU, and the definition
must not fork between them.

This module deliberately imports nothing from the rest of the repo (only numpy/torch), so
it is safe to import from `_v6_common.py`, which pins an OLD model class on `sys.path` and
must not risk pulling the current one in behind it.
"""
from __future__ import annotations

import numpy as np
import torch

# 3-quantile CRPS. The model predicts only q10/q50/q90, so the CRPS integral
#     CRPS(F, y) = 2 * int_0^1 pinball_tau(F^-1(tau), y) dtau
# is approximated by the midpoint rule on this tau grid: each tau represents the interval
# out to the midpoints of its neighbours (boundaries at 0.3 and 0.7, endpoints 0 and 1),
# giving weights that sum to 1. This is a COARSE 3-point rule -- the true integrand is
# unrepresented in the tails beyond q10/q90 -- so read the result as a relative comparison
# metric (lower is better, across pixels/samples/checkpoints), not an absolute CRPS.
CRPS_TAUS = (0.1, 0.5, 0.9)
CRPS_WEIGHTS = (0.3, 0.4, 0.3)


def pinball(q, truth, tau: float):
    """Per-element pinball (tilted-L1) loss at quantile level tau.

    err = truth - q; loss = max(tau*err, (tau-1)*err), i.e. tau*err where the truth is
    above the prediction and (1-tau)*(q-truth) where it is below. Differentiable in q for
    torch inputs. Accepts numpy arrays or torch tensors; the return matches the input.
    """
    err = truth - q
    maximum = torch.maximum if isinstance(err, torch.Tensor) else np.maximum
    return maximum(tau * err, (tau - 1.0) * err)


def crps_from_pinball(*pinball_by_tau):
    """Combine already-computed per-tau pinball losses into CRPS: 2 * sum_k w_k * p_k.

    Separate from `crps_3q` so a caller that already needs the individual pinball terms
    (e.g. reporting pin10/pin90 alongside CRPS) does not recompute them.
    """
    if len(pinball_by_tau) != len(CRPS_WEIGHTS):
        raise ValueError(f"expected {len(CRPS_WEIGHTS)} pinball terms (one per tau in "
                         f"{CRPS_TAUS}), got {len(pinball_by_tau)}")
    return 2.0 * sum(w * p for w, p in zip(CRPS_WEIGHTS, pinball_by_tau))


def crps_3q(q10, q50, q90, truth):
    """Per-element 3-quantile CRPS of (q10, q50, q90) against `truth`.

    The quantiles must be MARGINAL quantiles of the same scalar quantity as `truth` (here:
    one wind component) -- this is well posed per component and says nothing about wind
    SPEED, whose quantiles would need a u/v dependence the model does not provide.

    numpy inputs accumulate in float64 (the CRPS map is a reported statistic, and its
    float32 inputs are often near-cancelling); torch inputs keep their own dtype, so a
    caller that has deliberately gone to double stays there and the training path stays in
    float32.
    """
    terms = [pinball(q, truth, tau) for q, tau in zip((q10, q50, q90), CRPS_TAUS)]
    if isinstance(truth, np.ndarray):
        terms = [t.astype(np.float64, copy=False) for t in terms]
    return crps_from_pinball(*terms)


def rank_cdf(x: torch.Tensor, n_leading: int = 2) -> torch.Tensor:
    """Empirical CDF -- rank / (N-1) -- over the flattened TRAILING dims of `x`, computed
    independently within each slice of the leading `n_leading` dims. Same shape as `x`.

    Independence per slice is the load-bearing part in both callers. With n_leading=2 on a
    (B, C, H, W) field, u and v are each ranked against their OWN distribution rather than
    pooled, so "extreme for this component" does not get redefined by the other component's
    spread; with n_leading=1 on a (B, 1, H, W) band magnitude, each sample is ranked against
    itself, so one unusually energetic sample cannot flatten the weights of the rest.

    Not differentiable (argsort); both callers use it under no_grad / detached.
    """
    lead = x.shape[:n_leading]
    flat = x.reshape(*lead, -1)
    n = flat.shape[-1]
    ranks = flat.argsort(dim=-1).argsort(dim=-1).to(x.dtype)
    return (ranks / max(n - 1, 1)).reshape(x.shape)


# --------------------------------------------------------------------------------------
# Wind-speed stratification -- "metrics on the windiest N% of pixels"
# --------------------------------------------------------------------------------------
# Added to sqrt() when forming wind speed. Carried over from the masks this replaced (see
# top_speed_mask); it keeps the gradient of sqrt finite at exactly zero wind and is ~1e-7
# relative at the speeds this data actually contains, so it never moves a threshold.
SPEED_EPS = 1e-6


def wind_speed(uv, eps: float = SPEED_EPS):
    """(B, 2, H, W) [u, v] -> (B, 1, H, W) wind speed. numpy or torch.

    Kept as (B, 1, H, W) rather than (B, H, W): speed is a per-PIXEL scalar shared by both
    components, and the singleton channel is what lets it broadcast against a (B, 2, H, W)
    per-component quantity without the caller reshaping.
    """
    sqrt = torch.sqrt if isinstance(uv, torch.Tensor) else np.sqrt
    return sqrt(uv[:, 0:1] ** 2 + uv[:, 1:2] ** 2 + eps)


def top_speed_frac_to_pct(top_frac: float) -> str:
    """0.10 -> '10', 0.005 -> '0.5'. The suffix that names a stratum in output columns."""
    return f"{top_frac * 100:g}"


def top_speed_mask(truth, top_frac: float, eps: float = SPEED_EPS):
    """Boolean (B, 1, H, W) marking the windiest `top_frac` of EACH SAMPLE's pixels.

    `truth` is (B, 2, H, W) [u, v] -- always the HIGH-RESOLUTION truth (WRF), never a
    prediction. Stratifying on the prediction would let a model define its own easy cases;
    stratifying on truth asks the fixed question "how does this model do where the wind
    actually was strongest".

    Threshold is on wind SPEED, and is therefore SHARED by u and v: it marks a region of the
    storm, not a per-component condition. Metrics computed inside it are still per component.
    (Note `bias_diagnostic.py` and `tune_sigma.py` deliberately stratify differently -- on
    per-component |value| -- because they are asking a per-component question. Those are not
    this, and were left alone.)

    PER SAMPLE, not pooled across the split: each snapshot contributes its own windiest
    `top_frac`. This answers "how good is the model in each storm's core?". A pooled
    threshold would instead answer "how good is it at high absolute wind speeds?", which
    puts nearly every pixel of a severe storm in the stratum and none of a mild one -- a
    different and much more sample-imbalanced question. Per-sample matches every existing
    mask in this repo, so the numbers stay comparable to the `_ext` columns that predate it.

    top_frac=0.10 reproduces the previous `ext_quantile=0.9` masks exactly.
    """
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
    speed = wind_speed(truth, eps)
    batch = speed.shape[0]
    flat = speed.reshape(batch, -1)
    if isinstance(speed, torch.Tensor):
        thresh = torch.quantile(flat, 1.0 - top_frac, dim=1).reshape(-1, 1, 1, 1)
    else:
        thresh = np.quantile(flat, 1.0 - top_frac, axis=1).reshape(-1, 1, 1, 1)
    return speed >= thresh


def masked_mean(values, mask):
    """Mean of `values` over `mask` -- a scalar. `mask` broadcasts against `values`, so a
    (B,1,H,W) speed mask applies to a (B,2,H,W) per-component quantity.

    NaN when the mask selects nothing, rather than 0: an empty stratum has no defined mean,
    and silently reporting 0.0 would read as a perfect score.
    """
    if isinstance(values, torch.Tensor):
        m = mask.to(values.dtype).expand_as(values)
        denom = m.sum()
        return (values * m).sum() / denom if denom > 0 else values.new_tensor(float("nan"))
    m = np.broadcast_to(np.asarray(mask, dtype=bool), values.shape)
    return values[m].mean() if m.any() else float("nan")


def stratified_error(pred, truth, top_fracs=(0.10,), *, metric: str = "l1",
                     per_component: bool = False, eps: float = SPEED_EPS) -> dict:
    """Error on all pixels and on each top-wind-speed stratum, in one pass.

    pred/truth: (B, 2, H, W). `metric` is "l1" (MAE, the default and the one the README
    reports) or "se" (mean squared error; take sqrt yourself for RMSE, AFTER any averaging
    over events -- sqrt does not commute with the mean).

    Returns {"all": ..., "top10": ..., "top5": ...} keyed by `top_speed_frac_to_pct`, with
    scalars when per_component is False and length-2 arrays [u, v] when it is True.

    This is the convenience wrapper for a one-shot report. A driver that needs to aggregate
    at the EVENT level (eval_model_scorecard.py) must not use it -- it should build the masks
    with `top_speed_mask` and accumulate its own sums, because averaging per-batch means
    would weight events by their sample count.
    """
    if metric == "l1":
        per_pixel = (pred - truth).__abs__()
    elif metric == "se":
        per_pixel = (pred - truth) ** 2
    else:
        raise ValueError(f"metric must be 'l1' or 'se', got {metric!r}")

    is_torch = isinstance(per_pixel, torch.Tensor)
    ones = (torch.ones_like(per_pixel[:, 0:1], dtype=torch.bool) if is_torch
            else np.ones_like(per_pixel[:, 0:1], dtype=bool))
    strata = {"all": ones}
    for frac in top_fracs:
        strata[f"top{top_speed_frac_to_pct(frac)}"] = top_speed_mask(truth, frac, eps)

    out = {}
    for name, mask in strata.items():
        if per_component:
            vals = [masked_mean(per_pixel[:, c:c + 1], mask) for c in range(per_pixel.shape[1])]
            out[name] = (torch.stack(vals) if is_torch else np.array([float(v) for v in vals]))
        else:
            out[name] = masked_mean(per_pixel, mask)
    return out


def _self_test() -> None:
    """numpy and torch must agree, and pinball's minimizer must be the empirical quantile."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(2, 2, 8, 8))
    q10, q50, q90 = truth - 1.5, truth + 0.2, truth + 1.7

    for tau in CRPS_TAUS:
        a = pinball(q50, truth, tau)
        b = pinball(torch.from_numpy(q50), torch.from_numpy(truth), tau).numpy()
        assert np.allclose(a, b), f"numpy/torch pinball disagree at tau={tau}"
    print("  pinball: numpy and torch agree on all three taus OK")

    c_np = crps_3q(q10, q50, q90, truth)
    c_t = crps_3q(*(torch.from_numpy(a) for a in (q10, q50, q90, truth))).numpy()
    assert np.allclose(c_np, c_t), "numpy/torch crps_3q disagree"
    # Reused-pinball path must match the from-quantiles path exactly.
    c_reuse = crps_from_pinball(*(pinball(q, truth, tau)
                                  for q, tau in zip((q10, q50, q90), CRPS_TAUS)))
    assert np.allclose(c_np, c_reuse), "crps_from_pinball disagrees with crps_3q"
    print("  crps_3q: numpy/torch agree; crps_from_pinball matches OK")

    # A perfect prediction scores 0; a worse one scores strictly higher.
    assert crps_3q(truth, truth, truth, truth).max() < 1e-12
    assert crps_3q(q10, q50, q90, truth).mean() < crps_3q(q10 - 5, q50 + 5, q90 + 5, truth).mean()
    print("  crps_3q: zero on a perfect forecast, larger on a worse one OK")

    # rank_cdf: uniform on [0,1] per slice, and independent across slices.
    x = torch.randn(3, 2, 16, 16)
    for n_leading in (1, 2):
        cdf = rank_cdf(x, n_leading=n_leading)
        flat = cdf.reshape(*x.shape[:n_leading], -1)
        assert torch.allclose(flat.min(dim=-1).values, torch.zeros(flat.shape[:-1]))
        assert torch.allclose(flat.max(dim=-1).values, torch.ones(flat.shape[:-1]))
    # Scaling one channel must not move the other's ranks (n_leading=2 independence).
    scaled = x.clone()
    scaled[:, 0] *= 100.0
    assert torch.equal(rank_cdf(x, 2)[:, 1], rank_cdf(scaled, 2)[:, 1]), \
        "rank_cdf(n_leading=2) leaked one channel's scale into the other's ranks"
    print("  rank_cdf: spans [0,1] per slice; slices are independent OK")

    _self_test_stratification()

    print("\nAll quantile_metrics self-tests passed.")


def _self_test_stratification() -> None:
    rng = np.random.default_rng(1)
    truth = rng.normal(scale=5.0, size=(6, 2, 40, 50))

    # The mask selects the right FRACTION, per sample, for numpy and torch alike.
    for frac in (0.10, 0.05, 0.01, 0.5):
        m_np = top_speed_mask(truth, frac)
        m_t = top_speed_mask(torch.from_numpy(truth), frac).numpy()
        assert m_np.shape == (6, 1, 40, 50), m_np.shape
        assert np.array_equal(m_np, m_t), f"numpy/torch masks differ at top_frac={frac}"
        per_sample = m_np.reshape(6, -1).mean(axis=1)
        assert np.allclose(per_sample, frac, atol=1.5 / (40 * 50)), \
            f"top_frac={frac}: selected {per_sample} of each sample"
    print(f"  top_speed_mask: correct per-sample fraction, numpy == torch OK")

    # It really is the WINDIEST pixels, and the threshold is per sample (not pooled):
    # scaling one sample up by 10x must not change any sample's selected COUNT.
    speed = wind_speed(truth)
    m = top_speed_mask(truth, 0.10)
    # Per SAMPLE -- the thresholds are per sample, so a pooled comparison would be
    # meaningless (one sample's selected pixels may be calmer than another's rejected ones).
    for i in range(speed.shape[0]):
        assert speed[i][m[i]].min() >= speed[i][~m[i]].max(), \
            f"sample {i}: mask is not selecting the top of the speed distribution"
    scaled = truth.copy()
    scaled[0] *= 10.0
    m2 = top_speed_mask(scaled, 0.10)
    assert np.array_equal(m.reshape(6, -1).sum(1), m2.reshape(6, -1).sum(1)), \
        "per-sample thresholds leaked across samples (pooled behaviour?)"
    assert np.array_equal(m[1:], m2[1:]), "rescaling one sample changed another's mask"
    print("  top_speed_mask: selects the windiest pixels; thresholds are per sample OK")

    # Shared across components: the mask must not depend on which component is larger.
    swapped = truth[:, ::-1].copy()  # swap u and v -- speed is unchanged
    assert np.array_equal(top_speed_mask(truth, 0.10), top_speed_mask(swapped, 0.10)), \
        "mask changed when u and v were swapped -- it is not speed-based"
    print("  top_speed_mask: depends on speed only, not on the u/v split OK")

    # stratified_error: a known-constant error reproduces exactly in every stratum, and the
    # windy stratum is harder than the bulk for an error that grows with wind speed.
    pred_const = truth + 2.0
    got = stratified_error(pred_const, truth, top_fracs=(0.10, 0.05))
    for name, v in got.items():
        assert abs(float(v) - 2.0) < 1e-12, f"{name}: constant error 2.0 read as {float(v)}"
    assert set(got) == {"all", "top10", "top5"}, got.keys()

    pred_scaled = truth * 1.10  # 10% low bias -> error proportional to |wind|
    s = stratified_error(pred_scaled, truth, top_fracs=(0.10, 0.01))
    assert s["all"] < s["top10"] < s["top1"], f"windy strata should be harder, got {s}"
    print(f"  stratified_error: exact on a constant error; windier strata score worse "
          f"(all {s['all']:.3f} < top10 {s['top10']:.3f} < top1 {s['top1']:.3f}) OK")

    # numpy and torch agree, and per_component returns [u, v].
    t = {k: float(v) for k, v in stratified_error(
        torch.from_numpy(pred_scaled), torch.from_numpy(truth), top_fracs=(0.10,)).items()}
    assert all(abs(t[k] - float(s[k])) < 1e-12 for k in t), f"numpy/torch disagree: {s} vs {t}"
    pc = stratified_error(pred_scaled, truth, top_fracs=(0.10,), per_component=True)
    assert pc["top10"].shape == (2,), pc["top10"].shape
    assert abs(pc["top10"].mean() - s["top10"]) < 0.15, "per-component and pooled means diverge"
    print("  stratified_error: numpy == torch; per_component returns [u, v] OK")

    # se metric, and the empty-stratum contract.
    se = stratified_error(pred_const, truth, top_fracs=(0.10,), metric="se")
    assert abs(float(se["all"]) - 4.0) < 1e-12, se
    assert np.isnan(masked_mean(np.ones((2, 2, 3, 3)), np.zeros((2, 1, 3, 3), dtype=bool))), \
        "an empty stratum must be NaN, not 0.0"
    print("  stratified_error: metric='se' squares; an empty stratum is NaN OK")


if __name__ == "__main__":
    _self_test()
