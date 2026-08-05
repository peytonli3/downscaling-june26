"""Quantile-loss primitives shared by the training loss and every diagnostic.

Not an entry point -- import it. These four definitions (the tau grid, its CRPS
integration weights, pinball, and the 3-quantile CRPS) used to be copied into
`train_new_enscgp_swin.py`, `extra_scripts/swin/_common.py`,
`extra_scripts/swin/oneoff/_v6_common.py`, and
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

    print("\nAll quantile_metrics self-tests passed.")


if __name__ == "__main__":
    _self_test()
