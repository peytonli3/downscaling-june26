"""One runner for the self-tests that are otherwise reachable only one file at a time.

Several modules end in a `if __name__ == "__main__"` self-test (see CLAUDE.md). Those are
real assertions, but before this file there was no way to run them as a SET -- so "did my
change break anything?" meant remembering which scripts to invoke and in what order.

Run it with no dependencies beyond the conda env you already have::

    python tests/test_selftests.py            # everything
    python tests/test_selftests.py --fast     # skip the ones that build a real model
    python tests/test_selftests.py -k quantile

pytest is NOT required, and is deliberately absent from `environment.yml` -- the invocations
above need nothing beyond the stdlib. Where pytest IS installed, this file is a valid pytest
module too and `pytest -q` picks up the same functions (see `pytest.ini` for `testpaths` and
the `slow` marker).

These are synthetic-data tests: no checkpoint, no GPU, no data/ tree required. The
diagnostics that need a real checkpoint (extra_scripts/swin/oneoff/) are deliberately NOT
here -- they are not self-contained, and a runner would report "no data" as a failure.

Adding a module's self-test here is the point of writing one; a self-test nothing runs is
documentation, not a test.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

try:  # pytest is optional -- see the module docstring
    import pytest
    slow = pytest.mark.slow
except ImportError:  # no-op stand-in so the decorators below still work
    def slow(fn):
        fn.slow = True
        return fn


def test_quantile_metrics_self_test():
    """pinball/CRPS agree between numpy and torch; rank_cdf's slices are independent."""
    import quantile_metrics
    quantile_metrics._self_test()


def test_paths_resolve():
    """resolve(): absolute wins, relative is repo-relative -- the invariant every config
    path in the repo depends on."""
    import paths
    assert paths.resolve("/tmp/x") == Path("/tmp/x")
    assert paths.resolve("data/y") == REPO / "data/y"
    # The two figure-directory spellings must name the same place.
    assert paths.run_figures("0714") == paths.figures_dir(paths.RUNS_DIR / "0714")


def test_terrain_encoder_smoke():
    """TerrainEncoder maps (1,4,1000,1000) -> the 200x200 model grid."""
    import torch
    from terrain_encoder import TerrainEncoder
    enc = TerrainEncoder()
    with torch.no_grad():
        out = enc(torch.randn(1, 4, 1000, 1000))
    assert out.shape[0] == 1 and out.shape[-2:] == (200, 200), f"unexpected shape {tuple(out.shape)}"
    assert out.shape[1] == enc.body[-1].out_channels


@slow
def test_multiscale_loss_self_test():
    """Band decompositions telescope, sliced-W is shift-tolerant, per-band config is
    aligned, and one real fwd+bwd reaches both heads."""
    import multiscale_loss
    multiscale_loss.self_test()


@slow
def test_model_smoke_test():
    """Quantile monotonicity by construction, gradient reaches both heads, and the pinball
    minimizer really is the empirical quantile."""
    import new_enscgp_swin
    new_enscgp_swin._smoke_test(new_enscgp_swin.load_config())


def _is_slow(fn) -> bool:
    if getattr(fn, "slow", False):
        return True
    marks = getattr(fn, "pytestmark", [])
    return any(getattr(m, "name", None) == "slow" for m in marks)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true", help="Skip tests marked slow")
    p.add_argument("-k", dest="pattern", default=None, help="Only run tests whose name contains this")
    args = p.parse_args()

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)
             and (args.pattern is None or args.pattern in n)]

    failures = []
    for name, fn in tests:
        if args.fast and _is_slow(fn):
            print(f"SKIP {name} (slow)")
            continue
        print(f"\n--- {name}")
        start = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - start:.1f}s)")
        except Exception:
            traceback.print_exc()
            print(f"FAIL {name} ({time.time() - start:.1f}s)")
            failures.append(name)

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
