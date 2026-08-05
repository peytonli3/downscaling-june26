# `_v6_0714_arch/` — frozen model class, do not edit

`new_enscgp_swin.py` here is a **byte-identical copy** of the v6 model class:

```bash
git show v6-0714:scripts/new_enscgp_swin.py
```

It exists so the diagnostics in this directory that read
`runs/0714/checkpoints/best.pth` stay runnable from a clean checkout, without
depending on git state at run time (a shallow clone or a source tarball has no
`v6-0714` tag to fetch from).

## Why a copy rather than the current class

0714 predates several architecture changes (see `CHANGELOG.md`: the v7
simplification dropped the gates, the EnsCGP-seeded spread, and the
bicubic-residual input decomposition). Loading its checkpoint with the *current*
`ProbabilisticSwin2SR` either fails `load_state_dict` outright or — worse —
silently succeeds where tensor shapes still match but the channels underneath
have changed meaning. `_v6_common.py` pins this directory ahead of `scripts/` on
`sys.path` and then asserts that the pin won, so that failure mode is loud.

`network_swin2sr.py` and `terrain_encoder.py` are unchanged since `v6-0714`
(verified with `git diff v6-0714 -- scripts/network_swin2sr.py scripts/terrain_encoder.py`),
so those are imported from the live `scripts/` rather than copied here.

## If you need to change it

You don't. It describes a checkpoint that already exists and cannot change. If
it ever drifts from the tag, the diagnostics silently start reporting numbers for
a model that never produced `runs/0714/checkpoints/best.pth`. Verify with:

```bash
git show v6-0714:scripts/new_enscgp_swin.py \
  | diff - extra_scripts/swin/oneoff/_v6_0714_arch/new_enscgp_swin.py
```
