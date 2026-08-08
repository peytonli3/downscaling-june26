# Third-party notices

This project is released under the MIT License (see `LICENSE`), which covers the
code written for it. It also redistributes third-party code under a different
licence, listed below. Where the two differ, the third-party terms govern that
file.

---

## Swin2SR — `scripts/network_swin2sr.py`

The Swin2SR backbone this project builds on. **Vendored verbatim**: the file is a
byte-for-byte copy of the upstream source, with no functional modification. The only
local change is the attribution comment added at the top of the file, which points
here.

| | |
|---|---|
| Upstream | https://github.com/mv-lab/swin2sr (`models/network_swin2sr.py`) |
| Authors | Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte |
| Licence | Apache License 2.0 — full text in `licenses/Apache-2.0.txt` |
| Paper | *Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration*, https://arxiv.org/abs/2209.11345 |

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

`scripts/new_enscgp_swin.py` subclasses this backbone rather than editing it — the
project's own changes (quantile heads, terrain input, `upscale=1` restoration) live
in that separate MIT-licensed file, which is why the vendored copy could stay
unmodified.

Upstream states that its code is heavily based on the
[Swin Transformer](https://github.com/microsoft/Swin-Transformer) and SwinV2
Transformer by Ze Liu et al., and also refers to
[KAIR](https://github.com/cszn/KAIR), [BasicSR](https://github.com/xinntao/BasicSR)
and [SwinIR](https://github.com/JingyunLiang/SwinIR/) — please follow their licences
as well.

---

## Ens-CGP — method, not code

The EnsCGP (ensemble-conditional Gaussian process) first-guess stage implemented in
`scripts/enscgp_train.py` follows the method of Ravela et al.,
https://arxiv.org/abs/2602.13871.

**No code from the authors' implementation is redistributed here.**
`scripts/enscgp_train.py` is an independent implementation written for this project
against the coarsening operator and neighbor ensembles described in the paper. An
earlier helper script that called the authors' reference implementation directly was
removed (2026-08-08) precisely so that this repository has no un-redistributable
dependency; see the "Considered and rejected: analytic GCV" note in
`extra_scripts/enscgp/tune_sigma.py`.

---

## Runtime dependencies

Installed from their own distributions via `environment.yml`, not redistributed
here — PyTorch, timm, NumPy, SciPy, matplotlib, scikit-learn, scikit-image,
rasterio, cartopy, shapely, h5py, python-pptx. Each carries its own licence.
