# Per-band placeability crossover -- findings

Checkpoint: `runs/0714/checkpoints/best.pth`. Test split: 100 events. See band_crossover_diagnostic.py's module docstring for the full method.

## Part 3 -- code check (answered first; determines how Part 1 is read)

F(m) in FreqBandLoss's down_extremes weighting is computed from the TARGET, not the prediction. scripts/multiscale_loss.py:389 binds `t = truth_band[:, c:c+1]` (NOT pred_band), and scripts/multiscale_loss.py:390 computes `pw = self._pixel_weights(t.abs().detach())` from that `t`. The weight map therefore depends only on truth and is IDENTICAL across whichever field (bicubic/EnsCGP/SWIN) is being scored against that truth, at a given band/component/sample. The comparison in Part 1 is like-for-like as specified -- no re-run with target-derived weights is needed, because target-derived weights are already what training and this diagnostic both use.

```python
def __call__(self, pred_mean: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """pred_mean/truth: (B, 2, H, W) → scalar."""
        B, C, H, W = pred_mean.shape
        sigma = self.sigma_band.to(pred_mean.device)
        masks = self._get_masks(H, W, pred_mean.device, pred_mean.dtype)

        pred_F = torch.fft.rfft2(pred_mean)
        truth_F = torch.fft.rfft2(truth)

        total = pred_mean.new_zeros(())
        for b, mask in enumerate(masks):
            w = self.band_weight[b]
            if w == 0.0:
                continue  # fully deactivated: skip this band's irfft2/pixel-weight cost too
            pred_band = torch.fft.irfft2(mask * pred_F, s=(H, W))
            truth_band = torch.fft.irfft2(mask * truth_F, s=(H, W))
            for c in range(2):
                p = pred_band[:, c:c + 1]
                t = truth_band[:, c:c + 1]
                pw = self._pixel_weights(t.abs().detach())
                total = total + w * (pw * (p - t).abs()).mean() / (sigma[b, c] + self.eps)
        return total
```

## Sigma floor check (finest band)

```
  finest band, u: raw sigma=0.698417, floor=0.001000, floor not binding
  finest band, v: raw sigma=0.759445, floor=0.001000, floor not binding
```

## Part 1 -- crossover band

**u component**: bicubic significantly beats SWIN-q50's freq_l1 starting at band 2 (57-175km) and finer.

  Per-band (finest->coarsest), freq_l1 bicubic vs swin_q50, diff=bicubic-swin (95% CI):
  - band 0 (6-25km): bicubic=0.2528 swin=0.3655 diff=-0.1127 [-0.1208, -0.1048] -> bicubic wins
  - band 1 (28-86km): bicubic=0.4039 swin=0.5069 diff=-0.1030 [-0.1124, -0.0935] -> bicubic wins
  - band 2 (57-175km): bicubic=0.4870 swin=0.5078 diff=-0.0207 [-0.0292, -0.0120] -> bicubic wins
  - band 3 (114-359km): bicubic=0.4735 swin=0.4329 diff=+0.0406 [+0.0317, +0.0504] -> swin wins
  - band 4 (>325km): bicubic=0.1798 swin=0.1513 diff=+0.0285 [+0.0212, +0.0355] -> swin wins

**v component**: bicubic significantly beats SWIN-q50's freq_l1 starting at band 2 (57-175km) and finer.

  Per-band (finest->coarsest), freq_l1 bicubic vs swin_q50, diff=bicubic-swin (95% CI):
  - band 0 (6-25km): bicubic=0.2597 swin=0.3702 diff=-0.1105 [-0.1198, -0.1012] -> bicubic wins
  - band 1 (28-86km): bicubic=0.4066 swin=0.5050 diff=-0.0984 [-0.1094, -0.0871] -> bicubic wins
  - band 2 (57-175km): bicubic=0.4825 swin=0.5006 diff=-0.0182 [-0.0278, -0.0078] -> bicubic wins
  - band 3 (114-359km): bicubic=0.4631 swin=0.4251 diff=+0.0380 [+0.0283, +0.0484] -> swin wins
  - band 4 (>325km): bicubic=0.1831 swin=0.1590 diff=+0.0241 [+0.0186, +0.0298] -> swin wins

## Part 1 -- weighted vs unweighted freq_l1 (bicubic minus SWIN, side by side)

Separates "pointwise L1 prefers blur" (unavoidable, ~sqrt(2), present in any pointwise metric) from "the down_extremes weighting prefers blur" (a design choice on top of that). If the gap shrinks substantially unweighted, the fix belongs on the weighting scheme, not (only) on the band range.

**u component**, diff = bicubic - freq_l1(swin) [weighted] vs bicubic - freq_l1_unweighted(swin) [unweighted], both bicubic-minus-SWIN (negative = bicubic wins), 95% CI:
  - band 0 (6-25km): weighted diff=-0.1127 [-0.1208, -0.1048] -> bic wins  |  unweighted diff=-0.0021 [-0.0226, +0.0201] -> ns, unweighted retains 2% of the weighted gap
  - band 1 (28-86km): weighted diff=-0.1030 [-0.1124, -0.0935] -> bic wins  |  unweighted diff=-0.0239 [-0.0414, -0.0053] -> bic wins, unweighted retains 23% of the weighted gap
  - band 2 (57-175km): weighted diff=-0.0207 [-0.0292, -0.0120] -> bic wins  |  unweighted diff=+0.0067 [-0.0065, +0.0210] -> ns, unweighted retains 32% of the weighted gap
  - band 3 (114-359km): weighted diff=+0.0406 [+0.0317, +0.0504] -> swin wins  |  unweighted diff=+0.0489 [+0.0366, +0.0623] -> swin wins, unweighted retains 120% of the weighted gap
  - band 4 (>325km): weighted diff=+0.0285 [+0.0212, +0.0355] -> swin wins  |  unweighted diff=+0.0274 [+0.0193, +0.0352] -> swin wins, unweighted retains 96% of the weighted gap

**v component**, diff = bicubic - freq_l1(swin) [weighted] vs bicubic - freq_l1_unweighted(swin) [unweighted], both bicubic-minus-SWIN (negative = bicubic wins), 95% CI:
  - band 0 (6-25km): weighted diff=-0.1105 [-0.1198, -0.1012] -> bic wins  |  unweighted diff=+0.0063 [-0.0154, +0.0285] -> ns, unweighted retains 6% of the weighted gap
  - band 1 (28-86km): weighted diff=-0.0984 [-0.1094, -0.0871] -> bic wins  |  unweighted diff=-0.0225 [-0.0420, -0.0023] -> bic wins, unweighted retains 23% of the weighted gap
  - band 2 (57-175km): weighted diff=-0.0182 [-0.0278, -0.0078] -> bic wins  |  unweighted diff=+0.0004 [-0.0148, +0.0163] -> ns, unweighted retains 2% of the weighted gap
  - band 3 (114-359km): weighted diff=+0.0380 [+0.0283, +0.0484] -> swin wins  |  unweighted diff=+0.0399 [+0.0264, +0.0539] -> swin wins, unweighted retains 105% of the weighted gap
  - band 4 (>325km): weighted diff=+0.0241 [+0.0186, +0.0298] -> swin wins  |  unweighted diff=+0.0205 [+0.0143, +0.0267] -> swin wins, unweighted retains 85% of the weighted gap

**Verdict**: at every band/component where the WEIGHTED freq_l1 shows a significant bicubic win, the UNWEIGHTED gap is:
  - u, band 0 (6-25km): unweighted gap is 2% of the weighted gap, and remains NOT significant.
  - u, band 1 (28-86km): unweighted gap is 23% of the weighted gap, and remains significant.
  - u, band 2 (57-175km): unweighted gap is 32% of the weighted gap, and remains NOT significant.
  - v, band 0 (6-25km): unweighted gap is 6% of the weighted gap, and remains NOT significant.
  - v, band 1 (28-86km): unweighted gap is 23% of the weighted gap, and remains significant.
  - v, band 2 (57-175km): unweighted gap is 2% of the weighted gap, and remains NOT significant.
  At least one band's bicubic win DISAPPEARS (no longer significant) once the down_extremes weighting is removed -- the weighting scheme, not just pointwise L1 itself, is doing real work in creating that band's crossover. Fixing the weighting (not only the band range) is indicated there.

## Part 1 -- swd / melr corroboration (is the crossover a metric artifact?)

The whole premise is that freq_l1's ranking is a metric property, not a real skill deficit -- swd (displacement-tolerant) and melr (amplitude-only, phase-blind) should then rank SWIN much better than freq_l1 does, on the SAME fields/bands/samples.

**swd** (lower = better), mean over u+v, SWIN vs bicubic:
  - band 0 (6-25km): bicubic=+0.3294 enscgp=+0.1473 swin=+0.0673 -> swin better
  - band 1 (28-86km): bicubic=+0.1424 enscgp=+0.0942 swin=+0.0352 -> swin better
  - band 2 (57-175km): bicubic=+0.0934 enscgp=+0.1149 swin=+0.0489 -> swin better
  - band 3 (114-359km): bicubic=+0.0912 enscgp=+0.1445 swin=+0.0820 -> swin better
  - band 4 (>325km): bicubic=+0.5060 enscgp=+0.5625 swin=+0.4374 -> swin better
  SWIN better than bicubic at 5/5 bands on swd.

**melr** (closer to 0 = better (unbiased power)), mean over u+v, SWIN vs bicubic:
  - band 0 (6-25km): bicubic=-0.4398 enscgp=-0.2703 swin=-0.0659 -> swin better
  - band 1 (28-86km): bicubic=-0.3022 enscgp=-0.3004 swin=-0.0665 -> swin better
  - band 2 (57-175km): bicubic=-0.1204 enscgp=-0.2737 swin=-0.0478 -> swin better
  - band 3 (114-359km): bicubic=-0.0151 enscgp=-0.2189 swin=-0.0370 -> bicubic better
  - band 4 (>325km): bicubic=+0.0136 enscgp=-0.0919 swin=-0.0445 -> bicubic better
  SWIN better than bicubic at 3/5 bands on melr.

Contrast with freq_l1 above (SWIN loses 3/5 bands, the finest ones) -- if swd/melr favor SWIN broadly while freq_l1 does not, that is direct evidence the crossover is freq_l1's own pointwise/phase-sensitive design, not a genuine SWIN placement deficit at those scales relative to bicubic.

## Part 1 -- monotonicity (bicubic <= EnsCGP-mean <= SWIN-q50 in freq_l1, per band)

**u**: NOT monotone at all bands -- band 0:Y, band 1:Y, band 2:N, band 3:N, band 4:N
**v**: NOT monotone at all bands -- band 0:Y, band 1:Y, band 2:N, band 3:N, band 4:N

## Part 2 -- synthetic control (shift vs blur, from truth alone)

- band 0 (6-25km), u: freq_l1 best-shift=0.4570 best-blur=0.0386 -> prefers blur (B)
- band 0 (6-25km), v: freq_l1 best-shift=0.4888 best-blur=0.0417 -> prefers blur (B)
- band 1 (28-86km), u: freq_l1 best-shift=0.4500 best-blur=0.0113 -> prefers blur (B)
- band 1 (28-86km), v: freq_l1 best-shift=0.4753 best-blur=0.0116 -> prefers blur (B)
- band 2 (57-175km), u: freq_l1 best-shift=0.2871 best-blur=0.0031 -> prefers blur (B)
- band 2 (57-175km), v: freq_l1 best-shift=0.2995 best-blur=0.0031 -> prefers blur (B)
- band 3 (114-359km), u: freq_l1 best-shift=0.1604 best-blur=0.0008 -> prefers blur (B)
- band 3 (114-359km), v: freq_l1 best-shift=0.1690 best-blur=0.0008 -> prefers blur (B)
- band 4 (>325km), u: freq_l1 best-shift=0.0297 best-blur=0.0001 -> prefers blur (B)
- band 4 (>325km), v: freq_l1 best-shift=0.0363 best-blur=0.0001 -> prefers blur (B)

**freq_l1 prefers blur over displacement at**: band 0 (6-25km, u), band 0 (6-25km, v), band 1 (28-86km, u), band 1 (28-86km, v), band 2 (57-175km, u), band 2 (57-175km, v), band 3 (114-359km, u), band 3 (114-359km, v), band 4 (>325km, u), band 4 (>325km, v)
This confirms the metric-level property directly (independent of the model): at these bands, freq_l1 rewards attenuating the signal over placing it correctly-but-shifted.

## Recommendation

Down-weight/zero freq_l1 at band(s) [0, 1, 2] (6-25km, 28-86km, 57-175km) -- bicubic (and often EnsCGP-mean, see monotonicity above) significantly beats SWIN-q50 there, consistent with the placeability effect. Keep freq_l1 active at band(s) [3, 4] (114-359km, >325km), where SWIN wins.