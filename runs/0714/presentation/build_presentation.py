#!/usr/bin/env python3
"""Build the q10/q50/q90 model presentation (v0714) as a .pptx, pulling the generated
figures from runs/0714/figures/. Regenerate with:
    python build_presentation.py
Output: q10_q50_q90_model.pptx next to this script. Speaker notes (with rough timing) are
attached to every slide."""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
OUT = HERE / "q10_q50_q90_model.pptx"

# ── palette ──
NAVY = RGBColor(0x14, 0x2A, 0x4A)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)   # blue rule / takeaway
GOOD = RGBColor(0x2E, 0x7D, 0x46)     # green (wins)
BAD = RGBColor(0xB3, 0x3A, 0x3A)      # red (limitations)
GRAY = RGBColor(0x33, 0x33, 0x33)
LIGHT = RGBColor(0xEC, 0xF2, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SW, SH = 13.333, 7.5  # 16:9 inches

prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]


def _txt(slide, l, t, w, h, anchor=None):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    return tf


def _rect(slide, l, t, w, h, color):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def _set(p, text, size, color=GRAY, bold=False, align=None):
    p.text = text
    r = p.runs[0]
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = "Calibri"
    if align is not None:
        p.alignment = align


def _footer(slide, n):
    tf = _txt(slide, SW - 1.2, SH - 0.5, 0.9, 0.35)
    _set(tf.paragraphs[0], f"{n}", 12, RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


def header(slide, title):
    tf = _txt(slide, 0.6, 0.32, SW - 1.2, 0.95, anchor=MSO_ANCHOR.MIDDLE)
    _set(tf.paragraphs[0], title, 30, NAVY, bold=True)
    _rect(slide, 0.62, 1.28, SW - 1.24, 0.045, ACCENT)


def fit(img_path, bl, bt, bw, bh):
    iw, ih = Image.open(img_path).size
    ar, box_ar = iw / ih, bw / bh
    if ar > box_ar:
        w, h = bw, bw / ar
    else:
        h, w = bh, bh * ar
    return bl + (bw - w) / 2, bt + (bh - h) / 2, w, h


def bullets(n, title, items, notes=""):
    """items: list of (text, level, color). level 0/1; color optional (defaults GRAY)."""
    slide = prs.slides.add_slide(BLANK)
    header(slide, title)
    tf = _txt(slide, 0.75, 1.55, SW - 1.5, SH - 2.1)
    first = True
    for item in items:
        text, level = item[0], item[1]
        color = item[2] if len(item) > 2 else GRAY
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        prefix = "▸  " if level == 0 else "–  "
        _set(p, prefix + text, 22 if level == 0 else 18, color, bold=(level == 0))
        p.level = level
        p.space_after = Pt(10 if level == 0 else 5)
        if level == 1:
            p.runs[0].font.bold = False
    _footer(slide, n)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def figure(n, title, img, takeaway, notes=""):
    slide = prs.slides.add_slide(BLANK)
    header(slide, title)
    l, t, w, h = fit(img, 0.5, 1.5, SW - 1.0, 5.0)
    slide.shapes.add_picture(str(img), Inches(l), Inches(t), Inches(w), Inches(h))
    # takeaway strip
    _rect(slide, 0.6, SH - 0.92, SW - 1.2, 0.62, LIGHT)
    _rect(slide, 0.6, SH - 0.92, 0.09, 0.62, ACCENT)
    tf = _txt(slide, 0.85, SH - 0.9, SW - 1.5, 0.58, anchor=MSO_ANCHOR.MIDDLE)
    _set(tf.paragraphs[0], takeaway, 15, NAVY, bold=True)
    _footer(slide, n)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


# ── 1. TITLE ──
s = prs.slides.add_slide(BLANK)
_rect(s, 0, 0, SW, SH, NAVY)
_rect(s, 0, 4.35, SW, 0.06, ACCENT)
tf = _txt(s, 0.9, 2.2, SW - 1.8, 2.0, anchor=MSO_ANCHOR.BOTTOM)
_set(tf.paragraphs[0], "Probabilistic Wind Downscaling", 44, WHITE, bold=True)
p = tf.add_paragraph()
_set(p, "with Direct Quantile Regression (q10 / q50 / q90)", 26, RGBColor(0xBF, 0xD7, 0xE6))
tf2 = _txt(s, 0.92, 4.6, SW - 1.8, 1.4)
_set(tf2.paragraphs[0], "Replacing the Gaussian/Cholesky head on the EnsCGP → Swin2SR refiner", 18,
     RGBColor(0xD8, 0xE4, 0xEC))
p = tf2.add_paragraph()
_set(p, "model version 0714  ·  git tag v6-0714  ·  July 2026", 15, RGBColor(0x9F, 0xB8, 0xC8))
s.notes_slide.notes_text_frame.text = (
    "~15 min talk. Goal: motivate the switch from a Gaussian uncertainty head to direct "
    "quantiles, show the architecture and loss, then the results and the one open problem "
    "(tail overconfidence). [~0:30]")

# ── 2. PROBLEM ──
bullets(2, "The problem", [
    ("Downscale coarse ERA5 wind to WRF resolution over the NE-US coastal domain", 0),
    ("ERA5 native 34×34  →  WRF 200×200 (u, v components)", 1),
    ("We need per-pixel UNCERTAINTY, not just a best guess", 0),
    ("Downstream use is damage / extremes — the tail is what matters", 1),
    ("Wind is heavy-tailed and right-skewed", 0),
    ("A per-pixel Gaussian (the old head) is a poor fit, especially at peaks", 1),
    ("Goal: distribution-free per-pixel uncertainty that behaves at the extremes", 0, ACCENT),
], notes=(
    "Set up why uncertainty matters and why Gaussian is the wrong tool: symmetric, light-"
    "tailed, and it forces a covariance we don't really trust at extremes. [~1:30]"))

# ── 3. PIPELINE ──
bullets(3, "The pipeline", [
    ("ERA5 (low-res)  →  EnsCGP posterior  →  Swin2SR refiner  →  q10 / q50 / q90", 0, NAVY),
    ("EnsCGP: analog-ensemble first guess (posterior mean + per-pixel covariance)", 0),
    ("Built from the spread across each sample's k WRF analogs", 1),
    ("Swin2SR: a ~2.1M-param transformer refiner at native resolution (upscale = 1)", 0),
    ("Terrain encoder trained jointly; fed bicubic ERA5 + EnsCGP residual + Cholesky", 1),
    ("This talk: only the OUTPUT HEAD and the uncertainty LOSS changed (v0714)", 0, ACCENT),
    ("Backbone, structural losses, and data pipeline are unchanged", 1),
], notes=(
    "Orient the audience: EnsCGP is the statistical first guess, Swin refines it. Emphasize "
    "the change is surgical — head + loss only, everything else identical, so results are "
    "attributable to the head swap. [~1:15]"))

# ── 4. ARCHITECTURE CHANGE ──
bullets(4, "v0714: Gaussian → quantiles", [
    ("Old head: mean μ + 2×2 Cholesky covariance, trained by Gaussian NLL", 0, BAD),
    ("Assumes a symmetric, light-tailed distribution per pixel", 1),
    ("New head: direct q10 / q50 / q90 per component (u and v)", 0, GOOD),
    ("Distribution-free — no shape assumption; naturally handles skew and tails", 1),
    ("q50 keeps the old mean's role (it is literally still the mean head's output)", 1),
    ("Checkpoint-incompatible change — versioned as git tag v6-0714", 0),
    ("Old backbone / terrain / mean-head weights transfer via a partial load", 1),
], notes=(
    "The crux. NLL is gone; we predict quantiles directly. q50 is the same central field the "
    "structural losses already trained, so we keep sharpness for free. [~1:15]"))

# ── 5. HEAD MECHANICS ──
bullets(5, "The head — monotone by construction", [
    ("q90 = q50 + up_offset,   q10 = q50 − down_offset", 0, NAVY),
    ("Offsets = softplus(…) + ε  ⇒  strictly positive", 1),
    ("So q10 ≤ q50 ≤ q90 ALWAYS — no quantile crossing, no penalty needed", 0, GOOD),
    ("Offsets seeded from the EnsCGP per-pixel σ (σ_u = L11, σ_v = √(L21²+L22²))", 0),
    ("A fresh model emits q50 ± 1.28σ — the EnsCGP posterior as an 80% band", 1),
    ("softplus (not ReLU): no dead zone, spread can't collapse to zero", 1),
], notes=(
    "Why monotonicity is guaranteed rather than hoped for: we add non-negative offsets to a "
    "shared center. The EnsCGP-σ seeding means training starts from a sensible band and only "
    "has to learn corrections. [~1:15]"))

# ── 6. LOSS ──
bullets(6, "The loss", [
    ("Structural losses (multi-scale Wasserstein + frequency) train q50 only", 0),
    ("q50 stays SHARP and displacement-tolerant — unchanged from before", 1),
    ("Pinball (quantile) loss trains q90 and q10", 0),
    ("q50 is deliberately NOT pinball-trained (would blur it toward the median)", 1),
    ("Extreme weighting: up-weight high-wind pixels in the q90 pinball", 0, ACCENT),
    ("Rare peaks are a tiny fraction of pixels — without it, q90 smooths off the peaks", 1),
    ("Weight = 1 + α·CDF(|wind|), detached & mean-normalized; α is the tuning knob", 1),
], notes=(
    "Two jobs split cleanly: structural loss owns the central field, pinball owns the band. "
    "Extreme weighting is the lever for the tail — remember α, it comes back in the results "
    "and next steps. [~1:30]"))

# ── 7. TRAINING ──
figure(7, "Training & calibration during training", FIG / "training_curves.png",
       "Val loss converges; coverage panel shows q90 ≈ 0.90 and q10 ≈ 0.09 — well calibrated in the bulk.",
       notes=(
        "Panel 1: val loss, smooth convergence. Panel 4 is the key one — coverage tracks the "
        "nominal .10/.50/.90 lines within a couple epochs and stays there. The dashed extreme-"
        "tail lines already hint at the tail problem we'll quantify. [~1:15]"))

# ── 8. QUALITATIVE ──
figure(8, "Qualitative results (test samples)", FIG / "swin_quantile_panels.png",
       "q50 sharpens the EnsCGP first guess toward WRF truth; the q10–q90 band brackets it; CRPS concentrates on high-wind regions.",
       notes=(
        "Left to right: ERA5 LR, EnsCGP, WRF truth, then q10/q50/q90 and CRPS. Note q50 recovers "
        "fine structure the first guess lacks. Caveat to state out loud: the q10/q90 panels are "
        "component 'scenario' fields, not speed percentiles. CRPS lights up exactly where the "
        "wind is strong. [~1:30]"))

# ── 9. SPECTRA ──
figure(9, "Spectral fidelity of the central field", FIG / "eigenspectra_aggregate_swin.png",
       "q50 (green) tracks WRF's power spectrum across all scales; bicubic and EnsCGP fall short below ~100 km.",
       notes=(
        "This is the deterministic-sharpness evidence. Radial power spectra: the model recovers "
        "high-wavenumber energy the baselines are missing — the multi-scale loss delivers "
        "texture, not just a smooth field. [~1:00]"))

# ── 10. PIT ──
figure(10, "Calibration: PIT / rank histogram", FIG / "quantile_pit_histogram.png",
       "Overall bins match nominal [.10 .40 .40 .10] — calibrated. But the extreme-tail ≥q90 bin hits ~0.15–0.18 (should be 0.10).",
       notes=(
        "How often truth lands in each of the 4 bins cut by the quantiles. Blue overall bars sit "
        "on the nominal steps — genuinely calibrated in bulk. Red extreme-tail bars: the ≥q90 bin "
        "is ~1.5–1.8× too tall — truth punches through q90 far too often at peaks. This is THE "
        "problem. [~1:30]"))

# ── 11. COVERAGE MAPS ──
figure(11, "Where it fails: spatial coverage maps", FIG / "quantile_coverage_maps.png",
       "q10/q90 near nominal spatially; the q50 maps reveal a systematic median bias with a coherent land/ocean pattern.",
       notes=(
        "Per-pixel coverage minus nominal, blue = under-covers. q10/q90 are mostly fine. The "
        "surprise is the q50 panels — a strong land/ocean structure means the median is "
        "systematically biased over terrain, which the aggregate number hides. A second, "
        "separate lead to chase. [~1:15]"))

# ── 12. FINDINGS ──
bullets(12, "Findings", [
    ("Distribution-free uncertainty, monotone by construction — no crossing", 0, GOOD),
    ("Calibrated in the bulk: q90 ≈ 0.90, q10 ≈ 0.09 overall", 0, GOOD),
    ("Central field is sharp: matches WRF's spectrum across scales", 0, GOOD),
    ("Tail overconfidence: truth exceeds q90 ~1.5–1.8× too often at peaks", 0, BAD),
    ("Systematic q50 (median) bias over terrain — hidden by aggregate metrics", 0, BAD),
], notes=(
    "Scorecard. Three wins, two open problems. Be honest that the tail — the thing we care "
    "most about — is where it's weakest, and that's the current focus. [~1:00]"))

# ── 13. NEXT STEPS ──
bullets(13, "Next steps", [
    ("Raise extreme-weight α (currently being tuned); re-check tail coverage", 0),
    ("Terrain-stratified coverage / CRPS — test the orographic-bias hypothesis", 0),
    ("CRPS skill score vs EnsCGP and bicubic baselines — quantify probabilistic value", 0),
    ("Threshold-exceedance reliability (Brier) for damage thresholds", 0),
    ("Later: Monte-Carlo speed quantiles for operational speed uncertainty", 1),
], notes=(
    "Concrete and prioritized. α is the immediate lever for the tail; terrain stratification "
    "explains the q50 bias; skill scores answer 'is it actually better than the first guess.' "
    "[~1:00]"))

# ── 14. SUMMARY ──
s = prs.slides.add_slide(BLANK)
_rect(s, 0, 0, SW, SH, NAVY)
tf = _txt(s, 0.9, 1.1, SW - 1.8, 1.0)
_set(tf.paragraphs[0], "Summary", 34, WHITE, bold=True)
_rect(s, 0.92, 2.05, 3.0, 0.05, ACCENT)
tf = _txt(s, 0.95, 2.4, SW - 1.9, 4.3)
lines = [
    ("v0714 replaces the Gaussian/Cholesky head with monotone q10/q50/q90 + pinball loss", True),
    ("Distribution-free, no quantile crossing, seeded from the EnsCGP posterior spread", False),
    ("Sharp central field (spectrum matches WRF) and calibrated in the bulk", False),
    ("Open problem: overconfident in the damaging tail — the current focus", False),
    ("Fully reproducible: git tag v6-0714; every figure comes from a committed script", False),
]
first = True
for text, lead in lines:
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    first = False
    _set(p, ("→  " if lead else "     •  ") + text, 20 if lead else 17,
         WHITE if lead else RGBColor(0xC9, 0xDA, 0xE6), bold=lead)
    p.space_after = Pt(12)
tf = _txt(s, 0.95, 6.7, SW - 1.9, 0.5)
_set(tf.paragraphs[0], "Thank you — questions welcome", 16, RGBColor(0x9F, 0xB8, 0xC8))
s.notes_slide.notes_text_frame.text = (
    "Land the plane: the head swap gives principled, calibrated uncertainty with a sharp "
    "center; the tail is the next milestone. Everything is scripted and versioned. [~0:45]")

n_slides = len(prs.slides._sldIdLst)
prs.save(str(OUT))
print(f"Saved {OUT}  ({n_slides} slides)")