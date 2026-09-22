#!/usr/bin/env python3
"""Build a short progress PPT for the three-line mixture + residual steps."""

from __future__ import annotations

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from project_paths import CLASSIFICATION_ROOT

OUT_DIR = CLASSIFICATION_ROOT / "slides"
OUT_PATH = OUT_DIR / "three_line_mixture_and_residuals.pptx"
MIX_DIR = CLASSIFICATION_ROOT / "figures" / "rotation_mixture"
if not MIX_DIR.exists():
    MIX_DIR = CLASSIFICATION_ROOT / "rotation_mixture"
RES_DIR = CLASSIFICATION_ROOT / "figures" / "mixture_residuals"
SIMPLEX_DIR = CLASSIFICATION_ROOT / "figures" / "mixture_simplex"

NAVY = RGBColor(0x1B, 0x36, 0x5D)
ACCENT = RGBColor(0x2B, 0x6C, 0xB0)
MUTED = RGBColor(0x5C, 0x67, 0x73)
DARK = RGBColor(0x1F, 0x29, 0x37)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)
RULE = RGBColor(0xD5, 0xDC, 0xE3)

FONT = "Calibri"
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def _set_run_font(run, size=18, bold=False, color=DARK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("a:latin"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("a:latin"), {})
        rPr.append(rFonts)
    rFonts.set("typeface", name)
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", name)


def _blank(prs) -> object:
    return prs.slide_layouts[6]


def add_rect(slide, left, top, width, height, fill):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def add_text(
    slide,
    left,
    top,
    width,
    height,
    text,
    size=18,
    bold=False,
    color=DARK,
    align=PP_ALIGN.LEFT,
):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _set_run_font(run, size=size, bold=bold, color=color)
    return box


def add_bullets(slide, left, top, width, height, items, size=18, color=DARK, spacing=8):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(spacing)
        p.level = 0
        run = p.add_run()
        run.text = "•  " + item
        _set_run_font(run, size=size, bold=False, color=color)
    return box


def header_bar(slide, title, subtitle=None):
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.92), NAVY)
    add_rect(slide, 0, Inches(0.92), SLIDE_W, Inches(0.06), ACCENT)
    add_text(
        slide,
        Inches(0.45),
        Inches(0.18),
        Inches(12.4),
        Inches(0.45),
        title,
        size=26,
        bold=True,
        color=WHITE,
    )
    if subtitle:
        add_text(
            slide,
            Inches(0.45),
            Inches(0.55),
            Inches(12.4),
            Inches(0.32),
            subtitle,
            size=13,
            color=RGBColor(0xD7, 0xE3, 0xF4),
        )


def caption(slide, text, top=Inches(6.95)):
    add_text(
        slide,
        Inches(0.45),
        top,
        Inches(12.4),
        Inches(0.4),
        text,
        size=14,
        color=MUTED,
    )


def add_picture_fit(slide, path, left, top, width, height):
    pic = slide.shapes.add_picture(str(path), left, top, width=width)
    # scale to box while keeping aspect; python-pptx sets width, height follows
    if pic.height > height:
        ratio = height / pic.height
        pic.height = height
        pic.width = int(pic.width * ratio)
        pic.left = int(left + (width - pic.width) / 2)
    else:
        pic.top = int(top + (height - pic.height) / 2)
    return pic


def title_slide(prs):
    slide = prs.slides.add_slide(_blank(prs))
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, NAVY)
    add_rect(slide, 0, Inches(5.85), SLIDE_W, Inches(1.65), RGBColor(0x14, 0x2A, 0x4A))
    add_text(
        slide,
        Inches(0.7),
        Inches(2.0),
        Inches(12),
        Inches(0.5),
        "State Classification on Winfree Oscillator Networks",
        size=36,
        bold=True,
        color=WHITE,
    )
    add_text(
        slide,
        Inches(0.7),
        Inches(2.7),
        Inches(12),
        Inches(0.45),
        "Three-line mixture scan and residual diagnostics — progress report",
        size=22,
        color=RGBColor(0xC5, 0xD4, 0xEA),
    )
    add_text(
        slide,
        Inches(0.7),
        Inches(6.15),
        Inches(12),
        Inches(0.9),
        "Reference: Tchinda Djeudjo, Muolo, Njougouo, Carletti,\n"
        "Classification of Chimera States via Fourier Analysis and Unsupervised Learning, arXiv:2605.09401 (2026)",
        size=15,
        color=RGBColor(0xD7, 0xE3, 0xF4),
    )


def paper_slide(prs):
    slide = prs.slides.add_slide(_blank(prs))
    header_bar(slide, "The paper we borrow from", "The shared goal: classify without hand-chosen thresholds")
    add_text(
        slide,
        Inches(0.5),
        Inches(1.25),
        Inches(12.3),
        Inches(0.7),
        "Tchinda Djeudjo, Muolo, Njougouo, Carletti (2026)\n"
        "Classification of Chimera States via Fourier Analysis and Unsupervised Learning",
        size=20,
        bold=True,
        color=NAVY,
    )
    add_bullets(
        slide,
        Inches(0.55),
        Inches(2.15),
        Inches(12.2),
        Inches(4.6),
        [
            "Problem: older chimera / coherent / incoherent scores depend on ad-hoc thresholds, so boundaries are unstable.",
            "Method: extract amplitude, phase, and frequency of each oscillator from the time series by Fourier analysis.",
            "Collapse the spatial profiles into three continuous scalars: normalized total variation V(a), V(θ), V(Ω).",
            "Min–max scale, then k-means; silhouette chooses the number of clusters. A coarse split first, then chimera is split further.",
            "Name each cluster from its median values. Never decide in advance how large is “large.”",
        ],
        size=20,
        spacing=12,
    )


def borrow_slide(prs):
    slide = prs.slides.add_slide(_blank(prs))
    header_bar(slide, "What we borrow, and what we do not copy")
    add_rect(slide, Inches(0.45), Inches(1.3), Inches(6.0), Inches(5.5), LIGHT)
    add_rect(slide, Inches(6.85), Inches(1.3), Inches(6.0), Inches(5.5), LIGHT)
    add_text(slide, Inches(0.65), Inches(1.45), Inches(5.6), Inches(0.4), "Borrowed skeleton", size=20, bold=True, color=ACCENT)
    add_text(slide, Inches(7.05), Inches(1.45), Inches(5.6), Inches(0.4), "Rewritten for Winfree", size=20, bold=True, color=ACCENT)
    add_bullets(
        slide,
        Inches(0.65),
        Inches(2.0),
        Inches(5.6),
        Inches(4.5),
        [
            "Use continuous scores, not |ρ|<0.01 or a 15-point plateau.",
            "Unsupervised: fit or cluster first, name the states afterwards.",
            "Scan the full parameter grid first, then inspect where the model fails.",
            "Treat Partial as a mixture, not as a leftover catch-all class.",
        ],
        size=17,
        spacing=10,
    )
    add_bullets(
        slide,
        Inches(7.05),
        Inches(2.0),
        Inches(5.6),
        Inches(4.5),
        [
            "Not chimera on a Rayleigh ring: ER / fully connected Winfree.",
            "The stored data are only (ω, ρ); we do not run Fourier analysis.",
            "Three lines: ρ=0 (death), ρ=c (lock), ρ≈ω (drift).",
            "Each trial yields three weights, not a hard five-color label.",
        ],
        size=17,
        spacing=10,
    )


def method_slide(prs):
    slide = prs.slides.add_slide(_blank(prs))
    header_bar(slide, "The two completed steps", "No re-integration: we reuse the existing numerical solutions")
    add_rect(slide, Inches(0.45), Inches(1.35), Inches(12.4), Inches(2.45), LIGHT)
    add_rect(slide, Inches(0.45), Inches(4.05), Inches(12.4), Inches(2.45), LIGHT)
    add_text(slide, Inches(0.7), Inches(1.5), Inches(12), Inches(0.4), "Step 1.  Three-line mixture scan", size=22, bold=True, color=NAVY)
    add_text(
        slide,
        Inches(0.7),
        Inches(2.0),
        Inches(11.9),
        Inches(1.5),
        "Each oscillator is one sample. Fit three lines — death, a horizontal lock, and ρ≈ω — and obtain\n"
        "(w_death, w_lock, w_drift). Death / Locking / Incoherence means one weight near 1.\n"
        "Partial Death / Partial Locking means two weights are both large: a mix, not a third kind of dynamics.",
        size=18,
        color=DARK,
    )
    add_text(slide, Inches(0.7), Inches(4.2), Inches(12), Inches(0.4), "Step 2.  Residual diagnostics", size=22, bold=True, color=NAVY)
    add_text(
        slide,
        Inches(0.7),
        Inches(4.7),
        Inches(11.9),
        Inches(1.5),
        "For each grid cell: RMSE, max residual, and the shape of residual versus ω.\n"
        "Dark / small = the three lines suffice. Bright band = leftover geometry (bent branches, slipping, …).\n"
        "Data: raw_trials, p = 0.2, 0.4, 0.6, 0.8, 1.0, scanning (K, δ).",
        size=18,
        color=DARK,
    )


def group_slide(prs, title, purpose, image_path, note):
    slide = prs.slides.add_slide(_blank(prs))
    header_bar(slide, title, purpose)
    add_picture_fit(
        slide,
        image_path,
        Inches(0.55),
        Inches(1.2),
        Inches(12.2),
        Inches(5.55),
    )
    caption(slide, note)


def two_image_slide(prs, title, purpose, left_path, right_path, note):
    slide = prs.slides.add_slide(_blank(prs))
    header_bar(slide, title, purpose)
    add_picture_fit(slide, left_path, Inches(0.35), Inches(1.2), Inches(6.3), Inches(5.5))
    add_picture_fit(slide, right_path, Inches(6.7), Inches(1.2), Inches(6.3), Inches(5.5))
    caption(slide, note)


def p_grid_slide(prs):
    slide = prs.slides.add_slide(_blank(prs))
    header_bar(
        slide,
        "Step 1 also scanned the other p values",
        "Same three-line mixture; only the weight maps, no panel-by-panel caption",
    )
    files = [
        (0.2, MIX_DIR / "mixture_weights_p_0.20.png"),
        (0.4, MIX_DIR / "mixture_weights_p_0.40.png"),
        (0.6, MIX_DIR / "mixture_weights_p_0.60.png"),
        (0.8, MIX_DIR / "mixture_weights_p_0.80.png"),
    ]
    positions = [
        (Inches(0.35), Inches(1.2)),
        (Inches(6.75), Inches(1.2)),
        (Inches(0.35), Inches(4.25)),
        (Inches(6.75), Inches(4.25)),
    ]
    for (p, path), (left, top) in zip(files, positions):
        add_text(slide, left, top - Inches(0.28), Inches(6.1), Inches(0.28), f"p = {p:.1f}", size=14, bold=True, color=NAVY)
        add_picture_fit(slide, path, left, top, Inches(6.15), Inches(2.65))


def simplex_grid_slide(prs):
    slide = prs.slides.add_slide(_blank(prs))
    header_bar(
        slide,
        "Weight simplex at the other p",
        "Same reading as p = 1; one triangle per p, no panel-by-panel caption",
    )
    files = [
        (0.2, SIMPLEX_DIR / "mixture_simplex_partial_p_0.20.png"),
        (0.4, SIMPLEX_DIR / "mixture_simplex_partial_p_0.40.png"),
        (0.6, SIMPLEX_DIR / "mixture_simplex_partial_p_0.60.png"),
        (0.8, SIMPLEX_DIR / "mixture_simplex_partial_p_0.80.png"),
    ]
    positions = [
        (Inches(0.35), Inches(1.15)),
        (Inches(6.75), Inches(1.15)),
        (Inches(0.35), Inches(4.20)),
        (Inches(6.75), Inches(4.20)),
    ]
    for (p, path), (left, top) in zip(files, positions):
        add_text(
            slide,
            left,
            top - Inches(0.26),
            Inches(6.1),
            Inches(0.26),
            f"p = {p:.1f}",
            size=14,
            bold=True,
            color=NAVY,
        )
        add_picture_fit(slide, path, left, top, Inches(6.15), Inches(2.70))


def closing_slide(prs):
    slide = prs.slides.add_slide(_blank(prs))
    header_bar(slide, "What we can say so far")
    add_bullets(
        slide,
        Inches(0.55),
        Inches(1.4),
        Inches(12.2),
        Inches(5.5),
        [
            "We borrow “continuous scores + unsupervised, name afterwards,” not the chimera Fourier / total-variation pipeline.",
            "The three-line mixture already separates the coarse phases: death at large K, lock at small δ and moderate K, drift at small K.",
            "Partial is kept: it is a mix of death+drift or lock+drift. The weights say how large “partial” is.",
            "Residual maps show bent branches on the Partial band: fast oscillators are neither ρ=0 nor ρ=ω.",
            "Next: refine only the high-RMSE bright band (free-slope branch / extra plateau). Do not rebuild the whole phase diagram.",
        ],
        size=20,
        spacing=14,
    )


def main() -> None:
    missing = []
    needed = [
        MIX_DIR / "mixture_weights_p_1.00.png",
        MIX_DIR / "mixture_vs_rules_p_1.00.png",
        MIX_DIR / "mixture_examples_p_1.00.png",
        MIX_DIR / "mixture_weights_p_0.20.png",
        MIX_DIR / "mixture_weights_p_0.40.png",
        MIX_DIR / "mixture_weights_p_0.60.png",
        MIX_DIR / "mixture_weights_p_0.80.png",
        RES_DIR / "mixture_residual_maps_p_1.00.png",
        RES_DIR / "mixture_residual_examples_p_1.00.png",
        SIMPLEX_DIR / "mixture_simplex_partial_p_1.00.png",
        SIMPLEX_DIR / "mixture_simplex_partial_p_0.20.png",
        SIMPLEX_DIR / "mixture_simplex_partial_p_0.40.png",
        SIMPLEX_DIR / "mixture_simplex_partial_p_0.60.png",
        SIMPLEX_DIR / "mixture_simplex_partial_p_0.80.png",
    ]
    for path in needed:
        if not path.exists():
            missing.append(path)
    if missing:
        raise SystemExit("Missing figures:\n" + "\n".join(str(p) for p in missing))

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    title_slide(prs)
    paper_slide(prs)
    borrow_slide(prs)
    method_slide(prs)
    group_slide(
        prs,
        "Step 1.  Three-line mixture weights (p = 1)",
        "This figure: at each (K, δ), how much is death / lock / drift",
        MIX_DIR / "mixture_weights_p_1.00.png",
        "Yellow ≈ 1, black ≈ 0. Bottom-right RGB: purple = death, blue = lock, green = drift. Boundaries are Partial.",
    )
    group_slide(
        prs,
        "Step 1.  Comparison with the old five-color rules (p = 1)",
        "This figure: same data; left = old threshold labels, right = mixture weights",
        MIX_DIR / "mixture_vs_rules_p_1.00.png",
        "Coarse phases match. The old red Partial Death band is a purple–green continuum on the right, not a hard extra phase.",
    )
    group_slide(
        prs,
        "Step 1.  Open single cells: ρ versus ω (p = 1)",
        "This figure: where the weights come from — each point is one oscillator",
        MIX_DIR / "mixture_examples_p_1.00.png",
        "Top row: pure states on ρ=0 / a horizontal lock / the diagonal. Bottom row: mixes, death+drift or lock+drift.",
    )
    p_grid_slide(prs)
    group_slide(
        prs,
        "Weight simplex of Partial cells (p = 1)",
        "This figure: each old Partial cell is one point in (w_death, w_lock, w_drift)",
        SIMPLEX_DIR / "mixture_simplex_partial_p_1.00.png",
        "Red = Partial Death (death–drift edge). Orange = Partial Locking (lock–drift edge). Empty interior: two-way mix, not a fifth geometry.",
    )
    simplex_grid_slide(prs)
    group_slide(
        prs,
        "Step 2.  Residual maps (p = 1)",
        "This figure: where the three lines already fail",
        RES_DIR / "mixture_residual_maps_p_1.00.png",
        "Not a new five-color phase diagram. Black = good fit; bright band = three lines are not enough. The band sits on the old Partial region.",
    )
    group_slide(
        prs,
        "Step 2.  Residual versus ω (p = 1)",
        "This figure: the 6 cells with largest RMSE, to see which branch is missing",
        RES_DIR / "mixture_residual_examples_p_1.00.png",
        "Vertical axis is ρ − prediction. Blue points near 0 stay locked; the green branch bends down — neither death nor ρ=ω.",
    )
    closing_slide(prs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"Saved {OUT_PATH}")
    print(f"slides = {len(prs.slides)}")


if __name__ == "__main__":
    main()
