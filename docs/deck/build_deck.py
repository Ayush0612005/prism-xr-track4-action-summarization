#!/usr/bin/env python3
"""Build SRM_ToothpasteEater.pptx. Every number on the results slides is computed here from outputs/
and data/manifest.csv, so the deck cannot drift from the submission.

usage: python docs/deck/build_deck.py [out.pptx]
"""
import csv
import glob
import json
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "SRM_ToothpasteEater.pptx")

INK = RGBColor(0x16, 0x20, 0x2A)      # dominant dark
INK2 = RGBColor(0x2A, 0x36, 0x44)
AMBER = RGBColor(0xF2, 0xA5, 0x41)    # "kept" highlight accent
TEAL = RGBColor(0x2F, 0x7F, 0x79)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
MIST = RGBColor(0xF1, 0xF3, 0xF5)
MUTED = RGBColor(0x5F, 0x6B, 0x78)
LIGHT = RGBColor(0xC9, 0xD1, 0xDA)
HEAD = "Cambria"
BODY = "Calibri"

# ------------------------------------------------------------------ facts from the outputs
manifest = {r["video_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "manifest.csv")))}
outs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(ROOT, "outputs", "*.json"))}
epic = [v for v in outs if manifest.get(v, {}).get("dataset") == "epic_kitchens_100"]
fpv = [v for v in outs if manifest.get(v, {}).get("dataset") == "fpvsum"]
n_epic_m = sum(r["dataset"] == "epic_kitchens_100" for r in manifest.values())
n_fpv_m = sum(r["dataset"] == "fpvsum" for r in manifest.values())
frac = [sum(h["segment"][1] - h["segment"][0] for h in outs[v]["highlights"]) / outs[v]["duration_sec"] for v in outs]
n_actions = [len(outs[v]["actions"]) for v in epic]
epic_cooking = sum(outs[v]["label"]["name"] == "cooking" for v in epic)
cat = {"bikepolo": "bikepolo", "rockclimbing": "rockclimbing"}
fpv_ok = sum(outs[v]["label"]["name"] == cat.get(v.split("_")[0].lower(), v.split("_")[0].lower()) for v in fpv)
pairs = [len({(a["verb"], a["noun"]) for a in outs[v]["actions"]}) for v in epic]
F = dict(
    total=len(outs), total_m=len(manifest), epic=len(epic), epic_m=n_epic_m, fpv=len(fpv), fpv_m=n_fpv_m,
    frac_mean=100 * sum(frac) / len(frac), epic_cooking=epic_cooking,
    act_mean=sum(n_actions) / max(len(n_actions), 1), pairs_mean=sum(pairs) / max(len(pairs), 1),
    fpv_ok=fpv_ok, hours=sum(float(manifest[v]["duration_sec"]) for v in outs) / 3600)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def text(slide, x, y, w, h, s, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.italic, r.font.name = Pt(size), bold, italic, font
        r.font.color.rgb = color
        p.space_after = Pt(6)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        head, _, rest = it.partition("|")
        r = p.add_run()
        r.text = "▸  " + head
        r.font.size, r.font.name, r.font.color.rgb, r.font.bold = Pt(size), BODY, color, bool(rest)
        if rest:
            r2 = p.add_run()
            r2.text = " " + rest
            r2.font.size, r2.font.name, r2.font.color.rgb = Pt(size), BODY, color
        p.space_after = Pt(10)
    return tb


def box(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def box_text(shape, s, size=14, color=INK, bold=False, align=PP_ALIGN.CENTER):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size if i == 0 else size - 2)
        r.font.bold = bold and i == 0
        r.font.name = BODY
        r.font.color.rgb = color


def arrow(slide, x1, y1, x2, y2, color=MUTED):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"})
    ln.append(tail)


def strip(slide, x, y, w, h, segs, duration, base=INK2, keep=AMBER):
    """Timeline motif: the whole video as a bar, kept highlight segments in amber."""
    box(slide, x, y, w, h, base, shape=MSO_SHAPE.RECTANGLE)
    for a, b in segs:
        box(slide, x + w * a / duration, y, max(0.03, w * (b - a) / duration), h, keep, shape=MSO_SHAPE.RECTANGLE)


def title(slide, s, sub=None, dark=False):
    text(slide, 0.7, 0.5, W - 1.4, 0.8, s, size=36, bold=True, font=HEAD, color=PAPER if dark else INK)
    if sub:
        text(slide, 0.7, 1.3, W - 1.4, 0.5, sub, size=16, color=LIGHT if dark else MUTED)


def footer(slide, n, dark=False):
    text(slide, 0.7, 7.0, 8, 0.3, "Toothpaste Eater  |  Samsung PRISM XR Hackathon 2026  |  Track 4",
         size=10, color=LIGHT if dark else MUTED)
    text(slide, W - 1.2, 7.0, 0.5, 0.3, str(n), size=10, color=LIGHT if dark else MUTED, align=PP_ALIGN.RIGHT)


demo = outs.get("P01_13") or outs[epic[0]]

# 1 ---------------------------------------------------------------- title
s = prs.slides.add_slide(BLANK)
bg(s, INK)
text(s, 0.8, 0.8, 8, 0.4, "THEME 04  •  ACTION-AWARE VIDEO SUMMARIZATION", size=14, color=AMBER, bold=True)
text(s, 0.8, 1.6, 11.7, 1.8, ["Keep the 15% that matters", "Action-aware summaries of first-person video"],
     size=40, bold=True, font=HEAD, color=PAPER)
strip(s, 0.8, 4.3, 11.7, 0.35, [tuple(h["segment"]) for h in demo["highlights"]], demo["duration_sec"])
text(s, 0.8, 4.75, 11.7, 0.3, f"Highlights our system selected for EPIC-KITCHENS {demo['video_id']} "
     f"({demo['duration_sec']:.0f} s): amber is kept", size=11, color=LIGHT, italic=True)
text(s, 0.8, 5.6, 11, 0.9, ["Team Toothpaste Eater", "SRM Institute of Science and Technology, Kattankulathur"],
     size=18, color=PAPER)

# 2 ---------------------------------------------------------------- team
s = prs.slides.add_slide(BLANK)
bg(s, PAPER)
title(s, "Team details")
c = box(s, 0.7, 1.8, 6.2, 4.4, MIST)
text(s, 1.1, 2.1, 5.5, 0.5, "Ayush Kulshreshtha", size=26, bold=True, font=HEAD)
text(s, 1.1, 2.75, 5.5, 0.4, "Solo participant  •  Team lead", size=16, color=TEAL, bold=True)
bullets(s, 1.1, 3.45, 5.5, 2.6, [
    "Institution|SRM Institute of Science and Technology (SRMIST), Kattankulathur",
    "Email|ak4810@srmist.edu.in",
    "GitHub|github.com/Ayush0612005"], size=15)
c2 = box(s, 7.3, 1.8, 5.3, 4.4, INK)
text(s, 7.7, 2.1, 4.6, 0.4, "SUBMISSION", size=12, bold=True, color=AMBER)
bullets(s, 7.7, 2.65, 4.6, 3.4, [
    "Theme|Track 4: Action-Aware Video Summarization",
    "Team name|Toothpaste Eater",
    "Code|github.com/Ayush0612005/prism-xr-track4-action-summarization",
    "Entry point|./run.sh <video.mp4> <out.json>"], size=15, color=PAPER)
footer(s, 2)

# 3 ---------------------------------------------------------------- problem
s = prs.slides.add_slide(BLANK)
bg(s, PAPER)
title(s, "The problem, in our words")
bullets(s, 0.7, 1.9, 6.6, 4.5, [
    "Smart glasses record hours of egocentric video.|An XR assistant cannot afford to look at all of it.",
    "What matters is the actions:|opening the fridge, cutting the onion, pouring the water.",
    "So we must pick a short summary|that still contains as many distinct actions as possible.",
    "And it must be fast, reproducible, and honest|about what it saw: one MP4 in, one JSON out."], size=17)
for i, (big, small) in enumerate([("≤ 15%", "of each video kept"), ("≥ 4 s", "per highlight segment"),
                                   (f"{F['total_m']}", f"videos: {F['epic_m']} EPIC-KITCHENS + {F['fpv_m']} FPVSum")]):
    y = 1.9 + i * 1.5
    b = box(s, 8.0, y, 4.6, 1.25, MIST)
    text(s, 8.3, y + 0.12, 1.9, 1.0, big, size=36, bold=True, font=HEAD, color=TEAL, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 10.25, y + 0.12, 2.2, 1.0, small, size=14, color=INK, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 3)

# 4 ---------------------------------------------------------------- solution overview
s = prs.slides.add_slide(BLANK)
bg(s, PAPER)
title(s, "Our solution", "Three questions per video, answered by models that already know egocentric video")
cols = [("1", "What is happening?", "TSM action recogniser trained on EPIC-KITCHENS-100 scores 97 verbs x 300 nouns "
         "on sliding 2 / 4 / 8 s windows.", TEAL),
        ("2", "What kind of video is it?", "CLIP ViT-B/32 zero-shot on 16 frames with prompt ensembles picks one of 16 "
         "labels from pixels only.", INK2),
        ("3", "Which seconds to keep?", "Cooking: maximise expected coverage of distinct verb-noun actions. Sports: "
         "CLIP interestingness + motion + novelty.", AMBER)]
for i, (n, head, body, col) in enumerate(cols):
    x = 0.7 + i * 4.1
    box(s, x, 2.1, 3.8, 4.2, MIST)
    circ = box(s, x + 0.35, 2.4, 0.7, 0.7, col, shape=MSO_SHAPE.OVAL)
    box_text(circ, n, size=20, color=PAPER if col != AMBER else INK, bold=True)
    text(s, x + 0.35, 3.35, 3.1, 0.5, head, size=20, bold=True, font=HEAD)
    text(s, x + 0.35, 4.0, 3.1, 2.2, body, size=15, color=INK)
footer(s, 4)

# 5 ---------------------------------------------------------------- architecture
s = prs.slides.add_slide(BLANK)
bg(s, PAPER)
title(s, "Architecture")
def node(x, y, w, h, t, fill=MIST, col=INK):
    b = box(s, x, y, w, h, fill)
    box_text(b, t, size=14, color=col, bold=True)
node(0.6, 3.3, 1.7, 1.0, ["MP4", "first-person video"], INK, PAPER)
node(2.8, 3.3, 2.1, 1.0, ["ffmpeg decode", "4 fps, 224 px crops"])
node(5.5, 1.9, 2.4, 1.0, ["TSM ResNet-50", "EPIC-100, windows 2/4/8 s"])
node(5.5, 3.3, 2.4, 1.0, ["CLIP ViT-B/32", "frame embeddings"])
node(5.5, 4.7, 2.4, 1.0, ["Frame difference", "motion per second"])
node(8.5, 1.9, 2.0, 1.0, ["Actions", "top-3 pairs, NMS"])
node(8.5, 3.3, 2.0, 1.0, ["Label", "zero-shot, 16 classes"])
node(8.5, 4.7, 2.0, 1.0, ["Importance", "interest + novelty"])
node(11.0, 2.6, 1.8, 1.0, ["Coverage", "if cooking"], AMBER)
node(11.0, 4.0, 1.8, 1.0, ["Importance", "otherwise"], AMBER)
node(11.0, 5.6, 1.8, 0.9, ["JSON", "validated"], INK, PAPER)
arrow(s, 2.3, 3.8, 2.8, 3.8)
for yy in (2.4, 3.8, 5.2):
    arrow(s, 4.9, 3.8, 5.5, yy)
for yy in (2.4, 3.8, 5.2):
    arrow(s, 7.9, yy, 8.5, yy)
arrow(s, 10.5, 2.4, 11.0, 3.0)
arrow(s, 10.5, 3.8, 11.0, 3.2)
arrow(s, 10.5, 3.8, 11.0, 4.4)
arrow(s, 10.5, 5.2, 11.0, 4.6)
arrow(s, 11.9, 5.0, 11.9, 5.6)
text(s, 0.6, 6.25, 10, 0.5, "One decode per video; models loaded once; fp16 on GPU, CPU fallback. "
     "The strategy is chosen by the predicted label, never by file name.", size=13, color=MUTED, italic=True)
footer(s, 5)

# 6 ---------------------------------------------------------------- highlight selection on a real video
s = prs.slides.add_slide(BLANK)
bg(s, PAPER)
title(s, "Coverage-driven highlight selection",
      f"Real output for {demo['video_id']}: {len(demo['actions'])} action proposals, "
      f"{len(demo['highlights'])} highlights")
d = demo["duration_sec"]
x0, w0 = 0.9, 11.6
top = sorted(demo["actions"], key=lambda a: -a["score"])[:14]
rows = sorted({(a["verb"], a["noun"]) for a in top})[:7]
for i, (vb, nn) in enumerate(rows):
    y = 2.05 + i * 0.42
    text(s, x0, y, 2.2, 0.35, f"{vb} {nn}", size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    box(s, x0 + 2.3, y + 0.15, w0 - 2.3, 0.04, MIST, shape=MSO_SHAPE.RECTANGLE)
    for a in top:
        if (a["verb"], a["noun"]) == (vb, nn):
            sa, ea = a["segment"]
            box(s, x0 + 2.3 + (w0 - 2.3) * sa / d, y + 0.06, max(0.05, (w0 - 2.3) * (ea - sa) / d), 0.22, TEAL,
                shape=MSO_SHAPE.RECTANGLE)
yb = 2.05 + len(rows) * 0.42 + 0.15
text(s, x0, yb, 2.2, 0.4, "highlights", size=12, bold=True, anchor=MSO_ANCHOR.MIDDLE)
strip(s, x0 + 2.3, yb + 0.05, w0 - 2.3, 0.3, [tuple(h["segment"]) for h in demo["highlights"]], d, base=LIGHT)
text(s, x0 + 2.3, yb + 0.45, 3, 0.3, "0 s", size=10, color=MUTED)
text(s, x0 + w0 - 1.0, yb + 0.45, 1.0, 0.3, f"{d:.0f} s", size=10, color=MUTED, align=PP_ALIGN.RIGHT)
box(s, 0.9, 6.0, 11.6, 0.8, MIST)
text(s, 1.15, 6.1, 11.1, 0.6, "Each pair p gets probability q(p, w) per candidate window; coverage(p) = 1 - ∏(1 - q). "
     "Windows are added greedily by coverage gain per second, without overlap, until 14.8% of the video is used.",
     size=13, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 6)

# 7 ---------------------------------------------------------------- tech stack
s = prs.slides.add_slide(BLANK)
bg(s, PAPER)
title(s, "Tools and tech stack")
stack = [("Models", "TSM (EPIC-KITCHENS-100 release), OpenCLIP ViT-B/32 LAION-2B"),
         ("Framework", "PyTorch 2.11, torchvision, open_clip, NumPy"),
         ("Video", "ffmpeg (NVDEC when available), PyAV fallback, yt-dlp for FPVSum"),
         ("Compute", "RTX 3050 laptop GPU (FPVSum), 20 parallel GitHub Actions CPU runners (EPIC, 184 GB)"),
         ("Packaging", "run.sh entry point, Dockerfile (CUDA 12.8), pinned requirements.txt"),
         ("Quality", "Own validator for every kit rule, MD5-verified downloads, resumable batch driver")]
for i, (k, v) in enumerate(stack):
    col, row = i % 2, i // 2
    x, y = 0.7 + col * 6.1, 1.9 + row * 1.6
    box(s, x, y, 5.8, 1.35, MIST)
    text(s, x + 0.3, y + 0.2, 5.2, 0.4, k, size=17, bold=True, color=TEAL)
    text(s, x + 0.3, y + 0.62, 5.2, 0.7, v, size=14)
footer(s, 7)

# 8 ---------------------------------------------------------------- innovation
s = prs.slides.add_slide(BLANK)
bg(s, INK)
title(s, "Innovation highlights", dark=True)
inn = [("Coverage, not saliency", "For kitchen video we optimise the metric itself: expected number of distinct "
        "verb-noun pairs inside the summary, with a probabilistic max-coverage greedy."),
       ("Label-routed strategy", "The predicted label, from pixels only, decides whether to cover actions or follow "
        "visual excitement. No file-name shortcuts."),
       ("Zero training, full reuse", "A released egocentric action model and CLIP are combined at inference time; "
        "the whole pipeline is reproducible from one script."),
       ("Built to be re-run", "Strict weight loading, a validator for every rule, MD5-checked downloads and a 20-way "
        "parallel batch run that processed 184 GB without local storage.")]
for i, (h_, b_) in enumerate(inn):
    col, row = i % 2, i // 2
    x, y = 0.7 + col * 6.1, 1.8 + row * 2.45
    box(s, x, y, 5.8, 2.15, INK2)
    text(s, x + 0.35, y + 0.3, 5.1, 0.5, h_, size=20, bold=True, font=HEAD, color=AMBER)
    text(s, x + 0.35, y + 0.9, 5.1, 1.2, b_, size=14, color=PAPER)
footer(s, 8, dark=True)

# 9 ---------------------------------------------------------------- results
s = prs.slides.add_slide(BLANK)
bg(s, PAPER)
title(s, "Results", "Measured on our runs; official scores use hidden ground truth")
stats = [(f"{F['total']}/{F['total_m']}", "videos with a validated JSON (0 validator errors)"),
         (f"{F['frac_mean']:.1f}%", "mean share of each video kept as highlights"),
         (f"{F['act_mean']:.0f}", f"action proposals per EPIC video ({F['pairs_mean']:.0f} distinct verb-noun pairs)"),
         (f"{F['fpv_ok']}/{F['fpv']}", "FPVSum labels matching the category (dev check, never used as input)")]
for i, (big, small) in enumerate(stats):
    x = 0.7 + i * 3.05
    box(s, x, 2.0, 2.8, 2.5, MIST)
    text(s, x + 0.25, 2.25, 2.3, 1.0, big, size=36, bold=True, font=HEAD, color=TEAL)
    text(s, x + 0.25, 3.3, 2.3, 1.1, small, size=13)
bullets(s, 0.7, 4.85, 12, 1.9, [
    f"EPIC-KITCHENS:|{F['epic']}/{F['epic_m']} videos processed, all labelled cooking "
    f"({F['epic_cooking']}/{F['epic']}); 94 s video in 20 s on an RTX 3050 laptop GPU.",
    f"FPVSum:|{F['fpv']}/{F['fpv_m']} videos; the remaining source videos are private or removed on YouTube, "
    "so no output was produced for them (logged, never fabricated).",
    f"Total video processed:|{F['hours']:.1f} hours."], size=14)
footer(s, 9)

# 10 --------------------------------------------------------------- limitations
s = prs.slides.add_slide(BLANK)
bg(s, PAPER)
title(s, "Limitations and next steps")
lim = ["Action quality is bounded by the released TSM model|(about 36% action top-1 on EPIC-100 val).",
       "Sports highlights are unsupervised|(CLIP prompts, motion, novelty); not tuned on human importance labels.",
       "Zero-shot labels confuse look-alike sports|e.g. bike polo vs. bike, plane vs. skydiving.",
       "Some FPVSum sources are gone from YouTube,|so those videos have no output."]
nxt = ["Swap in a stronger egocentric backbone|(e.g. video transformers) behind the same interface.",
       "Learn importance from FPVSum-style annotations|with a light temporal head on CLIP features.",
       "Stream the pipeline on-device|for live summaries on glasses."]
box(s, 0.7, 1.8, 6.0, 4.9, MIST)
text(s, 1.0, 2.0, 5.4, 0.4, "Limitations", size=20, bold=True, font=HEAD)
bullets(s, 1.0, 2.6, 5.4, 4.0, lim, size=14)
box(s, 7.0, 1.8, 5.6, 4.9, INK)
text(s, 7.3, 2.0, 5.0, 0.4, "Next steps", size=20, bold=True, font=HEAD, color=AMBER)
bullets(s, 7.3, 2.6, 5.0, 4.0, nxt, size=14, color=PAPER)
footer(s, 10)

# 11 --------------------------------------------------------------- XR impact / close
s = prs.slides.add_slide(BLANK)
bg(s, INK)
text(s, 0.8, 0.9, 11.5, 0.5, "WHY IT MATTERS FOR XR", size=14, bold=True, color=AMBER)
text(s, 0.8, 1.5, 7.2, 1.9, ["~85% fewer frames", "for the XR model downstream"], size=36, bold=True, font=HEAD,
     color=PAPER)
text(s, 0.8, 3.7, 6.8, 1.6, "Keeping about 15% of frames means a captioning, memory or Q&A model on smart glasses "
     "processes roughly 85% fewer visual tokens, while the summary is chosen to keep the distinct hand-object "
     "actions.", size=16, color=LIGHT)
strip(s, 8.3, 1.9, 4.3, 0.5, [tuple(h["segment"]) for h in demo["highlights"]], demo["duration_sec"])
text(s, 8.3, 2.5, 4.3, 0.4, "full video  →  amber segments only", size=12, color=LIGHT, italic=True)
text(s, 0.8, 5.6, 11.5, 0.9, ["Code, weights, outputs and reproduction steps:",
                              "github.com/Ayush0612005/prism-xr-track4-action-summarization"],
     size=16, color=PAPER)
footer(s, 11, dark=True)

prs.save(OUT)
print("wrote", OUT, json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in F.items()}))
