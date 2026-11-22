#!/usr/bin/env python3
"""Assemble the <=3 min demo video from real material: deck slides (PDF), the real run.sh/Docker log,
and side-by-side clips of real videos vs. the highlights our system selected. Narration uses the offline
Windows speech synthesizer (System.Speech), so no text leaves the machine.

usage: python docs/demo/make_demo.py --deck-pdf docs/SRM_ToothpasteEater.pdf --work D:/xrdata/demo
         --epic-video <P01_13.MP4> --ski-video <Ski_-Z5a4UQmZLs.mp4> --run-log run_log.txt
         --validate-log validate_log.txt --out demo.mp4
"""
import argparse
import json
import os
import subprocess
import wave

import imageio_ffmpeg
import pypdfium2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
INK, INK2, AMBER, PAPER, LIGHT = (22, 32, 42), (42, 54, 68), (242, 165, 65), (255, 255, 255), (201, 209, 218)
FONT_DIR = "C:/Windows/Fonts"

NARRATION = {
    "title": "Hi, we are team Toothpaste Eater, from S R M Institute of Science and Technology. This is our Track 4 "
             "submission: action-aware summarization of first-person video.",
    "problem": "Smart glasses record hours of egocentric video, but an XR assistant cannot look at all of it. What "
               "matters are the actions, like opening a cupboard, washing a cup, or pouring water. So we keep at most "
               "fifteen percent of each video, in segments of at least four seconds, while keeping as many distinct "
               "actions as possible.",
    "arch": "Each video is decoded once, at four frames per second. A temporal shift network trained on EPIC KITCHENS "
            "slides two, four and eight second windows over the video, and scores ninety seven verbs and three "
            "hundred nouns. CLIP predicts the video category zero-shot, from pixels only. The predicted category "
            "picks the highlight strategy: action coverage for cooking, and visual interest and motion for sports.",
    "run": "Running it is one command. Here is our Docker image on a ninety-four second kitchen video, with the "
           "network switched off. It finishes in under a minute on a laptop GPU, and the output passes our "
           "validator, which checks every rule in the starter kit.",
    "sbs1": "On the left, the full kitchen video, sped up. On the right, only the twelve seconds our system kept: "
            "the windows that cover the most distinct actions the model detected.",
    "coverage": "For cooking, choosing highlights is a weighted max-coverage problem. Each verb-noun pair gets a "
                "probability of appearing in each candidate window, and windows are added greedily by coverage gain "
                "per second, until fourteen point eight percent of the video is used.",
    "sbs2": "For a GoPro ski video, the category is predicted as ski, so the importance strategy keeps the stretches "
            "with the most visual interest, motion and novelty.",
    "results": "We produced validated outputs for one hundred eighty-five of the one hundred eighty-seven videos: all "
               "one hundred thirty-eight EPIC KITCHENS videos, and forty-seven FPVSum videos. The other two are no "
               "longer on YouTube, so we did not produce outputs for them. Fifteen and a half hours of video were "
               "processed, with zero validator errors.",
    "xr": "Keeping about fifteen percent of frames means a downstream XR model processes roughly eighty-five percent "
          "fewer visual tokens, while the summary still keeps the distinct hand-object actions. Code, weights and "
          "outputs are on GitHub. Thank you.",
}


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def tts(text, wav):
    ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$s.SelectVoice('Microsoft David Desktop'); $s.Rate=0; "
          f"$s.SetOutputToWaveFile('{wav}'); $s.Speak([IO.File]::ReadAllText('{wav}.txt')); $s.Dispose()")
    open(wav + ".txt", "w", encoding="utf-8").write(text)
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    with wave.open(wav) as w:
        return w.getnframes() / w.getframerate()


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def slide_images(pdf, work):
    doc = pypdfium2.PdfDocument(pdf)
    out = []
    for i in range(len(doc)):
        img = doc[i].render(scale=W / doc[i].get_width()).to_pil().convert("RGB").resize((W, H))
        p = os.path.join(work, f"slide{i + 1:02d}.png")
        img.save(p)
        out.append(p)
    return out


def terminal_frames(run_log, validate_log, out_json, work):
    cmd1 = "$ docker run --rm --gpus all --network none -v ./videos:/data track4 /data/P01_13.MP4 /data/P01_13.json"
    cmd2 = "$ python tools/validate.py --file videos/P01_13.json -v"
    d = json.load(open(out_json))
    snippet = [f'  "label": {json.dumps(d["label"])},',
               f'  "actions": [{json.dumps(d["actions"][0])}, ... {len(d["actions"])} items],',
               f'  "highlights": {json.dumps(d["highlights"])}']
    rl = open(run_log, encoding="utf-8-sig").read().splitlines()
    vl = open(validate_log, encoding="utf-8-sig").read().splitlines()
    stages = [[cmd1], [cmd1] + rl,
              [cmd1] + rl + ["", cmd2] + vl
              + ["", "$ cat videos/P01_13.json", "{"] + snippet + ["}"]]
    mono, head = font("consola.ttf", 26), font("calibrib.ttf", 34)
    paths = []
    for k, lines in enumerate(stages):
        img = Image.new("RGB", (W, H), INK)
        dr = ImageDraw.Draw(img)
        dr.text((90, 60), "./run.sh in our Docker image (real output, network disabled)", font=head, fill=AMBER)
        dr.rounded_rectangle((70, 130, W - 70, H - 70), radius=18, fill=(12, 16, 22))
        y = 165
        for ln in lines:
            for chunk in [ln[i:i + 118] for i in range(0, max(len(ln), 1), 118)]:
                col = AMBER if chunk.startswith("$") else (PAPER if "error" not in chunk else (140, 220, 160))
                dr.text((105, y), chunk, font=mono, fill=col)
                y += 38
        p = os.path.join(work, f"term{k}.png")
        img.save(p)
        paths.append(p)
    return paths


def sbs_background(title, left_cap, right_cap, work, name):
    img = Image.new("RGB", (W, H), INK)
    dr = ImageDraw.Draw(img)
    dr.text((90, 70), title, font=font("cambriab.ttf", 54), fill=PAPER)
    dr.text((90, 250), left_cap, font=font("calibri.ttf", 32), fill=LIGHT)
    dr.text((990, 250), right_cap, font=font("calibrib.ttf", 32), fill=AMBER)
    p = os.path.join(work, name)
    img.save(p)
    return p


def still_scene(img, dur, wav, out):
    run([FF, "-y", "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.2f}", "-i", img, "-i", wav,
         "-filter_complex", f"[1:a]apad,atrim=0:{dur:.2f},aresample=48000[a]", "-map", "0:v", "-map", "[a]",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "160k", "-ac", "2", out])


def multi_still_scene(imgs, dur, wav, out, work):
    """Show imgs one after another (terminal 'typing' effect), last one held."""
    seg = [0.12 * dur, 0.30 * dur, 0.58 * dur]
    lst = os.path.join(work, "term_list.txt")
    with open(lst, "w") as f:
        for p, s in zip(imgs, seg):
            f.write(f"file '{p.replace(os.sep, '/')}'\nduration {s:.2f}\n")
        f.write(f"file '{imgs[-1].replace(os.sep, '/')}'\n")
    run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
         "-filter_complex", f"[0:v]fps={FPS},format=yuv420p[v];[1:a]apad,atrim=0:{dur:.2f},aresample=48000[a]",
         "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", "-ac", "2", out])


def sbs_scene(video, segs, duration, bg, dur, wav, out):
    """Left: full video sped up to the reel length. Right: the selected highlight segments, concatenated."""
    reel = sum(b - a for a, b in segs)
    speed = duration / reel
    n = len(segs)
    parts = [f"[0:v]split={n + 1}[full]" + "".join(f"[s{i}]" for i in range(n))]
    parts.append(f"[full]setpts=PTS/{speed:.4f},fps={FPS},scale=860:484:force_original_aspect_ratio=decrease,"
                 f"pad=860:484:(ow-iw)/2:(oh-ih)/2[left]")
    for i, (a, b) in enumerate(segs):
        parts.append(f"[s{i}]trim=start={a}:end={b},setpts=PTS-STARTPTS,fps={FPS},"
                     f"scale=860:484:force_original_aspect_ratio=decrease,pad=860:484:(ow-iw)/2:(oh-ih)/2[r{i}]")
    parts.append("".join(f"[r{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[right]")
    parts.append(f"[1:v]fps={FPS}[bg];[bg][left]overlay=90:320:shortest=0[t];[t][right]overlay=990:320,"
                 f"tpad=stop_mode=clone:stop_duration=30,trim=0:{dur:.2f},format=yuv420p[v]")
    parts.append(f"[2:a]apad,atrim=0:{dur:.2f},aresample=48000[a]")
    run([FF, "-y", "-i", video, "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.2f}", "-i", bg, "-i", wav,
         "-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "160k", "-ac", "2", out])
    return reel, speed


def main():
    ap = argparse.ArgumentParser()
    for k in ("deck-pdf", "work", "epic-video", "ski-video", "run-log", "validate-log", "out", "epic-json"):
        ap.add_argument("--" + k, required=k != "epic-json")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    slides = slide_images(a.deck_pdf, a.work)
    epic_json = a.epic_json or os.path.join(ROOT, "outputs", "P01_13.json")
    ski = json.load(open(os.path.join(ROOT, "outputs", "Ski_-Z5a4UQmZLs.json")))
    epic = json.load(open(epic_json))

    durs = {}
    for k, t in NARRATION.items():
        durs[k] = tts(t, os.path.join(a.work, f"{k}.wav"))
    wav = lambda k: os.path.join(a.work, f"{k}.wav")  # noqa: E731
    pad = 0.7
    scenes = []
    for key, slide in (("title", 0), ("problem", 2), ("arch", 4)):
        p = os.path.join(a.work, f"sc_{key}.mp4")
        still_scene(slides[slide], durs[key] + pad, wav(key), p)
        scenes.append(p)
    term = terminal_frames(a.run_log, a.validate_log, epic_json, a.work)
    p = os.path.join(a.work, "sc_run.mp4")
    multi_still_scene(term, durs["run"] + pad, wav("run"), p, a.work)
    scenes.append(p)

    e_segs = [tuple(h["segment"]) for h in epic["highlights"]]
    e_reel = sum(b - c for c, b in e_segs)
    bg = sbs_background("EPIC-KITCHENS P01_13: coverage strategy",
                        f"Full video, {epic['duration_sec']:.0f} s (sped up {epic['duration_sec'] / e_reel:.1f}x)",
                        f"Our highlights: {e_reel:.0f} s ({100 * e_reel / epic['duration_sec']:.1f}%)", a.work, "bg1.png")
    p = os.path.join(a.work, "sc_sbs1.mp4")
    sbs_scene(a.epic_video, e_segs, epic["duration_sec"], bg, max(durs["sbs1"], e_reel) + pad, wav("sbs1"), p)
    scenes.append(p)

    p = os.path.join(a.work, "sc_coverage.mp4")
    still_scene(slides[5], durs["coverage"] + pad, wav("coverage"), p)
    scenes.append(p)

    s_segs = [tuple(h["segment"]) for h in ski["highlights"]]
    s_reel = sum(b - c for c, b in s_segs)
    bg = sbs_background(f"FPVSum ski video: predicted label '{ski['label']['name']}', importance strategy",
                        f"Full video, {ski['duration_sec']:.0f} s (sped up {ski['duration_sec'] / s_reel:.1f}x)",
                        f"Our highlights: {s_reel:.0f} s ({100 * s_reel / ski['duration_sec']:.1f}%)", a.work, "bg2.png")
    p = os.path.join(a.work, "sc_sbs2.mp4")
    sbs_scene(a.ski_video, s_segs, ski["duration_sec"], bg, max(durs["sbs2"], s_reel) + pad, wav("sbs2"), p)
    scenes.append(p)

    for key, slide in (("results", 8), ("xr", 10)):
        p = os.path.join(a.work, f"sc_{key}.mp4")
        still_scene(slides[slide], durs[key] + pad, wav(key), p)
        scenes.append(p)

    lst = os.path.join(a.work, "scenes.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{s.replace(os.sep, '/')}'\n" for s in scenes)
    run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", a.out])
    total = sum(durs.values()) + pad * len(durs) + max(0, e_reel - durs["sbs1"]) + max(0, s_reel - durs["sbs2"])
    print(f"wrote {a.out}  (~{total:.0f} s)  narration per scene: " +
          ", ".join(f"{k}={v:.1f}" for k, v in durs.items()))


if __name__ == "__main__":
    main()
