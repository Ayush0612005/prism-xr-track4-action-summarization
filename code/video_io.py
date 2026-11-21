"""Video probing and decoding to a fixed-rate stack of 224x224 RGB crops."""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np

CROP = 224
SHORT_SIDE = 256


def _ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe_duration(path: str) -> float:
    """Container duration in seconds (ffprobe if present, else PyAV)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                capture_output=True, text=True, check=True).stdout
            return float(json.loads(out)["format"]["duration"])
        except Exception:
            pass
    import av
    with av.open(path) as c:
        if c.duration:
            return c.duration / 1e6
        s = c.streams.video[0]
        return float(s.duration * s.time_base)


def _vf(fps: float) -> str:
    # scale the short side to 256 (portrait or landscape), then centre-crop 224
    return (f"fps={fps},scale='if(gt(iw,ih),-2,{SHORT_SIDE})':'if(gt(iw,ih),{SHORT_SIDE},-2)',"
            f"crop={CROP}:{CROP}")


def decode_ffmpeg(path: str, fps: float, hwaccel: bool) -> np.ndarray:
    exe = _ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg not found")
    cmd = [exe, "-v", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", path, "-an", "-sn", "-vf", _vf(fps), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = CROP * CROP * 3
    chunks = []
    while True:  # stream so RAM holds only the small decoded crops
        buf = proc.stdout.read(frame_bytes * 256)
        if not buf:
            break
        chunks.append(buf)
    err = proc.stderr.read().decode(errors="replace")
    proc.wait()
    data = b"".join(chunks)
    n = len(data) // frame_bytes
    if proc.returncode != 0 or n == 0:
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {err[-500:]}")
    return np.frombuffer(data[: n * frame_bytes], dtype=np.uint8).reshape(n, CROP, CROP, 3)


def decode_pyav(path: str, fps: float) -> np.ndarray:
    import av
    frames, next_t = [], 0.0
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for fr in c.decode(s):
            if fr.time is None or fr.time + 1e-6 < next_t:
                continue
            w, h = fr.width, fr.height
            if w >= h:
                nw, nh = max(CROP, round(w * SHORT_SIDE / h / 2) * 2), SHORT_SIDE
            else:
                nw, nh = SHORT_SIDE, max(CROP, round(h * SHORT_SIDE / w / 2) * 2)
            img = fr.to_ndarray(width=nw, height=nh, format="rgb24")
            y0, x0 = (nh - CROP) // 2, (nw - CROP) // 2
            frames.append(img[y0:y0 + CROP, x0:x0 + CROP])
            next_t += 1.0 / fps
            while fr.time + 1e-6 >= next_t:  # skip if the source rate is below `fps`
                frames.append(frames[-1])
                next_t += 1.0 / fps
    if not frames:
        raise RuntimeError("PyAV decoded no frames")
    return np.stack(frames)


def decode(path: str, fps: float = 4.0) -> np.ndarray:
    """Returns uint8 array (N, 224, 224, 3), frame i at time i/fps."""
    errors = []
    for hw in (True, False):
        try:
            return decode_ffmpeg(path, fps, hw)
        except Exception as e:  # NVDEC unavailable or ffmpeg missing
            errors.append(str(e))
    try:
        return decode_pyav(path, fps)
    except Exception as e:
        errors.append(str(e))
    raise RuntimeError("could not decode video: " + " | ".join(errors))
