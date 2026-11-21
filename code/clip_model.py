"""CLIP zero-shot video label and per-frame interestingness signals (open_clip)."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Prompt ensembles per label (names must match supported_classes.json "labels").
LABEL_PROMPTS = {
    "bike": ["a first-person video of riding a mountain bike on a trail",
             "a helmet camera view while cycling downhill", "a GoPro video from a bicycle"],
    "bikepolo": ["a first-person video of playing bike polo on a court",
                 "a helmet camera view of bicycle polo with mallets and a ball"],
    "boxing": ["a first-person video of boxing in a ring", "a head camera view of a boxer sparring with gloves"],
    "cooking": ["a first-person video of cooking in a kitchen", "a head-mounted camera view of hands preparing food",
                "an egocentric video of washing dishes at a kitchen sink"],
    "horse": ["a first-person video of riding a horse", "a helmet camera view of horseback riding"],
    "jumping": ["a first-person video of skydiving or base jumping", "a GoPro view of free falling from a plane",
                "a first-person video of jumping off a cliff"],
    "longboarding": ["a first-person video of longboarding down a road", "a GoPro view of downhill skateboarding on asphalt"],
    "motor": ["a first-person video of riding a motorcycle", "a helmet camera view of motocross or a motorbike race"],
    "parkour": ["a first-person video of parkour running and jumping over rooftops",
                "a head camera view of free running in a city"],
    "plane": ["a first-person video from the cockpit of a small airplane", "a view of flying a plane"],
    "rockclimbing": ["a first-person video of rock climbing a cliff", "a helmet camera view of climbing a rock wall"],
    "scuba": ["a first-person video of scuba diving underwater", "an underwater GoPro view of a diver and fish"],
    "skate": ["a first-person video of skateboarding in a skate park", "a GoPro view of skateboard tricks"],
    "ski": ["a first-person video of skiing down a snowy slope", "a helmet camera view of snowboarding in the mountains"],
    "surfing": ["a first-person video of surfing a wave", "a GoPro view from a surfboard in the ocean"],
    "other": ["a first-person video", "a video of everyday life"],
}
EXCITING = ["an exciting moment", "a dramatic action moment", "a jump or a fast turn", "an impressive view",
            "a crash or a near miss", "a person doing something interesting"]
BORING = ["a boring idle moment", "a blurry dark frame", "a static view with nothing happening",
          "the camera pointing at the ground"]


class ClipScorer:
    def __init__(self, device: str, arch: str | None = None, pretrained: str | None = None,
                 cache_dir: str | None = None):
        import open_clip
        arch = arch or os.environ.get("CLIP_ARCH", "ViT-B-32")
        pretrained = pretrained or os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")
        self.model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, cache_dir=cache_dir)
        self.model = self.model.eval().to(device)
        self.tok = open_clip.get_tokenizer(arch)
        self.device = device
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def text(self, prompts: list[str]) -> torch.Tensor:
        t = self.model.encode_text(self.tok(prompts).to(self.device)).float()
        return F.normalize(t, dim=-1)

    @torch.no_grad()
    def images(self, frames: np.ndarray, batch: int = 128) -> torch.Tensor:
        """frames uint8 (N,224,224,3) -> L2-normalised embeddings (N, D) on device."""
        out = []
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i:i + batch]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std
            with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
                e = self.model.encode_image(x)
            out.append(F.normalize(e.float(), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def label(self, emb: torch.Tensor, labels: list[str], n_frames: int = 16):
        idx = np.linspace(0, len(emb) - 1, min(n_frames, len(emb))).round().astype(int)
        v = F.normalize(emb[idx].mean(0, keepdim=True), dim=-1)
        protos = torch.stack([F.normalize(self.text(LABEL_PROMPTS[l]).mean(0), dim=-1) for l in labels])
        logits = 100.0 * v @ protos.T
        probs = logits.softmax(-1)[0].cpu().numpy()
        k = int(probs.argmax())
        return labels[k], float(probs[k])

    @torch.no_grad()
    def interestingness(self, emb: torch.Tensor, label: str) -> np.ndarray:
        """Per-frame similarity to 'exciting' prompts minus 'boring' prompts."""
        ex = self.text([f"{p}, {LABEL_PROMPTS[label][0]}" for p in EXCITING]).mean(0)
        bo = self.text(BORING).mean(0)
        return ((emb @ ex) - (emb @ bo)).cpu().numpy()


def novelty(emb: torch.Tensor, frames_per_sec: int, window_sec: int = 10) -> np.ndarray:
    """1 - cosine similarity between a frame and the mean of the preceding window."""
    e = emb.float()
    w = max(1, frames_per_sec * window_sec)
    c = torch.cumsum(torch.cat([torch.zeros_like(e[:1]), e]), 0)
    idx = torch.arange(len(e), device=e.device)
    lo = (idx - w).clamp(min=0)
    cnt = (idx - lo).clamp(min=1).unsqueeze(1).float()
    prev = (c[idx] - c[lo]) / cnt
    prev[0] = e[0]
    return (1.0 - F.cosine_similarity(e, prev, dim=-1)).cpu().numpy()
