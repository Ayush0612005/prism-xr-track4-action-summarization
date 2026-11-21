"""Temporal Shift Module (TSM) action recogniser for EPIC-KITCHENS-100.

A small re-implementation of the architecture used by the official EPIC-KITCHENS-100
TSM RGB baseline (Lin et al., "TSM: Temporal Shift Module for Efficient Video
Understanding", ICCV 2019; checkpoint released by the EPIC-KITCHENS team in
github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM). Written from the
paper and the checkpoint layout so that the released weights load strictly.

Layout: torchvision ResNet-50, every bottleneck's conv1 wrapped by a temporal shift
(residual "blockres" placement, 8 segments, 1/8 channels shifted each direction),
the ImageNet fc replaced by dropout, a new 2048 -> 397 linear head (97 verbs then
300 nouns) and average consensus over the segments.
"""
from __future__ import annotations

import torch
from torch import nn
import torchvision

NUM_VERBS = 97
NUM_NOUNS = 300
NUM_SEGMENTS = 8

# Normalisation exactly as in the released config (the std is the ImageNet mean;
# this is an upstream quirk, and the weights were trained with it).
INPUT_MEAN = (0.485, 0.456, 0.406)
INPUT_STD = (0.485, 0.456, 0.406)


def temporal_shift(x: torch.Tensor, n_segment: int, fold_div: int) -> torch.Tensor:
    """Shift 1/fold_div of the channels one step back in time and 1/fold_div forward."""
    nt, c, h, w = x.shape
    x = x.view(nt // n_segment, n_segment, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]                  # take from the future
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # take from the past
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]             # untouched
    return out.view(nt, c, h, w)


class TemporalShift(nn.Module):
    def __init__(self, net: nn.Module, n_segment: int = NUM_SEGMENTS, fold_div: int = 8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        return self.net(temporal_shift(x, self.n_segment, self.fold_div))


class TSM(nn.Module):
    def __init__(self, n_segment: int = NUM_SEGMENTS):
        super().__init__()
        self.n_segment = n_segment
        base = torchvision.models.resnet50(weights=None)
        for layer in (base.layer1, base.layer2, base.layer3, base.layer4):
            for block in layer:
                block.conv1 = TemporalShift(block.conv1, n_segment, 8)
        base.fc = nn.Dropout(p=0.5)
        self.base_model = base
        self.new_fc = nn.Linear(2048, NUM_VERBS + NUM_NOUNS)

    def forward(self, x: torch.Tensor):
        """x: (N, T, 3, H, W) normalised float. Returns (verb_logits, noun_logits)."""
        n, t = x.shape[:2]
        feats = self.base_model(x.flatten(0, 1))
        logits = self.new_fc(feats).view(n, t, -1).mean(dim=1)
        return logits[:, :NUM_VERBS], logits[:, NUM_VERBS:]


def _strip_prefix(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        out[k[len("model."):] if k.startswith("model.") else k] = v
    return out


def load_tsm(path: str, device: str = "cpu") -> TSM:
    """Load either the original Lightning checkpoint or our fp16 state dict. Strict."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    state = _strip_prefix(state)
    model = TSM()
    expected = model.state_dict()
    missing = [k for k in expected if k not in state and not k.endswith("num_batches_tracked")]
    unexpected = [k for k in state if k not in expected]
    if missing or unexpected:
        raise RuntimeError(f"TSM checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    for k, v in state.items():
        if tuple(v.shape) != tuple(expected[k].shape):
            raise RuntimeError(f"TSM shape mismatch at {k}: {tuple(v.shape)} vs {tuple(expected[k].shape)}")
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    return model.eval().to(device)
