"""Segment-level MIL detection head on frozen features (the baseline detector).

    context     [LayerNorm(D)] -> Linear(D, 256) -> ReLU -> Dropout(p_context)
    classifier  Dropout(p_mil) -> Linear(256, 128) -> ReLU -> Dropout(p_mil / 2) -> Linear(128, 1)
    attention   Linear(256, 128) -> Tanh -> Linear(128, 1)

Each segment is scored from its own features only (no temporal context); the attention
pooling forms the video logit used by the training loss and is not part of the
segment scores.

This is the detector of the earlier method without its auxiliary heads. For exact
reproduction of earlier runs, the constructor consumes the random stream as the earlier
implementation did: that implementation created ``legacy_aux_heads`` auxiliary heads
(Linear(256, 128) and Linear(128, 2) each) between the context and the classifier.
Layers of the same shapes are created and discarded at that point, so a given seed yields
the same initial weights. Set ``legacy_aux_heads=0`` to drop this once exact
reproduction no longer matters.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

HIDDEN = 256
CLASSIFIER_HIDDEN = 128
LEGACY_PREFIXES = {"shared_temporal_context.": "context.",
                   "mil_segment_classifier.": "classifier.",
                   "mil_attention.": "attention."}
LEGACY_DROPPED = ("anticipation_heads.",)
GLOBAL_MODES = ("none", "mean", "stats")


class SegmentHead(nn.Module):
    def __init__(self, feature_dim: int, use_input_ln: bool = False, p_context: float = 0.3,
                 p_mil: float = 0.5, legacy_aux_heads: int = 3, global_mode: str = "none"):
        super().__init__()
        if global_mode not in GLOBAL_MODES:
            raise ValueError(f"global_mode {global_mode!r} not in {GLOBAL_MODES}")
        self.global_mode = global_mode
        layers: list[nn.Module] = [nn.LayerNorm(feature_dim)] if use_input_ln else []
        layers += [nn.Linear(feature_dim, HIDDEN), nn.ReLU(), nn.Dropout(p_context)]
        self.context = nn.Sequential(*layers)
        for _ in range(legacy_aux_heads):          # random-stream parity only, discarded
            nn.Linear(HIDDEN, CLASSIFIER_HIDDEN)
            nn.Linear(CLASSIFIER_HIDDEN, 2)
        self.classifier = nn.Sequential(
            nn.Dropout(p_mil), nn.Linear(HIDDEN, CLASSIFIER_HIDDEN), nn.ReLU(),
            nn.Dropout(p_mil / 2), nn.Linear(CLASSIFIER_HIDDEN, 1))
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN, CLASSIFIER_HIDDEN), nn.Tanh(), nn.Linear(CLASSIFIER_HIDDEN, 1))
        # Global offset (stage D, E016), created last and from zeros: no random draws, so the
        # other layers' initial weights and the data order equal the baseline's, and the
        # initial output is unchanged. Names differ per mode (strict state-dict loading).
        # Registered on the module itself, they come first in parameters() / state_dict().
        if global_mode == "mean":
            self.offset_gamma = nn.Parameter(torch.zeros(()))
        elif global_mode == "stats":
            self.offset_w = nn.Parameter(torch.zeros(2 * HIDDEN))
            self.offset_b = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """features (B, S, D) -> video logits (B,), segment logits (B, S), segment scores (B, S).

        With a global mode the segment logits are z_t + c(v) and the scores sigma(z_t + c(v));
        the video logit stays the attention pooling of the offset-free z_t."""
        if self.global_mode != "none":
            video_logits, local_logits, offset = self._parts(features)
            segment_logits = local_logits + offset.to(local_logits.dtype)[:, None]
            return video_logits, segment_logits, torch.sigmoid(segment_logits)
        b, s, _ = features.shape
        context = self.context(features.reshape(b * s, -1)).reshape(b, s, HIDDEN)
        segment_logits = self.classifier(context).squeeze(-1)
        segment_scores = torch.sigmoid(segment_logits)
        weights = torch.softmax(self.attention(context).squeeze(-1), dim=1)
        video_logits = (weights * segment_logits).sum(dim=1)
        return video_logits, segment_logits, segment_scores

    def _parts(self, features: torch.Tensor):
        """Video logits, offset-free segment logits z (B, S) and the offset c(v) (B,), fp32."""
        b, s, _ = features.shape
        flat = features.reshape(b * s, -1)
        pre = self.context[:-1](flat)            # same modules and order as self.context
        context = self.context[-1](pre).reshape(b, s, HIDDEN)
        local_logits = self.classifier(context).squeeze(-1)
        weights = torch.softmax(self.attention(context).squeeze(-1), dim=1)
        video_logits = (weights * local_logits).sum(dim=1)
        with torch.autocast(device_type=features.device.type, enabled=False):
            if self.global_mode == "mean":
                offset = self.offset_gamma * local_logits.float().mean(dim=1)
            else:                                # statistics before dropout, per video
                p = pre.reshape(b, s, HIDDEN).float()
                stats = torch.cat([p.mean(dim=1), torch.std(p, dim=1, correction=0)], dim=1)
                offset = stats @ self.offset_w + self.offset_b
        return video_logits, local_logits, offset

    def offset_parts(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Offset c(v) (B,) and offset-free scores sigma(z) (B, S) for the validation record
        (call in eval mode within the validation loop: no random draws)."""
        if self.global_mode == "none":
            raise ValueError("offset_parts needs a global mode")
        _, local_logits, offset = self._parts(features)
        return offset.float(), torch.sigmoid(local_logits.float())

    @staticmethod
    def convert_legacy_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rename an earlier checkpoint's detector weights; auxiliary heads are dropped."""
        out = {}
        for key, value in state.items():
            if key.startswith(LEGACY_DROPPED):
                continue
            for old, new in LEGACY_PREFIXES.items():
                if key.startswith(old):
                    out[new + key[len(old):]] = value
                    break
            else:
                raise KeyError(f"unexpected key in an earlier checkpoint: {key}")
        return out

    def load_legacy_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.load_state_dict(self.convert_legacy_state_dict(state), strict=True)
