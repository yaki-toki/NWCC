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


class SegmentHead(nn.Module):
    def __init__(self, feature_dim: int, use_input_ln: bool = False, p_context: float = 0.3,
                 p_mil: float = 0.5, legacy_aux_heads: int = 3):
        super().__init__()
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

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """features (B, S, D) -> video logits (B,), segment logits (B, S), segment scores (B, S)."""
        b, s, _ = features.shape
        context = self.context(features.reshape(b * s, -1)).reshape(b, s, HIDDEN)
        segment_logits = self.classifier(context).squeeze(-1)
        segment_scores = torch.sigmoid(segment_logits)
        weights = torch.softmax(self.attention(context).squeeze(-1), dim=1)
        video_logits = (weights * segment_logits).sum(dim=1)
        return video_logits, segment_logits, segment_scores

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
