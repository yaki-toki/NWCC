"""Segment pseudo-labels from the current model (self-training of the baseline).

For every training video the model scores its segments (eval mode, mixed precision on
CUDA as in the earlier implementation):

* anomalous video: the ``top_k`` highest-scoring segments with score >= ``confidence_min``
  become positives (1.0); if any was labeled, the ``pseudo_neg_k`` lowest-scoring
  segments become negatives (0.0) unless already positive;
* normal video: with ``top_k_neg`` > 0 its ``top_k_neg`` highest-scoring segments become
  negatives, otherwise it gets no pseudo-labels (``None``).

Unlabeled segments are NaN.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.amp import autocast


@dataclass(frozen=True)
class PseudoLabelStats:
    positive_videos: int
    positive_segments: int
    mined_negative_segments: int
    normal_videos: int
    normal_negative_segments: int


def generate_pseudo_labels(model: torch.nn.Module, features: Sequence[torch.Tensor],
                           labels: Sequence[int], device: torch.device, top_k: int,
                           confidence_min: float, use_amp: bool, top_k_neg: int = 0,
                           pseudo_neg_k: int = 0
                           ) -> tuple[list[torch.Tensor | None], PseudoLabelStats]:
    model.eval()
    out: list[torch.Tensor | None] = []
    pos_videos = pos_segs = mined = neg_videos = neg_segs = 0
    with torch.no_grad():
        for feats, label in zip(features, labels, strict=True):
            x = feats.float().unsqueeze(0).to(device)
            with autocast(device_type=device.type, enabled=use_amp):
                _, _, seg_scores = model(x)
            scores = seg_scores[0].float().cpu()
            if label == 1:
                pseudo = torch.full((len(scores),), float("nan"))
                values, idx = scores.topk(min(top_k, len(scores)))
                assigned = 0
                for v, i in zip(values, idx, strict=True):
                    if v.item() >= confidence_min:
                        pseudo[i] = 1.0
                        assigned += 1
                if assigned > 0:
                    pos_videos += 1
                    pos_segs += assigned
                if pseudo_neg_k > 0 and assigned > 0:
                    _, bottom = scores.topk(min(pseudo_neg_k, len(scores)), largest=False)
                    for i in bottom:
                        if torch.isnan(pseudo[i]):
                            pseudo[i] = 0.0
                            mined += 1
                out.append(pseudo)
            elif top_k_neg > 0:
                k = min(top_k_neg, len(scores))
                _, idx = scores.topk(k)
                pseudo = torch.full((len(scores),), float("nan"))
                for i in idx:
                    pseudo[i] = 0.0
                neg_videos += 1
                neg_segs += k
                out.append(pseudo)
            else:
                out.append(None)
    return out, PseudoLabelStats(pos_videos, pos_segs, mined, neg_videos, neg_segs)
