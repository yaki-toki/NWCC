"""Training losses of the baseline (weak video labels only).

The functions reproduce the earlier implementation operation by operation, including
the summation order, so that earlier runs are reproduced exactly; vectorized rewrites
are deliberate design changes for later stages.

    video_bce            BCE on the attention-pooled video logit
    max_ranking_loss     hinge between the top segment of every anomalous video and the
                         top segment of every normal video (pairs averaged)
    dense_ranking_loss   hinge between the top segment of every anomalous video and every
                         segment of every normal video ("dense normal-to-anomaly ranking";
                         called TC-MIL earlier, it uses no temporal order)
    sparsity_loss        |mean anomalous-video segment score - target|
    smoothness_loss      mean squared difference of adjacent segment scores
    pseudo_label_bce     BCE on segments carrying pseudo-labels of one value
    normal_segment_bce   BCE (target 0) on every segment of every normal video: segments of
                         a normal video are sure negatives under weak labels (stage D1, E011)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def video_bce(video_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(video_logits, labels)


def max_ranking_loss(anomaly_scores: torch.Tensor, normal_scores: torch.Tensor,
                     margin: float = 0.1) -> torch.Tensor:
    anomaly_max = anomaly_scores.max(dim=1)[0]
    normal_max = normal_scores.max(dim=1)[0]
    loss = torch.tensor(0.0, device=anomaly_scores.device)
    count = 0
    for a_score in anomaly_max:                    # pairwise loop: summation order as before
        for n_score in normal_max:
            loss = loss + F.relu(margin - (a_score - n_score))
            count += 1
    if count > 0:
        loss = loss / count
    return loss


def dense_ranking_loss(anomaly_scores: torch.Tensor, normal_scores: torch.Tensor,
                       margin: float = 0.5) -> torch.Tensor:
    if anomaly_scores.shape[0] == 0 or normal_scores.shape[0] == 0:
        return torch.tensor(0.0, device=anomaly_scores.device)
    hard_pos = anomaly_scores.max(dim=1)[0]
    neg_flat = normal_scores.reshape(-1)
    return F.relu(neg_flat.unsqueeze(0) - hard_pos.unsqueeze(1) + margin).mean()


def sparsity_loss(scores: torch.Tensor, target: float = 0.1) -> torch.Tensor:
    return torch.abs(scores.mean() - target)


def smoothness_loss(scores: torch.Tensor) -> torch.Tensor:
    diff = scores[:, 1:] - scores[:, :-1]
    return (diff ** 2).mean()


def pseudo_label_bce(segment_logits: torch.Tensor, pseudo: torch.Tensor,
                     value: float) -> torch.Tensor:
    """BCE over the segments whose pseudo-label equals ``value`` (NaN = no label)."""
    mask = ~torch.isnan(pseudo) & (pseudo == value)
    if mask.sum() > 0:
        return F.binary_cross_entropy_with_logits(segment_logits[mask], pseudo[mask].float())
    return torch.tensor(0.0, device=segment_logits.device)


def normal_segment_bce(normal_logits: torch.Tensor) -> torch.Tensor:
    """Mean BCE-with-logits against 0 over all segment logits (n_normal, S) of normal videos."""
    return F.binary_cross_entropy_with_logits(normal_logits, torch.zeros_like(normal_logits))
