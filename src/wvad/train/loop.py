"""One training epoch, validation scoring and seeding, as in the earlier implementation.

Operation order, mixed precision, gradient scaling and clipping follow the earlier
trainers exactly, so a seed reproduces an earlier run. In particular every DataLoader is
iterated the same way (each iterator draws from the global random stream).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from wvad.data.features import FeatureSet
from wvad.eval.metrics import max_video_auc
from wvad.losses import (
    dense_ranking_loss,
    max_ranking_loss,
    normal_segment_bce,
    pseudo_label_bce,
    smoothness_loss,
    sparsity_loss,
    topk_video_bce,
    video_bce,
)


@dataclass(frozen=True)
class LossWeights:
    mil: float = 0.1               # weight of the whole MIL term in the total loss
    rank_margin: float = 0.1       # max-ranking hinge margin
    dense: float = 0.0             # dense normal-to-anomaly ranking (inside the MIL term)
    dense_margin: float = 0.5
    sparse: float = 0.01
    smooth: float = 0.001
    pseudo: float = 0.0            # pseudo-positive BCE (stage 2)
    neg_pseudo: float = 0.0        # pseudo-negative BCE (stage 2)
    normal_seg: float = 0.0        # BCE on every normal-video segment (stage D1; 0 = off)
    video_topk: int = 0            # video BCE: 0 = attention-pooled logit; k >= 1 = mean
                                   # probability of the top-k segments (stage D2, E015)


class VideoBags(Dataset):
    """Items (features, label, segment labels[, pseudo-labels]) of one feature set."""

    def __init__(self, features: FeatureSet, with_pseudo: bool = False):
        self.fs = features
        self.with_pseudo = with_pseudo
        self.pseudo: list[torch.Tensor | None] = [None] * len(features)

    def __len__(self) -> int:
        return len(self.fs)

    def set_pseudo_labels(self, pseudo: list[torch.Tensor | None]) -> None:
        if len(pseudo) != len(self.fs):
            raise ValueError(f"{len(pseudo)} pseudo-label rows for {len(self.fs)} videos")
        self.pseudo = list(pseudo)

    def __getitem__(self, idx: int):
        feats = self.fs.features[idx].float()
        item = (feats, torch.tensor(self.fs.labels[idx], dtype=torch.long),
                self.fs.seg_labels[idx].float())
        if not self.with_pseudo:
            return item
        p = self.pseudo[idx]
        return item + (p.float() if p is not None else torch.full((feats.shape[0],), float("nan")),)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(ds: Dataset, batch_size: int, train: bool, device: torch.device) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0,
                      pin_memory=device.type == "cuda", drop_last=train)


def _losses(model, batch, w: LossWeights, device: torch.device):
    feats, labels = batch[0].to(device), batch[1].float().to(device)
    video_logits, seg_logits, seg_scores = model(feats)
    anomaly, normal = labels == 1, labels == 0
    zero = torch.tensor(0.0, device=device)
    if w.video_topk == 0:          # baseline path, unchanged
        l_bce = video_bce(video_logits, labels)
    else:                          # the attention branch then gets no gradient at all
        l_bce = topk_video_bce(seg_logits, labels, w.video_topk)
    l_rank, l_dense = zero, zero
    if anomaly.sum() > 0 and normal.sum() > 0:
        l_rank = max_ranking_loss(seg_scores[anomaly], seg_scores[normal], w.rank_margin)
        if w.dense > 0:
            l_dense = dense_ranking_loss(seg_scores[anomaly], seg_scores[normal], w.dense_margin)
    l_sparse = sparsity_loss(seg_scores[anomaly]) if anomaly.sum() > 0 else zero
    l_smooth = smoothness_loss(seg_scores)
    l_mil = l_bce + l_rank + w.dense * l_dense + w.sparse * l_sparse + w.smooth * l_smooth
    total = w.mil * l_mil
    l_normal = None                 # not computed at all when the weight is 0 (baseline)
    if w.normal_seg > 0:
        l_normal = normal_segment_bce(seg_logits[normal]) if normal.sum() > 0 else zero
        total = total + w.normal_seg * l_normal
    l_pos = zero
    if len(batch) == 4:
        pseudo = batch[3].to(device)
        l_pos = pseudo_label_bce(seg_logits, pseudo, 1.0)
        l_neg = pseudo_label_bce(seg_logits, pseudo, 0.0)
        total = total + w.pseudo * l_pos + w.neg_pseudo * l_neg
    return total, l_mil, l_pos, l_normal


def train_one_epoch(model, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler | None, w: LossWeights, device: torch.device,
                    use_amp: bool) -> dict:
    model.train()
    optimizer.zero_grad()
    sums = np.zeros(3)
    normal_sum = 0.0
    for batch in loader:
        with autocast(device_type=device.type, enabled=use_amp):
            total, l_mil, l_pos, l_normal = _losses(model, batch, w, device)
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
        sums += (total.item(), l_mil.item(), l_pos.item())
        if l_normal is not None:
            normal_sum += l_normal.item()
    means = sums / max(len(loader), 1)
    out = {"loss_total": float(means[0]), "loss_mil": float(means[1]),
           "loss_pseudo": float(means[2])}
    if w.normal_seg > 0:            # key absent for the baseline (its history is unchanged)
        out["loss_normal_seg"] = normal_sum / max(len(loader), 1)
    return out


def validation_video_auc(model, loader: DataLoader, device: torch.device,
                         record: list | None = None) -> float:
    """Video AUC on validation videos: max over segments of p, fp32 (autocast off).

    With ``record`` given, the (n_videos, S) segment scores of this same pass are appended
    to it; nothing else changes (no extra pass over the loader, no extra random draws).
    """
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch[0].to(device).float()
            with autocast(device_type=device.type, enabled=False):
                _, _, seg_scores = model(feats)
            scores.extend(seg_scores.float().cpu().numpy())
            labels.extend(batch[1].tolist())
    if record is not None:
        record.append(np.stack(scores).astype(np.float32))
    return max_video_auc(scores, labels)
