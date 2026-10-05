"""Cross-fitted video-level validation metrics (stage-D candidate comparison).

Checkpoint selection stays "highest validation video AUC over all validation videos".
To compare designs without rewarding volatile validation curves, a run is scored by
cross-fitting: the validation videos are split R times into stratified halves (normal /
anomalous, from the fixed validation order of the split file, ``default_rng(seed)``
restarted per run so every run uses the same splits); on each split the epoch is chosen by
video AUC on one half and the metrics are measured on the other half, then the halves are
swapped and the two directions averaged (E010 PLAN.md).

Video-label metrics only (no temporal labels): V-AUC / V-AP / V-pAUC use the video score
max over segments; NS-AUC compares anomalous videos' maxima with all segments of normal
videos (every segment of a normal video is a sure negative).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def video_auc(seg: np.ndarray, labels: np.ndarray) -> float:
    """seg: (n, S) segment scores; labels: (n,) 0/1. Ties count one half. Exact rank
    formula, so equal AUCs of two models are bit-identical (a difference is exactly 0)."""
    return float(rank_auc(seg.max(axis=1)[None], labels)[0])


def video_ap(seg: np.ndarray, labels: np.ndarray) -> float:
    return float(average_precision_score(labels, seg.max(axis=1)))


def video_pauc(seg: np.ndarray, labels: np.ndarray, max_fpr: float = 0.1) -> float:
    """McClish-standardized partial AUC up to ``max_fpr`` (sklearn)."""
    return float(roc_auc_score(labels, seg.max(axis=1), max_fpr=max_fpr))


def normal_segment_auc(seg: np.ndarray, labels: np.ndarray) -> float:
    """Anomalous videos' segment maxima (positives) vs every normal-video segment."""
    pos = seg[labels == 1].max(axis=1)
    neg = seg[labels == 0].ravel()
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


METRICS: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "v_auc": video_auc, "v_ap": video_ap, "ns_auc": normal_segment_auc, "v_pauc": video_pauc}


def stratified_half_splits(labels: np.ndarray, n_splits: int, seed: int = 0) -> np.ndarray:
    """(n_splits, n) boolean masks of half A; each label stratum split in half (floor)."""
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(labels == c) for c in (0, 1)]
    out = np.zeros((n_splits, len(labels)), dtype=bool)
    for r in range(n_splits):
        for s in strata:
            perm = rng.permutation(s)
            out[r, perm[:len(perm) // 2]] = True
    return out


def rank_auc(vmax: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Video AUC of every row of ``vmax`` (E, m) by the Mann-Whitney rank formula (ties one
    half). Equal AUCs are bit-identical (half-integer rank sums over one denominator)."""
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    ranks = rankdata(vmax, axis=1)
    return (ranks[:, pos].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def select_epoch(vmax: np.ndarray, labels: np.ndarray, rows: np.ndarray) -> tuple[int, bool]:
    """vmax: (E, n) video scores. Epoch index with the highest video AUC on ``rows``
    (earliest on ties), and whether the maximum was tied."""
    aucs = rank_auc(vmax[:, rows], labels[rows])
    best = np.flatnonzero(aucs == aucs.max())
    return int(best[0]), len(best) > 1


@dataclass(frozen=True)
class CrossFit:
    """Per-split values (average of the two directions) and the per-direction values."""

    values: dict[str, np.ndarray]           # metric -> (R,)
    directions: dict[str, np.ndarray]       # metric -> (R, 2): evaluated on half A, half B
    epochs: np.ndarray                      # (R, 2) selected index into the candidate axis of
                                            # ``scores`` (selected on B for A, on A for B)
    tie_rate: float

    def mean(self, metric: str) -> float:
        return float(self.values[metric].mean())

    def mc_se(self, metric: str) -> float:
        v = self.values[metric]
        return float(v.std(ddof=1) / np.sqrt(len(v)))


def crossfit(scores: np.ndarray, labels: np.ndarray, splits: np.ndarray,
             metrics: Mapping[str, Callable] = METRICS) -> CrossFit:
    """scores: (E, n, S) candidate epochs; labels: (n,); splits: (R, n) half-A masks."""
    r = len(splits)
    vmax = scores.max(axis=2)
    dirs = {m: np.empty((r, 2)) for m in metrics}
    epochs = np.empty((r, 2), dtype=int)
    ties = 0
    for k, half_a in enumerate(splits):
        a, b = np.flatnonzero(half_a), np.flatnonzero(~half_a)
        for j, (sel, ev) in enumerate(((b, a), (a, b))):
            e, tie = select_epoch(vmax, labels, sel)
            epochs[k, j], ties = e, ties + tie
            for m, fn in metrics.items():
                dirs[m][k, j] = fn(scores[e][ev], labels[ev])
    return CrossFit({m: d.mean(axis=1) for m, d in dirs.items()}, dirs, epochs,
                    ties / (2 * r))
