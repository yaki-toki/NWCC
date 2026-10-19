"""E023 pooled frame-AUC decomposition (PLAN.md "분해", gate (d1), question 4; Song & Lee,
arXiv 2608.21854). New code (add-only rule); judgement-analysis code (PLAN.md 코드 규칙 3).

The pooled frame AUC over all (positive frame, negative frame) pairs of a set of videos splits
into the pairs inside one video and the pairs across videos::

    AUC = w * W_pair + (1 - w) * X
    W_pair  AUC over the (positive, negative) pairs of the same video
    X       AUC over the pairs of two different videos
    w       share of within-video pairs: sum_v P_v N_v / (P N)

A pair counts 1 when the positive frame scores higher, 1/2 on a tie. Counts come from average
ranks (scipy.stats.rankdata): U = sum of the positives' ranks - P (P + 1) / 2, over all frames
and inside each video. Every rank sum is a sum of half-integers below 2^52, so U and the within
part are exact in float64 and the identity residual is pure rounding (gate (d1): <= 1e-12). W_pair
is NaN when no video holds both classes (w = 0); X is NaN when every pair is within one video
(w = 1); the residual then compares AUC with the defined part. AP has no such identity.

    decompose(scores, labels)       per-video frame scores and 0 / 1 frame labels (same lengths)
                                    -> Decomposition
    difference(a, b)                AUC_a - AUC_b = w dW + (1 - w) dX for two decompositions of
                                    the same frames (same labels: same w), with the cross share
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

RESIDUAL_TOL = 1e-12


@dataclass(frozen=True)
class Decomposition:
    auc: float
    w_pair: float               # NaN without a within-video pair
    x: float                    # NaN without a cross-video pair
    w: float
    positives: int
    negatives: int
    pairs: int                  # P N
    pairs_within: int           # sum_v P_v N_v
    u: float                    # pair count of all pairs (ties 1/2)
    u_within: float             # pair count of the within-video pairs
    videos_both: int            # videos holding both classes

    @property
    def recomposed(self) -> float:
        """w W_pair + (1 - w) X (the defined parts only)."""
        if self.pairs_within == 0:
            return self.x
        if self.pairs_within == self.pairs:
            return self.w_pair
        return self.w * self.w_pair + (1.0 - self.w) * self.x

    @property
    def residual(self) -> float:
        return self.auc - self.recomposed

    def to_json(self) -> dict:
        return {"auc": self.auc, "w_pair": self.w_pair, "x": self.x, "w": self.w,
                "positives": self.positives, "negatives": self.negatives, "pairs": self.pairs,
                "pairs_within": self.pairs_within, "u": self.u, "u_within": self.u_within,
                "videos_both": self.videos_both, "residual": self.residual}


def _u(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int, int]:
    """(pair count with ties 1/2, positives, negatives) of one set of frames."""
    pos = labels == 1
    p, n = int(pos.sum()), int((~pos).sum())
    if p == 0 or n == 0:
        return 0.0, p, n
    ranks = rankdata(scores)
    return float(ranks[pos].sum() - p * (p + 1) / 2.0), p, n


def _labels(y) -> np.ndarray:
    a = np.asarray(y)
    if a.ndim != 1 or not np.isin(a, (0, 1)).all():
        raise ValueError("frame labels must be a 1-D array of 0 / 1")
    return a.astype(np.int8)


def decompose(scores: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> Decomposition:
    """The decomposition of the pooled frame AUC of these videos (module docstring)."""
    if len(scores) != len(labels) or not len(scores):
        raise ValueError(f"{len(scores)} score arrays for {len(labels)} label arrays")
    s_list, y_list = [], []
    for s, y in zip(scores, labels, strict=True):
        s64 = np.asarray(s, dtype=np.float64)
        y8 = _labels(y)
        if s64.ndim != 1 or len(s64) != len(y8):
            raise ValueError(f"{len(s64)} scores for {len(y8)} frame labels")
        if not np.isfinite(s64).all():
            raise ValueError("frame scores must be finite")
        s_list.append(s64)
        y_list.append(y8)
    u, p, n = _u(np.concatenate(s_list), np.concatenate(y_list))
    if p == 0 or n == 0:
        raise ValueError("the pooled frames need both classes")
    u_within, pairs_within, both = 0.0, 0, 0
    for s64, y8 in zip(s_list, y_list, strict=True):
        uv, pv, nv = _u(s64, y8)
        if pv and nv:
            u_within += uv
            pairs_within += pv * nv
            both += 1
    pairs = p * n
    cross = pairs - pairs_within
    return Decomposition(auc=u / pairs,
                         w_pair=u_within / pairs_within if pairs_within else math.nan,
                         x=(u - u_within) / cross if cross else math.nan,
                         w=pairs_within / pairs, positives=p, negatives=n, pairs=pairs,
                         pairs_within=pairs_within, u=u, u_within=u_within, videos_both=both)


def identity_holds(d: Decomposition, tol: float = RESIDUAL_TOL) -> bool:
    """Gate (d1): |AUC - (w W_pair + (1 - w) X)| <= tol."""
    r = d.residual
    return math.isfinite(r) and abs(r) <= tol


@dataclass(frozen=True)
class Difference:
    d_auc: float
    within_part: float          # w * dW_pair
    cross_part: float           # (1 - w) * dX
    cross_share: float          # cross_part / d_auc (NaN when d_auc = 0)


def difference(a: Decomposition, b: Decomposition) -> Difference:
    """AUC_a - AUC_b split into the within- and cross-video parts (same frames and labels)."""
    if (a.positives, a.negatives, a.pairs_within) != (b.positives, b.negatives, b.pairs_within):
        raise ValueError("the two decompositions are not over the same frames and labels")
    w = a.w
    within = w * (a.w_pair - b.w_pair) if a.pairs_within else 0.0
    cross = (1.0 - w) * (a.x - b.x) if a.pairs_within != a.pairs else 0.0
    d = a.auc - b.auc
    return Difference(d, within, cross, cross / d if d != 0 else math.nan)
