"""Frame-level metrics under UCA proxy labels, computed from per-segment frame counts.

A video's segment scores are repeated over its frames (evaluator rule: T // 32 frames per
segment, the last segment also takes T % 32). Frame AP with segment-constant scores equals
a weighted AP over segments whose weights are the positive / negative frame counts, which
is what this module computes (identical to ``sklearn.metrics.average_precision_score`` on
the expanded frames; checked in the local unit tests). Proxy frame labels always come from
``Proxy.frame_labels`` (half-open intervals). ``support_frame_counts`` gives the same counts for
units on any frame supports (E021: 16-frame snippet supports, evaluator windows).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wvad.data.manifests import TrainManifest
from wvad.diagnostics.uca import Proxy

N_SEGMENTS = 32
POSITIVE_SEGMENT_MIN_FRACTION = 0.5


def segment_frame_counts(labels: np.ndarray, n_segments: int = N_SEGMENTS
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(positive, negative) frame counts per segment of one video's 0/1 frame labels."""
    t = len(labels)
    seg = np.minimum(np.arange(t) // (t // n_segments), n_segments - 1)
    pos = np.bincount(seg, weights=labels.astype(np.float64), minlength=n_segments)
    tot = np.bincount(seg, minlength=n_segments).astype(np.float64)
    return pos, tot - pos


def support_frame_counts(labels: np.ndarray, starts: Sequence[int] | np.ndarray,
                         ends: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(positive, negative) frame counts of one video's 0/1 frame labels inside each support
    interval [start, end) (any intervals within [0, T), e.g. the 16-frame snippet supports or the
    evaluator windows; E021). When the intervals partition [0, T) and each unit's score covers
    its support, ``weighted_ap`` / ``weighted_auc`` of the counts equal the frame metrics on the
    expanded frames."""
    lab = np.asarray(labels)
    s, e = np.asarray(starts), np.asarray(ends)
    if lab.ndim != 1:
        raise ValueError("labels must be one video's 1-D frame labels")
    if s.ndim != 1 or s.shape != e.shape:
        raise ValueError(f"starts {s.shape} and ends {e.shape} must be 1-D of the same length")
    if s.size and not (np.issubdtype(s.dtype, np.integer) and np.issubdtype(e.dtype, np.integer)):
        raise ValueError("support bounds must be integers")
    if s.size and not ((s >= 0).all() and (s <= e).all() and (e <= len(lab)).all()):
        raise ValueError(f"support intervals must satisfy 0 <= start <= end <= {len(lab)}")
    s, e = s.astype(np.int64), e.astype(np.int64)
    cum = np.concatenate([[0.0], np.cumsum(lab, dtype=np.float64)])
    pos = cum[e] - cum[s]
    return pos, (e - s) - pos


def weighted_ap(scores: np.ndarray, pos_w: np.ndarray, neg_w: np.ndarray) -> float:
    """Average precision (percent) of items with scores and positive / negative weights;
    tied scores form one threshold, as in sklearn."""
    s, p, n = (np.asarray(a, dtype=np.float64).ravel() for a in (scores, pos_w, neg_w))
    order = np.argsort(-s, kind="mergesort")
    s, p, n = s[order], p[order], n[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]       # end of each tie group
    tp, fp = np.cumsum(p)[last], np.cumsum(n)[last]
    if tp[-1] <= 0:
        raise ValueError("no positive weight; AP undefined")
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / tp[-1]
    return float(100 * np.sum(np.diff(np.r_[0.0, recall]) * precision))


def weighted_auc(scores: np.ndarray, pos_w: np.ndarray, neg_w: np.ndarray) -> float:
    """ROC AUC (percent) with weights; ties count one half."""
    s, p, n = (np.asarray(a, dtype=np.float64).ravel() for a in (scores, pos_w, neg_w))
    order = np.argsort(s, kind="mergesort")
    s, p, n = s[order], p[order], n[order]
    uniq, inv = np.unique(s, return_inverse=True)
    pg, ng = np.bincount(inv, weights=p), np.bincount(inv, weights=n)
    neg_below = np.cumsum(ng) - ng
    return float(100 * np.sum(pg * (neg_below + 0.5 * ng)) / (pg.sum() * ng.sum()))


@dataclass(frozen=True)
class ProxyVideoSet:
    """Videos used for a proxy metric, with per-segment positive / negative frame counts."""

    indices: np.ndarray          # manifest (training-cache) indices
    pos: np.ndarray              # (n, 32)
    neg: np.ndarray              # (n, 32)
    anomalous: np.ndarray        # (n,) bool
    categories: tuple[str, ...]

    def ap(self, segment_scores: np.ndarray, rows: np.ndarray | None = None) -> float:
        """segment_scores: (n, 32) aligned with ``indices``; rows: optional subset."""
        sel = slice(None) if rows is None else rows
        return weighted_ap(segment_scores[sel], self.pos[sel], self.neg[sel])

    def positive_segments(self) -> np.ndarray:
        tot = self.pos + self.neg
        return self.pos / tot >= POSITIVE_SEGMENT_MIN_FRACTION


def proxy_video_set(manifest: TrainManifest, proxy: Proxy, variant: str,
                    indices: Sequence[int], min_frames: int = N_SEGMENTS,
                    include_normal: bool = True) -> ProxyVideoSet:
    """Anomalous videos with a proxy in ``variant`` (+ normal videos), at least min_frames."""
    keep, pos, neg, anom = [], [], [], []
    for i in indices:
        t = manifest.total_frames[i]
        if t < min_frames:
            continue
        if manifest.labels[i] == 1:
            if not proxy.has(manifest.names[i], variant):
                continue
            labels = proxy.frame_labels(manifest.names[i], variant, t)
        elif include_normal:
            labels = np.zeros(t, dtype=bool)
        else:
            continue
        p, n = segment_frame_counts(labels)
        keep.append(i)
        pos.append(p)
        neg.append(n)
        anom.append(manifest.labels[i] == 1)
    return ProxyVideoSet(np.asarray(keep), np.stack(pos), np.stack(neg), np.asarray(anom),
                         tuple(manifest.categories[i] for i in keep))
