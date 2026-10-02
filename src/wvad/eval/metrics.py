"""Frame- and video-level detection metrics.

Conventions (identical to the previous project's evaluators, without half precision):

* frame labels: a video of ``T`` frames is 0 everywhere except the annotated intervals
  ``[start, end]``, whose end frame is included; intervals are clipped to ``[0, T)``;
* segment -> frame expansion: ``S`` segment scores cover ``T`` frames, each segment
  ``T // S`` frames and the last segment also the remaining ``T % S``;
* frame AUC / AP pool all frames of all videos with at least ``min_frames`` frames
  (videos shorter than the number of segments are skipped, as before);
* video AUC (validation selection): video score = max over its segment scores.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_MIN_FRAMES = 32


def frame_labels(total_frames: int, intervals: Iterable[Sequence[int]]) -> np.ndarray:
    """0/1 frame labels of one video; ``intervals`` hold inclusive [start, end] frames."""
    t = int(total_frames)
    if t < 1:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    gt = np.zeros(t, dtype=np.float32)
    for interval in intervals:
        start, end = interval
        lo, hi = max(0, int(start)), min(t, int(end) + 1)
        if lo < hi:
            gt[lo:hi] = 1.0
    return gt


def expand_segments(scores: Sequence[float], total_frames: int) -> np.ndarray:
    """Repeat segment scores over frames; the last segment takes the remainder."""
    s = np.asarray(scores)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty 1-D sequence")
    t = int(total_frames)
    if t < s.size:
        raise ValueError(f"{t} frames cannot hold {s.size} segments")
    counts = np.full(s.size, t // s.size, dtype=int)
    counts[-1] += t % s.size
    return np.repeat(s, counts)


def frame_auc_ap(segment_scores: Sequence[Sequence[float]], frame_gts: Sequence[np.ndarray],
                 min_frames: int = DEFAULT_MIN_FRAMES) -> dict:
    """Pooled frame AUC and AP (percent) over the videos with at least ``min_frames`` frames."""
    if len(segment_scores) != len(frame_gts):
        raise ValueError(f"{len(segment_scores)} score rows for {len(frame_gts)} videos")
    keep = [i for i, g in enumerate(frame_gts) if len(g) >= min_frames]
    if not keep:
        raise ValueError(f"no video has at least {min_frames} frames")
    scores = np.concatenate([expand_segments(segment_scores[i], len(frame_gts[i]))
                             for i in keep])
    gt = np.concatenate([np.asarray(frame_gts[i]) for i in keep])
    if gt.min() == gt.max():
        raise ValueError("frame labels contain a single class; AUC/AP undefined")
    return {"frame_auc": float(100 * roc_auc_score(gt, scores)),
            "frame_ap": float(100 * average_precision_score(gt, scores)),
            "n_videos": len(keep)}


def max_video_auc(segment_scores: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Video ROC-AUC with video score = max over segments; NaN if any score is not finite."""
    if len(segment_scores) != len(labels):
        raise ValueError(f"{len(segment_scores)} score rows for {len(labels)} labels")
    video = np.array([float(np.max(np.asarray(s, dtype=np.float64))) for s in segment_scores])
    if not np.all(np.isfinite(video)):
        return math.nan
    y = np.asarray([int(x) for x in labels])
    if y.min() == y.max():
        raise ValueError("labels contain a single class; video AUC undefined")
    return float(roc_auc_score(y, video))
