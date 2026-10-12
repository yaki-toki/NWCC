"""Frame arrays of E022 (PLAN.md "프레임 점수와 지표", gate (v2)): the pooled frame AUC / AP of
frame arrays and the within-video localisation metric L. A new analysis module (code rule 3); the
evaluator (``wvad.eval.metrics``, imported by the training) is not changed.

Frame arrays come from unit values by a unit name or a replication function (``unit``):
"segment" and "window" repeat 32 values over T // 32 frames each with the remainder in the last
(``metrics.expand_segments``; R0, R1, B and C.W, = ``dense_readout.window_to_frames``);
"snippet" repeats each dense snippet's value over its support [16 j, min(16 j + 16, T)) from the
frozen grid (``grid.snippet_support``), the rule of ``dense_readout.snippet_to_frames`` (the frame
score of C, C.G, C.last, C.o32 and A1.S; equal bit for bit, unit test); a callable (for example
``dense_readout.snippet_to_frames`` itself) is used as given. This module imports neither
``dense_readout`` nor ``crossfit``: every other module of src/wvad/eval counts as frozen code
that must not import them (the E021 rule test).

    pooled_frame_metrics  frame AUC and AP (percent) over all frames of the videos with at least
                          ``min_frames`` frames, pooled once: the computation of
                          ``metrics.frame_auc_ap`` after its segment expansion (the same frame
                          labels and sklearn calls, hence the same tie handling); on frame
                          arrays from ``expand_segments`` it equals ``frame_auc_ap`` (gate (v2))
    within_video          L: per anomalous video (video label 1) the frame AUC (UCF) or AP (XD)
                          of its own frames, with the evaluator's labels and ties; an anomalous
                          video whose frame labels are all 1 or all 0 is left out and counted
                          (fixed by the labels alone, so the same for every arm and seed); the
                          included videos are averaged with equal weight, in points
    localisation          L of the logit readout (PLAN.md N12): the stored unit values z + c(v)
                          (or z alone, the z-only readout) repeated over the frames. A per-video
                          constant c(v) leaves the ranks inside a video unchanged in exact
                          arithmetic; the fp32 sigma readout saturates at 1.0 and creates ties,
                          so L is not computed on sigma values.

The z-only readout of the main metric is ``pooled_frame_metrics`` on the frame arrays of the
stored z (no offset); the frame labels are ``metrics.frame_labels`` of the test manifest's
intervals (end frame included, clipped to the video), as the evaluator's.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from wvad.eval.metrics import DEFAULT_MIN_FRAMES, expand_segments
from wvad.extract import grid

Expand = Callable[[np.ndarray, int], np.ndarray]       # (unit values, T) -> (T,) frame values
UNITS = ("segment", "window", "snippet")
METRICS = {"auc": roc_auc_score, "ap": average_precision_score}
PRIMARY = {"ucf": "frame_auc", "xdv": "frame_ap"}       # the main metric of each dataset
L_METRIC = {"ucf": "auc", "xdv": "ap"}                  # L uses the main metric's kind
OPPOSITE = {"ucf": "ap", "xdv": "auc"}                  # the descriptive other kind


def snippet_frames(values: Sequence[float] | np.ndarray, total_frames: int) -> np.ndarray:
    """(T,) frame values: each snippet's value over its support [16 j, min(16 j + 16, T)) (the
    rule of dense_readout.snippet_to_frames, from the frozen grid; dtype kept)."""
    v = np.asarray(values)
    sup = grid.snippet_support(total_frames)
    if len(v) != len(sup):
        raise ValueError(f"{len(v)} values for {len(sup)} snippets")
    return np.repeat(v, sup[:, 1] - sup[:, 0])


def unit_frames(values: Sequence[float] | np.ndarray, total_frames: int,
                unit: str | Expand) -> np.ndarray:
    """(T,) frame values of one video's unit values: ``unit`` "segment" or "window" (32 values,
    the evaluator rule), "snippet" (n_v values over their supports), or a replication function
    (values, T) -> (T,) frame values; checked to hold T values."""
    if callable(unit):
        f = unit(np.asarray(values), int(total_frames))
    elif unit in ("segment", "window"):
        f = expand_segments(values, total_frames)
    elif unit == "snippet":
        f = snippet_frames(values, total_frames)
    else:
        raise ValueError(f"unknown unit {unit!r} (known: {UNITS}, or a function)")
    if np.asarray(f).shape != (int(total_frames),):
        raise ValueError(f"replication gave {np.asarray(f).shape} values for T = {total_frames}")
    return f


def frame_arrays(unit_values: Sequence[Sequence[float] | np.ndarray], totals: Sequence[int],
                 unit: str | Expand) -> list[np.ndarray]:
    """Frame values of every video (manifest order), by ``unit_frames``."""
    if len(unit_values) != len(totals):
        raise ValueError(f"{len(unit_values)} score rows for {len(totals)} videos")
    return [unit_frames(v, int(t), unit) for v, t in zip(unit_values, totals, strict=True)]


def _checked(frame_scores: Sequence[np.ndarray], frame_gts: Sequence[np.ndarray]) -> None:
    if len(frame_scores) != len(frame_gts):
        raise ValueError(f"{len(frame_scores)} score rows for {len(frame_gts)} videos")
    for i, (s, g) in enumerate(zip(frame_scores, frame_gts, strict=True)):
        if np.asarray(s).shape != (len(g),):
            raise ValueError(f"video {i}: {np.asarray(s).shape} frame scores for {len(g)} frames")


def pooled_frame_metrics(frame_scores: Sequence[np.ndarray], frame_gts: Sequence[np.ndarray],
                         min_frames: int = DEFAULT_MIN_FRAMES) -> dict:
    """Pooled frame AUC and AP (percent) of frame arrays over the videos with at least
    ``min_frames`` frames (``metrics.frame_auc_ap`` without its expansion)."""
    _checked(frame_scores, frame_gts)
    keep = [i for i, g in enumerate(frame_gts) if len(g) >= min_frames]
    if not keep:
        raise ValueError(f"no video has at least {min_frames} frames")
    scores = np.concatenate([np.asarray(frame_scores[i]) for i in keep])
    gt = np.concatenate([np.asarray(frame_gts[i]) for i in keep])
    if gt.min() == gt.max():
        raise ValueError("frame labels contain a single class; AUC/AP undefined")
    return {"frame_auc": float(100 * roc_auc_score(gt, scores)),
            "frame_ap": float(100 * average_precision_score(gt, scores)),
            "n_videos": len(keep)}


def video_metric(scores: np.ndarray, gt: np.ndarray, metric: str) -> float:
    """Frame AUC or AP (points) of one video's frames."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r} (known: {tuple(METRICS)})")
    return float(100 * METRICS[metric](np.asarray(gt), np.asarray(scores)))


def within_video(frame_scores: Sequence[np.ndarray], frame_gts: Sequence[np.ndarray],
                 labels: Sequence[int], metric: str) -> dict:
    """L (points): the equal-weight mean of ``video_metric`` over the anomalous videos whose
    frame labels hold both classes; the others are counted in ``n_excluded``."""
    _checked(frame_scores, frame_gts)
    if len(labels) != len(frame_gts):
        raise ValueError(f"{len(labels)} video labels for {len(frame_gts)} videos")
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r} (known: {tuple(METRICS)})")
    per_video, excluded = {}, []
    for i, (s, g, y) in enumerate(zip(frame_scores, frame_gts, labels, strict=True)):
        if int(y) != 1:
            continue
        g = np.asarray(g)
        if g.min() == g.max():
            excluded.append(i)
            continue
        per_video[i] = video_metric(s, g, metric)
    values = list(per_video.values())
    return {"metric": metric, "value": float(np.mean(values)) if values else math.nan,
            "n_videos": len(values), "n_excluded": len(excluded), "excluded": excluded,
            "per_video": per_video}


def localisation(unit_logits: Sequence[Sequence[float] | np.ndarray], totals: Sequence[int],
                 frame_gts: Sequence[np.ndarray], labels: Sequence[int], unit: str | Expand,
                 metric: str) -> dict:
    """L of the logit readout: ``within_video`` of the stored unit logits (z + c(v), or z for
    the z-only readout) repeated over the frames (``unit_frames``)."""
    return within_video(frame_arrays(unit_logits, totals, unit), frame_gts, labels, metric)
