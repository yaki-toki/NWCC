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

Length-aware metrics (E021 PLAN.md, "판독과 지표") also take each video's length q = ceil(T / 16)
/ 32 (16-frame snippets per evaluator window, T = manifest total_frames). The length-stratified
video AUC (LS-AUC) counts only the (anomalous, normal) pairs inside one length bin, so its weights
are the bins' pair counts. ``crossfit_with_length`` adds such metrics to ``crossfit`` with the
same epoch selection; the existing metrics and ``crossfit`` are unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
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


# --- length-aware metrics (E021) ------------------------------------------------------------
# Analysis functions on recorded scores (E021 PLAN.md, two-stage code rule). delta^LS of a
# pipeline and dataset is DELTA_FACTOR (e010) times the df-weighted pooled SD (e018 ``pooled``) of
# R0's cross-fitted LS-AUC over the 6 seeds of each pair ("문턱"); it is computed by the analysis.

SNIPPET_FRAMES, WINDOWS = 16, 32
LS_EDGES = (2.0, 4.0, 8.0, 16.0, 32.0)    # (0, 2], (2, 4], (4, 8], (8, 16], (16, 32], (32, inf)


def snippets_per_window(total_frames: Sequence[int] | np.ndarray) -> np.ndarray:
    """q = ceil(T / 16) / 32 per video: dense snippets per evaluator window (exact in float64, an
    integer over a power of two)."""
    t = np.asarray(total_frames)
    if t.ndim != 1 or not np.issubdtype(t.dtype, np.integer) or (t < 1).any():
        raise ValueError("total_frames must be a 1-D array of positive integers")
    return ((t.astype(np.int64) + SNIPPET_FRAMES - 1) // SNIPPET_FRAMES) / WINDOWS


def length_bins(q: Sequence[float] | np.ndarray, edges: Sequence[float] = LS_EDGES) -> np.ndarray:
    """Bin of every length q: bin 0 = (0, e_0], bin i = (e_(i-1), e_i], the last (e_(-1), inf)."""
    e = np.asarray(edges, dtype=np.float64)
    if (e.ndim != 1 or e.size == 0 or not np.isfinite(e).all() or e[0] <= 0
            or not np.all(np.diff(e) > 0)):
        raise ValueError(f"edges must be finite, positive and strictly increasing, got {edges}")
    lengths = np.asarray(q, dtype=np.float64)
    if lengths.ndim != 1 or not np.all(lengths > 0):           # also rejects NaN
        raise ValueError("lengths q must be a 1-D array of positive values")
    return np.searchsorted(e, lengths, side="left")


def _video_scores(seg: np.ndarray, labels: np.ndarray, q: Sequence[float] | np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Video scores (max over segments, float64), 0/1 labels and lengths, checked for length."""
    s = np.asarray(seg, dtype=np.float64).max(axis=1)
    y = np.asarray(labels)
    lengths = np.asarray(q, dtype=np.float64)
    if not len(s) == len(y) == len(lengths):
        raise ValueError(f"{len(s)} score rows, {len(y)} labels and {len(lengths)} lengths")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return s, y, lengths


def _wins(a: np.ndarray, n: np.ndarray) -> float:
    """Sum over the (a_i, n_j) pairs of [a_i > n_j] + 1/2 [a_i = n_j] (exact: half-integers)."""
    return float((a[:, None] > n[None, :]).sum() + 0.5 * (a[:, None] == n[None, :]).sum())


def _bin_parts(s: np.ndarray, y: np.ndarray, lengths: np.ndarray,
               edges: Sequence[float]) -> dict[str, np.ndarray]:
    bins = length_bins(lengths, edges)
    nb = len(edges) + 1
    n_a = np.bincount(bins[y == 1], minlength=nb)
    n_n = np.bincount(bins[y == 0], minlength=nb)
    wins = np.array([_wins(s[(bins == b) & (y == 1)], s[(bins == b) & (y == 0)])
                     for b in range(nb)])
    return {"n_anomalous": n_a, "n_normal": n_n, "pairs": n_a * n_n, "wins": wins}


def length_bin_parts(seg: np.ndarray, labels: np.ndarray, q: Sequence[float] | np.ndarray,
                     edges: Sequence[float] = LS_EDGES) -> dict[str, np.ndarray]:
    """Per length bin: anomalous and normal video counts, their pair count and the pair wins
    U = sum([s_a > s_n] + 1/2 [s_a = s_n]) of the video scores s (max over segments, finite);
    the bin's AUC is wins / pairs."""
    s, y, lengths = _video_scores(seg, labels, q)
    if not np.isfinite(s).all():
        raise ValueError("video scores must be finite")
    return _bin_parts(s, y, lengths, edges)


def length_stratified_auc(seg: np.ndarray, labels: np.ndarray, q: Sequence[float] | np.ndarray,
                          edges: Sequence[float] = LS_EDGES) -> float:
    """LS-AUC = sum_b U_b / sum_b |A_b| |N_b|: the video AUC (score = max over segments, ties one
    half) over the (anomalous, normal) pairs inside one length bin only, so the bins' pair counts
    are the weights. NaN when a video score is not finite or no bin holds a pair."""
    s, y, lengths = _video_scores(seg, labels, q)
    if not np.isfinite(s).all():
        return math.nan
    parts = _bin_parts(s, y, lengths, edges)
    pairs = int(parts["pairs"].sum())
    return float(parts["wins"].sum() / pairs) if pairs else math.nan


def length_matched_auc(seg: np.ndarray, labels: np.ndarray, q: Sequence[float] | np.ndarray,
                       max_ratio: float = 2.0) -> float:
    """Video AUC over the (anomalous, normal) pairs of similar length, q_n / r <= q_a <= r q_n
    (r = 2: |log2(q_a / q_n)| <= 1; E021 length diagnostic, descriptive). NaN when a video score
    is not finite or no pair qualifies."""
    s, y, lengths = _video_scores(seg, labels, q)
    if not np.isfinite(s).all():
        return math.nan
    qa, qn = lengths[y == 1][:, None], lengths[y == 0][None, :]
    ok = (qa <= max_ratio * qn) & (qn <= max_ratio * qa)
    sa, sn = s[y == 1][:, None], s[y == 0][None, :]
    pairs = int(ok.sum())
    wins = float(((sa > sn) & ok).sum() + 0.5 * ((sa == sn) & ok).sum())
    return wins / pairs if pairs else math.nan


LENGTH_METRICS: Mapping[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "ls_auc": length_stratified_auc}


def crossfit_with_length(scores: np.ndarray, labels: np.ndarray, splits: np.ndarray,
                         q: Sequence[float] | np.ndarray,
                         metrics: Mapping[str, Callable] = METRICS,
                         length_metrics: Mapping[str, Callable] = LENGTH_METRICS) -> CrossFit:
    """``crossfit`` (the same selected epochs, tie rate and values of ``metrics``) plus metrics
    fn(seg, labels, q) measured on the same evaluated half at the same selected epoch, the two
    directions averaged: LS-AUC on the half that the epoch chosen by V-AUC on the other half is
    evaluated on (E021). q: (n,) video lengths (``snippets_per_window``)."""
    clash = set(metrics) & set(length_metrics)
    if clash:
        raise ValueError(f"metric names used twice: {sorted(clash)}")
    lengths = np.asarray(q, dtype=np.float64)
    if lengths.shape != (scores.shape[1],):
        raise ValueError(f"q has shape {lengths.shape}, expected ({scores.shape[1]},)")
    base = crossfit(scores, labels, splits, metrics)
    dirs = {m: np.empty((len(splits), 2)) for m in length_metrics}
    for k, half_a in enumerate(splits):
        for j, ev in enumerate((np.flatnonzero(half_a), np.flatnonzero(~half_a))):
            e = base.epochs[k, j]
            for m, fn in length_metrics.items():
                dirs[m][k, j] = fn(scores[e][ev], labels[ev], lengths[ev])
    return CrossFit({**base.values, **{m: d.mean(axis=1) for m, d in dirs.items()}},
                    {**base.directions, **dirs}, base.epochs, base.tie_rate)
