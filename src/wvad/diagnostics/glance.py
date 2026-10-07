"""GlanceVAD glance points: the temporal-annotation diagnostics of E021 (descriptive only).

GlanceVAD (Zhang et al., ICME 2025, arXiv:2403.06154; MIT, see data/README.md) marks one frame
inside every anomalous event of the UCF-Crime and XD-Violence training videos. The two CSV files
are kept byte for byte under ``data/annotations/glancevad`` and their sha256 is checked on
reading. Only the rows of the validation part's videos are parsed, looked up by the training
manifest's names; the training part's points are skipped unread. Frame numbers are used as
0-based indices (E021 PLAN.md, "시간 주석 진단").

Computed once, after the E021 verdict file is committed (``describe``); never used for a verdict,
gate, selection or flag, nor for a later plan's choices. A run's frame scores (logits) are given
per validation video as unit scores constant on contiguous frame supports (``UnitScores``: the 32
evaluator windows, or the 16-frame snippet supports):

    GN-AUC  glance points (positives) vs every frame of the normal validation videos
            (negatives), ties one half
    GP      a point g's mid-rank percentile among its video's frames,
            (#{t: f(t) < f(g)} + 1/2 #{t: f(t) = f(g)}) / T; mean over the points and over the
            videos (mean of each video's point mean); chance 0.5
    Hit@w   the top unit (earliest on ties), its center c* = start + floor(length / 2); a hit
            when min_g |c* - g| <= w (w = 32, 128, 512 frames); share of the anomalous videos,
            with the chance level: mean over the videos of the share of frames t with
            min_g |t - g| <= w

Bootstrap of a contrast X - Y: 2,000 resamples from ``default_rng(0)`` (restarted per call): the
anomalous and the normal videos are resampled separately with replacement, then the development
seeds with replacement; both arms use the same videos and the same seeds (seed-paired). Row 0 of
every draw matrix is the original sample. Statistics are evaluated with the draw counts as video
weights, which equals evaluating them on the resampled videos.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wvad.data.catalog import split_file
from wvad.data.manifests import TrainManifest, load_manifest, manifest_file
from wvad.data.splits import ValSplit, check_labels, load_split
from wvad.paths import REPO_ROOT

GLANCE_DIR = REPO_ROOT / "data" / "annotations" / "glancevad"
GLANCE_FILES = {   # dataset -> (file, sha256); upstream master 18ff774d (data/README.md)
    "ucf": ("ucf_glance_annotations.csv",
            "c8c06b24468f6dc3fad9e699d10cf609711d3af612815eccb709e2fefb0fc6b7"),
    "xdv": ("xd_glance_annotations.csv",
            "46bfa2269b5dcc81f9407d8e2c958b0c13372105753753c32c3cd9656dbd2283"),
}
COLUMNS = ("video-id", "glance")
HIT_WINDOWS = (32, 128, 512)
STATISTICS = ("gn_auc", "gp_point", "gp_video", *(f"hit_{w}" for w in HIT_WINDOWS))
N_BOOT, BOOT_SEED = 2000, 0


class GlanceError(ValueError):
    """A glance file or its points do not match the pinned file or the validation videos."""


# --- reading --------------------------------------------------------------------------------

def read_glance_csv(path: Path, sha256: str, names: Collection[str]) -> dict[str, list[int]]:
    """Glance frames of the videos in ``names`` after checking the file's sha256; the rows of
    every other video are skipped without reading their frame."""
    data = Path(path).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise GlanceError(f"{path}: sha256 {got} differs from the pinned {sha256}")
    rows = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    if tuple(next(rows, ())) != COLUMNS:
        raise GlanceError(f"{path}: header must be {','.join(COLUMNS)}")
    wanted = set(names)
    out: dict[str, list[int]] = {}
    for line, row in enumerate(rows, start=2):
        if len(row) != len(COLUMNS):
            raise GlanceError(f"{path}:{line}: {len(row)} fields, expected {len(COLUMNS)}")
        if row[0] in wanted:
            try:
                out.setdefault(row[0], []).append(int(row[1]))
            except ValueError as exc:
                raise GlanceError(f"{path}:{line}: frame '{row[1]}' is not an integer") from exc
    return out


@dataclass(frozen=True, eq=False)
class GlanceSet:
    """The validation videos of one dataset, in the split's order (the order of the recorded
    validation scores), with the glance points of the anomalous ones."""

    dataset: str
    val_indices: np.ndarray             # (n,) training-manifest (cache) indices
    names: tuple[str, ...]
    total_frames: np.ndarray            # (n,) T from the manifest
    labels: np.ndarray                  # (n,) 0 normal, 1 anomalous
    points: tuple[np.ndarray, ...]      # (n,) sorted 0-based glance frames; empty when normal
    sha256: str

    @property
    def anomalous(self) -> np.ndarray:
        return np.flatnonzero(self.labels == 1)

    @property
    def normal(self) -> np.ndarray:
        return np.flatnonzero(self.labels == 0)

    def coverage(self) -> dict:
        """Counts of the kept points (the pre-freeze facts of the plan, re-measured)."""
        pts = [self.points[i] for i in self.anomalous]
        n = np.array([len(p) for p in pts])
        return {"dataset": self.dataset, "sha256": self.sha256,
                "n_validation": len(self.labels), "n_anomalous": len(pts),
                "n_normal": len(self.normal), "n_points": int(n.sum()),
                "points_per_video_median": float(np.median(n)),
                "points_per_video_max": int(n.max()),
                "min_point": int(min(p[0] for p in pts)),
                "min_frames_after_last_point": int(min(
                    self.total_frames[i] - 1 - self.points[i][-1] for i in self.anomalous))}


def load_glance(dataset: str, manifest: TrainManifest | None = None,
                split: ValSplit | None = None) -> GlanceSet:
    """Glance points of the validation part's anomalous videos. Checks the file's sha256, the
    manifest labels against the split, at least one point for every anomalous validation video,
    none for the normal ones, 0 <= g < T and no duplicate point."""
    if dataset not in GLANCE_FILES:
        raise KeyError(f"no glance file for dataset '{dataset}' (known: {', '.join(GLANCE_FILES)})")
    if manifest is None:
        manifest = load_manifest(manifest_file(dataset))
    if split is None:
        split = load_split(split_file(dataset))
    check_labels(split, manifest.labels, f"{dataset} training manifest")
    val = np.asarray(split.val_indices, dtype=np.int64)
    names = tuple(manifest.names[i] for i in val)
    labels = np.asarray([manifest.labels[i] for i in val], dtype=np.int64)
    total = np.asarray([manifest.total_frames[i] for i in val], dtype=np.int64)
    fname, sha = GLANCE_FILES[dataset]
    found = read_glance_csv(GLANCE_DIR / fname, sha, names)
    points = []
    for name, y, t in zip(names, labels, total, strict=True):
        g = np.sort(np.asarray(found.get(name, []), dtype=np.int64))
        if y == 0 and g.size:
            raise GlanceError(f"{name}: a normal validation video has glance points")
        if y == 1 and not g.size:
            raise GlanceError(f"{name}: an anomalous validation video has no glance point")
        if g.size and (g[0] < 0 or g[-1] >= t):
            raise GlanceError(f"{name}: glance frame outside [0, {t})")
        if np.unique(g).size != g.size:
            raise GlanceError(f"{name}: duplicate glance frames")
        points.append(g)
    return GlanceSet(dataset, val, names, total, labels, tuple(points), sha)


# --- frame scores ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class UnitScores:
    """Frame scores of one video, constant on contiguous units: unit u covers the frames
    [bounds[u], bounds[u + 1]) with the score values[u] (bounds[0] = 0, bounds[-1] = T, every unit
    at least one frame). Values must be finite; they are compared as float64."""

    values: np.ndarray
    bounds: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        b = np.asarray(self.bounds)
        if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all():
            raise ValueError("unit scores must be a non-empty 1-D array of finite values")
        if b.shape != (v.size + 1,) or not np.issubdtype(b.dtype, np.integer):
            raise ValueError(f"bounds must be {v.size + 1} integers, got {b.shape} {b.dtype}")
        if b[0] != 0 or (np.diff(b) < 1).any():
            raise ValueError("bounds must start at 0 and increase strictly")
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "bounds", b.astype(np.int64))

    @classmethod
    def from_counts(cls, values: Sequence[float] | np.ndarray,
                    counts: Sequence[int] | np.ndarray) -> UnitScores:
        """Unit u covers the next counts[u] frames, e.g. the evaluator windows (T // 32 frames
        each, the last also T % 32; ``wvad.eval.metrics.expand_segments``)."""
        c = np.asarray(counts)
        if c.ndim != 1 or not np.issubdtype(c.dtype, np.integer):
            raise ValueError("counts must be a 1-D array of integers")
        return cls(values, np.concatenate([[0], np.cumsum(c)]))

    @classmethod
    def from_supports(cls, values: Sequence[float] | np.ndarray,
                      starts: Sequence[int] | np.ndarray,
                      ends: Sequence[int] | np.ndarray) -> UnitScores:
        """Units on the supports [starts[u], ends[u]) that tile [0, T) in order, e.g. the 16-frame
        snippet supports [16 j, min(16 j + 16, T))."""
        s, e = np.asarray(starts), np.asarray(ends)
        if s.ndim != 1 or s.shape != e.shape or s.size == 0:
            raise ValueError("starts and ends must be non-empty 1-D arrays of the same length")
        if s[0] != 0 or (s[1:] != e[:-1]).any():
            raise ValueError("supports must tile [0, T) in order")
        return cls(values, np.concatenate([s, e[-1:]]))

    @property
    def total_frames(self) -> int:
        return int(self.bounds[-1])

    def at(self, frames: Sequence[int] | np.ndarray) -> np.ndarray:
        """Scores of the given frames (0 <= frame < T)."""
        f = np.asarray(frames, dtype=np.int64)
        if f.size and (f.min() < 0 or f.max() >= self.total_frames):
            raise ValueError(f"frames outside [0, {self.total_frames})")
        return self.values[np.searchsorted(self.bounds, f, side="right") - 1]

    def below_equal(self, x: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Numbers of frames whose score is below / equal to each value of x."""
        order = np.argsort(self.values, kind="stable")
        v = self.values[order]
        cum = np.concatenate([[0], np.cumsum(np.diff(self.bounds)[order])])
        xs = np.asarray(x, dtype=np.float64)
        lo, hi = np.searchsorted(v, xs, side="left"), np.searchsorted(v, xs, side="right")
        return cum[lo], cum[hi] - cum[lo]

    def top_center(self) -> int:
        """Center frame start + floor(length / 2) of the highest unit (the earliest on ties)."""
        u = int(np.argmax(self.values))
        a, b = int(self.bounds[u]), int(self.bounds[u + 1])
        return a + (b - a) // 2


# --- statistics -----------------------------------------------------------------------------

def covered_share(points: np.ndarray, total_frames: int, w: int) -> float:
    """Share of the frames t in [0, T) with min_g |t - g| <= w."""
    covered, end = 0, -1
    for g in np.sort(np.asarray(points, dtype=np.int64)).tolist():
        a, b = max(g - w, end + 1, 0), min(g + w, total_frames - 1)
        if b >= a:
            covered += b - a + 1
        end = max(end, b)
    return covered / total_frames


@dataclass(frozen=True, eq=False)
class Draws:
    """Bootstrap draw counts, rows 0..R; row 0 is the original sample (everything once)."""

    anomalous: np.ndarray     # (R + 1, n_anomalous)
    normal: np.ndarray        # (R + 1, n_normal)
    seeds: np.ndarray         # (R + 1, n_seeds)
    rng_seed: int


def _draw_counts(rng: np.random.Generator, n_resamples: int, n: int) -> np.ndarray:
    draws = rng.integers(0, n, size=(n_resamples, n))
    flat = (draws + n * np.arange(n_resamples)[:, None]).ravel()
    counts = np.bincount(flat, minlength=n_resamples * n).reshape(n_resamples, n)
    return np.vstack([np.ones((1, n), dtype=np.int64), counts.astype(np.int64)])


def bootstrap_draws(n_anomalous: int, n_normal: int, n_seeds: int, n_resamples: int = N_BOOT,
                    seed: int = BOOT_SEED) -> Draws:
    """Draw counts from default_rng(seed): for the resamples 1..R, the anomalous videos, then the
    normal videos, then the seeds, each drawn with replacement to its own number (one (R, n)
    block each, in that order)."""
    if min(n_anomalous, n_normal, n_seeds, n_resamples) < 1:
        raise ValueError("bootstrap needs at least one anomalous video, normal video, seed and "
                         "resample")
    rng = np.random.default_rng(seed)
    anomalous = _draw_counts(rng, n_resamples, n_anomalous)
    normal = _draw_counts(rng, n_resamples, n_normal)
    seeds = _draw_counts(rng, n_resamples, n_seeds)
    return Draws(anomalous, normal, seeds, seed)


def _check_frames(gs: GlanceSet, frames: Sequence[UnitScores]) -> None:
    if len(frames) != len(gs.labels):
        raise ValueError(f"{len(frames)} frame-score rows for {len(gs.labels)} validation videos")
    for f, t, name in zip(frames, gs.total_frames, gs.names, strict=True):
        if f.total_frames != t:
            raise ValueError(f"{name}: frame scores cover {f.total_frames} frames, manifest {t}")


def _weights(gs: GlanceSet, draws: Draws | None) -> tuple[np.ndarray, np.ndarray]:
    if draws is None:
        return np.ones((1, len(gs.anomalous))), np.ones((1, len(gs.normal)))
    if draws.anomalous.shape[1] != len(gs.anomalous) or draws.normal.shape[1] != len(gs.normal):
        raise ValueError("draws do not match the numbers of anomalous and normal videos")
    return draws.anomalous.astype(np.float64), draws.normal.astype(np.float64)


def run_statistics(gs: GlanceSet, frames: Sequence[UnitScores],
                   draws: Draws | None = None) -> dict[str, np.ndarray]:
    """GN-AUC, GP (point and video mean) and Hit@w of one run: ``frames`` holds the frame scores
    of every validation video in ``gs`` order. Returns (R + 1,) arrays, one value per row of the
    draws (row 0 = the original sample; without draws, row 0 only). Pass the same draws to both
    arms of a contrast."""
    _check_frames(gs, frames)
    wa, wn = _weights(gs, draws)
    anomalous, normal = gs.anomalous, gs.normal
    pts = [gs.points[i] for i in anomalous]
    if not pts or not len(normal) or min(len(p) for p in pts) < 1:
        raise ValueError("needs normal videos and anomalous videos with at least one point each")
    owner = np.repeat(np.arange(len(anomalous)), [len(p) for p in pts])     # point -> video
    score, gp = [], []
    for i, p in zip(anomalous, pts, strict=True):
        s = frames[i].at(p)
        below, equal = frames[i].below_equal(s)
        score.append(s)
        gp.append((below + 0.5 * equal) / gs.total_frames[i])
    s, gp_points = np.concatenate(score), np.concatenate(gp)
    wins = np.empty((len(s), len(normal)))          # normal frames below + 1/2 equal, per point
    for k, v in enumerate(normal):
        below, equal = frames[v].below_equal(s)
        wins[:, k] = below + 0.5 * equal
    wp = wa[:, owner]
    out = {"gn_auc": ((wp @ wins) * wn).sum(axis=1)
               / (wp.sum(axis=1) * (wn @ gs.total_frames[normal].astype(np.float64))),
           "gp_point": wp @ gp_points / wp.sum(axis=1),
           "gp_video": wa @ (np.bincount(owner, weights=gp_points) / np.bincount(owner))
               / wa.sum(axis=1)}
    centers = np.array([frames[i].top_center() for i in anomalous])
    nearest = np.array([np.abs(p - c).min() for p, c in zip(pts, centers, strict=True)])
    for w in HIT_WINDOWS:
        out[f"hit_{w}"] = wa @ (nearest <= w).astype(np.float64) / wa.sum(axis=1)
    return out


def hit_chance(gs: GlanceSet, draws: Draws | None = None) -> dict[str, np.ndarray]:
    """Chance level of Hit@w (the same for every run): mean over the anomalous videos of the
    share of frames within w of a point, per row of the draws."""
    wa, _ = _weights(gs, draws)
    out = {}
    for w in HIT_WINDOWS:
        share = np.array([covered_share(gs.points[i], int(gs.total_frames[i]), w)
                          for i in gs.anomalous])
        out[f"chance_{w}"] = wa @ share / wa.sum(axis=1)
    return out


# --- reporting ------------------------------------------------------------------------------

def contrast(x: Sequence[np.ndarray], y: Sequence[np.ndarray], draws: Draws) -> dict:
    """X - Y of one statistic. x[s], y[s]: the arms' (R + 1,) values of seed s (``run_statistics``
    with ``draws``, or any statistic evaluated with the draws' video counts as weights, e.g. the
    UCA proxy diagnostics; the same seed order). Per row r: sum_s m[r, s] (x_s[r] - y_s[r]) /
    sum_s m[r, s] with the seed draw counts m. Row 0, the seed mean of the paired differences on
    the original sample, is the estimate; the 95% percentile interval is over rows 1..R."""
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = draws.seeds.astype(np.float64)
    if xs.shape != ys.shape or xs.shape != (m.shape[1], m.shape[0]):
        raise ValueError(f"arm values {xs.shape} / {ys.shape} do not match the draws "
                         f"({m.shape[1]} seeds, {m.shape[0]} rows)")
    d = ((xs - ys).T * m).sum(axis=1) / m.sum(axis=1)
    if not np.isfinite(d).all():
        raise ValueError(f"{int((~np.isfinite(d)).sum())} non-finite bootstrap differences")
    lo, hi = (float(v) for v in np.percentile(d[1:], [2.5, 97.5]))
    return {"estimate": float(d[0]), "by_seed": (xs[:, 0] - ys[:, 0]).tolist(),
            "ci95": [lo, hi], "n_resamples": len(d) - 1, "rng_seed": draws.rng_seed}


def contrasts(x: Sequence[Mapping[str, np.ndarray]], y: Sequence[Mapping[str, np.ndarray]],
              draws: Draws) -> dict[str, dict]:
    """``contrast`` of every statistic of ``run_statistics`` (x[s], y[s]: seed s of each arm)."""
    return {k: contrast([r[k] for r in x], [r[k] for r in y], draws) for k in STATISTICS}


def seed_summary(values: Mapping[int, float]) -> dict:
    """Per-seed values, their mean and SD (ddof 1; NaN with fewer than two seeds): an arm's
    development seeds, or H0's six seeds as the null-scale reference of the plan."""
    v = np.asarray(list(values.values()), dtype=np.float64)
    return {"by_seed": {int(s): float(x) for s, x in values.items()},
            "mean": float(v.mean()) if v.size else math.nan,
            "sd": float(v.std(ddof=1)) if v.size >= 2 else math.nan}
