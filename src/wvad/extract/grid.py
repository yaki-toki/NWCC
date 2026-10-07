"""Snippet grid, evaluator windows and window assignment of the dense representation (E021).

This module is the single definition of every index rule of E021 (PLAN.md "조밀 표현의 정의",
"창·배정 규칙"): extraction, derived caches, dense scoring and the diagnostics use only these
functions. Frames are 0-based, intervals half-open [start, end), and T is always the manifest
``total_frames`` (PLAN.md R3; never the header count read on the extraction machine). Arrays
are int64 unless noted.

Constants
    SNIPPET = 16            frames per snippet (gap-free tiling of the video)
    CLIP_FRAMES = 8         encoder input frames per snippet
    CLIP_STRIDE = 2         frame step inside a snippet
    WINDOWS = 32            evaluator windows per video (= segments of the earlier caches)
    M_PICK = 2              size of the count-matched subsets (Bm, Wm)
    DRAW_SEED = 20261007    d16r draws use default_rng(DRAW_SEED + 3 r + ENCODER_INDEX[encoder])
    ENCODER_INDEX           {"xcb": 0, "xcl": 1, "siglip": 2}
    D16R_DRAWS              draws r per dataset: UCF (0, 1, 2) = d16r, d16r1, d16r2; XD (0,)

Members (window_members, m_members)
    .sets                   tuple of 32 int64 arrays (snippet indices in time order)
    .replaced               (32,) bool: empty window given its d16c snippet
    .sizes()                (32,) set sizes;  .union(): distinct snippets of all sets, sorted

Snippets of a video with T frames (T >= 1)
    n_snippets(T)           n_v = ceil(T / 16)
    snippet_frames(T)       (n_v, 8) encoder input frames min(16 j + 2 i, T - 1), i = 0..7
    snippet_support(T)      (n_v, 2) support [16 j, min(16 j + 16, T)); every frame lies in
                            exactly one support (the last snippet may be shorter)
    snippet_centres(T)      (n_v,) support start + floor(support length / 2)
    used_frames(T)          sorted distinct frames read by the encoder (every even frame, T - 1)
    frame_snippet(f)        snippet whose support holds frame(s) f, i.e. f // 16

Evaluator windows (T >= 32; the frames of wvad.eval.metrics.expand_segments)
    window_bounds(T)        (32, 2) [k L, (k + 1) L) with L = T // 32; the last window ends at T
    window_centres(T)       (32,) window start + floor(window length / 2)
    snippet_windows(T)      (n_v,) window holding each snippet centre (the snippet's assignment)
    window_members(T)       Members: J_k = snippets assigned to window k, in time order; an empty
                            window (only when L < 16) takes the snippet supporting its centre
                            frame (the d16c snippet), flagged in Members.replaced. The same J_k
                            serve the B / Bm features and the W / Wm readouts
    centre_snippets(T)      (32,) d16c: the snippet supporting each window centre frame
    start_snippets(T)       (32,) the snippet supporting each window start frame k L (XD d16r)
    pick_m(J, m=2)          count-matched subset: J[floor((i + 1/2) |J| / m)], i = 0..m-1, when
                            |J| >= m, else J itself
    m_members(T, m=2)       Members of pick_m applied to every J_k (after the empty-window rule)
    snippets_per_window(T)  q = n_v / 32 (the length strata of the LS-AUC)

d16r draws (training-split videos in manifest order; identical in both stages)
    d16r_snippets(dataset, totals, r, encoder)
                            list of (32,) snippet indices, one per video. UCF: one generator
                            default_rng(20261007 + 3 r + e) per (draw r, encoder) walks the
                            videos in manifest order and takes 32 uniforms per video, u_k for
                            window k in window order; window k gets J_k[floor(u_k |J_k|)]. Every
                            video consumes exactly 32 uniforms, so its draw depends only on its
                            manifest position and its own T. XD: r = 0 only, start_snippets
                            (no draw, shared by the encoders)

1-s grid (derived from the snippets, no re-extraction)
    one_second_snippets(T, fps)
                            (ceil(T / fps),) the snippet supporting frame floor(s fps) of every
                            second s
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SNIPPET = 16
CLIP_FRAMES = 8
CLIP_STRIDE = 2
WINDOWS = 32
M_PICK = 2
DRAW_SEED = 20261007
ENCODER_INDEX = {"xcb": 0, "xcl": 1, "siglip": 2}
D16R_DRAWS = {"ucf": (0, 1, 2), "xdv": (0,)}          # r of d16r (0), d16r1 and d16r2 (UCF)


def _frames(total_frames: int, minimum: int = 1) -> int:
    t = int(total_frames)
    if t < minimum:
        raise ValueError(f"total_frames must be >= {minimum}, got {total_frames}")
    return t


# --- snippets -------------------------------------------------------------------------------

def n_snippets(total_frames: int) -> int:
    """n_v = ceil(T / 16); a trailing partial snippet is kept."""
    return -(-_frames(total_frames) // SNIPPET)


def snippet_frames(total_frames: int) -> np.ndarray:
    """(n_v, 8) encoder input frames of every snippet: min(16 j + 2 i, T - 1) (the clip rule of
    the earlier caches: 8 frames, step 2, 16-frame span, clipped at the end)."""
    t = _frames(total_frames)
    starts = np.arange(n_snippets(t), dtype=np.int64)[:, None] * SNIPPET
    return np.minimum(starts + CLIP_STRIDE * np.arange(CLIP_FRAMES, dtype=np.int64), t - 1)


def snippet_support(total_frames: int) -> np.ndarray:
    """(n_v, 2) half-open support [16 j, min(16 j + 16, T))."""
    t = _frames(total_frames)
    starts = np.arange(n_snippets(t), dtype=np.int64) * SNIPPET
    return np.stack([starts, np.minimum(starts + SNIPPET, t)], axis=1)


def snippet_centres(total_frames: int) -> np.ndarray:
    """(n_v,) support start + floor(support length / 2)."""
    sup = snippet_support(total_frames)
    return sup[:, 0] + (sup[:, 1] - sup[:, 0]) // 2


def used_frames(total_frames: int) -> np.ndarray:
    """Sorted distinct frames the encoder reads (every even frame below T, and T - 1)."""
    return np.unique(snippet_frames(total_frames))


def frame_snippet(frames) -> np.ndarray:
    """Snippet index whose support holds each frame (frames >= 0)."""
    f = np.asarray(frames, dtype=np.int64)
    if np.any(f < 0):
        raise ValueError("frames must be >= 0")
    return f // SNIPPET


# --- evaluator windows ----------------------------------------------------------------------

def window_bounds(total_frames: int) -> np.ndarray:
    """(32, 2) evaluator windows: [k L, (k + 1) L), L = T // 32, the last one ending at T
    (T // 32 frames per window and the remainder in the last, as expand_segments)."""
    t = _frames(total_frames, WINDOWS)
    length = t // WINDOWS
    starts = np.arange(WINDOWS, dtype=np.int64) * length
    ends = starts + length
    ends[-1] = t
    return np.stack([starts, ends], axis=1)


def window_centres(total_frames: int) -> np.ndarray:
    """(32,) window start + floor(window length / 2)."""
    b = window_bounds(total_frames)
    return b[:, 0] + (b[:, 1] - b[:, 0]) // 2


def snippet_windows(total_frames: int) -> np.ndarray:
    """(n_v,) window of every snippet: the window holding its centre frame."""
    t = _frames(total_frames, WINDOWS)
    return np.minimum(snippet_centres(t) // (t // WINDOWS), WINDOWS - 1)


@dataclass(frozen=True)
class Members:
    """Per-window snippet sets of one video (each an int64 array in time order)."""

    sets: tuple[np.ndarray, ...]          # 32 arrays
    replaced: np.ndarray                  # (32,) bool: empty window given its d16c snippet

    def sizes(self) -> np.ndarray:
        return np.array([len(s) for s in self.sets], dtype=np.int64)

    def union(self) -> np.ndarray:
        """Distinct snippets of all windows, sorted (each counted once)."""
        return np.unique(np.concatenate(self.sets))


def centre_snippets(total_frames: int) -> np.ndarray:
    """(32,) d16c: the snippet supporting each window centre frame (shared by the encoders)."""
    return frame_snippet(window_centres(total_frames))


def start_snippets(total_frames: int) -> np.ndarray:
    """(32,) the snippet supporting each window start frame k L (XD d16r)."""
    return frame_snippet(window_bounds(total_frames)[:, 0])


def window_members(total_frames: int) -> Members:
    """J_k of every window (centre rule); an empty window takes its d16c snippet."""
    t = _frames(total_frames, WINDOWS)
    assign = snippet_windows(t)
    centre = centre_snippets(t)
    sets, replaced = [], np.zeros(WINDOWS, dtype=bool)
    for k in range(WINDOWS):
        members = np.flatnonzero(assign == k).astype(np.int64)
        if members.size == 0:
            members = centre[k:k + 1].copy()
            replaced[k] = True
        sets.append(members)
    return Members(tuple(sets), replaced)


def pick_m(members: np.ndarray, m: int = M_PICK) -> np.ndarray:
    """Count-matched subset: with |J| >= m the elements at floor((i + 1/2) |J| / m), i < m
    (J in time order); otherwise J itself."""
    j = np.asarray(members, dtype=np.int64)
    n = len(j)
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if n < m:
        return j.copy()
    return j[((2 * np.arange(m, dtype=np.int64) + 1) * n) // (2 * m)]


def m_members(total_frames: int, m: int = M_PICK) -> Members:
    """pick_m applied to every J_k of window_members (empty-window rule included)."""
    full = window_members(total_frames)
    return Members(tuple(pick_m(s, m) for s in full.sets), full.replaced.copy())


def snippets_per_window(total_frames: int) -> float:
    """q = ceil(T / 16) / 32."""
    return n_snippets(total_frames) / WINDOWS


# --- d16r draws -----------------------------------------------------------------------------

def d16r_generator(r: int, encoder: str) -> np.random.Generator:
    """default_rng(20261007 + 3 r + e), e = 0 XCB, 1 XCL, 2 SigLIP."""
    if encoder not in ENCODER_INDEX:
        raise KeyError(f"unknown encoder '{encoder}' (known: {', '.join(ENCODER_INDEX)})")
    return np.random.default_rng(DRAW_SEED + 3 * int(r) + ENCODER_INDEX[encoder])


def _ucf_draws(totals: Sequence[int], r: int, encoder: str) -> list[np.ndarray]:
    rng = d16r_generator(r, encoder)
    out = []
    for t in totals:
        u = rng.random(WINDOWS)                       # exactly 32 uniforms per video
        sets = window_members(t).sets
        picks = [s[min(int(uk * len(s)), len(s) - 1)] for uk, s in zip(u, sets, strict=True)]
        out.append(np.asarray(picks, dtype=np.int64))
    return out


def d16r_snippets(dataset: str, totals: Sequence[int], r: int, encoder: str
                  ) -> list[np.ndarray]:
    """(32,) d16r snippet indices per video; ``totals`` = manifest T of every training-split video
    in manifest order (the whole manifest, so both stages draw the same values)."""
    if dataset not in D16R_DRAWS:
        raise KeyError(f"unknown dataset '{dataset}'")
    if int(r) not in D16R_DRAWS[dataset]:
        raise ValueError(f"{dataset} has d16r draws {D16R_DRAWS[dataset]}, not r = {r}")
    if encoder not in ENCODER_INDEX:
        raise KeyError(f"unknown encoder '{encoder}' (known: {', '.join(ENCODER_INDEX)})")
    if dataset == "xdv":
        return [start_snippets(t) for t in totals]
    return _ucf_draws(totals, int(r), encoder)


# --- 1-s grid -------------------------------------------------------------------------------

def one_second_snippets(total_frames: int, fps: float) -> np.ndarray:
    """(ceil(T / fps),) snippet supporting frame floor(s * fps) of every second s."""
    t = _frames(total_frames)
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps must be finite and > 0, got {fps}")
    n = math.ceil(t / fps)
    frames = np.minimum(np.floor(np.arange(n) * float(fps)).astype(np.int64), t - 1)
    return frame_snippet(frames)
