"""Readouts of the E021 dense scores (PLAN.md "판독", "결과의 쓰임"): local analysis functions
computed from the stored offset-free snippet logits z and gamma (the server's frozen scoring
code, ``wvad.eval.dense``, writes them). This module is not part of the frozen extraction,
training and scoring code (PLAN.md two-stage code rule: the readout aggregation W / Wm / G may be
fixed between the stages; ``wvad.diagnostics.e021_gates.FROZEN_PATHS`` excludes it).

    W     u_k = mean of z over J_k (float64 -> float32), window logit u_k + c(v); M10:
          c(v) = gamma * float32(mean of all the video's snippet z in float64), in float32
    Wm    the same over pick_m(J_k, 2); M10 c(v) from the distinct picked snippets
    G     every snippet: z_j + c(v), c(v) from all snippets
The probability readout is sigma in fp32 (torch on the CPU: values above about 16.6 become
exactly 1.0f, as in the training records), the video score its maximum; the logit readout is
the maximum logit. Heads without an offset have c(v) = 0 (the logits are u_k, z_j).
Frame scores: snippet values repeated over their support (``snippet_to_frames``; the
confirmation plan's frame score of C), window values over the evaluator windows
(``window_to_frames`` = ``wvad.eval.metrics.expand_segments``, unchanged).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from wvad.eval.metrics import expand_segments
from wvad.extract import grid

READOUTS = ("W", "Wm", "G")


def mean32(values: np.ndarray) -> np.float32:
    """float64 mean of float32 values, stored as float32."""
    return np.float32(np.asarray(values, dtype=np.float32).mean(dtype=np.float64))


def offset(z_units: np.ndarray, gamma: float | None) -> np.float32:
    """c(v) = gamma * mean(z) (float32 product of float32 gamma and the float32 mean); 0 when
    the head has no offset."""
    if gamma is None:
        return np.float32(0.0)
    return np.float32(np.float32(gamma) * mean32(z_units))


def sigmoid32(x: np.ndarray) -> np.ndarray:
    """fp32 sigmoid on the CPU (torch), as the head's scores."""
    return torch.sigmoid(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()


@dataclass(frozen=True)
class Readout:
    logits: np.ndarray          # (32,) window logits (W, Wm) or (n_v,) snippet logits (G)
    offset: np.float32          # c(v) used (0 without an offset)

    @property
    def scores(self) -> np.ndarray:
        return sigmoid32(self.logits)

    @property
    def video_score(self) -> float:
        return float(self.scores.max())

    @property
    def video_logit(self) -> float:
        return float(self.logits.max())


def readout(z: np.ndarray, total_frames: int, gamma: float | None, kind: str) -> Readout:
    """The W, Wm or G readout of one video's snippet logits z (n_v,)."""
    z = np.asarray(z, dtype=np.float32)
    if len(z) != grid.n_snippets(total_frames):
        raise ValueError(f"{len(z)} snippet logits for T = {total_frames} "
                         f"({grid.n_snippets(total_frames)} snippets)")
    if kind == "G":
        c = offset(z, gamma)
        return Readout((z + c).astype(np.float32), c)
    if kind == "W":
        sets = grid.window_members(total_frames).sets
        c = offset(z, gamma)
    elif kind == "Wm":
        members = grid.m_members(total_frames)
        sets = members.sets
        c = offset(z[members.union()], gamma)
    else:
        raise ValueError(f"unknown readout {kind!r} (known: {READOUTS})")
    u = np.array([mean32(z[s]) for s in sets], dtype=np.float32)
    return Readout((u + c).astype(np.float32), c)


def window_readouts(z_list: Sequence[np.ndarray], totals: Sequence[int],
                    gamma: float | None, kind: str) -> np.ndarray:
    """(videos, 32) window logits of W or Wm (the analysis applies sigmoid32 or not)."""
    if kind not in ("W", "Wm"):
        raise ValueError("window_readouts takes W or Wm")
    return np.stack([readout(z, t, gamma, kind).logits for z, t in zip(z_list, totals,
                                                                       strict=True)])


def snippet_to_frames(values: np.ndarray, total_frames: int) -> np.ndarray:
    """(T,) frame values: each snippet's value over its support [16 j, min(16 j + 16, T))."""
    v = np.asarray(values)
    sup = grid.snippet_support(total_frames)
    if len(v) != len(sup):
        raise ValueError(f"{len(v)} values for {len(sup)} snippets")
    return np.repeat(v, sup[:, 1] - sup[:, 0])


def window_to_frames(values: np.ndarray, total_frames: int) -> np.ndarray:
    """(T,) frame values of 32 window values (the evaluator rule, expand_segments)."""
    return expand_segments(values, total_frames)
