"""Dense scoring and readouts of E021 (PLAN.md "판독", "결과의 쓰임", gate (s3)).

Scoring (server; frozen scoring code): a trained head scores single units, the dense snippets,
in eval mode, fp32, autocast off, through its own forward path (``SegmentHead`` scores every unit
from its own features, so a snippet's logit does not depend on the other units passed with it;
only the floating-point batch shape differs from the 32-unit path). Heads of the mean variants
(d16m, d16m2) see scale-matched snippets: build their ``DenseSet`` with ``scale_maps`` (the map
is the head's fixed input layer). ``snippet_logits`` returns the offset-free logits z; an M10
head's gamma is ``head_gamma``.

Readouts (local analysis; from stored logits z and gamma):
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
(s3): ``s3_check`` passes the scale-matched snippets averaged over J_k (d16m) or over the m-picks
(d16m2) through the head by the standard 32-unit path and compares z with the d16m (d16m2)
cache's: |dz| <= 1e-5 * max(1, |z|).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.amp import autocast

from wvad.data.dense import DenseSet, ScaleMap, load_scale_map, variant_dir
from wvad.data.features import FeatureSet
from wvad.eval.metrics import expand_segments
from wvad.eval.scoring import score
from wvad.extract import grid
from wvad.models.head import SegmentHead

ROWS_PER_PASS = 4096
READOUTS = ("W", "Wm", "G")
REL_TOL = 1e-5


# --- scoring --------------------------------------------------------------------------------

def head_gamma(model: SegmentHead) -> float | None:
    """The M10 offset scale gamma (None for a head without an offset); a stats head is not a
    dense-readout head."""
    if model.global_mode == "none":
        return None
    if model.global_mode != "mean":
        raise ValueError(f"dense readouts define c(v) for global_mode mean, not "
                         f"{model.global_mode!r}")
    return float(model.offset_gamma.detach().float().cpu())


def snippet_logits(model: SegmentHead, feats: torch.Tensor, device: torch.device,
                   rows: int = ROWS_PER_PASS) -> np.ndarray:
    """(n,) float32 offset-free logits z of n single units (n, D), in passes of ``rows`` units
    shaped (1, rows, D): eval mode, fp32, autocast off, no gradient."""
    head_gamma(model)                                   # rejects a stats head
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, feats.shape[0], rows):
            x = feats[a:a + rows].to(device).float().unsqueeze(0)
            with autocast(device_type=device.type, enabled=False):
                if model.global_mode == "none":
                    z = model(x)[1]
                else:
                    z = model._parts(x)[1]
            out.append(z[0].float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def scale_maps(dense_dirs: Sequence, variant: str) -> list[ScaleMap] | None:
    """The input maps of a d16m / d16m2 head (one per encoder folder); None for other heads."""
    if variant not in ("d16m", "d16m2"):
        return None
    return [load_scale_map(variant_dir(d, variant)) for d in dense_dirs]


def score_dense(model: SegmentHead, dense: DenseSet, videos: Sequence[int],
                device: torch.device) -> list[np.ndarray]:
    """Snippet logits z of every listed video (``dense`` carries the head's input map)."""
    return [snippet_logits(model, dense.video(v), device) for v in videos]


# --- readouts -------------------------------------------------------------------------------

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


# --- checks -------------------------------------------------------------------------------

def relative_dz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| / max(1, |b|) elementwise (float64)."""
    a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.abs(a64 - b64) / np.maximum(1.0, np.abs(b64))


def compare_logits(a: np.ndarray, b: np.ndarray, tol: float = REL_TOL) -> dict:
    """The (s2) / (s3) comparison: every |dz| <= tol * max(1, |z|)."""
    r = relative_dz(a, b)
    worst = float(r.max()) if r.size else 0.0
    return {"n": int(r.size), "max_rel_dz": worst,
            "pass": bool(r.size and math.isfinite(worst) and worst <= tol)}


def standard_z(model: SegmentHead, features: FeatureSet, device: torch.device) -> np.ndarray:
    """(videos, 32) offset-free logits of 32-unit features by the standard scoring path."""
    out = score(model, features, device)
    return out["local_logits"] if "local_logits" in out else out["logits"]


def averaged_units(dense: DenseSet, video: int, variant: str) -> torch.Tensor:
    """(32, D) float32: the video's (scale-matched) snippets averaged per window in float64,
    over J_k (d16m) or pick_m(J_k, 2) (d16m2), empty-window rule included."""
    t = int(dense.total_frames[video])
    sets = (grid.window_members(t) if variant == "d16m" else grid.m_members(t)).sets
    x = dense.video(video).numpy().astype(np.float64)
    return torch.from_numpy(np.stack([x[s].mean(axis=0) for s in sets]).astype(np.float32))


def s3_check(model: SegmentHead, dense: DenseSet, cache: FeatureSet, videos: Sequence[int],
             variant: str, device: torch.device, tol: float = REL_TOL) -> dict:
    """Gate (s3) of one head: ``dense`` built with the head's scale maps, ``cache`` the derived
    32-unit cache (``variant``) holding the same training-split videos (manifest order)."""
    if variant not in ("d16m", "d16m2"):
        raise ValueError("(s3) checks the d16m and d16m2 heads")
    if dense.scales is None or any(s is None for s in dense.scales):
        raise ValueError("(s3) needs the dense snippets with the head's scale maps")
    idx = [int(v) for v in videos]
    nan = torch.full((grid.WINDOWS,), float("nan"))
    averaged = FeatureSet(tuple(averaged_units(dense, v, variant) for v in idx),
                          tuple(int(dense.labels[v]) for v in idx), (nan,) * len(idx),
                          tuple(int(dense.total_frames[v]) for v in idx), ("dense",))
    z_dense = standard_z(model, averaged, device)
    z_cache = standard_z(model, cache.subset(idx), device)
    return {"variant": variant, "videos": len(idx), **compare_logits(z_dense, z_cache, tol)}
