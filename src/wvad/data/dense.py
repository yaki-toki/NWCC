"""Dense snippet caches of E021: memory-mapped reading and the scale-match map.

One folder per (dataset, encoder), ``<feature_root>/<dataset>_<encoder>_d16/`` (written by
``wvad.extract.dense`` and ``wvad.extract.derive``)::

    dense16/train/features.npy     (rows, D), native dtype (XCB / XCL float32, SigLIP float16):
                                   the 16-frame snippets of every training-split video in
                                   manifest order (video i: rows start[i] .. start[i] + n[i] - 1);
                                   created full size (NaN) by the first stage, filled per stage
    dense16/train/index.npz        per video: start, n (= ceil(T / 16)), total_frames (manifest
                                   T), header_frames, fps, grab_attempts, grab_successes,
                                   used_frames, replaced, replaced_tail, labels, done, phase;
                                   per row: snippet_replaced (replaced frames of its 8)
    dense16/train/meta.json        layout, model revision, grid, and one entry per extraction run
                                   (phase, commit, environment, precision, TF32, batch sizes)
    dense16/train/progress.jsonl   one line per finished video (the source of index.npz)
    <variant>/mil/train/data.pt    derived 32-unit caches (same format as the earlier caches)
    <variant>/meta.json            filled ("val" or "all"), rule, scale-match map (d16m, d16m2)

Reading converts to float32 and concatenates the encoders along the last dimension in the given
order, as ``wvad.data.features.load_feature_set`` does for the 32-unit caches; every encoder must
list the same videos (labels, T, snippet counts and row starts).

Scale match (PLAN.md "척도 맞춤"): per dimension, x' = mu_c + (sigma_c / sigma_V) (x - mu_V)
(ratio 1 where sigma_V = 0), computed in float64 and stored as float32. The same map is the
fixed input layer of the d16m / d16m2 heads when they score single snippets.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DENSE_SUBDIR = Path("dense16") / "train"
FEATURES_FILE = "features.npy"
INDEX_FILE = "index.npz"
META_FILE = "meta.json"
PROGRESS_FILE = "progress.jsonl"
SCALED_VARIANTS = ("d16m", "d16m2")
# per-video columns of index.npz (dtype); "snippet_replaced" is per row
VIDEO_COLUMNS = {"start": np.int64, "n": np.int64, "total_frames": np.int64,
                 "header_frames": np.int64, "fps": np.float64, "grab_attempts": np.int64,
                 "grab_successes": np.int64, "used_frames": np.int64, "replaced": np.int64,
                 "replaced_tail": np.int64, "labels": np.int64, "done": np.bool_,
                 "phase": np.int64}
PHASES = {"none": 0, "val": 1, "trainpart": 2, "sample": 3}     # index.npz "phase" codes


class DenseCacheError(ValueError):
    """A dense cache is missing, incomplete, or inconsistent with another encoder's."""


def store_dir(dense_dir: str | os.PathLike[str]) -> Path:
    """``<dense folder>/dense16/train`` (features, index, meta, progress)."""
    return Path(dense_dir) / DENSE_SUBDIR


def variant_dir(dense_dir: str | os.PathLike[str], variant: str) -> Path:
    return Path(dense_dir) / variant


def read_json(path: str | os.PathLike[str]) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class DenseIndex:
    columns: Mapping[str, np.ndarray]        # VIDEO_COLUMNS, one entry per video
    snippet_replaced: np.ndarray             # (rows,) uint8

    def __len__(self) -> int:
        return len(self.columns["n"])

    def __getattr__(self, name: str) -> np.ndarray:
        cols = self.__dict__.get("columns", {})
        if name in cols:
            return cols[name]
        raise AttributeError(name)

    @property
    def rows(self) -> int:
        return int(self.columns["n"].sum())


def read_index(dense_dir: str | os.PathLike[str]) -> DenseIndex:
    path = store_dir(dense_dir) / INDEX_FILE
    if not path.is_file():
        raise DenseCacheError(f"dense index not found: {path}")
    with np.load(path) as z:
        missing = [c for c in (*VIDEO_COLUMNS, "snippet_replaced") if c not in z]
        if missing:
            raise DenseCacheError(f"{path}: missing columns {missing}")
        cols = {c: np.asarray(z[c]).astype(t) for c, t in VIDEO_COLUMNS.items()}
        rep = np.asarray(z["snippet_replaced"]).astype(np.uint8)
    n = cols["n"]
    if np.any(cols["start"] != np.concatenate([[0], np.cumsum(n)[:-1]])) or len(rep) != n.sum():
        raise DenseCacheError(f"{path}: row starts / snippet rows inconsistent with n")
    return DenseIndex(cols, rep)


def write_index(dense_dir: str | os.PathLike[str], columns: Mapping[str, np.ndarray],
                snippet_replaced: np.ndarray) -> Path:
    """Write index.npz atomically (temporary file, then rename)."""
    path = store_dir(dense_dir) / INDEX_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **{c: np.asarray(columns[c]).astype(t) for c, t in VIDEO_COLUMNS.items()},
                 snippet_replaced=np.asarray(snippet_replaced, dtype=np.uint8))
    os.replace(tmp, path)
    return path


# --- scale match ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleMap:
    """Per-dimension affine map of one encoder: x' = mu_c + ratio (x - mu_v), float64."""

    mu_c: np.ndarray
    sigma_c: np.ndarray
    mu_v: np.ndarray
    sigma_v: np.ndarray

    @property
    def ratio(self) -> np.ndarray:
        safe = np.where(self.sigma_v == 0, 1.0, self.sigma_v)
        return np.where(self.sigma_v == 0, 1.0, self.sigma_c / safe)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """(..., D) values (stored dtype or float32) -> float32 after the map in float64."""
        return (self.mu_c + self.ratio * (np.asarray(x, dtype=np.float64) - self.mu_v)
                ).astype(np.float32)

    def to_json(self) -> dict:
        return {"mu_c": self.mu_c.tolist(), "sigma_c": self.sigma_c.tolist(),
                "mu_v": self.mu_v.tolist(), "sigma_v": self.sigma_v.tolist()}

    @classmethod
    def from_json(cls, raw: Mapping[str, Sequence[float]]) -> ScaleMap:
        arrays = [np.asarray(raw[k], dtype=np.float64) for k in ("mu_c", "sigma_c", "mu_v",
                                                                   "sigma_v")]
        if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 1:
            raise DenseCacheError("scale map arrays must be 1-D of one length")
        return cls(*arrays)

    @classmethod
    def fit(cls, centre_rows: np.ndarray, variant_rows: np.ndarray) -> ScaleMap:
        """Mean / population SD (float64) of the d16c rows and of the variant rows (training-part
        rows only, chosen by the caller). With equal row counts the ratio does not depend on
        the SD's ddof."""
        c = np.asarray(centre_rows, dtype=np.float64)
        v = np.asarray(variant_rows, dtype=np.float64)
        return cls(c.mean(axis=0), c.std(axis=0), v.mean(axis=0), v.std(axis=0))


def load_scale_map(variant_path: str | os.PathLike[str]) -> ScaleMap:
    """The map recorded in ``<variant>/meta.json`` (d16m / d16m2)."""
    meta = read_json(Path(variant_path) / META_FILE)
    if "scale" not in meta:
        raise DenseCacheError(f"{variant_path}: meta.json has no scale map")
    return ScaleMap.from_json(meta["scale"])


# --- reading ----------------------------------------------------------------------------------

class DenseSet:
    """Snippet features of one or more encoders over the training split (memory-mapped).

    ``scales`` (optional, one per encoder) applies each encoder's scale-match map before the
    concatenation (the input layer of a d16m / d16m2 head)."""

    def __init__(self, directories: Sequence[str | os.PathLike[str]],
                 scales: Sequence[ScaleMap | None] | None = None):
        if not directories:
            raise DenseCacheError("no dense directories given")
        self.directories = tuple(Path(d) for d in directories)
        self.indexes = tuple(read_index(d) for d in self.directories)
        self.arrays = tuple(self._open(d, ix) for d, ix in zip(self.directories, self.indexes,
                                                                 strict=True))
        self.scales = tuple(scales) if scales is not None else (None,) * len(self.directories)
        if len(self.scales) != len(self.directories):
            raise DenseCacheError(f"{len(self.scales)} scale maps for {len(self.directories)} "
                                  "encoders")
        first = self.indexes[0]
        for d, ix in zip(self.directories[1:], self.indexes[1:], strict=True):
            for col in ("start", "n", "total_frames", "labels"):
                if not np.array_equal(getattr(ix, col), getattr(first, col)):
                    raise DenseCacheError(f"{d}: '{col}' differs from {self.directories[0]}")
        for s, a in zip(self.scales, self.arrays, strict=True):
            if s is not None and len(s.mu_c) != a.shape[1]:
                raise DenseCacheError(f"scale map of dimension {len(s.mu_c)} for features of "
                                      f"dimension {a.shape[1]}")

    @staticmethod
    def _open(directory: Path, index: DenseIndex) -> np.ndarray:
        path = store_dir(directory) / FEATURES_FILE
        if not path.is_file():
            raise DenseCacheError(f"dense features not found: {path}")
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] != index.rows:
            raise DenseCacheError(f"{path}: shape {arr.shape} does not hold {index.rows} rows")
        return arr

    def __len__(self) -> int:
        return len(self.indexes[0])

    @property
    def labels(self) -> np.ndarray:
        return self.indexes[0].labels

    @property
    def total_frames(self) -> np.ndarray:
        return self.indexes[0].total_frames

    @property
    def counts(self) -> np.ndarray:
        return self.indexes[0].n

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(int(a.shape[1]) for a in self.arrays)

    @property
    def dim(self) -> int:
        return sum(self.dims)

    def done(self, video: int) -> bool:
        return all(bool(ix.done[video]) for ix in self.indexes)

    def video(self, video: int, snippets: Sequence[int] | np.ndarray | None = None
              ) -> torch.Tensor:
        """(n_v, D) float32 snippet features of one video (or the given snippet rows of it)."""
        i = int(video)
        if not self.done(i):
            raise DenseCacheError(f"video {i} is not extracted in every encoder")
        start, n = int(self.indexes[0].start[i]), int(self.indexes[0].n[i])
        if snippets is None:
            rows: slice | np.ndarray = slice(start, start + n)
        else:
            j = np.asarray(snippets, dtype=np.int64)
            if j.size and (j.min() < 0 or j.max() >= n):
                raise DenseCacheError(f"video {i}: snippet index out of range 0..{n - 1}")
            rows = start + j
        parts = []
        for arr, scale in zip(self.arrays, self.scales, strict=True):
            x = arr[rows]
            parts.append(torch.from_numpy(scale.apply(x) if scale is not None
                                          else np.array(x, dtype=np.float32)))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def matches(self, cache_features: Sequence[torch.Tensor], indices: np.ndarray,
                videos: Sequence[int]) -> bool:
        """The concatenated snippet rows at ``indices`` (videos, 32) equal the fused 32-unit
        features of ``load_feature_set`` (``FeatureSet.features``, float32) bit for bit: the
        same concatenation, for a selection variant (d16c, d16r*) of the same encoders."""
        return all(torch.equal(self.video(v, indices[v]), cache_features[v]) for v in videos)
