"""Per-video segment features cached by the extraction pipeline.

One directory per encoder holds ``mil/<split>/data.pt`` with

    features      list of (S, D_e) tensors, one per video (S = 32 segments)
    labels        list of video labels (0 normal, 1 anomalous)
    seg_labels    list of (S,) tensors (training videos: NaN, no temporal labels)
    total_frames  optional list of frame counts

Several encoders are fused by concatenating their features along the last dimension, in
the order given, after conversion to float32. Every encoder cache must list the same
videos (same count, same labels, same number of segments).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch


class FeatureCacheError(ValueError):
    """A cache is missing, malformed, or inconsistent with the other encoders."""


@dataclass(frozen=True)
class FeatureSet:
    features: tuple[torch.Tensor, ...]        # (S, D) float32 per video
    labels: tuple[int, ...]
    seg_labels: tuple[torch.Tensor, ...]
    total_frames: tuple[int | None, ...]
    sources: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def dim(self) -> int:
        return int(self.features[0].shape[-1]) if self.features else 0

    def subset(self, indices: Sequence[int]) -> FeatureSet:
        idx = [int(i) for i in indices]
        return FeatureSet(tuple(self.features[i] for i in idx),
                          tuple(self.labels[i] for i in idx),
                          tuple(self.seg_labels[i] for i in idx),
                          tuple(self.total_frames[i] for i in idx),
                          self.sources)


def cache_file(directory: str | os.PathLike[str], split: str) -> str:
    return os.path.join(os.fspath(directory), "mil", split, "data.pt")


def load_cache(directory: str | os.PathLike[str], split: str) -> dict:
    path = cache_file(directory, split)
    if not os.path.isfile(path):
        raise FeatureCacheError(f"feature cache not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in ("features", "labels", "seg_labels") if k not in data]
    if missing:
        raise FeatureCacheError(f"{path}: missing keys {missing}")
    if not (len(data["features"]) == len(data["labels"]) == len(data["seg_labels"])):
        raise FeatureCacheError(f"{path}: features/labels/seg_labels lengths differ")
    return data


def load_feature_set(directories: Sequence[str | os.PathLike[str]], split: str) -> FeatureSet:
    """Load one split of one or more encoder caches and fuse them by concatenation."""
    if not directories:
        raise FeatureCacheError("no feature directories given")
    caches = [load_cache(d, split) for d in directories]
    first = caches[0]
    labels = tuple(int(x) for x in first["labels"])
    for d, c in zip(directories[1:], caches[1:], strict=True):
        if tuple(int(x) for x in c["labels"]) != labels:
            raise FeatureCacheError(f"{cache_file(d, split)}: videos/labels differ from "
                                    f"{cache_file(directories[0], split)}")
    fused = []
    for i in range(len(labels)):
        parts = [c["features"][i].float() for c in caches]
        if any(p.dim() != 2 or p.shape[0] != parts[0].shape[0] for p in parts):
            raise FeatureCacheError(f"video {i}: segment tensors are not (S, D) with equal S")
        fused.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1))
    total = first.get("total_frames") or [None] * len(labels)
    return FeatureSet(tuple(fused), labels,
                      tuple(s.float() for s in first["seg_labels"]),
                      tuple(None if t is None else int(t) for t in total),
                      tuple(os.fspath(d) for d in directories))
