"""Training-video manifests: name, category, label and frame count per cache index.

A manifest lists the training videos in the order of the training feature cache (which
stores no names). Columns are parallel lists:

    names          video name without extension
    categories     UCF-Crime category ("Normal" for normal videos) / XD-Violence label code
    labels         0 normal, 1 anomalous
    total_frames   frame count read from the video header (as at feature extraction)
    fps            frame rate from the video header (UCF; null when not recorded)
    rel_paths      path of the video inside the dataset root (UCF; null when not recorded)

Only training videos are listed; test videos never appear in a training manifest. The test-set
manifests of the confirmation experiment (stage F) are separate files read by
``wvad.data.eval_manifests``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from wvad.paths import REPO_ROOT

MANIFEST_FILES = {"ucf": "data/manifests/ucf_train.json",
                  "xdv": "data/manifests/xdv_train.json",
                  # E024 stage 2: the official UBnormal training and validation videos (332),
                  # categories "normal" / "abnormal", rel_paths UBnormal/Scene<s>/<name>.mp4
                  "ubn": "data/manifests/ubn_train.json"}
COLUMNS = ("names", "categories", "labels", "total_frames", "fps", "rel_paths")


class ManifestError(ValueError):
    """A manifest file is malformed or inconsistent."""


@dataclass(frozen=True)
class TrainManifest:
    names: tuple[str, ...]
    categories: tuple[str, ...]
    labels: tuple[int, ...]
    total_frames: tuple[int, ...]
    fps: tuple[float | None, ...]
    rel_paths: tuple[str | None, ...]
    meta: Mapping[str, object]

    def __len__(self) -> int:
        return len(self.names)

    def index_of(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}


def manifest_file(dataset: str) -> Path:
    if dataset not in MANIFEST_FILES:
        raise KeyError(f"no manifest for dataset '{dataset}' (known: {', '.join(MANIFEST_FILES)})")
    return REPO_ROOT / MANIFEST_FILES[dataset]


def parse_manifest(raw: Mapping[str, object], source: str = "<manifest>") -> TrainManifest:
    try:
        names = tuple(str(x) for x in raw["names"])  # type: ignore[union-attr]
        n = len(names)
        cols = {c: list(raw.get(c) or [None] * n) for c in COLUMNS[1:]}  # type: ignore[union-attr]
        labels = tuple(int(x) for x in cols["labels"])
        total = tuple(int(x) for x in cols["total_frames"])
        fps = tuple(None if x is None else float(x) for x in cols["fps"])
        meta = dict(raw.get("meta") or {})  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"{source}: malformed manifest ({exc})") from exc
    for c, values in cols.items():
        if len(values) != n:
            raise ManifestError(f"{source}: column '{c}' has {len(values)} rows for {n} names")
    if len(set(names)) != n:
        raise ManifestError(f"{source}: duplicate video names")
    if any(x not in (0, 1) for x in labels):
        raise ManifestError(f"{source}: labels must be 0 or 1")
    if any(t < 1 for t in total):
        raise ManifestError(f"{source}: total_frames must be positive")
    return TrainManifest(names, tuple(str(x) for x in cols["categories"]), labels, total, fps,
                         tuple(None if x is None else str(x) for x in cols["rel_paths"]),
                         MappingProxyType(meta))


def load_manifest(path: str | os.PathLike[str]) -> TrainManifest:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{path}: top level must be a mapping")
    return parse_manifest(raw, source=str(path))


def manifest_to_json(m: TrainManifest) -> dict:
    return {"meta": dict(m.meta), "names": list(m.names), "categories": list(m.categories),
            "labels": list(m.labels), "total_frames": list(m.total_frames), "fps": list(m.fps),
            "rel_paths": list(m.rel_paths)}
