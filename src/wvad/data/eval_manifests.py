"""Test-set manifests (stage F): name, category, label, annotated intervals and frame count
per row of the test feature cache, which stores no names.

The test set stays sealed until the confirmation experiment: these manifests carry only the
dataset's own annotations, never model scores. Columns are parallel lists:

    names          video name without extension
    categories     UCF-Crime category ("Normal" for normal videos) / XD-Violence label code
    labels         0 normal, 1 anomalous
    intervals      annotated anomalous intervals, [start, end] frames with the end included
    total_frames   frame count (UCF: video header as at extraction; XD: the cache's own count)
    fps            frame rate
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from wvad.paths import REPO_ROOT

EVAL_MANIFEST_FILES = {"ucf": "data/manifests/ucf_test.json",
                       "xdv": "data/manifests/xdv_test.json",
                       "ubn": "data/manifests/ubn_test.json"}      # E024 stage 2 (UBnormal)
COLUMNS = ("names", "categories", "labels", "intervals", "total_frames", "fps")
# meta key of a manifest whose intervals are not stored in the file (UBnormal, CC BY-NC-ND: the
# tracks files are read by the dataset's own reader at the sealed scoring); every row's
# intervals column is then empty and ``frame_gts`` refuses.
EXTERNAL_INTERVALS = "intervals_external"


class EvalManifestError(ValueError):
    """A test manifest file is malformed or inconsistent."""


@dataclass(frozen=True)
class EvalManifest:
    names: tuple[str, ...]
    categories: tuple[str, ...]
    labels: tuple[int, ...]
    intervals: tuple[tuple[tuple[int, int], ...], ...]
    total_frames: tuple[int, ...]
    fps: tuple[float, ...]
    meta: Mapping[str, object]

    def __len__(self) -> int:
        return len(self.names)

    @property
    def intervals_external(self) -> bool:
        return bool(self.meta.get(EXTERNAL_INTERVALS))

    def frame_gts(self) -> list[np.ndarray]:
        """0/1 frame labels per video (``wvad.eval.metrics.frame_labels``)."""
        if self.intervals_external:
            raise EvalManifestError(f"intervals are external ({EXTERNAL_INTERVALS}): use the "
                                    "dataset's own label reader")
        from wvad.eval.metrics import frame_labels  # sklearn: not needed to read a manifest

        return [frame_labels(t, iv) for t, iv in zip(self.total_frames, self.intervals,
                                                     strict=True)]


def eval_manifest_file(dataset: str) -> Path:
    if dataset not in EVAL_MANIFEST_FILES:
        raise KeyError(f"no test manifest for dataset '{dataset}' "
                       f"(known: {', '.join(EVAL_MANIFEST_FILES)})")
    return REPO_ROOT / EVAL_MANIFEST_FILES[dataset]


def parse_eval_manifest(raw: Mapping[str, object], source: str = "<manifest>") -> EvalManifest:
    try:
        names = tuple(str(x) for x in raw["names"])  # type: ignore[union-attr]
        cols = {c: list(raw[c]) for c in COLUMNS[1:]}  # type: ignore[index]
        labels = tuple(int(x) for x in cols["labels"])
        intervals = tuple(tuple((int(s), int(e)) for s, e in v) for v in cols["intervals"])
        total = tuple(int(x) for x in cols["total_frames"])
        fps = tuple(float(x) for x in cols["fps"])
        meta = dict(raw.get("meta") or {})  # type: ignore[union-attr]
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalManifestError(f"{source}: malformed test manifest ({exc})") from exc
    n = len(names)
    for c, values in cols.items():
        if len(values) != n:
            raise EvalManifestError(f"{source}: column '{c}' has {len(values)} rows for {n}")
    if len(set(names)) != n:
        raise EvalManifestError(f"{source}: duplicate video names")
    if any(x not in (0, 1) for x in labels):
        raise EvalManifestError(f"{source}: labels must be 0 or 1")
    if any(t < 1 for t in total):
        raise EvalManifestError(f"{source}: total_frames must be positive")
    external = bool(meta.get(EXTERNAL_INTERVALS))
    for i, (y, iv) in enumerate(zip(labels, intervals, strict=True)):
        if external:
            if iv:
                raise EvalManifestError(f"{source}: row {i} stores intervals although the "
                                        f"manifest marks them {EXTERNAL_INTERVALS}")
            continue
        if bool(iv) != bool(y):
            raise EvalManifestError(f"{source}: row {i} label {y} with {len(iv)} intervals")
        if any(s < 0 or e < s for s, e in iv):
            raise EvalManifestError(f"{source}: row {i} has an invalid interval {iv}")
    return EvalManifest(names, tuple(str(x) for x in cols["categories"]), labels, intervals,
                        total, fps, MappingProxyType(meta))


def load_eval_manifest(path: str | os.PathLike[str]) -> EvalManifest:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, Mapping):
        raise EvalManifestError(f"{path}: top level must be a mapping")
    return parse_eval_manifest(raw, source=str(path))


def eval_manifest_to_json(m: EvalManifest) -> dict:
    return {"meta": dict(m.meta), "names": list(m.names), "categories": list(m.categories),
            "labels": list(m.labels), "intervals": [[list(x) for x in v] for v in m.intervals],
            "total_frames": list(m.total_frames), "fps": list(m.fps)}
