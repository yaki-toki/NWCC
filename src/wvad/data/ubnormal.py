"""UBnormal (E024 stage 2): names, paths, the official split lists and frame labels.

Layout under ``raw_video_root``: ``UBnormal/Scene<s>/<name>.mp4`` and, for an abnormal video,
``UBnormal/Scene<s>/<name>_annotations/<name>_tracks.txt`` with one anomaly per line,
``object_id,start_frame,end_frame`` (end frame included; the official
``scripts/write_frame_level_ground_truth.py``). Frame labels are zeros of length T with ones on
every [start, end], clipped to [0, T) (``wvad.eval.metrics.frame_labels``, the rule of the
UCF-Crime and XD-Violence test manifests). Names are
``{normal|abnormal}_scene_<s>_scenario_<k>[_<j>]``; the label is the prefix.

The six official split lists (``scripts/*_video_names.txt`` of the upstream repository) are
pinned by sha256 and size: training 186 normal + 82 abnormal, validation 26 + 38, test 53 + 158.

Licence CC BY-NC-ND 4.0 (``license.txt`` of the archive and the repository): metrics are
published; features, scores and the parsed intervals are not redistributed. The test manifest
therefore stores no intervals (``eval_manifests.EXTERNAL_INTERVALS``); ``test_frame_gts`` reads
the tracks files at the sealed scoring and logs every access.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

DATASET = "ubn"
FOLDER = "UBnormal"
UPSTREAM = {"repository": "https://github.com/lilygeorgescu/UBnormal",
            "commit": "8c77642bb72615988ace0451b94ec42f8953a525", "checked": "2026-10-26",
            "paper": "Acsintoae et al., CVPR 2022, arXiv:2111.08644"}
LICENCE = "CC BY-NC-ND 4.0"
LICENCE_SHA256 = {"repository license.txt":
                  "727f1dcf53278cdad47f7a92ed47c3f9197e10e45f2c70acd21e787df07c3d2d",
                  "archive license.txt":
                  "4c397c016fde73e4d9bb523a0f820bc00cd7e4914832514b6f9aa616308b3a28"}
SPLITS = ("train", "val", "test")
LIST_FILES = {("train", 0): "normal_training_video_names.txt",
              ("train", 1): "abnormal_training_video_names.txt",
              ("val", 0): "normal_validation_video_names.txt",
              ("val", 1): "abnormal_validation_video_names.txt",
              ("test", 0): "normal_test_video_names.txt",
              ("test", 1): "abnormal_test_video_names.txt"}
LIST_SHA256 = {"normal_training_video_names.txt":
               "329813830c3fe5765ac35bb31a691b7ec69eb9c963174d14e643898aed59dd1e",
               "abnormal_training_video_names.txt":
               "79bd655775a6a5c24dfa04220f969ebc11790dc293bdea441bbc74f5dd55f145",
               "normal_validation_video_names.txt":
               "3112a825a325c900f5616c43837c7178bc087290eee35b9d544c61a332c85324",
               "abnormal_validation_video_names.txt":
               "9de909b5aa35b99f372829efb57326586b99367752c3c150946ed587bc27d580",
               "normal_test_video_names.txt":
               "c5adc274b9551773d9c6ebf61e92b56749e69b3a5e6e02de3149303bf44291f8",
               "abnormal_test_video_names.txt":
               "da4f0655ebfbcbb3123e320d20a97ffbeb81288fcc0a3c784bb9036133877035"}
LIST_COUNTS = {("train", 0): 186, ("train", 1): 82, ("val", 0): 26, ("val", 1): 38,
               ("test", 0): 53, ("test", 1): 158}
CATEGORY = {0: "normal", 1: "abnormal"}
NAME_RE = re.compile(r"^(normal|abnormal)_scene_([1-9][0-9]*)_scenario_([1-9][0-9]*)"
                     r"(?:_([0-9]+))?$")
TRACKS_SUFFIX = "_tracks.txt"
ANNOTATIONS_SUFFIX = "_annotations"


class UBnormalError(ValueError):
    """A list, name or tracks file is malformed or does not match the pinned facts."""


# --- names and paths ------------------------------------------------------------------------

def _match(name: str) -> re.Match:
    m = NAME_RE.match(name)
    if m is None:
        raise UBnormalError(f"not a UBnormal video name: {name!r}")
    return m


def label_of(name: str) -> int:
    """1 for ``abnormal_…``, 0 for ``normal_…``."""
    return 1 if _match(name).group(1) == "abnormal" else 0


def scene_of(name: str) -> int:
    return int(_match(name).group(2))


def rel_path(name: str) -> str:
    """``UBnormal/Scene<s>/<name>.mp4`` (posix, relative to raw_video_root)."""
    return str(PurePosixPath(FOLDER) / f"Scene{scene_of(name)}" / f"{name}.mp4")


def tracks_rel_path(name: str) -> str:
    """``UBnormal/Scene<s>/<name>_annotations/<name>_tracks.txt`` (abnormal videos only)."""
    if label_of(name) != 1:
        raise UBnormalError(f"{name}: a normal video has no tracks file")
    return str(PurePosixPath(FOLDER) / f"Scene{scene_of(name)}" / f"{name}{ANNOTATIONS_SUFFIX}"
               / f"{name}{TRACKS_SUFFIX}")


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- the official split lists ---------------------------------------------------------------

def read_list(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Video names of one list (one per line, blank lines ignored, order kept)."""
    names = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n]
    for n in names:
        _match(n)
    if len(set(names)) != len(names):
        raise UBnormalError(f"{path}: duplicate names")
    return tuple(names)


def read_lists(directory: str | os.PathLike[str], strict: bool = True) -> dict:
    """The six lists: ``{(split, label): {"file", "sha256", "names"}}``. ``strict`` checks the
    pinned sha256 and sizes. Always: every name carries the list's label and no name is in two
    lists."""
    d = Path(directory)
    out: dict = {}
    seen: dict[str, tuple[str, int]] = {}
    for key, fname in LIST_FILES.items():
        f = d / fname
        if not f.is_file():
            raise UBnormalError(f"missing list {f}")
        sha, names = sha256_file(f), read_list(f)
        if strict and sha != LIST_SHA256[fname]:
            raise UBnormalError(f"{f}: sha256 {sha} differs from the pinned {LIST_SHA256[fname]}")
        if strict and len(names) != LIST_COUNTS[key]:
            raise UBnormalError(f"{f}: {len(names)} names, the pinned list has {LIST_COUNTS[key]}")
        for n in names:
            if label_of(n) != key[1]:
                raise UBnormalError(f"{f}: {n} does not carry label {key[1]}")
            if n in seen:
                raise UBnormalError(f"{n} is in {seen[n]} and {key}")
            seen[n] = key
        out[key] = {"file": fname, "sha256": sha, "names": names}
    return out


def manifest_rows(lists: Mapping, order: Sequence[tuple[str, int]]) -> list[str]:
    """Names in the order of the given (split, label) lists (the manifest order)."""
    return [n for key in order for n in lists[key]["names"]]


# --- tracks and frame labels ----------------------------------------------------------------

def _int(token: str, where: str) -> int:
    t = token.strip()
    try:
        v = float(t)
    except ValueError as exc:
        raise UBnormalError(f"{where}: not a number: {t!r}") from exc
    if not v.is_integer():
        raise UBnormalError(f"{where}: not an integer: {t!r}")
    return int(v)


def parse_tracks(text: str, source: str = "<tracks>") -> tuple[tuple[int, int, int], ...]:
    """Rows ``object_id,start_frame,end_frame`` (ints; blank lines ignored; brackets and spaces
    tolerated). A row with another shape, a negative frame or end < start raises."""
    rows = []
    for k, line in enumerate(text.splitlines(), start=1):
        s = line.strip().strip("[]").strip()
        if not s:
            continue
        parts = [p for p in s.split(",") if p.strip()]
        if len(parts) != 3:
            raise UBnormalError(f"{source} line {k}: {len(parts)} fields, expected 3")
        obj, start, end = (_int(p, f"{source} line {k}") for p in parts)
        if start < 0 or end < start:
            raise UBnormalError(f"{source} line {k}: invalid frames {start}..{end}")
        rows.append((obj, start, end))
    if not rows:
        raise UBnormalError(f"{source}: no track")
    return tuple(rows)


def intervals_of(tracks: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """(start, end) of every track, end included; object ids dropped."""
    return tuple((int(t[1]), int(t[2])) for t in tracks)


def frame_labels(total_frames: int, tracks: Sequence[Sequence[int]]) -> np.ndarray:
    """0/1 frame labels (float32, as ``metrics.frame_labels``) of an abnormal video: ones on
    every track's [start, end], clipped to [0, T) as the official script does."""
    from wvad.eval.metrics import frame_labels as _frame_labels  # sklearn: scoring hosts only

    return _frame_labels(int(total_frames), intervals_of(tracks))


def read_tracks(path: str | os.PathLike[str]) -> tuple[tuple[int, int, int], ...]:
    return parse_tracks(Path(path).read_text(encoding="utf-8"), str(path))


def tracks_bounds(total_frames: int, tracks: Sequence[Sequence[int]]) -> dict:
    """Descriptive check of a tracks file against the video's frame count."""
    ends = [int(t[2]) for t in tracks]
    return {"n_tracks": len(tracks), "max_end": max(ends),
            "ends_beyond_video": int(sum(e >= int(total_frames) for e in ends))}


def test_frame_gts(video_root: str | os.PathLike[str], names: Sequence[str],
                   labels: Sequence[int], total_frames: Sequence[int],
                   access_log: str | os.PathLike[str] | None = None,
                   command: str = "tracks") -> tuple[list[np.ndarray | None], dict]:
    """Frame labels of the test videos from the tracks files (the sealed scoring only): zeros for
    a normal video, ``frame_labels`` for an abnormal one. Every tracks path is appended to
    ``access_log`` (purpose "test labels") before any is read. A missing or malformed tracks
    file does not stop the scoring: its video gets None and is listed in ``record["failed"]``
    (the plan excludes it from L and reports the name)."""
    root = Path(video_root)
    rows = [{"command": command, "dataset": DATASET, "index": i, "name": n,
             "path": tracks_rel_path(n), "purpose": "test labels"}
            for i, (n, y) in enumerate(zip(names, labels, strict=True)) if int(y) == 1]
    if access_log is not None and rows:
        from wvad.extract.dense import log_access, utc_now

        log_access(access_log, [{"utc": utc_now(), **r} for r in rows])
    gts: list[np.ndarray | None] = []
    record: dict = {"read": 0, "failed": {}, "sha256": {}, "bounds": {}}
    for n, y, t in zip(names, labels, total_frames, strict=True):
        if int(y) != 1:
            gts.append(np.zeros(int(t), dtype=np.float32))
            continue
        f = root / tracks_rel_path(n)
        try:
            tracks = read_tracks(f)
        except (OSError, UBnormalError) as exc:
            record["failed"][n] = str(exc)
            gts.append(None)
            continue
        record["read"] += 1
        record["sha256"][n] = sha256_file(f)
        record["bounds"][n] = tracks_bounds(int(t), tracks)
        gts.append(frame_labels(int(t), tracks))
    return gts, record
