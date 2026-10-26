"""UBnormal (E024 stage 2): manifests and split, tracks check, test-store extraction.

    python -m wvad.prep.ubnormal manifests --lists <dir> --access_log <csv> [--out_dir <repo>]
    python -m wvad.prep.ubnormal tracks-check --lists <dir> --access_log <csv> --out <json>
    python -m wvad.prep.ubnormal extract-test --access_log <csv> [--encoders ...] [--workers 32]
        [--chunk 32] [--batch xcb=8,xcl=8,siglip=4] [--normalize lut]

``manifests`` reads the six official lists (sha256 and sizes pinned in ``wvad.data.ubnormal``)
and the video headers (OpenCV frame count and fps, no frame decoded) of all 543 videos under
``raw_video_root/UBnormal`` and writes

    data/manifests/ubn_train.json      the official training then validation videos (332 rows:
                                       normal_training, abnormal_training, normal_validation,
                                       abnormal_validation), categories "normal" / "abnormal",
                                       rel_paths UBnormal/Scene<s>/<name>.mp4
    data/splits/ubn_val_official.json  val_indices = the 64 official validation videos
    data/manifests/ubn_test.json       the 211 test videos (normal_test, abnormal_test); no
                                       intervals (meta ``intervals_external``: the tracks files
                                       are read at the sealed scoring; CC BY-NC-ND 4.0)

No annotation is read by ``manifests``. Every video path is appended to the access log before
its header is opened (access type (0): listing, headers, sha256 only).

``tracks-check`` (access type (0g-ubn)) parses the tracks files of the validation abnormal
videos (format, bounds against the manifest frame counts) and, for the test abnormal videos,
records existence, size, sha256 and line count only, never a value. ``extract-test`` runs the
dense 16-frame extraction (``wvad.extract.dense``) of the test videos into
``<feature_root>/ubn_<encoder>_d16_test`` (phase "sample", pixels only).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from wvad.data import ubnormal as ub
from wvad.data.catalog import ENCODERS, dense_name
from wvad.data.eval_manifests import (
    EXTERNAL_INTERVALS,
    eval_manifest_file,
    load_eval_manifest,
    parse_eval_manifest,
)
from wvad.data.manifests import TrainManifest, load_manifest, manifest_file, parse_manifest
from wvad.data.splits import parse_split, split_sha256
from wvad.paths import REPO_ROOT, load_paths

TRAIN_ORDER = (("train", 0), ("train", 1), ("val", 0), ("val", 1))
TEST_ORDER = (("test", 0), ("test", 1))
TEST_SUFFIX = "_test"                  # <dataset>_<encoder>_d16_test (wvad.data.dense_test rule)
ACCESS_COMMAND = "ubn-prep"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_header(path: Path) -> tuple[int, float]:
    from wvad.extract.video import read_header as _read_header

    return _read_header(path)


def _log(access_log: str | None, rows: Sequence[dict]) -> None:
    if access_log is None or not rows:
        return
    from wvad.extract.dense import log_access

    log_access(access_log, [{"utc": utc_now(), **r} for r in rows])


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")


def _headers(video_root: Path, names: Sequence[str], access_log: str | None,
             split: str) -> tuple[list[int], list[float]]:
    """Frame counts and fps of ``names`` (paths logged before any header is opened)."""
    rels = [ub.rel_path(n) for n in names]
    _log(access_log, [{"command": ACCESS_COMMAND, "dataset": ub.DATASET, "index": i, "name": n,
                       "path": r, "purpose": f"header ({split})"}
                      for i, (n, r) in enumerate(zip(names, rels, strict=True))])
    frames, fps = [], []
    for r in rels:
        f = video_root / r
        if not f.is_file():
            raise ub.UBnormalError(f"missing video {f}")
        t, rate = read_header(f)
        if t < 1:
            raise ub.UBnormalError(f"{f}: header frame count {t}")
        frames.append(int(t))
        fps.append(float(rate))
    return frames, fps


def _source(lists) -> dict:
    return {**ub.UPSTREAM, "lists": {v["file"]: v["sha256"] for v in lists.values()},
            "licence": ub.LICENCE, "licence_sha256": ub.LICENCE_SHA256}


def build_manifests(lists_dir: Path, video_root: Path, out_dir: Path = REPO_ROOT,
                    access_log: str | None = None, strict: bool = True) -> dict:
    """Write the three files (module docstring) and return their paths and a summary."""
    import cv2  # the header reader's library, recorded in meta

    lists = ub.read_lists(lists_dir, strict=strict)
    created = utc_now()
    headers = f"cv2 {cv2.__version__} CAP_PROP_FRAME_COUNT, CAP_PROP_FPS"
    # training + validation manifest
    names = ub.manifest_rows(lists, TRAIN_ORDER)
    n_train = sum(len(lists[k]["names"]) for k in TRAIN_ORDER[:2])
    frames, fps = _headers(video_root, names, access_log, "train+val")
    labels = [ub.label_of(n) for n in names]
    train = {"meta": {"dataset": ub.DATASET, "split": "train+val", "n": len(names),
                      "n_train": n_train, "n_val": len(names) - n_train,
                      "source": _source(lists), "headers": headers,
                      "order": "official lists: "
                               + ", ".join(lists[k]["file"] for k in TRAIN_ORDER),
                      "created": created},
             "names": names, "categories": [ub.CATEGORY[y] for y in labels], "labels": labels,
             "total_frames": frames, "fps": fps, "rel_paths": [ub.rel_path(n) for n in names]}
    parse_manifest(train, "ubn_train")
    # the official validation split over that manifest
    val_idx, train_idx = list(range(n_train, len(names))), list(range(n_train))
    split = {"meta": {"source": "official validation lists (UBnormal scripts)",
                      "n_total": len(names), "n_val": len(val_idx),
                      "n_val_anom": int(sum(labels[i] for i in val_idx)),
                      "n_val_norm": int(sum(1 - labels[i] for i in val_idx)),
                      "split_sha256": split_sha256(train_idx, val_idx),
                      "labels_source": "ubn_train.json labels", "created": created},
             "val_indices": val_idx, "train_indices": train_idx, "labels": labels}
    parse_split(split, "ubn_val_official")
    # the test manifest, without intervals
    t_names = ub.manifest_rows(lists, TEST_ORDER)
    t_frames, t_fps = _headers(video_root, t_names, access_log, "test")
    t_labels = [ub.label_of(n) for n in t_names]
    test = {"meta": {"dataset": ub.DATASET, "split": "test", "n": len(t_names),
                     "source": _source(lists), "headers": headers,
                     "order": "official lists: "
                              + ", ".join(lists[k]["file"] for k in TEST_ORDER),
                     EXTERNAL_INTERVALS: {
                         "file": f"UBnormal/Scene<s>/<name>{ub.ANNOTATIONS_SUFFIX}/<name>"
                                 f"{ub.TRACKS_SUFFIX}",
                         "read": "at the sealed scoring of E024 stage 3 only "
                                 "(wvad.data.ubnormal.test_frame_gts, access logged)",
                         "reason": f"{ub.LICENCE}: the intervals are not redistributed"},
                     "created": created},
            "names": t_names, "categories": [ub.CATEGORY[y] for y in t_labels],
            "labels": t_labels, "intervals": [[] for _ in t_names], "total_frames": t_frames,
            "fps": t_fps}
    parse_eval_manifest(test, "ubn_test")
    files = {"train": out_dir / "data" / "manifests" / "ubn_train.json",
             "split": out_dir / "data" / "splits" / "ubn_val_official.json",
             "test": out_dir / "data" / "manifests" / "ubn_test.json"}
    _write_json(files["train"], train)
    _write_json(files["split"], split)
    _write_json(files["test"], test)
    return {"files": {k: str(v) for k, v in files.items()},
            "sha256": {k: ub.sha256_file(v) for k, v in files.items()},
            "n_train": n_train, "n_val": len(val_idx), "n_test": len(t_names),
            "split_sha256": split["meta"]["split_sha256"],
            "frames": {"train+val": int(sum(frames)), "test": int(sum(t_frames))},
            "fps": sorted({*fps, *t_fps})}


def tracks_check(lists_dir: Path, video_root: Path, access_log: str | None = None,
                 strict: bool = True, train_manifest: Path | None = None,
                 test_manifest: Path | None = None) -> dict:
    """Validation abnormal videos: parse the tracks and compare the bounds with the manifest
    frame counts. Test abnormal videos: existence, size, sha256 and line count only."""
    lists = ub.read_lists(lists_dir, strict=strict)
    tm = load_manifest(train_manifest or manifest_file(ub.DATASET))
    frames = dict(zip(tm.names, tm.total_frames, strict=True))
    out: dict = {"check": "E024 stage 2 (0g-ubn) tracks check", "utc": utc_now(),
                 "validation": {}, "test": {}}
    val_names = lists[("val", 1)]["names"]
    _log(access_log, [{"command": ACCESS_COMMAND, "dataset": ub.DATASET, "index": i, "name": n,
                       "path": ub.tracks_rel_path(n),
                       "purpose": "tracks format check (validation)"}
                      for i, n in enumerate(val_names)])
    problems = []
    for n in val_names:
        f = video_root / ub.tracks_rel_path(n)
        try:
            tracks = ub.read_tracks(f)
            b = ub.tracks_bounds(frames[n], tracks)
            out["validation"][n] = {**b, "ok": b["ends_beyond_video"] == 0}
            if b["ends_beyond_video"]:
                problems.append(f"{n}: {b['ends_beyond_video']} track(s) end beyond "
                                f"T={frames[n]}")
        except (OSError, ub.UBnormalError) as exc:
            out["validation"][n] = {"ok": False, "error": str(exc)}
            problems.append(f"{n}: {exc}")
    em = load_eval_manifest(test_manifest or eval_manifest_file(ub.DATASET))
    test_names = lists[("test", 1)]["names"]
    _log(access_log, [{"command": ACCESS_COMMAND, "dataset": ub.DATASET, "index": i, "name": n,
                       "path": ub.tracks_rel_path(n),
                       "purpose": "tracks existence, sha256, line count (test; no value read)"}
                      for i, n in enumerate(test_names)])
    missing = []
    for n in test_names:
        f = video_root / ub.tracks_rel_path(n)
        if not f.is_file():
            out["test"][n] = {"exists": False}
            missing.append(n)
            continue
        text = f.read_bytes()
        out["test"][n] = {"exists": True, "size": len(text), "sha256": ub.sha256_file(f),
                          "lines": sum(1 for ln in text.splitlines() if ln.strip())}
    out["summary"] = {"validation_videos": len(val_names),
                      "validation_ok": sum(1 for v in out["validation"].values() if v["ok"]),
                      "validation_problems": problems, "test_videos": len(test_names),
                      "test_missing": missing, "test_manifest_n": len(em),
                      "ok": not problems and not missing}
    return out


def extract_test(access_log: str, encoders: Sequence[str], workers: int, chunk: int,
                 batch: dict, normalize: str, paths_file: str | None = None) -> dict:
    """Dense extraction of the test videos into <feature_root>/ubn_<encoder>_d16_test."""
    from wvad.extract import dense
    from wvad.extract.__main__ import _device, _encoders, _provenance

    paths = load_paths(paths_file)
    m = load_eval_manifest(eval_manifest_file(ub.DATASET))
    if not m.intervals_external:
        raise ub.UBnormalError("the test manifest should store no intervals")
    tm = TrainManifest(m.names, m.categories, m.labels, m.total_frames, m.fps,
                       tuple(ub.rel_path(n) for n in m.names),
                       MappingProxyType({"dataset": ub.DATASET, "split": "test"}))
    layout = dense.Layout.from_manifest(ub.DATASET, tm)
    dirs = {e: paths.feature_root / f"{dense_name(ub.DATASET, e)}{TEST_SUFFIX}"
            for e in encoders}
    device = _device()
    encs = _encoders(list(encoders), device)
    commit, env = _provenance()
    result = dense.run_extraction(dirs, layout,
                                  dense.tasks_for(paths, ub.DATASET, tm, range(len(tm))),
                                  encs, device, "autocast", batch, "sample", access_log,
                                  "dense-test", workers, chunk, environment=env, commit=commit,
                                  normalize=normalize)
    summary = {k: result[k] for k in ("phase", "videos_requested", "videos_pending",
                                      "videos_finished", "snippets", "seconds", "errors",
                                      "worker_failures")}
    print(json.dumps(summary))
    if result["errors"] or result["worker_failures"]:
        raise SystemExit(f"{len(result['errors'])} videos failed, worker failures "
                         f"{result['worker_failures']} (left undone; see meta.json)")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    from wvad.extract.__main__ import NORMALIZE, PRODUCTION_BATCH, _batch

    p = argparse.ArgumentParser(prog="python -m wvad.prep.ubnormal",
                                description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifests", help="ubn_train.json, ubn_val_official.json, ubn_test.json")
    m.add_argument("--lists", required=True, help="folder of the six official list files")
    m.add_argument("--access_log", required=True)
    m.add_argument("--out_dir", default=str(REPO_ROOT), help="repository root (default)")
    m.add_argument("--paths", help="paths file (default configs/paths.yaml)")
    m.add_argument("--no_strict", action="store_true", help="tests: skip the pinned sha256/sizes")
    t = sub.add_parser("tracks-check", help="(0g-ubn) validation tracks parsed, test tracks "
                                            "existence/sha256/lines only")
    t.add_argument("--lists", required=True)
    t.add_argument("--access_log", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--paths")
    t.add_argument("--no_strict", action="store_true")
    x = sub.add_parser("extract-test", help="dense test store ubn_<encoder>_d16_test")
    x.add_argument("--access_log", required=True)
    x.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    x.add_argument("--workers", type=int, default=32)
    x.add_argument("--chunk", type=int, default=32)
    x.add_argument("--batch", type=_batch, default=_batch(PRODUCTION_BATCH))
    x.add_argument("--normalize", choices=NORMALIZE, default="lut")
    x.add_argument("--paths")
    a = p.parse_args(argv)
    if a.command == "manifests":
        root = load_paths(a.paths).video_root()
        res = build_manifests(Path(a.lists), root, Path(a.out_dir), a.access_log,
                              strict=not a.no_strict)
        print(json.dumps(res, indent=1))
    elif a.command == "tracks-check":
        root = load_paths(a.paths).video_root()
        res = tracks_check(Path(a.lists), root, a.access_log, strict=not a.no_strict)
        _write_json(Path(a.out), res)
        print(json.dumps(res["summary"], indent=1))
        if not res["summary"]["ok"]:
            raise SystemExit("tracks check: problems found (see the record)")
    else:
        extract_test(a.access_log, a.encoders, a.workers, a.chunk, a.batch, a.normalize, a.paths)


if __name__ == "__main__":
    main()
