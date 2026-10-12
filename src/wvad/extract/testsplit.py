"""E022 test-split dense extraction (PLAN.md "시험 특징", "실행 순서와 시험셋 봉인" 3, 8 and 9,
"관문", "구현"). New code only (add-only rule): the frozen E021 functions are called unchanged.

    python -m wvad.extract.testsplit inputs --out <results/e022_train_inputs.json> [--e021_runs D]
        3a, not a test access: sha256 of the 42 E021 training inputs, the six scale-map hashes,
        the data_sha256 self-consistency and the gate (c) config checkpoints (``--e021_runs``)
    python -m wvad.extract.testsplit check --test_log LOG [--out JSON] [--listing FILE ...]
                                           [--e020_runs D] [--e020_scores D]
        test access (0), before the freeze: the restricted read of both test manifests, their
        bytes against E020's git blobs and the annotation files against E020's sha256, the test
        path rule against this machine's file listing (names only; ``--listing`` adds a list
        made elsewhere): the match count and the unmatched names; the E020 m10 chain files and
        test score files (bytes only) against the E020 records
    python -m wvad.extract.testsplit header --test_log LOG --video_log CSV --out CSV [--sha_only]
        gate (y3): header frame counts (no decoding) and sha256 of every test video
    python -m wvad.extract.testsplit dense --test_log LOG --video_log CSV --access_log CSV
                                           [--repro SCRATCH]
        production extraction of the test videos into <feature_root>/<ds>_<enc>_d16_test with
        the frozen run_extraction (phase "sample", command "e022-test-dense", commit mapping
        with split "test"); ``--repro``: gate (y2), the E021 throughput sample (10 validation
        videos per dataset) extracted again into SCRATCH by the same command and compared with
        the E021 rows (its own access CSV, command "e022-y2"); the production extraction opens
        no test video unless --record shows (y2) and (y3) passed for each dataset and this
        environment passes (y1), or the user approved the exception
        (results/e022_approvals.json; PLAN.md 8-9 and "시험 추출 관문")
    python -m wvad.extract.testsplit derive --test_log LOG [--overwrite REASON]
        the derived test caches d16c and d16m (<variant>/mil/test/data.pt with their
        e022_test_derive.json; the frozen pure functions of wvad.extract.derive, the E021 d16m
        scale map read only) and the label-free distribution statistics of descriptive 11

header, dense and derive take --datasets / --encoders (default all), --record (the extraction
gate record, results/e022_extract_gate.json: summaries merged under their keys) and the access
device: --selection_sha256 (required), --freeze_commit and --lineage (a deployed copy),
--train_inputs (the committed record). Every call checks items 1-6, dense (production) and
derive also item 8 at their start and again at their end. ``--paths`` names the paths file.

Test access log (--test_log, outside the repository, appended): a "refused" line when the access
device stops a call (it has then hashed test files at most), a "start" line before any test file
is read or opened, and an "end" line (or "error") with the outcome. Video files opened are
appended to --video_log (path, size, sha256, time) before extraction; run_extraction keeps its
own access CSV (--access_log). Module-level imports are light (stdlib and numpy): the spawned
decode workers import this module again.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt_time
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

DATASETS = ("ucf", "xdv")
ENCODERS = ("xcb", "xcl", "siglip")
PAIRS = ("xcb_xcl", "xcb_siglip")
PRODUCTION_BATCH = "xcb=8,xcl=8,siglip=4"
NORMALIZE = ("lut", "numpy")
Y2_COMMAND = "e022-y2"                                  # access-CSV command of gate (y2)
Y2_L2, Y2_COS = 1e-3, 0.99999                           # gate (y2) tolerance per snippet
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)                 # as E021 descriptive 7
VIDEO_FIELDS = ("utc", "command", "dataset", "index", "name", "path", "size", "sha256", "purpose")
HEADER_FIELDS = ("dataset", "index", "name", "path", "size", "sha256", "header_frames",
                 "manifest_frames", "delta", "fps", "utc")
E020_SEEDS = (42, 123, 777, 1000, 2025, 7, 13, 99, 555, 31337)
E020_RULE_FILES = {"best": "ckpt_best_val.pt", "last": "ckpt_last.pt"}
E020_CHAIN_FILES = ("ckpt_best_val.pt", "ckpt_last.pt", "selection.json")
APPROVALS_FILE = "e022_approvals.json"                  # results/: user-approved gate exceptions


def utc_now() -> str:
    return dt_time.datetime.now(dt_time.timezone.utc).isoformat(timespec="seconds")


def _sha256(path: str | os.PathLike[str]) -> str:
    from wvad.prep.manifests import sha256_file

    return sha256_file(path)


def write_json(path: str | os.PathLike[str], obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def merge_record(path: str | os.PathLike[str] | None, key: str, value) -> None:
    """Set ``key`` of the JSON gate record (created when missing)."""
    if path is None:
        return
    p = Path(path)
    record = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    record[key] = value
    write_json(p, record)


def append_rows(path: str | os.PathLike[str], fields: Sequence[str],
                rows: Sequence[Mapping]) -> None:
    """Append CSV rows (header once), flushed to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
        fh.flush()
        os.fsync(fh.fileno())


def _paths(path: str | None = None):
    from wvad.paths import load_paths

    return load_paths(path)


# --- 3a: training inputs --------------------------------------------------------------------

def record_inputs(paths, out: str | os.PathLike[str],
                  e021_runs: str | os.PathLike[str] | None = None) -> dict:
    """Write the training-input record under 'train_inputs' of ``out`` (other keys kept)."""
    from wvad.data import dense_test as dt
    from wvad.diagnostics.common import code_version

    rec = dt.train_input_record(paths, e021_runs) | {"utc": utc_now(), "code": code_version()}
    merge_record(out, "train_inputs", rec)
    return rec


# --- access (0) -----------------------------------------------------------------------------

def list_files(root: str | os.PathLike[str], rel_root: str) -> set[str]:
    """Relative posix paths (to ``root``) of the files under root/rel_root: names only, no file
    is opened."""
    base, out = Path(root) / rel_root, set()
    for folder, _, files in os.walk(base):
        for f in files:
            out.add((Path(folder) / f).relative_to(root).as_posix())
    return out


def read_listing(path: str | os.PathLike[str]) -> set[str]:
    """A listing made elsewhere: one path relative to raw_video_root per line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {PurePosixPath(x.strip().replace("\\", "/")).as_posix() for x in lines if x.strip()}


def match_listing(expected: Sequence[str], listed: set[str], scope: str) -> dict:
    """The test path rule against a listing: matched count and unmatched names only; files of
    the same name elsewhere in the scope are counted (the rule picks one path)."""
    inside = {p for p in listed if p == scope or p.startswith(scope + "/")}
    names = Counter(PurePosixPath(p).name for p in inside)
    matched = [p for p in expected if p in inside]
    return {"expected": len(expected), "matched": len(matched),
            "unmatched": sorted(PurePosixPath(p).stem for p in expected if p not in inside),
            "same_name_elsewhere": sum(names[PurePosixPath(p).name] > 1 for p in matched),
            "ok": len(matched) == len(expected) == len(set(expected))}


def read_sha256_list(path: str | os.PathLike[str]) -> dict[str, str]:
    """sha256sum format: '<sha256>  <relative path>' per line."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, rel = line.split(maxsplit=1)
            out[rel.strip()] = sha
    return out


def e020_chains() -> list[str]:
    return [f"{d}_{p}_s{s}" for d in DATASETS for p in PAIRS for s in E020_SEEDS]


def _hash_or_none(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def e020_run_files(runs: str | os.PathLike[str]) -> dict:
    """sha256 of the E020 m10 chain files (checkpoints, selection.json) of the 40 chains."""
    files = {f"{c}/m10/{f}": _hash_or_none(Path(runs) / c / "m10" / f)
             for c in e020_chains() for f in E020_CHAIN_FILES}
    return {"root": str(runs), "files": files,
            "missing": sorted(k for k, v in files.items() if v is None)}


def e020_score_files(scores: str | os.PathLike[str], run_files: Mapping[str, str | None] | None,
                     e020_results: str | os.PathLike[str]) -> dict:
    """The E020 m10 test score files (bytes only) against results/e020_test_scores.sha256; the
    m10 validation score files against the list gate (s) bound (its sha256 in e020_gate.json)
    and, with the chain files, each checkpoint against the ckpt_sha256 they recorded."""
    res = Path(e020_results)
    listed = read_sha256_list(res / "e020_test_scores.sha256")
    test = {}
    for c in e020_chains():
        for rule in E020_RULE_FILES:
            rel = f"{c}/m10_{rule}_i.npz"
            got = _hash_or_none(Path(scores) / "test" / rel)
            test[rel] = {"sha256": got, "e020": listed.get(rel),
                         "ok": got is not None and got == listed.get(rel)}
    out = {"root": str(scores), "test_scores": test,
           "test_scores_ok": all(v["ok"] for v in test.values())}
    gate = json.loads((res / "e020_gate.json").read_text(encoding="utf-8"))
    val_list = Path(scores) / "val" / "scores.sha256"
    out["val_list"] = {"sha256": _hash_or_none(val_list),
                       "e020": gate["s"]["val_list_sha256"]}
    out["val_list"]["ok"] = out["val_list"]["sha256"] == out["val_list"]["e020"]
    if run_files is None or not out["val_list"]["ok"]:
        return out
    val_listed = read_sha256_list(val_list)
    ckpts = {}
    for c in e020_chains():
        for rule, name in E020_RULE_FILES.items():
            rel, key = f"{c}/m10_{rule}_i.npz", f"{c}/m10/{name}"
            vf = Path(scores) / "val" / rel
            in_list = vf.is_file() and _sha256(vf) == val_listed.get(rel)
            want = None
            if in_list:
                with np.load(vf) as z:                  # a validation score file
                    want = str(z["ckpt_sha256"])
            ckpts[key] = {"val_file_listed": in_list, "e020_ckpt_sha256": want,
                          "sha256": run_files.get(key), "ok": in_list and want is not None
                          and run_files.get(key) == want}
    out["checkpoints"] = ckpts
    out["checkpoints_ok"] = all(v["ok"] for v in ckpts.values())
    return out


def access_zero(paths, test_log: str | os.PathLike[str], *, listings: Sequence[str] = (),
                e020_runs: str | os.PathLike[str] | None = None,
                e020_scores: str | os.PathLike[str] | None = None,
                plan_dir=None, e020_results=None) -> dict:
    """Test access (0) (PLAN.md 3b): refused unless PLAN.md records D1 as "승인"; one "start"
    line before any test file is read, one "end" line with the outcome."""
    from wvad.data import dense_test as dt
    from wvad.diagnostics.common import code_version

    plan_dir = Path(plan_dir or dt.E022_DIR)
    e020_results = Path(e020_results or dt.REPO_ROOT / "experiments" / "E020_confirmation_f1"
                        / "results")
    dt.outside_repo(test_log)
    approval = dt.d1_approval(plan_dir)
    start = {"time_utc": utc_now(), "access": "0", "command": "check", "event": "start",
             "code": code_version(), "plan_sha256_now": dt.plan_sha256(plan_dir / "PLAN.md"),
             "approval": approval, "purpose": "pre-freeze list and hash check (PLAN.md 3b)",
             "files": {"manifests": list(dt.MANIFEST_FILES.values()),
                       "annotations": list(dt.ANNOTATION_FILES.values()),
                       "listing_roots": list(dt.LISTING_ROOTS.values()),
                       "listings": [str(x) for x in listings],
                       "e020_runs": None if e020_runs is None else str(e020_runs),
                       "e020_scores": None if e020_scores is None else str(e020_scores)}}
    dt.append_test_log(test_log, start)
    try:
        out = _zero_body(paths, dt, listings, e020_runs, e020_scores, e020_results)
    except BaseException as exc:
        dt.append_test_log(test_log, {"time_utc": utc_now(), "access": "0", "command": "check",
                                      "event": "error", "error": repr(exc)})
        raise
    dt.append_test_log(test_log, {"time_utc": utc_now(), "access": "0", "command": "check",
                                  "event": "end", "result": out})
    return out


def _zero_body(paths, dt, listings, e020_runs, e020_scores, e020_results) -> dict:
    hashes = dt.manifest_hashes(paths)
    extra = set().union(*(read_listing(x) for x in listings)) if listings else None
    out: dict = {}
    for d in DATASETS:
        m = dt.read_manifest(d)
        want = dt.E020_TEST[d]
        expected = dt.video_rel_paths(m)
        local = list_files(paths.video_root(), dt.LISTING_ROOTS[d])
        out[d] = {"manifest": m.summary(),
                  "counts_ok": (len(m), m.anomalous, m.frames) == (want["n"], want["anomalous"],
                                                                   want["frames"]),
                  "hashes": hashes[d],
                  "listing": match_listing(expected, local, dt.LISTING_ROOTS[d])}
        if extra is not None:
            out[d]["listing_given"] = match_listing(expected, extra, dt.LISTING_ROOTS[d])
    checks = [out[d]["counts_ok"] and out[d]["listing"]["ok"]
              and all(out[d]["hashes"][k]["ok"] for k in ("manifest", "provenance",
                                                          "annotation"))
              and out[d].get("listing_given", {"ok": True})["ok"] for d in DATASETS]
    if e020_runs is not None:
        out["e020_runs"] = e020_run_files(e020_runs)
        checks.append(not out["e020_runs"]["missing"])
    if e020_scores is not None:
        run_files = out["e020_runs"]["files"] if e020_runs is not None else None
        s = out["e020_scores"] = e020_score_files(e020_scores, run_files, e020_results)
        checks += [s["test_scores_ok"], s["val_list"]["ok"], s.get("checkpoints_ok", True)]
    out["ok"] = all(checks)
    return out


# --- test videos ----------------------------------------------------------------------------

def video_tasks(paths, manifest) -> list:
    """Frozen VideoTask of every test video (manifest order; paths from the test rule only)."""
    from wvad.data import dense_test as dt
    from wvad.extract.dense import VideoTask

    root = paths.video_root()
    return [VideoTask(i, os.fspath(root / rel), int(t), name, rel)
            for i, (name, t, rel) in enumerate(zip(manifest.names, manifest.total_frames,
                                                   dt.video_rel_paths(manifest), strict=True))]


def check_access_csv(path: str | os.PathLike[str], command: str) -> None:
    """run_extraction's access CSV of one purpose only: every existing row has ``command`` (the
    production CSV holds "e022-test-dense" rows only, gate (y7); (y2) keeps its own)."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return
    with open(p, newline="", encoding="utf-8") as fh:
        other = sorted({r.get("command") for r in csv.DictReader(fh)} - {command})
    if other:
        raise SystemExit(f"{p} holds rows of {other}: use a separate access CSV for {command}")


def pending_tasks(tasks: Sequence, dirs: Mapping[str, Path]) -> list:
    """The tasks some encoder has not finished (as run_extraction decides from progress)."""
    from wvad.extract.dense import read_progress

    done = {e: set(read_progress(d)) for e, d in dirs.items()}
    return [t for t in tasks if any(t.index not in done[e] for e in dirs)]


def hash_videos(paths, dataset: str, items: Sequence[tuple[int, str, str]], allowed: set[str],
                video_log: str | os.PathLike[str], command: str, purpose: str) -> list[dict]:
    """(index, name, relative path) of videos about to be opened: each path must be in
    ``allowed`` (the positive check), then its size and sha256 are appended to the video log."""
    outside = [rel for _, _, rel in items if rel not in allowed]
    if outside:
        raise SystemExit(f"{len(outside)} paths outside the allowed set: {outside[:5]}")
    rows = []
    for index, name, rel in items:
        path = paths.video_root() / rel
        row = {"utc": utc_now(), "command": command, "dataset": dataset, "index": index,
               "name": name, "path": rel, "size": path.stat().st_size, "sha256": _sha256(path),
               "purpose": purpose}
        append_rows(video_log, VIDEO_FIELDS, [row])
        rows.append(row)
    return rows


def read_headers(paths, manifest, video_log: str | os.PathLike[str],
                 sha_only: bool = False) -> list[dict]:
    """Gate (y3) rows of every test video: size and sha256, and (unless ``sha_only``) the header
    frame count, fps and delta = header T - manifest T (no frame is decoded)."""
    from wvad.data import dense_test as dt
    from wvad.extract.video import read_header

    rels = dt.video_rel_paths(manifest)
    items = list(zip(range(len(manifest)), manifest.names, rels, strict=True))
    purpose = "sha256" if sha_only else "header"
    logged = hash_videos(paths, manifest.dataset, items, dt.video_path_set(manifest), video_log,
                         f"e022-{purpose}", purpose)
    rows = []
    for row, t in zip(logged, manifest.total_frames, strict=True):
        out = {k: row[k] for k in ("dataset", "index", "name", "path", "size", "sha256")}
        out |= {"manifest_frames": int(t), "utc": utc_now()}
        if not sha_only:
            frames, fps = read_header(paths.video_root() / row["path"])
            out |= {"header_frames": frames, "delta": frames - int(t), "fps": fps}
        rows.append(out)
    return rows


def header_summary(rows: Sequence[Mapping]) -> dict:
    """|header T - manifest T| <= 16 for every video (gate (y3), the E021 (x3a) bound)."""
    from wvad.extract.gate import X3A_DELTA

    deltas = [int(r["delta"]) for r in rows]
    bad = [{k: r[k] for k in ("dataset", "index", "name", "header_frames", "manifest_frames")}
           for r, d in zip(rows, deltas, strict=True) if abs(d) > X3A_DELTA]
    return {"videos": len(rows), "nonzero_delta": sum(d != 0 for d in deltas),
            "max_abs_delta": max((abs(d) for d in deltas), default=0), "bound": X3A_DELTA,
            "violations": bad, "status": "fail" if bad else "pass"}


# --- extraction -----------------------------------------------------------------------------

def _summary(result: Mapping) -> dict:
    keys = ("id", "phase", "videos_requested", "videos_pending", "videos_finished", "snippets",
            "seconds", "errors", "worker_failures")
    return {k: result.get(k) for k in keys}


def extract_split(paths, manifest, encoders: Mapping, device, batch: Mapping[str, int], *,
                  access_log, video_log, workers: int, chunk: int, environment: Mapping | None,
                  commit: Mapping, normalize: str = "lut", mode: str | None = None) -> dict:
    """The test videos of one dataset into the test stores by the frozen run_extraction (phase
    "sample", command "e022-test-dense"); the pending videos are hashed into the video log
    first (positive check)."""
    from wvad.data import dense_test as dt
    from wvad.extract import dense
    from wvad.extract.encoders import PRODUCTION_PRECISION

    if commit.get("split") != dt.SPLIT:
        raise ValueError("the run's commit mapping must carry split 'test'")
    check_access_csv(access_log, dt.COMMAND)
    layout = dt.layout_of(manifest)
    dirs = {e: dt.store_folder(paths, manifest.dataset, e) for e in encoders}
    tasks = video_tasks(paths, manifest)
    todo = pending_tasks(tasks, dirs)
    hashed = hash_videos(paths, manifest.dataset, [(t.index, t.name, t.rel_path) for t in todo],
                         dt.video_path_set(manifest), video_log, dt.COMMAND, "dense")
    result = dense.run_extraction(dirs, layout, tasks, encoders, device,
                                  mode or PRODUCTION_PRECISION, batch, dt.PHASE, access_log,
                                  dt.COMMAND, workers, chunk, environment=environment,
                                  commit=dict(commit), normalize=normalize)
    return {**_summary(result), "dataset": manifest.dataset, "videos_hashed": len(hashed),
            "stores": {e: str(d) for e, d in dirs.items()}, "names_sha256": layout.names_sha256()}


def repro_compare(scratch: Mapping[str, Path], e021: Mapping[str, Path], sample: Sequence[int],
                  layout) -> dict:
    """Gate (y2) per encoder: the re-extracted rows of the sample videos (``layout`` = the
    subset layout of ``scratch``) against the E021 rows: bit-equal rows, else per snippet
    relative L2 <= 1e-3 and cosine >= 0.99999."""
    from wvad.data.dense import FEATURES_FILE, read_index, store_dir
    from wvad.extract.gate import FAIL, PASS_BIT, PASS_TOL, bit_equal, cosine, rel_l2

    out = {}
    for e in scratch:
        new_ix, ref_ix = read_index(scratch[e]), read_index(e021[e])
        new = np.load(store_dir(scratch[e]) / FEATURES_FILE, mmap_mode="r")
        ref = np.load(store_dir(e021[e]) / FEATURES_FILE, mmap_mode="r")
        problems, eq, l2, cos = [], [], [], []
        for k, v in enumerate(sample):
            n = int(layout.counts[k])
            if not (new_ix.done[k] and ref_ix.done[v]) or int(ref_ix.n[v]) != n:
                problems.append(f"video {v}: not extracted or row counts differ")
                continue
            a = np.asarray(new[int(layout.starts[k]):int(layout.starts[k]) + n])
            b = np.asarray(ref[int(ref_ix.start[v]):int(ref_ix.start[v]) + n])
            with np.errstate(divide="ignore", invalid="ignore"):
                eq.append(bit_equal(a, b))
                l2.append(rel_l2(a, b))
                cos.append(cosine(a, b))
        eq_a = np.concatenate(eq) if eq else np.zeros(0, bool)
        l2_a = np.concatenate(l2) if l2 else np.zeros(0)
        cos_a = np.concatenate(cos) if cos else np.zeros(0)
        row_ok = eq_a | ((l2_a <= Y2_L2) & (cos_a >= Y2_COS))
        finite = np.isfinite(l2_a) & np.isfinite(cos_a)
        ok = not problems and eq_a.size > 0 and bool(row_ok.all())
        out[e] = {"videos": len(sample), "rows": int(eq_a.size), "bit_equal_rows": int(eq_a.sum()),
                  "max_rel_l2": float(l2_a[finite].max()) if finite.any() else None,
                  "min_cos": float(cos_a[finite].min()) if finite.any() else None,
                  "nonfinite_rows": int((~finite & ~eq_a).sum()), "problems": problems,
                  "status": (PASS_BIT if eq_a.all() else PASS_TOL) if ok else FAIL}
    return out


def repro_sample(paths, dataset: str, scratch: str | os.PathLike[str], encoders: Mapping, device,
                 batch: Mapping[str, int], *, access_log, video_log, workers: int, chunk: int,
                 environment: Mapping | None, commit: Mapping, normalize: str = "lut",
                 mode: str | None = None, manifest=None, split=None) -> dict:
    """Gate (y2): the E021 throughput sample of ``dataset`` (bench_sample: 10 validation
    videos, default_rng(20261007)) extracted again by the production call into ``scratch`` and
    compared with the E021 rows. Paths come from the training manifest (``manifest`` /
    ``split``: the repository's by default); only these 10 videos are allowed."""
    from wvad.data import dense_test as dt
    from wvad.data.catalog import dense_name, split_file
    from wvad.data.manifests import load_manifest, manifest_file
    from wvad.data.splits import check_labels, load_split
    from wvad.extract import dense
    from wvad.extract.encoders import PRODUCTION_PRECISION

    root = Path(scratch).resolve()
    if root == Path(paths.feature_root).resolve():
        raise SystemExit("--repro must be a scratch folder, not the feature_root")
    check_access_csv(access_log, Y2_COMMAND)
    manifest = manifest if manifest is not None else load_manifest(manifest_file(dataset))
    split = split if split is not None else load_split(split_file(dataset))
    check_labels(split, manifest.labels, f"{dataset} training manifest")
    sample = dense.bench_sample(split)
    layout = dense.Layout.from_manifest(dataset, manifest).subset(sample)
    tasks = dense.tasks_for(paths, dataset, manifest, sample, renumber=True)
    dirs = {e: root / dense_name(dataset, e) for e in encoders}
    allowed = {dense.rel_video_path(dataset, manifest, i) for i in sample}
    todo = pending_tasks(tasks, dirs)
    hash_videos(paths, dataset, [(sample[t.index], t.name, t.rel_path) for t in todo], allowed,
                video_log, Y2_COMMAND, "y2")
    result = dense.run_extraction(dirs, layout, tasks, encoders, device,
                                  mode or PRODUCTION_PRECISION, batch, dt.PHASE, access_log,
                                  Y2_COMMAND, workers, chunk, environment=environment,
                                  commit=dict(commit), normalize=normalize)
    e021 = {e: dt.e021_dir(paths, dataset, e) for e in encoders}
    return {"dataset": dataset, "sample": sample, "names": list(layout.names),
            "extraction": _summary(result), "comparison": repro_compare(dirs, e021, sample,
                                                                         layout)}


# --- derived test caches and descriptive 11 -------------------------------------------------

class Moments:
    """Pooled per-dimension mean / population variance (Chan's update, float64) and per-row
    squared L2 norms of float32 rows added block by block (no labels)."""

    def __init__(self, dim: int):
        self.n, self.mean, self.m2 = 0, np.zeros(dim), np.zeros(dim)
        self._sq: list[np.ndarray] = []

    def add(self, rows: np.ndarray) -> None:
        x = np.asarray(rows, dtype=np.float32).astype(np.float64)
        nb = x.shape[0]
        if nb == 0:
            return
        mb = x.mean(axis=0)
        delta, n = mb - self.mean, self.n + nb
        self.m2 = self.m2 + ((x - mb) ** 2).sum(axis=0) + delta ** 2 * (self.n * nb / n)
        self.mean = self.mean + delta * (nb / n)
        self.n = n
        self._sq.append((x * x).sum(axis=1))

    @property
    def var(self) -> np.ndarray:
        return self.m2 / self.n if self.n else np.full_like(self.m2, np.nan)

    @property
    def sq_norms(self) -> np.ndarray:
        return np.concatenate(self._sq) if self._sq else np.zeros(0)


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std()),
            **{f"q{int(round(100 * q))}": float(np.quantile(x, q)) for q in QUANTILES}}


def _part(sq: np.ndarray, var: np.ndarray) -> dict:
    return {"rows": int(sq.size), "norm": _stats(np.sqrt(sq)), "dimvar_mean": float(var.mean())}


def _ratio(single: np.ndarray, window: np.ndarray) -> dict:
    keep = window > 0
    return {"dims": int(keep.sum()), **_stats(single[keep] / window[keep])}


def desc11(moments: Mapping[str, Mapping[str, Moments]]) -> dict:
    """Descriptive 11 (no labels, every test video pooled): norm and per-dimension variance of
    C's scale-matched single snippets ("snippet") and of the d16m test windows ("window"), per
    encoder and per pair (the concatenated head input), with the per-dimension variance ratio
    snippet / window."""
    from wvad.data.catalog import ENCODER_PAIRS

    enc = {}
    for e, m in moments.items():
        s, w = m["snippet"], m["window"]
        enc[e] = {"snippet": _part(s.sq_norms, s.var), "window": _part(w.sq_norms, w.var),
                  "snippet_over_window_variance": _ratio(s.var, w.var),
                  "dimvar": {"snippet": s.var.tolist(), "window": w.var.tolist()}}
    pairs = {}
    for p in PAIRS:
        es = ENCODER_PAIRS[p]
        if not all(e in moments for e in es):
            continue
        sq = {k: sum(moments[e][k].sq_norms for e in es) for k in ("snippet", "window")}
        var = {k: np.concatenate([moments[e][k].var for e in es]) for k in ("snippet", "window")}
        pairs[p] = {k: _part(sq[k], var[k]) for k in ("snippet", "window")}
        pairs[p]["snippet_over_window_variance"] = _ratio(var["snippet"], var["window"])
        pairs[p]["norm_ratio"] = float(np.sqrt(sq["snippet"]).mean() / np.sqrt(sq["window"]).mean())
    return {"labels_used": False, "quantiles": list(QUANTILES), "encoders": enc, "pairs": pairs}


def check_store(folder: Path, layout, encoder: str) -> dict:
    """The dense test store before deriving: every test video done, in manifest order (index
    columns, meta names), written by phase-"sample" runs whose commit mapping says "test"."""
    from wvad.data import dense_test as dt
    from wvad.data.dense import META_FILE, read_index, read_json, store_dir

    ix = read_index(folder)
    meta = read_json(store_dir(folder) / META_FILE)
    problems = []
    same = (len(ix) == len(layout.names)
            and all(np.array_equal(getattr(ix, c), getattr(layout, a)) for c, a in
                    (("start", "starts"), ("n", "counts"), ("total_frames", "total_frames"),
                     ("labels", "labels"))))
    if not same:
        problems.append("index columns differ from the test manifest")
    want = {"dataset": layout.dataset, "encoder": encoder, "rows": layout.rows,
            "videos": len(layout.names), "names_sha256": layout.names_sha256()}
    problems += [f"meta {k} {meta.get(k)} != {v}" for k, v in want.items() if meta.get(k) != v]
    if same and not ix.done.all():
        problems.append(f"{int((~ix.done).sum())} test videos not extracted")
    runs = meta.get("runs") or []
    if not runs or any(r.get("phase") != dt.PHASE or (r.get("commit") or {}).get("split")
                       != dt.SPLIT for r in runs):
        problems.append("store runs are not all phase 'sample' with split 'test'")
    if problems:
        raise SystemExit(f"{folder}: {problems}")
    return {"videos": len(ix), "rows": ix.rows, "runs": [r.get("id") for r in runs]}


def e021_scale(paths, dataset: str, encoder: str, record: Mapping | None):
    """The E021 d16m scale map of one encoder (read only, never refitted) and its identity; with
    the committed training-input record, its meta.json and array sha256 must equal it."""
    from wvad.data import dense_test as dt
    from wvad.data.catalog import dense_name
    from wvad.data.dense import META_FILE, load_scale_map, variant_dir

    vd = variant_dir(dt.e021_dir(paths, dataset, encoder), dt.SCALED)
    scale = load_scale_map(vd)
    rec = {"source": f"{dense_name(dataset, encoder)}/{dt.SCALED}/{META_FILE}",
           "meta_sha256": _sha256(vd / META_FILE), "array_sha256": dt.scale_array_sha256(scale)}
    if record is not None:
        want_meta = (record.get("files") or {}).get(rec["source"])
        want_array = (record.get("scale_arrays") or {}).get(dense_name(dataset, encoder))
        if (rec["meta_sha256"], rec["array_sha256"]) != (want_meta, want_array):
            raise SystemExit(f"{rec['source']}: scale map differs from the committed record")
    return scale, rec


def write_derived(folder: Path, variant: str, features: Sequence, layout,
                  record: Mapping) -> dict:
    """<folder>/<variant>/mil/test/data.pt (keys of the current caches; seg_labels the
    placeholder) and its e022_test_derive.json (never a meta.json)."""
    import torch

    from wvad.data import dense_test as dt
    from wvad.extract import grid

    path = dt.derived_file(folder, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    torch.save({"features": list(features), "labels": [int(y) for y in layout.labels],
                "seg_labels": [torch.full((grid.WINDOWS,), dt.PLACEHOLDER) for _ in features],
                "total_frames": [int(t) for t in layout.total_frames]}, tmp)
    os.replace(tmp, path)
    rec = {**record, "data_sha256": _sha256(path)}
    write_json(dt.derive_record(folder, variant), rec)
    return rec


def derive_encoder(paths, manifest, encoder: str, *, commit: Mapping | None,
                   train_record: Mapping | None, overwrite: str | None = None
                   ) -> tuple[dict, dict[str, Moments]]:
    """d16c (the snippet supporting each window centre, stored dtype) and d16m (the float64 means
    over J_k, then the E021 map) of one encoder's test store; their records and the moments of
    descriptive 11. Labels and T come from the restricted read."""
    import torch

    from wvad.data import dense_test as dt
    from wvad.data.dense import FEATURES_FILE, INDEX_FILE, META_FILE, store_dir
    from wvad.extract import derive

    ds, layout = manifest.dataset, dt.layout_of(manifest)
    folder = dt.store_folder(paths, ds, encoder)
    existing = [str(dt.derived_file(folder, v)) for v in dt.VARIANTS
                if dt.derived_file(folder, v).exists()]
    if existing and not overwrite:
        raise SystemExit(f"derived test caches exist ({existing}): --overwrite '<reason>'")
    store_info = check_store(folder, layout, encoder)
    scale, scale_rec = e021_scale(paths, ds, encoder, train_record)
    store = np.load(store_dir(folder) / FEATURES_FILE, mmap_mode="r")
    centre = derive.snippet_indices(ds, encoder, "d16c", layout.total_frames)
    moments = {"snippet": Moments(store.shape[1]), "window": Moments(store.shape[1])}
    feats: dict[str, list] = {"d16c": [], "d16m": []}
    for v in range(len(layout.names)):
        a, t = int(layout.starts[v]), int(layout.total_frames[v])
        rows = np.asarray(store[a:a + int(layout.counts[v])])
        if not np.isfinite(rows.astype(np.float32)).all():
            raise SystemExit(f"{folder}: non-finite rows in test video {v}")
        feats["d16c"].append(torch.from_numpy(np.ascontiguousarray(rows[centre[v]])))
        means = scale.apply(derive.window_means(rows, derive.mean_members("d16m", t)))
        feats["d16m"].append(torch.from_numpy(np.ascontiguousarray(means)))
        moments["snippet"].add(scale.apply(rows))
        moments["window"].add(means)
    base = {"dataset": ds, "encoder": encoder, "split": dt.SPLIT, "videos": len(layout.names),
            "names_sha256": layout.names_sha256(), "dense_rows": layout.rows,
            "store": {f: _sha256(store_dir(folder) / f) for f in (INDEX_FILE, META_FILE)},
            "store_runs": store_info["runs"], "manifest_sha256": manifest.sha256,
            "labels": "test manifest categories (restricted read)",
            "total_frames": "test manifest (restricted read)",
            "seg_labels": f"placeholder {dt.PLACEHOLDER} (read by no scoring or evaluation)",
            "commit": dict(commit) if commit else None, "created_utc": utc_now(),
            "overwrite": overwrite}
    idx = hashlib.sha256(np.ascontiguousarray(centre, dtype="<i8").tobytes()).hexdigest()
    recs = {"d16c": write_derived(folder, "d16c", feats["d16c"], layout, base | {
                "variant": "d16c", "rule": derive.RULES["d16c"], "indices_sha256": idx}),
            "d16m": write_derived(folder, "d16m", feats["d16m"], layout, base | {
                "variant": "d16m", "rule": derive.RULES["d16m"], "scale": scale_rec})}
    return recs, moments


def derive_split(paths, manifest, encoders: Sequence[str], *, commit: Mapping | None,
                 train_record: Mapping | None, overwrite: str | None = None) -> dict:
    """Every encoder of one dataset, then descriptive 11 and the test-cache sha256."""
    from wvad.data import dense_test as dt

    records, moments = {}, {}
    for e in encoders:
        records[e], moments[e] = derive_encoder(paths, manifest, e, commit=commit,
                                                train_record=train_record, overwrite=overwrite)
    return {"dataset": manifest.dataset, "records": records, "desc11": desc11(moments),
            "sha256": dt.store_hashes(paths, [manifest.dataset], list(encoders))}


# --- pre-gates of the production extraction (PLAN.md 8) -------------------------------------

def read_approvals(path: str | os.PathLike[str] | None = None) -> dict:
    """The user-approved gate exceptions (results/e022_approvals.json of this tree, as
    ``wvad.diagnostics.e022 gates --stage extract`` reads it: {"<gate>": {"decision": "<the
    user's words>", ...}}); {} when absent. In a git checkout the file must be committed
    unchanged (a deployed copy holds committed files only)."""
    from wvad.data import dense_test as dt
    from wvad.diagnostics.common import code_version

    p = Path(path) if path is not None else dt.RESULTS / APPROVALS_FILE
    if not p.is_file():
        return {}
    if path is None and code_version().get("source") == "git":
        rel = p.relative_to(dt.REPO_ROOT).as_posix()
        if (dt._git(dt.REPO_ROOT, "ls-files", "--error-unmatch", "--", rel).returncode != 0
                or dt._git(dt.REPO_ROOT, "diff", "--quiet", "HEAD", "--", rel).returncode != 0):
            raise SystemExit(f"{rel} is not committed unchanged")
    return json.loads(p.read_text(encoding="utf-8"))


def production_gates(record: Mapping, datasets: Sequence[str], encoders: Sequence[str],
                     env: Mapping, approvals: Mapping) -> dict:
    """PLAN.md 8-9: the production extraction opens test videos only after the pre-gates passed
    on this server: (y1) this extraction environment = E021's env_extract.json (a different
    driver only is recorded and passes; (y2) decides), (y2) the re-extracted E021 sample of
    each dataset bit-equal or within the tolerance for every encoder extracted, (y3) every
    header of each dataset within |dT| <= 16. A failed gate passes only as the user's approved
    exception (``approvals`` has the gate: "통과(사용자 승인 예외)" / "(추출 환경 변경)")."""
    from wvad.diagnostics.e022_gates import gate_y1
    from wvad.extract.gate import PASS_BIT, PASS_TOL

    y1 = gate_y1(env)
    gates = {"y1": {"status": y1["status"], "differences": y1["differences"], "ok": y1["ok"]}}
    for d in datasets:
        comp = (record.get(f"y2_{d}") or {}).get("comparison") or {}
        status = {e: (comp.get(e) or {}).get("status") for e in encoders}
        gates[f"y2_{d}"] = {"status": status,
                            "ok": all(s in (PASS_BIT, PASS_TOL) for s in status.values())}
        y3 = record.get(f"y3_{d}") or {}
        gates[f"y3_{d}"] = {"status": y3.get("status"), "videos": y3.get("videos"),
                            "ok": y3.get("status") == "pass"}
    for name, res in gates.items():
        approval = approvals.get(name.split("_")[0])
        if not res["ok"] and approval:
            res |= {"ok": True, "approved": dict(approval)}
    return {"gates": gates, "failed": sorted(k for k, v in gates.items() if not v["ok"]),
            "ok": all(v["ok"] for v in gates.values())}


# --- command line ---------------------------------------------------------------------------

def _batch(text: str) -> dict[str, int]:
    """Batch sizes as xcb=8,xcl=8,siglip=4; an encoder not named keeps the production batch."""
    out = {"xcb": 8, "xcl": 8, "siglip": 4}
    for item in text.split(","):
        name, _, value = item.partition("=")
        if name not in ENCODERS or not value.isdigit() or int(value) < 1:
            raise argparse.ArgumentTypeError(f"batch must look like {PRODUCTION_BATCH}: {text}")
        out[name] = int(value)
    return out


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encoders(names: Sequence[str], device) -> dict:
    from wvad.extract.encoders import Encoder

    return {e: Encoder.load(e, device) for e in names}


def _environment() -> dict:
    from wvad.extract.encoders import extraction_environment

    return extraction_environment()


def _guard(args, paths, check_inputs: bool) -> dict:
    from wvad.data import dense_test as dt

    return dt.access_guard(paths, freeze=args.freeze_commit, lineage=args.lineage,
                           selection_sha256=args.selection_sha256, train_inputs=args.train_inputs,
                           check_inputs=check_inputs)


def _guard_summary(guard: Mapping) -> dict:
    return {"descent": guard["descent"], "plan": guard["plan"], "approval": guard["approval"],
            "test_data_ok": guard["test_data"]["ok"], "old_caches_ok": guard["old_caches"]["ok"],
            "selection": guard["selection"],
            "train_inputs": {k: v for k, v in guard["train_inputs"].items()
                             if k in ("checked", "ok", "record", "record_sha256", "files",
                                      "scale_arrays")}}


def _run_access(args, command: str, check_inputs: bool, body, precheck=None) -> dict:
    """Access device, ``precheck(paths, guard)`` (a SystemExit refuses the call), then the start
    line, ``body(paths, guard)``, the item-8 recheck and the end line (or a refused / error
    line)."""
    from wvad.data import dense_test as dt
    from wvad.diagnostics.common import code_version

    paths = _paths(args.paths)
    dt.outside_repo(args.test_log, getattr(args, "video_log", None),
                    getattr(args, "access_log", None), getattr(args, "repro", None))
    line = {"access": "A", "command": command, "datasets": list(args.datasets)}
    try:
        guard = _guard(args, paths, check_inputs)
        pre = precheck(paths, guard) if precheck is not None else None
    except SystemExit as exc:
        dt.append_test_log(args.test_log, line | {"time_utc": utc_now(), "event": "refused",
                                                  "reason": str(exc), "code": code_version()})
        raise
    dt.outside_repo(paths.feature_root)
    merge_record(args.record, f"guard_{command}", guard | {"time_utc": utc_now()})
    if pre is not None:
        merge_record(args.record, f"pregates_{command}", pre | {"time_utc": utc_now()})
    files = {k: str(getattr(args, k)) for k in ("video_log", "access_log", "repro", "out",
                                                "record") if getattr(args, k, None)}
    dt.append_test_log(args.test_log, line | {"time_utc": utc_now(), "event": "start",
                                              "guard": _guard_summary(guard), "files": files,
                                              "encoders": list(getattr(args, "encoders", []))}
                       | ({"pregates": pre} if pre is not None else {}))
    try:
        result = body(paths, guard)
    except BaseException as exc:
        dt.append_test_log(args.test_log, line | {"time_utc": utc_now(), "event": "error",
                                                  "error": repr(exc)})
        raise
    after = (dt.check_train_inputs(dt.load_train_inputs(args.train_inputs), paths)
             if check_inputs else {"checked": False})
    merge_record(args.record, f"train_inputs_after_{command}", after)
    dt.append_test_log(args.test_log, line | {"time_utc": utc_now(), "event": "end",
                                              "result": result, "train_inputs_after": after})
    if check_inputs and not after["ok"]:
        raise SystemExit(f"E021 training inputs changed during the call: {after}")
    return result


def _commit(guard: Mapping, args) -> dict:
    d = guard["descent"]
    return {"commit": d["deployed_commit"], "source": d["code_source"], "dirty": d["dirty"],
            "freeze_commit": d["freeze_commit"], "plan_sha256": guard["plan"]["sha256"],
            "selection_sha256": args.selection_sha256, "split": "test"}


def cmd_inputs(args) -> None:
    rec = record_inputs(_paths(args.paths), args.out, args.e021_runs)
    print(json.dumps({"files": len(rec["files"]), "scale_arrays": len(rec["scale_arrays"]),
                      "ok": rec["ok"]}))
    if not rec["ok"]:
        raise SystemExit(f"data_sha256 self-consistency failed: {rec['self_consistency']}")


def cmd_check(args) -> None:
    out = access_zero(_paths(args.paths), args.test_log, listings=args.listing or (),
                      e020_runs=args.e020_runs, e020_scores=args.e020_scores)
    if args.out:
        write_json(args.out, out)
    print(json.dumps({d: out[d]["listing"] for d in DATASETS})[:2000])
    if not out["ok"]:
        raise SystemExit("access (0) found a difference; see the test log")


def cmd_header(args) -> None:
    from wvad.data import dense_test as dt

    def body(paths, guard):
        rows, summary = [], {}
        for d in args.datasets:
            r = read_headers(paths, dt.read_manifest(d), args.video_log, args.sha_only)
            rows += r
            summary[d] = {"videos": len(r)} if args.sha_only else header_summary(r)
            merge_record(args.record, f"{'sha' if args.sha_only else 'y3'}_{d}", summary[d])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=HEADER_FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in HEADER_FIELDS} for r in rows)
        return summary

    summary = _run_access(args, "header", False, body)
    print(json.dumps(summary)[:2000])
    failed = [d for d, s in summary.items() if s.get("status") == "fail"]
    if failed:
        raise SystemExit(f"gate (y3) failed for {failed}: stop and report (PLAN.md (y3))")


def cmd_dense(args) -> None:
    from wvad.data import dense_test as dt

    state: dict = {}

    def precheck(paths, guard):
        """The production extraction: (y1), (y2), (y3) passed (or approved) before any test
        video is opened (``production_gates``); (y2) itself (--repro) opens training videos."""
        if args.repro:
            return None
        if not args.record or not Path(args.record).is_file():
            raise SystemExit("the production extraction needs --record with the (y2) and (y3) "
                             "records (PLAN.md 8: header, then dense --repro)")
        state["env"] = _environment()
        res = production_gates(json.loads(Path(args.record).read_text(encoding="utf-8")),
                               args.datasets, args.encoders, state["env"], read_approvals())
        if not res["ok"]:
            raise SystemExit(f"pre-gates of the production extraction not passed: "
                             f"{res['failed']} (PLAN.md 8: stop and report, or the user's "
                             "approval in results/e022_approvals.json)")
        return res

    def body(paths, guard):
        device = _device()
        encoders = _encoders(args.encoders, device)
        env = state["env"] if "env" in state else _environment()
        commit, out = _commit(guard, args), {}
        for d in args.datasets:
            kw = {"access_log": args.access_log, "video_log": args.video_log,
                  "workers": args.workers, "chunk": args.chunk, "environment": env,
                  "commit": commit, "normalize": args.normalize, "mode": _precision()}
            if args.repro:
                out[d] = repro_sample(paths, d, args.repro, encoders, device, args.batch, **kw)
                merge_record(args.record, f"y2_{d}", out[d])
            else:
                out[d] = extract_split(paths, dt.read_manifest(d), encoders, device, args.batch,
                                       **kw)
                merge_record(args.record, f"dense_{d}", out[d])
        return out

    out = _run_access(args, "repro" if args.repro else "dense", not args.repro, body, precheck)
    print(json.dumps(out)[:2000])
    failed = {d: (r.get("errors"), r.get("worker_failures")) for d, r in out.items()
              if not args.repro and (r.get("errors") or r.get("worker_failures"))}
    if failed:
        raise SystemExit(f"videos failed or workers failed (left undone): {failed}")


def _precision() -> str:
    from wvad.extract.encoders import PRODUCTION_PRECISION

    return PRODUCTION_PRECISION


def cmd_derive(args) -> None:
    from wvad.data import dense_test as dt

    def body(paths, guard):
        record = dt.load_train_inputs(args.train_inputs)
        out = {}
        for d in args.datasets:
            res = derive_split(paths, dt.read_manifest(d), args.encoders,
                               commit=_commit(guard, args), train_record=record,
                               overwrite=args.overwrite)
            merge_record(args.record, f"derive_{d}", res["records"])
            merge_record(args.record, f"desc11_{d}", res["desc11"])
            merge_record(args.record, f"test_cache_sha256_{d}", res["sha256"])
            out[d] = {e: {v: r["data_sha256"] for v, r in recs.items()}
                      for e, recs in res["records"].items()}
        return out

    print(json.dumps(_run_access(args, "derive", True, body))[:2000])


def _access_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--test_log", required=True, help="test access log (outside the repository)")
    p.add_argument("--record", help="extraction gate record JSON to merge the summaries into")
    p.add_argument("--selection_sha256", required=True,
                   help="sha256 of the committed results/e022_selection.json")
    p.add_argument("--freeze_commit", help="the freeze commit (required in a deployed copy)")
    p.add_argument("--lineage", default=None, help="deploy lineage file (default "
                                                   "DEPLOYED_LINEAGE.json in the deploy root)")
    p.add_argument("--train_inputs", default=None,
                   help="committed results/e022_train_inputs.json (default: in this tree)")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m wvad.extract.testsplit")
    ap.add_argument("--paths", help="paths file (default: $WVAD_PATHS or configs/paths.yaml)")
    sub = ap.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inputs", help="3a: training-input hashes (no test access)")
    i.add_argument("--out", required=True)
    i.add_argument("--e021_runs", help="E021 runs root (gate (c) config checkpoints)")
    c = sub.add_parser("check", help="test access (0): the pre-freeze list and hash check")
    c.add_argument("--test_log", required=True)
    c.add_argument("--out")
    c.add_argument("--listing", nargs="+", help="file listings made elsewhere (one relative "
                                                "path per line)")
    c.add_argument("--e020_runs")
    c.add_argument("--e020_scores")
    h = sub.add_parser("header", help="(y3): header frame counts and sha256 of the test videos")
    _access_args(h)
    h.add_argument("--video_log", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--sha_only", action="store_true", help="sha256 only (a local copy, (y7))")
    d = sub.add_parser("dense", help="production test extraction, or (y2) with --repro")
    _access_args(d)
    d.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    d.add_argument("--video_log", required=True)
    d.add_argument("--access_log", required=True, help="run_extraction's access CSV")
    d.add_argument("--repro", help="(y2): scratch folder of the re-extracted sample")
    d.add_argument("--workers", type=int, default=32)
    d.add_argument("--chunk", type=int, default=32)
    d.add_argument("--batch", type=_batch, default=_batch(PRODUCTION_BATCH))
    d.add_argument("--normalize", choices=NORMALIZE, default="lut")
    r = sub.add_parser("derive", help="derived test caches d16c, d16m and descriptive 11")
    _access_args(r)
    r.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    r.add_argument("--overwrite", help="reason to write the derived test caches again")
    args = ap.parse_args(argv)
    if getattr(args, "train_inputs", "unset") is None:
        from wvad.data.dense_test import TRAIN_INPUTS_FILE

        args.train_inputs = str(TRAIN_INPUTS_FILE)
    if getattr(args, "lineage", "unset") is None:
        from wvad.data.dense_test import LINEAGE_FILE

        args.lineage = str(LINEAGE_FILE)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    {"inputs": cmd_inputs, "check": cmd_check, "header": cmd_header, "dense": cmd_dense,
     "derive": cmd_derive}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
