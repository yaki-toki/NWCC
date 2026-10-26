"""Production dense extraction (E021): training-split videos -> decode worker processes -> GPU
batches -> one memory-mapped snippet cache per (dataset, encoder) (layout: ``wvad.data.dense``).

* Videos: the training-split videos of a phase (``val`` = the validation split, stage 1;
  ``trainpart`` = the rest, stage 2) in manifest order. Paths are built from the training
  manifest only (``video_path``) and appended to the access log before a worker may open them.
* Workers (spawned processes, OpenCV with one thread; ``video.decode_worker``) decode with the
  production rule and send uint8 RGB snippets at each needed input size in chunks of whole
  batches, so batch boundaries are fixed snippet indices of every video.
* The consumer normalises with the lookup tables on the device, encodes each chunk in batches
  under the precision mode, writes the rows into the memory map and, after a video's last
  chunk, flushes the map and appends one line to each encoder's progress.jsonl: the completion
  flag, with the id and commit of the run that wrote it (gate (x4) checks every such run, also
  one that crashed later). A rerun skips (video, encoder) pairs already done (resume); a video
  whose decoding failed stays undone and is reported. An unterminated last progress line (a
  crash during its append) is ignored and removed before the next append.
* The first run of a cache creates features.npy full size (every training-split snippet) filled
  with NaN, under a temporary name renamed after meta.json is written (a crash during the fill
  leaves no partial cache); later phases fill their rows of the same file. index.npz is rebuilt
  from the manifest and progress.jsonl at the end of every run; meta.json keeps one entry per
  run (phase, commit, environment, precision, TF32, batch and chunk sizes, encoder revisions).
* A failure of the decode workers (an unreadable message, a worker that exits abnormally while
  no message arrives) ends the run cleanly: the videos in progress are reported (left undone)
  and the failure is recorded in the run's ``worker_failures``.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import queue
import threading
import time
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from wvad.data.catalog import NEW_DATASETS, dense_dir
from wvad.data.dense import (
    FEATURES_FILE,
    META_FILE,
    PHASES,
    PROGRESS_FILE,
    VIDEO_COLUMNS,
    read_json,
    store_dir,
    write_index,
)
from wvad.data.manifests import TrainManifest
from wvad.data.splits import ValSplit
from wvad.extract import grid
from wvad.extract.encoders import Encoder, precision, tf32_settings
from wvad.extract.preprocess import clips_to_pixels, make_normalizer
from wvad.extract.video import decode_worker, video_messages
from wvad.paths import Paths

UCF_FOLDER = "Anomaly-Detection-Dataset"
XD_FOLDER = PurePosixPath("XD-Violence") / "train"
DEFAULT_CHUNK = 32                     # snippets per worker message (a multiple of each batch)
NAN_FILL_ROWS = 1 << 16
WORKER_POLL_SECONDS = 30.0             # decode-queue polling interval
WORKER_STALL_SECONDS = 120.0           # no message this long after an abnormal worker exit: stop
ACCESS_FIELDS = ("utc", "command", "dataset", "index", "name", "path", "purpose")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --- videos and the access log --------------------------------------------------------------

def rel_video_path(dataset: str, manifest: TrainManifest, index: int) -> str:
    """Path of a training video relative to raw_video_root (posix), from the manifest only."""
    if dataset == "ucf":
        rel = manifest.rel_paths[index]
        if rel is None:
            raise ValueError(f"UCF manifest row {index} has no rel_path")
        return str(PurePosixPath(UCF_FOLDER) / rel)
    if dataset == "xdv":
        return str(XD_FOLDER / f"{manifest.names[index]}.mp4")
    if dataset in NEW_DATASETS:            # E024 stage 2 (UBnormal): the manifest's rel_paths
        rel = manifest.rel_paths[index]
        if rel is None:
            raise ValueError(f"{dataset} manifest row {index} has no rel_path")
        return str(PurePosixPath(rel))
    raise KeyError(f"unknown dataset '{dataset}'")


def video_path(paths: Paths, dataset: str, manifest: TrainManifest, index: int) -> Path:
    return paths.video_root() / rel_video_path(dataset, manifest, index)


def manifest_video_paths(dataset: str, manifest: TrainManifest) -> set[str]:
    """Every relative path the training manifest can produce (gate (x6) positive check)."""
    return {rel_video_path(dataset, manifest, i) for i in range(len(manifest))}


def log_access(log: str | os.PathLike[str], rows: Sequence[Mapping]) -> None:
    """Append rows (ACCESS_FIELDS) to the CSV access log; the header is written once."""
    path = Path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ACCESS_FIELDS, lineterminator="\n")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ACCESS_FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def phase_videos(split: ValSplit, phase: str) -> list[int]:
    """Manifest indices of a phase, in manifest order."""
    if phase == "val":
        return sorted(split.val_indices)
    if phase == "trainpart":
        return sorted(split.train_indices)
    raise ValueError(f"unknown phase {phase!r} (val, trainpart)")


# --- layout and store -----------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Row layout of one dataset's training split (manifest order, T = manifest frames)."""

    dataset: str
    names: tuple[str, ...]
    labels: np.ndarray
    total_frames: np.ndarray
    counts: np.ndarray
    starts: np.ndarray

    @classmethod
    def from_manifest(cls, dataset: str, manifest: TrainManifest) -> Layout:
        t = np.asarray(manifest.total_frames, dtype=np.int64)
        n = np.array([grid.n_snippets(x) for x in t], dtype=np.int64)
        return cls(dataset, tuple(manifest.names), np.asarray(manifest.labels, dtype=np.int64),
                   t, n, np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64))

    @property
    def rows(self) -> int:
        return int(self.counts.sum())

    def subset(self, indices: Sequence[int]) -> Layout:
        """A layout of some videos only (throughput runs into a scratch folder)."""
        idx = [int(i) for i in indices]
        n = self.counts[idx]
        return Layout(self.dataset, tuple(self.names[i] for i in idx), self.labels[idx],
                      self.total_frames[idx], n,
                      np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64))

    def names_sha256(self) -> str:
        return names_sha256(self.names)


def names_sha256(names: Sequence[str]) -> str:
    """Identity of a video list and its order (meta.json, gate (x4))."""
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


GRID = {"snippet_frames": grid.SNIPPET, "clip_frames": grid.CLIP_FRAMES,
        "clip_stride": grid.CLIP_STRIDE, "total_frames": "manifest total_frames",
        "rows": "ceil(T / 16) per video, manifest order"}


def write_meta(path: Path, meta: Mapping) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def open_store(directory: Path, layout: Layout, encoder: str, encoder_info: Mapping,
               dim: int, dtype: str) -> np.memmap:
    """The cache of one encoder, created full size (NaN) on first use, else checked against
    this layout, encoder and dtype. A new features.npy is filled under a temporary name and
    renamed after meta.json is written (an interrupted fill is redone by the next run)."""
    sd = store_dir(directory)
    sd.mkdir(parents=True, exist_ok=True)
    features, meta_path = sd / FEATURES_FILE, sd / META_FILE
    want = {"dataset": layout.dataset, "encoder": encoder, "dim": int(dim), "dtype": dtype,
            "rows": layout.rows, "videos": len(layout.names),
            "names_sha256": layout.names_sha256(), "grid": GRID,
            "model_id": encoder_info.get("model_id"), "revision": encoder_info.get("revision")}
    if not features.exists():
        if meta_path.exists() and read_json(meta_path).get("runs"):
            raise ValueError(f"{sd}: meta.json lists runs but {FEATURES_FILE} is missing")
        part = features.with_name(features.name + ".part")
        mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.dtype(dtype),
                                       shape=(layout.rows, int(dim)))
        for a in range(0, layout.rows, NAN_FILL_ROWS):
            mm[a:a + NAN_FILL_ROWS] = np.nan
        mm.flush()
        del mm                                          # closed before the rename (Windows)
        write_meta(meta_path, {**want, "created_utc": utc_now(), "runs": []})
        os.replace(part, features)
    meta = read_json(meta_path)
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"{sd}: existing cache does not match this run: {bad}")
    mm = np.lib.format.open_memmap(features, mode="r+")
    if mm.shape != (layout.rows, int(dim)) or mm.dtype != np.dtype(dtype):
        raise ValueError(f"{features}: shape {mm.shape} / dtype {mm.dtype} differ from meta")
    return mm


def read_progress(directory: Path) -> dict[int, dict]:
    """Finished videos of one encoder cache (the last line of a video wins). An unterminated
    last line (a crash during its append) is ignored with a warning; ``repair_progress`` removes
    it before the next append."""
    path = store_dir(directory) / PROGRESS_FILE
    done: dict[int, dict] = {}
    if path.exists():
        complete, _, torn = path.read_bytes().rpartition(b"\n")
        if torn.strip():
            warnings.warn(f"{path}: unterminated last line ignored ({len(torn)} bytes)",
                          stacklevel=2)
        for line in complete.decode("utf-8").split("\n"):
            if line.strip():
                row = json.loads(line)
                done[int(row["video"])] = row
    return done


def repair_progress(directory: Path) -> int:
    """Cut an unterminated last line off progress.jsonl (before appending); returns the bytes
    removed."""
    path = store_dir(directory) / PROGRESS_FILE
    if not path.exists():
        return 0
    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep == len(data):
        return 0
    warnings.warn(f"{path}: unterminated last line removed ({len(data) - keep} bytes)",
                  stacklevel=2)
    with open(path, "r+b") as fh:
        fh.truncate(keep)
        fh.flush()
        os.fsync(fh.fileno())
    return len(data) - keep


def _append_progress(directory: Path, row: Mapping) -> None:
    with open(store_dir(directory) / PROGRESS_FILE, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def rebuild_index(directory: Path, layout: Layout) -> Path:
    """index.npz from the layout and progress.jsonl (VIDEO_COLUMNS and snippet_replaced)."""
    n = len(layout.names)
    cols = {c: np.zeros(n, dtype=t) for c, t in VIDEO_COLUMNS.items()}
    cols.update(start=layout.starts, n=layout.counts, total_frames=layout.total_frames,
                labels=layout.labels, header_frames=np.full(n, -1, dtype=np.int64),
                fps=np.full(n, np.nan))
    replaced = np.zeros(layout.rows, dtype=np.uint8)
    for v, row in read_progress(directory).items():
        for c in ("header_frames", "fps", "grab_attempts", "grab_successes", "used_frames",
                  "replaced", "replaced_tail"):
            cols[c][v] = row[c]
        cols["done"][v] = True
        cols["phase"][v] = PHASES[row["phase"]]
        rep = np.asarray(row["snippet_replaced"], dtype=np.uint8)
        if len(rep) != layout.counts[v]:
            raise ValueError(f"{directory}: video {v} has {len(rep)} snippet counts, "
                             f"expected {layout.counts[v]}")
        replaced[layout.starts[v]:layout.starts[v] + layout.counts[v]] = rep
    return write_index(directory, cols, replaced)


def _update_meta(directory: Path, run: Mapping, key: str | None = None) -> None:
    """Append ``run`` to meta.json's runs, or replace the run whose "id" equals ``key``."""
    path = store_dir(directory) / META_FILE
    meta = read_json(path)
    runs = [r for r in meta["runs"] if key is None or r.get("id") != key]
    meta["runs"] = runs + [dict(run)]
    write_meta(path, meta)


# --- extraction -----------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoTask:
    index: int                   # video index of the layout (manifest index in production)
    path: str
    total_frames: int            # manifest T
    name: str = ""
    rel_path: str = ""           # relative to raw_video_root (the access log)


class _Consumer:
    """Encodes the chunks of the decode messages and writes the rows (one process, one GPU)."""

    def __init__(self, stores: Mapping[str, np.memmap], dirs: Mapping[str, Path],
                 layout: Layout, encoders: Mapping[str, Encoder], device: torch.device,
                 mode: str, batch: Mapping[str, int], dtypes: Mapping[str, str], phase: str,
                 todo: Mapping[int, tuple[str, ...]], normalize: str = "lut",
                 tags: Mapping | None = None):
        self.stores, self.dirs, self.layout, self.encoders = stores, dirs, layout, encoders
        self.device, self.mode, self.batch, self.dtypes = device, mode, batch, dtypes
        self.phase, self.todo = phase, todo
        self.tags = dict(tags or {})              # run id and commit on every progress row
        self.normalizers = {e: make_normalizer(enc.spec.mean, enc.spec.std, device, normalize)
                            for e, enc in encoders.items()}
        self.replaced: dict[int, list[np.ndarray]] = {}
        self.finished: list[int] = []
        self.errors: dict[int, str] = {}
        self.failures: list[str] = []
        self.encode_seconds = {e: 0.0 for e in encoders}
        self.snippets = 0

    def handle(self, message: tuple) -> None:
        kind, video = message[0], int(message[1])
        if kind == "start":
            self.replaced[video] = []
        elif kind == "chunk":
            _, _, first, frames, replaced = message
            self._encode(video, int(first), frames)
            self.replaced[video].append(replaced)
        elif kind == "done":
            self._finish(video, message[2])
        elif kind == "error":
            self.errors[video] = message[2]
            self.replaced.pop(video, None)

    def _encode(self, video: int, first: int, frames: Mapping[int, np.ndarray]) -> None:
        start = int(self.layout.starts[video]) + first
        self.snippets += len(next(iter(frames.values())))
        for e in self.todo[video]:
            enc = self.encoders[e]
            clips = torch.from_numpy(frames[enc.spec.input_size]).to(self.device)
            t0 = time.perf_counter()
            parts = []
            for b in range(0, clips.shape[0], self.batch[e]):
                pixels = clips_to_pixels(clips[b:b + self.batch[e]], self.normalizers[e])
                with precision(self.mode, self.device):
                    parts.append(enc.encode(pixels))
            feats = torch.cat(parts).numpy()
            self.encode_seconds[e] += time.perf_counter() - t0
            if feats.dtype != np.dtype(self.dtypes[e]):
                raise RuntimeError(f"{e}: encoder output {feats.dtype} != store dtype "
                                   f"{self.dtypes[e]}")
            self.stores[e][start:start + len(feats)] = feats

    def _finish(self, video: int, stats: Mapping) -> None:
        rep = np.concatenate(self.replaced.pop(video)) if self.replaced.get(video) else \
            np.zeros(0, dtype=np.uint8)
        if len(rep) != self.layout.counts[video]:
            self.errors[video] = f"{len(rep)} snippets for n_v {self.layout.counts[video]}"
            return
        row = {"video": video, "name": self.layout.names[video], "phase": self.phase,
               "utc": utc_now(), **stats, **self.tags, "snippet_replaced": rep.tolist()}
        for e in self.todo[video]:
            self.stores[e].flush()
            _append_progress(self.dirs[e], row)
        self.finished.append(video)


def _check_batches(batch: Mapping[str, int], chunk: int, encoders: Sequence[str]) -> None:
    for e in encoders:
        if batch[e] < 1 or chunk % batch[e]:
            raise ValueError(f"chunk {chunk} must be a multiple of the {e} batch {batch[e]}")


def _run_inline(tasks, sizes_of, chunk, consumer: _Consumer) -> None:
    for t in tasks:
        for message in video_messages(t.index, t.path, t.total_frames, sizes_of[t.index], chunk):
            consumer.handle(message)


def _abnormal_exits(procs) -> list[str]:
    return [f"decode worker {p.pid} exited with code {p.exitcode}" for p in procs
            if p.exitcode not in (None, 0)]


def _run_workers(tasks, sizes_of, chunk, consumer: _Consumer, workers: int,
                 queue_size: int) -> None:
    """Spawned decode workers (``decode_worker``) feed the consumer through a receive thread.
    A receive failure (an unreadable message, a broken pipe) or a worker that exited abnormally
    while no message arrived for WORKER_STALL_SECONDS (a worker killed mid-message can leave the
    queue unreadable) ends the run: the failure goes to ``consumer.failures``, the videos in
    progress to ``consumer.errors`` (left undone, resumable) and the workers are stopped."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    task_q, result_q = ctx.Queue(), ctx.Queue(maxsize=queue_size)
    for t in tasks:
        task_q.put((t.index, t.path, t.total_frames, sizes_of[t.index]))
    for _ in range(workers):
        task_q.put(None)
    procs = [ctx.Process(target=decode_worker, args=(task_q, result_q, chunk), daemon=True)
             for _ in range(workers)]
    for p in procs:
        p.start()
    local: queue.Queue = queue.Queue(maxsize=4)        # unpickling overlaps the GPU work
    poll, stall = WORKER_POLL_SECONDS, WORKER_STALL_SECONDS

    def receive() -> None:
        exited = 0
        try:
            while exited < workers:
                try:
                    message = result_q.get(timeout=poll)
                except queue.Empty:
                    if not any(p.is_alive() for p in procs):
                        break
                    continue
                if message[0] == "exit":
                    exited += 1
                else:
                    local.put(message)
        except Exception as exc:                       # recorded; the run ends cleanly
            consumer.failures.append(f"receiving decode messages failed: {exc!r}")
        finally:
            local.put(None)

    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    try:
        waited = 0.0
        while True:
            try:
                message = local.get(timeout=poll)
            except queue.Empty:
                waited += poll
                dead = _abnormal_exits(procs)
                if dead and waited >= stall:
                    consumer.failures.append(f"no decode message for {waited:g} s after an "
                                             f"abnormal worker exit ({'; '.join(dead)})")
                    break
                continue
            waited = 0.0
            if message is None:
                break
            consumer.handle(message)
    finally:
        for p in procs:
            p.join(timeout=10)
        dead = _abnormal_exits(procs)
        alive = [p for p in procs if p.is_alive()]
        for p in alive:
            p.terminate()
        if dead or alive:
            task_q.cancel_join_thread()                # tasks left in the pipe: do not block exit
        consumer.failures += [d for d in dead if not any(d in f for f in consumer.failures)]
        if alive:
            consumer.failures.append(f"{len(alive)} decode worker(s) stopped by the consumer")
    for v in [v for v in consumer.replaced if v not in consumer.errors]:
        consumer.errors[v] = "decode worker ended before the video finished"


def run_extraction(dirs: Mapping[str, Path], layout: Layout, tasks: Sequence[VideoTask],
                   encoders: Mapping[str, Encoder], device: torch.device, mode: str,
                   batch: Mapping[str, int], phase: str, access_log: str | os.PathLike[str],
                   command: str = "dense", workers: int = 0, chunk: int = DEFAULT_CHUNK,
                   queue_size: int = 16, environment: Mapping | None = None,
                   commit: Mapping | None = None, normalize: str = "lut") -> dict:
    """Extract ``tasks`` (videos of ``phase``) into the caches ``dirs`` of ``encoders``. The
    paths still to decode are appended to ``access_log`` before any is opened. ``workers`` 0
    decodes in this process (tests); otherwise spawned decode processes."""
    names = list(encoders)
    _check_batches(batch, chunk, names)
    if phase not in PHASES or phase == "none":
        raise ValueError(f"unknown phase {phase!r}")
    dtypes = {e: (encoders[e].spec.autocast_dtype if mode == "autocast" else "float32")
              for e in names}
    infos = {e: encoders[e].info() for e in names}
    stores = {e: open_store(dirs[e], layout, e, infos[e], encoders[e].spec.dim, dtypes[e])
              for e in names}
    repaired = {e: n for e in names if (n := repair_progress(dirs[e]))}
    done = {e: set(read_progress(dirs[e])) for e in names}
    todo = {t.index: tuple(e for e in names if t.index not in done[e]) for t in tasks}
    pending = [t for t in tasks if todo[t.index]]
    sizes_of = {t.index: tuple(sorted({encoders[e].spec.input_size for e in todo[t.index]}))
                for t in pending}
    run_id = f"{phase}-{utc_now()}-{uuid.uuid4().hex[:8]}"          # unique per run
    run = {"id": run_id, "phase": phase, "started_utc": utc_now(), "commit": commit,
           "environment": environment, "precision": mode, "tf32": tf32_settings(),
           "batch": {e: batch[e] for e in names}, "chunk": chunk, "workers": workers,
           "normalize": normalize, "progress_repaired_bytes": repaired,
           "encoder": None, "videos_requested": len(tasks), "videos_pending": len(pending)}
    for e in names:
        _update_meta(dirs[e], {**run, "encoder": infos[e]})
    consumer = _Consumer(stores, dirs, layout, encoders, device, mode, batch, dtypes, phase,
                         todo, normalize,
                         tags={"run": run_id, "commit": (commit or {}).get("commit")})
    log_access(access_log, [{"utc": utc_now(), "command": command, "dataset": layout.dataset,
                             "index": t.index, "name": t.name, "path": t.rel_path,
                             "purpose": phase} for t in pending])
    t0 = time.perf_counter()
    if workers > 0:
        _run_workers(pending, sizes_of, chunk, consumer, workers, queue_size)
    else:
        _run_inline(pending, sizes_of, chunk, consumer)
    seconds = time.perf_counter() - t0
    for e in names:
        stores[e].flush()
        rebuild_index(dirs[e], layout)
    end = {"finished_utc": utc_now(), "seconds": round(seconds, 3),
           "videos_finished": len(consumer.finished), "snippets": consumer.snippets,
           "errors": {str(k): v for k, v in sorted(consumer.errors.items())},
           "worker_failures": list(consumer.failures),
           "encode_seconds": {e: round(s, 3) for e, s in consumer.encode_seconds.items()}}
    for e in names:
        _update_meta(dirs[e], {**run, "encoder": infos[e], **end}, key=run_id)
    return {**run, **end, "encoder": infos}


def store_dirs(paths: Paths, dataset: str, encoders: Sequence[str]) -> dict[str, Path]:
    return {e: dense_dir(paths, dataset, e) for e in encoders}


def tasks_for(paths: Paths, dataset: str, manifest: TrainManifest, indices: Sequence[int],
              renumber: bool = False) -> list[VideoTask]:
    """Tasks of manifest videos (paths from the manifest only). ``renumber``: task k is video k
    of ``Layout.subset(indices)`` (throughput runs)."""
    return [VideoTask(k if renumber else int(i),
                      os.fspath(video_path(paths, dataset, manifest, int(i))),
                      int(manifest.total_frames[int(i)]), manifest.names[int(i)],
                      rel_video_path(dataset, manifest, int(i)))
            for k, i in enumerate(indices)]


BENCH_SEED, BENCH_PER_DATASET = 20261007, 10


def bench_sample(split: ValSplit) -> list[int]:
    """The fixed throughput sample: 10 validation videos, default_rng(20261007), in order."""
    rng = np.random.default_rng(BENCH_SEED)
    return sorted(int(i) for i in rng.choice(sorted(split.val_indices), BENCH_PER_DATASET,
                                             replace=False))


def projected_hours(result: Mapping, stage_snippets: Mapping[str, int]) -> dict:
    """Hours of each stage at the measured snippets per second (all encoders of the run)."""
    rate = result["snippets"] / result["seconds"] if result["seconds"] > 0 else float("nan")
    return {"snippets_per_second": rate,
            **{k: (n / rate / 3600 if rate > 0 else None) for k, n in stage_snippets.items()}}
