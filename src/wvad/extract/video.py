"""Production decoding of the dense extraction (E021 PLAN.md "해독(생산)").

Every frame 0 .. T - 1 (T = manifest total_frames) is grabbed in order, also after a failed
grab, and no frame after T - 1 is read; only the frames the encoder uses are retrieved. A used
frame whose grab or retrieve failed is replaced by the last successfully decoded frame of the
whole video or, before the first success, by the next successful one (black when the video has
none). Recorded per video: grab attempts and successes, the distinct used frames and how many
of them were replaced, and how many of those lie at or after the header frame count (the
[header T, manifest T) part that gate (x3) counts separately, PLAN.md N3); per snippet, how
many of its 8 inputs were replaced.

The two earlier readers (``wvad.extract.legacy``) handle failures differently; on a video
without a decode failure all three return the same frames (gate (x0)).

The decode workers of the dense extraction (``video_messages``, ``decode_worker``) live here
too: they need numpy and OpenCV only (no torch), so spawned worker processes stay light.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from wvad.extract import grid

Convert = Callable[[np.ndarray | None], object]


def _cv2():
    import cv2  # optional dependency (extract); imported where videos are opened

    return cv2


@dataclass
class DecodeStats:
    total_frames: int                  # manifest T: the number of grab attempts
    header_frames: int = -1            # CAP_PROP_FRAME_COUNT of this machine's decoder
    fps: float = float("nan")          # CAP_PROP_FPS
    attempts: int = 0
    successes: int = 0
    used: int = 0                      # distinct used frames
    replaced: int = 0                  # used frames replaced (grab or retrieve failed)
    replaced_tail: int = 0             # of those, frames >= header_frames (header < T only)

    def as_dict(self) -> dict:
        return {"total_frames": self.total_frames, "header_frames": self.header_frames,
                "fps": self.fps, "grab_attempts": self.attempts,
                "grab_successes": self.successes, "used_frames": self.used,
                "replaced": self.replaced, "replaced_tail": self.replaced_tail}


def open_capture(path: str | os.PathLike[str]):
    cv2 = _cv2()
    cap = cv2.VideoCapture(os.fspath(path))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"cannot open video: {path}")
    return cap


def read_header(path: str | os.PathLike[str]) -> tuple[int, float]:
    """Header frame count and fps (no frame is decoded)."""
    cv2 = _cv2()
    cap = open_capture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def frame_sha256(frame: np.ndarray) -> str:
    """sha256 of a decoded uint8 frame (shape, then the bytes in C order)."""
    a = np.ascontiguousarray(frame)
    return hashlib.sha256(f"{a.shape}|{a.dtype}|".encode() + a.tobytes()).hexdigest()


class FrameDecoder:
    """Iterate (frame, image, replaced) over the used frames of one video, in increasing order.

    ``image`` is ``convert(frame)`` of the decoded BGR frame (``convert(None)`` once when the
    video has no decodable used frame: the caller's black frame); a replaced frame carries the
    image of its substitute. ``stats`` is complete when the iteration ends; with
    ``hashes=True`` ``frame_hashes`` maps every successfully decoded used frame to its
    ``frame_sha256``."""

    def __init__(self, path: str | os.PathLike[str], total_frames: int,
                 used: Sequence[int] | np.ndarray, convert: Convert | None = None,
                 hashes: bool = False):
        t = int(total_frames)
        u = np.asarray(used, dtype=np.int64)
        if t < 1:
            raise ValueError(f"total_frames must be >= 1, got {total_frames}")
        if u.size == 0 or np.any(np.diff(u) <= 0) or u[0] < 0 or u[-1] >= t:
            raise ValueError("used frames must be strictly increasing within [0, T)")
        self.path, self.total_frames, self.used = path, t, u
        self.convert: Convert = convert if convert is not None else (lambda frame: frame)
        self.stats = DecodeStats(total_frames=t, used=int(u.size))
        self.frame_hashes: dict[int, str] | None = {} if hashes else None

    def _failed(self, frame: int) -> None:
        s = self.stats
        s.replaced += 1
        if 0 <= s.header_frames < s.total_frames and frame >= s.header_frames:
            s.replaced_tail += 1

    def __iter__(self) -> Iterator[tuple[int, object, bool]]:
        cv2 = _cv2()
        s = self.stats
        cap = open_capture(self.path)
        try:
            s.header_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            s.fps = float(cap.get(cv2.CAP_PROP_FPS))
            position, last, pending = 0, None, []
            for f in self.used.tolist():
                ok = False
                while position <= f:                       # grab every frame up to f
                    ok = bool(cap.grab())
                    s.attempts += 1
                    s.successes += ok
                    position += 1
                frame = None
                if ok:
                    got, frame = cap.retrieve()
                    frame = frame if got else None
                if frame is not None:
                    if self.frame_hashes is not None:
                        self.frame_hashes[f] = frame_sha256(frame)
                    last = self.convert(frame)
                    for p in pending:                      # failed before the first success
                        yield p, last, True
                    pending = []
                    yield f, last, False
                    continue
                self._failed(f)
                if last is None:
                    pending.append(f)
                else:
                    yield f, last, True
            while position < self.total_frames:            # the rest up to T - 1, never beyond
                s.successes += bool(cap.grab())
                s.attempts += 1
                position += 1
            if pending:                                    # no used frame decoded at all
                black = self.convert(None)
                for p in pending:
                    yield p, black, True
        finally:
            cap.release()


def iter_snippets(decoder: FrameDecoder) -> Iterator[tuple[int, list, int]]:
    """(snippet j, its 8 images in input order, replaced inputs of the 8) for every snippet of
    the grid, from a decoder over ``grid.used_frames(T)``; frames are dropped once used. The
    iteration also exhausts the decoder, so its stats cover all T grabs."""
    t = decoder.total_frames
    if not np.array_equal(decoder.used, grid.used_frames(t)):
        raise ValueError("the decoder must cover the used frames of the snippet grid")
    frames = grid.snippet_frames(t)
    stream = iter(decoder)
    have: dict[int, tuple[object, bool]] = {}
    for j, need in enumerate(frames.tolist()):
        while need[-1] not in have:                        # frames arrive in increasing order
            f, image, replaced = next(stream)
            have[f] = (image, replaced)
        yield j, [have[f][0] for f in need], sum(have[f][1] for f in need)
        for f in [f for f in have if f < grid.SNIPPET * (j + 1)]:
            del have[f]
    for _ in stream:                                       # remaining grabs (stats only)
        pass


# --- decode workers (dense extraction) ------------------------------------------------------

def rgb_sizes(sizes: Sequence[int]) -> Convert:
    """convert(frame) -> {size: uint8 RGB frame} for every input size (black for None)."""
    from wvad.extract.preprocess import black, to_rgb

    return lambda frame: {s: black(s) if frame is None else to_rgb(frame, s) for s in sizes}


def video_messages(video: int, path: str, total_frames: int, sizes: Sequence[int],
                   chunk: int) -> Iterator[tuple]:
    """Messages of one video for the extraction consumer, in order:

        ("start", video)
        ("chunk", video, first snippet, {size: (k, 8, size, size, 3) uint8}, (k,) uint8 replaced)
        ...                                   k = chunk except for the last chunk
        ("done", video, DecodeStats.as_dict(), number of chunks)
    or ("error", video, message) instead of the rest when decoding fails."""
    yield ("start", video)
    try:
        decoder = FrameDecoder(path, total_frames, grid.used_frames(total_frames),
                               convert=rgb_sizes(sizes))
        frames: dict[int, list[np.ndarray]] = {s: [] for s in sizes}
        replaced: list[int] = []
        first, n_chunks = 0, 0
        for _, images, n_replaced in iter_snippets(decoder):
            for s in sizes:
                frames[s].append(np.stack([image[s] for image in images]))
            replaced.append(n_replaced)
            if len(replaced) == chunk:
                yield ("chunk", video, first, {s: np.stack(frames[s]) for s in sizes},
                       np.asarray(replaced, dtype=np.uint8))
                frames, replaced = {s: [] for s in sizes}, []
                first, n_chunks = first + chunk, n_chunks + 1
        if replaced:
            yield ("chunk", video, first, {s: np.stack(frames[s]) for s in sizes},
                   np.asarray(replaced, dtype=np.uint8))
            n_chunks += 1
        yield ("done", video, decoder.stats.as_dict(), n_chunks)
    except Exception as exc:                       # reported; the video stays undone
        yield ("error", video, f"{type(exc).__name__}: {exc}")


def decode_worker(tasks, results, chunk: int) -> None:
    """Worker process: (video, path, T, sizes) tasks until None; OpenCV with one thread."""
    import cv2

    cv2.setNumThreads(1)
    while True:
        task = tasks.get()
        if task is None:
            break
        video, path, total_frames, sizes = task
        for message in video_messages(video, path, total_frames, sizes, chunk):
            results.put(message)
    results.put(("exit", os.getpid()))
