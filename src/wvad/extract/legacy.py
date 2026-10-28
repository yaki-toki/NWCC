"""Clip positions and decode rules of the earlier 32-clip caches (E021 gates (x0)-(x2) only).

Ported from the previous project (the previous project's ``extract/reader.py`` ``read_video_segments_fast`` for
UCF-Crime, ``extract_features_xdviolence.py`` ``read_segments_from_video`` for XD-Violence).
Both use 32 segments of L = max(1, T // 32) frames and 8-frame clips with step 2 (frames
min(s + 2 i, T - 1)); T is the header frame count in the earlier code and is passed explicitly
here (the gates pass the manifest T, which is that header count of the extraction machine).

Positions
    ucf_centre_starts   s = k L + (L - 16) // 2, clipped to [0, T - 16] (UCF test caches)
    ucf_random_starts   np.random.randint over [k L, max(k L, min((k + 1) L - 16, T - 16))],
                        one draw per segment in order from the global numpy stream (UCF
                        training caches, drawn independently per encoder, no seed)
    ucf_allowed_starts  that range for one segment
    xd_starts           s = k L (XD caches)

Decode rules (failures handled differently, so both are kept)
    UCF  the needed frames in increasing order: grab() over skipped frames (failures ignored),
         read() at needed ones; a failed read leaves the frame missing, filled with the
         previous decoded frame of the same clip, else black
    XD   grab() frame by frame up to the last needed frame, retrieve() at needed ones; decoding
         stops at the first failed grab; a missing frame takes the last decoded frame seen in
         clip order (from segment 0), else black

The readers return each clip slot as ``convert(frame)`` (the caller's uint8 RGB conversion) and
which slots were filled, so the earlier float tensor is ``legacy_tensor`` of the slots.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from wvad.extract.preprocess import black, normalize_numpy, to_rgb
from wvad.extract.video import frame_sha256

SEGMENTS = 32
CLIP_LEN = 8
CLIP_STRIDE = 2
CLIP_SPAN = CLIP_LEN * CLIP_STRIDE
Convert = Callable[[np.ndarray], object]


def segment_length(total_frames: int) -> int:
    return max(1, int(total_frames) // SEGMENTS)


def ucf_centre_starts(total_frames: int) -> np.ndarray:
    t, length = int(total_frames), segment_length(total_frames)
    starts = [k * length + (length - CLIP_SPAN) // 2 for k in range(SEGMENTS)]
    return np.array([max(0, min(s, t - CLIP_SPAN)) for s in starts], dtype=np.int64)


def ucf_allowed_starts(total_frames: int, k: int) -> np.ndarray:
    """Start positions the earlier training draw could take in segment k."""
    t, length = int(total_frames), segment_length(total_frames)
    seg_start, seg_end = k * length, (k + 1) * length
    max_start = max(seg_start, min(seg_end - CLIP_SPAN, t - CLIP_SPAN))
    return np.arange(seg_start, max(seg_start + 1, max_start + 1), dtype=np.int64)


def ucf_random_starts(total_frames: int, rng=np.random) -> np.ndarray:
    """One ``rng.randint`` per segment, in order, as the earlier training reader drew them."""
    out = []
    for k in range(SEGMENTS):
        allowed = ucf_allowed_starts(total_frames, k)
        out.append(int(rng.randint(int(allowed[0]), int(allowed[-1]) + 1)))
    return np.array(out, dtype=np.int64)


def xd_starts(total_frames: int) -> np.ndarray:
    return np.arange(SEGMENTS, dtype=np.int64) * segment_length(total_frames)


def clip_frames(starts: Sequence[int], total_frames: int) -> np.ndarray:
    """(n_clips, 8) frames max(0, min(s + 2 i, T - 1))."""
    s = np.asarray(starts, dtype=np.int64)[:, None]
    return np.maximum(0, np.minimum(s + CLIP_STRIDE * np.arange(CLIP_LEN), int(total_frames) - 1))


@dataclass
class LegacyClips:
    slots: list[list[object]]          # [clip][frame] convert(frame) (or the black frame)
    missing: np.ndarray                # (n_clips, 8) bool: slot filled by the fallback rule
    total_frames: int
    frame_hashes: dict[int, str] | None = None


def _capture(path: str | os.PathLike[str]):
    import cv2  # optional dependency (extract)

    cap = cv2.VideoCapture(os.fspath(path))
    if not cap.isOpened():
        cap.release()
        return cv2, None
    return cv2, cap


def _total(cv2, cap, total_frames: int | None) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if total_frames is None else int(total_frames)


def read_ucf(path: str | os.PathLike[str], starts: Sequence[int], convert: Convert,
             black_frame: object, total_frames: int | None = None,
             hashes: bool = False) -> LegacyClips | None:
    """The earlier UCF decode rule (None when the video cannot be opened or T < 1)."""
    cv2, cap = _capture(path)
    if cap is None:
        return None
    t = _total(cv2, cap, total_frames)
    if t < 1:
        cap.release()
        return None
    frames = clip_frames(starts, t)
    decoded: dict[int, np.ndarray] = {}
    current = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for target in sorted(set(frames.ravel().tolist())):
        if target < current:                              # never: targets are increasing
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current = target
        while current < target:
            cap.grab()
            current += 1
        ok, frame = cap.read()
        current += 1
        if ok and frame is not None:
            decoded[target] = frame
    cap.release()
    converted = {f: convert(fr) for f, fr in decoded.items()}
    slots, missing = [], np.zeros(frames.shape, dtype=bool)
    for c, clip in enumerate(frames.tolist()):
        row, last_good = [], None
        for i, f in enumerate(clip):
            if f in converted:
                last_good = converted[f]
                row.append(last_good)
            else:
                missing[c, i] = True
                row.append(last_good if last_good is not None else black_frame)
        slots.append(row)
    hashed = {f: frame_sha256(fr) for f, fr in decoded.items()} if hashes else None
    return LegacyClips(slots, missing, t, hashed)


def read_xd(path: str | os.PathLike[str], starts: Sequence[int], convert: Convert,
            black_frame: object, total_frames: int | None = None,
            hashes: bool = False) -> LegacyClips | None:
    """The earlier XD decode rule (None when the video cannot be opened or T < 1)."""
    cv2, cap = _capture(path)
    if cap is None:
        return None
    t = _total(cv2, cap, total_frames)
    if t < 1:
        cap.release()
        return None
    frames = clip_frames(starts, t)
    wanted = {f: None for f in frames.ravel().tolist()}
    pos, last_needed = 0, max(wanted)
    while pos <= last_needed:
        if not cap.grab():
            break
        if pos in wanted:
            _, frame = cap.retrieve()
            wanted[pos] = frame
        pos += 1
    cap.release()
    decoded = {f: fr for f, fr in wanted.items() if fr is not None}
    converted = {f: convert(fr) for f, fr in decoded.items()}
    slots, missing = [], np.zeros(frames.shape, dtype=bool)
    last_good = None
    for c, clip in enumerate(frames.tolist()):
        row = []
        for i, f in enumerate(clip):
            if f in converted:
                last_good = converted[f]
                row.append(last_good)
            else:
                missing[c, i] = True
                row.append(last_good if last_good is not None else black_frame)
        slots.append(row)
    hashed = {f: frame_sha256(fr) for f, fr in decoded.items()} if hashes else None
    return LegacyClips(slots, missing, t, hashed)


def rgb_converter(size: int) -> tuple[Convert, np.ndarray]:
    """(convert, black frame) producing uint8 RGB frames of one input size."""
    return (lambda frame: to_rgb(frame, size)), black(size)


def legacy_tensor(clips: LegacyClips, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """The earlier readers' output: (n_clips, 3, 8, H, W) float32, numpy normalisation."""
    arr = np.stack([np.stack(row) for row in clips.slots])          # (n, 8, H, W, 3) uint8
    norm = normalize_numpy(arr, mean, std)
    return torch.from_numpy(np.ascontiguousarray(norm.transpose(0, 4, 1, 2, 3)))
