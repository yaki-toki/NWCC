"""Extraction gates of E021 (PLAN.md "추출 관문"): (x1) code equivalence, (x2) environment
equivalence, floors, frame hashes, and the checks (x3a), (x3), (x4), (x5), (x6).

(x1) (local GPU, earlier precision and batches; legacy decode rules): XD, 12 training-part
videos (6 violent, 6 normal, T <= 6000) at the segment starts, all 32 clips at once in fp32 with
the TF32 flags at their defaults; per (dataset, encoder) max e <= 1e-4 and median e <= 1e-5,
and the start shifted by +2 frames (-2 at the end) gives e >= 1e-3 for at least half the clips.
UCF, 10 validation videos (5 anomalous, 5 normal, T <= 3000): per encoder and segment every
allowed start of the earlier training draw is encoded (autocast, batches 8 / 8 / 4) and the
start s* with the smallest e is kept; max e(s*) <= 5e-3, median <= 5e-4, and among segments
with >= 2 allowed starts the second smallest e exceeds 5e-3 for at least half. e is the clip's
relative L2 distance to the cache row; the new feature is first cast to the cache dtype (UCF
SigLIP: float16 values compared). Samples: a fresh default_rng(20261007) per dataset, anomalous
videos drawn first. Floors: the same clips in the other precision modes.
(x2) (extraction environment, production decoder, precision and batches): the (x1) videos at
the XD starts and the UCF s* of each encoder; every clip relative L2 <= 1e-2 and cosine >=
0.9999; decoded-frame sha256 compared with the (x1) record (mismatch share reported).
Status codes: "pass_bit" (every clip bit-equal), "pass_tolerance", "fail".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from wvad.data.dense import FEATURES_FILE, read_index, read_json, store_dir, variant_dir
from wvad.data.features import load_cache
from wvad.data.manifests import TrainManifest
from wvad.data.splits import ValSplit, check_labels
from wvad.extract import grid, legacy
from wvad.extract.dense import names_sha256
from wvad.extract.derive import (
    INDICES_FILE,
    MEAN_VARIANTS,
    STAGE1_KEEP,
    data_file,
    fit_scale,
    snippet_indices,
)
from wvad.extract.encoders import (
    LEGACY_BATCH,
    LEGACY_PRECISION,
    Encoder,
    precision,
    same_environment,
)
from wvad.extract.encoders import spec as encoder_spec
from wvad.extract.preprocess import black, clips_to_pixels, make_normalizer
from wvad.extract.video import FrameDecoder, rgb_sizes
from wvad.prep.manifests import sha256_file

SEED = 20261007
X1_SAMPLE = {"xdv": ("trainpart", 6, 6, 6000), "ucf": ("val", 5, 5, 3000)}
TAU_FP32, TAU_FP32_MEDIAN, TAU_AMP, TAU_AMP_MEDIAN = 1e-4, 1e-5, 5e-3, 5e-4
X2_L2, X2_COS = 1e-2, 0.9999
X3_REPLACED, X3A_DELTA = 0.01, 16
PASS_BIT, PASS_TOL, FAIL = "pass_bit", "pass_tolerance", "fail"


def x1_sample(dataset: str, manifest: TrainManifest, split: ValSplit) -> list[int]:
    """The (x1) / (x2) videos (manifest indices, sorted)."""
    part, n_anom, n_norm, max_t = X1_SAMPLE[dataset]
    pool = sorted(split.val_indices if part == "val" else split.train_indices)
    pool = [i for i in pool if manifest.total_frames[i] <= max_t]
    rng = np.random.default_rng(SEED)
    picks = []
    for label, k in ((1, n_anom), (0, n_norm)):
        group = [i for i in pool if manifest.labels[i] == label]
        picks += [int(i) for i in rng.choice(group, k, replace=False)]
    return sorted(picks)


# --- clips, encoding, errors ----------------------------------------------------------------

@dataclass
class Clips:
    """Decoded clips: slot [clip][frame] = {input size: uint8 RGB frame} (shared arrays)."""

    slots: list[list[dict]]
    hashes: dict[str, str]

    def __len__(self) -> int:
        return len(self.slots)

    def stack(self, size: int, idx: Sequence[int]) -> np.ndarray:
        """(len(idx), 8, size, size, 3) uint8 clips."""
        return np.stack([np.stack([slot[size] for slot in self.slots[i]]) for i in idx])


def read_clips(path: str, dataset: str, starts: Sequence[int], total_frames: int,
               sizes: Sequence[int], decoder: str) -> Clips:
    """Clips at ``starts`` and the decoded-frame sha256s. ``decoder``: "legacy" (the dataset's
    earlier rule) or "production"."""
    convert = rgb_sizes(sizes)
    frames = legacy.clip_frames(starts, total_frames)
    if decoder == "legacy":
        read = legacy.read_ucf if dataset == "ucf" else legacy.read_xd
        out = read(path, starts, convert, {s: black(s) for s in sizes}, total_frames,
                   hashes=True)
        if out is None:
            raise OSError(f"cannot read {path}")
        slots, hashes = out.slots, out.frame_hashes
    elif decoder == "production":
        dec = FrameDecoder(path, total_frames, np.unique(frames), convert=convert, hashes=True)
        images = {f: image for f, image, _ in dec}
        slots, hashes = [[images[f] for f in clip] for clip in frames.tolist()], dec.frame_hashes
    else:
        raise ValueError(f"unknown decoder {decoder!r}")
    return Clips(slots, {str(k): v for k, v in (hashes or {}).items()})


def encode_clips(encoder: Encoder, clips: Clips, device: torch.device, mode: str, batch: int,
                 idx: Sequence[int] | None = None, normalize: str = "lut") -> np.ndarray:
    """(len(idx), D) features of the clips ``idx`` (default all) in the encoder's output dtype;
    a short last batch is padded with copies of its last clip, so every forward pass has
    ``batch`` clips (as the earlier runs: 32 clips in batches of 8, 4 or 32)."""
    norm = make_normalizer(encoder.spec.mean, encoder.spec.std, device, normalize)
    order = list(range(len(clips))) if idx is None else [int(i) for i in idx]
    out = []
    for a in range(0, len(order), batch):
        part = order[a:a + batch]
        pad = batch - len(part)
        x = clips.stack(encoder.spec.input_size, part + [part[-1]] * pad)
        with precision(mode, device):
            feats = encoder.encode(clips_to_pixels(torch.from_numpy(x).to(device), norm))
        out.append(feats[:batch - pad].numpy())
    return np.concatenate(out)


def rel_l2(new: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """Per row ||new - cache|| / ||cache||, new cast to the cache dtype, in float64."""
    a = np.asarray(new).astype(cache.dtype).astype(np.float64)
    b = np.asarray(cache, dtype=np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def cosine(new: np.ndarray, cache: np.ndarray) -> np.ndarray:
    a = np.asarray(new).astype(cache.dtype).astype(np.float64)
    b = np.asarray(cache, dtype=np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def bit_equal(new: np.ndarray, cache: np.ndarray) -> np.ndarray:
    return np.all(np.asarray(new).astype(cache.dtype) == cache, axis=-1)


def shifted_starts(starts: np.ndarray, total_frames: int) -> np.ndarray:
    """+2 frames, or -2 when the shifted clip would pass the last frame."""
    span = legacy.CLIP_STRIDE * (legacy.CLIP_LEN - 1)
    return np.where(starts + 2 + span <= total_frames - 1, starts + 2, starts - 2)


def _status(ok: bool, all_bit: bool) -> str:
    return (PASS_BIT if all_bit else PASS_TOL) if ok else FAIL


def summarize_x1(dataset: str, rows: Sequence[Mapping]) -> dict:
    """Pass rule of one (dataset, encoder) from its clip rows (PLAN.md (x1))."""
    e = np.array([r["e"] for r in rows])
    out = {"clips": len(rows), "max_e": float(e.max()), "median_e": float(np.median(e)),
           "bit_equal": int(sum(r["bit_equal"] for r in rows))}
    if dataset == "xdv":
        shift = np.array([r["e_shift"] for r in rows])
        info = float(np.mean(shift >= 10 * TAU_FP32))
        ok = out["max_e"] <= TAU_FP32 and out["median_e"] <= TAU_FP32_MEDIAN and info >= 0.5
        return {**out, "info_share": info, "info_denominator": len(rows),
                "status": _status(ok, out["bit_equal"] == len(rows))}
    second = [r["e_second"] for r in rows if r["n_allowed"] >= 2]
    info = float(np.mean(np.array(second) > TAU_AMP)) if second else None
    ok = (out["max_e"] <= TAU_AMP and out["median_e"] <= TAU_AMP_MEDIAN
          and (info is None or info >= 0.5))
    return {**out, "info_share": "n/a" if info is None else info,
            "info_denominator": len(second), "status": _status(ok, out["bit_equal"] == len(rows))}


def _cache_rows(paths, dataset: str, encoders: Sequence[str]) -> dict[str, list[torch.Tensor]]:
    return {e: load_cache(paths.feature_dir(f"{dataset}_{e}"), "train")["features"]
            for e in encoders}


def run_x1(paths, dataset: str, manifest: TrainManifest, split: ValSplit,
           encoders: Mapping[str, Encoder], device: torch.device, video_path,
           floors: Sequence[str] = ("fp32", "fp32_strict", "autocast"),
           mode: str | None = None, batch: Mapping[str, int] | None = None) -> dict:
    """(x1) on this machine; ``video_path(index)`` gives the file (manifest paths only).
    ``mode`` / ``batch`` default to the earlier precision and batches (PLAN.md)."""
    mode = mode or LEGACY_PRECISION[dataset]
    batch = batch or LEGACY_BATCH[dataset]
    floor_modes = [m for m in floors if m != mode]
    cache = _cache_rows(paths, dataset, list(encoders))
    sizes = sorted({enc.spec.input_size for enc in encoders.values()})
    videos = x1_sample(dataset, manifest, split)
    rows = {e: [] for e in encoders}
    floor_e = {e: {m: [] for m in floor_modes} for e in encoders}
    positions: dict[str, dict[str, list[int]]] = {e: {} for e in encoders}
    hashes = {}
    for v in videos:
        t = int(manifest.total_frames[v])
        if dataset == "xdv":
            starts = legacy.xd_starts(t)
            clips = read_clips(video_path(v), dataset, starts, t, sizes, "legacy")
            shifted = read_clips(video_path(v), dataset, shifted_starts(starts, t), t, sizes,
                                 "legacy")
            hashes[str(v)] = clips.hashes
            for e, enc in encoders.items():
                ref = cache[e][v].numpy()
                new = encode_clips(enc, clips, device, mode, batch[e])
                e_main = rel_l2(new, ref)
                e_shift = rel_l2(encode_clips(enc, shifted, device, mode, batch[e]), ref)
                eq = bit_equal(new, ref)
                rows[e] += [{"video": v, "k": k, "start": int(starts[k]), "e": float(e_main[k]),
                             "bit_equal": bool(eq[k]), "e_shift": float(e_shift[k])}
                            for k in range(len(starts))]
                for m in floor_modes:
                    floor_e[e][m] += rel_l2(encode_clips(enc, clips, device, m, batch[e]),
                                            ref).tolist()
            continue
        allowed = [legacy.ucf_allowed_starts(t, k) for k in range(legacy.SEGMENTS)]
        all_starts = np.concatenate(allowed)
        clips = read_clips(video_path(v), dataset, all_starts, t, sizes, "legacy")
        hashes[str(v)] = clips.hashes
        owner = np.repeat(np.arange(legacy.SEGMENTS), [len(a) for a in allowed])
        for e, enc in encoders.items():
            ref = cache[e][v].numpy()
            new = encode_clips(enc, clips, device, mode, batch[e])
            err = rel_l2(new, ref[owner])
            best = []
            for k in range(legacy.SEGMENTS):
                idx = np.flatnonzero(owner == k)
                order = idx[np.argsort(err[idx], kind="stable")]
                j = int(order[0])
                best.append(j)
                rows[e].append({"video": v, "k": k, "start": int(all_starts[j]),
                                "e": float(err[j]), "bit_equal": bool(bit_equal(new[j], ref[k])),
                                "n_allowed": len(idx),
                                "e_second": float(err[order[1]]) if len(idx) > 1 else None})
            positions[e][str(v)] = [int(all_starts[j]) for j in best]
            for m in floor_modes:
                floor_e[e][m] += rel_l2(encode_clips(enc, clips, device, m, batch[e], idx=best),
                                        ref).tolist()
    return {"check": "x1", "dataset": dataset, "videos": videos,
            "names": [manifest.names[v] for v in videos], "precision": mode,
            "batch": {e: batch[e] for e in encoders}, "decoder": "legacy",
            "encoders": {e: {"summary": summarize_x1(dataset, rows[e]), "rows": rows[e],
                             "floors": {m: _dist(floor_e[e][m]) for m in floor_modes},
                             "info": enc.info()} for e, enc in encoders.items()},
            "positions": positions if dataset == "ucf" else None, "frame_hashes": hashes}


def _dist(values: Sequence[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)) if a.size else None,
            "max": float(a.max()) if a.size else None}


def run_x2(paths, dataset: str, manifest: TrainManifest, x1: Mapping,
           encoders: Mapping[str, Encoder], device: torch.device, video_path,
           batch: Mapping[str, int], mode: str = "autocast", normalize: str = "lut") -> dict:
    """(x2) in this environment, on the (x1) record's videos and positions."""
    cache = _cache_rows(paths, dataset, list(encoders))
    rows = {e: [] for e in encoders}
    hashes, mismatch, compared = {}, 0, 0
    for v in x1["videos"]:
        t = int(manifest.total_frames[v])
        for e, enc in encoders.items():                  # UCF positions differ per encoder
            starts = (legacy.xd_starts(t) if dataset == "xdv"
                      else np.asarray(x1["positions"][e][str(v)], dtype=np.int64))
            clips = read_clips(video_path(v), dataset, starts, t, [enc.spec.input_size],
                               "production")
            hashes.setdefault(str(v), {}).update(clips.hashes)
            ref = cache[e][v].numpy()
            new = encode_clips(enc, clips, device, mode, batch[e], normalize=normalize)
            l2, cos, eq = rel_l2(new, ref), cosine(new, ref), bit_equal(new, ref)
            rows[e] += [{"video": v, "k": k, "start": int(starts[k]), "e": float(l2[k]),
                         "cos": float(cos[k]), "bit_equal": bool(eq[k]),
                         "ok": bool(l2[k] <= X2_L2 and cos[k] >= X2_COS)}
                        for k in range(len(starts))]
        local = x1["frame_hashes"].get(str(v), {})
        for f, sha in hashes[str(v)].items():
            if f in local:
                compared += 1
                mismatch += sha != local[f]
    summary = {}
    for e in encoders:
        r = rows[e]
        ok = all(x["ok"] for x in r)
        summary[e] = {"clips": len(r), "max_e": max(x["e"] for x in r),
                      "median_e": float(np.median([x["e"] for x in r])),
                      "min_cos": min(x["cos"] for x in r),
                      "status": _status(ok, all(x["bit_equal"] for x in r))}
    return {"check": "x2", "dataset": dataset, "videos": list(x1["videos"]), "precision": mode,
            "batch": dict(batch), "decoder": "production", "normalize": normalize,
            "encoders": {e: {"summary": summary[e], "rows": rows[e],
                             "info": encoders[e].info()} for e in encoders},
            "frame_hashes": hashes,
            "frame_hash_mismatch": {"compared": compared, "different": mismatch,
                                    "share": mismatch / compared if compared else None}}


# --- (x3a), (x6): headers, sha256 and the access log ----------------------------------------

def check_x3a(rows: Sequence[Mapping]) -> dict:
    """|header T - manifest T| <= 16 for every video (rows of the header command)."""
    bad = [r for r in rows if abs(int(r["header_frames"]) - int(r["manifest_frames"]))
           > X3A_DELTA]
    deltas = [int(r["header_frames"]) - int(r["manifest_frames"]) for r in rows]
    return {"videos": len(rows), "nonzero_delta": sum(d != 0 for d in deltas),
            "max_abs_delta": max((abs(d) for d in deltas), default=0),
            "violations": [{k: r[k] for k in ("dataset", "index", "name", "header_frames",
                                             "manifest_frames")} for r in bad],
            "status": FAIL if bad else "pass"}


def check_x6(server: Sequence[Mapping], local: Sequence[Mapping], access: Sequence[Mapping],
             allowed_paths: Mapping[str, set[str]]) -> dict:
    """Server sha256 = local sha256 for every file the server opened; every logged path is a
    training-manifest path (positive check)."""
    loc = {(r["dataset"], r["path"]): r["sha256"] for r in local}
    srv = {(r["dataset"], r["path"]): r["sha256"] for r in server}
    differ = [k for k, sha in srv.items() if loc.get(k) != sha]
    outside = [r for r in access if r["path"] not in allowed_paths.get(r["dataset"], set())]
    opened = {(r["dataset"], r["path"]) for r in access}
    unhashed = sorted(opened - set(srv))
    ok = not differ and not outside and not unhashed
    return {"server_files": len(srv), "sha_mismatch": [list(k) for k in differ],
            "outside_training_manifest": len(outside), "opened_without_sha": len(unhashed),
            "status": "pass" if ok else FAIL}


# --- (x3), (x4): the dense caches -----------------------------------------------------------

def check_x3(folder, manifest: TrainManifest, videos: Sequence[int],
             x3a_passed: Mapping[int, bool]) -> dict:
    """Every phase video grabbed T times; replaced share of its used frames <= 1% (frames at
    or after the header count excluded for videos that passed (x3a))."""
    ix = read_index(folder)
    over, short = [], []
    for v in videos:
        t, h = int(manifest.total_frames[v]), int(ix.header_frames[v])
        if int(ix.grab_attempts[v]) != t:
            short.append(v)
        used, rep = int(ix.used_frames[v]), int(ix.replaced[v])
        if x3a_passed.get(v, False) and 0 <= h < t:
            used -= int((grid.used_frames(t) >= h).sum())
            rep -= int(ix.replaced_tail[v])
        if used <= 0 or rep / used > X3_REPLACED:
            over.append({"index": v, "name": manifest.names[v], "replaced": rep, "used": used})
    return {"videos": len(videos), "grab_attempts_not_T": short, "over_1pct": over,
            "status": FAIL if over or short else "pass"}


def check_x4(folder, manifest: TrainManifest, videos: Sequence[int], encoder: str,
             commit: str, environment: Mapping) -> dict:
    """Completeness of one encoder cache for a phase (PLAN.md (x4))."""
    s = encoder_spec(encoder)
    ix, meta = read_index(folder), read_json(store_dir(folder) / "meta.json")
    t = np.asarray(manifest.total_frames)
    problems = []
    if meta.get("names_sha256") != names_sha256(manifest.names) or \
            meta.get("videos") != len(manifest):
        problems.append("videos or their order differ from the manifest")
    if not np.array_equal(ix.labels, np.asarray(manifest.labels)):
        problems.append("labels differ from the manifest")
    if not np.array_equal(ix.n, [grid.n_snippets(x) for x in t]):
        problems.append("n_v != ceil(T / 16)")
    arr = np.load(store_dir(folder) / FEATURES_FILE, mmap_mode="r")
    if arr.shape[0] != ix.rows:
        problems.append("rows != sum of n_v")
    if not all(ix.done[v] for v in videos):
        problems.append(f"{sum(not ix.done[v] for v in videos)} videos not done")
    nonfinite = [v for v in videos if ix.done[v] and not np.isfinite(
        np.asarray(arr[int(ix.start[v]):int(ix.start[v] + ix.n[v])], dtype=np.float32)).all()]
    if nonfinite:
        problems.append(f"non-finite rows in videos {nonfinite[:5]}")
    if meta.get("revision") != s.revision:
        problems.append("model revision differs from the pinned one")
    runs = [r for r in meta["runs"] if r.get("videos_finished")]
    for r in runs:
        enc = r.get("encoder") or {}
        if enc.get("loaded_commit") not in (None, s.revision):
            problems.append(f"run {r['id']}: loaded revision {enc.get('loaded_commit')}")
        if (r.get("commit") or {}).get("commit") != commit:
            problems.append(f"run {r['id']}: commit {(r.get('commit') or {}).get('commit')}")
        if not same_environment(r.get("environment") or {}, environment):
            problems.append(f"run {r['id']}: environment differs from env_extract.json")
        want_attn = (environment.get("encoders") or {}).get(encoder, {}).get(
            "attn_implementation")
        if want_attn is not None and enc.get("attn_implementation") != want_attn:
            problems.append(f"run {r['id']}: attention {enc.get('attn_implementation')} != "
                            f"{want_attn}")
    return {"videos": len(videos), "runs": len(runs), "problems": problems,
            "status": FAIL if problems else "pass"}


# --- (x5): derived caches -------------------------------------------------------------------

def check_x5(folder, layout, split: ValSplit, dataset: str, encoder: str, variants: Sequence[str],
             stage: int, n_random: int = 20) -> dict:
    """Derived caches of one encoder against the snippets (PLAN.md (x5))."""
    store = np.load(store_dir(folder) / FEATURES_FILE, mmap_mode="r")
    val, train = sorted(split.val_indices), sorted(split.train_indices)
    rng = np.random.default_rng(SEED)
    sample = sorted(int(i) for i in rng.choice(val if stage == 1 else range(len(layout.names)),
                                               min(n_random, len(val)), replace=False))
    out = {}
    for variant in variants:
        vd = variant_dir(folder, variant)
        meta = read_json(vd / "meta.json")
        data = load_cache(vd, "train")
        problems = []
        if meta.get("data_sha256") != sha256_file(data_file(vd)):
            problems.append("data.pt sha256 differs from meta")
        try:
            check_labels(split, data["labels"], f"{vd}")
        except ValueError as exc:
            problems.append(str(exc))
        if len(data.get("total_frames") or []) != len(layout.names) or \
                list(data["total_frames"]) != [int(x) for x in layout.total_frames]:
            problems.append("total_frames differ from the manifest")
        if len(data["seg_labels"]) != len(layout.names):
            problems.append("seg_labels missing")
        feats = data["features"]
        want_filled = "val" if (stage == 1 or variant in ("d16r1", "d16r2")) else "all"
        if meta.get("filled") != want_filled:
            problems.append(f"filled {meta.get('filled')} != {want_filled}")
        rows = val if meta.get("filled") == "val" else range(len(layout.names))
        if meta.get("filled") == "val" and not all(torch.isnan(feats[v]).all() for v in train):
            problems.append("training-part rows are not all NaN")
        if not all(torch.isfinite(feats[v].float()).all() for v in rows):
            problems.append("non-finite rows among the filled rows")
        if variant in MEAN_VARIANTS:
            problems += _check_means(store, layout, dataset, encoder, variant, meta, feats,
                                     sample, train)
        else:
            idx = snippet_indices(dataset, encoder, variant, layout.total_frames)
            if not np.array_equal(np.load(vd / INDICES_FILE), idx):
                problems.append("indices.npy differs from the grid rule")
            for v in rows:
                a = int(layout.starts[v])
                ref = torch.from_numpy(np.array(store[a + idx[v]]))
                if not torch.equal(feats[v], ref):
                    problems.append(f"video {v}: rows differ from the snippets")
                    break
            if stage == 2 and variant in ("d16c", "d16r"):
                keep = load_cache(vd / STAGE1_KEEP, "train")["features"]
                if not all(torch.equal(keep[v], feats[v]) for v in val):
                    problems.append("validation rows differ from the stage-1 file")
        out[variant] = {"problems": problems, "status": FAIL if problems else "pass"}
    return out


def _check_means(store, layout, dataset, encoder, variant, meta, feats, sample, train) -> list:
    problems = []
    scale = fit_scale(store, layout, dataset, encoder, variant, train)
    rec = meta.get("scale") or {}
    for k in ("mu_c", "sigma_c", "mu_v", "sigma_v"):
        if not np.allclose(np.asarray(rec.get(k, []), dtype=np.float64), getattr(scale, k),
                           rtol=1e-12, atol=0):
            problems.append(f"scale {k} differs from a refit on the training part")
    if rec.get("fit_rows") != len(train) * grid.WINDOWS:
        problems.append("scale map not fitted on exactly the training-part rows")
    worst = 0.0
    for v in sample:                      # recomputed independently of derive.window_means
        a, t = int(layout.starts[v]), int(layout.total_frames[v])
        rows = np.asarray(store[a:a + int(layout.counts[v])], dtype=np.float64)
        sets = (grid.window_members(t) if variant == "d16m" else grid.m_members(t)).sets
        means = np.stack([rows[j].sum(axis=0) / len(j) for j in sets])
        want = scale.apply(means.astype(np.float32))
        worst = max(worst, float(np.abs(feats[v].numpy() - want).max()))
    if not math.isfinite(worst) or worst > 1e-6:
        problems.append(f"recomputed means differ by {worst}")
    return problems
