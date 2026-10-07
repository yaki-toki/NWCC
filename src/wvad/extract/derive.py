"""32-unit caches derived from the dense snippets (E021 PLAN.md "팔", "척도 맞춤", R8).

    d16c    window k: the snippet supporting the window centre frame (grid.centre_snippets)
    d16r    UCF: draw r = 0 per encoder (grid.d16r_snippets); XD: the window-start snippet
    d16r1, d16r2   UCF draws r = 1, 2 (scoring only; written in stage 1 only)
    d16m    window k: the mean of its snippets J_k (empty window: its d16c snippet), scale-matched
    d16m2   window k: the mean of pick_m(J_k, 2), scale-matched

Written in the format of the current caches (``wvad.data.features``):
``<dense folder>/<variant>/mil/train/data.pt`` with features (one (32, D) tensor per
training-split video, manifest order), labels (manifest), seg_labels (copied from the earlier
cache of the same encoder) and total_frames (manifest T). ``<variant>/meta.json`` records how
the file was filled, the rule, the source cache, the sha256 of data.pt and, per variant, the
snippet indices (``indices.npy``, (videos, 32)) or the scale-match map; ``wvad.data.dense``
reads the map back.

Stage 1 (filled "val"): d16c, d16r (and UCF d16r1, d16r2) as full-size files whose training-part
rows are NaN (training on them gives a non-finite loss: a numerical failure) and whose
validation rows come from the snippets. Stage 2 (filled "all"): d16c, d16r, d16m, d16m2 with all
rows; the stage-1 files of d16c / d16r are first moved to ``<variant>/stage1_val/`` (gate (x5)
compares the validation rows); d16r1 / d16r2 stay as written in stage 1.

Copies (d16c, d16r*) keep the stored dtype (SigLIP float16) and equal the snippet rows bit for
bit. Means: stored values -> float32 -> mean in float64 -> float32; then the scale-match map
x' = mu_c + (sigma_c / sigma_V)(x - mu_V), fitted per dimension in float64 on the training-part
rows of the variant and of d16c (no labels), is applied to every row.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from wvad.data.dense import (
    FEATURES_FILE,
    META_FILE,
    ScaleMap,
    read_index,
    read_json,
    store_dir,
    variant_dir,
)
from wvad.data.features import load_cache
from wvad.data.splits import ValSplit
from wvad.extract import grid
from wvad.extract.dense import Layout, utc_now, write_meta
from wvad.prep.manifests import sha256_file

SELECT_VARIANTS = ("d16c", "d16r", "d16r1", "d16r2")
MEAN_VARIANTS = ("d16m", "d16m2")
DRAW_OF = {"d16r": 0, "d16r1": 1, "d16r2": 2}
STAGE_VARIANTS = {1: {"ucf": ("d16c", "d16r", "d16r1", "d16r2"), "xdv": ("d16c", "d16r")},
                  2: {"ucf": ("d16c", "d16r", "d16m", "d16m2"),
                      "xdv": ("d16c", "d16r", "d16m", "d16m2")}}
FILLED = {1: "val", 2: "all"}
STAGE1_KEEP = "stage1_val"
INDICES_FILE = "indices.npy"
RULES = {"d16c": "snippet supporting the window centre frame",
         "d16r": "UCF: uniform draw r over J_k per encoder; XD: snippet of the window start",
         "d16r1": "UCF: uniform draw r = 1 over J_k per encoder",
         "d16r2": "UCF: uniform draw r = 2 over J_k per encoder",
         "d16m": "mean over J_k (float64), scale-matched to d16c",
         "d16m2": "mean over pick_m(J_k, 2) (float64), scale-matched to d16c"}


def data_file(directory: Path) -> Path:
    return Path(directory) / "mil" / "train" / "data.pt"


def snippet_indices(dataset: str, encoder: str, variant: str,
                    totals: Sequence[int]) -> np.ndarray:
    """(videos, 32) snippet index per window for a selection variant (all training videos)."""
    if variant == "d16c":
        return np.stack([grid.centre_snippets(t) for t in totals])
    return np.stack(grid.d16r_snippets(dataset, totals, DRAW_OF[variant], encoder))


def window_means(rows: np.ndarray, members: grid.Members) -> np.ndarray:
    """(32, D) float32: per window the float64 mean of the float32 rows of its set."""
    x = np.asarray(rows).astype(np.float32)
    return np.stack([x[s].mean(axis=0, dtype=np.float64) for s in members.sets]
                    ).astype(np.float32)


def mean_members(variant: str, total_frames: int) -> grid.Members:
    return grid.window_members(total_frames) if variant == "d16m" else \
        grid.m_members(total_frames, grid.M_PICK)


def _video_rows(store: np.ndarray, layout: Layout, video: int) -> np.ndarray:
    a = int(layout.starts[video])
    return np.asarray(store[a:a + int(layout.counts[video])])


def _unscaled(store, layout: Layout, variant: str, video: int,
              indices: np.ndarray | None) -> np.ndarray:
    rows = _video_rows(store, layout, video)
    if variant in SELECT_VARIANTS:
        return rows[indices[video]]
    return window_means(rows, mean_members(variant, int(layout.total_frames[video])))


def fit_scale(store, layout: Layout, dataset: str, encoder: str, variant: str,
              train_videos: Sequence[int]) -> ScaleMap:
    """The variant's map from the training-part rows of the variant and of d16c."""
    centre = snippet_indices(dataset, encoder, "d16c", layout.total_frames)
    c_rows = np.concatenate([_unscaled(store, layout, "d16c", v, centre) for v in train_videos])
    v_rows = np.concatenate([_unscaled(store, layout, variant, v, None) for v in train_videos])
    return ScaleMap.fit(c_rows, v_rows)


def _keep_stage1(vd: Path) -> None:
    """Before a stage-2 write: move a stage-1 (filled "val") file aside, once."""
    meta_path = vd / META_FILE
    if not meta_path.exists() or read_json(meta_path).get("filled") != "val":
        return
    keep = vd / STAGE1_KEEP
    if keep.exists():
        raise FileExistsError(f"{keep} exists already; a stage-1 file is not overwritten")
    keep.mkdir()
    for name in ("mil", META_FILE, INDICES_FILE):
        if (vd / name).exists():
            shutil.move(os.fspath(vd / name), os.fspath(keep / name))


def _write(vd: Path, features: list[torch.Tensor], labels: Sequence[int],
           seg_labels: Sequence[torch.Tensor], totals: Sequence[int], meta: dict) -> Path:
    path = data_file(vd)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    torch.save({"features": features, "labels": [int(y) for y in labels],
                "seg_labels": list(seg_labels), "total_frames": [int(t) for t in totals]}, tmp)
    os.replace(tmp, path)
    write_meta(vd / META_FILE, {**meta, "data_sha256": sha256_file(path)})
    return path


def _refuse_after_stage2(folder: Path, dataset: str) -> None:
    """Stage 1 is not derived again once a stage-2 file (filled "all") exists: it would overwrite
    the stage-2 d16c / d16r files (and rewrite d16r1 / d16r2)."""
    done = [v for v in STAGE_VARIANTS[2][dataset]
            if (variant_dir(folder, v) / META_FILE).exists()
            and read_json(variant_dir(folder, v) / META_FILE).get("filled") == FILLED[2]]
    if done:
        raise ValueError(f"{folder}: stage-2 files exist ({', '.join(done)}); stage 1 is not "
                         "derived again")


def derive_variant(folder: Path, layout: Layout, split: ValSplit, dataset: str, encoder: str,
                   variant: str, stage: int, seg_labels: Sequence[torch.Tensor],
                   commit: Mapping | None = None) -> dict:
    """Write one variant of one encoder for stage 1 or 2 (``FILLED``); returns its meta."""
    if variant not in STAGE_VARIANTS[stage][dataset]:
        raise ValueError(f"{variant} is not derived in stage {stage} for {dataset}")
    if stage == 1:
        _refuse_after_stage2(folder, dataset)
    filled = FILLED[stage]
    index = read_index(folder)
    rows_of = sorted(split.val_indices) if filled == "val" else list(range(len(layout.names)))
    missing = [v for v in rows_of if not index.done[v]]
    if missing:
        raise ValueError(f"{folder}: {len(missing)} videos needed for stage {stage} are not "
                         f"extracted (first {missing[:5]})")
    store = np.load(store_dir(folder) / FEATURES_FILE, mmap_mode="r")
    vd = variant_dir(folder, variant)
    if stage == 2:
        _keep_stage1(vd)
    vd.mkdir(parents=True, exist_ok=True)
    meta: dict = {"variant": variant, "dataset": dataset, "encoder": encoder, "filled": filled,
                  "stage": stage, "rule": RULES[variant], "videos": len(layout.names),
                  "split_sha256": split.sha256, "dense_rows": layout.rows,
                  "dense_names_sha256": layout.names_sha256(), "created_utc": utc_now(),
                  "commit": commit}
    indices, scale = None, None
    if variant in SELECT_VARIANTS:
        indices = snippet_indices(dataset, encoder, variant, layout.total_frames)
        np.save(vd / INDICES_FILE, indices)
        meta["indices_file"] = INDICES_FILE
        if variant in DRAW_OF and dataset == "ucf":
            r = DRAW_OF[variant]
            meta["draw"] = {"r": r, "seed": grid.DRAW_SEED + 3 * r + grid.ENCODER_INDEX[encoder]}
    else:
        train = sorted(split.train_indices)
        scale = fit_scale(store, layout, dataset, encoder, variant, train)
        meta["scale"] = {**scale.to_json(), "fit_videos": "training part (split train_indices)",
                         "fit_rows": len(train) * grid.WINDOWS, "ddof": 0}
    dtype = store.dtype if variant in SELECT_VARIANTS else np.dtype(np.float32)
    nan = torch.from_numpy(np.full((grid.WINDOWS, store.shape[1]), np.nan, dtype=dtype))
    wanted = set(rows_of)
    features = []
    for v in range(len(layout.names)):
        if v not in wanted:
            features.append(nan.clone())
            continue
        x = _unscaled(store, layout, variant, v, indices)
        features.append(torch.from_numpy(np.ascontiguousarray(
            scale.apply(x) if scale is not None else x)))
    _write(vd, features, layout.labels, seg_labels, layout.total_frames, meta)
    return read_json(vd / META_FILE)


def legacy_seg_labels(legacy_dir: str | os.PathLike[str], layout: Layout) -> list[torch.Tensor]:
    """seg_labels of the earlier cache of the same encoder (labels must match the manifest)."""
    cache = load_cache(legacy_dir, "train")
    labels = [int(y) for y in cache["labels"]]
    if labels != [int(y) for y in layout.labels]:
        raise ValueError(f"{legacy_dir}: earlier cache labels differ from the manifest")
    return [s.float().clone() for s in cache["seg_labels"]]


def derive_stage(folder: Path, legacy_dir: Path, layout: Layout, split: ValSplit, dataset: str,
                 encoder: str, stage: int, commit: Mapping | None = None) -> dict[str, dict]:
    """Every variant of ``stage`` for one encoder (stage 1 refused after stage 2)."""
    if stage == 1:
        _refuse_after_stage2(folder, dataset)
    seg = legacy_seg_labels(legacy_dir, layout)
    return {v: derive_variant(folder, layout, split, dataset, encoder, v, stage, seg, commit)
            for v in STAGE_VARIANTS[stage][dataset]}
