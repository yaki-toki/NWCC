"""E023 derived training inputs at the comparison methods' official units (PLAN.md "같은 특징
규칙과 학습 단위", "검증·시험 입력", (3c), D1 scope (i)). New code (add-only rule); the frozen
E021 pure functions are called unchanged (``derive.window_means``, ``grid.window_members`` /
``grid.centre_snippets`` / ``grid.Members``, ``ScaleMap.fit`` / ``ScaleMap.apply``,
``read_index``). It reads only the E021 training-split snippet store of each encoder
(``<feature_root>/<dataset>_<encoder>_d16/dense16/train/{features.npy, index.npz}``), the
training manifest and the validation split file; no test file. It writes only into new E023
folders, never into an E021 folder::

    <feature_root>/<dataset>_<encoder>_d16_e023/<variant>/mil/train/data.pt
    <feature_root>/<dataset>_<encoder>_d16_e023/<variant>/meta.json

data.pt has the format of the E021 derived caches (read by the frozen ``load_feature_set(dirs,
"train")``): features (one (rows, D) float32 tensor per training-split video, manifest order;
training and validation part), labels, seg_labels (NaN, no temporal labels) and total_frames.

Variants ``d16<kind><N>[l2][r]``
    kind m      the official window mean over N windows (RTFM / UR-DMU process_feat, VadCLIP
                uniform_extract): boundaries r = np.linspace(0, n_v, N + 1, dtype=int64)
                (numpy's own function, floor), window i = mean of snippets r[i] .. r[i+1] - 1,
                an empty window (r[i] = r[i+1]) takes snippet r[i]; N rows per video
    kind p      VadCLIP process_feat (no random extract): n_v > N as kind m; n_v <= N the n_v raw
                snippets (unpadded here; the sampler pads to N and masks by length)
    kind g      the E021 rule (grid.window_members, centre assignment; N = 32 only): the
                pipeline's own reproduction of E021 d16m (unit tests), not a training input
    l2          every snippet l2-normalised per encoder first (float64, then float32; the
                Sultani-style clip normalisation, PLAN.md "입력 처리 표"), at training, validation
                and test alike (``l2_rows``)
    r           raw: no scale match (identity input layer); otherwise scale-matched
Means: rows float32 -> mean in float64 -> float32 (``derive.window_means``).

Scale match (E021 definition, refitted per variant on the training part): per encoder and
dimension, x' = mu_c + (sigma_c / sigma_V)(x - mu_V), ratio 1 where sigma_V = 0; mu_c, sigma_c of
the d16c reference rows (the snippet supporting each of the 32 evaluator-window centres,
grid.centre_snippets; after l2 for an l2 variant), mu_V, sigma_V of the variant's rows (every row
of kind p), both over the training-part videos of the split only, float64, ddof 0. Fitted by
``ScaleMap.fit`` on column blocks (the dimensions are independent; equal to one call on all
columns, unit-tested), applied to every row of the variant (stored scale-matched, as E021 d16m),
recorded in meta.json "scale" and used again as the fixed input layer of the head's validation
and test snippets (``load_variant_scales``).

    python -m wvad.extract.derive_n --dataset ucf --variants d16m200 d16p256 [--encoders ...]
                                    [--record results/e023_train_inputs.json] [--overwrite]
        every listed variant of every listed encoder; the record (key "derived_<dataset>") holds
        the sha256 of each data.pt, meta.json and scale-map array (dense_test.scale_array_sha256
        byte order) and of the source store files
"""

from __future__ import annotations

import argparse
import datetime as dt_time
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wvad.data.catalog import DATASETS, ENCODER_PAIRS, ENCODERS, dense_dir, dense_name
from wvad.data.dense import (
    FEATURES_FILE,
    INDEX_FILE,
    META_FILE,
    ScaleMap,
    read_index,
    read_json,
    store_dir,
)
from wvad.data.features import cache_file
from wvad.data.splits import ValSplit, check_labels
from wvad.extract import derive, grid

E023_SUFFIX = "_e023"                       # <dataset>_<encoder>_d16_e023
SPLIT = "train"
FIT_BLOCK = 128                             # columns per ScaleMap.fit call
SCALE_FIELDS = ("mu_c", "sigma_c", "mu_v", "sigma_v")
VARIANT_RE = re.compile(r"^d16(?P<kind>[mpg])(?P<unit>[1-9][0-9]*)(?P<l2>l2)?(?P<raw>r)?$")
RULES = {"m": "official window mean: r = np.linspace(0, n_v, N + 1, dtype=int64), window i = "
              "mean of snippets r[i]..r[i+1]-1 (float64 -> float32), empty window: snippet r[i]",
         "p": "VadCLIP process_feat: n_v > N the window mean of kind m, n_v <= N the raw "
              "snippets (unpadded)",
         "g": "E021 window_members (centre assignment, empty window: d16c snippet), N = 32"}


class VariantError(ValueError):
    """An unknown or malformed variant name."""


@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str          # "m" | "p" | "g"
    unit: int          # N
    l2: bool
    scaled: bool

    def to_json(self) -> dict:
        return {"name": self.name, "kind": self.kind, "unit": self.unit, "l2": self.l2,
                "scaled": self.scaled, "rule": RULES[self.kind]}


def variant_spec(name: str) -> VariantSpec:
    m = VARIANT_RE.match(str(name))
    if m is None:
        raise VariantError(f"variant {name!r} is not d16<m|p|g><N>[l2][r]")
    kind, unit = m["kind"], int(m["unit"])
    if kind == "g" and unit != grid.WINDOWS:
        raise VariantError(f"variant {name!r}: the E021 rule (g) has {grid.WINDOWS} windows")
    return VariantSpec(str(name), kind, unit, m["l2"] is not None, m["raw"] is None)


# --- pure functions -------------------------------------------------------------------------

def l2_rows(x: np.ndarray) -> np.ndarray:
    """Each row divided by its l2 norm in float64 (a zero row stays zero), as float32."""
    a = np.asarray(x, dtype=np.float64)
    norm = np.sqrt(np.square(a).sum(axis=-1, keepdims=True))
    return (a / np.where(norm == 0, 1.0, norm)).astype(np.float32)


def linspace_bounds(n_rows: int, unit: int) -> np.ndarray:
    """(N + 1,) window boundaries of the official rule (numpy's linspace, integer dtype)."""
    if n_rows < 1 or unit < 1:
        raise ValueError(f"need n_rows >= 1 and unit >= 1, got {n_rows}, {unit}")
    return np.linspace(0, int(n_rows), int(unit) + 1, dtype=np.int64)


def linspace_members(n_rows: int, unit: int) -> grid.Members:
    """The official windows as grid.Members (an empty window holds snippet r[i], flagged)."""
    r = linspace_bounds(n_rows, unit)
    sets, replaced = [], np.zeros(int(unit), dtype=bool)
    for i in range(int(unit)):
        if r[i] != r[i + 1]:
            sets.append(np.arange(r[i], r[i + 1], dtype=np.int64))
        else:
            sets.append(np.array([r[i]], dtype=np.int64))
            replaced[i] = True
    return grid.Members(tuple(sets), replaced)


def variant_rows(rows: np.ndarray, total_frames: int, spec: VariantSpec) -> np.ndarray:
    """The unscaled training rows of one video: (N, D) window means, or (n_v, D) raw snippets
    (kind p, n_v <= N); ``rows`` are the video's float32 snippets (after l2 for an l2 variant)."""
    x = np.asarray(rows, dtype=np.float32)
    n = len(x)
    if n != grid.n_snippets(total_frames):
        raise ValueError(f"{n} snippet rows for T = {total_frames}")
    if spec.kind == "g":
        return derive.window_means(x, grid.window_members(int(total_frames)))
    if spec.kind == "p" and n <= spec.unit:
        return x.copy()
    return derive.window_means(x, linspace_members(n, spec.unit))


def centre_rows(rows: np.ndarray, total_frames: int) -> np.ndarray:
    """(32, D) the d16c reference rows (snippet supporting each evaluator-window centre)."""
    return np.asarray(rows, dtype=np.float32)[grid.centre_snippets(int(total_frames))]


def column_blocks(dim: int, block: int = FIT_BLOCK) -> list[tuple[int, int]]:
    """[a, b) column blocks of ``block`` columns; never a block of one column when dim > 1 (a
    one-column array is reduced along its contiguous axis by numpy's pairwise sum, which can
    differ in the last bit from the column-wise sum of a wider array): a trailing one-column
    block joins the previous block."""
    if dim < 1 or block < 2:
        raise ValueError(f"need dim >= 1 and block >= 2, got {dim}, {block}")
    edges = list(range(0, dim, int(block))) + [dim]
    if len(edges) > 2 and edges[-1] - edges[-2] == 1:
        edges.pop(-2)
    return list(zip(edges[:-1], edges[1:], strict=True))


def fit_blocked(c_rows: Sequence[np.ndarray], v_rows: Sequence[np.ndarray],
                block: int = FIT_BLOCK) -> ScaleMap:
    """ScaleMap.fit over column blocks of the concatenated reference and variant rows (equal to
    one ScaleMap.fit on all columns, bit for bit: unit-tested)."""
    dim = int(np.asarray(c_rows[0]).shape[1])
    parts: dict[str, list[np.ndarray]] = {k: [] for k in SCALE_FIELDS}
    for a, b in column_blocks(dim, block):
        c = np.concatenate([np.asarray(r)[:, a:b] for r in c_rows])
        v = np.concatenate([np.asarray(r)[:, a:b] for r in v_rows])
        s = ScaleMap.fit(c, v)
        for k in SCALE_FIELDS:
            parts[k].append(getattr(s, k))
    return ScaleMap(*(np.concatenate(parts[k]) for k in SCALE_FIELDS))


def scale_array_sha256(scale: ScaleMap) -> str:
    """mu_c, sigma_c, mu_v, sigma_v as float64 little-endian bytes (dense_test byte order)."""
    data = b"".join(np.ascontiguousarray(getattr(scale, k), dtype="<f8").tobytes()
                    for k in SCALE_FIELDS)
    return hashlib.sha256(data).hexdigest()


# --- folders --------------------------------------------------------------------------------

def e023_dir(paths, dataset: str, encoder: str) -> Path:
    """<feature_root>/<dataset>_<encoder>_d16_e023 (new folder; never an E021 folder)."""
    return paths.feature_root / f"{dense_name(dataset, encoder)}{E023_SUFFIX}"


def variant_dir(paths, dataset: str, encoder: str, variant: str) -> Path:
    variant_spec(variant)
    return e023_dir(paths, dataset, encoder) / variant


def variant_dirs(paths, dataset: str, pair: str, variant: str) -> tuple[Path, ...]:
    """Folders of a derived variant of an encoder pair (fusion order), for load_feature_set."""
    if pair not in ENCODER_PAIRS:
        raise KeyError(f"unknown encoder pair '{pair}'")
    return tuple(variant_dir(paths, dataset, e, variant) for e in ENCODER_PAIRS[pair])


def load_variant_scales(dirs: Sequence[str | os.PathLike[str]]) -> list[ScaleMap] | None:
    """The input maps recorded in each folder's meta.json (None for a raw variant)."""
    metas = [read_json(Path(d) / META_FILE) for d in dirs]
    scaled = {bool(m.get("spec", {}).get("scaled")) for m in metas}
    if len(scaled) != 1:
        raise VariantError(f"folders {list(map(str, dirs))} mix scaled and raw variants")
    if not scaled.pop():
        return None
    return [ScaleMap.from_json(m["scale"]) for m in metas]


def sha256_file(path: str | os.PathLike[str]) -> str:
    from wvad.prep.manifests import sha256_file as _sha

    return _sha(path)


# --- derivation -----------------------------------------------------------------------------

def utc_now() -> str:
    return dt_time.datetime.now(dt_time.timezone.utc).isoformat(timespec="seconds")


def _check_store(index, split: ValSplit, manifest, source: str) -> None:
    check_labels(split, index.labels, source)
    undone = [v for v in range(len(index)) if not bool(index.done[v])]
    if undone:
        raise ValueError(f"{source}: {len(undone)} training videos are not extracted "
                         f"(first {undone[:5]})")
    if manifest is not None:
        if len(manifest) != len(index) or not (
                np.array_equal(np.asarray(manifest.total_frames), index.total_frames)
                and np.array_equal(np.asarray(manifest.labels), index.labels)):
            raise ValueError(f"{source}: the store differs from the training manifest")


def _write_atomic_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def derive_encoder(source: str | os.PathLike[str], target: str | os.PathLike[str],
                   variant: str, split: ValSplit, *, dataset: str, encoder: str,
                   manifest=None, commit: Mapping | None = None, overwrite: bool = False,
                   source_sha256: Mapping[str, str] | None = None) -> dict:
    """One variant of one encoder: ``source`` the E021 folder (its training-split store only),
    ``target`` the E023 variant folder. Returns the written meta (with data_sha256)."""
    spec = variant_spec(variant)
    src, out = Path(source), Path(target)
    if out.resolve() == src.resolve() or src.resolve() in out.resolve().parents:
        raise ValueError(f"{out}: derived inputs are never written into an E021 folder")
    data_path = Path(cache_file(out, SPLIT))
    if (data_path.exists() or (out / META_FILE).exists()) and not overwrite:
        raise FileExistsError(f"{out} exists (pass overwrite to replace it)")
    index = read_index(src)
    _check_store(index, split, manifest, str(src))
    store = np.load(store_dir(src) / FEATURES_FILE, mmap_mode="r")
    train = set(int(i) for i in split.train_indices)
    unscaled, c_fit, v_fit = [], [], []
    for v in range(len(index)):
        a, n, t = int(index.start[v]), int(index.n[v]), int(index.total_frames[v])
        rows = np.asarray(store[a:a + n]).astype(np.float32)
        if spec.l2:
            rows = l2_rows(rows)
        x = variant_rows(rows, t, spec)
        unscaled.append(x)
        if spec.scaled and v in train:
            c_fit.append(centre_rows(rows, t))
            v_fit.append(x)
    scale = fit_blocked(c_fit, v_fit) if spec.scaled else None
    rows = [int(len(x)) for x in unscaled]
    fit_rows = (int(sum(len(r) for r in c_fit)), int(sum(len(r) for r in v_fit)))
    del c_fit, v_fit
    features = []
    for i in range(len(unscaled)):              # each unscaled array freed once converted
        x, unscaled[i] = unscaled[i], None
        features.append(torch.from_numpy(np.ascontiguousarray(
            scale.apply(x) if scale is not None else x)))
    labels = [int(y) for y in index.labels]
    totals = [int(t) for t in index.total_frames]
    seg = [torch.full((n,), float("nan")) for n in rows]
    _write_atomic_torch(data_path, {"features": features, "labels": labels, "seg_labels": seg,
                                    "total_frames": totals})
    names = None if manifest is None else hashlib.sha256(
        "\n".join(manifest.names).encode("utf-8")).hexdigest()
    meta = {"variant": variant, "spec": spec.to_json(), "dataset": dataset, "encoder": encoder,
            "split": SPLIT, "filled": "all", "videos": len(index),
            "rows": rows, "split_sha256": split.sha256,
            "names_sha256": names,
            "source": {"folder": src.name, "files": dict(source_sha256 or {})},
            "scale": None if scale is None else {
                **scale.to_json(), "reference": "d16c: grid.centre_snippets (32 per video)",
                "fit_videos": "training part (split train_indices)",
                "fit_rows_c": fit_rows[0], "fit_rows_v": fit_rows[1], "ddof": 0,
                "block": FIT_BLOCK},
            "scale_array_sha256": None if scale is None else scale_array_sha256(scale),
            "created_utc": utc_now(), "commit": dict(commit) if commit else None,
            "data_sha256": sha256_file(data_path)}
    _write_atomic_json(out / META_FILE, meta)
    return meta


def source_hashes(source: str | os.PathLike[str]) -> dict[str, str]:
    """sha256 of the E021 store files derive_n reads (features.npy, index.npz)."""
    sd = store_dir(source)
    return {f"dense16/train/{f}": sha256_file(sd / f) for f in (FEATURES_FILE, INDEX_FILE)}


def derive_dataset(paths, dataset: str, variants: Sequence[str],
                   encoders: Sequence[str] = ENCODERS, *, split: ValSplit | None = None,
                   manifest=None, commit: Mapping | None = None,
                   overwrite: bool = False) -> dict:
    """Every listed variant of every listed encoder of one dataset; the record of its files."""
    from wvad.data import load_split, split_file

    for v in variants:
        variant_spec(v)
    split = split if split is not None else load_split(split_file(dataset))
    out: dict = {"dataset": dataset, "split_sha256": split.sha256, "variants": {}}
    for e in encoders:
        src = dense_dir(paths, dataset, e)
        hashes = source_hashes(src)
        out.setdefault("sources", {})[dense_name(dataset, e)] = hashes
        for v in variants:
            target = variant_dir(paths, dataset, e, v)
            meta = derive_encoder(src, target, v, split, dataset=dataset, encoder=e,
                                  manifest=manifest, commit=commit, overwrite=overwrite,
                                  source_sha256=hashes)
            key = f"{target.parent.name}/{v}"
            out["variants"][key] = {"data_sha256": meta["data_sha256"],
                                    "meta_sha256": sha256_file(target / META_FILE),
                                    "scale_array_sha256": meta["scale_array_sha256"],
                                    "videos": meta["videos"], "rows": int(sum(meta["rows"]))}
    return out


def check_derived(record: Mapping, paths) -> dict:
    """The live derived files against a ``derive_dataset`` record (data.pt and meta.json)."""
    changed = []
    for key, want in dict(record.get("variants") or {}).items():
        folder, variant = key.split("/")
        d = paths.feature_root / folder / variant
        got = {"data_sha256": sha256_file(cache_file(d, SPLIT)) if Path(
                   cache_file(d, SPLIT)).is_file() else None,
               "meta_sha256": sha256_file(d / META_FILE) if (d / META_FILE).is_file() else None}
        if got["data_sha256"] != want.get("data_sha256") or \
                got["meta_sha256"] != want.get("meta_sha256"):
            changed.append(key)
    return {"files": len(dict(record.get("variants") or {})), "changed": changed,
            "ok": not changed}


def merge_derived(old: Mapping | None, new: Mapping) -> dict:
    """One ``derived_<dataset>`` entry from an earlier record and a new ``derive_dataset``
    record: the variants of both (a re-derived key takes the new value), the sources of both
    (refused when a source's hashes changed), the same split (refused otherwise) and every
    call's time, code and variant keys under "calls"."""
    if not old:
        return {**dict(new), "calls": [{"utc": new.get("utc"), "code": new.get("code"),
                                        "variants": sorted(dict(new.get("variants") or {}))}]}
    if old.get("dataset") != new.get("dataset") or old.get("split_sha256") != new.get(
            "split_sha256"):
        raise VariantError(f"derived record of {old.get('dataset')} / split "
                           f"{old.get('split_sha256')} cannot take {new.get('dataset')} / "
                           f"{new.get('split_sha256')}")
    sources = dict(old.get("sources") or {})
    for k, v in dict(new.get("sources") or {}).items():
        if k in sources and sources[k] != v:
            raise VariantError(f"source {k} changed since the earlier derive_n call")
        sources[k] = v
    calls = list(old.get("calls") or [{"utc": old.get("utc"), "code": old.get("code"),
                                       "variants": sorted(dict(old.get("variants") or {}))}])
    calls.append({"utc": new.get("utc"), "code": new.get("code"),
                  "variants": sorted(dict(new.get("variants") or {}))})
    return {**dict(old), **dict(new), "sources": sources,
            "variants": {**dict(old.get("variants") or {}), **dict(new.get("variants") or {})},
            "calls": calls}


def _merge_record(path: str | os.PathLike[str], key: str, value) -> None:
    """``value`` into ``path`` under ``key``; a ``derived_<dataset>`` key is merged with the
    variants already recorded (``merge_derived``), never replaced."""
    p = Path(path)
    rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rec[key] = merge_derived(rec.get(key), value) if key.startswith("derived_") else value
    _write_atomic_json(p, rec)


def main(argv: Sequence[str] | None = None) -> None:
    from wvad.data.manifests import load_manifest, manifest_file
    from wvad.diagnostics.common import code_version
    from wvad.paths import load_paths

    p = argparse.ArgumentParser(prog="python -m wvad.extract.derive_n",
                                description="E023 derived training inputs (training split only)")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=ENCODERS)
    p.add_argument("--record", type=Path, help="results/e023_train_inputs.json (merged)")
    p.add_argument("--paths", help="paths file (default configs/paths.yaml)")
    p.add_argument("--overwrite", action="store_true", help="replace existing derived files")
    a = p.parse_args(argv)
    paths = load_paths(a.paths)
    manifest = load_manifest(manifest_file(a.dataset))
    rec = derive_dataset(paths, a.dataset, a.variants, a.encoders, manifest=manifest,
                         commit=code_version(), overwrite=a.overwrite)
    rec |= {"utc": utc_now(), "code": code_version()}
    if a.record is not None:
        _merge_record(a.record, f"derived_{a.dataset}", rec)
    print(f"[derive_n] {a.dataset}: {len(rec['variants'])} files"
          + (f" -> {a.record}" if a.record else ""))


if __name__ == "__main__":
    main()
