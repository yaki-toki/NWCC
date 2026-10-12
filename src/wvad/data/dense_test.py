"""E022 test split of the dense representation (PLAN.md "시험 특징", "실행 순서와 시험셋 봉인",
"구현"): the restricted test-manifest reader, the test video paths, the test store layout, the
training-input hash record (review N3) and the test access device, shared by the test-split
extraction (``wvad.extract.testsplit``) and the E022 scoring. A new module (add-only rule): the
frozen E021 code is called unchanged and no attribute of a frozen module is ever assigned.
Module-level imports stay inside S_extract (the extraction entry point imports this module);
only ``dense_snippets`` (scoring) imports ``wvad.eval.dense``, inside the function. No public
name starts with "test" (pytest would collect it from a test module that imports it).

Restricted reader (accesses (0) and (A))
    read_manifest(dataset)      names, categories and total_frames of data/manifests/<ds>_test.json
                                and the video labels from the categories (UCF "Normal", XD "A":
                                0; else 1). Every other top-level field (intervals, fps, meta)
                                is skipped unread: its extent is scanned (strings, brackets)
                                without decoding it, so no interval is parsed, returned or
                                logged. The file bytes are read whole (sha256, git blob id).
    read_provenance(dataset)    only the meta's source / annotation sha256 (gate (t))
Test videos, relative to raw_video_root (the training rules of wvad.extract.dense take a
TrainManifest, so the test rule is new; UCF as the earlier test cache was built)
    UCF anomalous   Anomaly-Detection-Dataset/<UCF_PARTS[category]>/<category>/<name>.mp4
    UCF normal      Anomaly-Detection-Dataset/Testing_Normal_Videos_Anomaly/<name>.mp4
    XD              XD-Violence/test/<name>.mp4
    video_path_set(manifest)    the positive-check set (every path the manifest can produce)
    layout_of(manifest)         the frozen Layout of the test videos (built by its constructor)
Test store (review N2; the frozen code fixes the subfolder dense16/train and the phases)
    <feature_root>/<dataset>_<encoder>_d16_test/dense16/train/{features.npy, index.npz,
                                    meta.json, progress.jsonl}     test videos only (phase sample)
    <...>_d16_test/<variant>/mil/test/data.pt          derived 32-unit test caches (d16c, d16m),
                                    read by the frozen load_feature_set(dirs, "test")
    <...>_d16_test/<variant>/e022_test_derive.json     their record; no meta.json is written
                                    there, so scale_maps(test folder) fails and the d16m map
                                    comes only from the E021 folder
    dense_snippets(...)         DenseSet(test folders, scales=scale_maps(E021 folders, "d16m"))
Training inputs (PLAN.md "실행 순서" 3a; N3): 7 files in each of the six E021 folders
    <feature_root>/<dataset>_<encoder>_d16/ (42) and the sha256 of each folder's d16m scale map
    (mu_c, sigma_c, mu_v, sigma_v of ScaleMap.from_json as float64 little-endian bytes, in this
    order; 6), with the data_sha256 self-consistency of d16m and d16c; check_train_inputs
    compares the live files with the committed record (gates (f), (p), (y6), access device 8).
Access device (PLAN.md "시험 접근 장치", items 1-6 and 8; item 7 belongs to scoring 2)
    1 deployed_descent      the commit is the freeze commit or a descendant: git merge-base in
                            a checkout, else the deploy's lineage file (the raw commit objects
                            from the freeze commit to DEPLOYED_COMMIT, each re-hashed here)
    2 plan_seal             PLAN.md before "## 변경 기록" has the sha256 of PLAN.sha256
    3 d1_approval           that part records the D1 decision as "승인"
    4 manifest_hashes       the test manifests are byte-identical to E020's (git blob ids of the
                            E020 freeze tree), their recorded source / annotation sha256 are
                            E020's (gate (t) record) and the annotation files have that sha256
    5 old_cache_hashes      the six earlier test caches have the sha256 of the E020 table
    6 selection_binding     results/e022_selection.json is in the (committed / deployed) tree
                            with the given sha256
    8 require_train_inputs  the training inputs equal the committed e022_train_inputs.json
Test access log: one JSON line per event, appended (append_test_log), outside the repository
(outside_repo).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import warnings
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wvad.data.catalog import DATASETS, ENCODER_PAIRS, ENCODERS, dense_dir, dense_name
from wvad.data.dense import (
    FEATURES_FILE,
    INDEX_FILE,
    META_FILE,
    PROGRESS_FILE,
    DenseSet,
    ScaleMap,
    load_scale_map,
    read_json,
    store_dir,
    variant_dir,
)
from wvad.data.features import cache_file
from wvad.diagnostics.common import code_version
from wvad.extract import grid
from wvad.paths import REPO_ROOT, Paths
from wvad.prep.manifests import sha256_file

E022_DIR = REPO_ROOT / "experiments" / "E022_dense_confirmation"
RESULTS = E022_DIR / "results"
TRAIN_INPUTS_FILE = RESULTS / "e022_train_inputs.json"
SELECTION_FILE = RESULTS / "e022_selection.json"
LINEAGE_FILE = REPO_ROOT / "DEPLOYED_LINEAGE.json"     # beside DEPLOYED_COMMIT in a deployed copy
CHANGELOG = "## 변경 기록"                               # PLAN.sha256 covers PLAN.md before it
SEAL_FILE = "PLAN.sha256"

MANIFEST_FILES = {"ucf": "data/manifests/ucf_test.json", "xdv": "data/manifests/xdv_test.json"}
READ_FIELDS = ("names", "categories", "total_frames")  # the only fields the reader decodes
NORMAL_CATEGORY = {"ucf": "Normal", "xdv": "A"}        # label 0; every other category 1
UCF_ROOT = "Anomaly-Detection-Dataset"
UCF_TEST_NORMAL = "Testing_Normal_Videos_Anomaly"
UCF_PARTS = {"Abuse": "Anomaly-Videos-Part-1", "Arrest": "Anomaly-Videos-Part-1",
             "Arson": "Anomaly-Videos-Part-1", "Assault": "Anomaly-Videos-Part-1",
             "Burglary": "Anomaly-Videos-Part-2", "Explosion": "Anomaly-Videos-Part-2",
             "Fighting": "Anomaly-Videos-Part-2", "RoadAccidents": "Anomaly-Videos-Part-3",
             "Robbery": "Anomaly-Videos-Part-3", "Shooting": "Anomaly-Videos-Part-3",
             "Shoplifting": "Anomaly-Videos-Part-4", "Stealing": "Anomaly-Videos-Part-4",
             "Vandalism": "Anomaly-Videos-Part-4"}
XD_TEST = "XD-Violence/test"
LISTING_ROOTS = {"ucf": UCF_ROOT, "xdv": XD_TEST}      # file-listing scope of access (0)
ANNOTATION_FILES = {"ucf": f"{UCF_ROOT}/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
                    "xdv": "XD-Violence/annotations/test_annotation.txt"}

FOLDER_SUFFIX = "_test"                                # <dataset>_<encoder>_d16_test
VARIANTS = ("d16c", "d16m")                            # derived test caches
SCALED = "d16m"                                        # the HB head's map (E021 folder)
DERIVE_RECORD = "e022_test_derive.json"
PHASE = "sample"                                       # the fitting frozen phase (index code 3)
COMMAND = "e022-test-dense"                            # access-CSV command of the extraction
SPLIT = "test"
PLACEHOLDER = -1.0                                     # seg_labels of the derived test caches
STORE_FILES = (FEATURES_FILE, INDEX_FILE, META_FILE, PROGRESS_FILE)

# training inputs (PLAN.md 3a), per E021 folder <feature_root>/<dataset>_<encoder>_d16/
TRAIN_INPUT_FILES = ("dense16/train/features.npy", "dense16/train/index.npz",
                     "dense16/train/meta.json", "d16m/mil/train/data.pt", "d16m/meta.json",
                     "d16c/mil/train/data.pt", "d16c/meta.json")
SCALE_FIELDS = ("mu_c", "sigma_c", "mu_v", "sigma_v")
# gate (c) references: the E021 M10 development chains (seed 0) of HB and H1, stages 1 and 2
E021_CONFIG_RUNS = ("hb_m10_stage1", "hb_m10", "h1_m10_stage1", "h1_m10")
E021_CONFIG_FILE = "ckpt_last.pt"
PAIRS = ("xcb_xcl", "xcb_siglip")

# E020 records (access device 4 and 5; equal to the E020 code, plan and git tree: unit tests)
E020_FREEZE = "2f1f80d9591b759b51f56b016eeda00824e28b47"
E020_MANIFEST_BLOB = {"ucf": "fd9c7386aa107c7878c0337e39d6367d301b346e",      # git blob ids in
                      "xdv": "35008a55b55d63acbab7d6a87be2d20627b5335d"}      # the E020 freeze
E020_TEST = {
    "ucf": {"n": 290, "anomalous": 140, "frames": 1_111_808,
            "source": "44db7fa4a212f9f965f447cfa560457f5963c7ab51fb13335460b872700f1b9a",
            "annotation": "3b9542413f2ed9e94f73bf0488c151b3d7d595a8d2ad30f524e9243a2ae2a17c"},
    "xdv": {"n": 800, "anomalous": 500, "frames": 2_335_801,
            "source": "f51a61c68bf4f340c4c824e3129d1cd8180cb2bd2ceed0d1d40f0377b650c4ab",
            "annotation": "7e6b430c3c49749ccf66ff516de8930bdd5dd475abf5929b74938a399db63599"},
}
E020_TEST_CACHE_SHA256 = {
    "ucf_xcb": "289abd32cb44900cb03ffbc2a3d61362e71c45926521a0589aaf86da01a6817e",
    "ucf_xcl": "e3b9b5361f00e5649163a3a941f6a3ac2d9e064cc729506530d7c72827018df2",
    "ucf_siglip": "83835d24993017c443d659c71bfff9d94f78aa52e1b59d2ccf43bad56dffbbc0",
    "xdv_xcb": "748108b2d2ae4dc82bd062fc502b965e9be7fb4a44be7ddb09b67a935b5efdf9",
    "xdv_xcl": "c35e8d8ec30c99c8e285eb9507f3f68bf88f7d01c3fd04514fc4a2a898729589",
    "xdv_siglip": "908fda83e496745d30fd13eb0b36bd47f3cd68762cf6c0b877cc117baf3feaef",
}
APPROVAL = re.compile(r'^- \*\*D1\((?P<scope>[^)\n]*)\): "(?P<answer>[^"\n]*)"\.\*\*', re.M)
APPROVED = "승인"


class SplitDataError(ValueError):
    """A test manifest, path or store does not have the form E022 expects."""


# --- restricted reader ----------------------------------------------------------------------

_WS = " \t\n\r"


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in _WS:
        i += 1
    return i


def _skip_string(text: str, i: int) -> int:
    """Index after the JSON string starting at text[i] (a quote); escapes are stepped over,
    nothing is decoded."""
    j = i + 1
    while j < len(text):
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == '"':
            return j + 1
        j += 1
    raise SplitDataError("unterminated string")


def _skip_value(text: str, i: int) -> int:
    """Index after the JSON value starting at text[i], found without decoding it: strings are
    stepped over and brackets matched; a scalar runs to the next delimiter."""
    if i >= len(text):
        raise SplitDataError("value expected")
    if text[i] == '"':
        return _skip_string(text, i)
    if text[i] not in "[{":
        j = i
        while j < len(text) and text[j] not in ",]}" and text[j] not in _WS:
            j += 1
        if j == i:
            raise SplitDataError("value expected")
        return j
    closers: list[str] = []
    j = i
    while j < len(text):
        c = text[j]
        if c == '"':
            j = _skip_string(text, j)
            continue
        if c in "[{":
            closers.append("]" if c == "[" else "}")
        elif c in "]}":
            if not closers or closers.pop() != c:
                raise SplitDataError("unbalanced brackets")
            if not closers:
                return j + 1
        j += 1
    raise SplitDataError("unterminated array or object")


def _members(text: str, keep: Collection[str], out: dict, keys: list[str]) -> int:
    """The members of the top-level object (text[0] is past its '{'); returns the index after
    its '}'."""
    decoder = json.JSONDecoder()
    i = _skip_ws(text, 0)
    if text[i] == "}":
        return i + 1
    while True:
        if text[i] != '"':
            raise SplitDataError("key expected")
        key, i = json.decoder.scanstring(text, i + 1)
        if key in keys:
            raise SplitDataError(f"duplicate key {key!r}")
        keys.append(key)
        i = _skip_ws(text, i)
        if text[i] != ":":
            raise SplitDataError("':' expected")
        i = _skip_ws(text, i + 1)
        if key in keep:
            out[key], i = decoder.raw_decode(text, i)
        else:
            i = _skip_value(text, i)
        i = _skip_ws(text, i)
        if text[i] == "}":
            return i + 1
        if text[i] != ",":
            raise SplitDataError("',' or '}' expected")
        i = _skip_ws(text, i + 1)


def skim_object(text: str, keep: Collection[str]) -> tuple[dict, tuple[str, ...]]:
    """Decode the values of the top-level keys in ``keep`` of a JSON object; every other value
    is skipped unread (``_skip_value``). Returns the decoded fields and all top-level key names
    (file order). A duplicate key or anything after the object is an error."""
    out: dict = {}
    keys: list[str] = []
    start = _skip_ws(text, 0)
    if start >= len(text) or text[start] != "{":
        raise SplitDataError("top level must be an object")
    try:
        end = start + 1 + _members(text[start + 1:], keep, out, keys)
    except IndexError as exc:
        raise SplitDataError("truncated JSON") from exc
    except json.JSONDecodeError as exc:
        raise SplitDataError(f"malformed JSON ({exc})") from exc
    if _skip_ws(text, end) != len(text):
        raise SplitDataError("data after the top-level object")
    return out, tuple(keys)


def git_blob_id(data: bytes) -> str:
    """git's object id of a blob with these bytes (sha1 of 'blob <size>\\0' + bytes)."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def manifest_path(dataset: str) -> Path:
    if dataset not in MANIFEST_FILES:
        raise KeyError(f"unknown dataset '{dataset}' (known: {', '.join(MANIFEST_FILES)})")
    return REPO_ROOT / MANIFEST_FILES[dataset]


def label_of(dataset: str, category: str) -> int:
    """Video label from the category: UCF "Normal", XD "A" are normal (0), every other category
    anomalous (1); an anomalous UCF category must have a video folder (UCF_PARTS)."""
    if dataset not in NORMAL_CATEGORY:
        raise KeyError(f"unknown dataset '{dataset}'")
    if category == NORMAL_CATEGORY[dataset]:
        return 0
    if dataset == "ucf" and category not in UCF_PARTS:
        raise SplitDataError(f"unknown UCF category {category!r}")
    return 1


@dataclass(frozen=True)
class RestrictedManifest:
    """The restricted view of a test manifest: no interval, fps or meta value."""

    dataset: str
    names: tuple[str, ...]
    categories: tuple[str, ...]
    total_frames: tuple[int, ...]
    labels: tuple[int, ...]            # from the categories (label_of)
    sha256: str                        # of the file bytes
    blob: str                          # git blob id of the file bytes
    keys: tuple[str, ...]              # top-level key names of the file

    def __len__(self) -> int:
        return len(self.names)

    @property
    def anomalous(self) -> int:
        return int(sum(self.labels))

    @property
    def frames(self) -> int:
        return int(sum(self.total_frames))

    def names_sha256(self) -> str:
        """Identity of the video list and order (the dense store meta's names_sha256)."""
        return hashlib.sha256("\n".join(self.names).encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        return {"dataset": self.dataset, "n": len(self), "anomalous": self.anomalous,
                "frames": self.frames, "sha256": self.sha256, "blob": self.blob,
                "names_sha256": self.names_sha256(), "fields_read": list(READ_FIELDS)}


def _strings(value, field: str, source: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise SplitDataError(f"{source}: '{field}' must be a list of strings")
    return tuple(value)


def read_manifest(dataset: str,
                  path: str | os.PathLike[str] | None = None) -> RestrictedManifest:
    """The restricted read (module docstring): names, categories, total_frames, labels from the
    categories, and the file's sha256 and git blob id."""
    p = Path(path) if path is not None else manifest_path(dataset)
    data = p.read_bytes()
    fields, keys = skim_object(data.decode("utf-8"), READ_FIELDS)
    missing = [k for k in READ_FIELDS if k not in fields]
    if missing:
        raise SplitDataError(f"{p}: missing {missing}")
    names = _strings(fields["names"], "names", str(p))
    categories = _strings(fields["categories"], "categories", str(p))
    totals = fields["total_frames"]
    if not isinstance(totals, list) or not all(isinstance(t, int) and not isinstance(t, bool)
                                               and t >= 1 for t in totals):
        raise SplitDataError(f"{p}: 'total_frames' must be a list of positive integers")
    if not len(names) == len(categories) == len(totals):
        raise SplitDataError(f"{p}: {len(names)} names, {len(categories)} categories, "
                             f"{len(totals)} frame counts")
    if len(set(names)) != len(names):
        raise SplitDataError(f"{p}: duplicate video names")
    labels = tuple(label_of(dataset, c) for c in categories)
    return RestrictedManifest(dataset, names, categories, tuple(int(t) for t in totals), labels,
                              hashlib.sha256(data).hexdigest(), git_blob_id(data), keys)


def read_provenance(dataset: str, path: str | os.PathLike[str] | None = None) -> dict:
    """The meta's source and annotation sha256 only (no other field is decoded)."""
    p = Path(path) if path is not None else manifest_path(dataset)
    fields, _ = skim_object(p.read_bytes().decode("utf-8"), ("meta",))
    meta = fields.get("meta") or {}
    try:
        return {"source": str(meta["source"]["sha256"]),
                "annotation": str(meta["annotation"]["sha256"])}
    except (KeyError, TypeError) as exc:
        raise SplitDataError(f"{p}: meta has no source / annotation sha256") from exc


# --- test videos ----------------------------------------------------------------------------

def video_rel_path(dataset: str, name: str, category: str) -> str:
    """Path of a test video relative to raw_video_root (posix), from its name and category."""
    if dataset == "ucf":
        if category == NORMAL_CATEGORY["ucf"]:
            return f"{UCF_ROOT}/{UCF_TEST_NORMAL}/{name}.mp4"
        if category not in UCF_PARTS:
            raise SplitDataError(f"unknown UCF category {category!r}")
        return f"{UCF_ROOT}/{UCF_PARTS[category]}/{category}/{name}.mp4"
    if dataset == "xdv":
        return f"{XD_TEST}/{name}.mp4"
    raise KeyError(f"unknown dataset '{dataset}'")


def video_rel_paths(manifest: RestrictedManifest) -> list[str]:
    """Relative paths of the test videos, manifest order."""
    return [video_rel_path(manifest.dataset, n, c)
            for n, c in zip(manifest.names, manifest.categories, strict=True)]


def video_path_set(manifest: RestrictedManifest) -> set[str]:
    """Every relative path the test manifest can produce (the positive check)."""
    paths = video_rel_paths(manifest)
    if len(set(paths)) != len(paths):
        raise SplitDataError(f"{manifest.dataset}: two test videos map to one path")
    return set(paths)


def layout_of(manifest: RestrictedManifest):
    """The frozen ``wvad.extract.dense.Layout`` of the test videos, built by its constructor
    (``Layout.from_manifest`` takes a TrainManifest): manifest order, T = manifest
    total_frames, n_v = ceil(T / 16), consecutive row starts."""
    from wvad.extract.dense import Layout

    t = np.asarray(manifest.total_frames, dtype=np.int64)
    n = np.array([grid.n_snippets(x) for x in t], dtype=np.int64)
    return Layout(manifest.dataset, tuple(manifest.names),
                  np.asarray(manifest.labels, dtype=np.int64), t, n,
                  np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64))


# --- test store layout ----------------------------------------------------------------------

def store_folder(paths: Paths, dataset: str, encoder: str) -> Path:
    """The test-only parent folder <feature_root>/<dataset>_<encoder>_d16_test."""
    return paths.feature_root / f"{dense_name(dataset, encoder)}{FOLDER_SUFFIX}"


def e021_dir(paths: Paths, dataset: str, encoder: str) -> Path:
    """The E021 folder <feature_root>/<dataset>_<encoder>_d16 (only read by E022)."""
    return dense_dir(paths, dataset, encoder)


def derived_dir(folder: str | os.PathLike[str], variant: str) -> Path:
    if variant not in VARIANTS:
        raise KeyError(f"no derived test cache '{variant}' (known: {', '.join(VARIANTS)})")
    return variant_dir(folder, variant)


def derived_file(folder: str | os.PathLike[str], variant: str) -> Path:
    """<test folder>/<variant>/mil/test/data.pt (read by load_feature_set(..., "test"))."""
    return Path(cache_file(derived_dir(folder, variant), SPLIT))


def derive_record(folder: str | os.PathLike[str], variant: str) -> Path:
    return derived_dir(folder, variant) / DERIVE_RECORD


def _encoders(pair: str) -> tuple[str, ...]:
    if pair not in ENCODER_PAIRS:
        raise KeyError(f"unknown encoder pair '{pair}'")
    return ENCODER_PAIRS[pair]


def store_folders(paths: Paths, dataset: str, pair: str) -> list[Path]:
    return [store_folder(paths, dataset, e) for e in _encoders(pair)]


def e021_dirs(paths: Paths, dataset: str, pair: str) -> list[Path]:
    return [e021_dir(paths, dataset, e) for e in _encoders(pair)]


def derived_dirs(paths: Paths, dataset: str, pair: str, variant: str) -> tuple[Path, ...]:
    """Folders of a derived test cache of a pair, for load_feature_set(dirs, "test")."""
    return tuple(derived_dir(d, variant) for d in store_folders(paths, dataset, pair))


def dense_snippets(paths: Paths, dataset: str, pair: str, scaled: bool = True) -> DenseSet:
    """The dense test snippets of a pair: DenseSet(test folders, scales=scale_maps(E021 folders,
    "d16m")) for the HB head (C, C.G, C.last), without a map for the H1 head (A1.S)."""
    from wvad.eval.dense import scale_maps  # scoring only (outside S_extract)

    scales = scale_maps(e021_dirs(paths, dataset, pair), SCALED) if scaled else None
    return DenseSet(store_folders(paths, dataset, pair), scales)


def check_dense(dense: DenseSet, manifest: RestrictedManifest) -> None:
    """The dense test set holds the manifest's videos (count, T, labels), all extracted."""
    if len(dense) != len(manifest) or not (
            np.array_equal(dense.total_frames, np.asarray(manifest.total_frames))
            and np.array_equal(dense.labels, np.asarray(manifest.labels))):
        raise SplitDataError(f"{manifest.dataset}: the dense test set differs from the manifest")
    undone = [v for v in range(len(dense)) if not dense.done(v)]
    if undone:
        raise SplitDataError(f"{len(undone)} test videos are not extracted (first {undone[:5]})")


def store_hashes(paths: Paths, datasets: Sequence[str] = DATASETS,
                 encoders: Sequence[str] = ENCODERS) -> dict[str, str | None]:
    """sha256 of the dense test stores and of the derived test caches with their records, keyed
    '<dataset>_<encoder>_d16_test/<relative path>' (None when missing)."""
    out: dict[str, str | None] = {}
    for d in datasets:
        for e in encoders:
            folder = store_folder(paths, d, e)
            files = [store_dir(folder) / f for f in STORE_FILES]
            files += [f for v in VARIANTS for f in (derived_file(folder, v),
                                                    derive_record(folder, v))]
            for f in files:
                out[f"{folder.name}/{f.relative_to(folder).as_posix()}"] = (
                    sha256_file(f) if f.is_file() else None)
    return out


# --- training inputs (N3) -------------------------------------------------------------------

def _folders() -> list[tuple[str, str]]:
    return [(d, e) for d in DATASETS for e in ENCODERS]


def train_input_files(paths: Paths) -> dict[str, Path]:
    """The 42 E021 training-split files, keyed '<dataset>_<encoder>_d16/<relative path>'."""
    return {f"{dense_name(d, e)}/{rel}": e021_dir(paths, d, e) / rel
            for d, e in _folders() for rel in TRAIN_INPUT_FILES}


def scale_array_sha256(scale: ScaleMap) -> str:
    """sha256 of mu_c, sigma_c, mu_v, sigma_v (ScaleMap.from_json) as float64 little-endian
    bytes, concatenated in this order."""
    parts = [np.ascontiguousarray(getattr(scale, k), dtype="<f8").tobytes() for k in SCALE_FIELDS]
    return hashlib.sha256(b"".join(parts)).hexdigest()


def scale_map_sha256(paths: Paths, dataset: str, encoder: str) -> str:
    """The scale-map hash of one E021 folder (its d16m/meta.json)."""
    return scale_array_sha256(load_scale_map(variant_dir(e021_dir(paths, dataset, encoder),
                                                         SCALED)))


def self_consistency(paths: Paths, files: Mapping[str, str | None]) -> dict:
    """Each recorded d16m / d16c data.pt sha256 equals its meta.json's data_sha256."""
    out = {}
    for d, e in _folders():
        for v in ("d16m", "d16c"):
            key = f"{dense_name(d, e)}/{v}"
            meta = read_json(variant_dir(e021_dir(paths, d, e), v) / META_FILE)
            got = files.get(f"{key}/mil/train/data.pt")
            out[key] = {"meta_data_sha256": meta.get("data_sha256"), "data_sha256": got,
                        "ok": got is not None and meta.get("data_sha256") == got}
    return out


def e021_config_files(e021_runs: str | os.PathLike[str]) -> dict[str, Path]:
    """The checkpoints holding the gate (c) reference configs: HB and H1 M10 development chains
    (seed 0) of every dataset and pair, stages 1 and 2 (16 files)."""
    return {f"{d}_{p}_s0/{r}/{E021_CONFIG_FILE}": Path(e021_runs) / f"{d}_{p}_s0" / r /
            E021_CONFIG_FILE for d in DATASETS for p in PAIRS for r in E021_CONFIG_RUNS}


def train_input_record(paths: Paths, e021_runs: str | os.PathLike[str] | None = None) -> dict:
    """The 3a record (not a test access): sha256 and size of the 42 files, the six scale-map
    hashes, the data_sha256 self-consistency and, with ``e021_runs``, the gate (c) configs."""
    found = train_input_files(paths)
    missing = [k for k, p in found.items() if not p.is_file()]
    if missing:
        raise SplitDataError(f"training inputs missing: {missing}")
    files = {k: sha256_file(p) for k, p in found.items()}
    consistency = self_consistency(paths, files)
    rec = {"feature_root": str(paths.feature_root), "files": files,
           "sizes": {k: p.stat().st_size for k, p in found.items()},
           "scale_arrays": {dense_name(d, e): scale_map_sha256(paths, d, e)
                            for d, e in _folders()},
           "scale_array_bytes": "mu_c, sigma_c, mu_v, sigma_v (ScaleMap.from_json of "
                                "d16m/meta.json 'scale') as float64 little-endian, concatenated",
           "self_consistency": consistency,
           "ok": all(c["ok"] for c in consistency.values())}
    if e021_runs is not None:
        cfg = e021_config_files(e021_runs)
        absent = [k for k, p in cfg.items() if not p.is_file()]
        if absent:
            raise SplitDataError(f"E021 config checkpoints missing: {absent}")
        rec["e021_configs"] = {"runs": str(e021_runs),
                               "files": {k: sha256_file(p) for k, p in cfg.items()}}
    return rec


def check_train_inputs(record: Mapping, paths: Paths) -> dict:
    """The live training inputs against a recorded ``train_input_record``: its 42 file hashes
    and six scale-map hashes (the record must list exactly those)."""
    want_files = dict(record.get("files") or {})
    want_scales = dict(record.get("scale_arrays") or {})
    found = train_input_files(paths)
    problems = []
    if set(want_files) != set(found):
        problems.append("the record does not list the 42 training inputs")
    if set(want_scales) != {dense_name(d, e) for d, e in _folders()}:
        problems.append("the record does not list the six scale maps")
    changed = [k for k, p in found.items()
               if not p.is_file() or sha256_file(p) != want_files.get(k)]
    scale_changed = []
    for d, e in _folders():
        try:
            got = scale_map_sha256(paths, d, e)
        except (OSError, ValueError, KeyError) as exc:
            got = f"unreadable: {exc!r}"
        if got != want_scales.get(dense_name(d, e)):
            scale_changed.append(dense_name(d, e))
    return {"files": len(found), "scale_arrays": len(_folders()), "changed": changed,
            "scale_changed": scale_changed, "problems": problems,
            "ok": not (problems or changed or scale_changed)}


def load_train_inputs(path: str | os.PathLike[str] = TRAIN_INPUTS_FILE) -> dict:
    """The 'train_inputs' part of the committed results/e022_train_inputs.json."""
    try:
        return dict(json.loads(Path(path).read_text(encoding="utf-8"))["train_inputs"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"{path}: no training-input record ({exc!r})") from exc


def require_train_inputs(path: str | os.PathLike[str], paths: Paths) -> dict:
    """Access device 8: the training inputs equal the committed record (else SystemExit)."""
    res = check_train_inputs(load_train_inputs(path), paths)
    if not res["ok"]:
        raise SystemExit(f"E021 training inputs differ from {path}: {res}")
    return {"record": str(path), "record_sha256": sha256_file(path), **res}


# --- plan seal, approval, deployed commit ---------------------------------------------------

def plan_frozen_part(data: bytes) -> bytes:
    """PLAN.md up to its first line starting with '## 변경 기록' (excluded), as E020 / E021."""
    marker = CHANGELOG.encode("utf-8")
    if data.startswith(marker):
        return b""
    pos = data.find(b"\n" + marker)
    if pos < 0:
        raise ValueError(f"no line starting with {CHANGELOG!r}")
    return data[:pos + 1]


def plan_sha256(plan: str | os.PathLike[str]) -> str:
    return hashlib.sha256(plan_frozen_part(Path(plan).read_bytes())).hexdigest()


def plan_seal(plan_dir: str | os.PathLike[str] = E022_DIR) -> dict:
    """Access device 2: PLAN.md before its change log has the sha256 of PLAN.sha256."""
    d = Path(plan_dir)
    try:
        want = (d / SEAL_FILE).read_text(encoding="utf-8").split()[0]
        got = plan_sha256(d / "PLAN.md")
    except (OSError, ValueError, IndexError) as exc:
        raise SystemExit(f"plan seal unreadable in {d} ({exc!r}): the plan is not frozen") \
            from exc
    if got != want:
        raise SystemExit(f"PLAN.md before {CHANGELOG!r} has sha256 {got}, {SEAL_FILE} {want}: "
                         "the plan changed after the freeze")
    return {"file": "PLAN.md", "sha256": got}


def d1_approval(plan_dir: str | os.PathLike[str] = E022_DIR) -> dict:
    """Access device 3: the part of PLAN.md before its change log records the user's D1
    decision as "승인" (exactly one D1 decision line)."""
    try:
        text = plan_frozen_part((Path(plan_dir) / "PLAN.md").read_bytes()).decode("utf-8")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"PLAN.md unreadable in {plan_dir} ({exc!r})") from exc
    found = list(APPROVAL.finditer(text))
    if len(found) != 1 or found[0]["answer"] != APPROVED:
        raise SystemExit(f"PLAN.md records the D1 decision {[m['answer'] for m in found]}, not "
                         f"exactly one '{APPROVED}': no test access (PLAN.md D1)")
    return {"decision": "D1", "answer": APPROVED, "scope": found[0]["scope"],
            "line_sha256": hashlib.sha256(found[0].group(0).encode("utf-8")).hexdigest()}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={Path(root).as_posix()}", *args],
                          cwd=root, capture_output=True, check=False)


def _git_text(root: Path, *args: str) -> str:
    res = _git(root, *args)
    return res.stdout.decode("utf-8").strip() if res.returncode == 0 else ""


def freeze_commit_of(root: Path = REPO_ROOT, plan_dir: Path = E022_DIR) -> str | None:
    """The commit that added PLAN.sha256 (git; None unless exactly one did)."""
    rel = (Path(plan_dir) / SEAL_FILE).relative_to(root).as_posix()
    hashes = _git_text(root, "log", "--diff-filter=A", "--format=%H", "--", rel).split()
    return hashes[0] if len(hashes) == 1 else None


def write_lineage(freeze: str, out: str | os.PathLike[str] | None = None, head: str = "HEAD",
                  root: Path = REPO_ROOT) -> dict:
    """Local (git), at deployment: the raw commit objects on every ancestry path from the freeze
    commit to ``head`` (base64), so that a deployed copy without git can check its descent;
    written to ``out`` (DEPLOYED_LINEAGE.json beside DEPLOYED_COMMIT)."""
    head_sha, freeze_sha = _git_text(root, "rev-parse", head), _git_text(root, "rev-parse", freeze)
    if not head_sha or not freeze_sha:
        raise SystemExit(f"unknown commit {head!r} or {freeze!r}")
    if _git(root, "merge-base", "--is-ancestor", freeze_sha, head_sha).returncode != 0:
        raise SystemExit(f"{head_sha} does not descend from {freeze_sha}")
    shas = [freeze_sha] + _git_text(root, "rev-list", "--ancestry-path",
                                    f"{freeze_sha}..{head_sha}").split()
    commits = {}
    for s in shas:
        res = _git(root, "cat-file", "commit", s)
        if res.returncode != 0:
            raise SystemExit(f"commit object {s} unreadable")
        commits[s] = base64.b64encode(res.stdout).decode("ascii")
    rec = {"freeze_commit": freeze_sha, "head": head_sha, "commits": commits}
    if out is not None:
        Path(out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    return rec


def _parents(raw: bytes) -> list[str]:
    header = raw.split(b"\n\n", 1)[0]
    return [line[7:].decode("ascii") for line in header.split(b"\n")
            if line.startswith(b"parent ")]


def verify_lineage(record: Mapping, deployed: str, freeze: str) -> bool:
    """``deployed`` is ``freeze`` or descends from it through the record's commit objects, each
    of which must hash to its id (sha1 of 'commit <size>\\0' + object)."""
    if deployed == freeze:
        return True
    raw: dict[str, bytes] = {}
    for sha, b64 in dict(record.get("commits") or {}).items():
        data = base64.b64decode(b64)
        if hashlib.sha1(b"commit %d\x00" % len(data) + data).hexdigest() != sha:
            return False
        raw[sha] = data
    stack, seen = [deployed], set()
    while stack:
        c = stack.pop()
        if c == freeze:
            return True
        if c in seen or c not in raw:
            continue
        seen.add(c)
        stack.extend(_parents(raw[c]))
    return False


def deployed_descent(freeze: str | None = None,
                     lineage: str | os.PathLike[str] | None = LINEAGE_FILE,
                     root: Path = REPO_ROOT, plan_dir: Path = E022_DIR) -> dict:
    """Access device 1: committed code (known commit, no uncommitted tracked change) on the
    freeze commit or a descendant. In a git checkout the freeze commit is the one that added
    PLAN.sha256 (``freeze``, when given, must equal it) and git merge-base decides; in a
    deployed copy (DEPLOYED_COMMIT, no git) ``freeze`` is required and the lineage file must
    lead from DEPLOYED_COMMIT to it (``verify_lineage``)."""
    code = code_version()
    deployed = code.get("commit")
    if not deployed or code.get("dirty") is True:
        raise SystemExit(f"test access needs committed code (known commit, clean tracked tree); "
                         f"code version {code}")
    if code.get("source") == "git":
        found = freeze_commit_of(root, plan_dir)
        if found is None:
            raise SystemExit("no freeze commit: PLAN.sha256 was not added by exactly one commit")
        if freeze is not None and freeze != found:
            raise SystemExit(f"--freeze_commit {freeze} is not the freeze commit {found}")
        ok = _git(root, "merge-base", "--is-ancestor", found, deployed).returncode == 0
        method, freeze = "git merge-base --is-ancestor", found
    else:
        if not freeze:
            raise SystemExit("a deployed copy needs --freeze_commit (and its lineage file)")
        try:
            rec = json.loads(Path(lineage).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"deploy lineage {lineage} unreadable ({exc!r})") from exc
        if rec.get("freeze_commit") != freeze or rec.get("head") != deployed:
            raise SystemExit(f"lineage {lineage} is for {rec.get('head')} from "
                             f"{rec.get('freeze_commit')}, not {deployed} from {freeze}")
        ok = verify_lineage(rec, deployed, freeze)
        method = "lineage file (commit objects re-hashed)"
    if not ok:
        raise SystemExit(f"deployed commit {deployed} does not descend from the freeze commit "
                         f"{freeze}")
    return {"deployed_commit": deployed, "code_source": code.get("source"),
            "dirty": code.get("dirty"), "freeze_commit": freeze, "method": method}


# --- test data and earlier test caches ------------------------------------------------------

def manifest_hashes(paths: Paths, datasets: Sequence[str] = DATASETS) -> dict:
    """Access device 4: each test manifest's bytes have E020's git blob id (the E020 freeze
    tree) and its recorded source / annotation sha256 (meta only) equal E020's record; each
    annotation file (bytes hashed, never parsed) has E020's sha256."""
    out: dict = {}
    for d in datasets:
        data = manifest_path(d).read_bytes()
        blob = git_blob_id(data)
        prov = read_provenance(d)
        want = {"source": E020_TEST[d].get("source"), "annotation": E020_TEST[d]["annotation"]}
        ann = paths.video_root() / ANNOTATION_FILES[d]
        sha = sha256_file(ann) if ann.is_file() else None
        out[d] = {"manifest": {"file": MANIFEST_FILES[d],
                               "sha256": hashlib.sha256(data).hexdigest(), "blob": blob,
                               "e020_blob": E020_MANIFEST_BLOB[d],
                               "ok": blob == E020_MANIFEST_BLOB[d]},
                  "provenance": {**prov, "ok": prov == want},
                  "annotation": {"file": ANNOTATION_FILES[d], "sha256": sha,
                                 "e020_sha256": E020_TEST[d]["annotation"],
                                 "ok": sha == E020_TEST[d]["annotation"]}}
    out["ok"] = all(out[d][k]["ok"] for d in datasets
                    for k in ("manifest", "provenance", "annotation"))
    return out


def old_cache_hashes(paths: Paths) -> dict:
    """Access device 5 (hashes only): the six earlier test caches (mil/test/data.pt) against the
    E020 table; no cache is deserialised."""
    out: dict = {}
    for name, want in E020_TEST_CACHE_SHA256.items():
        try:
            f = Path(cache_file(paths.feature_dir(name), SPLIT))
        except ValueError as exc:                       # PathsConfigError: no such entry
            out[name] = {"file": None, "sha256": None, "ok": False, "error": str(exc)}
            continue
        got = sha256_file(f) if f.is_file() else None
        out[name] = {"file": str(f), "sha256": got, "ok": got == want}
    out["ok"] = all(out[n]["ok"] for n in E020_TEST_CACHE_SHA256)
    return out


def selection_binding(expected: str | None, file: Path = SELECTION_FILE,
                      root: Path = REPO_ROOT) -> dict:
    """Access device 6: results/e022_selection.json is in this tree with the given sha256; in a
    git checkout it must also be tracked and unchanged (a deployed copy holds committed files
    only, and its descent is item 1)."""
    if not Path(file).is_file():
        raise SystemExit(f"{file} missing: commit the checkpoint selection first (PLAN.md 7)")
    got = sha256_file(file)
    if not expected or got != expected:
        raise SystemExit(f"{file} has sha256 {got}, --selection_sha256 {expected}")
    rel = (Path(file).relative_to(root).as_posix() if Path(file).is_relative_to(root)
           else str(file))
    out = {"file": rel, "sha256": got}
    if code_version().get("source") == "git":
        if (_git(root, "ls-files", "--error-unmatch", "--", rel).returncode != 0
                or _git(root, "diff", "--quiet", "HEAD", "--", rel).returncode != 0):
            raise SystemExit(f"{rel} is not committed unchanged")
        out["last_commit"] = _git_text(root, "log", "-1", "--format=%H", "--", rel)
    return out


def access_guard(paths: Paths, *, freeze: str | None, lineage: str | os.PathLike[str] | None,
                 selection_sha256: str | None,
                 train_inputs: str | os.PathLike[str] | None = TRAIN_INPUTS_FILE,
                 check_inputs: bool = True, plan_dir: Path = E022_DIR,
                 root: Path = REPO_ROOT, selection: Path = SELECTION_FILE) -> dict:
    """Access device items 1-6 and, with ``check_inputs``, 8 (PLAN.md); SystemExit at the first
    failure. Items 4 and 5 hash test files (bytes only, nothing is decoded)."""
    out = {"descent": deployed_descent(freeze, lineage, root, plan_dir),
           "plan": plan_seal(plan_dir), "approval": d1_approval(plan_dir)}
    data = manifest_hashes(paths)
    if not data["ok"]:
        raise SystemExit(f"test manifests or annotations differ from E020's record: {data}")
    caches = old_cache_hashes(paths)
    if not caches["ok"]:
        raise SystemExit(f"earlier test caches differ from the E020 table: {caches}")
    out |= {"test_data": data, "old_caches": caches,
            "selection": selection_binding(selection_sha256, selection, root)}
    if check_inputs:
        if train_inputs is None:
            raise SystemExit("access device 8 needs the training-input record")
        out["train_inputs"] = require_train_inputs(train_inputs, paths)
    else:
        out["train_inputs"] = {"checked": False}
    return out


# --- test access log ------------------------------------------------------------------------

def outside_repo(*paths: str | os.PathLike[str] | None, root: Path = REPO_ROOT) -> None:
    """Test-access logs, scores and stores stay outside the (deployed) repository."""
    base = Path(root).resolve()
    for p in paths:
        if p is not None and Path(p).expanduser().resolve().is_relative_to(base):
            raise SystemExit(f"{p} is inside the repository {root}: use a folder outside it")


def append_test_log(path: str | os.PathLike[str], entry: Mapping) -> int:
    """Append one JSON line (sorted keys); returns the log's line count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(entry), sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return len(read_test_log(p))


def read_test_log(path: str | os.PathLike[str]) -> list[dict]:
    """The log's entries (a torn or unreadable line is left out with a warning)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    out = []
    for line in text.splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                warnings.warn(f"{path}: unreadable log line left out", stacklevel=2)
    return out
