"""E020 (stage F1) scoring on the server (PLAN.md, order after the freeze). Stage-2 checkpoints
of the confirmation runs (2 datasets x 2 feature pairs x 10 confirmation seeds x 4 cells x 2
checkpoint rules) are scored by one or both paths of wvad.eval.scoring:

    (i)  primary: the server GPU, fp32, batch 32 (the training validation path)
    (ii) P000:    the same server's CPU, fp32, batch 256, under P000's thread set-up (its
         run_dcrc_eval.sh exported OMP_NUM_THREADS = MKL_NUM_THREADS = 6; eval_dcrc.py sets
         no thread count): both variables must be 6 when Python starts and
         torch.get_num_threads() 6, checked (never set) before scoring; torch's parallel
         info and MKL version are recorded in the meta

    --split val --p000_root <P000 outputs/> [--rescore <reason>]
        path (i) for the 160 runs x 2 rules (gate (s)) and path (ii) for the 40 base runs
        (pre-check against P000's val_video_auc); the stage-2 best and last weights of the
        base runs are compared with P000's checkpoints (gate (v0), val/meta.json). Written
        once; --rescore (after a rerun the plan allows, before test scoring began under this
        --out, i.e. before an access-log line names <out>/test; later, a new --out) moves all
        of val/ to val/superseded/<UTC time>/ and scores every run again
    --split test --cells base --gate_s <copy of results/e020_gate.json>
        the 40 base runs x 2 rules by both paths, refused unless that gate file shows (a)-(f),
        (t) and (s) passed (no unreadable run, clean tree) on this server's validation scores
        (sha256 of <out>/val/scores.sha256); gate (v) is checked locally next
    --split test --cells p000_base --gate_s <...> --p000_root <P000 outputs/>
        P000's own abl_base checkpoints by path (ii): gate (v) when (v0) failed for all 40
    --split test --cells l20 m10b m10 --gate_v <copy of results/e020_gate.json>
        path (i); refused unless that gate file has (v) passed on base scores made with this
        deployed commit and plan hash

One npz per run, rule and path, written outside the repository (under a temporary name, then
renamed, so an interrupted write leaves no truncated score file):

    <out>/<split>/<dataset>_<pair>_s<seed>/<cell>_<rule>_<path>.npz
        logits         (n, S) float32  segment logits (offset heads: z + c), primary readout
        probs          (n, S) float32  sigma(logits), the readout of the training records
        local_logits   (n, S) float32  offset heads only: offset-free logits z (_parts)
        offsets        (n,)   float32  offset heads only: c(v)
        parts_max_abs_diff             offset heads only: max |fp32(z + c) - logits|
        labels, indices (validation: training-cache indices; test: rows), epoch and
        val_video_auc of the checkpoint, ckpt_sha256, config (JSON), path, device,
        batch_size, num_threads
    <out>/<split>/scores.sha256    sha256 of every current npz (sha256sum format)
    <out>/val/meta.json, <out>/test/meta_<k>.json (k: the call's line in the access log;
        written with "complete": false right after that line, before any scoring, and again
        with "complete": true at the end, so a call cut short keeps its meta)

Test guards (PLAN.md): the sha256 of the deployed PLAN.md before its change log equals
PLAN.sha256; the commit is known (a deployed copy has no git: DEPLOYED_COMMIT; the local
analysis checks that it descends from the freeze commit) and is the commit of the validation
scoring in <out>/val, in the same environment; every checkpoint names its own run (config,
init and out_dir) and has the sha256 its validation file recorded (P000's: loadable by the
scoring); a cell already in the access log is scored again only with --rescore <reason>:
base (p000_base) in full at most once more and never after a later cell was scored, its
earlier files moved to <out>/test/superseded/<UTC time>/; with --missing_only only absent or
unreadable (not zip) files, the only re-read of the later cells, after an infrastructure
failure. Right before any test feature is deserialised, after the six test caches' sha256
(bytes only) matched the frozen plan's table, one JSON line (UTC time, DEPLOYED_COMMIT, plan
hash, cells, paths, runs, out dir, rescore reason, access-log realpath) is appended to
<runs>/test_access.log; refused if <out>/test already has a meta of that line or later, or an
earlier call used another access log. Scores and log stay outside the deploy tree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np
import torch

from wvad.data import DATASETS, FeatureSet, feature_dirs, load_feature_set
from wvad.data.features import cache_file
from wvad.diagnostics.common import code_version, gpu_environment, write_json
from wvad.diagnostics.e014 import GATE_ERRORS
from wvad.diagnostics.e018 import CONFIRMATION_SEEDS
from wvad.diagnostics.e020_seal import (
    E020_DIR,
    append_log,
    check_log_index,
    code_guard,
    full_reads,
    gate_s_unlock,
    gate_v_unlock,
    outside_repo,
    path_ii_threads,
    plan_frozen_part,
    plan_seal,
    read_log,
    same_scoring_env,
)
from wvad.diagnostics.runs import PAIRS, selection
from wvad.eval.scoring import BATCH_SIZE, load_head, score
from wvad.models.head import SegmentHead
from wvad.paths import REPO_ROOT, Paths, load_paths
from wvad.prep.manifests import sha256_file
from wvad.train.stages import INPUT_LAYER_NORM, _train_val

E020_RESULTS = E020_DIR / "results"
P000_RESULTS = REPO_ROOT / "experiments" / "P000_prior_validation_ablation" / "results"
ACCESS_LOG = "test_access.log"              # in the runs root, outside the deploy tree
SEEDS = CONFIRMATION_SEEDS
BEST, LAST = "ckpt_best_val.pt", "ckpt_last.pt"
RULES = {"best": BEST, "last": LAST}               # stage-2 checkpoint rules (primary: best)
# cell -> (stage-1 folder, stage-1 checkpoint stage 2 starts from, offset mode); factors A x B
CELLS = {"base": ("l20_stage1", BEST, "none"), "l20": ("l20_stage1", LAST, "none"),
         "m10b": ("m10_stage1", BEST, "mean"), "m10": ("m10_stage1", LAST, "mean")}
LATER = ("l20", "m10b", "m10")                     # scored only after gate (v) passed
P000_CELL = "p000_base"                            # P000's own abl_base checkpoints
BASE_CELLS = ("base", P000_CELL)                   # read in full at most twice, before LATER
TEST_CELLS = ("base", P000_CELL, *LATER)
PATHS = {"i": ("cuda", BATCH_SIZE), "ii": ("cpu", 256)}      # path -> (device, batch size)
P000_FEAT = {("ucf", "xcb_xcl"): "ucf_s0d1", ("ucf", "xcb_siglip"): "ucf_s0d1_sig",
             ("xdv", "xcb_xcl"): "xdv_xcb_xcl", ("xdv", "xcb_siglip"): "xdv_xcb_siglip"}
P000_ROLE = {"best": "best", "last": "final"}      # our rule -> P000 role (dcrc_*_runs.csv)
RETRY = "_retry1"            # P000 kept interrupted XD s31337 runs in <folder>_retry1
SPLITS = ("val", "test")
ENCODERS = ("xcb", "xcl", "siglip")
# sha256 of mil/test/data.pt per logical cache name (PLAN.md table, by cache folder)
CACHE_FOLDERS = {"ucf_xcb": "features_xclip_base", "ucf_xcl": "features_xclip_large",
                 "ucf_siglip": "features_siglip_so400m", "xdv_xcb": "features_xdviolence_xcb",
                 "xdv_xcl": "features_xdviolence_xcl",
                 "xdv_siglip": "features_xdviolence_siglip_so400m"}
TEST_CACHE_SHA256 = {
    "ucf_xcb": "289abd32cb44900cb03ffbc2a3d61362e71c45926521a0589aaf86da01a6817e",
    "ucf_xcl": "e3b9b5361f00e5649163a3a941f6a3ac2d9e064cc729506530d7c72827018df2",
    "ucf_siglip": "83835d24993017c443d659c71bfff9d94f78aa52e1b59d2ccf43bad56dffbbc0",
    "xdv_xcb": "748108b2d2ae4dc82bd062fc502b965e9be7fb4a44be7ddb09b67a935b5efdf9",
    "xdv_xcl": "c35e8d8ec30c99c8e285eb9507f3f68bf88f7d01c3fd04514fc4a2a898729589",
    "xdv_siglip": "908fda83e496745d30fd13eb0b36bd47f3cd68762cf6c0b877cc117baf3feaef",
}

Key = tuple[str, str, int, str]                    # (dataset, pair, seed, cell)
RUN_KEYS: tuple[Key, ...] = tuple((d, p, s, c) for d in DATASETS for p in PAIRS for s in SEEDS
                                  for c in CELLS)
P000_KEYS: tuple[Key, ...] = tuple((d, p, s, P000_CELL) for d in DATASETS for p in PAIRS
                                   for s in SEEDS)


class Job(NamedTuple):
    key: Key
    rule: str
    path: str
    ckpt: Path


# --- layout ---------------------------------------------------------------------------------

def chain_name(dataset: str, pair: str, seed: int) -> str:
    return f"{dataset}_{pair}_s{seed}"


def run_name(key: Key) -> str:
    """'<chain>/<cell>', the stage-2 run folder relative to the runs root."""
    d, p, s, c = key
    return f"{chain_name(d, p, s)}/{c}"


def score_rel(key: Key, rule: str, path: str = "i") -> str:
    return f"{run_name(key)}_{rule}_{path}.npz"


def score_path(root: Path, split: str, key: Key, rule: str, path: str = "i") -> Path:
    return Path(root) / split / score_rel(key, rule, path)


def access_log(runs: Path) -> Path:
    return Path(runs) / ACCESS_LOG


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def base_keys(excluded: Sequence[str] = ()) -> tuple[Key, ...]:
    return tuple(k for k in RUN_KEYS if k[3] == "base" and run_name(k) not in excluded)


def cell_paths(cell: str) -> tuple[str, ...]:
    """Scoring paths of a cell (both splits): base by both, P000's checkpoints by (ii)."""
    return {"base": ("i", "ii"), P000_CELL: ("ii",)}.get(cell, ("i",))


def scorable_rules(runs: Path, key: Key) -> tuple[str, ...]:
    """Rules with a checkpoint: a stage-2 run without any finite V-AUC saved no best checkpoint
    (selection.json best_epoch null), a numerical failure the analysis handles."""
    if key[3] == P000_CELL:
        return tuple(RULES)
    best = selection(Path(runs) / run_name(key)).get("best_epoch")
    return tuple(r for r in RULES if r != "best" or best is not None)


def keys_of(cells: Sequence[str], excluded: Sequence[str] = ()) -> list[Key]:
    keys = [k for k in RUN_KEYS if k[3] in cells and run_name(k) not in excluded]
    return keys + (list(P000_KEYS) if P000_CELL in cells else [])


def expected_files(runs: Path, cells: Sequence[str],
                   excluded: Sequence[str] = ()) -> dict[str, tuple[Key, str, str]]:
    """Relative score file -> (key, rule, path) for the cells (runs of --infra_failed left
    out)."""
    return {score_rel(k, r, p): (k, r, p) for k in keys_of(cells, excluded)
            for r in scorable_rules(runs, k) for p in cell_paths(k[3])}


def check_excluded(runs: Path, names: Sequence[str]) -> list[str]:
    """--infra_failed: stage-2 runs ('<chain>/<cell>') whose rerun also failed for infrastructure
    reasons (PLAN.md failure handling); each must lack selection.json, so no finished run can be
    dropped after the fact."""
    valid = {run_name(k) for k in RUN_KEYS}
    for name in names:
        if name not in valid or (Path(runs) / name / "selection.json").is_file():
            raise SystemExit(f"--infra_failed {name}: not an E020 stage-2 run, or it finished "
                             "(PLAN.md failure handling)")
    return sorted(set(names))


# --- test guards ----------------------------------------------------------------------------

def check_cells(split: str, cells: Sequence[str] | None, p000_root: Path | None) -> list[str]:
    """--split val scores every cell; --split test scores base (and/or p000_base) first, then
    the later cells, never both in one call (PLAN.md order)."""
    if split == "val":
        if cells or p000_root is None:
            raise SystemExit("--split val scores every cell (no --cells) and needs --p000_root "
                             "(gate (v0) weight comparison)")
        return list(CELLS)
    cells = list(dict.fromkeys(cells or ()))
    if not cells or (set(cells) & set(LATER) and set(cells) - set(LATER)):
        raise SystemExit("--split test needs --cells: base [p000_base] first, later l20 m10b m10 "
                         "in a separate call (PLAN.md order)")
    if P000_CELL in cells and p000_root is None:
        raise SystemExit("--cells p000_base needs --p000_root")
    return cells


def check_rescore(log: Sequence[Mapping], cells: Sequence[str], rescore: str | None,
                  missing_only: bool) -> list[str]:
    """Cells already in the access log are scored again only with --rescore <reason> (PLAN.md):
    base (p000_base) in full at most once more (the gate (v) rule) and never once a later cell
    was scored; the later cells only for the files an infrastructure failure left missing
    (--missing_only)."""
    logged = [set(e.get("cells", [])) for e in log]
    done = sorted(set().union(*logged) & set(cells))
    if (done or missing_only) and not rescore:
        raise SystemExit(f"cells {done or cells} need --rescore '<reason>' (PLAN.md: the test "
                         "set is scored once; again only under the gate (v) rule, or the missing "
                         "files after an infrastructure failure)")
    if set(done) & set(LATER) and not missing_only:
        raise SystemExit(f"{done} were scored: the later cells are read again only for files an "
                         "infrastructure failure left missing (--missing_only, PLAN.md)")
    base = sorted(set(cells) & set(BASE_CELLS))
    if base and any(c & set(LATER) for c in logged):
        raise SystemExit(f"{base}: the later cells were scored, base is not read again (PLAN.md)")
    twice = [c for c in base if not missing_only and full_reads(log, c) >= 2]
    if twice:
        raise SystemExit(f"{twice} were read in full twice: no further full read (PLAN.md: once "
                         "more after a gate (v) failure)")
    return done


def plan_cache_table(plan_dir: Path | None = None) -> dict[str, str | None]:
    """The test-cache sha256 table of the frozen part of PLAN.md, by logical cache name."""
    text = plan_frozen_part((Path(plan_dir or E020_DIR) / "PLAN.md").read_bytes()).decode(
        "utf-8")
    rows = dict(re.findall(r"^\s*\| (features_\w+) \| ([0-9a-f]{64}) \|\s*$", text, re.M))
    return {name: rows.get(folder) for name, folder in CACHE_FOLDERS.items()}


def cache_hashes(paths: Paths) -> dict:
    """sha256 of the six test caches against the plan's table (hashing reads no model score)."""
    out = {}
    for d in DATASETS:
        for e in ENCODERS:
            name = f"{d}_{e}"
            f = Path(cache_file(paths.feature_dir(name), "test"))
            got = sha256_file(f)
            out[name] = {"file": str(f), "sha256": got, "ok": got == TEST_CACHE_SHA256[name]}
    return out


def prepare_test(args: argparse.Namespace, env: Mapping, cells: Sequence[str]) -> dict:
    """Every refusal of the test scoring that reads no checkpoint and no test byte (the caches
    are hashed right before the access-log line)."""
    plan = plan_seal()
    code = code_guard()
    log = read_log(access_log(args.runs))
    done = check_rescore(log, cells, args.rescore, args.missing_only)
    later = bool(set(cells) & set(LATER))
    unlock = gate_v_unlock(args.gate_v, code["commit"], plan["sha256"]) if later else None
    gate_s = None if later else gate_s_unlock(args.gate_s, args.out)
    val_env = same_scoring_env(args.out, env, code["commit"])
    if plan_cache_table() != TEST_CACHE_SHA256:
        raise SystemExit("the code's test-cache sha256 differ from the frozen plan's table")
    return {"plan": plan, "code": code, "prior_log_entries": len(log), "rescored_cells": done,
            "rescore": args.rescore, "missing_only": args.missing_only, "gate_v": unlock,
            "gate_s": gate_s, "val_environment": val_env}


def val_binding(jobs: Sequence[Job], out: Path) -> list[str]:
    """Test jobs score the checkpoints gate (s) checked: each checkpoint's sha256 equals the one
    its validation file (<out>/val, path (i)) recorded (P000's checkpoints have none)."""
    problems = []
    for j in jobs:
        if j.key[3] == P000_CELL:
            continue
        f = score_path(out, "val", j.key, j.rule, "i")
        try:
            with np.load(f) as z:
                want = str(z["ckpt_sha256"])
            got = sha256_file(j.ckpt)
        except GATE_ERRORS as exc:
            problems.append(f"{f}: {exc!r}")
            continue
        if got != want:
            problems.append(f"{j.ckpt}: sha256 {got}, validated {want}")
    return problems


# --- P000 checkpoints -----------------------------------------------------------------------

def p000_record(key: Key) -> dict:
    """P000's per-run test record of the key's setting and seed (abl_base)."""
    d, p, s, _ = key
    feat = P000_FEAT[(d, p)]
    rec = json.loads((P000_RESULTS / "per_run" / d / f"abl_base_{feat}_s{s}.json").read_text(
        encoding="utf-8"))
    if (rec.get("dataset"), rec.get("config"), rec.get("feat"), rec.get("seed")) != (
            d, "abl_base", feat, s):
        raise ValueError(f"P000 record of {key} names another run")
    return rec


def p000_ckpt(root: Path | None, key: Key, rule: str) -> Path:
    """The checkpoint P000 evaluated (per-run record, 'outputs/...' of the previous project)
    under ``root`` = that outputs/ folder."""
    if root is None:
        raise ValueError("no --p000_root")
    rel = PurePosixPath(p000_record(key)["ckpts"][P000_ROLE[rule]])
    if rel.parts[0] != "outputs":
        raise ValueError(f"P000 checkpoint path {rel} is not under outputs/")
    return Path(root).joinpath(*rel.parts[1:])


def load_p000_head(ckpt: Path, dim: int, pair: str,
                   device: torch.device) -> tuple[SegmentHead, dict]:
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = SegmentHead(dim, use_input_ln=INPUT_LAYER_NORM[pair])
    model.load_legacy_state_dict(obj["model_state_dict"])
    return model.to(device).eval(), obj


def p000_problem(job: Job, obj: Mapping) -> str | None:
    """P000's checkpoint loads as the scoring loads it (legacy keys converted, strict, input
    dimension from the first context Linear) and has a numeric val_video_auc; raises on a
    conversion failure."""
    state = SegmentHead.convert_legacy_state_dict(obj["model_state_dict"])
    dim = next((v.shape[1] for k, v in state.items() if k.startswith("context.") and v.ndim == 2),
               None)
    if dim is None:
        return f"{job.ckpt}: no context Linear weight"
    SegmentHead(dim, use_input_ln=INPUT_LAYER_NORM[job.key[1]]).load_state_dict(state,
                                                                                 strict=True)
    float(obj.get("val_video_auc", math.nan))
    return None


def same_weights(ours: Path, theirs: Path) -> bool:
    """torch.equal on every tensor, P000's legacy key names converted."""
    a = torch.load(ours, map_location="cpu", weights_only=True)["model_state_dict"]
    b = SegmentHead.convert_legacy_state_dict(
        torch.load(theirs, map_location="cpu", weights_only=True)["model_state_dict"])
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def compare_folder(runs: Path, key: Key, files: Mapping[str, Path]) -> dict[str, bool | str]:
    out: dict[str, bool | str] = {}
    for rule, theirs in files.items():
        try:
            out[rule] = same_weights(Path(runs) / run_name(key) / RULES[rule], theirs)
        except GATE_ERRORS as exc:
            out[rule] = repr(exc)
    return out


def p000_weight_row(runs: Path, root: Path, key: Key) -> dict:
    """Gate (v0), weights: the base run's stage-2 best and last checkpoints against the folder
    P000 evaluated; a '_retry1' sibling (interrupted P000 runs) is compared and recorded too."""
    files = {r: p000_ckpt(root, key, r) for r in RULES}
    folders = {files["best"].parent.name: files}
    retry = files["best"].parent.with_name(files["best"].parent.name + RETRY)
    if retry.is_dir():
        folders[retry.name] = {r: retry / f.name for r, f in files.items()}
    found = {name: compare_folder(runs, key, fs) for name, fs in folders.items()}
    own = found[files["best"].parent.name]
    return {"run": run_name(key), "p000_dir": str(files["best"].parent),
            "best_equal": own["best"] is True, "last_equal": own["last"] is True,
            "folders": found,
            "matched": [n for n, v in found.items() if all(x is True for x in v.values())]}


def p000_weight_rows(runs: Path, root: Path, excluded: Sequence[str]) -> list[dict]:
    rows = []
    for key in base_keys(excluded):
        try:
            rows.append(p000_weight_row(runs, root, key))
        except GATE_ERRORS as exc:
            rows.append({"run": run_name(key), "error": repr(exc), "best_equal": False,
                         "last_equal": False})
    return rows


# --- scoring --------------------------------------------------------------------------------

def path_device(path: str) -> torch.device:
    """(i): the GPU (CPU where there is none: the environment checks refuse that); (ii): CPU."""
    kind = PATHS[path][0]
    return torch.device(kind if kind == "cpu" or torch.cuda.is_available() else "cpu")


def score_run(job: Job, features: FeatureSet, indices: np.ndarray,
              device: torch.device) -> dict:
    if job.key[3] == P000_CELL:
        model, obj = load_p000_head(job.ckpt, features.dim, job.key[1], device)
    else:
        model, obj = load_head(job.ckpt, features.dim, device)
    batch = PATHS[job.path][1]
    out = score(model, features, device, batch)
    if "offsets" in out:
        out["parts_max_abs_diff"] = np.float64(np.max(np.abs(
            out["local_logits"] + out["offsets"][:, None] - out["logits"])))
    cfg = obj.get("config", obj.get("args", {}))
    return out | {
        "labels": np.asarray(features.labels, dtype=np.int64), "indices": indices,
        "epoch": np.int64(obj.get("epoch", -1)),
        "val_video_auc": np.float64(obj.get("val_video_auc", math.nan)),
        "ckpt_sha256": np.str_(sha256_file(job.ckpt)),
        "config": np.str_(json.dumps(cfg, sort_keys=True, default=str)),
        "path": np.str_(job.path), "device": np.str_(str(device)),
        "batch_size": np.int64(batch), "num_threads": np.int64(torch.get_num_threads())}


def features_for(dataset: str, pair: str, split: str) -> tuple[FeatureSet, np.ndarray]:
    """Validation videos exactly as training builds them (wvad.train.stages._train_val, with the
    split-file label check), or the test cache in its own order."""
    if split == "val":
        _, val, val_split, _ = _train_val(dataset, pair)
        return val, np.asarray(val_split.val_indices)
    fs = load_feature_set(feature_dirs(load_paths(), dataset, pair), "test")
    return fs, np.arange(len(fs))


def make_jobs(runs: Path, cells: Sequence[str], excluded: Sequence[str],
              p000_root: Path | None) -> list[Job]:
    jobs = []
    for key, rule, path in expected_files(runs, cells, excluded).values():
        ck = (p000_ckpt(p000_root, key, rule) if key[3] == P000_CELL
              else Path(runs) / run_name(key) / RULES[rule])
        jobs.append(Job(key, rule, path, ck))
    return jobs


def checkpoint_problem(job: Job) -> str | None:
    """A checkpoint must be readable and (ours) name its own dataset, pair, seed and offset
    mode, start from its cell's stage-1 checkpoint (init: <stage-1 folder>/<checkpoint>) and
    write to its cell (out_dir): a misplaced checkpoint would be scored on the wrong features.
    P000's must load as the scoring loads it."""
    try:
        obj = torch.load(job.ckpt, map_location="cpu", weights_only=True)
        if job.key[3] == P000_CELL:
            return p000_problem(job, obj)
        cfg = obj["config"]
        init, out_dir = PurePosixPath(cfg["init"]), PurePosixPath(cfg["out_dir"])
        got = (cfg["dataset"], cfg["pair"], int(cfg["seed"]), cfg.get("global_mode", "none"),
               init.parent.name, init.name, out_dir.name)
    except GATE_ERRORS as exc:
        return f"{job.ckpt}: {exc!r}"
    want = (*job.key[:3], CELLS[job.key[3]][2], *CELLS[job.key[3]][:2], job.key[3])
    return None if got == want else f"{job.ckpt}: config {got}, expected {want}"


def save_npz(f: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write under a temporary name outside the *.npz pattern, then rename: an interrupted
    write leaves no truncated score file."""
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, f)


def score_jobs(jobs: Sequence[Job], out: Path, split: str) -> list[Path]:
    written = []
    for d in DATASETS:
        for p in PAIRS:
            mine = [j for j in jobs if j.key[:2] == (d, p)]
            if not mine:
                continue
            fs, idx = features_for(d, p, split)
            for j in mine:
                f = score_path(out, split, j.key, j.rule, j.path)
                save_npz(f, score_run(j, fs, idx, path_device(j.path)))
                written.append(f)
            print(f"[e020_score] {split} {d}/{p}: {len(mine)} files", flush=True)
    return written


def sha256_list(files: Sequence[Path], base: Path) -> str:
    rows = sorted((f.relative_to(base).as_posix(), sha256_file(f)) for f in files)
    return "".join(f"{sha}  {rel}\n" for rel, sha in rows)


def current_files(target: Path) -> list[Path]:
    """Score files of a split folder, superseded ones left out."""
    return sorted(f for f in Path(target).rglob("*.npz")
                  if "superseded" not in f.relative_to(target).parts)


def split_files(target: Path) -> list[Path]:
    """Every file of a split folder (scores, metas, sha256 list), superseded ones left out."""
    return sorted(f for f in Path(target).rglob("*") if f.is_file()
                  and "superseded" not in f.relative_to(target).parts)


def supersede(files: Sequence[Path], target: Path) -> str:
    """--rescore: move the earlier files of the rescored cells aside (kept for the report)."""
    dest = Path(target) / "superseded" / dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    for f in files:
        new = dest / f.relative_to(target)
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(new))
    return str(dest)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# --- main -----------------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E020 scoring of the confirmation runs (server)")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="score root outside the repository")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--cells", nargs="+", choices=TEST_CELLS,
                    help="test only: base [p000_base] first; l20 m10b m10 after gate (v)")
    ap.add_argument("--gate_s", type=Path, help="test, base / p000_base: copy of results/"
                                                "e020_gate.json with (a)-(f), (t), (s) passed")
    ap.add_argument("--gate_v", type=Path, help="test, later cells: copy of results/"
                                                "e020_gate.json with gate (v) passed")
    ap.add_argument("--p000_root", type=Path, help="P000 outputs/ (~/sear_phase1/outputs)")
    ap.add_argument("--infra_failed", nargs="*", default=[],
                    help="stage-2 runs '<chain>/<cell>' whose rerun also failed for "
                         "infrastructure reasons (PLAN.md failure handling)")
    ap.add_argument("--rescore", help="reason to score again: test, already scored cells; val, "
                                      "all of val/ (moved aside)")
    ap.add_argument("--missing_only", action="store_true",
                    help="test, with --rescore: score only absent or unreadable files "
                         "(infrastructure failure)")
    return ap.parse_args(argv)


def missing_jobs(jobs: Sequence[Job], args: argparse.Namespace) -> tuple[list, list]:
    """--missing_only: jobs whose file is absent or not a readable zip (its values unread),
    and those unreadable files (moved aside)."""
    todo, broken = [], []
    for j in jobs:
        f = score_path(args.out, args.split, j.key, j.rule, j.path)
        if f.exists() and zipfile.is_zipfile(f):
            continue
        todo.append(j)
        if f.exists():
            broken.append(f)
    return todo, broken


def select_jobs(jobs: list[Job], target: Path, args: argparse.Namespace) -> tuple[list, list]:
    """(jobs to score, existing files to move aside): existing files need --rescore; validation
    scores are written once, again only in full (all of val/ moved aside) and only before test
    scoring began under this --out: no access-log line names <out>/test (a call cut short
    counts; a later code change: a new --out root)."""
    if args.split == "val":
        old = split_files(target)
        if args.missing_only or (old and not args.rescore):
            raise SystemExit(f"{target}: validation scores are written once, again only in full "
                             "with --rescore '<reason>' (no --missing_only)")
        test = os.path.realpath(Path(args.out) / "test")
        if old and any(e.get("out_dir") and os.path.realpath(e["out_dir"]) == test
                       for e in read_log(access_log(args.runs))):
            raise SystemExit(f"test scoring began under {args.out}: score the validation videos "
                             "again under a new --out root (PLAN.md failure rule)")
        return jobs, old
    existing = [score_path(args.out, args.split, j.key, j.rule, j.path) for j in jobs]
    existing = [f for f in existing if f.exists()]
    if existing and not args.rescore:
        raise SystemExit(f"{len(existing)} score files exist (e.g. {existing[0]}): --rescore")
    if args.missing_only:
        return missing_jobs(jobs, args)
    return jobs, existing


def runnable_jobs(args: argparse.Namespace, cells: Sequence[str], excluded: Sequence[str],
                  target: Path) -> tuple[list, list]:
    """The jobs (every checkpoint readable and in place) and the existing files to move aside."""
    try:
        jobs = make_jobs(args.runs, cells, excluded, args.p000_root)
    except GATE_ERRORS as exc:
        raise SystemExit(f"runs or P000 records unreadable, nothing scored: {exc!r}") from exc
    problems = [x for x in map(checkpoint_problem, jobs) if x]
    if problems:
        raise SystemExit(f"{len(problems)} checkpoint problem(s), nothing scored: {problems}")
    return select_jobs(jobs, target, args)


def access_entry(meta: Mapping, cells: Sequence[str], jobs: Sequence[Job], target: Path) -> dict:
    names = sorted({run_name(j.key) for j in jobs})
    return {"time_utc": now_utc(), "deployed_commit": meta["code"]["commit"],
            "code_source": meta["code"]["source"], "dirty": meta["code"]["dirty"],
            "plan_sha256": meta["plan"]["sha256"], "cells": list(cells),
            "paths": {c: list(cell_paths(c)) for c in cells}, "n_runs": len(names),
            "n_files": len(jobs), "out_dir": str(target), "rescore": meta["rescore"],
            "missing_only": meta["missing_only"], "access_log": meta["access_log"],
            "runs": names}


def log_test_call(args: argparse.Namespace, meta: Mapping, cells: Sequence[str],
                  jobs: Sequence[Job], target: Path) -> dict:
    """Last test guards, then the access-log line right before any test feature is
    deserialised: checkpoints bound to their validation files, the meta index and the six
    test caches' sha256 (bytes only) against the plan's table."""
    unbound = val_binding(jobs, args.out)
    if unbound:
        raise SystemExit(f"{len(unbound)} checkpoint(s) differ from the validated ones, nothing "
                         f"scored: {unbound}")
    path = access_log(args.runs)
    head = {"access_log": check_log_index(target, path)}
    head["test_caches"] = cache_hashes(load_paths())
    if not all(v["ok"] for v in head["test_caches"].values()):
        raise SystemExit(f"test caches differ from the plan's sha256: {head['test_caches']}")
    entry = access_entry({**meta, **head}, cells, jobs, target)
    return head | {"access_log_line": append_log(path, entry)}


def meta_path(meta: Mapping, target: Path) -> Path:
    """<out>/val/meta.json, or <out>/test/meta_<k>.json (k: the call's access-log line)."""
    return target / ("meta.json" if meta["split"] == "val"
                     else f"meta_{meta['access_log_line']}.json")


def finish(meta: dict, files: Sequence[Path], target: Path) -> None:
    meta |= {"n_files_written": len(files), "time_utc_end": now_utc(), "complete": True}
    write_json(meta_path(meta, target), meta)
    write_text(target / "scores.sha256", sha256_list(current_files(target), target))
    print(f"[e020_score] {meta['split']} {meta['cells']}: {len(files)} files -> {target}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outside_repo(args.runs, args.out)
    excluded = check_excluded(args.runs, args.infra_failed)
    cells = check_cells(args.split, args.cells, args.p000_root)
    target = args.out / args.split
    env = gpu_environment()
    meta = {"split": args.split, "cells": cells, "time_utc": now_utc(), "code": code_version(),
            "environment": env, "runs": str(args.runs), "out": str(target),
            "infra_failed": excluded, "rescore": args.rescore, "missing_only": args.missing_only}
    if args.split == "test":
        meta |= prepare_test(args, env, cells)
    jobs, aside = runnable_jobs(args, cells, excluded, target)
    meta |= path_ii_threads(any(j.path == "ii" for j in jobs))
    if args.split == "val":
        meta["p000_weights"] = p000_weight_rows(args.runs, args.p000_root, excluded)
    else:
        meta |= log_test_call(args, meta, cells, jobs, target)
        # before any scoring: a call cut short (commit, plan, threads fixed) keeps its meta
        write_json(meta_path(meta, target), meta | {"complete": False})
    meta["superseded"] = supersede(aside, target) if aside else None
    finish(meta, score_jobs(jobs, args.out, args.split), target)


if __name__ == "__main__":
    main()
