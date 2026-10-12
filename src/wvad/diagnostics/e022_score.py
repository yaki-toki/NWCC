"""E022 server code (PLAN.md "체크포인트 선택", "시험 특징", "실행 순서와 시험셋 봉인"
6-11, "구현"): the batch environment records, the validation dense scoring, the checkpoint
selection, the server part of the extraction gates and the two test scorings. Fixed numeric
code from the freeze commit on (PLAN.md "코드 규칙" 1): the forward path (``forward_*``: a head
on its inputs) is kept apart from the writing and indexing of the score files (``*_arrays``,
``write_scores``, ``arm_path``, ``rescore_*``), so exception 1 (gate (v) failed for a
forward-path reason) and exception 2 ((v3) / (s3t) failed for a writing or indexing reason)
touch separate functions.

Chains (README "Stage F — E022"): <runs>/<dataset>_<pair>_s<seed>/{hb_m10_stage1, hb_m10,
h1_m10_stage1, h1_m10}: HB trained on d16m, H1 on d16c, both M10 (--global_mode mean
--offset_lr_mult 10), the 10 confirmation seeds, --save_epoch_ckpts (ckpt_epoch_01..30.pt). R0 is
the E020 m10 chain <e020_runs>/<dataset>_<pair>_s<seed>/m10 (D2 (a)).

    env --out <runs>/env_e022_b<k>.json
        before each training batch (gate (f)): this machine's torch, CUDA, GPU and driver, the
        deployed commit and the sha256 of the 12 d16m / d16c training caches; the record is
        written, then the batch is refused (exit 1) unless the environment equals E021's (g1)
        record (a different driver only is recorded) and the caches equal the committed record
    score --split val --runs_root R --out S --e020_runs E [--rescore REASON] [--infra_failed ..]
        every stage-2 epoch checkpoint (1-30) of HB and H1 on its own 32-unit validation features,
        every validation dense snippet (HB through the E021 d16m map) and, for HB, the (s3) input:
        wvad.diagnostics.e021_score.score_run, the E021 path (eval mode, fp32, autocast off);
        the E021 training inputs are checked first (gate (p)); the sha256 of the E020 m10 chain
        files, their selection and last V-AUC are recorded (gate (r0)); no test data is read
    select --scores S
        per HB chain the epoch of the highest full-validation V-AUC of the W readout (C) and of
        the G readout (C.G), per H1 chain of W (A1.S) (wvad.eval.dense_readout; video score =
        max fp32 sigma; crossfit.rank_auc; earliest epoch on ties) and the sha256 of every
        checkpoint a test scoring reads -> results/e022_selection.json (commit it before the
        test access (A); a run with non-finite dense logits or gamma failed numerically)
    extract-gates --test_log LOG --record X --headers CSV --env ENV --selection_sha256 S ...
        test access (A), server: gates (y4), (y5), (y6) on the test stores and descriptive 9
        (per-video grab and replacement counts), merged into the extraction gate record
    score --split test --cells r0 --runs_root R --e020_runs E --out S --test_log LOG ...
        scoring 1, access (B): the E020 m10 checkpoints (best, last) on the earlier test caches,
        path (i) (GPU, fp32, batch 32 = BATCH_SIZE); access device items 1-6
    score --split test --cells c r1 b a1s ...
        scoring 2, access (C), one call: C, C.G, C.last (HB checkpoints on the scale-matched
        dense test snippets), B (HB ckpt_best_val.pt on the test d16m cache), R1 (H1
        ckpt_best_val.pt and ckpt_last.pt on the test d16c cache), A1.S (H1 W checkpoint on the
        raw dense test snippets); with C also its test d16m 32-unit logits (C.o32's c_32) and the
        (s3t) input (the scale-matched snippets averaged over J_k, the 32-unit path) and the
        frozen s3_check; access device items 1-8 (item 8 again at the end)

Test score files <out>/test/<chain>/<arm>.npz (arm R0, R0.last, C, C.G, C.last, B, R1, R1.last,
A1.S; written under a temporary name, then renamed), with names / total_frames / labels of the
restricted test-manifest read (manifest order) and the checkpoint's sha256, epoch and config:

    32-unit arms   logits (n, 32) z + c, probs sigma(z + c) (fp32, the head's), local_logits z,
                   offsets c(v) (n,), gamma
    snippet arms   z (rows,) offset-free snippet logits (videos in manifest order), counts (n,)
                   snippets per video, gamma, c (n,) = gamma x mean z (dense_readout.offset),
                   zc (rows,) z + c, sigma (rows,) sigmoid32(z + c)
    C also         o32_logits / o32_probs / o32_local_logits / o32_offsets (the test d16m cache),
                   s3t_z (n, 32) and s3t_check (the frozen s3_check, JSON)

The test calls append "refused", "start" and "end" (or "error") lines to the test access log
(outside the deploy tree; the "start" line before any test feature is deserialised) and write
<out>/test/meta_<k>.json (k: the start line's number; "complete" false, then true; the sha256
of each file it wrote, recorded right after the file is written: ``written_sha256``, which binds
every score file to the call that wrote it) and <out>/test/scores.sha256. Scoring 1 may read R0
in full twice at most (the second time only with --rescore, exception 1) and never after scoring
2; scoring 2 reads in full once, after which only --missing_only (infrastructure: absent,
unreadable or unrecorded files) or --runs CHAIN/ARM ... (exception 2: the affected runs, the same
checkpoints) re-score files, with --rescore.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt_time
import json
import math
import os
import shutil
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np
import torch

from wvad.data import FeatureSet, feature_dirs, load_feature_set, load_split, split_file
from wvad.data import dense_test as dt
from wvad.data.dense import DenseSet, read_index
from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics import e021_score as es
from wvad.diagnostics import e022_gates as g
from wvad.diagnostics.common import code_version, gpu_environment, write_json
from wvad.eval.crossfit import rank_auc, select_epoch
from wvad.eval.dense import averaged_units, head_gamma, s3_check, score_dense, standard_z
from wvad.eval.dense_readout import offset, readout, sigmoid32
from wvad.eval.scoring import BATCH_SIZE, load_head, score
from wvad.extract import grid
from wvad.paths import REPO_ROOT, Paths, load_paths
from wvad.prep.manifests import sha256_file

E022_DIR = dt.E022_DIR
RESULTS = dt.RESULTS
SELECTION_FILE = dt.SELECTION_FILE
GATE_FILE = RESULTS / "e022_gate.json"
EXTRACT_GATE_FILE = RESULTS / "e022_extract_gate.json"
DATASETS = ("ucf", "xdv")
PAIRS = ("xcb_xcl", "xcb_siglip")
SEEDS = g.SEEDS                                   # the 10 confirmation seeds
PIPELINE = "m10"
HEADS = ("hb", "h1")
HEAD_FEATURES = {"hb": "d16m", "h1": "d16c"}      # = e021_score.HEADS
EPOCHS = es.EPOCHS
BEST, LAST = "ckpt_best_val.pt", "ckpt_last.pt"
R0_CELL = "m10"                                   # the E020 cell of R0
R0_FILES = ("ckpt_best_val.pt", "ckpt_last.pt", "selection.json")
READOUTS = {"hb": ("W", "G"), "h1": ("W",)}
# arm -> (head, checkpoint rule, test input); rules: W / G selected epoch, best, last
ARMS = {"R0": ("h0", "best", "legacy"), "R0.last": ("h0", "last", "legacy"),
        "C": ("hb", "W", "dense_scaled"), "C.G": ("hb", "G", "dense_scaled"),
        "C.last": ("hb", "last", "dense_scaled"), "B": ("hb", "best", "d16m"),
        "R1": ("h1", "best", "d16c"), "R1.last": ("h1", "last", "d16c"),
        "A1.S": ("h1", "W", "dense_raw")}
CELLS = {"r0": ("R0", "R0.last"), "c": ("C", "C.G", "C.last"), "b": ("B",),
         "r1": ("R1", "R1.last"), "a1s": ("A1.S",)}
SCORING_2 = ("c", "r1", "b", "a1s")               # D3 (a), D6 (c): one call, every cell
SNIPPET_ARMS = ("C", "C.G", "C.last", "A1.S")
COMMAND = "score-test"
ACCESS = {"r0": "B", "scoring_2": "C"}
NONFINITE_KEYS = ("logits", "probs", "local_logits", "offsets", "gamma", "z", "c", "zc", "sigma",
                  "o32_logits", "o32_probs", "o32_local_logits", "o32_offsets", "s3t_z")
TRAIN_ENV_KEYS = g.TRAIN_ENV_KEYS


def now_utc() -> str:
    return dt_time.datetime.now(dt_time.timezone.utc).isoformat(timespec="seconds")


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def chain_name(dataset: str, pair: str, seed: int) -> str:
    return f"{dataset}_{pair}_s{seed}"


def split_chain(chain: str) -> tuple[str, str, int]:
    """'<dataset>_<pair>_s<seed>' -> (dataset, pair, seed)."""
    d, rest = chain.split("_", 1)
    pair, seed = rest.rsplit("_s", 1)
    return d, pair, int(seed)


def head_keys(datasets: Sequence[str] | None = None, pairs: Sequence[str] | None = None,
              seeds: Sequence[int] | None = None,
              heads: Sequence[str] | None = None) -> list[es.RunKey]:
    """The new chains' stage-2 runs (E021 RunKey: head hb / h1, pipeline m10); the module's
    datasets, pairs, seeds and heads unless given."""
    return [es.RunKey(d, p, s, h, PIPELINE) for d in (datasets or DATASETS)
            for p in (pairs or PAIRS) for s in (seeds or SEEDS) for h in (heads or HEADS)]


def run_label(key: es.RunKey) -> str:
    return f"{key.chain}/{key.run}"


def val_path(root: Path, key: es.RunKey) -> Path:
    return Path(root) / "val" / key.chain / f"{key.run}.npz"


def arm_path(root: Path, chain: str, arm: str) -> Path:
    return Path(root) / "test" / chain / f"{arm}.npz"


def write_scores(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write under a temporary name outside the *.npz pattern, then rename."""
    es.save_npz(Path(path), arrays)


def sha256_list(files: Sequence[Path], base: Path) -> str:
    rows = sorted((Path(f).relative_to(base).as_posix(), sha256_file(f)) for f in files)
    return "".join(f"{sha}  {rel}\n" for rel, sha in rows)


def current_files(target: Path) -> list[Path]:
    """Score files of a split folder, superseded ones left out."""
    return sorted(f for f in Path(target).rglob("*.npz")
                  if "superseded" not in f.relative_to(target).parts)


def write_list(target: Path) -> Path:
    path = Path(target) / "scores.sha256"
    path.write_text(sha256_list(current_files(target), target), encoding="utf-8", newline="\n")
    return path


def _outside(*paths) -> None:
    dt.outside_repo(*paths)


def _record(path: Path | str | None = None) -> dict:
    """The committed training-input record (results/e022_train_inputs.json)."""
    return dt.load_train_inputs(path or dt.TRAIN_INPUTS_FILE)


# --- gate (f): batch environment ----------------------------------------------------------------

def batch_env(paths: Paths, out: Path, record: Mapping, want: Mapping | None = None) -> dict:
    """Gate (f) record of a training batch (written first): environment, deployed commit and
    the sha256 of the d16m / d16c training caches; SystemExit (do not launch) unless the
    environment equals E021's training environment ((g1) record; a different driver alone is
    recorded and allowed, PLAN.md (f)) and the caches equal the committed record."""
    env = gpu_environment()
    code = code_version()
    current = g.train_input_hashes(paths.feature_root, g.INPUT_SUBSETS["f"])
    inputs = g.inputs_gate("f", record, current)
    want = dict(want if want is not None else g.e021_training_env())
    diff = [k for k in TRAIN_ENV_KEYS if env.get(k) != want.get(k)]
    rec = {**env, "commit": code.get("commit"), "code": code, "train_inputs": current,
           "inputs_check": inputs, "want": want, "env_differences": diff, "utc": now_utc()}
    write_json(Path(out), rec)
    if not inputs["ok"] or diff not in ([], ["driver"]) or not code.get("commit"):
        raise SystemExit(f"environment {diff} or training inputs ({inputs['different']}, "
                         f"{inputs['missing']}) differ, or no commit: do not launch the batch")
    return rec


# --- validation scoring -------------------------------------------------------------------------

def check_excluded(runs: Path, names: Sequence[str]) -> list[str]:
    """--infra_failed: '<chain>/<run>' of the new chains whose rerun also failed for
    infrastructure reasons (PLAN.md); each must lack selection.json (no finished run is
    dropped after the fact)."""
    valid = {run_label(k) for k in head_keys()}
    for name in names:
        if name not in valid or (Path(runs) / name / "selection.json").is_file():
            raise SystemExit(f"--infra_failed {name}: not an E022 run, or it finished")
    return sorted(set(names))


def final_ckpt_problem(path: Path, key: es.RunKey, rule: str) -> str | None:
    """ckpt_best_val.pt / ckpt_last.pt of a new run names its run (dataset, pair, seed, the
    head's features, the M10 options, out_dir, the chain's stage-1 ckpt_last.pt as init); the
    last checkpoint holds epoch 30."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        cfg = obj["config"]
        out_dir, init = PurePosixPath(cfg["out_dir"]), PurePosixPath(cfg["init"])
        got = (cfg["dataset"], cfg["pair"], int(cfg["seed"]), cfg.get("features", "legacy"),
               cfg.get("global_mode", "none"), float(cfg.get("offset_lr_mult", 1.0)),
               out_dir.name, init.parent.name, init.name)
        epoch = int(obj["epoch"])
    except es.GATE_ERRORS as exc:
        return f"{path}: {exc!r}"
    mode, mult = es.PIPELINES[key.pipeline]
    want = (key.dataset, key.pair, key.seed, HEAD_FEATURES[key.head], mode, mult, key.run,
            key.stage1, LAST)
    if got != want:
        return f"{path}: {got}, expected {want}"
    if rule == "last" and epoch != EPOCHS:
        return f"{path}: epoch {epoch}, expected {EPOCHS}"
    return None


def final_files(runs: Path, key: es.RunKey) -> dict[str, np.ndarray]:
    """sha256 of the run's ckpt_best_val.pt, ckpt_last.pt, selection.json and history.json and
    the selected epoch (-1 when selection.json has none)."""
    d = es.run_dir(runs, key)
    sel = _json(d / "selection.json")
    best = sel.get("best_epoch")
    out = {"best_epoch": np.int64(-1 if best is None else int(best)),
           "last_sha256": np.str_(sha256_file(d / LAST)),
           "selection_sha256": np.str_(sha256_file(d / "selection.json")),
           "history_sha256": np.str_(sha256_file(d / "history.json"))}
    out["best_sha256"] = np.str_(sha256_file(d / BEST) if (d / BEST).is_file() else "")
    return out


def check_val_runs(runs: Path, keys: Sequence[es.RunKey]) -> None:
    """Every epoch checkpoint and the final checkpoints exist and name their run."""
    es.check_runs(runs, keys)
    problems = []
    for k in keys:
        d = es.run_dir(runs, k)
        problems += [x for r, f in (("best", BEST), ("last", LAST)) if (d / f).is_file()
                     and (x := final_ckpt_problem(d / f, k, r))]
        if not (d / LAST).is_file() or not (d / "selection.json").is_file():
            problems.append(f"{run_label(k)}: no {LAST} or selection.json")
    if problems:
        raise SystemExit(f"{len(problems)} checkpoint problem(s), nothing scored: {problems[:5]}")


def e020_m10(e020_runs: Path, datasets: Sequence[str] | None = None) -> dict:
    """Gate (r0) inputs on this machine: sha256 of the E020 m10 chain files, their selection and
    last-epoch V-AUC (no test data)."""
    out = {}
    for d in datasets or DATASETS:
        for p in PAIRS:
            for s in SEEDS:
                c = chain_name(d, p, s)
                m = Path(e020_runs) / c / R0_CELL
                files = {f"{c}/{R0_CELL}/{f}": (sha256_file(m / f) if (m / f).is_file() else None)
                         for f in R0_FILES}
                try:
                    sel = _json(m / "selection.json")
                    hist = _json(m / "history.json")
                    last = float(hist[-1]["val_video_auc"]) if hist else None
                except (OSError, ValueError, KeyError, IndexError, TypeError):
                    sel, last = None, None
                out[c] = {"files": files, "selection": sel, "last_val_video_auc": last}
    return out


def score_val_group(runs: Path, out: Path, dataset: str, pair: str, keys: Sequence[es.RunKey],
                    device: torch.device) -> list[Path]:
    """The runs of one (dataset, pair): 32-unit validation features and dense validation
    snippets loaded once (e021_score.unit_features / dense_val / averaged_set)."""
    split = load_split(split_file(dataset))
    manifest = load_manifest(manifest_file(dataset))
    val = np.asarray(split.val_indices, dtype=np.int64)
    totals = [manifest.total_frames[int(i)] for i in val]
    labels = [split.labels[int(i)] for i in val]
    heads = sorted({k.head for k in keys})
    units = {h: es.unit_features(dataset, pair, HEAD_FEATURES[h]) for h in heads}
    for h, fs in units.items():
        if list(fs.labels) != labels or len(fs) != len(val):
            raise SystemExit(f"{dataset}/{pair}/{HEAD_FEATURES[h]}: validation videos differ "
                             "from the split")
    dense = {h: es.dense_val(dataset, pair, val, totals, labels,
                             "d16m" if h == "hb" else "raw", device) for h in heads}
    averaged = es.averaged_set(dense["hb"], "d16m") if "hb" in dense else None
    written = []
    for k in keys:
        arrays = es.score_run(runs, k, {HEAD_FEATURES[k.head]: units[k.head]}, dense[k.head],
                              averaged if k.head == "hb" else None, device)
        arrays |= {"val_indices": val, "total_frames": np.asarray(totals, dtype=np.int64)}
        arrays |= final_files(runs, k)
        f = val_path(out, k)
        write_scores(f, arrays)
        written.append(f)
        print(f"[e022_score] val {run_label(k)}", flush=True)
    return written


def score_val(args: argparse.Namespace) -> None:
    """--split val (no test data): gate (p)'s training inputs first, then every missing run
    file (an infrastructure rerun keeps finished files; --rescore moves val/ aside)."""
    _outside(args.runs_root, args.out, args.e020_runs)
    paths = load_paths(args.paths)
    excluded = check_excluded(args.runs_root, args.infra_failed)
    keys = [k for k in head_keys() if run_label(k) not in excluded]
    check_val_runs(args.runs_root, keys)
    current = g.train_input_hashes(paths.feature_root)
    p_check = g.inputs_gate("p", _record(args.train_inputs), current)
    target = Path(args.out) / "val"
    meta = _json(target / "meta.json") if (target / "meta.json").is_file() else {"calls": []}
    call = {"time_utc": now_utc(), "code": code_version(), "environment": gpu_environment(),
            "runs": str(args.runs_root), "out": str(target), "infra_failed": excluded,
            "train_inputs": current, "inputs_p": p_check, "rescore": args.rescore}
    if not p_check["ok"]:
        write_json(target / "meta.json", {**meta, "calls": [*meta["calls"],
                                                            call | {"refused": "gate (p)"}]})
        raise SystemExit(f"E021 training inputs differ from the record (gate (p)): {p_check}")
    if args.rescore:
        call["superseded"] = es.supersede(target, args.rescore)
    call["e020_m10"] = e020_m10(args.e020_runs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    call["device"] = str(device)
    todo = [k for k in keys if not val_path(args.out, k).is_file()]
    written = []
    for d in DATASETS:
        for p in PAIRS:
            mine = [k for k in todo if (k.dataset, k.pair) == (d, p)]
            if mine:
                written += score_val_group(args.runs_root, args.out, d, p, mine, device)
    call |= {"time_utc_end": now_utc(), "kept_existing": sorted(run_label(k) for k in keys
                                                                if k not in todo),
             "written": [f.relative_to(target).as_posix() for f in written]}
    write_json(target / "meta.json", {**meta, "calls": [*meta["calls"], call]})
    write_list(target)
    print(f"[e022_score] val: {len(written)} files -> {target}")


# --- checkpoint selection ------------------------------------------------------------------------

def _npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def dense_failure(sf: Mapping[str, np.ndarray]) -> str | None:
    """Non-finite validation dense logits or gamma: the run's dense rules failed numerically."""
    bad = [k for k in ("dense_z", "gamma") if k in sf and not np.isfinite(sf[k]).all()]
    return f"non-finite {' and '.join(bad)} (validation dense scoring)" if bad else None


def readout_scores(sf: Mapping[str, np.ndarray], kind: str) -> np.ndarray:
    """(epochs, n) video scores of the W or G readout (max fp32 sigma over windows / snippets)
    of every epoch and validation video (wvad.eval.dense_readout.readout)."""
    z, counts = sf["dense_z"], np.asarray(sf["dense_counts"], dtype=np.int64)
    totals = np.asarray(sf["total_frames"], dtype=np.int64)
    gamma = sf.get("gamma")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    out = np.empty((z.shape[0], len(counts)), dtype=np.float64)
    for e in range(z.shape[0]):
        ge = None if gamma is None else float(gamma[e])
        for i, (a, n, t) in enumerate(zip(starts, counts, totals, strict=True)):
            out[e, i] = readout(z[e, a:a + n], int(t), ge, kind).video_score
    return out


def select_readout(sf: Mapping[str, np.ndarray], kind: str) -> dict:
    """Full-validation V-AUC per epoch (rank formula, fractions) and the selected epoch (1-based,
    the first maximum: strict >, earliest on ties)."""
    labels = np.asarray(sf["labels"])
    vs = readout_scores(sf, kind)
    aucs = rank_auc(vs, labels)
    best, tied = select_epoch(vs, labels, np.arange(len(labels)))
    return {"epoch": best + 1, "v_auc": [float(x) for x in aucs],
            "selected_v_auc": float(aucs[best]), "tied": bool(tied)}


def _ckpt_entry(run: str, file: str, sha: str, epoch: int | None) -> dict:
    return {"file": f"{run}/{file}", "sha256": sha, "epoch": epoch}


def select_chain(sf: Mapping[str, np.ndarray], key: es.RunKey) -> dict:
    """One run's selection record: readouts and the checkpoints of its arms."""
    run = key.run
    failed = dense_failure(sf)
    out: dict = {"head": key.head, "failed": failed, "readouts": {}, "checkpoints": {}}
    shas = [str(x) for x in sf["ckpt_sha256"]]
    if failed is None:
        for kind in READOUTS[key.head]:
            out["readouts"][kind] = select_readout(sf, kind)
    names = {"hb": {"W": "C", "G": "C.G"}, "h1": {"W": "A1.S"}}[key.head]
    for kind, arm in names.items():
        r = out["readouts"].get(kind)
        if r is not None:
            e = r["epoch"]
            out["checkpoints"][arm] = _ckpt_entry(run, f"ckpt_epoch_{e:02d}.pt", shas[e - 1], e)
    best = int(sf["best_epoch"])
    last_arm, best_arm = ("C.last", "B") if key.head == "hb" else ("R1.last", "R1")
    out["checkpoints"][last_arm] = _ckpt_entry(run, LAST, str(sf["last_sha256"]), EPOCHS)
    if best >= 1 and str(sf["best_sha256"]):
        out["checkpoints"][best_arm] = _ckpt_entry(run, BEST, str(sf["best_sha256"]), best)
    return out


def r0_entries(e020: Mapping[str, Mapping]) -> dict:
    """R0 / R0.last checkpoints of the E020 m10 chains (hashes of the validation call)."""
    out = {}
    for chain, rec in sorted(e020.items()):
        files = rec.get("files") or {}
        best = (rec.get("selection") or {}).get("best_epoch")
        out[chain] = {"R0": _ckpt_entry(R0_CELL, BEST, files.get(f"{chain}/{R0_CELL}/{BEST}"),
                                        best),
                      "R0.last": _ckpt_entry(R0_CELL, LAST,
                                             files.get(f"{chain}/{R0_CELL}/{LAST}"), EPOCHS)}
    return out


def selection_record(scores: Path, excluded: Sequence[str] = ()) -> dict:
    """The selection of every run with a validation score file (excluded runs: infrastructure
    failures) and the R0 checkpoints recorded by the last validation call."""
    target = Path(scores) / "val"
    meta = _json(target / "meta.json")
    calls = [c for c in meta.get("calls", []) if "e020_m10" in c]
    if not calls:
        raise SystemExit(f"{target}/meta.json has no completed validation call")
    chains = {}
    for k in head_keys():
        if run_label(k) in excluded:
            chains[run_label(k)] = {"head": k.head, "failed": "infrastructure (PLAN.md)",
                                    "readouts": {}, "checkpoints": {}}
            continue
        chains[run_label(k)] = select_chain(_npz(val_path(scores, k)), k)
    return {"chains": chains, "r0": r0_entries(calls[-1]["e020_m10"]),
            "infra_failed": list(excluded),
            "val_list_sha256": sha256_file(target / "scores.sha256"),
            "val_calls": [{"time_utc": c.get("time_utc"), "commit": (c.get("code") or {}).get(
                "commit")} for c in meta.get("calls", [])]}


def select_command(args: argparse.Namespace) -> None:
    rec = selection_record(args.scores, args.infra_failed)
    out = {"check": "E022 checkpoint selection (PLAN.md 체크포인트 선택; validation only)",
           "code": code_version(), "time_utc": now_utc(),
           "rule": "full-validation V-AUC of the readout's video scores (max fp32 sigma; "
                   "dense_readout W / G), first maximum (strict >, earliest epoch on ties)",
           **rec}
    write_json(Path(args.out), out)
    failed = [k for k, v in rec["chains"].items() if v["failed"]]
    print(f"[e022_score] selection -> {args.out} ({len(rec['chains'])} runs, failed {failed})")


# --- extraction gates (y4)-(y6), server, access (A) -------------------------------------------

def header_passes(path: Path, dataset: str) -> dict[int, bool]:
    """Per test video of the header CSV (testsplit header): |header T - manifest T| <= 16."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["dataset"] == dataset]
    return {int(r["index"]): abs(int(r["header_frames"]) - int(r["manifest_frames"]))
            <= g.Y3_DELTA for r in rows}


def index_columns(test_dir: Path) -> dict:
    """Descriptive 9 of one test store: per video grab attempts / successes, used, replaced and
    tail-replaced frames and the header frame count (no feature value)."""
    ix = read_index(test_dir)
    return {c: [int(x) for x in getattr(ix, c)] for c in (
        "header_frames", "grab_attempts", "grab_successes", "used_frames", "replaced",
        "replaced_tail")}


def guard_commit(record: Mapping, command: str = "dense") -> str | None:
    return ((record.get(f"guard_{command}") or {}).get("descent") or {}).get("deployed_commit")


def extraction_gates(paths: Paths, record: Mapping, headers: Path, env: Mapping,
                     train_record: Mapping, datasets: Sequence[str] | None = None,
                     encoders: Sequence[str] = dt.ENCODERS) -> dict:
    """(y4), (y5), (y6) of every (dataset, encoder) test store and descriptive 9. (y5)'s E021
    check: the training inputs now equal the record, and the extraction and derive calls found
    them unchanged at their start and end (their guard and after-call records)."""
    current = g.train_input_hashes(paths.feature_root)
    calls = {c: {"start": bool(((record.get(f"guard_{c}") or {}).get("train_inputs") or {})
                               .get("ok")),
                 "end": bool((record.get(f"train_inputs_after_{c}") or {}).get("ok"))}
             for c in ("dense", "derive")}
    commit = guard_commit(record)
    out = {}
    for d in datasets or DATASETS:
        m = dt.read_manifest(d)
        y3 = header_passes(headers, d)
        rows = {"y4": {}, "y5": {}, "y6": {}, "desc9": {}}
        for e in encoders:
            test_dir, e021 = dt.store_folder(paths, d, e), dt.e021_dir(paths, d, e)
            rows["y4"][e] = g.gate_y4(read_index(test_dir), m.names, m.total_frames, y3)
            y5 = g.gate_y5(test_dir, dataset=d, encoder=e, names=m.names, totals=m.total_frames,
                           labels=m.labels, commit=commit, environment=env, e021_before=current,
                           e021_after=current, recorded=train_record)
            y5["e021_calls"] = calls
            y5["ok"] = bool(y5["ok"] and all(v["start"] and v["end"] for v in calls.values()))
            rows["y5"][e] = y5
            rows["y6"][e] = g.gate_y6(test_dir, e021, dataset=d, names=m.names,
                                      categories=m.categories, totals=m.total_frames,
                                      recorded=train_record)
            rows["desc9"][e] = index_columns(test_dir)
        for k in ("y4", "y5", "y6"):
            rows[k]["ok"] = all(v["ok"] for v in rows[k].values())
        out[d] = rows
    return out


def extract_gates_command(args: argparse.Namespace) -> None:
    paths = load_paths(args.paths)
    _outside(args.test_log, args.record)
    line = {"access": "A", "command": "extract-gates", "datasets": list(args.datasets)}
    try:
        guard = dt.access_guard(paths, freeze=args.freeze_commit, lineage=args.lineage,
                                selection_sha256=args.selection_sha256,
                                train_inputs=args.train_inputs)
    except SystemExit as exc:
        dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "refused",
                                                  "reason": str(exc), "code": code_version()})
        raise
    record = _json(args.record)
    dt.append_test_log(args.test_log, line | {
        "time_utc": now_utc(), "event": "start", "guard": guard_summary(guard),
        "files": {"record": str(args.record), "headers": str(args.headers),
                  "env": str(args.env)}})
    try:
        res = extraction_gates(paths, record, Path(args.headers), _json(args.env),
                               dt.load_train_inputs(args.train_inputs), args.datasets)
    except BaseException as exc:
        dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "error",
                                                  "error": repr(exc)})
        raise
    after = dt.check_train_inputs(dt.load_train_inputs(args.train_inputs), paths)
    record = _json(args.record)
    for d, rows in res.items():
        for k, v in rows.items():
            record[f"{k}_{d}"] = v
    record["guard_extract_gates"] = guard | {"time_utc": now_utc()}
    record["train_inputs_after_extract_gates"] = after
    write_json(Path(args.record), record)
    status = {f"{k}_{d}": rows[k]["ok"] for d, rows in res.items() for k in ("y4", "y5", "y6")}
    dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "end",
                                              "result": status, "train_inputs_after": after})
    print(json.dumps(status))
    if not all(status.values()) or not after["ok"]:
        raise SystemExit(f"extraction gates failed: {status} (PLAN.md: stop and report)")


# --- test scoring: jobs, binding, order ---------------------------------------------------------

class Job(NamedTuple):
    chain: str
    arm: str
    ckpt: Path
    sha256: str
    epoch: int | None

    @property
    def dataset(self) -> str:
        return split_chain(self.chain)[0]

    @property
    def pair(self) -> str:
        return split_chain(self.chain)[1]

    @property
    def label(self) -> str:
        return f"{self.chain}/{self.arm}"


def arms_of(cells: Sequence[str]) -> list[str]:
    return [a for c in cells for a in CELLS[c]]


def jobs_of(runs: Path, e020_runs: Path, selection: Mapping, arms: Sequence[str],
            chains: Sequence[str] | None = None) -> tuple[list[Job], list[str]]:
    """The test jobs of the arms (every chain unless ``chains``) from the committed selection,
    and the runs without a checkpoint for an arm (numerical or infrastructure failure)."""
    chains = chains or [chain_name(d, p, s) for d in DATASETS for p in PAIRS for s in SEEDS]
    jobs, absent = [], []
    for chain in chains:
        for arm in arms:
            head = ARMS[arm][0]
            if head == "h0":
                entry = ((selection.get("r0") or {}).get(chain) or {}).get(arm)
                root = Path(e020_runs) / chain
            else:
                entry = (((selection.get("chains") or {}).get(f"{chain}/{head}_{PIPELINE}")
                          or {}).get("checkpoints") or {}).get(arm)
                root = Path(runs) / chain
            if not entry or not entry.get("sha256"):
                absent.append(f"{chain}/{arm}")
                continue
            jobs.append(Job(chain, arm, root / entry["file"], entry["sha256"], entry.get("epoch")))
    return jobs, absent


def job_problem(job: Job) -> str | None:
    """The checkpoint has the selected sha256 and names its run (E022 chains: e021_score's epoch
    check or final_ckpt_problem; R0: the E020 m10 cell)."""
    if not job.ckpt.is_file():
        return f"{job.label}: {job.ckpt} missing"
    if sha256_file(job.ckpt) != job.sha256:
        return f"{job.label}: {job.ckpt} has another sha256 than the selection's"
    d, p, s = split_chain(job.chain)
    head, rule, _ = ARMS[job.arm]
    if head == "h0":
        return r0_problem(job.ckpt, (d, p, s), rule)
    key = es.RunKey(d, p, s, head, PIPELINE)
    if rule in ("W", "G"):
        return es.checkpoint_problem(job.ckpt, key, int(job.epoch))
    return final_ckpt_problem(job.ckpt, key, rule)


def r0_problem(path: Path, ident: tuple[str, str, int], rule: str) -> str | None:
    """An E020 m10 checkpoint: its dataset, pair and seed, the M10 options, the legacy features,
    out_dir m10, init m10_stage1/ckpt_last.pt (last rule: epoch 30)."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        cfg = obj["config"]
        out_dir, init = PurePosixPath(cfg["out_dir"]), PurePosixPath(cfg["init"])
        got = (cfg["dataset"], cfg["pair"], int(cfg["seed"]), cfg.get("global_mode", "none"),
               float(cfg.get("offset_lr_mult", 1.0)), cfg.get("features", "legacy"),
               out_dir.name, init.parent.name, init.name)
        epoch = int(obj["epoch"])
    except es.GATE_ERRORS as exc:
        return f"{path}: {exc!r}"
    want = (*ident, "mean", 10.0, "legacy", R0_CELL, f"{R0_CELL}_stage1", LAST)
    if got != want:
        return f"{path}: {got}, expected {want}"
    return None if rule != "last" or epoch == EPOCHS else f"{path}: epoch {epoch}"


def scoring_calls(log: Sequence[Mapping]) -> list[dict]:
    """The "start" lines of the test scoring calls, in order."""
    return [e for e in log if e.get("command") == COMMAND and e.get("event") == "start"]


def scoring_kind(cells: Sequence[str]) -> str:
    """'r0' or 'scoring_2'; anything else is refused (PLAN.md order: R0 first, then every new
    cell in one call)."""
    cells = list(dict.fromkeys(cells))
    if cells == ["r0"]:
        return "r0"
    if sorted(cells) == sorted(SCORING_2):
        return "scoring_2"
    raise SystemExit(f"--cells {cells}: scoring 1 is 'r0' alone, scoring 2 'c r1 b a1s' in one "
                     "call (PLAN.md 10-11)")


def check_order(log: Sequence[Mapping], kind: str, rescore: str | None, missing_only: bool,
                runs_subset: Sequence[str]) -> dict:
    """PLAN.md order and re-reads: R0 in full at most twice (the second time with --rescore,
    exception 1) and never after scoring 2; scoring 2 after an R0 read, in full once; later
    only --missing_only (infrastructure) or --runs (exception 2), each with --rescore."""
    calls = scoring_calls(log)
    full = {k: [c for c in calls if c.get("kind") == k and c.get("mode") == "full"]
            for k in ("r0", "scoring_2")}
    mode = "missing_only" if missing_only else "runs" if runs_subset else "full"
    if missing_only and runs_subset:
        raise SystemExit("--missing_only and --runs are separate re-reads")
    if mode != "full" and not rescore:
        raise SystemExit(f"{mode}: needs --rescore '<reason>' (PLAN.md re-reads)")
    if kind == "r0":
        if any(c.get("kind") == "scoring_2" for c in calls):
            raise SystemExit("scoring 2 began: R0 is not read again (PLAN.md)")
        if mode == "runs":
            raise SystemExit("--runs re-scores scoring-2 runs only (exception 2)")
        if mode == "full" and len(full["r0"]) >= 2:
            raise SystemExit("R0 was read in full twice (PLAN.md: once more after (v) failed)")
        if mode == "full" and full["r0"] and not rescore:
            raise SystemExit("R0 was read: again only with --rescore '<reason>' (exception 1)")
        if mode == "missing_only" and not full["r0"]:
            raise SystemExit("--missing_only follows a full read")
    else:
        if not full["r0"]:
            raise SystemExit("scoring 2 needs scoring 1 (R0) first (PLAN.md 10-11)")
        if mode == "full" and full["scoring_2"]:
            raise SystemExit("scoring 2 was read in full: only --missing_only or --runs with "
                             "--rescore (PLAN.md)")
        if mode != "full" and not full["scoring_2"]:
            raise SystemExit(f"{mode} follows the full scoring-2 read")
    return {"mode": mode, "prior_calls": len(calls), "prior_full_r0": len(full["r0"]),
            "prior_full_scoring_2": len(full["scoring_2"])}


def _committed(path: Path, root: Path) -> dict:
    """A results file of this tree: present; in a git checkout tracked and unchanged (a deployed
    copy holds committed files only; its descent is access item 1)."""
    if not Path(path).is_file():
        raise SystemExit(f"{path} missing: commit it and deploy (PLAN.md access device 7)")
    rel = Path(path).relative_to(root).as_posix() if Path(path).is_relative_to(root) else str(path)
    if code_version().get("source") == "git":
        if (g.git("ls-files", "--error-unmatch", "--", rel, repo=root).returncode != 0
                or g.git("diff", "--quiet", "HEAD", "--", rel, repo=root).returncode != 0):
            raise SystemExit(f"{rel} is not committed unchanged")
    return {"file": rel, "sha256": sha256_file(path)}


def unlock_scoring_2(paths: Paths, plan_sha: str, gate_file: Path | None = None,
                     extract_file: Path | None = None, root: Path | None = None,
                     datasets: Sequence[str] | None = None) -> dict:
    """Access device item 7: the committed gate record shows gate (v) passed on R0 scores made
    under this plan, and the committed extraction gate record shows the extraction gates passed
    with the dense and derived test caches' sha256 equal to the live files."""
    gate_file, extract_file = Path(gate_file or GATE_FILE), Path(extract_file or EXTRACT_GATE_FILE)
    root = Path(root or REPO_ROOT)
    files = {"gate": _committed(gate_file, root), "extract": _committed(extract_file, root)}
    gate, ext = _json(gate_file), _json(extract_file)
    v = gate.get("v") or {}
    if (gate.get("status") or {}).get("v") is not True or v.get("ok") is not True:
        raise SystemExit(f"{gate_file}: gate (v) has not passed: scoring 2 is not run")
    if (v.get("scored_with") or {}).get("plan_sha256") != plan_sha:
        raise SystemExit(f"{gate_file}: (v) checked scores of plan {v.get('scored_with')}")
    if (ext.get("gates") or {}).get("ok") is not True:
        raise SystemExit(f"{extract_file}: the extraction gates have not passed")
    datasets = list(datasets or DATASETS)
    live = dt.store_hashes(paths, datasets)
    want = {}
    for d in datasets:
        want |= dict(ext.get(f"test_cache_sha256_{d}") or {})
    bad = sorted(k for k in live if live[k] is None or want.get(k) != live[k])
    if bad or set(want) != set(live):
        raise SystemExit(f"test caches differ from {extract_file}: {bad[:5]}")
    return {"files": files, "test_caches": len(live)}


def guard_summary(guard: Mapping) -> dict:
    return {"descent": guard["descent"], "plan": guard["plan"], "approval": guard["approval"],
            "test_data_ok": guard["test_data"]["ok"], "old_caches_ok": guard["old_caches"]["ok"],
            "old_caches": {k: v.get("sha256") for k, v in guard["old_caches"].items()
                           if isinstance(v, Mapping)},
            "selection": guard["selection"],
            "train_inputs": {k: v for k, v in guard["train_inputs"].items()
                             if k in ("checked", "ok", "record", "record_sha256", "changed",
                                      "scale_changed")}}


def input_folders(paths: Paths, kind: str) -> list[str]:
    """The test-feature folders a scoring call reads (access-log record)."""
    if kind == "r0":
        dirs = [x for d in DATASETS for p in PAIRS for x in feature_dirs(paths, d, p)]
    else:
        dirs = [x for d in DATASETS for e in dt.ENCODERS
                for x in (dt.store_folder(paths, d, e), *(dt.derived_dir(
                    dt.store_folder(paths, d, e), v) for v in dt.VARIANTS))]
    return sorted({str(x) for x in dirs})


# --- test scoring: forward path (exception 1) ----------------------------------------------------

def forward_units(model, features: FeatureSet, device: torch.device) -> dict[str, np.ndarray]:
    """The 32-unit path (wvad.eval.scoring.score, batch 32: path (i) on the GPU) and gamma."""
    out = score(model, features, device, BATCH_SIZE)
    return out | {"gamma": np.float32(head_gamma(model))}


def forward_snippets(model, dense: DenseSet, n: int, device: torch.device
                     ) -> tuple[list[np.ndarray], np.float32]:
    """Offset-free snippet logits z of every test video (frozen score_dense) and gamma."""
    return score_dense(model, dense, range(n), device), np.float32(head_gamma(model))


def averaged_set(dense: DenseSet, n: int) -> FeatureSet:
    """(s3t) input: the scale-matched test snippets averaged over J_k (empty-window rule
    included; wvad.eval.dense.averaged_units), as s3_check builds it."""
    nan = torch.full((grid.WINDOWS,), float("nan"))
    return FeatureSet(tuple(averaged_units(dense, v, "d16m") for v in range(n)),
                      tuple(int(dense.labels[v]) for v in range(n)), (nan,) * n,
                      tuple(int(dense.total_frames[v]) for v in range(n)), ("dense",))


def forward_s3t(model, dense: DenseSet, cache: FeatureSet, averaged: FeatureSet,
                device: torch.device) -> tuple[np.ndarray, dict]:
    """(s3t): z of the averaged path (standard_z) and the frozen s3_check against the d16m
    cache."""
    z = standard_z(model, averaged, device)
    check = s3_check(model, dense, cache, range(len(averaged)), "d16m", device)
    return z, check


# --- test scoring: writing and indexing (exception 2) ---------------------------------------------

def manifest_arrays(manifest) -> dict[str, np.ndarray]:
    return {"names": np.asarray(manifest.names), "total_frames": np.asarray(
        manifest.total_frames, dtype=np.int64), "labels": np.asarray(manifest.labels,
                                                                     dtype=np.int64)}


def job_arrays(job: Job, obj: Mapping, device: torch.device) -> dict[str, np.ndarray]:
    cfg = obj.get("config", {})
    return {"chain": np.str_(job.chain), "arm": np.str_(job.arm), "ckpt": np.str_(str(job.ckpt)),
            "ckpt_sha256": np.str_(job.sha256), "epoch": np.int64(obj.get("epoch", -1)),
            "val_video_auc": np.float64(obj.get("val_video_auc", math.nan)),
            "config": np.str_(json.dumps(cfg, sort_keys=True, default=str)),
            "device": np.str_(str(device)), "batch_size": np.int64(BATCH_SIZE),
            "rows_per_pass": np.int64(es.ROWS_PER_PASS)}


def unit_arrays(out: Mapping[str, np.ndarray], manifest, prefix: str = "") -> dict:
    """A 32-unit forward result (n, 32) as stored: logits, probs, local_logits, offsets (and
    gamma unless ``prefix``)."""
    n = len(manifest)
    for k in ("logits", "probs", "local_logits"):
        if np.asarray(out[k]).shape != (n, grid.WINDOWS):
            raise ValueError(f"{k}: shape {np.asarray(out[k]).shape} for {n} videos")
    arrays = {f"{prefix}{k}": np.asarray(out[k], dtype=np.float32)
              for k in ("logits", "probs", "local_logits", "offsets")}
    if not prefix:
        arrays["gamma"] = np.float32(out["gamma"])
    return arrays


def snippet_arrays(z_list: Sequence[np.ndarray], gamma: np.float32, manifest) -> dict:
    """Snippet logits in manifest order with their counts (ceil(T / 16) per video), c(v) =
    gamma x mean z (dense_readout.offset), z + c and sigmoid32(z + c)."""
    totals = manifest.total_frames
    if len(z_list) != len(totals):
        raise ValueError(f"{len(z_list)} videos scored for {len(totals)} in the manifest")
    counts = np.array([len(z) for z in z_list], dtype=np.int64)
    want = np.array([grid.n_snippets(int(t)) for t in totals], dtype=np.int64)
    if not np.array_equal(counts, want):
        raise ValueError("snippet counts differ from ceil(T / 16) of the manifest")
    g32 = float(gamma)
    c = np.array([offset(z, g32) for z in z_list], dtype=np.float32)
    z = np.concatenate([np.asarray(x, dtype=np.float32) for x in z_list])
    zc = (z + np.repeat(c, counts)).astype(np.float32)
    return {"z": z, "counts": counts, "gamma": np.float32(gamma), "c": c, "zc": zc,
            "sigma": sigmoid32(zc)}


def nonfinite(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Non-finite values per stored score array (gate (v3) counts them; not a stop)."""
    return {k: int(np.size(arrays[k]) - np.isfinite(np.asarray(arrays[k], dtype=np.float64)
                                                    ).sum())
            for k in NONFINITE_KEYS if k in arrays}


# --- test scoring: one (dataset, pair) ---------------------------------------------------------

def _check_set(fs: FeatureSet, manifest, what: str, totals: bool = True) -> FeatureSet:
    if len(fs) != len(manifest) or list(fs.labels) != list(manifest.labels):
        raise SystemExit(f"{what}: videos or labels differ from the test manifest")
    if totals and [int(t) for t in fs.total_frames] != list(manifest.total_frames):
        raise SystemExit(f"{what}: total_frames differ from the test manifest")
    return fs


class Inputs:
    """The test inputs of one (dataset, pair), each loaded at its first use."""

    def __init__(self, paths: Paths, manifest, pair: str):
        self.paths, self.manifest, self.pair = paths, manifest, pair
        self._cache: dict = {}

    def get(self, name: str):
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name: str):
        d, p, m = self.manifest.dataset, self.pair, self.manifest
        if name == "legacy":
            return _check_set(load_feature_set(feature_dirs(self.paths, d, p), "test"), m,
                              f"{d}/{p} earlier test cache", totals=False)
        if name in ("d16m", "d16c"):
            return _check_set(load_feature_set(dt.derived_dirs(self.paths, d, p, name), "test"),
                              m, f"{d}/{p} test {name}")
        if name in ("dense_scaled", "dense_raw"):
            dense = dt.dense_snippets(self.paths, d, p, scaled=name == "dense_scaled")
            dt.check_dense(dense, m)
            return dense
        if name == "averaged":
            return averaged_set(self.get("dense_scaled"), len(m))
        raise KeyError(name)


def score_job(job: Job, inputs: Inputs, device: torch.device) -> dict[str, np.ndarray]:
    """One checkpoint on its test inputs; the arrays of its score file."""
    if sha256_file(job.ckpt) != job.sha256:
        raise SystemExit(f"{job.label}: checkpoint changed after the access line")
    _, _, source = ARMS[job.arm]
    m = inputs.manifest
    first = inputs.get(source)
    dim = first.dim if isinstance(first, FeatureSet) else sum(first.dims)
    model, obj = load_head(job.ckpt, dim, device)
    arrays = manifest_arrays(m) | job_arrays(job, obj, device)
    if job.arm in SNIPPET_ARMS:
        z_list, gamma = forward_snippets(model, first, len(m), device)
        arrays |= snippet_arrays(z_list, gamma, m)
        if job.arm == "C":
            cache = inputs.get("d16m")
            arrays |= unit_arrays(forward_units(model, cache, device), m, prefix="o32_")
            s3_z, check = forward_s3t(model, first, cache, inputs.get("averaged"), device)
            arrays |= {"s3t_z": np.asarray(s3_z, dtype=np.float32),
                       "s3t_check": np.str_(json.dumps(check, sort_keys=True))}
    else:
        arrays |= unit_arrays(forward_units(model, first, device), m)
    return arrays


def score_jobs(paths: Paths, jobs: Sequence[Job], out: Path, device: torch.device,
               on_written: Callable[[Path], None] | None = None) -> dict:
    """Every job by (dataset, pair): its file and its non-finite counts; ``on_written`` is
    called with each file right after it is renamed into place (the call's meta records it)."""
    written, counts = [], {}
    for d in DATASETS:
        m = dt.read_manifest(d)
        for p in PAIRS:
            mine = [j for j in jobs if (j.dataset, j.pair) == (d, p)]
            if not mine:
                continue
            inputs = Inputs(paths, m, p)
            for j in mine:
                arrays = score_job(j, inputs, device)
                f = arm_path(out, j.chain, j.arm)
                write_scores(f, arrays)
                if on_written is not None:
                    on_written(f)
                written.append(f)
                counts[j.label] = nonfinite(arrays)
            print(f"[e022_score] test {d}/{p}: {len(mine)} files", flush=True)
    return {"written": written, "nonfinite": counts}


def recorded_files(target: Path, kind: str) -> dict[str, set[str]]:
    """'<chain>/<arm>.npz' -> the sha256 values the calls of a kind (scoring 1 'r0', scoring 2)
    recorded for the files they wrote (meta ``written_sha256``, each file recorded right after
    it was written, so an interrupted call's files are recorded too)."""
    out: dict[str, set[str]] = {}
    for f in Path(target).glob("meta_*.json"):
        meta = _json(f)
        if meta.get("kind") == kind:
            for rel, sha in (meta.get("written_sha256") or {}).items():
                out.setdefault(rel, set()).add(sha)
    return out


def rescore_files(jobs: Sequence[Job], out: Path, mode: str, rescore: str | None = None,
                  recorded: Mapping[str, set[str]] | None = None
                  ) -> tuple[list[Job], list[Path]]:
    """The jobs a call scores and the existing files it moves aside: a full read writes every
    file (existing files: refused, or with --rescore, R0's second read of exception 1, moved
    aside), --missing_only takes absent or unreadable files and (with ``recorded``) files no
    call of the kind recorded with their sha256 (a call stopped between writing a file and
    recording it), --runs the named runs."""
    paths_ = {j.label: arm_path(out, j.chain, j.arm) for j in jobs}
    if mode == "full":
        existing = [p for p in paths_.values() if p.exists()]
        if existing and not rescore:
            raise SystemExit(f"{len(existing)} score files exist (e.g. {existing[0]})")
        return list(jobs), existing
    if mode == "missing_only":
        target = Path(out) / "test"

        def missing(p: Path) -> bool:
            if not (p.exists() and zipfile.is_zipfile(p)):
                return True
            rel = p.relative_to(target).as_posix()
            return recorded is not None and sha256_file(p) not in recorded.get(rel, ())
        todo = [j for j in jobs if missing(paths_[j.label])]
        return todo, [paths_[j.label] for j in todo if paths_[j.label].exists()]
    return list(jobs), [paths_[j.label] for j in jobs if paths_[j.label].exists()]


def supersede(files: Sequence[Path], out: Path) -> str | None:
    if not files:
        return None
    target = Path(out) / "test"
    dest = target / "superseded" / dt_time.datetime.now(dt_time.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    for f in files:
        new = dest / Path(f).relative_to(target)
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(new))
    return str(dest)


# --- test scoring: the call ------------------------------------------------------------------

def parse_runs(items: Sequence[str], kind: str) -> list[str]:
    """--runs '<chain>/<arm>' of scoring 2 (exception 2)."""
    out = []
    for it in items:
        chain, _, arm = it.partition("/")
        try:
            split_chain(chain)
        except ValueError as exc:
            raise SystemExit(f"--runs {it}: '<dataset>_<pair>_s<seed>/<arm>'") from exc
        if kind != "scoring_2" or arm not in arms_of(SCORING_2):
            raise SystemExit(f"--runs {it}: a scoring-2 run (exception 2)")
        out.append(f"{chain}/{arm}")
    return sorted(set(out))


def score_test(args: argparse.Namespace) -> None:
    kind = scoring_kind(args.cells)
    paths = load_paths(args.paths)
    _outside(args.runs_root, args.out, args.test_log, args.e020_runs)
    subset = parse_runs(args.runs or (), kind)
    target = Path(args.out) / "test"
    line = {"access": ACCESS[kind], "command": COMMAND, "kind": kind, "cells": list(args.cells)}

    def refuse(reason: str):
        dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "refused",
                                                  "reason": reason, "code": code_version()})
        raise SystemExit(reason)

    try:
        order = check_order(dt.read_test_log(args.test_log), kind, args.rescore,
                            args.missing_only, subset)
        guard = dt.access_guard(paths, freeze=args.freeze_commit, lineage=args.lineage,
                                selection_sha256=args.selection_sha256,
                                train_inputs=args.train_inputs,
                                check_inputs=kind == "scoring_2")
        unlock = (unlock_scoring_2(paths, guard["plan"]["sha256"]) if kind == "scoring_2"
                  else None)
        selection = _json(SELECTION_FILE)
        jobs, absent = jobs_of(args.runs_root, args.e020_runs, selection, arms_of(args.cells))
        if subset:
            jobs = [j for j in jobs if j.label in subset]
            if len(jobs) != len(subset):
                raise SystemExit(f"--runs names runs without a selected checkpoint: {subset}")
        problems = [x for x in map(job_problem, jobs) if x]
        if problems:
            raise SystemExit(f"{len(problems)} checkpoint problem(s): {problems[:5]}")
        todo, aside = rescore_files(jobs, args.out, order["mode"], args.rescore,
                                    recorded_files(target, kind))
        k = len(dt.read_test_log(args.test_log)) + 1
        if any(int(f.stem.split("_", 1)[1]) >= k for f in target.glob("meta_*.json")):
            raise SystemExit(f"{target} has a meta of access-log line {k} or later: another "
                             "access log was used")
    except SystemExit as exc:
        refuse(str(exc))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = {"split": "test", "kind": kind, "cells": list(args.cells), "mode": order["mode"],
            "order": order, "time_utc": now_utc(), "code": code_version(),
            "environment": gpu_environment(), "device": str(device), "rescore": args.rescore,
            "runs_subset": subset, "guard": guard_summary(guard), "unlock": unlock,
            "absent": absent, "jobs": [j.label for j in todo],
            "access_log": os.path.realpath(args.test_log), "out": str(target)}
    entry = line | {"time_utc": now_utc(), "event": "start", "mode": order["mode"],
                    "rescore": args.rescore, "runs": [j.label for j in todo], "absent": absent,
                    "inputs": input_folders(paths, kind), "out_dir": str(target),
                    "deployed_commit": guard["descent"]["deployed_commit"],
                    "plan_sha256": guard["plan"]["sha256"], "guard": guard_summary(guard)}
    line_no = dt.append_test_log(args.test_log, entry)
    meta |= {"access_log_line": line_no,
             "access_log_entry": json.loads(json.dumps(entry, sort_keys=True, default=str))}
    meta_file = target / f"meta_{line_no}.json"
    meta["written_sha256"] = {}
    write_json(meta_file, meta | {"complete": False})

    def recorded(f: Path) -> None:
        """Bind the file to this call (the analysis's per-file binding, gates (v) / (v3))."""
        meta["written_sha256"][f.relative_to(target).as_posix()] = sha256_file(f)
        write_json(meta_file, meta | {"complete": False})

    try:
        meta["superseded"] = supersede(aside, args.out)
        res = score_jobs(paths, todo, args.out, device, on_written=recorded)
        after = (dt.check_train_inputs(dt.load_train_inputs(args.train_inputs), paths)
                 if kind == "scoring_2" else {"checked": False})
    except BaseException as exc:
        dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "error",
                                                  "error": repr(exc), "line": line_no})
        raise
    write_list(target)
    meta |= {"written": [f.relative_to(target).as_posix() for f in res["written"]],
             "nonfinite": res["nonfinite"], "train_inputs_after": after,
             "time_utc_end": now_utc(), "complete": True}
    write_json(meta_file, meta)
    dt.append_test_log(args.test_log, line | {
        "time_utc": now_utc(), "event": "end", "line": line_no, "written": len(res["written"]),
        "nonfinite_runs": sorted(k for k, v in res["nonfinite"].items() if any(v.values())),
        "train_inputs_after": after})
    print(f"[e022_score] test {kind}: {len(res['written'])} files -> {target}")
    if kind == "scoring_2" and not after["ok"]:
        raise SystemExit(f"E021 training inputs changed during scoring 2: {after}")


# --- command line ------------------------------------------------------------------------------

def _access_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--test_log", required=True, help="test access log (outside the repository)")
    p.add_argument("--selection_sha256", required=True,
                   help="sha256 of the committed results/e022_selection.json")
    p.add_argument("--freeze_commit", help="the freeze commit (required in a deployed copy)")
    p.add_argument("--lineage", default=str(dt.LINEAGE_FILE),
                   help="deploy lineage file (DEPLOYED_LINEAGE.json in the deploy root)")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e022_score")
    ap.add_argument("--paths", help="paths file (default: $WVAD_PATHS or configs/paths.yaml)")
    ap.add_argument("--train_inputs", default=str(dt.TRAIN_INPUTS_FILE),
                    help="committed results/e022_train_inputs.json")
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("env", help="gate (f): a training batch's environment record")
    e.add_argument("--out", required=True)
    s = sub.add_parser("score", help="--split val (validation) or test (scoring 1 / 2)")
    s.add_argument("--split", required=True, choices=("val", "test"))
    s.add_argument("--runs_root", required=True, type=Path, help="E022 runs root")
    s.add_argument("--out", required=True, type=Path, help="score root outside the repository")
    s.add_argument("--e020_runs", required=True, type=Path, help="E020 runs root (R0)")
    s.add_argument("--cells", nargs="+", choices=tuple(CELLS))
    s.add_argument("--infra_failed", nargs="*", default=[],
                   help="val: '<chain>/<run>' whose rerun also failed (infrastructure)")
    s.add_argument("--rescore", help="reason: val, move val/ aside; test, a permitted re-read")
    s.add_argument("--missing_only", action="store_true",
                   help="test: absent or unreadable files only (infrastructure failure)")
    s.add_argument("--runs", nargs="+", metavar="CHAIN/ARM",
                   help="test, scoring 2, with --rescore: re-score these runs only (exception 2)")
    s.add_argument("--test_log")
    s.add_argument("--selection_sha256")
    s.add_argument("--freeze_commit")
    s.add_argument("--lineage", default=str(dt.LINEAGE_FILE))
    c = sub.add_parser("select", help="W / G checkpoint selection (validation only)")
    c.add_argument("--scores", required=True, type=Path, help="score root (its val/)")
    c.add_argument("--out", default=str(dt.SELECTION_FILE))
    c.add_argument("--infra_failed", nargs="*", default=[])
    x = sub.add_parser("extract-gates", help="(y4)-(y6) on the test stores (server, access A)")
    _access_args(x)
    x.add_argument("--record", required=True, help="the server's extraction gate record")
    x.add_argument("--headers", required=True, help="the (y3) header CSV of testsplit header")
    x.add_argument("--env", required=True, help="the extraction environment (gate env)")
    x.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = ap.parse_args(argv)
    if args.command == "score" and args.split == "test":
        missing = [k for k in ("cells", "test_log", "selection_sha256") if not getattr(args, k)]
        if missing:
            ap.error(f"--split test needs {', '.join('--' + m for m in missing)}")
    if args.command == "score" and args.split == "val" and (args.cells or args.missing_only
                                                            or args.runs):
        ap.error("--split val scores every run (no --cells, --missing_only, --runs)")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    if args.command == "env":
        paths = load_paths(args.paths)
        rec = batch_env(paths, Path(args.out), _record(args.train_inputs))
        print(f"{args.out}: environment {rec['env_differences'] or 'as E021'}, inputs ok")
    elif args.command == "score":
        (score_val if args.split == "val" else score_test)(args)
    elif args.command == "select":
        select_command(args)
    else:
        extract_gates_command(args)


if __name__ == "__main__":
    main()
