"""E024 stage 3: the confirmation on UBnormal (experiments/E024_localisation_method/stage3/PLAN.md;
the sealed test read, once).

Arms, per dataset (ubn the confirmation; ucf and xdv the fourth read of their test sets, auxiliary
and non-independent, never in the outcome) x pair (xcb_xcl, xcb_siglip) x new seed (3, 4, 5), in
<runs>/<ds>_<pair>_s<seed>/<folder>:
    P     k_rg_t0.25              UR-DMU + K (rho global, tau 0.25; stage 1's candidate):
                                  ``train --arm p``
    U     urdmu_u200              UR-DMU (= P - K): ``python -m wvad.baselines.train --method urdmu
                                  --unit 200 ... --lr_mult <lr> --scale_mult 1 --n_chunk 4096
                                  --deterministic`` (stage 0a's command with the lr multiplier)
    C6    ctl_trainonly_rg_t0.25  K in training only, scored by z (stage 1's final_control):
                                  ``train --arm c6``
    gate  <runs>/ubn_xcb_xcl_s3/gate_k0 (``train --arm k0``, mode off): gate (k0) against U
Multipliers (the stage-2 development run: lr in {0.5, 1, 2}, scale 1): the ubn chains use the lr
multiplier m chosen by ``pins`` and scale 1, the ucf and xdv chains lr 1 and scale 1 as in stage 1
(the inputs record's ``multiplier.by_dataset``, ``settings_of``): ``train`` is
``e024_stage1.train_chain`` when both are 1 and the same steps with lr x m otherwise.

Scores: the sigma checkpoint of ``selection.json`` (P selected by s = z + k, C6 by z, U by z),
rebuilt as stage 1's ``rescore`` does and scored through ``common.score_video`` (whole videos,
N_chunk pieces) in deterministic mode: P scores s, C6 and U z; P is also scored with k = 0 (I2,
descriptive). Snippet logits are repeated over their supports [16 j, min(16 j + 16, T)).

Metrics per chain (``wvad.eval.frames``), in points: L, the equal-weight mean over the anomalous
test videos whose frames hold both classes of their within-video frame AUC (ubn, ucf) or AP
(xdv); the pooled frame AUC and AP (videos >= 32 frames); AUC_A, the pooled frame AUC of the
anomalous videos. Frame labels: ubn from the tracks files (``wvad.data.ubnormal.test_frame_gts``:
a video whose tracks fail is left out of every metric and named), ucf and xdv from the test
manifest's intervals. ``labels.npz`` keeps per snippet the positive frame count of its support:
frames share their snippet's score, so these counts give every metric exactly (the analysis
recomputes L from them and must meet summary.json's values).

Statistics (PLAN.md "통계와 판정"): every numerically complete chain enters the main analysis (a
missing chain or a non-finite value stops); a chain whose selected sigma V-AUC is below the
inputs record's collapse threshold is only flagged (``collapsed``), and the descriptive block
``reference_excluding_flagged`` repeats the two contrasts on the seeds where P and B are unflagged
in both pairs (fewer than two: ``seeds_insufficient``, no interval; never the outcome). A
contrast P - B (B = U, C6) uses the seeds 3, 4, 5; Delta_{v,s} = the pair mean of L_v(P) minus
that of L_v(B); the estimate is the mean over the L videos and the seeds. Main interval: the
video x seed bootstrap of glance.py and stage 1's sensitivity rows: rows 1..R each draw the L
videos and the paired seeds with replacement (one video draw and one seed draw per row, shared:
``glance._draw_counts`` from np.random.default_rng(20261026), videos then seeds; R = 2000; row 0
the sample), row r = sum_v sum_s w_rv m_rs Delta_vs / (sum_v w_rv sum_s m_rs), the 95% percentile
interval over rows 1..R. Also: the video-only interval (seeds averaged first, the same video
draws), the per-seed and per-pair point estimates, the pooled-metric contrasts (rows of
``glance.bootstrap_draws``: anomalous and normal videos separately, then the seeds), I2 (P - I2,
I2 - U) and length tertiles (descriptive). Class: 양 (lower > 0), 열세 (upper < 0), else 차이
불확정, with PLAN.md's wording ("해석 문구"). Outcome, ubn only: the first matching row of PLAN.md's
table (no_verdict, met, partial_u, partial_c6, not_met) and its statement composed from the two
classes ("UR-DMU 대비 <class>, C6 대비 <class>"; met: the claim).

Commands:
    train       one P / C6 / gate_k0 chain (server); seeds 3-5 only, the registered folder of its
                cell, the dataset's lr and scale multipliers (``multiplier.by_dataset``);
                stage3.json first, beside stage 1's k_config.json
    pins        results/e024_stage3_inputs.json before the freeze, once (gate (p)): the UBnormal
                manifests and split (sha256, meta), the auxiliary test manifests (sha256), the six
                official lists (pinned sha256), the stage-2 tracks check (--tracks_check: the
                sha256 of the 158 test tracks files, the 38 validation entries), the licence
                record, the archive sha256, the stage-2 development U chains
                (<u_dev>/<pair>_s<seed>/urdmu_u200_lr<m>; their config.json must hold lr_mult m of
                the grid and scale_mult 1), the collapse flag thresholds (ubn: 0.5 + (m_U - 0.5) /
                2, m_U the median sigma V-AUC of the six lr x 1 chains; ucf, xdv: stage 1's
                0.60), the lr multiplier (highest mean sigma V-AUC of its six chains; ties 1,
                0.5, 2) with ``by_dataset``, their environments, stage 1's candidate and
                final_control
    freeze      stage3/PLAN.sha256 (clean tree; the plan, the inputs record and the code committed)
    gate-k0     gate_k0 against its cell's U chain: val_scores z (30 points) and selection, bit for
                bit (<runs>/gate_k0.json)
    gates       (h) the arms of a cell share their input hashes; (e) trained parameters P - U = C6
                - U = 65,793 = k_config.json's g count, gate_k0 - U = 0; (n) folder names against
                k_config.json, stage3.json and every configuration against the inputs record (the
                dataset's multipliers); (v) one environment for every chain
                (<runs>/gate_{h,e,n,v}.json, every entry with the chain's selection.json sha256; a
                missing or unregistered chain makes a gate fail)
    score-test  THE sealed test read, ubn, ucf, xdv: once (a completed read is never repeated;
                after an incomplete one, --infra_rerun <reason>, one rerun, counted by the access
                log's start rows whatever --out was used). Before any test read: the seal and the
                deploy (PLAN.sha256 = the sealed part's hash, a code commit, a clean tree where
                git is available, the inputs record), the running commit = the stage-3 freeze
                commit (``dense_test.deployed_descent``; --freeze_commit and DEPLOYED_LINEAGE.json
                in a deployed copy), every chain's configuration, multipliers, code commit (=
                the freeze commit) and environment, the gates recomputed from the
                chain folders and the stored gate files, then gate (s) for every chain on
                validation (<runs>/gate_s.json); then a start row and per dataset the scores of
                every chain (<out>/<ds>/<pair>_s<seed>/<arm>.npz; the store bound to the manifest
                by its names sha256), the labels after them (ubn: each tracks file's sha256
                against the inputs record before any is parsed), labels.npz, summary.json; an end
                row and done.json (lineage). Every test-side read is appended to the access log
                first. <out> and the log stay outside the repository, the log outside <out>.
    analyze     results/e024_stage3.json and e024_stage3_runs.csv on the frozen state: clean tree,
                the freeze commit (the seal added alone, the inputs record of its parent, src/
                unchanged since), the inputs files, done.json (complete, every dataset, its commit
                the freeze commit or a descendant whose diff touches nothing under src/,
                stage3/PLAN.md, PLAN.sha256 or the inputs record; the sealed plan, the inputs
                record of HEAD, every score file's sha256), the access log (--access_log: its
                sha256, start rows = attempts, the end row last, the attempt's rows), the gate
                files ok, listing every chain with its current selection.json and as done.json
                recorded them, every chain's record (configuration, multipliers, done.json's
                commit, checkpoint and point as scored), the per-video L recomputed from
                labels.npz equal to summary.json's
"""

from __future__ import annotations

import argparse
import csv
import datetime as dtm
import hashlib
import json
import math
import os
import platform
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wvad.baselines import common
from wvad.baselines import train as tr
from wvad.baselines import urdmu_k as uk
from wvad.baselines.recipes import parse_mult
from wvad.data import dense_test as dt
from wvad.data import load_split
from wvad.data import ubnormal as ub
from wvad.data.catalog import split_file
from wvad.data.dense import META_FILE, DenseSet, store_dir
from wvad.data.dense_test import CHANGELOG, freeze_commit_of, plan_sha256
from wvad.data.eval_manifests import eval_manifest_file, load_eval_manifest
from wvad.diagnostics import e024_stage0 as s0
from wvad.diagnostics import e024_stage1 as s1
from wvad.diagnostics import glance as gl
from wvad.diagnostics.common import code_version, write_csv, write_json
from wvad.diagnostics.e023_prefreeze import split_rows
from wvad.eval import frames as fr
from wvad.eval.metrics import DEFAULT_MIN_FRAMES
from wvad.extract import grid
from wvad.paths import REPO_ROOT, load_paths

STAGE3_DIR = s0.E024_DIR / "stage3"
RESULTS = s0.RESULTS
SEAL_FILE = s0.SEAL_FILE
INPUTS = RESULTS / "e024_stage3_inputs.json"
RESULT_FILE = RESULTS / "e024_stage3.json"
RUNS_CSV = RESULTS / "e024_stage3_runs.csv"
STAGE1_RESULT = RESULTS / "e024_stage1.json"
UBN_FILES = (REPO_ROOT / "data" / "manifests" / "ubn_train.json",       # sha256 and meta
             REPO_ROOT / "data" / "manifests" / "ubn_test.json",
             REPO_ROOT / "data" / "splits" / "ubn_val_official.json")
AUX_FILES = (REPO_ROOT / "data" / "manifests" / "ucf_test.json",        # sha256 only
             REPO_ROOT / "data" / "manifests" / "xdv_test.json")

DATASETS = ("ubn", "ucf", "xdv")             # scoring order: UBnormal first
CONFIRMATION = "ubn"                         # the others: auxiliary, non-independent
PAIRS = s0.PAIRS
SEEDS = (3, 4, 5)
DEV_SEEDS = s0.DEV_SEEDS                     # the stage-2 development U chains
LR_GRID = ("0.5", "1", "2")                  # the stage-2 learning-rate check: lr x m, scale 1
SCALE = "1"                                  # the scale multiplier of every chain
MULT_ORDER = ("1", "0.5", "2")               # ties: 1 first
DEV_FOLDER = "urdmu_u200_lr{}"
ARMS = {"p": s1.k_folder(None, 0.25), "u": s0.U_RUN,
        "c6": s1.control_folder("trainonly", None, 0.25)}
LABEL = {"p": "P", "u": "U", "c6": "C6", "i2": "I2"}
TRAIN = {"p": ("k", None, 0.25), "c6": ("trainonly", None, 0.25), "k0": ("off", None, 1.0)}
TRAIN_FOLDER = {"p": ARMS["p"], "c6": ARMS["c6"], "k0": s1.GATE_K0}
METHOD = {"p": uk.NAME, "c6": uk.NAME, "k0": uk.NAME, "u": "urdmu"}
SCORED_K = {"p": True, "c6": False, "u": None}     # k_in_logit of the logits a chain scored with
COMPARED = {"p": "s", "c6": "z0", "u": "z"}
GATE_CELL = ("ubn", "xcb_xcl", 3)
CANDIDATE, FINAL_CONTROL = ARMS["p"], "trainonly"   # stage 1's result (checked by pins)
CHAIN_CONFIG = {"unit": 200, "deterministic": True, "n_chunk": common.N_CHUNK_DEFAULT,
                "iterations": None, "eval_points": common.EVAL_POINTS, "evaluate": True,
                "float64": False}
DEV_CONFIG = {"method": "urdmu", "unit": 200, "deterministic": True,
              "n_chunk": common.N_CHUNK_DEFAULT, "dataset": "ubn"}
CONTRASTS = {"P-U": ("p", "u"), "P-C6": ("p", "c6")}
DESCRIPTIVE = {"P-I2": ("p", "i2"), "I2-U": ("i2", "u")}
POOLED = ("frame_auc", "frame_ap")
N_BOOT, BOOT_SEED, LEVEL = 2000, 20261026, 95
GATE_FILES = ("gate_k0.json", "gate_h.json", "gate_e.json", "gate_n.json", "gate_v.json")
GATE_S = "gate_s.json"
DONE, STARTED, SUMMARY, LABELS_FILE = "done.json", "started.json", "summary.json", "labels.npz"
SUPERSEDED = "superseded"
STAGE_FILE = "stage3.json"                   # beside stage 1's k_config.json in P, C6, gate_k0
STAGE = "E024 stage 3"
COMMAND = "e024-stage3-score-test"
START, END = "sealed read start", "sealed read end"      # access-log purposes of an attempt
MANIFEST_PURPOSE = "test manifest (names, labels, frame counts; no intervals)"
AUX_COLLAPSE = s0.COLLAPSE                   # 0.60, stage 1's flag threshold: ucf and xdv
SRC = "src"                                  # the code a freeze covers (git diff -- src)
HEX64 = re.compile(r"[0-9a-f]{64}")
CODE_FILES =(Path(__file__).resolve(), Path(s1.__file__).resolve(), Path(uk.__file__).resolve(),
              Path(ub.__file__).resolve())
NOT_REDISTRIBUTED = ("test scores and labels derived from the test annotations: kept outside the "
                     "repository, not redistributed (UBnormal CC BY-NC-ND 4.0)")
# PLAN.md "통계와 판정", verbatim: the classes, the wording and the outcome table
CLASS_POSITIVE, CLASS_UNDETERMINED, CLASS_NEGATIVE = "양", "차이 불확정", "열세"
INTERPRETATION = {"P-U": "K의 이점은 확인되지 않았다",
                  "P-C6": "추론 시 K를 사용하는 추가 이점은 확인되지 않았다"}
CAUSE = "원인은 단정하지 않는다"
CLAIM = ("UBnormal에서 P(K와 함께 학습하고 s로 선택한 절차)는 UR-DMU와 C6(학습 전용 대안)보다 L이 "
         "높다(95% 하한 > 0)")
ROWS = {"no_verdict": ("관문 실패·판정 없음", "보고하고 사용자 결정"),
        "met": ("두 하한 모두 > 0", "원고 결과·주장 작성(아래 \"결과의 쓰임\")"),
        "partial_u": ("P − U 하한 > 0, P − C6 하한 ≤ 0",
                      "미충족. 두 대비의 분류를 결과 문장으로 적고 "
                      "추론 시 K의 이점은 주장하지 않음"),
        "partial_c6": ("P − C6 하한 > 0, P − U 하한 ≤ 0",
                       "미충족. 두 대비의 분류를 결과 문장으로 그대로 적음"),
        "not_met": ("둘 다 하한 ≤ 0",
                    "미충족. 두 대비의 분류를 결과 문장으로 그대로 보고; 사용자 결정")}
# PLAN.md "결과 문장": composed from the two classes (a contrast with upper < 0 stays 열세)
STATEMENT = {"no_verdict": "판정 없음(관문 실패)",
             "met": "UBnormal에서 P는 UR-DMU와 C6보다 L이 높다(95% 하한 > 0)",
             "partial_u": "UR-DMU 대비 양, C6 대비 {c6}",
             "partial_c6": "C6 대비 양, UR-DMU 대비 {u}",
             "not_met": "UR-DMU 대비 {u}, C6 대비 {c6}"}


def utc_now() -> str:
    return dtm.datetime.now(dtm.timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(Path(REPO_ROOT).resolve()).as_posix()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _same_mult(value, mult: str) -> bool:
    try:
        return parse_mult(value) == parse_mult(mult)
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _one(values: set, what: str):
    if len(values) != 1:
        raise SystemExit(f"{what}: {sorted(map(str, values))}, expected one value")
    return next(iter(values))


# --- folders --------------------------------------------------------------------------------

def cells() -> list[tuple[str, str, int]]:
    return [(d, p, s) for d in DATASETS for p in PAIRS for s in SEEDS]


def chain_key(d: str, p: str, s: int, folder: str) -> str:
    return f"{s0.chain_name(d, p, s)}/{folder}"


def expected_chains(runs: Path) -> list[tuple[str, tuple[str, str, int], str, Path]]:
    """(key, cell, arm, folder) of every registered chain: the three arms of every cell, then
    the gate chain (arm "k0")."""
    out = [(chain_key(*c, ARMS[a]), c, a, Path(runs) / s0.chain_name(*c) / ARMS[a])
           for c in cells() for a in ARMS]
    return out + [(chain_key(*GATE_CELL, s1.GATE_K0), GATE_CELL, "k0",
                   Path(runs) / s0.chain_name(*GATE_CELL) / s1.GATE_K0)]


def unregistered(runs: Path) -> list[str]:
    """Folders inside the registered cells of ``runs`` that are no registered chain."""
    want = {k for k, *_ in expected_chains(runs)}
    out = []
    for name in dict.fromkeys(s0.chain_name(*c) for c in cells()):
        c = Path(runs) / name
        out += [f"{name}/{f.name}" for f in sorted(c.iterdir()) if f.is_dir()
                and f"{name}/{f.name}" not in want] if c.is_dir() else []
    return out


# --- the inputs record (gate (p)) -----------------------------------------------------------

def input_files() -> dict:
    """sha256 of the UBnormal manifests and split with their meta, and of the auxiliary test
    manifests (bytes only: no interval is decoded)."""
    out = {_rel(f): {"sha256": s0.sha256_file(f), "meta": _json(f).get("meta")}
           for f in UBN_FILES}
    return out | {_rel(f): {"sha256": s0.sha256_file(f)} for f in AUX_FILES}


def check_input_files(inputs: Mapping) -> dict:
    """Gate (p) for the repository's input files: as the inputs record (SystemExit otherwise)."""
    now, rec = input_files(), inputs.get("files") or {}
    bad = sorted(k for k in set(now) | set(rec) if now.get(k) != rec.get(k))
    if bad:
        raise SystemExit(f"input files differ from {INPUTS.name}: {bad}")
    return {"ok": True, "files": sorted(now)}


def read_inputs() -> dict:
    if not INPUTS.is_file():
        raise SystemExit(f"{INPUTS}: no inputs record (pins, before the freeze)")
    return _json(INPUTS)


def by_dataset(lr: str) -> dict[str, dict[str, str]]:
    """The multipliers per dataset: the ubn chains lr x ``lr`` and scale 1, the ucf and xdv
    chains lr 1 and scale 1 (stage 1)."""
    return {d: {"lr_mult": str(lr) if d == CONFIRMATION else "1", "scale_mult": SCALE}
            for d in ("ubn", "ucf", "xdv")}


def settings_of(dataset: str, inputs: Mapping) -> dict[str, str]:
    """The lr and scale multipliers of the chains of ``dataset``: the inputs record's
    ``multiplier.by_dataset`` (SystemExit unless the chosen lr is of the grid, the scale is 1 and
    by_dataset is ``by_dataset(chosen_lr_mult)``)."""
    m = inputs.get("multiplier") or {}
    lr, recorded = m.get("chosen_lr_mult"), m.get("by_dataset")
    if lr not in LR_GRID or m.get("scale_mult") != SCALE or recorded != by_dataset(lr) or \
            dataset not in recorded:
        raise SystemExit(f"inputs record multiplier {m} for {dataset}: by_dataset must give ubn "
                         f"the chosen lr multiplier of {LR_GRID} with scale {SCALE}, ucf and xdv "
                         "lr 1 and scale 1")
    return dict(recorded[dataset])


def dev_dir(u_dev: Path, pair: str, seed: int, lr: str) -> Path:
    return Path(u_dev) / f"{pair}_s{seed}" / DEV_FOLDER.format(lr)


def _dev_chain(f: Path, pair: str, seed: int, lr: str) -> tuple[dict | None, str | None]:
    """One stage-2 development U chain's record, or the reason it cannot be used: its
    config.json must hold the folder's lr multiplier and scale 1 as given to the train command,
    and selection.json the same configuration."""
    rec = s1._record(f)
    if not rec:
        return None, "no selection.json"
    if not (f / tr.FILES["config"]).is_file():
        return None, "no config.json"
    ran = {k: _json(f / tr.FILES["config"]).get(k) for k in ("lr_mult", "scale_mult")}
    if ran != {"lr_mult": lr, "scale_mult": SCALE}:
        return None, (f"config.json {ran}: the development run is lr_mult {lr} (its folder, of "
                      f"{LR_GRID}) with scale_mult {SCALE}")
    cfg = rec.get("config") or {}
    want = {**DEV_CONFIG, "pair": pair, "seed": seed, **ran}
    wrong = {k: cfg.get(k) for k, v in want.items() if cfg.get(k) != v}
    sig = (rec.get("selection") or {}).get("sigma")
    if wrong:
        return None, f"selection.json config {wrong}"
    if rec.get("failed") or not sig or int(rec.get("nonfinite_losses") or 0):
        return None, f"numerical failure ({rec.get('failed') or 'no sigma selection'})"
    val = f / tr.FILES["val"]
    if not val.is_file() or s0.sha256_file(val) != rec.get("val_scores_sha256"):
        return None, "val_scores.npz missing or not selection.json's"
    return {**ran, "pair": pair, "seed": seed, "v_auc": float(sig["v_auc"]),
            "point": int(sig["point"]), "parameters": rec.get("parameters"),
            "selection_sha256": s0.sha256_file(f / tr.FILES["record"]),
            "val_scores_sha256": rec["val_scores_sha256"], "environment": rec.get("environment"),
            "code": rec.get("code")}, None


def dev_chains(u_dev: Path) -> dict[str, dict]:
    """The 18 stage-2 development U chains (validation only; lr multipliers 0.5, 1, 2 with scale 1
    x pairs x seeds 0-2); SystemExit, naming them, on a missing, misconfigured or failed one."""
    s0.refuse_test_paths(u_dev)
    out, problems = {}, []
    for m in LR_GRID:
        for p in PAIRS:
            for s in DEV_SEEDS:
                f = dev_dir(u_dev, p, s, m)
                rec, why = _dev_chain(f, p, s, m)
                if why:
                    problems.append(f"{p}_s{s}/{f.name}: {why}")
                else:
                    out[f"{p}_s{s}/{f.name}"] = rec
    if problems:
        raise SystemExit(f"stage-2 development chains: {problems}; stop and report")
    return out


def collapse_threshold(dev: Mapping[str, Mapping]) -> dict:
    """PLAN.md "붕괴와 실패": 0.5 + (m_U - 0.5) / 2, m_U the median sigma V-AUC of the
    lr x 1 development chains (a flag: a chain below it is reported, never left out)."""
    v = [r["v_auc"] for r in dev.values() if r["lr_mult"] == "1"]
    m_u = float(np.median(v))
    t = 0.5 + 0.5 * (m_u - 0.5)
    return {"rule": "0.5 + 0.5 (m_U - 0.5); m_U = median selected sigma V-AUC of the lr x 1 "
                    "development U chains; a chain below it is flagged collapsed (reported; the "
                    "main analysis keeps it)",
            "chains": len(v), "m_u": m_u, "threshold": t,
            "by_dataset": collapse_by_dataset(t),
            "aux_rule": f"ucf and xdv: stage 1's {AUX_COLLAPSE}"}


def collapse_by_dataset(threshold: float) -> dict[str, float]:
    """The flag thresholds: the stage-2 value for ubn, stage 1's 0.60 for ucf and xdv."""
    return {d: float(threshold) if d == CONFIRMATION else AUX_COLLAPSE
            for d in ("ubn", "ucf", "xdv")}


def threshold_of(dataset: str, inputs: Mapping) -> float:
    """The collapse flag threshold of ``dataset`` (SystemExit unless the inputs record's
    ``collapse.by_dataset`` is ``collapse_by_dataset(collapse.threshold)``)."""
    c = inputs.get("collapse") or {}
    recorded = c.get("by_dataset")
    if recorded != collapse_by_dataset(c.get("threshold")) or dataset not in recorded:
        raise SystemExit(f"inputs record collapse {c}: by_dataset must give ubn the stage-2 "
                         f"threshold, ucf and xdv {AUX_COLLAPSE}")
    return float(recorded[dataset])


def choose_multiplier(dev: Mapping[str, Mapping]) -> dict:
    """The lr multiplier of the grid with the highest mean sigma V-AUC over its six chains (ties:
    1, then 0.5, then 2; scale 1), for the ubn chains; ucf and xdv 1 (``by_dataset``)."""
    means = {m: float(np.mean([r["v_auc"] for r in dev.values() if r["lr_mult"] == m]))
             for m in LR_GRID}
    chosen = s1.first_largest(means, MULT_ORDER)
    return {"grid_lr_mult": list(LR_GRID), "scale_mult": SCALE, "chosen_lr_mult": chosen,
            "mean_sigma_vauc": means, "by_dataset": by_dataset(chosen)}


def stage1_choice() -> dict:
    """Stage 1's candidate and final control from its committed result (stop unless they are
    k_rg_t0.25 and trainonly)."""
    rec = _json(STAGE1_RESULT)
    cand = ((rec.get("stage_a") or {}).get("decision") or {}).get("candidate")
    final = (rec.get("stage_b") or {}).get("final_control")
    if cand != CANDIDATE or final != FINAL_CONTROL:
        raise SystemExit(f"{STAGE1_RESULT.name}: candidate {cand} / final_control {final}, the "
                         f"plan's arms need {CANDIDATE} / {FINAL_CONTROL}; stop and report")
    return {"file": _rel(STAGE1_RESULT), "sha256": s0.sha256_file(STAGE1_RESULT),
            "candidate": cand, "final_control": final}


def tracks_record(path: Path, listed: Mapping) -> dict:
    """The stage-2 tracks check (``prep.ubnormal tracks-check``): the sha256 of every official
    abnormal test video's tracks file (pinned: the sealed read compares each before parsing any)
    and the validation entries as checked (that record holds no sha256 for them); SystemExit
    unless both name sets are the official lists' and every entry is complete and ok."""
    rec = _json(path)
    test, val = rec.get("test") or {}, rec.get("validation") or {}
    problems = []
    for what, got, want in (("test", test, listed[("test", 1)]["names"]),
                            ("validation", val, listed[("val", 1)]["names"])):
        if set(got) != set(want):
            problems.append(f"{what} names differ from the official abnormal list "
                            f"({len(got)} for {len(want)})")
    problems += [f"test {n}: no file or sha256" for n, e in sorted(test.items())
                 if not (e.get("exists") and HEX64.fullmatch(str(e.get("sha256"))))]
    problems += [f"validation {n}: not ok" for n, e in sorted(val.items())
                 if e.get("ok") is not True]
    if (rec.get("summary") or {}).get("ok") is not True:
        problems.append("summary not ok")
    if problems:
        raise SystemExit(f"{path}: {problems}; stop and report")
    return {"file": Path(path).name, "sha256": s0.sha256_file(path), "check": rec.get("check"),
            "utc": rec.get("utc"), "summary": rec.get("summary"),
            "test": {n: test[n]["sha256"] for n in sorted(test)},
            "validation": {n: val[n] for n in sorted(val)}}


def pins_record(u_dev: Path, zip_sha256: str, lists: Path, tracks_check: Path) -> dict:
    """The pre-freeze inputs record (module docstring, ``pins``)."""
    if not HEX64.fullmatch(str(zip_sha256)):
        raise SystemExit(f"--zip_sha256 {zip_sha256!r}: 64 lowercase hex digits")
    try:
        listed = ub.read_lists(lists, strict=True)
    except ub.UBnormalError as exc:
        raise SystemExit(f"official lists: {exc}") from exc
    tracks = tracks_record(tracks_check, listed)
    dev = dev_chains(u_dev)
    envs = list({json.dumps(s1._env(r["environment"]), sort_keys=True): s1._env(
        r["environment"]) for r in dev.values()}.values())
    return {"stage": "E024 stage 3 inputs (stage3/PLAN.md 관문 (p))", "files": input_files(),
            "lists": {v["file"]: v["sha256"] for v in listed.values()}, "tracks": tracks,
            "licence": {"licence": ub.LICENCE, "sha256": dict(ub.LICENCE_SHA256),
                        "upstream": dict(ub.UPSTREAM)},
            "archive": {"file": "UBnormal.zip", "sha256": zip_sha256},
            "stage1": stage1_choice(), "u_dev": dev, "collapse": collapse_threshold(dev),
            "multiplier": choose_multiplier(dev),
            "environment": {"keys": list(s1.ENV_KEYS), "distinct": envs,
                            "shared": len(envs) == 1},
            "code": code_version()}


def write_pins(u_dev: Path, zip_sha256: str, lists: Path, tracks_check: Path,
               out: Path | None = None) -> Path:
    """Gate (p), once, before the freeze."""
    out = Path(out) if out is not None else INPUTS
    if out.exists() or (STAGE3_DIR / SEAL_FILE).exists():
        raise SystemExit(f"{out.name} or the seal exists: the inputs record is written once, "
                         "before the freeze")
    write_json(out, pins_record(u_dev, zip_sha256, lists, tracks_check))
    return out


# --- seal -----------------------------------------------------------------------------------

def _committed(path: Path) -> bool:
    rel = _rel(path)
    return s0._git("cat-file", "-e", f"HEAD:{rel}").returncode == 0 \
        and not s0._git("status", "--porcelain", "--", rel).stdout.strip()


def _clean() -> dict:
    code = code_version()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed; code version {code}")
    return code


def freeze() -> str:
    """stage3/PLAN.sha256 = sha256 of PLAN.md before its change log (clean tree; the plan, the
    inputs record and the code files committed and unmodified; commit the seal file alone: that
    commit is the stage-3 freeze commit)."""
    _clean()
    for f in (STAGE3_DIR / "PLAN.md", INPUTS, *CODE_FILES):
        if not _committed(f):
            raise SystemExit(f"{_rel(f)} must be committed and unmodified at HEAD before the "
                             "freeze")
    seal = STAGE3_DIR / SEAL_FILE
    if seal.exists():
        raise SystemExit(f"{seal} exists: the plan is frozen once")
    h = plan_sha256(STAGE3_DIR / "PLAN.md")
    seal.write_text(f"{h}  PLAN.md (before {CHANGELOG})\n", encoding="utf-8", newline="\n")
    return h


def require_frozen() -> dict:
    """Clean tree; stage3/PLAN.sha256 added once, alone, by the freeze commit; the plan has the
    sealed hash; the inputs record as committed in the freeze commit's parent; ``src/``
    unchanged between the freeze commit and HEAD."""
    code = _clean()
    seal = dt.plan_seal(STAGE3_DIR)
    freeze_commit = freeze_commit_of(REPO_ROOT, STAGE3_DIR)
    seal_rel, inputs_rel = _rel(STAGE3_DIR / SEAL_FILE), _rel(INPUTS)
    changed = s0._git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root",
                      freeze_commit or "HEAD").stdout.split()
    if freeze_commit is None or changed != [seal_rel]:
        raise SystemExit(f"the stage-3 freeze commit must add {seal_rel} alone "
                         f"({freeze_commit}, {changed})")
    at_freeze = s0._git("show", f"{freeze_commit}:{seal_rel}").stdout.split()
    if not at_freeze or at_freeze[0] != seal["sha256"]:
        raise SystemExit(f"the plan hash {seal['sha256']} is not the one sealed by "
                         f"{freeze_commit}")
    parent = s0._git("show", f"{freeze_commit}^:{inputs_rel}")
    if parent.returncode != 0 or json.loads(parent.stdout) != _json(INPUTS):
        raise SystemExit(f"{inputs_rel} differs from the one committed before the freeze")
    if s0._git("diff", "--quiet", freeze_commit, "HEAD", "--", SRC).returncode != 0:
        raise SystemExit(f"{SRC}/ changed after the freeze commit {freeze_commit} (HEAD)")
    return {"plan": seal, "freeze_commit": freeze_commit, "code": code}


# --- training (server) ----------------------------------------------------------------------

def train_config(dataset: str, pair: str, seed: int, out: Path, settings: Mapping[str, str], *,
                 n_chunk: int, iterations: int | None, paths_file: str | None,
                 device: str | None) -> tr.TrainConfig:
    """``e024_stage1.train_chain``'s TrainConfig with the lr and scale multipliers of
    ``settings``."""
    return tr.TrainConfig(method=uk.NAME, unit=s1.CHAIN_CONFIG["unit"], dataset=dataset,
                          pair=pair, seed=int(seed), out_dir=str(out),
                          lr_mult=settings["lr_mult"], scale_mult=settings["scale_mult"],
                          n_chunk=int(n_chunk), deterministic=True, iterations=iterations,
                          paths=paths_file, device=device)


def stage_record(dataset: str, pair: str, seed: int, arm: str, settings: Mapping) -> dict:
    """stage3.json of a P / C6 / gate_k0 chain: the stage, the arm, the cell and the
    multipliers (stage 1's k_config.json beside it is unchanged)."""
    return {"stage": STAGE, "plan": "experiments/E024_localisation_method/stage3/PLAN.md",
            "arm": arm, "dataset": dataset, "pair": pair, "seed": int(seed),
            "multipliers": dict(settings), "code": code_version()}


def train_arm(dataset: str, pair: str, seed: int, arm: str, out_dir: str | Path, *,
              lr_mult: str | None = None, scale_mult: str | None = None,
              n_chunk: int = common.N_CHUNK_DEFAULT, paths_file: str | None = None,
              device: str | None = None, method: uk.URDMUK | None = None, paths=None,
              split=None, iterations: int | None = None, log=print) -> dict:
    """One chain of arm p, c6 or k0 (gate_k0: ubn xcb_xcl seed 3 only) in <runs>/<ds>_<pair>_s
    <seed>/<folder>, with the dataset's multipliers (``settings_of``: ubn lr m and scale 1, ucf
    and xdv 1 and 1; --lr_mult / --scale_mult, when given, must be them):
    ``e024_stage1.train_chain`` when both are 1, otherwise its steps (test paths refused, the
    folder of (mode, rho, tau), a completed chain refused, k_config.json first) with lr x m.
    stage3.json (``stage_record``) is written first, beside stage 1's k_config.json.
    ``method``, ``paths``, ``split`` and ``iterations`` are the Python API's (tests)."""
    if int(seed) not in SEEDS:
        raise SystemExit(f"seed {seed}: stage 3 trains the new seeds {SEEDS} only")
    if dataset not in DATASETS or pair not in PAIRS or arm not in TRAIN:
        raise SystemExit(f"{dataset} / {pair} / arm {arm}: not a stage-3 chain")
    out, cell = Path(out_dir), s0.chain_name(dataset, pair, seed)
    s0.refuse_test_paths(out)
    if out.name != TRAIN_FOLDER[arm] or out.parent.name != cell:
        raise SystemExit(f"{out}: arm {arm} of {cell} is <runs>/{cell}/{TRAIN_FOLDER[arm]}")
    if arm == "k0" and (dataset, pair, int(seed)) != GATE_CELL:
        raise SystemExit(f"gate_k0 is the chain of {s0.chain_name(*GATE_CELL)} only")
    settings = settings_of(dataset, read_inputs())
    for key, given in (("lr_mult", lr_mult), ("scale_mult", scale_mult)):
        if given is not None and not _same_mult(given, settings[key]):
            raise SystemExit(f"--{key} {given}: the inputs record gives {dataset} {key} "
                             f"{settings[key]}")
    mode, rho, tau = TRAIN[arm]
    kcfg = uk.KConfig.of(mode, rho, tau)
    method = method or uk.URDMUK(kcfg, seed)
    if method.kcfg != kcfg or method.seed != int(seed):
        raise SystemExit(f"method {method.kcfg} seed {method.seed} for {kcfg} seed {seed}")
    if (out / tr.FILES["record"]).exists():
        raise SystemExit(f"{out / tr.FILES['record']} exists: a completed chain is not rerun")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / STAGE_FILE, stage_record(dataset, pair, seed, arm, settings))
    if settings == {"lr_mult": "1", "scale_mult": "1"}:
        return s1.train_chain(dataset, pair, seed, rho, tau, mode, out, n_chunk=n_chunk,
                              paths_file=paths_file, device=device, method=method, paths=paths,
                              split=split, iterations=iterations, log=log)
    write_json(out / s1.K_CONFIG, s1.k_config_record(kcfg, {"mode": mode, "rho": rho,
                                                            "tau": tau}, seed))
    cfg = train_config(dataset, pair, seed, out, settings, n_chunk=n_chunk,
                       iterations=iterations, paths_file=paths_file, device=device)
    return tr.run(cfg, method=method, paths=paths, split=split, log=log)


# --- gates after training (server) ----------------------------------------------------------

def gate_k0(runs: Path, u_runs: Path | None = None) -> dict:
    """Gate (k0): the mode-off chain against the U chain of its cell (ubn xcb_xcl seed 3; U in
    ``u_runs``, default ``runs``): val_scores z (every point) and selection, bit for bit."""
    s0.refuse_test_paths(runs, *(() if u_runs is None else (u_runs,)))
    cell = s0.chain_name(*GATE_CELL)
    chain = Path(runs) / cell / s1.GATE_K0
    rec = s1.check_name(chain)
    u = Path(u_runs if u_runs is not None else runs) / cell / ARMS["u"]
    return {**s0.gate_b(u, chain), "chain": f"{cell}/{s1.GATE_K0}", "kconfig": rec["kconfig"]}


def _gate_h(rec: Mapping, u: Mapping) -> dict:
    if not u:
        return {"ok": False, "differ": "no U chain in the cell"}
    got, ref = rec.get("inputs") or {}, u.get("inputs") or {}
    bad = [k for k in s0.INPUT_KEYS if not got.get(k) or got.get(k) != ref.get(k)]
    return {"ok": not bad, "differ": bad}


def _gate_e(path: Path, arm: str, rec: Mapping, u: Mapping) -> dict:
    got, ref = rec.get("parameters"), (u or {}).get("parameters")
    if arm == "u":
        return {"ok": got is not None, "parameters": got}
    try:
        kc = s1.check_name(path)
    except SystemExit as exc:
        return {"ok": False, "reason": str(exc)}
    diff = None if got is None or ref is None else int(got) - int(ref)
    want = 0 if kc["kconfig"]["mode"] == "off" else uk.G_PARAMETERS
    return {"ok": diff == want and kc.get("g_parameters") == want, "parameters": got,
            "difference": diff, "expected": want}


def config_problems(rec: Mapping, kc: Mapping | None, cell: tuple[str, str, int], arm: str,
                    settings: Mapping[str, str]) -> dict:
    """What of a chain's configuration differs from its registration: CHAIN_CONFIG, the arm's
    method, the cell, the lr and scale multipliers of ``settings`` (``settings_of``; compared by
    value), and k_config.json's seed (empty: none)."""
    d, p, s = cell
    cfg = rec.get("config") or {}
    want = {**CHAIN_CONFIG, "method": METHOD[arm], "dataset": d, "pair": p, "seed": s}
    wrong = {k: cfg.get(k) for k, v in want.items() if cfg.get(k) != v}
    wrong |= {k: cfg.get(k) for k in ("lr_mult", "scale_mult")
              if not _same_mult(cfg.get(k), settings[k])}
    if kc is not None and kc.get("seed") != s:
        wrong["k_config seed"] = kc.get("seed")
    return wrong


def stage_problems(path: Path, cell: tuple, arm: str, settings: Mapping) -> dict:
    """What of a P / C6 / gate_k0 chain's stage3.json differs from ``stage_record`` (code
    aside; empty: none)."""
    f = Path(path) / STAGE_FILE
    got = _json(f) if f.is_file() else {}
    want = {k: v for k, v in stage_record(*cell, arm, settings).items() if k != "code"}
    return {k: got.get(k) for k, v in want.items() if got.get(k) != v}


def _gate_n(path: Path, cell: tuple, arm: str, rec: Mapping, settings: Mapping) -> dict:
    kc = None
    if arm != "u":
        try:
            kc = s1.check_name(path)
        except SystemExit as exc:
            return {"ok": False, "reason": str(exc)}
    wrong = config_problems(rec, kc, cell, arm, settings)
    if kc is None:
        return {"ok": not wrong, "differ": wrong}
    stage = stage_problems(path, cell, arm, settings)
    return {"ok": not wrong and not stage and kc["kconfig"] == asdict(uk.KConfig.of(*TRAIN[arm])),
            "differ": wrong, "stage3": stage, "kconfig": kc["kconfig"], "seed": kc.get("seed")}


def _gate_v(recs: Mapping[str, Mapping], inputs: Mapping) -> tuple[dict, dict]:
    envs = {k: s1._env(r.get("environment")) for k, r in recs.items()}
    distinct = list({json.dumps(e, sort_keys=True): e for e in envs.values()}.values())
    one = len(distinct) == 1 and distinct[0].get("torch") is not None
    pinned = (inputs.get("environment") or {}).get("distinct") or []
    return ({k: {"ok": one} for k in envs},
            {"environment": distinct[0] if one else None, "distinct": distinct,
             "equals_pinned": one and distinct[0] in pinned})


def gate_records(runs: Path) -> dict[str, dict]:
    """Gates (h), (e), (n), (v) over every registered chain (module docstring; (n) also the
    configuration against the inputs record, ``config_problems``, and stage3.json); every entry
    holds the chain's selection.json sha256; a missing chain (no selection.json) or an
    unregistered folder makes every gate fail."""
    s0.refuse_test_paths(runs)
    inputs = read_inputs()
    chains = expected_chains(runs)
    recs = {k: s1._record(f) for k, _, _, f in chains}
    missing, other = [k for k, r in recs.items() if not r], unregistered(runs)
    present = [(k, c, a, f) for k, c, a, f in chains if recs[k]]
    selection = {k: s0.sha256_file(f / tr.FILES["record"]) for k, _, _, f in present}
    u_of = {c: recs.get(chain_key(*c, ARMS["u"])) or {} for _, c, _, _ in chains}
    v, shared = _gate_v({k: recs[k] for k, *_ in present}, inputs)

    def record(entries: Mapping, **extra) -> dict:
        ok = not missing and not other and bool(entries) and all(x["ok"] for x in
                                                                 entries.values())
        return {"ok": ok, "chains": {k: {**x, "selection_sha256": selection[k]}
                                     for k, x in entries.items()},
                "missing": missing, "unregistered": other, **extra}

    return {"gate_h": record({k: _gate_h(recs[k], u_of[c]) for k, c, _, _ in present}),
            "gate_e": record({k: _gate_e(f, a, recs[k], u_of[c]) for k, c, a, f in present}),
            "gate_n": record({k: _gate_n(f, c, a, recs[k], settings_of(c[0], inputs))
                              for k, c, a, f in present}),
            "gate_v": record(v, **shared)}


def write_gates(runs: Path) -> dict[str, bool]:
    """<runs>/gate_h.json, gate_e.json, gate_n.json, gate_v.json; returns their ok flags."""
    recs = gate_records(runs)
    for name, rec in recs.items():
        write_json(Path(runs) / f"{name}.json", {"check": f"E024 stage 3 {name} (stage3/PLAN.md "
                                                 "관문)", **rec, "code": code_version()})
    return {name: rec["ok"] for name, rec in recs.items()}


# --- chains under the scoring and analysis preconditions --------------------------------------

def check_chain(path: Path, cell: tuple[str, str, int], arm: str, settings: Mapping[str, str],
                commit: str | None) -> dict:
    """One scored chain: its configuration (``config_problems`` with the dataset's multipliers
    ``settings``), gate (n) (P, C6), the code commit of selection.json and k_config.json =
    ``commit``, a sigma selection without numerical failure (a missing chain or a numerical
    failure stops: PLAN.md "붕괴와 실패"). Returns what scoring and analysis use."""
    rec = s1._record(path)
    if not rec:
        raise SystemExit(f"{path}: no selection.json (a missing chain stops; rerun it once)")
    kc = None if arm == "u" else s1.check_name(path)
    wrong = config_problems(rec, kc, cell, arm, settings)
    if wrong:
        raise SystemExit(f"{path}: {wrong} differ from the registration (lr_mult "
                         f"{settings['lr_mult']}, scale_mult {settings['scale_mult']})")
    codes = {tr.FILES["record"]: (rec.get("code") or {}).get("commit"),
             **({} if kc is None else {s1.K_CONFIG: (kc.get("code") or {}).get("commit")})}
    if commit is not None and any(c != commit for c in codes.values()):
        raise SystemExit(f"{path}: code commits {codes} are not {commit}")
    sig = (rec.get("selection") or {}).get("sigma")
    nonfinite = int(rec.get("nonfinite_losses") or 0)
    if rec.get("failed") or not sig or nonfinite:
        why = rec.get("failed") or (f"{nonfinite} non-finite losses" if nonfinite
                                    else "no sigma selection")
        raise SystemExit(f"{path}: numerical failure ({why}): stop and report (PLAN.md 붕괴와 "
                         "실패)")
    return {"path": Path(path), "cell": cell, "arm": arm, "record": rec, "kconfig": kc,
            "point": int(sig["point"]), "v_auc": float(sig["v_auc"]),
            "checkpoint_sha256": rec["checkpoints"]["sigma"]["sha256"],
            "parameters": rec.get("parameters"), "environment": rec.get("environment")}


def scored_chains(runs: Path, datasets: Sequence[str], inputs: Mapping, commit: str | None
                  ) -> dict[str, dict]:
    """``check_chain`` of every arm of every cell of ``datasets`` with the dataset's multipliers
    (dataset, pair, seed, arm order)."""
    return {chain_key(d, p, s, ARMS[a]): check_chain(Path(runs) / s0.chain_name(d, p, s) /
                                                     ARMS[a], (d, p, s), a,
                                                     settings_of(d, inputs), commit)
            for d in datasets for p in PAIRS for s in SEEDS for a in ARMS}


def selection_hashes(runs: Path, keys: Sequence[str]) -> dict[str, str]:
    """The current selection.json sha256 of the chains ``keys`` (a missing one stops)."""
    out = {}
    for k in keys:
        f = Path(runs) / k / tr.FILES["record"]
        if not f.is_file():
            raise SystemExit(f"{f}: missing (a missing chain stops)")
        out[k] = s0.sha256_file(f)
    return out


def require_gates(runs: Path, names: Sequence[str], keys: Sequence[str],
                  selection: Mapping[str, str]) -> dict:
    """The gate records in the runs root (SystemExit otherwise): each ok; every one but
    gate_k0.json with a "chains" list holding every key as passed, recorded with the chain's
    current selection.json sha256 (``selection``); their sha256."""
    out = {}
    for n in names:
        f = Path(runs) / n
        rec = _json(f) if f.is_file() else {}
        if rec.get("ok") is not True:
            raise SystemExit(f"{f}: missing or not ok; stop and report")
        if n != GATE_FILES[0]:
            listed = rec.get("chains")
            if not isinstance(listed, Mapping):
                raise SystemExit(f"{n}: no chains list; stop and report")
            absent = [k for k in keys if (listed.get(k) or {}).get("ok") is not True]
            stale = [k for k in keys if k not in absent and (listed.get(k) or {}).get(
                "selection_sha256") != selection[k]]
            if absent or stale:
                raise SystemExit(f"{n}: no passed entry for {absent}; entries of another "
                                 f"selection.json for {stale}")
        out[n] = {"ok": True, "sha256": s0.sha256_file(f)}
    return out


def chain_model(chk: Mapping, dims: Sequence[int], device: torch.device):
    """The chain's network with its sigma checkpoint (point and sha256 as selection.json), fp32,
    eval mode: UR-DMU (U) or ``URDMUK(KConfig, seed)`` (P, C6), as stage 1's rescore."""
    rec = chk["record"]
    cfg = rec["config"]
    ck, sha = s1._load_sigma(chk["path"], rec)
    method = common.method_on(METHOD["u"], cfg["dataset"]) if chk["arm"] == "u" else \
        uk.URDMUK(uk.KConfig(**chk["kconfig"]["kconfig"]), int(cfg["seed"]))
    model = method.build(int(sum(dims)), unit=int(cfg["unit"]),
                         input_ln=common.input_ln(cfg["pair"]), scale=dict(cfg["scale"]))
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device=device, dtype=torch.float32)
    method.set_mode(model, False)
    return method, model, sha


def chain_logits(chk: Mapping, method, model, xs: Sequence[torch.Tensor], device: torch.device,
                 k_in_logit: bool | None) -> list[np.ndarray]:
    """Snippet logits of every video (``common.score_video``, the chain's unit and N_chunk);
    ``k_in_logit`` sets the K test branch (None for U)."""
    if k_in_logit is not None:
        model.k_in_logit = k_in_logit
    cfg = chk["record"]["config"]
    return [common.score_video(method, model, x, unit=int(cfg["unit"]),
                               n_chunk=int(cfg["n_chunk"]), device=device) for x in xs]


def _groups(chains: Mapping[str, Mapping]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for k, c in chains.items():
        out.setdefault(c["cell"][:2], []).append(k)
    return out


def _variant(chains: Mapping[str, Mapping], keys: Sequence[str], what: str) -> str:
    return _one({chains[k]["record"]["config"]["variant"] for k in keys}, f"{what} variants")


def _gate_s_chain(chk: Mapping, val, device: torch.device, split_sha: str) -> dict:
    rec = chk["record"]
    f = chk["path"] / tr.FILES["val"]
    if rec["config"].get("split_sha256") != split_sha:
        return {"ok": False, "reason": "the split differs from the chain's"}
    if s0.sha256_file(f) != rec.get("val_scores_sha256"):
        return {"ok": False, "reason": "val_scores.npz is not selection.json's"}
    with np.load(f) as z:
        stored, idx = z["z"][chk["point"] - 1], [int(v) for v in z["val_indices"]]
    if idx != list(val.indices):
        return {"ok": False, "reason": "validation videos differ from val_scores.npz"}
    method, model, _ = chain_model(chk, val.dims, device)
    v = np.concatenate(chain_logits(chk, method, model, val.x, device, SCORED_K[chk["arm"]]))
    same = v.dtype == stored.dtype and v.shape == stored.shape and v.tobytes() == \
        stored.tobytes()
    return {"ok": same, "compared": COMPARED[chk["arm"]],
            "differing_snippets": int((v != stored).sum()) if v.shape == stored.shape else None}


def gate_s(runs: Path, chains: Mapping[str, Mapping], paths, device: torch.device,
           splits: Mapping | None = None) -> dict:
    """Gate (s), before any test read: every chain's sigma checkpoint re-scores its validation
    videos (``train.validation_videos``) to the logits it was selected with (val_scores.npz
    z[point - 1]: s for P, z0 for C6, z for U), bit for bit. Writes <runs>/gate_s.json;
    SystemExit when a chain fails."""
    out = {}
    for (d, p), keys in _groups(chains).items():
        split = (splits or {}).get(d) or load_split(split_file(d))
        val = tr.validation_videos(paths, d, p, _variant(chains, keys, f"{d}/{p}"), split)
        for k in keys:
            out[k] = {**_gate_s_chain(chains[k], val, device, split.sha256),
                      "selection_sha256": s0.sha256_file(chains[k]["path"] / tr.FILES["record"])}
    rec = {"check": "E024 stage 3 gate (s) (stage3/PLAN.md 관문; validation only)",
           "ok": bool(out) and all(v["ok"] for v in out.values()), "chains": out,
           "utc": utc_now(), "code": code_version(), "environment": tr._environment(device)}
    write_json(Path(runs) / GATE_S, rec)
    if not rec["ok"]:
        bad = [k for k, v in out.items() if not v["ok"]]
        raise SystemExit(f"gate (s) failed for {bad}: stop and report (no test read)")
    return rec


# --- the sealed test read (server) ----------------------------------------------------------

def _log(access_log: Path, dataset: str, rows: Sequence[Mapping]) -> None:
    """Rows appended to the access log (``wvad.extract.dense.log_access``) before the read."""
    from wvad.extract.dense import log_access

    log_access(access_log, [{"utc": utc_now(), "command": COMMAND, "dataset": dataset,
                             "index": "", **r} for r in rows])


def manifest_view(d: str, manifest: Path) -> dict:
    """Names, video labels and frame counts of the test videos without decoding an interval:
    the ubn test manifest stores none (``intervals_external``); ucf and xdv through the
    restricted reader (``dense_test.read_manifest``)."""
    if d == CONFIRMATION:
        m = load_eval_manifest(manifest)
        if not m.intervals_external:
            raise SystemExit(f"{manifest}: the ubn test manifest must store no intervals")
    else:
        m = dt.read_manifest(d, manifest)
    return {"names": tuple(m.names), "labels": np.asarray(m.labels, np.int64),
            "total_frames": np.asarray(m.total_frames, np.int64),
            "sha256": s0.sha256_file(manifest), "names_sha256": _names_sha256(m.names)}


def check_tracks(view: Mapping, paths, pinned: Mapping[str, str], access_log: Path,
                 target: Path) -> dict:
    """Before any tracks file is parsed: every abnormal test video's tracks file has the sha256
    pinned in the inputs record (read as bytes after one access-log row each); the pinned names
    are the manifest's abnormal videos. A mismatch is written to <out>/ubn/gate_tracks.json and
    stops (SystemExit)."""
    names = [n for n, y in zip(view["names"], view["labels"], strict=True) if int(y) == 1]
    _log(access_log, CONFIRMATION, [{"name": n, "path": ub.tracks_rel_path(n),
                                     "purpose": "test labels: tracks sha256 against the inputs "
                                                "record (bytes, not parsed)"} for n in names])
    root = paths.video_root()
    found = {n: s0.sha256_file(root / ub.tracks_rel_path(n))
             if (root / ub.tracks_rel_path(n)).is_file() else None for n in names}
    rec = {"ok": False, "checked": len(names),
           "mismatch": {n: {"pinned": pinned.get(n), "found": found[n]} for n in names
                        if found[n] != pinned.get(n)},
           "pinned_not_in_manifest": sorted(set(pinned) - set(names))}
    rec["ok"] = not rec["mismatch"] and not rec["pinned_not_in_manifest"]
    if not rec["ok"]:
        write_json(Path(target) / "gate_tracks.json", rec)
        raise SystemExit(f"tracks files differ from the inputs record: {rec}; stop and report "
                         "(no tracks file parsed)")
    return rec


def frame_labels_of(d: str, view: Mapping, manifest: Path, paths, access_log: Path,
                    pinned: Mapping[str, str], target: Path
                    ) -> tuple[list[np.ndarray | None], dict]:
    """The test frame labels (after every score file is written): ubn from the tracks files
    (``check_tracks`` first, then ``ubnormal.test_frame_gts``, which logs every path before
    reading; a failed video gets None and is named; the parsed files must still have the pinned
    sha256), ucf and xdv from the manifest intervals (one log row first)."""
    totals = [int(t) for t in view["total_frames"]]
    if d == CONFIRMATION:
        checked = check_tracks(view, paths, pinned, access_log, target)
        gts, rec = ub.test_frame_gts(paths.video_root(), view["names"],
                                     [int(y) for y in view["labels"]], totals, access_log,
                                     command=COMMAND)
        moved = sorted(n for n, sha in rec["sha256"].items() if sha != pinned.get(n))
        if moved:
            write_json(Path(target) / "gate_tracks.json", {**checked, "ok": False,
                                                           "changed_while_read": moved})
            raise SystemExit(f"tracks files {moved} changed between the check and the parse")
        return gts, {"source": "tracks files", "pinned_sha256": checked, **rec}
    _log(access_log, d, [{"name": manifest.name, "path": str(manifest),
                          "purpose": "test labels (manifest intervals)"}])
    m = load_eval_manifest(manifest)
    if tuple(m.names) != view["names"] or list(m.labels) != view["labels"].tolist() or \
            list(m.total_frames) != totals:
        raise SystemExit(f"{manifest}: the manifest differs from its restricted read")
    return m.frame_gts(), {"source": "manifest intervals", "read": len(m), "failed": {}}


def label_arrays(gts: Sequence[np.ndarray | None], totals: Sequence[int]) -> dict:
    """Per snippet the positive frame count of its support (``pos``; 0 for a video without
    labels, ``valid`` False), the snippet counts and the frame counts."""
    pos, valid = [], []
    for g, t in zip(gts, totals, strict=True):
        sup = grid.snippet_support(int(t))
        if g is None:
            pos.append(np.zeros(len(sup), np.int64))
            valid.append(False)
            continue
        g = np.asarray(g)
        if g.shape != (int(t),):
            raise ValueError(f"{g.shape} frame labels for T = {t}")
        pos.append(np.add.reduceat(g.astype(np.int64), sup[:, 0]))
        valid.append(True)
    return {"pos": np.concatenate(pos), "valid": np.asarray(valid, bool),
            "counts": np.array([grid.n_snippets(int(t)) for t in totals], np.int64),
            "total_frames": np.asarray(totals, np.int64)}


def video_frames(z: np.ndarray, pos: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray]:
    """(T,) frame scores (each snippet's over its support) and frame labels (a snippet's
    positive count first): the same (score, label) pairs as the true frames, hence the same
    AUC and AP (sklearn's ties are by score)."""
    sup = grid.snippet_support(int(total))
    n = sup[:, 1] - sup[:, 0]
    offset = np.arange(int(total)) - np.repeat(sup[:, 0], n)
    return np.repeat(np.asarray(z), n), (offset < np.repeat(np.asarray(pos), n)).astype(
        np.float32)


def l_videos(lab: Mapping, labels: Sequence[int]) -> list[int]:
    """The L videos: anomalous, labelled, both classes among their frames (by the labels only,
    the same for every chain)."""
    rows = split_rows(np.asarray(lab["pos"]), lab["counts"])
    return [v for v, (y, ok, p, t) in enumerate(zip(labels, lab["valid"], rows,
                                                    lab["total_frames"], strict=True))
            if int(y) == 1 and ok and 0 < int(p.sum()) < int(t)]


def chain_metrics(fs: Mapping[int, np.ndarray], gs: Mapping[int, np.ndarray],
                  labels: Sequence[int], metric: str, lv: Sequence[int]) -> dict:
    """One chain on its labelled test videos (``fs``, ``gs``: video -> frame scores, labels): L
    (``frames.within_video``) with its values on the L videos ``lv`` (in that order), the pooled
    frame AUC and AP (``frames.pooled_frame_metrics``, videos >= 32 frames), AUC_A (the pooled
    frame AUC of the anomalous videos; None when their frames hold one class)."""
    vids = sorted(fs)
    lw = fr.within_video([fs[v] for v in vids], [gs[v] for v in vids],
                         [int(labels[v]) for v in vids], metric)
    per = {vids[i]: x for i, x in lw["per_video"].items()}
    if sorted(per) != sorted(lv):
        raise ValueError("the L videos differ from the labels' L videos")
    pooled = fr.pooled_frame_metrics([fs[v] for v in vids], [gs[v] for v in vids])
    anom = [v for v in vids if int(labels[v]) == 1 and len(gs[v]) >= DEFAULT_MIN_FRAMES]
    ga = np.concatenate([gs[v] for v in anom]) if anom else np.zeros(0)
    auc_a = fr.pooled_frame_metrics([fs[v] for v in anom], [gs[v] for v in anom])[
        "frame_auc"] if ga.size and ga.min() != ga.max() else None
    return {"L": lw["value"], "n_videos": lw["n_videos"], "n_excluded": lw["n_excluded"],
            "per_video": [per[v] for v in lv], "frame_auc": pooled["frame_auc"],
            "frame_ap": pooled["frame_ap"], "pooled_videos": pooled["n_videos"],
            "auc_anomalous": auc_a}


def score_structure(z: Mapping[str, Sequence[np.ndarray]], view: Mapping) -> dict:
    """Gate (z): one logit per snippet (ceil(T / 16)) of every video, every logit finite."""
    want = [grid.n_snippets(int(t)) for t in view["total_frames"]]
    out = {k: {"counts_ok": [len(x) for x in zl] == want,
               "nonfinite": int(sum(int((~np.isfinite(x)).sum()) for x in zl))}
           for k, zl in z.items()}
    return {"ok": all(v["counts_ok"] and not v["nonfinite"] for v in out.values()),
            "logits": out}


def write_scores(f: Path, z: Mapping[str, Sequence[np.ndarray]], view: Mapping, key: str,
                 chk: Mapping, ck_sha: str) -> None:
    """<out>/<ds>/<pair>_s<seed>/<arm>.npz: the snippet logits (z; P also z0) concatenated in
    manifest order, counts, frame counts, video labels, the names' sha256, the point, the
    checkpoint sha256 (written whole, then renamed)."""
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.stem}.part.npz")
    np.savez(tmp, **{k: np.concatenate(v).astype(np.float32) for k, v in z.items()},
             counts=np.array([len(x) for x in z["z"]], np.int64),
             total_frames=view["total_frames"], labels=view["labels"],
             names_sha256=np.str_(view["names_sha256"]), point=np.int64(chk["point"]),
             checkpoint_sha256=np.str_(ck_sha), chain=np.str_(key), arm=np.str_(chk["arm"]))
    tmp.replace(f)


def score_file(out: Path, d: str, p: str, s: int, arm: str) -> Path:
    return Path(out) / d / f"{p}_s{s}" / f"{arm}.npz"


def _test_inputs(d: str, p: str, paths, view: Mapping, access_log: Path, variant: str
                 ) -> list[torch.Tensor]:
    """The pair's dense test store (logged first; bound to the manifest by the names sha256 of
    its meta.json, the counts and the labels; every video extracted) through the chains'
    InputPipeline (``train.input_pipeline``: the training folders' scale maps)."""
    folders = dt.store_folders(paths, d, p)
    _log(access_log, d, [{"name": f.name, "path": str(f),
                          "purpose": "test features (dense snippet store)"} for f in folders])
    for f in folders:
        meta = store_dir(f) / META_FILE
        got = (_json(meta) if meta.is_file() else {}).get("names_sha256")
        if got != view["names_sha256"]:
            raise SystemExit(f"{meta}: names_sha256 {got} is not the test manifest's "
                             f"{view['names_sha256']}")
    dense = DenseSet(folders)
    n = len(view["names"])
    if len(dense) != n or not np.array_equal(dense.total_frames, view["total_frames"]) or \
            not np.array_equal(dense.labels, view["labels"]):
        raise SystemExit(f"{d}/{p}: the dense test store differs from the test manifest")
    undone = [v for v in range(n) if not dense.done(v)]
    if undone:
        raise SystemExit(f"{d}/{p}: {len(undone)} test videos not extracted ({undone[:5]})")
    pipe = tr.input_pipeline(paths, d, p, variant, dense.dims)
    return [pipe.apply(dense.video(v)) for v in range(n)]


def score_pair(d: str, p: str, keys: Sequence[str], chains: Mapping[str, Mapping], out: Path,
               paths, view: Mapping, access_log: Path, device: torch.device) -> dict[str, Path]:
    """Every chain of (d, p) on the test videos (P also with k = 0); gate (z) per chain (a
    failure writes <out>/<ds>/gate_z.json and stops)."""
    xs = _test_inputs(d, p, paths, view, access_log, _variant(chains, keys, f"{d}/{p}"))
    written = {}
    for k in keys:
        chk = chains[k]
        method, model, sha = chain_model(chk, [int(xs[0].shape[1])], device)
        z = {"z": chain_logits(chk, method, model, xs, device, SCORED_K[chk["arm"]])}
        if chk["arm"] == "p":
            z["z0"] = chain_logits(chk, method, model, xs, device, False)
        f = score_file(out, d, p, chk["cell"][2], chk["arm"])
        write_scores(f, z, view, k, chk, sha)
        st = score_structure(z, view)
        if not st["ok"]:
            write_json(Path(out) / d / "gate_z.json", {"ok": False, "chain": k, **st})
            raise SystemExit(f"gate (z) failed for {k}: {st}; stop and report")
        written[k] = f
        del model
    return written


def dataset_summary(d: str, view: Mapping, gts: Sequence[np.ndarray | None], lab: Mapping,
                    label_rec: Mapping, out: Path, files: Mapping[str, Path]) -> dict:
    """summary.json of one dataset: the videos, the label record, the L videos and every
    chain's metrics (P: also I2), from the written score files and the true frame labels
    ``gts`` (the analysis recomputes them from labels.npz and must meet these values)."""
    labels = [int(y) for y in view["labels"]]
    lv = l_videos(lab, labels)
    gs = {v: np.asarray(g) for v, g in enumerate(gts) if g is not None}
    metric = fr.L_METRIC[d]
    chains = {}
    for k, f in files.items():
        z = s0._npz(f)
        entry = {"file": Path(f).relative_to(out).as_posix(), "point": int(z["point"]),
                 "checkpoint_sha256": str(z["checkpoint_sha256"])}
        for zk, name in (("z", "metrics"), ("z0", "i2")):
            if zk in z:
                zl = split_rows(z[zk], z["counts"])
                fs = {v: fr.snippet_frames(zl[v], len(g)) for v, g in gs.items()}
                entry[name] = chain_metrics(fs, gs, labels, metric, lv)
        chains[k] = entry
    return {"dataset": d, "names": list(view["names"]), "names_sha256": view["names_sha256"],
            "manifest_sha256": view["sha256"], "metric": metric, "videos": len(labels),
            "anomalous": int(sum(labels)), "labels": dict(label_rec), "l_videos": lv,
            "chains": chains,
            "note": NOT_REDISTRIBUTED}


def score_dataset(d: str, chains: Mapping[str, Mapping], out: Path, paths, access_log: Path,
                  manifest: Path, device: torch.device, pinned: Mapping[str, str]) -> dict:
    """One dataset of the sealed read: the manifest view (logged first), the scores of every
    chain, then the frame labels (ubn: the tracks files against ``pinned``), labels.npz and
    summary.json (module docstring)."""
    _log(access_log, d, [{"name": manifest.name, "path": str(manifest),
                          "purpose": MANIFEST_PURPOSE}])
    view = manifest_view(d, manifest)
    files: dict[str, Path] = {}
    for (dd, p), keys in _groups(chains).items():
        if dd == d:
            files |= score_pair(d, p, keys, chains, out, paths, view, access_log, device)
    gts, label_rec = frame_labels_of(d, view, manifest, paths, access_log, pinned,
                                     Path(out) / d)
    lab = write_labels(Path(out) / d, gts, view)
    summary = dataset_summary(d, view, gts, lab, label_rec, Path(out), files)
    write_json(Path(out) / d / SUMMARY, summary)
    return summary


def write_labels(target: Path, gts: Sequence[np.ndarray | None], view: Mapping) -> dict:
    """<out>/<ds>/labels.npz: ``label_arrays`` with the video labels and the names' sha256."""
    lab = label_arrays(gts, [int(t) for t in view["total_frames"]])
    Path(target).mkdir(parents=True, exist_ok=True)
    np.savez(Path(target) / LABELS_FILE, **lab, labels=view["labels"],
             names_sha256=np.str_(view["names_sha256"]), note=np.str_(NOT_REDISTRIBUTED))
    return lab


def file_hashes(out: Path, datasets: Sequence[str]) -> dict[str, str]:
    """sha256 of every file under <out>/<ds> (posix paths relative to <out>)."""
    return {f.relative_to(out).as_posix(): s0.sha256_file(f)
            for d in datasets for f in sorted((Path(out) / d).rglob("*")) if f.is_file()}


def command_rows(access_log: Path) -> list[dict]:
    """The access-log rows of this command (score-test), in order."""
    if not Path(access_log).is_file():
        return []
    with open(access_log, encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("command") == COMMAND]


def _is(row: Mapping, purpose: str) -> bool:
    return str(row.get("purpose") or "").startswith(purpose)


def _attempt(out: Path, access_log: Path, reason: str | None) -> dict:
    """The sealed read happens once (SystemExit otherwise): the access log lies outside <out>;
    a completed read (done.json complete in <out>, or an end row of this command in the access
    log) is never repeated; an earlier attempt (rows of this command in the access log, whatever
    <out> it used, or started.json in <out>) needs ``--infra_rerun <reason>``, and one earlier
    attempt at most (counted by the log's start rows; its files in <out> move to
    <out>/superseded/<utc>)."""
    out, log = Path(out), Path(access_log)
    if log.resolve().is_relative_to(out.resolve()):
        raise SystemExit(f"{log} lies inside {out}: the access log stays outside the scores "
                         "folder")
    rows = command_rows(log)
    done = out / DONE
    if (done.is_file() and _json(done).get("complete") is True) or any(_is(r, END) for r in rows):
        raise SystemExit("the sealed test read is complete (done.json or the access log's end "
                         "row): it is never repeated")
    starts = sum(_is(r, START) for r in rows)
    in_out = [f.name for f in (out / STARTED, done) if f.exists()]
    earlier = max(starts, 1 if rows or in_out else 0)
    if not earlier:
        if reason:
            raise SystemExit("--infra_rerun given, but no earlier attempt (access log, <out>)")
        return {"attempt": 1, "infra_rerun": None, "earlier_in_out": []}
    if not reason:
        raise SystemExit(f"an earlier attempt exists ({starts} start rows of {COMMAND} in {log}; "
                         f"{in_out} in {out}): no silent rerun: --infra_rerun <reason> "
                         "(infrastructure failures, once)")
    sup = out / SUPERSEDED
    if earlier >= 2 or (sup.is_dir() and any(sup.iterdir())):
        raise SystemExit("an earlier rerun exists; the plan allows one rerun only")
    return {"attempt": earlier + 1, "infra_rerun": str(reason), "earlier_in_out": in_out}


def _supersede(out: Path) -> str:
    """The earlier attempt's files in <out> to <out>/superseded/<utc> (returned relative)."""
    dest = Path(out) / SUPERSEDED / dtm.datetime.now(dtm.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    dest.mkdir(parents=True)
    for f in sorted(Path(out).iterdir()):
        if f.name != SUPERSEDED:
            shutil.move(str(f), str(dest / f.name))
    return dest.relative_to(out).as_posix()


def deploy_record() -> dict:
    """The seal and the deploy, before any test read (SystemExit otherwise): PLAN.sha256 holds
    the hash of PLAN.md's sealed part (``dense_test.plan_seal``, as stage 1's seal checks),
    ``code_version`` yields a commit (DEPLOYED_COMMIT on the server, git HEAD in a checkout) and a
    git checkout is clean, the inputs record exists."""
    plan = dt.plan_seal(STAGE3_DIR)
    code = code_version()
    if not code.get("commit"):
        raise SystemExit(f"no code commit (DEPLOYED_COMMIT or git HEAD): {code}")
    if code.get("source") == "git" and code.get("dirty") is not False:
        raise SystemExit(f"the git tree is not clean ({code}): the sealed read runs the frozen "
                         "code")
    read_inputs()
    return {"plan": plan, "plan_sha256_file": (STAGE3_DIR / SEAL_FILE).read_text(
        encoding="utf-8"), "code": code, "inputs_sha256": s0.sha256_file(INPUTS)}


def freeze_check(freeze: str | None = None, lineage: Path | None = None) -> dict:
    """Before any test read: E022's access device 1 (``dense_test.deployed_descent`` with the
    stage-3 plan folder: in a git checkout the commit that added stage3/PLAN.sha256, ``freeze``
    if given must be it; in a deployed copy ``freeze`` and the lineage file written by
    ``dense_test.write_lineage`` beside DEPLOYED_COMMIT), and the running commit equal to the
    freeze commit (SystemExit otherwise)."""
    lineage = Path(lineage) if lineage is not None else dt.LINEAGE_FILE
    rec = dt.deployed_descent(freeze, lineage=lineage, root=REPO_ROOT, plan_dir=STAGE3_DIR)
    if rec["deployed_commit"] != rec["freeze_commit"]:
        raise SystemExit(f"the running commit {rec['deployed_commit']} is not the stage-3 freeze "
                         f"commit {rec['freeze_commit']}: the sealed read runs at the freeze "
                         "commit (no test read)")
    head = None
    if rec["code_source"] != "git":
        head = json.loads(lineage.read_text(encoding="utf-8")).get("head")
    return {**rec, "running_equals_freeze": True,
            "lineage": None if head is None else str(lineage), "lineage_head": head}


def _preconditions(runs: Path, freeze: str | None = None, lineage: Path | None = None) -> dict:
    """Everything checked before any test read: the seal and the deploy (``deploy_record``),
    the running commit = the stage-3 freeze commit (``freeze_check``), the inputs files, every
    chain (``check_chain``: recorded at the freeze commit, the dataset's multipliers), the gates
    recomputed from the chain folders (h, e, n, v, k0: ok) and the stored gate files (ok, every
    chain passed with its current selection.json)."""
    dep = deploy_record()
    frz = freeze_check(freeze, lineage)
    if dep["code"]["commit"] != frz["freeze_commit"]:
        raise SystemExit(f"code version {dep['code']} is not the freeze commit "
                         f"{frz['freeze_commit']}")
    inputs = read_inputs()
    files = check_input_files(inputs)
    chains = scored_chains(runs, DATASETS, inputs, frz["freeze_commit"])
    keys = [*chains, chain_key(*GATE_CELL, s1.GATE_K0)]
    recomputed = {**gate_records(runs), "gate_k0": gate_k0(runs)}
    failed = [n for n, r in recomputed.items() if r.get("ok") is not True]
    if failed:
        raise SystemExit(f"gates recomputed from the chain folders fail: {failed}; stop and "
                         "report (no test read)")
    selection = selection_hashes(runs, keys)
    return {**dep, "freeze": frz, "chains_commit": frz["freeze_commit"], "input_files": files,
            "settings": {d: settings_of(d, inputs) for d in DATASETS}, "chains": chains,
            "gates": require_gates(runs, GATE_FILES, keys, selection),
            "gates_recomputed": sorted(recomputed), "selection": selection,
            "tracks": dict((inputs.get("tracks") or {}).get("test") or {})}


def score_test(runs: Path, out: Path, access_log: Path, *, freeze_commit: str | None = None,
               infra_rerun: str | None = None, paths=None, paths_file: str | None = None,
               device: str | None = None, lineage: Path | None = None,
               splits: Mapping | None = None, manifests: Mapping[str, Path] | None = None
               ) -> dict:
    """THE sealed test read (module docstring, ``score-test``), always ubn, ucf, xdv in that
    order, at the stage-3 freeze commit (``freeze_commit``: required in a deployed copy, with
    ``lineage``, by default DEPLOYED_LINEAGE.json beside DEPLOYED_COMMIT); returns done.json's
    record. ``paths``, ``splits`` and ``manifests`` are the Python API's (tests)."""
    runs, out, access_log = Path(runs), Path(out), Path(access_log)
    ds = tuple(DATASETS)
    dt.outside_repo(out, access_log)
    attempt = _attempt(out, access_log, infra_rerun)
    paths = paths if paths is not None else load_paths(paths_file)
    pre = _preconditions(runs, freeze_commit, lineage)
    if not pre["tracks"]:
        raise SystemExit("the inputs record pins no test tracks sha256 (pins --tracks_check)")
    chains = pre["chains"]
    det = common.deterministic(True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = tr._environment(dev)
    differ = [k for k, c in chains.items() if s1._env(c["environment"]) != s1._env(env)]
    if differ:
        raise SystemExit(f"environment differs from the chains' {differ[:3]}: score where they "
                         "were trained (no test read)")
    gs = gate_s(runs, chains, paths, dev, splits)
    attempt["superseded"] = _supersede(out) if attempt["earlier_in_out"] else None
    lineage = {"utc_start": utc_now(), "datasets": list(ds), **attempt,
               **{k: pre[k] for k in ("code", "freeze", "plan", "plan_sha256_file",
                                      "inputs_sha256", "chains_commit")},
               "gate_s_sha256": s0.sha256_file(runs / GATE_S), "runs": str(runs)}
    n = attempt["attempt"]
    rows0 = len(command_rows(access_log))
    _log(access_log, "", [{"name": f"attempt {n}", "path": str(out),
                           "purpose": f"{START}: attempt {n}" + (
                               f"; infra rerun: {infra_rerun}" if infra_rerun else "")}])
    write_json(out / STARTED, {"check": "E024 stage 3 sealed test read started", **lineage})
    for d in ds:
        manifest = Path((manifests or {}).get(d) or eval_manifest_file(d))
        score_dataset(d, {k: c for k, c in chains.items() if c["cell"][0] == d}, out, paths,
                      access_log, manifest, dev, pre["tracks"])
    _log(access_log, "", [{"name": f"attempt {n}", "path": str(out),
                           "purpose": f"{END}: attempt {n}"}])
    done = {"check": "E024 stage 3 sealed test read (stage3/PLAN.md 시험 접근)", "complete": True,
            **lineage, "utc_end": utc_now(), "environment": {**env, **det}, "device": str(dev),
            "multipliers": pre["settings"], "gates": pre["gates"],
            "gates_recomputed": pre["gates_recomputed"], "selection": pre["selection"],
            "gate_s": {"file": GATE_S, "ok": gs["ok"], "sha256": lineage["gate_s_sha256"]},
            "checkpoints": {k: c["checkpoint_sha256"] for k, c in chains.items()},
            "files": file_hashes(out, ds),
            "access_log": {"path": os.path.realpath(access_log),
                           "sha256": s0.sha256_file(access_log),
                           "rows": len(command_rows(access_log)) - rows0},
            "note": NOT_REDISTRIBUTED}
    write_json(out / DONE, done)
    return done


# --- statistics -----------------------------------------------------------------------------

def collapsed(v_auc: float, threshold: float) -> bool:
    return not (math.isfinite(v_auc) and v_auc >= threshold)


def l_draws(n_videos: int, n_seeds: int) -> dict[str, np.ndarray]:
    """The video x seed draw counts (rows 0..R, row 0 the sample): from
    default_rng(BOOT_SEED), the L videos, then the paired seeds (glance's counts), so the video
    block does not depend on the number of seeds."""
    rng = np.random.default_rng(BOOT_SEED)
    return {"videos": gl._draw_counts(rng, N_BOOT, n_videos),
            "seeds": gl._draw_counts(rng, N_BOOT, n_seeds)}


def l_rows(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R + 1,) rows of the video x seed bootstrap and of the video-only bootstrap (seeds
    averaged first, the same video draws) of one contrast; ``delta`` (videos, paired seeds)."""
    dr = l_draws(*delta.shape)
    w, m = dr["videos"].astype(np.float64), dr["seeds"].astype(np.float64)
    rows = ((w @ delta) * m).sum(axis=1) / (w.sum(axis=1) * m.sum(axis=1))
    return rows, w @ delta.mean(axis=1) / w.sum(axis=1)


def interval(rows: np.ndarray) -> list[float]:
    """The 95% percentile interval over rows 1..R (finite rows)."""
    r = np.asarray(rows[1:], np.float64)
    r = r[np.isfinite(r)]
    return [float(v) for v in np.percentile(r, [2.5, 97.5])] if r.size else [math.nan, math.nan]


def contrast_class(lo: float, hi: float) -> str:
    return CLASS_POSITIVE if lo > 0 else CLASS_NEGATIVE if hi < 0 else CLASS_UNDETERMINED


def interpretation(name: str, cls: str) -> str | None:
    """PLAN.md "해석 문구": an undetermined contrast is limited to its sentence, an upper bound
    below 0 is 열세; a positive contrast has no sentence of its own."""
    if cls == CLASS_UNDETERMINED:
        return INTERPRETATION.get(name)
    return CLASS_NEGATIVE if cls == CLASS_NEGATIVE else None


def unflagged_seeds(data: Mapping, a: str, b: str) -> list[int]:
    """Seeds where neither arm is flagged collapsed in either pair (I2 shares P's chains): the
    seeds of ``reference_excluding_flagged`` only."""
    arm = {"i2": "p"}
    return [s for s in SEEDS if all(not data[(arm.get(x, x), p, s)]["collapsed"] for x in (a, b)
                                    for p in PAIRS)]


def _l(data: Mapping, arm: str, p: str, s: int) -> np.ndarray:
    return data[("p", p, s)]["L0"] if arm == "i2" else data[(arm, p, s)]["L"]


def l_contrast(name: str, a: str, b: str, data: Mapping, seeds: Sequence[int],
               verdict: bool = False) -> dict:
    """A - B on L over ``seeds`` (module docstring): estimate, main and video-only intervals, per
    seed and per pair, class and wording; ``_delta`` (videos, seeds) for the tertiles. A
    non-finite bootstrap row stops a ``verdict`` contrast; a descriptive one drops it and
    records how many (``dropped_rows``)."""
    seeds = list(seeds)
    if not seeds:
        return {"no_verdict": True, "seeds": []}
    delta = np.stack([np.mean([_l(data, a, p, s) - _l(data, b, p, s) for p in PAIRS], axis=0)
                      for s in seeds], axis=1)
    rows, video = l_rows(delta)
    dropped = int((~np.isfinite(rows)).sum() + (~np.isfinite(video)).sum())
    if verdict and dropped:
        raise SystemExit(f"{name}: {dropped} non-finite bootstrap rows in a verdict contrast; "
                         "stop and report")
    lo, hi = interval(rows)
    cls = contrast_class(lo, hi)
    return {"seeds": seeds, "n_videos": int(delta.shape[0]), "estimate": float(rows[0]),
            "dropped_rows": dropped,
            "interval": [lo, hi], "interval_video_only": interval(video),
            "by_seed": {s: float(delta[:, i].mean()) for i, s in enumerate(seeds)},
            "by_pair": {p: float(np.mean([_l(data, a, p, s) - _l(data, b, p, s)
                                          for s in seeds])) for p in PAIRS},
            "arms": {"a": float(np.mean([np.mean([_l(data, a, p, s) for p in PAIRS], axis=0)
                                         for s in seeds])),
                     "b": float(np.mean([np.mean([_l(data, b, p, s) for p in PAIRS], axis=0)
                                         for s in seeds]))},
            "class": cls, "interpretation": interpretation(name, cls), "_delta": delta}


def reference_excluding_flagged(data: Mapping) -> dict:
    """Descriptive, never the outcome: the two contrasts on the seeds where neither arm has a
    chain flagged collapsed in either pair; fewer than two such seeds: ``seeds_insufficient`` and
    no interval."""
    flagged = sorted(f"{x['key']}" for x in data.values() if x["collapsed"])
    out = {}
    for name, (a, b) in CONTRASTS.items():
        seeds = unflagged_seeds(data, a, b)
        out[name] = {"seeds": seeds, "seeds_insufficient": True} if len(seeds) < 2 else {
            **_public(l_contrast(name, a, b, data, seeds)), "seeds_insufficient": False}
    return {"note": "descriptive; the main analysis keeps every numerically complete chain and "
                    "the outcome never uses this block", "flagged": flagged, "contrasts": out}


def pooled_videos(lab: Mapping, labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """The anomalous and the normal videos of the pooled metrics (labelled, >= 32 frames)."""
    ok = np.asarray(lab["valid"]) & (np.asarray(lab["total_frames"]) >= DEFAULT_MIN_FRAMES)
    y = np.asarray(labels)
    return np.flatnonzero(ok & (y == 1)), np.flatnonzero(ok & (y == 0))


def pooled_rows(z: np.ndarray, lab: Mapping, labels: Sequence[int], wa: np.ndarray,
                wn: np.ndarray, chunk: int = 32) -> dict[str, np.ndarray]:
    """(R + 1,) pooled frame AUC and AP (points) of one chain's snippet logits ``z`` for every
    row of the video draws (``wa``, ``wn``: the counts of ``pooled_videos``): a video drawn k
    times counts its frames k times. Frames share their snippet's score, so a snippet enters
    with its positive and negative frame counts; equal scores form one threshold (AUC: one
    half; AP: one step), as sklearn's roc_auc_score and average_precision_score (row 0 equals
    ``frames.pooled_frame_metrics``)."""
    anom, norm = pooled_videos(lab, labels)
    col = np.full(len(lab["counts"]), -1)
    col[anom], col[norm] = np.arange(len(anom)), len(anom) + np.arange(len(norm))
    owner = np.repeat(np.arange(len(lab["counts"])), lab["counts"])
    size = np.concatenate([np.diff(grid.snippet_support(int(t)), axis=1)[:, 0]
                           for t in lab["total_frames"]])
    keep = col[owner] >= 0
    order = np.argsort(np.asarray(z, np.float64)[keep], kind="stable")
    s = np.asarray(z, np.float64)[keep][order]
    pos = np.asarray(lab["pos"], np.float64)[keep][order]
    neg = size[keep][order] - pos
    c = col[owner][keep][order]
    starts = np.flatnonzero(np.r_[True, np.diff(s) != 0])
    w_all = np.hstack([wa, wn]).astype(np.float64)
    auc, ap = [], []
    for a in range(0, len(w_all), chunk):
        w = w_all[a:a + chunk][:, c]
        pg, ng = np.add.reduceat(w * pos, starts, axis=1), np.add.reduceat(w * neg, starts, axis=1)
        tp, tn = pg.sum(axis=1), ng.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            auc.append((pg * (np.cumsum(ng, axis=1) - 0.5 * ng)).sum(axis=1) / (tp * tn))
            cp, cn = np.cumsum(pg[:, ::-1], axis=1), np.cumsum(ng[:, ::-1], axis=1)
            prec = np.divide(cp, cp + cn, out=np.zeros_like(cp), where=(cp + cn) > 0)
            ap.append((pg[:, ::-1] * prec).sum(axis=1) / tp)
    return {"frame_auc": 100 * np.concatenate(auc), "frame_ap": 100 * np.concatenate(ap)}


def pooled_draws(lab: Mapping, labels: Sequence[int], n_seeds: int) -> gl.Draws:
    anom, norm = pooled_videos(lab, labels)
    return gl.bootstrap_draws(len(anom), len(norm), n_seeds, N_BOOT, BOOT_SEED)


def pooled_contrast(a: str, b: str, data: Mapping, rows_of: Callable, lab: Mapping,
                    labels: Sequence[int]) -> dict:
    """A - B on the pooled frame AUC and AP over the seeds 3, 4, 5: per row the seed-weighted
    mean (the seed block of ``glance.bootstrap_draws``) of the pair means, its percentile
    interval over the finite rows 1..R (descriptive)."""
    seeds = list(SEEDS)
    m = pooled_draws(lab, labels, len(seeds)).seeds.astype(np.float64)
    out: dict = {"seeds": seeds}
    for stat in POOLED:
        x = np.stack([np.mean([rows_of(a, p, s)[stat] - rows_of(b, p, s)[stat] for p in PAIRS],
                              axis=0) for s in seeds], axis=1)
        r = (x * m).sum(axis=1) / m.sum(axis=1)
        lo, hi = interval(r)
        out[stat] = {"estimate": float(r[0]), "interval": [lo, hi],
                     "n_resamples": int(np.isfinite(r[1:]).sum()),
                     "dropped_rows": int((~np.isfinite(r[1:])).sum()),
                     "by_seed": {s: float(x[0, i]) for i, s in enumerate(seeds)},
                     "class": contrast_class(lo, hi)}
    return out


def outcome(contrasts: Mapping[str, Mapping]) -> dict:
    """PLAN.md "결과 행" (ubn): the first matching row, the two classes and the "결과 문장"
    composed from them (met also the manuscript claim of "결과의 쓰임")."""
    if any(c.get("no_verdict") for c in contrasts.values()):
        return {"row": "no_verdict", "result": ROWS["no_verdict"][0],
                "next": ROWS["no_verdict"][1], "classes": None,
                "statement": STATEMENT["no_verdict"]}
    cls = {k: contrasts[k]["class"] for k in CONTRASTS}
    u, c6 = (contrasts[k]["interval"][0] > 0 for k in CONTRASTS)
    row = "met" if u and c6 else "partial_u" if u else "partial_c6" if c6 else "not_met"
    out = {"row": row, "result": ROWS[row][0], "next": ROWS[row][1], "classes": cls,
           "statement": STATEMENT[row].format(u=cls["P-U"], c6=cls["P-C6"])}
    return out | ({"claim": CLAIM} if row == "met" else {})


# --- analysis -------------------------------------------------------------------------------

def read_done(scores: Path, frozen: Mapping) -> dict:
    """done.json of the sealed read and its deploy record: complete, every dataset, its freeze
    check's freeze commit the one git finds, its commit (``deployed_commit``) the chains'
    commit, the sealed plan, the inputs record of HEAD (the tree is clean), every file under
    <scores>/<ds> as recorded (SystemExit otherwise)."""
    f = Path(scores) / DONE
    if not f.is_file():
        raise SystemExit(f"{f}: no done.json (the sealed test read is not complete)")
    done = _json(f)
    commit = (done.get("code") or {}).get("commit")
    deploy = deployed_commit(frozen["freeze_commit"], commit)
    checks = {"complete": done.get("complete") is True,
              "datasets": list(done.get("datasets") or []) == list(DATASETS),
              "freeze commit": (done.get("freeze") or {}).get("freeze_commit") == frozen[
                  "freeze_commit"],
              "chains commit": done.get("chains_commit") == commit,
              "plan": (done.get("plan") or {}).get("sha256") == frozen["plan"]["sha256"],
              "inputs": done.get("inputs_sha256") == s0.sha256_file(INPUTS),
              "files": file_hashes(scores, DATASETS) == done.get("files")}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"{f}: {bad} differ from the frozen state; stop and report")
    return done, deploy


def deployed_commit(freeze: str, commit: str | None) -> dict:
    """The commit of the sealed read (= the chains'): the freeze commit, or a descendant of it
    whose diff from it touches nothing under ``src/``, stage3/PLAN.md, stage3/PLAN.sha256 or the
    inputs record (git; SystemExit otherwise)."""
    if not commit:
        raise SystemExit("done.json records no code commit")
    if commit == freeze:
        return {"commit": commit, "freeze_commit": freeze, "descendant": False, "changed": []}
    if s0._git("merge-base", "--is-ancestor", freeze, commit).returncode != 0:
        raise SystemExit(f"done.json's commit {commit} is neither the freeze commit {freeze} nor "
                         "a descendant of it")
    changed = s0._git("diff", "--name-only", freeze, commit).stdout.splitlines()
    protected = {_rel(f) for f in (*CODE_FILES, STAGE3_DIR / "PLAN.md", STAGE3_DIR / SEAL_FILE,
                                   INPUTS)}
    touched = sorted(c for c in changed if c in protected or c.startswith(f"{SRC}/"))
    if touched:
        raise SystemExit(f"done.json's commit {commit} changes {touched} after the freeze commit "
                         f"{freeze}: not the sealed code, plan or inputs")
    return {"commit": commit, "freeze_commit": freeze, "descendant": True, "changed": changed}


def access_record(done: Mapping, access_log: Path, scores: Path) -> dict:
    """The access log of the sealed read against done.json (SystemExit otherwise): its sha256;
    of this command's rows, one start row per attempt, one end row and it last, the rows from
    the last start on as many as done.json counts with a manifest row per dataset; the
    superseded attempt in <scores> (if any) the earlier one."""
    log = dict(done.get("access_log") or {})
    if not Path(access_log).is_file() or s0.sha256_file(access_log) != log.get("sha256"):
        raise SystemExit(f"{access_log}: not the access log recorded by done.json")
    rows = command_rows(access_log)
    starts = [i for i, r in enumerate(rows) if _is(r, START)]
    ends = [i for i, r in enumerate(rows) if _is(r, END)]
    last = rows[starts[-1]:] if starts else []
    sup = done.get("superseded")
    checks = {"start rows": len(starts) == done.get("attempt"),
              "end row": ends == [len(rows) - 1],
              "rows of the attempt": len(last) == log.get("rows"),
              "manifest rows": [r["dataset"] for r in last if _is(r, MANIFEST_PURPOSE)] == list(
                  DATASETS),
              "superseded attempt": sup is None or (
                  (Path(scores) / sup / STARTED).is_file() and _json(Path(scores) / sup / STARTED)
                  .get("attempt") == done.get("attempt", 0) - 1)}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"{access_log}: {bad} differ from done.json; stop and report")
    return {**log, "verified": True, "command_rows": len(rows), "attempts": len(starts)}


def chain_data(d: str, runs: Path, scores: Path, done: Mapping, summary: Mapping, lab: Mapping,
               settings: Mapping[str, str], commit: str, threshold: float) -> dict:
    """(arm, pair, seed) -> the chain's record (``check_chain``: the dataset's multipliers, the
    sealed read's commit), its collapse flag, logits and per-video L recomputed from labels.npz
    (stop unless it is summary.json's and the score file is the scored checkpoint)."""
    lv, pos = summary["l_videos"], split_rows(lab["pos"], lab["counts"])
    metric = fr.L_METRIC[d]
    out = {}
    for p in PAIRS:
        for s in SEEDS:
            for a in ARMS:
                key = chain_key(d, p, s, ARMS[a])
                chk = check_chain(Path(runs) / s0.chain_name(d, p, s) / ARMS[a], (d, p, s), a,
                                  settings, commit)
                z = s0._npz(score_file(scores, d, p, s, a))
                if (str(z["checkpoint_sha256"]), int(z["point"]), str(z["names_sha256"])) != \
                        (chk["checkpoint_sha256"], chk["point"], summary["names_sha256"]) or \
                        done["checkpoints"].get(key) != chk["checkpoint_sha256"]:
                    raise SystemExit(f"{key}: the score file is not the scored sigma checkpoint")
                entry, vals = summary["chains"][key], {}
                for zk, name in (("z", "metrics"), ("z0", "i2")):
                    if zk in z:
                        zl = split_rows(z[zk], z["counts"])
                        v = [fr.video_metric(*video_frames(zl[i], pos[i], int(
                            lab["total_frames"][i])), metric) for i in lv]
                        if v != entry[name]["per_video"]:
                            raise SystemExit(f"{key} {zk}: L from labels.npz differs from "
                                             f"{SUMMARY}")
                        vals[zk] = np.asarray(v, np.float64)
                out[(a, p, s)] = {"key": key, "check": chk, "z": z, "entry": entry,
                                  "L": vals["z"], "L0": vals.get("z0"),
                                  "collapsed": collapsed(chk["v_auc"], threshold)}
    return out


def _mean(values: Sequence) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def arm_block(a: str, data: Mapping) -> dict:
    """An arm's chains (V-AUC with its collapse flag, metrics), the number flagged and its means
    over every chain (the flag excludes nothing)."""
    src, metrics = ("p", "i2") if a == "i2" else (a, "metrics")
    chains = {f"{p}_s{s}": {"point": data[(src, p, s)]["check"]["point"],
                            "v_auc": data[(src, p, s)]["check"]["v_auc"],
                            "collapsed": data[(src, p, s)]["collapsed"],
                            **{k: data[(src, p, s)]["entry"][metrics][k]
                               for k in ("L", "n_videos", "n_excluded", "frame_auc", "frame_ap",
                                         "auc_anomalous")}}
              for p in PAIRS for s in SEEDS}
    records = [data[(src, p, s)]["check"]["record"] for p in PAIRS for s in SEEDS]
    return {"folder": ARMS[src], "chains": chains,
            "collapsed": sum(c["collapsed"] for c in chains.values()),
            "numerical_failures": sum(bool(r.get("failed") or r.get("nonfinite_losses"))
                                      for r in records),
            "numerical_failures_rule": "from the chain records; one stops before analysis",
            "v_auc_mean": _mean([c["v_auc"] for c in chains.values()]),
            **{f"{k}_mean": _mean([c[k] for c in chains.values()])
               for k in ("L", "frame_auc", "frame_ap", "auc_anomalous")}}


def tertiles(data: Mapping, lv: Sequence[int], totals: Sequence[int], cons: Mapping) -> list:
    """L by video-length tertile (descriptive): the L videos ranked by frame count (stable),
    three near-equal groups; each arm's mean over its chains, each contrast's estimate."""
    t = np.asarray([int(totals[v]) for v in lv])
    out = []
    for g in np.array_split(np.argsort(t, kind="stable"), 3):
        if not g.size:
            out.append({"videos": 0})
            continue
        arms = {LABEL[a]: _mean([float(data[(a, p, s)]["L"][g].mean()) for p in PAIRS
                                 for s in SEEDS]) for a in ARMS}
        out.append({"videos": int(g.size), "frames": [int(t[g].min()), int(t[g].max())],
                    "arms": arms, "contrasts": {k: None if c.get("no_verdict") else float(
                        c["_delta"][g].mean()) for k, c in cons.items()}})
    return out


def _public(c: Mapping) -> dict:
    return {k: v for k, v in c.items() if not k.startswith("_")}


def dataset_block(d: str, runs: Path, scores: Path, done: Mapping, settings: Mapping[str, str],
                  commit: str, threshold: float, rows: list) -> dict:
    """One dataset's arms, contrasts (L over seeds 3-5 and pooled), the reference block without
    the flagged chains, I2, tertiles; its rows appended."""
    summary = _json(Path(scores) / d / SUMMARY)
    lab = s0._npz(Path(scores) / d / LABELS_FILE)
    labels = [int(y) for y in lab["labels"]]
    lv = l_videos(lab, labels)
    if lv != summary["l_videos"] or str(lab["names_sha256"]) != summary["names_sha256"]:
        raise SystemExit(f"{d}: {LABELS_FILE} and {SUMMARY} disagree")
    data = chain_data(d, runs, scores, done, summary, lab, settings, commit, threshold)
    draws = pooled_draws(lab, labels, 1)
    cache: dict = {}

    def rows_of(arm: str, p: str, s: int) -> dict:
        if (arm, p, s) not in cache:
            src, zk, name = ("p", "z0", "i2") if arm == "i2" else (arm, "z", "metrics")
            r = pooled_rows(data[(src, p, s)]["z"][zk], lab, labels, draws.anomalous,
                            draws.normal)
            want = data[(src, p, s)]["entry"][name]
            if any(abs(r[k][0] - want[k]) > 1e-9 for k in POOLED):
                raise SystemExit(f"{d} {arm} {p} s{s}: pooled rows differ from {SUMMARY}")
            cache[(arm, p, s)] = r
        return cache[(arm, p, s)]

    aux = d != CONFIRMATION
    cons = {k: l_contrast(k, a, b, data, SEEDS, verdict=not aux)
            for k, (a, b) in CONTRASTS.items()}
    desc = {k: _public(l_contrast(k, a, b, data, SEEDS)) for k, (a, b) in DESCRIPTIVE.items()}
    for (a, p, s), x in data.items():
        rows.append(run_row(d, a, p, s, x, done))
    return {"role": "auxiliary" if aux else "confirmation", "non_independent": aux,
            "metric": summary["metric"], "collapse_threshold": threshold,
            "videos": {"test": summary["videos"], "anomalous": summary["anomalous"],
                       "label_failed": summary["labels"].get("failed") or {},
                       "l_videos": len(lv),
                       "l_excluded_one_class": int(summary["anomalous"]) - len(lv) - len(
                           summary["labels"].get("failed") or {}),
                       "pooled": dict(zip(("anomalous", "normal"), (int(len(x)) for x in
                                                                    pooled_videos(lab, labels)),
                                          strict=True))},
            "arms": {LABEL[a]: arm_block(a, data) for a in ARMS},
            "contrasts": {k: _public(c) for k, c in cons.items()},
            "reference_excluding_flagged": reference_excluding_flagged(data),
            "pooled_contrasts": {k: pooled_contrast(a, b, data, rows_of, lab, labels)
                                 for k, (a, b) in CONTRASTS.items()},
            "descriptive": {"I2": {"arm": arm_block("i2", data), **desc}},
            "tertiles": tertiles(data, lv, lab["total_frames"], cons)}


def run_row(d: str, a: str, p: str, s: int, x: Mapping, done: Mapping) -> dict:
    """One chain's row (JSON runs and CSV); P also its I2 metrics."""
    chk, m = x["check"], x["entry"]["metrics"]
    row = {"dataset": d, "pair": p, "seed": s, "arm": LABEL[a], "folder": ARMS[a],
           "point": chk["point"], "v_auc": chk["v_auc"], "collapsed": x["collapsed"],
           "parameters": chk["parameters"], "checkpoint_sha256": chk["checkpoint_sha256"],
           "score_sha256": done["files"].get(x["entry"]["file"]),
           **{k: m[k] for k in ("L", "n_videos", "n_excluded", "frame_auc", "frame_ap",
                                "auc_anomalous")}}
    if "i2" in x["entry"]:
        row |= {f"{k}_i2": x["entry"]["i2"][k] for k in ("L", "frame_auc", "frame_ap",
                                                         "auc_anomalous")}
    return row


def analyze(runs: Path, scores: Path, access_log: Path, out: Path | None = None,
            csv_out: Path | None = None) -> dict:
    """results/e024_stage3.json and e024_stage3_runs.csv (module docstring, ``analyze``)."""
    for root in (runs, scores):
        if not Path(root).is_dir():
            raise SystemExit(f"{root}: no folder")
    frozen = require_frozen()
    inputs = read_inputs()
    files = check_input_files(inputs)
    done, deploy = read_done(scores, frozen)
    access = access_record(done, access_log, scores)
    keys = [k for k, *_ in expected_chains(runs)]          # the gate chain last
    selection = selection_hashes(runs, keys)
    gates = require_gates(runs, GATE_FILES, keys, selection) | require_gates(
        runs, (GATE_S,), keys[:-1], selection)
    recorded = {**(done.get("gates") or {}), GATE_S: done.get("gate_s") or {}}
    moved = [n for n in gates if (recorded.get(n) or {}).get("sha256") != gates[n]["sha256"]]
    if moved:
        raise SystemExit(f"gate records {moved} are not those of the sealed read (done.json)")
    rows: list[dict] = []
    blocks = {d: dataset_block(d, runs, scores, done, settings_of(d, inputs), deploy["commit"],
                               threshold_of(d, inputs), rows) for d in DATASETS}
    report = {"check": "E024 stage 3 (stage3/PLAN.md)", **frozen, "deploy": deploy,
              "inputs": {"file": _rel(INPUTS), "sha256": s0.sha256_file(INPUTS),
                         "files": files, "collapse": inputs["collapse"],
                         "collapse_threshold": {d: threshold_of(d, inputs) for d in DATASETS},
                         "multiplier": inputs["multiplier"], "stage1": inputs["stage1"],
                         "archive": inputs["archive"], "lists": inputs["lists"],
                         "tracks_sha256": (inputs.get("tracks") or {}).get("sha256")},
              "gates": {**gates, "p": files["ok"], "chains": len(keys),
                        "gates_recomputed_before_the_read": done.get("gates_recomputed")},
              "access": access,
              "lineage": {k: done.get(k) for k in ("code", "plan", "plan_sha256_file",
                                                   "inputs_sha256", "chains_commit", "attempt",
                                                   "infra_rerun", "utc_start", "utc_end",
                                                   "environment", "checkpoints", "files")},
              "datasets": {CONFIRMATION: blocks[CONFIRMATION]},
              "outcome": outcome(blocks[CONFIRMATION]["contrasts"]),
              "auxiliary": {d: b for d, b in blocks.items() if d != CONFIRMATION},
              "interpretation": {"classes": {"lower > 0": CLASS_POSITIVE,
                                             "upper < 0": CLASS_NEGATIVE,
                                             "otherwise": CLASS_UNDETERMINED},
                                 "undetermined": INTERPRETATION, "cause": CAUSE},
              "runs": rows,
              "bootstrap": {"n_resamples": N_BOOT, "rng_seed": BOOT_SEED, "level": LEVEL,
                            "main": "video x seed (glance.py, stage 1 sensitivity): per row one "
                                    "draw of the L videos and one draw of the paired seeds, "
                                    "shared (default_rng(rng_seed): videos, then seeds)",
                            "video_only": "the same video draws, seeds averaged first",
                            "pooled": "glance.bootstrap_draws: anomalous and normal pooled "
                                      "videos, then the paired seeds",
                            "interval": "percentile 2.5 / 97.5 over rows 1..R"},
              "libraries": {"python": platform.python_version(), "numpy": np.__version__}}
    write_json(Path(out) if out is not None else RESULT_FILE, report)
    write_csv(Path(csv_out) if csv_out is not None else RUNS_CSV, rows)
    return report


# --- command line ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e024_stage3")
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="one P / C6 / gate_k0 chain (server)")
    t.add_argument("--dataset", required=True, choices=DATASETS)
    t.add_argument("--pair", required=True, choices=PAIRS)
    t.add_argument("--seed", required=True, type=int, choices=SEEDS)
    t.add_argument("--arm", required=True, choices=tuple(TRAIN))
    t.add_argument("--out_dir", required=True)
    t.add_argument("--lr_mult", help="the dataset's multiplier (ubn: the inputs record's; ucf, "
                                     "xdv: 1; default: it)")
    t.add_argument("--scale_mult", help="as --lr_mult")
    t.add_argument("--n_chunk", type=int, default=common.N_CHUNK_DEFAULT)
    t.add_argument("--paths", help="paths file (default configs/paths.yaml)")
    t.add_argument("--device", help="cuda / cpu (default: cuda when available)")
    pi = sub.add_parser("pins", help="write the pre-freeze inputs record (gate (p)), once")
    pi.add_argument("--u_dev", type=Path, required=True,
                    help="the stage-2 runs root (<pair>_s<seed>/urdmu_u200_lr<m>)")
    pi.add_argument("--zip_sha256", required=True, help="sha256 of UBnormal.zip")
    pi.add_argument("--lists", type=Path, required=True, help="the six official list files")
    pi.add_argument("--tracks_check", type=Path, required=True,
                    help="the stage-2 tracks_check.json (prep.ubnormal tracks-check)")
    pi.add_argument("--out", type=Path, help=f"default {_rel(INPUTS)}")
    sub.add_parser("freeze", help="write stage3/PLAN.sha256")
    gk = sub.add_parser("gate-k0", help="gate_k0 against U, bit for bit")
    gk.add_argument("--runs", type=Path, required=True, help="the stage-3 runs root")
    gk.add_argument("--u_runs", type=Path, help="the U chains' root (default --runs)")
    gk.add_argument("--out", type=Path, help="default <runs>/gate_k0.json")
    ga = sub.add_parser("gates", help="gates (h), (e), (n), (v)")
    ga.add_argument("--runs", type=Path, required=True)
    st = sub.add_parser("score-test", help="the sealed test read, once (server)")
    st.add_argument("--runs", type=Path, required=True)
    st.add_argument("--out", type=Path, required=True, help="scores folder outside the repo")
    st.add_argument("--access_log", type=Path, required=True, help="the access-log CSV")
    st.add_argument("--paths", help="paths file (default configs/paths.yaml)")
    st.add_argument("--device", help="cuda / cpu (default: cuda when available)")
    st.add_argument("--freeze_commit", help="the stage-3 freeze commit (required in a deployed "
                                            "copy, with DEPLOYED_LINEAGE.json beside "
                                            "DEPLOYED_COMMIT; checked against git otherwise)")
    st.add_argument("--infra_rerun", help="reason of the one infrastructure rerun")
    an = sub.add_parser("analyze")
    an.add_argument("--runs", type=Path, required=True, help="local copy of the runs root")
    an.add_argument("--scores", type=Path, required=True, help="local copy of score-test --out")
    an.add_argument("--access_log", type=Path, required=True,
                    help="local copy of the access log, checked against done.json")
    an.add_argument("--out", type=Path, default=RESULT_FILE)
    an.add_argument("--csv", type=Path, default=RUNS_CSV)
    a = ap.parse_args(argv)
    if a.command == "train":
        rec = train_arm(a.dataset, a.pair, a.seed, a.arm, a.out_dir, lr_mult=a.lr_mult,
                        scale_mult=a.scale_mult, n_chunk=a.n_chunk, paths_file=a.paths,
                        device=a.device)
        sel = {k: v["point"] for k, v in rec["selection"].items()}
        print(f"[e024_stage3] train done: failed={rec['failed']} selection={sel}")
    elif a.command == "pins":
        out = write_pins(a.u_dev, a.zip_sha256, a.lists, a.tracks_check, a.out)
        print(f"[e024_stage3] pins -> {out}")
    elif a.command == "freeze":
        print(f"[e024_stage3] PLAN.sha256 {freeze()}: commit it alone (the freeze commit)")
    elif a.command == "gate-k0":
        res = gate_k0(a.runs, a.u_runs)
        write_json(a.out or Path(a.runs) / GATE_FILES[0], res)
        print(f"[e024_stage3] gate-k0: {'passed' if res['ok'] else 'FAILED'}")
        if not res["ok"]:
            raise SystemExit(1)
    elif a.command == "gates":
        oks = write_gates(a.runs)
        print(f"[e024_stage3] gates: {oks}")
        if not all(oks.values()):
            raise SystemExit(1)
    elif a.command == "score-test":
        done = score_test(a.runs, a.out, a.access_log, freeze_commit=a.freeze_commit,
                          infra_rerun=a.infra_rerun, paths_file=a.paths, device=a.device)
        print(f"[e024_stage3] score-test done: {len(done['files'])} files -> {a.out}")
    else:
        rep = analyze(a.runs, a.scores, a.access_log, a.out, a.csv)
        print(json.dumps(rep["outcome"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
