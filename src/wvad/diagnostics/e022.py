"""E022 (stage F): local analysis of the pre-registered dense confirmation on the test set
(experiments/E022_dense_confirmation/PLAN.md). Analysis code fixed at the freeze commit (PLAN.md
"코드 규칙" 3); the statistics are wvad.diagnostics.e022_stats, the frame arrays
wvad.eval.frames, the descriptives wvad.diagnostics.e022_desc.

    freeze     PLAN.sha256 = sha256 of PLAN.md before "## 변경 기록" (PLAN.md committed unmodified
               at HEAD, clean tracked tree; once)
    prefreeze  (3c) the tests/ tree sha256 and run summary of a pytest JUnit XML, the server's
               training and extraction environment records and the driver against E021's
               (580.126.20), merged into results/e022_train_inputs.json beside the 3a record
    log        results/e022_test_access.log: the access logs of both machines merged in time
               order (append-only copies; identical lines once)
    gates      --stage train    (a)-(g), (f), (r0) (with the local E020 copy), (s), (s3), (p),
                                (w), (w2); --infra_failed / --gate_failed runs left out and
                                recorded; only before the first test read: refused, the gate
                                record untouched, once results/e022_gate.json holds a (v), (v2),
                                (v3) or (s3t) section or the committed access-log copy or
                                --test_log (optional here) has an (A)-(D) line (access (0) lines
                                do not count), so step 7's --gate_failed list cannot be rewritten
                                after the test scores exist                       (PLAN.md 7)
               --stage extract  (t), (y0)-(y7) (the server's (y4)-(y6) records)  (9)
               --stage v        (v), (v2) on the scoring-1 files, part of access (B) (10)
               --stage v3       (v3), (s3t) on the scoring-2 files, before any metric, part of
                                access (C); --gate_failed runs left out and recorded (11)
               (v) and (v3) need a clean working tree and the frozen plan (else a "refused"
               line) and log "start" / "end" lines in --test_log; (v3) also needs --gate_failed
               to be exactly the list step 7 recorded (train.gate_failed of the committed
               results/e022_gate.json, else a "refused" line); results/e022_gate.json
               (the extraction gates: results/e022_extract_gate.json, key "gates"); written
               also when a gate fails
    analyze    access (D), clean working tree: the seal history and the refusals (code changed
               since the freeze, --gate_failed other than step 7's list, an earlier summary or
               (D) start line without --recompute), the (D) start line, then the committed
               records, every gate again ((v3), (s3t) without the --gate_failed runs) and the
               access log, the verdict (C - R0, UCF frame AUC / XD frame AP, logit readout),
               flags, condition (2), decomposition and descriptives: results/e022_summary.json
               (verdict first) and results/e022_runs.csv; once (again only with --recompute
               REASON, PLAN.md re-computation)

Score roots (copies of the server's ~/wvad_runs/e022_scores): val/<chain>/<head>_m10.npz
(validation dense scoring), test/<chain>/<arm>.npz with meta_<k>.json and scores.sha256 (test
scorings). Arms: R0, R0.last (E020 m10 checkpoints), C, C.G, C.last, B (HB), R1, R1.last, A1.S
(H1); C.W and C.o32 are read from C's file (window readout of the same logits; C's snippet
logits with c_32(v) = gamma x mean of its test d16m 32-unit z).
"""

from __future__ import annotations

import argparse
import datetime as dt_time
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

from wvad.data import dense_test as dt
from wvad.data.eval_manifests import eval_manifest_file, load_eval_manifest
from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics import e021_gates as eg
from wvad.diagnostics import e021_score as es
from wvad.diagnostics import e022_desc as desc
from wvad.diagnostics import e022_gates as g
from wvad.diagnostics import e022_score as sc
from wvad.diagnostics import e022_stats as st
from wvad.diagnostics.common import code_version, write_csv, write_json
from wvad.eval import frames as fr
from wvad.eval.dense import compare_logits
from wvad.eval.dense_readout import offset, readout, sigmoid32
from wvad.eval.metrics import frame_auc_ap, max_video_auc
from wvad.extract import grid
from wvad.extract.dense import rel_video_path
from wvad.paths import REPO_ROOT
from wvad.prep.manifests import sha256_file

E022_DIR, RESULTS = dt.E022_DIR, dt.RESULTS
PLAN, SEAL = E022_DIR / "PLAN.md", E022_DIR / "PLAN.sha256"
CHANGELOG = dt.CHANGELOG
OUT = {"gate": RESULTS / "e022_gate.json", "extract": RESULTS / "e022_extract_gate.json",
       "summary": RESULTS / "e022_summary.json", "runs": RESULTS / "e022_runs.csv",
       "inputs": RESULTS / "e022_train_inputs.json", "selection": RESULTS / "e022_selection.json",
       "approvals": RESULTS / "e022_approvals.json",
       "access": RESULTS / "e022_test_access.log",
       "scores": RESULTS / "e022_test_scores.sha256"}
E020_RESULTS = REPO_ROOT / "experiments" / "E020_confirmation_f1" / "results"
E020_RUNS_CSV = E020_RESULTS / "e020_runs.csv"
E020_SCORES_LIST = E020_RESULTS / "e020_test_scores.sha256"
PRIMARY = dict(fr.PRIMARY)
ANOMALOUS = {"ucf": "anom_frame_auc", "xdv": "anom_frame_ap"}
READOUTS = ("logit", "sigma")
IGNORED_CONFIG = ("seed", "out_dir", "init")       # gate (c): the only keys that may differ
START_EPOCH = 20
V_REL, V_SIGMA, V_POINTS, V2_TOL = 1e-5, 1e-6, 1e-4, 1e-9
BOOT_POINT_TOL = 1e-6
TRAIN_GATES = ("a", "b", "c", "d", "e", "f", "g", "r0", "s", "s3", "p", "w", "w2")
# the fixed server code (PLAN.md "코드 규칙" 1): unchanged from the freeze commit on
FIXED_SERVER = ("src/wvad/extract/testsplit.py", "src/wvad/data/dense_test.py",
                "src/wvad/diagnostics/e022_score.py", "src/wvad/diagnostics/e022_gates.py")
HEAD_ARMS = {"hb": ("C", "C.G", "C.last", "C.W", "C.o32", "B"),
             "h1": ("R1", "R1.last", "A1.S")}
ALL_ARMS = ("R0", "R0.last", "C", "C.G", "C.last", "C.W", "C.o32", "B", "R1", "R1.last", "A1.S")
FILE_OF = {"C.W": "C", "C.o32": "C"}               # arms read from another arm's score file
UNIT_KEYS = ("logits", "probs", "local_logits", "offsets", "gamma")
SNIPPET_KEYS = ("z", "c", "zc", "sigma", "gamma")
# the stored arrays each arm's frame scores come from (their non-finite values: its failure)
ARM_KEYS = {**{a: SNIPPET_KEYS for a in ("C", "C.G", "C.last", "A1.S", "C.W")},
            "C.o32": (*SNIPPET_KEYS, "o32_local_logits")}


def now_utc() -> str:
    return dt_time.datetime.now(dt_time.timezone.utc).isoformat(timespec="seconds")


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_or(path: Path, default):
    return _json(path) if Path(path).is_file() else default


def _npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def clean(obj):
    """JSON-safe copy: numpy to Python, tuple keys to 'a/b', NaN and infinities to None."""
    if isinstance(obj, Mapping):
        return {("/".join(map(str, k)) if isinstance(k, tuple) else str(k)): clean(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def git(*args: str) -> subprocess.CompletedProcess:
    """git in this repository (REPO_ROOT read at call time)."""
    return g.git(*args, repo=REPO_ROOT)


def git_bytes(*args: str) -> bytes:
    """Raw stdout of a git command (file contents byte for byte); empty on failure."""
    res = subprocess.run(["git", "-c", f"safe.directory={Path(REPO_ROOT).as_posix()}", *args],
                         cwd=REPO_ROOT, capture_output=True, check=False)
    return res.stdout if res.returncode == 0 else b""


def is_ancestor(a: str | None, b: str | None) -> bool:
    """a is b or an ancestor of b (git merge-base --is-ancestor); False when either is unknown."""
    return bool(a and b) and git("merge-base", "--is-ancestor", a, b).returncode == 0


# --- seal ------------------------------------------------------------------------------------

def plan_sha() -> str:
    return dt.plan_sha256(PLAN)


def require_clean_tree() -> dict:
    code = code_version()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed (PLAN.md); code version {code}")
    return code


def freeze_command(args: argparse.Namespace) -> None:
    require_clean_tree()
    rel = PLAN.relative_to(REPO_ROOT).as_posix()
    if (git("cat-file", "-e", f"HEAD:{rel}").returncode != 0
            or git("status", "--porcelain", "--", rel).stdout.strip()):
        raise SystemExit(f"{rel} must be committed and unmodified at HEAD before the freeze")
    if SEAL.exists():
        raise SystemExit(f"{SEAL} exists: the plan is frozen once")
    h = plan_sha()
    SEAL.write_text(f"{h}  PLAN.md (before {CHANGELOG})\n", encoding="utf-8", newline="\n")
    print(f"PLAN.sha256 written ({h}); commit it alone: that commit is the freeze commit")


# the freeze commit's parent tree (PLAN.md "고정 규칙"): the plan, the E022 code, README and the
# pre-freeze records (3a / 3c, access (0)); paths relative to the E022 folder or the repository
FREEZE_PARENT = {"plan": ("PLAN.md",),
                 "records": ("results/e022_train_inputs.json", "results/e022_test_access.log"),
                 "repository": ("README.md", "src/wvad/data/dense_test.py",
                                "src/wvad/extract/testsplit.py", "src/wvad/eval/frames.py",
                                *(f"src/wvad/diagnostics/{m}.py" for m in (
                                    "e022", "e022_score", "e022_gates", "e022_stats",
                                    "e022_desc")))}


def seal_history() -> dict:
    """PLAN.sha256 was added once, by the freeze commit, which changes nothing else and whose
    parent tree holds the plan, the code, README and the pre-freeze records (``FREEZE_PARENT``;
    PLAN.md "고정 규칙"), and holds the hash of the current plan and of PLAN.md in that commit
    (no re-freeze)."""
    rel = {n: (E022_DIR / n).relative_to(REPO_ROOT).as_posix() for n in ("PLAN.sha256",
                                                                         "PLAN.md")}
    touched = git("log", "--format=%H", "--", rel["PLAN.sha256"]).stdout.split()
    freeze = dt.freeze_commit_of(REPO_ROOT, E022_DIR)
    if len(touched) != 1 or touched[0] != freeze:
        raise SystemExit(f"PLAN.sha256 was touched by {touched}: only the freeze commit adds it")
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root",
                  freeze).stdout.split()
    if changed != [rel["PLAN.sha256"]]:
        raise SystemExit(f"the freeze commit {freeze} changes {changed}: PLAN.sha256 alone "
                         "(PLAN.md 고정 규칙)")
    e022 = E022_DIR.relative_to(REPO_ROOT).as_posix()
    parent = [*(f"{e022}/{p}" for k in ("plan", "records") for p in FREEZE_PARENT[k]),
              *FREEZE_PARENT["repository"]]
    absent = [p for p in parent if git("cat-file", "-e", f"{freeze}^:{p}").returncode != 0]
    if absent:
        raise SystemExit(f"the freeze commit's parent lacks {absent}: commit the plan, code, "
                         "README and the pre-freeze records first (PLAN.md 고정 규칙)")
    frozen = git_bytes("show", f"{freeze}:{rel['PLAN.sha256']}").decode("utf-8").split()
    shown = git_bytes("show", f"{freeze}:{rel['PLAN.md']}")
    at_freeze = hashlib.sha256(dt.plan_frozen_part(shown)).hexdigest()
    now = dt.plan_seal(E022_DIR)["sha256"]
    if not frozen or not frozen[0] == now == at_freeze:
        raise SystemExit(f"plan hash now {now}, frozen {frozen[:1]}, at the freeze {at_freeze}")
    return {"freeze_commit": freeze, "sha256": now, "seal_commits": touched,
            "freeze_changes": changed, "parent_holds": parent}


# --- (3c) pre-freeze record and the access-log copy -----------------------------------------

def prefreeze(junit: Path, env_train: Mapping, env_extract: Mapping,
              inputs_file: Path | None = None) -> dict:
    """The (3c) record: tests/ tree and run summary, the server environments and the driver
    comparison (PLAN.md 3c, review N13), merged into the 3a record's file."""
    inputs_file = Path(inputs_file or OUT["inputs"])
    rec = _json(inputs_file)
    if "train_inputs" not in rec:
        raise SystemExit(f"{inputs_file}: no 3a record (testsplit inputs) yet")
    want = g.e021_training_env()
    rec["tests"] = g.tests_record(Path(junit))
    rec["environment"] = {
        "training": dict(env_train), "extraction": dict(env_extract),
        "training_differences": [k for k in g.TRAIN_ENV_KEYS
                                 if env_train.get(k) != want.get(k)],
        "extraction_y1_preview": g.gate_y1(env_extract), "utc": now_utc()}
    rec["driver"] = {"server_training": env_train.get("driver"),
                     "server_extraction": env_extract.get("driver"), "e021": want.get("driver"),
                     "same": env_train.get("driver") == want.get("driver")
                     == env_extract.get("driver"),
                     "handling": "PLAN.md (f): recorded; training determinism by (a), (g), (s), "
                                 "scoring by (v); the verdict tagged '(드라이버 변경)'"}
    write_json(Path(inputs_file), rec)
    return rec


def _merged(texts: Sequence[str]) -> list[str]:
    """Log lines of several copies: every line once, ordered by its time (stable)."""
    rows, seen = [], set()
    for i, text in enumerate(texts):
        for j, line in enumerate(text.splitlines()):
            if not line.strip() or line in seen:
                continue
            seen.add(line)
            try:
                t = str(json.loads(line).get("time_utc") or "")
            except ValueError:
                t = ""
            rows.append((t, i, j, line))
    return [r[3] for r in sorted(rows)]


def merge_logs(parts: Sequence[Path], out: Path | None = None) -> int:
    """The access logs merged into one copy: every line once, ordered by its time (stable)."""
    out = Path(out or OUT["access"])
    lines = _merged([Path(p).read_text(encoding="utf-8") for p in parts])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    return len(lines)


def merged_entries(*parts: Path) -> list[dict]:
    """The entries of the committed copy and this machine's log merged as ``merge_logs`` does
    (the current analysis's (D) lines are in the local log only); absent parts are empty."""
    texts = [Path(p).read_text(encoding="utf-8") for p in parts if Path(p).is_file()]
    out = []
    for line in _merged(texts):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def log_entries(path: Path | None = None) -> list[dict]:
    """The committed access-log copy (results/e022_test_access.log)."""
    return dt.read_test_log(path or OUT["access"])


def access0_files(entries: Sequence[Mapping]) -> dict[str, str]:
    """The E020 m10 chain hashes of the last completed access (0) (testsplit check)."""
    ends = [e for e in entries if e.get("access") == "0" and e.get("event") == "end"]
    if not ends:
        return {}
    res = ends[-1].get("result") or {}
    return {k: v for k, v in ((res.get("e020_runs") or {}).get("files") or {}).items() if v}


# --- gates of the training and validation outputs (PLAN.md 7) ------------------------------

def _config(d: Path) -> dict:
    return torch.load(Path(d) / sc.LAST, map_location="cpu", weights_only=True)["config"]


def references(e021_runs: Path, record: Mapping) -> dict[tuple[str, str, str, int], dict]:
    """Gate (c) references: the E021 M10 development chain (seed 0) of the same dataset, pair
    and head, stages 1 and 2, each checkpoint with the sha256 recorded before the freeze."""
    files = ((record.get("train_inputs") or {}).get("e021_configs") or {}).get("files") or {}
    out = {}
    for d in sc.DATASETS:
        for p in sc.PAIRS:
            for h in sc.HEADS:
                for stage, run in ((1, f"{h}_m10_stage1"), (2, f"{h}_m10")):
                    rel = f"{d}_{p}_s0/{run}/{sc.LAST}"
                    f = Path(e021_runs) / rel
                    if sha256_file(f) != files.get(rel):
                        raise ValueError(f"{rel}: sha256 differs from the pre-freeze record")
                    out[(d, p, h, stage)] = _config(f.parent)
    return out


def config_ok(cfg: Mapping, ref: Mapping, key: es.RunKey) -> bool:
    """Gate (c): the E021 reference config except seed, out_dir and init (the key set
    included), with the head's features, save_epoch_ckpts, mean offset and multiplier 10."""
    a, b = set(cfg) - set(IGNORED_CONFIG), set(ref) - set(IGNORED_CONFIG)
    return bool(cfg.get("seed") == key.seed and a == b and all(cfg[k] == ref[k] for k in a)
                and cfg.get("features") == sc.HEAD_FEATURES[key.head]
                and cfg.get("save_epoch_ckpts") is True and cfg.get("global_mode") == "mean"
                and float(cfg.get("offset_lr_mult", 0)) == 10.0)


@dataclass
class ValData:
    dataset: str
    idx: np.ndarray
    labels: np.ndarray
    split_sha: str
    totals: np.ndarray


def val_data(dataset: str) -> ValData:
    idx, labels, sha = eg.val_split(dataset)
    return ValData(dataset, idx, labels, sha, eg.val_totals(dataset))


def keys_of(excluded: Sequence[str] = ()) -> list[es.RunKey]:
    return [k for k in sc.head_keys() if sc.run_label(k) not in excluded]


def load_runs(runs: Path, keys: Sequence[es.RunKey]) -> tuple[dict, list[str]]:
    out, errors = {}, []
    for k in keys:
        try:
            out[k] = eg.load_run(runs, k)
        except es.GATE_ERRORS as exc:
            errors.append(f"{sc.run_label(k)}: {exc!r}")
    return out, errors


def chain_rows(runs_root: Path, runs: Mapping, refs: Mapping, vals: Mapping[str, ValData]
               ) -> tuple[list[dict], list[str]]:
    """(a)-(e), (g) of every run."""
    rows, errors = [], []
    for key, run in runs.items():
        v = vals[key.dataset]
        d1, d2 = es.stage1_dir(runs_root, key), es.run_dir(runs_root, key)
        try:
            rows.append({
                "run": sc.run_label(key), "failed": run.failed,
                "a": eg.same_record_at(d2, 0, d1, START_EPOCH), "b": eg.init_is_own(d1, d2),
                "c": (config_ok(_config(d1), refs[(key.dataset, key.pair, key.head, 1)], key)
                      and config_ok(run.config, refs[(key.dataset, key.pair, key.head, 2)],
                                    key)),
                "d": eg.order_ok(runs_root, key, v.idx), "e": eg.gamma_learned(run),
                "g": eg.selection_ok(run, v.labels, v.split_sha, d2)})
        except es.GATE_ERRORS as exc:
            errors.append(f"{sc.run_label(key)}: {exc!r}")
    return rows, errors


def s3_row(run, sf: Mapping, selected: Mapping) -> dict:
    """(s3) of an HB run at its selected epochs (C: W, C.G: G): the averaged scale-matched
    snippets' z = the d16m cache's z (|dz| <= 1e-5 max(1, |z|)); every epoch recorded."""
    head = {"run": sc.run_label(run.key)}
    failed = run.failed or eg.dense_failure(sf)
    if failed:
        return head | {"skipped": failed, "ok": True}
    epochs = sorted({int(c["epoch"]) for a, c in (selected.get("checkpoints") or {}).items()
                     if a in ("C", "C.G")})
    at = {e: compare_logits(sf["s3_z"][e - 1], sf["d16m_z"][e - 1]) for e in epochs}
    every = compare_logits(sf["s3_z"], sf["d16m_z"])
    return head | {"epochs": {str(e): r for e, r in at.items()}, "all_epochs": every,
                   "ok": bool(epochs) and all(r["pass"] for r in at.values())}


def score_rows(runs: Mapping, scores: Path, selection: Mapping, vals: Mapping[str, ValData]
               ) -> tuple[dict, list[str], dict]:
    """(s), (s3), (p) from the validation score files; a run with non-finite dense scores
    failed numerically (its (s3) skipped, (p) keeps the shape check)."""
    rows = {k: [] for k in ("s", "s3", "p")}
    errors, numerical = [], {}
    for key, run in runs.items():
        v, label = vals[key.dataset], sc.run_label(key)
        try:
            sf = _npz(sc.val_path(scores, key))
            rows["s"].append(eg.s_row(run, sf, v.idx, v.labels))
            rows["p"].append(eg.p_row(run, sf, v.totals))
            if key.head == "hb":
                rows["s3"].append(s3_row(run, sf, selection["chains"][label]))
            why = eg.dense_failure(sf)
            if why and not run.failed:
                numerical[label] = why
        except es.GATE_ERRORS as exc:
            errors.append(f"{label}: {exc!r}")
    return rows, errors, numerical


def val_calls(scores: Path) -> list[dict]:
    """The validation scoring calls that scored (a call refused by gate (p) scored nothing; it
    stays in val/meta.json)."""
    return [c for c in _json(Path(scores) / "val" / "meta.json").get("calls") or []
            if "refused" not in c]


def p_inputs(calls: Sequence[Mapping], record: Mapping) -> dict:
    """(p): every validation scoring call found the E021 inputs equal to the record."""
    rows = [g.inputs_gate("p", record, dict(c.get("train_inputs") or {})) for c in calls]
    return {"calls": rows, "ok": bool(rows) and all(r["ok"] for r in rows)}


BATCH_RECORDS = "env_e022_b*.json"                 # e022_score env --out <runs>/env_e022_b<k>.json
BATCH_SIZE_CHAINS = 24                             # PLAN.md 6: 24 chains per batch


def batches(runs: Path, calls: Sequence[Mapping]) -> dict[str, dict]:
    """(f): every training batch's environment record (env_e022_b*.json; the 3c pre-freeze
    records in the same folder are not batches) and every validation scoring call."""
    out = {f.name: _json(f) for f in sorted(Path(runs).glob(BATCH_RECORDS))}
    for i, c in enumerate(calls):
        env = dict(c.get("environment") or {})
        out[f"val_call_{i}"] = env | {"commit": (c.get("code") or {}).get("commit"),
                                      "train_inputs": c.get("train_inputs") or {}}
    return out


def gate_r0(calls: Sequence[Mapping], entries: Sequence[Mapping]) -> dict:
    """(r0): the E020 m10 files of the validation call (the server's) against access (0), their
    selection and last V-AUC against e020_runs.csv."""
    e020 = (calls[-1] if calls else {}).get("e020_m10") or {}
    current = {k: v for rec in e020.values() for k, v in (rec.get("files") or {}).items()
               if v}
    selections = {c: rec.get("selection") for c, rec in e020.items() if rec.get("selection")}
    histories = {c: [{"val_video_auc": rec["last_val_video_auc"]}] for c, rec in e020.items()
                 if rec.get("last_val_video_auc") is not None}
    return g.gate_r0(current, access0_files(entries), selections,
                     g.runs_csv_rows(E020_RUNS_CSV), histories)


R0_LOCAL = ("m10/history.json", "m10_stage1/history.json", "m10/record_val_scores.npz")


def r0_local(e020_runs: Path, entries: Sequence[Mapping]) -> dict:
    """(r0), the local copy the analysis reads (--e020_runs): for every expected E020 m10 chain
    the histories of both stages parse and the validation record exists, and the chain files
    (checkpoints, selection.json) have the sha256 access (0) recorded on the server (E020 records
    no history hash). A wrong path or an incomplete copy fails here, not as numerical failures
    of every R0 run (PLAN.md "수치 실패")."""
    want = access0_files(entries)
    missing, unreadable, different = [], [], []
    for d in sc.DATASETS:
        for p in sc.PAIRS:
            for s in sc.SEEDS:
                chain = sc.chain_name(d, p, s)
                root = Path(e020_runs) / chain
                for rel in R0_LOCAL:
                    if not (root / rel).is_file():
                        missing.append(f"{chain}/{rel}")
                    elif rel.endswith("history.json"):
                        try:
                            _history_values((root / rel).parent)
                        except SystemExit:
                            unreadable.append(f"{chain}/{rel}")
                for f in sc.R0_FILES:
                    key, path = f"{chain}/{sc.R0_CELL}/{f}", root / sc.R0_CELL / f
                    if not path.is_file():
                        missing.append(key)
                    elif sha256_file(path) != want.get(key):
                        different.append(key)
    return {"root": str(e020_runs), "missing": missing, "unreadable": unreadable,
            "different_from_access_0": different, "access_0_files": len(want),
            "ok": bool(want) and not (missing or unreadable or different)}


def w_gate(scores: Path, selection: Mapping, excluded: Sequence[str],
           skip: Sequence[str] = ()) -> dict:
    """(w): the selection recomputed by the same function from the stored logits equals the
    committed results/e022_selection.json (readouts, epochs, checkpoint sha256, R0); ``excluded``
    as at the selection (infrastructure failures), ``skip`` runs left out of the gates
    afterwards (--gate_failed)."""
    again = clean(sc.selection_record(scores, excluded))
    diff = [k for k in again["chains"] if k not in skip and again["chains"][k] != clean(
        selection["chains"].get(k))]
    same_r0 = again["r0"] == clean(selection.get("r0"))
    return {"runs": len(again["chains"]) - len(set(skip)), "different": diff, "r0_same": same_r0,
            "val_list_same": again["val_list_sha256"] == selection.get("val_list_sha256"),
            "ok": not diff and same_r0 and again["val_list_sha256"] == selection.get(
                "val_list_sha256")}


def w2_gate(scores: Path, selection: Mapping, runs: Mapping) -> dict:
    """(w2): e022_gates' independent recomputation of c(v), W and G and the V-AUC."""
    chains = {}
    for key in runs:
        label = sc.run_label(key)
        sf = _npz(sc.val_path(scores, key))
        rec = (selection["chains"].get(label) or {}).get("readouts") or {}
        chains[label] = {"dense_z": sf["dense_z"], "counts": sf["dense_counts"],
                         "totals": sf["total_frames"], "labels": sf["labels"],
                         "gamma": sf.get("gamma"), "recorded": rec,
                         "readouts": sc.READOUTS[key.head]}
    return g.gate_w2(chains)


def check_gate_failed(names: Sequence[str], infra: Sequence[str]) -> list[str]:
    """--gate_failed: '<chain>/<head>_m10' of the new chains dropped after a gate failure that
    could not be fixed (PLAN.md "관문 실패"; R0 has none: (r0) failing stops E022)."""
    valid = {sc.run_label(k) for k in sc.head_keys()}
    for name in names:
        if name not in valid or name in infra:
            raise SystemExit(f"--gate_failed {name}: not an E022 run, or an infrastructure "
                             "failure already")
    return sorted(set(names))


def committed_gate_failed() -> list[str]:
    """The runs step 7 dropped: ``train.gate_failed`` of results/e022_gate.json. It is read where
    the working tree is clean (``gates --stage v3``, ``analyze``) and before ``write_gate``
    rewrites the train section from the command line, so it is the committed value; SystemExit
    when the record, its train section or the list is missing."""
    try:
        return sorted(set(_json(OUT["gate"])["train"]["gate_failed"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"{OUT['gate'].name}: no train gate record with its gate_failed list "
                         f"(step 7) to check --gate_failed against ({exc!r})") from exc


def require_committed_gate_failed(names: Sequence[str]) -> list[str]:
    """--gate_failed of ``gates --stage v3`` and of ``analyze`` is exactly the list step 7
    recorded (``committed_gate_failed``). A chain is dropped after a gate failure that cannot be
    fixed before any test score exists (PLAN.md "관문 실패"), and every other cause of a failed
    (v3) / (s3t) stops E022: a run is neither dropped nor taken back once the scores can be
    read. The list cannot be rewritten after a test read either: ``gates --stage train`` is
    refused then (``require_before_test_access``)."""
    given, dropped = sorted(set(names)), committed_gate_failed()
    if given != dropped:
        raise SystemExit(f"--gate_failed {given} is not the list step 7 recorded, {dropped} "
                         f"(train.gate_failed of the committed {OUT['gate'].name}); the dropped "
                         "runs are not changed once the test scores exist (PLAN.md 관문 실패)")
    return given


TEST_GATE_SECTIONS = ("v", "v2", "v3", "s3t")      # the gate record's sections of the test gates
TEST_ACCESSES = ("A", "B", "C", "D")               # logged test reads; access (0) reads none


def require_before_test_access(test_log: Path | None = None) -> None:
    """``gates --stage train`` (step 7) runs only before the first test read: SystemExit
    otherwise, before anything is run or written. The command rewrites ``train.gate_failed`` of
    results/e022_gate.json, the list ``gates --stage v3`` and ``analyze`` hold --gate_failed to
    (``require_committed_gate_failed``); run again after a (v3) / (s3t) failure with another list
    it would turn that failure into a drop. A test read began when the gate record holds a (v),
    (v2), (v3) or (s3t) section (the local test gates write them, failed or not) or when the
    committed access-log copy (results/e022_test_access.log, if present) or ``test_log`` (if
    given) has an (A)-(D) line, whatever its event; access (0) lines are no test read. Only the
    command is checked: ``train_gates``, which ``analysis_gates`` runs again after the test
    reads, is not."""
    try:
        record = _json_or(OUT["gate"], {})
        if not isinstance(record, Mapping):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{OUT['gate'].name} is unreadable ({exc!r}): that no test read began "
                         "cannot be shown, so the train gates are not run") from exc
    found, held = [], [repr(s) for s in TEST_GATE_SECTIONS if s in record]
    if held:
        found.append(f"{OUT['gate'].name} holds the test gates' sections {', '.join(held)}")
    for what, path in (("the committed access-log copy", OUT["access"]),
                       ("the local test log", test_log)):
        if path is None:
            continue
        letters = sorted({e["access"] for e in log_entries(path)
                          if isinstance(e, Mapping) and e.get("access") in TEST_ACCESSES})
        if letters:
            named = ", ".join(f"({x})" for x in letters)
            found.append(f"{what} {Path(path).name} has access {named} lines")
    if found:
        raise SystemExit(f"gates --stage train runs only before the first test read (PLAN.md 7), "
                         f"but {'; '.join(found)}: it would rewrite train.gate_failed, the list "
                         "gates --stage v3 and analyze hold --gate_failed to, and the dropped "
                         "runs are not changed once the test scores exist (PLAN.md 관문 실패)")


def train_gates(args: argparse.Namespace) -> dict:
    """PLAN.md 7: the gates of the training and validation outputs (before the test access).
    Runs named by --infra_failed (no selection.json) and --gate_failed (dropped after an unfixed
    gate failure) are left out of the gates and recorded; the analysis counts their seeds as
    dropped for a reason other than the treatment (PLAN.md "주 검정")."""
    record = _json(OUT["inputs"])
    selection = _json(OUT["selection"])
    excluded = sc.check_excluded(args.runs, args.infra_failed)
    dropped = check_gate_failed(getattr(args, "gate_failed", None) or [], excluded)
    trained = keys_of(excluded)
    keys = [k for k in trained if sc.run_label(k) not in dropped]
    vals = {d: val_data(d) for d in sc.DATASETS}
    runs, errors = load_runs(args.runs, keys)
    try:
        refs = references(args.e021_runs, record)
    except es.GATE_ERRORS as exc:
        refs, errors = {}, [*errors, f"gate (c) references: {exc!r}"]
    rows, errs = chain_rows(args.runs, runs, refs, vals) if refs else ([], [])
    srows, serrs, numerical = score_rows(runs, args.scores, selection, vals)
    calls = val_calls(args.scores)
    tests_now = g.tests_record(Path(args.junit))
    f = g.gate_f(batches(args.runs, calls), recorded=record, tests_now=tests_now,
                 tests_before=record.get("tests") or {"tree": {"sha256": None}},
                 reason=args.tests_reason, repo=REPO_ROOT)
    freeze = dt.freeze_commit_of(REPO_ROOT, E022_DIR)
    descent = {name: is_ancestor(freeze, b.get("commit"))
               for name, b in batches(args.runs, calls).items()}
    n_batches = len(list(Path(args.runs).glob(BATCH_RECORDS)))
    need = math.ceil(len(trained) / BATCH_SIZE_CHAINS)
    status = {x: bool(rows) and all(r.get(x, True) for r in rows) for x in "abcdeg"}
    status |= {"f": bool(f["ok"] and descent and all(descent.values()) and n_batches >= need),
               "s": bool(srows["s"]) and all(r["ok"] for r in srows["s"]),
               "s3": bool(srows["s3"]) and all(r["ok"] for r in srows["s3"]),
               "p": bool(srows["p"]) and all(r["ok"] for r in srows["p"])}
    p = p_inputs(calls, record)
    status["p"] = status["p"] and p["ok"]
    entries = log_entries()
    r0 = gate_r0(calls, entries)
    r0["local"] = r0_local(args.e020_runs, entries)
    w = w_gate(args.scores, selection, excluded, dropped)
    w2 = w2_gate(args.scores, selection, runs)
    status |= {"r0": bool(r0["ok"] and r0["local"]["ok"]), "w": w["ok"], "w2": w2["ok"]}
    return {"time_utc": now_utc(), "infra_failed": excluded, "gate_failed": dropped,
            "runs_expected": len(keys),
            "errors": [*errors, *errs, *serrs], "chains": rows, "scores": srows,
            "numerically_failed": {**{sc.run_label(k): r.failed for k, r in runs.items()
                                      if r.failed}, **numerical},
            "f": f, "batch_records": {"found": n_batches, "needed": need},
            "freeze_commit": freeze, "freeze_descent": descent, "p_inputs": p,
            "r0": r0, "w": w, "w2": w2, "status": status,
            "ok": not (errors or errs or serrs) and all(status.values())}


# --- extraction gates, local part (PLAN.md 8-9) --------------------------------------------

def _csv(path: Path) -> list[dict]:
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def guard_commits(record: Mapping) -> dict[str, str | None]:
    return {k[len("guard_"):]: ((v or {}).get("descent") or {}).get("deployed_commit")
            for k, v in record.items() if k.startswith("guard_")}


def y2_paths(record: Mapping, datasets: Sequence[str] | None = None) -> dict[str, set[str]]:
    """The training-split paths of the (y2) sample (training manifest, E021 rule)."""
    out = {}
    for d in datasets or sc.DATASETS:
        y2 = record.get(f"y2_{d}") or {}
        m = load_manifest(manifest_file(d))
        out[d] = {rel_video_path(d, m, int(i)) for i in y2.get("sample") or ()}
    return out


def y2_status(record: Mapping, datasets: Sequence[str] | None = None) -> dict:
    """(y2) from the testsplit dense --repro records: every encoder bit-equal or within the
    tolerance, no problem."""
    datasets = list(datasets or sc.DATASETS)
    out = {}
    for d in datasets:
        comp = (record.get(f"y2_{d}") or {}).get("comparison") or {}
        out[d] = {e: {"status": c.get("status"), "rows": c.get("rows"),
                      "max_rel_l2": c.get("max_rel_l2"), "min_cos": c.get("min_cos")}
                  for e, c in comp.items()}
    ok = all(out[d] and all(v["status"] in (g.PASS_BIT, g.PASS_TOL) for v in out[d].values())
             for d in datasets)
    status = (g.FAIL if not ok else g.PASS_BIT if all(
        v["status"] == g.PASS_BIT for d in datasets for v in out[d].values()) else g.PASS_TOL)
    return {"datasets": out, "status": status, "ok": ok}


APPROVABLE = {"y1": "(추출 환경 변경)", "y2": "통과(사용자 승인 예외: {videos})",
              "y3": "통과(사용자 승인 예외: {videos})", "y4": "통과(사용자 승인 예외: {videos})"}


def approved(name: str, result: Mapping, approvals: Mapping) -> dict:
    """PLAN.md "시험 추출 관문": a failed (y2), (y3) or (y4) passes as an approved exception when
    the user approved it (results/e022_approvals.json: {"<gate>": {"decision": "<the user's
    words>", "videos": [...], "date": ...}}); (y1): another extraction environment, tagged
    "(추출 환경 변경)" ((y2) must then pass in it)."""
    a = approvals.get(name)
    if result.get("ok") or not a or name not in APPROVABLE:
        return dict(result)
    tag = APPROVABLE[name].format(videos=", ".join(map(str, a.get("videos") or [])))
    return dict(result) | {"ok": True, "status": "approved", "approval": dict(a), "tag": tag}


def extract_gates(record: Mapping, *, headers: Path, video_access: Path, extract_access: Path,
                  local_sha: Path | None, junit: Path, env_extract: Mapping,
                  tests_before: Mapping, reason: str | None = None,
                  approvals: Mapping | None = None) -> dict:
    """(t), (y0)-(y7) (PLAN.md "시험 추출 관문"); (y4)-(y6) as the server evaluated them
    (e022_score extract-gates); user-approved exceptions of (y1)-(y4) (``approved``)."""
    manifests = {d: dt.read_manifest(d) for d in sc.DATASETS}
    summaries = {}
    for d, m in manifests.items():
        prov = dt.read_provenance(d)
        summaries[d] = g.manifest_summary(d, m.names, m.categories, m.total_frames,
                                          {"source": {"sha256": prov["source"]},
                                           "annotation": {"sha256": prov["annotation"]}})
    first = record.get("guard_header") or record.get("guard_dense") or {}
    caches = {k: v.get("sha256") for k, v in (first.get("old_caches") or {}).items()
              if isinstance(v, Mapping)}
    out: dict = {"t": g.gate_t(summaries, caches)}
    tests_now = g.tests_record(Path(junit))
    commits = guard_commits(record)
    out["y0"] = {c: g.gate_y0(c, tests_now, tests_before, reason=reason, repo=REPO_ROOT)
                 for c in sorted({c for c in commits.values() if c})}
    out["y0"]["ok"] = bool(commits) and all(c and out["y0"][c]["ok"] for c in commits.values())
    out["y1"] = g.gate_y1(env_extract)
    out["y2"] = y2_status(record)
    out["y3"] = g.gate_y3(_csv(headers))
    for k in ("y4", "y5", "y6"):
        rows = {d: record.get(f"{k}_{d}") for d in sc.DATASETS}
        out[k] = {"datasets": {d: (r or {}).get("ok") for d, r in rows.items()},
                  "ok": all((r or {}).get("ok") is True for r in rows.values())}
    local = None
    if local_sha is not None:
        local = {(r["dataset"], r["path"]): r["sha256"] for r in _csv(local_sha)}
    out["y7"] = g.gate_y7(_csv(video_access), _csv(extract_access),
                          {d: dt.video_path_set(m) for d, m in manifests.items()},
                          y2_paths(record), local_sha256=local)
    for k in APPROVABLE:
        out[k] = approved(k, out[k], approvals or {})
    out["status"] = {k: bool(out[k]["ok"]) for k in ("t", "y0", "y1", "y2", "y3", "y4", "y5",
                                                     "y6", "y7")}
    out["tags"] = [out[k]["tag"] for k in APPROVABLE if out[k].get("tag")]
    out["driver_only_y1"] = out["y1"].get("status") == g.DRIVER_ONLY
    out["ok"] = all(out["status"].values())
    return out


# --- gates of the test scorings (PLAN.md 10-11) --------------------------------------------

def scoring_metas(scores: Path) -> list[dict]:
    files = sorted((Path(scores) / "test").glob("meta_*.json"), key=lambda f: int(f.stem[5:]))
    return [_json(f) | {"meta_file": f.name} for f in files]


def _meta_line(meta: Mapping) -> int:
    return int(str(meta["meta_file"])[len("meta_"):-len(".json")])


def scored_with(metas: Sequence[Mapping], kind: str, files: Mapping[str, str | None]) -> dict:
    """Each current score file of a kind ('<chain>/<arm>.npz' -> its sha256 now) bound to the
    call that wrote it: the latest call of the kind whose meta recorded the file with that
    sha256 (``written_sha256``, recorded right after each write). Bound when every file has a
    writer and every writer ran under one plan hash; the writers' deployed commits are listed
    and a change between them is surfaced (``commit_changed``): the access-log check allows it
    only as PLAN.md "코드 규칙" 1 exceptions 1 / 2 (``access_check``, --allow_code_change)."""
    mine = sorted((m for m in metas if m.get("kind") == kind), key=_meta_line)
    bound, unbound = {}, []
    for rel, sha in sorted(files.items()):
        writer = next((m for m in reversed(mine) if sha is not None
                       and (m.get("written_sha256") or {}).get(rel) == sha), None)
        if writer is None:
            unbound.append(rel)
        else:
            bound[rel] = writer
    writers = {m["meta_file"]: m for m in bound.values()}
    commits = sorted({str((m.get("code") or {}).get("commit")) for m in writers.values()})
    plans = sorted({str(((m.get("guard") or {}).get("plan") or {}).get("sha256"))
                    for m in writers.values()})
    return {"files": {rel: m["meta_file"] for rel, m in bound.items()}, "unbound": unbound,
            "meta": sorted(writers, key=lambda f: _meta_line({"meta_file": f})),
            "commits": commits, "commit": commits[0] if len(commits) == 1 else None,
            "commit_changed": len(commits) > 1,
            "plan_sha256": plans[0] if len(plans) == 1 else None,
            "ok": bool(files) and not unbound and len(plans) == 1 and "None" not in plans
            and "None" not in commits}


def e020_runs_rows(path: Path | None = None) -> dict:
    """(dataset, pair, seed, rule, readout) -> E020 m10 frame AUC / AP (points)."""
    out = {}
    for r in _csv(path or E020_RUNS_CSV):
        if r["cell"] == "m10" and not r.get("failed"):
            out[(r["dataset"], r["pair"], int(r["seed"]), r["rule"], r["readout"])] = {
                "frame_auc": float(r["frame_auc"]), "frame_ap": float(r["frame_ap"])}
    return out


def read_sha256_list(path: Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, rel = line.split(None, 1)
            out[rel.strip()] = sha
    return out


def e020_file(e020_scores: Path, chain: str, cell: str, rule: str, listed: Mapping) -> Path:
    """An E020 path (i) test score file, its sha256 checked against the committed list before it
    is read (PLAN.md)."""
    rel = f"{chain}/{cell}_{rule}_i.npz"
    f = Path(e020_scores) / "test" / rel
    if listed.get(rel) is None or sha256_file(f) != listed[rel]:
        raise ValueError(f"{rel}: sha256 differs from results/e020_test_scores.sha256")
    return f


def _rel_ok(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    r = compare_logits(np.asarray(a), np.asarray(b), V_REL)
    return bool(r["pass"]), r["max_rel_dz"]


def v_row(ours: Mapping, theirs: Mapping, gts: Sequence, want: Mapping | None,
          want_device: str) -> dict:
    """Gate (v) of one R0 run and rule: z + c, z and c within 1e-5 max(1, |value|), sigma within
    1e-6, frame AUC / AP (logit and sigma) within 1e-4 points of e020_runs.csv, path (i)."""
    row = {}
    ok = True
    for k in ("logits", "local_logits", "offsets"):
        same, worst = _rel_ok(ours[k], theirs[k])
        row[f"{k}_max_rel"] = worst
        ok = ok and same
    dp = np.abs(np.asarray(ours["probs"], np.float64) - np.asarray(theirs["probs"], np.float64))
    row["probs_max_abs"] = float(dp.max()) if np.isfinite(dp).all() else math.inf
    ok = ok and row["probs_max_abs"] <= V_SIGMA
    row["bit"] = all(np.asarray(ours[k]).dtype == np.asarray(theirs[k]).dtype
                     and np.array_equal(ours[k], theirs[k])
                     for k in ("logits", "local_logits", "offsets", "probs"))
    for ro, k in (("logit", "logits"), ("sigma", "probs")):
        got = frame_auc_ap(ours[k], gts)
        for m in ("frame_auc", "frame_ap"):
            row[f"{ro}_{m}"] = got[m]
            if want is not None and ro in want:
                d = abs(got[m] - want[ro][m])
                row[f"{ro}_{m}_diff"] = d
                ok = ok and d <= V_POINTS
    if want is None:
        ok = False
        row["reason"] = "no e020_runs.csv row"
    cond = str(ours["device"]).startswith(want_device) and int(ours["batch_size"]) == \
        sc.BATCH_SIZE
    row["path_i"] = cond
    return row | {"ok": bool(ok and cond)}


def v2_row(arr: Mapping, gts: Sequence, totals: Sequence[int]) -> dict:
    """Gate (v2): frame arrays through wvad.eval.frames = the expand_segments evaluator."""
    worst = 0.0
    for k in ("logits", "probs"):
        a = fr.pooled_frame_metrics(fr.frame_arrays(arr[k], totals, "segment"), gts)
        b = frame_auc_ap(arr[k], gts)
        worst = max(worst, *(abs(a[m] - b[m]) for m in ("frame_auc", "frame_ap")))
    return {"max_abs_diff": worst, "ok": worst <= V2_TOL}


def gates_v(scores: Path, e020_scores: Path, manifests: Mapping, selection: Mapping,
            want_device: str = "cuda") -> dict:
    """(v) and (v2) on the R0 files of scoring 1 (PLAN.md 10): every R0 run and rule with a
    checkpoint in the selection (the absent ones listed)."""
    listed = read_sha256_list(E020_SCORES_LIST)
    table = e020_runs_rows()
    jobs, absent = sc.jobs_of(Path("."), Path("."), selection, sc.CELLS["r0"])
    rows, v2rows, errors, current = [], [], [], {}
    gts = {d: manifests[d].frame_gts() for d in sc.DATASETS}
    for j in jobs:
        d, p, s = sc.split_chain(j.chain)
        m, rule = manifests[d], "best" if j.arm == "R0" else "last"
        head = {"run": j.label}
        f = sc.arm_path(scores, j.chain, j.arm)
        current[f"{j.chain}/{j.arm}.npz"] = sha256_file(f) if f.is_file() else None
        try:
            ours = _npz(f)
            if list(ours["names"]) != list(m.names) or str(ours["ckpt_sha256"]) != j.sha256:
                raise ValueError("names or checkpoint differ from the manifest / selection")
            theirs = _npz(e020_file(e020_scores, j.chain, sc.R0_CELL, rule, listed))
            want = {ro: table[(d, p, s, rule, ro)] for ro in READOUTS
                    if (d, p, s, rule, ro) in table}
            rows.append(head | v_row(ours, theirs, gts[d], want or None, want_device))
            v2rows.append(head | v2_row(ours, gts[d], m.total_frames))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{head['run']}: {exc!r}")
    bound = scored_with(scoring_metas(scores), "r0", current)
    n_bit = sum(r["bit"] for r in rows)
    ok = bool(rows) and not errors and all(r["ok"] for r in rows) and bound["ok"]
    v = {"rows": rows, "errors": errors, "absent": absent, "n": len(rows),
         "bit_equal_runs": n_bit,
         "status": ("pass_bit" if ok and n_bit == len(rows) else "pass_tolerance" if ok
                    else "fail"), "scored_with": bound, "ok": ok}
    v2 = {"rows": v2rows, "max_abs_diff": max((r["max_abs_diff"] for r in v2rows),
                                              default=None),
          "ok": bool(v2rows) and not errors and all(r["ok"] for r in v2rows)}
    return {"v": v, "v2": v2}


def expected_scoring_2(selection: Mapping) -> tuple[list[sc.Job], list[str]]:
    return sc.jobs_of(Path("."), Path("."), selection, sc.arms_of(sc.SCORING_2))


def v3_file(arr: Mapping, arm: str, m) -> dict:
    """Gate (v3) of one scoring-2 file: names and order = the restricted manifest read; snippet
    arms ceil(T / 16) rows per video, 32-unit arms (and C's d16m units) 32 per video; the
    non-finite values counted only (numerical failure)."""
    n = len(m)
    problems = []
    if list(np.asarray(arr.get("names", []))) != list(m.names):
        problems.append("names or order differ from the test manifest")
    if not np.array_equal(arr.get("total_frames", []), m.total_frames):
        problems.append("total_frames differ from the test manifest")
    if arm in sc.SNIPPET_ARMS:
        want = np.array([grid.n_snippets(int(t)) for t in m.total_frames])
        counts = np.asarray(arr.get("counts", []))
        if not np.array_equal(counts, want):
            problems.append("snippet counts differ from ceil(T / 16)")
        for k in ("z", "zc", "sigma"):
            if np.asarray(arr.get(k, [])).shape != (int(want.sum()),):
                problems.append(f"{k} does not hold sum n_v rows")
        if np.asarray(arr.get("c", [])).shape != (n,):
            problems.append("c is not one value per video")
    keys = ["logits", "probs", "local_logits"] if arm not in sc.SNIPPET_ARMS else []
    if arm == "C":
        keys += ["o32_logits", "o32_probs", "o32_local_logits", "s3t_z"]
    for k in keys:
        if np.asarray(arr.get(k, [])).shape != (n, 32):
            problems.append(f"{k} is not (videos, 32)")
    counts = sc.nonfinite(arr)
    return {"problems": problems, "nonfinite": counts,
            "numerical_failure": any(counts.values()), "ok": not problems}


def job_run(job: sc.Job) -> str:
    """The '<chain>/<head>_m10' run a score file belongs to (the name --gate_failed uses)."""
    return f"{job.chain}/{sc.ARMS[job.arm][0]}_{sc.PIPELINE}"


def gates_v3(scores: Path, selection: Mapping, gate_failed: Sequence[str] = ()) -> dict:
    """(v3) and (s3t) on the scoring-2 files (PLAN.md 11), before any metric: every expected
    file listed in scores.sha256 with its sha256; structure per file; (s3t) per C file. The files
    of the ``gate_failed`` runs (--gate_failed: dropped after an unfixed gate failure) are left
    out of the evaluation, the binding to their writers included, and recorded (``gate_failed``,
    ``skipped``): a failure confined to a dropped run does not stop E022, its seeds count as
    dropped for a reason other than the treatment (PLAN.md "주 검정")."""
    dropped = check_gate_failed(gate_failed, ())
    jobs, absent = expected_scoring_2(selection)
    skipped = sorted(j.label for j in jobs if job_run(j) in dropped)
    jobs = [j for j in jobs if j.label not in skipped]
    target = Path(scores) / "test"
    listed = read_sha256_list(target / "scores.sha256")
    manifests = {d: dt.read_manifest(d) for d in sc.DATASETS}
    rows, s3t, errors, current = {}, {}, [], {}
    for j in jobs:
        rel = f"{j.chain}/{j.arm}.npz"
        f = target / rel
        current[rel] = sha256_file(f) if f.is_file() else None
        try:
            if current[rel] is None or listed.get(rel) != current[rel]:
                raise ValueError("not listed in scores.sha256 with its sha256")
            arr = _npz(f)
            if str(arr["ckpt_sha256"]) != j.sha256:
                raise ValueError("checkpoint sha256 differs from the selection")
            rows[j.label] = v3_file(arr, j.arm, manifests[j.dataset])
            if j.arm == "C":
                s3t[j.label] = s3t_row(arr, rows[j.label])
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{j.label}: {exc!r}")
    errs = [r for r in rows.values() if not r["ok"]]
    s3_ok = bool(s3t) and all(r["ok"] for r in s3t.values())
    metas = scoring_metas(scores)
    bound = scored_with(metas, "scoring_2", current)
    mine = [m for m in metas if m.get("kind") == "scoring_2"]
    after = {m["meta_file"]: (m.get("train_inputs_after") or {}).get("ok") is True
             for m in mine if m.get("complete")}
    return {"v3": {"rows": rows, "errors": errors, "absent": absent, "files": len(rows),
                   "nonfinite_runs": sorted(k for k, r in rows.items()
                                            if r["numerical_failure"]),
                   "scored_with": bound, "commit_changed": bound["commit_changed"],
                   "train_inputs_after": after,
                   "calls_cut_short": [m["meta_file"] for m in mine if not m.get("complete")],
                   "gate_failed": dropped, "skipped": skipped,
                   "ok": bool(rows) and not errors and not errs and bound["ok"]
                   and bool(after) and all(after.values())},
            "s3t": {"rows": s3t, "max_rel_dz": max((r.get("max_rel_dz", 0.0)
                                                    for r in s3t.values()), default=None),
                    "gate_failed": dropped,
                    "skipped": [x for x in skipped if x.endswith("/C")], "ok": s3_ok}}


def s3t_row(arr: Mapping, v3: Mapping) -> dict:
    """(s3t) of one C file: the averaged-path z = the test d16m z (|dz| <= 1e-5 max(1, |z|)) and
    the server's s3_check passed; non-finite values: numerical failure, skipped."""
    if any(v3["nonfinite"].get(k) for k in ("o32_local_logits", "s3t_z")):
        return {"skipped": "non-finite (numerical failure)", "ok": True}
    r = compare_logits(arr["s3t_z"], arr["o32_local_logits"])
    server = json.loads(str(arr["s3t_check"]))
    return r | {"server": server, "ok": bool(r["pass"] and server.get("pass") is True)}


# --- gate record ---------------------------------------------------------------------------

def write_gate(section: str, result: Mapping, statuses: Mapping[str, bool]) -> dict:
    """Merge a stage's result and statuses into results/e022_gate.json (written also on a
    failure: it is a record)."""
    rec = _json_or(OUT["gate"], {"status": {}})
    rec["code"] = code_version()
    rec[section] = clean(result)
    rec["status"] = {**rec.get("status", {}), **{k: bool(v) for k, v in statuses.items()}}
    write_json(OUT["gate"], rec)
    return rec


LOCAL_GATES = {"v": ("B", "gates-v"), "v3": ("C", "gates-v3")}   # stage -> access, command
LOCAL_COMMANDS = tuple(c for _, c in LOCAL_GATES.values())


def logged_access(test_log: Path, access: str, command: str, files: Mapping, run,
                  precheck=None) -> dict:
    """A local test access (gates (v) / (v3): part of accesses (B) / (C)): a clean git working
    tree (the committed code is what runs: the pass that unlocks scoring 2 and the pass the
    analysis relies on come from it) and the frozen plan hash, then ``precheck()`` (a check of
    the arguments against the committed records, run on that clean tree), else a "refused" line
    and SystemExit; a "start" line before any test file is read, ``run()``, then an "end" line
    with its outcome (or "error")."""
    dt.outside_repo(test_log)
    code = code_version()
    line = {"access": access, "command": command, "commit": code.get("commit"), "code": code}
    try:
        require_clean_tree()
        plan = dt.plan_seal(E022_DIR)["sha256"]
        if precheck is not None:
            precheck()
    except SystemExit as exc:
        dt.append_test_log(test_log, line | {"time_utc": now_utc(), "event": "refused",
                                             "reason": str(exc)})
        raise
    line["plan_sha256"] = plan
    dt.append_test_log(test_log, line | {"time_utc": now_utc(), "event": "start",
                                         "files": dict(files)})
    try:
        res = run()
    except BaseException as exc:
        dt.append_test_log(test_log, line | {"time_utc": now_utc(), "event": "error",
                                             "error": repr(exc)})
        raise
    dt.append_test_log(test_log, line | {"time_utc": now_utc(), "event": "end",
                                         "ok": {k: v.get("ok") for k, v in res.items()}})
    return res


def gates_command(args: argparse.Namespace) -> None:
    if args.stage == "train":
        require_before_test_access(getattr(args, "test_log", None))    # not in train_gates
        res = train_gates(args)
        write_gate("train", res, res["status"])
        ok = res["ok"]
    elif args.stage == "extract":
        record = _json(OUT["extract"])
        tests_before = (_json(OUT["inputs"]).get("tests") or {"tree": {"sha256": None}})
        res = extract_gates(record, headers=args.headers, video_access=args.video_access,
                            extract_access=args.extract_access, local_sha=args.local_sha,
                            junit=args.junit, env_extract=_json(args.env_extract),
                            tests_before=tests_before, reason=args.tests_reason,
                            approvals=_json_or(OUT["approvals"], {}))
        record["gates"] = clean(res) | {"code": code_version(), "time_utc": now_utc()}
        write_json(OUT["extract"], record)
        ok = res["ok"]
    elif args.stage == "v":
        def run_v() -> dict:
            manifests = {d: load_eval_manifest(eval_manifest_file(d)) for d in sc.DATASETS}
            return gates_v(args.scores, args.e020_scores, manifests, _json(OUT["selection"]),
                           args.device)
        files = {"scores": str(Path(args.scores) / "test"), "e020_scores": str(
            args.e020_scores), "manifests": list(dt.MANIFEST_FILES.values())}
        res = logged_access(args.test_log, *LOCAL_GATES["v"], files, run_v)
        write_gate("v", res["v"], {"v": res["v"]["ok"]})
        write_gate("v2", res["v2"], {"v2": res["v2"]["ok"]})
        ok = res["v"]["ok"] and res["v2"]["ok"]
    else:
        dropped = check_gate_failed(args.gate_failed, args.infra_failed)
        files = {"scores": str(Path(args.scores) / "test"),
                 "manifests (restricted read)": list(dt.MANIFEST_FILES.values()),
                 "left out (--gate_failed)": dropped}
        res = logged_access(args.test_log, *LOCAL_GATES["v3"], files,
                            lambda: gates_v3(args.scores, _json(OUT["selection"]), dropped),
                            precheck=lambda: require_committed_gate_failed(dropped))
        write_gate("v3", res["v3"], {"v3": res["v3"]["ok"]})
        write_gate("s3t", res["s3t"], {"s3t": res["s3t"]["ok"]})
        ok = res["v3"]["ok"] and res["s3t"]["ok"]
    print(f"gates {args.stage}: {'passed' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(f"gate stage {args.stage} failed (PLAN.md failure rules); see the record")


# --- access log (analysis) ------------------------------------------------------------------

def fixed_code_same(freeze: str | None, commit: str | None) -> bool | None:
    """The fixed server code and the identity sets at ``commit`` equal the freeze commit's."""
    if not freeze or not commit:
        return None
    res = git("diff", "--quiet", freeze, commit, "--", *FIXED_SERVER)
    same = {0: True, 1: False}.get(res.returncode)
    ident = g.code_identity(commit, ("S_train", "S_score", "S_extract"), repo=REPO_ROOT)
    return bool(same and ident["ok"]) if same is not None else None


SCORE_CODE = "src/wvad/diagnostics/e022_score.py"   # the one fixed file exceptions 1, 2 may fix


def fixed_code_changes(freeze: str | None, commit: str | None) -> dict:
    """The fixed server files that differ between the freeze commit and ``commit``, and whether
    the identity sets (S_train, S_score, S_extract) are unchanged there (code rule 0)."""
    if not freeze or not commit:
        return {"changed": None, "identity_ok": False}
    res = git("diff", "--name-only", freeze, commit, "--", *FIXED_SERVER)
    ident = g.code_identity(commit, ("S_train", "S_score", "S_extract"), repo=REPO_ROOT)
    return {"changed": res.stdout.split() if res.returncode == 0 else None,
            "identity_ok": bool(ident["ok"])}


def local_gate_failed_before(entries: Sequence[Mapping], line: int, command: str,
                       gates: Sequence[str]) -> bool:
    """The last "end" line of the local gate ``command`` before log line ``line`` (1-based)
    shows one of ``gates`` failed (``ok`` false). A passed gate, an "error" or "refused" line,
    or no line at all is not a failed gate."""
    for e in reversed(entries[:line - 1]):
        if e.get("command") == command and e.get("event") == "end":
            ok = e.get("ok") or {}
            return any(ok.get(x) is False for x in gates)
    return False


def exception_entry(row: Mapping, entries: Sequence[Mapping]) -> str | None:
    """The re-read with which a call on changed fixed code may begin (PLAN.md "코드 규칙" 1),
    each only after the gate that names its cause failed: exception 1, R0's second full read
    (--rescore) after a forward-path fix of e022_score.py, when the last gate (v) call before it
    ended with (v) failed; exception 2, a scoring-2 --runs re-read (--rescore) after a writing /
    indexing fix, when the last gate (v3) call before it ended with (v3) or (s3t) failed."""
    if (row["access"] == "B" and row["kind"] == "r0" and row["mode"] == "full"
            and row["prior_full_r0"] >= 1 and row["rescore"]
            and local_gate_failed_before(entries, row["line"], LOCAL_GATES["v"][1], ("v",))):
        return "exception 1"
    if (row["access"] == "C" and row["kind"] == "scoring_2" and row["mode"] == "runs"
            and row["rescore"]
            and local_gate_failed_before(entries, row["line"], LOCAL_GATES["v3"][1],
                                         ("v3", "s3t"))):
        return "exception 2"
    return None


def logged_dirty(e: Mapping) -> bool:
    """The call's logged code version had uncommitted tracked changes (a local line's ``code``,
    a server line's access-device ``descent``)."""
    descent = (e.get("guard") or {}).get("descent") or {}
    return (e.get("code") or {}).get("dirty") is True or descent.get("dirty") is True


def _call_row(i: int, e: Mapping, plan: str, freeze: str | None, prior_full_r0: int) -> dict:
    commit = (e.get("deployed_commit") or e.get("commit")
              or ((e.get("guard") or {}).get("descent") or {}).get("deployed_commit"))
    plan_e = e.get("plan_sha256") or ((e.get("guard") or {}).get("plan") or {}).get("sha256")
    return {"line": i, "access": e.get("access"), "command": e.get("command"),
            "kind": e.get("kind"), "mode": e.get("mode"), "rescore": e.get("rescore"),
            "prior_full_r0": prior_full_r0, "commit": commit, "dirty": logged_dirty(e),
            "descends_from_freeze": is_ancestor(freeze, commit), "plan_ok": plan_e == plan,
            "fixed_code_same": fixed_code_same(freeze, commit)}


def _d_rule(entries: Sequence[Mapping], tests: Sequence[int]) -> dict:
    """Access (D): the first analysis after every test read of (A)-(C); a later one only as a
    recorded re-computation (--recompute REASON; PLAN.md "(D)", code rules)."""
    d = [(i, e) for i, e in enumerate(entries, 1) if e.get("access") == "D"
         and e.get("event") == "start"]
    first = d[0][0] if d else None
    return {"d_starts": [i for i, _ in d],
            "tests_before_d": first is None or not tests or max(tests) < first,
            "d_later_with_reason": all(bool(e.get("recompute")) for _, e in d[1:])}


def access_check(entries: Sequence[Mapping], metas: Sequence[Mapping], plan: str,
                 freeze: str | None, allow_change: bool = False) -> dict:
    """The access log: every test call (A, B, C; the local gates (v) / (v3) included) on the
    freeze commit or a descendant, from committed code (a logged ``dirty`` fails the call), with
    the frozen plan hash and the fixed code of the freeze commit; with --allow_code_change a
    changed e022_score.py only (identity sets unchanged) from the re-read of exception 1 or 2 on
    (``exception_entry``: after the failed gate that names its cause). Extraction before scoring
    1, scoring 1 before scoring 2, R0 read in full at most twice, scoring 2 in full once,
    gate (v) logged after the last R0 read and before scoring 2, gate (v3) after the last
    scoring-2 call; every (A)-(C) read before the first (D), a later (D) only with a reason;
    every scoring meta names its own start line."""
    starts = [(i, e) for i, e in enumerate(entries, 1) if e.get("event") == "start"
              and e.get("access") in ("A", "B", "C")]
    rows, entered, full_r0 = [], None, 0
    for i, e in starts:
        row = _call_row(i, e, plan, freeze, full_r0)
        permitted = False
        if not row["fixed_code_same"] and allow_change:
            entered = entered or exception_entry(row, entries)
            ch = fixed_code_changes(freeze, row["commit"])
            permitted = bool(entered and row["access"] in ("B", "C") and ch["identity_ok"]
                             and ch["changed"] is not None
                             and set(ch["changed"]) <= {SCORE_CODE})
            row["code_change"] = ch | {"exception": entered, "permitted": permitted}
        row["ok"] = bool(not row["dirty"] and row["descends_from_freeze"] and row["plan_ok"]
                         and (row["fixed_code_same"] or permitted))
        rows.append(row)
        full_r0 += row["access"] == "B" and row["mode"] == "full"
    server = [r for r in rows if r["command"] not in LOCAL_COMMANDS]
    a = [r["line"] for r in server if r["access"] == "A"]
    b = [r for r in server if r["access"] == "B"]
    c = [r for r in server if r["access"] == "C"]
    gv = [r["line"] for r in rows if r["command"] == LOCAL_GATES["v"][1]]
    gv3 = [r["line"] for r in rows if r["command"] == LOCAL_GATES["v3"][1]]
    last_r0 = max((r["line"] for r in b if r["mode"] == "full"), default=None)
    order = {"access_0": any(e.get("access") == "0" and e.get("event") == "end"
                             for e in entries),
             "extraction_before_scoring": bool(a) and bool(b) and max(a) < min(
                 r["line"] for r in b),
             "scoring_1_before_2": bool(b) and bool(c) and max(
                 r["line"] for r in b) < min(r["line"] for r in c),
             "r0_full_reads": sum(r["mode"] == "full" for r in b),
             "scoring_2_full_reads": sum(r["mode"] == "full" for r in c),
             "v_logged": last_r0 is not None and any(
                 last_r0 < x < min((r["line"] for r in c), default=math.inf) for x in gv),
             "v3_logged": bool(c) and any(x > max(r["line"] for r in c) for x in gv3),
             **_d_rule(entries, [r["line"] for r in rows])}
    started = [e for _, e in starts]
    metas_ok = all(m.get("access_log_entry") in started
                   and (m.get("access_log_entry") or {}).get("deployed_commit")
                   == (m.get("code") or {}).get("commit") for m in metas)
    logs = {m.get("access_log") for m in metas}
    ok = (all(r["ok"] for r in rows) and order["access_0"]
          and order["extraction_before_scoring"] and order["scoring_1_before_2"]
          and 1 <= order["r0_full_reads"] <= 2 and order["scoring_2_full_reads"] == 1
          and order["v_logged"] and order["v3_logged"] and order["tests_before_d"]
          and order["d_later_with_reason"] and metas_ok and len(logs) == 1)
    return {"freeze_commit": freeze, "calls": rows, **order, "metas_match_log": metas_ok,
            "one_access_log": len(logs) == 1, "ok": bool(ok)}


# --- test data, runs and metrics -------------------------------------------------------------

@dataclass
class EvalSet:
    dataset: str
    names: tuple
    labels: np.ndarray
    totals: np.ndarray
    gts: list = field(repr=False)

    @property
    def anomalous(self) -> np.ndarray:
        return np.flatnonzero(self.labels == 1)


def eval_sets() -> dict[str, EvalSet]:
    """The full test manifests (access (D): the interval fields are read here only)."""
    out = {}
    for d in sc.DATASETS:
        m = load_eval_manifest(eval_manifest_file(d))
        out[d] = EvalSet(d, tuple(m.names), np.asarray(m.labels), np.asarray(m.total_frames),
                         m.frame_gts())
    return out


@dataclass
class Units:
    """One run's per-video unit values: ``kind`` segment / window (32 per video, evaluator
    windows) or snippet (n_v per video, supports)."""

    kind: str
    logit: list
    z: list | None = None
    c: np.ndarray | None = None
    prob: list | None = None

    def values(self, readout_: str) -> list:
        if readout_ == "logit":
            return self.logit
        if readout_ == "sigma":
            return self.prob if self.prob is not None else [sigmoid32(x) for x in self.logit]
        if readout_ == "z":
            return self.z
        if readout_ == "c":
            return [np.full(len(x), v, dtype=np.float32) for x, v in zip(self.logit, self.c,
                                                                           strict=True)]
        raise ValueError(readout_)


def split_rows(flat: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return [np.asarray(flat[a:a + n], dtype=np.float32) for a, n in zip(starts, counts,
                                                                          strict=True)]


def arm_units(arm: str, arr: Mapping, totals: np.ndarray) -> Units:
    """The readout units of an arm from its score file (C.W and C.o32 from C's)."""
    if arm not in sc.SNIPPET_ARMS and arm not in FILE_OF:
        return Units("segment", list(arr["logits"]), list(arr["local_logits"]),
                     np.asarray(arr["offsets"], dtype=np.float32), list(arr["probs"]))
    gamma = float(arr["gamma"])
    zs = split_rows(arr["z"], np.asarray(arr["counts"]))
    if arm == "C.W":
        w = [readout(z, int(t), gamma, "W") for z, t in zip(zs, totals, strict=True)]
        return Units("window", [r.logits for r in w], None,
                     np.array([r.offset for r in w], dtype=np.float32))
    if arm == "C.o32":
        c32 = np.array([offset(u, gamma) for u in arr["o32_local_logits"]], dtype=np.float32)
        return Units("snippet", [(z + c).astype(np.float32) for z, c in zip(zs, c32,
                                                                           strict=True)],
                     zs, c32)
    rd = [readout(z, int(t), gamma, "G") for z, t in zip(zs, totals, strict=True)]
    return Units("snippet", [r.logits for r in rd], zs,
                 np.array([r.offset for r in rd], dtype=np.float32))


def frames_of(units: Units, readout_: str, totals: np.ndarray) -> list[np.ndarray]:
    return fr.frame_arrays(units.values(readout_), totals, units.kind)


def run_metrics(units: Units, ts: EvalSet, readout_: str) -> dict:
    """Frame AUC / AP (points) over all test videos, the anomalous videos only (pooled), video
    AUC (max), and for the logit readout the within-video L and its opposite kind."""
    frames = frames_of(units, readout_, ts.totals)
    full = fr.pooled_frame_metrics(frames, ts.gts)
    anom = ts.anomalous
    only = fr.pooled_frame_metrics([frames[i] for i in anom], [ts.gts[i] for i in anom])
    vals = units.values(readout_)
    out = {"frame_auc": full["frame_auc"], "frame_ap": full["frame_ap"],
           "anom_frame_auc": only["frame_auc"], "anom_frame_ap": only["frame_ap"],
           "video_auc": 100 * max_video_auc(vals, ts.labels)}
    if readout_ == "logit":
        within = fr.within_video(frames, ts.gts, ts.labels, fr.L_METRIC[ts.dataset])
        other = fr.within_video(frames, ts.gts, ts.labels, fr.OPPOSITE[ts.dataset])
        out |= {"L": within["value"], "L_excluded": within["n_excluded"],
                "L_opposite": other["value"], "L_per_video": within["per_video"]}
    return out


RunKey = tuple[str, str, int, str]            # (dataset, pair, seed, arm)


@dataclass
class Loaded:
    units: dict = field(default_factory=dict)      # RunKey -> Units
    files: dict = field(default_factory=dict)      # RunKey -> score-file sha256 / epoch
    nonfinite: dict = field(default_factory=dict)  # RunKey -> bool


def load_test(scores: Path, sets: Mapping[str, EvalSet], selection: Mapping,
              gate_failed: Sequence[str] = ()) -> Loaded:
    """Every test score file of the two scorings, bound to its selected checkpoint, except the
    files of the ``gate_failed`` runs (--gate_failed): like (v3), the analysis does not open
    them, so a structure failure or an absent file confined to a dropped run does not stop it
    (``failures`` counts the run's arms as gate failures)."""
    out = Loaded()
    dropped = set(gate_failed)
    jobs, _ = sc.jobs_of(Path("."), Path("."), selection, [*sc.CELLS["r0"],
                                                            *sc.arms_of(sc.SCORING_2)])
    for j in jobs:
        if job_run(j) in dropped:
            continue
        ts = sets[j.dataset]
        arr = _npz(sc.arm_path(scores, j.chain, j.arm))
        if str(arr["ckpt_sha256"]) != j.sha256 or list(arr["names"]) != list(ts.names):
            raise ValueError(f"{j.label}: not the selected checkpoint or not the manifest order")
        d, p, s = sc.split_chain(j.chain)
        counts = sc.nonfinite(arr)
        arms = [j.arm, *(a for a, src in FILE_OF.items() if src == j.arm)]
        for a in arms:
            bad = any(counts.get(k) for k in ARM_KEYS.get(a, UNIT_KEYS))
            out.nonfinite[(d, p, s, a)] = bad
            out.files[(d, p, s, a)] = {"epoch": int(arr["epoch"]), "file_arm": j.arm}
            if not bad:
                out.units[(d, p, s, a)] = arm_units(a, arr, ts.totals)
    return out


def metrics_of(loaded: Loaded, sets: Mapping[str, EvalSet]) -> dict:
    """(dataset, pair, seed, arm, readout) -> metrics; C and R0 also the z-only and c-only
    readouts (descriptive 3, the z-only flag)."""
    out = {}
    for key, units in loaded.units.items():
        ts = sets[key[0]]
        names = list(READOUTS)
        if key[3] in ("C", "R0"):
            names += ["z", "c"]
        for ro in names:
            out[(*key, ro)] = run_metrics(units, ts, ro)
    return out


def values(metrics: Mapping, dataset: str, readout_: str, metric: str,
           arms: Sequence[str] = ALL_ARMS) -> dict:
    """(pair, seed, arm) -> metric of one dataset and readout (failed runs absent)."""
    return {(k[1], k[2], k[3]): m[metric] for k, m in metrics.items()
            if k[0] == dataset and k[4] == readout_ and k[3] in arms and metric in m}


# --- failures ------------------------------------------------------------------------------

def _history_values(d: Path) -> list[float]:
    """The training losses and V-AUC of a run's history.json (every epoch). A missing,
    unreadable or malformed history (no epoch, a value that is not a number) is not a numerical
    failure: SystemExit (a wrong --runs / --e020_runs path or an incomplete copy)."""
    f = Path(d) / "history.json"
    try:
        hist = _json(f)
        values = [v for h in hist for k, v in h.items()
                  if k.startswith("loss_") or k == "val_video_auc"]
        if not hist or not values or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            raise ValueError("no epoch, or a loss / V-AUC that is not a number")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise SystemExit(f"{f}: history missing or unreadable ({exc!r}): a wrong runs path or "
                         "an incomplete copy, not a numerical failure (PLAN.md)") from exc
    return [float(v) for v in values]


def history_failed(d: Path) -> bool:
    """PLAN.md "수치 실패": a non-finite training loss or V-AUC in the parsed history (only
    then); an absent or unreadable history stops the analysis (``_history_values``)."""
    return not all(math.isfinite(v) for v in _history_values(d))


def _finite(x) -> bool:
    return bool(np.isfinite(np.asarray(x, dtype=np.float64)).all())


def val_failed(scores: Path | None, e020_runs: Path, chain: str) -> set[str]:
    """Arms whose rule checkpoint has non-finite validation scores (PLAN.md "수치 실패(체크포인트
    규칙별)"): C.last / R1.last at epoch 30 (32-unit record, and for HB the dense logits and
    gamma), B / R1 at the training's best epoch; R0 / R0.last from the E020 record. (C, C.G,
    A1.S: the selection fails on any non-finite dense value.)"""
    out = set()
    if scores is not None:
        d, p, s = sc.split_chain(chain)
        for head, best_arm, last_arm in (("hb", "B", "C.last"), ("h1", "R1", "R1.last")):
            f = sc.val_path(scores, es.RunKey(d, p, s, head, sc.PIPELINE))
            if not f.is_file():
                continue
            sf = _npz(f)
            own = sf[f"{sc.HEAD_FEATURES[head]}_probs"]
            last = [own[-1], *((sf["dense_z"][-1], sf["gamma"][-1]) if head == "hb" else ())]
            if not all(_finite(x) for x in last):
                out.add(last_arm)
            best = int(sf["best_epoch"])
            if best >= 1 and not _finite(own[best - 1]):
                out.add(best_arm)
    rec = Path(e020_runs) / chain / sc.R0_CELL / "record_val_scores.npz"
    if rec.is_file():
        z = _npz(rec)
        sel = _json_or(rec.parent / "selection.json", {})
        epochs = list(z["epochs"])
        if 30 in epochs and not _finite(z["scores"][epochs.index(30)]):
            out.add("R0.last")
        best = sel.get("best_epoch")
        if best is not None and best in epochs and not _finite(z["scores"][epochs.index(best)]):
            out.add("R0")
    return out


def failures(runs: Path, e020_runs: Path, selection: Mapping, loaded: Loaded,
             infra: Sequence[str], gate_failed: Sequence[str],
             scores: Path | None = None) -> list[st.Failure]:
    """The runs left out per arm (PLAN.md "수치 실패", "인프라 실패", "관문 실패"): an
    infrastructure or gate failure of a chain drops its head's arms (its history is not read);
    a non-finite history (either stage) every arm of the head; no selected checkpoint
    (non-finite validation dense scores, or no best epoch) the arm; non-finite validation
    scores of the rule's checkpoint (``val_failed``) or non-finite test scores ((v3) counts)
    the arm. A missing or unreadable history stops (``history_failed``)."""
    out = []
    for d in sc.DATASETS:
        for p in sc.PAIRS:
            for s in sc.SEEDS:
                chain = sc.chain_name(d, p, s)
                bad_val = val_failed(scores, e020_runs, chain)
                for head, arms in (("hb", HEAD_ARMS["hb"]), ("h1", HEAD_ARMS["h1"]),
                                   ("h0", ("R0", "R0.last"))):
                    label = f"{chain}/{head}_m10" if head != "h0" else f"{chain}/R0"
                    if label in infra or label in gate_failed:
                        kind = st.INFRASTRUCTURE if label in infra else st.GATE
                        out += [st.Failure(d, p, s, a, kind) for a in arms]
                        continue
                    if head == "h0":
                        hist = history_failed(Path(e020_runs) / chain / sc.R0_CELL) or \
                            history_failed(Path(e020_runs) / chain / "m10_stage1")
                    else:
                        hist = (history_failed(Path(runs) / label) or history_failed(
                            Path(runs) / f"{label}_stage1"))
                    out += [st.Failure(d, p, s, a, st.NUMERICAL) for a in arms
                            if hist or (d, p, s, a) not in loaded.units or a in bad_val]
    return out


def drop_failed(loaded: Loaded, fails: Sequence[st.Failure]) -> Loaded:
    """The loaded runs without the failed ones: a run left out of its rule (numerical,
    infrastructure or gate failure) enters no metric, test or descriptive (PLAN.md)."""
    gone = {(f.dataset, f.pair, f.seed, f.arm) for f in fails}
    return Loaded({k: v for k, v in loaded.units.items() if k not in gone}, dict(loaded.files),
                  dict(loaded.nonfinite))


# --- bootstrap, E020 baselines -----------------------------------------------------------------

def bootstraps(loaded: Loaded, sets: Mapping[str, EvalSet], primary: Mapping,
               decisions: st.Decisions = st.RECORDED, n_resamples: int = st.BOOT_R) -> dict:
    """Video bootstrap of C - R0 (logit readout, primary metric) over the complete seeds;
    resample 0 must reproduce the evaluator's values."""
    out = {}
    for d in sc.DATASETS:
        ts, prim = sets[d], primary[d]
        seeds, _ = st.seed_combination(prim, {"C": 1, "R0": -1})
        atoms = {}
        for p in sc.PAIRS:
            for s in seeds:
                atoms[(p, s, "C")] = _atoms(loaded.units[(d, p, s, "C")], ts)
                atoms[(p, s, "R0")] = _atoms(loaded.units[(d, p, s, "R0")], ts)
        res = st.bootstrap_contrast(atoms, ts.labels, PRIMARY[d], seeds, "C", "R0",
                                    n_resamples=n_resamples,
                                    levels=(decisions.boot_level, 0.95),
                                    point={k: prim[k] for k in atoms})
        if seeds and not res["point_check_max_abs_diff"] <= BOOT_POINT_TOL:
            raise ValueError(f"{d}: bootstrap resample 0 differs from the evaluator by "
                             f"{res['point_check_max_abs_diff']}")
        out[d] = res
    return out


def _atoms(units: Units, ts: EvalSet):
    def build():
        if units.kind == "snippet":
            return st.snippet_atoms(units.logit, ts.gts)
        return st.window_atoms(np.stack(units.logit), ts.gts)
    return build


def e020_values(e020_scores: Path, sets: Mapping[str, EvalSet]) -> tuple[dict, list[dict]]:
    """D7 (a): the E020 l20 and base cells (best rule, path (i)), logit readout, primary
    metric; every file's sha256 checked against the committed E020 list before it is read."""
    listed = read_sha256_list(E020_SCORES_LIST)
    out, read = {d: {} for d in sc.DATASETS}, []
    for d in sc.DATASETS:
        ts = sets[d]
        for p in sc.PAIRS:
            for s in sc.SEEDS:
                chain = sc.chain_name(d, p, s)
                for cell, arm in (("l20", "L20"), ("base", "base")):
                    f = e020_file(e020_scores, chain, cell, "best", listed)
                    arr = _npz(f)
                    read.append({"file": f"{chain}/{cell}_best_i.npz", "sha256": listed[
                        f"{chain}/{cell}_best_i.npz"]})
                    if not np.isfinite(arr["logits"]).all():
                        continue
                    frames = fr.frame_arrays(arr["logits"], ts.totals, "segment")
                    out[d][(p, s, arm)] = fr.pooled_frame_metrics(frames, ts.gts)[PRIMARY[d]]
    return out, read


# --- analysis --------------------------------------------------------------------------------

def libraries() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__}


def committed(path: Path) -> bool:
    rel = Path(path).relative_to(REPO_ROOT).as_posix()
    return (git("cat-file", "-e", f"HEAD:{rel}").returncode == 0
            and not git("status", "--porcelain", "--", rel).stdout.strip())


def code_changes(freeze: str) -> list[str]:
    res = git("diff", "--name-only", freeze, "HEAD", "--", "src", "data/manifests")
    if res.returncode != 0:
        raise SystemExit(f"git diff from the freeze commit {freeze} failed")
    return res.stdout.split()


def records_check(scores: Path) -> dict:
    """PLAN.md: the analysis needs the committed (v3) / (s3t) passes, the score sha256 list
    (equal to the copy's), the selection and the extraction gates."""
    gate = _json_or(OUT["gate"], {})
    listed = Path(scores) / "test" / "scores.sha256"
    out = {"gate_committed": committed(OUT["gate"]) if OUT["gate"].is_file() else False,
           "v3_s3t_passed": (gate.get("status") or {}).get("v3") is True
           and (gate.get("status") or {}).get("s3t") is True,
           "scores_list_committed": committed(OUT["scores"]) if OUT["scores"].is_file()
           else False,
           "scores_list_equal": OUT["scores"].is_file() and listed.is_file()
           and OUT["scores"].read_text(encoding="utf-8") == listed.read_text(encoding="utf-8"),
           "selection_committed": committed(OUT["selection"]),
           "extraction_gates": ((_json_or(OUT["extract"], {}).get("gates") or {}).get("ok")
                                is True)}
    out["ok"] = all(out.values())
    return out


def run_rows(metrics: Mapping, loaded: Loaded, fails: Sequence[st.Failure],
             info: Mapping) -> list[dict]:
    """results/e022_runs.csv: one row per run and readout (metrics in points)."""
    failed = {(f.dataset, f.pair, f.seed, f.arm): f.kind for f in fails}
    rows = []
    for d in sc.DATASETS:
        for p in sc.PAIRS:
            for s in sc.SEEDS:
                for a in ALL_ARMS:
                    base = {"dataset": d, "pair": p, "seed": s, "arm": a,
                            **(info.get((d, p, s, a)) or {})}
                    if (d, p, s, a) in failed:
                        rows.append(base | {"failed": failed[(d, p, s, a)]})
                        continue
                    for ro in ("logit", "sigma", "z", "c"):
                        m = metrics.get((d, p, s, a, ro))
                        if m is not None:
                            rows.append(base | {"readout": ro, "failed": "", **{
                                k: v for k, v in m.items() if k != "L_per_video"}})
    return rows


def selection_info(selection: Mapping, runs: Path, e020_runs: Path) -> dict:
    """Per run: the selected epoch and its validation V-AUC (points; C, C.G, A1.S: the
    readout's full-validation V-AUC; B, R1, R0: the training selection; .last: epoch 30)."""
    out = {}
    for label, rec in (selection.get("chains") or {}).items():
        chain, run = label.split("/")
        d, p, s = sc.split_chain(chain)
        for arm, ck in (rec.get("checkpoints") or {}).items():
            kind = {"C": "W", "C.G": "G", "A1.S": "W"}.get(arm)
            if kind:
                v = rec["readouts"][kind]["selected_v_auc"]
            else:
                v = _selected_auc(Path(runs) / chain / run, arm.endswith(".last"))
            out[(d, p, s, arm)] = {"epoch": ck.get("epoch"), "val_v_auc": 100 * v
                                   if v is not None else None}
        if "C" in (rec.get("checkpoints") or {}):
            out[(d, p, s, "C.W")] = out[(d, p, s, "C")]
            out[(d, p, s, "C.o32")] = out[(d, p, s, "C")]
    for chain, rec in (selection.get("r0") or {}).items():
        d, p, s = sc.split_chain(chain)
        for arm in ("R0", "R0.last"):
            v = _selected_auc(Path(e020_runs) / chain / sc.R0_CELL, arm == "R0.last")
            out[(d, p, s, arm)] = {"epoch": (rec.get(arm) or {}).get("epoch"),
                                   "val_v_auc": 100 * v if v is not None else None}
    return out


def _selected_auc(d: Path, last: bool) -> float | None:
    try:
        if last:
            return float(_json(d / "history.json")[-1]["val_video_auc"])
        return float(_json(d / "selection.json")["best_val_video_auc"])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def analysis(args: argparse.Namespace, gate: Mapping, stamp: Mapping) -> dict:
    """Metrics, failures, bootstrap, the evaluation (verdict, flags, condition (2), decomposition)
    and the descriptives."""
    selection = _json(OUT["selection"])
    sets = eval_sets()
    loaded = load_test(args.scores, sets, selection, args.gate_failed)
    fails = failures(args.runs, args.e020_runs, selection, loaded, args.infra_failed,
                     args.gate_failed, args.scores)
    loaded = drop_failed(loaded, fails)
    metrics = metrics_of(loaded, sets)

    def by_d(ro: str, arms: Sequence[str] = ALL_ARMS, metric: Mapping | None = None) -> dict:
        return {d: values(metrics, d, ro, (metric or PRIMARY)[d], arms) for d in sc.DATASETS}

    primary = by_d("logit")
    within = {d: values(metrics, d, "logit", "L", ("C", "R0")) for d in sc.DATASETS}
    boot = bootstraps(loaded, sets, primary, n_resamples=args.boot_resamples)
    e020, e020_read = e020_values(args.e020_scores, sets)
    res = st.evaluate(primary, sigma=by_d("sigma", ("C", "R0")), z_only=by_d("z", ("C", "R0")),
                      within=within, boot=boot, failures=fails, e020=e020)
    tags = []
    if (gate.get("train") or {}).get("f", {}).get("tag"):
        tags.append(gate["train"]["f"]["tag"])
    tags += list(((_json_or(OUT["extract"], {}).get("gates") or {}).get("tags")) or [])
    info = selection_info(selection, args.runs, args.e020_runs)
    summary = {"check": "E022 stage F: dense confirmation C - R0 (PLAN.md 판정)", **stamp,
               "gate": gate.get("status"), "margins": st.MARGIN, "tags": tags,
               "primary": {"metric": PRIMARY, "readout": "logit (z + c)",
                           "checkpoints": "C: W full-validation V-AUC; R0: ckpt_best_val.pt"},
               "reported": desc.reported(metrics, ("C", "R0")),
               **clean(res), "e020_files_read": e020_read, "complete": False}
    if summary.get("verdict") != st.BETTER_SUB:
        summary["note"] = ("not 'better (substantive)': secondary, per-setting and "
                           "decomposition results are not claimed as improvements (PLAN.md)")
    write_json(OUT["summary"], clean(summary))                 # 1. verdict, flags, condition (2)
    write_csv(OUT["runs"], [clean(r) for r in run_rows(metrics, loaded, fails, info)])
    summary["descriptives"] = desc.descriptives(
        metrics=metrics, loaded=loaded, sets=sets, info=info,
        extract_record=_json_or(OUT["extract"], {}), gate=gate, scores=args.scores,
        e020_runs=args.e020_runs)
    summary["complete"] = True
    write_json(OUT["summary"], clean(summary))                 # 2. descriptives
    return summary


def prior_analyses(test_log: Path) -> list[dict]:
    """The (D) start lines of earlier analyses (the committed copy and the local log)."""
    return [e for e in merged_entries(OUT["access"], test_log)
            if e.get("access") == "D" and e.get("event") == "start"]


def analysis_gates(args: argparse.Namespace, seal: Mapping) -> dict:
    """Inside access (D): the committed records, every gate again ((v), (v2), (v3), (s3t) read
    the test scores and the manifest intervals; (v3), (s3t) without the --gate_failed runs) and
    the access log (with this analysis's (D) start line); SystemExit after recording when one
    fails (PLAN.md failure rules). --gate_failed is first compared with the committed train
    record (``require_committed_gate_failed``): ``write_gate("train", ...)`` at the end
    rewrites that section from these arguments."""
    require_committed_gate_failed(args.gate_failed)
    records = records_check(args.scores)
    train = train_gates(args)
    manifests = {d: load_eval_manifest(eval_manifest_file(d)) for d in sc.DATASETS}
    vres = gates_v(args.scores, args.e020_scores, manifests, _json(OUT["selection"]),
                   args.device)
    v3 = gates_v3(args.scores, _json(OUT["selection"]), args.gate_failed)
    access = access_check(merged_entries(OUT["access"], args.test_log),
                          scoring_metas(args.scores), seal["sha256"], seal["freeze_commit"],
                          bool(args.allow_code_change))
    gate = write_gate("train", train, train["status"])
    for name, part in (("v", vres["v"]), ("v2", vres["v2"]), ("v3", v3["v3"]),
                       ("s3t", v3["s3t"]), ("access", access), ("records", records)):
        gate = write_gate(name, part, {name: part["ok"]})
    if not all(gate["status"].values()):
        raise SystemExit(f"a gate, a record or the access log failed: "
                         f"{[k for k, v in gate['status'].items() if not v]}; see "
                         f"{OUT['gate'].name} (PLAN.md failure rules)")
    return gate


def analyze_command(args: argparse.Namespace) -> None:
    """Access (D), once: the refusals first (clean tree, seal history, code changed since the
    freeze, --gate_failed other than step 7's list, an earlier analysis), then the (D) start
    line before any test score or interval is read, the gates, the analysis and the end (or
    error) line. A refusal reads no test data and so leaves the once-only analysis unused."""
    code = require_clean_tree()
    seal = seal_history()
    changed = code_changes(seal["freeze_commit"])
    if changed and not args.allow_code_change:
        raise SystemExit(f"{changed} changed after the freeze commit: --allow_code_change "
                         "'<reason>' (and the reason in the plan's change log)")
    dt.outside_repo(args.test_log)
    require_committed_gate_failed(args.gate_failed)
    prior = prior_analyses(args.test_log)
    if (OUT["summary"].exists() or prior) and not args.recompute:
        raise SystemExit(f"the analysis ran (once, PLAN.md; summary {OUT['summary'].exists()}, "
                         f"(D) start lines {len(prior)}): --recompute '<reason>' only in the "
                         "cases of the code rules")
    stamp = {"code": code, "plan": {"file": "PLAN.md", "sha256": seal["sha256"]},
             "seal": seal, "code_changed_since_freeze": changed,
             "allow_code_change": args.allow_code_change, "recompute": args.recompute,
             "libraries": libraries()}
    line = {"access": "D", "command": "analyze", "commit": code.get("commit"),
            "plan_sha256": seal["sha256"], "recompute": args.recompute}
    dt.append_test_log(args.test_log, line | {
        "time_utc": now_utc(), "event": "start", "code": code,
        "allow_code_change": args.allow_code_change,
        "files": {"scores": str(Path(args.scores) / "test"),
                  "manifests": list(dt.MANIFEST_FILES.values()),
                  "e020_scores": str(args.e020_scores), "e020_runs": str(args.e020_runs)}})
    try:
        gate = analysis_gates(args, seal)
        summary = analysis(args, gate, stamp)
    except BaseException as exc:
        dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "error",
                                                  "error": repr(exc)})
        raise
    dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "end",
                                              "verdict": summary.get("verdict"),
                                              "e020_files_read": summary.get("e020_files_read")})
    print(summary.get("verdict_plan"), (summary.get("condition_2") or {}).get("plan"),
          summary.get("flags_raised_plan"))


# --- command line ------------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser, scores: bool = True) -> None:
    p.add_argument("--runs", type=Path, required=True, help="local copy of ~/wvad_runs/e022")
    if scores:
        p.add_argument("--scores", type=Path, required=True,
                       help="local copy of the e022 score root (val/, test/)")
    p.add_argument("--infra_failed", nargs="*", default=[],
                   help="'<chain>/<head>_m10' whose rerun also failed (infrastructure)")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e022")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="write PLAN.sha256 (PLAN.md before its change log)")
    pf = sub.add_parser("prefreeze", help="(3c): tests record, server environments, driver")
    pf.add_argument("--junit", required=True, type=Path)
    pf.add_argument("--env_train", required=True, type=Path)
    pf.add_argument("--env_extract", required=True, type=Path)
    lg = sub.add_parser("log", help="merge the access logs into results/e022_test_access.log")
    lg.add_argument("parts", nargs="+", type=Path)
    gt = sub.add_parser("gates")
    gt.add_argument("--stage", required=True, choices=("train", "extract", "v", "v3"),
                    help="train: the gates of the training and validation outputs (step 7), only "
                         "before the first test read: refused, the gate record untouched, once "
                         "results/e022_gate.json holds a (v), (v2), (v3) or (s3t) section or the "
                         "committed access-log copy or --test_log has an (A)-(D) line (access (0) "
                         "lines do not count); extract: the extraction gates (9); v: gates (v), "
                         "(v2) (10); v3: gates (v3), (s3t) (11)")
    gt.add_argument("--runs", type=Path, help="train: local copy of ~/wvad_runs/e022")
    gt.add_argument("--scores", type=Path, help="local copy of the e022 score root")
    gt.add_argument("--e021_runs", type=Path, help="train: local copy of ~/wvad_runs/e021")
    gt.add_argument("--e020_runs", type=Path,
                    help="train: local copy of ~/wvad_runs/e020 (R0 histories, (r0))")
    gt.add_argument("--e020_scores", type=Path, help="v: local copy of the E020 score root")
    gt.add_argument("--test_log", type=Path,
                    help="v / v3: the local test access log (outside the repository); train: "
                         "optional, an (A)-(D) line in it refuses the stage")
    gt.add_argument("--junit", type=Path, help="train / extract: a pytest JUnit XML of tests/")
    gt.add_argument("--tests_reason", help="why the tests/ tree differs from the pre-freeze one")
    gt.add_argument("--infra_failed", nargs="*", default=[])
    gt.add_argument("--gate_failed", nargs="*", default=[],
                    help="train / v3: '<chain>/<head>_m10' dropped after an unfixed gate "
                         "failure (left out of the gates and recorded); v3: exactly the list "
                         "--stage train recorded")
    gt.add_argument("--headers", type=Path, help="extract: the (y3) header CSV")
    gt.add_argument("--video_access", type=Path, help="extract: results/e022_video_access.csv")
    gt.add_argument("--extract_access", type=Path,
                    help="extract: results/e022_extract_access.csv")
    gt.add_argument("--local_sha", type=Path, help="extract: the local sha256 CSV (header "
                                                  "--sha_only), (y7)")
    gt.add_argument("--env_extract", type=Path, help="extract: results/env_extract_test.json")
    gt.add_argument("--device", default="cuda", help="v: device prefix of path (i)")
    an = sub.add_parser("analyze")
    _common(an)
    an.add_argument("--e021_runs", type=Path, required=True)
    an.add_argument("--e020_runs", type=Path, required=True,
                    help="local copy of ~/wvad_runs/e020 (R0 histories)")
    an.add_argument("--e020_scores", type=Path, required=True)
    an.add_argument("--junit", type=Path, required=True)
    an.add_argument("--tests_reason")
    an.add_argument("--test_log", type=Path, required=True,
                    help="the local test access log (outside the repository)")
    an.add_argument("--gate_failed", nargs="*", default=[],
                    help="'<chain>/<head>_m10' dropped after an unfixed gate failure: exactly "
                         "the list gates --stage train recorded")
    an.add_argument("--allow_code_change", metavar="REASON")
    an.add_argument("--recompute", metavar="REASON")
    an.add_argument("--device", default="cuda")
    an.add_argument("--boot_resamples", type=int, default=st.BOOT_R)
    args = ap.parse_args(argv)
    if args.command == "gates":
        need = {"train": ("runs", "scores", "e021_runs", "e020_runs", "junit"),
                "extract": ("headers", "video_access", "extract_access", "junit",
                            "env_extract"),
                "v": ("scores", "e020_scores", "test_log"),
                "v3": ("scores", "test_log")}[args.stage]
        missing = [k for k in need if getattr(args, k) is None]
        if missing:
            ap.error(f"--stage {args.stage} needs {', '.join('--' + m for m in missing)}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    if args.command == "freeze":
        freeze_command(args)
    elif args.command == "prefreeze":
        rec = prefreeze(args.junit, _json(args.env_train), _json(args.env_extract))
        if not rec["driver"]["same"]:
            print(f"DRIVER DIFFERS from E021 ({rec['driver']}): tell the user before the freeze "
                  "and record the handling under the plan's user decisions (PLAN.md 3c)")
        print("prefreeze record written:", OUT["inputs"])
    elif args.command == "log":
        print(merge_logs(args.parts), "lines ->", OUT["access"])
    elif args.command == "gates":
        gates_command(args)
    else:
        analyze_command(args)


if __name__ == "__main__":
    main()
