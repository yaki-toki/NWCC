"""E023 (stage F2): local commands of the pre-registered localisation comparison
(experiments/E023_localisation_baselines/PLAN.md). Analysis code fixed at the freeze commit
(PLAN.md "코드 규칙" 3); the statistics are wvad.diagnostics.e023_stats, the frame arrays
wvad.eval.frames (through E022's pure loaders), the decomposition wvad.eval.decompose and the
descriptives wvad.diagnostics.e023_desc. New code (add-only rule).

    freeze     PLAN.sha256 = sha256 of PLAN.md before "## 변경 기록" (PLAN.md committed unmodified
               at HEAD, clean tracked tree; once); refused while D5 / D6 are undecided
               (``decisions_check``: e023_stats.RECORDED and the plan's decision table)
    prefreeze  (3e) the tests/ tree and run summary of a pytest JUnit XML and the server's
               training environment, merged into results/e023_prefreeze.json (key "3e")
    q7         (q7) one port's code review and its Jev judgement, merged into
               results/e023_q1b.json under "q7" -> <method> (--method, --review FILE outside
               the repository, hashed and named only, --jev P, --verdict TEXT, --findings ...)
    gates      --stage prefreeze  (q2), (q3) (per dataset), (q5), (q6), (q7) and the (q1d)
                                  record (complete, every family method; e023_score q1d)
               --stage train      (a), (b), (d), (f), (g), (p), (s), (w), (w2), (n); only before
                                  the first test read (refused once the gate record holds a (v3)
                                  or (k) section or an access log holds an (A) or (D) line)
               --stage v3         (v3) and (k) on the access (A) files, before any metric: a
                                  logged part of access (A) (clean working tree, frozen plan,
                                  "start" / "end" lines); structure only (restricted manifest
                                  read, no interval)
               results/e023_gate.json (written also when a gate fails)
    analyze    access (D), once (clean working tree): the refusals first (seal history, code
               changed since the freeze, D5 / D6 undecided, --gate_failed, an earlier summary or
               (D) start line without --recompute), the (D) start line before any test score or
               interval is
               read, the committed records ((v3) / (k) passed, the score sha256 list), gate (h)
               (E022 C / C.last score files and C's selected checkpoints against E022's lists),
               gate (h2) (C / C.last L and pooled main metric per setting and seed equal to
               e022_summary.json descriptives.1_cells within 1e-9), gate (d1) (the decomposition
               identity of every run within 1e-12 and, for UCF, the decomposition's AUC equal to
               the evaluator's pooled frame AUC within 1e-12), then condition (2), condition (3),
               the
               mechanism, the flags, the problem description and the descriptives (D14
               included): results/e023_summary.json (verdict first), results/e023_runs.csv; a
               failed (h), (h2) or (d1) writes a minimal summary with the category "관문 실패(판정
               없음)" and stops

Score roots (local copies): E023 <scores>/test/<chain>/<run>__<rule>.npz (access (A)), E022
<e022_scores>/test/<chain>/{C,C.last}.npz (read only after their sha256 matched
e022_test_scores.sha256). C's frame scores are E022's: snippet logits z + c(v) (G readout); the new
arms' snippet logits z (c(v) = 0), each repeated over the snippet supports.
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
import torch

from wvad.baselines import train as bt
from wvad.baselines.recipes import dataset_recipe
from wvad.data import dense_test as dt
from wvad.data import load_split, split_file
from wvad.diagnostics import e020_stats as st20
from wvad.diagnostics import e022 as e22
from wvad.diagnostics import e022_gates as g22
from wvad.diagnostics import e022_stats as st22
from wvad.diagnostics import e023_access as ea
from wvad.diagnostics import e023_desc as desc
from wvad.diagnostics import e023_gates as gt
from wvad.diagnostics import e023_score as sc
from wvad.diagnostics import e023_stats as st
from wvad.diagnostics.common import code_version, write_csv, write_json
from wvad.eval import decompose as dc
from wvad.eval import frames as fr
from wvad.eval.metrics import DEFAULT_MIN_FRAMES
from wvad.paths import REPO_ROOT

E023_DIR, RESULTS = gt.E023_DIR, gt.RESULTS
PLAN, SEAL = E023_DIR / "PLAN.md", E023_DIR / "PLAN.sha256"
CHANGELOG = dt.CHANGELOG
OUT = gt.FILES
E022_RESULTS = dt.RESULTS
E022_SCORES_LIST = E022_RESULTS / "e022_test_scores.sha256"
E022_SELECTION = E022_RESULTS / "e022_selection.json"
E022_SUMMARY = E022_RESULTS / "e022_summary.json"
PRIMARY = dict(fr.PRIMARY)
H2_TOL, D1_TOL, BOOT_POINT_TOL = 1e-9, dc.RESIDUAL_TOL, 1e-6
D1_AUC_TOL = 1e-12                    # (d1): |decomposition AUC - evaluator frame AUC| (UCF)
DECIDED_BEFORE_FREEZE = ("D5", "D6")
TEST_GATE_SECTIONS = ("v3", "k")
TEST_ACCESSES = ("A", "D")
LOCAL_V3 = "gates-v3"
FREEZE_PARENT = {"plan": ("PLAN.md",),
                 "records": ("results/e023_licences.json", "results/e023_recipes.json",
                             "results/e023_train_inputs.json", "results/e023_blocked_ckpts.json",
                             "results/e023_prefreeze.json", "results/e023_q1b.json",
                             "results/e023_test_access.log"),
                 "repository": ("README.md", "src/wvad/baselines/common.py",
                                "src/wvad/baselines/train.py", "src/wvad/eval/decompose.py",
                                *(f"src/wvad/diagnostics/{m}.py" for m in (
                                    "e023", "e023_score", "e023_gates", "e023_stats",
                                    "e023_desc", "e023_access", "e023_prefreeze")))}


def now_utc() -> str:
    return dt_time.datetime.now(dt_time.timezone.utc).isoformat(timespec="seconds")


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_or(path: Path, default):
    try:
        return _json(path)
    except (OSError, ValueError):
        return default


def _npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def clean(obj):
    return e22.clean(obj)


def git(*args: str) -> subprocess.CompletedProcess:
    return g22.git(*args, repo=REPO_ROOT)


def git_bytes(*args: str) -> bytes:
    res = subprocess.run(["git", "-c", f"safe.directory={Path(REPO_ROOT).as_posix()}", *args],
                         cwd=REPO_ROOT, capture_output=True, check=False)
    return res.stdout if res.returncode == 0 else b""


# --- seal -----------------------------------------------------------------------------------------

def require_clean_tree() -> dict:
    code = code_version()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed (PLAN.md); code version {code}")
    return code


def plan_answers(plan: Path = PLAN, keys: Sequence[str] = DECIDED_BEFORE_FREEZE
                 ) -> dict[str, str | None]:
    """The user-answer column of the plan's decision table rows ``| D5 | ... |`` (None: no
    row)."""
    out: dict[str, str | None] = {k: None for k in keys}
    for line in Path(plan).read_text(encoding="utf-8").splitlines():
        for k in keys:
            if line.startswith(f"| {k} |"):
                cells = line.strip().strip("|").split("|")
                out[k] = cells[-1].strip()
    return out


def decisions_check(plan: Path = PLAN, decisions: st.Decisions | None = None) -> dict:
    """D5 and D6 answered (PLAN.md "사용자 결정": before the freeze): the decisions the code uses
    (e023_stats.RECORDED) are decided and the plan's decision table holds an answer, not
    "미정", in the D5 and D6 rows."""
    dec = st.RECORDED if decisions is None else decisions
    answers = plan_answers(plan)
    open_rows = sorted(k for k, v in answers.items() if v is None or "미정" in v)
    return {"code_decided": bool(dec.decided), "plan_answers": answers,
            "plan_undecided": open_rows, "ok": bool(dec.decided) and not open_rows}


def require_decisions(plan: Path = PLAN) -> dict:
    res = decisions_check(plan)
    if not res["ok"]:
        raise SystemExit(f"D5 / D6 are undecided ({res}): record the answers in PLAN.md "
                         "'사용자 결정 기록' and e023_stats.RECORDED before the freeze / analysis")
    return res


def freeze_command(_args: argparse.Namespace) -> None:
    require_clean_tree()
    require_decisions()
    rel = PLAN.relative_to(REPO_ROOT).as_posix()
    if (git("cat-file", "-e", f"HEAD:{rel}").returncode != 0
            or git("status", "--porcelain", "--", rel).stdout.strip()):
        raise SystemExit(f"{rel} must be committed and unmodified at HEAD before the freeze")
    if SEAL.exists():
        raise SystemExit(f"{SEAL} exists: the plan is frozen once")
    h = dt.plan_sha256(PLAN)
    SEAL.write_text(f"{h}  PLAN.md (before {CHANGELOG})\n", encoding="utf-8", newline="\n")
    print(f"PLAN.sha256 written ({h}); commit it alone: that commit is the freeze commit")


def seal_history() -> dict:
    """PLAN.sha256 was added once, by the freeze commit, which changes nothing else and whose
    parent tree holds the plan, the code, README and the pre-freeze records (PLAN.md "고정
    규칙"); the current plan has the frozen hash."""
    rel = {n: (E023_DIR / n).relative_to(REPO_ROOT).as_posix() for n in ("PLAN.sha256",
                                                                         "PLAN.md")}
    touched = git("log", "--format=%H", "--", rel["PLAN.sha256"]).stdout.split()
    freeze = dt.freeze_commit_of(REPO_ROOT, E023_DIR)
    if len(touched) != 1 or touched[0] != freeze:
        raise SystemExit(f"PLAN.sha256 was touched by {touched}: only the freeze commit adds it")
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root",
                  freeze).stdout.split()
    if changed != [rel["PLAN.sha256"]]:
        raise SystemExit(f"the freeze commit {freeze} changes {changed}: PLAN.sha256 alone")
    base = E023_DIR.relative_to(REPO_ROOT).as_posix()
    parent = [*(f"{base}/{p}" for k in ("plan", "records") for p in FREEZE_PARENT[k]),
              *FREEZE_PARENT["repository"]]
    absent = [p for p in parent if git("cat-file", "-e", f"{freeze}^:{p}").returncode != 0]
    if absent:
        raise SystemExit(f"the freeze commit's parent lacks {absent} (PLAN.md 고정 규칙)")
    frozen = git_bytes("show", f"{freeze}:{rel['PLAN.sha256']}").decode("utf-8").split()
    at_freeze = hashlib.sha256(dt.plan_frozen_part(git_bytes(
        "show", f"{freeze}:{rel['PLAN.md']}"))).hexdigest()
    now = dt.plan_seal(E023_DIR)["sha256"]
    if not frozen or not frozen[0] == now == at_freeze:
        raise SystemExit(f"plan hash now {now}, frozen {frozen[:1]}, at the freeze {at_freeze}")
    return {"freeze_commit": freeze, "sha256": now, "seal_commits": touched,
            "freeze_changes": changed, "parent_holds": parent}


def code_changes(freeze: str) -> list[str]:
    res = git("diff", "--name-only", freeze, "HEAD", "--", "src", "data/manifests")
    if res.returncode != 0:
        raise SystemExit(f"git diff from the freeze commit {freeze} failed")
    return res.stdout.split()


def committed(path: Path) -> bool:
    rel = Path(path).relative_to(REPO_ROOT).as_posix()
    return (git("cat-file", "-e", f"HEAD:{rel}").returncode == 0
            and not git("status", "--porcelain", "--", rel).stdout.strip())


# --- (3e) ---------------------------------------------------------------------------------------

def prefreeze(junit: Path, env_train: Mapping, out: Path = OUT["prefreeze"]) -> dict:
    """(3e): the tests/ tree and run summary and the server's training environment against
    E022's, merged into results/e023_prefreeze.json (key "3e")."""
    want = sc.e022_training_env()
    rec = _json_or(out, {})
    rec["3e"] = {"tests": g22.tests_record(Path(junit)), "environment": dict(env_train),
                 "e022_environment": want,
                 "differences": [k for k in gt.TRAIN_ENV_KEYS if env_train.get(k) != want.get(k)],
                 "precision": "comparison methods fp32 (recipes); C AMP (E022, disclosed)",
                 "utc": now_utc()}
    write_json(Path(out), rec)
    return rec["3e"]


# --- gate record ------------------------------------------------------------------------------

def write_gate(section: str, result: Mapping, statuses: Mapping[str, bool]) -> dict:
    rec = _json_or(OUT["gate"], {"status": {}})
    rec["code"] = code_version()
    rec[section] = clean(result)
    rec["status"] = {**rec.get("status", {}), **{k: bool(v) for k, v in statuses.items()}}
    write_json(OUT["gate"], rec)
    return rec


def merged_entries(*parts: Path | None) -> list[dict]:
    texts = [Path(p).read_text(encoding="utf-8") for p in parts if p is not None
             and Path(p).is_file()]
    out = []
    for line in ea.merged_lines(texts):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def require_before_test_access(test_log: Path | None = None) -> None:
    """``gates --stage train`` runs only before the first test read (it records the dropped
    runs that (v3) and the analysis hold to)."""
    record = _json_or(OUT["gate"], {})
    found = [f"{OUT['gate'].name} holds {s!r}" for s in TEST_GATE_SECTIONS if s in record]
    letters = sorted({e.get("access") for e in merged_entries(OUT["access"], test_log)
                      if e.get("access") in TEST_ACCESSES})
    if letters:
        found.append(f"an access log has {', '.join(f'({x})' for x in letters)} lines")
    if found:
        raise SystemExit(f"gates --stage train runs only before the first test read: "
                         f"{'; '.join(found)}")


def require_committed_gate_failed(names: Sequence[str]) -> list[str]:
    """--gate_failed of (v3) and analyze is exactly the committed train record's list."""
    rec = _json_or(OUT["gate"], {})
    want = sorted((rec.get("train") or {}).get("gate_failed") or [])
    if sorted(set(names)) != want:
        raise SystemExit(f"--gate_failed {sorted(set(names))} differs from the train gates' list "
                         f"{want}")
    return want


# --- pre-freeze gates ---------------------------------------------------------------------------

def prefreeze_gates(args: argparse.Namespace) -> dict:
    """(q2), (q3) (every method on both datasets: common.method_on), (q5), (q6), (q7) and the
    (q1d) record (the ported methods through the registry)."""
    from wvad.baselines import common as bc

    methods = list(args.methods)
    licences = _json(OUT["licences"])
    recipes_rec = _json(OUT["recipes"])
    recipes = {m: {d: bc.method_on(m, d).recipe.to_json() for d in gt.DATASETS}
               for m in methods}
    defaults = {}
    for m in methods:
        cfg = bt.parse(["--method", m, "--unit", str(bc.get_method(m).official_unit),
                        "--dataset", "ucf", "--pair", "xcb_xcl", "--seed", "0", "--out_dir", "x"])
        defaults[m] = {"lr_mult": cfg.lr_mult, "scale_mult": cfg.scale_mult}
    q5 = {}
    for item in args.q5 or []:
        name, _, dirs = item.partition("=")
        a, _, b = dirs.partition(",")
        q5[name] = gt.gate_q5(Path(a), Path(b))
    access0 = _json(args.access0)
    totals = {d: access0[d]["total_frames"] for d in gt.DATASETS}
    recorded = {d: access0[d]["chunk_targets"].get(str(args.n_chunk), [])
                for d in gt.DATASETS}
    sources = {"train": Path(bt.__file__).read_text(encoding="utf-8"),
               "e023_score": Path(sc.__file__).read_text(encoding="utf-8")}
    q6 = gt.gate_q6(args.n_chunk, totals, None, recorded, sources)
    q1b = _json_or(OUT["q1b"], {})
    q7 = gt.gate_q7((q1b.get("q7") or {}), methods)
    res = {"q2": gt.gate_q2(licences), "q3": gt.gate_q3(recipes_rec, recipes, defaults),
           "q5": {"chains": q5, "ok": bool(q5) and all(v["ok"] for v in q5.values())},
           "q6": q6, "q7": q7, "q1d": q1d_check(q1b.get("q1d"), methods)}
    res["ok"] = all(v["ok"] for v in res.values())
    return res


def q1d_check(rec: Mapping | None, methods: Sequence[str]) -> dict:
    """The (q1d) record is present, complete and covers every family method (its flags are
    labels, not a pass condition: PLAN.md "실패 처리")."""
    rows = (rec or {}).get("methods") or {}
    missing = sorted(m for m in methods if m not in rows)
    ok = bool(rec) and bool(rec.get("complete")) and not missing
    return {"recorded": bool(rec), "complete": bool((rec or {}).get("complete")),
            "missing_methods": missing,
            "flagged": sorted(m for m, r in rows.items() if r.get("flag")), "ok": ok}


def record_q7(method: str, review: Path, jev: float, verdict: str,
              findings: Sequence[str] = (), out: Path | None = None) -> dict:
    """(q7): one port's code review (a file outside the repository, recorded by name and
    sha256) and its Jev judgement, merged into results/e023_q1b.json "q7" -> ``method``."""
    from wvad.baselines import common as bc

    if method not in gt.PLAN_NAMES:
        raise SystemExit(f"--method {method!r}: not an E023 method")
    review = Path(review)
    dt.outside_repo(review)
    if not review.is_file():
        raise SystemExit(f"{review}: no review file")
    if not 0.0 <= float(jev) <= 1.0:
        raise SystemExit(f"--jev {jev}: a probability")
    entry = {"jev": float(jev), "verdict": str(verdict), "findings": list(findings),
             "review": review.name, "review_sha256": sc.sha256_file(review),
             "utc": now_utc(), "code": code_version()}
    bc.update_json(out or OUT["q1b"],
                   lambda r: r.setdefault("q7", {}).__setitem__(method, entry),
                   default={"check": "E023 (q1b) official-pipeline comparisons (PLAN.md 관문)",
                            "methods": {}})
    return entry


# --- training and selection gates ---------------------------------------------------------------

def _cfg_of(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)["config"]


def baseline_labels(selection: Mapping, dropped: Sequence[str]) -> list[tuple[str, Mapping]]:
    return [(k, e) for k, e in (selection.get("chains") or {}).items()
            if e.get("role") != "mechanism" and k not in dropped
            and e.get("failed") != "infrastructure (PLAN.md)"]


def cl20_labels(selection: Mapping, dropped: Sequence[str]) -> list[tuple[str, Mapping]]:
    return [(k, e) for k, e in (selection.get("chains") or {}).items()
            if e.get("role") == "mechanism" and k not in dropped
            and e.get("failed") != "infrastructure (PLAN.md)"]


def gate_s_rows(runs: Path, scores: Path, selection: Mapping, book: Mapping,
                dropped: Sequence[str]) -> dict:
    """(s) for every chain from the --split val files."""
    rows = {}
    for label, _e in baseline_labels(selection, dropped):
        chain, run = label.split("/")
        sf = _npz(gt.val_file(scores, chain, run))
        stored = _npz(Path(runs) / label / bt.FILES["val"])["z"]
        rescored = {r: sf[f"z_{r}"] for r in gt.RULES if f"z_{r}" in sf}
        points = {r: int(sf[f"point_{r}"]) for r in rescored}
        shas = {r: f"{sf[f'sha256_recorded_{r}']}|{sf[f'sha256_{r}']}" for r in rescored}
        rec_book = (book.get("chains") or {}).get(label)
        eval_rec = None if rec_book is None else {k: v["sha256"] for k, v in
                                                 rec_book["points"].items()}
        eval_cur = None if "eval_points" not in sf else dict(zip(
            [str(x) for x in sf["eval_points"]], [str(x) for x in sf["eval_sha256"]],
            strict=True))
        rows[label] = gt.gate_s(rescored, stored, points, shas, eval_cur, eval_rec)
    for label, e in cl20_labels(selection, dropped):
        chain, run = label.split("/")
        sf = _npz(gt.val_file(scores, chain, run))
        last = (e.get("checkpoints") or {}).get("last") or {}
        rows[label] = gt.gate_s({"last": sf["last_dense_z"]}, sf["dense_z"],
                                {"last": gt.EVAL_POINTS},
                                {"last": f"{last.get('sha256', '')}|{sf['last_sha256']}"})
    return {"chains": rows, "ok": bool(rows) and all(r["ok"] for r in rows.values())}


def stored_logits(runs: Path, scores: Path, label: str, entry: Mapping) -> dict:
    """The stored validation logits of a chain (points, rows), counts, totals, labels."""
    chain, run = label.split("/")
    if entry.get("role") == "mechanism":
        sf = _npz(gt.val_file(scores, chain, run))
        return {"z": sf["dense_z"], "counts": sf["dense_counts"], "totals": sf["total_frames"],
                "labels": sf["labels"], "steps": sf["epochs"], "val_indices": sf["val_indices"]}
    z = _npz(Path(runs) / label / bt.FILES["val"])
    return {"z": z["z"], "counts": z["counts"], "totals": z["total_frames"],
            "labels": z["labels"], "steps": z["eval_steps"], "val_indices": z["val_indices"]}


def train_gates(args: argparse.Namespace) -> dict:
    """(a), (b), (d), (f), (g), (p), (s), (w), (w2), (n) (PLAN.md "학습·선택 관문"); runs named
    by --gate_failed are left out and recorded."""
    selection, book = _json(OUT["selection"]), _json_or(OUT["all_ckpts"], {"chains": {}})
    arms = gt.arms_from_json(selection["arms"])
    dropped = sorted(set(args.gate_failed))
    devlr, recipes = _json(OUT["devlr"]), _json(OUT["recipes"])
    n_chunk = int(args.n_chunk)
    base = baseline_labels(selection, dropped)
    configs = {k: _json(Path(args.runs) / k / bt.FILES["config"]) for k, _ in base}
    res: dict = {"gate_failed": dropped}
    res["a"] = gt.gate_a(configs, arms, devlr, recipes, n_chunk)
    refs = {f"{d}/{p}": _cfg_of(Path(args.e021_runs) / gt.chain_name(d, p, 0) / gt.CL20_RUN /
                                "ckpt_last.pt") for d in gt.DATASETS for p in gt.PAIRS}
    cl20 = {k: _cfg_of(Path(args.runs) / k / "ckpt_last.pt") for k, _ in
            cl20_labels(selection, dropped)}
    res["a_cl20"] = gt.gate_a_cl20(cl20, refs)
    reps = {}
    for k, _ in base:
        rep = Path(args.runs) / "repeat" / k
        if (rep / bt.FILES["record"]).is_file():
            reps[k] = gt.gate_b(Path(args.runs) / k, rep)
    covered = {(a.name, d) for k, _ in base for a in [arms[selection["chains"][k]["arm"]]]
               for d in [gt.split_chain(k.split("/")[0])[0]] if k in reps}
    want_cov = {(a.name, d) for a in arms.values() if a.role == "main" for d in gt.DATASETS}
    res["b"] = {"chains": reps, "missing": sorted(f"{a}/{d}" for a, d in want_cov - covered),
                "ok": bool(reps) and want_cov <= covered and all(r["ok"] for r in reps.values())}
    splits = {d: load_split(split_file(d)) for d in gt.DATASETS}
    stored = {k: stored_logits(args.runs, args.scores, k, e) for k, e in
              (*base, *cl20_labels(selection, dropped))}
    res["d"] = {"chains": {k: gt.gate_d(v["val_indices"], splits[gt.split_chain(
        k.split("/")[0])[0]].val_indices) for k, v in stored.items()}}
    res["d"]["ok"] = bool(stored) and all(r["ok"] for r in res["d"]["chains"].values())
    envs = {Path(f).stem: _json(f) for f in sorted(Path(args.runs).glob("env_e023_b*.json"))}
    commits = sorted({e.get("commit") for e in envs.values()})
    inputs = {"ok": bool(envs) and all((e.get("inputs_check") or {}).get("ok")
                                       for e in envs.values())}
    tests_before = (_json_or(OUT["prefreeze"], {}).get("3e") or {}).get("tests") or \
        {"tree": {"sha256": None, "files": {}}}
    res["f"] = gt.gate_f(commits[0] if len(commits) == 1 else None, inputs=inputs, env=envs,
                         want_env=sc.e022_training_env(),
                         dtypes={k: c.get("dtype") for k, c in configs.items()},
                         tests_now=g22.tests_record(Path(args.junit)),
                         tests_before=tests_before, reason=args.tests_reason)
    res["g"] = {"chains": {k: gt.gate_g(Path(args.runs) / k, bt.reselect) for k, _ in base}}
    res["g"]["ok"] = bool(base) and all(r["ok"] for r in res["g"]["chains"].values())
    res["p"] = {"chains": {k: gt.gate_p(configs[k], arms[e["arm"]], dataset_recipe(
        recipes["methods"][arms[e["arm"]].method], gt.split_chain(k.split("/")[0])[0]))
                           for k, e in base}}
    res["p"]["ok"] = bool(base) and all(r["ok"] for r in res["p"]["chains"].values())
    res["s"] = gate_s_rows(args.runs, args.scores, selection, book, dropped)
    w, w2 = {}, {}
    for k, v in stored.items():
        entry = selection["chains"][k]
        again = sc.readout_selection(v["z"], v["counts"], v["totals"], v["labels"], v["steps"])
        w[k] = gt.gate_w(again, entry)
        w2[k] = {"z_points": v["z"], "counts": v["counts"], "totals": v["totals"],
                 "labels": v["labels"], "entry": entry}
    res["w"] = {"chains": w, "ok": bool(w) and all(r["ok"] for r in w.values())}
    res["w2"] = gt.gate_w2(w2)
    hist = {k: _json(Path(args.runs) / k / bt.FILES["history"])["points"] for k, _ in base}
    res["n"] = gt.gate_n(hist, {k: v["z"] for k, v in stored.items()})
    names = ("a", "a_cl20", "b", "d", "f", "g", "p", "s", "w", "w2", "n")
    res["status"] = {n: bool(res[n]["ok"]) for n in names}
    res["ok"] = all(res["status"].values())
    return res


# --- (v3), (k): part of access (A) ----------------------------------------------------------------

def v3_gates(scores: Path, selection: Mapping, book: Mapping | None, access0: Mapping,
             n_chunk: int, blocked: frozenset, dropped: Sequence[str] = ()) -> dict:
    """(v3) on every expected test score file (structure only; the restricted manifest read)
    and (k) on the loaded checkpoints the scoring calls recorded."""
    target = Path(scores) / "test"
    metas = [_json(f) for f in sorted(target.glob("meta_*.json"))]
    bound = {}
    for m in metas:
        for rel, sha in (m.get("written_sha256") or {}).items():
            bound.setdefault(rel, set()).add(sha)
    manifests = {d: dt.read_manifest(d) for d in gt.DATASETS}
    jobs, absent = sc.jobs_of(selection, book)
    rows, ok_bound = {}, {}
    for j in jobs:
        if f"{j.chain}/{j.run}" in dropped:
            continue
        f = gt.score_file(scores, j.chain, j.run, j.rule)
        rel = f.relative_to(target).as_posix()
        if not f.is_file():
            rows[rel] = None
            continue
        rows[rel] = gt.v3_file(_npz(f), manifests[j.dataset],
                               gt.EVAL_POINTS if j.rule == gt.EVAL_RULE else None)
        ok_bound[rel] = sc.sha256_file(f) in bound.get(rel, set())
    full = [m for m in metas if m.get("mode") == "full"]
    targets = (full[-1].get("chunk_targets") or {}) if full else {}
    a0 = {d: (access0.get(d) or {}).get("chunk_targets", {}).get(str(n_chunk), [])
          for d in gt.DATASETS}
    v3 = gt.gate_v3(rows, {d: m.names for d, m in manifests.items()}, targets, a0, ok_bound)
    v3["absent_rules"] = absent
    loaded = [r for m in metas for r in (m.get("loaded") or [])]
    return {"v3": v3, "k": gt.gate_k(loaded, blocked)}


def logged_access(test_log: Path, access: str, command: str, files: Mapping, run,
                  precheck=None) -> dict:
    """A local part of a test access: clean tree, frozen plan and ``precheck`` (else a
    "refused" line), the "start" line before any test file is read, ``run()``, "end" (or
    "error")."""
    dt.outside_repo(test_log)
    code = code_version()
    line = {"access": access, "command": command, "commit": code.get("commit"), "code": code}
    try:
        require_clean_tree()
        plan = dt.plan_seal(E023_DIR)["sha256"]
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


# --- analysis: inputs -------------------------------------------------------------------------

@dataclass
class Loaded:
    units: dict = field(default_factory=dict)       # (d, p, s, arm) -> e022.Units
    per_video_sigma: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)       # (d, p, s, arm) -> file sha256
    nonfinite: dict = field(default_factory=dict)
    d14: dict = field(default_factory=dict)         # (d, p, s, arm) -> (30, rows) z
    sigma_ones: dict = field(default_factory=dict)
    constant_share: dict = field(default_factory=dict)


def e022_checks(e022_scores: Path, listed_file: Path | None = None,
                selection_file: Path | None = None) -> dict:
    """Gate (h): every E022 C / C.last score file has its listed sha256 (bytes) and, once
    matched, records C's selected checkpoint of E022's committed selection (defaults: E022's
    committed results/e022_test_scores.sha256 and e022_selection.json)."""
    listed_file = Path(listed_file or E022_SCORES_LIST)
    listed = ea.read_sha256_list(listed_file)
    sel = _json(selection_file or E022_SELECTION)
    files, bad = {}, []
    for chain in ea.chains():
        for arm in gt.C_ARMS:
            rel = f"{chain}/{arm}.npz"
            f = Path(e022_scores) / "test" / rel
            got = sc.sha256_file(f) if f.is_file() else None
            row = {"sha256": got, "listed": listed.get(rel), "ok": got is not None
                   and got == listed.get(rel)}
            if row["ok"]:
                ck = (((sel.get("chains") or {}).get(f"{chain}/hb_m10") or {})
                      .get("checkpoints") or {}).get(arm) or {}
                stored = str(_npz(f)["ckpt_sha256"])
                row |= {"ckpt_sha256": stored, "selected": ck.get("sha256"),
                        "ok": stored == ck.get("sha256")}
            files[rel] = row
            if not row["ok"]:
                bad.append(rel)
    return {"files": len(files), "mismatched": bad, "list_sha256": sc.sha256_file(listed_file),
            "ok": not bad and len(files) == len(gt.C_ARMS) * len(ea.chains())}


def split_rows(flat: np.ndarray, counts: Sequence[int]) -> list[np.ndarray]:
    return e22.split_rows(np.asarray(flat), np.asarray(counts))


def new_units(arr: Mapping[str, np.ndarray]) -> e22.Units:
    """The readout units of a new arm's test file: logit readout zc (= z, c(v) = 0), its z,
    the stored fp32 sigma scores."""
    counts = np.asarray(arr["counts"])
    zc = split_rows(arr["zc"], counts)
    return e22.Units("snippet", zc, split_rows(arr["z"], counts),
                     np.zeros(len(counts), dtype=np.float32), split_rows(arr["sigma"], counts))


def load_e023(scores: Path, selection: Mapping, book: Mapping | None, sets: Mapping,
              dropped: Sequence[str] = (), listed: Mapping[str, str] | None = None) -> Loaded:
    """Every E023 test score file of the selection, its bytes first checked against the
    committed sha256 list (``listed``: results/e023_test_scores.sha256), then bound to its
    checkpoint and the manifest order; non-finite files recorded (numerical failure of that
    rule)."""
    out = Loaded()
    jobs, _ = sc.jobs_of(selection, book)
    for j in jobs:
        if f"{j.chain}/{j.run}" in dropped:
            continue
        d, p, s = gt.split_chain(j.chain)
        ts = sets[d]
        f = gt.score_file(scores, j.chain, j.run, j.rule)
        if listed is not None and sc.sha256_file(f) != listed.get(
                f.relative_to(Path(scores) / "test").as_posix()):
            raise ValueError(f"{j.label}: not the file of the committed sha256 list")
        arr = _npz(f)
        if list(map(str, arr["names"])) != list(ts.names):
            raise ValueError(f"{j.label}: not the manifest order")
        if j.rule == gt.EVAL_RULE:
            if [str(x) for x in arr["ckpt_sha256"]] != [x[1] for x in j.files]:
                raise ValueError(f"{j.label}: not the selected evaluation-point checkpoints")
            out.d14[(d, p, s, j.arm)] = (np.asarray(arr["z"]), np.asarray(arr["counts"]))
            continue
        if str(arr["ckpt_sha256"]) != j.files[0][1]:
            raise ValueError(f"{j.label}: not the selected checkpoint")
        arm = gt.rule_arm(j.arm, j.rule)
        key = (d, p, s, arm)
        bad = any(sc.nonfinite(arr).values())
        out.nonfinite[key] = bad
        out.files[key] = sc.sha256_file(f)
        if bad:
            continue
        out.units[key] = new_units(arr)
        out.sigma_ones[key] = int((np.asarray(arr["sigma"]) == np.float32(1.0)).sum())
        zs = split_rows(arr["z"], arr["counts"])
        out.constant_share[key] = float(np.mean([np.ptp(z) == 0 for z in zs]))
    return out


def load_e022(e022_scores: Path, sets: Mapping, loaded: Loaded) -> dict:
    """E022's C and C.last units (E022's own loader, e022.arm_units) and C's c(v) per run."""
    offsets = {}
    for chain in ea.chains():
        d, p, s = gt.split_chain(chain)
        ts = sets[d]
        for arm in gt.C_ARMS:
            arr = _npz(Path(e022_scores) / "test" / chain / f"{arm}.npz")
            if list(map(str, arr["names"])) != list(ts.names):
                raise ValueError(f"{chain}/{arm}: not the manifest order")
            bad = any(np.size(arr[k]) - np.isfinite(np.asarray(arr[k], np.float64)).sum()
                      for k in ("z", "c", "zc", "sigma") if k in arr)
            loaded.nonfinite[(d, p, s, arm)] = bool(bad)
            if bad:
                continue
            loaded.units[(d, p, s, arm)] = e22.arm_units(arm, arr, ts.totals)
            if arm == "C":
                offsets[(d, p, s)] = (np.asarray(arr["c"], np.float64), np.asarray(ts.totals),
                                      np.asarray(ts.labels))
    return offsets


def history_nonfinite(runs: Path, label: str, entry: Mapping) -> bool:
    """A non-finite training loss in the chain's record (C.L20: either stage's history)."""
    d = Path(runs) / label
    if entry.get("role") == "mechanism":
        return e22.history_failed(d) or e22.history_failed(d.parent / gt.CL20_STAGE1)
    rec = _json(d / bt.FILES["record"])
    return int(rec.get("nonfinite_losses") or 0) > 0


def failures(runs: Path, selection: Mapping, loaded: Loaded, gate_failed: Sequence[str]
             ) -> list[st22.Failure]:
    """The runs left out per rule (PLAN.md "실패 처리"): infrastructure (selection), gate
    (--gate_failed), numerical (a non-finite training loss: every rule of the chain; no selected
    checkpoint; non-finite test scores)."""
    out = []
    for label, e in (selection.get("chains") or {}).items():
        chain, _run = label.split("/")
        d, p, s = gt.split_chain(chain)
        rules = [gt.rule_arm(e["arm"], r) for r in gt.RULES]
        if e.get("failed") == "infrastructure (PLAN.md)" or label in gate_failed:
            kind = st22.INFRASTRUCTURE if label not in gate_failed else st22.GATE
            out += [st22.Failure(d, p, s, a, kind) for a in rules]
            continue
        hist = history_nonfinite(runs, label, e)
        out += [st22.Failure(d, p, s, a, st22.NUMERICAL) for a in rules
                if hist or (d, p, s, a) not in loaded.units]
    for (d, p, s, a), bad in loaded.nonfinite.items():
        if a in gt.C_ARMS and bad:
            out.append(st22.Failure(d, p, s, a, st22.NUMERICAL))
    return out


def run_metrics(units: e22.Units, ts, readouts: Sequence[str]) -> dict:
    """E022's per-run metrics (e022.run_metrics) and, for the sigma readout, L of the fp32
    sigma frame scores (the readout flag)."""
    out = {}
    for ro in readouts:
        m = e22.run_metrics(units, ts, ro)
        if ro == "sigma":
            frames = fr.frame_arrays(units.values("sigma"), ts.totals, units.kind)
            w = fr.within_video(frames, ts.gts, ts.labels, fr.L_METRIC[ts.dataset])
            m |= {"L": w["value"], "L_per_video": w["per_video"]}
        out[ro] = m
    return out


def decomposition(units: e22.Units, ts) -> dc.Decomposition:
    """The pooled frame-AUC decomposition of one run's logit frames (the evaluator's videos:
    at least DEFAULT_MIN_FRAMES frames)."""
    frames = e22.frames_of(units, "logit", ts.totals)
    keep = [i for i, g in enumerate(ts.gts) if len(g) >= DEFAULT_MIN_FRAMES]
    return dc.decompose([frames[i] for i in keep], [np.asarray(ts.gts[i]) for i in keep])


# --- analysis: gates (h2), (d1) -------------------------------------------------------------------

def h2_gate(metrics: Mapping, summary: Mapping, tol: float = H2_TOL) -> dict:
    """(h2): C and C.last L and the pooled main metric per setting and seed (logit readout)
    against e022_summary.json descriptives.1_cells (seed order of the confirmation seeds)."""
    cells = ((summary.get("descriptives") or {}).get("1_cells") or {}).get("cells") or {}
    rows, worst = {}, 0.0
    for d in gt.DATASETS:
        for p in gt.PAIRS:
            for arm in gt.C_ARMS:
                for name in ("L", PRIMARY[d]):
                    ours = [metrics[(d, p, s, arm, "logit")][name] for s in gt.SEEDS
                            if (d, p, s, arm, "logit") in metrics]
                    want = (cells.get(f"{d}/{p}/{arm}/logit/{name}") or {}).get("values")
                    k = f"{d}/{p}/{arm}/{name}"
                    if want is None or len(want) != len(ours):
                        rows[k] = {"ok": False, "reason": "missing or different seed count"}
                        worst = math.inf
                        continue
                    diff = max((abs(a - b) for a, b in zip(ours, want, strict=True)),
                               default=0.0)
                    worst = max(worst, diff)
                    rows[k] = {"n": len(ours), "max_abs_diff": diff, "ok": diff <= tol}
    return {"rows": rows, "max_abs_diff": worst, "tol": tol,
            "ok": bool(rows) and all(r["ok"] for r in rows.values())}


def d1_gate(decomp: Mapping[tuple, dc.Decomposition], tol: float = D1_TOL,
            metrics: Mapping | None = None, auc_tol: float = D1_AUC_TOL) -> dict:
    """(d1): the identity of every run within ``tol`` and, for the UCF runs (``metrics``: the
    evaluator's per-run metrics, logit readout), |decomposition AUC - pooled frame AUC / 100| <=
    ``auc_tol`` (the decomposition reproduces the evaluator's frames, not only itself)."""
    res = {"/".join(map(str, k)): v.residual for k, v in decomp.items()}
    bad = sorted(k for k, k2 in zip(res, decomp, strict=True)
                 if not dc.identity_holds(decomp[k2], tol))
    worst = max((abs(r) for r in res.values() if math.isfinite(r)), default=0.0)
    auc_rows, auc_bad, auc_worst = {}, [], 0.0
    if metrics is not None:
        for k, v in decomp.items():
            if k[0] != "ucf":
                continue
            m = metrics.get((*k, "logit")) or {}
            name = "/".join(map(str, k))
            if "frame_auc" not in m:
                auc_rows[name], auc_worst = math.inf, math.inf
                auc_bad.append(name)
                continue
            diff = abs(float(v.auc) - float(m["frame_auc"]) / 100.0)
            auc_rows[name] = diff
            auc_worst = max(auc_worst, diff)
            if not diff <= auc_tol:
                auc_bad.append(name)
    return {"runs": len(res), "max_abs_residual": worst, "failed": bad, "tol": tol,
            "auc_vs_evaluator": {"runs": len(auc_rows), "max_abs_diff": auc_worst,
                                 "failed": sorted(auc_bad), "tol": auc_tol,
                                 "checked": metrics is not None},
            "ok": bool(res) and not bad and not auc_bad}


# --- analysis -----------------------------------------------------------------------------------

def values(metrics: Mapping, dataset: str, readout: str, metric: str,
           arms: Sequence[str] | None = None) -> dict:
    return {(k[1], k[2], k[3]): m[metric] for k, m in metrics.items()
            if k[0] == dataset and k[4] == readout and metric in m
            and (arms is None or k[3] in arms)}


def per_video_arrays(metrics: Mapping, readout: str = "logit") -> tuple[dict, dict]:
    """(dataset, pair, seed, arm) -> per-video L in the included order, and the order."""
    order: dict[str, list[int]] = {}
    out = {}
    for k, m in metrics.items():
        if k[4] != readout or "L_per_video" not in m:
            continue
        idx = sorted(int(i) for i in m["L_per_video"])
        if order.setdefault(k[0], idx) != idx:
            raise ValueError(f"{k}: another set of included videos")
        out[k[:4]] = np.array([m["L_per_video"][i] for i in idx], dtype=np.float64)
    return out, order


def bootstraps(metrics: Mapping, loaded: Loaded, sets: Mapping, fam: Sequence[str],
               n_resamples: int) -> tuple[dict, dict]:
    """L and pooled video bootstraps of C - B_k (logit readout) over each arm's complete seeds;
    resample 0 reproduces the evaluator's values (within BOOT_POINT_TOL)."""
    per_video, order = per_video_arrays(metrics)
    boot_l, boot_p = {}, {}
    for d in gt.DATASETS:
        ts = sets[d]
        w_l = st.l_weights(len(order.get(d, [])), n_resamples)
        w_p = st20.bootstrap_weights(ts.labels, n_resamples, st.BOOT_SEED)
        l_vals, p_vals = values(metrics, d, "logit", "L"), values(metrics, d, "logit",
                                                                  PRIMARY[d])
        pv = {(k[1], k[2], k[3]): v for k, v in per_video.items() if k[0] == d}
        atoms = {(p, s, a): (lambda u=loaded.units[(d, p, s, a)], g=ts.gts: st22.snippet_atoms(
            u.logit, g)) for (dd, p, s, a) in loaded.units if dd == d}
        seeds_c = list(st.pair_means(l_vals, "C"))
        boot_l[d], boot_p[d] = {}, {}
        cache_l, cache_p = {}, {}
        for b in fam:
            seeds_b = list(st.pair_means(l_vals, b))
            r = st.l_bootstrap(pv, w_l, "C", b, seeds_c, seeds_b, point=l_vals, cache=cache_l)
            seeds_pb = list(st.pair_means(p_vals, b))
            q = st.pooled_bootstrap(atoms, w_p, PRIMARY[d], "C", b, list(st.pair_means(
                p_vals, "C")), seeds_pb, point=p_vals, cache=cache_p)
            for name, x in (("L", r), ("pooled", q)):
                if x["seeds_a"] and x["seeds_b"] and not x["point_check_max_abs_diff"] <= \
                        BOOT_POINT_TOL:
                    raise ValueError(f"{d} {name} C-{b}: bootstrap resample 0 differs from the "
                                     f"evaluator by {x['point_check_max_abs_diff']}")
            boot_l[d][b], boot_p[d][b] = r, q
    return boot_l, boot_p


def d14_points(loaded: Loaded, sets: Mapping, selection: Mapping, book: Mapping) -> dict:
    """D14 descriptive inputs: per main-family run the test L and pooled metric (snippet atoms,
    resample 0) of each evaluation point, its validation V-AUCs and the selected point."""
    out = {}
    for (d, p, s, arm), (z, counts) in loaded.d14.items():
        ts = sets[d]
        label = next(k for k, e in selection["chains"].items() if e["arm"] == arm
                     and k.split("/")[0] == gt.chain_name(d, p, s))
        entry, pts = selection["chains"][label], book["chains"][label]["points"]
        ls, pooled = [], []
        ones = np.ones((1, len(ts.labels)), dtype=np.int64)
        for k in range(z.shape[0]):
            zs = split_rows(z[k], counts)
            if not all(np.isfinite(x).all() for x in zs):
                ls.append(math.nan)
                pooled.append(math.nan)
                continue
            frames = fr.frame_arrays(zs, ts.totals, "snippet")
            ls.append(fr.within_video(frames, ts.gts, ts.labels, fr.L_METRIC[d])["value"])
            at = st22.snippet_atoms(zs, ts.gts)
            auc, ap = st20.weighted_auc_ap(at, ones)
            pooled.append(float((auc if PRIMARY[d] == "frame_auc" else ap)[0]))
        out[(d, p, s, arm)] = {"L": ls, "pooled": pooled,
                               "v_auc_sigma": [pts[k]["v_auc_sigma"] for k in sorted(pts)],
                               "selected": int(entry["selection"]["sigma"]["point"])}
    return out


def weak_arms(arms: Mapping[str, gt.Arm]) -> list[str]:
    """(q1d) "적응 하의 약함" from results/e023_q1b.json ("q1d": e023_score q1d, the
    e023_gates.q1d_record format; "methods" -> method -> {"flag": bool}). A missing, incomplete
    or partial record stops the analysis (the flag could otherwise never be raised)."""
    rec = _json_or(OUT["q1b"], {}).get("q1d")
    main = [a for a in arms.values() if a.role == "main"]
    chk = q1d_check(rec, [a.method for a in main])
    if not chk["ok"]:
        raise SystemExit(f"(q1d) record missing or incomplete in {OUT['q1b'].name}: {chk}")
    return [a.name for a in main if rec["methods"][a.method].get("flag")]


def run_rows(metrics: Mapping, fails: Sequence[st22.Failure], info: Mapping) -> list[dict]:
    failed = {(f.dataset, f.pair, f.seed, f.arm): f.kind for f in fails}
    keys = sorted({k[:4] for k in metrics} | set(failed), key=str)
    rows = []
    for k in keys:
        base = {"dataset": k[0], "pair": k[1], "seed": k[2], "arm": k[3], **(info.get(k) or {})}
        if k in failed:
            rows.append(base | {"failed": failed[k]})
            continue
        for ro in ("logit", "sigma", "z"):
            m = metrics.get((*k, ro))
            if m is not None:
                rows.append(base | {"readout": ro, "failed": "", **{
                    x: v for x, v in m.items() if x != "L_per_video"}})
    return rows


def selection_info(selection: Mapping) -> dict:
    """(dataset, pair, seed, arm rule) -> selected point and its validation V-AUC (points)."""
    out = {}
    for label, e in (selection.get("chains") or {}).items():
        d, p, s = gt.split_chain(label.split("/")[0])
        for r in gt.RULES:
            ck = (e.get("checkpoints") or {}).get(r)
            if not ck:
                continue
            v = None
            if r != "last" and e.get("selection"):
                v = 100 * e["selection"][r]["v_auc"]
            elif e.get("v_auc"):
                v = 100 * e["v_auc"]["sigma"][-1]
            k = (d, p, s, gt.rule_arm(e["arm"], r))
            out[k] = {"point": ck.get("point"), "val_v_auc": v}
    return out


def gate_failure_summary(stamp: Mapping, gate_name: str, reason: str,
                         gates: Mapping[str, bool], selection: Mapping | None = None) -> dict:
    """The minimal results/e023_summary.json of a failed (h), (h2) or (d1): the category
    "관문 실패(판정 없음)" (PLAN.md "관문 실패 처리"), no verdict, no descriptives."""
    plan = st.STATUS_NAMES[st.GATE_FAILED]
    summary = {"check": "E023 stage F2: localisation against same-feature comparison methods "
                        "(PLAN.md 판정)", **stamp,
               "family": (selection or {}).get("family"),
               "verdict": {"category": plan, "failed_gate": gate_name, "reason": reason},
               "condition_2": {"status": st.GATE_FAILED, "plan": plan},
               "condition_3": {"status": "not_checked", "plan": f"점검 안 함({plan})"},
               "gates": dict(gates), "claims": {"secondary_improvements_allowed": False},
               "complete": False}
    write_json(OUT["summary"], clean(summary))
    return summary


def _stop(stamp: Mapping, name: str, reason: str, gates: Mapping[str, bool],
          selection: Mapping) -> None:
    gate_failure_summary(stamp, name, reason, gates, selection)
    raise SystemExit(f"{reason} (PLAN.md: stop and report; summary '관문 실패(판정 없음)')")


def analysis(args: argparse.Namespace, stamp: Mapping, gate: Mapping) -> dict:
    """Gates (h), (h2), (d1), the judgement and the descriptives (PLAN.md order)."""
    selection = _json(OUT["selection"])
    book = _json_or(OUT["all_ckpts"], {"chains": {}})
    arms = gt.arms_from_json(selection["arms"])
    fam = gt.main_arms(arms)
    x32 = {a.name: gt.arm_name(a.method, a.official_unit, a.official_unit)
           for a in arms.values() if a.role == "x32"}
    h = e022_checks(args.e022_scores)
    write_gate("h", h, {"h": h["ok"]})
    if not h["ok"]:
        _stop(stamp, "h", f"gate (h) failed: {h['mismatched'][:5]}", {"h": False}, selection)
    sets = e22.eval_sets()
    loaded = load_e023(args.scores, selection, book, sets, args.gate_failed,
                       ea.read_sha256_list(OUT["scores"]))
    offsets = load_e022(args.e022_scores, sets, loaded)
    fails = failures(args.runs, selection, loaded, args.gate_failed)
    gone = {(f.dataset, f.pair, f.seed, f.arm) for f in fails}
    units = {k: u for k, u in loaded.units.items() if k not in gone}
    metrics, per_sigma = {}, {}
    for k, u in units.items():
        ros = ("logit", "sigma", "z") if k[3] == "C" else ("logit", "sigma")
        for ro, m in run_metrics(u, sets[k[0]], ros).items():
            metrics[(*k, ro)] = m
            if ro == "sigma":
                per_sigma[k] = m["L"]
    h2 = h2_gate(metrics, _json(E022_SUMMARY))
    write_gate("h2", h2, {"h2": h2["ok"]})
    if not h2["ok"]:
        _stop(stamp, "h2", "gate (h2) failed: E023's C / C.last values differ from E022's",
              {"h": True, "h2": False}, selection)
    decomp = {k: decomposition(u, sets[k[0]]) for k, u in units.items()}
    d1 = d1_gate(decomp, metrics=metrics)
    write_gate("d1", d1, {"d1": d1["ok"]})
    if not d1["ok"]:
        bad = d1["failed"] or d1["auc_vs_evaluator"]["failed"]
        _stop(stamp, "d1", f"gate (d1) failed: {bad[:5]}", {"h": True, "h2": True, "d1": False},
              selection)
    loaded.units = units
    boot_l, boot_p = bootstraps(metrics, loaded, sets, fam, args.boot_resamples)
    l_logit = {d: values(metrics, d, "logit", "L") for d in gt.DATASETS}
    l_sigma = {d: {(k[1], k[2], k[3]): v for k, v in per_sigma.items()
                   if k[0] == d and k[3] in fam} for d in gt.DATASETS}
    info = selection_info(selection)
    sel_v = {(d, p, s, a): (None if v["val_v_auc"] is None else v["val_v_auc"] / 100)
             for (d, p, s, a), v in info.items() if a in fam}
    res = st.evaluate(l_logit=l_logit, l_sigma=l_sigma,
                      l_opposite={d: values(metrics, d, "logit", "L_opposite")
                                  for d in gt.DATASETS},
                      pooled={d: values(metrics, d, "logit", PRIMARY[d]) for d in gt.DATASETS},
                      family_arms=fam, x32_arms=x32, boot_l=boot_l, boot_pooled=boot_p,
                      degenerate_rows=st.degenerate(sel_v, loaded.constant_share, fam),
                      weak=weak_arms(arms), gates_ok=True)
    summary = {"check": "E023 stage F2: localisation against same-feature comparison methods "
                        "(PLAN.md 판정)", **stamp, "gate": gate.get("status"),
               "family": selection.get("family"), "arms": selection.get("arms"),
               "primary": {"gate_metric": "L (UCF within-video frame AUC, XD within-video "
                                          "frame AP; logit readout; pair mean)",
                           "check_metric": PRIMARY},
               "gates": {"h": h["ok"], "h2": h2["ok"], "d1": d1["ok"]},
               "failures": [list(f) for f in fails], **clean(res), "complete": False}
    write_json(OUT["summary"], clean(summary))                  # 1. verdict first
    write_csv(OUT["runs"], [clean(r) for r in run_rows(metrics, fails, info)])
    per_video, order = per_video_arrays(metrics)
    included = {d: [int(sets[d].totals[i]) for i in order.get(d, [])] for d in gt.DATASETS}
    metas = [_json(f) for f in sorted((Path(args.scores) / "test").glob("meta_*.json"))]
    full = [m for m in metas if m.get("mode") == "full"]
    cut_names = {d: set((full[-1].get("chunk_targets") or {}).get(d, [])) if full else set()
                 for d in gt.DATASETS}
    chunked = {d: [sets[d].names[i] in cut_names[d] for i in order.get(d, [])]
               for d in gt.DATASETS}
    records = {}
    for label, e in selection["chains"].items():
        f = Path(args.runs) / label / bt.FILES["record"]
        if e.get("role") != "mechanism" and f.is_file():
            r = _json(f)
            records[label] = {"arm": e["arm"], "parameters": r.get("parameters"),
                              "seconds": r.get("seconds")}
    secs = None
    if full and full[-1].get("time_utc") and full[-1].get("time_utc_end"):
        t0 = dt_time.datetime.fromisoformat(full[-1]["time_utc"])
        secs = (dt_time.datetime.fromisoformat(full[-1]["time_utc_end"]) - t0).total_seconds()
    summary["descriptives"] = desc.descriptives(
        metrics=metrics, decomp={k: v.to_json() for k, v in decomp.items()},
        per_video=per_video, included_totals=included, chunked=chunked, selection=selection,
        devlr=_json_or(OUT["devlr"], {}), val_auc=sel_v, c_offsets=offsets,
        chain_records=records, scoring_seconds=secs, sigma_ones=loaded.sigma_ones, h2=h2,
        prefreeze=_json_or(OUT["prefreeze"], {}), boot_l=boot_l,
        d14_points=d14_points(loaded, sets, selection, book), family_arms=fam,
        all_arms=[a for a in sorted({k[3] for k in units}) if a != "C"],
        condition3=res["condition_3"])
    summary["complete"] = True
    write_json(OUT["summary"], clean(summary))                  # 2. descriptives
    return summary


def records_check(scores: Path) -> dict:
    """The committed records the analysis relies on: (v3) and (k) passed, the score sha256 list
    committed and equal to the copy's, the selection files, E022's lists."""
    gate = _json_or(OUT["gate"], {})
    listed = Path(scores) / "test" / "scores.sha256"
    out = {"gate_committed": OUT["gate"].is_file() and committed(OUT["gate"]),
           "v3_k_passed": (gate.get("status") or {}).get("v3") is True
           and (gate.get("status") or {}).get("k") is True,
           "scores_list_committed": OUT["scores"].is_file() and committed(OUT["scores"]),
           "scores_list_equal": OUT["scores"].is_file() and listed.is_file()
           and OUT["scores"].read_text(encoding="utf-8") == listed.read_text(encoding="utf-8"),
           "selection_committed": OUT["selection"].is_file() and committed(OUT["selection"]),
           "e022_lists_committed": committed(E022_SCORES_LIST) and committed(E022_SELECTION),
           "q1d_recorded": q1d_check(_json_or(OUT["q1b"], {}).get("q1d"), [
               a.method for a in gt.arms_from_json(
                   _json_or(OUT["selection"], {}).get("arms") or {}).values()
               if a.role == "main"])["ok"]}
    out["ok"] = all(out.values())
    return out


def prior_analyses(test_log: Path) -> list[dict]:
    return [e for e in merged_entries(OUT["access"], test_log)
            if e.get("access") == "D" and e.get("event") == "start"]


def analyze_command(args: argparse.Namespace) -> None:
    """Access (D), once (module docstring)."""
    code = require_clean_tree()
    seal = seal_history()
    changed = code_changes(seal["freeze_commit"])
    if changed and not args.allow_code_change:
        raise SystemExit(f"{changed} changed after the freeze commit: --allow_code_change "
                         "'<reason>' (and the reason in the plan's change log)")
    dt.outside_repo(args.test_log)
    decisions = require_decisions()
    require_committed_gate_failed(args.gate_failed)
    prior = prior_analyses(args.test_log)
    if (OUT["summary"].exists() or prior) and not args.recompute:
        raise SystemExit(f"the analysis ran (once, PLAN.md; summary {OUT['summary'].exists()}, "
                         f"(D) start lines {len(prior)}): --recompute '<reason>' only in the "
                         "cases of the code rules")
    stamp = {"code": code, "plan": {"file": "PLAN.md", "sha256": seal["sha256"]},
             "seal": seal, "decisions": decisions, "code_changed_since_freeze": changed,
             "allow_code_change": args.allow_code_change, "recompute": args.recompute,
             "libraries": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "torch": torch.__version__}}
    line = {"access": "D", "command": "analyze", "commit": code.get("commit"),
            "plan_sha256": seal["sha256"], "recompute": args.recompute}
    dt.append_test_log(args.test_log, line | {
        "time_utc": now_utc(), "event": "start", "code": code,
        "files": {"scores": str(Path(args.scores) / "test"),
                  "e022_scores": f"{args.e022_scores}/test/<chain>/{{C,C.last}}.npz",
                  "manifests": list(dt.MANIFEST_FILES.values())}})
    try:
        records = records_check(args.scores)
        gate = write_gate("records", records, {"records": records["ok"]})
        if not records["ok"]:
            raise SystemExit(f"records missing or uncommitted: {records}")
        summary = analysis(args, stamp, gate)
    except BaseException as exc:
        dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "error",
                                                  "error": repr(exc)})
        raise
    dt.append_test_log(args.test_log, line | {
        "time_utc": now_utc(), "event": "end",
        "condition_2": (summary.get("condition_2") or {}).get("status"),
        "condition_3": (summary.get("condition_3") or {}).get("status")})
    print((summary.get("condition_2") or {}).get("plan"),
          (summary.get("condition_3") or {}).get("plan"))


# --- command line -------------------------------------------------------------------------------

def gates_command(args: argparse.Namespace) -> None:
    if args.stage == "prefreeze":
        res = prefreeze_gates(args)
        write_gate("prefreeze", res, {f"prefreeze_{k}": v["ok"] for k, v in res.items()
                                      if isinstance(v, Mapping)})
        ok = res["ok"]
    elif args.stage == "train":
        require_before_test_access(args.test_log)
        res = train_gates(args)
        write_gate("train", res, res["status"])
        ok = res["ok"]
    else:
        dropped = require_committed_gate_failed(args.gate_failed)
        selection = _json(OUT["selection"])
        book = _json(OUT["all_ckpts"]) if OUT["all_ckpts"].is_file() else None
        files = {"scores": str(Path(args.scores) / "test"),
                 "manifests (restricted read)": list(dt.MANIFEST_FILES.values()),
                 "left out (--gate_failed)": dropped}
        res = logged_access(args.test_log, "A", LOCAL_V3, files,
                            lambda: v3_gates(args.scores, selection, book, _json(args.access0),
                                             args.n_chunk, ea.load_blocked(OUT["blocked"]),
                                             dropped))
        write_gate("v3", res["v3"], {"v3": res["v3"]["ok"]})
        write_gate("k", res["k"], {"k": res["k"]["ok"]})
        ok = res["v3"]["ok"] and res["k"]["ok"]
    print(f"gates {args.stage}: {'passed' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(f"gate stage {args.stage} failed (PLAN.md failure rules); see the record")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e023")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="write PLAN.sha256 (PLAN.md before its change log)")
    pf = sub.add_parser("prefreeze", help="(3e): tests record and training environment")
    pf.add_argument("--junit", required=True, type=Path)
    pf.add_argument("--env_train", required=True, type=Path)
    q7 = sub.add_parser("q7", help="(q7): one port's code review and its Jev judgement")
    q7.add_argument("--method", required=True, choices=sorted(gt.PLAN_NAMES))
    q7.add_argument("--review", required=True, type=Path, help="the review file (outside the "
                                                                "repository; name and sha256)")
    q7.add_argument("--jev", required=True, type=float, help="Jev probability of the verdict")
    q7.add_argument("--verdict", required=True, help="e.g. 'accept' / 'fix and re-review'")
    q7.add_argument("--findings", nargs="*", default=[], help="finding ids, e.g. H1 M3")
    gt_ = sub.add_parser("gates")
    gt_.add_argument("--stage", required=True, choices=("prefreeze", "train", "v3"))
    gt_.add_argument("--runs", type=Path, help="train: local copy of ~/wvad_runs/e023")
    gt_.add_argument("--scores", type=Path, help="train / v3: local copy of the e023 score root")
    gt_.add_argument("--e021_runs", type=Path, help="train: the E021 HB L20 reference chains")
    gt_.add_argument("--junit", type=Path, help="train: a pytest JUnit XML of tests/")
    gt_.add_argument("--tests_reason")
    gt_.add_argument("--test_log", type=Path, help="v3: the local test access log")
    gt_.add_argument("--access0", type=Path, help="prefreeze / v3: the access (0) record")
    gt_.add_argument("--n_chunk", type=int, default=4096, help="N_chunk of (3j)")
    gt_.add_argument("--gate_failed", nargs="*", default=[],
                     help="'<chain>/<run>' dropped after an unfixed gate failure")
    gt_.add_argument("--methods", nargs="+", default=list(gt.MAIN),
                     help="prefreeze: the ported methods")
    gt_.add_argument("--q5", nargs="*", help="prefreeze: METHOD=DIR_EVAL,DIR_NO_EVAL")
    an = sub.add_parser("analyze")
    an.add_argument("--runs", type=Path, required=True, help="local copy of ~/wvad_runs/e023")
    an.add_argument("--scores", type=Path, required=True, help="local copy of the e023 scores")
    an.add_argument("--e022_scores", type=Path, required=True,
                    help="local copy of the E022 score root (test/<chain>/{C,C.last}.npz)")
    an.add_argument("--test_log", type=Path, required=True)
    an.add_argument("--gate_failed", nargs="*", default=[])
    an.add_argument("--allow_code_change", metavar="REASON")
    an.add_argument("--recompute", metavar="REASON")
    an.add_argument("--boot_resamples", type=int, default=st.BOOT_R)
    args = ap.parse_args(argv)
    if args.command == "gates":
        need = {"prefreeze": ("access0",), "train": ("runs", "scores", "e021_runs", "junit"),
                "v3": ("scores", "test_log", "access0")}[args.stage]
        missing = [k for k in need if getattr(args, k) is None]
        if missing:
            ap.error(f"--stage {args.stage} needs {', '.join('--' + m for m in missing)}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    if args.command == "freeze":
        freeze_command(args)
    elif args.command == "prefreeze":
        rec = prefreeze(args.junit, _json(args.env_train))
        print(f"(3e) recorded; environment differences from E022: {rec['differences']}")
    elif args.command == "q7":
        rec = record_q7(args.method, args.review, args.jev, args.verdict, args.findings)
        print(f"(q7) {args.method}: Jev {rec['jev']} '{rec['verdict']}' -> {OUT['q1b']}")
    elif args.command == "gates":
        gates_command(args)
    else:
        analyze_command(args)


if __name__ == "__main__":
    main()
