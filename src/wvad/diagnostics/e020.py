"""E020 (stage F1): the pre-registered confirmation experiment on the test set (PLAN.md). 2x2
factorial of the stage-2 start (A: stage-1 best-validation vs last checkpoint) and the offset
(B: off vs M10) over 2 datasets x 2 feature pairs x the 10 confirmation seeds; main question
M10 vs L20.

``freeze`` writes PLAN.sha256 (sha256 of PLAN.md before its change log) once PLAN.md is
committed unmodified at HEAD on a clean tree; the commit adding it is the freeze commit, the
only commit that may touch it. ``gates`` (before test scoring): (a)-(f) on the training
outputs, (t) on the test data (no model score), (v0) against P000 and, when --scores holds the
server's validation scoring (e020_score --split val), (s) with the path (ii) pre-check;
results/e020_gate.json, copied to the server, unlocks base. ``gates --check v`` (right after
e020_score --split test --cells base, before any other test scoring): the same gates, then gate
(v) on the base scores; the gate file unlocks the later cells. ``analyze`` (after every test
scoring): the same gates and (v), the test score files (each bound to the checkpoint gate (s)
validated), the access log (each call's DEPLOYED_COMMIT is the freeze commit or a descendant,
plan hash, base first and read in full at most twice, the later cells on the commit and plan of
(v), every call of this score root with its meta), then the verdict (UCF frame AUC / XD frame
AP, logit readout, stage-2 best checkpoint; seed-paired t on the pair-averaged M10 - L20, Holm
over the datasets, margins m_d), the flags, the secondary checkpoint rule, the factorial, the
video bootstrap and the descriptives. ``gates --scores`` (its gate file unlocks base),
``gates --check v`` and ``analyze`` need a clean git working tree, the latter two also the seal
history (PLAN.sha256 added once, by the freeze commit, with the plan's frozen hash); ``analyze``
refuses code or test-manifest changes since the freeze unless --allow_code_change gives the
reason. Score files stay outside the repository; their sha256 lists are checked.

    python -m wvad.diagnostics.e020 freeze
    python -m wvad.diagnostics.e020 gates --runs <e020 copy> --e019_gate <e019 gate copy> \
        --deploy_evidence <evidence.json> [--scores <score copy> [--check v]]
    python -m wvad.diagnostics.e020 analyze --runs <e020 copy> --e019_gate <e019 gate copy> \
        --deploy_evidence <evidence.json> --scores <score copy> [--allow_code_change <reason>]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr

from wvad.data import DATASETS
from wvad.data.eval_manifests import EvalManifest, eval_manifest_file, load_eval_manifest
from wvad.diagnostics import e020_stats as st
from wvad.diagnostics.common import code_version, write_csv, write_json
from wvad.diagnostics.e010 import splits_sha256
from wvad.diagnostics.e014 import GATE_ERRORS
from wvad.diagnostics.e015 import ENV_KEYS
from wvad.diagnostics.e020_gates import (
    PRE_SCORING,
    STAGE2_EPOCHS,
    START_EPOCH,
    _json,
    _npz,
    gate_v,
    path_ii_ok,
    run_gates,
    scoring_env,
    test_metas,
    val_split,
    verify_scores,
)
from wvad.diagnostics.e020_score import (
    BASE_CELLS,
    BEST,
    CELLS,
    E020_RESULTS,
    LAST,
    LATER,
    P000_CELL,
    PATHS,
    RULES,
    RUN_KEYS,
    SEEDS,
    TEST_CACHE_SHA256,
    Key,
    chain_name,
    check_excluded,
    expected_files,
    run_name,
    scorable_rules,
    score_path,
    score_rel,
)
from wvad.diagnostics.e020_seal import (
    E020_DIR,
    UNLOCK,
    full_reads,
    plan_frozen_part,
    plan_seal,
    read_log,
    write_plan_seal,
)
from wvad.diagnostics.runs import PAIRS, selection
from wvad.diagnostics.stage_d import calibration
from wvad.eval.crossfit import crossfit, stratified_half_splits, video_auc
from wvad.eval.metrics import frame_auc_ap, max_video_auc
from wvad.paths import REPO_ROOT
from wvad.prep.manifests import sha256_file

BOOT_POINT_TOL = 1e-6
PRIMARY = {"ucf": "frame_auc", "xdv": "frame_ap"}
ANOMALOUS = {"ucf": "anom_frame_auc", "xdv": "anom_frame_ap"}
READOUTS = ("logit", "sigma")
PARTS = ("z", "c")                       # readout decomposition (offset cells, best rule)
INFRA_FAILED = "infrastructure failure twice (PLAN.md)"
ACCESS_COPY = E020_RESULTS / "e020_test_access.log"
SCORES_COPY = E020_RESULTS / "e020_test_scores.sha256"
CALL_CHECKS = ("environment_ok", "same_environment_as_val", "same_commit_as_val", "caches_ok",
               "plan_ok", "descends_from_freeze", "gate_v_ok")


def write_gate(gate: Mapping) -> None:
    write_json(E020_RESULTS / "e020_gate.json", {"code": code_version(), **gate})


# --- seal: clean tree, freeze commit, test calls ----------------------------------------------

def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def git_bytes(*args: str) -> bytes:
    """Raw stdout of a git command (file contents byte for byte); empty on failure."""
    res = subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
                         cwd=REPO_ROOT, capture_output=True, check=False)
    return res.stdout if res.returncode == 0 else b""


def freeze_commit() -> str | None:
    """The commit that added PLAN.sha256 (None unless exactly one did)."""
    rel = (E020_DIR / "PLAN.sha256").relative_to(REPO_ROOT).as_posix()
    hashes = git("log", "--diff-filter=A", "--format=%H", "--", rel).stdout.split()
    return hashes[0] if len(hashes) == 1 else None


def seal_history() -> dict:
    """The plan was frozen once (PLAN.md): exactly one commit touched PLAN.sha256 (the freeze
    commit), and the hash it holds there equals the current plan seal and the sha256 of PLAN.md
    before its change log in that commit."""
    rel = {n: (E020_DIR / n).relative_to(REPO_ROOT).as_posix() for n in ("PLAN.sha256",
                                                                         "PLAN.md")}
    touched = git("log", "--format=%H", "--", rel["PLAN.sha256"]).stdout.split()
    freeze = freeze_commit()
    if len(touched) != 1 or touched[0] != freeze:
        raise SystemExit(f"PLAN.sha256 was touched by commits {touched}: only the freeze commit "
                         "may add it and none may change it (PLAN.md)")
    frozen = (git_bytes("show", f"{freeze}:{rel['PLAN.sha256']}").split() or [b""])[0].decode()
    try:
        at_freeze = hashlib.sha256(plan_frozen_part(
            git_bytes("show", f"{freeze}:{rel['PLAN.md']}"))).hexdigest()
    except ValueError as exc:
        raise SystemExit(f"PLAN.md of the freeze commit {freeze} unreadable ({exc!r})") from exc
    now = plan_seal(E020_DIR)["sha256"]
    if not frozen == now == at_freeze:
        raise SystemExit(f"plan hash now {now}, frozen {frozen!r}, of PLAN.md at the freeze "
                         f"{at_freeze}: the seal changed after the freeze (PLAN.md)")
    return {"freeze_commit": freeze, "sha256": frozen, "seal_commits": touched}


def code_changes(freeze: str) -> list[str]:
    """Tracked files under src/ and data/manifests/ changed from the freeze commit to HEAD."""
    res = git("diff", "--name-only", freeze, "HEAD", "--", "src", "data/manifests")
    if res.returncode != 0:
        raise SystemExit(f"git diff from the freeze commit {freeze} failed: {res.stderr.strip()}")
    return res.stdout.split()


def is_ancestor(a: str | None, b: str | None) -> bool:
    """a is b or an ancestor of b (git merge-base --is-ancestor)."""
    return bool(a and b) and git("merge-base", "--is-ancestor", a, b).returncode == 0


def require_clean_tree() -> dict:
    """PLAN.md: the local analysis runs on a clean git working tree (tracked files)."""
    code = code_version()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed (PLAN.md); code version {code}")
    return code


def caches_ok(meta: Mapping) -> bool:
    caches = meta.get("test_caches") or {}
    return set(caches) == set(TEST_CACHE_SHA256) and all(
        caches[k].get("sha256") == v for k, v in TEST_CACHE_SHA256.items())


def call_rows(scores: Path, plan_sha: str, freeze: str | None,
              scored_with: Mapping | None) -> list[dict]:
    """Every test call: training environment, the environment and commit of the validation
    scoring, the plan's test caches, the frozen plan hash, a commit descending from the freeze
    and (later cells) the gate (v) binding it was unlocked with equal to the one (v) found."""
    val = _json(Path(scores) / "val" / "meta.json")
    val_env, val_commit = val.get("environment") or {}, (val.get("code") or {}).get("commit")
    rows = []
    for m in test_metas(scores):
        env, commit = m.get("environment") or {}, (m.get("code") or {}).get("commit")
        later = bool(set(m.get("cells") or ()) & set(LATER))
        unlocked = (m.get("gate_v") or {}).get("scored_with")
        rows.append({"meta": m["meta_file"], "cells": m.get("cells"), "commit": commit,
                     "environment_ok": scoring_env(m)["ok"],
                     "same_environment_as_val": all(env.get(k) == val_env.get(k)
                                                    for k in ENV_KEYS),
                     "same_commit_as_val": commit is not None and commit == val_commit,
                     "caches_ok": caches_ok(m),
                     "plan_ok": (m.get("plan") or {}).get("sha256") == plan_sha,
                     "descends_from_freeze": is_ancestor(freeze, commit),
                     "gate_v_ok": not later or (scored_with is not None
                                                and unlocked == scored_with)})
    return rows


def path_ok(arr: Mapping, path: str) -> bool:
    """Scoring conditions of the file's path: (i) GPU, batch 32; (ii) P000's."""
    if path == "ii":
        return path_ii_ok(arr)
    return bool(str(arr["path"]) == "i" and str(arr["device"]).startswith("cuda")
                and int(arr["batch_size"]) == PATHS["i"][1])


def file_binding(runs: Path, scores: Path, key: Key, rule: str, path: str = "i",
                 arr: Mapping | None = None) -> str | None:
    """Why a test score file is not bound to the checkpoint gate (s) validated (None: it is):
    its ckpt_sha256 equals the local run copy's checkpoint and the validation file's, its epoch
    the selected one (last rule: 30), its val_video_auc the recorded one (NaN equal to NaN) and
    its scoring conditions those of its path."""
    rel = score_rel(key, rule, path)
    try:
        arr = _npz(score_path(scores, "test", key, rule, path)) if arr is None else arr
        d2 = Path(runs) / run_name(key)
        sel = selection(d2)
        epoch, auc = ((sel["best_epoch"], sel["best_val_video_auc"]) if rule == "best" else
                      (STAGE2_EPOCHS, _json(d2 / "history.json")[-1]["val_video_auc"]))
        with np.load(score_path(scores, "val", key, rule, "i")) as z:
            validated = str(z["ckpt_sha256"])
        sha, got_auc = str(arr["ckpt_sha256"]), float(arr["val_video_auc"])
        bad = [name for name, ok in (
            ("checkpoint sha256 (local copy)", sha == sha256_file(d2 / RULES[rule])),
            ("checkpoint sha256 (validation file)", sha == validated),
            ("epoch", int(arr["epoch"]) == int(epoch)),
            ("val_video_auc", got_auc == float(auc) or math.isnan(got_auc) and math.isnan(auc)),
            ("scoring conditions", path_ok(arr, path))) if not ok]
    except GATE_ERRORS as exc:
        return f"{rel}: {exc!r}"
    return f"{rel}: {', '.join(bad)}" if bad else None


def test_scores_check(scores: Path, runs: Path, excluded: Sequence[str], cells: Sequence[str],
                      plan_sha: str, final: bool, scored_with: Mapping | None) -> dict:
    """Test score files (sha256 lists; ``final``: every cell, equal to the committed
    results/e020_test_scores.sha256), each bound to its validated checkpoint (P000's
    excepted), and every test call's conditions."""
    allowed = (P000_CELL,) if final else (P000_CELL, *LATER)
    expected = expected_files(runs, cells, excluded)
    files = verify_scores(scores, "test", expected, allowed, SCORES_COPY if final else None)
    unbound = [x for x in (file_binding(runs, scores, *v) for v in expected.values()
                           if v[0][3] != P000_CELL) if x]
    freeze = freeze_commit()
    calls = call_rows(scores, plan_sha, freeze, scored_with)
    ok = files["ok"] and not unbound and calls and all(c[k] for c in calls for k in CALL_CHECKS)
    return {"files": files, "unbound": unbound, "freeze_commit": freeze, "calls": calls,
            "ok": bool(ok)}


def access_order(rows: Sequence[Mapping], entries: Sequence[Mapping],
                 scored_with: Mapping | None) -> dict:
    """Base first, base (p000_base) read in full at most twice and never after a later cell,
    the later cells on the deployed commit and plan hash of gate (v)'s base scores."""
    first_base = next((r["line"] for r in rows if "base" in r["cells"]), None)
    later = [r for r in rows if set(r["cells"]) & set(LATER)]
    base = [r["line"] for r in rows if set(r["cells"]) & set(BASE_CELLS)]
    bound = (None if scored_with is None else (scored_with["commit"], scored_with["plan_sha256"]))
    return {"base_first": first_base is not None and all(r["line"] > first_base for r in later),
            "no_base_after_later": not later or all(x < later[0]["line"] for x in base),
            "full_base_reads": {c: full_reads(entries, c) for c in BASE_CELLS},
            "later_bound_to_gate_v": bound is not None and all(
                (r["deployed_commit"], r["plan_sha256"]) == bound for r in later)}


def access_check(entries: Sequence[Mapping], metas: Sequence[Mapping], plan_sha: str,
                 scored_with: Mapping | None, test_dir: str | None = None) -> dict:
    """The committed access log (results/e020_test_access.log): every call's DEPLOYED_COMMIT is
    the freeze commit or a descendant and its plan hash the frozen one (``plan_sha``, the
    freeze commit's PLAN.sha256); the order and binding of access_order; every test meta names
    its own log line and every line of this score root (out_dir ``test_dir``, the server's
    <out>/test, or one a meta records) has its meta; every call used one access log."""
    freeze = freeze_commit()
    rows = [{"line": i, "time_utc": e.get("time_utc"), "cells": list(e.get("cells") or []),
             "rescore": e.get("rescore"), "missing_only": bool(e.get("missing_only")),
             "deployed_commit": e.get("deployed_commit"), "plan_sha256": e.get("plan_sha256"),
             "out_dir": e.get("out_dir"),
             "descends_from_freeze": is_ancestor(freeze, e.get("deployed_commit")),
             "plan_ok": e.get("plan_sha256") == plan_sha} for i, e in enumerate(entries, 1)]
    order = access_order(rows, entries, scored_with)
    by_line = {r["line"]: r["deployed_commit"] for r in rows}
    metas_ok = all(by_line.get(m.get("access_log_line")) == (m.get("code") or {}).get("commit")
                   for m in metas)
    roots = {test_dir, *(m.get("out") for m in metas)} - {None}
    have = {m.get("access_log_line") for m in metas}
    no_meta = [r["line"] for r in rows if r["out_dir"] in roots and r["line"] not in have]
    one_log = len({e.get("access_log") for e in entries} | {m.get("access_log") for m in metas})
    ok = (bool(rows) and all(r["descends_from_freeze"] and r["plan_ok"] for r in rows)
          and order["base_first"] and order["no_base_after_later"]
          and all(n <= 2 for n in order["full_base_reads"].values())
          and order["later_bound_to_gate_v"] and metas_ok and not no_meta and one_log == 1)
    return {"freeze_commit": freeze, "entries": rows, **order, "metas_match_log": metas_ok,
            "lines_without_meta": no_meta, "same_access_log": one_log == 1, "ok": bool(ok)}


def manifests_and_gts() -> tuple[dict, dict]:
    manifests = {d: load_eval_manifest(eval_manifest_file(d)) for d in DATASETS}
    return manifests, {d: m.frame_gts() for d, m in manifests.items()}


def add_gate_v(gate: dict, scores: Path, runs: Path, excluded: Sequence[str],
               cells: Sequence[str], plan_sha: str, final: bool, gts: Mapping) -> None:
    """Gate (v) (PLAN.md order 4), then the test score checks bound to it."""
    gate["v"] = gate_v(scores, gate["v0"], gts, excluded)
    gate["test_scores"] = test_scores_check(scores, runs, excluded, cells, plan_sha, final,
                                            gate["v"]["scored_with"])
    gate["status"]["test_scores"] = gate["test_scores"]["ok"]
    gate["status"]["v"] = gate["v"]["ok"]


# --- test scores, failures and metrics ------------------------------------------------------

def load_scores(runs: Path, scores: Path, split: str, excluded: Sequence[str],
                manifests: Mapping[str, EvalManifest] | None = None) -> dict:
    """Path (i) score files of every run and scorable rule; test files are checked against the
    manifest order (rows, labels) and their validated checkpoint (file_binding)."""
    out = {}
    for key in RUN_KEYS:
        if run_name(key) in excluded:
            continue
        for rule in scorable_rules(runs, key):
            arr = _npz(score_path(scores, split, key, rule, "i"))
            m = manifests[key[0]] if manifests else None
            if m is not None and (arr["logits"].shape[0] != len(m)
                                  or list(arr["labels"]) != list(m.labels)
                                  or not np.array_equal(arr["indices"], np.arange(len(m)))):
                raise ValueError(f"{score_rel(key, rule)}: rows differ from the test manifest")
            why = file_binding(runs, scores, key, rule, "i", arr) if m is not None else None
            if why:
                raise ValueError(f"{why}: not the validated checkpoint")
            out[(key, rule)] = arr
    return out


def history_failure(runs: Path, key: Key) -> str | None:
    """Non-finite loss or V-AUC in the history of either stage (fails every rule)."""
    c = Path(runs) / chain_name(*key[:3])
    hist = [*_json(c / CELLS[key[3]][0] / "history.json"), *_json(c / key[3] / "history.json")]
    if all(math.isfinite(v) for h in hist for k, v in h.items()
           if k.startswith("loss_") or k == "val_video_auc"):
        return None
    return "non-finite training loss or V-AUC (history)"


def score_failure(arr: Mapping | None, what: str) -> str | None:
    if arr is None:
        return f"no {what} score file (no checkpoint)"
    if not all(np.isfinite(arr[k]).all() for k in ("logits", "probs")):
        return f"non-finite {what} score"
    return None


def failures(runs: Path, val: Mapping, test: Mapping,
             excluded: Sequence[str]) -> dict[tuple[Key, str], str]:
    """Numerical failure per checkpoint rule (PLAN.md): history losses / V-AUC, or the rule's
    validation or test scores, non-finite -> the run is dropped under that rule."""
    out = {}
    for key in RUN_KEYS:
        if run_name(key) in excluded:
            out |= {(key, r): INFRA_FAILED for r in RULES}
            continue
        hist = history_failure(runs, key)
        for rule in RULES:
            why = (hist or score_failure(val.get((key, rule)), "validation")
                   or score_failure(test.get((key, rule)), "test"))
            if why:
                out[(key, rule)] = why
    return out


def treatment_failures(failed: Mapping[tuple[Key, str], str],
                       rule: str | None = None) -> list[tuple[Key, str]]:
    """Numerical failures of the offset cells (m10, m10b), all rules or one."""
    return sorted((k, r) for (k, r), why in failed.items() if CELLS[k[3]][2] != "none"
                  and why != INFRA_FAILED and rule in (None, r))


def readout(arr: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Frame-score source: logit (z + c), sigma (fp32 sigma(z + c)), z or c (offset heads)."""
    if name == "c":
        return np.repeat(arr["offsets"][:, None], arr["logits"].shape[1], axis=1)
    return arr[{"logit": "logits", "sigma": "probs", "z": "local_logits"}[name]]


def run_metrics(scores: np.ndarray, gts: Sequence[np.ndarray], labels: np.ndarray) -> dict:
    """Frame AUC / AP (all videos; anomalous videos only, their frames pooled) and video AUC
    (max), in points."""
    anom = np.flatnonzero(np.asarray(labels) == 1)
    full = frame_auc_ap(scores, gts)
    only = frame_auc_ap(scores[anom], [gts[i] for i in anom])
    return {"frame_auc": full["frame_auc"], "frame_ap": full["frame_ap"],
            "anom_frame_auc": only["frame_auc"], "anom_frame_ap": only["frame_ap"],
            "video_auc": 100 * max_video_auc(scores, labels)}


def compute_metrics(test: Mapping, failed: Mapping, gts: Mapping[str, list],
                    labels: Mapping[str, np.ndarray]) -> dict[tuple[Key, str, str], dict]:
    out = {}
    for (key, rule), arr in test.items():
        if (key, rule) in failed:
            continue
        names = READOUTS + (PARTS if CELLS[key[3]][2] != "none" and rule == "best" else ())
        for ro in names:
            out[(key, rule, ro)] = run_metrics(readout(arr, ro), gts[key[0]], labels[key[0]])
    return out


def values(metrics: Mapping, dataset: str, rule: str, ro: str, metric: str) -> dict:
    """(pair, seed, cell) -> metric of one dataset, rule and readout (failed runs absent)."""
    return {(k[1], k[2], k[3]): m[metric] for (k, r, o), m in metrics.items()
            if k[0] == dataset and r == rule and o == ro}


def run_info(runs: Path, key: Key) -> dict:
    c = Path(runs) / chain_name(*key[:3])
    s1, ck, _ = CELLS[key[3]]
    sel1, sel2 = selection(c / s1), selection(c / key[3])
    hist = _json(c / key[3] / "history.json")
    return {"s2_epoch": sel2.get("best_epoch"), "s2_v_auc": float(sel2["best_val_video_auc"]),
            "last_v_auc": float(hist[-1]["val_video_auc"]) if hist else math.nan,
            "s1_best_epoch": sel1.get("best_epoch"),
            "s1_start_epoch": START_EPOCH if ck == LAST else sel1.get("best_epoch")}


# --- verdict and flags ----------------------------------------------------------------------

def bootstraps(test: Mapping, metrics: Mapping, gts: Mapping, labels: Mapping,
               rule: str) -> dict:
    """Video bootstrap of D_d (M10 - L20, primary metric, logit readout) per dataset; resample 0
    must reproduce the evaluator's values."""
    out = {}
    for d in DATASETS:
        prim = values(metrics, d, rule, "logit", PRIMARY[d])
        seeds, _ = st.seed_diffs(prim, "m10", "l20")
        runs = {(p, s, c): test[((d, p, s, c), rule)]["logits"]
                for p in PAIRS for s in seeds for c in ("m10", "l20")}
        res = st.bootstrap_diff(runs, gts[d], labels[d], PRIMARY[d], seeds,
                                point={k: prim[k] for k in runs})
        if seeds and not res["point_check_max_abs_diff"] <= BOOT_POINT_TOL:
            raise ValueError(f"{d}: bootstrap resample 0 differs from the evaluator by "
                             f"{res['point_check_max_abs_diff']}")
        out[d] = res
    return out


def evaluate(metrics: Mapping, rule: str, boot: Mapping[str, dict], failed: Mapping) -> dict:
    """Main test, its readout / distribution / sample checks, the flags and, with treatment-
    related numerical failures, the imputed sensitivity result for one rule."""
    def by_d(ro: str, metric: Mapping[str, str]) -> dict:
        return {d: values(metrics, d, rule, ro, metric[d]) for d in DATASETS}
    prim = by_d("logit", PRIMARY)
    out = {"rule": rule, "main": st.main_test(prim),
           "sigma_readout": st.main_test(by_d("sigma", PRIMARY)),
           "wilcoxon": st.contrast(prim, "m10", "l20", kind="wilcoxon"),
           "m10_minus_base": st.contrast(prim, "m10", "base"),
           "localisation": st.contrast(by_d("logit", ANOMALOUS), "m10", "l20"),
           "settings": st.setting_contrast(prim, "m10", "l20"), "bootstrap": dict(boot)}
    out["flags"] = st.flags(main=out["main"], sigma=out["sigma_readout"],
                            wilcoxon=out["wilcoxon"], vs_base=out["m10_minus_base"],
                            localisation=out["localisation"], settings=out["settings"],
                            boot=boot)
    treat = treatment_failures(failed, rule)
    if treat:
        imputed = {d: st.impute_minimum(prim[d], [k[1:] for k, _ in treat if k[0] == d])
                   for d in DATASETS}
        out["treatment_failure_sensitivity"] = {
            "imputed": [f"{run_name(k)}/{r}" for k, r in treat], **st.main_test(imputed)}
    return out


def decide(res: Mapping[str, Mapping], failed: Mapping) -> dict:
    """Verdict on the primary rule, condition (2) (its exceptions: the setting-dependent
    settings, PLAN.md), checkpoint-rule dependence and M10 numerical instability (condition (2)
    then goes to the user; counted per (run, rule) and per run, and whether the primary test,
    m10 under the best rule, is hit)."""
    best, last = res["best"], res["last"]
    pairs = treatment_failures(failed)
    treat = [f"{run_name(k)}/{r}" for k, r in pairs]
    g_best, g_last = best["main"]["grades"], last["main"]["grades"]
    v = best["main"]["verdict"]
    cond = "not_met" if v != "better_substantive" else ("deferred_to_user" if treat else "met")
    fl = dict(best["flags"]) | {
        "checkpoint_rule_dependent": [d for d in DATASETS if g_best[d] != g_last[d]],
        "m10_numerically_unstable": bool(treat), "n_treatment_related_failures": len(treat),
        "n_runs_with_treatment_failure": len({run_name(k) for k, _ in pairs}),
        "treatment_failure_in_primary_test": any(k[3] == "m10" and r == "best" for k, r in pairs),
        "treatment_related_failures": treat}
    out = {"verdict": v, "condition_2": cond,
           "condition_2_exceptions": list(fl["setting_dependent"]) if cond == "met" else [],
           "grades": g_best, "grades_last_rule": g_last, "flags": fl}
    sens = {r: {k: res[r]["treatment_failure_sensitivity"][k] for k in ("verdict", "grades")}
            for r in RULES if "treatment_failure_sensitivity" in res[r]}
    return out | ({"treatment_failure_sensitivity": sens} if sens else {})


# --- descriptives ---------------------------------------------------------------------------

def _mean_sd(v: Sequence[float]) -> dict:
    return {"n": len(v), "mean": float(np.mean(v)) if v else math.nan,
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else math.nan}


def cell_table(metrics: Mapping) -> dict:
    """Descriptive 1: mean and seed SD per dataset / pair / cell / rule / readout / metric (the
    seed values are in e020_runs.csv)."""
    groups: dict[str, list[float]] = {}
    for (k, rule, ro), m in metrics.items():
        for name, v in m.items():
            groups.setdefault(f"{k[0]}/{k[1]}/{k[3]}/{rule}/{ro}/{name}", []).append(v)
    return {g: _mean_sd(v) for g, v in groups.items()}


def crossfit_v_auc(runs: Path, dataset: str, keys: Sequence[Key], cal: Mapping) -> dict:
    """Cross-fitted validation V-AUC (stage-2 epochs 1-30, E010 splits) of the given runs."""
    _, labels, _ = val_split(dataset)
    splits = stratified_half_splits(labels, cal["R"], seed=0)
    if splits_sha256(splits) != cal["splits_sha256"]:
        raise ValueError(f"{dataset}: splits differ from E010")
    return {k: crossfit(_npz(Path(runs) / run_name(k) / "record_val_scores.npz")["scores"][1:],
                        labels, splits, {"v_auc": video_auc}).mean("v_auc") for k in keys}


def _sign_share(diffs: Sequence[tuple[float, float]]) -> float:
    return float(np.mean([np.sign(a) == np.sign(b) for a, b in diffs])) if diffs else math.nan


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation; NaN with fewer than 3 runs or a constant input."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(spearmanr(x, y)[0])


def val_test(runs: Path, metrics: Mapping, info: Mapping, cal: Mapping) -> dict:
    """Descriptive 2: per setting, Spearman correlation of the stage-2 selected V-AUC with the
    test primary metric over its runs; sign agreement of the cross-fitted validation M10 - L20
    with the test M10 - L20 per seed (per setting and pair-averaged)."""
    out = {"rank_correlation": {}, "sign_agreement": {}}
    for d in DATASETS:
        prim = values(metrics, d, "best", "logit", PRIMARY[d])
        cf = crossfit_v_auc(runs, d, [(d, *k) for k in prim if k[2] in ("m10", "l20")], cal[d])
        per_seed = {}
        for p in PAIRS:
            ks = [k for k in prim if k[0] == p]
            rho = _spearman([info[(d, *k)]["s2_v_auc"] for k in ks], [prim[k] for k in ks])
            out["rank_correlation"][f"{d}/{p}"] = {"n": len(ks), "spearman": rho}
            seeds = [s for s in SEEDS if all((p, s, c) in prim for c in ("m10", "l20"))]
            per_seed[p] = {s: (cf[(d, p, s, "m10")] - cf[(d, p, s, "l20")],
                               prim[(p, s, "m10")] - prim[(p, s, "l20")]) for s in seeds}
            out["sign_agreement"][f"{d}/{p}"] = _sign_share(list(per_seed[p].values()))
        both = [s for s in SEEDS if all(s in per_seed[p] for p in PAIRS)]
        out["sign_agreement"][f"{d}/pair_mean"] = _sign_share(
            [tuple(np.mean([per_seed[p][s][i] for p in PAIRS]) for i in (0, 1)) for s in both])
    return out


def decomposition(metrics: Mapping, test: Mapping, labels: Mapping) -> dict:
    """Descriptive 3 (E017 risk check), offset cells, best rule: seed means of the primary and
    anomalous-only metrics for z, c and z + c, and the share of anomalous videos with c < 0."""
    out = {}
    for d in DATASETS:
        for p in PAIRS:
            for c in (x for x in CELLS if CELLS[x][2] != "none"):
                keys = [(d, p, s, c) for s in SEEDS if ((d, p, s, c), "best", "z") in metrics]
                if not keys:
                    continue
                entry = {ro: {m: float(np.mean([metrics[(k, "best", ro)][m] for k in keys]))
                              for m in (PRIMARY[d], ANOMALOUS[d])} for ro in (*PARTS, "logit")}
                entry["anomalous_share_c_below_0"] = float(np.mean(
                    [np.mean(test[(k, "best")]["offsets"][labels[d] == 1] < 0) for k in keys]))
                out[f"{d}/{p}/{c}"] = {"n": len(keys), **entry}
    return out


def epochs(info: Mapping) -> dict:
    """Descriptive 4: selected stage-2 epochs per cell (and the stage-1 start of best cells)."""
    out = {}
    for d in DATASETS:
        for p in PAIRS:
            for c in CELLS:
                keys = [(d, p, s, c) for s in SEEDS if (d, p, s, c) in info]
                e2 = [info[k]["s2_epoch"] for k in keys]
                entry = {"stage2_selected": e2, "stage2_counts": {
                    str(e): e2.count(e) for e in sorted({x for x in e2 if x is not None})}}
                if CELLS[c][1] == BEST:
                    e1 = [info[k]["s1_start_epoch"] for k in keys]
                    entry |= {"stage1_start": e1, "stage1_start_is_last": e1.count(START_EPOCH)}
                out[f"{d}/{p}/{c}"] = entry
    return out


def reported(metrics: Mapping) -> dict:
    """The numbers SEAR-D reports whatever the outcome (PLAN.md, use of results): the m10 cell,
    stage-2 best checkpoint, logit readout; seed mean and SD per dataset and pair."""
    out = {}
    for d in DATASETS:
        for p in PAIRS:
            rows = [m for (k, r, o), m in metrics.items()
                    if k[:2] == (d, p) and k[3] == "m10" and r == "best" and o == "logit"]
            out[f"{d}/{p}"] = {name: _mean_sd([x[name] for x in rows])
                               for name in (rows[0] if rows else {})}
    return out


def descriptives(runs: Path, metrics: Mapping, test: Mapping, info: Mapping, labels: Mapping,
                 gate_v_result: Mapping) -> dict:
    return {"reported_m10": reported(metrics), "cells": cell_table(metrics),
            "val_test": val_test(runs, metrics, info, calibration()),
            "readout_decomposition": decomposition(metrics, test, labels),
            "selected_epochs": epochs(info),
            "p000": {"gate_v_rows": list(gate_v_result["rows"]),
                     "path_i_minus_ii": list(gate_v_result["path_i_minus_ii"]),
                     "note": "base by paths (i) and (ii) vs P000 abl_base under the same "
                             "evaluator and ground truth (gate (v)); the earlier manuscript's "
                             "87.78 (UCF) and 75.75 (XD) are not compared: checkpoints chosen "
                             "on the test set, and XD AP on segment-quantised ground truth "
                             "(PLAN.md)"}}


def csv_rows(metrics: Mapping, info: Mapping, failed: Mapping) -> list[dict]:
    """e020_runs.csv: one row per run, rule and readout (metrics and val_video_auc in points)."""
    rows = []
    for key in RUN_KEYS:
        base = {"dataset": key[0], "pair": key[1], "seed": key[2], "cell": key[3]}
        for rule in RULES:
            if (key, rule) in failed:
                rows.append(base | {"rule": rule, "failed": failed[(key, rule)]})
                continue
            last = rule == "last"
            for ro in (*READOUTS, *PARTS):
                m = metrics.get((key, rule, ro))
                if m is None:
                    continue
                rows.append(base | {
                    "rule": rule, "readout": ro, "failed": "", **m,
                    "s1_start_epoch": info[key]["s1_start_epoch"],
                    "s2_epoch": STAGE2_EPOCHS if last else info[key]["s2_epoch"],
                    "val_video_auc": 100 * info[key]["last_v_auc" if last else "s2_v_auc"]})
    return rows


# --- commands -------------------------------------------------------------------------------

def freeze_command(args: argparse.Namespace) -> None:
    """PLAN.md, the reviewed code and the manifests are committed first (PLAN.md tracked at
    HEAD and unmodified, clean tracked tree); the freeze commit then adds PLAN.sha256 alone, its
    tree holding that PLAN.md (seal_history)."""
    require_clean_tree()
    rel = (E020_DIR / "PLAN.md").relative_to(REPO_ROOT).as_posix()
    if (git("cat-file", "-e", f"HEAD:{rel}").returncode != 0
            or git("status", "--porcelain", "--", rel).stdout.strip()):
        raise SystemExit(f"{rel} must be committed and unmodified at HEAD before the freeze: "
                         "commit the plan, code, manifests and README first (README)")
    h = write_plan_seal(E020_DIR)
    print(f"PLAN.sha256 written ({h}); commit it alone: that commit is the freeze commit")


def gates_command(args: argparse.Namespace) -> None:
    excluded = check_excluded(args.runs, args.infra_failed)
    check_v = args.check == "v"
    if check_v and args.scores is None:
        raise SystemExit("--check v needs --scores")
    if args.scores is not None:                # the gate file unlocks test scoring (server)
        require_clean_tree()
    plan = seal_history() if check_v else None
    gate = run_gates(args.runs, args.e019_gate, excluded, args.scores, args.deploy_evidence)
    if check_v:
        add_gate_v(gate, args.scores, args.runs, excluded, ("base",), plan["sha256"], False,
                   manifests_and_gts()[1])
    write_gate(gate)
    print("gates:", gate["status"])
    blocking = UNLOCK if check_v else (*PRE_SCORING, "s")
    if gate["errors"] or not all(gate["status"].get(g, not check_v) for g in blocking):
        raise SystemExit(f"gate failed or runs unreadable {gate['errors']} (PLAN.md failure "
                         "handling); see results/e020_gate.json")


def libraries() -> dict:
    """Versions behind the statistics (SciPy's Wilcoxon 'auto' rule changed across versions)."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def seal_stamp(code: Mapping, reason: str | None) -> dict:
    """The seal history and the analysed code: files under src/ and data/manifests/ changed
    since the freeze are refused unless --allow_code_change gives the reason (also for the
    plan's change log)."""
    seal = seal_history()
    changed = code_changes(seal["freeze_commit"])
    if changed and not reason:
        raise SystemExit(f"{changed} changed after the freeze commit {seal['freeze_commit']}: "
                         "--allow_code_change '<reason>' (and the reason in the plan's change "
                         "log)")
    return {"code": code, "plan": {"file": "PLAN.md", "sha256": seal["sha256"]}, "seal": seal,
            "code_changed_since_freeze": changed, "allow_code_change": reason,
            "libraries": libraries()}


def analyze_command(args: argparse.Namespace) -> None:
    stamp = seal_stamp(require_clean_tree(), args.allow_code_change)
    plan_sha = stamp["plan"]["sha256"]
    excluded = check_excluded(args.runs, args.infra_failed)
    gate = run_gates(args.runs, args.e019_gate, excluded, args.scores, args.deploy_evidence)
    manifests, gts = manifests_and_gts()
    add_gate_v(gate, args.scores, args.runs, excluded, ("base", *LATER), plan_sha, True, gts)
    val_out = _json(Path(args.scores) / "val" / "meta.json").get("out")     # server <out>/val
    test_dir = val_out and str(PurePosixPath(val_out).with_name("test"))
    gate["access"] = access_check(read_log(ACCESS_COPY), test_metas(args.scores), plan_sha,
                                  gate["v"]["scored_with"], test_dir)
    gate["status"]["access"] = gate["access"]["ok"]
    write_gate(gate)
    if gate["errors"] or not all(gate["status"].get(g, False) for g in (*UNLOCK, "access")):
        raise SystemExit("a gate, the test score files or the access log failed (PLAN.md); "
                         "see results/e020_gate.json")
    analysis(args, gate, manifests, gts, excluded, stamp)


def analysis(args: argparse.Namespace, gate: Mapping, manifests: Mapping, gts: Mapping,
             excluded: Sequence[str], stamp: Mapping) -> None:
    labels = {d: np.asarray(m.labels) for d, m in manifests.items()}
    test = load_scores(args.runs, args.scores, "test", excluded, manifests)
    failed = failures(args.runs, load_scores(args.runs, args.scores, "val", excluded), test,
                      excluded)
    info = {k: run_info(args.runs, k) for k in RUN_KEYS if run_name(k) not in excluded}
    metrics = compute_metrics(test, failed, gts, labels)
    res = {r: evaluate(metrics, r, bootstraps(test, metrics, gts, labels, r), failed)
           for r in RULES}
    summary = {"check": "E020 stage-F1 confirmation: M10 vs L20 and the 2x2 factorial "
                        "(PLAN.md)", **stamp, "gate": gate["status"], "margins": st.MARGIN,
               "alpha": st.ALPHA,
               "primary": {"metric": PRIMARY, "readout": "logit (z + c)", "rule": "best"},
               **decide(res, failed),
               "failed": {f"{run_name(k)}/{r}": why for (k, r), why in sorted(failed.items())},
               "best_rule": res["best"], "last_rule": res["last"]}
    if summary["verdict"] != "better_substantive":
        summary["note"] = ("not better (substantive): auxiliary metrics and per-setting results "
                           "are exploratory, not claimed as improvements (PLAN.md)")
    write_json(E020_RESULTS / "e020_summary.json", summary)        # 1. verdict and flags
    print(summary["verdict"], summary["condition_2"], summary["condition_2_exceptions"],
          summary["grades"], {k: v for k, v in summary["flags"].items() if v})
    write_csv(E020_RESULTS / "e020_runs.csv", csv_rows(metrics, info, failed))
    summary["factorial"] = st.factorial({d: values(metrics, d, "best", "logit", PRIMARY[d])
                                         for d in DATASETS})
    summary["descriptives"] = descriptives(args.runs, metrics, test, info, labels, gate["v"])
    write_json(E020_RESULTS / "e020_summary.json", summary)        # 2. factorial, descriptives


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E020 stage-F1 confirmation analysis (local)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="write PLAN.sha256 (PLAN.md before its change log)")
    for name in ("gates", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--runs", required=True, type=Path, help="local copy of ~/wvad_runs/e020")
        p.add_argument("--e019_gate", required=True, type=Path,
                       help="local copy of the E019 no-change gate folder (e019_gate_70d41384)")
        p.add_argument("--scores", required=name == "analyze", type=Path,
                       help="local copy of the e020_score output root (val/, test/)")
        p.add_argument("--deploy_evidence", type=Path,
                       help="gate (e), 13:48 batch: JSON evidence of the training deploy")
        p.add_argument("--infra_failed", nargs="*", default=[],
                       help="stage-2 runs '<chain>/<cell>' whose rerun also failed for "
                            "infrastructure reasons (PLAN.md failure handling)")
        if name == "gates":
            p.add_argument("--check", choices=("v",),
                           help="v: gate (v) on the base test scores (unlocks the later cells)")
        else:
            p.add_argument("--allow_code_change", metavar="REASON",
                           help="analyse although src/ or data/manifests/ changed after the "
                                "freeze (the reason also goes to the plan's change log)")
    args = ap.parse_args(argv)
    {"freeze": freeze_command, "gates": gates_command, "analyze": analyze_command}[
        args.command](args)


if __name__ == "__main__":
    main()
