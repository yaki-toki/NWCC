"""E021 (stage E3 pre-check): local analysis of the validation-only dense-snippet pilot
(experiments/E021_dense_snippet_pilot/PLAN.md). Video labels only, no test data; the temporal
annotations (GlanceVAD glance points, UCA proxies) are read only by ``describe``, after the verdict
file is committed: ``stage1`` and ``verdict`` import neither the glance nor the UCA module
(import-graph unit test) and the verdict code reads no annotation file.

    freeze    PLAN.sha256 = sha256 of PLAN.md before its change log (PLAN.md committed unmodified
              at HEAD, clean tracked tree; once)
    stage1    gates (a)-(h), (s), (s2), (p), (f), the extraction gates (x0)-(x6) of stage 1, the
              no-change gates (g1), (g2) and the freeze descent of the (g1), (x4), batch, scoring
              and analysis commits -> stage-1 verdict (A0) -> flags -> descriptives and the
              stage-2 start conditions: results/e021_stage1_gate.json, e021_stage1_runs.csv,
              e021_stage1_summary.json
    verdict   after stage 2: the gates of both stages (and (s3)) -> final verdict (C; eight final
              categories) -> flags -> descriptives: results/e021_gate.json, e021_runs.csv,
              e021_verdict.json. A stage-1 NO-GO gives the final category without stage-2 data;
              --not_run {gate,time,videos,defect} records a stage 2 that is not run (a gate
              failure without verdict, or start condition (3), (4) or (5) failed)
    choice    the user's option after a hold / no-verdict category, with whether it came before
              or after ``describe`` (appended to e021_verdict.json)
    describe  the temporal-annotation diagnostics, once, only when e021_verdict.json is committed
              unmodified; its sha256 is recorded at the start and at the end

    python -m wvad.diagnostics.e021 stage1 --runs <e021 copy> --scores <e021_scores copy> \
        [--extract_meta <dense meta.json copies>]
    python -m wvad.diagnostics.e021 verdict --runs <e021 copy> --scores <e021_scores copy>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from wvad.data import load_split, split_file
from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics import e021_arms as arms
from wvad.diagnostics import e021_desc as desc
from wvad.diagnostics import e021_gates as gates
from wvad.diagnostics import e021_rules as rules
from wvad.diagnostics.common import code_version, write_csv, write_json
from wvad.diagnostics.e021_score import (
    DATASETS,
    DEV_SEEDS,
    E021_DIR,
    PAIRS,
    RunKey,
    features_path,
    run_keys,
    score_path,
)
from wvad.eval.crossfit import METRICS, crossfit, snippets_per_window, stratified_half_splits
from wvad.eval.crossfit import video_auc as v_auc_metric
from wvad.eval.dense_readout import sigmoid32
from wvad.extract import grid
from wvad.paths import REPO_ROOT

RESULTS = gates.RESULTS
PLAN, SEAL = E021_DIR / "PLAN.md", E021_DIR / "PLAN.sha256"
CHANGELOG = "## 변경 기록"
OUT = {"stage1_gate": RESULTS / "e021_stage1_gate.json",
       "stage1_runs": RESULTS / "e021_stage1_runs.csv",
       "stage1_summary": RESULTS / "e021_stage1_summary.json",
       "gate": RESULTS / "e021_gate.json", "runs": RESULTS / "e021_runs.csv",
       "verdict": RESULTS / "e021_verdict.json", "describe": RESULTS / "e021_describe.json"}
EXTRACT_GATE = RESULTS / "e021_extract_gate.json"
APPROVALS = RESULTS / "e021_approvals.json"
OPTIONS = {rules.HOLD: ("i", "ii", "iii"), rules.HOLD_READOUT: ("i", "ii", "iii"),
           rules.NOT_RUN: ("i", "ii", "iii"), rules.HOLD_LOSS: ("i", "ii"),
           rules.GATE_FAILED: ("ii", "iii", "iv")}


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_or(path: Path, default):
    return _json(path) if Path(path).is_file() else default


def _clean(obj):
    """JSON-safe copy: numpy scalars and arrays to Python, tuple keys to 'a/b' strings, NaN to
    None."""
    if isinstance(obj, Mapping):
        return {("/".join(map(str, k)) if isinstance(k, tuple) else str(k)): _clean(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- plan seal ------------------------------------------------------------------------------

def plan_frozen_part(data: bytes) -> bytes:
    """PLAN.md up to its first line starting with '## 변경 기록' (excluded), as E020."""
    marker = CHANGELOG.encode("utf-8")
    if data.startswith(marker):
        return b""
    pos = data.find(b"\n" + marker)
    if pos < 0:
        raise ValueError(f"no line starting with {CHANGELOG!r}")
    return data[:pos + 1]


def plan_seal(plan: Path | None = None, seal: Path | None = None) -> dict:
    """The frozen plan is unchanged: PLAN.md before its change log has PLAN.sha256's hash."""
    plan, seal = plan or PLAN, seal or SEAL
    try:
        want = seal.read_text(encoding="utf-8").split()[0]
        got = hashlib.sha256(plan_frozen_part(plan.read_bytes())).hexdigest()
    except (OSError, ValueError, IndexError) as exc:
        raise SystemExit(f"plan seal unreadable ({exc!r}): freeze the plan first") from exc
    if got != want:
        raise SystemExit(f"PLAN.md before {CHANGELOG!r} has sha256 {got}, PLAN.sha256 {want}")
    rel = seal.relative_to(REPO_ROOT).as_posix() if seal.is_relative_to(REPO_ROOT) else None
    commits = gates.git("log", "--diff-filter=A", "--format=%H", "--", rel).stdout.split() \
        if rel else []
    return {"sha256": got, "freeze_commit": commits[0] if len(commits) == 1 else None}


def freeze_command(args: argparse.Namespace) -> None:
    code = code_version()
    rel = PLAN.relative_to(REPO_ROOT).as_posix()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed; code version {code}")
    if (gates.git("cat-file", "-e", f"HEAD:{rel}").returncode != 0
            or gates.git("status", "--porcelain", "--", rel).stdout.strip()):
        raise SystemExit(f"{rel} must be committed and unmodified at HEAD before the freeze")
    if SEAL.exists():
        raise SystemExit(f"{SEAL} exists: the plan is frozen once")
    h = hashlib.sha256(plan_frozen_part(PLAN.read_bytes())).hexdigest()
    SEAL.write_text(f"{h}  PLAN.md (before {CHANGELOG})\n", encoding="utf-8", newline="\n")
    print(f"PLAN.sha256 written ({h}); commit it alone: that commit is the freeze commit")


# --- analysis -------------------------------------------------------------------------------

def dataset_inputs(dataset: str) -> dict:
    """Validation videos in split order: labels, manifest T, lengths q = ceil(T / 16) / 32."""
    split = load_split(split_file(dataset))
    manifest = load_manifest(manifest_file(dataset))
    idx = np.asarray(split.val_indices, dtype=np.int64)
    totals = np.array([manifest.total_frames[i] for i in idx], dtype=np.int64)
    return {"labels": np.array([split.labels[i] for i in idx]), "totals": totals,
            "q": snippets_per_window(totals), "names": manifest.names,
            "trainpart_snippets": int(sum(grid.n_snippets(manifest.total_frames[i])
                                          for i in split.train_indices))}


def load_features(scores: Path, stage: int) -> dict[tuple[str, str], dict]:
    """The scoring's feature files of the stages so far (stage-2 keys added to stage 1's)."""
    out: dict = {}
    for d in DATASETS:
        for p in PAIRS:
            merged = {}
            for k in range(1, stage + 1):
                f = features_path(scores, k, d, p)
                with np.load(f) as z:
                    merged |= {key: z[key] for key in z.files}
            out[(d, p)] = merged
    return out


DENSE_W = {"h0": "A0", "h1": "A1", "hb": "C"}


def _decomposition(dec: Mapping, df: arms.DatasetFits, labels: np.ndarray) -> dict:
    """Descriptive 4 (M10 W arms): cross-fitted V-AUC of z only, c only and z + c."""
    splits = stratified_half_splits(labels, df.R, seed=0)
    out: dict = {}
    for (arm, part), runs in dec.items():
        for (_p, pair, _s), v in runs.items():
            cf = crossfit(v, labels, splits, {"v_auc": v_auc_metric})
            out.setdefault(pair, {}).setdefault(arm, {}).setdefault(part, []).append(
                cf.mean("v_auc"))
    for pair, by_arm in out.items():
        for arm, parts in by_arm.items():
            full = [df.fits[(arm, rules.PRIMARY)].get(("m10", pair, s)) for s in DEV_SEEDS]
            parts["full"] = [None if f is None else f.mean("v_auc") for f in full]
            for k in list(parts):
                v = parts[k]
                parts[k] = {"per_seed": v, "mean": desc._mean(v)}
    return out


def analyze(runs_root: Path, scores: Path, stage: int, runs: Mapping[RunKey, gates.Run],
            lineage: Mapping) -> dict:
    """Readouts, cross-fits, z tables and the per-dataset descriptives of the arms up to
    ``stage`` (one dataset's readout arrays in memory at a time)."""
    features = load_features(scores, stage)
    fits, descs, data_desc, inputs = {}, {}, {}, {}
    for d in DATASETS:
        inp = inputs[d] = dataset_inputs(d)
        labels, q, totals = inp["labels"], inp["q"], inp["totals"]
        names = arms.arms_of(stage, d)
        values: dict = defaultdict(dict)
        dec: dict = defaultdict(dict)
        dd: dict = {}
        for key in sorted(k for k in runs if k.dataset == d):
            run, fk = runs[key], (key.pipeline, key.pair, key.seed)
            mine = [a for a in names if arms.ARMS[a].head == key.head]
            if run.failed:
                for a in mine:
                    for ro in rules.READOUTS:
                        values[(a, ro)][fk] = None
                continue
            with np.load(score_path(scores, key.stage, key)) as z:
                sf = {k: z[k] for k in z.files}
            dense = arms.dense_readouts(sf) if "dense_z" in sf else None
            replaced = features[(d, key.pair)].get("snippet_replaced")
            for a in mine:
                vals = arms.arm_values(a, run, sf, dense)
                for ro in rules.READOUTS:
                    values[(a, ro)][fk] = vals[ro]
                dd[(a, *fk)] = desc.run_desc(a, vals, labels, q, totals, sf, dense, replaced)
            w = DENSE_W.get(key.head)
            if dense is not None and "W_free" in dense and key.seed in DEV_SEEDS and w in mine:
                dec[(w, "z_only")][fk] = sigmoid32(dense["W_free"])
                dec[(w, "c_only")][fk] = dense["c_W"][..., None]
        fits[d] = arms.fit_dataset(d, values, labels, q, lineage)
        decomposition = _decomposition(dec, fits[d], labels)
        data_desc[d] = desc.dataset_descriptives(
            d, fits[d], dd, names, labels, q, totals, features,
            {(d, pair): v for pair, v in decomposition.items()})
        descs[d] = dd
        del values, dec
    ls = arms.ls_thresholds(fits)
    delta, z = arms.tables(fits, lineage, ls, arms.contrasts_of(stage))
    return {"fits": fits, "ls": ls, "delta": delta, "z": z, "descs": descs,
            "dataset_descriptives": data_desc, "features": features, "inputs": inputs}


def h_gate(fits: Mapping[str, arms.DatasetFits]) -> dict:
    return gates.gate_h({(p, d, pair, s): cf for d, df in fits.items()
                         for (p, pair, s), cf in df.initial_r0.items()})


def run_rows(an: Mapping) -> list[dict]:
    rows = []
    for d, df in an["fits"].items():
        for (arm, ro), runs in sorted(df.fits.items()):
            for (p, pair, s), cf in sorted(runs.items()):
                row = {"dataset": d, "pipeline": p, "arm": arm, "readout": ro, "pair": pair,
                       "seed": s, "failed": cf is None, "n_splits": df.R}
                if cf is not None:
                    row |= {m: cf.mean(m) for m in (*METRICS, "ls_auc") if m in cf.values}
                    row |= {"mc_se_v_auc": cf.mc_se("v_auc"), "tie_rate": cf.tie_rate}
                    r = an["descs"][d].get((arm, p, pair, s))
                    if r is not None and ro == rules.PRIMARY:
                        row["full_validation_epoch"] = r["epoch"]
                rows.append(row)
    return rows


def z_report(an: Mapping, contrasts: Sequence[str]) -> dict:
    out = {}
    for c in contrasts:
        for ro in rules.READOUTS:
            for p in rules.PIPELINE_NAMES:
                for m in rules.VERDICT_METRICS:
                    out[f"{c}/{ro}/{p}/{m}"] = {
                        f"{d}/{pr}": {"delta": an["delta"][(ro, p, c, m)][(d, pr)],
                                      "z": an["z"][(ro, p, c, m)][(d, pr)]}
                        for d, pr in rules.SETTINGS}
    return out


def projected_hours(meta_files: Sequence[Path] | None,
                    inputs: Mapping[str, Mapping]) -> dict | None:
    """Start condition (3): the stage-2 extraction time at the throughput measured in the
    stage-1 production run (val phase) of each dataset (one dense meta.json per dataset)."""
    if not meta_files:
        return None
    rate = {}
    for f in meta_files:
        meta = _json(f)
        runs = [r for r in meta.get("runs", []) if r.get("phase") == "val"
                and r.get("seconds") and r.get("videos_finished")]
        if runs:
            rate[meta["dataset"]] = sum(r["snippets"] for r in runs) / sum(r["seconds"]
                                                                          for r in runs)
    if set(rate) != set(DATASETS):
        return {"hours": None, "rate": rate, "reason": "a dataset's val-phase run is missing"}
    hours = sum(inputs[d]["trainpart_snippets"] / rate[d] / 3600 for d in DATASETS)
    return {"hours": hours, "snippets_per_second": rate,
            "trainpart_snippets": {d: inputs[d]["trainpart_snippets"] for d in DATASETS}}


def all_gates(args: argparse.Namespace, stage: int, runs: Mapping, errors: Sequence[str],
              identity: Mapping, names: Mapping, freeze: str | None) -> dict:
    """Every gate the stage needs before its verdict: analysis gates (without (h)), the
    extraction gate record with the user approvals, the no-change gates and the freeze descent
    of the (g1), (x4), batch, scoring and analysis commits (``freeze``: the freeze commit). A run
    whose dense scores are non-finite is marked failed by the score gates (numerical failure)."""
    keys = [k for st in range(1, stage + 1) for k in run_keys(st)]
    code = code_version()
    record = _json_or(EXTRACT_GATE, {})
    tg = gates.training_gates(args.runs, runs)
    sg = {st: gates.score_gates(runs, args.scores, st) for st in range(1, stage + 1)}
    files = {st: gates.verify_scores(args.scores, st, run_keys(st)) for st in range(1, stage + 1)}
    fg = gates.gate_f(args.runs, args.scores, stage, identity)
    ext = rules.extraction_gates(record, _json_or(APPROVALS, {}), stage, names)
    descent = gates.freeze_descent(freeze, identity, record, args.runs, args.scores, stage,
                                   code.get("commit"))
    _, _, ident = gates.identity_target(identity)
    status = {**{g: tg["status"][g] for g in "abcdeg"},
              **{g: all(sg[st]["status"].get(g, True) for st in sg) for g in ("s", "s2", "p")},
              "f": fg["ok"], "score_files": all(f["ok"] for f in files.values()),
              "extraction": ext["ok"], "g1": ident["g1"], "g2": ident["g2"],
              "freeze_descent": descent["ok"]}
    if stage == 2:
        status["s3"] = sg[2]["status"]["s3"]
    return {"code": code, "stage": stage, "runs_expected": len(keys),
            "errors": [*errors, *tg["errors"], *(e for s in sg.values() for e in s["errors"])],
            "status": status, "training": tg, "scores": sg, "score_files": files, "f": fg,
            "extraction": ext, "identity": ident, "freeze_descent": descent,
            "numerically_failed": sorted(f"{k.chain}/{k.run}" for k, r in runs.items()
                                         if r.failed)}


def gated_analysis(args: argparse.Namespace, stage: int, gate_file: Path,
                   plan: Mapping) -> tuple[dict, dict]:
    """Gates, then (if they pass) the analysis and gate (h); stops with the gate file written
    when a gate fails (PLAN.md: no verdict without the gates). ``plan``: ``plan_seal()``."""
    lineage = rules.lineage_thresholds()
    runs, errors = gates.stage_runs(args.runs, range(1, stage + 1))
    identity = _json_or(gates.IDENTITY, {})
    names = {d: load_manifest(manifest_file(d)).names for d in DATASETS}
    gate = all_gates(args, stage, runs, errors, identity, names, plan.get("freeze_commit"))
    write_json(gate_file, _clean(gate))
    if gate["errors"] or not all(gate["status"].values()):
        raise SystemExit(f"gate failed or runs unreadable (PLAN.md failure handling); see "
                         f"{gate_file.name}: {[k for k, v in gate['status'].items() if not v]}")
    an = analyze(args.runs, args.scores, stage, runs, lineage)
    gate["h"] = h_gate(an["fits"])
    gate["status"]["h"] = gate["h"]["ok"]
    write_json(gate_file, _clean(gate))
    if not gate["h"]["ok"]:
        raise SystemExit(f"gate (h) failed: the h0 runs differ from the earlier records; see "
                         f"{gate_file.name}")
    an |= {"runs": runs, "lineage": lineage, "gate": gate, "identity": identity}
    return an, gate


def descriptives(an: Mapping, stage: int) -> dict:
    return {"datasets": an["dataset_descriptives"],
            "contrasts": desc.contrast_summary(an["fits"], an["delta"], an["z"], stage,
                                               an["lineage"], an["ls"]),
            "training": an["train"], "features": desc.feature_stats(an["features"]),
            "extraction": desc.extraction(_json_or(EXTRACT_GATE, {}), an["features"])}


def flags(an: Mapping, stage: int, per_pipeline: Mapping) -> dict:
    approx = [(t["dataset"], t["encoder"]) for t in an["gate"]["extraction"]["tags"]
              if t["kind"] == "approximate_equivalence"]
    norm = {d: v["feature_norm_v_auc"] for d, v in an["dataset_descriptives"].items()}
    names = sorted({a for d in DATASETS for a in arms.arms_of(stage, d)})
    return desc.interpretation_flags(stage, an["z"], an["delta"], an["fits"], an["lineage"],
                                     norm, an["train"], approx, per_pipeline, names)


def header(an: Mapping, plan: Mapping) -> dict:
    return {"code": code_version(), "plan": plan,
            "lineage": an["lineage"], "ls_thresholds": an["ls"],
            "cross_fit": {d: {"R": f.R, "splits_sha256": f.splits_sha256, "mc": f.mc}
                          for d, f in an["fits"].items()},
            "numerically_failed": an["gate"]["numerically_failed"],
            "tags": an["gate"]["extraction"]["tags"]}


def stage1_command(args: argparse.Namespace) -> None:
    plan = plan_seal()
    an, gate = gated_analysis(args, 1, OUT["stage1_gate"], plan)
    an["train"] = desc.training(an["runs"], {d: v["labels"] for d, v in an["inputs"].items()},
                                an["lineage"])
    verdict = rules.stage1_verdict(an["z"])
    per_pipeline = {p: rules.stage1_category(an["z"], "A0", (p,))[0]
                    for p in rules.PIPELINE_NAMES}
    hours = projected_hours(args.extract_meta, an["inputs"])
    gate_commit = gates.identity_target(an["identity"])[1]
    start = rules.start_conditions(verdict["category"], True,
                                   None if hours is None else hours.get("hours"),
                                   code_same=gates.frozen_code_unchanged(gate_commit))
    write_csv(OUT["stage1_runs"], [_clean(r) for r in run_rows(an)])
    write_json(OUT["stage1_summary"], _clean({
        "check": "E021 stage 1 (A0, validation videos; PLAN.md 1단계 판정)", **header(an, plan),
        "verdict": verdict, "per_pipeline_category": per_pipeline,
        "start_conditions": start | {"projection": hours,
                                     "pending_note": "(3) user decision when over 24 h; (4) "
                                                     "check x6 --recheck before stage 2"},
        "tables": z_report(an, arms.VERDICT_CONTRASTS[1]),
        "flags": flags(an, 1, per_pipeline), "descriptives": descriptives(an, 1)}))
    print("stage 1:", verdict["category"], verdict["flags"], "start:", start["conditions"])


def not_run_verdict(args: argparse.Namespace, plan: Mapping, s1: Mapping | None) -> dict:
    if s1 is not None and s1["verdict"]["category"] == rules.STAGE1_NOGO:
        raise SystemExit("stage 1 was NO-GO: run `verdict` without --not_run (1단계 NO-GO)")
    if args.not_run != "gate" and (s1 is None or
                                   s1["verdict"]["category"] != rules.CONTINUE):
        raise SystemExit("start conditions (3)-(5) apply after a stage-1 'continue' only")
    category = rules.GATE_FAILED if args.not_run == "gate" else rules.NOT_RUN
    reason = None if args.not_run == "gate" else args.not_run
    tags = [] if s1 is None else s1.get("tags", [])
    return {"category": category, "reason": reason,
            "category_plan": rules.plan_name(category, reason),
            "label": rules.label(category, reason, tags), "tags": tags,
            "stated_reason": args.reason, "stage2_run": False}


def verdict_command(args: argparse.Namespace) -> None:
    plan = plan_seal()
    s1 = _json_or(OUT["stage1_summary"], None)
    base = {"check": "E021 final verdict (PLAN.md 최종 판정)", "code": code_version(),
            "plan": plan, "stage1_summary_sha256": (sha256_of(OUT["stage1_summary"])
                                                    if s1 is not None else None)}
    if args.not_run:
        if not args.reason:
            raise SystemExit("--not_run needs --reason '<what failed>'")
        out = base | {"verdict": not_run_verdict(args, plan, s1)}
    elif s1 is None:
        raise SystemExit("no stage-1 summary: a stage-1 gate failure is recorded with "
                         "--not_run gate --reason '<gate>'")
    elif s1["verdict"]["category"] == rules.STAGE1_NOGO:
        tags = s1.get("tags", [])
        out = base | {"verdict": {"category": rules.STAGE1_NOGO, "reason": None,
                                  "category_plan": rules.plan_name(rules.STAGE1_NOGO),
                                  "label": rules.label(rules.STAGE1_NOGO, None, tags),
                                  "tags": tags, "stage2_run": False}}
    else:
        if args.runs is None or args.scores is None:
            raise SystemExit("the final verdict needs --runs and --scores")
        out = base | final_analysis(args, plan)
    write_json(OUT["verdict"], _clean(out))
    print("final:", out["verdict"]["label"])


def final_analysis(args: argparse.Namespace, plan: Mapping) -> dict:
    an, gate = gated_analysis(args, 2, OUT["gate"], plan)
    an["train"] = desc.training(an["runs"], {d: v["labels"] for d, v in an["inputs"].items()},
                                an["lineage"])
    fv = rules.final_verdict(an["z"], gates_ok=True,
                             numerical_ok=not gate["numerically_failed"])
    tags = gate["extraction"]["tags"]
    verdict = {**fv, "reason": None, "label": rules.label(fv["category"], None, tags),
               "tags": tags, "stage2_run": True,
               "stage1": rules.stage1_verdict(an["z"])["category"]}
    write_csv(OUT["runs"], [_clean(r) for r in run_rows(an)])
    contrasts = (*arms.VERDICT_CONTRASTS[1], *arms.VERDICT_CONTRASTS[2])
    return {**header(an, plan), "verdict": verdict,
            "tables": z_report(an, contrasts),
            "flags": flags(an, 2, fv["per_pipeline"]) | {
                "extreme_readout_sensitive": fv["extreme_readout_sensitive"]},
            "descriptives": descriptives(an, 2)}


# --- after the verdict ------------------------------------------------------------------------

def committed_unmodified(path: Path) -> bool:
    rel = Path(path).relative_to(REPO_ROOT).as_posix()
    return (gates.git("cat-file", "-e", f"HEAD:{rel}").returncode == 0
            and not gates.git("status", "--porcelain", "--", rel).stdout.strip())


def choice_command(args: argparse.Namespace) -> None:
    """The user's option after a hold or no-verdict category (PLAN.md "결과의 쓰임"), and whether
    it was taken before or after ``describe`` (not enforced; disclosed)."""
    if not OUT["verdict"].is_file():
        raise SystemExit("no e021_verdict.json")
    before = sha256_of(OUT["verdict"])
    v = _json(OUT["verdict"])
    cat = v["verdict"]["category"]
    if cat not in OPTIONS or args.option not in OPTIONS[cat]:
        raise SystemExit(f"category {cat}: options {OPTIONS.get(cat, ())}")
    d = _json_or(OUT["describe"], None)
    after = bool(d and d.get("complete"))
    v.setdefault("user_choices", []).append({
        "option": args.option, "user_words": args.quote,
        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "after_describe": after, "verdict_sha256_before_this_entry": before})
    write_json(OUT["verdict"], v)
    print(f"choice {args.option} recorded ({'after' if after else 'before'} describe)")


def describe_command(args: argparse.Namespace) -> None:
    """Temporal-annotation diagnostics (PLAN.md "시간 주석 진단"), once, after the verdict file
    is committed; the glance and UCA modules are imported here only."""
    verdict = OUT["verdict"]
    if not verdict.is_file() or not committed_unmodified(verdict):
        raise SystemExit("describe runs only after results/e021_verdict.json is committed "
                         "(unmodified at HEAD)")
    previous = _json_or(OUT["describe"], None)
    if previous is not None and previous.get("complete"):
        raise SystemExit("describe ran already (once, PLAN.md)")
    start = {"verdict_sha256_start": sha256_of(verdict),
             "time_utc_start": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "code": code_version(), "complete": False,
             "earlier_incomplete": previous}
    write_json(OUT["describe"], start)
    from wvad.diagnostics import e021_describe  # annotations: imported here only

    v = _json(verdict)
    result = e021_describe.describe(args.runs, args.scores, bool(v["verdict"].get("stage2_run")))
    end = sha256_of(verdict)
    write_json(OUT["describe"], _clean({
        **start, **result, "verdict_sha256_end": end,
        "verdict_unchanged": end == start["verdict_sha256_start"],
        "time_utc_end": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "complete": True}))
    print("describe:", OUT["describe"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E021 dense-snippet pilot: local analysis")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    s1 = sub.add_parser("stage1")
    v = sub.add_parser("verdict")
    for p in (s1, v):
        p.add_argument("--runs", type=Path, required=p is s1)
        p.add_argument("--scores", type=Path, required=p is s1)
    s1.add_argument("--extract_meta", nargs="+", type=Path,
                    help="copies of the dense caches' meta.json (one per dataset): stage-1 "
                         "throughput for start condition (3)")
    v.add_argument("--not_run", choices=("gate", "time", "videos", "defect"))
    v.add_argument("--reason")
    c = sub.add_parser("choice")
    c.add_argument("--option", required=True, choices=("i", "ii", "iii", "iv"))
    c.add_argument("--quote", required=True, help="the user's words")
    d = sub.add_parser("describe")
    d.add_argument("--runs", type=Path, required=True)
    d.add_argument("--scores", type=Path, required=True)
    args = ap.parse_args(argv)
    {"freeze": freeze_command, "stage1": stage1_command, "verdict": verdict_command,
     "choice": choice_command, "describe": describe_command}[args.command](args)


if __name__ == "__main__":
    main()
