"""E017 (stage D/E): the E016 G1a head (z' = z + gamma * mean z, gamma from 0) with a learning-rate
multiplier on gamma only (`--offset_lr_mult`), each arm retrained from stage 1 (20 epochs) ->
ckpt_last.pt -> stage 2 (30 epochs, no pseudo labels).

Gates (a)-(f) first; then the E010 common rule per candidate on Delta alone (outcome_delta)
against the reused E013 L20 runs (cross-checked against e013_runs.csv at the E010 R), and the
choice between adopted candidates (PLAN.md). The E016 G1a runs (multiplier 1) are a reference
arm for the dose-response descriptive only (cross-checked against e016_runs.csv). Video labels
only; the UCF UCA proxy AP is computed last, after the verdict is written, and is descriptive
only (decision 0004). No test data.

    python -m wvad.diagnostics.e017 --e003 <e003 copy> --e013 <e013 copy> --e016 <e016 copy> \
        --runs <e017 copy>
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from wvad.diagnostics import e016
from wvad.diagnostics.common import (
    DEV_SEEDS,
    code_version,
    gpu_environment,
    write_csv,
    write_json,
)
from wvad.diagnostics.e011 import score_shape
from wvad.diagnostics.e012 import ZERO, _expect2
from wvad.diagnostics.e013 import (
    COMMON1,
    LAST,
    XCL,
    init_parts,
    same_record_at,
    sd_validity,
    selection_share,
    start_value,
)
from wvad.diagnostics.e014 import GATE_ERRORS, e013_reference
from wvad.diagnostics.e015 import (
    convergence_flag,
    curves,
    decided,
    differ,
    env_check,
    epoch_scores,
    identity_target,
    loss_change,
    same_env,
    stage1_end,
    tidy,
    top_scores,
)
from wvad.diagnostics.runs import PAIRS
from wvad.diagnostics.stage_d import (
    DATASETS,
    SETTINGS,
    Arm,
    DatasetData,
    _fit,
    calibration,
    check_reference,
    choose,
    classify,
    compare,
    fit_dataset,
    load_dataset,
    run_rows,
)
from wvad.eval.crossfit import METRICS
from wvad.paths import REPO_ROOT

E017_RESULTS = REPO_ROOT / "experiments" / "E017_offset_scale" / "results"
E016_RUNS_CSV = REPO_ROOT / "experiments" / "E016_global_offset" / "results" / "e016_runs.csv"
BASE, REF = "l20", "g1a"                 # baseline (verdict) and E016 G1a (descriptive only)
MULT = {"m3": 3.0, "m10": 10.0}          # candidates; stage_d.choose order (tie -> smaller m)
CANDIDATES = tuple(MULT)
START_EPOCH = 20
ADAM_BOUND = {"ucf": 0.12, "xdv": 0.30}  # gamma reachable at multiplier 1 (E016 PLAN.md)
BUDGET_FRACTION = 0.8
GAMMA_EPOCHS = {"stage1": (5, 10, 15, 20), "stage2": (10, 20, 30)}


def arms(e003: Path, e013: Path, e016_runs: Path, runs: Path) -> dict[str, Arm]:
    mean = {"global_mode": "mean"}
    return {BASE: Arm(BASE, e013, "stage1", BASE, ZERO, stage1_root=e003, expect1=COMMON1,
                      expect2=_expect2(10), init_ckpt=LAST),
            REF: Arm(REF, e016_runs, f"{REF}_stage1", REF, ZERO, expect1={**COMMON1, **mean},
                     expect2={**_expect2(10), **mean}, init_ckpt=LAST),
            **{a: Arm(a, runs, f"{a}_stage1", a, ZERO,
                      expect1={**COMMON1, **mean, "offset_lr_mult": m},
                      expect2={**_expect2(10), **mean, "offset_lr_mult": m}, init_ckpt=LAST)
               for a, m in MULT.items()}}


def bound(dataset: str, arm: str) -> float:
    """Distance Adam can move gamma within the budget for this arm (PLAN.md table)."""
    return ADAM_BOUND[dataset] * (MULT[arm] if arm in MULT else 1.0)


# --- gates ----------------------------------------------------------------------------------

def _config(d: Path) -> dict:
    return torch.load(d / LAST, map_location="cpu", weights_only=True)["config"]


def gate_chain(e003: Path, e013: Path, e016_runs: Path, runs: Path, chain: str) -> dict:
    c, ref, g1a = runs / chain, e003 / chain / "stage1", e016_runs / chain / f"{REF}_stage1"
    row = {}
    for a, m in MULT.items():
        s1, s2 = c / f"{a}_stage1", c / a
        e016.check_offset_record(s2)
        row[f"a_{a}"] = same_record_at(s2, 0, s1, START_EPOCH)
        row[f"b_{a}"] = init_parts(s2) == (chain, f"{a}_stage1", LAST)
        row[f"c_{a}"] = all(_config(d).get("global_mode") == "mean"
                            and _config(d).get("offset_lr_mult") == m for d in (s1, s2))
        row[f"d_{a}"] = differ(epoch_scores(s1, 1), epoch_scores(g1a, 1))
        row[f"e_{a}"] = e016.learned("g1a", e016.offsets_of(s1), e016.offsets_of(s2))
    row["c_base"] = all(_config(d).get("global_mode", "none") == "none"
                        for d in (ref, e013 / chain / BASE))
    names = list(MULT)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row[f"d_{a}_{b}"] = differ(epoch_scores(c / f"{a}_stage1", 1),
                                       epoch_scores(c / f"{b}_stage1", 1))
    return row


GATE_KEYS = ([f"{g}_{a}" for g in "abcde" for a in CANDIDATES] + ["c_base"]
             + [f"d_{a}_{b}" for i, a in enumerate(CANDIDATES) for b in CANDIDATES[i + 1:]])


def gates(e003: Path, e013: Path, e016_runs: Path, runs: Path, identity: Path) -> dict:
    rows = []
    for d in DATASETS:
        for p in PAIRS:
            for s in DEV_SEEDS:
                chain = f"{d}_{p}_s{s}"
                try:
                    row = {"chain": chain, **gate_chain(e003, e013, e016_runs, runs, chain)}
                except GATE_ERRORS as exc:
                    row = {"chain": chain, "error": repr(exc)}
                rows.append(row)
    status = {k: all(r.get(k, False) for r in rows if "error" not in r) for k in GATE_KEYS}
    env = env_check(runs, identity)
    status["f"] = env["ok"]
    return {"chains": rows, "status": status, "environment": env,
            "errors": [r["chain"] for r in rows if "error" in r]}


def record_env(path: str, identity: str | None = None) -> None:
    """Server, before a candidate batch (README): write this machine's environment and deployed
    commit to ``path`` and require them to equal the passed no-change gate (gate (f)); exits
    non-zero otherwise, so the batch is not launched."""
    env = {**gpu_environment(), "commit": code_version()["commit"]}
    write_json(Path(path), env)
    want, passed = identity_target(Path(identity) if identity
                                   else E017_RESULTS / "e017_identity.json")
    if not (passed and same_env(env, want)):
        raise SystemExit(f"environment or commit differs from the no-change gate: {env} vs "
                         f"{want} (gate passed: {passed}); do not launch")
    print(f"{path}: matches the no-change gate")


# --- reference arm --------------------------------------------------------------------------

def e016_reference(dataset: str, path: Path | None = None) -> dict[tuple, dict[str, float]]:
    """Cross-fitted values of the E016 G1a runs as written by E016 (at the E010 R)."""
    with open(path or E016_RUNS_CSV, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["arm"] == REF and r["dataset"] == dataset]
    return {(r["pair"], int(r["seed"])): {m: float(r[m]) for m in METRICS} for r in rows}


def check_ref_arm(dataset: str, cell: Arm, data: DatasetData, fits, cal) -> str:
    """The reused G1a runs must reproduce e016_runs.csv at the E010 R (before any escalation)."""
    at_r = fits if fits.R == cal["R"] else _fit(data, cal["R"])
    check_reference(dataset, cell, at_r, e016_reference(dataset))
    return f"passed at R={cal['R']} ({', '.join(METRICS)})"


# --- descriptives ---------------------------------------------------------------------------

def gamma_track(data: DatasetData, arm: str, pair: str) -> list[dict]:
    """Per seed: gamma at the planned epochs, the stage-2 movement ratio (last 5 epochs over the
    first 5) and the final value."""
    out = []
    for s in DEV_SEEDS:
        run = data.runs[(arm, pair, s)]
        g1 = {h["epoch"]: h["offset_gamma"] for h in run.history1}
        g2 = [h["offset_gamma"] for h in run.history2]
        first, last = g2[4] - g1[max(g1)], g2[-1] - g2[-6]
        out.append({"stage1": {str(e): g1[e] for e in GAMMA_EPOCHS["stage1"]},
                    "stage2": {str(e): g2[e - 1] for e in GAMMA_EPOCHS["stage2"]},
                    "stage2_last5_over_first5": (last / first) if first != 0 else None,
                    "end": g2[-1]})
    return out


def budget(data: Mapping[str, DatasetData], arms_: Sequence[str]) -> dict:
    """Budget rule per arm (PLAN.md): any of its 12 runs with |gamma| at the end of stage 2 >=
    0.8 x the arm's reachable distance -> 'budget-bound'."""
    out = {}
    for a in arms_:
        hits = []
        for d in DATASETS:
            for p in PAIRS:
                for s in DEV_SEEDS:
                    g = data[d].runs[(a, p, s)].history2[-1]["offset_gamma"]
                    if abs(g) >= BUDGET_FRACTION * bound(d, a):
                        hits.append({"run": f"{d}_{p}_s{s}", "gamma": g, "bound": bound(d, a)})
        out[a] = {"budget_bound": bool(hits), "runs": hits}
    return out


def arm_minus(fits: Mapping, a: str, b: str, thr: Mapping[str, float]) -> dict:
    """Arm a - arm b per setting (Delta/delta, per seed) and the common-rule class (descriptive)."""
    deltas, out = {}, {}
    for d, p in SETTINGS:
        f2 = fits[d].stage2
        per = [f2[(a, p, s)].mean("v_auc") - f2[(b, p, s)].mean("v_auc") for s in DEV_SEEDS]
        deltas[(d, p)] = float(np.mean(per))
        out[f"{d}/{p}"] = {"delta_over_threshold": deltas[(d, p)] / thr[d],
                           "per_seed_over_threshold": [x / thr[d] for x in per]}
    return {"settings": out, "class_descriptive": classify(deltas, thr)}


def dose_response(fits: Mapping, data: Mapping[str, DatasetData], ok: Sequence[str],
                  thr: Mapping[str, float]) -> dict:
    """Delta/delta vs the baseline and mean final gamma for multiplier 1 (E016 G1a), and each
    non-failed candidate; same baseline, same splits."""
    out = {}
    for d, p in SETTINGS:
        f2 = fits[d].stage2
        base = np.mean([f2[(BASE, p, s)].mean("v_auc") for s in DEV_SEEDS])
        row = {}
        for a in (REF, *ok):
            v = np.mean([f2[(a, p, s)].mean("v_auc") for s in DEV_SEEDS])
            g = np.mean([data[d].runs[(a, p, s)].history2[-1]["offset_gamma"] for s in DEV_SEEDS])
            row[a] = {"multiplier": MULT.get(a, 1.0),
                      "delta_over_threshold": float((v - base) / thr[d]),
                      "gamma_end_mean": float(g)}
        out[f"{d}/{p}"] = row
    return out


def descriptives(cells, data, fits, cal, compared: Mapping[str, dict], ok: list[str],
                 verdicts: Mapping[str, dict]) -> dict:
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    arms_all = (BASE, REF, *ok)
    out, dec_by_arm = {}, {a: {} for a in ok}
    for d, p in SETTINGS:
        k = f"{d}/{p}"
        tracks = {a: gamma_track(data[d], a, p) for a in (REF, *ok)}
        entry = {
            "curves": curves(data[d], arms_all, p),
            "stage1_epoch20_minus_base_over_delta": {a: stage1_end(data[d], a, p, thr[d])
                                                     for a in (REF, *ok)},
            "loss_mil_change": {a: loss_change(data[d], a, p) for a in arms_all},
            "tie_rate_selected": {a: float(np.mean([e016.tie_rate(data[d].runs[(a, p, s)].s2_best,
                                                                  data[d].labels)
                                                    for s in DEV_SEEDS])) for a in arms_all},
            "gamma": tracks,
            "gamma_positive_seeds": {a: sum(r["end"] > 0 for r in t) for a, t in tracks.items()},
            "amp_skipped_steps": {a: e016.amp_skips(data[d], a, p) for a in ok},
            "offset_distribution": {a: e016.offset_distribution(cells[a], data[d], d, p)
                                    for a in ok},
            "decomposition": {},
        }
        for a in ok:
            dec = e016.decomposition(cells[a], data[d], fits[d], d, p, thr[d])
            entry["decomposition"][a] = dec
            dec_by_arm[a][k] = dec
        if p == XCL:
            entry["convergence_flag"] = {a: convergence_flag(data[d], a, p, thr[d]) for a in ok}
            entry["share_last_10pct"] = {a: selection_share(fits[d], a, p, 28, 30)
                                         for a in arms_all}
        else:
            entry["share_epochs_1_3"] = {a: selection_share(fits[d], a, p, 1, 3)
                                         for a in arms_all}
        keys = ("delta", "delta_over_threshold", "aux_opposite_2delta",
                "split_half_sign_agreement", "candidate_seed_sd_over_sd_d",
                "paired_delta_se_over_sigma", "per_seed_delta", "selection")
        entry["compare"] = {a: {x: compared[a]["settings"][k][x] for x in keys} for a in ok}
        out[k] = entry
    pairs = [(b, a) for i, a in enumerate(CANDIDATES) for b in CANDIDATES[i + 1:]
             if a in ok and b in ok]
    return {"settings": out,
            "dose_response": dose_response(fits, data, ok, thr),
            "arm_differences": {f"{a}_minus_{b}": arm_minus(fits, a, b, thr) for a, b in pairs},
            "budget": budget(data, (REF, *ok)),
            "characterization": {a: e016.characterize(dec_by_arm[a]) for a in ok
                                 if verdicts[a]["verdict"] == "adopt"}}     # adoptions only


def csv_rows(cells: Mapping[str, Arm], data, fits) -> list[dict]:
    rows = run_rows([cells[BASE], cells[REF], *(cells[a] for a in CANDIDATES)], data, fits)
    for row in rows:
        if row["failed"]:
            continue
        d, arm, p, s = row["dataset"], row["arm"], row["pair"], row["seed"]
        best = data[d].runs[(arm, p, s)].s2_best
        row |= {"s1_epoch": START_EPOCH, "s1_optimism": 0.0,       # no stage-1 selection
                "s1_selected": start_value(data[d], arm, p, s), "s1_crossfit_v_auc": ""}
        row |= score_shape(best, data[d].labels) | top_scores(best, data[d].labels)
        if arm != BASE:
            row["offset_lr_mult"] = MULT.get(arm, 1.0)
            row["stage2_end_offset_gamma"] = data[d].runs[(arm, p, s)].history2[-1][
                "offset_gamma"]
    return rows


# --- main -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E017 learning-rate multiplier on the G1a gamma")
    ap.add_argument("--e003", required=True, type=Path)
    ap.add_argument("--e013", required=True, type=Path)
    ap.add_argument("--e016", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--no_proxy", action="store_true",
                    help="skip the UCA proxy AP (descriptive only; the verdict is unchanged)")
    args = ap.parse_args(argv)
    gate = gates(args.e003, args.e013, args.e016, args.runs, E017_RESULTS / "e017_identity.json")
    write_json(E017_RESULTS / "e017_gate.json", {"code": code_version(), **gate})
    if gate["errors"]:
        raise SystemExit(f"runs missing or unreadable for {gate['errors']} (PLAN.md); "
                         "see results/e017_gate.json")
    if not all(gate["status"].values()):
        raise SystemExit("gate failed (config/init: rerun; (d)/(e): code defect; (f): commit "
                         "or environment; PLAN.md); see results/e017_gate.json")
    cells = arms(args.e003, args.e013, args.e016, args.runs)
    cal = calibration()
    data = {d: load_dataset(list(cells.values()), d) for d in DATASETS}
    fits = {d: fit_dataset(data[d], cal[d], d, baseline=cells[BASE],
                           reference=e013_reference(d)) for d in DATASETS}
    ref_checks = {d: check_ref_arm(d, cells[REF], data[d], fits[d], cal[d]) for d in DATASETS}
    compared = {a: compare(cells[a], cells[BASE], data, fits, cal) for a in CANDIDATES}
    verdicts = {a: decided(compared[a]) for a in CANDIDATES}
    chosen = choose([verdicts[a] for a in CANDIDATES])
    ok = [a for a in CANDIDATES if verdicts[a]["verdict"] != "failed"]
    for a in ok:
        tidy(compared[a], a, data)
    summary = {
        "check": "E017 learning-rate multiplier on the G1a gamma (PLAN.md)",
        "code": code_version(), "calibration": cal, "gate": gate["status"],
        "cross_fit": {d: {"R": f.R, "splits_sha256": f.splits_sha256, "mc": f.mc,
                          "checks": {**f.checks, "reference_arm_vs_e016": ref_checks[d]}}
                      for d, f in fits.items()},
        "verdicts": verdicts,
        "deltas": {a: {k: st["delta"] for k, st in compared[a].get("settings", {}).items()}
                   for a in CANDIDATES},
        "deltas_over_threshold": {a: {k: st["delta_over_threshold"]["v_auc"]
                                      for k, st in compared[a].get("settings", {}).items()}
                                  for a in CANDIDATES},
        "chosen": chosen,
        "sd_validity": sd_validity(fits, chosen, cal) if chosen else None,
        "notes": ["verdict = outcome_delta (Delta alone); the Delta' fields and verdicts of "
                  "stage_d.compare are dropped: no arm selects a stage-1 checkpoint",
                  "stage-1 fields relabelled: start epoch 20, optimism 0",
                  "the E016 G1a runs (multiplier 1) are a descriptive reference arm only"],
    }
    write_json(E017_RESULTS / "e017_summary.json", summary)     # 1. verdict and choice
    for a in CANDIDATES:
        print(a, verdicts[a]["verdict"],
              {k: round(v, 2) for k, v in summary["deltas_over_threshold"][a].items()})
    print("chosen:", chosen)
    write_csv(E017_RESULTS / "e017_runs.csv", csv_rows(cells, data, fits))
    summary["descriptives"] = descriptives(cells, data, fits, cal, compared, ok, verdicts)
    write_json(E017_RESULTS / "e017_summary.json", summary)     # 2. descriptives
    if not args.no_proxy:                                         # 3. after the verdict (0004)
        summary["uca_proxy_ap"] = {
            "note": "descriptive only (decision 0004): not used for the verdict, the choice, "
                    "the pipeline, the next candidate, reruns or claims about localisation; "
                    "low split-half reliability in E003 C5",
            "ucf": e016.proxy_summary(data["ucf"], ok)}
        write_json(E017_RESULTS / "e017_summary.json", summary)


if __name__ == "__main__":
    main()
