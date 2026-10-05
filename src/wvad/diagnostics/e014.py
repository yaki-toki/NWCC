"""E014 (stage D3): pseudo-label self-training added to the new reference pipeline (E013 L20);
arms ST3 (pseudo weights 1.0 / 0.3) and STw (0.1 / 0.03), both 3 rounds x 10 epochs with the
curriculum, started from the E003 stage-1 last checkpoint like the baseline.

Gates (a)-(c) first; then the E010 common rule per candidate against the reused E013 L20 runs
(cross-checked against e013_runs.csv on all metrics at the E010 R) and the choice between
adopted candidates (PLAN.md). Video labels only; the UCF UCA proxy AP is computed last, after
the verdict is written, and is descriptive only (decision 0004). No test data.

    python -m wvad.diagnostics.e014 --e003 <e003 copy> --e013 <e013 copy> --runs <e014 copy>
"""

from __future__ import annotations

import argparse
import csv
import pickle
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from wvad.diagnostics.common import DEV_SEEDS, code_version, write_csv, write_json
from wvad.diagnostics.e011 import proxy_aps, score_shape
from wvad.diagnostics.e012 import PSEUDO_STATS, ZERO, _expect2, _json
from wvad.diagnostics.e013 import (
    COMMON1,
    LAST,
    XCL,
    init_parts,
    relabel_stage1,
    same_record_at,
    sd_validity,
    selection_share,
    start_value,
)
from wvad.diagnostics.runs import PAIRS
from wvad.diagnostics.stage_d import (
    DATASETS,
    SETTINGS,
    Arm,
    DatasetData,
    calibration,
    choose,
    compare,
    fit_dataset,
    load_dataset,
    run_rows,
)
from wvad.eval.crossfit import METRICS, rank_auc
from wvad.paths import REPO_ROOT

E013_RESULTS = REPO_ROOT / "experiments" / "E013_stage1_last_checkpoint" / "results"
E014_RESULTS = REPO_ROOT / "experiments" / "E014_self_training" / "results"
BASE = "l20"
CANDIDATES = ("stw", "st3")             # stage_d.choose order: a tie goes to STw (PLAN.md)
WEIGHTS = {"st3": {**ZERO, "pseudo": 1.0, "neg_pseudo": 0.3},
           "stw": {**ZERO, "pseudo": 0.1, "neg_pseudo": 0.03}}
START_EPOCH = 20
MIL_WEIGHT = 0.1
PSEUDO_NEG_K = 4
CURVE_EPOCHS = (0, 1, 10, 20, 30)
DIFF_EPOCHS = (10, 20, 30)
LOSS_EPOCHS = (1, 10, 11, 20, 21, 30)
ROUNDS = ((1, 10), (11, 20), (21, 30))
DIFF_PAIRS = (("st3", BASE), ("stw", BASE), ("st3", "stw"))
ROUND1_KEYS = ("scores", "labels", "top_k", "confidence")


def arms(e003: Path, e013: Path, runs: Path) -> dict[str, Arm]:
    e2 = _expect2(10)
    common = {"stage1_root": e003, "expect1": COMMON1, "expect2": e2, "init_ckpt": LAST}
    return {BASE: Arm(BASE, e013, "stage1", BASE, ZERO, **common),
            **{a: Arm(a, runs, "stage1", a, WEIGHTS[a], weights1=ZERO, **common)
               for a in CANDIDATES}}


# --- gates ----------------------------------------------------------------------------------

def round1(d: Path) -> dict[str, np.ndarray]:
    """Round-1 arrays of a stage-2 run's pseudo-label record (stored labels, no regeneration)."""
    z = np.load(d / "record_pseudo.npz")
    return {**{k: z[k][0] for k in ROUND1_KEYS}, "train_indices": z["train_indices"]}


def same_arrays(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> bool:
    """Same keys, dtypes, shapes and values (NaN = NaN, so NaN positions must match)."""
    return not differing(a, b)


def differing(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> list[str]:
    """Keys whose arrays differ (missing on one side, dtype, shape or values; NaN = NaN)."""
    out = sorted(set(a) ^ set(b))
    for k in a:
        if k in b and not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and bool(
                np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"))):
            out.append(k)
    return out


def mil_weight(d: Path) -> float:
    return float(torch.load(d / LAST, map_location="cpu", weights_only=True)["config"]
                 ["weights"]["mil"])


def gate_chain(e003: Path, e013: Path, runs: Path, chain: str) -> dict:
    s1, l20 = e003 / chain / "stage1", e013 / chain / BASE
    row = {}
    for a in CANDIDATES:
        d = runs / chain / a
        row[f"a_{a}"] = same_record_at(d, 0, s1, START_EPOCH)
        row[f"init_{a}_ok"] = init_parts(d) == (chain, "stage1", LAST)
        row[f"mil_{a}_ok"] = mil_weight(d) == MIL_WEIGHT        # loss-share formula (desc. 6)
    r3, rw, rl = round1(runs / chain / "st3"), round1(runs / chain / "stw"), round1(l20)
    row["b_diff"], row["c_diff"] = differing(r3, rw), differing(r3, rl)
    row["b"], row["c"] = not row["b_diff"], not row["c_diff"]
    return row


GATE_ERRORS = (OSError, KeyError, TypeError, ValueError, IndexError, RuntimeError, EOFError,
               zipfile.BadZipFile, pickle.UnpicklingError)


def server_env(runs: Path) -> dict | None:
    """The server environment recorded before training (README); None if missing or empty."""
    try:
        env = _json(runs / "env.json")
    except (OSError, ValueError):
        return None
    return env if isinstance(env, dict) and env.get("torch") and env.get("cuda") else None


def gates(e003: Path, e013: Path, runs: Path) -> dict:
    rows = []
    for d in DATASETS:
        for p in PAIRS:
            for s in DEV_SEEDS:
                chain = f"{d}_{p}_s{s}"
                try:
                    row = {"chain": chain, **gate_chain(e003, e013, runs, chain)}
                except GATE_ERRORS as exc:
                    row = {"chain": chain, "error": repr(exc)}
                rows.append(row)
    keys = [f"{g}_{a}{ok}" for g, ok in (("a", ""), ("init", "_ok"), ("mil", "_ok"))
            for a in CANDIDATES] + ["b", "c"]
    status = {k: all(r.get(k, False) for r in rows if "error" not in r) for k in keys}
    env = server_env(runs)
    return {"chains": rows, "status": status,
            "errors": [r["chain"] for r in rows if "error" in r]
            + ([] if env else ["env.json"]),
            "server_env": env}


# --- verdicts -------------------------------------------------------------------------------

def e013_reference(dataset: str, path: Path | None = None) -> dict[tuple, dict[str, float]]:
    """Cross-fitted values of the E013 L20 runs as written by E013 (at the E010 R)."""
    with open(path or E013_RESULTS / "e013_runs.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["arm"] == BASE and r["dataset"] == dataset]
    return {(r["pair"], int(r["seed"])): {m: float(r[m]) for m in METRICS} for r in rows}


def check_shared(compared: Mapping[str, dict]) -> None:
    """All three arms start from the same stage-1 run: every setting must list all seeds as
    shared and an optimism difference of exactly 0 (so Delta' = Delta)."""
    for a, c in compared.items():
        for k, st in c.get("settings", {}).items():
            o = st["stage1_optimism"]
            if o["shared_stage1_seeds"] != list(DEV_SEEDS) or o["difference"] != 0.0:
                raise ValueError(f"{a} {k}: stage 1 not shared with the baseline ({o})")


def relabel(compared: dict, arm: str, data: Mapping[str, DatasetData]) -> None:
    """No arm selects a stage-1 epoch: start epoch 20, optimism 0 for both sides (PLAN.md)."""
    relabel_stage1(compared, arm, data)
    relabel_stage1(compared, BASE, data)
    for st in compared.get("settings", {}).values():
        st["stage1_optimism"] |= {"candidate": 0.0, "baseline": 0.0}


# --- descriptives ---------------------------------------------------------------------------

def full_val_auc(data: DatasetData, arm: str, pair: str, seed: int, epoch: int) -> float:
    """Full-validation V-AUC of stage-2 epoch ``epoch`` (0 = the stage-1 epoch-20 start)."""
    run = data.runs[(arm, pair, seed)]
    scores = run.s1[START_EPOCH - 1] if epoch == 0 else run.s2[epoch - 1]
    return float(rank_auc(scores.max(axis=1)[None], data.labels)[0])


def curves(data: DatasetData, arms_: Sequence[str], pair: str) -> dict:
    return {a: {e: float(np.mean([full_val_auc(data, a, pair, s, e) for s in DEV_SEEDS]))
                for e in CURVE_EPOCHS} for a in arms_}


def differences(data: DatasetData, ok: Sequence[str], pair: str, thr: float) -> dict:
    """Per-seed and seed-mean differences at epochs 10, 20, 30 in delta units."""
    out = {}
    for a, b in DIFF_PAIRS:
        if a not in (BASE, *ok) or b not in (BASE, *ok):
            continue
        per = {e: [(full_val_auc(data, a, pair, s, e) - full_val_auc(data, b, pair, s, e)) / thr
                   for s in DEV_SEEDS] for e in DIFF_EPOCHS}
        out[f"{a}-{b}"] = {e: {"per_seed": v, "mean": float(np.mean(v))} for e, v in per.items()}
    return out


def late_decline(diffs: Mapping[str, dict], arm: str) -> dict:
    """P3 descriptive rule (PLAN.md): settings with seed-mean (arm - L20) <= -1 delta at epoch
    30 and below its epoch-10 value; 3 or more -> 'decline added in later rounds'."""
    key = f"{arm}-{BASE}"
    counted = [k for k, dd in diffs.items() if key in dd
               and dd[key][30]["mean"] <= -1 and dd[key][30]["mean"] < dd[key][10]["mean"]]
    return {"settings": counted, "decline_added_in_later_rounds": len(counted) >= 3}


def loss_terms(h: Mapping, w: Mapping[str, float]) -> dict:
    """Pseudo-label / MIL term ratio and the positive and negative pseudo losses of one epoch
    (dense and normal-segment terms are 0 in both arms)."""
    mil = MIL_WEIGHT * h["loss_mil"]
    pseudo = h["loss_total"] - mil
    l_pos = h["loss_pseudo"]
    l_neg = (pseudo - w["pseudo"] * l_pos) / w["neg_pseudo"]
    return {"pseudo_over_mil": pseudo / mil, "pseudo_share": pseudo / h["loss_total"],
            "l_pos": l_pos, "l_neg": l_neg}


def loss_summary(data: DatasetData, arm: str, pair: str) -> dict:
    out = {}
    for e in LOSS_EPOCHS:
        per = [loss_terms(next(h for h in data.runs[(arm, pair, s)].history2 if h["epoch"] == e),
                          WEIGHTS[arm]) for s in DEV_SEEDS]
        out[e] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    return out


def tie_stats(scores: np.ndarray, labels: np.ndarray, k: int,
              neg_k: int = PSEUDO_NEG_K) -> dict[str, int]:
    """Ties among the pseudo-positive videos of one round (videos with a stored label 1):
    k-th = (k+1)-th highest score, k+1 or more segments at 1.0, neg_k-th = (neg_k+1)-th
    lowest score."""
    pos = (labels == 1).any(axis=1)
    sc = scores[pos]
    desc, asc = -np.sort(-sc, axis=1), np.sort(sc, axis=1)
    return {"pseudo_positive_videos": int(pos.sum()),
            "top_k_boundary_ties": int(np.sum(desc[:, k - 1] == desc[:, k])),
            "ones_at_least_k_plus_1": int(np.sum((sc == 1.0).sum(axis=1) >= k + 1)),
            "bottom_boundary_ties": int(np.sum(asc[:, neg_k - 1] == asc[:, neg_k]))}


def pseudo_summary(cell: Arm, data: DatasetData, dataset: str, pair: str) -> dict:
    """Per round: history pseudo-label counts and tie counts (per seed)."""
    out = {}
    for r, (lo, _) in enumerate(ROUNDS, start=1):
        rows = []
        for s in DEV_SEEDS:
            h = next(x for x in data.runs[(cell.name, pair, s)].history2 if x["epoch"] == lo)
            z = np.load(cell.root / f"{dataset}_{pair}_s{s}" / cell.stage2 / "record_pseudo.npz")
            rows.append({**{k: h[k] for k in PSEUDO_STATS},
                         **tie_stats(z["scores"][r - 1], z["labels"][r - 1],
                                     int(z["top_k"][r - 1]))})
        out[r] = {k: [x[k] for x in rows] for k in rows[0]}
    return out


def descriptives(cells: Mapping[str, Arm], data, fits, cal, compared: Mapping[str, dict],
                 ok: list[str]) -> dict:
    """Descriptives 3 and 5-8 per setting for the baseline and the arms in ``ok`` (not
    failed), plus the P3 rule over settings."""
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    arms_all = (BASE, *ok)
    out = {}
    for d, p in SETTINGS:
        k = f"{d}/{p}"
        entry = {
            "stage2_curve": curves(data[d], arms_all, p),
            "differences_over_delta": differences(data[d], ok, p, thr[d]),
            "selection_rounds": {a: [selection_share(fits[d], a, p, lo, hi) for lo, hi in ROUNDS]
                                 for a in arms_all},
            "loss": {a: loss_summary(data[d], a, p) for a in ok},
            "pseudo_labels": {a: pseudo_summary(cells[a], data[d], d, p) for a in arms_all},
        }
        if p == XCL:
            entry["share_last_10pct"] = {a: selection_share(fits[d], a, p, 28, 30)
                                         for a in arms_all}
        else:
            entry["share_epochs_1_3"] = {a: selection_share(fits[d], a, p, 1, 3)
                                         for a in arms_all}
        keys = ("delta", "delta_over_threshold", "aux_opposite_2delta",
                "split_half_sign_agreement", "candidate_seed_sd_over_sd_d",
                "paired_delta_se_over_sigma", "per_seed_delta", "stage1_optimism", "selection")
        entry["compare"] = {a: {x: compared[a]["settings"][k][x] for x in keys} for a in ok}
        out[k] = entry
    diffs = {k: v["differences_over_delta"] for k, v in out.items()}
    return {"settings": out, "late_round_decline": {a: late_decline(diffs, a) for a in ok}}


def csv_rows(cells: Mapping[str, Arm], data, fits) -> list[dict]:
    rows = run_rows([cells[BASE], *(cells[a] for a in CANDIDATES)], data, fits)
    for row in rows:
        if row["failed"]:
            continue
        d, arm, p, s = row["dataset"], row["arm"], row["pair"], row["seed"]
        row |= {"s1_epoch": START_EPOCH, "s1_optimism": 0.0,       # no stage-1 selection
                "s1_selected": start_value(data[d], arm, p, s), "s1_crossfit_v_auc": ""}
        row |= score_shape(data[d].runs[(arm, p, s)].s2_best, data[d].labels)
    return rows


def proxy_summary(data: DatasetData, ok: Sequence[str]) -> dict:
    """UCF UCA proxy frame AP (descriptive only, decision 0004): per seed and the seed-mean
    candidate - baseline difference per variant."""
    ap = proxy_aps(data)
    out = {}
    for p in PAIRS:
        per = {a: [ap[(a, p, s)] for s in DEV_SEEDS] for a in (BASE, *ok)}
        variants = list(per[BASE][0])
        out[p] = {"per_seed": per, "minus_base": {
            a: {v: float(np.mean([x[v] for x in per[a]]) - np.mean([x[v] for x in per[BASE]]))
                for v in variants} for a in ok}}
    return out


# --- main -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E014 self-training in the reference pipeline")
    ap.add_argument("--e003", required=True, type=Path)
    ap.add_argument("--e013", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--no_proxy", action="store_true",
                    help="skip the UCA proxy AP (descriptive only; the verdict is unchanged)")
    args = ap.parse_args(argv)
    gate = gates(args.e003, args.e013, args.runs)
    write_json(E014_RESULTS / "e014_gate.json", {"code": code_version(), **gate})
    if gate["errors"]:
        raise SystemExit(f"runs or env.json missing or unreadable for {gate['errors']} "
                         "(PLAN.md); see results/e014_gate.json")
    if not all(gate["status"].values()):
        raise SystemExit("gate failed (init path: rerun that run; otherwise environment: "
                         "verdict suspended, PLAN.md); see results/e014_gate.json")
    cells = arms(args.e003, args.e013, args.runs)
    cal = calibration()
    data = {d: load_dataset(list(cells.values()), d) for d in DATASETS}
    fits = {d: fit_dataset(data[d], cal[d], d, baseline=cells[BASE],
                           reference=e013_reference(d)) for d in DATASETS}
    compared = {a: compare(cells[a], cells[BASE], data, fits, cal) for a in CANDIDATES}
    check_shared(compared)
    ok = [a for a in CANDIDATES if compared[a]["verdict"] != "failed"]
    for a in ok:
        relabel(compared[a], a, data)
    chosen = choose([compared[a] for a in CANDIDATES])
    summary = {
        "check": "E014 self-training ST3 vs STw (PLAN.md)", "code": code_version(),
        "calibration": cal, "gate": gate["status"],
        "cross_fit": {d: {"R": f.R, "splits_sha256": f.splits_sha256, "mc": f.mc,
                          "checks": f.checks} for d, f in fits.items()},
        "verdicts": {a: {k: compared[a][k] for k in ("verdict", "outcome_delta",
                                                    "outcome_delta_prime",
                                                    "mean_delta_over_threshold")
                         if k in compared[a]} for a in CANDIDATES},
        "deltas": {a: {k: st["delta"] for k, st in compared[a].get("settings", {}).items()}
                   for a in CANDIDATES},
        "deltas_over_threshold": {a: {k: st["delta_over_threshold"]["v_auc"]
                                      for k, st in compared[a].get("settings", {}).items()}
                                  for a in CANDIDATES},
        "chosen": chosen,
        "sd_validity": sd_validity(fits, chosen, cal) if chosen else None,
        "notes": ["stage 1 shared by all arms (checked): Delta' = Delta",
                  "stage-1 fields relabelled: start epoch 20, optimism 0 (no stage-1 selection)"],
    }
    write_json(E014_RESULTS / "e014_summary.json", summary)     # 1. verdict and choice
    for a in CANDIDATES:
        print(a, compared[a]["verdict"],
              {k: round(v, 2) for k, v in summary["deltas_over_threshold"][a].items()})
    print("chosen:", chosen)
    write_csv(E014_RESULTS / "e014_runs.csv", csv_rows(cells, data, fits))
    summary["descriptives"] = descriptives(cells, data, fits, cal, compared, ok)
    write_json(E014_RESULTS / "e014_summary.json", summary)     # 2. descriptives
    if not args.no_proxy:                                         # 3. after the verdict (0004)
        summary["uca_proxy_ap"] = {
            "note": "descriptive only (decision 0004): not used for the verdict, the choice, "
                    "the pipeline, the next candidate, reruns or claims about localisation; "
                    "low split-half reliability in E003 C5",
            "ucf": proxy_summary(data["ucf"], ok)}
        write_json(E014_RESULTS / "e014_summary.json", summary)


if __name__ == "__main__":
    main()
