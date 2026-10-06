"""E018: recalibrate the stage-D seed noise SD_d and threshold delta_d on the adopted M10 pipeline
(E017) from 6 seeds per (dataset, pair): dev seeds 0-2 (the E017 runs, cross-checked against
e017_runs.csv) and the calibration-only seeds 100-102 (E018 runs). Same metric, splits and MC
rule as E010 (PLAN.md). Video labels only, no test data, no training here.

    python -m wvad.diagnostics.e018 --e017 <e017 copy> --runs <e018 copy>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from wvad.data import load_split, split_file
from wvad.diagnostics.common import DEV_SEEDS, code_version, write_csv, write_json
from wvad.diagnostics.e010 import (
    DELTA_FACTOR,
    EFFECTS,
    check_selection,
    outcome_probabilities,
    pooled_sd,
    splits_sha256,
)
from wvad.diagnostics.e013 import LAST, init_parts, same_record_at
from wvad.diagnostics.e014 import GATE_ERRORS
from wvad.diagnostics.e015 import env_check
from wvad.diagnostics.runs import PAIRS, selection, val_record
from wvad.diagnostics.stage_d import DATASETS, MC_LIMIT, R_LARGE, calibration
from wvad.eval.crossfit import METRICS, crossfit, stratified_half_splits
from wvad.paths import REPO_ROOT

E018_RESULTS = REPO_ROOT / "experiments" / "E018_delta_recalibration" / "results"
E017_RESULTS = REPO_ROOT / "experiments" / "E017_offset_scale" / "results"
ARM = "m10"
CAL_SEEDS = (100, 101, 102)
SEEDS = (*DEV_SEEDS, *CAL_SEEDS)
SD_DF = len(PAIRS) * (len(SEEDS) - 1)                      # 2 pairs x 5 = 10
IGNORED_CONFIG = ("seed", "init", "out_dir")               # differ by construction
START_EPOCH = 20
REF_TOL = 1e-12

Key = tuple[str, int]                                      # (pair, seed)


def chain_dir(e017: Path, runs: Path, dataset: str, pair: str, seed: int) -> Path:
    return (e017 if seed in DEV_SEEDS else runs) / f"{dataset}_{pair}_s{seed}"


def _config(d: Path) -> dict:
    return torch.load(d / LAST, map_location="cpu", weights_only=True)["config"]


def same_config(d: Path, ref: Path, seed: int) -> bool:
    """Gate (c): every config field equals the E017 M10 run's (seed 0) except seed / init /
    out_dir, and the seed is the expected one."""
    cfg, want = _config(d), _config(ref)
    keys = (set(cfg) | set(want)) - set(IGNORED_CONFIG)
    return int(cfg["seed"]) == seed and all(cfg.get(k) == want.get(k) for k in keys)


# --- gates ----------------------------------------------------------------------------------

def gate_chain(e017: Path, runs: Path, dataset: str, pair: str, seed: int) -> dict:
    c = runs / f"{dataset}_{pair}_s{seed}"
    ref = e017 / f"{dataset}_{pair}_s0"
    s1, s2 = c / f"{ARM}_stage1", c / ARM
    return {"a": same_record_at(s2, 0, s1, START_EPOCH),
            "b": init_parts(s2) == (c.name, f"{ARM}_stage1", LAST),
            "c": same_config(s1, ref / f"{ARM}_stage1", seed) and same_config(s2, ref / ARM, seed)}


def gates(e017: Path, runs: Path) -> dict:
    rows = []
    for d in DATASETS:
        for p in PAIRS:
            for s in CAL_SEEDS:
                try:
                    row = {"chain": f"{d}_{p}_s{s}", **gate_chain(e017, runs, d, p, s)}
                except GATE_ERRORS as exc:
                    row = {"chain": f"{d}_{p}_s{s}", "error": repr(exc)}
                rows.append(row)
    status = {k: all(r.get(k, False) for r in rows if "error" not in r) for k in "abc"}
    env = env_check(runs, E017_RESULTS / "e017_identity.json")
    status["d"] = env["ok"]
    return {"chains": rows, "status": status, "environment": env,
            "errors": [r["chain"] for r in rows if "error" in r]}


# --- loading and cross-fit ------------------------------------------------------------------

def load(e017: Path, runs: Path, dataset: str) -> tuple[np.ndarray, dict, dict]:
    """Labels, stage-2 scores of epochs 1-30 per (pair, seed), and the checked selection V-AUC
    (gate (e): the record reproduces the run's own selection)."""
    split = load_split(split_file(dataset))
    val_idx = np.asarray(split.val_indices)
    labels = np.array([split.labels[i] for i in val_idx])
    scores, selected = {}, {}
    for p in PAIRS:
        for s in SEEDS:
            d = chain_dir(e017, runs, dataset, p, s) / ARM
            rec, sel = val_record(d), selection(d)
            if sel["val_split_sha"] != split.sha256 or not np.array_equal(rec.val_indices,
                                                                           val_idx):
                raise ValueError(f"{d}: split or validation order differs from the split file")
            scores[(p, s)] = rec.scores[rec.epochs >= 1]
            selected[(p, s)] = check_selection(d, scores[(p, s)], labels, sel)
    return labels, scores, selected


def fit(labels: np.ndarray, scores: Mapping[Key, np.ndarray], n_splits: int):
    splits = stratified_half_splits(labels, n_splits, seed=0)
    return {k: crossfit(v, labels, splits) for k, v in scores.items()}, splits_sha256(splits)


def e017_reference(dataset: str, path: Path | None = None) -> dict[tuple, dict[str, float]]:
    with open(path or E017_RESULTS / "e017_runs.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["arm"] == ARM and r["dataset"] == dataset]
    return {(r["pair"], int(r["seed"])): {m: float(r[m]) for m in METRICS} for r in rows}


def check_dev_reference(fits: Mapping, reference: Mapping) -> bool:
    """Gate (f): the reused dev-seed runs reproduce e017_runs.csv (E010 R, all metrics)."""
    return all(abs(fits[(p, s)].mean(m) - reference[(p, s)][m]) <= REF_TOL
               for p in PAIRS for s in DEV_SEEDS for m in METRICS)


def noise(fits: Mapping, seeds) -> dict[str, float]:
    return {m: pooled_sd([[fits[(p, s)].mean(m) for s in seeds] for p in PAIRS])
            for m in METRICS}


def analyze_dataset(dataset: str, labels, scores, cal: Mapping, reference: Mapping) -> dict:
    fits, sha = fit(labels, scores, cal["R"])
    if sha != cal["splits_sha256"]:
        raise ValueError(f"{dataset}: splits sha {sha} != E010 {cal['splits_sha256']}")
    dev_ok = check_dev_reference(fits, reference)
    sd = noise(fits, SEEDS)
    worst = max(cf.mc_se("v_auc") for cf in fits.values()) / sd["v_auc"]
    mc = {"first_R": cal["R"], "first_mc_se_over_sd": worst, "escalated": False}
    if worst > MC_LIMIT and cal["R"] < R_LARGE:
        fits, sha = fit(labels, scores, R_LARGE)
        sd = noise(fits, SEEDS)
        mc |= {"escalated": True, "R": R_LARGE, "splits_sha256": sha,
               "mc_se_over_sd": max(cf.mc_se("v_auc") for cf in fits.values()) / sd["v_auc"]}
    delta = {m: DELTA_FACTOR * v for m, v in sd.items()}
    dev, calib = noise(fits, DEV_SEEDS), noise(fits, CAL_SEEDS)
    group = {}
    for p in PAIRS:
        d_mean = np.mean([fits[(p, s)].mean("v_auc") for s in DEV_SEEDS])
        c_mean = np.mean([fits[(p, s)].mean("v_auc") for s in CAL_SEEDS])
        vals = [fits[(p, s)].mean("v_auc") for s in SEEDS]
        group[p] = {"cal_minus_dev": float(c_mean - d_mean),
                    "cal_minus_dev_over_delta": float((c_mean - d_mean) / delta["v_auc"]),
                    "mean_6": float(np.mean(vals)), "sd_6": float(np.std(vals, ddof=1))}
    rows = [{"dataset": dataset, "pair": p, "seed": s, "group": "dev" if s in DEV_SEEDS else
             "calibration", "n_splits": len(cf.epochs), **{m: cf.mean(m) for m in METRICS},
             **{f"mc_se_{m}": cf.mc_se(m) for m in METRICS}, "tie_rate": cf.tie_rate}
            for (p, s), cf in fits.items()]
    return {"rows": rows, "dev_reference_ok": dev_ok, "summary": {
        "n_splits": len(next(iter(fits.values())).epochs), "splits_sha256": sha, "mc": mc,
        "sd": sd, "delta": delta, "df": SD_DF,
        "sd_dev_only": dev, "sd_calibration_only": calib,
        "sd_over_e010": {m: sd[m] / cal["sd"][m] for m in METRICS},
        "seed_groups": group,
        "max_mc_se_over_sd": max(cf.mc_se("v_auc") for cf in fits.values()) / sd["v_auc"],
        "tie_rate_max": float(max(cf.tie_rate for cf in fits.values()))}}


def e017_in_new_delta(datasets: Mapping[str, dict]) -> dict:
    """Descriptive 5: E017 M10's Delta (vs L20) in units of the new delta (verdict unchanged)."""
    s = json.loads((E017_RESULTS / "e017_summary.json").read_text(encoding="utf-8"))
    return {k: v["v_auc"] / datasets[k.split("/")[0]]["delta"]["v_auc"]
            for k, v in s["deltas"][ARM].items()}


# --- main -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E018 delta recalibration on the M10 pipeline")
    ap.add_argument("--e017", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    gate = gates(args.e017, args.runs)
    write_json(E018_RESULTS / "e018_gate.json", {"code": code_version(), **gate})
    if gate["errors"]:
        raise SystemExit(f"runs missing or unreadable for {gate['errors']}; see e018_gate.json")
    if not all(gate["status"].values()):
        raise SystemExit("gate failed (config/init: rerun; environment: report; PLAN.md); "
                         "see results/e018_gate.json")
    cal = calibration()
    loaded = {d: load(args.e017, args.runs, d) for d in DATASETS}      # gate (e) checks
    out = {d: analyze_dataset(d, *loaded[d][:2], cal[d], e017_reference(d)) for d in DATASETS}
    gate["status"] |= {"e": True, "f": all(o["dev_reference_ok"] for o in out.values())}
    write_json(E018_RESULTS / "e018_gate.json", {"code": code_version(), **gate})
    if not gate["status"]["f"]:
        raise SystemExit("gate (f) failed: dev-seed M10 runs differ from e017_runs.csv")
    datasets = {d: o["summary"] for d, o in out.items()}
    power = {f"{e:g}": {"normal": outcome_probabilities(e),
                        f"t{SD_DF}": outcome_probabilities(e, SD_DF)} for e in EFFECTS}
    write_csv(E018_RESULTS / "e018_runs.csv", [r for o in out.values() for r in o["rows"]])
    write_json(E018_RESULTS / "e018_summary.json", {
        "check": "E018 delta recalibration on the M10 pipeline (PLAN.md)",
        "code": code_version(), "gate": gate["status"],
        "primary_metric": "cross-fitted validation video AUC (v_auc), stage-2 epochs 1-30",
        "seeds": {"dev": list(DEV_SEEDS), "calibration": list(CAL_SEEDS)},
        "datasets": datasets,
        "power": power,
        "e017_m10_delta_over_new_delta": e017_in_new_delta(datasets),
        "use": "delta_d (and delta_d^m) for stage-D experiments whose baseline is the M10 "
               "pipeline, from E019; E017's verdict is not revisited; calibration seeds "
               "100-102 are excluded from stage-D comparisons and stage-F confirmation"})
    for d, s in datasets.items():
        print(f"{d}: R={s['n_splits']} SD={s['sd']['v_auc']:.6f} delta={s['delta']['v_auc']:.6f}"
              f" ({s['sd_over_e010']['v_auc']:.2f} x E010) MC/SD={s['max_mc_se_over_sd']:.3f}")


if __name__ == "__main__":
    main()
