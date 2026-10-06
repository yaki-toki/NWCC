"""E018: recalibrate the stage-D seed noise SD_d and threshold delta_d on the adopted M10 pipeline
(E017) from 6 seeds per (dataset, pair): dev seeds 0-2 (the E017 runs, cross-checked against
e017_runs.csv) and the calibration-only seeds 100-102 (E018 runs). Same metric, splits and MC
rule as E010 (PLAN.md). Video labels only, no test data, no training here.

Calibration seeds never reach a stage-D decision: they are read only here, and stage_d keeps its
development-seed check (common.check_dev_seed). A numerically failed run is dropped and the SD is
the df-weighted pooled SD of the remaining runs (PLAN.md failure handling).

    python -m wvad.diagnostics.e018 --e017 <e017 copy> --runs <e018 copy>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from wvad.data import load_split, split_file
from wvad.diagnostics.common import DEV_SEEDS, code_version, write_csv, write_json
from wvad.diagnostics.e010 import (
    DELTA_FACTOR,
    EFFECTS,
    check_selection,
    outcome_probabilities,
    splits_sha256,
)
from wvad.diagnostics.e013 import LAST, init_parts, same_record_at
from wvad.diagnostics.e014 import GATE_ERRORS
from wvad.diagnostics.e015 import differ, env_check, epoch_scores
from wvad.diagnostics.e017 import r_end
from wvad.diagnostics.runs import PAIRS, selection
from wvad.diagnostics.stage_d import DATASETS, MC_LIMIT, R_LARGE, calibration
from wvad.eval.crossfit import METRICS, crossfit, stratified_half_splits
from wvad.paths import REPO_ROOT

E018_RESULTS = REPO_ROOT / "experiments" / "E018_delta_recalibration" / "results"
E017_RESULTS = REPO_ROOT / "experiments" / "E017_offset_scale" / "results"
ARM = "m10"
CAL_SEEDS = (100, 101, 102)
CONFIRMATION_SEEDS = (42, 123, 777, 1000, 2025, 7, 13, 99, 555, 31337)   # stage F (ROADMAP)
assert not set(CAL_SEEDS) & (set(DEV_SEEDS) | set(CONFIRMATION_SEEDS))
SEEDS = (*DEV_SEEDS, *CAL_SEEDS)
SD_DF = len(PAIRS) * (len(SEEDS) - 1)                      # 2 pairs x 5 = 10 without failures
SENSITIVITY_DF = 5
IGNORED_CONFIG = ("seed", "init", "out_dir")               # differ by construction
START_EPOCH = 20
REF_TOL = 1e-12

Key = tuple[str, int]                                      # (pair, seed)


def chain_dir(e017: Path, runs: Path, dataset: str, pair: str, seed: int) -> Path:
    if seed in DEV_SEEDS:
        return e017 / f"{dataset}_{pair}_s{seed}"
    if seed in CAL_SEEDS:
        return runs / f"{dataset}_{pair}_s{seed}"
    raise ValueError(f"seed {seed} is neither a development nor a calibration seed")


def _config(d: Path) -> dict:
    return torch.load(d / LAST, map_location="cpu", weights_only=True)["config"]


def same_config(d: Path, ref: Path, seed: int) -> bool:
    """Gate (c): the whole config equals the E017 M10 run's (seed 0) except seed / init /
    out_dir, and the seed is the expected one."""
    cfg, want = _config(d), _config(ref)
    keys = (set(cfg) | set(want)) - set(IGNORED_CONFIG)
    return int(cfg["seed"]) == seed and all(cfg.get(k) == want.get(k) for k in keys)


# --- gates ----------------------------------------------------------------------------------

def gate_chain(e017: Path, runs: Path, dataset: str, pair: str, seed: int) -> dict:
    c = runs / f"{dataset}_{pair}_s{seed}"
    ref = e017 / f"{dataset}_{pair}_s0"
    s1, s2 = c / f"{ARM}_stage1", c / ARM
    first = epoch_scores(s1, 1)
    others = [chain_dir(e017, runs, dataset, pair, o) / f"{ARM}_stage1"
              for o in SEEDS if o != seed]
    return {"a": same_record_at(s2, 0, s1, START_EPOCH),
            "b": init_parts(s2) == (c.name, f"{ARM}_stage1", LAST),
            "c": same_config(s1, ref / f"{ARM}_stage1", seed) and same_config(s2, ref / ARM, seed),
            "h": all(differ(first, epoch_scores(o, 1)) for o in others)}


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
    status = {k: all(r.get(k, False) for r in rows if "error" not in r) for k in "abch"}
    env = env_check(runs, E017_RESULTS / "e017_identity.json")
    status["d"] = env["ok"]
    return {"chains": rows, "status": status, "environment": env,
            "errors": [r["chain"] for r in rows if "error" in r]}


# --- loading and cross-fit ------------------------------------------------------------------

def _history(d: Path) -> list[dict]:
    return json.loads((d / "history.json").read_text(encoding="utf-8"))


def numerically_failed(d: Path, scores: np.ndarray) -> bool:
    hist = [*_history(d.parent / f"{ARM}_stage1"), *_history(d)]
    losses = [v for h in hist for k, v in h.items() if k.startswith("loss_")]
    return not (np.isfinite(scores).all() and all(math.isfinite(v) for v in losses))


def load(e017: Path, runs: Path, dataset: str) -> tuple[np.ndarray, dict, dict, list]:
    """Labels, stage-2 scores of epochs 1-30 per (pair, seed), the checked selection V-AUC
    (gate (e)) and the numerically failed runs (dropped)."""
    split = load_split(split_file(dataset))
    val_idx = np.asarray(split.val_indices)
    labels = np.array([split.labels[i] for i in val_idx])
    scores, selected, failed = {}, {}, []
    for p in PAIRS:
        for s in SEEDS:
            d = chain_dir(e017, runs, dataset, p, s) / ARM
            z, sel = np.load(d / "record_val_scores.npz"), selection(d)
            if not np.array_equal(z["epochs"], np.arange(0, 31)):
                raise ValueError(f"{d}: stage-2 record should hold epochs 0-30")
            if sel["val_split_sha"] != split.sha256 or not np.array_equal(z["val_indices"],
                                                                           val_idx):
                raise ValueError(f"{d}: split or validation order differs from the split file")
            sc = z["scores"][z["epochs"] >= 1]
            if numerically_failed(d, sc):
                failed.append(f"{dataset}_{p}_s{s}")
                continue
            scores[(p, s)] = sc
            selected[(p, s)] = check_selection(d, sc, labels, sel)
    return labels, scores, selected, failed


def fit(labels: np.ndarray, scores: Mapping[Key, np.ndarray], n_splits: int):
    splits = stratified_half_splits(labels, n_splits, seed=0)
    return {k: crossfit(v, labels, splits) for k, v in scores.items()}, splits_sha256(splits)


def e017_reference(dataset: str, path: Path | None = None) -> dict[tuple, dict[str, float]]:
    with open(path or E017_RESULTS / "e017_runs.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["arm"] == ARM and r["dataset"] == dataset]
    return {(r["pair"], int(r["seed"])): {m: float(r[m]) for m in METRICS} for r in rows}


def check_dev_reference(fits: Mapping, reference: Mapping) -> bool:
    """Gate (f): the reused dev-seed runs reproduce e017_runs.csv (E010 R, all metrics); a dev
    run that failed numerically cannot reproduce it."""
    return all((p, s) in fits and abs(fits[(p, s)].mean(m) - reference[(p, s)][m]) <= REF_TOL
               for p in PAIRS for s in DEV_SEEDS for m in METRICS)


def pooled(groups: Sequence[Sequence[float]]) -> tuple[float, int]:
    """df-weighted pooled SD and its df: sum of squared deviations from each group's mean over
    sum (n_g - 1); groups with fewer than 2 values are dropped."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    df = sum(len(g) - 1 for g in groups)
    ss = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    return math.sqrt(ss / df), df


def noise(fits: Mapping, seeds) -> dict[str, float]:
    return {m: pooled([[fits[(p, s)].mean(m) for s in seeds if (p, s) in fits]
                       for p in PAIRS])[0] for m in METRICS}


def reliability(fits: Mapping, delta: float) -> dict:
    """Descriptive 7: cross-pair correlation of seed deviations, between-pair variance ratio
    with F bounds, leave-one-seed-out SD range, within-group pooled SD (dev / calibration)."""
    v = {p: {s: fits[(p, s)].mean("v_auc") for s in SEEDS if (p, s) in fits} for p in PAIRS}
    common = [s for s in SEEDS if all(s in v[p] for p in PAIRS)]
    dev = {p: np.array([v[p][s] for s in common]) - np.mean([v[p][s] for s in common])
           for p in PAIRS}
    a, b = (dev[p] for p in PAIRS)
    var = {p: float(np.var(list(v[p].values()), ddof=1)) for p in PAIRS}
    n = [len(v[p]) for p in PAIRS]
    loo = [pooled([[x for t, x in v[p].items() if t != s] for p in PAIRS])[0] for s in SEEDS]
    within = pooled([[v[p][s] for s in grp if s in v[p]] for p in PAIRS
                     for grp in (DEV_SEEDS, CAL_SEEDS)])
    return {"cross_pair_seed_correlation": float(np.corrcoef(a, b)[0, 1]),
            "variance_ratio": var[PAIRS[0]] / var[PAIRS[1]],
            "variance_ratio_f_bounds": [float(stats.f.ppf(0.025, n[0] - 1, n[1] - 1)),
                                        float(stats.f.ppf(0.975, n[0] - 1, n[1] - 1))],
            "leave_one_seed_out_sd": [float(min(loo)), float(max(loo))],
            "within_group_pooled_sd": within[0], "within_group_df": within[1],
            "within_group_pooled_sd_over_sd_6": within[0] / pooled(
                [list(v[p].values()) for p in PAIRS])[0], "delta": delta}


def analyze_dataset(dataset: str, labels, scores, cal: Mapping, reference: Mapping) -> dict:
    fits, sha = fit(labels, scores, cal["R"])
    sha_ok = sha == cal["splits_sha256"]
    dev_ok = check_dev_reference(fits, reference)
    sd = noise(fits, SEEDS)
    worst = max(cf.mc_se("v_auc") for cf in fits.values()) / sd["v_auc"]
    mc = {"first_R": cal["R"], "first_mc_se_over_sd": worst, "escalated": False}
    if worst > MC_LIMIT and cal["R"] < R_LARGE:
        fits, sha = fit(labels, scores, R_LARGE)
        sd = noise(fits, SEEDS)
        mc |= {"escalated": True, "R": R_LARGE, "splits_sha256": sha,
               "mc_se_over_sd": max(cf.mc_se("v_auc") for cf in fits.values()) / sd["v_auc"]}
    counts = [sum((p, s) in fits for s in SEEDS) for p in PAIRS]
    df = sum(n - 1 for n in counts if n >= 2)
    delta = {m: DELTA_FACTOR * v for m, v in sd.items()}
    dev, calib = noise(fits, DEV_SEEDS), noise(fits, CAL_SEEDS)
    group = {}
    for p in PAIRS:
        d_vals = [fits[(p, s)].mean("v_auc") for s in DEV_SEEDS if (p, s) in fits]
        c_vals = [fits[(p, s)].mean("v_auc") for s in CAL_SEEDS if (p, s) in fits]
        vals = d_vals + c_vals
        diff = float(np.mean(c_vals) - np.mean(d_vals))
        group[p] = {"cal_minus_dev": diff, "cal_minus_dev_over_delta": diff / delta["v_auc"],
                    "mean_6": float(np.mean(vals)), "sd_6": float(np.std(vals, ddof=1))}
    rows = [{"dataset": dataset, "pair": p, "seed": s, "group": "dev" if s in DEV_SEEDS else
             "calibration", "n_splits": len(cf.epochs), **{m: cf.mean(m) for m in METRICS},
             **{f"mc_se_{m}": cf.mc_se(m) for m in METRICS}, "tie_rate": cf.tie_rate}
            for (p, s), cf in fits.items()]
    return {"rows": rows, "dev_reference_ok": dev_ok, "splits_ok": sha_ok, "summary": {
        "n_splits": len(next(iter(fits.values())).epochs), "splits_sha256": sha, "mc": mc,
        "sd": sd, "delta": delta, "df": df,
        "sd_dev_only": dev, "sd_calibration_only": calib,
        "sd_over_e010": {m: sd[m] / cal["sd"][m] for m in METRICS},
        "seed_groups": group,
        "reliability": reliability(fits, delta["v_auc"]),
        "max_mc_se_over_sd": max(cf.mc_se("v_auc") for cf in fits.values()) / sd["v_auc"],
        "tie_rate_max": float(max(cf.tie_rate for cf in fits.values()))}}


def calibration_gamma(runs: Path, dataset: str) -> dict:
    """Descriptive 8: stage-2 end gamma and r_end of the calibration seeds."""
    out = {}
    for p in PAIRS:
        rows = []
        for s in CAL_SEEDS:
            g2 = [h["offset_gamma"] for h in _history(runs / f"{dataset}_{p}_s{s}" / ARM)]
            rows.append({"seed": s, "gamma_end": g2[-1], "r_end": r_end(g2, dataset, ARM)})
        out[p] = rows
    return out


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
        raise SystemExit("gate failed (PLAN.md failure handling); see results/e018_gate.json")
    cal = calibration()
    loaded = {d: load(args.e017, args.runs, d) for d in DATASETS}      # gate (e) checks
    out = {d: analyze_dataset(d, *loaded[d][:2], cal[d], e017_reference(d)) for d in DATASETS}
    gate["status"] |= {"e": True, "f": all(o["dev_reference_ok"] for o in out.values()),
                       "g": all(o["splits_ok"] for o in out.values())}
    gate["numerically_failed"] = [f for d in DATASETS for f in loaded[d][3]]
    write_json(E018_RESULTS / "e018_gate.json", {"code": code_version(), **gate})
    if not (gate["status"]["f"] and gate["status"]["g"]):
        raise SystemExit("gate (f) or (g) failed: dev-seed M10 runs differ from e017_runs.csv or "
                         "the splits differ from E010; see results/e018_gate.json")
    datasets = {d: o["summary"] for d, o in out.items()}
    power = {f"{e:g}": {"normal": outcome_probabilities(e),
                        f"t{SD_DF}": outcome_probabilities(e, SD_DF),
                        f"t{SENSITIVITY_DF}": outcome_probabilities(e, SENSITIVITY_DF)}
             for e in EFFECTS}
    write_csv(E018_RESULTS / "e018_runs.csv", [r for o in out.values() for r in o["rows"]])
    write_json(E018_RESULTS / "e018_summary.json", {
        "check": "E018 delta recalibration on the M10 pipeline (PLAN.md)",
        "code": code_version(), "gate": gate["status"],
        "numerically_failed": gate["numerically_failed"],
        "primary_metric": "cross-fitted validation video AUC (v_auc), stage-2 epochs 1-30",
        "seeds": {"dev": list(DEV_SEEDS), "calibration": list(CAL_SEEDS)},
        "datasets": datasets,
        "power": power,
        "calibration_gamma": {d: calibration_gamma(args.runs, d) for d in DATASETS},
        "e017_m10_delta_over_new_delta": e017_in_new_delta(datasets),
        "use": "exact delta_d and delta_d^m for stage-D decisions of the M10 pipeline lineage "
               "from E019 (validity rule after each adoption with this SD as reference; R and "
               "splits from here); not for stage F or L20-baseline comparisons; E017's verdict "
               "is not revisited; calibration seeds 100-102 are never used in a decision"})
    for d, s in datasets.items():
        print(f"{d}: R={s['n_splits']} df={s['df']} SD={s['sd']['v_auc']:.6f} "
              f"delta={s['delta']['v_auc']:.6f} ({s['sd_over_e010']['v_auc']:.2f} x E010) "
              f"MC/SD={s['max_mc_se_over_sd']:.3f}")


if __name__ == "__main__":
    main()
