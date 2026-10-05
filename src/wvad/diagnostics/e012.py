"""E012 (stage D): training-budget check of the base pipeline, 2 x 2 nested design.

Cells: base = (stage 1 20, stage 2 30) = the E003 runs; S1 = (40, 30); S2 = (20, 60, started
from the E003 stage-1 checkpoint); S12 = (40, 60). S1 and S12 share one 40-epoch stage 1.
Determinism gates (a)-(e) are checked first (PLAN.md); the verdict uses the budget rule
(condition A on Delta, condition B on Delta and Delta' in the settings with convergence
evidence), Delta' with per-chain stage-1 optimism set to 0 for chains that share the
baseline's stage-1 checkpoint. Video labels only; no UCA proxy, no test data.

    python -m wvad.diagnostics.e012 --base_runs <e003 copy> --runs <e012 copy>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from wvad.diagnostics.common import DEV_SEEDS, code_version, write_csv, write_json
from wvad.diagnostics.e010 import pooled_sd
from wvad.diagnostics.e011 import score_shape
from wvad.diagnostics.runs import PAIRS
from wvad.diagnostics.stage_d import (
    DATASETS,
    SETTINGS,
    Arm,
    DatasetData,
    DatasetFits,
    calibration,
    compare,
    fit_dataset,
    load_dataset,
    run_rows,
)
from wvad.eval.crossfit import crossfit, rank_auc, stratified_half_splits, video_auc
from wvad.paths import REPO_ROOT

E012_RESULTS = REPO_ROOT / "experiments" / "E012_training_budget" / "results"
ZERO = {"dense": 0.0, "pseudo": 0.0, "neg_pseudo": 0.0, "normal_seg": 0.0}
CONVERGENCE = (("ucf", "xcb_xcl"), ("xdv", "xcb_xcl"))
HIST1 = ("epoch", "loss_total", "loss_mil", "loss_pseudo", "val_video_auc")
HIST2 = ("epoch", "loss_total", "loss_mil", "val_video_auc")
PSEUDO_STATS = ("positive_videos", "positive_segments", "mined_negative_segments",
                "normal_videos", "normal_negative_segments")
SELECTION_KEYS = ("best_val_video_auc", "best_epoch", "val_split_sha")
SD_RATIO_LIMIT = 1.5
COMMON = {"weight_decay": 1e-4, "batch_size": 32}


def _expect2(epochs_per_iter: int) -> dict:
    return {**COMMON, "lr": 3e-5, "iterations": 3, "epochs_per_iter": epochs_per_iter,
            "curriculum": True, "top_k_neg": 0, "pseudo_neg_k": 4}


def arms(base_root: Path, runs_root: Path) -> dict[str, Arm]:
    e1 = {**COMMON, "lr": 1e-4}
    return {
        "base": Arm("base", base_root, "stage1", "base", ZERO, reuse_e010=True,
                    expect1=e1, expect2=_expect2(10)),
        "s1": Arm("s1", runs_root, "s40_stage1", "s1", ZERO, stage1_epochs=40,
                  stage2_epochs=30, expect1=e1, expect2=_expect2(10)),
        "s2": Arm("s2", runs_root, "stage1", "s2", ZERO, stage1_root=base_root,
                  stage2_epochs=60, expect1=e1, expect2=_expect2(20)),
        "s12": Arm("s12", runs_root, "s40_stage1", "s12", ZERO, stage1_epochs=40,
                   stage2_epochs=60, expect1=e1, expect2=_expect2(20)),
    }


# --- determinism gates (a)-(e) --------------------------------------------------------------

def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_record(a: Path, b: Path, epochs: Sequence[int]) -> bool:
    """Recorded validation scores of ``epochs`` equal (values and dtype)."""
    za, zb = np.load(a / "record_val_scores.npz"), np.load(b / "record_val_scores.npz")
    ia = [int(np.flatnonzero(za["epochs"] == e)[0]) for e in epochs]
    ib = [int(np.flatnonzero(zb["epochs"] == e)[0]) for e in epochs]
    sa, sb = za["scores"][ia], zb["scores"][ib]
    return sa.dtype == sb.dtype and bool(np.array_equal(sa, sb))


def same_history(a: Path, b: Path, epochs: Sequence[int], fields: Sequence[str]) -> bool:
    ha = {h["epoch"]: h for h in _json(a / "history.json")}
    hb = {h["epoch"]: h for h in _json(b / "history.json")}
    return all(e in ha and e in hb and all(ha[e].get(f) == hb[e].get(f) for f in fields)
               for e in epochs)


def same_state(a: Path, b: Path) -> bool:
    x = torch.load(a, map_location="cpu", weights_only=True)["model_state_dict"]
    y = torch.load(b, map_location="cpu", weights_only=True)["model_state_dict"]
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def gate_chain(e3: Path, e12: Path) -> dict:
    """Gates (a)-(e) of one chain; (b) and (d) only when the 40-epoch stage 1 selects an
    epoch <= 20 (then its checkpoint should be the E003 one)."""
    s40, b1 = e12 / "s40_stage1", e3 / "stage1"
    row = {"a": same_record(s40, b1, range(1, 21))
           and same_history(s40, b1, range(1, 21), HIST1)}
    sel40, sel3 = _json(s40 / "selection.json"), _json(b1 / "selection.json")
    row["s40_best_epoch"] = int(sel40["best_epoch"])
    if row["s40_best_epoch"] <= 20:
        row["b"] = (all(sel40[k] == sel3[k] for k in SELECTION_KEYS)
                    and same_state(s40 / "ckpt_best_val.pt", b1 / "ckpt_best_val.pt"))
        row["d"] = (same_record(e12 / "s1", e3 / "base", range(0, 31))
                    and same_history(e12 / "s1", e3 / "base", range(1, 31),
                                     HIST1 + PSEUDO_STATS)
                    and _json(e12 / "s1" / "selection.json")
                    == _json(e3 / "base" / "selection.json"))
    row["c"] = (same_record(e12 / "s12", e12 / "s1", range(0, 31))
                and same_history(e12 / "s12", e12 / "s1", range(1, 31), HIST2))
    row["e"] = (same_record(e12 / "s2", e3 / "base", range(0, 31))
                and same_history(e12 / "s2", e3 / "base", range(1, 31), HIST2))
    return row


def gates(base_root: Path, runs_root: Path) -> tuple[dict, dict[tuple, bool]]:
    """Gate results per chain and the stage-1 sharing map of the 40-epoch stage 1."""
    rows, shared = [], {}
    for d in DATASETS:
        for p in PAIRS:
            for s in DEV_SEEDS:
                chain = f"{d}_{p}_s{s}"
                row = {"chain": chain, **gate_chain(base_root / chain, runs_root / chain)}
                shared[(d, p, s)] = bool(row.get("b", False))
                rows.append(row)
    status = {k: all(r[k] for r in rows if k in r) for k in "abcde"}
    return {"chains": rows, "status": status,
            "shared_stage1_chains": int(sum(shared.values()))}, shared


# --- budget rule ----------------------------------------------------------------------------

def budget_rule(result: Mapping, thr: Mapping[str, float]) -> dict:
    """Condition A (Delta) and condition B (Delta and Delta' in the convergence settings)."""
    if result.get("verdict") == "failed":
        return {"adopt": False, "failed": True}
    st = result["settings"]

    def t(k: str) -> float:
        return thr[k.split("/")[0]]

    delta = {k: v["delta"]["v_auc"] for k, v in st.items()}
    prime = {k: v["delta_prime_v_auc"] for k, v in st.items()}
    worse = [k for k in delta if delta[k] <= -t(k)]
    conv = [f"{d}/{p}" for d, p in CONVERGENCE]
    gain = [k for k in conv if delta[k] >= t(k) and prime[k] >= t(k)]
    gain_delta = [k for k in conv if delta[k] >= t(k)]
    return {"adopt": not worse and bool(gain), "condition_A": not worse,
            "condition_B": bool(gain), "worse_settings": worse, "gain_settings": gain,
            "condition_B_delta_only": bool(gain_delta), "gain_settings_delta_only": gain_delta,
            "mean_delta_over_threshold": result["mean_delta_over_threshold"]}


def choose_cell(rules: Mapping[str, dict], s12_vs_small: Mapping[str, dict]) -> str | None:
    """S12 only if adopted and adopted against every adopted smaller cell; otherwise the
    adopted S1/S2 with the larger mean Delta/delta (ties to S1)."""
    small = [c for c in ("s1", "s2") if rules[c]["adopt"]]
    if rules["s12"]["adopt"] and all(s12_vs_small[c]["adopt"] for c in small):
        return "s12"
    if not small:
        return None
    return max(small, key=lambda c: (rules[c]["mean_delta_over_threshold"], c == "s1"))


def sd_validity(fits: Mapping[str, DatasetFits], cell: str, cal: Mapping) -> dict:
    out = {}
    for d in DATASETS:
        sd = pooled_sd([[fits[d].stage2[(cell, p, s)].mean("v_auc") for s in DEV_SEEDS]
                        for p in PAIRS])
        out[d] = {"pooled_sd": sd, "ratio_to_sd_d": sd / cal[d]["sd"]["v_auc"]}
    out["recalibrate"] = any(out[d]["ratio_to_sd_d"] > SD_RATIO_LIMIT for d in DATASETS)
    return out


# --- descriptives ---------------------------------------------------------------------------

def window_start(budget: int) -> int:
    """First epoch of the last 10% of a budget: 30 -> 28, 60 -> 55, 20 -> 19, 40 -> 37."""
    return budget - budget // 10 + 1


def curves(data: DatasetData, cell: Arm, pair: str) -> dict:
    """Seed-mean full-validation V-AUC at every 10th epoch of both stages."""
    def at(stage: str, budget: int) -> dict:
        vals = {}
        for e in range(10, budget + 1, 10):
            v = [rank_auc(getattr(data.runs[(cell.name, pair, s)], stage)[e - 1]
                          .max(axis=1)[None], data.labels)[0] for s in DEV_SEEDS]
            vals[e] = float(np.mean(v))
        return vals
    return {"stage1": at("s1", cell.stage1_epochs), "stage2": at("s2", cell.stage2_epochs)}


def truncated_crossfit(data: DatasetData, cell: Arm, pair: str, R: int) -> dict:
    """Seed-mean cross-fitted V-AUC with stage-2 candidates cut to epochs 1..k."""
    splits = stratified_half_splits(data.labels, R, seed=0)
    return {k: float(np.mean([crossfit(data.runs[(cell.name, pair, s)].s2[:k], data.labels,
                                       splits, {"v_auc": video_auc}).mean("v_auc")
                              for s in DEV_SEEDS]))
            for k in range(10, cell.stage2_epochs + 1, 10)}


def convergence(fits: DatasetFits, cell: Arm, pair: str) -> dict:
    """Share of cross-fit selections (splits x directions x seeds) in the last 10% of the
    budget (CrossFit.epochs is a 0-based index into the candidate epochs)."""
    def share(table: dict, budget: int) -> float:
        sel = np.concatenate([table[(cell.name, pair, s)].epochs.ravel() + 1
                              for s in DEV_SEEDS])
        return float(np.mean(sel >= window_start(budget)))
    s1, s2 = share(fits.stage1, cell.stage1_epochs), share(fits.stage2, cell.stage2_epochs)
    return {"stage1_share": s1, "stage2_share": s2, "still_converging": s1 >= 0.5 or s2 >= 0.5}


def split_half_excluding_identical(fits: DatasetFits, cand: str, base: str, pair: str) -> dict:
    """Split-half sign agreement of cand - base without chains identical to the baseline."""
    same, agree = [], []
    for s in DEV_SEEDS:
        a = fits.stage2[(cand, pair, s)].directions["v_auc"]
        b = fits.stage2[(base, pair, s)].directions["v_auc"]
        if np.array_equal(a, b):
            same.append(s)
            continue
        sign = np.sign(a - b)
        agree.append(float(np.mean(sign[:, 0] == sign[:, 1])))
    return {"identical_seeds": same, "agreement": float(np.mean(agree)) if agree else None}


def descriptives(cells: Mapping[str, Arm], data: Mapping[str, DatasetData],
                 fits: Mapping[str, DatasetFits], rows: list[dict]) -> dict:
    out = {}
    for d, p in SETTINGS:
        for name, cell in cells.items():
            if any(data[d].runs[(name, p, s)].failed for s in DEV_SEEDS):
                continue
            entry = {"curves": curves(data[d], cell, p),
                     "truncated_crossfit": truncated_crossfit(data[d], cell, p, fits[d].R),
                     "convergence": convergence(fits[d], cell, p)}
            if name != "base":
                entry["vs_base"] = split_half_excluding_identical(fits[d], name, "base", p)
            out[f"{d}/{p}/{name}"] = entry
    for row in rows:
        if not row["failed"]:
            run = data[row["dataset"]].runs[(row["arm"], row["pair"], row["seed"])]
            row |= score_shape(run.s2_best, data[row["dataset"]].labels)
    return out


# --- main -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E012 training-budget check")
    ap.add_argument("--base_runs", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    cells = arms(args.base_runs, args.runs)
    gate, shared = gates(args.base_runs, args.runs)
    write_json(E012_RESULTS / "e012_gate.json", {"code": code_version(), **gate})
    if not (gate["status"]["a"] and gate["status"]["b"]):
        raise SystemExit("gate (a)/(b) failed: environment changed; re-run the baseline and S2 "
                         "(PLAN.md); see results/e012_gate.json")
    if not (gate["status"]["c"] and gate["status"]["d"] and gate["status"]["e"]):
        raise SystemExit("gate (c)/(d)/(e) failed: the stage-2 extension cannot be "
                         "interpreted; verdict suspended (PLAN.md); see results/e012_gate.json")
    cal = calibration()
    data = {d: load_dataset(list(cells.values()), d) for d in DATASETS}
    fits = {d: fit_dataset(data[d], cal[d], d, baseline=cells["base"]) for d in DATASETS}
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    all_shared = {k: True for k in shared}
    vs_base = {"s1": compare(cells["s1"], cells["base"], data, fits, cal, shared),
               "s2": compare(cells["s2"], cells["base"], data, fits, cal, all_shared),
               "s12": compare(cells["s12"], cells["base"], data, fits, cal, shared)}
    rules = {c: budget_rule(r, thr) for c, r in vs_base.items()}
    s12_vs = {}                     # S12 against each adopted smaller cell (PLAN.md)
    if rules["s12"]["adopt"]:
        smaller = {"s1": all_shared, "s2": shared}
        for c in ("s1", "s2"):
            if rules[c]["adopt"]:
                s12_vs[c] = budget_rule(
                    compare(cells["s12"], cells[c], data, fits, cal, smaller[c]), thr)
    chosen = choose_cell(rules, s12_vs)
    rows = run_rows(list(cells.values()), data, fits)
    desc = descriptives(cells, data, fits, rows)
    write_csv(E012_RESULTS / "e012_runs.csv", rows)
    write_json(E012_RESULTS / "e012_summary.json", {
        "check": "E012 training-budget check (PLAN.md)", "code": code_version(),
        "calibration": cal, "gate": gate["status"],
        "shared_stage1_chains": gate["shared_stage1_chains"],
        "cross_fit": {d: {"R": f.R, "splits_sha256": f.splits_sha256, "mc": f.mc,
                          "checks": f.checks} for d, f in fits.items()},
        "vs_base": vs_base, "budget_rule": rules, "s12_vs_smaller": s12_vs,
        "chosen": chosen,
        "sd_validity": sd_validity(fits, chosen, cal) if chosen else None,
        "descriptives": desc})
    for c in ("s1", "s2", "s12"):
        r = rules[c]
        print(c, "adopt" if r["adopt"] else "no", "| A", r.get("condition_A"), "B",
              r.get("condition_B"), "(B on Delta only:", r.get("condition_B_delta_only"), ")",
              "| worse", r.get("worse_settings"), "gain", r.get("gain_settings"))
    print("S12 vs smaller:", {k: v["adopt"] for k, v in s12_vs.items()}, "| chosen:", chosen)


if __name__ == "__main__":
    main()
