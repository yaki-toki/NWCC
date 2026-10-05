"""E013 (stage D): stage 2 started from the stage-1 last checkpoint instead of the
validation-selected one; arms L20 (E003 stage 1, 20 epochs), L40 (E012 40-epoch stage 1)
and the per-pair Lmix (XCB+XCL from L40, XCB+SigLIP from L20; no extra training).

Gates (a)-(c) first; then non-inferiority against the E003 base runs on Delta (Delta+ adds
the baseline's stage-1 optimism as a sensitivity: Delta+ only -> hold), and the arm choice on
Delta alone (PLAN.md). Video labels only; no UCA proxy, no test data.

    python -m wvad.diagnostics.e013 --e003 <e003 copy> --e012 <e012 copy> --runs <e013 copy>
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from wvad.diagnostics.common import DEV_SEEDS, code_version, write_csv, write_json
from wvad.diagnostics.e010 import pooled_sd
from wvad.diagnostics.e011 import score_shape
from wvad.diagnostics.e012 import (
    HIST1,
    PSEUDO_STATS,
    SD_RATIO_LIMIT,
    ZERO,
    _expect2,
    _json,
    same_history,
    same_record,
    same_state,
    split_half_excluding_identical,
)
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
from wvad.eval.crossfit import rank_auc
from wvad.paths import REPO_ROOT

E013_RESULTS = REPO_ROOT / "experiments" / "E013_stage1_last_checkpoint" / "results"
XCL, SIGLIP = "xcb_xcl", "xcb_siglip"
LAST = "ckpt_last.pt"
COMMON1 = {"weight_decay": 1e-4, "batch_size": 32, "lr": 1e-4}
NAMES = ("l20", "l40", "lmix")
PLANNED_B = {"ucf_xcb_xcl_s0", "ucf_xcb_xcl_s1", "xdv_xcb_xcl_s0", "xdv_xcb_xcl_s1"}
PLANNED_C = {"xdv_xcb_xcl_s0"}


def arms(e003: Path, e012: Path, runs: Path) -> dict[str, Arm]:
    e2 = _expect2(10)
    return {
        "base": Arm("base", e003, "stage1", "base", ZERO, reuse_e010=True,
                    expect1=COMMON1, expect2=e2),
        "l20": Arm("l20", runs, "stage1", "l20", ZERO, stage1_root=e003, expect1=COMMON1,
                   expect2=e2, init_ckpt=LAST),
        "l40": Arm("l40", runs, "s40_stage1", "l40", ZERO, stage1_root=e012, stage1_epochs=40,
                   expect1=COMMON1, expect2=e2, init_ckpt=LAST),
        "e012_s1": Arm("e012_s1", e012, "s40_stage1", "s1", ZERO, stage1_epochs=40,
                       expect1=COMMON1, expect2=e2),
    }


LMIX = Arm("lmix", Path("."), "", "lmix", ZERO)      # composed from l40 (XCL) and l20 (SigLIP)
LMIX_SOURCE = {XCL: "l40", SIGLIP: "l20"}


# --- gates ----------------------------------------------------------------------------------

def same_record_at(a: Path, ea: int, b: Path, eb: int) -> bool:
    """Recorded scores of epoch ``ea`` of run a equal epoch ``eb`` of run b (NaN = NaN)."""
    za, zb = np.load(a / "record_val_scores.npz"), np.load(b / "record_val_scores.npz")
    sa = za["scores"][int(np.flatnonzero(za["epochs"] == ea)[0])]
    sb = zb["scores"][int(np.flatnonzero(zb["epochs"] == eb)[0])]
    return sa.dtype == sb.dtype and bool(np.array_equal(sa, sb, equal_nan=True))


def same_stage2(a: Path, b: Path) -> bool:
    return (same_record(a, b, range(0, 31))
            and same_history(a, b, range(1, 31), HIST1 + PSEUDO_STATS)
            and _json(a / "selection.json") == _json(b / "selection.json"))


def init_parts(d: Path) -> tuple[str, ...]:
    """Last three parts of the stage-2 run's init path (diagnoses a gate (a) failure)."""
    import torch
    cfg = torch.load(d / LAST, map_location="cpu", weights_only=True)["config"]
    return Path(cfg["init"]).parts[-3:]


def gate_chain(e003: Path, e012: Path, runs: Path, chain: str) -> dict:
    e3, e12, e13 = e003 / chain, e012 / chain, runs / chain
    s1_3, s1_12 = e3 / "stage1", e12 / "s40_stage1"
    row = {"a_l20": same_record_at(e13 / "l20", 0, s1_3, 20),
           "a_l40": same_record_at(e13 / "l40", 0, s1_12, 40),
           "init_l20_ok": init_parts(e13 / "l20") == (chain, "stage1", LAST),
           "init_l40_ok": init_parts(e13 / "l40") == (chain, "s40_stage1", LAST)}
    row["e003_best_epoch"] = int(_json(s1_3 / "selection.json")["best_epoch"])
    if row["e003_best_epoch"] == 20:
        row["b"] = (same_state(s1_3 / "ckpt_best_val.pt", s1_3 / LAST)
                    and same_stage2(e13 / "l20", e3 / "base"))
    row["e012_s40_best_epoch"] = int(_json(s1_12 / "selection.json")["best_epoch"])
    if row["e012_s40_best_epoch"] == 40:
        row["c"] = (same_state(s1_12 / "ckpt_best_val.pt", s1_12 / LAST)
                    and same_stage2(e13 / "l40", e12 / "s1"))
    return row


def gates(e003: Path, e012: Path, runs: Path) -> tuple[dict, dict[tuple, bool]]:
    """Gate rows and the map of L20 chains whose start weights equal the baseline's."""
    rows, same_start = [], {}
    for d in DATASETS:
        for p in PAIRS:
            for s in DEV_SEEDS:
                chain = f"{d}_{p}_s{s}"
                try:
                    row = {"chain": chain, **gate_chain(e003, e012, runs, chain)}
                except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
                    row = {"chain": chain, "error": repr(exc)}
                same_start[(d, p, s)] = bool(row.get("b", False))
                rows.append(row)
    status = {k: all(r[k] for r in rows if k in r) for k in ("a_l20", "a_l40", "b", "c")}
    b_chains = [r["chain"] for r in rows if "b" in r]
    c_chains = [r["chain"] for r in rows if "c" in r]
    status["b_chains_as_planned"] = set(b_chains) == PLANNED_B        # change log
    status["c_chains_as_planned"] = set(c_chains) == PLANNED_C
    return {"chains": rows, "status": status,
            "errors": [r["chain"] for r in rows if "error" in r],
            "b_chains": b_chains, "c_chains": c_chains}, same_start


# --- Lmix -----------------------------------------------------------------------------------

def add_lmix(data: Mapping[str, DatasetData], fits: Mapping[str, DatasetFits]) -> None:
    """Register the per-pair arm under the name 'lmix' (same run objects, no copies).

    Called after cross-fitting and the MC rule: nothing iterates over all runs afterwards, so
    the aliases are never counted twice. Keep it that way."""
    for d in DATASETS:
        for p in PAIRS:
            src = LMIX_SOURCE[p]
            for s in DEV_SEEDS:
                data[d].runs[("lmix", p, s)] = data[d].runs[(src, p, s)]
                if (src, p, s) in fits[d].stage2:
                    fits[d].stage2[("lmix", p, s)] = fits[d].stage2[(src, p, s)]
                    fits[d].stage1[("lmix", p, s)] = fits[d].stage1[(src, p, s)]


# --- verdicts -------------------------------------------------------------------------------

def failed(data: Mapping[str, DatasetData], arm: str, pairs=PAIRS) -> bool:
    return any(data[d].runs[(arm, p, s)].failed for d in DATASETS for p in pairs
               for s in DEV_SEEDS)


def seed_mean(fits: DatasetFits, arm: str, pair: str) -> float:
    return float(np.mean([fits.stage2[(arm, pair, s)].mean("v_auc") for s in DEV_SEEDS]))


def base_optimism(data: DatasetData, fits: DatasetFits, pair: str) -> list[float]:
    """Per-seed stage-1 optimism of the baseline (self-selected - cross-fitted, epochs 1-20)."""
    return [data.runs[("base", pair, s)].s1_selected
            - fits.stage1[("base", pair, s)].mean("v_auc") for s in DEV_SEEDS]


def same_start_of(arm: str, d: str, p: str, s: int, same_start: Mapping[tuple, bool]) -> bool:
    """Start weights equal the baseline's: only L20 chains whose E003 stage 1 selected its last
    epoch (gate b); Lmix inherits this for its XCB+SigLIP (L20) settings."""
    src = LMIX_SOURCE[p] if arm == "lmix" else arm
    return src == "l20" and bool(same_start[(d, p, s)])


def deltas(arm: str, data, fits, same_start: Mapping[tuple, bool]) -> dict:
    """Delta and Delta+ of an arm against the baseline per setting (V-AUC)."""
    out = {}
    for d, p in SETTINGS:
        delta = seed_mean(fits[d], arm, p) - seed_mean(fits[d], "base", p)
        opt = base_optimism(data[d], fits[d], p)
        shared = [same_start_of(arm, d, p, s, same_start) for s in DEV_SEEDS]
        added = [0.0 if sh else o for sh, o in zip(shared, opt, strict=True)]
        plus = delta + float(np.mean(added))
        out[f"{d}/{p}"] = {"delta": delta, "delta_plus": plus,
                           "same_start_seeds": [s for s in DEV_SEEDS
                                                if same_start_of(arm, d, p, s, same_start)]}
    return out


def non_inferiority(d: Mapping[str, dict], thr: Mapping[str, float]) -> str:
    ok = all(v["delta"] > -thr[k.split("/")[0]] for k, v in d.items())
    ok_plus = all(v["delta_plus"] > -thr[k.split("/")[0]] for k, v in d.items())
    return "non_inferior" if ok else ("hold" if ok_plus else "not_non_inferior")


def l40_vs_l20(fits: Mapping[str, DatasetFits], pair: str) -> dict[str, float]:
    return {d: seed_mean(fits[d], "l40", pair) - seed_mean(fits[d], "l20", pair)
            for d in DATASETS}


def choose_arm(status: Mapping[str, str], xcl_diff: Mapping[str, float] | None,
               siglip_diff: Mapping[str, float] | None, thr: Mapping[str, float]) -> dict:
    """Steps 1-3 of PLAN.md on Delta alone (L40 - L20 per dataset). A diff is None when a
    run it needs failed; a needed but unavailable comparison makes the verdict hold
    (change log)."""
    if xcl_diff is None:
        return {"verdict": "hold", "chosen": None,
                "reason": "XCB+XCL comparison unavailable (failed runs)"}
    steps = {"xcl_40": (any(xcl_diff[d] >= thr[d] for d in DATASETS)
                        and not any(xcl_diff[d] <= -thr[d] for d in DATASETS))}
    if steps["xcl_40"]:
        if siglip_diff is None:
            return {**steps, "verdict": "hold", "chosen": None,
                    "reason": "XCB+SigLIP comparison unavailable (failed runs)"}
        steps["siglip_40_ok"] = not any(siglip_diff[d] <= -thr[d] for d in DATASETS)
        picked = "l40" if steps["siglip_40_ok"] else "lmix"
    else:
        picked = "l20"
    order = [picked, *{"l40": ["lmix", "l20"], "lmix": ["l20"], "l20": []}[picked]]
    chosen = next((a for a in order if status[a] == "non_inferior"), None)
    if chosen is not None:
        verdict = "adopt"
    elif any(status[a] == "hold" for a in order):
        verdict = "hold"
    else:
        verdict = "keep_current"
    return {**steps, "picked_by_steps": picked, "order": order, "chosen": chosen,
            "verdict": verdict}


def sd_validity(fits: Mapping[str, DatasetFits], arm: str, cal: Mapping) -> dict:
    out = {}
    for d in DATASETS:
        sd = pooled_sd([[fits[d].stage2[(arm, p, s)].mean("v_auc") for s in DEV_SEEDS]
                        for p in PAIRS])
        out[d] = {"pooled_sd": sd, "ratio_to_sd_d": sd / cal[d]["sd"]["v_auc"]}
    out["recalibrate"] = any(out[d]["ratio_to_sd_d"] > SD_RATIO_LIMIT for d in DATASETS)
    return out


# --- descriptives ---------------------------------------------------------------------------

def start_epoch(arm: str, pair: str) -> int:
    src = LMIX_SOURCE[pair] if arm == "lmix" else arm
    return 40 if src == "l40" else 20


def stage1_level(data: DatasetData, fits: DatasetFits, arm: str, pair: str, thr: float) -> float:
    """Seed mean of (start-epoch full-validation V-AUC - baseline stage-1 cross-fitted V-AUC),
    in delta units."""
    e = start_epoch(arm, pair)
    v = [rank_auc(data.runs[(arm, pair, s)].s1[e - 1].max(axis=1)[None], data.labels)[0]
         - fits.stage1[("base", pair, s)].mean("v_auc") for s in DEV_SEEDS]
    return float(np.mean(v)) / thr


def stage2_curve(data: DatasetData, arm: str, pair: str) -> dict:
    """Seed-mean full-validation V-AUC at stage-2 epochs 0, 10, 20, 30 (epoch 0 = start)."""
    out = {}
    for e in (0, 10, 20, 30):
        v = []
        for s in DEV_SEEDS:
            run = data.runs[(arm, pair, s)]
            if e > 0:
                scores = run.s2[e - 1]
            elif arm in ("base", "e012_s1"):
                scores = run.s1[run.s1_epoch - 1]
            else:
                scores = run.s1[start_epoch(arm, pair) - 1]
            v.append(rank_auc(scores.max(axis=1)[None], data.labels)[0])
        out[e] = float(np.mean(v))
    return out


def selection_share(fits: DatasetFits, arm: str, pair: str, lo: int, hi: int) -> float:
    sel = np.concatenate([fits.stage2[(arm, pair, s)].epochs.ravel() + 1 for s in DEV_SEEDS])
    return float(np.mean((sel >= lo) & (sel <= hi)))


def start_value(data: DatasetData, arm: str, pair: str, seed: int) -> float:
    """Full-validation V-AUC of the stage-1 epoch the arm's stage 2 starts from."""
    e = start_epoch(arm, pair)
    return float(rank_auc(data.runs[(arm, pair, seed)].s1[e - 1].max(axis=1)[None],
                          data.labels)[0])


def relabel_stage1(compared: dict, arm: str, data: Mapping[str, DatasetData]) -> None:
    """Arms without stage-1 selection: report the start epoch, its V-AUC and optimism 0 in
    the compare output's selection summary (change log); drop the stage-1 cross-fit."""
    for k, st in compared.get("settings", {}).items():
        d, p = k.split("/")
        sel = st["selection"][arm]
        sel |= {"s1_selected": float(np.mean([start_value(data[d], arm, p, s)
                                              for s in DEV_SEEDS])),
                "s1_optimism": 0.0, "s1_epochs": [start_epoch(arm, p)] * len(DEV_SEEDS)}
        sel.pop("s1_crossfit", None)


def descriptives(data, fits, cal, compared: Mapping[str, dict], ok: list[str]) -> dict:
    """Descriptives for the baseline and the arms in ``ok`` (not failed)."""
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    arms_all = ("base", *ok)
    out = {}
    for d, p in SETTINGS:
        k = f"{d}/{p}"
        entry = {
            "stage2_curve": {a: stage2_curve(data[d], a, p) for a in arms_all},
            "stage1_level_over_delta": {a: stage1_level(data[d], fits[d], a, p, thr[d])
                                        for a in ok},
            "split_half_excluding_identical": {
                a: split_half_excluding_identical(fits[d], a, "base", p) for a in ok},
        }
        if "l40" in ok:
            vs_s1 = seed_mean(fits[d], "l40", p) - seed_mean(fits[d], "e012_s1", p)
            entry["l40_vs_e012_s1"] = {
                "delta": vs_s1, "delta_over_threshold": vs_s1 / thr[d],
                "identical_seeds": split_half_excluding_identical(
                    fits[d], "l40", "e012_s1", p)["identical_seeds"]}
        if p == XCL:
            entry["share_last_10pct"] = {a: selection_share(fits[d], a, p, 28, 30)
                                         for a in arms_all}
        else:
            entry["share_epochs_1_3"] = {a: selection_share(fits[d], a, p, 1, 3)
                                         for a in arms_all}
        keys = ("delta_over_threshold", "candidate_seed_sd_over_sd_d",
                "paired_delta_se_over_sigma", "per_seed_delta", "selection")
        entry["compare"] = {a: {x: compared[a]["settings"][k][x] for x in keys} for a in ok}
        out[k] = entry
    return out


def csv_rows(cells: Mapping[str, Arm], data, fits) -> list[dict]:
    rows = run_rows([cells["base"], cells["l20"], cells["l40"]], data, fits)
    for row in rows:
        if row["failed"]:
            continue
        d, arm, p, s = row["dataset"], row["arm"], row["pair"], row["seed"]
        run = data[d].runs[(arm, p, s)]
        if arm in ("l20", "l40"):                  # no stage-1 selection (change log)
            row |= {"s1_epoch": start_epoch(arm, p), "s1_optimism": 0.0,
                    "s1_selected": start_value(data[d], arm, p, s), "s1_crossfit_v_auc": ""}
        row |= score_shape(run.s2_best, data[d].labels)
    return rows


# --- main -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E013 stage-1 last-checkpoint pipeline")
    ap.add_argument("--e003", required=True, type=Path)
    ap.add_argument("--e012", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    gate, same_start = gates(args.e003, args.e012, args.runs)
    write_json(E013_RESULTS / "e013_gate.json", {"code": code_version(), **gate})
    if gate["errors"]:
        raise SystemExit(f"runs missing or unreadable for {gate['errors']} (PLAN.md); "
                         "see results/e013_gate.json")
    if not all(gate["status"].values()):
        raise SystemExit("gate failed (environment, start point or gate chains); verdict "
                         "suspended (PLAN.md); see results/e013_gate.json (init_*_ok)")
    cells = arms(args.e003, args.e012, args.runs)
    cal = calibration()
    data = {d: load_dataset(list(cells.values()), d) for d in DATASETS}
    fits = {d: fit_dataset(data[d], cal[d], d, baseline=cells["base"]) for d in DATASETS}
    add_lmix(data, fits)
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    is_failed = {a: failed(data, a) for a in NAMES}
    ok = [a for a in NAMES if not is_failed[a]]
    d_all = {a: deltas(a, data, fits, same_start) for a in ok}
    for v in d_all.values():
        for k, x in v.items():
            t_ = thr[k.split("/")[0]]
            x |= {"delta_over_threshold": x["delta"] / t_,
                  "delta_plus_over_threshold": x["delta_plus"] / t_}
    status = {a: "failed" if is_failed[a] else non_inferiority(d_all[a], thr) for a in NAMES}
    diffs = {p: None if failed(data, "l20", (p,)) or failed(data, "l40", (p,))
             else l40_vs_l20(fits, p) for p in (XCL, SIGLIP)}
    choice = choose_arm(status, diffs[XCL], diffs[SIGLIP], thr)
    cell = {**cells, "lmix": LMIX}
    compared = {}
    for a in ok:
        shared = {(d, p, s): same_start_of(a, d, p, s, same_start) for d in DATASETS
                  for p in PAIRS for s in DEV_SEEDS}
        compared[a] = compare(cell[a], cells["base"], data, fits, cal, shared)
        relabel_stage1(compared[a], a, data)
    write_csv(E013_RESULTS / "e013_runs.csv", csv_rows(cells, data, fits))
    chosen = choice.get("chosen")
    notes = ["descriptives['*']['compare'] holds stage_d.compare output; its common-rule "
             "verdict fields are not used (E013 uses non-inferiority and the arm choice)"]
    if chosen == "l20":
        notes.append("sd_validity is partly circular for L20: 4 of its 6 XCB+XCL chains are "
                     "the E003 runs that set SD_d")
    write_json(E013_RESULTS / "e013_summary.json", {
        "check": "E013 stage-1 last-checkpoint pipeline (PLAN.md)", "code": code_version(),
        "calibration": cal, "gate": gate["status"], "gate_b_chains": gate["b_chains"],
        "gate_c_chains": gate["c_chains"],
        "cross_fit": {d: {"R": f.R, "splits_sha256": f.splits_sha256, "mc": f.mc,
                          "checks": f.checks} for d, f in fits.items()},
        "deltas": d_all, "status": status, "l40_minus_l20": diffs, "choice": choice,
        "sd_validity": sd_validity(fits, chosen, cal) if chosen else None,
        "notes": notes, "descriptives": descriptives(data, fits, cal, compared, ok)})
    for a in NAMES:
        shown = {k: (round(v["delta_over_threshold"], 2), round(v["delta_plus_over_threshold"], 2))
                 for k, v in d_all.get(a, {}).items()}
        print(a, status[a], shown)
    print("choice:", choice)


if __name__ == "__main__":
    main()
