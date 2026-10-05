"""E016 (stage D/E): a learned per-video offset added to the segment logits (GL-Score); arms
G1a (gamma x mean segment logit, gamma from 0) and G1b (linear in mean/std of the pre-dropout
context features, from 0), each retrained from stage 1 (20 epochs) -> ckpt_last.pt -> stage 2
(30 epochs, no pseudo labels).

Gates (a)-(f) first; then the E010 common rule per candidate on Delta alone (outcome_delta)
against the reused E013 L20 runs (cross-checked against e013_runs.csv at the E010 R), and the
choice between adopted candidates (PLAN.md). Descriptives include the readout decomposition
(offset-free max, offset alone, full readout) from record_val_offset.npz and the inference-only
offset G0, which is selection-contaminated (chosen on the same validation videos). Video labels
only; the UCF UCA proxy AP is computed last, after the verdict is written, and is descriptive
only (decision 0004). No test data.

    python -m wvad.diagnostics.e016 --e003 <e003 copy> --e013 <e013 copy> --runs <e016 copy>
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from wvad.data import load_split, split_file
from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.c0 import PROXY_FILE
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
from wvad.diagnostics.proxy_eval import proxy_video_set
from wvad.diagnostics.runs import PAIRS, rows_of
from wvad.diagnostics.stage_d import (
    DATASETS,
    SETTINGS,
    Arm,
    DatasetData,
    calibration,
    choose,
    classify,
    compare,
    fit_dataset,
    load_dataset,
    run_rows,
)
from wvad.diagnostics.uca import VARIANTS, load_proxy
from wvad.eval.crossfit import crossfit, rank_auc, stratified_half_splits, video_auc
from wvad.paths import REPO_ROOT

E016_RESULTS = REPO_ROOT / "experiments" / "E016_global_offset" / "results"
BASE = "l20"
CANDIDATES = ("g1a", "g1b")             # stage_d.choose order: a tie goes to G1a (PLAN.md)
MODE = {"g1a": "mean", "g1b": "stats"}
OFFSET_KEYS = {"g1a": ("offset_gamma",), "g1b": ("offset_w", "offset_b")}
START_EPOCH = 20
ADAM_BOUND = {"ucf": 0.12, "xdv": 0.30}         # |gamma| reachable within the budget (PLAN.md)
BUDGET_FRACTION = 0.8
G0_LAMBDA, CLIP = 0.5, 1e-7


def arms(e003: Path, e013: Path, runs: Path) -> dict[str, Arm]:
    return {BASE: Arm(BASE, e013, "stage1", BASE, ZERO, stage1_root=e003, expect1=COMMON1,
                      expect2=_expect2(10), init_ckpt=LAST),
            **{a: Arm(a, runs, f"{a}_stage1", a, ZERO,
                      expect1={**COMMON1, "global_mode": MODE[a]},
                      expect2={**_expect2(10), "global_mode": MODE[a]}, init_ckpt=LAST)
               for a in CANDIDATES}}


# --- gates ----------------------------------------------------------------------------------

def _ckpt(d: Path) -> dict:
    return torch.load(d / LAST, map_location="cpu", weights_only=True)


def offsets_of(d: Path) -> dict[str, torch.Tensor]:
    return {k: v for k, v in _ckpt(d)["model_state_dict"].items() if k.startswith("offset_")}


def learned(arm: str, s1: Mapping[str, torch.Tensor], s2: Mapping[str, torch.Tensor]) -> bool:
    """Gate (e): the arm's offset tensors exist, are non-zero after stage 1 and moved in
    stage 2 (keys checked first: an empty map never passes). A non-finite offset passes here:
    it is a numerical failure, and its non-finite scores mark the arm failed (PLAN.md)."""
    keys = set(OFFSET_KEYS[arm])
    if set(s1) != keys or set(s2) != keys:
        return False
    if not all(bool(torch.isfinite(t).all()) for t in (*s1.values(), *s2.values())):
        return True
    main = s1["offset_gamma"] if arm == "g1a" else s1["offset_w"]
    return bool(main.abs().sum() > 0) and any(not torch.equal(s1[k], s2[k]) for k in keys)


def check_offset_record(d: Path) -> None:
    """The stage-2 offset record exists and matches the score record (epochs, validation
    order, shapes); otherwise the chain is reported as unreadable."""
    z, ref = np.load(d / "record_val_offset.npz"), np.load(d / "record_val_scores.npz")
    if not (np.array_equal(z["epochs"], ref["epochs"])
            and np.array_equal(z["val_indices"], ref["val_indices"])
            and z["local_scores"].shape == ref["scores"].shape
            and z["offsets"].shape == ref["scores"].shape[:2]):
        raise ValueError(f"{d}: record_val_offset.npz does not match record_val_scores.npz")


def gate_chain(e003: Path, e013: Path, runs: Path, chain: str) -> dict:
    c, ref = runs / chain, e003 / chain / "stage1"
    row = {}
    for a in CANDIDATES:
        s1, s2 = c / f"{a}_stage1", c / a
        check_offset_record(s2)
        row[f"a_{a}"] = same_record_at(s2, 0, s1, START_EPOCH)
        row[f"b_{a}"] = init_parts(s2) == (chain, f"{a}_stage1", LAST)
        row[f"c_{a}"] = all(_ckpt(d)["config"].get("global_mode") == MODE[a] for d in (s1, s2))
        row[f"d_{a}"] = differ(epoch_scores(s1, 1), epoch_scores(ref, 1))
        row[f"e_{a}"] = learned(a, offsets_of(s1), offsets_of(s2))
    row["c_base"] = all(_ckpt(d)["config"].get("global_mode", "none") == "none"
                        for d in (ref, e013 / chain / BASE))
    row["d_g1a_g1b"] = differ(epoch_scores(c / "g1a_stage1", 1), epoch_scores(c / "g1b_stage1", 1))
    return row


GATE_KEYS = [f"{g}_{a}" for g in "abcde" for a in CANDIDATES] + ["c_base", "d_g1a_g1b"]


def gates(e003: Path, e013: Path, runs: Path, identity: Path) -> dict:
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
                                   else E016_RESULTS / "e016_identity.json")
    if not (passed and same_env(env, want)):
        raise SystemExit(f"environment or commit differs from the no-change gate: {env} vs "
                         f"{want} (gate passed: {passed}); do not launch")
    print(f"{path}: matches the no-change gate")


# --- descriptives ---------------------------------------------------------------------------

def tie_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """Share of (anomalous, normal) pairs whose video max scores are exactly equal."""
    m = scores.max(axis=1)
    a, n = m[labels == 1], m[labels == 0]
    return float(np.mean(a[:, None] == n[None, :]))


def offset_record(cell: Arm, dataset: str, pair: str, seed: int) -> dict:
    z = np.load(cell.root / f"{dataset}_{pair}_s{seed}" / cell.stage2 / "record_val_offset.npz")
    return {k: z[k] for k in ("epochs", "offsets", "local_scores")}


def learned_offsets(cells: Mapping[str, Arm], arm: str, dataset: str, pair: str) -> list[dict]:
    """Per seed: offset parameters at the end of stage 1 and stage 2 (gamma, or ||w|| and b)."""
    out = []
    for s in DEV_SEEDS:
        chain = cells[arm].root / f"{dataset}_{pair}_s{s}"
        row = {}
        for tag, d in (("stage1", chain / f"{arm}_stage1"), ("stage2", chain / arm)):
            t = offsets_of(d)
            if arm == "g1a":
                row[tag] = {"gamma": float(t["offset_gamma"])}
            else:
                row[tag] = {"w_norm": float(t["offset_w"].norm()), "b": float(t["offset_b"])}
        out.append(row)
    return out


def gamma_positive(rows: Sequence[dict]) -> dict:
    """Descriptive 8: seeds whose G1a gamma is positive (the post-hoc probe's sign), per stage."""
    return {tag: sum(r[tag]["gamma"] > 0 for r in rows) for tag in ("stage1", "stage2")} | {
        "of": len(rows)}


def amp_skips(data: DatasetData, arm: str, pair: str) -> list[dict]:
    """GradScaler skipped steps (PLAN.md), per seed and stage: epochs after the first with
    skipped steps, and the largest per-epoch count."""
    out = []
    for s in DEV_SEEDS:
        run = data.runs[(arm, pair, s)]
        row = {}
        for tag, hist in (("stage1", run.history1), ("stage2", run.history2)):
            k = [int(h["amp_skipped_steps"]) for h in hist]
            row[tag] = {"epochs": len(k),
                        "epochs_after_first_with_skips": sum(x > 0 for x in k[1:]),
                        "max_per_epoch": max(k)}
        out.append(row)
    return out


def budget_bound(cells: Mapping[str, Arm]) -> dict:
    """G1a budget rule (PLAN.md): any of its 12 runs with |gamma| at the end of stage 2 >= 0.8
    x the Adam bound of its dataset -> the arm is 'budget-bound'."""
    hits = []
    for d in DATASETS:
        for p in PAIRS:
            for s in DEV_SEEDS:
                g = float(offsets_of(cells["g1a"].root / f"{d}_{p}_s{s}" / "g1a")["offset_gamma"])
                if abs(g) >= BUDGET_FRACTION * ADAM_BOUND[d]:
                    hits.append({"run": f"{d}_{p}_s{s}", "gamma": g})
    return {"budget_bound": bool(hits), "runs": hits}


def offset_distribution(cell: Arm, data: DatasetData, dataset: str, pair: str) -> dict:
    """c(v) at each seed's selected stage-2 checkpoint: anomalous / normal means, SD across
    videos and the full-validation AUC of c alone (not cross-fitted); seed means."""
    rows = []
    for s in DEV_SEEDS:
        rec = offset_record(cell, dataset, pair, s)
        e = data.runs[(cell.name, pair, s)].s2_epoch
        c = rec["offsets"][int(np.flatnonzero(rec["epochs"] == e)[0])]
        rows.append({"anomalous_mean": float(c[data.labels == 1].mean()),
                     "normal_mean": float(c[data.labels == 0].mean()),
                     "sd": float(c.std()),
                     "auc_full_validation": float(rank_auc(c[None], data.labels)[0])})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def _cf(scores: np.ndarray, labels: np.ndarray, splits: np.ndarray) -> float:
    return crossfit(scores, labels, splits, {"v_auc": video_auc}).mean("v_auc")


def decomposition(cell: Arm, data: DatasetData, fits, dataset: str, pair: str,
                  thr: float) -> dict:
    """Cross-fitted V-AUC (same splits, epoch chosen by each readout) of (i) the offset-free
    max, (ii) the offset alone, (iii) the full readout, from recorded values only; seed means
    and Delta/delta vs the baseline."""
    splits = stratified_half_splits(data.labels, fits.R, seed=0)
    vals = {"i_offset_free": [], "ii_offset_only": [], "iii_full": []}
    for s in DEV_SEEDS:
        rec = offset_record(cell, dataset, pair, s)
        keep = rec["epochs"] >= 1
        vals["i_offset_free"].append(_cf(rec["local_scores"][keep], data.labels, splits))
        vals["ii_offset_only"].append(_cf(rec["offsets"][keep][:, :, None], data.labels, splits))
        vals["iii_full"].append(_cf(data.runs[(cell.name, pair, s)].s2, data.labels, splits))
    base = float(np.mean([fits.stage2[(BASE, pair, s)].mean("v_auc") for s in DEV_SEEDS]))
    mean = {k: float(np.mean(v)) for k, v in vals.items()}
    full_matches = all(abs(v - fits.stage2[(cell.name, pair, s)].mean("v_auc")) <= 1e-12
                       for v, s in zip(vals["iii_full"], DEV_SEEDS, strict=True))
    return {"v_auc": mean, "baseline_v_auc": base,
            "delta_over_threshold": {k: (v - base) / thr for k, v in mean.items()},
            "iii_equals_verdict_value": full_matches}


def characterize(dec: Mapping[str, dict]) -> str:
    """Adoption wording (PLAN.md), means over the settings: (iii) - (i) >= (i) - baseline ->
    'global evidence', otherwise 'training change'."""
    full = np.mean([v["v_auc"]["iii_full"] for v in dec.values()])
    free = np.mean([v["v_auc"]["i_offset_free"] for v in dec.values()])
    base = np.mean([v["baseline_v_auc"] for v in dec.values()])
    return "global_evidence" if full - free >= free - base else "training_change"


def g0_readout(scores: np.ndarray) -> np.ndarray:
    """Inference-only offset G0 (selection-contaminated): max_t z'_t + 0.5 mean_s z'_s, from the
    recorded probabilities clipped to [1e-7, 1 - 1e-7]; shape (..., n, 1)."""
    p = np.clip(scores.astype(np.float64), CLIP, 1 - CLIP)
    z = np.log(p) - np.log1p(-p)
    return (z.max(axis=-1) + G0_LAMBDA * z.mean(axis=-1))[..., None]


def g0_minus_own(cell: Arm, data: DatasetData, fits, pair: str, thr: float) -> dict:
    splits = stratified_half_splits(data.labels, fits.R, seed=0)
    per = [(_cf(g0_readout(data.runs[(cell.name, pair, s)].s2), data.labels, splits)
            - fits.stage2[(cell.name, pair, s)].mean("v_auc")) / thr for s in DEV_SEEDS]
    return {"per_seed_over_threshold": per, "mean_over_threshold": float(np.mean(per)),
            "label": "selection-contaminated (transform chosen on the same validation videos)"}


def g1b_minus_g1a(fits: Mapping, ok: Sequence[str], thr: Mapping[str, float]) -> dict | None:
    if not all(a in ok for a in CANDIDATES):
        return None
    deltas, out = {}, {}
    for d, p in SETTINGS:
        f2 = fits[d].stage2
        per = [f2[("g1b", p, s)].mean("v_auc") - f2[("g1a", p, s)].mean("v_auc")
               for s in DEV_SEEDS]
        deltas[(d, p)] = float(np.mean(per))
        out[f"{d}/{p}"] = {"delta_over_threshold": deltas[(d, p)] / thr[d],
                           "per_seed_over_threshold": [x / thr[d] for x in per]}
    return {"settings": out, "class_descriptive": classify(deltas, thr),
            "note": "the arms also differ in effective learning speed of c (PLAN.md)"}


def descriptives(cells, data, fits, cal, compared: Mapping[str, dict], ok: list[str],
                 verdicts: Mapping[str, dict]) -> dict:
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    arms_all = (BASE, *ok)
    out, dec_by_arm = {}, {a: {} for a in ok}
    for d, p in SETTINGS:
        k = f"{d}/{p}"
        entry = {
            "curves": curves(data[d], arms_all, p),
            "stage1_epoch20_minus_base_over_delta": {a: stage1_end(data[d], a, p, thr[d])
                                                     for a in ok},
            "loss_mil_change": {a: loss_change(data[d], a, p) for a in arms_all},
            "tie_rate_selected": {a: float(np.mean([tie_rate(data[d].runs[(a, p, s)].s2_best,
                                                             data[d].labels)
                                                    for s in DEV_SEEDS])) for a in arms_all},
            "learned_offsets": {a: learned_offsets(cells, a, d, p) for a in ok},
            "amp_skipped_steps": {a: amp_skips(data[d], a, p) for a in ok},
            "offset_distribution": {a: offset_distribution(cells[a], data[d], d, p) for a in ok},
            "decomposition": {},
            "g0_minus_own_readout": {a: g0_minus_own(cells[a], data[d], fits[d], p, thr[d])
                                     for a in ok},
        }
        if "g1a" in ok:
            entry["g1a_gamma_positive_seeds"] = gamma_positive(entry["learned_offsets"]["g1a"])
        for a in ok:
            dec = decomposition(cells[a], data[d], fits[d], d, p, thr[d])
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
    return {"settings": out, "g1b_minus_g1a": g1b_minus_g1a(fits, ok, thr),
            "characterization": {a: characterize(dec_by_arm[a]) for a in ok
                                 if verdicts[a]["verdict"] == "adopt"},     # adoptions only
            "g1a_budget": budget_bound(cells) if "g1a" in ok else None}


def csv_rows(cells: Mapping[str, Arm], data, fits) -> list[dict]:
    rows = run_rows([cells[BASE], *(cells[a] for a in CANDIDATES)], data, fits)
    for row in rows:
        if row["failed"]:
            continue
        d, arm, p, s = row["dataset"], row["arm"], row["pair"], row["seed"]
        best = data[d].runs[(arm, p, s)].s2_best
        row |= {"s1_epoch": START_EPOCH, "s1_optimism": 0.0,       # no stage-1 selection
                "s1_selected": start_value(data[d], arm, p, s), "s1_crossfit_v_auc": ""}
        row |= score_shape(best, data[d].labels) | top_scores(best, data[d].labels)
        if arm != BASE:
            t = offsets_of(cells[arm].root / f"{d}_{p}_s{s}" / arm)
            row |= {(f"stage2_end_{k}" if v.ndim == 0 else f"stage2_end_{k}_norm"):
                    (float(v) if v.ndim == 0 else float(v.norm())) for k, v in t.items()}
    return rows


def proxy_summary(data: DatasetData, ok: Sequence[str]) -> dict:
    """UCF UCA proxy frame AP (descriptive only, decision 0004) of each run's selected stage-2
    checkpoint: four variants, and the AP over anomalous videos only."""
    manifest, proxy = load_manifest(manifest_file("ucf")), load_proxy(PROXY_FILE)
    val_idx = np.asarray(load_split(split_file("ucf")).val_indices)
    sets = {v: proxy_video_set(manifest, proxy, v, val_idx) for v in VARIANTS}
    out = {}
    for p in PAIRS:
        per = {}
        for a in (BASE, *ok):
            rows = []
            for s in DEV_SEEDS:
                best = data.runs[(a, p, s)].s2_best
                r = {}
                for v, vs in sets.items():
                    sc = best[rows_of(val_idx, vs.indices)]
                    r[v] = vs.ap(sc)
                    r[f"{v}_anomalous_only"] = vs.ap(sc, rows=vs.anomalous)
                rows.append(r)
            per[a] = rows
        keys = list(per[BASE][0])
        out[p] = {"per_seed": per, "minus_base": {
            a: {k: float(np.mean([x[k] for x in per[a]]) - np.mean([x[k] for x in per[BASE]]))
                for k in keys} for a in ok}}
    return out


# --- main -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E016 global offset on segment logits (G1a, G1b)")
    ap.add_argument("--e003", required=True, type=Path)
    ap.add_argument("--e013", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--no_proxy", action="store_true",
                    help="skip the UCA proxy AP (descriptive only; the verdict is unchanged)")
    args = ap.parse_args(argv)
    gate = gates(args.e003, args.e013, args.runs, E016_RESULTS / "e016_identity.json")
    write_json(E016_RESULTS / "e016_gate.json", {"code": code_version(), **gate})
    if gate["errors"]:
        raise SystemExit(f"runs missing or unreadable for {gate['errors']} (PLAN.md); "
                         "see results/e016_gate.json")
    if not all(gate["status"].values()):
        raise SystemExit("gate failed (config/init: rerun; (d)/(e): code defect; (f): commit "
                         "or environment; PLAN.md); see results/e016_gate.json")
    cells = arms(args.e003, args.e013, args.runs)
    cal = calibration()
    data = {d: load_dataset(list(cells.values()), d) for d in DATASETS}
    fits = {d: fit_dataset(data[d], cal[d], d, baseline=cells[BASE],
                           reference=e013_reference(d)) for d in DATASETS}
    compared = {a: compare(cells[a], cells[BASE], data, fits, cal) for a in CANDIDATES}
    verdicts = {a: decided(compared[a]) for a in CANDIDATES}
    chosen = choose([verdicts[a] for a in CANDIDATES])
    ok = [a for a in CANDIDATES if verdicts[a]["verdict"] != "failed"]
    for a in ok:
        tidy(compared[a], a, data)
    summary = {
        "check": "E016 global offset on segment logits: G1a vs G1b (PLAN.md)",
        "code": code_version(), "calibration": cal, "gate": gate["status"],
        "cross_fit": {d: {"R": f.R, "splits_sha256": f.splits_sha256, "mc": f.mc,
                          "checks": f.checks} for d, f in fits.items()},
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
                  "G0 values are selection-contaminated and not used for interpretation"],
    }
    write_json(E016_RESULTS / "e016_summary.json", summary)     # 1. verdict and choice
    for a in CANDIDATES:
        print(a, verdicts[a]["verdict"],
              {k: round(v, 2) for k, v in summary["deltas_over_threshold"][a].items()})
    print("chosen:", chosen)
    write_csv(E016_RESULTS / "e016_runs.csv", csv_rows(cells, data, fits))
    summary["descriptives"] = descriptives(cells, data, fits, cal, compared, ok, verdicts)
    write_json(E016_RESULTS / "e016_summary.json", summary)     # 2. descriptives
    if not args.no_proxy:                                         # 3. after the verdict (0004)
        summary["uca_proxy_ap"] = {
            "note": "descriptive only (decision 0004): not used for the verdict, the choice, "
                    "the pipeline, the next candidate, reruns or claims about localisation; "
                    "low split-half reliability in E003 C5",
            "ucf": proxy_summary(data["ucf"], ok)}
        write_json(E016_RESULTS / "e016_summary.json", summary)


if __name__ == "__main__":
    main()
