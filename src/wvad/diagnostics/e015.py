"""E015 (stage D2): the video BCE on segment scores instead of the attention-pooled logit;
arms KB1 (top segment probability) and KB3 (mean of the top-3 segment probabilities), each
retrained from stage 1 (20 epochs) -> ckpt_last.pt -> stage 2 (30 epochs, no pseudo labels).

Gates (a)-(f) first; then the E010 common rule per candidate on Delta alone (outcome_delta;
the Delta' path of stage_d.compare is not used: no arm selects a stage-1 checkpoint) against
the reused E013 L20 runs (cross-checked against e013_runs.csv at the E010 R), and the choice
between adopted candidates (PLAN.md). Video labels only; the UCF UCA proxy AP is computed
last, after the verdict is written, and is descriptive only (decision 0004). No test data.

    python -m wvad.diagnostics.e015 --e003 <e003 copy> --e013 <e013 copy> --runs <e015 copy>
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from wvad.diagnostics.common import (
    DEV_SEEDS,
    code_version,
    gpu_environment,
    write_csv,
    write_json,
)
from wvad.diagnostics.e011 import score_shape
from wvad.diagnostics.e012 import ZERO, _expect2, _json
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
from wvad.diagnostics.e014 import GATE_ERRORS, e013_reference, proxy_summary
from wvad.diagnostics.runs import PAIRS
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
from wvad.eval.crossfit import crossfit, rank_auc, stratified_half_splits, video_auc
from wvad.paths import REPO_ROOT

E015_RESULTS = REPO_ROOT / "experiments" / "E015_topk_mil" / "results"
BASE = "l20"
CANDIDATES = ("kb1", "kb3")             # stage_d.choose order: a tie goes to KB1 (PLAN.md)
TOPK = {"kb1": 1, "kb3": 3}
START_EPOCH = 20
S1_CURVE = (1, 5, 10, 15, 20)
S2_CURVE = (10, 20, 30)
ENV_KEYS = ("torch", "cuda", "gpu", "driver")
UNUSED = ("delta_prime_v_auc", "stage1_optimism")     # Delta' path of compare (PLAN.md)


def arms(e003: Path, e013: Path, runs: Path) -> dict[str, Arm]:
    common = {"expect1": COMMON1, "expect2": _expect2(10), "init_ckpt": LAST}
    return {BASE: Arm(BASE, e013, "stage1", BASE, ZERO, stage1_root=e003, **common),
            **{a: Arm(a, runs, f"{a}_stage1", a, {**ZERO, "video_topk": TOPK[a]}, **common)
               for a in CANDIDATES}}


# --- gates ----------------------------------------------------------------------------------

def _ckpt(d: Path) -> dict:
    return torch.load(d / LAST, map_location="cpu", weights_only=True)


def attention(d: Path) -> dict[str, torch.Tensor]:
    return {k: v for k, v in _ckpt(d)["model_state_dict"].items() if k.startswith("attention.")}


def same_tensors(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    return bool(a) and a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def epoch_scores(d: Path, epoch: int) -> np.ndarray:
    z = np.load(d / "record_val_scores.npz")
    return z["scores"][int(np.flatnonzero(z["epochs"] == epoch)[0])]


def differ(a: np.ndarray, b: np.ndarray) -> bool:
    """NaN != NaN: two numerically failed records count as different here, so the failure is
    handled by the run checks (failed arm), not reported as an inactive option."""
    return not np.array_equal(a, b)


def gate_chain(e003: Path, runs: Path, chain: str) -> dict:
    c, ref = runs / chain, e003 / chain / "stage1"
    row = {}
    for a in CANDIDATES:
        s1, s2 = c / f"{a}_stage1", c / a
        row[f"a_{a}"] = same_record_at(s2, 0, s1, START_EPOCH)
        row[f"b_{a}"] = init_parts(s2) == (chain, f"{a}_stage1", LAST)
        row[f"c_{a}"] = all(int(_ckpt(d)["config"]["weights"].get("video_topk", 0)) == TOPK[a]
                            for d in (s1, s2))
        row[f"d_{a}"] = differ(epoch_scores(s1, 1), epoch_scores(ref, 1))
        row[f"e_{a}"] = same_tensors(attention(s1), attention(s2))
    row["d_kb1_kb3"] = differ(epoch_scores(c / "kb1_stage1", 1), epoch_scores(c / "kb3_stage1", 1))
    row["e_kb1_kb3"] = same_tensors(attention(c / "kb1_stage1"), attention(c / "kb3_stage1"))
    return row


def identity_target(identity: Path) -> tuple[dict, bool]:
    """Environment and deployed commit recorded by the no-change gate, and whether it passed."""
    ident = _json(identity)
    want = {**{k: ident["environment"][k] for k in ENV_KEYS}, "commit": ident["code"]["commit"]}
    return want, ident.get("all_ok") is True


def same_env(env, want: Mapping) -> bool:
    return isinstance(env, dict) and all(v and env.get(k) == v for k, v in want.items())


def env_check(runs: Path, identity: Path) -> dict:
    """Gate (f): each batch's env_<dataset>.json (environment and deployed commit) equals the
    no-change gate record e015_identity.json, and that gate passed."""
    try:
        want, passed = identity_target(identity)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "reason": f"e015_identity.json unreadable: {exc!r}"}
    out = {"identity_all_ok": passed, "want": want, "batches": {}}
    for d in DATASETS:
        try:
            env = _json(runs / f"env_{d}.json")
        except (OSError, ValueError):
            env = None
        out["batches"][d] = {"env": env, "same": same_env(env, want)}
    out["ok"] = passed and all(b["same"] for b in out["batches"].values())
    return out


def record_env(path: str) -> None:
    """Server, before a candidate batch (README): write this machine's environment and deployed
    commit to ``path``, then require them to equal the passed no-change gate (gate (f));
    exits non-zero otherwise, so the batch is not launched."""
    env = {**gpu_environment(), "commit": code_version()["commit"]}
    write_json(Path(path), env)
    want, passed = identity_target(E015_RESULTS / "e015_identity.json")
    if not (passed and same_env(env, want)):
        raise SystemExit(f"environment or commit differs from the no-change gate: {env} vs "
                         f"{want} (gate passed: {passed}); do not launch")
    print(f"{path}: matches the no-change gate")


GATE_KEYS = ([f"{g}_{a}" for g in "abcde" for a in CANDIDATES] + ["d_kb1_kb3", "e_kb1_kb3"])


def gates(e003: Path, runs: Path, identity: Path) -> dict:
    rows = []
    for d in DATASETS:
        for p in PAIRS:
            for s in DEV_SEEDS:
                chain = f"{d}_{p}_s{s}"
                try:
                    row = {"chain": chain, **gate_chain(e003, runs, chain)}
                except GATE_ERRORS as exc:
                    row = {"chain": chain, "error": repr(exc)}
                rows.append(row)
    status = {k: all(r.get(k, False) for r in rows if "error" not in r) for k in GATE_KEYS}
    env = env_check(runs, identity)
    status["f"] = env["ok"]
    return {"chains": rows, "status": status, "environment": env,
            "errors": [r["chain"] for r in rows if "error" in r]}


# --- verdicts -------------------------------------------------------------------------------

def decided(c: Mapping) -> dict:
    """compare() output judged on Delta alone: verdict = outcome_delta; a failed arm stays
    failed (it has no outcome_delta)."""
    if c["verdict"] == "failed":
        return {"candidate": c["candidate"], "verdict": "failed"}
    return {"candidate": c["candidate"], "verdict": c["outcome_delta"],
            "mean_delta_over_threshold": c["mean_delta_over_threshold"]}


def tidy(compared: dict, arm: str, data: Mapping[str, DatasetData]) -> None:
    """Relabel stage 1 (start epoch 20, optimism 0) for both sides and drop the Delta' fields
    and verdicts of compare(), which are not used (PLAN.md)."""
    relabel_stage1(compared, arm, data)
    relabel_stage1(compared, BASE, data)
    for k in ("verdict", "outcome_delta_prime", "stage1_optimism_sensitive"):
        compared.pop(k, None)
    for st in compared.get("settings", {}).values():
        for k in UNUSED:
            st.pop(k, None)


# --- descriptives ---------------------------------------------------------------------------

def full_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(rank_auc(scores.max(axis=1)[None], labels)[0])


def curves(data: DatasetData, arms_: Sequence[str], pair: str) -> dict:
    out = {}
    for a in arms_:
        runs = [data.runs[(a, pair, s)] for s in DEV_SEEDS]
        out[a] = {"stage1": {e: float(np.mean([full_auc(r.s1[e - 1], data.labels) for r in runs]))
                             for e in S1_CURVE},
                  "stage2": {e: float(np.mean([full_auc(r.s2[e - 1], data.labels) for r in runs]))
                             for e in S2_CURVE}}
    return out


def stage1_end(data: DatasetData, arm: str, pair: str, thr: float) -> list[float]:
    """Per seed: stage-1 epoch-20 full-validation V-AUC, candidate - baseline, in delta."""
    return [(full_auc(data.runs[(arm, pair, s)].s1[START_EPOCH - 1], data.labels)
             - full_auc(data.runs[(BASE, pair, s)].s1[START_EPOCH - 1], data.labels)) / thr
            for s in DEV_SEEDS]


def convergence_flag(data: DatasetData, arm: str, pair: str, thr: float) -> dict:
    """XCB+XCL rule (PLAN.md): seed mean of (candidate - baseline) stage-1 rise from epoch 15 to
    20, in delta; |.| >= 1 -> 'convergence-speed confound not excluded'."""
    def rise(a: str, s: int) -> float:
        r = data.runs[(a, pair, s)].s1
        return full_auc(r[19], data.labels) - full_auc(r[14], data.labels)
    v = float(np.mean([rise(arm, s) - rise(BASE, s) for s in DEV_SEEDS])) / thr
    return {"rise_difference_over_delta": v, "confound_not_excluded": abs(v) >= 1}


def loss_change(data: DatasetData, arm: str, pair: str) -> dict:
    def at(hist, e):
        return next(h["loss_mil"] for h in hist if h["epoch"] == e)
    runs = [data.runs[(arm, pair, s)] for s in DEV_SEEDS]
    return {"stage1": [float(np.mean([at(r.history1, e) for r in runs])) for e in (1, 20)],
            "stage2": [float(np.mean([at(r.history2, e) for r in runs])) for e in (1, 30)]}


def top_scores(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Selected checkpoint: mean of the 1st-3rd highest anomalous-video segment scores and of
    the top normal-video segment score."""
    desc = -np.sort(-scores, axis=1)
    a, n = desc[labels == 1], desc[labels == 0]
    return {"anomalous_top1": float(a[:, 0].mean()), "anomalous_top2": float(a[:, 1].mean()),
            "anomalous_top3": float(a[:, 2].mean()), "normal_top1": float(n[:, 0].mean())}


def k_effect(fits: Mapping, ok: Sequence[str], thr: Mapping[str, float]) -> dict | None:
    """KB3 - KB1 per setting (Delta/delta, per seed) with the common-rule class (descriptive)."""
    if not all(a in ok for a in CANDIDATES):
        return None
    deltas, out = {}, {}
    for d, p in SETTINGS:
        f2 = fits[d].stage2
        per = [f2[("kb3", p, s)].mean("v_auc") - f2[("kb1", p, s)].mean("v_auc")
               for s in DEV_SEEDS]
        deltas[(d, p)] = float(np.mean(per))
        out[f"{d}/{p}"] = {"delta_over_threshold": deltas[(d, p)] / thr[d],
                           "per_seed_over_threshold": [x / thr[d] for x in per]}
    return {"settings": out, "class_descriptive": classify(deltas, thr)}


def top3_readout(data: DatasetData, fits, arms_: Sequence[str], pair: str, thr: float) -> dict:
    """Cross-fitted V-AUC with the readout 'mean of the top-3 segment scores' (epochs selected
    by the same readout, same splits as the main analysis); Delta/delta vs the baseline."""
    splits = stratified_half_splits(data.labels, fits.R, seed=0)
    val = {}
    for a in arms_:
        per = []
        for s in DEV_SEEDS:
            s2 = data.runs[(a, pair, s)].s2
            top3 = np.sort(s2, axis=2)[:, :, -3:].mean(axis=2, keepdims=True)
            per.append(crossfit(top3, data.labels, splits, {"v_auc": video_auc}).mean("v_auc"))
        val[a] = float(np.mean(per))
    return {"v_auc": val, "delta_over_threshold": {a: (val[a] - val[BASE]) / thr
                                                   for a in arms_ if a != BASE}}


def descriptives(data, fits, cal, compared: Mapping[str, dict], ok: list[str]) -> dict:
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    arms_all = (BASE, *ok)
    out = {}
    for d, p in SETTINGS:
        k = f"{d}/{p}"
        entry = {
            "curves": curves(data[d], arms_all, p),
            "stage1_epoch20_minus_base_over_delta": {a: stage1_end(data[d], a, p, thr[d])
                                                     for a in ok},
            "loss_mil_change": {a: loss_change(data[d], a, p) for a in arms_all},
            "top3_readout": top3_readout(data[d], fits[d], arms_all, p, thr[d]),
        }
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
    return {"settings": out, "kb3_minus_kb1": k_effect(fits, ok, thr)}


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
    return rows


# --- main -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E015 video BCE on segment scores (KB1, KB3)")
    ap.add_argument("--e003", required=True, type=Path)
    ap.add_argument("--e013", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--no_proxy", action="store_true",
                    help="skip the UCA proxy AP (descriptive only; the verdict is unchanged)")
    args = ap.parse_args(argv)
    gate = gates(args.e003, args.runs, E015_RESULTS / "e015_identity.json")
    write_json(E015_RESULTS / "e015_gate.json", {"code": code_version(), **gate})
    if gate["errors"]:
        raise SystemExit(f"runs missing or unreadable for {gate['errors']} (PLAN.md); "
                         "see results/e015_gate.json")
    if not all(gate["status"].values()):
        raise SystemExit("gate failed (config/init: rerun; (d)/(e): code defect; (f): commit "
                         "or environment; PLAN.md); see results/e015_gate.json")
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
        "check": "E015 video BCE on segment scores: KB1 vs KB3 (PLAN.md)", "code": code_version(),
        "calibration": cal, "gate": gate["status"],
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
                  "stage-1 fields relabelled: start epoch 20, optimism 0"],
    }
    write_json(E015_RESULTS / "e015_summary.json", summary)     # 1. verdict and choice
    for a in CANDIDATES:
        print(a, verdicts[a]["verdict"],
              {k: round(v, 2) for k, v in summary["deltas_over_threshold"][a].items()})
    print("chosen:", chosen)
    write_csv(E015_RESULTS / "e015_runs.csv", csv_rows(cells, data, fits))
    summary["descriptives"] = descriptives(data, fits, cal, compared, ok)
    write_json(E015_RESULTS / "e015_summary.json", summary)     # 2. descriptives
    if not args.no_proxy:                                         # 3. after the verdict (0004)
        summary["uca_proxy_ap"] = {
            "note": "descriptive only (decision 0004): not used for the verdict, the choice, "
                    "the pipeline, the next candidate, reruns or claims about localisation; "
                    "low split-half reliability in E003 C5",
            "ucf": proxy_summary(data["ucf"], ok)}
        write_json(E015_RESULTS / "e015_summary.json", summary)


if __name__ == "__main__":
    main()
