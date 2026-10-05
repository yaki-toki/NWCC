"""E011 (stage D1): BCE on every normal-video segment, two weights, against the base pipeline.

The verdict follows the stage-D common rule with the stage-1 optimism condition
(``wvad.diagnostics.stage_d``). Descriptives specific to E011, never used for the verdict
or any choice: score shape at the selected stage-2 checkpoint, the weighted ratio of the
normal-segment loss to the MIL term, and (UCF only, decision 0004) the UCA proxy frame AP.

The reused E003 baseline requires the no-change gate (``results/e011_identity.json``,
``all_ok``). If the gate failed for environment reasons only, the plan re-runs the baseline
with the candidates' commit: pass those runs with ``--fresh_baseline`` (same layout,
``stage1`` / ``base``; the E010 cross-check is then skipped). Command (README, E011):

    python -m wvad.diagnostics.e011 --base_runs <e003 copy> --runs <e011 copy>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from wvad.data import load_split, split_file
from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.c0 import PROXY_FILE
from wvad.diagnostics.common import DEV_SEEDS, code_version, write_csv, write_json
from wvad.diagnostics.proxy_eval import proxy_video_set
from wvad.diagnostics.runs import PAIRS, rows_of
from wvad.diagnostics.stage_d import (
    DATASETS,
    Arm,
    DatasetData,
    calibration,
    choose,
    compare,
    fit_dataset,
    load_dataset,
    run_rows,
)
from wvad.diagnostics.uca import VARIANTS, load_proxy
from wvad.paths import REPO_ROOT

E011_RESULTS = REPO_ROOT / "experiments" / "E011_dense_normal_negatives" / "results"
ZERO = {"dense": 0.0, "pseudo": 0.0, "neg_pseudo": 0.0}
CANDIDATES = (("d1a", 0.1), ("d1b", 1.0))           # ascending weight (tie rule)
MIL_WEIGHT = 0.1
DESCRIPTIVE_PREFIXES = ("normal_", "anomalous_", "loss_ratio_", "proxy_ap_")


def arms(base_root: Path, runs_root: Path, reuse_e003: bool = True) -> list[Arm]:
    base = Arm("base", base_root, "stage1", "base", {**ZERO, "normal_seg": 0.0},
               reuse_e010=reuse_e003)
    return [base, *(Arm(n, runs_root, f"{n}_stage1", n, {**ZERO, "normal_seg": lam})
                    for n, lam in CANDIDATES)]


def gate_status() -> str:
    """The reused E003 baseline needs a passed no-change gate."""
    path = E011_RESULTS / "e011_identity.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no-change gate result missing (README, E011)")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("all_ok") is not True:
        raise ValueError(f"{path}: no-change gate did not pass; see PLAN.md for the fallback")
    return f"passed (code {gate['code'].get('commit')})"


def score_shape(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Validation segment scores (n, S) of the selected checkpoint."""
    normal, anomalous = scores[labels == 0], scores[labels == 1]
    flat = normal.ravel()
    return {"normal_mean": float(flat.mean()), "normal_p99": float(np.quantile(flat, 0.99)),
            "anomalous_max_mean": float(anomalous.max(axis=1).mean()),
            "anomalous_mean": float(anomalous.mean()),
            "normal_frac_ge_0999": float(np.mean(flat >= 0.999)),
            "normal_frac_le_0001": float(np.mean(flat <= 0.001)),
            "anomalous_frac_ge_0999": float(np.mean(anomalous >= 0.999)),
            "anomalous_frac_le_0001": float(np.mean(anomalous <= 0.001))}


def loss_ratio(history: Sequence[Mapping], lam: float) -> dict:
    """lambda * L_n / (0.1 * L_mil) at the first and last epoch (candidates only)."""
    if lam == 0:
        return {}
    r = [lam * h["loss_normal_seg"] / (MIL_WEIGHT * h["loss_mil"]) for h in history]
    return {"first": r[0], "last": r[-1]}


def proxy_aps(data: DatasetData) -> dict[tuple, dict[str, float]]:
    """UCA proxy frame AP (percent) of each UCF run's selected stage-2 checkpoint."""
    manifest, proxy = load_manifest(manifest_file("ucf")), load_proxy(PROXY_FILE)
    val_idx = np.asarray(load_split(split_file("ucf")).val_indices)
    sets = {v: proxy_video_set(manifest, proxy, v, val_idx) for v in VARIANTS}
    return {key: {v: vs.ap(run.s2_best[rows_of(val_idx, vs.indices)]) for v, vs in sets.items()}
            for key, run in data.runs.items() if not run.failed}


def descriptives(all_arms: Sequence[Arm], data: Mapping[str, DatasetData],
                 rows: list[dict]) -> dict:
    """Adds per-run descriptives 5-7 to ``rows``; returns seed-mean summaries."""
    lam = {a.name: float(a.weights["normal_seg"]) for a in all_arms}
    proxy = proxy_aps(data["ucf"])
    for row in rows:
        if row["failed"]:
            continue
        key = (row["arm"], row["pair"], row["seed"])
        run = data[row["dataset"]].runs[key]
        row |= score_shape(run.s2_best, data[row["dataset"]].labels)
        for stage, hist in (("s1", run.history1), ("s2", run.history2)):
            row |= {f"loss_ratio_{stage}_{k}": v for k, v in loss_ratio(hist, lam[key[0]]).items()}
        if row["dataset"] == "ucf":
            row |= {f"proxy_ap_{v}": ap for v, ap in proxy[key].items()}
    summary = {}
    for d in DATASETS:
        for p in PAIRS:
            for a in all_arms:
                rr = [r for r in rows if (r["dataset"], r["pair"], r["arm"]) == (d, p, a.name)
                      and not r["failed"]]
                if len(rr) != len(DEV_SEEDS):
                    continue
                keys = [k for k in rr[0] if k.startswith(DESCRIPTIVE_PREFIXES)]
                summary[f"{d}/{p}/{a.name}"] = {k: float(np.mean([r[k] for r in rr])) for k in keys}
    for p in PAIRS:
        base = summary.get(f"ucf/{p}/base")
        for name, _ in CANDIDATES:
            cand = summary.get(f"ucf/{p}/{name}")
            if cand and base:
                cand["proxy_ap_minus_base"] = {v: cand[f"proxy_ap_{v}"] - base[f"proxy_ap_{v}"]
                                               for v in VARIANTS}
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E011 stage D1 analysis")
    ap.add_argument("--base_runs", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--fresh_baseline", type=Path,
                    help="re-run baseline root (gate failed for environment reasons only)")
    args = ap.parse_args(argv)
    if args.fresh_baseline is None:
        baseline = {"source": "E003 base runs (reused)", "gate": gate_status()}
        all_arms = arms(args.base_runs, args.runs)
    else:
        baseline = {"source": f"fresh baseline runs: {args.fresh_baseline}"}
        all_arms = arms(args.fresh_baseline, args.runs, reuse_e003=False)
    cal = calibration()
    data = {d: load_dataset(all_arms, d) for d in DATASETS}       # every run checked first
    fits = {d: fit_dataset(data[d], cal[d], d, baseline=all_arms[0]) for d in DATASETS}
    results = [compare(c, all_arms[0], data, fits, cal) for c in all_arms[1:]]
    rows = run_rows(all_arms, data, fits)
    desc = descriptives(all_arms, data, rows)
    write_csv(E011_RESULTS / "e011_runs.csv", rows)
    write_json(E011_RESULTS / "e011_summary.json", {
        "check": "E011 stage D1 (PLAN.md)", "code": code_version(),
        "calibration": cal, "baseline": baseline,
        "cross_fit": {d: {"R": f.R, "splits_sha256": f.splits_sha256, "mc": f.mc,
                          "checks": f.checks} for d, f in fits.items()},
        "candidates": results, "chosen": choose(results),
        "descriptives": desc,
        "note": "descriptives (score shape, loss ratio, UCA proxy frame AP) are not used for "
                "the verdict or any choice (PLAN.md)"})
    for r in results:
        print(r["candidate"], r["verdict"], "| delta:", r.get("outcome_delta"),
              "delta':", r.get("outcome_delta_prime"),
              "| mean D/delta:", round(r.get("mean_delta_over_threshold", float("nan")), 2))
    print("chosen:", choose(results))


if __name__ == "__main__":
    main()
