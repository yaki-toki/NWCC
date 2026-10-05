"""E010: stage-D development-metric calibration on the recorded E003 runs (no training).

Primary metric: cross-fitted validation video AUC (``wvad.eval.crossfit``). Per dataset:
seed noise SD_d (root mean over the 4 (pair, cell) combinations of the 3-seed variance,
8 df), threshold delta_d = 2 * SD_d * sqrt(2/3), approximate power and false adoption of the
stage-D adoption rule, split-half sign reliability of the known full - base difference, and
the same noise for the auxiliary metrics (descriptive). Video labels only: no UCA proxy, no
test data (E010 PLAN.md).

    python -m wvad.diagnostics.e010 --runs <local copy of ~/wvad_runs/e003>
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy import stats

from wvad.data import load_split, split_file
from wvad.diagnostics.common import DEV_SEEDS, code_version, write_csv, write_json
from wvad.diagnostics.runs import CELLS, PAIRS, run_dir, selection, val_record
from wvad.eval.crossfit import METRICS, CrossFit, crossfit, rank_auc, stratified_half_splits
from wvad.paths import REPO_ROOT

E010_RESULTS = REPO_ROOT / "experiments" / "E010_dev_metric_calibration" / "results"
DATASETS = ("ucf", "xdv")
R_DEFAULT, R_LARGE, MC_LIMIT = 200, 1000, 0.25
DELTA_FACTOR = 2 * math.sqrt(2 / 3)
EFFECTS = (0.0, 1.0, 2.0, 3.0)      # true difference / SD of a difference of 3-seed means
SELECTION_TOL = 1e-6

Key = tuple[str, str, int]          # (pair, cell, seed)
KEYS: tuple[Key, ...] = tuple(itertools.product(PAIRS, CELLS, DEV_SEEDS))


def pooled_sd(groups: Iterable[Sequence[float]]) -> float:
    """Square root of the mean within-group sample variance (ddof=1)."""
    return float(math.sqrt(np.mean([np.var(g, ddof=1) for g in groups])))


def outcome_probabilities(effect: float, df: int | None = None) -> dict[str, float]:
    """Probabilities of the four stage-D outcomes over 4 independent settings.

    ``effect`` is the true difference in units of sigma, the SD of a difference of two
    3-seed means; the threshold is delta = 2 sigma. With ``df`` the threshold uses an
    estimated sigma (``df`` degrees of freedom): d / sigma_hat is noncentral t.
    """
    if df is None:
        dist = stats.norm(loc=effect)
    else:
        dist = stats.t(df) if effect == 0 else stats.nct(df, effect)
    hi, lo = float(dist.sf(2)), float(dist.cdf(-2))        # d >= delta, d <= -delta
    mid = 1 - hi - lo                                      # |d| < delta
    prob = (hi, mid, lo)
    p = {"single_setting_ge_delta": hi, "adopt": 0.0, "worse": 0.0, "not_detected": mid ** 4}
    for combo in itertools.product(range(3), repeat=4):        # 0 = hi, 1 = mid, 2 = lo
        n_hi, n_lo = combo.count(0), combo.count(2)
        if n_hi >= 3 and n_lo == 0:
            p["adopt"] += math.prod(prob[c] for c in combo)
        elif n_lo >= 3:
            p["worse"] += math.prod(prob[c] for c in combo)
    p["hold"] = 1 - p["adopt"] - p["worse"] - p["not_detected"]
    return p


def load_dataset(runs: Path, dataset: str) -> tuple[np.ndarray, dict[Key, np.ndarray],
                                                    dict[Key, float]]:
    """Validation labels (split-file order), stage-2 epoch 1-30 scores and the recorded
    selection AUC of the 12 development runs of one dataset."""
    split = load_split(split_file(dataset))
    val_idx = np.asarray(split.val_indices)
    labels = np.array([split.labels[i] for i in val_idx])
    scores, recorded = {}, {}
    for key in KEYS:
        pair, cell, seed = key
        d = run_dir(runs, dataset, pair, seed, cell)
        rec = val_record(d)
        if not np.array_equal(rec.val_indices, val_idx):
            raise ValueError(f"{d}: validation order differs from the split file")
        scores[key] = rec.scores[rec.epochs >= 1]
        recorded[key] = float(selection(d)["best_val_video_auc"])
    return labels, scores, recorded


def crossfit_all(labels: np.ndarray, scores: Mapping[Key, np.ndarray],
                 n_splits: int) -> dict[Key, CrossFit]:
    splits = stratified_half_splits(labels, n_splits, seed=0)
    return {k: crossfit(v, labels, splits) for k, v in scores.items()}


def selected_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Full-validation-set selection value (highest video AUC over the candidate epochs)."""
    return float(rank_auc(scores.max(axis=2), labels).max())


def run_rows(dataset: str, fits: Mapping[Key, CrossFit], scores: Mapping[Key, np.ndarray],
             labels: np.ndarray, recorded: Mapping[Key, float]) -> list[dict]:
    rows = []
    for key, cf in fits.items():
        pair, cell, seed = key
        sel = selected_auc(scores[key], labels)
        if abs(sel - recorded[key]) > SELECTION_TOL:
            raise ValueError(f"{dataset}/{pair}/{cell}/s{seed}: recorded scores give selection "
                             f"AUC {sel:.6f}, selection.json {recorded[key]:.6f}")
        rows.append({"dataset": dataset, "pair": pair, "cell": cell, "seed": seed,
                     "n_splits": len(cf.epochs), **{m: cf.mean(m) for m in METRICS},
                     **{f"mc_se_{m}": cf.mc_se(m) for m in METRICS},
                     "tie_rate": cf.tie_rate, "selected_val_auc": sel})
    return rows


def sign_agreement(full: CrossFit, base: CrossFit) -> float:
    """Share of splits where full - base has the same sign on half A and on half B."""
    d = full.directions["v_auc"] - base.directions["v_auc"]
    return float(np.mean(np.sign(d[:, 0]) == np.sign(d[:, 1])))


def summarize(fits: Mapping[Key, CrossFit]) -> dict:
    sd = {m: pooled_sd([[fits[(p, c, s)].mean(m) for s in DEV_SEEDS]
                        for p in PAIRS for c in CELLS]) for m in METRICS}
    diffs = {m: {p: [fits[(p, "full", s)].mean(m) - fits[(p, "base", s)].mean(m)
                     for s in DEV_SEEDS] for p in PAIRS} for m in METRICS}
    agree = {p: float(np.mean([sign_agreement(fits[(p, "full", s)], fits[(p, "base", s)])
                               for s in DEV_SEEDS])) for p in PAIRS}
    mc = max(cf.mc_se("v_auc") for cf in fits.values())
    ties = [cf.tie_rate for cf in fits.values()]
    return {"n_splits": len(next(iter(fits.values())).epochs),
            "sd": sd, "delta": {m: DELTA_FACTOR * v for m, v in sd.items()},
            "full_minus_base": {m: {p: float(np.mean(v)) for p, v in diffs[m].items()}
                                for m in METRICS},
            "full_minus_base_paired_sd": {m: pooled_sd(diffs[m].values()) for m in METRICS},
            "unpaired_difference_sd": {m: math.sqrt(2) * v for m, v in sd.items()},
            "split_half_sign_agreement": {**agree, "mean": float(np.mean(list(agree.values())))},
            "max_mc_se_v_auc": mc, "mc_se_over_sd": mc / sd["v_auc"],
            "tie_rate_mean": float(np.mean(ties)), "tie_rate_max": float(np.max(ties))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E010 development-metric calibration")
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    rows, datasets = [], {}
    for dataset in DATASETS:
        labels, scores, recorded = load_dataset(args.runs, dataset)
        fits = crossfit_all(labels, scores, R_DEFAULT)
        summary = summarize(fits)
        if summary["mc_se_over_sd"] > MC_LIMIT:
            fits = crossfit_all(labels, scores, R_LARGE)
            summary = summarize(fits)
        rows += run_rows(dataset, fits, scores, labels, recorded)
        datasets[dataset] = summary
        print(f"{dataset}: R={summary['n_splits']} SD={summary['sd']['v_auc']:.5f} "
              f"delta={summary['delta']['v_auc']:.5f} MC/SD={summary['mc_se_over_sd']:.3f} "
              f"agree={summary['split_half_sign_agreement']['mean']:.3f}", flush=True)
    power = {f"{e:g}": {"normal": outcome_probabilities(e), "t8": outcome_probabilities(e, 8)}
             for e in EFFECTS}
    write_csv(E010_RESULTS / "e010_runs.csv", rows)
    write_json(E010_RESULTS / "e010_summary.json", {
        "check": "E010 development-metric calibration (PLAN.md)", "code": code_version(),
        "primary_metric": "cross-fitted validation video AUC (v_auc), stage-2 epochs 1-30",
        "datasets": datasets,
        "power": power,
        "power_note": "keys = true difference in units of sigma (SD of a difference of two "
                      "3-seed means); delta = 2 sigma, so key 2 is an effect of exactly delta; "
                      "the 4 settings are treated as independent",
        "rule": {"adopt": ">= 3 of 4 settings with D >= delta_d, the rest D > -delta_d",
                 "worse": ">= 3 of 4 settings with D <= -delta_d",
                 "not_detected": "all 4 settings |D| < delta_d",
                 "hold": "otherwise"}})
    for e, p in power.items():
        print(f"effect {e} sigma:", {k: f"{v:.4f}" for k, v in p["normal"].items()},
              "t8 adopt", f"{p['t8']['adopt']:.4f}")


if __name__ == "__main__":
    main()
