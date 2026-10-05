"""E010: stage-D development-metric calibration on the recorded E003 runs (no training).

Primary metric: cross-fitted validation video AUC (``wvad.eval.crossfit``). Per dataset:
seed noise SD_d (root mean over the 4 (pair, cell) combinations of the 3-seed variance,
8 df), threshold delta_d = 2 * SD_d * sqrt(2/3), approximate power and false adoption of the
stage-D adoption rule, split-half sign reliability of the known full - base difference, and
the same noise for the auxiliary metrics (descriptive). Video labels only: no UCA proxy, no
test data (E010 PLAN.md). Every run is loaded and checked before anything is computed or
printed.

    python -m wvad.diagnostics.e010 --runs <local copy of ~/wvad_runs/e003>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import integrate, stats

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
SD_DF = 8                           # 4 (pair, cell) groups x (3 seeds - 1)
SELECTION_TOL = 1e-6

Key = tuple[str, str, int]          # (pair, cell, seed)
KEYS: tuple[Key, ...] = tuple(itertools.product(PAIRS, CELLS, DEV_SEEDS))


def pooled_sd(groups: Iterable[Sequence[float]]) -> float:
    """Square root of the mean within-group sample variance (ddof=1)."""
    return float(math.sqrt(np.mean([np.var(g, ddof=1) for g in groups])))


# --- power of the adoption rule -------------------------------------------------------------

def setting_probs(effect: float, u: float) -> np.ndarray:
    """P(D >= delta), P(|D| < delta), P(D <= -delta) for one setting, in units of sigma
    (SD of a difference of two 3-seed means): D ~ N(effect, 1), delta = 2u, where
    u = sigma_hat / sigma (u = 1: threshold from the true sigma)."""
    hi, lo = stats.norm.sf(2 * u - effect), stats.norm.cdf(-2 * u - effect)
    return np.array([hi, 1 - hi - lo, lo])


def pair_matrix(effect: float, df: int | None = None) -> np.ndarray:
    """(3, 3) joint outcome probabilities of the two settings of one dataset, which share
    that dataset's threshold. ``df`` None: true sigma; otherwise sigma_hat^2 / sigma^2 ~
    chi2(df) / df, integrated out (the D of the two settings are independent given it)."""
    if df is None:
        p = setting_probs(effect, 1.0)
        return np.outer(p, p)
    out = np.empty((3, 3))
    for i, j in itertools.product(range(3), repeat=2):
        def f(v, i=i, j=j):
            p = setting_probs(effect, math.sqrt(v / df))
            return p[i] * p[j] * stats.chi2.pdf(v, df)
        out[i, j] = integrate.quad(f, 0, np.inf, epsabs=1e-13, epsrel=1e-10, limit=200)[0]
    return out


def outcome_probabilities(effect: float, df: int | None = None) -> dict[str, float]:
    """Probabilities of the four stage-D outcomes over the 4 settings (2 datasets x 2 pairs).

    Settings of one dataset share the estimated threshold (``pair_matrix``); the two
    datasets' thresholds are estimated independently.
    """
    m = pair_matrix(effect, df)
    p = {"single_setting_ge_delta": float(m[0].sum()), "adopt": 0.0, "worse": 0.0,
         "not_detected": float(m[1, 1] ** 2)}
    for combo in itertools.product(range(3), repeat=4):        # 0 = hi, 1 = mid, 2 = lo
        prob = m[combo[0], combo[1]] * m[combo[2], combo[3]]
        n_hi, n_lo = combo.count(0), combo.count(2)
        if n_hi >= 3 and n_lo == 0:
            p["adopt"] += float(prob)
        elif n_lo >= 3:
            p["worse"] += float(prob)
    p["hold"] = 1 - p["adopt"] - p["worse"] - p["not_detected"]
    return p


# --- loading and integrity checks -----------------------------------------------------------

@dataclass(frozen=True)
class DatasetRuns:
    labels: np.ndarray                      # (n_val,) video labels, split-file order
    scores: dict[Key, np.ndarray]           # (30, n_val, S) stage-2 epochs 1-30
    selected: dict[Key, float]              # full-validation selection AUC (checked)


def check_selection(d: Path, scores: np.ndarray, labels: np.ndarray, sel: Mapping) -> float:
    """The recorded scores must reproduce the run's own checkpoint selection."""
    aucs = rank_auc(scores.max(axis=2), labels)
    best = float(aucs.max())
    if abs(best - float(sel["best_val_video_auc"])) > SELECTION_TOL:
        raise ValueError(f"{d}: recorded scores give selection AUC {best:.6f}, "
                         f"selection.json {sel['best_val_video_auc']:.6f}")
    if not 1 <= int(sel["best_epoch"]) <= len(aucs):
        raise ValueError(f"{d}: selected epoch {sel['best_epoch']} outside 1-{len(aucs)}")
    if abs(float(aucs[int(sel["best_epoch"]) - 1]) - best) > 1e-12:
        raise ValueError(f"{d}: selected epoch {sel['best_epoch']} is not a maximum of the "
                         f"recorded scores")
    return best


def load_dataset(runs: Path, dataset: str) -> DatasetRuns:
    split = load_split(split_file(dataset))
    val_idx = np.asarray(split.val_indices)
    labels = np.array([split.labels[i] for i in val_idx])
    scores, selected = {}, {}
    for key in KEYS:
        pair, cell, seed = key
        d = run_dir(runs, dataset, pair, seed, cell)
        rec, sel = val_record(d), selection(d)
        if sel["val_split_sha"] != split.sha256:
            raise ValueError(f"{d}: trained on split {sel['val_split_sha']}, not {split.sha256}")
        if not np.array_equal(rec.val_indices, val_idx):
            raise ValueError(f"{d}: validation order differs from the split file")
        if rec.scores.ndim != 3 or rec.scores.shape[1] != len(val_idx):
            raise ValueError(f"{d}: score array {rec.scores.shape} for {len(val_idx)} videos")
        scores[key] = rec.scores[rec.epochs >= 1]
        selected[key] = check_selection(d, scores[key], labels, sel)
    return DatasetRuns(labels, scores, selected)


# --- analysis -------------------------------------------------------------------------------

def splits_sha256(splits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(splits).tobytes()).hexdigest()


def crossfit_all(labels: np.ndarray, scores: Mapping[Key, np.ndarray],
                 n_splits: int) -> tuple[dict[Key, CrossFit], str]:
    splits = stratified_half_splits(labels, n_splits, seed=0)
    return {k: crossfit(v, labels, splits) for k, v in scores.items()}, splits_sha256(splits)


def run_rows(dataset: str, fits: Mapping[Key, CrossFit],
             selected: Mapping[Key, float]) -> list[dict]:
    return [{"dataset": dataset, "pair": pair, "cell": cell, "seed": seed,
             "n_splits": len(cf.epochs), **{m: cf.mean(m) for m in METRICS},
             **{f"mc_se_{m}": cf.mc_se(m) for m in METRICS},
             "tie_rate": cf.tie_rate, "selected_val_auc": selected[(pair, cell, seed)]}
            for (pair, cell, seed), cf in fits.items()]


def sign_agreement(full: CrossFit, base: CrossFit) -> tuple[float, float]:
    """Share of splits where V-AUC full - base has the same sign on half A and on half B
    (np.sign: a zero difference is its own sign, so (0, 0) agrees and (0, +) does not),
    and the share of splits with a zero difference on at least one half."""
    d = full.directions["v_auc"] - base.directions["v_auc"]
    s = np.sign(d)
    return float(np.mean(s[:, 0] == s[:, 1])), float(np.mean((s == 0).any(axis=1)))


def summarize(fits: Mapping[Key, CrossFit]) -> dict:
    sd = {m: pooled_sd([[fits[(p, c, s)].mean(m) for s in DEV_SEEDS]
                        for p in PAIRS for c in CELLS]) for m in METRICS}
    diffs = {m: {p: [fits[(p, "full", s)].mean(m) - fits[(p, "base", s)].mean(m)
                     for s in DEV_SEEDS] for p in PAIRS} for m in METRICS}
    signs = {p: [sign_agreement(fits[(p, "full", s)], fits[(p, "base", s)])
                 for s in DEV_SEEDS] for p in PAIRS}
    agree = {p: float(np.mean([a for a, _ in v])) for p, v in signs.items()}
    mc = max(cf.mc_se("v_auc") for cf in fits.values())
    ties = [cf.tie_rate for cf in fits.values()]
    return {"n_splits": len(next(iter(fits.values())).epochs),
            "sd": sd, "delta": {m: DELTA_FACTOR * v for m, v in sd.items()},
            "full_minus_base": {m: {p: float(np.mean(v)) for p, v in diffs[m].items()}
                                for m in METRICS},
            "full_minus_base_paired_sd": {m: pooled_sd(diffs[m].values()) for m in METRICS},
            "unpaired_difference_sd": {m: math.sqrt(2) * v for m, v in sd.items()},
            "split_half_sign_agreement": {**agree, "mean": float(np.mean(list(agree.values())))},
            "split_half_zero_share": float(np.mean([z for v in signs.values() for _, z in v])),
            "max_mc_se_v_auc": mc, "mc_se_over_sd": mc / sd["v_auc"],
            "tie_rate_mean": float(np.mean(ties)), "tie_rate_max": float(np.max(ties))}


def analyze_dataset(dataset: str, data: DatasetRuns) -> tuple[list[dict], dict]:
    """Cross-fit with R = 200; recompute with R = 1000 if any run's MC SE > 0.25 SD_d."""
    fits, sha = crossfit_all(data.labels, data.scores, R_DEFAULT)
    summary = summarize(fits)
    escalation = {"r200_mc_se_over_sd": summary["mc_se_over_sd"],
                  "escalated": summary["mc_se_over_sd"] > MC_LIMIT}
    if escalation["escalated"]:
        fits, sha = crossfit_all(data.labels, data.scores, R_LARGE)
        summary = summarize(fits)
    summary |= {"mc_escalation": escalation, "splits_sha256": sha,
                "n_val": int(len(data.labels)), "n_val_anomalous": int(data.labels.sum())}
    return run_rows(dataset, fits, data.selected), summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E010 development-metric calibration")
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    data = {d: load_dataset(args.runs, d) for d in DATASETS}      # all checks first
    rows, datasets = [], {}
    for dataset in DATASETS:
        r, datasets[dataset] = analyze_dataset(dataset, data[dataset])
        rows += r
    per_pair = [v for s in datasets.values()
                for k, v in s["split_half_sign_agreement"].items() if k != "mean"]
    power = {f"{e:g}": {"normal": outcome_probabilities(e),
                        f"t{SD_DF}": outcome_probabilities(e, SD_DF)} for e in EFFECTS}
    write_csv(E010_RESULTS / "e010_runs.csv", rows)
    write_json(E010_RESULTS / "e010_summary.json", {
        "check": "E010 development-metric calibration (PLAN.md)", "code": code_version(),
        "primary_metric": "cross-fitted validation video AUC (v_auc), stage-2 epochs 1-30",
        "datasets": datasets,
        "split_half_sign_agreement_all": float(np.mean(per_pair)),
        "power": power,
        "power_note": "keys = true difference in units of sigma (SD of a difference of two "
                      "3-seed means); delta = 2 sigma, so key 2 is an effect of exactly delta. "
                      f"normal: threshold from the true sigma; t{SD_DF}: estimated threshold "
                      f"(sigma_hat^2/sigma^2 ~ chi2({SD_DF})/{SD_DF}), shared by the two "
                      "settings of a dataset, independent between datasets",
        "rule": {"adopt": ">= 3 of 4 settings with D >= delta_d, the rest D > -delta_d",
                 "worse": ">= 3 of 4 settings with D <= -delta_d",
                 "not_detected": "all 4 settings |D| < delta_d",
                 "hold": "otherwise"}})
    for dataset, s in datasets.items():
        print(f"{dataset}: R={s['n_splits']} SD={s['sd']['v_auc']:.5f} "
              f"delta={s['delta']['v_auc']:.5f} MC/SD={s['mc_se_over_sd']:.3f} "
              f"agree={s['split_half_sign_agreement']['mean']:.3f}")
    for e, p in power.items():
        print(f"effect {e} sigma:", {k: f"{v:.4f}" for k, v in p["normal"].items()},
              f"t{SD_DF} adopt {p[f't{SD_DF}']['adopt']:.4f}")


if __name__ == "__main__":
    main()
