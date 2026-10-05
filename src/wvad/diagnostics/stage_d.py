"""Stage-D comparison of candidate training designs with the baseline (E010 common rule).

Each arm is one training design run as (stage 1, stage 2) chains for the two datasets, the
two feature pairs and the development seeds. Every run is checked first (configuration,
stage-2 init, records, selection, finite losses); a run with non-finite losses or scores
marks its arm as failed. The primary metric is the cross-fitted validation video AUC over
the arm's stage-2 epochs (1-30 by default) on the splits fixed by E010; Delta = candidate
seed mean - baseline seed mean per setting, judged with the per-dataset threshold delta_d.
Because stage 2 starts from a stage-1 checkpoint selected on all validation videos, the
stage-1 selection optimism (self-selected - cross-fitted V-AUC over stage-1 epochs) is
measured as well and Delta' = Delta - (candidate - baseline stage-1 optimism) must also pass
for adoption (E011 PLAN.md). Video labels only: no UCA proxy, no test data.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from wvad.data import load_split, split_file
from wvad.diagnostics.common import DEV_SEEDS, check_dev_seed
from wvad.diagnostics.e010 import check_selection, splits_sha256
from wvad.diagnostics.runs import PAIRS
from wvad.eval.crossfit import METRICS, CrossFit, crossfit, stratified_half_splits, video_auc
from wvad.paths import REPO_ROOT

E010_RESULTS = REPO_ROOT / "experiments" / "E010_dev_metric_calibration" / "results"
DATASETS = ("ucf", "xdv")
SETTINGS = tuple(itertools.product(DATASETS, PAIRS))
STAGE1_EPOCHS, STAGE2_EPOCHS = 20, 30
R_LARGE, MC_LIMIT = 1000, 0.25
WEIGHT_KEYS = ("dense", "pseudo", "neg_pseudo", "normal_seg")
INIT_TOL = 1e-5
SIGMA_FACTOR = math.sqrt(2 / 3)


@dataclass(frozen=True)
class Arm:
    name: str
    root: Path
    stage1: str                             # sub-folder of each chain for stage 1
    stage2: str                             # sub-folder for stage 2
    weights: Mapping[str, float]            # expected loss weights, both stages (absent = 0)
    reuse_e010: bool = False                # E003 runs: cross-check against e010_runs.csv
    stage1_root: Path | None = None         # where this arm's stage-1 runs live (default root)
    stage1_epochs: int = STAGE1_EPOCHS
    stage2_epochs: int = STAGE2_EPOCHS      # iterations x epochs_per_iter
    expect1: Mapping[str, object] = field(default_factory=dict)   # expected stage-1 config
    expect2: Mapping[str, object] = field(default_factory=dict)   # expected stage-2 config
    init_ckpt: str = "ckpt_best_val.pt"     # stage-1 checkpoint stage 2 starts from (E013)

    def stage1_dir(self, chain: str) -> Path:
        return Path(self.stage1_root or self.root) / chain / self.stage1


@dataclass
class RunData:
    s1: np.ndarray | None = None            # (stage1_epochs, n, S), epochs 1..
    s2: np.ndarray | None = None            # (stage2_epochs, n, S), epochs 1..
    s1_selected: float = math.nan
    s1_epoch: int = 0
    s2_selected: float = math.nan
    s2_epoch: int = 0
    history1: list = field(default_factory=list)
    history2: list = field(default_factory=list)
    failed: str | None = None

    @property
    def s2_best(self) -> np.ndarray:        # scores of the selected stage-2 checkpoint
        return self.s2[self.s2_epoch - 1]


# --- calibration (E010) ---------------------------------------------------------------------

def calibration() -> dict:
    s = json.loads((E010_RESULTS / "e010_summary.json").read_text(encoding="utf-8"))
    return {d: {"sd": v["sd"], "delta": v["delta"], "R": int(v["n_splits"]),
                "splits_sha256": v["splits_sha256"]} for d, v in s["datasets"].items()}


def e010_baseline_v_auc() -> dict[tuple, float]:
    with open(E010_RESULTS / "e010_runs.csv", encoding="utf-8") as fh:
        return {(r["dataset"], r["pair"], int(r["seed"])): float(r["v_auc"])
                for r in csv.DictReader(fh) if r["cell"] == "base"}


# --- loading and checks ---------------------------------------------------------------------

def _config(d: Path) -> dict:
    return torch.load(d / "ckpt_last.pt", map_location="cpu", weights_only=True)["config"]


def check_config(cfg: Mapping, dataset: str, pair: str, seed: int,
                 weights: Mapping[str, float], where: Path,
                 expect: Mapping[str, object] | None = None) -> None:
    check_dev_seed(cfg["seed"])
    if (cfg["dataset"], cfg["pair"], int(cfg["seed"])) != (dataset, pair, int(seed)):
        raise ValueError(f"{where}: config {cfg['dataset']}/{cfg['pair']}/s{cfg['seed']} does "
                         f"not match the folder")
    w = cfg["weights"]
    for k in WEIGHT_KEYS:
        if float(w.get(k, 0.0)) != float(weights.get(k, 0.0)):
            raise ValueError(f"{where}: loss weight {k}={w.get(k, 0.0)}, expected "
                             f"{weights.get(k, 0.0)}")
    for k, v in (expect or {}).items():
        if cfg.get(k) != v:
            raise ValueError(f"{where}: config {k}={cfg.get(k)!r}, expected {v!r}")


def _record(d: Path, epochs: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    z = np.load(d / "record_val_scores.npz")
    if not np.array_equal(z["epochs"], epochs):
        raise ValueError(f"{d}: record epochs {z['epochs'][:3]}... expected {epochs[:3]}...")
    if not np.array_equal(z["val_indices"], val_idx):
        raise ValueError(f"{d}: validation order differs from the split file")
    if z["scores"].ndim != 3 or z["scores"].shape[1] != len(val_idx):
        raise ValueError(f"{d}: score array {z['scores'].shape} for {len(val_idx)} videos")
    return z["scores"]


def _history(d: Path) -> list[dict]:
    return json.loads((d / "history.json").read_text(encoding="utf-8"))


def _finite_history(history: list[dict]) -> bool:
    return all(math.isfinite(v) for h in history for k, v in h.items() if k.startswith("loss_"))


def load_run(arm: Arm, dataset: str, pair: str, seed: int, val_idx: np.ndarray,
             labels: np.ndarray, split_sha: str) -> RunData:
    """Check one (stage 1, stage 2) chain and load its validation records."""
    check_dev_seed(seed)
    chain = arm.root / f"{dataset}_{pair}_s{seed}"
    d1, d2 = arm.stage1_dir(chain.name), chain / arm.stage2
    cfg1 = _config(d1)
    check_config(cfg1, dataset, pair, seed, arm.weights, d1,
                 {"epochs": arm.stage1_epochs, **arm.expect1})
    run = RunData(history1=_history(d1))
    s1 = _record(d1, np.arange(1, arm.stage1_epochs + 1), val_idx)
    if not (np.isfinite(s1).all() and _finite_history(run.history1)):
        # stage 2 may not exist (no stage-1 checkpoint is ever selected)
        run.failed = "non-finite loss or score in stage 1"
        return run
    cfg2 = _config(d2)
    check_config(cfg2, dataset, pair, seed, arm.weights, d2, arm.expect2)
    if int(cfg2["iterations"]) * int(cfg2["epochs_per_iter"]) != arm.stage2_epochs:
        raise ValueError(f"{d2}: stage-2 budget {cfg2['iterations']} x {cfg2['epochs_per_iter']}"
                         f" != {arm.stage2_epochs}")
    if Path(cfg2["init"]).parts[-3:] != (chain.name, arm.stage1, arm.init_ckpt):
        raise ValueError(f"{d2}: init {cfg2['init']} is not this chain's stage-1 checkpoint")
    run.history2 = _history(d2)
    s2 = _record(d2, np.arange(0, arm.stage2_epochs + 1), val_idx)
    if not (np.isfinite(s2).all() and _finite_history(run.history2)):
        run.failed = "non-finite loss or score in stage 2"
        return run
    lam = float(arm.weights.get("normal_seg", 0.0))
    for stage, hist in ((arm.stage1, run.history1), (arm.stage2, run.history2)):
        for h in hist:
            if lam > 0 and not h.get("loss_normal_seg", 0.0) > 0:
                raise ValueError(f"{chain}/{stage} epoch {h.get('epoch')}: normal-segment loss "
                                 f"missing or not positive")
            if lam == 0 and "loss_normal_seg" in h:
                raise ValueError(f"{chain}/{stage} epoch {h.get('epoch')}: normal-segment loss "
                                 f"logged for a run without it")
    sel1 = json.loads((d1 / "selection.json").read_text(encoding="utf-8"))
    sel2 = json.loads((d2 / "selection.json").read_text(encoding="utf-8"))
    for sel, d in ((sel1, d1), (sel2, d2)):
        if sel["val_split_sha"] != split_sha:
            raise ValueError(f"{d}: trained on split {sel['val_split_sha']}, not {split_sha}")
    run.s1, run.s2 = s1, s2[1:]
    run.s1_selected = check_selection(d1, run.s1, labels, sel1)
    run.s2_selected = check_selection(d2, run.s2, labels, sel2)
    run.s1_epoch, run.s2_epoch = int(sel1["best_epoch"]), int(sel2["best_epoch"])
    start = arm.stage1_epochs if arm.init_ckpt == "ckpt_last.pt" else run.s1_epoch
    if not np.allclose(s2[0], s1[start - 1], atol=INIT_TOL, rtol=0):
        raise ValueError(f"{d2}: epoch-0 scores differ from stage-1 epoch {start} "
                         f"({arm.init_ckpt})")
    return run


@dataclass(frozen=True)
class DatasetData:
    labels: np.ndarray
    runs: dict[tuple[str, str, int], RunData]     # (arm, pair, seed)


def load_dataset(arms: Sequence[Arm], dataset: str) -> DatasetData:
    split = load_split(split_file(dataset))
    val_idx = np.asarray(split.val_indices)
    labels = np.array([split.labels[i] for i in val_idx])
    runs = {(a.name, p, s): load_run(a, dataset, p, s, val_idx, labels, split.sha256)
            for a in arms for p in PAIRS for s in DEV_SEEDS}
    return DatasetData(labels, runs)


# --- cross-fit ------------------------------------------------------------------------------

@dataclass
class DatasetFits:
    R: int
    splits_sha256: str
    stage2: dict = field(default_factory=dict)    # (arm, pair, seed) -> CrossFit
    stage1: dict = field(default_factory=dict)    # (arm, pair, seed) -> CrossFit (v_auc)
    mc: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)


def _fit(data: DatasetData, R: int) -> DatasetFits:
    splits = stratified_half_splits(data.labels, R, seed=0)
    out = DatasetFits(R, splits_sha256(splits))
    for key, run in data.runs.items():
        if run.failed:
            continue
        out.stage2[key] = crossfit(run.s2, data.labels, splits)
        out.stage1[key] = crossfit(run.s1, data.labels, splits, {"v_auc": video_auc})
    return out


def _worst_mc(fits: DatasetFits, sd: float) -> float:
    return max(cf.mc_se("v_auc") for cf in fits.stage2.values()) / sd


def fit_dataset(data: DatasetData, cal: Mapping, dataset: str,
                baseline: Arm | None = None) -> DatasetFits:
    """Cross-fit on the E010 splits; a reused E003 baseline is cross-checked against E010 at
    the E010 R before any escalation; then the MC rule with the fixed E010 SD_d."""
    fits = _fit(data, cal["R"])
    if fits.splits_sha256 != cal["splits_sha256"]:
        raise ValueError(f"splits sha {fits.splits_sha256} != E010 {cal['splits_sha256']}")
    checks = {"reused_baseline_vs_e010": "not applicable (no reused baseline)"}
    if baseline is not None and baseline.reuse_e010:
        check_reused_baseline(dataset, baseline, fits)
        checks["reused_baseline_vs_e010"] = f"passed at R={fits.R}"
    elif baseline is not None:
        checks["reused_baseline_vs_e010"] = "skipped (fresh baseline runs)"
    sd = cal["sd"]["v_auc"]
    worst = _worst_mc(fits, sd)
    mc = {"first_R": cal["R"], "first_mc_se_over_sd": worst, "escalated": False}
    if worst > MC_LIMIT and cal["R"] < R_LARGE:
        prefix = stratified_half_splits(data.labels, R_LARGE, seed=0)[:cal["R"]]
        if splits_sha256(prefix) != cal["splits_sha256"]:
            raise ValueError("first splits of the larger set differ from E010's")
        fits = _fit(data, R_LARGE)
        mc |= {"escalated": True, "R": R_LARGE, "splits_sha256": fits.splits_sha256,
               "mc_se_over_sd": _worst_mc(fits, sd)}
        if mc["mc_se_over_sd"] > MC_LIMIT:
            mc["exceeds_at_max_R"] = True         # recorded only (plan)
    elif worst > MC_LIMIT:
        mc["exceeds_at_max_R"] = True             # recorded only (plan)
    fits.mc, fits.checks = mc, checks
    return fits


def check_reused_baseline(dataset: str, arm: Arm, fits: DatasetFits) -> None:
    """Re-computed cross-fitted V-AUC of reused E003 runs (at the E010 R) must equal E010's."""
    want = e010_baseline_v_auc()
    for p in PAIRS:
        for s in DEV_SEEDS:
            got = fits.stage2[(arm.name, p, s)].mean("v_auc")
            if abs(got - want[(dataset, p, s)]) > 1e-12:
                raise ValueError(f"{dataset}/{p}/s{s}: cross-fitted V-AUC {got} != E010 "
                                 f"{want[(dataset, p, s)]}")


# --- comparison -----------------------------------------------------------------------------

def classify(deltas: Mapping[tuple, float], thresholds: Mapping[str, float]) -> str:
    """E010 rule over the 4 settings; thresholds per dataset."""
    hi = sum(deltas[k] >= thresholds[k[0]] for k in deltas)
    lo = sum(deltas[k] <= -thresholds[k[0]] for k in deltas)
    if hi >= 3 and lo == 0:
        return "adopt"
    if lo >= 3:
        return "worse"
    if all(abs(deltas[k]) < thresholds[k[0]] for k in deltas):
        return "not_detected"
    return "hold"


def final_verdict(outcome: str, outcome_prime: str, failed: bool) -> dict:
    if failed:
        return {"verdict": "failed", "stage1_optimism_sensitive": False}
    if outcome == "adopt":
        ok = outcome_prime == "adopt"
        return {"verdict": "adopt" if ok else "hold", "stage1_optimism_sensitive": not ok}
    return {"verdict": outcome,
            "stage1_optimism_sensitive": outcome == "worse" and outcome_prime != "worse"}


def sign_agreement(cand: CrossFit, base: CrossFit) -> float:
    s = np.sign(cand.directions["v_auc"] - base.directions["v_auc"])
    return float(np.mean(s[:, 0] == s[:, 1]))


def selection_summary(arm: Arm, pair: str, f1: Mapping, f2: Mapping, runs: Mapping) -> dict:
    """Seed means of the stage-1 / stage-2 self-selected and cross-fitted V-AUC (descriptive 3)
    and the selected epochs."""
    keys = [(arm.name, pair, s) for s in DEV_SEEDS]
    s1c = [f1[k].mean("v_auc") for k in keys]
    s2c = [f2[k].mean("v_auc") for k in keys]
    s1s = [runs[k].s1_selected for k in keys]
    s2s = [runs[k].s2_selected for k in keys]
    return {"s1_selected": float(np.mean(s1s)), "s1_crossfit": float(np.mean(s1c)),
            "s1_optimism": float(np.mean(s1s) - np.mean(s1c)),
            "s2_selected": float(np.mean(s2s)), "s2_crossfit": float(np.mean(s2c)),
            "s2_optimism": float(np.mean(s2s) - np.mean(s2c)),
            "s1_epochs": [runs[k].s1_epoch for k in keys],
            "s2_epochs": [runs[k].s2_epoch for k in keys]}


def compare(cand: Arm, base: Arm, data: Mapping[str, DatasetData],
            fits: Mapping[str, DatasetFits], cal: Mapping,
            shared_s1: Mapping[tuple, bool] | None = None) -> dict:
    """Delta, Delta', verdict and descriptives (1-4) of one candidate against the baseline.

    Delta' subtracts the seed mean of the per-chain stage-1 optimism difference; a chain whose
    stage-1 checkpoint is the baseline's (``shared_s1[(dataset, pair, seed)]``; by default:
    the same stage-1 run folder) contributes 0, since the same start carries the same
    optimism into stage 2 (E012 PLAN.md)."""
    failed = any(data[d].runs[(cand.name, p, s)].failed
                 for d in DATASETS for p in PAIRS for s in DEV_SEEDS)
    if failed:
        return {"candidate": cand.name, **final_verdict("", "", True)}
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    settings, delta, delta_prime = {}, {}, {}
    for d, p in SETTINGS:
        f2, f1, runs = fits[d].stage2, fits[d].stage1, data[d].runs

        def mean(arm: Arm, metric: str, f2=f2, p=p) -> float:
            return float(np.mean([f2[(arm.name, p, s)].mean(metric) for s in DEV_SEEDS]))

        def s1_opt(arm: Arm, f1=f1, runs=runs, p=p) -> list[float]:
            return [runs[(arm.name, p, s)].s1_selected - f1[(arm.name, p, s)].mean("v_auc")
                    for s in DEV_SEEDS]

        def shared(s: int, d=d, p=p) -> bool:
            if shared_s1 is not None:
                return bool(shared_s1[(d, p, s)])
            chain = f"{d}_{p}_s{s}"
            return cand.stage1_dir(chain) == base.stage1_dir(chain)

        dm = {m: mean(cand, m) - mean(base, m) for m in METRICS}
        opt_c, opt_b = s1_opt(cand), s1_opt(base)
        shared_seeds = [s for s in DEV_SEEDS if shared(s)]
        keep = np.array([s not in shared_seeds for s in DEV_SEEDS])
        # mean(c) - mean(b) with shared chains zeroed: bit-identical to the E011 formula
        # when no chain is shared
        d_opt = float(np.mean(np.where(keep, opt_c, 0.0)) - np.mean(np.where(keep, opt_b, 0.0)))
        per_seed = [f2[(cand.name, p, s)].mean("v_auc") - f2[(base.name, p, s)].mean("v_auc")
                    for s in DEV_SEEDS]
        sigma = cal[d]["sd"]["v_auc"] * SIGMA_FACTOR
        aux = {m: dm[m] / cal[d]["delta"][m] for m in METRICS}
        delta[(d, p)], delta_prime[(d, p)] = dm["v_auc"], dm["v_auc"] - d_opt
        settings[f"{d}/{p}"] = {
            "delta": dm, "delta_over_threshold": aux,
            "delta_prime_v_auc": dm["v_auc"] - d_opt,
            "stage1_optimism": {"candidate": float(np.mean(opt_c)),
                                "baseline": float(np.mean(opt_b)), "difference": d_opt,
                                "difference_over_delta": d_opt / thr[d],
                                "shared_stage1_seeds": shared_seeds},
            "aux_opposite_2delta": [m for m in METRICS if m != "v_auc"
                                    and np.sign(dm[m]) == -np.sign(dm["v_auc"]) != 0
                                    and abs(aux[m]) >= 2],
            "split_half_sign_agreement": float(np.mean(
                [sign_agreement(f2[(cand.name, p, s)], f2[(base.name, p, s)]) for s in DEV_SEEDS])),
            "candidate_seed_sd_over_sd_d": float(np.std(
                [f2[(cand.name, p, s)].mean("v_auc") for s in DEV_SEEDS], ddof=1))
                / cal[d]["sd"]["v_auc"],
            "paired_delta_se_over_sigma": float(np.std(per_seed, ddof=1) / math.sqrt(3)) / sigma,
            "per_seed_delta": per_seed,
            "selection": {a.name: selection_summary(a, p, f1, f2, runs) for a in (cand, base)},
        }
    outcome, outcome_prime = classify(delta, thr), classify(delta_prime, thr)
    return {"candidate": cand.name, "outcome_delta": outcome, "outcome_delta_prime": outcome_prime,
            **final_verdict(outcome, outcome_prime, False),
            "mean_delta_over_threshold": float(np.mean([delta[k] / thr[k[0]] for k in delta])),
            "settings": settings}


def choose(results: Sequence[dict]) -> str | None:
    """Among adopted candidates (in ascending weight order): largest mean Delta/delta_d,
    ties to the earlier (smaller weight) one."""
    adopted = [r for r in results if r["verdict"] == "adopt"]
    if not adopted:
        return None
    return max(adopted, key=lambda r: (r["mean_delta_over_threshold"],
                                       -results.index(r)))["candidate"]


def run_rows(arms: Sequence[Arm], data: Mapping[str, DatasetData],
             fits: Mapping[str, DatasetFits]) -> list[dict]:
    rows = []
    for d in DATASETS:
        for a in arms:
            for p in PAIRS:
                for s in DEV_SEEDS:
                    run = data[d].runs[(a.name, p, s)]
                    row = {"dataset": d, "arm": a.name, "pair": p, "seed": s,
                           "failed": run.failed or ""}
                    if not run.failed:
                        f2, f1 = fits[d].stage2[(a.name, p, s)], fits[d].stage1[(a.name, p, s)]
                        row |= {**{m: f2.mean(m) for m in METRICS},
                                "mc_se_v_auc": f2.mc_se("v_auc"), "tie_rate": f2.tie_rate,
                                "s2_selected": run.s2_selected, "s2_epoch": run.s2_epoch,
                                "s1_crossfit_v_auc": f1.mean("v_auc"),
                                "s1_selected": run.s1_selected, "s1_epoch": run.s1_epoch,
                                "s1_optimism": run.s1_selected - f1.mean("v_auc"),
                                "s2_optimism": run.s2_selected - f2.mean("v_auc")}
                    rows.append(row)
    return rows
