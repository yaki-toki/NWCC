"""E021 arms, readouts and cross-fits (PLAN.md "팔", "판독", "판독과 지표", "문턱"). Local
analysis code: no annotation file, neither the glance nor the UCA module.

An arm is a head read on validation units (PLAN.md "팔과 대비"):

    R0 (h0, 32 own legacy)   R1v (h0, 32 d16c)   R1rv (h0, 32 d16r)   R1rv1 / R1rv2 (UCF draws)
    A0 (h0, W)   A0m (h0, Wm)   A0.G (h0, G; extreme-readout sensitivity)
    R1 (h1, 32 own d16c)   A1 (h1, W)   R1r (h1r, 32 own d16r)
    B (hb, 32 own d16m)   C (hb, W)   C.G (hb, G)   Bm (hbm, 32 own d16m2)   Cm (hbm, Wm)

Readouts: primary = fp32 sigma (32-unit arms on their own features: the training record, the
stage-D standard; other 32-unit arms: the scoring's sigma; W / Wm / G:
wvad.eval.dense_readout.readout, sigma in fp32), logit = the same without sigma (selection and
metrics). The video score is the maximum over the 32 units (G: over all snippets; NS-AUC is not
computed for G).

Cross-fit (wvad.eval.crossfit): stage-2 epochs 1-30, the E010 / E018 splits (UCF R = 1000, XD R =
200), the epoch selected per split by the arm's own readout (V-AUC), LS-AUC on the same halves.
Development seeds get every metric; calibration seeds V-AUC and LS-AUC (R0: every metric, for gate
(h)). MC rule (E018): when the worst MC SE of a dataset's V-AUC (primary readout, any arm and
seed) exceeds 0.25 SD_{P,d} of its pipeline lineage and R < 1000, the dataset is fitted again at
R = 1000 (the first splits unchanged).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from wvad.diagnostics.e010 import splits_sha256
from wvad.diagnostics.e021_gates import Run, centre_rows, snippet_rows
from wvad.diagnostics.e021_rules import (
    LOGIT,
    PRIMARY,
    READOUTS,
    SETTINGS,
    ls_threshold,
    pooled_sd,
    z_of,
)
from wvad.diagnostics.e021_score import (
    CAL_SEEDS,
    DEV_SEEDS,
    HEAD_SEEDS,
    PAIRS,
    PIPELINES,
)
from wvad.eval.crossfit import (
    METRICS,
    CrossFit,
    crossfit_with_length,
    select_epoch,
    stratified_half_splits,
    video_auc,
)
from wvad.eval.dense_readout import readout, sigmoid32

R_LARGE, MC_LIMIT = 1000, 0.25


class Arm(NamedTuple):
    head: str
    kind: str                    # record | unit | W | Wm | G
    variant: str | None = None   # 32-unit variant (record, unit)


ARMS = {"R0": Arm("h0", "record", "legacy"), "R1v": Arm("h0", "unit", "d16c"),
        "R1rv": Arm("h0", "unit", "d16r"), "R1rv1": Arm("h0", "unit", "d16r1"),
        "R1rv2": Arm("h0", "unit", "d16r2"), "A0": Arm("h0", "W"), "A0m": Arm("h0", "Wm"),
        "A0.G": Arm("h0", "G"),
        "R1": Arm("h1", "record", "d16c"), "A1": Arm("h1", "W"),
        "R1r": Arm("h1r", "record", "d16r"),
        "B": Arm("hb", "record", "d16m"), "C": Arm("hb", "W"), "C.G": Arm("hb", "G"),
        "Bm": Arm("hbm", "record", "d16m2"), "Cm": Arm("hbm", "Wm")}
STAGE_ARMS = {1: ("R0", "R1v", "R1rv", "R1rv1", "R1rv2", "A0", "A0m", "A0.G"),
              2: ("R1", "A1", "R1r", "B", "C", "C.G", "Bm", "Cm")}
UCF_ONLY = ("R1rv1", "R1rv2")
VERDICT_CONTRASTS = {1: ("A0-R1v", "A0-R0", "A0.G-R1v", "A0.G-R0"),
                     2: ("C-R1", "C-R0", "C.G-R1", "C.G-R0")}
DESCRIPTIVE_CONTRASTS = {1: ("R1v-R0", "R1rv-R0", "R1rv1-R0", "R1rv2-R0", "R1v-R1rv", "A0m-R1v",
                             "A0-A0m"),
                         2: ("R1-R0", "R1r-R0", "R1-R1r", "A1-R1", "B-R1", "Bm-R1", "C-B", "C-A1",
                             "Cm-R1", "C-Cm", "A1-A0")}
DENSE_ARMS = ("A0", "A1", "C")             # W arms with a snippet resolution (descriptives)
ALL_METRICS = (*METRICS, "ls_auc")

FitKey = tuple[str, str, int]              # (pipeline, pair, seed)


def arms_of(stage: int, dataset: str) -> tuple[str, ...]:
    """Arms of the analysis up to ``stage`` (the final analysis also reads the stage-1 arms)."""
    names = [a for k in range(1, stage + 1) for a in STAGE_ARMS[k]]
    return tuple(a for a in names if dataset == "ucf" or a not in UCF_ONLY)


def contrasts_of(stage: int) -> tuple[str, ...]:
    return tuple(c for k in range(1, stage + 1)
                 for c in (*VERDICT_CONTRASTS[k], *DESCRIPTIVE_CONTRASTS[k]))


def seeds_of(arm: str) -> tuple[int, ...]:
    return HEAD_SEEDS[ARMS[arm].head]


# --- readouts -------------------------------------------------------------------------------

def dense_readouts(sf: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """W and Wm window logits (E, n, 32), the G video maximum logit and fp32 sigma (E, n) of every
    epoch and validation video (wvad.eval.dense_readout.readout); M10 heads also the offset-free
    W (z only) and the c(v) of W and Wm (E, n)."""
    z, counts, totals = sf["dense_z"], np.asarray(sf["dense_counts"]), sf["total_frames"]
    gamma = sf["gamma"] if "gamma" in sf else None
    n_ep, n = z.shape[0], len(counts)
    starts = snippet_rows(counts)
    out = {k: np.empty((n_ep, n, 32), np.float32) for k in ("W", "Wm")}
    out |= {k: np.empty((n_ep, n), np.float32) for k in ("G_logit", "G_prob")}
    if gamma is not None:
        out |= {"W_free": np.empty((n_ep, n, 32), np.float32),
                "c_W": np.empty((n_ep, n), np.float32), "c_Wm": np.empty((n_ep, n), np.float32)}
    for e in range(n_ep):
        g = None if gamma is None else float(gamma[e])
        for i in range(n):
            zi, t = z[e, starts[i]:starts[i] + counts[i]], int(totals[i])
            w, wm, gr = readout(zi, t, g, "W"), readout(zi, t, g, "Wm"), readout(zi, t, g, "G")
            out["W"][e, i], out["Wm"][e, i] = w.logits, wm.logits
            out["G_logit"][e, i], out["G_prob"][e, i] = gr.video_logit, gr.video_score
            if gamma is not None:
                out["W_free"][e, i] = readout(zi, t, None, "W").logits
                out["c_W"][e, i], out["c_Wm"][e, i] = w.offset, wm.offset
    return out


def arm_values(arm: str, run: Run, sf: Mapping[str, np.ndarray],
               dense: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    """(30, n, S) values of an arm's primary and logit readouts on one run."""
    a = ARMS[arm]
    if a.kind == "record":
        return {PRIMARY: run.record, LOGIT: sf[f"{a.variant}_logits"]}
    if a.kind == "unit":
        return {PRIMARY: sf[f"{a.variant}_probs"], LOGIT: sf[f"{a.variant}_logits"]}
    if a.kind == "G":
        return {PRIMARY: dense["G_prob"][..., None], LOGIT: dense["G_logit"][..., None]}
    logits = dense[a.kind]
    return {PRIMARY: sigmoid32(logits), LOGIT: logits}


def full_epoch(values: np.ndarray, labels: np.ndarray) -> int:
    """Index of the epoch with the highest full-validation V-AUC (earliest on ties): the
    checkpoint rule of the confirmation plan for arms without their own ckpt_best_val.pt."""
    return select_epoch(values.max(axis=2), labels, np.arange(len(labels)))[0]


# --- cross-fit ------------------------------------------------------------------------------

def fit_one(values: np.ndarray, labels: np.ndarray, splits: np.ndarray, q: np.ndarray,
            arm: str, seed: int) -> CrossFit:
    full = seed in DEV_SEEDS or arm == "R0"
    metrics = dict(METRICS) if full else {"v_auc": video_auc}
    if ARMS.get(arm, Arm("", "")).kind == "G":
        metrics.pop("ns_auc", None)        # its negatives would be every snippet (PLAN.md F19)
    return crossfit_with_length(values, labels, splits, q, metrics)


@dataclass
class DatasetFits:
    dataset: str
    R: int
    splits_sha256: str
    fits: dict = field(default_factory=dict)     # (arm, readout) -> {FitKey: CrossFit | None}
    initial_r0: dict = field(default_factory=dict)   # FitKey -> CrossFit at the E010 R
    mc: dict = field(default_factory=dict)


def _fit_all(values: Mapping, labels: np.ndarray, splits: np.ndarray, q: np.ndarray) -> dict:
    out: dict = {}
    for (arm, ro), runs in values.items():
        out[(arm, ro)] = {k: (None if v is None else fit_one(v, labels, splits, q, arm, k[2]))
                          for k, v in runs.items()}
    return out


def _worst_mc(fits: Mapping, sd: Mapping[str, float]) -> float:
    worst = 0.0
    for (_arm, ro), runs in fits.items():
        if ro != PRIMARY:
            continue
        for (p, _, _), cf in runs.items():
            if cf is not None:
                worst = max(worst, cf.mc_se("v_auc") / sd[p])
    return worst


def fit_dataset(dataset: str, values: Mapping, labels: np.ndarray, q: np.ndarray,
                lineage: Mapping[str, Mapping]) -> DatasetFits:
    """Cross-fit every (arm, readout, run) of a dataset; the MC rule with each pipeline's
    lineage SD (``lineage[P][dataset]``); the R0 primary fits at the first R kept for (h)."""
    first = lineage[next(iter(lineage))][dataset]
    r0, sha0 = first["R"], first["splits_sha256"]
    splits = stratified_half_splits(labels, r0, seed=0)
    if splits_sha256(splits) != sha0:
        raise ValueError(f"{dataset}: splits differ from the lineage's {sha0}")
    out = DatasetFits(dataset, r0, sha0, _fit_all(values, labels, splits, q))
    out.initial_r0 = dict(out.fits.get(("R0", PRIMARY), {}))
    sd = {p: lineage[p][dataset]["sd"]["v_auc"] for p in lineage}
    worst = _worst_mc(out.fits, sd)
    out.mc = {"first_R": r0, "first_mc_se_over_sd": worst, "escalated": False}
    if worst > MC_LIMIT and r0 < R_LARGE:
        big = stratified_half_splits(labels, R_LARGE, seed=0)
        if splits_sha256(big[:r0]) != sha0:
            raise ValueError(f"{dataset}: the first splits of R = {R_LARGE} differ")
        out.fits = _fit_all(values, labels, big, q)
        out.R, out.splits_sha256 = R_LARGE, splits_sha256(big)
        out.mc |= {"escalated": True, "R": R_LARGE, "splits_sha256": out.splits_sha256,
                   "mc_se_over_sd": _worst_mc(out.fits, sd)}
        out.mc["exceeds_at_max_R"] = out.mc["mc_se_over_sd"] > MC_LIMIT     # recorded only
    elif worst > MC_LIMIT:
        out.mc["exceeds_at_max_R"] = True                                     # recorded only
    return out


# --- seed means, contrasts, z ---------------------------------------------------------------

def seed_values(df: DatasetFits, arm: str, readout_: str, pipeline: str, pair: str,
                metric: str, seeds: Sequence[int] = DEV_SEEDS) -> list[float | None]:
    runs = df.fits.get((arm, readout_), {})
    out = []
    for s in seeds:
        cf = runs.get((pipeline, pair, s))
        out.append(None if cf is None or metric not in cf.values else cf.mean(metric))
    return out


def seed_mean(df: DatasetFits, arm: str, readout_: str, pipeline: str, pair: str,
              metric: str) -> float | None:
    """Development-seed mean; None if a run failed or the arm has no fit there."""
    vals = seed_values(df, arm, readout_, pipeline, pair, metric)
    if not vals or any(v is None or not math.isfinite(v) for v in vals):
        return None
    return float(np.mean(vals))


def split_contrast(name: str) -> tuple[str, str]:
    x, y = name.split("-")
    return x, y


def contrast_delta(fits: Mapping[str, DatasetFits], name: str, readout_: str, pipeline: str,
                   metric: str) -> dict[tuple[str, str], float | None]:
    x, y = split_contrast(name)
    out = {}
    for d, p in SETTINGS:
        a = seed_mean(fits[d], x, readout_, pipeline, p, metric) if d in fits else None
        b = seed_mean(fits[d], y, readout_, pipeline, p, metric) if d in fits else None
        out[(d, p)] = None if a is None or b is None else a - b
    return out


def ls_thresholds(fits: Mapping[str, DatasetFits]) -> dict:
    """delta^LS_{P,d} per readout from R0's LS-AUC over its six seeds (PLAN.md "문턱")."""
    out = {}
    for ro in READOUTS:
        for p in PIPELINES:
            for d, df in fits.items():
                vals = {pair: seed_values(df, "R0", ro, p, pair, "ls_auc",
                                          DEV_SEEDS + CAL_SEEDS) for pair in PAIRS}
                out[(ro, p, d)] = ls_threshold(vals)
    return out


def threshold(lineage: Mapping, ls: Mapping, readout_: str, pipeline: str, dataset: str,
              metric: str) -> float | None:
    if metric == "ls_auc":
        return ls[(readout_, pipeline, dataset)]["delta"]
    return lineage[pipeline][dataset]["delta"][metric]


def tables(fits: Mapping[str, DatasetFits], lineage: Mapping, ls: Mapping,
           contrasts: Sequence[str], metrics: Sequence[str] = ALL_METRICS) -> tuple[dict, dict]:
    """Delta and z per (readout, pipeline, contrast, metric) and setting."""
    delta, z = {}, {}
    for ro in READOUTS:
        for p in PIPELINES:
            for c in contrasts:
                for m in metrics:
                    dl = contrast_delta(fits, c, ro, p, m)
                    delta[(ro, p, c, m)] = dl
                    z[(ro, p, c, m)] = {k: z_of(v, threshold(lineage, ls, ro, p, k[0], m))
                                        for k, v in dl.items()}
    return delta, z


def arm_sd(df: DatasetFits, arm: str, pipeline: str, metric: str = "v_auc",
           seeds: Sequence[int] | None = None) -> tuple[float, int]:
    """Pooled SD over the pairs of an arm's seeds (H0 / H1 / HB arms: development +
    calibration, df 10; R1r / Bm / Cm: development, df 4); failed runs left out."""
    seeds = seeds_of(arm) if seeds is None else seeds
    return pooled_sd([[v for v in seed_values(df, arm, PRIMARY, pipeline, pair, metric, seeds)
                       if v is not None and math.isfinite(v)] for pair in PAIRS])


def per_seed_deltas(df: DatasetFits, name: str, readout_: str, pipeline: str, pair: str,
                    metric: str = "v_auc") -> list[float | None]:
    x, y = split_contrast(name)
    a = seed_values(df, x, readout_, pipeline, pair, metric)
    b = seed_values(df, y, readout_, pipeline, pair, metric)
    return [None if u is None or v is None else u - v for u, v in zip(a, b, strict=True)]


def window_top_rows(sf: Mapping[str, np.ndarray], epoch: int) -> dict[str, np.ndarray]:
    """At one epoch: each video's top dense snippet (argmax of z, the logit readout of every
    snippet; earliest on ties) as a row index, and whether it is one of the video's d16c
    snippets (R1's / R1v's 32 units)."""
    z, counts, totals = sf["dense_z"][epoch], np.asarray(sf["dense_counts"]), sf["total_frames"]
    starts = snippet_rows(counts)
    top = np.array([starts[i] + int(np.argmax(z[starts[i]:starts[i] + counts[i]]))
                    for i in range(len(counts))])
    centres = centre_rows(totals, counts)
    return {"top_row": top, "top_in_d16c": np.array([top[i] in centres[i]
                                                     for i in range(len(top))])}
