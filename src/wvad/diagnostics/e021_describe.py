"""E021 temporal-annotation diagnostics (PLAN.md "시간 주석 진단"; descriptive only, user decision
3): computed once by ``python -m wvad.diagnostics.e021 describe`` after the verdict file is
committed, never used for the verdict, a gate, a selection, a flag or a later plan's choices.
This is the only E021 module that reads the GlanceVAD points (wvad.diagnostics.glance) and the
UCA proxies (wvad.diagnostics.uca); the verdict code imports neither.

Frame scores (logits, no saturation; M10 z + c(v)) of each arm's checkpoint: the 32-unit arms
R0, R1, R1r, B and Bm use the training's ckpt_best_val.pt (selection.json), every other arm the
epoch of its readout's full-validation V-AUC (earliest on ties). 32-unit arms (R0, R1v, R1rv, R1,
R1r, B, Bm): unit logits over the evaluator windows; W / Wm arms (A0, A0m, A1, C, Cm): window
logits over the windows; snippet resolution S (A0.S, A1.S, C.S, the W arm's epoch): snippet
logits z_j + c(v) over their supports.

GlanceVAD (both datasets): GN-AUC, GP (point and video mean), Hit@32/128/512 with the chance
level (wvad.diagnostics.glance). UCA (UCF; M1A, M1B, M2A, M2B): per-video frame AP and AUC (video
means, the main statistics), the pooled AP of the anomalous videos and AP / AUC of all
validation videos, the "event stands out" rate by event length over the window length L (shape A
connected components; < 0.5, 0.5-1, 1-2, >= 2) and the share of event frames the encoder sees.
Contrasts: seed-paired X - Y with 95% bootstrap intervals (2,000 resamples of the anomalous and
the normal videos and of the development seeds, default_rng(0)); null scale: the SD of the same
statistic over h0's six seeds (development + calibration).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from wvad.diagnostics import e021_arms as arms
from wvad.diagnostics.e021_gates import dense_failure, load_run, snippet_rows, val_split
from wvad.diagnostics.e021_rules import PRIMARY
from wvad.diagnostics.e021_score import (
    CAL_SEEDS,
    DATASETS,
    DEV_SEEDS,
    PAIRS,
    PIPELINES,
    RunKey,
    score_path,
)
from wvad.diagnostics.glance import (
    STATISTICS,
    UnitScores,
    bootstrap_draws,
    contrast,
    contrasts,
    hit_chance,
    load_glance,
    run_statistics,
    seed_summary,
)
from wvad.diagnostics.proxy_eval import support_frame_counts, weighted_ap, weighted_auc
from wvad.diagnostics.uca import VARIANTS, load_proxy
from wvad.eval.dense_readout import readout
from wvad.extract import grid
from wvad.paths import REPO_ROOT

PROXY_FILE = REPO_ROOT / "data" / "annotations" / "ucf_train_uca_proxy.json"
STAGE_ARMS = {1: ("R0", "R1v", "R1rv", "A0", "A0m", "A0.S"),
              2: ("R1", "R1r", "B", "Bm", "A1", "A1.S", "C", "Cm", "C.S")}
CONTRASTS = {1: ("A0-R1v", "A0-R0", "R1v-R0", "A0m-R1v", "A0-A0m", "A0.S-R1v", "A0.S-A0"),
             2: ("C-R1", "C-R0", "A1-R1", "C-Cm", "C-B", "R1-R0", "R1r-R0", "R1-R1r", "A1-A0",
                 "C.S-R1", "C.S-C", "A1.S-R1")}
EVENT_BINS = (0.5, 1.0, 2.0)                    # event length / L: <0.5, 0.5-1, 1-2, >=2
UCA_STATS = ("video_ap", "video_auc")


def base_arm(arm: str) -> str:
    return arm[:-2] if arm.endswith(".S") else arm


def head_of(arm: str) -> str:
    return arms.ARMS[base_arm(arm)].head


def window_counts(total_frames: int) -> np.ndarray:
    """Frames of the 32 evaluator windows (T // 32 each, the last also T % 32)."""
    c = np.full(grid.WINDOWS, total_frames // grid.WINDOWS, dtype=np.int64)
    c[-1] += total_frames % grid.WINDOWS
    return c


def arm_frames(arm: str, run, sf: Mapping[str, np.ndarray], dense: Mapping | None,
               labels: np.ndarray) -> tuple[int, list[UnitScores]]:
    """The arm's checkpoint epoch (1-based) and its frame scores per validation video."""
    a = arms.ARMS[base_arm(arm)]
    totals, n = sf["total_frames"], len(sf["total_frames"])
    if a.kind == "record":
        e = int(run.selection["best_epoch"]) - 1
    else:
        e = arms.full_epoch(arms.arm_values(base_arm(arm), run, sf, dense)[PRIMARY], labels)
    if a.kind in ("record", "unit"):
        logits = sf[f"{a.variant}_logits"][e]
        return e + 1, [UnitScores.from_counts(logits[i], window_counts(int(totals[i])))
                       for i in range(n)]
    gamma = None if "gamma" not in sf else float(sf["gamma"][e])
    counts = np.asarray(sf["dense_counts"])
    starts = snippet_rows(counts)
    frames = []
    for i in range(n):
        z, t = sf["dense_z"][e, starts[i]:starts[i] + counts[i]], int(totals[i])
        if arm.endswith(".S"):
            sup = grid.snippet_support(t)
            frames.append(UnitScores.from_supports(readout(z, t, gamma, "G").logits,
                                                   sup[:, 0], sup[:, 1]))
        else:
            frames.append(UnitScores.from_counts(readout(z, t, gamma, a.kind).logits,
                                                 window_counts(t)))
    return e + 1, frames


# --- UCA ------------------------------------------------------------------------------------

class UcaSet:
    """One proxy variant on the UCF validation videos: eligible anomalous videos (with a proxy
    interval) and the normal videos, their frame labels and events (shape A components)."""

    def __init__(self, proxy, variant: str, names: Sequence[str], labels: np.ndarray,
                 totals: np.ndarray):
        self.variant = variant
        self.anomalous = [i for i in range(len(names)) if labels[i] == 1
                          and proxy.has(names[i], variant)]
        self.normal = [i for i in range(len(names)) if labels[i] == 0]
        self.frame_labels = {i: proxy.frame_labels(names[i], variant, int(totals[i]))
                             for i in self.anomalous}
        self.events = {i: [tuple(iv) for iv in proxy.intervals(names[i], variant)]
                       for i in self.anomalous} if variant.endswith("A") else {}
        self.totals = totals

    def seen_share(self, seen: Mapping[int, np.ndarray]) -> float:
        """Share of the event frames that are encoder input frames (``seen[i]``: frames)."""
        hit = tot = 0
        for i in self.anomalous:
            lab = self.frame_labels[i]
            hit += int(lab[seen[i]].sum())
            tot += int(lab.sum())
        return hit / tot if tot else math.nan


def _counts(u: UnitScores, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return support_frame_counts(lab, u.bounds[:-1], u.bounds[1:])


def uca_statistics(us: UcaSet, frames: Sequence[UnitScores], draws=None) -> dict:
    """Per-video AP and AUC means (fractions; bootstrap rows when ``draws``), pooled AP / AUC and
    the event standout rate per length bin (row 0 only for the pooled values)."""
    ap, auc = [], []
    for i in us.anomalous:
        pos, neg = _counts(frames[i], us.frame_labels[i])
        ap.append(weighted_ap(frames[i].values, pos, neg) / 100)
        auc.append(weighted_auc(frames[i].values, pos, neg) / 100 if neg.sum() > 0 else math.nan)
    w = (np.ones((1, len(us.anomalous))) if draws is None
         else draws.anomalous.astype(np.float64))
    out = {}
    for k, v in (("video_ap", np.array(ap)), ("video_auc", np.array(auc))):
        ok = np.isfinite(v)
        out[k] = (w[:, ok] @ v[ok]) / w[:, ok].sum(axis=1)
    vals = [frames[i].values for i in us.anomalous]
    cnt = [_counts(frames[i], us.frame_labels[i]) for i in us.anomalous]
    out["pooled_anomalous_ap"] = weighted_ap(np.concatenate(vals), np.concatenate(
        [c[0] for c in cnt]), np.concatenate([c[1] for c in cnt])) / 100
    nv = [frames[i].values for i in us.normal]
    nn = [np.diff(frames[i].bounds).astype(np.float64) for i in us.normal]
    s = np.concatenate(vals + nv)
    pos = np.concatenate([c[0] for c in cnt] + [np.zeros_like(x) for x in nn])
    neg = np.concatenate([c[1] for c in cnt] + nn)
    out["pooled_all_ap"], out["pooled_all_auc"] = (weighted_ap(s, pos, neg) / 100,
                                                   weighted_auc(s, pos, neg) / 100)
    if us.events:
        out["standout_by_length"] = standout(us, frames)
    return out


def standout(us: UcaSet, frames: Sequence[UnitScores]) -> list[dict]:
    """Per event-length bin (event / L): events whose highest frame score exceeds the highest
    frame score outside the event in the same video (videos fully covered are skipped)."""
    bins = [[0, 0] for _ in range(len(EVENT_BINS) + 1)]
    for i in us.anomalous:
        t = int(us.totals[i])
        f = frames[i].at(np.arange(t))
        for a, b in us.events.get(i, []):
            outside = np.r_[f[:a], f[b:]]
            if not len(outside):
                continue
            k = int(np.searchsorted(EVENT_BINS, (b - a) / (t // grid.WINDOWS), side="right"))
            bins[k][1] += 1
            bins[k][0] += int(f[a:b].max() > outside.max())
    return [{"events": n, "standout": s, "rate": (s / n if n else None)} for s, n in bins]


# --- runs -----------------------------------------------------------------------------------

def _run_inputs(runs_root: Path, scores: Path, key: RunKey):
    run = load_run(runs_root, key)
    with np.load(score_path(scores, key.stage, key)) as z:
        sf = {k: z[k] for k in z.files}
    run.failed = run.failed or dense_failure(sf)        # as the verdict's gates (PLAN.md 수치 실패)
    dense = arms.dense_readouts(sf) if "dense_z" in sf and not run.failed else None
    return run, sf, dense


def collect(runs_root: Path, scores: Path, dataset: str, arm_names: Sequence[str],
            seeds: Sequence[int]) -> dict:
    """Frame scores per (arm, pipeline, pair, seed) of the listed seeds (failed runs left out)."""
    labels = val_split(dataset)[1]
    out = {}
    for p in PIPELINES:
        for pair in PAIRS:
            for head in sorted({head_of(a) for a in arm_names}):
                for s in seeds:
                    if s not in arms.HEAD_SEEDS[head]:
                        continue
                    key = RunKey(dataset, pair, s, head, p)
                    run, sf, dense = _run_inputs(runs_root, scores, key)
                    if run.failed:
                        continue
                    for a in arm_names:
                        if head_of(a) == head:
                            out[(a, p, pair, s)] = arm_frames(a, run, sf, dense, labels)
    return out


def describe_dataset(runs_root: Path, scores: Path, dataset: str, stage2: bool) -> dict:
    gs = load_glance(dataset)
    val_idx, labels, _ = val_split(dataset)
    if not np.array_equal(gs.val_indices, val_idx):
        raise ValueError(f"{dataset}: glance videos are not the validation videos in order")
    names = [*STAGE_ARMS[1], *(STAGE_ARMS[2] if stage2 else ())]
    frames = collect(runs_root, scores, dataset, names, DEV_SEEDS)
    h0 = [a for a in names if head_of(a) == "h0"]
    null = collect(runs_root, scores, dataset, h0, CAL_SEEDS) | {
        k: v for k, v in frames.items() if k[0] in h0}
    draws = bootstrap_draws(len(gs.anomalous), len(gs.normal), len(DEV_SEEDS))
    out = {"coverage": gs.coverage(), "chance": {k: float(v[0]) for k, v in
                                                 hit_chance(gs).items()}, "glance": {}}
    uca = None
    if dataset == "ucf":
        proxy = load_proxy(PROXY_FILE)
        uca = {v: UcaSet(proxy, v, list(gs.names), labels, gs.total_frames) for v in VARIANTS}
        out["uca"] = {"videos": {v: len(u.anomalous) for v, u in uca.items()},
                      "seen_share": seen_shares(uca, gs.total_frames), "settings": {}}
    contrast_names = [*CONTRASTS[1], *(CONTRASTS[2] if stage2 else ())]
    for p in PIPELINES:
        for pair in PAIRS:
            k = f"{dataset}/{pair}/{p}"
            stats = {a: [run_statistics(gs, frames[(a, p, pair, s)][1], draws)
                         for s in DEV_SEEDS] for a in names
                     if all((a, p, pair, s) in frames for s in DEV_SEEDS)}
            entry = {"epochs": {a: [frames[(a, p, pair, s)][0] for s in DEV_SEEDS]
                                for a in stats},
                     "statistics": {a: {m: seed_summary({s: float(x[m][0]) for s, x in
                                                         zip(DEV_SEEDS, v, strict=True)})
                                        for m in STATISTICS} for a, v in stats.items()},
                     "null_scale_h0": null_scale(gs, null, h0, p, pair),
                     "contrasts": {c: contrasts(stats[c.split("-")[0]], stats[c.split("-")[1]],
                                                draws)
                                   for c in contrast_names
                                   if all(x in stats for x in c.split("-"))}}
            out["glance"][k] = entry
            if uca is not None:
                out["uca"]["settings"][k] = uca_setting(uca, frames, null, names, h0, p, pair,
                                                        contrast_names)
    return out


def null_scale(gs, null: Mapping, h0: Sequence[str], p: str, pair: str) -> dict:
    """SD of each statistic over h0's six seeds (the null-scale reference)."""
    out = {}
    for a in h0:
        rows = {s: run_statistics(gs, null[(a, p, pair, s)][1]) for s in (*DEV_SEEDS, *CAL_SEEDS)
                if (a, p, pair, s) in null}
        out[a] = {m: seed_summary({s: float(r[m][0]) for s, r in rows.items()})["sd"]
                  for m in STATISTICS}
    return out


def seen_shares(uca: Mapping[str, UcaSet], totals: np.ndarray) -> dict:
    """Event frames the encoder reads: every snippet (dense, about every other frame) and the
    32 d16c snippets (R1v / R1); the earlier UCF clips' positions are unknown (E002 C2)."""
    dense = {i: grid.used_frames(int(t)) for i, t in enumerate(totals)}
    centre = {i: np.unique(grid.snippet_frames(int(t))[grid.centre_snippets(int(t))])
              for i, t in enumerate(totals)}
    return {v: {"dense": u.seen_share(dense), "d16c": u.seen_share(centre)}
            for v, u in uca.items()}


def uca_setting(uca: Mapping[str, UcaSet], frames: Mapping, null: Mapping,
                names: Sequence[str], h0: Sequence[str], p: str, pair: str,
                contrast_names: Sequence[str]) -> dict:
    out = {}
    for v, us in uca.items():
        draws = bootstrap_draws(len(us.anomalous), len(us.normal), len(DEV_SEEDS))
        stats = {a: [uca_statistics(us, frames[(a, p, pair, s)][1], draws) for s in DEV_SEEDS]
                 for a in names if all((a, p, pair, s) in frames for s in DEV_SEEDS)}
        point = {a: {m: seed_summary({s: float(x[m][0] if m in UCA_STATS else x[m])
                                      for s, x in zip(DEV_SEEDS, r, strict=True)})
                     for m in (*UCA_STATS, "pooled_anomalous_ap", "pooled_all_ap",
                               "pooled_all_auc")} for a, r in stats.items()}
        standout_rates = {a: [x.get("standout_by_length") for x in r] for a, r in stats.items()}
        nulls = {}
        for a in h0:
            rows = {s: uca_statistics(us, null[(a, p, pair, s)][1])
                    for s in (*DEV_SEEDS, *CAL_SEEDS) if (a, p, pair, s) in null}
            nulls[a] = {m: seed_summary({s: float(r[m][0]) for s, r in rows.items()})["sd"]
                        for m in UCA_STATS}
        out[v] = {"statistics": point, "standout_by_length": standout_rates,
                  "null_scale_h0": nulls,
                  "contrasts": {c: {m: contrast([x[m] for x in stats[c.split("-")[0]]],
                                                [x[m] for x in stats[c.split("-")[1]]], draws)
                                    for m in UCA_STATS}
                                for c in contrast_names
                                if all(x in stats for x in c.split("-"))}}
    return out


def describe(runs_root: Path, scores: Path, stage2: bool) -> dict:
    """Every dataset's glance (and, for UCF, UCA) diagnostics of the arms that were run."""
    return {"stage2_run": stage2, "datasets": {d: describe_dataset(runs_root, scores, d, stage2)
                                               for d in DATASETS}}
