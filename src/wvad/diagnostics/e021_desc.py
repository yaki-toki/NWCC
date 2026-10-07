"""E021 descriptives and interpretation flags (PLAN.md "기술 (판정 밖, 시간 주석 제외)" 1-10 and
"해석 표지"): written beside the verdict, never used for it, for a gate or for a later choice.
Local analysis code: no annotation file, neither the glance nor the UCA module.

Per run and arm, the checkpoint of the arm's own full-validation V-AUC (primary readout, earliest
on ties) gives the per-video quantities (length strata, length tertiles, ties at 1.0f, score
differences, top dense snippets); seed means over the development seeds unless noted.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from wvad.data import ENCODER_PAIRS
from wvad.diagnostics.e021_arms import (
    ARMS,
    DENSE_ARMS,
    DatasetFits,
    arm_sd,
    contrasts_of,
    full_epoch,
    per_seed_deltas,
    seed_values,
    threshold,
    window_top_rows,
)
from wvad.diagnostics.e021_gates import snippet_rows
from wvad.diagnostics.e021_rules import (
    LOGIT,
    PIPELINE_NAMES,
    PRIMARY,
    READOUTS,
    SETTINGS,
    any_abs_ge1,
    four_type,
    one_dataset_only,
    sd_ratio_flag,
    sign_mismatches,
)
from wvad.diagnostics.e021_score import DEV_SEEDS, PAIRS, PIPELINES
from wvad.eval.crossfit import (
    length_bin_parts,
    length_matched_auc,
    length_stratified_auc,
    rank_auc,
)
from wvad.extract import grid

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
HEAVY_REPLACED = 5                     # replaced frames of a snippet's 8 (PLAN.md, heavy snippets)
N_BATCHES = {"ucf": 42, "xdv": 104}    # E017 r_end: training batches per epoch (drop_last)
LR2 = 3e-5                             # stage-2 learning rate
CURVE_EPOCHS = (5, 10, 15, 20)
GAMMA_EPOCHS = (10, 20, 30)
NORM_FLAG = 0.05                       # length-shortcut flag: feature-norm V-AUC difference
ONE = np.float32(1.0)


def _mean(values: Sequence[float | None]) -> float | None:
    v = [x for x in values if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v and len(v) == len(values) else None


def _quantiles(x: np.ndarray) -> dict:
    if not len(x):
        return {}
    return {"mean": float(np.mean(x)), **{f"q{int(100 * q)}": float(np.quantile(x, q))
                                          for q in QUANTILES}}


def full_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return math.nan
    return float(rank_auc(np.asarray(scores, dtype=np.float64)[None], labels)[0])


# --- per run (arm, checkpoint of its own full-validation V-AUC) --------------------------------

def run_desc(arm: str, values: Mapping[str, np.ndarray], labels: np.ndarray, q: np.ndarray,
             totals: np.ndarray, sf: Mapping[str, np.ndarray] | None = None,
             dense: Mapping[str, np.ndarray] | None = None,
             replaced: np.ndarray | None = None) -> dict:
    """Per-video quantities of one run of an arm at its full-validation epoch (primary
    readout): video scores, ties at 1.0f, LS bin AUCs, length-matched AUC, score - log T
    Spearman within each class, V-AUC per length tertile; dense W arms also their top snippet
    (in the d16c set or not), heavily replaced top snippets and the M10 offsets."""
    prim = values[PRIMARY]
    e = full_epoch(prim, labels)
    s = prim[e].max(axis=1)
    out = {"epoch": e + 1, "scores": s, "logits": values[LOGIT][e].max(axis=1),
           "tied_videos": int((s == ONE).sum()),
           "tied_pairs": int((s[labels == 1] == ONE).sum() * (s[labels == 0] == ONE).sum())}
    parts = length_bin_parts(prim[e], labels, q)
    out["bin_auc"] = [float(w / n) if n else None for w, n in zip(parts["wins"], parts["pairs"],
                                                                  strict=True)]
    out["length_matched_auc"] = length_matched_auc(prim[e], labels, q)
    logt = np.log(totals.astype(np.float64))
    out["spearman_log_t"] = {c: float(spearmanr(s[labels == y], logt[labels == y])[0])
                             for c, y in (("normal", 0), ("anomalous", 1))}
    cuts = np.quantile(totals, [1 / 3, 2 / 3])
    tert = np.digitize(totals, cuts, right=True)
    out["tertile_v_auc"] = [full_auc(s[tert == t], labels[tert == t]) for t in range(3)]
    if arm in DENSE_ARMS and sf is not None and dense is not None:
        top = window_top_rows(sf, e)
        out["top_outside_d16c"] = {c: float(np.mean(~top["top_in_d16c"][labels == y]))
                                   for c, y in (("normal", 0), ("anomalous", 1))}
        if replaced is not None:
            out |= heavy_tops(sf, e, replaced, top["top_row"])
        if "c_W" in dense:
            own = sf[f"{own_variant(arm)}_c"][e]
            out["c_minus_32"] = {"W": _quantiles(dense["c_W"][e] - own),
                                 "Wm": _quantiles(dense["c_Wm"][e] - own)}
    return out


def own_variant(arm: str) -> str:
    """The 32-unit features of the dense arm's head (the c(v) compared in descriptive 4)."""
    return {"h0": "legacy", "h1": "d16c", "hb": "d16m", "hbm": "d16m2"}[ARMS[arm].head]


def heavy_tops(sf: Mapping[str, np.ndarray], epoch: int, replaced: np.ndarray,
               top_rows: np.ndarray) -> dict:
    """Snippets with >= 5 of 8 frames replaced: how many, how often one is a video's top
    snippet and how often a window's top snippet (argmax of z over J_k) at this epoch."""
    z, counts, totals = sf["dense_z"][epoch], np.asarray(sf["dense_counts"]), sf["total_frames"]
    heavy = np.asarray(replaced) >= HEAVY_REPLACED
    starts = snippet_rows(counts)
    window_tops = 0
    for i, t in enumerate(totals):
        zi, hi = z[starts[i]:starts[i] + counts[i]], heavy[starts[i]:starts[i] + counts[i]]
        if hi.any():
            sets = grid.window_members(int(t)).sets
            window_tops += int(sum(hi[s[np.argmax(zi[s])]] for s in sets))
    return {"heavy_snippets": int(heavy.sum()), "heavy_video_tops": int(heavy[top_rows].sum()),
            "heavy_window_tops": window_tops}


# --- per dataset ------------------------------------------------------------------------------

def length_only(labels: np.ndarray, totals: np.ndarray, q: np.ndarray) -> dict:
    """Descriptive 2 (i): the length alone (shorter = anomalous) as a score."""
    score = (-q)[:, None]
    return {"v_auc": full_auc(-totals.astype(np.float64), labels),
            "ls_auc": length_stratified_auc(score, labels, q),
            "length_matched_auc": length_matched_auc(score, labels, q)}


def norm_aucs(features: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
              labels: np.ndarray, dataset: str) -> dict:
    """Descriptive 2 (ii): V-AUC of each head input's per-video mean L2 norm."""
    out = {}
    for (d, p), f in features.items():
        if d != dataset:
            continue
        out[p] = {k[:-5]: full_auc(v, labels) for k, v in f.items() if k.endswith("_norm")}
    return out


def arm_summary(df: DatasetFits, descs: Mapping, arm: str, pipeline: str, pair: str,
                labels: np.ndarray) -> dict:
    """Descriptives 1, 2, 3, 6, 8 of one arm in one setting and pipeline."""
    out = {}
    for ro in READOUTS:
        runs = df.fits.get((arm, ro), {})
        out[ro] = {}
        for m in ("v_auc", "v_ap", "ns_auc", "v_pauc", "ls_auc"):
            vals = seed_values(df, arm, ro, pipeline, pair, m)
            out[ro][m] = {"mean": _mean(vals), "per_seed": vals}
        cfs = [runs.get((pipeline, pair, s)) for s in DEV_SEEDS]
        out[ro]["tie_rate"] = _mean([None if c is None else c.tie_rate for c in cfs])
        out[ro]["mc_se_v_auc"] = _mean([None if c is None else c.mc_se("v_auc") for c in cfs])
        out[ro]["selected_epochs"] = dict(sorted(Counter(
            int(x) + 1 for c in cfs if c is not None for x in c.epochs.ravel()).items()))
    for m in ("v_auc", "ls_auc"):
        out[f"sd_{m}"] = {"dev": arm_sd(df, arm, pipeline, m, DEV_SEEDS)[0],
                          "all_seeds": arm_sd(df, arm, pipeline, m)[0]}
    rd = [descs.get((arm, pipeline, pair, s)) for s in DEV_SEEDS]
    if all(r is not None for r in rd):
        out["full_validation_epochs"] = [r["epoch"] for r in rd]
        out["ties_1f"] = {"videos": _mean([r["tied_videos"] for r in rd]),
                          "pairs": _mean([r["tied_pairs"] for r in rd])}
        n_bins = len(rd[0]["bin_auc"])
        out["bin_auc"] = [_mean([r["bin_auc"][b] for r in rd]) for b in range(n_bins)]
        out["length_matched_auc"] = _mean([r["length_matched_auc"] for r in rd])
        out["spearman_log_t"] = {c: _mean([r["spearman_log_t"][c] for r in rd])
                                 for c in ("normal", "anomalous")}
        out["tertile_v_auc"] = [_mean([r["tertile_v_auc"][t] for r in rd]) for t in range(3)]
        for k in ("top_outside_d16c",):
            if k in rd[0]:
                out[k] = {c: _mean([r[k][c] for r in rd]) for c in ("normal", "anomalous")}
        for k in ("heavy_snippets", "heavy_video_tops", "heavy_window_tops"):
            if k in rd[0]:
                out[k] = [r[k] for r in rd]
        if "c_minus_32" in rd[0]:
            out["c_minus_32"] = [r["c_minus_32"] for r in rd]
    return out


def score_differences(descs: Mapping, x: str, y: str, pipeline: str, pair: str,
                      labels: np.ndarray) -> dict | None:
    """Descriptive 3: per validation video X's score minus Y's (each at its own full-validation
    epoch, primary readout), pooled over the development seeds, by class."""
    rows = [(descs.get((x, pipeline, pair, s)), descs.get((y, pipeline, pair, s)))
            for s in DEV_SEEDS]
    if any(a is None or b is None for a, b in rows):
        return None
    diff = np.stack([a["scores"].astype(np.float64) - b["scores"] for a, b in rows])
    return {c: _quantiles(diff[:, labels == yv].ravel()) for c, yv in (("normal", 0),
                                                                     ("anomalous", 1))}


def bin_delta(summaries: Mapping[str, dict], x: str, y: str) -> list[float | None] | None:
    """Descriptive 2: X - Y of the length-bin AUCs (each arm at its full-validation epoch,
    seed means)."""
    a, b = summaries[x].get("bin_auc"), summaries[y].get("bin_auc")
    if a is None or b is None:
        return None
    return [None if u is None or v is None else u - v for u, v in zip(a, b, strict=True)]


def dataset_descriptives(dataset: str, df: DatasetFits, descs: Mapping, arms: Sequence[str],
                         labels: np.ndarray, q: np.ndarray, totals: np.ndarray,
                         features: Mapping, decomposition: Mapping) -> dict:
    out = {"length_only": length_only(labels, totals, q),
           "feature_norm_v_auc": norm_aucs(features, labels, dataset), "settings": {}}
    for p in PIPELINES:
        for pair in PAIRS:
            k = f"{dataset}/{pair}/{p}"
            entry = {"arms": {a: arm_summary(df, descs, a, p, pair, labels) for a in arms}}
            entry["score_differences"] = {f"{x}-{y}": score_differences(descs, x, y, p, pair,
                                                                         labels)
                                          for x, y in (("A0", "R1v"), ("C", "R1")) if x in arms}
            entry["bin_delta"] = {f"{x}-{y}": bin_delta(entry["arms"], x, y)
                                  for x, y in (("A0", "R1v"), ("A0", "R0"), ("C", "R1"),
                                               ("C", "R0")) if x in arms}
            if p == "m10":
                entry["readout_decomposition"] = decomposition.get((dataset, pair), {})
            out["settings"][k] = entry
    return out


# --- training ---------------------------------------------------------------------------------

def r_end(gamma2: Sequence[float], dataset: str, mult: float) -> float:
    """E017 descriptive: mean gamma change over stage-2 epochs 26-30 as a fraction of the most
    Adam can move gamma per epoch, times sign(final gamma)."""
    per_epoch = (gamma2[-1] - gamma2[-6]) / 5
    return float(np.sign(gamma2[-1])) * per_epoch / (N_BATCHES[dataset] * LR2 * mult)


def training(runs: Mapping, labels: Mapping[str, np.ndarray], lineage: Mapping) -> dict:
    """Descriptive 6: stage-1 curves, the convergence-confound inputs (stage-1 rise from epoch
    15 to 20 minus h0's, in delta_{P,d}), GradScaler-skipped steps, M10 gamma and r_end."""
    out, rise = {}, {}
    for key, run in runs.items():
        if run.failed:
            continue
        lab = labels[key.dataset]
        curve = {e: full_auc(run.s1[e - 1].max(axis=1), lab) for e in CURVE_EPOCHS}
        rise[key] = curve[20] - curve[15]
        row = {"stage1_curve": curve,
               "amp_skipped_steps": int(sum(h.get("amp_skipped_steps", 0)
                                            for h in (*run.history1, *run.history2)))}
        if PIPELINES[key.pipeline][0] == "mean":
            g2 = [h["offset_gamma"] for h in run.history2]
            row["gamma"] = {"stage1_end": run.history1[-1]["offset_gamma"],
                            **{f"stage2_{e}": g2[e - 1] for e in GAMMA_EPOCHS}}
            row["r_end"] = r_end(g2, key.dataset, PIPELINES[key.pipeline][1])
        out[f"{key.chain}/{key.run}"] = row
    conv = {}
    for key, v in rise.items():
        if key.head == "h0" or key.seed not in DEV_SEEDS:
            continue
        base = rise.get(key._replace(head="h0"))
        if base is None:
            continue
        conv.setdefault((key.pipeline, key.dataset, key.pair, key.head), []).append(v - base)
    confound = {}
    for (p, d, pair, h), v in conv.items():
        if len(v) == len(DEV_SEEDS):
            x = float(np.mean(v)) / lineage[p][d]["delta"]["v_auc"]
            confound[f"{d}/{pair}/{p}/{h}"] = {"rise_difference_over_delta": x,
                                               "confound": abs(x) >= 1}
    return {"runs": out, "convergence": confound}


def feature_stats(features: Mapping) -> dict:
    """Descriptive 7: mean norm and mean per-dimension variance of every head input, the
    scale-match ratios, and C's training (window means) vs scoring (single snippets) spread."""
    out = {}
    for (d, p), f in features.items():
        e = {k[:-5]: {"norm_mean": float(np.mean(v))} for k, v in f.items() if k.endswith("_norm")}
        for k, v in f.items():
            if k.endswith("_dimvar"):
                e.setdefault(k[:-7], {})["dimvar_mean"] = float(np.mean(v))
            if k.startswith("scale_ratio_"):
                e[k] = {"min": float(np.min(v)), "median": float(np.median(v)),
                        "max": float(np.max(v))}
        for var in ("d16m", "d16m2"):
            if f"{var}_dimvar" in f and f"dense_{var}_dimvar" in f:
                ratio = f[f"dense_{var}_dimvar"] / np.where(f[f"{var}_dimvar"] > 0,
                                                            f[f"{var}_dimvar"], np.nan)
                e[f"snippet_over_window_mean_variance_{var}"] = _quantiles(
                    ratio[np.isfinite(ratio)])
        out[f"{d}/{p}"] = e
    return out


def extraction(record: Mapping, features: Mapping) -> dict:
    """Descriptive 5: the extraction gate summaries (errors, bit-equal shares, floors, frame-hash
    mismatches, throughput) and, from the dense index, grab and replacement counts."""
    keep = {k: v for k, v in record.items() if not k.startswith("x5_")}
    index = {}
    for (d, _p), f in features.items():
        if "index_grab_attempts" not in f or d in index:
            continue
        done = np.asarray(f["index_done"], dtype=bool)
        used = np.maximum(f["index_used_frames"][done], 1)
        index[d] = {"videos_done": int(done.sum()),
                    "grab_failures": int((f["index_grab_attempts"][done]
                                          - f["index_grab_successes"][done]).sum()),
                    "replaced_share": _quantiles(f["index_replaced"][done] / used),
                    "header_minus_manifest": _quantiles(
                        (f["index_header_frames"][done] - f["index_total_frames"][done]).astype(
                            np.float64)),
                    "heavy_validation_snippets": int((f["snippet_replaced"]
                                                      >= HEAVY_REPLACED).sum())}
    return {"gate_record": keep, "dense_index": index}


# --- contrasts and flags --------------------------------------------------------------------

def contrast_summary(fits: Mapping[str, DatasetFits], delta: Mapping, z: Mapping,
                     stage: int, lineage: Mapping, ls: Mapping) -> dict:
    """Descriptives 1, 9, 10: z, z^LS, four-type flags and the paired SD of per-seed Delta (in
    delta units) of every contrast; for the verdict contrasts the margin to the floor z > -2."""
    out = {}
    for c in contrasts_of(stage):
        for ro in READOUTS:
            for p in PIPELINE_NAMES:
                e = {"z": {f"{d}/{pr}": z[(ro, p, c, "v_auc")][(d, pr)] for d, pr in SETTINGS},
                     "z_ls": {f"{d}/{pr}": z[(ro, p, c, "ls_auc")][(d, pr)]
                              for d, pr in SETTINGS},
                     "type": four_type(z[(ro, p, c, "v_auc")]),
                     "type_ls": four_type(z[(ro, p, c, "ls_auc")]),
                     "aux_z": {m: {f"{d}/{pr}": z[(ro, p, c, m)][(d, pr)] for d, pr in SETTINGS}
                               for m in ("v_ap", "ns_auc", "v_pauc")}}
                paired = {}
                for d, pr in SETTINGS:
                    if d not in fits:
                        continue
                    per = per_seed_deltas(fits[d], c, ro, p, pr)
                    zv = z[(ro, p, c, "v_auc")][(d, pr)]
                    thr = threshold(lineage, ls, ro, p, d, "v_auc")
                    live = [x for x in per if x is not None]
                    paired[f"{d}/{pr}"] = {
                        "per_seed_delta": per,
                        "paired_sd_over_delta": (float(np.std(live, ddof=1)) / thr
                                                 if len(live) == len(per) else None),
                        "margin_to_floor": None if zv is None else zv + 2}
                e["paired"] = paired
                out[f"{c}/{ro}/{p}"] = e
    draws = {}
    for p in PIPELINE_NAMES:
        for pr in PAIRS:
            zs = {r: z[(PRIMARY, p, f"{a}-R0", "v_auc")][("ucf", pr)]
                  for r, a in ((0, "R1rv"), (1, "R1rv1"), (2, "R1rv2"))}
            zls = {r: z[(PRIMARY, p, f"{a}-R0", "ls_auc")][("ucf", pr)]
                   for r, a in ((0, "R1rv"), (1, "R1rv1"), (2, "R1rv2"))}
            live = [v for v in zs.values() if v is not None]
            draws[f"ucf/{pr}/{p}"] = {"z": zs, "z_ls": zls,
                                      "range": (max(live) - min(live)) if live else None}
    return {"contrasts": out, "ucf_draw_effect": draws}


def delta_validity(fits: Mapping[str, DatasetFits], lineage: Mapping,
                   arms: Sequence[str]) -> dict:
    """Arm SD / lineage SD (V-AUC) per pipeline and dataset; flagged outside [1/1.5, 1.5]."""
    out = {}
    for d, df in fits.items():
        for p in PIPELINE_NAMES:
            for a in arms:
                sd, dof = arm_sd(df, a, p)
                ratio = sd / lineage[p][d]["sd"]["v_auc"] if dof else None
                out[f"{a}/{p}/{d}"] = {"sd": sd, "df": dof, "ratio": ratio,
                                       "flag": sd_ratio_flag(ratio)}
    return out


def interpretation_flags(stage: int, z: Mapping, delta: Mapping, fits: Mapping,
                         lineage: Mapping, norm_auc: Mapping, train: Mapping,
                         approx: Sequence[tuple[str, str]], per_pipeline: Mapping,
                         arms: Sequence[str]) -> dict:
    """PLAN.md "해석 표지" (primary readout, descriptive; the verdict is unchanged)."""
    main, ref = ("A0-R1v", "A0-R0") if stage == 1 else ("C-R1", "C-R0")
    effect, split_c, count_c = (("R1v-R0", ("R1rv-R0", "R1v-R1rv"), "A0-A0m") if stage == 1 else
                                ("R1-R0", ("R1r-R0", "R1-R1r"), "C-Cm"))
    flags: dict = {}
    approx_settings = {(d, p) for d, e in approx for p in PAIRS if e in ENCODER_PAIRS[p]}
    flags["reextraction_effect"] = {}
    for p in PIPELINE_NAMES:
        ze = z[(PRIMARY, p, effect, "v_auc")]
        flags["reextraction_effect"][p] = {
            "flag": any_abs_ge1(ze),
            "decomposition": {c: {f"{d}/{pr}": ("reextraction (approximate equivalence)"
                                                 if (d, pr) in approx_settings else
                                                 z[(PRIMARY, p, c, "v_auc")][(d, pr)])
                                  for d, pr in SETTINGS} for c in split_c}}
    flags["count_dependent"] = {p: any_abs_ge1(z[(PRIMARY, p, count_c, "v_auc")])
                                for p in PIPELINE_NAMES}
    flags["one_dataset_only"] = {p: one_dataset_only(z[(PRIMARY, p, main, "v_auc")])
                                 for p in PIPELINE_NAMES}
    mism = {f"{c}/{p}": [f"{d}/{pr}" for d, pr in sign_mismatches(
        delta[(PRIMARY, p, c, "v_auc")], delta[(PRIMARY, p, c, "ls_auc")])]
        for c in (main, ref) for p in PIPELINE_NAMES}
    norm_diff = {}
    if stage == 2:
        for d, per_pair in norm_auc.items():
            for pr, v in per_pair.items():
                for var in ("d16m", "d16m2"):
                    if var in v and "d16c" in v:
                        norm_diff[f"{d}/{pr}/{var}"] = v[var] - v["d16c"]
    flags["length_shortcut_possible"] = {
        "flag": any(abs(x) >= NORM_FLAG for x in norm_diff.values() if x == x)
        or any(len(v) >= 2 for v in mism.values()),
        "norm_v_auc_minus_r1": norm_diff, "sign_mismatch_settings": mism}
    flags["delta_validity"] = delta_validity(fits, lineage, arms)
    if stage == 2:
        flags["video_level_adopt"] = {p: four_type(z[(PRIMARY, p, main, "v_auc")]) == "adopt"
                                      and four_type(z[(PRIMARY, p, main, "ls_auc")]) == "adopt"
                                      for p in PIPELINE_NAMES}
        flags["convergence_confound"] = {k: v["confound"] for k, v in
                                         train["convergence"].items()}
    flags["pipeline_dependent"] = {"categories": dict(per_pipeline),
                                   "flag": len(set(per_pipeline.values())) > 1}
    return flags

