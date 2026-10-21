"""E023 descriptives (PLAN.md "기술 (판정 밖)" 1-12, "문제 기술", "D14 기술"): computed after the
verdict and written beside it, never used for condition (2), condition (3), a flag or a later
choice. Descriptive-only code (PLAN.md "코드 규칙" 4); pure functions of the analysis's per-run
values. New code (add-only rule).

    1   per arm / setting / readout / metric (L, pooled, AUC_A or AP_A, the opposite within-video
        metric, video AUC): mean, seed SD, the seed values; d_z of arm - C per setting (common
        seeds; C.last for the .last rules)
    2   problem description: per run (pooled AUC, W_pair, X, w, L, AUC_A); per method pair the
        pooled-AUC difference split into w dW_pair + (1 - w) dX with the cross share; per dataset
        the pairs whose pooled and L order differ and Kendall's tau of the two orders
    3   mechanism: C, C.L20 (and .last) pooled metric, W_pair, X, the z-only readout of C, the
        Spearman correlation of C's c(v) with log T within each class
    4   per setting the Spearman correlation of the selected validation V-AUC with test L and the
        pooled metric
    5   the selected evaluation points; the D4 choices (with the widened values)
    6   C - B_k per test-length tertile of the included anomalous videos; L without the videos
        cut into N_chunk pieces
    7   efficiency: trainable parameters, seconds per chain, test-scoring seconds
    8   sigma ties at 1.0f per arm (snippet values)
    9   the (h2) table
    10  (3f) / (3g) estimates against the actual 99% L bootstrap half-widths
    11  D14 (comparison methods only): test-selection inflation (best of the 30 points - the
        validation-selected point, L and pooled), validation-test Spearman over the points, the
        method order per point against the selected order
    12  condition (3)'s 98% Welch lower bounds

Single-seed mode (PLAN.md "단일 seed 실행과 판정"; ``single``): the per-run inputs hold each
chain's chosen seed only (the pairs and arms may hold different seeds), so the arm - C pairings
of 1 and 2 pair on (dataset, pair) with each arm's own chosen seed and record the seeds used;
the non-chosen attempts that still have values are listed in 1 under "non_chosen" (marked, used
by nothing else).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

DATASETS = ("ucf", "xdv")
PAIRS = ("xcb_xcl", "xcb_siglip")
PRIMARY = {"ucf": "frame_auc", "xdv": "frame_ap"}
ANOMALOUS = {"ucf": "anom_frame_auc", "xdv": "anom_frame_ap"}
METRICS = ("L", "frame_auc", "frame_ap", "anom_frame_auc", "anom_frame_ap", "L_opposite",
           "video_auc")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def mean_sd(v: Sequence[float]) -> dict:
    x = [float(a) for a in v if a is not None and math.isfinite(float(a))]
    return {"n": len(x), "mean": float(np.mean(x)) if x else math.nan,
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else math.nan, "values": x}


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return math.nan
    return float(spearmanr(a[keep], b[keep])[0])


def quantiles(x: Sequence[float]) -> dict:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            **{f"q{int(round(100 * q))}": float(np.quantile(a, q)) for q in QUANTILES}}


def _base(arm: str) -> str:
    return "C.last" if arm.endswith(".last") else "C"


# --- 1 ---------------------------------------------------------------------------------------

def cell_table(metrics: Mapping[tuple, Mapping], single: bool = False,
               non_chosen: Mapping[tuple, Mapping] | None = None) -> dict:
    """Descriptive 1 (``metrics``: (dataset, pair, seed, arm, readout) -> metric values;
    ``single``: pair an arm with C on (dataset, pair), each its own chosen seed, the seeds
    recorded; ``non_chosen``: the non-chosen attempts' values, listed and marked)."""
    groups: dict[str, list] = {}
    by: dict[tuple, float] = {}
    for (d, p, s, a, ro), m in metrics.items():
        for name in METRICS:
            if name in m:
                groups.setdefault(f"{d}/{p}/{a}/{ro}/{name}", []).append(m[name])
        if "L" in m:
            by[(d, p, s, a, ro)] = m["L"]
    d_z = {}
    for (d, p, _s, a, ro) in {(k[0], k[1], None, k[3], k[4]) for k in by}:
        base = _base(a)
        if a == base:
            continue
        if single:
            mine = [s for (dd, pp, s, aa, rr) in by if (dd, pp, aa, rr) == (d, p, a, ro)]
            theirs = [s for (dd, pp, s, aa, rr) in by if (dd, pp, aa, rr) == (d, p, base, "logit")]
            pairs = [(sa, sb) for sa in mine for sb in theirs]
        else:
            pairs = [(s, s) for (dd, pp, s, aa, rr) in by if (dd, pp, aa, rr) == (d, p, a, ro)
                     and (d, p, s, base, "logit") in by]
        diffs = [by[(d, p, sa, a, ro)] - by[(d, p, sb, base, "logit")] for sa, sb in pairs]
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else math.nan
        d_z[f"{d}/{p}/{a}/{ro}"] = {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs
                                    else math.nan,
                                    "d_z": float(np.mean(diffs)) / sd if sd and math.isfinite(sd)
                                    else math.nan,
                                    "seeds": [[sa, sb] for sa, sb in pairs]}
    out = {"cells": {k: mean_sd(v) for k, v in sorted(groups.items())},
           "d_z_vs_c": dict(sorted(d_z.items()))}
    if non_chosen is not None:
        out["non_chosen"] = {f"{d}/{p}/s{s}/{a}/{ro}": {"chosen": False, **{
            n: m[n] for n in METRICS if n in m}}
            for (d, p, s, a, ro), m in sorted(non_chosen.items(), key=str)}
    return out


# --- 2 problem description -------------------------------------------------------------------

def problem(decomp: Mapping[tuple, Mapping], metrics: Mapping[tuple, Mapping],
            arms: Sequence[str], single: bool = False) -> dict:
    """Descriptive 2 (``decomp``: (dataset, pair, seed, arm) -> Decomposition.to_json();
    ``single``: a method pair is paired on (dataset, pair), each arm its own chosen seed, the
    seeds recorded)."""
    runs = {}
    for (d, p, s, a), dc in decomp.items():
        m = metrics.get((d, p, s, a, "logit")) or {}
        runs[f"{d}/{p}/s{s}/{a}"] = {"auc": dc["auc"], "w_pair": dc["w_pair"], "x": dc["x"],
                                     "w": dc["w"], "L": m.get("L"),
                                     "auc_a": m.get(ANOMALOUS[d])}
    pairs_out, flips = {}, {}
    for d in DATASETS:
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                if single:
                    keys = [((d, p, sa, a), (d, p, sb, b))
                            for (dd, p, sa, aa) in decomp if dd == d and aa == a
                            for (d2, p2, sb, bb) in decomp if (d2, p2, bb) == (d, p, b)]
                else:
                    keys = [((d, p, s, a), (d, p, s, b)) for (dd, p, s, aa) in decomp
                            if dd == d and aa == a and (d, p, s, b) in decomp]
                rows = [(decomp[ka], decomp[kb]) for ka, kb in keys]
                if not rows:
                    continue
                w = np.array([x["w"] for x, _ in rows])
                dauc = np.array([x["auc"] - y["auc"] for x, y in rows])
                within = np.array([x["w"] * (x["w_pair"] - y["w_pair"]) for x, y in rows])
                cross = np.array([(1 - x["w"]) * (x["x"] - y["x"]) for x, y in rows])
                mean_d = float(dauc.mean())
                pairs_out[f"{d}/{a}-{b}"] = {
                    "n": len(rows), "d_auc": mean_d, "within_part": float(within.mean()),
                    "cross_part": float(cross.mean()), "w_mean": float(w.mean()),
                    "cross_share": float(cross.mean()) / mean_d if mean_d else math.nan,
                    "seeds": [[ka[1], ka[2], kb[2]] for ka, kb in keys]}
        flips[d] = order_flips(metrics, d, arms)
    return {"runs": runs, "pairs": pairs_out, "order": flips}


def order_flips(metrics: Mapping[tuple, Mapping], dataset: str, arms: Sequence[str]) -> dict:
    """Pairs of arms ordered differently by the seed-mean pooled metric and by the seed-mean L,
    and Kendall's tau of the two orders (logit readout)."""
    pm, lm = {}, {}
    for a in arms:
        vals = [m for (d, _p, _s, aa, ro), m in metrics.items()
                if d == dataset and aa == a and ro == "logit"]
        if vals:
            pm[a] = float(np.mean([m[PRIMARY[dataset]] for m in vals]))
            lm[a] = float(np.mean([m["L"] for m in vals if "L" in m]))
    names = sorted(pm)
    flipped = [f"{a}-{b}" for i, a in enumerate(names) for b in names[i + 1:]
               if (pm[a] - pm[b]) * (lm[a] - lm[b]) < 0]
    tau = float(kendalltau([pm[a] for a in names], [lm[a] for a in names])[0]) \
        if len(names) >= 2 else math.nan
    return {"arms": names, "pooled": pm, "L": lm, "flipped_pairs": flipped,
            "n_flipped": len(flipped), "kendall_tau": tau}


# --- 3 mechanism -----------------------------------------------------------------------------

def mechanism(decomp: Mapping[tuple, Mapping], metrics: Mapping[tuple, Mapping],
              c_offsets: Mapping[tuple, tuple]) -> dict:
    """Descriptive 3: per dataset the seed means of the pooled metric, W_pair and X of C, C.L20
    (and .last), the z-only readout of C, and the Spearman of C's c(v) with log T within each
    class (``c_offsets``: (dataset, pair, seed) -> (c (n,), totals (n,), labels (n,)))."""
    out = {}
    for d in DATASETS:
        row = {}
        for a in ("C", "C.L20", "C.last", "C.L20.last"):
            dcs = [v for (dd, _p, _s, aa), v in decomp.items() if dd == d and aa == a]
            ms = [m for (dd, _p, _s, aa, ro), m in metrics.items()
                  if dd == d and aa == a and ro == "logit"]
            row[a] = {"pooled": mean_sd([m[PRIMARY[d]] for m in ms]),
                      "w_pair": mean_sd([x["w_pair"] for x in dcs]),
                      "x": mean_sd([x["x"] for x in dcs])}
        zs = [m for (dd, _p, _s, aa, ro), m in metrics.items() if dd == d and aa == "C"
              and ro == "z"]
        row["C_z_only"] = mean_sd([m[PRIMARY[d]] for m in zs])
        rho = {}
        for cls in (0, 1):
            vals = []
            for (dd, _p, _s), (c, t, y) in c_offsets.items():
                if dd != d:
                    continue
                keep = np.asarray(y) == cls
                vals.append(spearman(np.asarray(c)[keep], np.log(np.asarray(t)[keep])))
            rho[str(cls)] = mean_sd(vals)
        row["spearman_c_logT"] = rho
        out[d] = row
    return out


# --- 4, 5 ------------------------------------------------------------------------------------

def validation_test(metrics: Mapping[tuple, Mapping], val_auc: Mapping[tuple, float]) -> dict:
    """Descriptive 4 (``val_auc``: (dataset, pair, seed, arm) -> the selected V-AUC)."""
    out = {}
    for d in DATASETS:
        for p in PAIRS:
            keys = [k for k in val_auc if k[0] == d and k[1] == p and (*k, "logit") in metrics]
            v = [val_auc[k] for k in keys]
            out[f"{d}/{p}"] = {
                "n": len(keys),
                "spearman_L": spearman(v, [metrics[(*k, "logit")]["L"] for k in keys]),
                "spearman_pooled": spearman(v, [metrics[(*k, "logit")][PRIMARY[d]]
                                                for k in keys])}
    return out


def selected_points(selection: Mapping, devlr: Mapping | None) -> dict:
    """Descriptive 5: per arm and rule the selected evaluation points; the D4 choices."""
    pts: dict[str, list] = {}
    for _label, e in (selection.get("chains") or {}).items():
        for r in ("sigma", "logit"):
            sel = (e.get("selection") or {}).get(r)
            if sel:
                pts.setdefault(f"{e['arm']}/{r}", []).append(int(sel["point"]))
    return {"points": {k: {"values": sorted(v), **quantiles(v)} for k, v in sorted(pts.items())},
            "d4": {k: {"choice": v.get("choice"), "lr": v.get("lr"), "scale": v.get("scale")}
                   for k, v in ((devlr or {}).get("settings") or {}).items()}}


# --- 6 length --------------------------------------------------------------------------------

def tertiles(totals: Sequence[int]) -> np.ndarray:
    """0 / 1 / 2: the length tertile of each included video (np.quantile at 1/3, 2/3)."""
    t = np.asarray(totals, dtype=np.float64)
    q1, q2 = np.quantile(t, [1 / 3, 2 / 3])
    return np.where(t <= q1, 0, np.where(t <= q2, 1, 2))


def length(per_video: Mapping[tuple, np.ndarray], totals: Mapping[str, Sequence[int]],
           arms: Sequence[str], chunked: Mapping[str, Sequence[bool]]) -> dict:
    """Descriptive 6 (``per_video``: (dataset, pair, seed, arm) -> per-video L in the included
    order; ``totals``: dataset -> T of the included videos; ``chunked``: dataset -> whether each
    included video was cut into N_chunk pieces)."""
    out = {}
    for d in DATASETS:
        if d not in totals:
            continue
        tt = tertiles(totals[d])
        cut = np.asarray(chunked.get(d) or [False] * len(tt), dtype=bool)
        rows = {}
        for a in ("C", *arms):
            runs = {k: np.asarray(v) for k, v in per_video.items() if k[0] == d and k[3] == a}
            if not runs:
                continue
            rows[a] = {"tertiles": [mean_sd([float(v[tt == i].mean()) if (tt == i).any()
                                             else math.nan for v in runs.values()])
                                    for i in range(3)],
                       "without_chunked": mean_sd([float(v[~cut].mean()) if (~cut).any()
                                                   else math.nan for v in runs.values()])}
        diff = {}
        for a in arms:
            if a in rows and "C" in rows:
                diff[a] = [rows["C"]["tertiles"][i]["mean"] - rows[a]["tertiles"][i]["mean"]
                           for i in range(3)]
        out[d] = {"arms": rows, "c_minus_arm": diff, "videos_chunked": int(cut.sum())}
    return out


# --- 7, 8 ------------------------------------------------------------------------------------

def efficiency(chain_records: Mapping[str, Mapping], scoring_seconds: float | None) -> dict:
    """Descriptive 7 (``chain_records``: '<chain>/<run>' -> {"arm", "parameters", "seconds"})."""
    by: dict[str, dict[str, list]] = {}
    for _label, r in chain_records.items():
        b = by.setdefault(r["arm"], {"parameters": [], "seconds": []})
        if r.get("parameters") is not None:
            b["parameters"].append(r["parameters"])
        if r.get("seconds") is not None:
            b["seconds"].append(r["seconds"])
    return {"arms": {a: {"parameters": sorted(set(v["parameters"])),
                         "seconds": mean_sd(v["seconds"])} for a, v in sorted(by.items())},
            "flops_per_snippet": "not measured by this code (reported from the ports)",
            "test_scoring_seconds": scoring_seconds}


def ties(sigma_ones: Mapping[tuple, int]) -> dict:
    """Descriptive 8 (``sigma_ones``: (dataset, pair, seed, arm) -> snippets with sigma 1.0f)."""
    by: dict[str, list] = {}
    for (d, _p, _s, a), n in sigma_ones.items():
        by.setdefault(f"{d}/{a}", []).append(int(n))
    return {k: {"total": int(sum(v)), "runs_with_ties": int(sum(1 for x in v if x))}
            for k, v in sorted(by.items())}


# --- 10, 11 ----------------------------------------------------------------------------------

def half_width(boot: Mapping, level_key: str = "99") -> float:
    lo, hi = (boot.get("intervals") or {}).get(level_key, [math.nan, math.nan])
    return (hi - lo) / 2 if math.isfinite(lo) and math.isfinite(hi) else math.nan


def estimates(prefreeze: Mapping, boot_l: Mapping[str, Mapping[str, Mapping]]) -> dict:
    """Descriptive 10: the (3f) UCF half-widths and the (3g) assumptions beside the actual 99%
    L bootstrap half-widths of C - B_k."""
    actual = {f"{a}/{d}": half_width(b) for d, rows in boot_l.items() for a, b in rows.items()}
    return {"actual_half_width_99": actual,
            "3f": (prefreeze.get("3f") or {}).get("methods"),
            "3g_designated": (prefreeze.get("3g") or {}).get("designated"),
            "3g_half_widths": (prefreeze.get("3g") or {}).get("half_widths")}


def d14(points: Mapping[tuple, Mapping], arms: Sequence[str]) -> dict:
    """Descriptive 11 (``points``: (dataset, pair, seed, arm) -> {"L": (30,), "pooled": (30,),
    "v_auc_sigma": (30,), "selected": 1-based point}): per method the test-selection inflation
    (best of the points - the selected point), the validation-test Spearman over the points and,
    per point, how many method pairs the seed-mean L orders unlike the selected checkpoints."""
    out = {}
    for d in DATASETS:
        rows = {}
        means_at: dict[str, np.ndarray] = {}
        selected_mean: dict[str, float] = {}
        for a in arms:
            runs = [v for k, v in points.items() if k[0] == d and k[3] == a]
            if not runs:
                continue
            infl_l = [float(np.max(v["L"]) - v["L"][v["selected"] - 1]) for v in runs]
            infl_p = [float(np.max(v["pooled"]) - v["pooled"][v["selected"] - 1]) for v in runs]
            rho = [spearman(v["v_auc_sigma"], v["L"]) for v in runs]
            rows[a] = {"inflation_L": mean_sd(infl_l), "inflation_pooled": mean_sd(infl_p),
                       "spearman_val_test_L": mean_sd(rho)}
            means_at[a] = np.mean([np.asarray(v["L"], dtype=np.float64) for v in runs], axis=0)
            selected_mean[a] = float(np.mean([v["L"][v["selected"] - 1] for v in runs]))
        names = sorted(means_at)
        flips = []
        for k in range(len(next(iter(means_at.values()))) if means_at else 0):
            flips.append(sum(1 for i, a in enumerate(names) for b in names[i + 1:]
                             if (means_at[a][k] - means_at[b][k])
                             * (selected_mean[a] - selected_mean[b]) < 0))
        out[d] = {"methods": rows, "flips_per_point": flips}
    return out


# --- everything ------------------------------------------------------------------------------

def descriptives(*, metrics: Mapping, decomp: Mapping, per_video: Mapping,
                 included_totals: Mapping, chunked: Mapping, selection: Mapping,
                 devlr: Mapping | None, val_auc: Mapping, c_offsets: Mapping,
                 chain_records: Mapping, scoring_seconds: float | None, sigma_ones: Mapping,
                 h2: Mapping, prefreeze: Mapping, boot_l: Mapping, d14_points: Mapping,
                 family_arms: Sequence[str], all_arms: Sequence[str], condition3: Mapping,
                 single: bool = False, non_chosen: Mapping | None = None) -> dict:
    """Descriptives 1-12 (module docstring)."""
    return {"1_cells": cell_table(metrics, single, non_chosen),
            "2_problem": problem(decomp, metrics, ["C", *all_arms], single),
            "3_mechanism": mechanism(decomp, metrics, c_offsets),
            "4_validation_test": validation_test(metrics, val_auc),
            "5_selected_points": selected_points(selection, devlr),
            "6_length": length(per_video, included_totals, family_arms, chunked),
            "7_efficiency": efficiency(chain_records, scoring_seconds),
            "8_ties": ties(sigma_ones),
            "9_h2": dict(h2),
            "10_estimates": estimates(prefreeze, boot_l),
            "11_d14": d14(d14_points, family_arms),
            "12_condition3_lower_98": dict(condition3.get("lower_98") or {})}
