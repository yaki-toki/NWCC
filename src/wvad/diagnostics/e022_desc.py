"""E022 descriptives (PLAN.md "기술 (판정 밖)" 1-12): computed after the verdict and written
beside it, never used for the verdict, a flag, condition (2) or a later choice. Descriptive-only
code (PLAN.md "코드 규칙" 4): no verdict or flag is computed here.

    1   per arm / setting / readout / metric: mean, seed SD, the seed values; d_z against R0
    2   the decomposition (``e022_stats.decomposition``, in the summary): a pointer
    3   E017 risk check: C and R0 primary and anomalous-only metrics of z only, c only, z + c, and
        the share of anomalous videos with c < 0
    4   validation and test: per setting the Spearman correlation of the selected V-AUC (C: W,
        R0: 32) with the test primary metric; per seed the sign agreement of the cross-fitted
        validation C - R0 (E010 splits) with the test C - R0
    5   selected epochs (C, C.G, B, R1, R0, A1.S)
    6   length: primary metric per test-length tertile (C, R0, C - R0), Spearman of c(v) with
        log T within each class, the tail frames [32 L, T) (L = T // 32) in the annotation
    7   offset bag: C.o32's primary metric and L, per video c - c_32 and its Spearman with log T
        within each class
    8   localisation: anomalous-only pooled metric, the opposite within-video metric, the videos
        left out of L
    9   extraction: per video grab attempts / successes and replaced frames, the (y3) deltas, the
        (y2) errors, throughput and time (the extraction gate record)
    10  ties at 1.0f of the sigma readout, per arm
    11  the label-free distribution statistics of the derived stage (extraction gate record)
    12  gate (v)'s bit-equal share and the (s3t) error distribution
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from wvad.diagnostics import e021_score as es
from wvad.diagnostics import e022_score as sc
from wvad.diagnostics.e010 import splits_sha256
from wvad.diagnostics.stage_d import calibration
from wvad.eval import frames as fr
from wvad.eval.crossfit import crossfit, stratified_half_splits, video_auc

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
ONE = np.float32(1.0)
METRICS = ("frame_auc", "frame_ap", "anom_frame_auc", "anom_frame_ap", "video_auc", "L",
           "L_opposite")


def _mean_sd(v: Sequence[float]) -> dict:
    x = [float(a) for a in v if a is not None and math.isfinite(float(a))]
    return {"n": len(x), "mean": float(np.mean(x)) if x else math.nan,
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else math.nan, "values": x}


def _quantiles(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()),
            **{f"q{int(round(100 * q))}": float(np.quantile(x, q)) for q in QUANTILES}}


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return math.nan
    return float(spearmanr(x[keep], y[keep])[0])


def primary(dataset: str) -> str:
    return fr.PRIMARY[dataset]


# --- 1 and the reported numbers -------------------------------------------------------------

def reported(metrics: Mapping, arms: Sequence[str] = ("C", "R0")) -> dict:
    """The numbers reported whatever the outcome (PLAN.md "결과의 쓰임"): seed mean and SD of
    the primary metric per dataset and pair (logit readout, primary checkpoint rules)."""
    out = {}
    for (d, p, _s, a, ro), m in metrics.items():
        if a in arms and ro == "logit":
            out.setdefault(f"{d}/{p}/{a}", []).append(m[primary(d)])
    return {k: _mean_sd(v) for k, v in sorted(out.items())}


def cell_table(metrics: Mapping) -> dict:
    """Descriptive 1: per dataset / pair / arm / readout / metric the mean, seed SD and seed
    values; per dataset / pair / arm / readout d_z of arm - R0 on the primary metric (common
    seeds; R0.last for the .last arms)."""
    groups: dict[str, list] = {}
    by: dict = {}
    for (d, p, s, a, ro), m in metrics.items():
        for name in METRICS:
            if name in m:
                groups.setdefault(f"{d}/{p}/{a}/{ro}/{name}", []).append(m[name])
        by[(d, p, s, a, ro)] = m[primary(d)]
    d_z = {}
    for (d, p, _s, a, ro) in {(k[0], k[1], None, k[3], k[4]) for k in by}:
        base = "R0.last" if a.endswith(".last") else "R0"
        if a == base:
            continue
        diffs = [by[(d, p, s, a, ro)] - by[(d, p, s, base, ro)] for s in sc.SEEDS
                 if (d, p, s, a, ro) in by and (d, p, s, base, ro) in by]
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else math.nan
        d_z[f"{d}/{p}/{a}/{ro}"] = {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs
                                    else math.nan, "d_z": float(np.mean(diffs)) / sd
                                    if sd and math.isfinite(sd) else math.nan}
    return {"cells": {k: _mean_sd(v) for k, v in sorted(groups.items())},
            "d_z_vs_r0": dict(sorted(d_z.items()))}


# --- 3 E017 risk check -------------------------------------------------------------------------

def readout_decomposition(metrics: Mapping, loaded, sets: Mapping) -> dict:
    """Descriptive 3 (C, R0): seed means of the primary and anomalous-only metrics for z only,
    c only and z + c (logit), and the share of anomalous videos with c(v) < 0."""
    out = {}
    for d, ts in sets.items():
        anom = ts.labels == 1
        for p in sc.PAIRS:
            for a in ("C", "R0"):
                entry = {}
                for ro in ("z", "c", "logit"):
                    keys = [(d, p, s, a, ro) for s in sc.SEEDS if (d, p, s, a, ro) in metrics]
                    entry[ro] = {m: float(np.mean([metrics[k][m] for k in keys])) if keys
                                 else math.nan for m in (primary(d), f"anom_{primary(d)}")}
                shares = [float(np.mean(u.c[anom] < 0)) for (dd, pp, _s, aa), u in
                          loaded.units.items() if (dd, pp, aa) == (d, p, a) and u.c is not None]
                entry["anomalous_share_c_below_0"] = _mean_sd(shares)
                out[f"{d}/{p}/{a}"] = entry
    return out


# --- 4 validation and test ------------------------------------------------------------------

def _w_scores(sf: Mapping) -> np.ndarray:
    """(epochs, n, 1) video scores of the W readout (max fp32 sigma), the selection's."""
    return sc.readout_scores(sf, "W")[..., None]


def _crossfit(values: np.ndarray, labels: np.ndarray, splits: np.ndarray) -> float:
    return crossfit(values, labels, splits, {"v_auc": video_auc}).mean("v_auc")


def val_test(metrics: Mapping, info: Mapping, scores: Path, e020_runs: Path,
             cal: Mapping | None = None) -> dict:
    """Descriptive 4: Spearman of the selected V-AUC with the test primary metric (per setting,
    per arm and C, R0 pooled); per seed the sign agreement of the cross-fitted validation
    V-AUC difference C - R0 (C: W readout, R0: its 32-unit record; E010 splits) with the test
    primary C - R0 (per setting and pair-averaged)."""
    cal = cal or calibration()
    out: dict = {"rank_correlation": {}, "sign_agreement": {}, "crossfit_errors": [],
                 "splits_equal_e010": {}}
    for d in sc.DATASETS:
        per_pair = {}
        for p in sc.PAIRS:
            pooled = {"x": [], "y": []}
            for a in ("C", "R0"):
                xs, ys = [], []
                for s in sc.SEEDS:
                    m = metrics.get((d, p, s, a, "logit"))
                    v = (info.get((d, p, s, a)) or {}).get("val_v_auc")
                    if m is not None and v is not None:
                        xs.append(v)
                        ys.append(m[primary(d)])
                out["rank_correlation"][f"{d}/{p}/{a}"] = {"n": len(xs),
                                                           "spearman": _spearman(xs, ys)}
                pooled["x"] += xs
                pooled["y"] += ys
            out["rank_correlation"][f"{d}/{p}/pooled"] = {
                "n": len(pooled["x"]), "spearman": _spearman(pooled["x"], pooled["y"])}
            per_pair[p] = _seed_signs(d, p, metrics, scores, e020_runs, cal[d],
                                      out["crossfit_errors"], out["splits_equal_e010"])
            out["sign_agreement"][f"{d}/{p}"] = _share(list(per_pair[p].values()))
        both = [s for s in sc.SEEDS if all(s in per_pair[p] for p in sc.PAIRS)]
        out["sign_agreement"][f"{d}/pair_mean"] = _share(
            [tuple(float(np.mean([per_pair[p][s][i] for p in sc.PAIRS])) for i in (0, 1))
             for s in both])
    return out


def _seed_signs(d: str, p: str, metrics: Mapping, scores: Path, e020_runs: Path, cal: Mapping,
                errors: list, equal: dict) -> dict:
    out = {}
    for s in sc.SEEDS:
        mc, mr = metrics.get((d, p, s, "C", "logit")), metrics.get((d, p, s, "R0", "logit"))
        if mc is None or mr is None:
            continue
        chain = sc.chain_name(d, p, s)
        try:
            with np.load(sc.val_path(scores, es.RunKey(d, p, s, "hb", sc.PIPELINE))) as z:
                sf = {k: z[k] for k in z.files}
            labels = np.asarray(sf["labels"])
            splits = stratified_half_splits(labels, cal["R"], seed=0)
            equal[d] = splits_sha256(splits) == cal["splits_sha256"]
            vc = _crossfit(_w_scores(sf), labels, splits)
            with np.load(Path(e020_runs) / chain / sc.R0_CELL / "record_val_scores.npz") as z:
                rec = z["scores"][1:]
            vr = _crossfit(rec, labels, splits)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{chain}: {exc!r}")
            continue
        out[s] = (vc - vr, mc[primary(d)] - mr[primary(d)])
    return out


def _share(pairs: Sequence[tuple[float, float]]) -> dict:
    if not pairs:
        return {"n": 0, "share": math.nan}
    return {"n": len(pairs), "share": float(np.mean([np.sign(a) == np.sign(b)
                                                     for a, b in pairs]))}


# --- 5 selected epochs -------------------------------------------------------------------------

def selected_epochs(info: Mapping) -> dict:
    """Descriptive 5: the checkpoint epochs per dataset / pair / arm and their counts."""
    out = {}
    for d in sc.DATASETS:
        for p in sc.PAIRS:
            for a in ("C", "C.G", "B", "R1", "R0", "A1.S"):
                e = [(info.get((d, p, s, a)) or {}).get("epoch") for s in sc.SEEDS]
                got = [x for x in e if x is not None]
                out[f"{d}/{p}/{a}"] = {"epochs": e, "counts": {str(k): v for k, v in sorted(
                    Counter(got).items())}}
    return out


# --- 6 length, 7 offset bag ---------------------------------------------------------------

def tertiles(totals: np.ndarray) -> np.ndarray:
    cuts = np.quantile(totals, [1 / 3, 2 / 3])
    return np.digitize(totals, cuts, right=True)


def length(loaded, sets: Mapping) -> dict:
    """Descriptive 6: per test-length tertile the primary metric of C and R0 (logit) and the
    seed mean of the pair-averaged C - R0; Spearman of c(v) with log T within each class (C,
    R0); the tail frames [32 L, T) and the annotation."""
    out = {"tertiles": {}, "c_log_t": {}, "tail": {}}
    for d, ts in sets.items():
        tert = tertiles(ts.totals)
        logt = np.log(ts.totals.astype(np.float64))
        per = {}
        for (dd, p, s, a), u in loaded.units.items():
            if dd != d or a not in ("C", "R0"):
                continue
            frames = fr.frame_arrays(u.logit, ts.totals, u.kind)
            vals = []
            for t in range(3):
                idx = np.flatnonzero(tert == t)
                try:
                    vals.append(fr.pooled_frame_metrics([frames[i] for i in idx],
                                                        [ts.gts[i] for i in idx])[primary(d)])
                except ValueError:
                    vals.append(math.nan)
            per[(p, s, a)] = vals
            if u.c is not None:
                out["c_log_t"].setdefault(f"{d}/{p}/{a}", {"normal": [], "anomalous": []})
                for name, y in (("normal", 0), ("anomalous", 1)):
                    keep = ts.labels == y
                    out["c_log_t"][f"{d}/{p}/{a}"][name].append(_spearman(u.c[keep],
                                                                          logt[keep]))
        for a in ("C", "R0"):
            for t in range(3):
                out["tertiles"][f"{d}/{a}/t{t}"] = _mean_sd(
                    [v[t] for (p, s, aa), v in per.items() if aa == a])
        diffs = {t: [float(np.mean([per[(p, s, "C")][t] - per[(p, s, "R0")][t]
                                    for p in sc.PAIRS]))
                     for s in sc.SEEDS if all((p, s, x) in per for p in sc.PAIRS
                                              for x in ("C", "R0"))] for t in range(3)}
        for t, v in diffs.items():
            out["tertiles"][f"{d}/C-R0/t{t}"] = _mean_sd(v)
        out["tertile_bounds"] = out.get("tertile_bounds", {}) | {
            d: [float(x) for x in np.quantile(ts.totals, [1 / 3, 2 / 3])]}
        out["tail"][d] = tail_frames(ts)
    out["c_log_t"] = {k: {c: _mean_sd(v) for c, v in r.items()}
                      for k, r in out["c_log_t"].items()}
    return out


def tail_frames(ts) -> dict:
    """Frames [32 L, T) (L = T // 32) that the earlier 32-clip extraction did not observe: their
    number, the share of them inside the annotation and the share of annotated frames there."""
    tail = pos_tail = pos = 0
    for t, gt in zip(ts.totals, ts.gts, strict=True):
        start = 32 * (int(t) // 32)
        tail += int(t) - start
        pos_tail += int(np.asarray(gt)[start:].sum())
        pos += int(np.asarray(gt).sum())
    return {"tail_frames": tail, "tail_share_annotated": pos_tail / tail if tail else math.nan,
            "annotated_share_in_tail": pos_tail / pos if pos else math.nan}


def offset_bag(metrics: Mapping, loaded, sets: Mapping) -> dict:
    """Descriptive 7: C.o32's primary metric and L, the per-video c - c_32 and its Spearman with
    log T within each class."""
    out = {"metrics": {}, "c_minus_c32": {}}
    for (d, p, _s, a, ro), m in metrics.items():
        if a == "C.o32" and ro == "logit":
            out["metrics"].setdefault(f"{d}/{p}", {"primary": [], "L": []})
            out["metrics"][f"{d}/{p}"]["primary"].append(m[primary(d)])
            out["metrics"][f"{d}/{p}"]["L"].append(m.get("L", math.nan))
    out["metrics"] = {k: {n: _mean_sd(v) for n, v in r.items()}
                      for k, r in out["metrics"].items()}
    for d, ts in sets.items():
        logt = np.log(ts.totals.astype(np.float64))
        for p in sc.PAIRS:
            diffs, rho = [], {"normal": [], "anomalous": []}
            for s in sc.SEEDS:
                c, o = loaded.units.get((d, p, s, "C")), loaded.units.get((d, p, s, "C.o32"))
                if c is None or o is None:
                    continue
                delta = c.c.astype(np.float64) - o.c.astype(np.float64)
                diffs.append(delta)
                for name, y in (("normal", 0), ("anomalous", 1)):
                    keep = ts.labels == y
                    rho[name].append(_spearman(delta[keep], logt[keep]))
            out["c_minus_c32"][f"{d}/{p}"] = {
                "per_video": _quantiles(np.concatenate(diffs)) if diffs else {"n": 0},
                "spearman_log_t": {k: _mean_sd(v) for k, v in rho.items()}}
    return out


# --- 8 localisation, 10 ties ------------------------------------------------------------------

def localisation(metrics: Mapping) -> dict:
    """Descriptive 8: anomalous-only pooled metric, the opposite within-video metric and the
    videos left out of L, per dataset / pair / arm (logit readout)."""
    groups: dict = {}
    for (d, p, _s, a, ro), m in metrics.items():
        if ro != "logit":
            continue
        g = groups.setdefault(f"{d}/{p}/{a}", {"anomalous_only": [], "L_opposite": [],
                                               "L_excluded": set()})
        g["anomalous_only"].append(m[f"anom_{primary(d)}"])
        g["L_opposite"].append(m.get("L_opposite", math.nan))
        g["L_excluded"].add(int(m.get("L_excluded", -1)))
    return {k: {"anomalous_only": _mean_sd(v["anomalous_only"]),
                "L_opposite": _mean_sd(v["L_opposite"]),
                "L_excluded": sorted(v["L_excluded"])} for k, v in sorted(groups.items())}


def ties(loaded) -> dict:
    """Descriptive 10: units at exactly 1.0f in the sigma readout (and videos whose maximum is
    1.0f), seed mean per dataset / pair / arm."""
    groups: dict = {}
    for (d, p, _s, a), u in loaded.units.items():
        sig = u.values("sigma")
        units = int(sum(int((np.asarray(x) == ONE).sum()) for x in sig))
        videos = int(sum(bool(np.max(x) == ONE) for x in sig))
        g = groups.setdefault(f"{d}/{p}/{a}", {"units": [], "videos": []})
        g["units"].append(units)
        g["videos"].append(videos)
    return {k: {n: _mean_sd(v) for n, v in r.items()} for k, r in sorted(groups.items())}


# --- 9, 11, 12 records -------------------------------------------------------------------------

def extraction(record: Mapping) -> dict:
    """Descriptive 9 from the extraction gate record: per dataset and encoder the grab and
    replacement counts (quantiles; videos with replaced frames), the (y3) deltas, the (y2)
    errors, and throughput and time of the production runs."""
    out = {}
    for d in sc.DATASETS:
        per = {}
        for e, cols in ((record.get(f"desc9_{d}") or {}).items()):
            arr = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
            per[e] = {k: _quantiles(v) for k, v in arr.items()}
            per[e]["videos_with_replaced"] = int((arr.get("replaced", np.zeros(0)) > 0).sum())
            per[e]["grab_not_equal_success"] = int(
                (arr.get("grab_attempts", np.zeros(0)) != arr.get("grab_successes",
                                                                   np.zeros(0))).sum())
        y3 = record.get(f"y3_{d}") or {}
        y2 = (record.get(f"y2_{d}") or {}).get("comparison") or {}
        dense = record.get(f"dense_{d}") or {}
        deltas = [x for x in ((record.get("gates") or {}).get("y3") or {}).get("deltas") or []
                  if x.get("dataset") == d]
        out[d] = {"index": per, "y3": {k: y3.get(k) for k in ("videos", "nonzero_delta",
                                                               "max_abs_delta", "violations")}
                  | {"deltas": deltas},
                  "y2": {e: {k: c.get(k) for k in ("status", "rows", "bit_equal_rows",
                                                    "max_rel_l2", "min_cos")}
                         for e, c in y2.items()},
                  "throughput": {k: dense.get(k) for k in ("videos_finished", "snippets",
                                                           "seconds")}
                  | {"snippets_per_second": (dense.get("snippets") / dense["seconds"]
                                             if dense.get("seconds") else None)}}
    return out


def distribution(record: Mapping) -> dict:
    """Descriptive 11: the derived stage's label-free statistics (not recomputed)."""
    return {d: record.get(f"desc11_{d}") for d in sc.DATASETS}


def reproduction(gate: Mapping) -> dict:
    """Descriptive 12: gate (v)'s bit-equal share and the (s3t) relative error distribution."""
    v = gate.get("v") or {}
    rows = list(v.get("rows") or [])
    s3t = [r.get("max_rel_dz") for r in ((gate.get("s3t") or {}).get("rows") or {}).values()
           if r.get("max_rel_dz") is not None]
    return {"v_bit_equal_share": (sum(bool(r.get("bit")) for r in rows) / len(rows)
                                  if rows else math.nan), "v_runs": len(rows),
            "v_status": v.get("status"), "s3t_max_rel_dz": _quantiles(np.asarray(s3t))}


# --- all -----------------------------------------------------------------------------------

def descriptives(*, metrics: Mapping, loaded, sets: Mapping, info: Mapping,
                 extract_record: Mapping, gate: Mapping, scores: Path, e020_runs: Path) -> dict:
    """Descriptives 1-12 (the selected epochs and V-AUC come through ``info``)."""
    return {"1_cells": cell_table(metrics),
            "2_decomposition": "see 'decomposition' (e022_stats.decomposition)",
            "3_readout_decomposition": readout_decomposition(metrics, loaded, sets),
            "4_validation_test": val_test(metrics, info, scores, e020_runs),
            "5_selected_epochs": selected_epochs(info),
            "6_length": length(loaded, sets),
            "7_offset_bag": offset_bag(metrics, loaded, sets),
            "8_localisation": localisation(metrics),
            "9_extraction": extraction(extract_record),
            "10_ties": ties(loaded),
            "11_distribution": distribution(extract_record),
            "12_reproduction": reproduction(gate)}
