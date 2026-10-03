"""E003/C5: does the selection criterion (validation video AUC) pick a worse epoch than the
proxy target would? (P10, UCF; base and full judged separately)

Candidates: stage-2 epochs 1-30 of each recorded run. Cross-fit: 1,000 half splits of the
validation videos, stratified (anomalous with a proxy / anomalous without / normal), with
default_rng(0) restarted for every run and variant. Per split: on half 1 pick e_A (best
video AUC, earliest on ties) and e_P (best proxy frame AP); on half 2
loss = AP(e_P) - AP(e_A) and power = AP(e_P) - mean over the 30 epochs of AP. Split-half
reliability: Spearman of half-1 and half-2 AP over the 30 epochs. Run values are means over
splits; cells (pair x variant) use seed means. Confirm: all 8 cells loss >= 1.0. Reject: all
8 cells loss < 0.3 with power >= 1.0 and reliability >= 0.5. Otherwise hold ("hold, low
power" when loss < 0.3 everywhere but power or reliability falls short) (PLAN.md, C5).

    python -m wvad.diagnostics.c5 --runs <local copy of ~/wvad_runs/e003>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.c0 import PROXY_FILE
from wvad.diagnostics.common import DEV_SEEDS, E003_RESULTS, code_version, write_csv, write_json
from wvad.diagnostics.proxy_eval import proxy_video_set, weighted_ap, weighted_auc
from wvad.diagnostics.runs import CELLS, PAIRS, rows_of, run_dir, selection, val_record
from wvad.diagnostics.uca import VARIANTS, load_proxy

N_SPLITS = 1000
CONFIRM_LOSS, REJECT_LOSS, POWER_MIN, RELIABILITY_MIN = 1.0, 0.3, 1.0, 0.5
ROUND_ENDS = (10, 20, 30)
SATURATION_TOL = 0.001


def video_auc(vmax: np.ndarray, labels: np.ndarray) -> float:
    return weighted_auc(vmax, labels.astype(float), 1.0 - labels)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def splits(strata: list[np.ndarray], n_rows: int, n: int = N_SPLITS) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        half1 = np.zeros(n_rows, dtype=bool)
        for stratum in strata:
            perm = rng.permutation(stratum)
            half1[perm[:len(perm) // 2]] = True
        out.append(half1)
    return out


def category_video_weights(categories: tuple[str, ...], anomalous: np.ndarray) -> np.ndarray:
    """Anomalous videos weighted by inverse category count (mean 1); normal videos 1."""
    w = np.ones(len(categories))
    cats = [c for c, a in zip(categories, anomalous, strict=True) if a]
    counts = {c: cats.count(c) for c in set(cats)}
    raw = np.array([1.0 / counts[c] if a else 0.0
                    for c, a in zip(categories, anomalous, strict=True)])
    w[anomalous] = raw[anomalous] / raw[anomalous].mean()
    return w


def analyze_run(scores: np.ndarray, labels: np.ndarray, vs, rows: np.ndarray) -> dict:
    """scores: (30, n_val, 32) stage-2 epochs 1-30; rows: proxy-set rows in the val order."""
    n_ep, n_val = scores.shape[:2]
    vmax = scores.max(axis=2)                                         # (30, n_val)
    in_proxy = np.zeros(n_val, dtype=bool)
    in_proxy[rows] = True
    anomalous = labels == 1
    strata = [np.flatnonzero(anomalous & in_proxy), np.flatnonzero(anomalous & ~in_proxy),
              np.flatnonzero(~anomalous)]
    w = category_video_weights(vs.categories, vs.anomalous)[:, None]
    ps = scores[:, rows]                                              # (30, n_proxy, 32)

    def ap_curve(sel: np.ndarray, weighted: bool = False) -> np.ndarray:
        pos, neg = (vs.pos * w, vs.neg * w) if weighted else (vs.pos, vs.neg)
        return np.array([weighted_ap(ps[e][sel], pos[sel], neg[sel]) for e in range(n_ep)])

    acc = {k: [] for k in ("loss", "power", "reliability", "tie", "loss_tie_expected",
                           "loss_videoset", "loss_category_weighted")}
    for half1 in splits(strata, n_val):
        p1, p2 = half1[rows], ~half1[rows]
        auc1 = np.array([video_auc(vmax[e, half1], labels[half1]) for e in range(n_ep)])
        aucp1 = np.array([video_auc(vmax[e, half1 & in_proxy], labels[half1 & in_proxy])
                          for e in range(n_ep)])
        ap1, ap2 = ap_curve(p1), ap_curve(p2)
        ap1w, ap2w = ap_curve(p1, True), ap_curve(p2, True)
        e_a, e_p = int(np.argmax(auc1)), int(np.argmax(ap1))
        ties = np.flatnonzero(auc1 == auc1.max())
        acc["loss"].append(ap2[e_p] - ap2[e_a])
        acc["power"].append(ap2[e_p] - ap2.mean())
        acc["reliability"].append(spearman(ap1, ap2))
        acc["tie"].append(len(ties) > 1)
        acc["loss_tie_expected"].append(ap2[e_p] - ap2[ties].mean())
        acc["loss_videoset"].append(ap2[e_p] - ap2[int(np.argmax(aucp1))])
        acc["loss_category_weighted"].append(ap2w[int(np.argmax(ap1w))] - ap2w[e_a])
    auc_all = np.array([video_auc(vmax[e], labels) for e in range(n_ep)])
    ap_all = ap_curve(np.ones(len(rows), dtype=bool))
    out = {k: float(np.nanmean(v)) for k, v in acc.items()}
    out["tie_rate"] = out.pop("tie")
    out["reliability_nan_splits"] = int(np.isnan(acc["reliability"]).sum())
    out |= {"spearman_full": spearman(auc_all, ap_all), "ap_range": float(np.ptp(ap_all)),
            "auc_by_epoch": auc_all.tolist(), "ap_by_epoch": ap_all.tolist()}
    return out


def cell_verdict(cells: list[dict]) -> str:
    if all(c["loss"] >= CONFIRM_LOSS for c in cells):
        return "confirm"
    small = all(c["loss"] < REJECT_LOSS for c in cells)
    powered = all(c["power"] >= POWER_MIN and c["reliability"] >= RELIABILITY_MIN for c in cells)
    if small and powered:
        return "reject"
    return "hold, low power" if small else "hold"


def criterion_saturation(runs: Path) -> list[dict]:
    rows = []
    for dataset in ("ucf", "xdv"):
        for pair in PAIRS:
            for cell in CELLS:
                for seed in DEV_SEEDS:
                    d = run_dir(runs, dataset, pair, seed, cell)
                    hist = json.loads((d / "history.json").read_text(encoding="utf-8"))
                    auc = np.array([h["val_video_auc"] for h in hist])
                    rows.append({"dataset": dataset, "pair": pair, "cell": cell, "seed": seed,
                                 "auc_min": float(auc.min()), "auc_max": float(auc.max()),
                                 "auc_range": float(np.ptp(auc)),
                                 "epochs_within_tol": int((auc >= auc.max() - SATURATION_TOL)
                                                          .sum())})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E003 C5 selection criterion (P10)")
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    manifest, proxy = load_manifest(manifest_file("ucf")), load_proxy(PROXY_FILE)
    sel_rows, end_rows = [], []
    for pair in PAIRS:
        for cell in CELLS:
            for seed in DEV_SEEDS:
                d = run_dir(args.runs, "ucf", pair, seed, cell)
                rec = val_record(d)
                keep = rec.epochs >= 1
                scores, epochs = rec.scores[keep], rec.epochs[keep]
                labels = np.array([manifest.labels[i] for i in rec.val_indices])
                best = selection(d)["best_epoch"]
                for variant in VARIANTS:
                    vs = proxy_video_set(manifest, proxy, variant, rec.val_indices)
                    rows = rows_of(rec.val_indices, vs.indices)
                    r = analyze_run(scores, labels, vs, rows)
                    sel_rows.append({"pair": pair, "cell": cell, "seed": seed,
                                     "variant": variant,
                                     **{k: v for k, v in r.items() if not k.endswith("epoch")}})
                    marks = [(str(e), e) for e in ROUND_ENDS] + [("selected", best)]
                    for mark, e in marks:
                        k = int(np.flatnonzero(epochs == e)[0])
                        end_rows.append({"pair": pair, "cell": cell, "seed": seed,
                                         "variant": variant, "mark": mark, "epoch": e,
                                         "val_video_auc": r["auc_by_epoch"][k],
                                         "proxy_ap": r["ap_by_epoch"][k]})
                    print(f"{pair} {cell} s{seed} {variant}: loss={r['loss']:.3f} "
                          f"power={r['power']:.3f} rel={r['reliability']:.3f}", flush=True)
    summary_cells, verdicts = [], {}
    for cell in CELLS:
        cells = []
        for pair in PAIRS:
            for variant in VARIANTS:
                rr = [r for r in sel_rows if (r["pair"], r["cell"], r["variant"]) ==
                      (pair, cell, variant)]
                cells.append({"cell": cell, "pair": pair, "variant": variant,
                              **{k: float(np.nanmean([r[k] for r in rr]))
                                 for k in ("loss", "power", "reliability", "tie_rate",
                                           "loss_tie_expected", "loss_videoset",
                                           "loss_category_weighted", "spearman_full",
                                           "ap_range")}})
        summary_cells += cells
        verdicts[cell] = cell_verdict(cells)
    result = {"check": "E003/C5 selection criterion (P10, UCF)", "code": code_version(),
              "splits": {"n": N_SPLITS, "rng": "default_rng(0) restarted per run and variant",
                         "strata": ["anomalous with proxy", "anomalous without proxy",
                                    "normal"]},
              "thresholds": {"confirm_loss": CONFIRM_LOSS, "reject_loss": REJECT_LOSS,
                             "power_min": POWER_MIN, "reliability_min": RELIABILITY_MIN},
              "category_weighting": "anomalous videos weighted by the inverse number of "
                                    "videos with a proxy per category (mean 1), normal 1; "
                                    "descriptive",
              "nan_policy": "NaN Spearman values (constant AP curves) are left out of means; "
                            "counts in c5_selection.csv (reliability_nan_splits)",
              "cells": summary_cells, "verdicts": verdicts,
              "criterion_saturation": criterion_saturation(args.runs)}
    write_csv(E003_RESULTS / "c5_selection.csv", sel_rows)
    write_csv(E003_RESULTS / "c5_round_ends.csv", end_rows)
    write_json(E003_RESULTS / "c5_summary.json", result)
    print("verdicts:", verdicts)


if __name__ == "__main__":
    main()
