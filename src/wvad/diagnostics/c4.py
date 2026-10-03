"""E003/C4: does pseudo-label training make the late-round segment ranking worse than
continued MIL training? (UCF, training-part anomalous videos with a proxy)

P@3: share of proxy-positive segments (>= 50% proxy frames) among a video's top-3 scored
segments, ties at the 3rd value counted by expectation:
P@3 = (p_gt + (3 - g) * p_tie / t) / 3. Rounds: 1 = stage-1 model (shared by base and full),
2 and 3 = after epochs 10 and 20, 4 = final model (descriptive). D_s = mean over videos of
P@3_full(3) - P@3_base(3) for seed s; D = mean over seeds. 95% CI: hierarchical bootstrap
(10,000; default_rng(0)): resample the 3 seeds, then the videos once per replicate (paired
across the sampled seeds). Confirm: all 8 cells (pair x variant) D <= -3 points, CI upper
< 0 and all D_s < 0. Reject: all 8 cells CI lower > -1. Otherwise hold (PLAN.md, C4).

    python -m wvad.diagnostics.c4 --runs <local copy of ~/wvad_runs/e003>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.c0 import PROXY_FILE
from wvad.diagnostics.common import DEV_SEEDS, E003_RESULTS, code_version, write_csv, write_json
from wvad.diagnostics.proxy_eval import proxy_video_set
from wvad.diagnostics.runs import CELLS, PAIRS, pseudo_record, rows_of, run_dir
from wvad.diagnostics.uca import VARIANTS, load_proxy

K, N_BOOT = 3, 10_000
CONFIRM_D, REJECT_CI_LOWER = -3.0, -1.0


def precision_at_k(scores: np.ndarray, positive: np.ndarray, k: int = K) -> np.ndarray:
    """Tie-aware precision@k per row of (n, S) scores against (n, S) boolean positives."""
    vk = -np.sort(-scores, axis=1)[:, k - 1:k]
    gt, eq = scores > vk, scores == vk
    g, t = gt.sum(1), eq.sum(1)
    p_gt, p_tie = (gt & positive).sum(1), (eq & positive).sum(1)
    return (p_gt + (k - g) * p_tie / t) / k


def hierarchical_ci(diff: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of (n_seeds, n_videos) differences: resample seeds, then videos."""
    rng = np.random.default_rng(seed)
    n_s, n_v = diff.shape
    stats = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, n_s, n_s)
        videos = rng.integers(0, n_v, n_v)
        stats[b] = diff[np.ix_(seeds, videos)].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def label_quality(labels: np.ndarray, positive: np.ndarray) -> dict:
    sel, mined = labels == 1, labels == 0
    return {"selected_precision": float((sel & positive).sum() / max(sel.sum(), 1)),
            "n_selected": int(sel.sum()),
            "mined_negative_error": float((mined & positive).sum() / max(mined.sum(), 1)),
            "n_mined": int(mined.sum())}


def category_weights(categories: tuple[str, ...]) -> np.ndarray:
    cats, counts = np.unique(categories, return_counts=True)
    w = np.array([1.0 / dict(zip(cats, counts, strict=True))[c] for c in categories])
    return w / w.sum()


def verdict(cells: list[dict]) -> str:
    if all(c["D"] <= CONFIRM_D and c["ci"][1] < 0 and all(d < 0 for d in c["D_by_seed"])
           for c in cells):
        return "confirm"
    if all(c["ci"][0] > REJECT_CI_LOWER for c in cells):
        return "reject"
    return "hold"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E003 C4 pseudo-label ranking (P3)")
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    manifest, proxy = load_manifest(manifest_file("ucf")), load_proxy(PROXY_FILE)
    rows, cells = [], []
    for pair in PAIRS:
        records = {(s, c): pseudo_record(run_dir(args.runs, "ucf", pair, s, c))
                   for s in DEV_SEEDS for c in CELLS}
        for variant in VARIANTS:
            order = records[(DEV_SEEDS[0], CELLS[0])].train_indices
            vs = proxy_video_set(manifest, proxy, variant, order, include_normal=False)
            positive = vs.positive_segments()
            p_at = {}
            r1_identical = True
            for s in DEV_SEEDS:
                for c in CELLS:
                    rec = records[(s, c)]
                    sc = rec.scores[:, rows_of(rec.train_indices, vs.indices)]
                    lab = rec.labels[:, rows_of(rec.train_indices, vs.indices)]
                    for r in range(4):
                        p_at[(s, c, r + 1)] = precision_at_k(sc[r], positive)
                        q = label_quality(lab[r], positive) if r < 3 else {}
                        rows.append({"pair": pair, "variant": variant, "cell": c, "seed": s,
                                     "round": r + 1, "n_videos": len(vs.indices),
                                     "p_at_3": float(100 * p_at[(s, c, r + 1)].mean()), **q})
                r1_identical &= bool(np.array_equal(p_at[(s, "base", 1)], p_at[(s, "full", 1)]))
            diff3 = np.stack([p_at[(s, "full", 3)] - p_at[(s, "base", 3)] for s in DEV_SEEDS])
            diff4 = np.stack([p_at[(s, "full", 4)] - p_at[(s, "base", 4)] for s in DEV_SEEDS])
            lo, hi = hierarchical_ci(diff3)
            within = {f"{c}_r{r}": float(100 * np.mean([(p_at[(s, c, r)] - p_at[(s, c, 1)]).mean()
                                                        for s in DEV_SEEDS]))
                      for c in CELLS for r in (3, 4)}
            cells.append({"pair": pair, "variant": variant, "n_videos": len(vs.indices),
                          "D": float(100 * diff3.mean()),
                          "D_by_seed": [float(100 * d.mean()) for d in diff3],
                          "ci": [100 * lo, 100 * hi], "within_arm": within,
                          "D_round4": float(100 * diff4.mean()),
                          "category_weighted_D": float(
                              100 * (diff3.mean(0) * category_weights(vs.categories)).sum()),
                          "base_rate": float(positive.mean()),
                          "round1_identical": r1_identical})
            print(json.dumps(cells[-1]), flush=True)
    result = {"check": "E003/C4 pseudo-label ranking (P3, UCF)", "code": code_version(),
              "k": K, "bootstrap": {"n": N_BOOT, "rng": "default_rng(0)",
                                    "levels": ["seed", "video (paired across seeds)"]},
              "thresholds": {"confirm_d": CONFIRM_D, "reject_ci_lower": REJECT_CI_LOWER},
              "cells": cells, "verdict": verdict(cells)}
    write_csv(E003_RESULTS / "c4_pseudo_ranking.csv", rows)
    write_json(E003_RESULTS / "c4_summary.json", result)
    print("verdict:", result["verdict"])


if __name__ == "__main__":
    main()
