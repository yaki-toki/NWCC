"""E002/C1: temporal-resolution ceiling under UCA proxy labels (no model, no training).

A video of T frames is cut into S segments of T // S frames, the last one also taking the
remaining T % S frames (the evaluator's rule). The oracle gives every segment the fraction
of its frames that are proxy-positive; repeated over the frames, this is the best score a
model with S segments can produce. Frame AP / AUC (percent) are pooled over all frames of
the population. Because oracle scores are constant within a segment, they are computed
exactly from per-segment positive/negative frame counts used as sample weights (identical
to expanding every frame; checked in the local unit tests).

Population (PLAN.md): anomalous training videos with a proxy in the variant and normal
training videos, all with at least 256 frames; reported for the whole training split and
for its validation part (used for the verdict).

    python -m wvad.diagnostics.c1
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from wvad.data.catalog import split_file
from wvad.data.manifests import TrainManifest, load_manifest, manifest_file
from wvad.data.splits import load_split
from wvad.diagnostics.c0 import PROXY_FILE
from wvad.diagnostics.common import E002_RESULTS, code_version, write_csv, write_json
from wvad.diagnostics.uca import VARIANTS, Proxy, load_proxy

RESOLUTIONS = (32, 64, 128, 256)
MIN_FRAMES = 256
GAP_LARGE, GAP_SMALL = 5.0, 2.0


def segment_counts(labels: np.ndarray, n_segments: int) -> tuple[np.ndarray, np.ndarray]:
    """(positive, total) frame counts per segment under the evaluator's segmentation."""
    t = len(labels)
    seg = np.minimum(np.arange(t) // (t // n_segments), n_segments - 1)
    pos = np.bincount(seg, weights=labels.astype(np.float64), minlength=n_segments)
    tot = np.bincount(seg, minlength=n_segments).astype(np.float64)
    return pos, tot


def oracle_ap_auc(videos: Sequence[np.ndarray], n_segments: int) -> dict:
    """Pooled frame AP / AUC (percent) of the segment-fraction oracle."""
    scores, ys, weights = [], [], []
    for labels in videos:
        pos, tot = segment_counts(labels, n_segments)
        frac = pos / tot
        for y, w in ((1, pos), (0, tot - pos)):
            keep = w > 0
            scores.append(frac[keep])
            ys.append(np.full(int(keep.sum()), y))
            weights.append(w[keep])
    s, y, w = np.concatenate(scores), np.concatenate(ys), np.concatenate(weights)
    return {"oracle_ap": float(100 * average_precision_score(y, s, sample_weight=w)),
            "oracle_auc": float(100 * roc_auc_score(y, s, sample_weight=w)),
            "n_frames": int(w.sum()), "n_pos_frames": int(w[y == 1].sum())}


def population(manifest: TrainManifest, proxy: Proxy, variant: str, indices: set[int],
               category: str | None = None) -> tuple[list[np.ndarray], int, int]:
    anom, n_norm = [], 0
    videos = []
    for i in sorted(indices):
        t = manifest.total_frames[i]
        if t < MIN_FRAMES:
            continue
        name = manifest.names[i]
        if manifest.labels[i] == 0:
            videos.append(np.zeros(t, dtype=bool))
            n_norm += 1
        elif proxy.has(name, variant) and category in (None, manifest.categories[i]):
            videos.append(proxy.frame_labels(name, variant, t))
            anom.append(i)
    return videos, len(anom), n_norm


def gap_label(deltas: Sequence[float]) -> str:
    if all(d >= GAP_LARGE for d in deltas):
        return "large"
    if all(d < GAP_SMALL for d in deltas):
        return "small"
    return "unclear"


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="E002 C1 resolution ceiling").parse_args(argv)
    manifest = load_manifest(manifest_file("ucf"))
    proxy = load_proxy(PROXY_FILE)
    split = load_split(split_file("ucf"))
    parts = {"all": set(range(len(manifest))), "val": set(split.val_indices)}
    categories = sorted({c for c, y in zip(manifest.categories, manifest.labels, strict=True)
                         if y == 1})
    rows, deltas = [], {}
    for variant in VARIANTS:
        for part, idx in parts.items():
            for category in [None, *categories]:
                videos, n_anom, n_norm = population(manifest, proxy, variant, idx, category)
                if n_anom == 0:
                    continue
                for s in RESOLUTIONS:
                    r = oracle_ap_auc(videos, s)
                    rows.append({"variant": variant, "part": part,
                                 "category": category or "all", "S": s,
                                 "n_anomalous": n_anom, "n_normal": n_norm, **r})
            ap = {r["S"]: r["oracle_ap"] for r in rows
                  if (r["variant"], r["part"], r["category"]) == (variant, part, "all")}
            deltas[(variant, part)] = ap[128] - ap[32]
    val_deltas = {v: deltas[(v, "val")] for v in VARIANTS}
    summary = {"check": "E002/C1 resolution ceiling (PLAN.md)", "code": code_version(),
               "min_frames": MIN_FRAMES, "resolutions": list(RESOLUTIONS),
               "delta_ap_128_minus_32": {f"{v}/{p}": round(d, 4) for (v, p), d in deltas.items()},
               "gap_label_val": gap_label(list(val_deltas.values())),
               "thresholds": {"large": GAP_LARGE, "small": GAP_SMALL},
               "note": "P9-resolution status is finished in E003 with the dev-seed base model AP"}
    write_csv(E002_RESULTS / "c1_resolution.csv", rows)
    write_json(E002_RESULTS / "c1_summary.json", summary)
    for r in rows:
        if r["category"] == "all":
            print(f"{r['variant']} {r['part']:3s} S={r['S']:3d} n={r['n_anomalous']}+"
                  f"{r['n_normal']} AP={r['oracle_ap']:.2f} AUC={r['oracle_auc']:.2f}")
    print("delta(128-32) val:", {v: round(d, 2) for v, d in val_deltas.items()},
          "->", summary["gap_label_val"])


if __name__ == "__main__":
    main()
