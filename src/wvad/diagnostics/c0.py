"""E002/C0: alignment of the training manifests with the feature caches (gates G1-G3),
per-category proxy coverage, and the summary of all C0 gates (G4-G5 come from the local
fingerprint check, results/c0_fingerprint.json).

    python -m wvad.diagnostics.c0 --ucf_train_list <UCF-Crime root>/Anomaly_Train.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

from wvad.data.catalog import split_file
from wvad.data.features import load_cache
from wvad.data.manifests import TrainManifest, load_manifest, manifest_file
from wvad.data.splits import load_split
from wvad.diagnostics.common import E002_RESULTS, code_version, write_csv, write_json
from wvad.diagnostics.uca import Proxy, load_proxy, load_uca, sha256_file
from wvad.paths import REPO_ROOT, load_paths

PROXY_FILE = REPO_ROOT / "data" / "annotations" / "ucf_train_uca_proxy.json"
DURATION_TOL_S, DURATION_MIN_FRACTION = 1.0, 0.95


def gate_labels(manifest: TrainManifest, cache_labels: list[int]) -> dict:
    got = tuple(int(y) for y in cache_labels)
    n_bad = sum(a != b for a, b in zip(got, manifest.labels, strict=False))
    return {"pass": got == manifest.labels, "n_manifest": len(manifest),
            "n_cache": len(got), "mismatched_positions": n_bad}


def gate_anomaly_order(manifest: TrainManifest, train_list: list[str]) -> dict:
    """Anomalous manifest names, in order, equal the anomalous lines of Anomaly_Train.txt.

    The list file also holds 800 normal-video lines ('Training_Normal_Videos_Anomaly/...');
    like the previous loader, only lines whose first path component is an anomaly
    category are kept.
    """
    categories = {c for c, y in zip(manifest.categories, manifest.labels, strict=True) if y == 1}
    paths = [PurePosixPath(line.strip().replace("\\", "/")) for line in train_list if line.strip()]
    listed = [p.stem for p in paths if p.parts[0] in categories]
    ours = [n for n, y in zip(manifest.names, manifest.labels, strict=True) if y == 1]
    first_diff = next((i for i, (a, b) in enumerate(zip(ours, listed, strict=False)) if a != b),
                      None)
    return {"pass": ours == listed, "n_manifest_anomalous": len(ours), "n_list": len(listed),
            "first_difference": first_diff}


def gate_durations(manifest: TrainManifest, uca: dict) -> dict:
    diffs = []
    for name, total, fps in zip(manifest.names, manifest.total_frames, manifest.fps,
                                strict=True):
        e = uca.get(name)
        if e is not None and e.get("duration") is not None and fps:
            diffs.append(abs(float(e["duration"]) - total / fps))
    d = np.asarray(diffs)
    frac = float((d <= DURATION_TOL_S).mean()) if len(d) else 0.0
    return {"pass": frac >= DURATION_MIN_FRACTION, "n_videos_with_uca": len(d),
            "fraction_within_1s": round(frac, 4),
            "abs_diff_seconds_median": round(float(np.median(d)), 3) if len(d) else None,
            "abs_diff_seconds_max": round(float(d.max()), 3) if len(d) else None}


def proxy_by_category(manifest: TrainManifest, proxy: Proxy, uca: dict,
                      val_indices: set[int]) -> list[dict]:
    rows = []
    for cat in sorted(set(manifest.categories)):
        idx = [i for i, c in enumerate(manifest.categories) if c == cat]
        row = {"category": cat, "n": len(idx),
               "n_train": sum(i not in val_indices for i in idx),
               "n_val": sum(i in val_indices for i in idx),
               "uca": sum(manifest.names[i] in uca for i in idx)}
        for m in ("M1", "M2"):
            has = [i for i in idx if proxy.has(manifest.names[i], f"{m}A")]
            row[f"{m}_train"] = sum(i not in val_indices for i in has)
            row[f"{m}_val"] = sum(i in val_indices for i in has)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E002 C0 alignment gates")
    ap.add_argument("--ucf_train_list", required=True, type=Path)
    ap.add_argument("--uca_dir", type=Path, help="default: paths.yaml annotations.uca")
    args = ap.parse_args(argv)
    paths = load_paths()
    ucf, xdv = load_manifest(manifest_file("ucf")), load_manifest(manifest_file("xdv"))
    uca, uca_hashes = load_uca(args.uca_dir or paths.annotation_dir("uca"))
    proxy = load_proxy(PROXY_FILE)
    val = set(load_split(split_file("ucf")).val_indices)

    g1 = gate_labels(ucf, load_cache(paths.feature_dir("ucf_xcb"), "train")["labels"])
    g1["anomaly_order"] = gate_anomaly_order(
        ucf, args.ucf_train_list.read_text(encoding="utf-8").splitlines())
    g1["pass"] = g1["pass"] and g1["anomaly_order"]["pass"]
    g2 = gate_labels(xdv, load_cache(paths.feature_dir("xdv_xcb"), "train")["labels"])
    g3 = gate_durations(ucf, uca)
    gates = {"G1": g1, "G2": g2, "G3": g3}
    fp_file = E002_RESULTS / "c0_fingerprint.json"
    if fp_file.is_file():
        fp = json.loads(fp_file.read_text(encoding="utf-8"))
        gates["G4"], gates["G5"] = fp["G4"], fp["G5"]
    else:
        gates["G4"] = gates["G5"] = {"pass": False, "note": "c0_fingerprint.json not found"}

    cats = proxy_by_category(ucf, proxy, uca, val)
    normal_uca = sum(n in uca for n, y in zip(ucf.names, ucf.labels, strict=True) if y == 0)
    summary = {
        "check": "E002/C0 alignment gates (PLAN.md)", "code": code_version(),
        "inputs": {"ucf_manifest": sha256_file(manifest_file("ucf")),
                   "xdv_manifest": sha256_file(manifest_file("xdv")),
                   "proxy": sha256_file(PROXY_FILE), "ucf_train_list": sha256_file(
                       args.ucf_train_list), "uca": uca_hashes},
        "gates": gates, "all_pass": all(g["pass"] for g in gates.values()),
        "proxy_counts": {"anomalous_with_uca": sum(r["uca"] for r in cats
                                                   if r["category"] != "Normal"),
                         "normal_with_uca": normal_uca,
                         "uca_entries_ignored": proxy.meta.get("n_uca_ignored")},
        "categories": dict(Counter(ucf.categories)),
    }
    write_json(E002_RESULTS / "c0_alignment.json", summary)
    write_csv(E002_RESULTS / "c0_proxy_by_category.csv", cats)
    for k, g in gates.items():
        print(k, "PASS" if g["pass"] else "FAIL",
              {x: y for x, y in g.items() if x not in ("pass",)})
    print("all_pass", summary["all_pass"])


if __name__ == "__main__":
    main()
