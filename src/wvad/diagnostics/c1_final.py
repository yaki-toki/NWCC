"""E003/C1 final: P9-resolution verdict with the dev-seed base model (rule fixed in E002).

model AP(v, p): proxy frame AP of base's selected-epoch validation scores (recorded), mean
over seeds 0-2, on variant v's E002 C1 validation population (>= 256 frames; anomalous
validation videos with a proxy + normal validation videos). share(v, p) =
delta(v) / (AP128(v) - model AP(v, p)); delta from E002 c1_summary.json, AP128 from E002
c1_resolution.csv (part=val, category=all, S=128). A cell with a denominator <= 0 is
undefined and counts as hold. Confirm: all 8 cells delta >= 5 and share >= 0.25. Reject:
all 8 cells delta < 2 or share < 0.10. Otherwise hold.

    python -m wvad.diagnostics.c1_final --runs <local copy of ~/wvad_runs/e003>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.c0 import PROXY_FILE
from wvad.diagnostics.common import DEV_SEEDS, E002_RESULTS, E003_RESULTS, code_version, write_json
from wvad.diagnostics.proxy_eval import proxy_video_set
from wvad.diagnostics.runs import PAIRS, rows_of, run_dir, selection, val_record
from wvad.diagnostics.uca import VARIANTS, load_proxy

MIN_FRAMES = 256
CONFIRM_DELTA, CONFIRM_SHARE, REJECT_DELTA, REJECT_SHARE = 5.0, 0.25, 2.0, 0.10


def e002_inputs() -> tuple[dict[str, float], dict[str, float]]:
    summary = json.loads((E002_RESULTS / "c1_summary.json").read_text(encoding="utf-8"))
    delta = {v: summary["delta_ap_128_minus_32"][f"{v}/val"] for v in VARIANTS}
    with open(E002_RESULTS / "c1_resolution.csv", encoding="utf-8") as fh:
        ap128 = {r["variant"]: float(r["oracle_ap"]) for r in csv.DictReader(fh)
                 if r["part"] == "val" and r["category"] == "all" and r["S"] == "128"}
    return delta, ap128


def verdict(cells: list[dict]) -> str:
    if all(c["defined"] and c["delta"] >= CONFIRM_DELTA and c["share"] >= CONFIRM_SHARE
           for c in cells):
        return "confirm"
    if all(c["defined"] and (c["delta"] < REJECT_DELTA or c["share"] < REJECT_SHARE)
           for c in cells):
        return "reject"
    return "hold"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E003 C1 final (P9 resolution)")
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    manifest, proxy = load_manifest(manifest_file("ucf")), load_proxy(PROXY_FILE)
    delta, ap128 = e002_inputs()
    cells = []
    for variant in VARIANTS:
        for pair in PAIRS:
            per_seed = []
            for seed in DEV_SEEDS:
                d = run_dir(args.runs, "ucf", pair, seed, "base")
                rec = val_record(d)
                vs = proxy_video_set(manifest, proxy, variant, rec.val_indices, MIN_FRAMES)
                scores = rec.at(selection(d)["best_epoch"])[rows_of(rec.val_indices, vs.indices)]
                per_seed.append(vs.ap(scores))
            model_ap = float(np.mean(per_seed))
            denom = ap128[variant] - model_ap
            cells.append({"variant": variant, "pair": pair, "delta": delta[variant],
                          "ap128": ap128[variant], "model_ap": model_ap,
                          "model_ap_by_seed": per_seed, "defined": denom > 0,
                          "share": delta[variant] / denom if denom > 0 else None,
                          "n_anomalous": int(vs.anomalous.sum()),
                          "n_normal": int((~vs.anomalous).sum())})
            print(json.dumps(cells[-1]), flush=True)
    result = {"check": "E003/C1 final, P9 resolution (E002 PLAN C1 rule)", "code": code_version(),
              "min_frames": MIN_FRAMES,
              "thresholds": {"confirm_delta": CONFIRM_DELTA, "confirm_share": CONFIRM_SHARE,
                             "reject_delta": REJECT_DELTA, "reject_share": REJECT_SHARE},
              "cells": cells, "verdict": verdict(cells)}
    write_json(E003_RESULTS / "c1_final.json", result)
    print("verdict:", result["verdict"])


if __name__ == "__main__":
    main()
