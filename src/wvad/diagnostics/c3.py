"""E003/C3: is score saturation the cause of the half-precision differences (P11)?

Scores of each run's selected checkpoint (base and full ckpt_best_val.pt) on validation
videos, three ways: F = fp32 (recorded), R = F rounded to fp16, A = one video at a time
with the whole forward pass under autocast (``rescore``, server GPU; the previous
evaluator's way). Measures: saturation (share of all validation segment scores >= 0.999,
<= 0.001, == 1.0 in F), fp16 ties (1 - distinct(R) / distinct(F)), and for UCF the proxy
frame AP of F, R, A (dR = AP(R) - AP(F), dA = AP(A) - AP(F), per variant).
Verdict on the reference cell UCF XCB+SigLIP base, seed means of signed dR, dA:
not reproduced: all variants |dA| < 0.2; confirm: all variants |dA| >= 0.5, dR and dA of the
same sign with dR/dA >= 0.5, and saturation (>= 0.999) >= 10%; reject: all variants
|dA| >= 0.5 and dR/dA < 0.2 ("reject (offset)" when some variant has the opposite sign with
|dR| >= 0.5); otherwise hold (PLAN.md, C3).

    python -m wvad.diagnostics.c3 rescore --runs ~/wvad_runs/e003     # server GPU
    python -m wvad.diagnostics.c3 analyze --runs <local copy>          # after copying back
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.c0 import PROXY_FILE
from wvad.diagnostics.common import DEV_SEEDS, E003_RESULTS, code_version, write_csv, write_json
from wvad.diagnostics.proxy_eval import proxy_video_set
from wvad.diagnostics.runs import CELLS, PAIRS, rows_of, run_dir, selection, val_record
from wvad.diagnostics.uca import VARIANTS, load_proxy

DATASETS = ("ucf", "xdv")
REFERENCE = ("ucf", "xcb_siglip", "base")
NOT_REPRODUCED, EFFECT, RATIO_CONFIRM, RATIO_REJECT, SATURATION = 0.2, 0.5, 0.5, 0.2, 0.10


def rescore(runs: Path) -> None:
    from wvad.models import SegmentHead
    from wvad.train.stages import INPUT_LAYER_NORM, _train_val

    device = torch.device("cuda")
    for dataset in DATASETS:
        for pair in PAIRS:
            _, val_fs, split, dim = _train_val(dataset, pair)
            for seed in DEV_SEEDS:
                for cell in CELLS:
                    d = run_dir(runs, dataset, pair, seed, cell)
                    ckpt = torch.load(d / "ckpt_best_val.pt", map_location="cpu",
                                      weights_only=True)
                    if ckpt["val_split_sha"] != split.sha256 or tuple(split.val_indices) != \
                            tuple(int(i) for i in val_record(d).val_indices):
                        raise ValueError(f"{d}: checkpoint split or validation order differs")
                    model = SegmentHead(dim, use_input_ln=INPUT_LAYER_NORM[pair])
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.to(device).eval()
                    out = []
                    with torch.no_grad():
                        for feats in val_fs.features:
                            with autocast(device_type="cuda", enabled=True):
                                _, _, seg = model(feats.float().unsqueeze(0).to(device))
                            out.append(seg[0].float().cpu().numpy())
                    np.savez_compressed(d / "record_val_autocast.npz", epoch=ckpt["epoch"],
                                        scores=np.stack(out),
                                        val_indices=np.asarray(split.val_indices))
                    print(f"rescored {d}", flush=True)


def saturation(f: np.ndarray) -> dict:
    r = f.astype(np.float16).astype(np.float32)
    return {"sat_hi": float((f >= 0.999).mean()), "sat_lo": float((f <= 0.001).mean()),
            "exact_one": float((f == 1.0).mean()),
            "fp16_tie_loss": float(1 - len(np.unique(r)) / len(np.unique(f)))}


def verdict(ref: dict[str, dict]) -> str:
    da = {v: ref[v]["dA"] for v in VARIANTS}
    dr = {v: ref[v]["dR"] for v in VARIANTS}
    if all(abs(da[v]) < NOT_REPRODUCED for v in VARIANTS):
        return "not reproduced"
    big = all(abs(da[v]) >= EFFECT for v in VARIANTS)
    if big and all(dr[v] * da[v] > 0 and dr[v] / da[v] >= RATIO_CONFIRM for v in VARIANTS) \
            and ref["sat_hi"] >= SATURATION:
        return "confirm"
    if big and all(dr[v] / da[v] < RATIO_REJECT for v in VARIANTS):
        offset = any(dr[v] * da[v] < 0 and abs(dr[v]) >= EFFECT for v in VARIANTS)
        return "reject (offset)" if offset else "reject"
    return "hold"


def analyze(runs: Path) -> None:
    manifest, proxy = load_manifest(manifest_file("ucf")), load_proxy(PROXY_FILE)
    rows = []
    for dataset in DATASETS:
        for pair in PAIRS:
            for cell in CELLS:
                for seed in DEV_SEEDS:
                    d = run_dir(runs, dataset, pair, seed, cell)
                    rec = val_record(d)
                    epoch = selection(d)["best_epoch"]
                    f = rec.at(epoch)
                    z = np.load(d / "record_val_autocast.npz")
                    if int(z["epoch"]) != epoch or not np.array_equal(z["val_indices"],
                                                                       rec.val_indices):
                        raise ValueError(f"{d}: autocast scores do not match the selection")
                    a = z["scores"]
                    row = {"dataset": dataset, "pair": pair, "cell": cell, "seed": seed,
                           "epoch": epoch, **saturation(f)}
                    if dataset == "ucf":
                        r = f.astype(np.float16).astype(np.float32)
                        for v in VARIANTS:
                            vs = proxy_video_set(manifest, proxy, v, rec.val_indices)
                            k = rows_of(rec.val_indices, vs.indices)
                            ap_f = vs.ap(f[k])
                            row |= {f"ap_F_{v}": ap_f, f"dR_{v}": vs.ap(r[k]) - ap_f,
                                    f"dA_{v}": vs.ap(a[k]) - ap_f}
                    rows.append(row)
    ref_rows = [r for r in rows if (r["dataset"], r["pair"], r["cell"]) == REFERENCE]
    ref = {v: {"dR": float(np.mean([r[f"dR_{v}"] for r in ref_rows])),
               "dA": float(np.mean([r[f"dA_{v}"] for r in ref_rows]))} for v in VARIANTS}
    ref["sat_hi"] = float(np.mean([r["sat_hi"] for r in ref_rows]))
    result = {"check": "E003/C3 saturation and half precision (P11)", "code": code_version(),
              "reference_cell": dict(zip(("dataset", "pair", "cell"), REFERENCE, strict=True)),
              "reference_seed_means": ref,
              "thresholds": {"not_reproduced": NOT_REPRODUCED, "effect": EFFECT,
                             "ratio_confirm": RATIO_CONFIRM, "ratio_reject": RATIO_REJECT,
                             "saturation": SATURATION},
              "verdict": verdict(ref)}
    write_csv(E003_RESULTS / "c3_saturation.csv", rows)
    write_json(E003_RESULTS / "c3_summary.json", result)
    print(json.dumps(result["reference_seed_means"]), "->", result["verdict"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E003 C3 saturation and half precision")
    ap.add_argument("action", choices=["rescore", "analyze"])
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    (rescore if args.action == "rescore" else analyze)(args.runs)


if __name__ == "__main__":
    main()
