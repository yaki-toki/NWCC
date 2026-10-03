"""E003/C0b: are the XCL and SigLIP caches in the same video order as the XCB cache?

For each dataset and encoder e, a ridge regression maps XCB segment features to e segment
features, fitted on segment pairs (same video, same segment index) of the training part;
inputs and outputs are standardized with training-part segment statistics and the penalty
is chosen by video-grouped 5-fold cross-validation (mean segment cosine). Each validation
video's predicted e sequence (32 x d, flattened) is ranked against the true e sequences of
all validation videos by cosine. A video passes when its own row is rank 1, or when it is
in the top 3 and the rank-1 video is a sibling (XD: same source before '__#'; UCF: cached
e rows more similar than the 99.9th percentile of validation pairs). Anomalous and normal
videos are judged separately. Positive control: the answer shifted by one video within the
label must fail under the same rule (PLAN.md, C0b).

    python -m wvad.diagnostics.c0b
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from wvad.data.catalog import split_file
from wvad.data.features import load_cache
from wvad.data.manifests import load_manifest, manifest_file
from wvad.data.splits import load_split
from wvad.diagnostics.common import E003_RESULTS, code_version, write_json
from wvad.paths import load_paths

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
FOLDS = 5
PASS_FRACTION, CONTROL_MAX_PASS = 0.80, 0.10
MAX_RANK, PERCENTILE = 3, 99.9


def stack(features: Sequence, idx: Sequence[int]) -> np.ndarray:
    return np.stack([features[i].float().numpy() for i in idx])           # (n, S, d)


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return [(a - mu) / sd for a in (train, *others)]


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge without intercept (inputs and outputs are centred)."""
    xtx = x.T @ x
    return np.linalg.solve(xtx + alpha * np.eye(len(xtx)), x.T @ y)


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float((num / den).mean())


def choose_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[float, float]]:
    """Video-grouped K-fold CV over ALPHAS; x, y are (n_videos, S, d). Folds: video i -> i % K."""
    fold = np.arange(len(x)) % FOLDS
    scores = {}
    for alpha in ALPHAS:
        cos = []
        for k in range(FOLDS):
            tr, te = fold != k, fold == k
            w = ridge(x[tr].reshape(-1, x.shape[-1]), y[tr].reshape(-1, y.shape[-1]), alpha)
            pred = x[te].reshape(-1, x.shape[-1]) @ w
            cos.append(mean_cosine(pred, y[te].reshape(-1, y.shape[-1])))
        scores[alpha] = float(np.mean(cos))
    best = max(ALPHAS, key=lambda a: (scores[a], -a))
    return best, scores


def unit_rows(a: np.ndarray) -> np.ndarray:
    flat = a.reshape(len(a), -1)
    return flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)


def judge(sim: np.ndarray, answers: np.ndarray, siblings: np.ndarray) -> dict:
    """sim: (n, n) query x candidate cosine; answers[i]: candidate counted as correct for i;
    siblings[a, b]: candidates a and b are siblings. Returns rank-1 and pass fractions."""
    order = np.argsort(-sim, axis=1, kind="stable")
    rank = np.argmax(order == answers[:, None], axis=1) + 1
    top1 = order[:, 0]
    via_sibling = (rank > 1) & (rank <= MAX_RANK) & siblings[top1, answers]
    passed = (rank == 1) | via_sibling
    return {"rank1": float((rank == 1).mean()), "pass_fraction": float(passed.mean()),
            "sibling_share": float(via_sibling.sum() / max(passed.sum(), 1)),
            "median_rank": float(np.median(rank))}


def sibling_matrix(dataset: str, names: Sequence[str], e_unit: np.ndarray) -> np.ndarray:
    if dataset == "xdv":
        src = np.array([n.split("__#")[0] for n in names])
        return src[:, None] == src[None, :]
    pair = e_unit @ e_unit.T
    iu = np.triu_indices(len(pair), k=1)
    return pair > np.percentile(pair[iu], PERCENTILE)


def check_cell(dataset: str, encoder: str, paths=None) -> list[dict]:
    paths = paths or load_paths()
    split = load_split(split_file(dataset))
    xcb = load_cache(paths.feature_dir(f"{dataset}_xcb"), "train")["features"]
    enc = load_cache(paths.feature_dir(f"{dataset}_{encoder}"), "train")["features"]
    names = load_manifest(manifest_file(dataset)).names
    tr, va = list(split.train_indices), list(split.val_indices)
    x_tr, x_va = standardize(stack(xcb, tr), stack(xcb, va))
    y_tr, y_va = standardize(stack(enc, tr), stack(enc, va))
    alpha, cv = choose_alpha(x_tr, y_tr)
    w = ridge(x_tr.reshape(-1, x_tr.shape[-1]), y_tr.reshape(-1, y_tr.shape[-1]), alpha)
    pred = (x_va.reshape(-1, x_va.shape[-1]) @ w).reshape(len(va), -1)
    true_unit = unit_rows(y_va)
    sim = unit_rows(pred) @ true_unit.T                                      # (n_val, n_val)
    sib = sibling_matrix(dataset, [names[i] for i in va], true_unit)
    labels = np.array([split.labels[i] for i in va])
    rows = []
    for label in (1, 0):
        q = np.flatnonzero(labels == label)
        real = judge(sim[q], q, sib)
        control = judge(sim[q], np.roll(q, -1), sib)                         # shifted by one
        rows.append({"dataset": dataset, "encoder": encoder,
                     "label": "anomalous" if label else "normal", "n": len(q),
                     "alpha": alpha, "cv_cosine": cv, **real,
                     "control_rank1": control["rank1"],
                     "control_pass_fraction": control["pass_fraction"],
                     "pass": real["pass_fraction"] >= PASS_FRACTION
                     and control["pass_fraction"] <= CONTROL_MAX_PASS})
    return rows


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="E003 C0b cross-encoder alignment").parse_args(argv)
    rows = []
    for dataset in ("ucf", "xdv"):
        for encoder in ("xcl", "siglip"):
            rows += check_cell(dataset, encoder)
            for r in rows[-2:]:
                print(f"{r['dataset']} {r['encoder']:6s} {r['label']:9s} n={r['n']} "
                      f"alpha={r['alpha']} rank1={r['rank1']:.3f} pass={r['pass_fraction']:.3f} "
                      f"(sibling {r['sibling_share']:.3f}) control={r['control_pass_fraction']:.3f}"
                      f" -> {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    summary = {"check": "E003/C0b cross-encoder alignment (PLAN.md)", "code": code_version(),
               "alphas": list(ALPHAS), "folds": FOLDS,
               "fold_rule": "training-part video i -> i % 5",
               "thresholds": {"pass_fraction": PASS_FRACTION,
                              "control_max_pass": CONTROL_MAX_PASS, "max_rank": MAX_RANK,
                              "ucf_sibling_percentile": PERCENTILE},
               "cells": rows, "all_pass": all(r["pass"] for r in rows)}
    write_json(E003_RESULTS / "c0b_cross_encoder.json", summary)
    print("all_pass", summary["all_pass"])


if __name__ == "__main__":
    main()
