"""E004/C6: does temporal context add information? Linear probe under UCA proxy labels.

UCF training split only: probes are fitted on the training part and scored on the
validation part. Segment labels (evaluator segment rule): positive = >= 50% proxy frames
(strict: 100%), negative = every segment of a normal video and zero-overlap segments of
anomalous videos with a proxy; partial overlaps and anomalous videos without a proxy are
left out. Inputs (each block standardized with its own training-part statistics):
(i) x_t, (ii) [x_t, x_{t-1}], (iii) [x_t, mean x_{t-4..t-1}], (iv) [x_t, x_{t-1}, x_{t+1}],
(v) [x_t, mean of 4 random same-video segments with |j - t| > 4] (control).
L2 logistic regression (lbfgs, max_iter 2000, tol 1e-4); C per (pair, input, shape) by
5-fold CV (folds contiguous within label and category), with a one-time grid-edge
extension. Verdicts and the stratified paired video bootstrap follow PLAN.md (E004).
Standardized blocks are written once as .npy files and memory-mapped by the parallel fits;
every fit is saved with the commit and the config hash (resume refuses a mismatch).

    python -m wvad.diagnostics.c6 [--out F:/Vision/wvad_runs/e004]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from wvad.data import feature_dirs, load_feature_set, load_split, split_file
from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.c0 import PROXY_FILE
from wvad.diagnostics.common import code_version, write_csv, write_json
from wvad.diagnostics.proxy_eval import segment_frame_counts
from wvad.diagnostics.uca import VARIANTS, load_proxy
from wvad.paths import REPO_ROOT, load_paths

EXP = REPO_ROOT / "experiments" / "E004_temporal_context_probe"
E004_RESULTS, CONFIG = EXP / "results", EXP / "config.yaml"
PAIRS = ("xcb_xcl", "xcb_siglip")
INPUTS = ("i", "ii", "iii", "iv", "v")
STRICT_INPUTS = ("i", "iii", "v")
INPUT_BLOCKS = {"i": ("x",), "ii": ("x", "prev1"), "iii": ("x", "past4"),
                "iv": ("x", "prev1", "next1"), "v": ("x", "rand4")}
SHAPE_LABELS = {"A": "M1A", "B": "M1B"}
C_GRID = (0.01, 0.1, 1.0)
EDGE = {0.01: 0.001, 1.0: 10.0}
FOLDS, N_BOOT, RAND_SEED, PAST, GAP, N_RAND = 5, 10_000, 20261003, 4, 4, 4
PARALLEL, BLAS_THREADS = 4, 8
CONFIRM_GAIN, ATTR_SCENE_UPPER = 1.0, 0.5


# ---------------------------------------------------------------- labels and inputs
def segment_labels(frac: np.ndarray, strict: bool) -> np.ndarray:
    """1 positive, 0 negative, -1 excluded, from per-segment proxy-frame fractions."""
    pos = frac >= 1.0 if strict else frac >= 0.5
    return np.where(pos, 1, np.where(frac == 0, 0, -1)).astype(np.int8)


def video_labels(manifest, proxy, variant: str, strict: bool) -> np.ndarray:
    """(n_videos, 32) labels for every training-cache video."""
    out = np.full((len(manifest), 32), -1, dtype=np.int8)
    for i in range(len(manifest)):
        if manifest.labels[i] == 0:
            out[i] = 0
        elif proxy.has(manifest.names[i], variant):
            lab = proxy.frame_labels(manifest.names[i], variant, manifest.total_frames[i])
            pos, neg = segment_frame_counts(lab)
            out[i] = segment_labels(pos / (pos + neg), strict)
    return out


def random_table(n_videos: int, n_seg: int = 32) -> np.ndarray:
    """(n_videos, S, 4) random same-video segments with |j - t| > GAP (fixed draw)."""
    rng = np.random.default_rng(RAND_SEED)
    table = np.empty((n_videos, n_seg, N_RAND), dtype=np.int64)
    for v in range(n_videos):
        for t in range(n_seg):
            cand = [j for j in range(n_seg) if abs(j - t) > GAP]
            table[v, t] = rng.choice(cand, N_RAND, replace=False)
    return table


def blocks(feats: np.ndarray, rand: np.ndarray) -> dict[str, np.ndarray]:
    """Per-video blocks, each (n, S, D); edges padded by repetition."""
    s = feats.shape[1]
    idx = np.arange(s)
    past = np.stack([feats[:, np.maximum(idx - k, 0)] for k in range(1, PAST + 1)]).mean(0)
    rnd = np.take_along_axis(feats[:, None], rand[..., None], axis=2).mean(axis=2)
    return {"x": feats, "prev1": feats[:, np.maximum(idx - 1, 0)],
            "next1": feats[:, np.minimum(idx + 1, s - 1)], "past4": past, "rand4": rnd}


def standardize_blocks(blk: dict[str, np.ndarray], fit_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Every block with its own per-dimension statistics over the training-part segments."""
    out = {}
    for name, b in blk.items():
        flat = b[fit_rows].reshape(-1, b.shape[-1])
        mu, sd = flat.mean(0), flat.std(0)
        out[name] = ((b - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)
    return out


def fold_ids(manifest, train_idx: np.ndarray) -> np.ndarray:
    """Contiguous folds within label and category (rank r of n in its group -> 5r // n)."""
    folds = np.empty(len(train_idx), dtype=np.int8)
    groups: dict[str, list[int]] = {}
    for row, i in enumerate(train_idx):
        groups.setdefault(manifest.categories[i], []).append(row)
    for rows in groups.values():
        for r, row in enumerate(rows):
            folds[row] = (FOLDS * r) // len(rows)
    return folds


def check_folds(folds: np.ndarray, labels: np.ndarray, is_normal: np.ndarray) -> None:
    """Every fold needs positives and normal-video negatives (labels: training-part rows)."""
    for k in range(FOLDS):
        sel = folds == k
        if not (labels[sel] == 1).any() or not (labels[sel & is_normal] == 0).any():
            raise ValueError(f"fold {k} lacks positives or normal-video negatives")


# ---------------------------------------------------------------- fitting (worker side)
def design(block_files: dict[str, str], inp: str, rows: np.ndarray, labels: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labeled segments of videos ``rows``: X (float32), y, video index per segment."""
    lab = labels[rows]
    keep = lab >= 0
    parts = [np.load(block_files[b], mmap_mode="r")[rows][keep] for b in INPUT_BLOCKS[inp]]
    vid = np.repeat(rows[:, None], lab.shape[1], axis=1)[keep]
    return np.concatenate(parts, axis=1).astype(np.float32), lab[keep], vid


def probe_job(block_files: dict[str, str], inp: str, fit_rows: np.ndarray,
              eval_rows: np.ndarray, labels: np.ndarray, c: float, mode: str) -> dict:
    """Fit on fit_rows, score eval_rows. mode 'cv' returns the held-out AP as 'value'."""
    x, y, _ = design(block_files, inp, fit_rows, labels)
    xe, ye, _ = design(block_files, inp, eval_rows, labels)
    model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, tol=1e-4,
                               class_weight=None, C=c)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    scores = model.predict_proba(xe)[:, 1].astype(np.float32)
    out = {"n_iter": int(model.n_iter_[0]),
           "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught)}
    if mode == "cv":
        return out | {"value": np.array([100 * average_precision_score(ye, scores)])}
    return out | {"value": scores}


class FitStore:
    """One npz per fit; refuses results from another commit or configuration."""

    def __init__(self, root: Path, commit: str, config_hash: str):
        self.root, self.commit, self.config_hash = root / "fits", commit, config_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> dict | None:
        p = self.root / f"{key}.npz"
        if not p.is_file():
            return None
        z = np.load(p)
        if str(z["commit"]) != self.commit or str(z["config_hash"]) != self.config_hash:
            raise ValueError(f"{p}: saved by another commit or configuration; "
                             f"refusing to resume")
        return {"value": z["value"], "n_iter": int(z["n_iter"]),
                "converged": bool(z["converged"])}

    def save(self, key: str, res: dict) -> None:
        np.savez(self.root / f"{key}.npz", value=res["value"], n_iter=res["n_iter"],
                 converged=res["converged"], commit=self.commit, config_hash=self.config_hash)


def run_jobs(store: FitStore, jobs: dict[str, tuple]) -> dict[str, dict]:
    """jobs: key -> probe_job arguments. Missing keys are fitted in parallel and saved."""
    done = {k: store.load(k) for k in jobs}
    todo = [k for k in jobs if done[k] is None]
    if todo:
        with parallel_config(backend="loky", inner_max_num_threads=BLAS_THREADS):
            results = Parallel(n_jobs=PARALLEL, verbose=5, return_as="generator")(
                delayed(probe_job)(*jobs[k]) for k in todo)
            for k, r in zip(todo, results, strict=True):
                store.save(k, r)
                done[k] = r
    return done


# ---------------------------------------------------------------- metrics and bootstrap
@dataclass(frozen=True)
class Sorted:
    """Score order and tie-group ends, computed once for repeated weighted AP."""

    order: np.ndarray
    last: np.ndarray

    @classmethod
    def of(cls, scores: np.ndarray) -> Sorted:
        order = np.argsort(-scores, kind="mergesort")
        s = scores[order]
        return cls(order, np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1])

    def ap(self, pos_w: np.ndarray, neg_w: np.ndarray) -> float:
        tp = np.cumsum(pos_w[self.order])[self.last]
        fp = np.cumsum(neg_w[self.order])[self.last]
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return float(100 * np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * prec))


def bootstrap_deltas(y: np.ndarray, vid: np.ndarray, normal: np.ndarray,
                     scores: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> dict:
    """Stratified (normal / anomalous videos) paired video bootstrap of AP(a) - AP(b).
    Resampling videos with replacement = weighting each segment by its video's count."""
    videos, seg_video = np.unique(vid, return_inverse=True)
    is_norm = np.isin(videos, normal)
    strata = [s for s in (np.flatnonzero(is_norm), np.flatnonzero(~is_norm)) if len(s)]
    yf = y.astype(np.float64)
    srt = {k: Sorted.of(v) for k, v in scores.items()}
    ones = np.ones(len(y))
    point = {f"{a}-{b}": srt[a].ap(yf, 1 - yf) - srt[b].ap(yf, 1 - yf) for a, b in pairs}
    boots = {k: np.empty(N_BOOT) for k in point}
    rng = np.random.default_rng(0)
    for t in range(N_BOOT):
        counts = np.zeros(len(videos))
        for s in strata:
            np.add.at(counts, rng.choice(s, len(s)), 1)
        w = counts[seg_video] * ones
        aps = {k: srt[k].ap(w * yf, w * (1 - yf)) for k in srt}
        for a, b in pairs:
            boots[f"{a}-{b}"][t] = aps[a] - aps[b]
    return {k: {"delta": float(point[k]),
                "ci": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]}
            for k in point}


def verdicts(cells: list[dict]) -> dict:
    ctx = [c["deltas"]["iii-i"] for c in cells]
    ctl = [c["deltas"]["iii-v"] for c in cells]
    if all(d["delta"] >= CONFIRM_GAIN and d["ci"][0] > 0 for d in ctx):
        p6 = "confirm"
    elif all(d["ci"][1] < CONFIRM_GAIN for d in ctx):
        p6 = "reject"
    else:
        p6 = "hold"
    if p6 != "confirm":
        attribution = "not applicable"
    elif all(d["ci"][0] > 0 for d in ctl):
        attribution = "local"
    elif all(d["ci"][1] < ATTR_SCENE_UPPER for d in ctl):
        attribution = "scene"
    else:
        attribution = "mixed"
    return {"p6": p6, "attribution": attribution}


# ---------------------------------------------------------------- main
def choose_c(store: FitStore, files: dict, pair: str, inp: str, shape: str,
             train_idx: np.ndarray, folds: np.ndarray, labels: np.ndarray,
             cv_rows: list[dict]) -> float:
    def evaluate(cs) -> dict[float, float]:
        jobs = {f"cv_{pair}_{inp}_{shape}_C{c:g}_f{k}":
                (files, inp, train_idx[folds != k], train_idx[folds == k], labels, c, "cv")
                for c in cs for k in range(FOLDS)}
        res = run_jobs(store, jobs)
        means = {}
        for c in cs:
            ap = [float(res[f"cv_{pair}_{inp}_{shape}_C{c:g}_f{k}"]["value"][0])
                  for k in range(FOLDS)]
            means[c] = float(np.mean(ap))
            cv_rows.append({"pair": pair, "input": inp, "shape": shape, "C": c,
                            **{f"fold{k}": a for k, a in enumerate(ap)}, "mean_ap": means[c]})
        return means

    means = evaluate(C_GRID)
    best = max(means, key=lambda c: (means[c], -c))
    if best in EDGE:                                    # one-time grid-edge extension
        means |= evaluate((EDGE[best],))
        best = max(means, key=lambda c: (means[c], -c))
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E004 C6 temporal-context probe")
    ap.add_argument("--out", type=Path, default=Path("F:/Vision/wvad_runs/e004"))
    ap.add_argument("--allow_dirty", action="store_true", help="for testing only")
    args = ap.parse_args(argv)
    code = code_version()
    if code["dirty"] and not args.allow_dirty:
        raise SystemExit("uncommitted changes: commit first (fits record the commit)")
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    store = FitStore(args.out, str(code["commit"]), config_hash)
    manifest, proxy = load_manifest(manifest_file("ucf")), load_proxy(PROXY_FILE)
    split = load_split(split_file("ucf"))
    train_idx, val_idx = np.asarray(split.train_indices), np.asarray(split.val_indices)
    rand = random_table(len(manifest))
    labels = {(v, s): video_labels(manifest, proxy, v, s) for v in VARIANTS
              for s in (False, True)}
    is_normal = np.array([y == 0 for y in manifest.labels])
    folds = fold_ids(manifest, train_idx)
    for v in VARIANTS:
        check_folds(folds, labels[(v, False)][train_idx], is_normal[train_idx])
    normal_val = val_idx[is_normal[val_idx]]

    probe_rows, cv_rows, cells, chosen = [], [], [], {}
    for pair in PAIRS:
        fs = load_feature_set(feature_dirs(load_paths(), "ucf", pair), "train")
        feats = np.stack([f.numpy() for f in fs.features]).astype(np.float32)
        std = standardize_blocks(blocks(feats, rand), train_idx)
        files = {}
        (args.out / "blocks").mkdir(parents=True, exist_ok=True)
        for name, b in std.items():
            files[name] = str(args.out / "blocks" / f"{pair}_{name}.npy")
            np.save(files[name], b)
        del feats, std
        for inp in INPUTS:
            for shape, var in SHAPE_LABELS.items():
                chosen[(pair, inp, shape)] = choose_c(store, files, pair, inp, shape, train_idx,
                                                      folds, labels[(var, False)], cv_rows)
        jobs, keys = {}, {}
        for kind, inputs in (("standard", INPUTS), ("strict", STRICT_INPUTS)):
            for var in VARIANTS:
                lab = labels[(var, kind == "strict")]
                for inp in inputs:
                    c = chosen[(pair, inp, var[2])]
                    key = f"final_{pair}_{kind}_{var}_{inp}_C{c:g}"
                    jobs[key] = (files, inp, train_idx, val_idx, lab, c, "final")
                    keys[(kind, var, inp)] = key
        res = run_jobs(store, jobs)
        for kind, inputs in (("standard", INPUTS), ("strict", STRICT_INPUTS)):
            for var in VARIANTS:
                lab = labels[(var, kind == "strict")]
                lv = lab[val_idx]
                keep = lv >= 0
                ye = lv[keep]
                vid = np.repeat(val_idx[:, None], 32, axis=1)[keep]
                anom = ~np.isin(vid, normal_val)
                scores = {inp: res[keys[(kind, var, inp)]]["value"] for inp in inputs}
                for inp in inputs:
                    r = res[keys[(kind, var, inp)]]
                    probe_rows.append({
                        "pair": pair, "input": inp, "variant": var, "labels": kind,
                        "C": chosen[(pair, inp, var[2])], "n_val_pos": int((ye == 1).sum()),
                        "n_val_neg": int((ye == 0).sum()),
                        "ap": float(100 * average_precision_score(ye, scores[inp])),
                        "ap_within_anomalous": float(100 * average_precision_score(
                            ye[anom], scores[inp][anom])),
                        "n_iter": r["n_iter"], "converged": r["converged"]})
                pairs = ([("iii", "i"), ("iii", "v"), ("ii", "i"), ("iv", "i")]
                         if kind == "standard" else [("iii", "i"), ("iii", "v")])
                within = {k: scores[k][anom] for k in ("i", "iii", "v")}
                cells.append({"pair": pair, "variant": var, "labels": kind,
                              "deltas": bootstrap_deltas(ye, vid, normal_val, scores, pairs),
                              "deltas_within_anomalous": bootstrap_deltas(
                                  ye[anom], vid[anom], normal_val, within,
                                  [("iii", "i"), ("iii", "v")])})
                print(pair, kind, var, json.dumps({k: round(d["delta"], 3) for k, d in
                                                   cells[-1]["deltas"].items()}), flush=True)

    result = {"check": "E004/C6 temporal-context probe (PLAN.md)", "code": code,
              "config_sha256": config_hash,
              "chosen_C": {f"{p}/{i}/{s}": c for (p, i, s), c in chosen.items()},
              "cells": cells, "verdicts": verdicts([c for c in cells if c["labels"] == "standard"]),
              "n_final_fits_not_converged": int(sum(not r["converged"] for r in probe_rows))}
    write_csv(E004_RESULTS / "c6_probe.csv", probe_rows)
    write_csv(E004_RESULTS / "c6_cv.csv", cv_rows)
    write_json(E004_RESULTS / "c6_summary.json", result)
    print("verdicts:", result["verdicts"])


if __name__ == "__main__":
    main()
