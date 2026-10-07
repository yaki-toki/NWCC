"""E021 dense scoring on the server (PLAN.md "실행 순서" 8 and 15; the inputs of gates (s), (s2),
(s3) and (p)). This module is frozen with the extraction, training and scoring code from the
stage-1 production to the end of stage 2 (PLAN.md two-stage code rule, gate (f)): it imports
only that code (wvad.{data,eval,extract,models,train}) and the writers of
wvad.diagnostics.common, never the local analysis (wvad.diagnostics.e021 and its helpers).

Runs (README "Stage E — E021"): <runs>/<dataset>_<pair>_s<seed>/<head>_<pipeline>_stage1 and
<head>_<pipeline> (stage 2 with ckpt_epoch_01.pt .. ckpt_epoch_30.pt from --save_epoch_ckpts).
Heads and their training features (``--features``): h0 legacy (plan stage 1), h1 d16c, h1r d16r,
hb d16m, hbm d16m2 (plan stage 2); pipelines l20 (E013) and m10 (E017: --global_mode mean
--offset_lr_mult 10); development seeds 0-2 for every head, calibration seeds 100-102 for h0, h1
and hb (SD estimation only).

Every epoch checkpoint passes the validation videos' 32-unit features of its variants through
the training validation path (wvad.eval.scoring: eval mode, fp32, autocast off, batch 32):

    h0   legacy (R0 logits; gate (s)), d16c (R1v; gate (s2)), d16r (R1rv), UCF d16r1 and d16r2
    h1   d16c (gate (s), (s2));  h1r  d16r (gate (s));  hb  d16m (gate (s));  hbm  d16m2 (gate (s))

and, for h0, h1, hb and hbm, every dense snippet of the validation videos (wvad.eval.dense:
offset-free z by the head's own path; hb and hbm through their scale-match map, the head's fixed
input layer). For hb and hbm, gate (s3) passes the scale-matched snippets averaged over J_k
(d16m) or over the m-picks (d16m2) through the 32-unit path. Validation videos only (the split's
order); no test data is read.

Per run one file <out>/stage<k>/<chain>/<head>_<pipeline>.npz (written under a temporary name,
then renamed):

    epochs, ckpt_epoch, ckpt_val_auc, ckpt_sha256       (30,) epoch checkpoints 1-30
    config                                               JSON of the checkpoints' config
    val_indices, labels, total_frames                    (n,) validation videos, split order
    variants                                             32-unit variants scored
    <variant>_logits, <variant>_probs                    (30, n, 32) float32 (M10: z + c)
    <variant>_z, <variant>_c                             M10 only: offset-free z, c(v)
    gamma                                                (30,) float32, M10 only
    dense_z, dense_counts                                (30, rows) float32 offset-free snippet
                                                         logits (videos in split order), (n,)
    s3_z                                                 hb, hbm: (30, n, 32) float32
    device, batch_size, rows_per_pass

Per dataset and pair, features_<dataset>_<pair>.npz: head-input feature statistics of the
validation videos (per-video mean L2 norm and per-dimension variance of every 32-unit variant
and of the dense snippets, raw and scale-matched; descriptives 2 and 7), the scale-match ratios,
the replaced frames of every validation snippet and the dense index columns of every video
(descriptive 5). <out>/stage<k>/meta.json lists every call (commit, environment, times, runs);
<out>/stage<k>/scores.sha256 holds the sha256 of every current file (sha256sum format).

    python -m wvad.diagnostics.e021_score --runs ~/wvad_runs/e021 --out ~/wvad_runs/e021_scores \
        --stage 1|2 [--datasets ucf xdv] [--pairs xcb_xcl xcb_siglip] [--rescore <reason>]

A rerun after an infrastructure failure is the same command: finished files are kept and only
the missing ones are scored. ``--rescore`` moves the stage's files to superseded/<UTC time>/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import shutil
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np
import torch

from wvad.data import ENCODER_PAIRS, FeatureSet, load_split, split_file
from wvad.data.catalog import dense_dir
from wvad.data.dense import VIDEO_COLUMNS, DenseSet, load_scale_map
from wvad.data.dense import variant_dir as dense_variant_dir
from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.common import code_version, gpu_environment, write_json
from wvad.eval.dense import (
    ROWS_PER_PASS,
    averaged_units,
    head_gamma,
    scale_maps,
    snippet_logits,
    standard_z,
)
from wvad.eval.scoring import BATCH_SIZE, load_head, score
from wvad.extract import grid
from wvad.paths import REPO_ROOT, load_paths
from wvad.prep.manifests import sha256_file
from wvad.train.stages import _train_val

E021_DIR = REPO_ROOT / "experiments" / "E021_dense_snippet_pilot"
DATASETS = ("ucf", "xdv")
PAIRS = ("xcb_xcl", "xcb_siglip")
HEADS = {"h0": "legacy", "h1": "d16c", "h1r": "d16r", "hb": "d16m", "hbm": "d16m2"}
STAGE_HEADS = {1: ("h0",), 2: ("h1", "h1r", "hb", "hbm")}
PIPELINES = {"l20": ("none", 1.0), "m10": ("mean", 10.0)}       # (global_mode, offset_lr_mult)
DEV_SEEDS, CAL_SEEDS = (0, 1, 2), (100, 101, 102)
HEAD_SEEDS = {"h0": DEV_SEEDS + CAL_SEEDS, "h1": DEV_SEEDS + CAL_SEEDS,
              "hb": DEV_SEEDS + CAL_SEEDS, "h1r": DEV_SEEDS, "hbm": DEV_SEEDS}
DENSE_INPUT = {"h0": "raw", "h1": "raw", "hb": "d16m", "hbm": "d16m2"}   # h1r: no dense arm
SCORING_ONLY = {"ucf": ("d16r1", "d16r2"), "xdv": ()}                  # UCF draws r = 1, 2
EPOCHS = 30
STAGE1_LAST = "ckpt_last.pt"
GATE_ERRORS = (OSError, KeyError, TypeError, ValueError, IndexError, RuntimeError, EOFError,
               zipfile.BadZipFile, pickle.UnpicklingError)
STAT_CHUNK = 16384                                 # rows per pass of the feature statistics


class RunKey(NamedTuple):
    dataset: str
    pair: str
    seed: int
    head: str
    pipeline: str

    @property
    def chain(self) -> str:
        return f"{self.dataset}_{self.pair}_s{self.seed}"

    @property
    def run(self) -> str:
        """Stage-2 run folder (the head's 30 epoch checkpoints)."""
        return f"{self.head}_{self.pipeline}"

    @property
    def stage1(self) -> str:
        return f"{self.head}_{self.pipeline}_stage1"

    @property
    def stage(self) -> int:
        return 1 if self.head in STAGE_HEADS[1] else 2


# --- layout ---------------------------------------------------------------------------------

def run_keys(stage: int, datasets: Sequence[str] = DATASETS,
             pairs: Sequence[str] = PAIRS) -> list[RunKey]:
    """Every run of a plan stage (1: 48 h0 runs; 2: 144 runs of h1, h1r, hb, hbm)."""
    return [RunKey(d, p, s, h, pl) for d in datasets for p in pairs for h in STAGE_HEADS[stage]
            for pl in PIPELINES for s in HEAD_SEEDS[h]]


def unit_variants(head: str, dataset: str) -> tuple[str, ...]:
    """32-unit validation features a head is scored on (its own variant first)."""
    if head == "h0":
        return ("legacy", "d16c", "d16r", *SCORING_ONLY[dataset])
    return (HEADS[head],)


def run_dir(runs: Path, key: RunKey) -> Path:
    return Path(runs) / key.chain / key.run


def stage1_dir(runs: Path, key: RunKey) -> Path:
    return Path(runs) / key.chain / key.stage1


def epoch_ckpt(runs: Path, key: RunKey, epoch: int) -> Path:
    return run_dir(runs, key) / f"ckpt_epoch_{epoch:02d}.pt"


def score_path(root: Path, stage: int, key: RunKey) -> Path:
    return Path(root) / f"stage{stage}" / key.chain / f"{key.run}.npz"


def features_path(root: Path, stage: int, dataset: str, pair: str) -> Path:
    return Path(root) / f"stage{stage}" / f"features_{dataset}_{pair}.npz"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def outside_repo(*paths: Path) -> None:
    """Scores stay outside the repository (and outside a deployed copy of it)."""
    repo = os.path.realpath(REPO_ROOT)
    for p in paths:
        real = os.path.realpath(p)
        if real == repo or real.startswith(repo + os.sep):
            raise SystemExit(f"{p} is inside the repository; use a folder outside it")


# --- checks before scoring ------------------------------------------------------------------

def checkpoint_problem(path: Path, key: RunKey, epoch: int) -> str | None:
    """An epoch checkpoint must name its own run: dataset, pair, seed, the head's features, the
    pipeline's offset options, save_epoch_ckpts, the run folder (out_dir), the chain's stage-1
    ckpt_last.pt (init) and its epoch; a misplaced checkpoint would score the wrong head."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        cfg = obj["config"]
        out_dir, init = PurePosixPath(cfg["out_dir"]), PurePosixPath(cfg["init"])
        got = (cfg["dataset"], cfg["pair"], int(cfg["seed"]), cfg.get("features", "legacy"),
               cfg.get("global_mode", "none"), float(cfg.get("offset_lr_mult", 1.0)),
               cfg.get("save_epoch_ckpts"), out_dir.name, init.parent.name, init.name,
               int(obj["epoch"]))
    except GATE_ERRORS as exc:
        return f"{path}: {exc!r}"
    mode, mult = PIPELINES[key.pipeline]
    want = (key.dataset, key.pair, key.seed, HEADS[key.head], mode, mult, True, key.run,
            key.stage1, STAGE1_LAST, epoch)
    return None if got == want else f"{path}: {got}, expected {want}"


def check_runs(runs: Path, keys: Sequence[RunKey]) -> None:
    """Every epoch checkpoint exists and names its run; nothing is scored otherwise."""
    missing = [f"{k.chain}/{k.run}" for k in keys
               if not all(epoch_ckpt(runs, k, e).is_file() for e in range(1, EPOCHS + 1))]
    if missing:
        raise SystemExit(f"{len(missing)} runs lack epoch checkpoints, nothing scored: {missing}")
    problems = [x for k in keys for e in range(1, EPOCHS + 1)
                if (x := checkpoint_problem(epoch_ckpt(runs, k, e), k, e))]
    if problems:
        raise SystemExit(f"{len(problems)} checkpoint problem(s), nothing scored: {problems[:5]}")


# --- features -------------------------------------------------------------------------------

def unit_features(dataset: str, pair: str, variant: str) -> FeatureSet:
    """Validation videos of a 32-unit variant exactly as training builds them
    (wvad.train.stages._train_val with the split-file label check; legacy: the earlier call)."""
    if variant == "legacy":
        return _train_val(dataset, pair)[1]
    return _train_val(dataset, pair, variant)[1]


@dataclass
class DenseVal:
    """Dense snippets of the validation videos (split order), concatenated on the device."""

    dense: DenseSet
    videos: np.ndarray            # (n,) manifest indices
    counts: np.ndarray            # (n,) snippets per video
    features: torch.Tensor        # (rows, D) float32


def dense_val(dataset: str, pair: str, videos: Sequence[int], totals: Sequence[int],
              labels: Sequence[int], mapped: str, device: torch.device) -> DenseVal:
    """``mapped``: "raw", or the variant whose scale-match maps form the input layer."""
    paths = load_paths()
    dirs = [dense_dir(paths, dataset, e) for e in ENCODER_PAIRS[pair]]
    ds = DenseSet(dirs, None if mapped == "raw" else scale_maps(dirs, mapped))
    v = np.asarray(videos, dtype=np.int64)
    if not all(ds.done(int(i)) for i in v):
        raise SystemExit(f"{dataset}/{pair}: validation videos missing from the dense caches")
    if not (np.array_equal(ds.total_frames[v], np.asarray(totals))
            and np.array_equal(ds.labels[v], np.asarray(labels))):
        raise SystemExit(f"{dataset}/{pair}: dense index T or labels differ from the manifest")
    feats = torch.cat([ds.video(int(i)) for i in v]).to(device)
    return DenseVal(ds, v, ds.counts[v].astype(np.int64), feats)


def averaged_set(dv: DenseVal, variant: str) -> FeatureSet:
    """(s3) input: the scale-matched snippets averaged per window (eval.dense.averaged_units),
    one (32, D) tensor per validation video, as eval.dense.s3_check builds it."""
    ds = dv.dense
    nan = torch.full((grid.WINDOWS,), float("nan"))
    idx = [int(i) for i in dv.videos]
    return FeatureSet(tuple(averaged_units(ds, v, variant) for v in idx),
                      tuple(int(ds.labels[v]) for v in idx), (nan,) * len(idx),
                      tuple(int(ds.total_frames[v]) for v in idx), ("dense",))


def _chunked_stats(x: torch.Tensor, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-video mean L2 norm (videos = consecutive row blocks of ``counts``) and the
    per-dimension population variance over all rows, in float64 chunks."""
    norms, total, n = [], None, x.shape[0]
    for a in range(0, n, STAT_CHUNK):
        part = x[a:a + STAT_CHUNK].detach().to("cpu", torch.float64)
        norms.append(part.norm(dim=1).numpy())
        s = part.sum(dim=0)
        total = s if total is None else total + s
    mean = total / n
    ss = None
    for a in range(0, n, STAT_CHUNK):
        part = x[a:a + STAT_CHUNK].detach().to("cpu", torch.float64) - mean
        q = (part * part).sum(dim=0)
        ss = q if ss is None else ss + q
    norm = np.concatenate(norms)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return np.add.reduceat(norm, starts) / counts, (ss / n).numpy()


def feature_stats(units: Mapping[str, FeatureSet], dense: Mapping[str, DenseVal]) -> dict:
    """Descriptives 2 and 7 (model-independent): per validation video the mean L2 norm of the
    head-input units and the per-dimension variance over all units, for every 32-unit variant
    (``<variant>_norm``, ``<variant>_dimvar``) and the dense snippets (``dense_<input>_*``); the
    scale-match ratios of the mapped inputs; the replaced frames of every validation snippet and
    the dense index columns of every training-split video (first encoder of the pair)."""
    out = {}
    for v, fs in units.items():
        x = torch.stack(fs.features).to(torch.float64)            # (n, 32, D)
        out[f"{v}_norm"] = x.norm(dim=2).mean(dim=1).numpy()
        out[f"{v}_dimvar"] = x.reshape(-1, x.shape[2]).var(dim=0, unbiased=False).numpy()
    for name, dv in dense.items():
        out[f"dense_{name}_norm"], out[f"dense_{name}_dimvar"] = _chunked_stats(dv.features,
                                                                                dv.counts)
        if name != "raw":
            out[f"scale_ratio_{name}"] = np.concatenate(
                [load_scale_map(dense_variant_dir(d, name)).ratio for d in dv.dense.directories])
    any_dv = next(iter(dense.values()), None)
    if any_dv is not None:
        ix = any_dv.dense.indexes[0]
        rows = [ix.snippet_replaced[int(ix.start[v]):int(ix.start[v] + ix.n[v])]
                for v in any_dv.videos]
        out["snippet_replaced"] = np.concatenate(rows).astype(np.uint8)
        out["dense_counts"] = any_dv.counts
        out |= {f"index_{c}": np.asarray(ix.columns[c]) for c in VIDEO_COLUMNS}
    return out


# --- scoring --------------------------------------------------------------------------------

def score_run(runs: Path, key: RunKey, units: Mapping[str, FeatureSet], dense: DenseVal | None,
              averaged: FeatureSet | None, device: torch.device) -> dict[str, np.ndarray]:
    """The 30 epoch checkpoints of one run on its 32-unit variants, the dense snippets and the
    (s3) input (see the module docstring for the arrays)."""
    m10 = PIPELINES[key.pipeline][0] != "none"
    dim = next(iter(units.values())).dim
    acc: dict[str, list] = defaultdict(list)
    config = None
    for e in range(1, EPOCHS + 1):
        path = epoch_ckpt(runs, key, e)
        model, obj = load_head(path, dim, device)
        acc["ckpt_sha256"].append(sha256_file(path))
        acc["ckpt_epoch"].append(int(obj["epoch"]))
        acc["ckpt_val_auc"].append(float(obj["val_video_auc"]))
        config = obj["config"]
        for v, fs in units.items():
            out = score(model, fs, device, BATCH_SIZE)
            acc[f"{v}_logits"].append(out["logits"])
            acc[f"{v}_probs"].append(out["probs"])
            if m10:
                acc[f"{v}_z"].append(out["local_logits"])
                acc[f"{v}_c"].append(out["offsets"])
        if m10:
            acc["gamma"].append(np.float32(head_gamma(model)))
        if dense is not None:
            acc["dense_z"].append(snippet_logits(model, dense.features, device))
        if averaged is not None:
            acc["s3_z"].append(standard_z(model, averaged, device))
    first = next(iter(units.values()))
    arrays = {k: (np.asarray(v) if k in ("ckpt_sha256", "ckpt_epoch", "ckpt_val_auc")
                  else np.stack(v).astype(np.float32)) for k, v in acc.items()}
    arrays |= {"epochs": np.arange(1, EPOCHS + 1), "variants": np.asarray(list(units)),
               "config": np.str_(json.dumps(config, sort_keys=True, default=str)),
               "labels": np.asarray(first.labels, dtype=np.int64),
               "device": np.str_(str(device)), "batch_size": np.int64(BATCH_SIZE),
               "rows_per_pass": np.int64(ROWS_PER_PASS)}
    if dense is not None:
        arrays["dense_counts"] = dense.counts
    return arrays


def save_npz(f: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write under a temporary name outside the *.npz pattern, then rename."""
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, f)


def score_group(runs: Path, out: Path, stage: int, dataset: str, pair: str,
                keys: Sequence[RunKey], device: torch.device) -> list[Path]:
    """Score the runs of one (dataset, pair); features and dense snippets are loaded once."""
    split = load_split(split_file(dataset))
    manifest = load_manifest(manifest_file(dataset))
    val = np.asarray(split.val_indices, dtype=np.int64)
    totals = [manifest.total_frames[int(i)] for i in val]
    labels = [split.labels[int(i)] for i in val]
    heads = STAGE_HEADS[stage]                      # every input of the stage (feature file)
    variants = list(dict.fromkeys(v for h in heads for v in unit_variants(h, dataset)))
    units = {v: unit_features(dataset, pair, v) for v in variants}
    for v, fs in units.items():
        if list(fs.labels) != labels or len(fs) != len(val):
            raise SystemExit(f"{dataset}/{pair}/{v}: validation videos differ from the split")
    inputs = sorted({DENSE_INPUT[h] for h in heads if h in DENSE_INPUT})
    dense = {m: dense_val(dataset, pair, val, totals, labels, m, device) for m in inputs}
    averaged = {m: averaged_set(dense[m], m) for m in inputs if m != "raw"}
    written = []
    fpath = features_path(out, stage, dataset, pair)
    if not fpath.exists():
        save_npz(fpath, feature_stats(units, dense) | {"val_indices": val})
        written.append(fpath)
    for k in keys:
        mine = {v: units[v] for v in unit_variants(k.head, dataset)}
        dv = dense.get(DENSE_INPUT.get(k.head, ""))
        arrays = score_run(runs, k, mine, dv, averaged.get(DENSE_INPUT.get(k.head, "")), device)
        arrays |= {"val_indices": val, "total_frames": np.asarray(totals, dtype=np.int64)}
        f = score_path(out, stage, k)
        save_npz(f, arrays)
        written.append(f)
        print(f"[e021_score] stage {stage} {k.chain}/{k.run}", flush=True)
    return written


# --- outputs --------------------------------------------------------------------------------

def current_files(target: Path) -> list[Path]:
    return sorted(f for f in Path(target).rglob("*.npz")
                  if "superseded" not in f.relative_to(target).parts)


def sha256_list(files: Sequence[Path], base: Path) -> str:
    rows = sorted((f.relative_to(base).as_posix(), sha256_file(f)) for f in files)
    return "".join(f"{sha}  {rel}\n" for rel, sha in rows)


def supersede(target: Path, reason: str) -> str | None:
    """--rescore: move the stage's current files aside (kept for the report)."""
    files = [f for f in Path(target).rglob("*") if f.is_file()
             and "superseded" not in f.relative_to(target).parts]
    if not files:
        return None
    dest = Path(target) / "superseded" / dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    for f in files:
        new = dest / f.relative_to(target)
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(new))
    (dest / "REASON.txt").write_text(reason + "\n", encoding="utf-8")
    return str(dest)


def read_meta(target: Path) -> dict:
    f = Path(target) / "meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {"calls": []}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E021 dense scoring of the epoch checkpoints")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="score root outside the repository")
    ap.add_argument("--stage", required=True, type=int, choices=(1, 2))
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--pairs", nargs="+", choices=PAIRS, default=list(PAIRS))
    ap.add_argument("--rescore", help="reason: move the stage's files aside and score again")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outside_repo(args.runs, args.out)
    target = Path(args.out) / f"stage{args.stage}"
    keys = run_keys(args.stage, args.datasets, args.pairs)
    check_runs(args.runs, keys)
    moved = supersede(target, args.rescore) if args.rescore else None
    todo = [k for k in keys if not score_path(args.out, args.stage, k).is_file()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    call = {"time_utc": now_utc(), "code": code_version(), "environment": gpu_environment(),
            "stage": args.stage, "datasets": list(args.datasets), "pairs": list(args.pairs),
            "runs": str(args.runs), "out": str(target), "device": str(device),
            "rescore": args.rescore, "superseded": moved,
            "kept_existing": sorted(f"{k.chain}/{k.run}" for k in keys if k not in todo)}
    written = []
    for d in args.datasets:
        for p in args.pairs:
            mine = [k for k in todo if (k.dataset, k.pair) == (d, p)]
            if mine or not features_path(args.out, args.stage, d, p).exists():
                written += score_group(args.runs, args.out, args.stage, d, p, mine, device)
    meta = read_meta(target)
    meta["calls"] = [*meta.get("calls", []), {**call, "time_utc_end": now_utc(),
                                              "written": [f.relative_to(target).as_posix()
                                                          for f in written]}]
    write_json(target / "meta.json", meta)
    (target / "scores.sha256").write_text(sha256_list(current_files(target), target),
                                          encoding="utf-8", newline="\n")
    print(f"[e021_score] stage {args.stage}: {len(written)} files -> {target}")


if __name__ == "__main__":
    main()
