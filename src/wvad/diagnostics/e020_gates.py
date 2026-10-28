"""E020 gates (PLAN.md gate table): (a)-(f) on the training outputs, (t) on the test data
(manifests, annotations, caches, label order; no model score), (v0) against P000 (validation
selection here, stage-2 weights from the server's validation scoring), (s) on the server's
validation re-scoring (with the path (ii) pre-check against P000's validation AUC) and (v) on
the base cell's test scores against P000, checked before the later cells are scored. Used by
wvad.diagnostics.e020.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from wvad.data import DATASETS, load_split, split_file
from wvad.data.eval_manifests import EvalManifest, eval_manifest_file, load_eval_manifest
from wvad.data.features import cache_file, load_cache
from wvad.diagnostics.common import write_json
from wvad.diagnostics.e010 import check_selection
from wvad.diagnostics.e013 import same_record_at
from wvad.diagnostics.e014 import GATE_ERRORS
from wvad.diagnostics.e015 import ENV_KEYS, identity_target, same_env
from wvad.diagnostics.e020_score import (
    CELLS,
    ENCODERS,
    LAST,
    P000_CELL,
    P000_FEAT,
    P000_KEYS,
    P000_ROLE,
    PATHS,
    RULES,
    RUN_KEYS,
    SEEDS,
    TEST_CACHE_SHA256,
    Key,
    base_keys,
    chain_name,
    current_files,
    expected_files,
    run_name,
    scorable_rules,
    score_path,
)
from wvad.diagnostics.e020_seal import PATH_II_THREADS, THREAD_VARS
from wvad.diagnostics.runs import PAIRS, selection
from wvad.eval.metrics import frame_auc_ap, max_video_auc
from wvad.paths import REPO_ROOT, load_paths
from wvad.prep.manifests import sha256_file

E019_IDENTITY = (REPO_ROOT / "experiments" / "E019_encoder_late_fusion" / "results"
                 / "e019_identity.json")
P000_RESULTS = REPO_ROOT / "experiments" / "P000_prior_validation_ablation" / "results"
# gate (c) references: E019 no-change gate chains of the deployed commit (e019_gate_70d41384)
GATE_CHAIN = "ucf_xcb_xcl_s0"                      # stage1, l20
M10_CHAIN = {"ucf": "ucf_xcb_xcl_s0", "xdv": "xdv_xcb_siglip_s0"}      # m10_stage1, m10
REF_OF = {"l20_stage1": "l20_stage1", "l20": "l20", "base": "l20",
          "m10_stage1": "m10_stage1", "m10": "m10", "m10b": "m10"}
STAGE1 = ("l20_stage1", "m10_stage1")
START_EPOCH, STAGE2_EPOCHS, SEGMENTS = 20, 30, 32
IGNORED = ("seed", "out_dir", "init", "dataset", "pair")
OFFSET_LR_MULT = 10.0
S_TOL, V_TOL = 1e-6, 1e-4
V_METRICS = ("frame_auc", "frame_ap")
PRE_SCORING = ("a", "b", "c", "d", "e", "f", "t")    # blocking; (v0) only selects (v)'s runs
BATCH_1348 = ("base", "m10b")             # 13:48 batch: stage 2 only, one log per run
# gate (t): expected test data (PLAN.md; manifest meta records the source and annotation files)
TEST_EXPECTED = {
    "ucf": {"n": 290, "anomalous": 140, "frames": 1_111_808,
            "source": "44db7fa4a212f9f965f447cfa560457f5963c7ab51fb13335460b872700f1b9a",
            "annotation": "3b9542413f2ed9e94f73bf0488c151b3d7d595a8d2ad30f524e9243a2ae2a17c"},
    "xdv": {"n": 800, "anomalous": 500, "frames": 2_335_801,
            "source": "f51a61c68bf4f340c4c824e3129d1cd8180cb2bd2ceed0d1d40f0377b650c4ab",
            "annotation": "7e6b430c3c49749ccf66ff516de8930bdd5dd475abf5929b74938a399db63599"},
}


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _config(d: Path) -> dict:
    return torch.load(d / LAST, map_location="cpu", weights_only=True)["config"]


def _gamma(path: Path) -> float:
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    return float(state["offset_gamma"])


def val_split(dataset: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Validation indices (split-file order), their video labels and the split identity."""
    split = load_split(split_file(dataset))
    idx = np.asarray(split.val_indices)
    return idx, np.array([split.labels[i] for i in idx]), split.sha256


# --- gates (a)-(f) --------------------------------------------------------------------------

def references(gate_root: Path, dataset: str) -> dict:
    """Gate (c) references (PLAN.md): base, l20 and l20_stage1 against the E019 no-change gate
    chain ucf_xcb_xcl_s0 {l20, stage1}; m10 and m10_stage1 against the same gate's M10 chain of
    the dataset (UCF: ucf_xcb_xcl_s0, XD: xdv_xcb_siglip_s0) {m10, m10_stage1}."""
    g, m = Path(gate_root) / GATE_CHAIN, Path(gate_root) / M10_CHAIN[dataset]
    return {"l20_stage1": _config(g / "stage1"), "l20": _config(g / "l20"),
            "m10_stage1": _config(m / "m10_stage1"), "m10": _config(m / "m10")}


def same_except(cfg: Mapping, ref: Mapping, ignored: Sequence[str] = IGNORED) -> bool:
    """Equal key sets and values except the ignored keys."""
    a, b = set(cfg) - set(ignored), set(ref) - set(ignored)
    return a == b and all(cfg[k] == ref[k] for k in a)


def is_m10(cfg: Mapping) -> bool:
    return cfg.get("global_mode") == "mean" and cfg.get("offset_lr_mult") == OFFSET_LR_MULT


def config_ok(folder: str, cfgs: Mapping[str, dict], refs: Mapping[str, dict],
              ident: tuple[str, str, int]) -> bool:
    """Gate (c) for one run folder: its own dataset / pair / seed; equal to its reference except
    seed, out_dir, init, dataset and pair (key set included); m10b equal to the chain's own m10
    config except init and out_dir (mean offset, multiplier 10)."""
    x = cfgs[folder]
    if (x.get("dataset"), x.get("pair"), x.get("seed")) != tuple(ident):
        return False
    if folder in ("m10_stage1", "m10", "m10b") and not is_m10(x):
        return False
    if folder == "m10b" and "m10" in cfgs:
        return same_except(x, cfgs["m10"], ("init", "out_dir"))
    return same_except(x, refs[REF_OF[folder]])


def init_is_own(d1: Path, d2: Path, ckpt: str) -> bool:
    """Gate (b): the stage-2 init is this chain's stage-1 checkpoint ``ckpt``, compared as the
    full path recorded at training time (E018 code review G3)."""
    out1 = PurePosixPath(_config(d1)["out_dir"])
    init = PurePosixPath(_config(d2)["init"])
    return out1.parts[-2:] == (d1.parent.name, d1.name) and init == out1 / ckpt


def selection_ok(d2: Path, labels: np.ndarray, val_idx: np.ndarray, split_sha: str) -> bool:
    """Gate (d): the stage-2 record (epochs 0-30, split order) reproduces selection.json (E010
    check); a non-finite record is a numerical failure, handled by the analysis."""
    z, sel = _npz(d2 / "record_val_scores.npz"), selection(d2)
    if not (np.array_equal(z["epochs"], np.arange(0, STAGE2_EPOCHS + 1))
            and np.array_equal(z["val_indices"], val_idx) and sel["val_split_sha"] == split_sha):
        return False
    scores = z["scores"][1:]
    if not np.isfinite(scores).all():
        return True
    try:
        check_selection(d2, scores, labels, sel)
    except ValueError:
        return False
    return True


def gamma_learned(d1: Path, d2: Path, start_ckpt: str) -> bool:
    """Gate (f): gamma non-zero at the end of stage 1 and moved in stage 2 (stage-2 end vs the
    cell's own start checkpoint); a non-finite gamma passes here (numerical failure)."""
    end1, start, end2 = _gamma(d1 / LAST), _gamma(d1 / start_ckpt), _gamma(d2 / LAST)
    if not all(math.isfinite(g) for g in (end1, start, end2)):
        return True
    return end1 != 0 and end2 != start


def gate_chain(runs: Path, refs: Mapping, ident: tuple[str, str, int], labels: np.ndarray,
               val_idx: np.ndarray, split_sha: str, excluded: Sequence[str]) -> dict:
    """Gates (a)-(d), (f) of one chain; excluded cells (and a stage-1 run no live cell uses) are
    skipped."""
    chain = chain_name(*ident)
    c = Path(runs) / chain
    live = [x for x in CELLS if f"{chain}/{x}" not in excluded]
    folders = [f for f in STAGE1 if any(CELLS[x][0] == f for x in live)] + live
    cfgs = {f: _config(c / f) for f in folders}
    row = {"chain": chain, "excluded": [x for x in CELLS if x not in live]}
    for f in folders:
        row[f"c_{f}"] = config_ok(f, cfgs, refs, ident)
    for cell in live:
        s1, ck, mode = CELLS[cell]
        d1, d2 = c / s1, c / cell
        start = START_EPOCH if ck == LAST else int(selection(d1)["best_epoch"])
        row[f"a_{cell}"] = same_record_at(d2, 0, d1, start)
        row[f"b_{cell}"] = init_is_own(d1, d2, ck)
        row[f"d_{cell}"] = selection_ok(d2, labels, val_idx, split_sha)
        if mode == "mean":
            row[f"f_{cell}"] = gamma_learned(d1, d2, ck)
    return row


def chain_gates(runs: Path, gate_root: Path, excluded: Sequence[str]) -> list[dict]:
    rows = []
    for d in DATASETS:
        val_idx, labels, sha = val_split(d)
        try:
            refs = references(gate_root, d)
        except GATE_ERRORS as exc:
            rows += [{"chain": chain_name(d, p, s), "error": f"references: {exc!r}"}
                     for p in PAIRS for s in SEEDS]
            continue
        for p in PAIRS:
            for s in SEEDS:
                try:
                    rows.append(gate_chain(runs, refs, (d, p, s), labels, val_idx, sha,
                                           excluded))
                except GATE_ERRORS as exc:
                    rows.append({"chain": chain_name(d, p, s), "error": repr(exc)})
    return rows


def letter_status(rows: Sequence[Mapping], letters: str = "abcdf") -> dict[str, bool]:
    ok = [r for r in rows if "error" not in r]
    return {g: all(bool(v) for r in ok for k, v in r.items() if k.startswith(f"{g}_"))
            for g in letters}


# --- gate (e) -------------------------------------------------------------------------------

def batch_env_check(runs: Path, identity: Path = E019_IDENTITY) -> dict:
    """13:14 batch: every environment record (env*.json in the runs root, at least
    env_<dataset>.json) equals the E019 no-change gate's environment and deployed commit, and
    that gate passed."""
    try:
        want, passed = identity_target(identity)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "reason": f"{Path(identity).name} unreadable: {exc!r}"}
    batches = {}
    for f in sorted(Path(runs).glob("env*.json")):
        try:
            env = _json(f)
        except (OSError, ValueError):
            env = None
        batches[f.name] = {"env": env, "same": same_env(env, want)}
    required = {f"env_{d}.json" for d in DATASETS}
    ok = passed and required <= set(batches) and all(b["same"] for b in batches.values())
    return {"identity_all_ok": passed, "want": want, "batches": batches, "ok": bool(ok)}


def expected_logs(excluded: Sequence[str] = ()) -> tuple[int, int]:
    """(fewest, most) base / m10b stage-2 logs of the 13:48 batch: one per run; a run of
    --infra_failed may or may not have left its log."""
    n = sum(k[3] in BATCH_1348 for k in RUN_KEYS)
    return n - sum(name.rsplit("/", 1)[-1] in BATCH_1348 for name in excluded), n


def _mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(os.path.getmtime(path)).astimezone().isoformat(
        timespec="seconds")


def write_deploy_evidence(deploy: str | Path, runs: str | Path, out: str | Path) -> dict:
    """Gate (e), 13:48 batch, on the server (README step 1): file times (local time with UTC
    offset) of the training deploy's DEPLOYED_COMMIT, of the second queue's start mark and of
    the first and last base / m10b stage-2 logs, their number and the deployed commit, written
    to ``out`` as JSON."""
    d, r = Path(deploy), Path(runs)
    logs = sorted((f for c in BATCH_1348 for f in r.glob(f"*.{c}.log")), key=os.path.getmtime)
    if not logs:
        raise SystemExit(f"no base / m10b stage-2 logs in {r}")
    ev = {"deployed_commit": (d / "DEPLOYED_COMMIT").read_text(encoding="utf-8").strip(),
          "deployed_commit_mtime": _mtime(d / "DEPLOYED_COMMIT"),
          "queue_start": _mtime(r / "queue_best.start"), "run_logs_first": _mtime(logs[0]),
          "run_logs_last": _mtime(logs[-1]), "n_logs": len(logs)}
    write_json(Path(out), ev)
    return ev


def deploy_evidence_check(path: Path | None, commit: str | None,
                          excluded: Sequence[str] = ()) -> dict:
    """13:48 batch (base, m10b; no environment record): --deploy_evidence, the JSON file of
    write_deploy_evidence: deployed_commit (the server's DEPLOYED_COMMIT),
    deployed_commit_mtime, run_logs_first and run_logs_last (the earliest and latest base /
    m10b log), n_logs and optionally queue_start (ISO 8601 with UTC offset). The commit must be
    the no-change gate's, the deploy must precede the runs (first <= last log) and every run of
    the batch must have its log (80; a run of --infra_failed may lack one)."""
    if path is None:
        return {"ok": False, "reason": "no --deploy_evidence: the 13:48 batch is unverified"}
    want = expected_logs(excluded)
    try:
        ev = _json(path)
        t = {k: dt.datetime.fromisoformat(ev[k]) for k in ("deployed_commit_mtime",
                                                             "run_logs_first", "run_logs_last")}
        before = t["deployed_commit_mtime"] < t["run_logs_first"]
        ordered = t["run_logs_first"] <= t["run_logs_last"]
        if "queue_start" in ev:
            q = dt.datetime.fromisoformat(ev["queue_start"])
            before = before and t["deployed_commit_mtime"] < q <= t["run_logs_first"]
        n_ok = want[0] <= int(ev["n_logs"]) <= want[1]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "reason": f"{path} unreadable: {exc!r}"}
    same = ev.get("deployed_commit") == commit
    return {"file": str(path), "sha256": sha256_file(path), "evidence": ev,
            "commit_is_gate_commit": same, "deploy_before_runs": bool(before),
            "logs_ordered": bool(ordered), "n_logs_expected": list(want), "n_logs_ok": n_ok,
            "ok": bool(same and before and ordered and n_ok)}


def gate_e(runs: Path, evidence: Path | None, identity: Path = E019_IDENTITY,
           excluded: Sequence[str] = ()) -> dict:
    """Gate (e): deployed commit and server environment = the E019 no-change gate, per batch."""
    env = batch_env_check(runs, identity)
    ev = deploy_evidence_check(evidence, (env.get("want") or {}).get("commit"), excluded)
    return {"batch_1314": env, "batch_1348": ev, "ok": bool(env["ok"] and ev["ok"])}


# --- gate (t): test data --------------------------------------------------------------------

def ucf_projection(intervals: Sequence[Sequence[int]], total_frames: int,
                   n: int = SEGMENTS) -> np.ndarray:
    """Previous UCF extractor (the previous project's extract/extractor.py): segment k covers frames
    [k d, (k + 1) d) with d = max(1, T // n) and is 1 if it overlaps an annotated interval
    (s < end and e > start, the annotation's numbers as given)."""
    step = max(1, int(total_frames) // n)
    return np.array([float(any(k * step < e and (k + 1) * step > s for s, e in intervals))
                     for k in range(n)])


def xdv_projection(intervals: Sequence[Sequence[int]], total_frames: int,
                   n: int = SEGMENTS) -> np.ndarray:
    """Previous XD extractor (extract_features_xdviolence.py): frame mask over
    [max(0, s), min(T, e)) per interval, boundaries linspace(0, T, n + 1) as int, 1 where the
    mask covers more than half of the segment."""
    t = int(total_frames)
    mask = np.zeros(t)
    for s, e in intervals:
        lo, hi = max(0, int(s)), min(t, int(e))
        if lo < hi:
            mask[lo:hi] = 1.0
    b = np.linspace(0, t, n + 1, dtype=int)
    return np.array([float(b[k + 1] > b[k] and mask[b[k]:b[k + 1]].mean() > 0.5)
                     for k in range(n)])


PROJECTIONS = {"ucf": ucf_projection, "xdv": xdv_projection}


def order_check(projected: np.ndarray, cached: np.ndarray) -> dict:
    """Mean Hamming distance (segments) between the projected and the cached segment labels of
    the anomalous videos, against the cached rows shifted by one (control); passes when it is
    at most a fifth of the control's."""
    if projected.shape != cached.shape:
        return {"ok": False, "reason": f"shapes {projected.shape} / {cached.shape}"}
    dist = float(np.abs(projected - cached).sum(axis=1).mean())
    ctrl = float(np.abs(projected - np.roll(cached, 1, axis=0)).sum(axis=1).mean())
    return {"mean_hamming": dist, "shifted_control": ctrl, "ok": bool(dist <= ctrl / 5)}


def manifest_summary(m: EvalManifest) -> dict:
    return {"n": len(m), "anomalous": int(sum(m.labels)), "frames": int(sum(m.total_frames)),
            "source": dict(m.meta.get("source") or {}).get("sha256"),
            "annotation": dict(m.meta.get("annotation") or {}).get("sha256")}


def check_test_data(dataset: str, manifest: EvalManifest, caches: Mapping[str, Mapping],
                    hashes: Mapping[str, str]) -> dict:
    """Gate (t) for one dataset from its manifest and the three encoders' test caches."""
    summary = manifest_summary(manifest)
    labels = {e: [int(x) for x in c["labels"]] for e, c in caches.items()}
    seg = {e: np.stack([np.asarray(x, dtype=np.float64) for x in c["seg_labels"]])
           for e, c in caches.items()}
    first = next(iter(caches))
    out = {"manifest": summary,
           "manifest_ok": all(summary[k] == v for k, v in TEST_EXPECTED[dataset].items()),
           "cache_sha256": dict(hashes),
           "cache_sha256_ok": all(hashes[e] == TEST_CACHE_SHA256[f"{dataset}_{e}"]
                                  for e in caches),
           "labels_equal_across_encoders": all(labels[e] == labels[first] for e in caches),
           "labels_match_manifest": labels[first] == list(manifest.labels),
           "seg_labels_equal_across_encoders": all(
               seg[e].shape == seg[first].shape
               and np.array_equal(seg[e], seg[first], equal_nan=True) for e in caches)}
    if out["labels_match_manifest"]:
        anom = np.flatnonzero(np.asarray(manifest.labels) == 1)
        proj = np.stack([PROJECTIONS[dataset](manifest.intervals[i], manifest.total_frames[i])
                         for i in anom])
        out["order"] = order_check(proj, seg[first][anom])
    else:
        out["order"] = {"ok": False, "reason": "cache labels differ from the manifest"}
    frames = caches[first].get("total_frames")
    out["cache_total_frames_match_manifest"] = (
        None if frames is None else [int(t) for t in frames] == list(manifest.total_frames))
    keys = ("manifest_ok", "cache_sha256_ok", "labels_equal_across_encoders",
            "labels_match_manifest", "seg_labels_equal_across_encoders")
    out["ok"] = bool(all(out[k] for k in keys) and out["order"]["ok"]
                     and (dataset != "xdv" or out["cache_total_frames_match_manifest"] is True))
    return out


def gate_t() -> dict:
    """Gate (t) on the local test data (the server's caches are checked by the test scoring)."""
    paths, out = load_paths(), {}
    for d in DATASETS:
        try:
            m = load_eval_manifest(eval_manifest_file(d))
            dirs = {e: paths.feature_dir(f"{d}_{e}") for e in ENCODERS}
            hashes = {e: sha256_file(cache_file(x, "test")) for e, x in dirs.items()}
            out[d] = check_test_data(d, m, {e: load_cache(x, "test") for e, x in dirs.items()},
                                     hashes)
        except GATE_ERRORS as exc:
            out[d] = {"ok": False, "error": repr(exc)}
    out["ok"] = all(out[d]["ok"] for d in DATASETS)
    return out


# --- P000 tables and score files ------------------------------------------------------------

def p000_rows(name: str) -> list[dict]:
    with open(P000_RESULTS / name, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["config"] == "abl_base"]


def _p000_key(row: Mapping) -> tuple[str, str] | None:
    hit = [k for k, v in P000_FEAT.items() if v == row["feat"] and k[0] == row["dataset"]]
    return hit[0] if hit else None


def p000_selection() -> dict[tuple[str, str, int], tuple[float, float, int]]:
    """(dataset, pair, seed) -> P000 abl_base (warm-start V-AUC, best V-AUC, best epoch)."""
    out = {}
    for r in p000_rows("val_selection.csv"):
        dp = _p000_key(r)
        if dp is not None:
            out[(*dp, int(r["seed"]))] = (float(r["warm_start_val_video_auc"]),
                                          float(r["best_val_video_auc"]), int(r["best_epoch"]))
    return out


def p000_test() -> dict[tuple[str, str, int, str], dict[str, float]]:
    """(dataset, pair, seed, role) -> P000 abl_base test frame AUC / AP and validation video
    AUC (points, 6 decimals; signal p)."""
    out = {}
    for d in DATASETS:
        for r in p000_rows(f"dcrc_{d}_runs.csv"):
            dp = _p000_key(r)
            if dp is not None and r["signal"] == "p":
                out[(*dp, int(r["seed"]), r["role"])] = {
                    k: float(r[k]) for k in ("frame_auc", "frame_ap", "val_video_auc")}
    return out


def file_cell(rel: str) -> str:
    """Cell of a score file '<chain>/<cell>_<rule>_<path>.npz'."""
    return PurePosixPath(rel).name.rsplit("_", 2)[0]


def verify_scores(scores: Path, split: str, expected: Mapping[str, object],
                  allowed: Sequence[str] = (), results_list: Path | None = None) -> dict:
    """The split's sha256 list holds every expected file (other files only of the ``allowed``
    cells), every listed file matches it, every current file is listed and, with
    ``results_list``, the list equals that committed copy."""
    base = Path(scores) / split
    text = (base / "scores.sha256").read_text(encoding="utf-8")
    listed = {}
    for line in text.splitlines():
        if line.strip():
            sha, rel = line.split(None, 1)
            listed[rel] = sha
    on_disk = {f.relative_to(base).as_posix() for f in current_files(base)}
    out = {"files": len(listed), "expected": len(expected),
           "missing": sorted(set(expected) - set(listed)),
           "unexpected": sorted(r for r in set(listed) - set(expected)
                                if file_cell(r) not in allowed),
           "unlisted": sorted(on_disk - set(listed)),
           "sha256_mismatch": sorted(r for r, sha in listed.items() if not (base / r).is_file()
                                     or sha256_file(base / r) != sha)}
    if results_list is not None:
        out["matches_results_list"] = (Path(results_list).is_file() and Path(
            results_list).read_text(encoding="utf-8") == text)
    out["ok"] = bool(not any(out[k] for k in ("missing", "unexpected", "unlisted",
                                              "sha256_mismatch"))
                     and out.get("matches_results_list", True))
    return out


def scoring_env(meta: Mapping, identity: Path = E019_IDENTITY) -> dict:
    """A scoring call ran in the training environment (E019 no-change gate: torch, CUDA, GPU,
    driver); its commit is recorded."""
    want, _ = identity_target(identity)
    env = meta.get("environment") or {}
    return {"environment": env, "code": meta.get("code"),
            "ok": all(env.get(k) == want[k] for k in ENV_KEYS)}


def path_ii_ok(arr: Mapping) -> bool:
    """Scored under P000's conditions: CPU, batch 256, 6 torch threads (OMP / MKL 6)."""
    return bool(str(arr.get("path")) == "ii" and str(arr.get("device")) == "cpu"
                and int(arr.get("batch_size", -1)) == PATHS["ii"][1]
                and int(arr.get("num_threads", -1)) == PATH_II_THREADS)


def test_metas(scores: Path) -> list[dict]:
    """Test-scoring metas in call order (meta_<k>.json, k = line of the access log), those of
    calls cut short ("complete": false) included."""
    files = sorted((Path(scores) / "test").glob("meta_*.json"), key=lambda f: int(f.stem[5:]))
    return [_json(f) | {"meta_file": f.name} for f in files]


def cell_calls(metas: Sequence[Mapping], cell: str) -> list[Mapping] | None:
    """The test calls that wrote a cell's current files: its last full read and the later
    calls of that cell (--missing_only); None if it was never read in full. A call cut short
    ("complete": false) counts: its commit, plan hash and threads were fixed before scoring."""
    hits = [m for m in metas if cell in m.get("cells", [])]
    full = [i for i, m in enumerate(hits) if not m.get("missing_only")]
    return hits[full[-1]:] if full else None


def threads_ok(meta: Mapping) -> bool:
    """A call that scored path (ii) ran under P000's thread set-up (OMP / MKL 6; recorded
    before scoring, so also in the meta of a call cut short)."""
    return (meta.get("threads_env") or {}) == dict.fromkeys(THREAD_VARS, str(PATH_II_THREADS))


# --- gate (v0) ------------------------------------------------------------------------------

def val_weights(scores: Path | None) -> dict[str, dict]:
    """Weight comparisons of the server's validation scoring (val/meta.json p000_weights)."""
    if scores is None or not (Path(scores) / "val" / "meta.json").is_file():
        return {}
    return {r["run"]: r for r in _json(Path(scores) / "val" / "meta.json").get(
        "p000_weights") or []}


def v0_row(runs: Path, key: Key, ref: Mapping, weights: Mapping[str, dict]) -> dict:
    sel = selection(Path(runs) / run_name(key))
    got = (sel.get("base_val_video_auc"), sel.get("best_val_video_auc"), sel.get("best_epoch"))
    want = tuple(ref[key[:3]])
    w = weights.get(run_name(key))
    w_ok = None if w is None else bool(w.get("best_equal") is True
                                       and w.get("last_equal") is True)
    return {"run": run_name(key), "start_val_video_auc": got[0], "p000_warm_start": want[0],
            "best_val_video_auc": got[1], "p000_best_val_video_auc": want[1],
            "best_epoch": got[2], "p000_best_epoch": want[2], "selection_equal": got == want,
            "weights_equal": w_ok, "p000_dirs_matched": (w or {}).get("matched"),
            "ok": bool(got == want and w_ok is True)}


def gate_v0(runs: Path, scores: Path | None, excluded: Sequence[str],
            ref: Mapping | None = None) -> dict:
    """Gate (v0), per base run, not blocking: start and best V-AUC and the selected epoch equal
    P000 val_selection.csv exactly, and the stage-2 best and last weights equal P000's
    checkpoints (torch.equal; compared on the server, val/meta.json). Only passing runs enter
    gate (v)."""
    ref = p000_selection() if ref is None else ref
    weights = val_weights(scores)
    rows = []
    for key in base_keys(excluded):
        try:
            rows.append(v0_row(runs, key, ref, weights))
        except GATE_ERRORS as exc:
            rows.append({"run": run_name(key), "ok": False, "error": repr(exc)})
    n = sum(bool(r["ok"]) for r in rows)
    return {"rows": rows, "n_runs": len(rows), "n_pass": n, "weights_checked": bool(weights),
            "ok": bool(rows) and n == len(rows)}


# --- gate (s) -------------------------------------------------------------------------------

def s_row(runs: Path, scores: Path, key: Key, rule: str, labels: np.ndarray,
          val_idx: np.ndarray) -> dict:
    """Gate (s) for one run and rule: path (i) sigma of the re-scored checkpoint vs the training
    record of its epoch (best: selected epoch; last: epoch 30), V-AUC and epoch as recorded."""
    d2 = Path(runs) / run_name(key)
    row = {"run": run_name(key), "rule": rule}
    if rule not in scorable_rules(runs, key):
        return row | {"skipped": "no best checkpoint (numerical failure)"}
    z = _npz(score_path(scores, "val", key, rule, "i"))
    rec, sel = _npz(d2 / "record_val_scores.npz"), selection(d2)
    if rule == "best":
        epoch, want = int(sel["best_epoch"]), float(sel["best_val_video_auc"])
    else:
        epoch, want = STAGE2_EPOCHS, float(_json(d2 / "history.json")[-1]["val_video_auc"])
    stored = rec["scores"][int(np.flatnonzero(rec["epochs"] == epoch)[0])]
    probs = z["probs"]
    row |= {"epoch": int(z["epoch"]), "epoch_recorded": epoch}
    if probs.shape != stored.shape:
        return row | {"ok": False, "reason": f"shapes {probs.shape} / {stored.shape}"}
    if not (np.isfinite(stored).all() and np.isfinite(probs).all()):
        return row | {"skipped": "non-finite scores (numerical failure)"}
    auc = max_video_auc(probs, labels)
    row |= {"max_abs_diff": float(np.max(np.abs(probs.astype(np.float64) - stored))),
            "v_auc": auc, "v_auc_recorded": want,
            "same_order": bool(np.array_equal(z["indices"], val_idx)
                               and np.array_equal(z["labels"], labels)),
            "same_checkpoint": str(z["ckpt_sha256"]) == sha256_file(d2 / RULES[rule])}
    row["ok"] = bool(row["max_abs_diff"] <= S_TOL and auc == want and row["epoch"] == epoch
                     and float(z["val_video_auc"]) == want and row["same_order"]
                     and row["same_checkpoint"])
    return row


def precheck_row(runs: Path, scores: Path, key: Key, rule: str, labels: np.ndarray,
                 val_idx: np.ndarray, ref: Mapping) -> dict:
    """Pre-check (PLAN.md order 2): path (ii) video AUC of a base checkpoint on the validation
    videos equals P000's val_video_auc (dcrc_*_runs.csv, points) to 6 decimals."""
    if rule not in scorable_rules(runs, key):
        return {"run": run_name(key), "rule": rule, "ok": False,
                "skipped": "no best checkpoint (numerical failure)"}
    z = _npz(score_path(scores, "val", key, rule, "ii"))
    auc = 100 * max_video_auc(z["probs"], labels)
    want = ref[(*key[:3], P000_ROLE[rule])]["val_video_auc"]
    order = bool(np.array_equal(z["indices"], val_idx) and np.array_equal(z["labels"], labels))
    return {"run": run_name(key), "rule": rule, "val_video_auc": auc,
            "p000_val_video_auc": want, "same_order": order, "p000_conditions": path_ii_ok(z),
            "ok": bool(order and path_ii_ok(z) and round(auc, 6) == want)}


def precheck_summary(rows: Sequence[Mapping], v0: Mapping) -> dict:
    """Not blocking (the plan's step 3 needs (a)-(f), (t), (s)); judged on the (v0)-passing
    runs, the runs gate (v) uses."""
    passing = {r["run"] for r in v0.get("rows", []) if r.get("ok")}
    used = [r for r in rows if r["run"] in passing]
    return {"rows": list(rows), "n_ok": sum(bool(r.get("ok")) for r in rows),
            "ok_all": bool(rows) and all(r.get("ok") for r in rows),
            "ok": (all(r.get("ok") for r in used) if used else None)}


def _guarded(head: Mapping, fn, *args) -> dict:
    try:
        return fn(*args)
    except GATE_ERRORS as exc:
        return {**head, "ok": False, "error": repr(exc)}


def gate_s(runs: Path, scores: Path, excluded: Sequence[str], v0: Mapping,
           ref: Mapping | None = None) -> dict:
    """Gate (s) on every run and rule (path (i)) and the path (ii) pre-check of the base runs;
    the score files must match the server's sha256 list, whose sha256 is recorded (base test
    scoring on the server needs it unchanged)."""
    ref = p000_test() if ref is None else ref
    files = verify_scores(scores, "val", expected_files(runs, list(CELLS), excluded))
    rows, pre = [], []
    for d in DATASETS:
        val_idx, labels, _ = val_split(d)
        for key in (k for k in RUN_KEYS if k[0] == d and run_name(k) not in excluded):
            for rule in RULES:
                head = {"run": run_name(key), "rule": rule}
                rows.append(_guarded(head, s_row, runs, scores, key, rule, labels, val_idx))
                if key[3] == "base":
                    pre.append(_guarded(head, precheck_row, runs, scores, key, rule, labels,
                                        val_idx, ref))
    env = scoring_env(_json(Path(scores) / "val" / "meta.json"))
    checked = [r for r in rows if "skipped" not in r]
    return {"files": files, "environment": env, "rows": rows,
            "val_list_sha256": sha256_file(Path(scores) / "val" / "scores.sha256"),
            "max_abs_diff": max((r.get("max_abs_diff", 0.0) for r in checked), default=None),
            "skipped": [f"{r['run']}/{r['rule']}" for r in rows if "skipped" in r],
            "precheck": precheck_summary(pre, v0),
            "ok": bool(files["ok"] and env["ok"] and all(r.get("ok", False) for r in checked))}


# --- gate (v) -------------------------------------------------------------------------------

def v_row(scores: Path, key: Key, rule: str, gts: Sequence, ref: Mapping) -> dict:
    """Gate (v) for one run and rule: path (ii) sigma readout of the test scores vs P000's
    frame AUC / AP (|difference| <= 1e-4 points)."""
    row = {"run": run_name(key), "rule": rule}
    try:
        arr = _npz(score_path(scores, "test", key, rule, "ii"))
        got = frame_auc_ap(arr["probs"], gts)
        want = ref[(*key[:3], P000_ROLE[rule])]
    except (OSError, ValueError, KeyError) as exc:
        return row | {"ok": False, "error": repr(exc)}
    diff = {m: got[m] - want[m] for m in V_METRICS}
    cond = path_ii_ok(arr)
    return row | {**{m: got[m] for m in V_METRICS}, **{f"p000_{m}": want[m] for m in V_METRICS},
                  **{f"diff_{m}": v for m, v in diff.items()}, "p000_conditions": cond,
                  "ok": bool(cond and all(abs(v) <= V_TOL for v in diff.values()))}


def path_diff_row(scores: Path, key: Key, rule: str, gts: Sequence) -> dict:
    """A base run's test frame AUC / AP by paths (i) and (ii) under the sigma and logit
    readouts and their differences (i) - (ii); descriptive only (PLAN.md descriptive 5)."""
    row = {"run": run_name(key), "rule": rule}
    try:
        a, b = (_npz(score_path(scores, "test", key, rule, p)) for p in ("i", "ii"))
        row["max_abs_prob_diff"] = float(np.max(np.abs(a["probs"].astype(np.float64)
                                                       - b["probs"])))
        for ro, field in (("sigma", "probs"), ("logit", "logits")):
            x, y = frame_auc_ap(a[field], gts), frame_auc_ap(b[field], gts)
            for m in V_METRICS:
                row |= {f"{ro}_{m}_i": x[m], f"{ro}_{m}_ii": y[m],
                        f"{ro}_{m}_i_minus_ii": x[m] - y[m]}
    except (OSError, ValueError, KeyError) as exc:
        row["error"] = repr(exc)
    return row


def binding(calls: Sequence[Mapping] | None) -> dict | None:
    """Deployed commit and plan hash shared by the test calls that wrote the score files gate
    (v) used, calls cut short included (None if a cell was never read in full or the calls
    differ); the server checks them before scoring the later cells."""
    if not calls:
        return None
    pairs = {(c["code"]["commit"], c["plan"]["sha256"]) for c in calls}
    if len(pairs) != 1:
        return None
    commit, plan = pairs.pop()
    return {"commit": commit, "plan_sha256": plan, "meta": [c["meta_file"] for c in calls]}


def p000_stand_in(v0: Mapping, excluded: Sequence[str]) -> str | None:
    """Why P000's checkpoints may not stand in for the base runs in gate (v) (None: they may):
    PLAN.md allows it only when (v0) was evaluated for every base run, without error and with
    the server's weight comparison, and every run failed."""
    if "error" in v0 or any("error" in r for r in v0.get("rows", [])):
        return "(v0) not evaluated for every base run (error)"
    if not v0.get("weights_checked"):
        return "(v0) weights not compared on the server (val/meta.json)"
    if v0.get("n_runs") != len(base_keys(excluded)) or v0.get("n_pass") != 0:
        return f"(v0) {v0.get('n_runs')} rows, {v0.get('n_pass')} passing: not all runs failed"
    return None


def v_basis(metas: Sequence[Mapping], v0: Mapping,
            excluded: Sequence[str]) -> tuple[list[Key], str, list[Mapping] | None]:
    """Gate (v)'s runs (the (v0)-passing base runs, else P000's checkpoints when PLAN.md allows
    it, else none), how they were chosen and the calls that wrote their score files."""
    passing = {r["run"] for r in v0.get("rows", []) if r.get("ok")}
    keys: list[Key] = [k for k in base_keys(excluded) if run_name(k) in passing]
    groups = [cell_calls(metas, "base")]
    basis = "v0_passing_base_runs"
    if not keys:
        why = p000_stand_in(v0, excluded)
        if why is not None:
            return [], f"none: {why}", None
        keys, basis = list(P000_KEYS), "p000_checkpoints (v0 failed for every base run)"
        groups.append(cell_calls(metas, P000_CELL))
    calls = None if any(g is None for g in groups) else [c for g in groups for c in g]
    return keys, basis, calls


def gate_v(scores: Path, v0: Mapping, gts: Mapping[str, Sequence], excluded: Sequence[str],
           ref: Mapping | None = None) -> dict:
    """Gate (v) (PLAN.md): the path (ii) sigma readout of every (v0)-passing base run equals
    P000 abl_base (best rule: role best; last rule: role final) within 1e-4 points; when (v0)
    failed for all runs, P000's own checkpoints scored by path (ii) (p000_base) stand in. The
    calls that wrote those files share one commit and plan hash and ran path (ii) under
    P000's threads. The (i) - (ii) differences are descriptive."""
    ref = p000_test() if ref is None else ref
    keys, basis, calls = v_basis(test_metas(scores), v0, excluded)
    rows = [v_row(scores, k, r, gts[k[0]], ref) for k in keys for r in RULES]
    bound = binding(calls)
    threads = bool(calls) and all(threads_ok(c) for c in calls)
    diffs = [abs(r[f"diff_{m}"]) for r in rows for m in V_METRICS if f"diff_{m}" in r]
    return {"basis": basis, "n_runs": len(keys), "rows": rows,
            "max_abs_diff": max(diffs, default=None),
            "path_i_minus_ii": [path_diff_row(scores, k, r, gts[k[0]])
                                for k in base_keys(excluded) for r in RULES],
            "scored_with": bound, "threads_env_ok": threads,
            "ok": bool(rows) and bound is not None and threads and all(r["ok"] for r in rows)}


# --- before test scoring --------------------------------------------------------------------

def run_gates(runs: Path, gate_root: Path, excluded: Sequence[str], scores: Path | None,
              evidence: Path | None) -> dict:
    """Gates before test scoring: (a)-(f), (t), (v0), and (s) when validation scores exist."""
    rows = chain_gates(runs, gate_root, excluded)
    status = letter_status(rows)
    env = gate_e(runs, evidence, excluded=excluded)
    status["e"] = env["ok"]
    t = gate_t()
    status["t"] = t["ok"]
    try:
        v0 = gate_v0(runs, scores, excluded)
    except GATE_ERRORS as exc:
        v0 = {"ok": False, "rows": [], "error": repr(exc)}
    status["v0"] = v0["ok"]
    out = {"chains": rows, "environment": env, "t": t, "v0": v0, "infra_failed": list(excluded),
           "errors": [r["chain"] for r in rows if "error" in r]}
    if scores is not None and (Path(scores) / "val").is_dir():
        try:
            out["s"] = gate_s(runs, scores, excluded, v0)
        except GATE_ERRORS as exc:
            out["s"] = {"ok": False, "error": repr(exc), "precheck": {"ok": None}}
        status["s"] = out["s"]["ok"]
        status["s_precheck"] = out["s"]["precheck"]["ok"]
    out["status"] = {k: status[k] for k in sorted(status)}
    return out
