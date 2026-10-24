"""E024 stage 0a: is the UR-DMU - C32 localisation gap visible on validation?
(experiments/E024_localisation_method/stage0/PLAN.md "0a"; validation only).

Arms, per dataset (ucf, xdv) x pair (xcb_xcl, xcb_siglip) x development seed (0, 1, 2):
    C32-L20  the E021 ``hb_l20`` score file: the epoch of the first maximum of the W-readout sigma
             full-validation V-AUC (``e022_score.select_readout(sf, "W")``, E022 C's rule), its
             dense snippet logits ``dense_z``
    U        ``wvad.baselines.train --method urdmu --unit 200 --lr_mult 1 --scale_mult 1
             --n_chunk 4096 --deterministic``: the sigma point of ``selection.json``,
             ``val_scores.npz`` z[point - 1]

Snippet logits are repeated over their supports [16 j, min(16 j + 16, T)). Statistics, per run, on
the validation videos (``glance.run_statistics``): GP (video mean) x 100 primary, GN-AUC and
Hit@128 descriptive; on UCF the UCA-proxy within-video frame AUC in points (M2A, secondary,
``e023_prefreeze.video_aucs``). Each statistic is evaluated for every row of one draw matrix
(``glance.bootstrap_draws(n_anomalous, n_normal, 1, 2000, 20261024)``: the anomalous and the normal
videos resampled with replacement, row 0 the original sample; the seed block is unused), shared by
every arm and pair. An arm's value per (dataset, pair) is the mean over its surviving seeds (at
least two; a run that collapsed, selected sigma V-AUC < 0.60, failed numerically or is missing is
left out, with its reason), the dataset value the mean of the two pairs. Delta_gap32 = U - C32-L20;
the 95% percentile interval over rows 1..R of the primary statistic classifies it: positive (lower >
0), negative (upper < 0), inconclusive. The other statistics are reported, not classified.

Commands:
    pins      results/e024_stage0a_inputs.json before the freeze: sha256 and counts of the inputs
              (no metric), the C32 files checked against E021's scores.sha256; written once
    freeze    stage0/PLAN.sha256 (clean tree; the plan, the pins and this module committed)
    gate-h    each U run's input hashes against the E023 confirmation chain urdmu_u200 record
    gate-b    two runs' val_scores z and selection, bit for bit
    analyze   results/e024_stage0a.json and e024_stage0a_runs.csv: clean tree, the freeze commit
              (PLAN.sha256 added alone), the pins as committed before it and equal to the inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from wvad.data.catalog import split_file
from wvad.data.dense_test import CHANGELOG, freeze_commit_of, plan_seal, plan_sha256
from wvad.diagnostics import glance as gl
from wvad.diagnostics.common import code_version, write_csv, write_json
from wvad.diagnostics.e022_score import dense_failure, select_readout
from wvad.diagnostics.e023_prefreeze import proxy_labels, split_rows, video_aucs
from wvad.extract.grid import snippet_support
from wvad.paths import REPO_ROOT

E024_DIR = REPO_ROOT / "experiments" / "E024_localisation_method"
STAGE0_DIR = E024_DIR / "stage0"
RESULTS = E024_DIR / "results"
SEAL_FILE = "PLAN.sha256"
PINS = RESULTS / "e024_stage0a_inputs.json"
UCA_PROXY = REPO_ROOT / "data" / "annotations" / "ucf_train_uca_proxy.json"
E021_LIST = "stage2/scores.sha256"          # E021's server-written list of its stage-2 score files

DATASETS = ("ucf", "xdv")
PAIRS = ("xcb_xcl", "xcb_siglip")
DEV_SEEDS = (0, 1, 2)
C32, U = "C32-L20", "U"
U_RUN = "urdmu_u200"
U_CONFIG = {"method": "urdmu", "unit": 200, "lr_mult": "1", "scale_mult": "1",
            "deterministic": True, "n_chunk": 4096}
C32_FILE = "hb_l20.npz"
UCA_VARIANT = "M2A"
COLLAPSE = 0.60               # selected sigma V-AUC below this: the run is left out (PLAN.md)
MIN_SEEDS = 2                 # surviving seeds per (dataset, pair) an arm needs
N_BOOT, BOOT_SEED = 2000, 20261024
STATISTICS = ("gp_video", "gn_auc", "hit_128")
PRIMARY = "gp_video"
NEXT = {"positive": "stage 0b on this dataset (after the 0b plan is frozen)",
        "inconclusive": "no stage 0b: the validation metric cannot resolve this gap",
        "negative": "no stage 0b: validation runs against the test direction (UR-DMU > C)",
        "no_verdict": "no stage 0b: an arm has fewer than two surviving seeds in a pair"}
TEST_MARKERS = ("_d16_test",)


# --- inputs ---------------------------------------------------------------------------------------

def refuse_test_paths(*paths: str | Path) -> None:
    """Stage 0a reads validation only: a test folder or store as an input (as given or after
    resolving links) is refused."""
    for p in paths:
        for q in (Path(p), Path(p).expanduser().resolve()):
            parts = q.as_posix().split("/")
            if "test" in parts or any(m in part for part in parts for m in TEST_MARKERS):
                raise SystemExit(f"{p}: a test-set path; stage 0a reads validation only")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def c32_name(dataset: str, pair: str, seed: int) -> str:
    return f"{dataset}_{pair}_s{seed}/{C32_FILE}"


def c32_file(e021_scores: Path, dataset: str, pair: str, seed: int) -> Path:
    return Path(e021_scores) / "stage2" / c32_name(dataset, pair, seed)


def u_dir(runs: Path, dataset: str, pair: str, seed: int) -> Path:
    return Path(runs) / f"{dataset}_{pair}_s{seed}" / U_RUN


def validation_set(dataset: str) -> gl.GlanceSet:
    return gl.load_glance(dataset)


def uca_labels(gs: gl.GlanceSet) -> dict[int, np.ndarray]:
    """Validation position -> UCA proxy frame labels of the anomalous videos with an M2A
    interval holding both classes (the videos ``video_aucs`` scores)."""
    from wvad.diagnostics.uca import load_proxy

    lab = proxy_labels(gs.names, gs.labels, gs.total_frames, load_proxy(UCA_PROXY), UCA_VARIANT)
    return {i: v for i, v in lab.items() if np.asarray(v).min() != np.asarray(v).max()}


def read_sha_list(path: Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = sha
    return out


def pins_record(e021_scores: Path) -> dict:
    """The pre-freeze input record (sha256 and counts; no score is read beyond its bytes). The
    C32 files must equal E021's own list (the server-written scores.sha256)."""
    refuse_test_paths(e021_scores)
    files = {c32_name(d, p, s): sha256_file(c32_file(e021_scores, d, p, s))
             for d in DATASETS for p in PAIRS for s in DEV_SEEDS}
    listed = read_sha_list(Path(e021_scores) / E021_LIST)
    bad = [k for k, v in files.items() if listed.get(k) != v]
    if bad:
        raise SystemExit(f"C32 files differ from E021's {E021_LIST}: {bad}")
    return {"stage": "E024 stage 0a inputs (stage0/PLAN.md 0a, gate (p))", "c32_files": files,
            "e021_list": {"file": E021_LIST,
                          "sha256": sha256_file(Path(e021_scores) / E021_LIST),
                          "c32_files_listed": len(files)},
            "splits": {d: sha256_file(split_file(d)) for d in DATASETS},
            "glance": {d: validation_set(d).coverage() for d in DATASETS},
            "uca": {"file": UCA_PROXY.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(UCA_PROXY), "variant": UCA_VARIANT,
                    "videos": len(uca_labels(validation_set("ucf")))}}


def write_pins(e021_scores: Path) -> Path:
    """Gate (p), once, before the freeze."""
    if PINS.exists() or (STAGE0_DIR / SEAL_FILE).exists():
        raise SystemExit(f"{PINS.name} or the seal exists: the input record is written once, "
                         "before the freeze")
    write_json(PINS, pins_record(e021_scores))
    return PINS


def check_pins(pins: Mapping, e021_scores: Path) -> dict:
    """Gate (p): the inputs equal the committed record (SystemExit otherwise)."""
    now = pins_record(e021_scores)
    bad = [k for k in ("c32_files", "e021_list", "splits", "glance", "uca")
           if now[k] != pins.get(k)]
    if bad:
        raise SystemExit(f"stage 0a inputs differ from {PINS.name}: {bad}")
    return {"ok": True, "record": PINS.relative_to(REPO_ROOT).as_posix(),
            "record_sha256": sha256_file(PINS)}


# --- runs -----------------------------------------------------------------------------------------

def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _failed(arm: str, seed: int, reason: str, sha: str | None = None,
            missing: bool = False) -> dict:
    return {"arm": arm, "seed": seed, "failed": reason, "missing": missing, "point": None,
            "v_auc": math.nan, "file_sha256": sha}


def c32_run(path: Path, seed: int) -> dict:
    """One E021 hb_l20 file: E022 C's W-readout selection and that epoch's snippet logits."""
    sf = _npz(path)
    sha = sha256_file(path)
    bad = dense_failure(sf)
    if bad:
        return _failed(C32, seed, bad, sha)
    sel = select_readout(sf, "W")
    e = int(sel["epoch"])
    return {"arm": C32, "seed": seed, "failed": None,
            "z": split_rows(sf["dense_z"][e - 1], sf["dense_counts"]),
            "point": e, "v_auc": float(sel["selected_v_auc"]),
            "val_indices": [int(x) for x in sf["val_indices"]],
            "total_frames": [int(x) for x in sf["total_frames"]],
            "labels": [int(x) for x in sf["labels"]], "file_sha256": sha}


def u_run(path: Path, seed: int, dataset: str, pair: str) -> dict:
    """One UR-DMU run: its sigma point (selection.json) and that point's validation logits; a
    numerically failed run is returned with its reason (left out like a collapse), a missing one
    marked missing (the analysis stops unless it is named an infrastructure failure)."""
    rec_file, z_file = Path(path) / "selection.json", Path(path) / "val_scores.npz"
    if not rec_file.is_file():
        return _failed(U, seed, "missing selection.json", missing=True)
    rec = json.loads(rec_file.read_text(encoding="utf-8"))
    cfg = rec.get("config") or {}
    want = {**U_CONFIG, "seed": seed, "dataset": dataset, "pair": pair}
    wrong = {k: cfg.get(k) for k, v in want.items() if cfg.get(k) != v}
    if wrong:
        raise SystemExit(f"{path}: config {wrong} differs from {want}")
    sig = (rec.get("selection") or {}).get("sigma")
    if rec.get("failed") or not sig:
        return _failed(U, seed, str(rec.get("failed") or "no sigma selection"))
    if not z_file.is_file():
        return _failed(U, seed, "missing val_scores.npz", missing=True)
    point = int(sig["point"])
    z = _npz(z_file)
    return {"arm": U, "seed": seed, "failed": None,
            "z": split_rows(z["z"][point - 1], z["counts"]),
            "point": point, "v_auc": float(sig["v_auc"]),
            "val_indices": [int(x) for x in z["val_indices"]],
            "total_frames": [int(x) for x in z["total_frames"]],
            "labels": [int(x) for x in z["labels"]], "file_sha256": sha256_file(z_file)}


def frames_of(run: Mapping, gs: gl.GlanceSet) -> list[gl.UnitScores]:
    """The run's frame scores in ``gs`` order: its validation videos, frame counts and labels must
    be the split's and the manifest's."""
    for key, want in (("val_indices", gs.val_indices), ("total_frames", gs.total_frames),
                      ("labels", gs.labels)):
        if key in run and list(run[key]) != [int(x) for x in want]:
            raise SystemExit(f"{run['arm']} seed {run['seed']}: {key} differ from the split / "
                             "manifest")
    out = []
    for z, t in zip(run["z"], gs.total_frames, strict=True):
        sup = snippet_support(int(t))
        if len(z) != len(sup):
            raise SystemExit(f"{run['arm']} seed {run['seed']}: {len(z)} logits for "
                             f"{len(sup)} snippets")
        out.append(gl.UnitScores.from_supports(np.asarray(z, np.float64), sup[:, 0], sup[:, 1]))
    return out


def uca_values(run: Mapping, gs: gl.GlanceSet, labels: Mapping[int, np.ndarray],
               draws: gl.Draws) -> np.ndarray:
    """(R + 1,) UCA-proxy within-video AUC in points (``video_aucs`` gives points): per row the
    draw-weighted mean over the anomalous videos with proxy labels holding both classes (NaN in
    a row that draws none of them)."""
    aucs = video_aucs(run["z"], gs.total_frames, labels)
    col = {int(v): k for k, v in enumerate(gs.anomalous)}
    idx = sorted(aucs)
    w = draws.anomalous[:, [col[i] for i in idx]].astype(np.float64)
    vals = np.array([aucs[i] for i in idx], dtype=np.float64)
    tot = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (w @ vals) / np.where(tot > 0, tot, 1.0), np.nan)


def run_values(run: Mapping, gs: gl.GlanceSet, draws: gl.Draws,
               labels: Mapping[int, np.ndarray] | None) -> dict[str, np.ndarray]:
    """Every statistic of one run, (R + 1,), on the shared draws (GP, GN-AUC, Hit x 100)."""
    st = gl.run_statistics(gs, frames_of(run, gs), draws)
    out = {k: 100.0 * np.asarray(st[k], np.float64) for k in STATISTICS}
    if labels:
        out["uca_auc"] = uca_values(run, gs, labels, draws)
    return out


# --- statistics -----------------------------------------------------------------------------------

def collapsed(v_auc: float) -> bool:
    return not (math.isfinite(v_auc) and v_auc >= COLLAPSE)


def arm_mean(values: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
    """Mean over an arm's surviving seeds in one (dataset, pair); None below MIN_SEEDS."""
    if len(values) < MIN_SEEDS:
        return None
    return {k: np.mean([v[k] for v in values], axis=0) for k in values[0]}


def pooled(pairs: Sequence[Mapping[str, np.ndarray] | None]) -> dict[str, np.ndarray] | None:
    """Mean of the two pairs (None when a pair has no arm value)."""
    if any(p is None for p in pairs):
        return None
    return {k: np.mean([p[k] for p in pairs], axis=0) for k in pairs[0]}


def classify(lo: float, hi: float) -> str:
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "inconclusive"


def contrast(a: np.ndarray, b: np.ndarray, primary: bool = True) -> dict:
    """a - b per row; row 0 the estimate, the 95% percentile interval over rows 1..R. The primary
    statistic must be finite in every row and is classified; a secondary one is summarised over
    its finite rows and not classified."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ok = np.isfinite(d)
    if primary and not ok.all():
        raise ValueError(f"{int((~ok).sum())} non-finite bootstrap differences")
    rows = d[1:][ok[1:]]
    lo, hi = (float(v) for v in np.percentile(rows, [2.5, 97.5])) if rows.size else (math.nan,
                                                                                       math.nan)
    out = {"estimate": float(d[0]), "ci95": [lo, hi], "n_resamples": int(rows.size)}
    if primary:
        out["class"] = classify(lo, hi)
    return out


def arm_contrasts(u: Mapping[str, np.ndarray] | None,
                  c: Mapping[str, np.ndarray] | None) -> dict:
    """Delta_gap32 = U - C32-L20 of every statistic, the primary classified ("no_verdict" when an
    arm is missing)."""
    if u is None or c is None:
        return {"no_verdict": True}
    return {k: contrast(u[k], c[k], primary=k == PRIMARY) for k in u if k in c}


def outcome(res: Mapping) -> dict:
    """PLAN.md "0a 결과 → 다음 단계" for one dataset (from its primary-statistic contrast)."""
    cls = "no_verdict" if res.get("no_verdict") else res[PRIMARY]["class"]
    return {"class": cls, "next": NEXT[cls], "stage0b": cls == "positive"}


def overall(outcomes: Mapping[str, Mapping]) -> dict:
    go = sorted(d for d, o in outcomes.items() if o["stage0b"])
    return {"stage0b_datasets": go,
            "next": ("stage 0b on " + ", ".join(go)) if go else
            "stage 0 ends; user decision (stage-1 large-effect screening, audit paper or stop)"}


def seed_summary(seeds: Sequence[int], values: Sequence[Mapping[str, np.ndarray]]) -> dict:
    v = [float(x[PRIMARY][0]) for x in values]
    return {"by_seed": {int(s): x for s, x in zip(seeds, v, strict=True)},
            "sd": float(np.std(v, ddof=1)) if len(v) >= 2 else math.nan}


def seed_signs(u: Mapping[int, float], c: Mapping[int, float]) -> dict:
    """Per development seed present in both arms: the sign of U - C32 (descriptive)."""
    common = sorted(set(u) & set(c))
    d = {s: u[s] - c[s] for s in common}
    return {"by_seed": d, "positive": sum(v > 0 for v in d.values()), "n": len(d)}


# --- analysis -------------------------------------------------------------------------------------

def chain_name(dataset: str, pair: str, seed: int) -> str:
    return f"{dataset}_{pair}_s{seed}"


def excluded(run: Mapping, chain: str = "", infra_failed: Sequence[str] = ()) -> str | None:
    """Why a run is left out (PLAN.md "붕괴와 실패"), or None. A missing run stops the analysis
    unless it is named in ``infra_failed`` (an infrastructure failure after its one rerun)."""
    if run.get("missing"):
        if chain not in infra_failed:
            raise SystemExit(f"{chain} {run['arm']}: {run['failed']}; rerun it once, or name it "
                             "with --infra_failed after the rerun failed")
        return f"infrastructure failure (named): {run['failed']}"
    if run.get("failed"):
        return f"failed: {run['failed']}"
    if collapsed(run["v_auc"]):
        return "collapsed (selected sigma V-AUC < 0.60)"
    return None


def dataset_report(d: str, e021_scores: Path, runs: Path, rows: list,
                   infra_failed: Sequence[str] = ()) -> dict:
    gs = validation_set(d)
    labels = uca_labels(gs) if d == "ucf" else None
    draws = gl.bootstrap_draws(len(gs.anomalous), len(gs.normal), 1, N_BOOT, BOOT_SEED)
    pairs, means = {}, {C32: [], U: []}
    for p in PAIRS:
        got: dict[str, list] = {C32: [], U: []}
        for s in DEV_SEEDS:
            for run in (c32_run(c32_file(e021_scores, d, p, s), s),
                        u_run(u_dir(runs, d, p, s), s, d, p)):
                why = excluded(run, chain_name(d, p, s), infra_failed)
                vals = None if why else run_values(run, gs, draws, labels)
                rows.append({"dataset": d, "pair": p, "seed": s, "arm": run["arm"],
                             "point": run["point"], "v_auc": run["v_auc"], "excluded": why,
                             "file_sha256": run["file_sha256"],
                             **({} if vals is None else {k: float(v[0]) for k, v in vals.items()})})
                if vals is not None:
                    got[run["arm"]].append((s, vals))
        pairs[p] = {}
        for arm, items in got.items():
            m = arm_mean([v for _, v in items])
            means[arm].append(m)
            pairs[p][arm] = None if m is None else {
                "seeds": seed_summary([s for s, _ in items], [v for _, v in items]),
                "value": {k: float(v[0]) for k, v in m.items()}}
        pairs[p]["delta_gap32"] = arm_contrasts(means[U][-1], means[C32][-1])
        pairs[p]["seed_signs"] = seed_signs(
            {s: float(v[PRIMARY][0]) for s, v in got[U]},
            {s: float(v[PRIMARY][0]) for s, v in got[C32]})
    res = arm_contrasts(pooled(means[U]), pooled(means[C32]))
    return {"delta_gap32": res, "pairs": pairs, "outcome": outcome(res),
            "glance": gs.coverage(), "uca_videos": None if labels is None else len(labels)}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def require_frozen() -> dict:
    """Clean tree; PLAN.sha256 added once, alone, by the freeze commit; the plan has the frozen
    hash; the pins as committed in the freeze commit's parent."""
    code = code_version()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed; code version {code}")
    seal = plan_seal(STAGE0_DIR)
    freeze = freeze_commit_of(REPO_ROOT, STAGE0_DIR)
    seal_rel = (STAGE0_DIR / SEAL_FILE).relative_to(REPO_ROOT).as_posix()
    pins_rel = PINS.relative_to(REPO_ROOT).as_posix()
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root",
                   freeze or "HEAD").stdout.split()
    if freeze is None or changed != [seal_rel]:
        raise SystemExit(f"the stage-0 freeze commit must add {seal_rel} alone ({freeze}, "
                         f"{changed})")
    at_freeze = _git("show", f"{freeze}:{seal_rel}").stdout.split()
    if not at_freeze or at_freeze[0] != seal["sha256"]:
        raise SystemExit(f"the plan hash {seal['sha256']} is not the one sealed by {freeze}")
    parent = _git("show", f"{freeze}^:{pins_rel}")
    if parent.returncode != 0 or json.loads(parent.stdout) != json.loads(
            PINS.read_text(encoding="utf-8")):
        raise SystemExit(f"{pins_rel} differs from the one committed before the freeze")
    return {"plan": seal, "freeze_commit": freeze, "code": code}


def analyze(e021_scores: Path, runs: Path, out: Path, csv_out: Path,
            infra_failed: Sequence[str] = ()) -> dict:
    refuse_test_paths(e021_scores, runs, out, csv_out)
    if not Path(runs).is_dir():
        raise SystemExit(f"{runs}: no runs folder")
    frozen = require_frozen()
    gate_p = check_pins(json.loads(PINS.read_text(encoding="utf-8")), e021_scores)
    rows: list[dict] = []
    report = {"check": "E024 stage 0a (stage0/PLAN.md 0a)", **frozen, "gate_p": gate_p,
              "infra_failed": list(infra_failed),
              "datasets": {d: dataset_report(d, e021_scores, runs, rows, infra_failed)
                           for d in DATASETS}}
    report["outcome"] = overall({d: r["outcome"] for d, r in report["datasets"].items()})
    report |= {"bootstrap": {"n_resamples": N_BOOT, "rng_seed": BOOT_SEED,
                             "resampled": "anomalous and normal videos; seeds averaged"},
               "libraries": {"python": platform.python_version(), "numpy": np.__version__}}
    write_json(out, report)
    write_csv(csv_out, rows)
    return report


# --- gates ----------------------------------------------------------------------------------------

INPUT_KEYS = ("train_data_sha256", "scale_array_sha256", "split_sha256", "val_indices_sha256")


def gate_h(runs: Path, e023_runs: Path) -> dict:
    """Each U run's training and validation inputs equal the E023 confirmation chain's
    (``<ds>_<pair>_s42/urdmu_u200``) record; every hash must be present."""
    refuse_test_paths(runs, e023_runs)
    out, ok = {}, True
    for d in DATASETS:
        for p in PAIRS:
            ref = json.loads((Path(e023_runs) / f"{d}_{p}_s42" / U_RUN / "selection.json")
                             .read_text(encoding="utf-8"))["inputs"]
            for s in DEV_SEEDS:
                f = u_dir(runs, d, p, s) / "selection.json"
                got = json.loads(f.read_text(encoding="utf-8"))["inputs"] if f.is_file() else {}
                bad = [k for k in INPUT_KEYS if not got.get(k) or got.get(k) != ref.get(k)]
                out[f"{d}_{p}_s{s}"] = {"ok": not bad, "differ": bad}
                ok = ok and not bad
    return {"ok": ok, "runs": out}


def gate_b(a: Path, b: Path) -> dict:
    """Two runs of the same command: val_scores z and the selection bit for bit."""
    refuse_test_paths(a, b)
    za, zb = _npz(Path(a) / "val_scores.npz")["z"], _npz(Path(b) / "val_scores.npz")["z"]
    sa, sb = (json.loads((Path(x) / "selection.json").read_text(encoding="utf-8"))["selection"]
              for x in (a, b))
    same_z = za.shape == zb.shape and za.dtype == zb.dtype and bool(np.array_equal(za, zb))
    return {"ok": same_z and bool(sa) and sa == sb, "z_equal": same_z, "selection_equal": sa == sb}


# --- freeze ---------------------------------------------------------------------------------------

def freeze() -> str:
    """stage0/PLAN.sha256 = sha256 of PLAN.md before its change log (clean tree; the plan, the pins
    and this module committed and unmodified; commit the seal file alone: that commit is the
    stage-0a freeze commit)."""
    code = code_version()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed; code version {code}")
    for f in (STAGE0_DIR / "PLAN.md", PINS, Path(__file__).resolve()):
        rel = f.relative_to(REPO_ROOT).as_posix()
        if _git("cat-file", "-e", f"HEAD:{rel}").returncode != 0 \
                or _git("status", "--porcelain", "--", rel).stdout.strip():
            raise SystemExit(f"{rel} must be committed and unmodified at HEAD before the freeze")
    seal = STAGE0_DIR / SEAL_FILE
    if seal.exists():
        raise SystemExit(f"{seal} exists: the plan is frozen once")
    h = plan_sha256(STAGE0_DIR / "PLAN.md")
    seal.write_text(f"{h}  PLAN.md (before {CHANGELOG})\n", encoding="utf-8", newline="\n")
    return h


# --- command line ---------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e024_stage0")
    sub = ap.add_subparsers(dest="command", required=True)
    pi = sub.add_parser("pins", help="write the pre-freeze input record (gate (p)), once")
    pi.add_argument("--e021_scores", type=Path, required=True)
    sub.add_parser("freeze", help="write stage0/PLAN.sha256")
    gh = sub.add_parser("gate-h", help="U runs' input hashes against the E023 records")
    gh.add_argument("--runs", type=Path, required=True)
    gh.add_argument("--e023_runs", type=Path, required=True)
    gh.add_argument("--out", type=Path)
    gb = sub.add_parser("gate-b", help="two runs bit for bit")
    gb.add_argument("--a", type=Path, required=True)
    gb.add_argument("--b", type=Path, required=True)
    gb.add_argument("--out", type=Path)
    an = sub.add_parser("analyze")
    an.add_argument("--e021_scores", type=Path, required=True)
    an.add_argument("--runs", type=Path, required=True, help="the U runs root")
    an.add_argument("--out", type=Path, default=RESULTS / "e024_stage0a.json")
    an.add_argument("--csv", type=Path, default=RESULTS / "e024_stage0a_runs.csv")
    an.add_argument("--infra_failed", nargs="*", default=[],
                    help="<ds>_<pair>_s<seed> U chains missing after their one rerun")
    a = ap.parse_args(argv)
    if a.command == "pins":
        print(f"[e024_stage0] pins -> {write_pins(a.e021_scores)}")
    elif a.command == "freeze":
        print(f"[e024_stage0] PLAN.sha256 {freeze()}: commit it alone (the freeze commit)")
    elif a.command in ("gate-h", "gate-b"):
        res = gate_h(a.runs, a.e023_runs) if a.command == "gate-h" else gate_b(a.a, a.b)
        if a.out:
            write_json(a.out, res)
        print(f"[e024_stage0] {a.command}: {'passed' if res['ok'] else 'FAILED'}")
        if not res["ok"]:
            raise SystemExit(1)
    else:
        rep = analyze(a.e021_scores, a.runs, a.out, a.csv, a.infra_failed)
        print(json.dumps({d: r["outcome"] for d, r in rep["datasets"].items()}, indent=1))


if __name__ == "__main__":
    main()
