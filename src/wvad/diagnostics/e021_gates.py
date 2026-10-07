"""E021 gates of the local analysis (PLAN.md "관문 (분석 전에 확인)"): (a)-(g) on the training
outputs, (h) against the earlier records, (s), (s2), (s3), (p) on the server's dense scoring, (f)
on the deployed commits and environments, and the score-file integrity. Local analysis code: no
annotation file and neither the glance nor the UCA module is imported.

    (a) stage-2 epoch-0 record = the run's own stage-1 epoch-20 record (bit for bit)
    (b) stage-2 init = the chain's own stage-1 ckpt_last.pt (full path recorded at training)
    (c) config = the no-change gate chain's config of the same pipeline and stage except seed,
        pair, dataset, out_dir, init and features; features = the head's; save_epoch_ckpts true
        (reference: the h0 development chain ucf_xcb_xcl_s0, the (g2) chain whose new config
        keys every later comparison uses, PLAN.md "무변경 관문")
    (d) the records' validation order = the split file
    (e) M10: gamma non-zero at the end of stage 1 and moved in stage 2
    (f) each batch's deployed commit has the no-change gate commit's extraction, training and
        scoring code (git), and the batch environment equals the gate's; the same for the
        scoring calls
    (g) the stage-2 record reproduces selection.json (E010 check)
    (h) h0 development chains = e013_runs.csv (l20) / e017_runs.csv (m10), h0 M10 calibration
        chains = e018_runs.csv, all four metrics at the E010 R within 1e-12
    (s) every epoch checkpoint re-scored on its own 32-unit features: sigma = the record of that
        epoch within 1e-6 (and its epoch, recorded V-AUC, config and validation order)
    (s2) h0 / h1: dense z of the d16c snippets = z of the d16c 32-unit cache,
         |dz| <= 1e-5 max(1, |z|)
    (s3) hb / hbm: z of the scale-matched snippets averaged per window = z of the own cache
    (p) dense score rows = the dense cache index (ceil(T / 16) per video), finite
    freeze  the (g1) commit, every (x4) run commit, every batch environment record, every scoring
            call and the analysis commit are the freeze commit or its descendants (PLAN.md
            "실행 순서" 2, git merge-base --is-ancestor, as E020)

A run with non-finite losses or records failed numerically: it is not re-run, its gate checks
that need finite values are skipped, and the verdict treats its settings as failed. Non-finite
dense snippet logits (or gamma) in the score file of a run whose losses and records are finite
are a numerical failure as well (PLAN.md "수치 실패(손실·점수에 NaN·무한대)"), not a gate (p)
failure: ``score_gates`` marks that run failed (its (s2), (s3) skipped; (p) keeps the shape check).
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from wvad.data import load_split, split_file
from wvad.data.manifests import load_manifest, manifest_file
from wvad.diagnostics.common import code_version, gpu_environment, write_json
from wvad.diagnostics.e010 import check_selection
from wvad.diagnostics.e021_score import (
    DATASETS,
    DEV_SEEDS,
    E021_DIR,
    EPOCHS,
    GATE_ERRORS,
    HEADS,
    PAIRS,
    PIPELINES,
    STAGE1_LAST,
    RunKey,
    current_files,
    run_dir,
    run_keys,
    score_path,
    stage1_dir,
)
from wvad.eval.crossfit import METRICS, CrossFit
from wvad.eval.dense import compare_logits
from wvad.extract import grid
from wvad.paths import REPO_ROOT
from wvad.prep.manifests import sha256_file

RESULTS = E021_DIR / "results"
IDENTITY = RESULTS / "e021_identity.json"
REF_CHAIN = "ucf_xcb_xcl_s0"
IGNORED_CONFIG = ("seed", "pair", "dataset", "out_dir", "init", "features")
START_EPOCH = 20
S_TOL, REF_TOL = 1e-6, 1e-12
ENV_KEYS = ("torch", "cuda", "gpu", "driver")
EXTRACT_ENCODERS = ("xcb", "xcl", "siglip")
# The frozen extraction, training and scoring code (PLAN.md two-stage code rule, gate (f), start
# condition (5)), without the analysis functions computed from saved logits that the rule lets
# the analysis fix between the stages: the LS-AUC (eval/crossfit.py) and the readout
# aggregation W / Wm / G (eval/dense_readout.py). No frozen module imports them (unit test).
ANALYSIS_EXCLUDED = ("src/wvad/eval/crossfit.py", "src/wvad/eval/dense_readout.py")
FROZEN_PATHS = ("src/wvad/extract", "src/wvad/train", "src/wvad/models", "src/wvad/data",
                "src/wvad/eval", "src/wvad/diagnostics/e021_score.py",
                *(f":(exclude){p}" for p in ANALYSIS_EXCLUDED))
REFERENCE_CSV = {"l20": REPO_ROOT / "experiments" / "E013_stage1_last_checkpoint" / "results"
                 / "e013_runs.csv",
                 "m10": REPO_ROOT / "experiments" / "E017_offset_scale" / "results"
                 / "e017_runs.csv",
                 "m10_cal": REPO_ROOT / "experiments" / "E018_delta_recalibration" / "results"
                 / "e018_runs.csv"}
ENV_FILES = {1: ("env_ucf.json", "env_xdv.json"), 2: ("env_s2_ucf_*.json", "env_s2_xdv_*.json")}


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _config(d: Path) -> dict:
    return _ckpt(Path(d) / STAGE1_LAST)["config"]


def _npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def val_split(dataset: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Validation indices (split-file order), their labels and the split identity."""
    split = load_split(split_file(dataset))
    idx = np.asarray(split.val_indices)
    return idx, np.array([split.labels[i] for i in idx]), split.sha256


def val_totals(dataset: str) -> np.ndarray:
    """Manifest T of the validation videos (split order): the grid's T (PLAN.md R3)."""
    manifest = load_manifest(manifest_file(dataset))
    return np.array([manifest.total_frames[int(i)] for i in val_split(dataset)[0]],
                    dtype=np.int64)


# --- runs -----------------------------------------------------------------------------------

def _finite_history(history: Sequence[Mapping]) -> bool:
    return all(math.isfinite(v) for h in history for k, v in h.items() if k.startswith("loss_"))


@dataclass
class Run:
    key: RunKey
    s1: np.ndarray                     # (20, n, 32) stage-1 record, epochs 1-20
    s2: np.ndarray                     # (31, n, 32) stage-2 record, epochs 0-30
    history1: list
    history2: list
    selection: dict
    config: dict                       # stage-2 ckpt_last.pt config
    failed: str | None = None
    gamma: dict = field(default_factory=dict)       # M10: stage-1 end, stage-2 end

    @property
    def record(self) -> np.ndarray:    # (30, n, 32) stage-2 epochs 1-30
        return self.s2[1:]


def load_run(runs: Path, key: RunKey) -> Run:
    """The records, histories, selection and config of one chain; ``failed`` names a numerical
    failure (non-finite loss or record score)."""
    d1, d2 = stage1_dir(runs, key), run_dir(runs, key)
    z1, z2 = _npz(d1 / "record_val_scores.npz"), _npz(d2 / "record_val_scores.npz")
    if not np.array_equal(z1["epochs"], np.arange(1, START_EPOCH + 1)):
        raise ValueError(f"{d1}: stage-1 record should hold epochs 1-{START_EPOCH}")
    if not np.array_equal(z2["epochs"], np.arange(0, EPOCHS + 1)):
        raise ValueError(f"{d2}: stage-2 record should hold epochs 0-{EPOCHS}")
    h1, h2 = _json(d1 / "history.json"), _json(d2 / "history.json")
    run = Run(key, z1["scores"], z2["scores"], h1, h2, _json(d2 / "selection.json"),
              _config(d2))
    if not (np.isfinite(z1["scores"]).all() and np.isfinite(z2["scores"]).all()
            and _finite_history(h1) and _finite_history(h2)):
        run.failed = "non-finite loss or record score"
    if PIPELINES[key.pipeline][0] == "mean":
        run.gamma = {"stage1_end": _gamma(d1 / STAGE1_LAST), "stage2_end": _gamma(d2 / STAGE1_LAST)}
    return run


def _gamma(path: Path) -> float:
    return float(_ckpt(path)["model_state_dict"]["offset_gamma"])


# --- gates (a)-(g) --------------------------------------------------------------------------

def same_record_at(a: Path, ea: int, b: Path, eb: int) -> bool:
    """Gate (a): recorded scores of epoch ``ea`` of run a = epoch ``eb`` of run b (NaN = NaN)."""
    za, zb = _npz(a / "record_val_scores.npz"), _npz(b / "record_val_scores.npz")
    sa = za["scores"][int(np.flatnonzero(za["epochs"] == ea)[0])]
    sb = zb["scores"][int(np.flatnonzero(zb["epochs"] == eb)[0])]
    return sa.dtype == sb.dtype and bool(np.array_equal(sa, sb, equal_nan=True))


def init_is_own(d1: Path, d2: Path) -> bool:
    """Gate (b): the stage-2 init is this chain's stage-1 ckpt_last.pt, compared as the full
    path recorded at training time."""
    out1 = PurePosixPath(_config(d1)["out_dir"])
    init = PurePosixPath(_config(d2)["init"])
    return out1.parts[-2:] == (d1.parent.name, d1.name) and init == out1 / STAGE1_LAST


def references(runs: Path) -> dict[tuple[str, int], dict]:
    """Gate (c) references: the configs of the h0 development chain ucf_xcb_xcl_s0 per pipeline
    and stage (passed (g2))."""
    out = {}
    for p in PIPELINES:
        key = RunKey("ucf", "xcb_xcl", 0, "h0", p)
        out[(p, 1)], out[(p, 2)] = _config(stage1_dir(runs, key)), _config(run_dir(runs, key))
    return out


def config_ok(cfg: Mapping, ref: Mapping, key: RunKey) -> bool:
    """Gate (c) for one run config: own dataset / pair / seed, the head's features,
    save_epoch_ckpts true, and every other key (the key set included) equal to the reference."""
    a, b = set(cfg) - set(IGNORED_CONFIG), set(ref) - set(IGNORED_CONFIG)
    return ((cfg.get("dataset"), cfg.get("pair"), cfg.get("seed")) ==
            (key.dataset, key.pair, key.seed)
            and cfg.get("features") == HEADS[key.head] and cfg.get("save_epoch_ckpts") is True
            and a == b and all(cfg[k] == ref[k] for k in a))


def order_ok(runs: Path, key: RunKey, val_idx: np.ndarray) -> bool:
    """Gate (d): every record of the chain lists the validation videos in split order."""
    files = [stage1_dir(runs, key) / "record_val_scores.npz",
             run_dir(runs, key) / "record_val_scores.npz"]
    if PIPELINES[key.pipeline][0] == "mean":
        files += [f.with_name("record_val_offset.npz") for f in files]
    return all(np.array_equal(_npz(f)["val_indices"], val_idx) for f in files)


def gamma_learned(run: Run) -> bool:
    """Gate (e): gamma non-zero at the end of stage 1 and moved in stage 2 (stage 2 starts from
    the stage-1 ckpt_last.pt); a non-finite gamma passes here (numerical failure)."""
    g1, g2 = run.gamma["stage1_end"], run.gamma["stage2_end"]
    if not (math.isfinite(g1) and math.isfinite(g2)):
        return True
    return g1 != 0 and g2 != g1


def selection_ok(run: Run, labels: np.ndarray, split_sha: str, d2: Path) -> bool:
    """Gate (g): the stage-2 record (epochs 1-30) reproduces selection.json (E010 check); a
    non-finite record is a numerical failure, handled by the verdict."""
    if run.selection.get("val_split_sha") != split_sha:
        return False
    if not np.isfinite(run.record).all():
        return True
    try:
        check_selection(d2, run.record, labels, run.selection)
    except ValueError:
        return False
    return True


def run_gates(runs: Path, run: Run, refs: Mapping, labels: np.ndarray, val_idx: np.ndarray,
              split_sha: str) -> dict:
    key = run.key
    d1, d2 = stage1_dir(runs, key), run_dir(runs, key)
    row = {"run": f"{key.chain}/{key.run}", "failed": run.failed,
           "a": same_record_at(d2, 0, d1, START_EPOCH), "b": init_is_own(d1, d2),
           "c": (config_ok(_config(d1), refs[(key.pipeline, 1)], key)
                 and config_ok(run.config, refs[(key.pipeline, 2)], key)),
           "d": order_ok(runs, key, val_idx), "g": selection_ok(run, labels, split_sha, d2)}
    if PIPELINES[key.pipeline][0] == "mean":
        row["e"] = gamma_learned(run)
    return row


# --- gate (h) -------------------------------------------------------------------------------

def reference_values(pipeline: str, dataset: str, calibration: bool = False,
                     files: Mapping[str, Path] | None = None) -> dict[tuple[str, int], dict]:
    """(pair, seed) -> the four cross-fitted metrics of the earlier runs."""
    f = (files or REFERENCE_CSV)["m10_cal" if calibration else pipeline]
    with open(f, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if calibration:
        rows = [r for r in rows if r["group"] == "calibration" and r["dataset"] == dataset]
    else:
        rows = [r for r in rows if r["arm"] == pipeline and r["dataset"] == dataset]
    return {(r["pair"], int(r["seed"])): {m: float(r[m]) for m in METRICS} for r in rows}


def gate_h(fits: Mapping[tuple[str, str, str, int], CrossFit | None],
           files: Mapping[str, Path] | None = None) -> dict:
    """``fits``: (pipeline, dataset, pair, seed) -> R0 primary-readout cross-fit at the E010 R
    (None: failed run). Development chains of both pipelines and the M10 calibration chains."""
    rows = []
    for p in PIPELINES:
        for d in DATASETS:
            refs = {False: reference_values(p, d, False, files)}
            if p == "m10":
                refs[True] = reference_values(p, d, True, files)
            for cal, ref in refs.items():
                for (pair, seed), want in sorted(ref.items()):
                    if (seed in DEV_SEEDS) == cal:
                        continue
                    cf = fits.get((p, d, pair, seed))
                    diff = (None if cf is None else
                            max(abs(cf.mean(m) - want[m]) for m in METRICS))
                    rows.append({"run": f"{d}_{pair}_s{seed}/h0_{p}", "max_abs_diff": diff,
                                 "ok": diff is not None and diff <= REF_TOL})
    n_want = len(PIPELINES) * len(DATASETS) * len(PAIRS) * len(DEV_SEEDS) + \
        len(DATASETS) * len(PAIRS) * 3
    return {"rows": rows, "ok": len(rows) == n_want and all(r["ok"] for r in rows)}


# --- gates (s), (s2), (s3), (p) and the score files -----------------------------------------

def own_z(sf: Mapping[str, np.ndarray], variant: str, m10: bool) -> np.ndarray:
    """Offset-free z of a 32-unit variant (L20: the logits)."""
    return sf[f"{variant}_z"] if m10 else sf[f"{variant}_logits"]


def snippet_rows(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


def centre_rows(totals: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(n, 32) dense row of each window's d16c snippet."""
    starts = snippet_rows(counts)
    return np.stack([starts[i] + grid.centre_snippets(int(t)) for i, t in enumerate(totals)])


def s_row(run: Run, sf: Mapping[str, np.ndarray], val_idx: np.ndarray,
          labels: np.ndarray) -> dict:
    """Gate (s) of one run: every epoch checkpoint's sigma on its own features = the record."""
    key = run.key
    own = HEADS[key.head]
    want_auc = [h["val_video_auc"] for h in run.history2]
    row = {"run": f"{key.chain}/{key.run}",
           "epochs_ok": bool(np.array_equal(sf["epochs"], np.arange(1, EPOCHS + 1))
                             and np.array_equal(sf["ckpt_epoch"], np.arange(1, EPOCHS + 1))),
           "val_auc_ok": [float(x) for x in sf["ckpt_val_auc"]] == want_auc,
           "order_ok": bool(np.array_equal(sf["val_indices"], val_idx)
                            and np.array_equal(sf["labels"], labels)),
           "config_ok": json.loads(str(sf["config"])) ==
           json.loads(json.dumps(run.config, sort_keys=True, default=str))}
    if run.failed:
        return row | {"skipped": run.failed, "ok": True}
    probs = sf[f"{own}_probs"]
    if probs.shape != run.record.shape:
        return row | {"ok": False, "reason": f"shapes {probs.shape} / {run.record.shape}"}
    diff = np.abs(probs.astype(np.float64) - run.record.astype(np.float64))
    row["max_abs_diff"] = float(diff.max()) if np.isfinite(diff).all() else math.inf
    row["ok"] = bool(row["max_abs_diff"] <= S_TOL and row["epochs_ok"] and row["val_auc_ok"]
                     and row["order_ok"] and row["config_ok"])
    return row


def dense_failure(sf: Mapping[str, np.ndarray]) -> str | None:
    """The numerical failure of a run's dense scoring (PLAN.md "수치 실패(손실·점수에 NaN·
    무한대)"): non-finite dense snippet logits or gamma in its score file; None when finite."""
    bad = [k for k in ("dense_z", "gamma") if k in sf and not np.isfinite(sf[k]).all()]
    return f"non-finite {' and '.join(bad)} (dense scoring)" if bad else None


def s2_row(run: Run, sf: Mapping[str, np.ndarray]) -> dict:
    """Gate (s2): the dense z of every window's d16c snippet = the d16c 32-unit z (30 epochs)."""
    m10 = PIPELINES[run.key.pipeline][0] == "mean"
    head = {"run": f"{run.key.chain}/{run.key.run}"}
    failed = run.failed or dense_failure(sf)
    if failed:
        return head | {"skipped": failed, "ok": True}
    rows = centre_rows(sf["total_frames"], sf["dense_counts"])
    r = compare_logits(sf["dense_z"][:, rows], own_z(sf, "d16c", m10))
    return head | r | {"ok": r["pass"]}


def s3_row(run: Run, sf: Mapping[str, np.ndarray]) -> dict:
    """Gate (s3): z of the scale-matched snippets averaged per window = the own cache's z."""
    m10 = PIPELINES[run.key.pipeline][0] == "mean"
    head = {"run": f"{run.key.chain}/{run.key.run}"}
    failed = run.failed or dense_failure(sf)
    if failed:
        return head | {"skipped": failed, "ok": True}
    r = compare_logits(sf["s3_z"], own_z(sf, HEADS[run.key.head], m10))
    return head | r | {"ok": r["pass"]}


def p_row(run: Run, sf: Mapping[str, np.ndarray], totals: np.ndarray) -> dict:
    """Gate (p): dense rows per video = ceil(T / 16) (the dense cache index rule) and the score
    array (30, sum); non-finite values are the run's numerical failure (``dense_failure``),
    recorded here, not a gate failure."""
    head = {"run": f"{run.key.chain}/{run.key.run}"}
    counts = np.asarray(sf["dense_counts"])
    want = np.array([grid.n_snippets(int(t)) for t in totals])
    shape_ok = bool(np.array_equal(counts, want)
                    and np.array_equal(sf["total_frames"], totals)
                    and sf["dense_z"].shape == (EPOCHS, int(want.sum())))
    if run.failed:
        return head | {"shape_ok": shape_ok, "skipped": run.failed, "ok": shape_ok}
    failed = dense_failure(sf)
    row = head | {"shape_ok": shape_ok, "finite": failed is None, "ok": shape_ok}
    return row | ({"numerical_failure": failed} if failed else {})


def verify_scores(root: Path, stage: int, keys: Sequence[RunKey]) -> dict:
    """The stage's score files: every expected run file and the four feature files listed in
    scores.sha256 with the listed sha256, no other file."""
    target = Path(root) / f"stage{stage}"
    try:
        listed = {}
        for line in (target / "scores.sha256").read_text(encoding="utf-8").splitlines():
            sha, rel = line.split("  ", 1)
            listed[rel] = sha
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"scores.sha256 unreadable: {exc!r}"}
    want = {score_path(root, stage, k).relative_to(target).as_posix() for k in keys}
    want |= {f"features_{d}_{p}.npz" for d in DATASETS for p in PAIRS}
    have = {f.relative_to(target).as_posix(): f for f in current_files(target)}
    missing = sorted(want - set(have))
    extra = sorted(set(have) - want)
    bad = sorted(r for r in want & set(have) if listed.get(r) != sha256_file(have[r]))
    return {"files": len(want), "missing": missing, "unexpected": extra, "sha256_mismatch": bad,
            "ok": not missing and not extra and not bad}


def score_gates(runs: Mapping[RunKey, Run], root: Path, stage: int) -> dict:
    """(s), (s2), (s3), (p) of the stage's runs from their score files. A run whose dense scores
    are non-finite (``dense_failure``) while its losses and records are finite failed
    numerically: ``Run.failed`` is set here (after its (s) row), so the verdict treats its
    settings as failed (PLAN.md 수치 실패), and it is listed in "numerical_failures"."""
    rows = {g: [] for g in ("s", "s2", "s3", "p")}
    errors, numerical = [], {}
    splits = {d: val_split(d) for d in DATASETS}
    totals = {d: val_totals(d) for d in DATASETS}
    for key, run in runs.items():
        if key.stage != stage:
            continue
        val_idx, labels, _ = splits[key.dataset]
        try:
            sf = _npz(score_path(root, stage, key))
            rows["s"].append(s_row(run, sf, val_idx, labels))
            if key.head in ("h0", "h1"):
                rows["s2"].append(s2_row(run, sf))
            if key.head in ("hb", "hbm"):
                rows["s3"].append(s3_row(run, sf))
            if key.head in ("h0", "h1", "hb", "hbm"):
                rows["p"].append(p_row(run, sf, totals[key.dataset]))
            failed = None if run.failed else dense_failure(sf)
            if failed:
                run.failed = failed
                numerical[f"{key.chain}/{key.run}"] = failed
        except GATE_ERRORS as exc:
            errors.append(f"{key.chain}/{key.run}: {exc!r}")
    status = {g: bool(r) and all(x["ok"] for x in r) for g, r in rows.items()}
    if stage == 1:
        status.pop("s3")
    return {"rows": rows, "status": status, "errors": errors, "numerical_failures": numerical,
            "s_max_abs_diff": max((r.get("max_abs_diff", 0.0) for r in rows["s"]), default=None)}


# --- gate (f): commits and environments ------------------------------------------------------

def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def code_equal(a: str | None, b: str | None) -> bool | None:
    """The frozen code (extraction, training, scoring) is the same at commits a and b; None when
    a commit is unknown to the local repository."""
    if not a or not b:
        return None
    res = git("diff", "--quiet", a, b, "--", *FROZEN_PATHS)
    return {0: True, 1: False}.get(res.returncode)


def frozen_code_unchanged(commit: str | None) -> bool | None:
    """Start condition (5): the frozen code of the working tree (uncommitted edits included, and
    no untracked file under the frozen paths) equals that of ``commit``; None when the commit is
    unknown."""
    if not commit:
        return None
    res = git("diff", "--quiet", commit, "--", *FROZEN_PATHS)
    same = {0: True, 1: False}.get(res.returncode)
    if same is not True:
        return same
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *FROZEN_PATHS)
    if untracked.returncode != 0:
        return None
    return not untracked.stdout.strip()


def is_ancestor(a: str | None, b: str | None) -> bool:
    """a is b or an ancestor of b (git merge-base --is-ancestor, as E020); False when either is
    unknown."""
    return bool(a and b) and git("merge-base", "--is-ancestor", a, b).returncode == 0


def freeze_descent(freeze: str | None, identity: Mapping, record: Mapping, runs: Path,
                   scores: Path, stage: int, analysis: str | None) -> dict:
    """PLAN.md "실행 순서" 2: the extraction, training and analysis commits are the freeze commit
    or its descendants. Checked for the (g1) commit, every (x4) run commit of the stage's phases
    (results/e021_extract_gate.json; an entry without them fails), every batch environment
    record (env*.json), every scoring call of the stages so far and the analysis commit."""
    rows = []

    def add(what: str, commit: str | None, reason: str | None = None) -> None:
        ok = reason is None and is_ancestor(freeze, commit)
        rows.append({"what": what, "commit": commit, "ok": ok}
                    | ({"reason": reason} if reason else {}))

    add("g1", ((identity.get("g1") or {}).get("code") or {}).get("commit"))
    for d in DATASETS:
        for ph in ("val",) if stage == 1 else ("val", "trainpart"):
            entry = record.get(f"x4_{d}_{ph}") or {}
            for e in EXTRACT_ENCODERS:
                commits = (entry.get(e) or {}).get("run_commits")
                if not commits:
                    add(f"x4_{d}_{ph}_{e}", None, "no run commits recorded")
                for c in commits or ():
                    add(f"x4_{d}_{ph}_{e}", c)
    for f in sorted(Path(runs).glob("env*.json")):
        try:
            add(f.name, _json(f).get("commit"))
        except (OSError, ValueError, AttributeError) as exc:
            add(f.name, None, f"unreadable: {exc!r}")
    for k in range(1, stage + 1):
        meta_file = Path(scores) / f"stage{k}" / "meta.json"
        try:
            calls = _json(meta_file)["calls"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            add(f"scoring stage{k}", None, f"{meta_file}: {exc!r}")
            continue
        if not calls:
            add(f"scoring stage{k}", None, "no scoring call recorded")
        for c in calls:
            add(f"scoring stage{k} {c.get('time_utc')}", (c.get("code") or {}).get("commit"))
    add("analysis", analysis)
    return {"freeze_commit": freeze, "rows": rows,
            "ok": bool(freeze) and all(r["ok"] for r in rows)}


def same_env(env: Mapping | None, want: Mapping) -> bool:
    return isinstance(env, Mapping) and all(want.get(k) and env.get(k) == want.get(k)
                                            for k in ENV_KEYS)


def identity_target(identity: Mapping) -> tuple[dict, str | None, dict]:
    """Environment and commit of the passed no-change gate (g1), and the pass state of (g1) and
    (g2)."""
    g1 = identity.get("g1") or {}
    want = {k: (g1.get("environment") or {}).get(k) for k in ENV_KEYS}
    status = {g: (identity.get(g) or {}).get("all_ok") is True for g in ("g1", "g2")}
    return want, (g1.get("code") or {}).get("commit"), status


def gate_f(runs: Path, scores: Path, stage: int, identity: Mapping) -> dict:
    """Every batch environment record (env*.json in the runs root; stage 1: env_ucf.json and
    env_xdv.json, stage 2 also env_s2_<dataset>_*.json) and every scoring call of the stages so
    far: environment = the no-change gate's, deployed commit with the gate commit's frozen code."""
    want, commit, passed = identity_target(identity)
    batches = {}
    for f in sorted(Path(runs).glob("env*.json")):
        try:
            env = _json(f)
        except (OSError, ValueError):
            env = None
        same = code_equal((env or {}).get("commit"), commit)
        batches[f.name] = {"env": env, "same_env": same_env(env, want), "same_code": same,
                           "ok": bool(same_env(env, want) and same)}
    needed = [pat for k in range(1, stage + 1) for pat in ENV_FILES[k]]
    present = {pat: bool(list(Path(runs).glob(pat))) for pat in needed}
    calls = []
    for k in range(1, stage + 1):
        meta_file = Path(scores) / f"stage{k}" / "meta.json"
        try:
            meta = _json(meta_file)["calls"]
        except (OSError, ValueError, KeyError) as exc:
            calls.append({"stage": k, "ok": False, "reason": f"{meta_file}: {exc!r}"})
            continue
        for c in meta:
            same = code_equal((c.get("code") or {}).get("commit"), commit)
            calls.append({"stage": k, "time_utc": c.get("time_utc"), "same_code": same,
                          "same_env": same_env(c.get("environment"), want),
                          "ok": bool(same and same_env(c.get("environment"), want))})
    ok = (passed["g1"] and all(present.values()) and all(b["ok"] for b in batches.values())
          and bool(calls) and all(c["ok"] for c in calls))
    return {"gate_commit": commit, "want": want, "identity": passed, "env_files": present,
            "batches": batches, "scoring_calls": calls, "ok": bool(ok)}


def record_env(path: str, need: Sequence[str] = ("g1",), identity: str | None = None) -> None:
    """Server, before a training batch (README): write this machine's environment and deployed
    commit to ``path``; exit non-zero unless the no-change gates ``need`` passed and the
    environment and commit equal the (g1) record, so the batch is not launched."""
    env = {**gpu_environment(), "commit": code_version()["commit"]}
    write_json(Path(path), env)
    ident = _json(Path(identity) if identity else IDENTITY)
    want, commit, passed = identity_target(ident)
    if not (all(passed[g] for g in need) and same_env(env, want) and env["commit"] == commit):
        raise SystemExit(f"environment or commit differs from the no-change gate, or {list(need)} "
                         f"not passed: {env} vs {want} / {commit} ({passed}); do not launch")
    print(f"{path}: matches the no-change gate")


# --- all gates of a stage -------------------------------------------------------------------

def stage_runs(runs: Path, stages: Sequence[int]) -> tuple[dict[RunKey, Run], list[str]]:
    """Every run of the given plan stages, loaded and checked for numerical failure; an
    unreadable run is an error (missing runs stop the analysis)."""
    out, errors = {}, []
    for st in stages:
        for key in run_keys(st):
            try:
                out[key] = load_run(runs, key)
            except GATE_ERRORS as exc:
                errors.append(f"{key.chain}/{key.run}: {exc!r}")
    return out, errors


def training_gates(runs_root: Path, runs: Mapping[RunKey, Run]) -> dict:
    """(a)-(e), (g) of every loaded run."""
    rows, errors = [], []
    try:
        refs = references(runs_root)
    except GATE_ERRORS as exc:
        return {"rows": [], "status": {g: False for g in "abcdeg"},
                "errors": [f"references: {exc!r}"]}
    splits = {d: val_split(d) for d in DATASETS}
    for key, run in runs.items():
        val_idx, labels, sha = splits[key.dataset]
        try:
            rows.append(run_gates(runs_root, run, refs, labels, val_idx, sha))
        except GATE_ERRORS as exc:
            errors.append(f"{key.chain}/{key.run}: {exc!r}")
    status = {g: bool(rows) and all(r.get(g, True) for r in rows) for g in "abcdeg"}
    return {"rows": rows, "status": status, "errors": errors}
