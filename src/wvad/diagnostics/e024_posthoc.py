"""E024 post-hoc inference diagnostics on the completed stage-3 models (user instruction of
2026-10-27). Exploratory, after the sealed read and after the archived post-hoc analysis
(``posthoc_20261026``): the verdict not_met, the sealed score files and the sealed access log are
unchanged, nothing is trained, no seed, hyperparameter or dataset is added.

    python -m wvad.diagnostics.e024_posthoc score-uniform --runs ~/wvad_runs/e024/stage3 \
        --scores ~/wvad_runs/e024/stage3_scores --out ~/wvad_runs/e024/posthoc_uniform \
        --access_log ~/wvad_runs/e024/posthoc_access.csv       # server GPU, a deployed copy
    python -m wvad.diagnostics.e024_posthoc analyze              # local, from the stored scores

score-uniform
    For every P chain (``k_rg_t0.25``) of the stage-3 runs root, the sigma checkpoint (sha256 as
    selection.json and as the sealed ``p.npz``) is scored on the test videos exactly as the sealed
    read scored it (the dense test stores bound to the test manifest, the chain's input pipeline,
    unit and N_chunk, ``common.score_video``, deterministic fp32) three ways:
      z_k  = z + k, mode k           (must equal the sealed file's "z" bit for bit, else stop)
      z0   = z, k = 0                (must equal the sealed file's "z0" bit for bit, else stop)
      z    = z + k_uni, the same checkpoint with KConfig(rho=None, tau=None, mode="tauinf"):
             the normality term of the context weights is removed, so the weights are uniform
             over the rows outside t's own run of identical input rows (the exclusion rule of
             mode k is unchanged); g, the residual h_t - c_t, the pieces and the readout are
             unchanged. No parameter is changed or trained.
    Writes <out>/<ds>/<pair>_s<seed>/uni.npz (the fields of the sealed score files, "z" the
    uniform-context logits, plus "z_k" and "z0") and <out>/record.json (checkpoint and file
    hashes, the equality checks, the environment, the code commit). Every test-side read (the
    manifests, the dense test stores) is appended to --access_log, a new file outside the
    sealed scores folder and outside the repository; the sealed access log is not touched. The
    command refuses when the environment differs from the chains' recorded one, when a sealed
    file's sha256 differs from done.json, or when a recomputed sealed logit differs.

analyze
    From the sealed scores, the uniform-context scores, the archived per-video values and the
    existing annotations (UBnormal: the tracks files of the archive against the pinned hashes of
    the sealed read), per L video and completed model: Precision@10% and Recall@5/10/20% of P,
    UR-DMU, C6, I2 and Uni (the rules of the archived analysis: frames ranked by snippet score,
    the boundary group split uniformly), and of four untrained temporal priors on the same videos
    and budgets (constant = the random-review expectation under the uniform tie rule, rising,
    falling and centre-peaked snippet scores, as the archived follow-up). Every arm value is
    checked against the archived per-video values and every prior mean against the archived
    follow-up before any new quantity is written.
    Contrasts (points, mean over the two encoder pairs and the three completed seeds, per L
    video): P - prior (four priors) and P - Uni, each with the per-video counts improved /
    unchanged / worsened (six-model mean per video, and per model), the per-seed and per-pair
    estimates, and two percentile bootstrap intervals: conditional on the six trained models
    (videos resampled, seeds fixed) and crossed over videos and seeds. Both are given nominal
    (95%) and Bonferroni-corrected over the family of the FAMILY new contrasts of a dataset
    (5 contrasts x 4 metrics). The archived contrasts P - UR-DMU, P - C6 and P - I2 are not
    recomputed; their intervals are not reused for the new contrasts. Uni - I2 is descriptive
    (estimate only). Nothing here is an independent confirmation.
    Writes experiments/E024_localisation_method/posthoc_20261027/results.json and per_video.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dtm
import hashlib
import json
import math
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from wvad.baselines import common
from wvad.baselines import train as tr
from wvad.baselines import urdmu_k as uk
from wvad.data import dense_test as dt
from wvad.data import ubnormal as ub
from wvad.data.dense import META_FILE, DenseSet, store_dir
from wvad.data.eval_manifests import eval_manifest_file, load_eval_manifest
from wvad.diagnostics import e024_stage0 as s0
from wvad.diagnostics import e024_stage1 as s1
from wvad.diagnostics import e024_stage3 as s3
from wvad.diagnostics.common import code_version, write_json
from wvad.extract import grid
from wvad.extract.dense import log_access
from wvad.paths import REPO_ROOT, load_paths

OUT_DIR = s0.E024_DIR / "posthoc_20261027"
ARCHIVE_DIR = s0.E024_DIR / "posthoc_20261026"          # the archived post-hoc analysis (repo)
ARCHIVE_RUN = Path(os.environ.get("WVAD_E024_POSTHOC",
                                  "F:/Vision/wvad_runs/e024_posthoc_20261026"))
SCORES = Path(os.environ.get("WVAD_E024_SCORES", "F:/Vision/wvad_runs/e024/stage3_scores"))
UNIFORM = Path(os.environ.get("WVAD_E024_UNIFORM", "F:/Vision/wvad_runs/e024/posthoc_uniform"))
UBNORMAL_ZIP = REPO_ROOT / "UBnormal.zip"
COMMAND = "e024-posthoc-score-uniform"
ARM_UNI = "uni"
UNIFORM_KCONFIG = {"rho": None, "tau": None, "mode": "tauinf"}
STORE_PURPOSE = "test features (dense snippet store); post-hoc uniform-context re-scoring"
DATASETS = s3.DATASETS
PAIRS = s3.PAIRS
SEEDS = s3.SEEDS
ARMS = ("p", "u", "c6", "i2", ARM_UNI)
ARCHIVED_ARMS = ("p", "u", "c6", "i2")                  # order of the archived values array
PRIORS = ("constant", "rising", "falling", "center")
METRIC_KEYS = ("precision10", "recall05", "recall10", "recall20")
BUDGETS = (0.05, 0.10, 0.20)
REVIEW = 0.10
SNIPPET = grid.SNIPPET
N_BOOT, BOOT_SEED, ALPHA = 2000, 20261027, 0.05
FAMILY = (len(PRIORS) + 1) * len(METRIC_KEYS)            # 20 new contrasts per dataset
TOL = 1e-9
NOT_REDISTRIBUTED = s3.NOT_REDISTRIBUTED


# --- small helpers --------------------------------------------------------------------------

def utc_now() -> str:
    return dtm.datetime.now(dtm.timezone.utc).isoformat(timespec="seconds")


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _log(access_log: Path, dataset: str, rows: Sequence[Mapping]) -> None:
    log_access(access_log, [{"utc": utc_now(), "command": COMMAND, "dataset": dataset,
                             "index": "", **r} for r in rows])


# --- score-uniform (server) -----------------------------------------------------------------

def test_inputs(d: str, p: str, paths, view: Mapping, access_log: Path, variant: str
                ) -> list[torch.Tensor]:
    """The pair's dense test store through the chains' input pipeline, as the sealed read
    (``e024_stage3._test_inputs``) but logged under this command."""
    folders = dt.store_folders(paths, d, p)
    _log(access_log, d, [{"name": f.name, "path": str(f), "purpose": STORE_PURPOSE}
                         for f in folders])
    for f in folders:
        meta = store_dir(f) / META_FILE
        got = (_json(meta) if meta.is_file() else {}).get("names_sha256")
        if got != view["names_sha256"]:
            raise SystemExit(f"{meta}: names_sha256 {got} is not the test manifest's "
                             f"{view['names_sha256']}")
    dense = DenseSet(folders)
    n = len(view["names"])
    if len(dense) != n or not np.array_equal(dense.total_frames, view["total_frames"]) or \
            not np.array_equal(dense.labels, view["labels"]):
        raise SystemExit(f"{d}/{p}: the dense test store differs from the test manifest")
    undone = [v for v in range(n) if not dense.done(v)]
    if undone:
        raise SystemExit(f"{d}/{p}: {len(undone)} test videos not extracted ({undone[:5]})")
    pipe = tr.input_pipeline(paths, d, p, variant, dense.dims)
    return [pipe.apply(dense.video(v)) for v in range(n)]


def p_chain(runs: Path, d: str, p: str, s: int, inputs: Mapping) -> dict:
    """The P chain of a cell as the sealed read checked it (configuration, selection, hashes;
    the code commit is not re-checked here: the deployed copy is not the frozen one)."""
    return s3.check_chain(Path(runs) / s0.chain_name(d, p, s) / s3.ARMS["p"], (d, p, s), "p",
                          s3.settings_of(d, inputs), None)


def uniform_model(chk: Mapping, model: torch.nn.Module, dims: Sequence[int],
                  device: torch.device):
    """The same checkpoint under KConfig(rho=None, tau=None, mode="tauinf"): uniform context
    weights over the rows outside t's own run; defined for the selected global mode k only."""
    cfg = chk["record"]["config"]
    kc = chk["kconfig"]["kconfig"]
    if kc.get("mode") != "k" or kc.get("rho") is not None or kc.get("tau") is None:
        raise SystemExit(f"{chk['path']}: the uniform diagnostic is defined for the global "
                         f"configuration of mode k, got {kc}")
    method = uk.URDMUK(uk.KConfig(**UNIFORM_KCONFIG), int(cfg["seed"]))
    um = method.build(int(sum(dims)), unit=int(cfg["unit"]),
                      input_ln=common.input_ln(cfg["pair"]), scale=dict(cfg["scale"]))
    um.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    um = um.to(device=device, dtype=torch.float32)
    method.set_mode(um, False)
    um.k_in_logit = True
    return method, um


def _cat(zl: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(z) for z in zl]).astype(np.float32)


def score_uniform(runs: Path, scores: Path, out: Path, access_log: Path, *,
                  datasets: Sequence[str] = DATASETS, device: str | None = None,
                  paths_file: str | None = None) -> dict:
    runs, scores, out, access_log = Path(runs), Path(scores), Path(out), Path(access_log)
    for f in (out, access_log):
        if f.resolve().is_relative_to(scores.resolve()):
            raise SystemExit(f"{f} lies inside the sealed scores folder {scores}")
    dt.outside_repo(out, access_log)
    done = _json(scores / s3.DONE)
    if done.get("complete") is not True:
        raise SystemExit(f"{scores / s3.DONE}: the sealed read is not complete")
    inputs = s3.read_inputs()
    paths = load_paths(paths_file)
    det = common.deterministic(True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = tr._environment(dev)
    record = {"check": "E024 post-hoc: uniform-context re-scoring of the completed P checkpoints "
                       "(inference sensitivity of fixed models; no training)",
              "utc_start": utc_now(), "code": code_version(), "runs": str(runs),
              "sealed_scores": str(scores), "sealed_done_sha256": s0.sha256_file(scores / s3.DONE),
              "kconfig_uniform": dict(UNIFORM_KCONFIG), "environment": {**env, **det},
              "device": str(dev), "datasets": {}, "complete": False, "note": NOT_REDISTRIBUTED}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "record.json", record)
    for d in datasets:
        manifest = Path(eval_manifest_file(d))
        _log(access_log, d, [{"name": manifest.name, "path": str(manifest),
                              "purpose": s3.MANIFEST_PURPOSE + "; " + STORE_PURPOSE}])
        view = s3.manifest_view(d, manifest)
        drec: dict[str, dict] = {}
        for p in PAIRS:
            chains = {s3.chain_key(d, p, s, s3.ARMS["p"]): p_chain(runs, d, p, s, inputs)
                      for s in SEEDS}
            keys = list(chains)
            differ = [k for k, c in chains.items() if s1._env(c["environment"]) != s1._env(env)]
            if differ:
                raise SystemExit(f"environment differs from the chains' {differ}: score where "
                                 "they were trained")
            xs = test_inputs(d, p, paths, view, access_log, s3._variant(chains, keys, f"{d}/{p}"))
            dims = [int(xs[0].shape[1])]
            for k, chk in chains.items():
                s = int(chk["cell"][2])
                sealed = s3.score_file(scores, d, p, s, "p")
                rel = sealed.relative_to(scores).as_posix()
                sealed_sha = s0.sha256_file(sealed)
                if done["files"].get(rel) != sealed_sha:
                    raise SystemExit(f"{sealed}: sha256 differs from the sealed read's done.json")
                stored = _npz(sealed)
                method, model, sha = s3.chain_model(chk, dims, dev)
                if sha != str(stored["checkpoint_sha256"]) or sha != chk["checkpoint_sha256"]:
                    raise SystemExit(f"{k}: checkpoint sha256 {sha} differs from the sealed file")
                zk_l = s3.chain_logits(chk, method, model, xs, dev, True)
                z0_l = s3.chain_logits(chk, method, model, xs, dev, False)
                zk, z0 = _cat(zk_l), _cat(z0_l)
                equal = {"z": bool(np.array_equal(zk, stored["z"])),
                         "z0": bool(np.array_equal(z0, stored["z0"]))}
                if not all(equal.values()):
                    raise SystemExit(f"{k}: the recomputed sealed logits differ from the stored "
                                     f"file ({equal}): stop and report")
                umethod, umodel = uniform_model(chk, model, dims, dev)
                zu_l = s3.chain_logits(chk, umethod, umodel, xs, dev, True)
                zu = _cat(zu_l)
                st = s3.score_structure({"z": zu_l}, view)
                if not st["ok"]:
                    raise SystemExit(f"{k}: uniform-context logits fail the structure check {st}")
                f = s3.score_file(out, d, p, s, ARM_UNI)
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_name(f"{f.stem}.part.npz")
                np.savez(tmp, z=zu, z_k=zk, z0=z0,
                         counts=np.array([len(x) for x in zu_l], np.int64),
                         total_frames=view["total_frames"], labels=view["labels"],
                         names_sha256=np.str_(view["names_sha256"]), point=np.int64(chk["point"]),
                         checkpoint_sha256=np.str_(sha), chain=np.str_(k), arm=np.str_(ARM_UNI),
                         kconfig=np.str_(json.dumps(UNIFORM_KCONFIG)))
                tmp.replace(f)
                drec[k] = {"pair": p, "seed": s, "point": int(chk["point"]),
                           "checkpoint_sha256": sha, "sealed_file": rel,
                           "sealed_sha256": sealed_sha, "recomputed_equal": equal,
                           "uniform_file": f.relative_to(out).as_posix(),
                           "uniform_sha256": s0.sha256_file(f), "structure_ok": st["ok"],
                           "k_abs_mean": float(np.abs(zk - z0).mean()),
                           "k_uniform_abs_mean": float(np.abs(zu - z0).mean()),
                           "kconfig": dict(chk["kconfig"]["kconfig"])}
                del model, umodel
            del xs
        record["datasets"][d] = {"manifest_sha256": view["sha256"],
                                 "names_sha256": view["names_sha256"], "chains": drec}
        write_json(out / "record.json", record)
    record["utc_end"] = utc_now()
    record["access_log"] = {"path": os.path.realpath(access_log),
                            "sha256": s0.sha256_file(access_log)}
    record["complete"] = True
    write_json(out / "record.json", record)
    return record


# --- analyze (local) -------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snippet_lengths(t: int) -> np.ndarray:
    return np.minimum(SNIPPET, t - np.arange(grid.n_snippets(t)) * SNIPPET)


def snippet_positives(g: np.ndarray) -> np.ndarray:
    return np.add.reduceat(g.astype(np.int64), np.arange(0, len(g), SNIPPET))


def video_values(z: np.ndarray, g: np.ndarray) -> dict[str, float]:
    """Per-video Recall@q (q in BUDGETS) and Precision@10% of snippet scores ``z`` on frame
    labels ``g``: frames ranked by their snippet's score, tied scores one group, the group that
    contains the budget boundary entering by the share that fills the budget (the rule of the
    archived analysis and of the sealed evaluation)."""
    z = np.asarray(z, float)
    g = np.asarray(g, np.uint8)
    t = len(g)
    p = float(g.sum())
    if not 0 < p < t:
        raise ValueError("a video of the L population holds both classes")
    lens = snippet_lengths(t)
    pos = snippet_positives(g)
    if len(z) != len(lens):
        raise ValueError(f"{len(z)} scores for {len(lens)} snippets")
    order = np.argsort(-z, kind="stable")
    zo, pp, nn = z[order], pos[order], lens[order]
    starts = np.r_[0, np.flatnonzero(np.diff(zo) != 0) + 1]
    pp = np.add.reduceat(pp, starts).astype(float)
    nn = np.add.reduceat(nn, starts).astype(float)
    cum_n, cum_p = np.r_[0, np.cumsum(nn)], np.r_[0, np.cumsum(pp)]
    out = {}
    for q in BUDGETS:
        rec = float(np.interp(q * t, cum_n, cum_p) / p)
        out[f"recall{int(round(q * 100)):02d}"] = rec
        if q == REVIEW:
            out["precision10"] = rec * p / (REVIEW * t)
    return out


def prior_scores(name: str, n: int) -> np.ndarray:
    """The untrained temporal priors of the archived follow-up on ``n`` snippets."""
    t = np.linspace(0.0, 1.0, n)
    return {"constant": np.zeros(n), "rising": t, "falling": -t,
            "center": -np.abs(t - 0.5)}[name]


def annotations(ds: str, lab: Mapping) -> tuple:
    """Frame labels of every test video from the existing annotations (UBnormal: the tracks
    files of the archive, checked against the pinned hashes of the sealed read), checked against
    the sealed labels file."""
    m = load_eval_manifest(REPO_ROOT / "data" / "manifests" / f"{ds}_test.json")
    if ds == s3.CONFIRMATION:
        pinned = _json(s3.INPUTS)["tracks"]["test"]
        gts = []
        with zipfile.ZipFile(UBNORMAL_ZIP) as archive:
            for name, y, t in zip(m.names, m.labels, m.total_frames, strict=True):
                if not y:
                    gts.append(np.zeros(t, np.uint8))
                    continue
                rel = ub.tracks_rel_path(name).removeprefix("UBnormal/")
                raw = archive.read(rel)
                if sha256_bytes(raw) != pinned[name]:
                    raise ValueError(f"{name}: tracks file differs from the sealed read")
                gts.append(ub.frame_labels(t, ub.parse_tracks(raw.decode("utf-8-sig"), rel))
                           .astype(np.uint8))
    else:
        gts = [g.astype(np.uint8) for g in m.frame_gts()]
    pos = np.concatenate([snippet_positives(g) for g in gts])
    if (not np.array_equal(pos, lab["pos"])
            or not np.array_equal(m.total_frames, lab["total_frames"])):
        raise ValueError(f"{ds}: annotations differ from the sealed labels file")
    return m, gts


def arm_scores(ds: str, pair: str, seed: int, arm: str) -> tuple[np.ndarray, dict]:
    """Snippet logits of one completed model and arm, with the file's identity."""
    if arm == ARM_UNI:
        f = UNIFORM / ds / f"{pair}_s{seed}" / "uni.npz"
    else:
        f = SCORES / ds / f"{pair}_s{seed}" / ("p.npz" if arm in ("p", "i2") else f"{arm}.npz")
    z = _npz(f)
    ident = {"file": str(f).replace("\\", "/"), "sha256": s0.sha256_file(f),
             "chain": str(z["chain"]), "checkpoint_sha256": str(z["checkpoint_sha256"]),
             "point": int(z["point"])}
    return (z["z0"] if arm == "i2" else z["z"]), ident


def _draws(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)


def _interval(rows: np.ndarray, alpha: float) -> list[float]:
    r = np.asarray(rows, np.float64)
    r = r[np.isfinite(r)]
    if not r.size:
        return [math.nan, math.nan]
    return [float(v) for v in np.percentile(r, [100 * alpha / 2, 100 * (1 - alpha / 2)])]


def _class(lo: float, hi: float) -> str:
    return "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive"


def _counts(d: np.ndarray) -> dict:
    return {"improved": int((d > TOL).sum()), "unchanged": int((np.abs(d) <= TOL).sum()),
            "worsened": int((d < -TOL).sum())}


def contrast(delta: np.ndarray, w: np.ndarray, m: np.ndarray, pair_delta: Mapping[str, float]
             ) -> dict:
    """``delta`` (videos, seeds): the per-video difference of one contrast and metric, pairs
    averaged; ``w`` (R, videos) and ``m`` (R, seeds) the shared draw counts. Points."""
    d6 = delta.mean(axis=1)
    cond = (w @ d6) / w.sum(axis=1)
    crossed = ((w @ delta) * m).sum(axis=1) / (w.sum(axis=1) * m.sum(axis=1))
    a_fam = ALPHA / FAMILY
    out = {"estimate": float(d6.mean() * 100),
           "by_seed": {str(s): float(delta[:, i].mean() * 100) for i, s in enumerate(SEEDS)},
           "by_pair": {p: float(v * 100) for p, v in pair_delta.items()},
           "counts_mean6": _counts(d6),
           "conditional": {"nominal": [100 * v for v in _interval(cond, ALPHA)],
                           "corrected": [100 * v for v in _interval(cond, a_fam)]},
           "crossed": {"nominal": [100 * v for v in _interval(crossed, ALPHA)],
                       "corrected": [100 * v for v in _interval(crossed, a_fam)]}}
    out["class_conditional_corrected"] = _class(*out["conditional"]["corrected"])
    out["class_crossed_corrected"] = _class(*out["crossed"]["corrected"])
    return out


def analyze_dataset(ds: str, followup: Mapping) -> tuple[dict, list[dict]]:
    lab = _npz(SCORES / ds / s3.LABELS_FILE)
    summary = _json(SCORES / ds / s3.SUMMARY)
    lv = [int(i) for i in summary["l_videos"]]
    off = np.r_[0, np.cumsum(lab["counts"])]
    m, gts = annotations(ds, lab)
    arch = _npz(ARCHIVE_RUN / f"{ds}_per_video.npz")
    if [int(i) for i in arch["indices"]] != lv:
        raise ValueError(f"{ds}: the archived population differs from the sealed L videos")
    arch_metrics = [str(x) for x in arch["metrics"]]
    mi = {k: arch_metrics.index(k) for k in METRIC_KEYS}
    nv = len(lv)
    vals = np.full((len(ARMS), len(PAIRS), len(SEEDS), nv, len(METRIC_KEYS)), np.nan)
    files: dict[str, dict] = {}
    for pi, p in enumerate(PAIRS):
        for si, s in enumerate(SEEDS):
            ids = {}
            for ai, a in enumerate(ARMS):
                z, ident = arm_scores(ds, p, s, a)
                ids[a] = ident
                if len(z) != off[-1]:
                    raise ValueError(f"{ds}/{p}_s{s}/{a}: {len(z)} logits for {off[-1]} snippets")
                for vi, i in enumerate(lv):
                    v = video_values(z[off[i]:off[i + 1]], gts[i])
                    vals[ai, pi, si, vi] = [v[k] for k in METRIC_KEYS]
            if ids[ARM_UNI]["checkpoint_sha256"] != ids["p"]["checkpoint_sha256"] or \
                    ids[ARM_UNI]["point"] != ids["p"]["point"]:
                raise ValueError(f"{ds}/{p}_s{s}: the uniform scores are not of P's checkpoint")
            u = _npz(UNIFORM / ds / f"{p}_s{s}" / "uni.npz")
            pz = _npz(SCORES / ds / f"{p}_s{s}" / "p.npz")
            if not (np.array_equal(u["z_k"], pz["z"]) and np.array_equal(u["z0"], pz["z0"])):
                raise ValueError(f"{ds}/{p}_s{s}: the uniform file's sealed logits differ")
            files[f"{p}_s{s}"] = ids
    # checks against the archive (the four archived arms) and the archived follow-up (priors)
    err = max(float(np.abs(vals[ai, :, :, :, ki] - arch["values"][ai, :, :, :, mi[k]]).max())
              for ai, a in enumerate(ARCHIVED_ARMS) for ki, k in enumerate(METRIC_KEYS))
    pri = np.full((len(PRIORS), nv, len(METRIC_KEYS)), np.nan)
    for qi, name in enumerate(PRIORS):
        for vi, i in enumerate(lv):
            v = video_values(prior_scores(name, int(lab["counts"][i])), gts[i])
            pri[qi, vi] = [v[k] for k in METRIC_KEYS]
    fu = followup["datasets"][ds]["untrained_baselines"]
    err_prior = max(abs(float(pri[qi, :, ki].mean() * 100) - float(fu[name][k]))
                    for qi, name in enumerate(PRIORS) for ki, k in enumerate(METRIC_KEYS))
    checks = {"archived_per_video_max_abs_err": err, "archived_per_video_ok": err < TOL,
              "archived_prior_means_max_abs_err": err_prior,
              "archived_prior_means_ok": err_prior < 1e-6}
    if not (checks["archived_per_video_ok"] and checks["archived_prior_means_ok"]):
        raise SystemExit(f"{ds}: the recomputed values differ from the archive {checks}")
    # levels (points): mean over pairs, seeds and videos; per model for every arm
    levels = {a: {k: float(vals[ai, :, :, :, ki].mean() * 100) for ki, k in enumerate(METRIC_KEYS)}
              for ai, a in enumerate(ARMS)}
    levels.update({f"prior_{name}": {k: float(pri[qi, :, ki].mean() * 100)
                                     for ki, k in enumerate(METRIC_KEYS)}
                   for qi, name in enumerate(PRIORS)})
    per_model = [{"pair": p, "seed": s,
                  **{f"{a}_{k}": float(vals[ai, pi, si, :, ki].mean() * 100)
                     for ai, a in enumerate(ARMS) for ki, k in enumerate(METRIC_KEYS)}}
                 for pi, p in enumerate(PAIRS) for si, s in enumerate(SEEDS)]
    # contrasts
    rng = np.random.default_rng(BOOT_SEED)
    w, mdraw = _draws(rng, nv), _draws(rng, len(SEEDS))
    ip = ARMS.index("p")
    contrasts: dict[str, dict] = {}
    for qi, name in enumerate(PRIORS):
        for ki, k in enumerate(METRIC_KEYS):
            d = vals[ip, :, :, :, ki] - pri[qi, None, None, :, ki]          # pair, seed, video
            c = contrast(d.mean(axis=0).T, w, mdraw,
                         {p: float(d[pi].mean()) for pi, p in enumerate(PAIRS)})
            c["counts_by_model"] = [{"pair": p, "seed": s, **_counts(d[pi, si])}
                                    for pi, p in enumerate(PAIRS) for si, s in enumerate(SEEDS)]
            contrasts[f"P-{name}/{k}"] = c
    iu = ARMS.index(ARM_UNI)
    ii = ARMS.index("i2")
    for ki, k in enumerate(METRIC_KEYS):
        d = vals[ip, :, :, :, ki] - vals[iu, :, :, :, ki]
        c = contrast(d.mean(axis=0).T, w, mdraw,
                     {p: float(d[pi].mean()) for pi, p in enumerate(PAIRS)})
        c["counts_by_model"] = [{"pair": p, "seed": s, **_counts(d[pi, si])}
                                for pi, p in enumerate(PAIRS) for si, s in enumerate(SEEDS)]
        c["same_checkpoint"] = True
        contrasts[f"P-Uni/{k}"] = c
        du = vals[iu, :, :, :, ki] - vals[ii, :, :, :, ki]
        contrasts[f"Uni-I2/{k}"] = {"descriptive": True, "estimate": float(du.mean() * 100),
                                    "by_seed": {str(s): float(du[:, si].mean() * 100)
                                                for si, s in enumerate(SEEDS)},
                                    "counts_mean6": _counts(du.mean(axis=0).mean(axis=0))}
    rows = []
    kp = METRIC_KEYS.index("precision10")
    for vi, i in enumerate(lv):
        row = {"dataset": ds, "index": i, "name": m.names[i], "frames": int(m.total_frames[i]),
               "snippets": int(lab["counts"][i]), "positives": int(gts[i].sum()),
               "prevalence": float(gts[i].mean())}
        for ai, a in enumerate(ARMS):
            for ki, k in enumerate(METRIC_KEYS):
                row[f"{a}_{k}_mean6"] = float(vals[ai, :, :, vi, ki].mean() * 100)
        for qi, name in enumerate(PRIORS):
            for ki, k in enumerate(METRIC_KEYS):
                row[f"{name}_{k}"] = float(pri[qi, vi, ki] * 100)
        row["d_P_Uni_precision10_mean6"] = float(
            (vals[ip, :, :, vi, kp] - vals[iu, :, :, vi, kp]).mean() * 100)
        row["d_P_center_precision10_mean6"] = float(
            (vals[ip, :, :, vi, kp] - pri[PRIORS.index("center"), vi, kp]).mean() * 100)
        rows.append(row)
    block = {"n_videos": nv, "l_videos": lv, "files": files, "checks": checks, "levels": levels,
             "per_model": per_model, "contrasts": contrasts,
             "random_precision10_archived": followup["datasets"][ds]["random_precision10"]}
    return block, rows


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def analyze() -> dict:
    done = _json(SCORES / s3.DONE)
    urec = _json(UNIFORM / "record.json")
    if urec.get("complete") is not True:
        raise SystemExit(f"{UNIFORM / 'record.json'}: the uniform re-scoring is not complete")
    followup = _json(ARCHIVE_RUN / "followup.json")
    results = {"check": "E024 post-hoc 2026-10-27: untrained temporal priors and the "
                        "uniform-context inference diagnostic on the completed models",
               "purpose": "exploratory; after the verdict; the verdict not_met, the sealed files "
                          "and the archived analysis are unchanged; no training",
               "utc": utc_now(), "code": code_version(),
               "inputs": {"sealed_done_sha256": s0.sha256_file(SCORES / s3.DONE),
                          "sealed_complete": done.get("complete"),
                          "uniform_record_sha256": s0.sha256_file(UNIFORM / "record.json"),
                          "uniform_code": urec.get("code"),
                          "uniform_environment": urec.get("environment"),
                          "archived_followup_sha256": s0.sha256_file(ARCHIVE_RUN / "followup.json"),
                          "archive_dir": ARCHIVE_DIR.relative_to(REPO_ROOT).as_posix()},
               "arms": {"p": "UR-DMU + K, scored with s = z + k (the sealed logits)",
                        "u": "UR-DMU", "c6": "the training-only control, scored with z",
                        "i2": "P's checkpoint scored with k = 0",
                        ARM_UNI: "P's checkpoint scored with uniform context weights "
                                 "(KConfig rho None, tau None, mode tauinf); same g, pieces, "
                                 "exclusion of t's own run"},
               "priors": {"constant": "all snippets tied: the random-review expectation under "
                                      "the uniform tie rule (Precision@10% = prevalence, "
                                      "Recall@q = q)",
                          "rising": "score increases with time",
                          "falling": "score decreases with time",
                          "center": "centre-peaked score -|t - 1/2|"},
               "metrics": list(METRIC_KEYS), "budgets": list(BUDGETS),
               "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "alpha": ALPHA,
                             "family": FAMILY,
                             "family_definition": "per dataset: 5 new contrasts (P - four "
                             "priors, P - Uni) x 4 metrics; Bonferroni alpha / 20; the "
                             "conditional interval resamples videos with the six models fixed, "
                             "the crossed interval resamples videos and seeds (pairs averaged "
                             "first); the archived contrasts P - U, P - C6 and P - I2 are not "
                             "recomputed and their intervals are not reused",
                             "classes": "by the corrected interval: positive (lower > 0), "
                             "negative (upper < 0), inconclusive otherwise; an inconclusive "
                             "contrast is neither equivalence nor evidence of no effect"},
               "interpretation": {"P-prior": "whether the completed models present more "
                                  "anomalous frames than an untrained temporal rule under the "
                                  "same budget",
                                  "P-Uni": "inference sensitivity of the fixed models to the "
                                  "normality weighting; not a comparison with a model trained "
                                  "with uniform context and not evidence of a general necessity "
                                  "of the weighting"},
               "datasets": {}, "note": NOT_REDISTRIBUTED}
    rows_all: list[dict] = []
    for ds in DATASETS:
        block, rows = analyze_dataset(ds, followup)
        results["datasets"][ds] = block
        rows_all += rows
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "results.json", results)
    write_csv(OUT_DIR / "per_video.csv", rows_all)
    for ds, b in results["datasets"].items():
        lv = b["levels"]
        print(ds, f"n={b['n_videos']}", "Prec@10:",
              " ".join(f"{a}={lv[a]['precision10']:.2f}" for a in ARMS),
              " ".join(f"{n}={lv[f'prior_{n}']['precision10']:.2f}" for n in PRIORS))
        for name in (*(f"P-{n}" for n in PRIORS), "P-Uni"):
            c = b["contrasts"][f"{name}/precision10"]
            print(f"  {name:12s} {c['estimate']:+.2f} cond {c['conditional']['corrected']} "
                  f"crossed {c['crossed']['corrected']} counts {c['counts_mean6']}")
    print(OUT_DIR / "results.json")
    return results


# --- command line ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("score-uniform", help="server: uniform-context re-scoring of the P "
                                             "checkpoints (module docstring)")
    u.add_argument("--runs", required=True)
    u.add_argument("--scores", required=True, help="the sealed scores folder (read only)")
    u.add_argument("--out", required=True, help="a new folder outside the sealed scores")
    u.add_argument("--access_log", required=True, help="a new access log (outside the repo)")
    u.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    u.add_argument("--device", default=None)
    u.add_argument("--paths", default=None, help="configs/paths.yaml (default: the usual one)")
    sub.add_parser("analyze", help="local: priors and the uniform diagnostic from the stored "
                                   "scores (module docstring)")
    a = ap.parse_args(argv)
    if a.cmd == "score-uniform":
        rec = score_uniform(Path(a.runs), Path(a.scores), Path(a.out), Path(a.access_log),
                            datasets=tuple(a.datasets), device=a.device, paths_file=a.paths)
        print(json.dumps({k: rec[k] for k in ("utc_start", "utc_end", "complete", "device")}))
        for d, b in rec["datasets"].items():
            print(d, len(b["chains"]), "chains",
                  all(c["recomputed_equal"]["z"] and c["recomputed_equal"]["z0"]
                      for c in b["chains"].values()))
    else:
        analyze()


if __name__ == "__main__":
    main()
