"""E024 stage 1: the within-video contrast module K on validation
(experiments/E024_localisation_method/stage1/PLAN.md; validation only).

Arms, per dataset (ucf, xdv) x pair (xcb_xcl, xcb_siglip) x development seed (0, 1, 2):
    U         the stage-0a chains <u_runs>/<ds>_<pair>_s<seed>/urdmu_u200 (not retrained)
    K(rho, tau)  UR-DMU + K (``wvad.baselines.urdmu_k``), rho in {global, 0.1, 0.02} x tau in
              {1, 0.25}: <runs>/<ds>_<pair>_s<seed>/k_r<g|0.1|0.02>_t<1|0.25>
    controls  of the candidate (rho*, tau*), stage B: required C1 ctl_mlp_r<rho*>_t<tau*>, C6
              ctl_trainonly_r<rho*>_t<tau*> and I1 (``smooth``: the U logits' 5-snippet centred
              moving average, <ds>_<pair>_s<seed>/i1_smooth.npz); optional C2 ctl_tauinf, C2b
              ctl_mean and C3 ctl_shuffle (C2b, C3 only when rho* is local), reported when run
              in all 12 cells (else ignored and listed as optional_controls_ignored)
    gates     gate_k0 (``--mode off``, arguments rho g and tau 1) and gate_b (a repeat of
              k_rg_t1), UCF xcb_xcl seed 0
Every K and control chain is ``train.run(TrainConfig(method="urdmu_k", unit 200, lr and scale x1,
n_chunk 4096, deterministic), method=URDMUK(kcfg, seed))``; ``train`` writes k_config.json (the
command-line arguments, the effective KConfig, the trained g parameter count (0 for off), the
code version) into the chain folder first.

Scores and statistics as stage 0a (``e024_stage0``): the sigma point's validation logits on their
snippet supports; glance GP (video mean) x 100 primary, GN-AUC, Hit@128 and on UCF the UCA-proxy
AUC descriptive, every run on the draw matrix ``glance.bootstrap_draws(n_a, n_n, 1, 2000,
20261024)`` (videos only, row 0 the sample) shared by every arm. A contrast A - B is computed per
(dataset, pair) on the seeds both arms kept (fewer than two: no verdict for the pair and the
dataset), the arm means over those seeds, then the two pairs. A collapsed run (selected sigma
V-AUC < 0.60) is left out and a missing one stops unless named with --infra_failed, as stage 0a
(``e024_stage0.excluded``); a numerically failed K or control chain stops the analysis (PLAN.md
"붕괴와 실패").
    stage A  a development candidate selection on validation data, not a demonstration of
             improvement. Reported for every configuration: D_{d,k} = K(k) - U; sd_{d,k} over rows
             1..R (ddof 1); t^r = (D^r - D^0) / sd; q_d = 95th percentile (np.percentile, linear)
             of max_k |t^r| over the configurations with a verdict (the family); interval D^0 +-
             q_d sd (an sd of 0 stops) and its class (positive / negative / inconclusive). The
             candidate: among the valid configurations (a verdict on both datasets) the largest
             score_k = mean of the two datasets' D^0 with score_k > 0, the registration order
             (rho global, 0.1, 0.02; tau 1, 0.25) on ties; none: the search ends. Descriptive:
             whether the candidate's two lower bounds are > 0, its V-AUC - U per (dataset,
             pair) on the paired seeds and v_auc_noninferior (all >= -0.01).
    stage B  input: the committed stage-A result (``--stage_a``), needed exactly when control
             chains exist. For every control run, K(rho*, tau*) - C_j with stage A's q_d (an sd
             of 0 stops) and its class (K 우위: lower > 0, 통제 우위: upper < 0, else 차이
             불확정), and C_j - U's point estimate; final_control (the comparator of the
             confirmation): the valid required control (C_j - U with a verdict on both datasets)
             with the largest mean of C_j - U over the datasets, C1, C6, I1 on ties. No adoption
             verdict.
    sensitivity (descriptive)  per (dataset, pair) the seed block of ``bootstrap_draws(n_a, n_n,
             n_seed, 2000, 20261024)``, n_seed the number of paired seeds (its video blocks are the
             main draws), weights both arms' paired seeds in their order: per row sum_i m_i (A_i -
             B_i) / sum_i m_i, column i the i-th paired seed (a position in the intersection, not
             a seed number); its own max-T q (a reference value) and interval.
Outcome: the first matching row of PLAN.md "결과 → 다음" (a gate failure stops before any
result is written).

Commands:
    train     one K / control / gate chain (server)
    pins      results/e024_stage1_inputs.json before the freeze, once (gate (p))
    freeze    stage1/PLAN.sha256 (clean tree; the plan, the pins and the code committed)
    gate-k0   gate_k0 against the 0a U chain: val_scores z (30 points) and selection, bit for bit
    gate-b    gate_b against k_rg_t1, bit for bit
    gate-h    every chain's input hashes against the 0a U record of its cell
    gate-e    parameters - the 0a U chain's = the trained g count of k_config.json (65,793; off 0)
    rescore   a chain's ckpt_sigma_best.pt scored through ``common.score_video`` with k (s = z +
              k, trainonly included) and with k = 0 (z0), in deterministic mode: rescore.npz and
              rescore.json. Gate (s): its environment equals the chain's (else "environment
              differs": rerun once where it does), then the logits the chain scored with (z0 for
              trainonly, s otherwise) equal the stored z[point - 1] bit for bit (server, before
              the checkpoints are deleted); rescore.json also records the validation input's
              repeated snippets and the share K masks (descriptive)
    smooth    I1 from the 0a U val_scores.npz
    analyze   results/e024_stage1.json and e024_stage1_runs.csv. Before any result is written
              it stops unless: a clean tree and the freeze commit; gate (p) the pins; the
              server's gate files in the runs root (gate_k0, gate_b, gate_h, gate_e .json; stage
              B also gate_h_b, gate_e_b) all ok, and every K / control chain with a selection
              passed in (h) and (e) (stage B: the _b files); every such chain's configuration
              (CHAIN_CONFIG) and recorded code commit (k_config.json, selection.json) equal the
              freeze commit, its val_scores.npz the sha256 of selection.json, gate (n) its
              folder name against k_config.json, gate (s) a passed rescore.json whose recorded
              val_scores sha256 is the current file's, gate (v) its environment against the
              pinned 0a U chain's (none missing)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wvad.baselines import common
from wvad.baselines import train as tr
from wvad.baselines import urdmu_k as uk
from wvad.data import load_split
from wvad.data.catalog import split_file
from wvad.data.dense_test import CHANGELOG, freeze_commit_of, plan_seal, plan_sha256
from wvad.diagnostics import e024_stage0 as s0
from wvad.diagnostics import glance as gl
from wvad.diagnostics.common import code_version, write_csv, write_json
from wvad.diagnostics.e023_prefreeze import split_rows
from wvad.paths import REPO_ROOT, load_paths

STAGE1_DIR = s0.E024_DIR / "stage1"
RESULTS = s0.RESULTS
SEAL_FILE = s0.SEAL_FILE
PINS = RESULTS / "e024_stage1_inputs.json"

DATASETS, PAIRS, DEV_SEEDS = s0.DATASETS, s0.PAIRS, s0.DEV_SEEDS
RHO_TOKEN = {None: "g", 0.1: "0.1", 0.02: "0.02"}           # registration order
TAU_TOKEN = {1.0: "1", 0.25: "0.25"}
RHO_OF = {v: k for k, v in RHO_TOKEN.items()}
TAU_OF = {v: k for k, v in TAU_TOKEN.items()}
CONFIGS = tuple((r, t) for r in RHO_TOKEN for t in TAU_TOKEN)
CONTROLS = ("mlp", "tauinf", "mean", "shuffle", "trainonly")    # C1, C2, C2b, C3, C6
LOCAL_ONLY = ("mean", "shuffle")                                 # only when rho* is local
REQUIRED = ("mlp", "trainonly", "smooth")                        # C1, C6, I1: stage B's core
SELECTION_NOTE = ("development selection on validation data: the candidate is the valid "
                  "configuration with the largest positive mean of the two datasets' K - U GP "
                  "point estimates; not a demonstration of improvement (the intervals and "
                  "classes are reported, not required)")
CLASS_K, CLASS_CONTROL, CLASS_UNDETERMINED = "K 우위", "통제 우위", "차이 불확정"
U, I1, I2 = s0.U, "I1", "I2"
LABEL = {"mlp": "C1", "tauinf": "C2", "mean": "C2b", "shuffle": "C3", "trainonly": "C6",
         "smooth": "I1"}
GATE_K0, GATE_B = "gate_k0", "gate_b"
GATE_CELL = ("ucf", "xcb_xcl", 0)
FOLDER = re.compile(r"(?:k|ctl_(mlp|tauinf|mean|shuffle|trainonly))_r(g|0\.1|0\.02)_t(1|0\.25)")
K_CONFIG, RESCORE_NPZ, RESCORE_JSON = "k_config.json", "rescore.npz", "rescore.json"
I1_FILE = "i1_smooth.npz"
CHAIN_CONFIG = {"method": uk.NAME, "unit": 200, "lr_mult": "1", "scale_mult": "1",
                "deterministic": True, "n_chunk": common.N_CHUNK_DEFAULT, "iterations": None,
                "eval_points": common.EVAL_POINTS, "evaluate": True, "float64": False}
GATE_FILES = ("gate_k0.json", "gate_b.json", "gate_h.json", "gate_e.json")
GATE_FILES_B = ("gate_h_b.json", "gate_e_b.json")               # stage B: the controls too
CODE_FILES = (Path(__file__).resolve(), Path(uk.__file__).resolve())   # committed before freeze
N_BOOT, BOOT_SEED = s0.N_BOOT, s0.BOOT_SEED      # 2000, 20261024
COLLAPSE, MIN_SEEDS = s0.COLLAPSE, s0.MIN_SEEDS  # 0.60, 2 (applied by e024_stage0)
PRIMARY = s0.PRIMARY
LEVEL = 95                                       # max-T: the 95th percentile of max |t|
VAUC_MARGIN = 0.01
SMOOTH_WIDTH = 5
ENV_KEYS = ("torch", "cuda", "gpu", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
            "float32_matmul_precision")
NEXT = {"no_verdict": "stop and report (no verdict); user decision",
        "no_candidate": "no positive-mean configuration; the search ends (no stage B, no new "
                        "configuration or loss)",
        "stage_a_only": "run C1 (ctl_mlp) and C6 (ctl_trainonly) for the candidate (24 chains) "
                        "and I1 (smooth), then analyze with --stage_a <this committed file>",
        "stage_b_done": "report the K - control classes and final_control; next: the "
                        "confirmation plan (UBnormal)"}
refuse_test_paths = s0.refuse_test_paths


# --- folders and k_config.json --------------------------------------------------------------

def cells() -> list[tuple[str, str, int]]:
    return [(d, p, s) for d in DATASETS for p in PAIRS for s in DEV_SEEDS]


def k_folder(rho: float | None, tau: float) -> str:
    return f"k_r{RHO_TOKEN[rho]}_t{TAU_TOKEN[tau]}"


def control_folder(name: str, rho: float | None, tau: float) -> str:
    return f"ctl_{name}_r{RHO_TOKEN[rho]}_t{TAU_TOKEN[tau]}"


def controls_of(rho: float | None) -> tuple[str, ...]:
    """The chain controls of a candidate's rho in registration order (C2b and C3 only when rho
    is local; with rho global C2b equals C2 and C3 equals K)."""
    return tuple(c for c in CONTROLS if rho is not None or c not in LOCAL_ONLY)


def optional_cells(candidate: str, runs: Path | None) -> dict[str, list[str]]:
    """Optional control of the candidate (C2, and C2b / C3 with a local rho) -> the cells of
    ``runs`` holding its folder."""
    cfg = parse_folder(candidate)
    return {c: [s0.chain_name(*x) for x in cells() if runs is not None and (
        Path(runs) / s0.chain_name(*x) / control_folder(c, cfg["rho"], cfg["tau"])).is_dir()]
            for c in controls_of(cfg["rho"]) if c not in REQUIRED}


def stage_b_arms(candidate: str, runs: Path | None = None) -> dict[str, str]:
    """Control name -> arm (its folder; I1 last) of the candidate: the required C1 and C6 and
    the optional C2 / C2b / C3 whose folder exists in all 12 cells of ``runs`` (never required;
    one in fewer cells is ignored and listed by the analysis)."""
    cfg = parse_folder(candidate)
    full = {c for c, cs in optional_cells(candidate, runs).items() if len(cs) == len(cells())}
    return {**{c: control_folder(c, cfg["rho"], cfg["tau"]) for c in controls_of(cfg["rho"])
               if c in REQUIRED or c in full}, "smooth": I1}


def parse_folder(name: str) -> dict:
    """The mode and the (rho, tau) arguments a registered folder name carries (gate (n))."""
    if name == GATE_K0:
        return {"mode": "off", "rho": None, "tau": 1.0}
    if name == GATE_B:
        return {"mode": "k", "rho": None, "tau": 1.0}
    m = FOLDER.fullmatch(name)
    if not m:
        raise SystemExit(f"{name}: not a registered stage-1 folder name")
    return {"mode": m[1] or "k", "rho": RHO_OF[m[2]], "tau": TAU_OF[m[3]]}


def trained_g(mode: str) -> int:
    """g's trained parameter count (gate (e)): 65,793, 0 in mode off (g frozen)."""
    return 0 if mode == "off" else uk.g_parameters()


def k_config_record(kcfg: uk.KConfig, args: Mapping, seed: int) -> dict:
    return {"stage": "E024 stage 1 chain (stage1/PLAN.md K의 정의, 통제)", "args": dict(args),
            "kconfig": asdict(kcfg), "seed": int(seed), "g_parameters": trained_g(kcfg.mode),
            "code": code_version()}


def check_name(chain: Path) -> dict:
    """Gate (n): the folder name's mode and (rho, tau) tokens are k_config.json's command-line
    arguments and its KConfig is the effective one of them (``KConfig.of``; SystemExit
    otherwise); returns the record."""
    chain = Path(chain)
    want = parse_folder(chain.name)
    f = chain / K_CONFIG
    if not f.is_file():
        raise SystemExit(f"{chain}: no {K_CONFIG} (gate (n))")
    rec = json.loads(f.read_text(encoding="utf-8"))
    args = rec.get("args") or {}
    try:
        kc = asdict(uk.KConfig.of(args.get("mode"), args.get("rho"), args.get("tau")))
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"{chain}: {K_CONFIG} arguments {args} ({exc}) (gate (n))") from exc
    bad = {k: args.get(k) for k, v in want.items() if args.get(k) != v}
    if bad or rec.get("kconfig") != kc:
        raise SystemExit(f"{chain}: the folder name {want} and {K_CONFIG} {args} / "
                         f"{rec.get('kconfig')} differ (gate (n))")
    return rec


def _record(chain: Path) -> dict:
    f = Path(chain) / tr.FILES["record"]
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


# --- training and re-scoring (server) -------------------------------------------------------

def train_chain(dataset: str, pair: str, seed: int, rho: float | None, tau: float | None,
                mode: str, out_dir: str | Path, *, n_chunk: int = common.N_CHUNK_DEFAULT,
                paths_file: str | None = None, device: str | None = None,
                method: uk.URDMUK | None = None, paths=None, split=None,
                iterations: int | None = None, log=print) -> dict:
    """One chain (PLAN.md "K 명령"): k_config.json, then ``train.run`` with 0a U's settings; the
    folder name must be the registered one of (mode, rho, tau). ``method``, ``paths``, ``split``
    and ``iterations`` are the Python API's (tests)."""
    out = Path(out_dir)
    refuse_test_paths(out)
    kcfg = uk.KConfig.of(mode, rho, tau)
    args = {"mode": mode, "rho": rho, "tau": tau}
    want = parse_folder(out.name)
    if any(args[k] != v for k, v in want.items()):
        raise SystemExit(f"{out.name} is the folder of {want}, not of {args}")
    method = method or uk.URDMUK(kcfg, seed)
    if method.kcfg != kcfg or method.seed != int(seed):
        raise SystemExit(f"method {method.kcfg} seed {method.seed} for {kcfg} seed {seed}")
    if (out / tr.FILES["record"]).exists():
        raise SystemExit(f"{out / tr.FILES['record']} exists: a completed chain is not rerun")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / K_CONFIG, k_config_record(kcfg, args, seed))
    cfg = tr.TrainConfig(method=uk.NAME, unit=CHAIN_CONFIG["unit"], dataset=dataset, pair=pair,
                         seed=int(seed), out_dir=str(out), lr_mult="1", scale_mult="1",
                         n_chunk=int(n_chunk), deterministic=True, iterations=iterations,
                         paths=paths_file, device=device)
    return tr.run(cfg, method=method, paths=paths, split=split, log=log)


def context_norms(model: torch.nn.Module, x: torch.Tensor, n_chunk: int,
                  device: torch.device) -> np.ndarray:
    """(n,) ||x_t - c_t|| of one video in its N_chunk pieces (NaN without context or in a mode
    without one; eval mode set by the caller)."""
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for a, b in common.chunk_bounds(int(x.shape[0]), n_chunk):
            v = model(x[a:b].to(device=device, dtype=torch.float32)[None]).get("xc_norm")
            out.append(np.full(b - a, np.nan, np.float32) if v is None
                       else v.reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def _load_sigma(chain: Path, rec: Mapping) -> tuple[dict, str]:
    sig = (rec.get("selection") or {}).get("sigma")
    if not sig:
        raise SystemExit(f"{chain}: no sigma selection to rescore")
    f = chain / tr.FILES["sigma"]
    ck = torch.load(f, map_location="cpu", weights_only=True)
    sha = s0.sha256_file(f)
    if int(ck["eval_point"]) != int(sig["point"]) or \
            sha != rec["checkpoints"]["sigma"]["sha256"]:
        raise SystemExit(f"{f}: point {ck['eval_point']} / sha256 differ from selection.json")
    return ck, sha


def repeat_shares(xs: Sequence[torch.Tensor], n_chunk: int, context: bool) -> dict:
    """PLAN.md "복제 행 마스크" (descriptive) on the chain's processed validation input: the
    snippets equal byte for byte to the previous one and those in such a run of two or more
    (whole videos), and the snippets K actually masks beyond themselves (a run of two or more
    by ``run_ids`` inside an N_chunk piece; None in a mode without a context)."""
    n = equal = in_runs = masked = 0
    for x in xs:
        rows = np.ascontiguousarray(x.numpy()).view(np.uint8).reshape(len(x), -1)
        same = (rows[1:] == rows[:-1]).all(axis=1)
        r = np.concatenate([[0], np.cumsum(~same)])
        n, equal = n + len(rows), equal + int(same.sum())
        in_runs += int((np.bincount(r)[r] > 1).sum())
        for a, b in common.chunk_bounds(len(rows), n_chunk):
            k = uk.run_ids(x[a:b][None])[0]
            masked += int((torch.bincount(k)[k] > 1).sum())
    return {"snippets": n, "equal_to_previous": equal, "share_equal_to_previous": equal / n,
            "in_runs_of_two_or_more": in_runs, "share_in_runs": in_runs / n,
            "masked": masked if context else None,
            "share_masked": masked / n if context else None}


def _rescore_record(chain: Path, kc: Mapping, point: int, ck_sha: str, det: Mapping,
                    env: Mapping, gate_s: Mapping, **extra) -> dict:
    record = {"check": "E024 stage 1 rescore (stage1/PLAN.md 체크포인트와 점수, gate (s))",
              "chain": chain.name, "point": point, "kconfig": kc["kconfig"],
              "checkpoint_sha256": ck_sha, "gate_s": dict(gate_s), **extra,
              "environment": {**env, **det}, "code": code_version()}
    write_json(chain / RESCORE_JSON, record)
    return record


def rescore(chain: str | Path, *, paths=None, paths_file: str | None = None, split=None,
            device: str | None = None) -> dict:
    """Gate (s) and the scores of I2 and |k| (module docstring): rescore.json always,
    rescore.npz (s, z0, xc_norm, counts, val_indices, point) when the environment equals the
    chain's; SystemExit (after writing them) on another environment or when the scoring logits
    differ from the stored z[point - 1] in any bit."""
    chain = Path(chain)
    refuse_test_paths(chain)
    kc, rec = check_name(chain), _record(chain)
    cfg = rec.get("config") or {}
    ck, ck_sha = _load_sigma(chain, rec)
    point = int(ck["eval_point"])
    det = common.deterministic(True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = tr._environment(dev)
    if _env(env) != _env(rec.get("environment")):
        _rescore_record(chain, kc, point, ck_sha, det, env,
                        {"ok": False, "reason": "environment differs",
                         "chain_environment": _env(rec.get("environment"))})
        raise SystemExit(f"{chain}: gate (s): the environment differs from the chain's; rerun "
                         "rescore once in the chain's environment (before the .pt are deleted)")
    split = split if split is not None else load_split(split_file(cfg["dataset"]))
    if split.sha256 != cfg.get("split_sha256"):
        raise SystemExit(f"{chain}: the split differs from the chain's")
    val = tr.validation_videos(paths if paths is not None else load_paths(paths_file),
                               cfg["dataset"], cfg["pair"], cfg["variant"], split)
    method = uk.URDMUK(uk.KConfig(**kc["kconfig"]), int(cfg["seed"]))
    model = method.build(sum(val.dims), unit=int(cfg["unit"]),
                         input_ln=common.input_ln(cfg["pair"]), scale=cfg["scale"])
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device=dev, dtype=torch.float32)
    method.set_mode(model, False)
    unit, n_chunk = int(cfg["unit"]), int(cfg["n_chunk"])
    scores = {}
    for key, k_in_logit in (("s", True), ("z0", False)):
        model.k_in_logit = k_in_logit
        scores[key] = np.concatenate([common.score_video(method, model, x, unit=unit,
                                                         n_chunk=n_chunk, device=dev)
                                      for x in val.x])
    scores["xc_norm"] = np.concatenate([context_norms(model, x, n_chunk, dev) for x in val.x])
    with np.load(chain / tr.FILES["val"]) as f:
        stored, stored_idx = f["z"][point - 1], [int(v) for v in f["val_indices"]]
    if stored_idx != list(val.indices):
        raise SystemExit(f"{chain}: validation videos differ from val_scores.npz")
    used = "z0" if kc["kconfig"]["mode"] == "trainonly" else "s"
    v = scores[used]
    same = v.dtype == stored.dtype and v.shape == stored.shape and v.tobytes() == stored.tobytes()
    out, tmp = chain / RESCORE_NPZ, chain / "rescore.part.npz"
    np.savez(tmp, **scores, counts=np.array([len(x) for x in val.x], np.int64),
             val_indices=np.asarray(val.indices, np.int64), point=np.int64(point))
    tmp.replace(out)
    differ = int((v != stored).sum()) if v.shape == stored.shape else None
    record = _rescore_record(
        chain, kc, point, ck_sha, det, env,
        {"ok": same, "compared": used, "differing_snippets": differ},
        val_scores_sha256=s0.sha256_file(chain / tr.FILES["val"]),
        rescore_sha256=s0.sha256_file(out),
        repeats=repeat_shares(val.x, n_chunk, kc["kconfig"]["mode"] in uk.CONTEXT_MODES))
    if not same:
        raise SystemExit(f"{chain}: gate (s) failed: {used} differs from z[{point - 1}] in "
                         f"{differ} snippets; stop and report")
    return record


# --- I1 -------------------------------------------------------------------------------------

def smooth_logits(z: np.ndarray, width: int = SMOOTH_WIDTH) -> np.ndarray:
    """I1 of one video: the centred moving average of ``width`` snippets over the whole video
    (pieces ignored), the existing neighbours only at the edges and in a shorter video
    (float64)."""
    h = width // 2
    pad = np.pad(np.asarray(z, np.float64), h, constant_values=np.nan)
    return np.nanmean(np.lib.stride_tricks.sliding_window_view(pad, width), axis=1)


def write_smooth(u_runs: Path, runs: Path) -> list[Path]:
    """<runs>/<ds>_<pair>_s<seed>/i1_smooth.npz of every 0a U run with a sigma selection (a
    failed U run has no I1; a collapsed one is left out with it by the analysis)."""
    refuse_test_paths(u_runs, runs)
    out = []
    for d, p, s in cells():
        u = s0.u_run(s0.u_dir(u_runs, d, p, s), s, d, p)
        if u["failed"]:
            continue
        f = Path(runs) / s0.chain_name(d, p, s) / I1_FILE
        f.parent.mkdir(parents=True, exist_ok=True)
        z = [smooth_logits(x) for x in u["z"]]
        np.savez(f, z=np.concatenate(z), counts=np.array([len(x) for x in z], np.int64),
                 point=np.int64(u["point"]), source_sha256=np.array(u["file_sha256"]))
        out.append(f)
    return out


# --- pins, seal -----------------------------------------------------------------------------

def long_videos(gs: gl.GlanceSet) -> dict:
    """Validation videos with ceil(T / 16) > N_chunk (scored in pieces with piece-wise K)."""
    pos = list(common.chunk_targets(gs.total_frames, common.N_CHUNK_DEFAULT))
    return {"n_chunk": common.N_CHUNK_DEFAULT, "videos": len(pos),
            "anomalous": int(gs.labels[pos].sum()) if pos else 0}


def pins_record(u_runs: Path) -> dict:
    """The pre-freeze input record: the 0a U chains (sha256 of val_scores.npz and selection.json,
    parameters, environment: the server's torch), the splits, glance and UCA counts as stage 0a,
    the long videos."""
    refuse_test_paths(u_runs)
    runs = {}
    for d, p, s in cells():
        u, name = s0.u_dir(u_runs, d, p, s), s0.chain_name(d, p, s)
        sel = _record(u)
        runs[name] = {"val_scores_sha256": s0.sha256_file(u / tr.FILES["val"]),
                      "selection_sha256": s0.sha256_file(u / tr.FILES["record"]),
                      "parameters": sel.get("parameters"), "environment": sel.get("environment")}
    gs = {d: s0.validation_set(d) for d in DATASETS}
    return {"stage": "E024 stage 1 inputs (stage1/PLAN.md, gate (p))", "u_runs": runs,
            "splits": {d: s0.sha256_file(split_file(d)) for d in DATASETS},
            "glance": {d: gs[d].coverage() for d in DATASETS},
            "uca": {"file": s0.UCA_PROXY.relative_to(REPO_ROOT).as_posix(),
                    "sha256": s0.sha256_file(s0.UCA_PROXY), "variant": s0.UCA_VARIANT,
                    "videos": len(s0.uca_labels(gs["ucf"]))},
            "long_videos": {d: long_videos(gs[d]) for d in DATASETS}}


def write_pins(u_runs: Path) -> Path:
    """Gate (p), once, before the freeze."""
    if PINS.exists() or (STAGE1_DIR / SEAL_FILE).exists():
        raise SystemExit(f"{PINS.name} or the seal exists: the input record is written once, "
                         "before the freeze")
    write_json(PINS, pins_record(u_runs))
    return PINS


def check_pins(pins: Mapping, u_runs: Path) -> dict:
    """Gate (p): the inputs equal the committed record (SystemExit otherwise)."""
    now = pins_record(u_runs)
    bad = [k for k in ("u_runs", "splits", "glance", "uca", "long_videos")
           if now[k] != pins.get(k)]
    if bad:
        raise SystemExit(f"stage 1 inputs differ from {PINS.name}: {bad}")
    return {"ok": True, "record": PINS.relative_to(REPO_ROOT).as_posix(),
            "record_sha256": s0.sha256_file(PINS)}


def _committed(path: Path) -> bool:
    rel = Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    return s0._git("cat-file", "-e", f"HEAD:{rel}").returncode == 0 \
        and not s0._git("status", "--porcelain", "--", rel).stdout.strip()


def freeze() -> str:
    """stage1/PLAN.sha256 = sha256 of PLAN.md before its change log (clean tree; the plan, the
    pins, this module and urdmu_k committed and unmodified; commit the seal file alone: that
    commit is the stage-1 freeze commit)."""
    code = code_version()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed; code version {code}")
    for f in (STAGE1_DIR / "PLAN.md", PINS, *CODE_FILES):
        if not _committed(f):
            raise SystemExit(f"{f.relative_to(REPO_ROOT).as_posix()} must be committed and "
                             "unmodified at HEAD before the freeze")
    seal = STAGE1_DIR / SEAL_FILE
    if seal.exists():
        raise SystemExit(f"{seal} exists: the plan is frozen once")
    h = plan_sha256(STAGE1_DIR / "PLAN.md")
    seal.write_text(f"{h}  PLAN.md (before {CHANGELOG})\n", encoding="utf-8", newline="\n")
    return h


def require_frozen() -> dict:
    """Clean tree; stage1/PLAN.sha256 added once, alone, by the freeze commit; the plan has the
    frozen hash; the pins as committed in the freeze commit's parent (as stage 0a)."""
    code = code_version()
    if code["source"] != "git" or code["dirty"] is not False:
        raise SystemExit(f"a clean git working tree is needed; code version {code}")
    seal = plan_seal(STAGE1_DIR)
    freeze_commit = freeze_commit_of(REPO_ROOT, STAGE1_DIR)
    seal_rel = (STAGE1_DIR / SEAL_FILE).relative_to(REPO_ROOT).as_posix()
    pins_rel = PINS.relative_to(REPO_ROOT).as_posix()
    changed = s0._git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root",
                      freeze_commit or "HEAD").stdout.split()
    if freeze_commit is None or changed != [seal_rel]:
        raise SystemExit(f"the stage-1 freeze commit must add {seal_rel} alone "
                         f"({freeze_commit}, {changed})")
    at_freeze = s0._git("show", f"{freeze_commit}:{seal_rel}").stdout.split()
    if not at_freeze or at_freeze[0] != seal["sha256"]:
        raise SystemExit(f"the plan hash {seal['sha256']} is not the one sealed by "
                         f"{freeze_commit}")
    parent = s0._git("show", f"{freeze_commit}^:{pins_rel}")
    if parent.returncode != 0 or json.loads(parent.stdout) != json.loads(
            PINS.read_text(encoding="utf-8")):
        raise SystemExit(f"{pins_rel} differs from the one committed before the freeze")
    return {"plan": seal, "freeze_commit": freeze_commit, "code": code}


# --- gates ----------------------------------------------------------------------------------

def chain_dirs(runs: Path) -> tuple[list[tuple[str, str, int, Path]], list[str]]:
    """Every registered chain folder <runs>/<ds>_<pair>_s<seed>/<name> with its cell, and the
    unregistered folders found beside them."""
    found, other = [], []
    for d, p, s in cells():
        c = Path(runs) / s0.chain_name(d, p, s)
        for f in sorted(x for x in c.iterdir() if x.is_dir()) if c.is_dir() else ():
            try:
                parse_folder(f.name)
                found.append((d, p, s, f))
            except SystemExit:
                other.append(f"{c.name}/{f.name}")
    return found, other


def gate_k0(runs: Path, u_runs: Path) -> dict:
    """Gate (k0): the --mode off chain's val_scores z (every point) and selection against the 0a
    U chain of UCF xcb_xcl seed 0, bit for bit."""
    refuse_test_paths(runs, u_runs)
    d, p, s = GATE_CELL
    chain = Path(runs) / s0.chain_name(d, p, s) / GATE_K0
    rec = check_name(chain)
    return {**s0.gate_b(s0.u_dir(u_runs, d, p, s), chain), "kconfig": rec["kconfig"]}


def gate_b(runs: Path) -> dict:
    """Gate (b): the gate_b repeat against k_rg_t1 (UCF xcb_xcl seed 0), bit for bit."""
    cell = Path(runs) / s0.chain_name(*GATE_CELL)
    a, b = cell / k_folder(None, 1.0), cell / GATE_B
    for c in (a, b):
        check_name(c)
    return s0.gate_b(a, b)


def gate_h(runs: Path, u_runs: Path) -> dict:
    """Gate (h): every chain's training and validation input hashes equal its cell's 0a U record
    (every hash present)."""
    refuse_test_paths(runs, u_runs)
    chains, other = chain_dirs(runs)
    out, ok = {}, not other and bool(chains)
    for d, p, s, f in chains:
        ref = _record(s0.u_dir(u_runs, d, p, s)).get("inputs") or {}
        got = _record(f).get("inputs") or {}
        bad = [k for k in s0.INPUT_KEYS if not got.get(k) or got.get(k) != ref.get(k)]
        out[f"{f.parent.name}/{f.name}"] = {"ok": not bad, "differ": bad}
        ok = ok and not bad
    return {"ok": ok, "chains": out, "unregistered": other}


def gate_e(runs: Path, u_runs: Path) -> dict:
    """Gate (e): selection.json's parameters minus the cell's 0a U chain's = k_config.json's
    trained g parameter count = 65,793 for k and the controls, 0 for off."""
    refuse_test_paths(runs, u_runs)
    chains, other = chain_dirs(runs)
    out, ok = {}, not other and bool(chains)
    for d, p, s, f in chains:
        kc = check_name(f)
        got, ref = _record(f).get("parameters"), _record(s0.u_dir(u_runs, d, p, s)).get(
            "parameters")
        diff = None if got is None or ref is None else int(got) - int(ref)
        want = 0 if kc["kconfig"]["mode"] == "off" else uk.G_PARAMETERS
        good = kc["g_parameters"] == want and diff == want
        out[f"{f.parent.name}/{f.name}"] = {"ok": good, "difference": diff, "expected": want}
        ok = ok and good
    return {"ok": ok, "chains": out, "unregistered": other}


# --- runs -----------------------------------------------------------------------------------

def chain_run(path: Path, seed: int, dataset: str, pair: str, commit: str) -> dict:
    """One stage-1 chain as ``e024_stage0.u_run`` reads a U chain, under the analysis
    preconditions: gate (n), the whole configuration (CHAIN_CONFIG), the code commit of
    k_config.json and selection.json = the freeze commit ``commit``, val_scores.npz = its
    recorded sha256; a numerical failure stops (PLAN.md "붕괴와 실패"); a missing chain is
    marked missing."""
    path = Path(path)
    if not (path / tr.FILES["record"]).is_file():
        return {**s0._failed(path.name, seed, "missing selection.json", missing=True),
                "path": path}
    kc, rec = check_name(path), _record(path)
    cfg = rec.get("config") or {}
    want = {**CHAIN_CONFIG, "seed": seed, "dataset": dataset, "pair": pair}
    wrong = {k: cfg.get(k) for k, v in want.items() if cfg.get(k) != v}
    if wrong or kc.get("seed") != seed:
        raise SystemExit(f"{path}: config {wrong} / k_config seed {kc.get('seed')} differ from "
                         f"{want}")
    codes = {f: (r.get("code") or {}).get("commit") for f, r in ((K_CONFIG, kc),
                                                                 (tr.FILES["record"], rec))}
    if any(c != commit for c in codes.values()):
        raise SystemExit(f"{path}: code commits {codes} are not the freeze commit {commit}")
    sig = (rec.get("selection") or {}).get("sigma")
    nonfinite = int(rec.get("nonfinite_losses") or 0)
    if rec.get("failed") or not sig or nonfinite > 0:
        why = rec.get("failed") or (f"{nonfinite} non-finite losses" if nonfinite
                                    else "no sigma selection")
        raise SystemExit(f"{path}: numerical failure ({why}): stop and report (PLAN.md "
                         "붕괴와 실패)")
    info = {"path": path, "kconfig": kc["kconfig"], "parameters": rec.get("parameters"),
            "environment": rec.get("environment")}
    z_file = path / tr.FILES["val"]
    if not z_file.is_file():
        return {**s0._failed(path.name, seed, "missing val_scores.npz", missing=True), **info}
    sha = s0.sha256_file(z_file)
    if sha != rec.get("val_scores_sha256"):
        raise SystemExit(f"{z_file}: sha256 {sha} is not selection.json's")
    point, z = int(sig["point"]), s0._npz(z_file)
    return {"arm": path.name, "seed": seed, "failed": None,
            "z": split_rows(z["z"][point - 1], z["counts"]), "point": point,
            "v_auc": float(sig["v_auc"]), "val_indices": [int(x) for x in z["val_indices"]],
            "total_frames": [int(x) for x in z["total_frames"]],
            "labels": [int(x) for x in z["labels"]], "file_sha256": sha, **info}


def i1_run(path: Path, seed: int, u: Mapping) -> dict:
    """I1 of one U run (``smooth``'s file, from that run's val_scores.npz and point); left out
    with its U run."""
    if u["failed"]:
        return {**u, "arm": I1}
    if not Path(path).is_file():
        return s0._failed(I1, seed, f"missing {I1_FILE} (run smooth)", missing=True)
    f = s0._npz(path)
    if str(f["source_sha256"]) != u["file_sha256"] or int(f["point"]) != u["point"]:
        raise SystemExit(f"{path}: not the I1 of the U run {u['file_sha256']} point {u['point']}")
    return {**u, "arm": I1, "z": split_rows(f["z"], f["counts"]),
            "file_sha256": s0.sha256_file(path)}


def i2_run(k: Mapping) -> dict:
    """I2 of one kept candidate-K run: its rescore z0 (descriptive; left out without a rescore,
    never stopping the analysis)."""
    f = k["path"] / RESCORE_NPZ
    if not f.is_file():
        return s0._failed(I2, k["seed"], f"missing {RESCORE_NPZ}")
    r = s0._npz(f)
    return {**k, "arm": I2, "z": split_rows(r["z0"], r["counts"]), "file_sha256": s0.sha256_file(f)}


def k_size(s: np.ndarray, z0: np.ndarray, counts: Sequence[int]) -> dict:
    """|k| = median |s - z0| / std(z0) over the validation snippets (std ddof 0), overall and per
    video-length tertile (videos ranked by snippet count, three near-equal groups)."""
    a = np.abs(s.astype(np.float64) - z0.astype(np.float64))
    sd = float(np.std(z0.astype(np.float64)))
    a = a / sd if sd > 0 else np.full_like(a, math.nan)
    owner = np.repeat(np.arange(len(counts)), counts)
    groups = np.array_split(np.argsort(np.asarray(counts), kind="stable"), 3)
    return {"k_size": float(np.median(a)),
            **{f"k_size_t{i + 1}": float(np.median(a[np.isin(owner, g)]))
               for i, g in enumerate(groups)}}


def rescore_info(chain: Path) -> dict:
    """A chain's rescore record (gate (s) passed, rescore.npz and val_scores.npz the files it
    recorded; SystemExit otherwise) and its descriptives; {"rescore": None} without one."""
    rj = Path(chain) / RESCORE_JSON
    if not rj.is_file():
        return {"rescore": None}
    rec = json.loads(rj.read_text(encoding="utf-8"))
    npz = Path(chain) / RESCORE_NPZ
    if not rec.get("gate_s", {}).get("ok") or s0.sha256_file(npz) != rec.get("rescore_sha256") \
            or s0.sha256_file(Path(chain) / tr.FILES["val"]) != rec.get("val_scores_sha256"):
        raise SystemExit(f"{chain}: gate (s) failed, or rescore.npz / val_scores.npz differ "
                         "from rescore.json; stop and report")
    r = s0._npz(npz)
    xc = r["xc_norm"][np.isfinite(r["xc_norm"])]
    return {"rescore": rec["rescore_sha256"], **k_size(r["s"], r["z0"], r["counts"]),
            "xc_norm_val": float(np.median(xc)) if xc.size else None}


def train_xc_norm(chain: Path, point: int) -> float | None:
    """The sigma point's evaluation interval in history.json: the mean of the per-step median
    ||x_t - c_t|| over the steps that emitted it (xc_norm / xc_norm_steps; None without)."""
    f = Path(chain) / tr.FILES["history"]
    if not f.is_file():
        return None
    for e in json.loads(f.read_text(encoding="utf-8")).get("points") or []:
        if e.get("point") == point:
            loss = e.get("loss") or {}
            return loss["xc_norm"] / loss["xc_norm_steps"] if loss.get("xc_norm_steps") \
                else None
    return None


def _env(e: Mapping | None) -> dict:
    return {k: (e or {}).get(k) for k in ENV_KEYS}


def _chain_key(r: Mapping) -> str:
    return f"{r['dataset']}_{r['pair']}_s{r['seed']}/{r['arm']}"


def _selected_chains(rows: Sequence[Mapping]) -> list[Mapping]:
    """The rows of the K and control chains with a sigma selection."""
    return [r for r in rows if r["arm"].startswith(("k_", "ctl_")) and r["point"] is not None]


def run_row(cell: tuple[str, str, int], arm: str, run: Mapping, why: str | None,
            vals: Mapping | None, u_env: Mapping | None) -> dict:
    """One run's row; a K / control chain with a selection also gets gate (v)'s comparison
    with the pinned U environment of its cell and its rescore and history descriptives."""
    d, p, s = cell
    row = {"dataset": d, "pair": p, "seed": s, "arm": arm,
           **{k: (run.get("kconfig") or {}).get(k) for k in ("mode", "rho", "tau")},
           "point": run.get("point"), "v_auc": run.get("v_auc"), "excluded": why,
           "parameters": run.get("parameters"), "file_sha256": run.get("file_sha256"),
           **({} if vals is None else {k: float(v[0]) for k, v in vals.items()})}
    if "path" in run and arm not in (I1, I2) and run.get("point") is not None:
        row["environment_ok"] = _env(run.get("environment")) == _env(u_env)
        row |= {**rescore_info(run["path"]),
                "xc_norm_train": train_xc_norm(run["path"], run["point"])}
    return row


def dataset_context(d: str, pins: Mapping, commit: str) -> dict:
    """The validation set, the UCA labels (UCF), the shared video draws, the sensitivity seed
    blocks per number of paired seeds (their video blocks are the main draws), the pinned U
    environments (none empty) and the freeze commit."""
    empty = [k for k, v in pins["u_runs"].items() if not v.get("environment")]
    if empty:
        raise SystemExit(f"gate (v): no pinned U environment for {empty}")
    gs = s0.validation_set(d)
    draws = gl.bootstrap_draws(len(gs.anomalous), len(gs.normal), 1, N_BOOT, BOOT_SEED)
    seeds = {}
    for n in range(MIN_SEEDS, len(DEV_SEEDS) + 1):
        sens = gl.bootstrap_draws(len(gs.anomalous), len(gs.normal), n, N_BOOT, BOOT_SEED)
        if not (np.array_equal(draws.anomalous, sens.anomalous)
                and np.array_equal(draws.normal, sens.normal)):
            raise RuntimeError("the sensitivity draws' video blocks differ from the main draws")
        seeds[n] = sens.seeds
    return {"gs": gs, "draws": draws, "seeds": seeds,
            "labels": s0.uca_labels(gs) if d == "ucf" else None,
            "u_env": {k: v["environment"] for k, v in pins["u_runs"].items()}, "commit": commit}


def dataset_values(d: str, u_runs: Path, runs: Path, arms: Sequence[str], ctx: Mapping,
                   rows: list, infra_failed: Sequence[str] = ()) -> tuple[dict, dict, dict]:
    """values[arm][pair][seed] (``run_values`` of the kept runs), v_auc[arm][pair][seed] and the
    kept runs, for U and ``arms`` (stage-1 folders and I1); one row per run."""
    keys = (U, *arms)
    vals, vauc, kept = ({a: {p: {} for p in PAIRS} for a in keys} for _ in range(3))
    for p in PAIRS:
        for s in DEV_SEEDS:
            c = s0.chain_name(d, p, s)
            u = s0.u_run(s0.u_dir(u_runs, d, p, s), s, d, p)
            for arm in keys:
                run = u if arm == U else i1_run(Path(runs) / c / I1_FILE, s, u) if arm == I1 \
                    else chain_run(Path(runs) / c / arm, s, d, p, ctx["commit"])
                why = s0.excluded(run, c if arm in (U, I1) else f"{c}/{arm}", infra_failed)
                v = None if why else s0.run_values(run, ctx["gs"], ctx["draws"], ctx["labels"])
                rows.append(run_row((d, p, s), arm, run, why, v, ctx["u_env"].get(c)))
                if v is not None:
                    vals[arm][p][s], vauc[arm][p][s], kept[arm][p][s] = v, run["v_auc"], run
    return vals, vauc, kept


# --- contrasts ------------------------------------------------------------------------------

def seed_weighted(a: Mapping, b: Mapping, seeds: Mapping[str, list[int]],
                  blocks: Mapping[int, np.ndarray]) -> np.ndarray:
    """(R + 1,) video x seed rows of the primary A - B: per pair sum_i m[r, i] (A_i - B_i)[r] /
    sum_i m[r, i] over the paired seeds in their order, m the seed block of their number, then
    the pair mean. Column i is the i-th paired seed, a position in the intersection, not a seed
    number: when the pairs' intersections differ, one column weights different seeds in the
    two pairs (descriptive only)."""
    out = []
    for p in PAIRS:
        w = np.asarray(blocks[len(seeds[p])], np.float64)
        diff = np.stack([a[p][s][PRIMARY] - b[p][s][PRIMARY] for s in seeds[p]], axis=1)
        out.append((w * diff).sum(axis=1) / w.sum(axis=1))
    return np.mean(out, axis=0)


def _seed_sd(by_seed: Mapping[str, Mapping[int, float]]) -> float:
    """SD (ddof 1) over the seeds paired in both pairs of the pair-mean per-seed difference."""
    common_seeds = sorted(set.intersection(*(set(v) for v in by_seed.values())))
    v = [np.mean([by_seed[p][s] for p in PAIRS]) for s in common_seeds]
    return float(np.std(v, ddof=1)) if len(v) >= 2 else math.nan


def pair_contrast(a: Mapping, b: Mapping, blocks: Mapping[int, np.ndarray]) -> dict:
    """A - B on the seeds both arms kept per pair (PLAN.md "seed 짝짓기"): the primary rows
    (``_diff``, row 0 the estimate), every statistic's percentile interval (descriptive), the
    per-seed differences, the sensitivity rows (``_sens``); no verdict below MIN_SEEDS paired
    seeds in a pair."""
    seeds = {p: sorted(set(a[p]) & set(b[p])) for p in PAIRS}
    if any(len(s) < MIN_SEEDS for s in seeds.values()):
        return {"no_verdict": True, "seeds": seeds}
    ma = s0.pooled([s0.arm_mean([a[p][s] for s in seeds[p]]) for p in PAIRS])
    mb = s0.pooled([s0.arm_mean([b[p][s] for s in seeds[p]]) for p in PAIRS])
    diff = ma[PRIMARY] - mb[PRIMARY]
    if not np.isfinite(diff).all():
        raise ValueError(f"{int((~np.isfinite(diff)).sum())} non-finite bootstrap differences")
    by_seed = {p: {s: float(a[p][s][PRIMARY][0] - b[p][s][PRIMARY][0]) for s in seeds[p]}
               for p in PAIRS}
    return {"seeds": seeds, "estimate": float(diff[0]),
            "arms": {"a": float(ma[PRIMARY][0]), "b": float(mb[PRIMARY][0])},
            "statistics": {k: s0.contrast(ma[k], mb[k], primary=False) for k in ma if k in mb},
            "by_pair": {p: {**s0.seed_signs({s: float(a[p][s][PRIMARY][0]) for s in seeds[p]},
                                            {s: float(b[p][s][PRIMARY][0]) for s in seeds[p]}),
                            "sd": gl.seed_summary(by_seed[p])["sd"]} for p in PAIRS},
            "seed_sd": _seed_sd(by_seed), "_diff": diff,
            "_sens": seed_weighted(a, b, seeds, blocks)}


def public(c: Mapping) -> dict:
    """A contrast without its row arrays (the result file)."""
    return {k: v for k, v in c.items() if not k.startswith("_")}


def family_q(diffs: Mapping[str, np.ndarray]) -> float:
    """q_d: the 95th percentile (linear) over rows 1..R of max_k |(D^r_k - D^0_k) / sd_k|, sd_k
    over rows 1..R with ddof 1; an sd of 0 stops (PLAN.md A단계)."""
    sd = {k: float(np.std(v[1:], ddof=1)) for k, v in diffs.items()}
    zero = [k for k, v in sd.items() if not v > 0]
    if zero:
        raise SystemExit(f"bootstrap sd 0 for {zero}: stop and report (PLAN.md A단계)")
    t = np.stack([(v[1:] - v[0]) / sd[k] for k, v in diffs.items()])
    return float(np.percentile(np.abs(t).max(axis=0), LEVEL))


def simultaneous(c: Mapping, q: float) -> dict:
    """The interval D^0 +- q sd of a contrast (sd over rows 1..R, ddof 1)."""
    sd = float(np.std(c["_diff"][1:], ddof=1))
    return {"sd": sd, "q": q, "interval": [c["estimate"] - q * sd, c["estimate"] + q * sd]}


def sens_family(cons: Mapping[str, Mapping]) -> dict:
    """The sensitivity max-T of a family (descriptive): on the rows where every member is
    finite, each member's sd over its finite rows."""
    if not cons:
        return {"q": math.nan, "rows": 0}
    r = np.stack([c["_sens"] for c in cons.values()])
    keep = np.isfinite(r[:, 1:]).all(axis=0)
    sd = np.array([np.std(v[1:][np.isfinite(v[1:])], ddof=1) for v in r])
    if keep.sum() < 2 or not (sd > 0).all():
        return {"q": math.nan, "rows": int(keep.sum())}
    t = (r[:, 1:][:, keep] - r[:, :1]) / sd[:, None]
    return {"q": float(np.percentile(np.abs(t).max(axis=0), LEVEL)), "rows": int(keep.sum())}


def sensitivity(c: Mapping, q: float) -> dict:
    v = c["_sens"]
    rows = v[1:][np.isfinite(v[1:])]
    sd = float(np.std(rows, ddof=1)) if rows.size > 1 else math.nan
    lo, hi = (float(x) for x in np.percentile(rows, [2.5, 97.5])) if rows.size else (math.nan,
                                                                                     math.nan)
    return {"estimate": float(v[0]), "rows": int(rows.size), "sd": sd, "ci95": [lo, hi],
            "q": q, "interval": [float(v[0]) - q * sd, float(v[0]) + q * sd]}


# --- stages ---------------------------------------------------------------------------------

def first_largest(scores: Mapping[str, float], order: Sequence[str]) -> str | None:
    """The key of the largest score, the first in ``order`` on ties (None when empty)."""
    best = None
    for k in order:
        if k in scores and (best is None or scores[k] > scores[best]):
            best = k
    return best


def stage_a(vals: Mapping[str, Mapping], seed_draws: Mapping[str, Mapping]) -> dict:
    """Stage A, the development candidate selection (module docstring): every configuration's
    contrast, max-T interval and class are reported; the candidate is the valid configuration
    (a verdict on both datasets) with the largest positive score (the two datasets' mean point
    estimate), the registration order on ties."""
    arms = [k_folder(*c) for c in CONFIGS]
    datasets = {}
    for d in DATASETS:
        cons = {a: pair_contrast(vals[d][a], vals[d][U], seed_draws[d]) for a in arms}
        fam = [a for a in arms if not cons[a].get("no_verdict")]
        q = family_q({a: cons[a]["_diff"] for a in fam}) if fam else math.nan
        sq = sens_family({a: cons[a] for a in fam})
        for a in fam:
            cons[a] |= simultaneous(cons[a], q)
            cons[a]["class"] = s0.classify(*cons[a]["interval"])
            cons[a]["sensitivity"] = sensitivity(cons[a], sq["q"])
        datasets[d] = {"q": q, "family": fam, "family_size": len(fam), "no_verdict": not fam,
                       "sensitivity": sq, "configs": cons}
    valid = [a for a in arms if all(a in datasets[d]["family"] for d in DATASETS)]
    score = {a: float(np.mean([datasets[d]["configs"][a]["estimate"] for d in DATASETS]))
             for a in valid}
    cand = first_largest({a: v for a, v in score.items() if v > 0}, arms)
    return {"datasets": datasets, "no_verdict": any(datasets[d]["no_verdict"] for d in DATASETS),
            "selection_note": SELECTION_NOTE,
            "decision": {"valid": valid, "score": score, "candidate": cand,
                         "q": {d: datasets[d]["q"] for d in DATASETS},
                         "family": {d: datasets[d]["family"] for d in DATASETS}},
            "candidate_both_lower_bounds_positive": None if cand is None else all(
                datasets[d]["configs"][cand]["interval"][0] > 0 for d in DATASETS)}


def vauc_report(vauc: Mapping[str, Mapping], arm: str) -> dict:
    """Descriptive (never a stop): per (dataset, pair) the paired-seed means of the selected
    V-AUC of ``arm`` and of U and their difference; ``v_auc_noninferior`` = every difference >=
    -0.01 (None when a cell has fewer than MIN_SEEDS paired seeds)."""
    cells_ = {}
    for d in DATASETS:
        for p in PAIRS:
            k, u = vauc[d][arm][p], vauc[d][U][p]
            s = sorted(set(k) & set(u))
            if len(s) < MIN_SEEDS:
                cells_[f"{d}_{p}"] = {"no_verdict": True, "seeds": s}
                continue
            mk, mu = float(np.mean([k[x] for x in s])), float(np.mean([u[x] for x in s]))
            cells_[f"{d}_{p}"] = {"seeds": s, "k": mk, "u": mu, "difference": mk - mu}
    nv = any(c.get("no_verdict") for c in cells_.values())
    return {"cells": cells_, "margin": VAUC_MARGIN, "v_auc_noninferior": None if nv else all(
        c["difference"] >= -VAUC_MARGIN for c in cells_.values())}


def control_class(lo: float, hi: float) -> str:
    """K - C_j by its interval: K ahead, control ahead or undetermined (never "equivalent")."""
    return CLASS_K if lo > 0 else CLASS_CONTROL if hi < 0 else CLASS_UNDETERMINED


def stage_b(vals: Mapping, seed_draws: Mapping, a: Mapping, arms: Mapping[str, str]) -> dict:
    """Stage B for the candidate (module docstring): per control and dataset K - C_j with stage
    A's q_d, its class, and C_j - U's point estimate; final_control = the valid required control
    (C_j - U with a verdict on both datasets) with the largest mean of C_j - U over the datasets,
    the order C1, C6, I1 on ties."""
    sel = a["decision"]["candidate"]
    q = a["decision"]["q"]
    sq = {d: a["datasets"][d]["sensitivity"]["q"] for d in DATASETS}
    cons: dict[str, dict] = {}
    vs_u: dict[str, dict] = {}
    for j, arm in arms.items():
        cons[j], vs_u[j] = {}, {}
        for d in DATASETS:
            c = pair_contrast(vals[d][sel], vals[d][arm], seed_draws[d])
            if not c.get("no_verdict"):
                c |= simultaneous(c, q[d])
                if not c["sd"] > 0:
                    raise SystemExit(f"bootstrap sd 0 for K - {LABEL[j]} on {d}: stop and report")
                c["class"] = control_class(*c["interval"])
                c["sensitivity"] = sensitivity(c, sq[d])
            cons[j][d] = c
            u = pair_contrast(vals[d][arm], vals[d][U], seed_draws[d])
            vs_u[j][d] = None if u.get("no_verdict") else u["estimate"]
    required = [j for j in arms if j in REQUIRED]
    valid = [j for j in required if all(vs_u[j][d] is not None for d in DATASETS)]
    score = {j: float(np.mean([vs_u[j][d] for d in DATASETS])) for j in valid}
    return {"candidate": sel, "config": parse_folder(sel), "controls": list(arms),
            "labels": {j: LABEL[j] for j in arms}, "required": required,
            "optional_run": [j for j in arms if j not in REQUIRED],
            "contrasts": {j: {d: public(c) for d, c in v.items()} for j, v in cons.items()},
            "control_minus_u": vs_u, "final_control_valid": valid,
            "final_control_score": score, "final_control": first_largest(score, required)}


def outcome(a: Mapping, b: Mapping | None) -> dict:
    """PLAN.md "결과 → 다음" (scope of 2026-10-25): the first matching row. The stopping "no
    verdict" is a dataset without a configuration with a verdict or no configuration with a
    verdict on both datasets (stage A), or no valid required control (stage B); the controls
    and V-AUC are reported, never an adoption verdict."""
    if a["no_verdict"] or not a["decision"]["valid"]:
        row = "no_verdict"
    elif a["decision"]["candidate"] is None:
        row = "no_candidate"
    elif b is None:
        row = "stage_a_only"
    elif b["final_control"] is None:
        row = "no_verdict"
    else:
        row = "stage_b_done"
    return {"row": row, "next": NEXT[row]}


# --- analysis -------------------------------------------------------------------------------

def stage_a_input(runs: Path, stage_a_file: Path | None) -> dict | None:
    """Stage B's input: the committed stage-A result (``--stage_a``), needed exactly when control
    chains exist; returns its stage_a block."""
    controls = sorted(f"{x.parent.name}/{x.name}" for x in Path(runs).glob("*/ctl_*")
                      if x.is_dir())
    if not controls:
        if stage_a_file is not None:
            raise SystemExit("--stage_a given, but no control chain exists: stage A only")
        return None
    if stage_a_file is None:
        raise SystemExit(f"{len(controls)} control chains exist: stage B needs --stage_a <the "
                         "committed stage-A result>")
    text = Path(stage_a_file).read_text(encoding="utf-8")
    try:
        rel = Path(stage_a_file).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise SystemExit(f"{stage_a_file}: not in the repository") from exc
    shown = s0._git("show", f"HEAD:{rel}")
    if shown.returncode != 0 or shown.stdout != text:
        raise SystemExit(f"{rel} must be the committed stage-A result")
    rec = json.loads(text)
    a = rec.get("stage_a") or {}
    if rec.get("stage_b") is not None or not (a.get("decision") or {}).get("candidate"):
        raise SystemExit(f"{rel} is no stage-A result with a candidate")
    return a


def _canonical(x) -> str:
    return json.dumps(json.loads(json.dumps(x)), sort_keys=True)


def check_environments(rows: Sequence[Mapping]) -> dict:
    """Gate (v): every K / control chain with a selection was compared and its recorded torch,
    CUDA, GPU and TF32 flags equal its cell's pinned 0a U chain's (SystemExit otherwise)."""
    chains = _selected_chains(rows)
    unchecked = [_chain_key(r) for r in chains if "environment_ok" not in r]
    bad = [_chain_key(r) for r in chains if r.get("environment_ok") is False]
    if unchecked or bad:
        raise SystemExit(f"gate (v): environments not compared {unchecked} or differing from "
                         f"the 0a U chains {bad}")
    return {"ok": True, "keys": list(ENV_KEYS), "chains": len(chains)}


def require_rescores(rows: Sequence[Mapping]) -> dict:
    """Gate (s): every K / control chain with a selection has a rescore.json that passed
    (``rescore_info`` stops on a failed or changed one; SystemExit when one is missing)."""
    chains = _selected_chains(rows)
    missing = [_chain_key(r) for r in chains if not r.get("rescore")]
    if missing:
        raise SystemExit(f"gate (s): no passed rescore.json for {missing}; run rescore before "
                         "the .pt are deleted")
    return {"ok": True, "chains": len(chains)}


def check_gate_files(runs: Path, rows: Sequence[Mapping], stage_b: bool) -> dict:
    """The server gates' records in the runs root, each ok (stage B also the _b records); every
    K / control chain with a selection passed in gates (h) and (e) (stage B: the _b records,
    written after the controls). SystemExit otherwise."""
    names = GATE_FILES + (GATE_FILES_B if stage_b else ())
    recs = {}
    for n in names:
        f = Path(runs) / n
        recs[n] = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        if recs[n].get("ok") is not True:
            raise SystemExit(f"{f}: missing or not ok; stop and report")
    keys = [_chain_key(r) for r in _selected_chains(rows)]
    for n in (GATE_FILES_B if stage_b else GATE_FILES[2:]):
        listed = recs[n].get("chains") or {}
        absent = [k for k in keys if (listed.get(k) or {}).get("ok") is not True]
        if absent:
            raise SystemExit(f"{n}: no passed entry for {absent}; rerun the gate on all chains")
    return {n: {"ok": True, "sha256": s0.sha256_file(Path(runs) / n)} for n in names}


def descriptive_i2(sel: str, kept: Mapping, vals: Mapping, ctxs: Mapping, rows: list) -> dict:
    """I2 (the candidate K chains' rescore z0, descriptive): K - I2 and I2 - U per dataset."""
    out = {}
    for d in DATASETS:
        i2 = {p: {} for p in PAIRS}
        for p in PAIRS:
            for s, k in kept[d][sel][p].items():
                run = i2_run(k)
                why = s0.excluded(run)
                v = None if why else s0.run_values(run, ctxs[d]["gs"], ctxs[d]["draws"],
                                                   ctxs[d]["labels"])
                rows.append(run_row((d, p, s), I2, run, why, v, None))
                if v is not None:
                    i2[p][s] = v
        blocks = ctxs[d]["seeds"]
        out[d] = {"K - I2": public(pair_contrast(vals[d][sel], i2, blocks)),
                  "I2 - U": public(pair_contrast(i2, vals[d][U], blocks))}
    return out


def analyze(u_runs: Path, runs: Path, out: Path, csv_out: Path, stage_a_file: Path | None = None,
            infra_failed: Sequence[str] = ()) -> dict:
    refuse_test_paths(u_runs, runs, out, csv_out, *([stage_a_file] if stage_a_file else []))
    for root in (u_runs, runs):
        if not Path(root).is_dir():
            raise SystemExit(f"{root}: no runs folder")
    frozen = require_frozen()
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    gate_p = check_pins(pins, u_runs)
    prior = stage_a_input(runs, stage_a_file)
    arms = [k_folder(*c) for c in CONFIGS]
    arms_b = None if prior is None else stage_b_arms(prior["decision"]["candidate"], runs)
    arms += [] if arms_b is None else list(arms_b.values())
    rows: list[dict] = []
    ctxs = {d: dataset_context(d, pins, frozen["freeze_commit"]) for d in DATASETS}
    vals, vauc, kept = {}, {}, {}
    for d in DATASETS:
        vals[d], vauc[d], kept[d] = dataset_values(d, u_runs, runs, arms, ctxs[d], rows,
                                                   infra_failed)
    gate_v, gate_s = check_environments(rows), require_rescores(rows)
    gate_files = check_gate_files(runs, rows, prior is not None)
    seed_draws = {d: ctxs[d]["seeds"] for d in DATASETS}
    a = stage_a(vals, seed_draws)
    if prior is not None and _canonical(a["decision"]) != _canonical(prior["decision"]):
        raise SystemExit("stage A recomputed differs from the committed stage-A decision")
    b = None if prior is None else stage_b(vals, seed_draws, a, arms_b)
    sel = a["decision"]["candidate"]
    if b is not None:
        b["optional_controls_ignored"] = {c: cs for c, cs in optional_cells(sel, runs).items()
                                          if cs and len(cs) < len(cells())}
    report = {"check": "E024 stage 1 (stage1/PLAN.md)", **frozen, "gate_p": gate_p,
              "gate_n": {"ok": True, "chains": sum(1 for r in rows
                                                   if r["mode"] is not None and r["arm"] != I2)},
              "gate_v": gate_v, "gate_s": gate_s, "gate_files": gate_files,
              "infra_failed": list(infra_failed),
              "configs": arms[:len(CONFIGS)],
              "stage_a": {"datasets": {d: {**{k: v for k, v in r.items() if k != "configs"},
                                           "configs": {c: public(x) for c, x in
                                                       r["configs"].items()}}
                                       for d, r in a["datasets"].items()},
                          "no_verdict": a["no_verdict"], "selection_note": a["selection_note"],
                          "decision": a["decision"],
                          "candidate_both_lower_bounds_positive":
                              a["candidate_both_lower_bounds_positive"],
                          "candidate_v_auc": None if sel is None else vauc_report(vauc, sel)},
              "stage_b": b, "outcome": outcome(a, b),
              "descriptive": {"I2": None if sel is None else descriptive_i2(sel, kept, vals,
                                                                            ctxs, rows)},
              "runs": rows,
              "bootstrap": {"n_resamples": N_BOOT, "rng_seed": BOOT_SEED, "level": LEVEL,
                            "resampled": "anomalous and normal videos; seeds paired and "
                                         "averaged",
                            "sensitivity": "video x seed (the seed block of the paired seeds' "
                                           "number; column i = the i-th paired seed, a position "
                                           "in the intersection, not a seed number), "
                                           "descriptive; its q is a reference value"},
              "libraries": {"python": platform.python_version(), "numpy": np.__version__}}
    write_json(out, report)
    write_csv(csv_out, rows)
    return report


# --- command line ---------------------------------------------------------------------------

def _gate_command(name: str, res: Mapping, out: Path | None) -> None:
    if out:
        write_json(out, res)
    print(f"[e024_stage1] {name}: {'passed' if res['ok'] else 'FAILED'}")
    if not res["ok"]:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e024_stage1")
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="one K / control / gate chain")
    t.add_argument("--dataset", required=True, choices=DATASETS)
    t.add_argument("--pair", required=True, choices=PAIRS)
    t.add_argument("--seed", required=True, type=int, choices=DEV_SEEDS)
    t.add_argument("--rho", required=True, choices=list(RHO_OF))
    t.add_argument("--tau", required=True, choices=list(TAU_OF))
    t.add_argument("--mode", required=True, choices=uk.MODES)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--n_chunk", type=int, default=common.N_CHUNK_DEFAULT)
    t.add_argument("--paths", help="paths file (default configs/paths.yaml)")
    t.add_argument("--device", help="cuda / cpu (default: cuda when available)")
    pi = sub.add_parser("pins", help="write the pre-freeze input record (gate (p)), once")
    pi.add_argument("--u_runs", type=Path, required=True, help="the 0a runs root")
    sub.add_parser("freeze", help="write stage1/PLAN.sha256")
    for name in ("gate-k0", "gate-b", "gate-h", "gate-e"):
        g = sub.add_parser(name)
        g.add_argument("--runs", type=Path, required=True, help="the stage-1 runs root")
        if name != "gate-b":
            g.add_argument("--u_runs", type=Path, required=True, help="the 0a runs root")
        g.add_argument("--out", type=Path)
    rs = sub.add_parser("rescore", help="gate (s), I2 and |k| scores of one chain")
    rs.add_argument("--chain", type=Path, required=True)
    rs.add_argument("--paths", help="paths file (default configs/paths.yaml)")
    rs.add_argument("--device", help="cuda / cpu (default: cuda when available)")
    sm = sub.add_parser("smooth", help="I1 from the 0a U logits")
    sm.add_argument("--u_runs", type=Path, required=True)
    sm.add_argument("--runs", type=Path, required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--u_runs", type=Path, required=True, help="the 0a runs root")
    an.add_argument("--runs", type=Path, required=True, help="the stage-1 runs root")
    an.add_argument("--stage_a", type=Path, help="the committed stage-A result (stage B)")
    an.add_argument("--out", type=Path, default=RESULTS / "e024_stage1.json")
    an.add_argument("--csv", type=Path, default=RESULTS / "e024_stage1_runs.csv")
    an.add_argument("--infra_failed", nargs="*", default=[],
                    help="<ds>_<pair>_s<seed>/<folder> chains missing after their one rerun")
    a = ap.parse_args(argv)
    if a.command == "train":
        rec = train_chain(a.dataset, a.pair, a.seed, RHO_OF[a.rho], TAU_OF[a.tau], a.mode,
                          a.out_dir, n_chunk=a.n_chunk, paths_file=a.paths, device=a.device)
        sel = {k: v["point"] for k, v in rec["selection"].items()}
        print(f"[e024_stage1] train done: failed={rec['failed']} selection={sel}")
    elif a.command == "pins":
        print(f"[e024_stage1] pins -> {write_pins(a.u_runs)}")
    elif a.command == "freeze":
        print(f"[e024_stage1] PLAN.sha256 {freeze()}: commit it alone (the freeze commit)")
    elif a.command == "gate-k0":
        _gate_command(a.command, gate_k0(a.runs, a.u_runs), a.out)
    elif a.command == "gate-b":
        _gate_command(a.command, gate_b(a.runs), a.out)
    elif a.command in ("gate-h", "gate-e"):
        fn = gate_h if a.command == "gate-h" else gate_e
        _gate_command(a.command, fn(a.runs, a.u_runs), a.out)
    elif a.command == "rescore":
        rec = rescore(a.chain, paths_file=a.paths, device=a.device)
        print(f"[e024_stage1] rescore {a.chain}: gate (s) passed, {rec['rescore_sha256']}")
    elif a.command == "smooth":
        print(f"[e024_stage1] smooth: {len(write_smooth(a.u_runs, a.runs))} I1 files")
    else:
        rep = analyze(a.u_runs, a.runs, a.out, a.csv, a.stage_a, a.infra_failed)
        print(json.dumps(rep["outcome"], indent=1))


if __name__ == "__main__":
    main()
