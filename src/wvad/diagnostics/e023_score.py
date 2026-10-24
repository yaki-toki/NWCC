"""E023 server code (PLAN.md "일정·초매개변수 규칙(기본 D4)", "체크포인트 선택", "실행 순서와
시험셋 봉인" 6-9, "구현"): the (3c) training-input record, the D4 validation check, the
validation re-scoring of the kept checkpoints (gate (s)) and the C.L20 post-hoc dense validation
scoring, the checkpoint selection with the D14 bookkeeping, and test access (A). Fixed numeric
code from the freeze on (PLAN.md "코드 규칙" 1): the forward path (``forward_*``: a model on its
inputs) is kept apart from the writing and indexing of the score files (``*_arrays``,
``write_scores``, ``gt.score_file``, ``rescore_files``), so the one permitted fix after a
writing / indexing failure of gate (v3) touches no forward function. New code (add-only rule);
the comparison methods are used only through the registry (``wvad.baselines.common.get_method``)
and always on a dataset (``common.dataset_method``: the dataset's official configuration, e.g.
VCV's XD option file; a chain's recorded recipe must equal it, ``chain_method``).

Runs (README "Stage F — E023"): <runs>/<chain>/<method>_u<unit> (wvad.baselines.train: the main
family at the official unit with --save_eval_ckpts, X.32 at 32), <runs>/<chain>/hb_l20 (C.L20:
the frozen wvad.train, E021's HB L20 command plus --save_epoch_ckpts), development chains for D4
<runs>/dev/<chain>/<method>_u<unit>_lr<m>_sc<m> (multiplier "/" written "d").

    inputs --out FILE --e021_runs DIR
        (3c) (server, no test access): the 42 E021 training inputs and 6 scale maps
        (dense_test.train_input_record) and the sha256 of the E021 HB L20 reference configs
        (seed 0, stages 1 and 2) merged into FILE (key "train_inputs"; derive_n writes
        "derived_<dataset>" into the same file)
    env --out FILE [--train_inputs FILE]
        gate (f) before each training batch: this machine's torch, CUDA, GPU and driver, the
        deployed commit and the training-input check, written first; exit 1 (do not launch)
        unless the environment equals E022's (a different driver alone is recorded)
    devlr --runs_root R [--out results/e023_devlr.json] [--family ...]
        D4 from the finished development chains (validation only): per method and setting the
        lr scores x{1/3, 1, 3} (widened once to x1/9 or x9 when an end value wins), then the
        scale constants x{1/10, 1, 10} at the chosen lr (recipes.d4_score / d4_pick) over the
        chains that did not collapse (PLAN.md "붕괴 제외"; recorded per setting under
        "collapsed", each value's surviving seeds under "surviving"; a value with fewer than
        D4.min_surviving is never chosen, and a setting without an eligible value is listed
        under "no_eligible" with no choice); "choices" holds every complete setting; each
        setting records the official lr and the scale constants of its dataset's
        configuration. Single-seed D4 (D4.seed_mode "single", PLAN.md "단일 seed 실행과 판정"):
        a value's score is that of its first development seed (0, then 1, then 2) that did not
        collapse, min_surviving 1
    q1d --runs_root R --record results/e023_q1b.json [--family ...]
        (q1d) sanity floor (pre-freeze, validation only) from the development chains at the
        official recipe (lr x1, scale x1; <runs>/dev/<dataset>_xcb_xcl_s<0-2>/<method>_u<unit>
        _lr1_sc1, the chains (3f) and D4 use too): per dataset each method's mean selected
        sigma V-AUC over its chains that did not collapse against the MIL port's mean - 2.0
        points (e023_gates.q1d_record), merged into the record under "q1d"; exit 1 (nothing
        written) while a chain or selection is missing
    jobs --stage d4|confirm --runs_root R [--family ...] [--devlr FILE] [--infra_failed ..]
        one line per chain still to train: "method unit dataset pair seed lr_mult scale_mult
        out_dir d14" (d4: the development chains D4 needs next; confirm: every main-family and
        X.32 arm's chains from the committed D4 choices; single-seed mode, e023_stats.SEED_MODE: per
        chain seed 42, or the next of 123, 777 after a collapsed, numerically failed or
        --infra_failed attempt, ``confirm_seeds``)
    score --split val --runs_root R --out S [--family ...] [--infra_failed ..] [--rescore REASON]
        no test data: every comparison-method chain's kept checkpoints (sigma, logit, last)
        re-scored on the validation videos (gate (s)), S/val/<chain>/<run>.npz; every C.L20
        chain's 30 stage-2 epoch checkpoints on its 32-unit and dense validation inputs
        (e021_score.score_run: eval mode, fp32, autocast off, the E021 d16m map) and its
        ckpt_last.pt on the dense validation snippets, S/val/<chain>/hb_l20.npz. Val and test:
        a comparison-method chain's forward path runs in its recorded training mode
        (chain_numerics; on CUDA CUBLAS_WORKSPACE_CONFIG=:4096:8, else refused first)
    select --runs_root R --scores S [--family ...] [--infra_failed ..] [--out ..] [--all_ckpts ..]
        per chain the evaluation point of the highest full-validation V-AUC of the W readout
        (sigma: max fp32 sigma(u_k), logit: max u_k; no offset), strict >, earliest point on
        ties; the last point; checkpoint sha256; per chain "collapsed" (PLAN.md "실패 처리"
        "붕괴 제외": selected sigma V-AUC < 0.60; E022 C's chains from its committed selection
        under "collapse"; single-seed mode: only each chain's attempts and the block
        "single_seed", nothing written while a fallback seed is still to train) ->
        results/e023_selection.json; D14: the 30
        evaluation-point checkpoints of every main-family chain with their V-AUCs ->
        results/e023_all_ckpts.json (commit both before access (A))
    score --split test --runs_root R --out S --test_log LOG --selection_sha256 X
          --all_ckpts_sha256 Y --memory_need_gb G [--freeze_commit F] [--n_chunk N] ...
        access (A), one call: gate (m) first (free GPU memory >= (3j) x 1.2, else a "refused"
        line: wait and call again), the access device (1 descent from the freeze commit, 2 the
        frozen plan, 3 the D1 approval, 4-5 gate (t), 6 the committed selection files and every
        checkpoint in them with its sha256, 7 the blocked-checkpoint guard, 8 the training
        inputs at the start and the end), the "start" line before any test feature is read,
        then every new arm's sigma-best, logit-best and last checkpoint and (D14) the main
        family's 30 evaluation-point checkpoints on the dense test stores (the chain's own
        input processing; whole videos, cut into N_chunk pieces only when n_v > N_chunk;
        C.L20 through the frozen score_dense with the E021 d16m map). Later reads only
        --missing_only (infrastructure) or --runs CHAIN/RUN[__RULE] (a writing / indexing fix
        after (v3) failed), each with --rescore

Test score files S/test/<chain>/<run>__<rule>.npz (under a temporary name, then renamed): names,
total_frames, labels of the restricted manifest read (manifest order), the checkpoint's sha256,
point and step; z (rows,) float32 snippet logits, counts (n,), zc (rows,) the logit readout
(z + c(v), c(v) = 0 for every new arm: no offset, no gamma stored), sigma (rows,) fp32 sigma(z);
D14 <run>__eval.npz: z (30, rows) snippet logits only, with the 30 checkpoints' sha256, points and
steps. The call's meta S/test/meta_<k>.json (k: the log line of its "start") records every file's
sha256 right after writing it, the loaded checkpoints (gate (k)), the videos cut into pieces
(gate (v3), (q6)), and S/test/scores.sha256 lists the current files.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt_time
import json
import os
import shutil
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from wvad.baselines import common as bc
from wvad.baselines import train as bt
from wvad.baselines.recipes import D4Grid, behaviour, d4_pick, d4_score, d4_widen
from wvad.data import dense_test as dt
from wvad.data import load_split, split_file
from wvad.data.catalog import ENCODER_PAIRS, dense_dir
from wvad.data.dense import DenseSet
from wvad.diagnostics import e021_score as es
from wvad.diagnostics import e023_access as ea
from wvad.diagnostics import e023_gates as gt
from wvad.diagnostics import e023_stats as st23
from wvad.diagnostics.common import code_version, gpu_environment, write_json
from wvad.diagnostics.e023_stats import COLLAPSE_V_AUC, collapsed
from wvad.eval.dense import score_dense, snippet_logits
from wvad.eval.dense_readout import sigmoid32
from wvad.extract import derive_n, grid
from wvad.models.head import SegmentHead
from wvad.paths import REPO_ROOT, load_paths
from wvad.prep.manifests import sha256_file
from wvad.train.stages import INPUT_LAYER_NORM

COMMAND = "e023-score-test"
# D4 in the experiment's seed mode (e023_stats.SEED_MODE, the one switch; PLAN.md "단일 seed 실행과
# 판정"): single -> min_surviving 1, multi -> 2 (recipes.D4Grid enforces both)
D4 = D4Grid(seed_mode=st23.SEED_MODE)
ACCESS = "A"
SCORE_KEYS = ("z", "zc", "sigma")
CL20_KEY = ("hb", "l20")                         # e021_score RunKey head / pipeline
CL20_CONFIG_RUNS = (gt.CL20_STAGE1, gt.CL20_RUN)
Resolver = Callable[[str], bc.BaselineMethod]


def now_utc() -> str:
    return dt_time.datetime.now(dt_time.timezone.utc).isoformat(timespec="seconds")


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _outside(*paths) -> None:
    dt.outside_repo(*paths)


def _merge(path: Path, key: str, value) -> None:
    p = Path(path)
    rec = _json(p) if p.exists() else {}
    rec[key] = value
    write_json(p, rec)


def official_units(family: Sequence[str], resolver: Resolver = bc.get_method) -> dict[str, int]:
    """The official training unit of each family method (through the registry)."""
    return {m: int(resolver(m).official_unit) for m in family}


def chain_method(resolver: Resolver, name: str, cfg: Mapping) -> bc.BaselineMethod:
    """The method a chain ran: ``name`` with its dataset's official configuration
    (common.dataset_method), checked against the chain's recorded recipe (config.json)."""
    method = bc.dataset_method(resolver(name), str(cfg["dataset"]))
    rec = cfg.get("recipe")
    if rec is not None and behaviour(dict(rec)) != behaviour(method.recipe.to_json()):
        raise SystemExit(f"{name} on {cfg['dataset']}: the chain's recorded recipe is not the "
                         "dataset configuration's (another configuration was trained)")
    return method


def family_arms(family: Sequence[str], resolver: Resolver = bc.get_method) -> dict[str, gt.Arm]:
    return gt.arm_table(gt.check_family(family), official_units(family, resolver))


# --- (3c) training-input record -----------------------------------------------------------------

def cl20_reference_files(e021_runs: Path) -> dict[str, Path]:
    """The E021 HB L20 development chains (seed 0) of every dataset and pair, stages 1 and 2:
    the reference configs of gate (a) for C.L20 (8 files)."""
    return {f"{gt.chain_name(d, p, 0)}/{r}/ckpt_last.pt":
            Path(e021_runs) / gt.chain_name(d, p, 0) / r / "ckpt_last.pt"
            for d in gt.DATASETS for p in gt.PAIRS for r in CL20_CONFIG_RUNS}


def inputs_record(paths, e021_runs: Path) -> dict:
    """(3c): dense_test.train_input_record (42 files, 6 scale maps) and the C.L20 references."""
    rec = dt.train_input_record(paths)
    files = cl20_reference_files(e021_runs)
    absent = [k for k, p in files.items() if not p.is_file()]
    if absent:
        raise SystemExit(f"E021 HB L20 reference checkpoints missing: {absent}")
    rec["cl20_configs"] = {"runs": "E021 runs root (not recorded as a path)",
                           "files": {k: sha256_file(p) for k, p in files.items()}}
    return rec | {"utc": now_utc(), "code": code_version()}


def inputs_check(record_file: Path, paths) -> dict:
    """Access device 8: the 42 E021 training inputs and 6 scale maps and every derived E023
    training input against the committed (3c) record."""
    try:
        rec = _json(record_file)
        base = dt.check_train_inputs(rec["train_inputs"], paths)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"{record_file}: no training-input record ({exc!r})") from exc
    derived = {k: derive_n.check_derived(v, paths) for k, v in rec.items()
               if k.startswith("derived_")}
    return {"record": Path(record_file).name, "record_sha256": sha256_file(record_file),
            "train_inputs": base, "derived": derived,
            "ok": bool(base["ok"] and all(v["ok"] for v in derived.values()))}


def require_inputs(record_file: Path, paths) -> dict:
    res = inputs_check(record_file, paths)
    if not res["ok"]:
        raise SystemExit(f"training inputs differ from {record_file}: {res}")
    return res


# --- gate (f): batch environment ------------------------------------------------------------------

def e022_training_env(results: Path = dt.RESULTS) -> dict:
    """E022's training environment (its first batch record results/env_e022_b*.json)."""
    files = sorted(Path(results).glob("env_e022_b*.json"))
    if not files:
        raise SystemExit(f"no E022 batch environment record in {results}")
    rec = _json(files[0])
    return {k: rec.get(k) for k in gt.TRAIN_ENV_KEYS}


def batch_env(paths, out: Path, train_inputs: Path, want: Mapping | None = None) -> dict:
    """Gate (f) record of a training batch (written first): this machine's torch, CUDA, GPU and
    driver, the deployed commit and the training-input check; SystemExit (do not launch) unless
    the environment equals E022's (a different driver alone is recorded) and the inputs equal
    the committed (3c) record."""
    env = gpu_environment()
    code = code_version()
    inputs = inputs_check(train_inputs, paths)
    want = dict(want if want is not None else e022_training_env())
    diff = [k for k in gt.TRAIN_ENV_KEYS if env.get(k) != want.get(k)]
    rec = {**env, "commit": code.get("commit"), "code": code, "inputs_check": inputs,
           "want": want, "env_differences": diff, "utc": now_utc()}
    write_json(Path(out), rec)
    if not inputs["ok"] or diff not in ([], ["driver"]) or not code.get("commit"):
        raise SystemExit(f"environment {diff}, training inputs ok={inputs['ok']} or no commit: "
                         "do not launch the batch")
    return rec


# --- D4 (validation only) -----------------------------------------------------------------------

def mult_code(m: str) -> str:
    return str(m).replace("/", "d")


def dev_run(method: str, unit: int, lr: str, scale: str) -> str:
    return f"{gt.run_name(method, unit)}_lr{mult_code(lr)}_sc{mult_code(scale)}"


def dev_dir(runs: Path, dataset: str, pair: str, seed: int, run: str) -> Path:
    return Path(runs) / "dev" / gt.chain_name(dataset, pair, seed) / run


def dev_seeds_next(runs: Path, method: str, unit: int, d: str, p: str, lr: str, sc: str
                   ) -> list[int]:
    """The development seeds of one value still to train: multi-seed D4 every unfinished seed;
    single-seed D4 (PLAN.md "단일 seed 실행과 판정", "세부 규칙") the first unfinished seed after
    collapsed or numerically failed ones (none once a seed is usable or every seed failed)."""
    todo = []
    for s in D4.seeds:
        f = dev_dir(runs, d, p, s, dev_run(method, unit, lr, sc))
        if not (f / bt.FILES["record"]).is_file():
            if D4.seed_mode == "single":
                return [s]
            todo.append(s)
            continue
        if D4.seed_mode == "single" and dev_failure(f) is None:
            return []
    return todo


def dev_failure(f: Path) -> str | None:
    """A finished development chain's failure for single-seed D4: "numerical" (a non-finite
    validation V-AUC: its D4 score would be NaN; PLAN.md "세부 규칙"), "collapsed" (selected
    sigma V-AUC < 0.60), None (usable)."""
    with np.load(Path(f) / bt.FILES["val"]) as z:
        r = np.asarray(z["v_auc_sigma"], dtype=np.float64)
    if not np.isfinite(r).all():
        return "numerical"
    return "collapsed" if collapsed(float(r.max())) else None


def _dev_score(runs: Path, method: str, unit: int, d: str, p: str, lr: str, sc: str,
               left_out: list[str] | None = None, numerical: list[str] | None = None
               ) -> tuple[float | None, int | None]:
    """(D4 score, surviving development seeds) of one value ((None, None) until all development
    seeds it needs finished): the score over its chains that did not collapse (PLAN.md "실패
    처리" "붕괴 제외": the selected, i.e. highest, sigma V-AUC below 0.60; their '<chain>/<run>'
    added to ``left_out``), NaN when every seed collapsed; d4_pick takes the value only with at
    least D4.min_surviving surviving seeds. Single-seed D4 (D4.seed_mode "single", PLAN.md "단일
    seed 실행과 판정", "세부 규칙"): the seeds in order, the score of the first that neither
    collapsed nor failed numerically (a non-finite V-AUC; added to ``numerical``); 1 surviving
    seed; later seeds are not read; none usable: NaN, 0 (not eligible)."""
    if D4.seed_mode == "single":
        for s in D4.seeds:
            f = dev_dir(runs, d, p, s, dev_run(method, unit, lr, sc))
            if not (f / bt.FILES["record"]).is_file():
                return None, None
            why = dev_failure(f)
            if why is None:
                with np.load(f / bt.FILES["val"]) as z:
                    r = np.asarray(z["v_auc_sigma"], dtype=np.float64)
                return d4_score(r[None, :], D4), 1
            into = left_out if why == "collapsed" else numerical
            if into is not None and f"{f.parent.name}/{f.name}" not in into:
                into.append(f"{f.parent.name}/{f.name}")
        return float("nan"), 0
    rows = []
    for s in D4.seeds:
        f = dev_dir(runs, d, p, s, dev_run(method, unit, lr, sc))
        if not (f / bt.FILES["record"]).is_file():
            return None, None
        with np.load(f / bt.FILES["val"]) as z:
            rows.append((f, np.asarray(z["v_auc_sigma"], dtype=np.float64)))
    keep = []
    for f, r in rows:
        if not collapsed(float(r.max())):
            keep.append(r)
        elif left_out is not None and f"{f.parent.name}/{f.name}" not in left_out:
            left_out.append(f"{f.parent.name}/{f.name}")
    return (d4_score(np.stack(keep), D4) if keep else float("nan")), len(keep)


def _no_eligible(stage: str) -> str:
    if D4.seed_mode == "single":
        return (f"{stage}: no value with a usable development seed (PLAN.md 붕괴 제외, 세부 "
                "규칙); the official value is used")
    return (f"{stage}: no value with at least {D4.min_surviving} surviving development seeds "
            "(PLAN.md 붕괴 제외); nothing chosen (user decision)")


def d4_state(runs: Path, method: str, unit: int, dataset: str, pair: str,
             has_scale: bool) -> dict:
    """The D4 state of one method and setting: lr scores (widened once), scale scores at the
    chosen lr, each value's surviving development seeds ("surviving"; a value with fewer than
    D4.min_surviving is never chosen), the values still to train, the choice once complete, the
    collapsed development chains left out ("collapsed"), the numerically failed ones
    ("numerical", single-seed D4) and "no_eligible" when no value of a stage is eligible (then,
    multi-seed, no choice and nothing more to train; single-seed, PLAN.md "세부 규칙", the
    official value is used for that stage and recorded under "official_used")."""
    left: list[str] = []
    bad: list[str] = []
    lr = {m: _dev_score(runs, method, unit, dataset, pair, m, "1", left, bad)
          for m in D4.lr_mults}
    lr_scores, lr_alive = {m: v[0] for m, v in lr.items()}, {m: v[1] for m, v in lr.items()}
    need = [(m, "1") for m, v in lr_scores.items() if v is None]
    out = {"lr": {"scores": lr_scores, "surviving": lr_alive, "pick": None, "widened": None,
                  "final": None},
           "scale": None, "need": need, "complete": False, "choice": None, "collapsed": left,
           "numerical": bad, "no_eligible": None, "official_used": []}
    if need:
        return out
    pick = d4_pick(lr_scores, D4, surviving=lr_alive)
    out["lr"]["pick"] = pick
    if pick is None:
        out["no_eligible"] = _no_eligible("lr")
        if D4.seed_mode != "single":
            return out
        pick = D4.official
        out["official_used"].append("lr")
    wide = d4_widen(pick, D4)
    final = pick
    if wide is not None:
        ws, wn = _dev_score(runs, method, unit, dataset, pair, wide, "1", left, bad)
        out["lr"]["widened"] = {"value": wide, "score": ws, "surviving": wn}
        if ws is None:
            out["need"] = [(wide, "1")]
            return out
        final = d4_pick({**lr_scores, wide: ws}, D4, surviving={**lr_alive, wide: wn})
    out["lr"]["final"] = final
    sc_final = "1"
    if has_scale:
        sc = {m: _dev_score(runs, method, unit, dataset, pair, final, m, left, bad)
              for m in D4.scale_mults}
        sc_scores, sc_alive = {m: v[0] for m, v in sc.items()}, {m: v[1] for m, v in sc.items()}
        out["scale"] = {"scores": sc_scores, "surviving": sc_alive, "pick": None}
        out["need"] = [(final, m) for m, v in sc_scores.items() if v is None]
        if out["need"]:
            return out
        sc_final = d4_pick(sc_scores, D4, surviving=sc_alive)
        out["scale"]["pick"] = sc_final
        if sc_final is None:
            out["no_eligible"] = "; ".join(x for x in (out["no_eligible"],
                                                       _no_eligible("scale")) if x)
            if D4.seed_mode != "single":
                return out
            sc_final = D4.official
            out["official_used"].append("scale")
    out["complete"] = True
    out["choice"] = {"lr_mult": final, "scale_mult": sc_final}
    return out


def devlr_record(runs: Path, family: Sequence[str], resolver: Resolver = bc.get_method) -> dict:
    """results/e023_devlr.json: every main-family method x setting (X.32 takes the choice of the
    same method and setting; C.L20 runs E021's HB L20 command)."""
    settings, choices = {}, {}
    for m in gt.check_family(family):
        base = resolver(m)
        unit = int(base.official_unit)
        for d in gt.DATASETS:
            method = bc.dataset_method(base, d)
            rec = method.recipe
            for p in gt.PAIRS:
                st = d4_state(runs, m, unit, d, p, bool(rec.scale_constants))
                st["official"] = {"lr": float(rec.optimizer.lr),
                                  "scale_constants": dict(rec.scale_constants)}
                settings[f"{m}/{d}/{p}"] = st
                if st["complete"]:
                    choices[f"{m}/{d}/{p}"] = st["choice"]
    return {"check": "E023 D4 validation check (PLAN.md 검증 점검; development seeds 0-2"
                     + (", the first that did not collapse: 단일 seed 실행과 판정)"
                        if D4.seed_mode == "single" else ")"),
            "code": code_version(), "time_utc": now_utc(), "grid": D4.to_json(),
            "family": list(family), "settings": settings, "choices": choices,
            "no_eligible": sorted(k for k, v in settings.items() if v["no_eligible"]),
            "complete": len(choices) == len(settings)}


def job_line(method: str, unit: int, d: str, p: str, seed: int, lr: str, sc: str, out: Path,
             d14: bool) -> str:
    return f"{method} {unit} {d} {p} {seed} {lr} {sc} {Path(out).as_posix()} {int(d14)}"


def q1d_chain_ok(cfg: Mapping, resolver: Resolver, method: str, unit: int, dataset: str,
                 seed: int) -> bool:
    """Whether a development chain's config.json is the official-recipe chain its path names:
    the method, unit, dataset and seed, XCB+XCL, lr x1, scale x1, no iterations override and the
    dataset's official recipe without its documentation fields (recipes.DOC_KEYS; gate (a)'s
    comparison for these keys)."""
    want = {"method": method, "unit": unit, "dataset": dataset, "pair": gt.Q1D_PAIR,
            "seed": seed, "lr_mult": "1", "scale_mult": "1", "iterations": None}
    recipe = bc.dataset_method(resolver(method), dataset).recipe.to_json()
    return (all(cfg.get(k) == v for k, v in want.items())
            and behaviour(cfg.get("recipe")) == behaviour(recipe))


def q1d_values(runs: Path, family: Sequence[str], resolver: Resolver = bc.get_method
               ) -> dict[str, dict[str, dict[int, float | None]]]:
    """(q1d) inputs: method -> dataset -> development seed -> the selected sigma V-AUC of the
    official-recipe development chain (XCB+XCL, lr x1, scale x1; None when the chain has no
    record or no selection, or its config.json is missing or not the official-recipe chain's,
    ``q1d_chain_ok``)."""
    out: dict = {}
    for m in gt.check_family(family):
        unit = int(resolver(m).official_unit)
        for d in gt.DATASETS:
            for s in gt.DEV_SEEDS:
                chain = dev_dir(runs, d, gt.Q1D_PAIR, s, dev_run(m, unit, "1", "1"))
                f, cfg = chain / bt.FILES["record"], chain / bt.FILES["config"]
                v = None
                if f.is_file() and cfg.is_file() and q1d_chain_ok(_json(cfg), resolver, m,
                                                                  unit, d, s):
                    sel = (_json(f).get("selection") or {}).get("sigma") or {}
                    v = sel.get("v_auc")
                out.setdefault(m, {}).setdefault(d, {})[s] = None if v is None else float(v)
    return out


def q1d_command(args: argparse.Namespace, resolver: Resolver = bc.get_method) -> dict:
    """``q1d``: e023_gates.q1d_record from the development chains, merged into --record
    ("q1d"); SystemExit (nothing written) while a value is missing."""
    family = gt.check_family(args.family)
    rec = gt.q1d_record(q1d_values(args.runs_root, family, resolver), family)
    rec |= {"code": code_version(), "time_utc": now_utc(),
            "chains": "<runs>/dev/<dataset>_xcb_xcl_s<seed>/<method>_u<unit>_lr1_sc1"}
    if not rec["complete"]:
        raise SystemExit(f"(q1d) incomplete, nothing written: missing {rec['missing']}")
    bc.update_json(args.record, lambda r: r.__setitem__("q1d", rec),
                   default={"check": "E023 (q1b) official-pipeline comparisons (PLAN.md 관문)",
                            "methods": {}})
    flagged = sorted(m for m, r in rec["methods"].items() if r["flag"])
    print(f"[e023_score] (q1d) -> {args.record}; flagged '적응 하의 약함': {flagged}")
    return rec


def d4_jobs(runs: Path, family: Sequence[str], resolver: Resolver = bc.get_method) -> list[str]:
    rec = devlr_record(runs, family, resolver)
    lines = []
    for label, st in rec["settings"].items():
        m, d, p = label.split("/")
        unit = int(resolver(m).official_unit)
        for lr, sc in st["need"]:
            for s in dev_seeds_next(runs, m, unit, d, p, lr, sc):
                out = dev_dir(runs, d, p, s, dev_run(m, unit, lr, sc))
                lines.append(job_line(m, unit, d, p, s, lr, sc, out, False))
    return lines


def confirm_jobs(runs: Path, devlr: Mapping, family: Sequence[str],
                 resolver: Resolver = bc.get_method, infra_failed: Sequence[str] = (),
                 gate_failed: Sequence[str] = (), incomplete: list[str] | None = None
                 ) -> list[str]:
    """The confirmation chains of every main-family and X.32 arm not finished yet
    (``confirm_seeds``; single-seed mode: seed 42, or the next fallback seed; a chain with an
    attempt still running or crashed is not listed, its label added to ``incomplete``)."""
    arms = family_arms(family, resolver)
    lines = []
    for a in arms.values():
        if a.role == "mechanism":
            continue
        for d in gt.DATASETS:
            for p in gt.PAIRS:
                ch = (devlr.get("choices") or {}).get(f"{a.method}/{d}/{p}")
                if ch is None:
                    raise SystemExit(f"no D4 choice for {a.method}/{d}/{p}: commit the complete "
                                     "results/e023_devlr.json first")
                for s in confirm_seeds(runs, a, d, p, infra_failed, gate_failed, incomplete):
                    out = Path(runs) / gt.chain_name(d, p, s) / a.run
                    lines.append(job_line(a.method, a.unit, d, p, s, ch["lr_mult"],
                                          ch["scale_mult"], out, a.d14))
    return lines


def run_status(out: Path) -> str | None:
    """The attempt outcome of a finished comparison-method chain from its own selection.json
    (validation only; e023_gates.ATTEMPT_*): None while unfinished, "numerical" (a failed
    record, a non-finite training loss or no sigma selection), "collapsed" (selected sigma V-AUC
    < 0.60), else "ok". ``selection_record`` recomputes the selection and decides."""
    f = Path(out) / bt.FILES["record"]
    if not f.is_file():
        return None
    rec = _json(f)
    v = ((rec.get("selection") or {}).get("sigma") or {}).get("v_auc")
    if rec.get("failed") or int(rec.get("nonfinite_losses") or 0) > 0 or v is None:
        return gt.ATTEMPT_NUMERICAL
    return gt.ATTEMPT_COLLAPSED if collapsed(v) else gt.ATTEMPT_OK


def started(runs: Path, label: str) -> bool:
    """Whether an attempt's run folder exists (a comparison-method run; C.L20: either stage)."""
    d = Path(runs) / label
    return d.exists() or (d.name == gt.CL20_RUN and (d.parent / gt.CL20_STAGE1).exists())


def confirm_seeds(runs: Path, arm: gt.Arm, d: str, p: str, infra_failed: Sequence[str] = (),
                  gate_failed: Sequence[str] = (), incomplete: list[str] | None = None
                  ) -> list[int]:
    """The confirmation seeds of one chain still to train: multi-seed every unfinished seed;
    single-seed (PLAN.md "단일 seed 실행과 판정") the next seed of 42, 123, 777 after collapsed,
    numerically failed, infrastructure-failed ('<chain>/<run>' in ``infra_failed``) or
    gate-failed (``gate_failed``, before access (A)) attempts, none once a seed is ok or all three
    failed. An attempt whose run folder exists without a finished record is running or crashed:
    nothing is listed for the chain (no second run into the same folder) and its label goes to
    ``incomplete`` (wait for the batch; a crash is rerun once, then named with --infra_failed)."""
    if st23.SEED_MODE != st23.SINGLE:
        return [s for s in gt.SEEDS if not (Path(runs) / gt.chain_name(d, p, s) / arm.run /
                                            bt.FILES["record"]).is_file()]
    status = {}
    for s in gt.SINGLE_SEEDS:
        label = f"{gt.chain_name(d, p, s)}/{arm.run}"
        if label in infra_failed or label in gate_failed:
            status[s] = gt.ATTEMPT_INFRA if label in infra_failed else gt.ATTEMPT_GATE
            continue
        status[s] = run_status(Path(runs) / label)
        if status[s] is None and started(runs, label):
            if incomplete is not None:
                incomplete.append(label)
            return []
        if status[s] in (None, gt.ATTEMPT_OK):
            break
    nxt = gt.single_seed(status)["next"]
    return [] if nxt is None else [nxt]


# --- validation: re-scoring (gate (s)) and C.L20 --------------------------------------------------

def val_inputs(paths, dataset: str, pair: str, variant: str, split) -> dict:
    """The validation snippets processed as in training (the chain's InputPipeline on the E021
    training-split store; ``wvad.baselines.train.load_inputs``'s validation part)."""
    dense = DenseSet([dense_dir(paths, dataset, e) for e in ENCODER_PAIRS[pair]])
    pipeline = bt.input_pipeline(paths, dataset, pair, variant, dense.dims)
    idx = [int(i) for i in split.val_indices]
    return {"x": [pipeline.apply(dense.video(v)) for v in idx], "val_indices": idx,
            "dim": int(sum(dense.dims))}


def build_model(method: bc.BaselineMethod, state: Mapping, cfg: Mapping, dim: int,
                device: torch.device) -> torch.nn.Module:
    """The chain's network (its config's unit, pair and scale constants) with ``state``, fp32,
    eval mode."""
    model = method.build(int(dim), unit=int(cfg["unit"]), input_ln=bc.input_ln(cfg["pair"]),
                         scale=dict(cfg["scale"]))
    model.load_state_dict(dict(state))
    model = model.to(device=device, dtype=torch.float32)
    method.set_mode(model, False)
    return model


def forward_baseline(method: bc.BaselineMethod, model, xs: Sequence[torch.Tensor], *, unit: int,
                     n_chunk: int, device: torch.device) -> list[np.ndarray]:
    """Snippet logits of every video (common.score_video: whole videos, N_chunk pieces only when
    n_v > N_chunk; the method's own pieces inside), eval mode, no grad, autocast off."""
    method.set_mode(model, False)
    return [bc.score_video(method, model, x, unit=unit, n_chunk=n_chunk, device=device)
            for x in xs]


NUMERICS = ("comparison-method chains scored in their recorded training mode (config "
            "'deterministic': common.deterministic; chain_numerics, PLAN.md 변경 기록 2026-10-24)")


@contextlib.contextmanager
def chain_numerics(cfg: Mapping):
    """A comparison-method chain's forward path in its own training mode (PLAN.md 변경 기록
    2026-10-24, gate (s)): a chain trained with --deterministic is scored after
    common.deterministic(True) too, so its kept checkpoints give the stored logits bit for bit
    (without it RTFM's Conv1d takes another cuDNN algorithm); the previous flags are restored."""
    prev = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    if bool(cfg.get("deterministic")):
        bc.deterministic(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev[2], prev[3]


def require_numerics_env(cuda: bool) -> None:
    """chain_numerics on CUDA needs CUBLAS_WORKSPACE_CONFIG=:4096:8 (common.deterministic):
    checked before any score file is moved, written or read."""
    if cuda and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("CUBLAS_WORKSPACE_CONFIG=:4096:8 is needed: the comparison-method "
                         "chains are scored in their deterministic training mode "
                         "(chain_numerics)")


def forward_hb(model: SegmentHead, dense: DenseSet, n: int, device: torch.device
               ) -> list[np.ndarray]:
    """C.L20's offset-free snippet logits (the frozen score_dense; the E021 d16m map in
    ``dense``)."""
    return score_dense(model, dense, range(n), device)


def hb_model(obj: Mapping, dim: int, device: torch.device) -> SegmentHead:
    """A SegmentHead from a loaded wvad.train checkpoint (wvad.eval.scoring.load_head's rule)."""
    cfg = obj["config"]
    model = SegmentHead(int(dim), use_input_ln=INPUT_LAYER_NORM[cfg["pair"]],
                        global_mode=cfg.get("global_mode", "none"))
    model.load_state_dict(obj["model_state_dict"])
    return model.to(device).eval()


def chain_record(runs: Path, chain: str, run: str) -> tuple[dict, dict]:
    """(selection.json, config.json) of a comparison-method chain."""
    d = Path(runs) / chain / run
    return _json(d / bt.FILES["record"]), _json(d / bt.FILES["config"])


def rescore_chain(runs: Path, chain: str, arm: gt.Arm, inputs: Mapping, device: torch.device,
                  resolver: Resolver) -> dict[str, np.ndarray]:
    """Gate (s) input of one chain: its kept checkpoints re-scored on the validation videos."""
    rec, cfg = chain_record(runs, chain, arm.run)
    method = chain_method(resolver, arm.method, cfg)
    out: dict[str, np.ndarray] = {"val_indices": np.asarray(inputs["val_indices"], np.int64),
                                  "counts": np.array([len(x) for x in inputs["x"]], np.int64)}
    for rule in gt.RULES:
        ck = (rec.get("checkpoints") or {}).get(rule)
        if not ck:
            continue
        path = Path(runs) / chain / arm.run / ck["file"]
        obj = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(method, obj["model_state_dict"], cfg, inputs["dim"], device)
        with chain_numerics(cfg):
            z = forward_baseline(method, model, inputs["x"], unit=int(cfg["unit"]),
                                 n_chunk=int(cfg["n_chunk"]), device=device)
        out |= {f"z_{rule}": np.concatenate(z).astype(np.float32),
                f"point_{rule}": np.int64(ck["point"]),
                f"sha256_recorded_{rule}": np.str_(ck["sha256"]),
                f"sha256_{rule}": np.str_(sha256_file(path))}
    if rec.get("eval_ckpts"):
        names = sorted(rec["eval_ckpts"])
        out["eval_points"] = np.asarray(names)
        out["eval_sha256"] = np.asarray([sha256_file(Path(runs) / chain / arm.run /
                                                     bt.EVAL_CKPT.format(int(k)))
                                         if (Path(runs) / chain / arm.run /
                                             bt.EVAL_CKPT.format(int(k))).is_file() else ""
                                         for k in names])
    return out


def cl20_key(dataset: str, pair: str, seed: int) -> es.RunKey:
    return es.RunKey(dataset, pair, seed, *CL20_KEY)


def score_cl20(runs: Path, key: es.RunKey, units, dense, device: torch.device) -> dict:
    """C.L20 post-hoc validation scoring (PLAN.md: E022 C's path): the 30 epoch checkpoints
    (e021_score.score_run), ckpt_last.pt on the dense snippets, the run files' sha256."""
    arrays = es.score_run(runs, key, {"d16m": units}, dense, None, device)
    d = es.run_dir(runs, key)
    obj = torch.load(d / "ckpt_last.pt", map_location="cpu", weights_only=True)
    model = hb_model(obj, int(dense.features.shape[1]), device)
    arrays |= {"last_dense_z": snippet_logits(model, dense.features, device),
               "last_sha256": np.str_(sha256_file(d / "ckpt_last.pt")),
               "last_epoch": np.int64(obj["epoch"]),
               "selection_sha256": np.str_(sha256_file(d / "selection.json")),
               "history_sha256": np.str_(sha256_file(d / "history.json"))}
    return arrays


def check_excluded(names: Sequence[str], valid: Sequence[str], runs: Path) -> list[str]:
    """--infra_failed '<chain>/<run>': runs of the experiment whose rerun also failed
    (infrastructure); a finished run (selection.json) is never dropped after the fact."""
    for n in names:
        if n not in valid or (Path(runs) / n / bt.FILES["record"]).is_file():
            raise SystemExit(f"--infra_failed {n}: not an E023 run, or it finished")
    return sorted(set(names))


def chain_labels(arms: Mapping[str, gt.Arm]) -> list[tuple[str, gt.Arm]]:
    """Every chain the experiment may hold: the 10 seeds (multi-seed) or the attempts 42, 123,
    777 (single-seed, PLAN.md "단일 seed 실행과 판정")."""
    seeds = gt.SINGLE_SEEDS if st23.SEED_MODE == st23.SINGLE else gt.SEEDS
    return [(f"{gt.chain_name(d, p, s)}/{a.run}", a) for a in arms.values()
            for d in gt.DATASETS for p in gt.PAIRS for s in seeds]


def score_val(args: argparse.Namespace, resolver: Resolver = bc.get_method) -> None:
    """--split val (no test data): every missing validation file (an infrastructure rerun keeps
    finished files; --rescore moves val/ aside)."""
    _outside(args.runs_root, args.out)
    require_numerics_env(torch.cuda.is_available())
    paths = load_paths(args.paths)
    arms = family_arms(args.family, resolver)
    labels = chain_labels(arms)
    excluded = check_excluded(args.infra_failed, [lb for lb, _ in labels], args.runs_root)
    target = Path(args.out) / "val"
    meta = _json(target / "meta.json") if (target / "meta.json").is_file() else {"calls": []}
    call = {"time_utc": now_utc(), "code": code_version(), "environment": gpu_environment(),
            "family": list(args.family), "infra_failed": excluded, "rescore": args.rescore,
            "numerics": NUMERICS}
    if args.rescore:
        call["superseded"] = es.supersede(target, args.rescore)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = [(lb, a) for lb, a in labels if lb not in excluded
            and not gt.val_file(args.out, *lb.split("/")).is_file()
            and (st23.SEED_MODE != st23.SINGLE
                 or (Path(args.runs_root) / lb / bt.FILES["record"]).is_file())]
    written = []
    for d in gt.DATASETS:
        split = load_split(split_file(d))
        for p in gt.PAIRS:
            mine = [(lb, a) for lb, a in todo if gt.split_chain(lb.split("/")[0])[:2] == (d, p)]
            hb = [(lb, a) for lb, a in mine if a.role == "mechanism"]
            if hb:
                keys = [cl20_key(*gt.split_chain(lb.split("/")[0])) for lb, _ in hb]
                es.check_runs(args.runs_root, keys)
                manifest = es.load_manifest(es.manifest_file(d))
                val = np.asarray(split.val_indices, dtype=np.int64)
                totals = [manifest.total_frames[int(i)] for i in val]
                labs = [split.labels[int(i)] for i in val]
                units = es.unit_features(d, p, "d16m")
                dense = es.dense_val(d, p, val, totals, labs, "d16m", device)
                for k, (lb, _) in zip(keys, hb, strict=True):
                    arrays = score_cl20(args.runs_root, k, units, dense, device)
                    arrays |= {"val_indices": val, "total_frames": np.asarray(totals, np.int64)}
                    f = gt.val_file(args.out, *lb.split("/"))
                    es.save_npz(f, arrays)
                    written.append(f)
            variants: dict[str, dict] = {}
            for lb, a in mine:
                if a.role == "mechanism":
                    continue
                chain = lb.split("/")[0]
                _, cfg = chain_record(args.runs_root, chain, a.run)
                if cfg["variant"] not in variants:
                    variants.clear()
                    variants[cfg["variant"]] = val_inputs(paths, d, p, cfg["variant"], split)
                arrays = rescore_chain(args.runs_root, chain, a, variants[cfg["variant"]],
                                       device, resolver)
                f = gt.val_file(args.out, chain, a.run)
                es.save_npz(f, arrays)
                written.append(f)
            print(f"[e023_score] val {d}/{p}: {len(mine)} runs", flush=True)
    call |= {"time_utc_end": now_utc(), "device": str(device),
             "written": [f.relative_to(target).as_posix() for f in written]}
    write_json(target / "meta.json", {**meta, "calls": [*meta["calls"], call]})
    write_list(target)
    print(f"[e023_score] val: {len(written)} files -> {target}")


# --- selection ------------------------------------------------------------------------------------

def split_rows(flat: np.ndarray, counts: Sequence[int]) -> list[np.ndarray]:
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return [np.asarray(flat[a:a + n]) for a, n in zip(starts, counts, strict=True)]


def readout_selection(z_points: np.ndarray, counts: Sequence[int], totals: Sequence[int],
                      labels: Sequence[int], steps: Sequence[int]) -> dict:
    """Both readouts' full-validation V-AUC per evaluation point (common.video_scores, rank
    formula) and the first maximum of each (strict >, earliest on ties); None selection when a
    logit is not finite (numerical failure)."""
    z = np.asarray(z_points, dtype=np.float32)
    if not np.isfinite(z).all():
        return {"v_auc": None, "selection": None,
                "failed": "non-finite validation logits (numerical failure)"}
    aucs: dict[str, list[float]] = {"sigma": [], "logit": []}
    for e in range(len(z)):
        vs = bc.video_scores(split_rows(z[e], counts), totals)
        for r in aucs:
            aucs[r].append(float(bc.v_aucs(vs[r][None, :], labels)[0]))
    sel = {}
    for r, a in aucs.items():
        i, tied = bc.select_point(a)
        sel[r] = {"point": i + 1, "step": int(steps[i]), "v_auc": float(a[i]), "tied": tied}
    return {"v_auc": aucs, "selection": sel, "failed": None}


def baseline_entry(runs: Path, chain: str, arm: gt.Arm) -> tuple[dict, dict | None, list[str]]:
    """The selection entry of one comparison-method chain, its D14 bookkeeping and problems."""
    d = Path(runs) / chain / arm.run
    rec = _json(d / bt.FILES["record"])
    z = _npz(d / bt.FILES["val"]) if (d / bt.FILES["val"]).is_file() else None
    problems = []
    entry: dict = {"arm": arm.name, "role": arm.role, "method": arm.method, "unit": arm.unit,
                   "config_sha256": sha256_file(d / bt.FILES["config"]),
                   "record_sha256": sha256_file(d / bt.FILES["record"]),
                   "val_scores_sha256": sha256_file(d / bt.FILES["val"]) if z is not None
                   else None, "checkpoints": {}}
    if z is None:
        return entry | {"failed": "no validation scores", "selection": None,
                        "v_auc": None}, None, [f"{chain}/{arm.run}: no val_scores.npz"]
    sel = readout_selection(z["z"], z["counts"], z["total_frames"], z["labels"],
                            z["eval_steps"])
    entry |= {"eval_steps": [int(x) for x in z["eval_steps"]], "v_auc": sel["v_auc"],
              "selection": sel["selection"], "failed": sel["failed"] or rec.get("failed")}
    rec_sel = rec.get("selection") or {}
    if sel["selection"] is not None and {r: v["point"] for r, v in sel["selection"].items()} \
            != {r: (rec_sel.get(r) or {}).get("point") for r in sel["selection"]}:
        problems.append(f"{chain}/{arm.run}: recomputed selection differs from selection.json")
    for rule in gt.RULES:
        ck = (rec.get("checkpoints") or {}).get(rule)
        if ck and (rule == "last" or entry["failed"] is None):
            entry["checkpoints"][rule] = {"file": f"{arm.run}/{ck['file']}",
                                          "sha256": ck["sha256"], "point": ck["point"],
                                          "step": ck["step"]}
    d14 = None
    if arm.d14:
        eval_ck = dict(rec.get("eval_ckpts") or {})
        want = [f"{k:02d}" for k in range(1, gt.EVAL_POINTS + 1)]
        if sorted(eval_ck) != want:
            problems.append(f"{chain}/{arm.run}: D14 needs the 30 evaluation-point checkpoints")
        steps = entry["eval_steps"]
        d14 = {"arm": arm.name, "points": {
            k: {"file": f"{arm.run}/{bt.EVAL_CKPT.format(int(k))}", "sha256": eval_ck[k],
                "step": steps[int(k) - 1],
                "v_auc_sigma": None if sel["v_auc"] is None else sel["v_auc"]["sigma"][int(k) - 1],
                "v_auc_logit": None if sel["v_auc"] is None else sel["v_auc"]["logit"][int(k) - 1]}
            for k in sorted(eval_ck)}}
    return entry, d14, problems


def cl20_entry(scores: Path, chain: str) -> dict:
    """The selection entry of one C.L20 chain from its post-hoc validation file (C's rule: W
    readout, no offset)."""
    f = gt.val_file(scores, chain, gt.CL20_RUN)
    sf = _npz(f)
    epochs = [int(x) for x in sf["epochs"]]
    sel = readout_selection(sf["dense_z"], sf["dense_counts"], sf["total_frames"], sf["labels"],
                            epochs)
    shas = [str(x) for x in sf["ckpt_sha256"]]
    entry = {"arm": gt.CL20, "role": "mechanism", "method": gt.CL20_RUN, "unit": gt.X32,
             "val_file_sha256": sha256_file(f), "eval_steps": epochs, "v_auc": sel["v_auc"],
             "selection": sel["selection"], "failed": sel["failed"], "checkpoints": {}}
    if sel["selection"] is not None:
        for r in ("sigma", "logit"):
            e = sel["selection"][r]["point"]
            entry["checkpoints"][r] = {"file": f"{gt.CL20_RUN}/ckpt_epoch_{e:02d}.pt",
                                       "sha256": shas[e - 1], "point": e, "step": e}
    entry["checkpoints"]["last"] = {"file": f"{gt.CL20_RUN}/ckpt_last.pt",
                                    "sha256": str(sf["last_sha256"]),
                                    "point": int(sf["last_epoch"]), "step": int(sf["last_epoch"])}
    return entry


def c_collapse(e022_selection: Path | None = None) -> tuple[dict, list[str]]:
    """PLAN.md "실패 처리" "붕괴 제외" for E022 C (every arm; validation only): per
    '<chain>/hb_m10' of the experiment its selected W-readout (sigma) validation V-AUC in E022's
    committed selection (default ea.E022_SELECTION) and the flag (all >= 0.958 when the rule was
    written: none collapses), and the problems (a chain without a selected V-AUC)."""
    rec = _json(e022_selection or ea.E022_SELECTION)
    rows, problems = {}, []
    for d in gt.DATASETS:
        for p in gt.PAIRS:
            for s in gt.SEEDS:
                label = f"{gt.chain_name(d, p, s)}/hb_m10"
                v = ((((rec.get("chains") or {}).get(label) or {}).get("readouts") or {})
                     .get("W") or {}).get("selected_v_auc")
                if v is None:
                    problems.append(f"{label}: no selected W V-AUC in E022's selection")
                    continue
                rows[label] = {"selected_v_auc": float(v), "collapsed": collapsed(v)}
    return rows, problems


def history_nonfinite(runs: Path, label: str, entry: Mapping) -> bool:
    """A non-finite training loss in the chain's record (C.L20: either stage's history, E022's
    parser)."""
    d = Path(runs) / label
    if entry.get("role") == "mechanism":
        from wvad.diagnostics import e022 as e22

        return e22.history_failed(d) or e22.history_failed(d.parent / gt.CL20_STAGE1)
    return int(_json(d / bt.FILES["record"]).get("nonfinite_losses") or 0) > 0


def attempt_status(runs: Path, label: str, entry: Mapping) -> str:
    """The outcome of one attempt (PLAN.md "단일 seed 실행과 판정"; validation only, before access
    (A)) from its selection entry: "infrastructure" (--infra_failed), "gate" (--gate_failed, a
    gate failure before access (A)), "collapsed" (its "collapsed" flag), "numerical" (no
    selection: non-finite validation logits or a failed record; or a non-finite training
    loss), else "ok"."""
    if entry.get("failed") == gt.INFRA_ENTRY:
        return gt.ATTEMPT_INFRA
    if entry.get("failed") == gt.GATE_ENTRY:
        return gt.ATTEMPT_GATE
    if entry.get("collapsed"):
        return gt.ATTEMPT_COLLAPSED
    if entry.get("selection") is None or entry.get("failed") or \
            history_nonfinite(runs, label, entry):
        return gt.ATTEMPT_NUMERICAL
    return gt.ATTEMPT_OK


def c_single(c_rows: Mapping[str, Mapping]) -> dict:
    """E022 C's chains under the single-seed rule: per dataset and pair the first of 42, 123,
    777 that did not collapse (C is not retrained; ``c_collapse`` rows)."""
    out = {}
    for d in gt.DATASETS:
        for p in gt.PAIRS:
            status = {}
            for s in gt.SINGLE_SEEDS:
                r = c_rows.get(f"{gt.chain_name(d, p, s)}/hb_m10")
                if r is not None:
                    status[s] = gt.ATTEMPT_COLLAPSED if r["collapsed"] else gt.ATTEMPT_OK
            k = gt.chain_key("C", d, p)
            out[k] = gt.single_seed(status)
    return out


def single_record_check(selection: Mapping) -> dict:
    """The single-seed block of a selection record against its chains: every attempt's chain
    is in the record with its status consistent with the chain's "collapsed" flag and failure,
    each decided chain equals e023_gates.single_seed of its attempts (nothing pending), no chain
    outside the attempts, and the C block equals ``c_single`` of the collapse record. A
    multi-seed record (no block): ok."""
    block = selection.get("single_seed")
    if block is None:
        return {"mode": "multi", "problems": [], "ok": True}
    chains = selection.get("chains") or {}
    problems, used = [], set()
    for k, ch in (block.get("chains") or {}).items():
        arm, d, p = k.split("/")
        status = {}
        for att in ch.get("attempts") or []:
            label = att["label"]
            used.add(label)
            e = chains.get(label)
            status[att["seed"]] = att["status"]
            if e is None or e.get("arm") != arm:
                problems.append(f"{k}: attempt {label} not in the record")
            elif bool(e.get("collapsed")) != (att["status"] == gt.ATTEMPT_COLLAPSED) or \
                    (att["status"] == gt.ATTEMPT_OK and (e.get("failed") or not e.get(
                        "selection"))) or \
                    (e.get("failed") == gt.INFRA_ENTRY) != (att["status"] == gt.ATTEMPT_INFRA) \
                    or (e.get("failed") == gt.GATE_ENTRY) != (att["status"] == gt.ATTEMPT_GATE):
                problems.append(f"{k}: attempt {label} status {att['status']} differs")
        again = gt.single_seed(status)
        if {x: ch.get(x) for x in ("seed", "next", "no_verdict")} != \
                {x: again[x] for x in ("seed", "next", "no_verdict")} or again["next"] is not None:
            problems.append(f"{k}: not the single-seed rule's choice ({again})")
    extra = sorted(set(chains) - used)
    if extra:
        problems.append(f"chains outside the attempts: {extra[:5]}")
    c_rows = (selection.get("collapse") or {}).get("c_chains") or {}
    if dict(block.get("c_chains") or {}) != c_single(c_rows):
        problems.append("the C block differs from the collapse record")
    return {"mode": st23.SINGLE, "problems": problems, "ok": not problems}


def _entry(runs: Path, scores: Path, label: str, a: gt.Arm, excluded: Sequence[str],
           problems: list[str], d14: dict, gate_failed: Sequence[str] = ()) -> dict | None:
    """The selection entry of one chain (None with a problem recorded), its D14 bookkeeping;
    an infrastructure (``excluded``) or pre-access gate failure (``gate_failed``) has no
    checkpoint (never test-scored)."""
    chain = label.split("/")[0]
    if label in excluded or label in gate_failed:
        return {"arm": a.name, "role": a.role,
                "failed": gt.INFRA_ENTRY if label in excluded else gt.GATE_ENTRY,
                "selection": None, "v_auc": None, "checkpoints": {}}
    if a.role == "mechanism":
        if not gt.val_file(scores, chain, a.run).is_file():
            problems.append(f"{label}: no validation file")
            return None
        return cl20_entry(scores, chain)
    if not (Path(runs) / label / bt.FILES["record"]).is_file():
        problems.append(f"{label}: no selection.json")
        return None
    entry, book, probs = baseline_entry(runs, chain, a)
    problems += probs
    if book is not None:
        d14[label] = book
    return entry


def _single_chains(runs: Path, scores: Path, arms: Mapping[str, gt.Arm],
                   excluded: Sequence[str], problems: list[str], d14: dict,
                   gate_failed: Sequence[str] = ()) -> tuple[dict, dict]:
    """Single-seed mode: per chain (arm, dataset, pair) the attempts 42, 123, 777 in order until
    one is ok (``attempt_status``; infrastructure and pre-access gate failures pass to the next
    seed like collapses); a seed still to train (the first, or the fallback after a failed
    attempt) is a problem naming it (nothing written: train it, score it, select again), and so
    is an attempt that started without a finished record (running, or crashed: rerun once,
    then --infra_failed)."""
    chains, block = {}, {}
    for a in arms.values():
        for d in gt.DATASETS:
            for p in gt.PAIRS:
                status, labels = {}, {}
                for s in gt.SINGLE_SEEDS:
                    label = f"{gt.chain_name(d, p, s)}/{a.run}"
                    trained = label in excluded or label in gate_failed or \
                        (Path(runs) / label / bt.FILES["record"]).is_file()
                    if not trained:
                        if started(runs, label):
                            problems.append(f"{label}: started without a finished record "
                                            "(running: wait for its batch; crashed: rerun once, "
                                            "then name it with --infra_failed)")
                        break
                    entry = _entry(runs, scores, label, a, excluded, problems, d14,
                                   gate_failed)
                    if entry is None:
                        break
                    entry["collapsed"] = collapsed(((entry.get("selection") or {}).get(
                        "sigma") or {}).get("v_auc"))
                    chains[label] = entry
                    labels[s] = label
                    status[s] = attempt_status(runs, label, entry)
                    if status[s] == gt.ATTEMPT_OK:
                        break
                ch = gt.single_seed(status)
                for att in ch["attempts"]:
                    att["label"] = labels[att["seed"]]
                k = gt.chain_key(a.name, d, p)
                block[k] = ch
                if ch["next"] is not None and not any(x.startswith(
                        f"{gt.chain_name(d, p, ch['next'])}/{a.run}") for x in problems):
                    problems.append(f"{gt.chain_key(a.name, d, p)}: seed {ch['next']} to train "
                                    f"(single-seed rule; {gt.chain_name(d, p, ch['next'])}/"
                                    f"{a.run})")
    return chains, block


def selection_record(runs: Path, scores: Path, family: Sequence[str],
                     excluded: Sequence[str] = (), resolver: Resolver = bc.get_method,
                     e022_selection: Path | None = None, gate_failed: Sequence[str] = ()
                     ) -> tuple[dict, dict]:
    """(e023_selection.json body, e023_all_ckpts.json body); SystemExit when a chain that is not
    an infrastructure failure has no record or the D14 bookkeeping is incomplete. Each chain
    records "collapsed" (PLAN.md "실패 처리" "붕괴 제외": its selected sigma V-AUC < 0.60; the
    chain then leaves every rule of the analysis, e023.failures), "collapse" the rule, the
    collapsed chains and E022 C's (``c_collapse``). Single-seed mode (PLAN.md "단일 seed 실행과
    판정"): only the attempts of each chain (``_single_chains``) and the block "single_seed"
    (per chain its attempts and chosen seed; C's from E022's chains, ``c_single``); SystemExit
    while a fallback seed is still to train. ``gate_failed``: chains dropped by a gate failure
    before access (A) (recorded without checkpoints; single-seed: the next seed replaces them)."""
    arms = family_arms(family, resolver)
    chains, d14, problems = {}, {}, []
    block = None
    if st23.SEED_MODE == st23.SINGLE:
        chains, block = _single_chains(runs, scores, arms, excluded, problems, d14,
                                       gate_failed)
    else:
        for label, a in chain_labels(arms):
            entry = _entry(runs, scores, label, a, excluded, problems, d14, gate_failed)
            if entry is not None:
                chains[label] = entry
    c_rows, c_problems = c_collapse(e022_selection)
    problems += c_problems
    if problems:
        raise SystemExit(f"{len(problems)} selection problem(s), nothing written: {problems[:5]}")
    for e in chains.values():
        e["collapsed"] = collapsed(((e.get("selection") or {}).get("sigma") or {}).get("v_auc"))
    rule = ("full-validation V-AUC of the W readout on the E021 32-window grid (sigma: max fp32 "
            "sigma(u_k); logit: max u_k; no offset), first maximum (strict >, earliest point on "
            "ties); last = the last evaluation point")
    after = ("the next seed of 42, 123, 777 replaces it (PLAN.md 단일 seed 실행과 판정)"
             if st23.SEED_MODE == st23.SINGLE else "no rerun, no replacement seed")
    collapse = {"rule": f"PLAN.md 실패 처리 '붕괴 제외': selected sigma validation V-AUC < "
                        f"{COLLAPSE_V_AUC} (validation only, fixed here before access (A)); the "
                        f"chain leaves every rule, {after}",
                "threshold": COLLAPSE_V_AUC,
                "collapsed": sorted(k for k, e in chains.items() if e["collapsed"]),
                "c_chains": c_rows,
                "c_collapsed": sorted(k for k, r in c_rows.items() if r["collapsed"])}
    common = {"code": code_version(), "time_utc": now_utc(), "family": list(family),
              "arms": {k: v.to_json() for k, v in arms.items()}}
    sel = {"check": "E023 checkpoint selection (PLAN.md 체크포인트 선택; validation only)",
           **common, "rule": rule, "infra_failed": list(excluded),
           "gate_failed_before_access": sorted(gate_failed), "collapse": collapse,
           "chains": chains}
    if block is not None:
        sel["single_seed"] = {
            "rule": "PLAN.md 단일 seed 실행과 판정: per chain (arm, dataset, pair) the first seed "
                    "of 42, 123, 777 that neither collapsed nor failed (numerical, "
                    "infrastructure or a gate before access (A)); every attempt failed: 판정 "
                    "없음 for that arm and dataset",
            "seeds": list(gt.SINGLE_SEEDS), "chains": block, "c_chains": c_single(c_rows),
            "no_verdict": sorted(k for k, v in block.items() if v["no_verdict"])}
    book = {"check": "E023 D14 evaluation-point checkpoints (temporary until access (A))",
            **common, "chains": d14}
    return sel, book


def select_command(args: argparse.Namespace, resolver: Resolver = bc.get_method) -> None:
    labels = [lb for lb, _ in chain_labels(family_arms(args.family, resolver))]
    excluded = sorted(set(args.infra_failed))
    gate_failed = sorted(set(args.gate_failed))
    unknown = [x for x in excluded + gate_failed if x not in labels]
    if unknown:
        raise SystemExit(f"--infra_failed / --gate_failed {unknown}: not E023 runs")
    sel, book = selection_record(args.runs_root, args.scores, args.family, excluded, resolver,
                                 gate_failed=gate_failed)
    vl = Path(args.scores) / "val" / "scores.sha256"
    sel["val_list_sha256"] = sha256_file(vl) if vl.is_file() else None
    write_json(Path(args.out), sel)
    write_json(Path(args.all_ckpts), book)
    failed = sorted(k for k, v in sel["chains"].items() if v.get("failed"))
    gone = sel["collapse"]["collapsed"] + sel["collapse"]["c_collapsed"]
    print(f"[e023_score] selection -> {args.out} ({len(sel['chains'])} chains, failed {failed}, "
          f"collapsed {gone}); D14 -> {args.all_ckpts} ({len(book['chains'])} chains)")
    if "single_seed" in sel:
        moved = sorted(k for k, v in sel["single_seed"]["chains"].items()
                       if v["seed"] not in (None, gt.SINGLE_SEEDS[0]))
        print(f"[e023_score] single seed: fallback seeds {moved}, 판정 없음 "
              f"{sel['single_seed']['no_verdict']}")


# --- test scoring: jobs, binding, order -----------------------------------------------------------

class Job(NamedTuple):
    chain: str
    run: str
    arm: str
    rule: str                                   # sigma | logit | last | eval
    files: tuple                                # ((relative file, sha256, point, step), ...)

    @property
    def dataset(self) -> str:
        return gt.split_chain(self.chain)[0]

    @property
    def pair(self) -> str:
        return gt.split_chain(self.chain)[1]

    @property
    def label(self) -> str:
        return f"{self.chain}/{self.run}__{self.rule}"


def jobs_of(selection: Mapping, all_ckpts: Mapping | None) -> tuple[list[Job], list[str]]:
    """Every test job of the committed selection (and the D14 evaluation points) and the rules
    without a checkpoint (numerical or infrastructure failures)."""
    jobs, absent = [], []
    for label, entry in (selection.get("chains") or {}).items():
        chain, run = label.split("/")
        for rule in gt.RULES:
            ck = (entry.get("checkpoints") or {}).get(rule)
            if not ck or not ck.get("sha256"):
                absent.append(f"{label}__{rule}")
                continue
            jobs.append(Job(chain, run, entry["arm"], rule,
                            ((ck["file"], ck["sha256"], ck.get("point"), ck.get("step")),)))
        book = ((all_ckpts or {}).get("chains") or {}).get(label)
        if book is not None:
            pts = book["points"]
            jobs.append(Job(chain, run, entry["arm"], gt.EVAL_RULE, tuple(
                (pts[k]["file"], pts[k]["sha256"], int(k), pts[k]["step"]) for k in sorted(pts))))
    return jobs, absent


def job_problem(job: Job, runs: Path, blocked: frozenset, selection: Mapping,
                n_chunk: int) -> str | None:
    """Every checkpoint of the job exists with its selected sha256 and is not blocked; a
    comparison-method chain's config.json is the selected one and has the call's N_chunk."""
    for rel, sha, _, _ in job.files:
        f = Path(runs) / job.chain / rel
        if not f.is_file():
            return f"{job.label}: {f} missing"
        got = sha256_file(f)
        if got != sha:
            return f"{job.label}: {rel} has another sha256 than the selection's"
        if got in blocked:
            return f"{job.label}: {rel} is on the blocked list (PLAN.md item 7)"
    entry = selection["chains"][f"{job.chain}/{job.run}"]
    if entry.get("role") != "mechanism":
        cfg_file = Path(runs) / job.chain / job.run / bt.FILES["config"]
        if not cfg_file.is_file() or sha256_file(cfg_file) != entry.get("config_sha256"):
            return f"{job.label}: config.json differs from the selected chain's"
        if int(_json(cfg_file)["n_chunk"]) != int(n_chunk):
            return f"{job.label}: the chain's N_chunk differs from --n_chunk {n_chunk}"
    return None


def scoring_calls(log: Sequence[Mapping]) -> list[dict]:
    return [e for e in log if e.get("command") == COMMAND and e.get("event") == "start"]


def check_order(log: Sequence[Mapping], rescore: str | None, missing_only: bool,
                runs_subset: Sequence[str]) -> dict:
    """PLAN.md: access (A) scores in full once; afterwards only --missing_only (infrastructure:
    absent, unreadable or unrecorded files) or --runs (the runs a writing / indexing fix of a
    failed (v3) affects), each with --rescore."""
    calls = scoring_calls(log)
    full = [c for c in calls if c.get("mode") == "full"]
    if missing_only and runs_subset:
        raise SystemExit("--missing_only and --runs are separate re-reads")
    mode = "missing_only" if missing_only else "runs" if runs_subset else "full"
    if mode != "full" and not rescore:
        raise SystemExit(f"{mode}: needs --rescore '<reason>' (PLAN.md re-reads)")
    if mode == "full" and full:
        raise SystemExit("access (A) was read in full: only --missing_only or --runs with "
                         "--rescore (PLAN.md)")
    if mode != "full" and not full:
        raise SystemExit(f"{mode} follows the full read")
    return {"mode": mode, "prior_calls": len(calls), "prior_full": len(full)}


def selection_files(selection_sha256: str | None, all_ckpts_sha256: str | None,
                    root: Path = REPO_ROOT, selection: Path = gt.FILES["selection"],
                    all_ckpts: Path = gt.FILES["all_ckpts"]) -> dict:
    """Access device 6: the committed selection (and D14 bookkeeping) with their sha256."""
    out = {"selection": dt.selection_binding(selection_sha256, selection, root)}
    if all_ckpts_sha256 is not None:
        out["all_ckpts"] = dt.selection_binding(all_ckpts_sha256, all_ckpts, root)
    return out


def access_device(paths, *, freeze: str | None, lineage, selection_sha256: str | None,
                  all_ckpts_sha256: str | None, train_inputs: Path,
                  plan_dir: Path = gt.E023_DIR, root: Path = REPO_ROOT,
                  selection: Path = gt.FILES["selection"],
                  all_ckpts: Path = gt.FILES["all_ckpts"]) -> dict:
    """Access device items 1-6 and 8 (item 7 is the guard at every load); SystemExit at the
    first failure. Gate (t) (items 4, 5) hashes test files, bytes only."""
    out = {"descent": dt.deployed_descent(freeze, lineage, root, plan_dir),
           "plan": dt.plan_seal(plan_dir), "approval": ea.d1_approval(plan_dir)}
    t = gt.gate_t(paths)
    if not t["ok"]:
        raise SystemExit(f"gate (t): test manifests, annotations or dense test stores differ "
                         f"from the E020 / E022 records: {t}")
    out["t"] = t
    out |= selection_files(selection_sha256, all_ckpts_sha256, root, selection, all_ckpts)
    out["train_inputs"] = require_inputs(train_inputs, paths)
    return out


def guard_summary(guard: Mapping) -> dict:
    return {"descent": guard["descent"], "plan": guard["plan"], "approval": guard["approval"],
            "t_ok": guard["t"]["ok"], "selection": guard["selection"],
            "all_ckpts": guard.get("all_ckpts"),
            "train_inputs": {k: guard["train_inputs"].get(k)
                             for k in ("record", "record_sha256", "ok")}}


def memory_gate(need_gb: float | None, device: torch.device) -> dict:
    """Gate (m) on this machine's device (free memory by torch.cuda.mem_get_info)."""
    if device.type != "cuda":
        return gt.gate_m(None, None, device.type)
    free, _total = torch.cuda.mem_get_info(device)
    need = None if need_gb is None else int(float(need_gb) * 2 ** 30)
    return gt.gate_m(int(free), need, "cuda")


# --- test scoring: inputs -------------------------------------------------------------------------

class TestInputs:
    """The dense test snippets of one (dataset, pair): raw (the comparison methods' input
    processing on top; one variant held at a time) and scale-matched by the E021 d16m map
    (C.L20)."""

    __test__ = False                             # not a pytest class

    def __init__(self, paths, manifest, pair: str):
        self.paths, self.manifest, self.pair = paths, manifest, pair
        self._raw: DenseSet | None = None
        self._hb: DenseSet | None = None
        self._variant: tuple[str, list] | None = None

    @property
    def n(self) -> int:
        return len(self.manifest)

    def raw(self) -> DenseSet:
        if self._raw is None:
            dense = DenseSet(dt.store_folders(self.paths, self.manifest.dataset, self.pair))
            dt.check_dense(dense, self.manifest)
            self._raw = dense
        return self._raw

    def processed(self, variant: str) -> list[torch.Tensor]:
        """Every test video through the variant's InputPipeline (wvad.baselines.train)."""
        if self._variant is None or self._variant[0] != variant:
            self._variant = None
            raw = self.raw()
            pipe = bt.input_pipeline(self.paths, self.manifest.dataset, self.pair, variant,
                                     raw.dims)
            self._variant = (variant, [pipe.apply(raw.video(v)) for v in range(self.n)])
        return self._variant[1]

    def hb(self) -> DenseSet:
        if self._hb is None:
            dense = dt.dense_snippets(self.paths, self.manifest.dataset, self.pair, scaled=True)
            dt.check_dense(dense, self.manifest)
            self._hb = dense
        return self._hb


# --- test scoring: writing and indexing ------------------------------------------------------

def manifest_arrays(manifest) -> dict[str, np.ndarray]:
    return {"names": np.asarray(manifest.names),
            "total_frames": np.asarray(manifest.total_frames, dtype=np.int64),
            "labels": np.asarray(manifest.labels, dtype=np.int64)}


def _counts(z_list: Sequence[np.ndarray], manifest) -> np.ndarray:
    counts = np.array([len(z) for z in z_list], dtype=np.int64)
    want = np.array([grid.n_snippets(int(t)) for t in manifest.total_frames], dtype=np.int64)
    if len(z_list) != len(manifest) or not np.array_equal(counts, want):
        raise ValueError("snippet counts differ from ceil(T / 16) of the manifest")
    return counts


def snippet_arrays(z_list: Sequence[np.ndarray], manifest) -> dict[str, np.ndarray]:
    """Snippet logits in manifest order, their counts, the logit readout zc (= z: c(v) = 0, no
    offset) and the fp32 sigma scores."""
    counts = _counts(z_list, manifest)
    z = np.concatenate([np.asarray(x, dtype=np.float32) for x in z_list]).astype(np.float32)
    return {"z": z, "counts": counts, "zc": z.copy(), "sigma": sigmoid32(z)}


def eval_arrays(z_lists: Sequence[Sequence[np.ndarray]], manifest, files: Sequence[tuple]
                ) -> dict[str, np.ndarray]:
    """D14: the snippet logits of the 30 evaluation-point checkpoints (z (points, rows)) with
    their sha256, points and steps."""
    counts = _counts(z_lists[0], manifest)
    z = np.stack([np.concatenate([np.asarray(x, dtype=np.float32) for x in zl])
                  for zl in z_lists]).astype(np.float32)
    for zl in z_lists:
        _counts(zl, manifest)
    return {"z": z, "counts": counts, "points": np.asarray([f[2] for f in files], np.int64),
            "steps": np.asarray([f[3] for f in files], np.int64),
            "ckpt_sha256": np.asarray([f[1] for f in files])}


def job_arrays(job: Job, n_chunk: int, device: torch.device) -> dict[str, np.ndarray]:
    rel, sha, point, step = job.files[0]
    return {"chain": np.str_(job.chain), "run": np.str_(job.run), "arm": np.str_(job.arm),
            "rule": np.str_(job.rule), "ckpt": np.str_(rel),
            "ckpt_sha256": np.str_(sha) if job.rule != gt.EVAL_RULE else np.str_(""),
            "point": np.int64(-1 if point is None else point),
            "step": np.int64(-1 if step is None else step), "n_chunk": np.int64(n_chunk),
            "device": np.str_(str(device))}


def nonfinite(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Non-finite values per stored score array (gate (v3) counts them; not a stop)."""
    return {k: int(np.size(arrays[k]) - np.isfinite(np.asarray(arrays[k], dtype=np.float64))
                   .sum()) for k in SCORE_KEYS if k in arrays}


def write_scores(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    es.save_npz(Path(path), arrays)


def sha256_list(files: Sequence[Path], base: Path) -> str:
    rows = sorted((Path(f).relative_to(base).as_posix(), sha256_file(f)) for f in files)
    return "".join(f"{sha}  {rel}\n" for rel, sha in rows)


def current_files(target: Path) -> list[Path]:
    return sorted(f for f in Path(target).rglob("*.npz")
                  if "superseded" not in f.relative_to(target).parts)


def write_list(target: Path) -> Path:
    path = Path(target) / "scores.sha256"
    path.write_text(sha256_list(current_files(target), target), encoding="utf-8", newline="\n")
    return path


def chunk_record(manifest, n_chunk: int) -> list[str]:
    """Names of the test videos cut into N_chunk pieces (n_v > N_chunk; frame counts only)."""
    return [manifest.names[i] for i in bc.chunk_targets(list(manifest.total_frames), n_chunk)]


def recorded_files(target: Path) -> dict[str, set[str]]:
    """'<chain>/<file>' -> the sha256 the scoring calls recorded right after writing it."""
    out: dict[str, set[str]] = {}
    for f in Path(target).glob("meta_*.json"):
        for rel, sha in (_json(f).get("written_sha256") or {}).items():
            out.setdefault(rel, set()).add(sha)
    return out


def rescore_files(jobs: Sequence[Job], out: Path, mode: str, rescore: str | None,
                  recorded: Mapping[str, set[str]] | None = None
                  ) -> tuple[list[Job], list[Path]]:
    """The jobs a call scores and the existing files it moves aside (full: every file, existing
    ones refused; --missing_only: absent, unreadable or unrecorded files; --runs: the named)."""
    target = Path(out) / "test"
    paths_ = {j.label: gt.score_file(out, j.chain, j.run, j.rule) for j in jobs}
    if mode == "full":
        existing = [p for p in paths_.values() if p.exists()]
        if existing:
            raise SystemExit(f"{len(existing)} score files exist (e.g. {existing[0]})")
        return list(jobs), []
    if mode == "missing_only":
        def missing(p: Path) -> bool:
            if not (p.exists() and zipfile.is_zipfile(p)):
                return True
            rel = p.relative_to(target).as_posix()
            return recorded is not None and sha256_file(p) not in recorded.get(rel, ())
        todo = [j for j in jobs if missing(paths_[j.label])]
        return todo, [paths_[j.label] for j in todo if paths_[j.label].exists()]
    return list(jobs), [paths_[j.label] for j in jobs if paths_[j.label].exists()]


def supersede(files: Sequence[Path], out: Path) -> str | None:
    if not files:
        return None
    target = Path(out) / "test"
    dest = target / "superseded" / dt_time.datetime.now(dt_time.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    for f in files:
        new = dest / Path(f).relative_to(target)
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(new))
    return str(dest)


def parse_runs(items: Sequence[str], jobs: Sequence[Job]) -> list[str]:
    """--runs '<chain>/<run>' (every rule) or '<chain>/<run>__<rule>'."""
    labels = {j.label for j in jobs}
    out = []
    for it in items:
        hits = [lb for lb in labels if lb == it or lb.startswith(f"{it}__")]
        if not hits:
            raise SystemExit(f"--runs {it}: no selected test job")
        out += hits
    return sorted(set(out))


# --- test scoring: one call -------------------------------------------------------------------

def score_job(job: Job, inputs: TestInputs, runs: Path, guard: ea.CheckpointGuard,
              selection: Mapping, n_chunk: int, device: torch.device,
              resolver: Resolver) -> dict[str, np.ndarray]:
    """One job's arrays: every checkpoint through the guard, its forward path, the arrays."""
    entry = selection["chains"][f"{job.chain}/{job.run}"]
    m = inputs.manifest
    z_lists = []
    if entry.get("role") == "mechanism":
        dense = inputs.hb()
        for rel, _, _, _ in job.files:
            obj = guard.load(Path(runs) / job.chain / rel)
            z_lists.append(forward_hb(hb_model(obj, sum(dense.dims), device), dense,
                                      inputs.n, device))
    else:
        cfg = _json(Path(runs) / job.chain / job.run / bt.FILES["config"])
        method = chain_method(resolver, entry["method"], cfg)
        xs = inputs.processed(cfg["variant"])
        dim = int(xs[0].shape[1])
        for rel, _, _, _ in job.files:
            obj = guard.load(Path(runs) / job.chain / rel)
            model = build_model(method, obj["model_state_dict"], cfg, dim, device)
            with chain_numerics(cfg):
                z_lists.append(forward_baseline(method, model, xs, unit=int(cfg["unit"]),
                                                n_chunk=n_chunk, device=device))
            del model
    arrays = manifest_arrays(m) | job_arrays(job, n_chunk, device)
    if job.rule == gt.EVAL_RULE:
        return arrays | eval_arrays(z_lists, m, job.files)
    return arrays | snippet_arrays(z_lists[0], m)


def _job_variant(job: Job, runs: Path, selection: Mapping) -> str:
    entry = selection["chains"][f"{job.chain}/{job.run}"]
    if entry.get("role") == "mechanism":
        return "~hb"
    return str(_json(Path(runs) / job.chain / job.run / bt.FILES["config"])["variant"])


def score_jobs(paths, jobs: Sequence[Job], runs: Path, out: Path, device: torch.device,
               guard: ea.CheckpointGuard, selection: Mapping, n_chunk: int,
               resolver: Resolver = bc.get_method,
               on_written: Callable[[Path], None] | None = None) -> dict:
    """Every job by (dataset, pair), one input variant held at a time; each file is reported to
    ``on_written`` right after it is renamed into place."""
    written, counts, targets = [], {}, {}
    for d in gt.DATASETS:
        m = dt.read_manifest(d)
        targets[d] = chunk_record(m, n_chunk)
        for p in gt.PAIRS:
            mine = sorted((j for j in jobs if (j.dataset, j.pair) == (d, p)),
                          key=lambda j: (_job_variant(j, runs, selection), j.label))
            if not mine:
                continue
            inputs = TestInputs(paths, m, p)
            for j in mine:
                arrays = score_job(j, inputs, runs, guard, selection, n_chunk, device, resolver)
                f = gt.score_file(out, j.chain, j.run, j.rule)
                write_scores(f, arrays)
                if on_written is not None:
                    on_written(f)
                written.append(f)
                counts[j.label] = nonfinite(arrays)
            print(f"[e023_score] test {d}/{p}: {len(mine)} files", flush=True)
    return {"written": written, "nonfinite": counts, "chunk_targets": targets}


def score_test(args: argparse.Namespace, resolver: Resolver = bc.get_method) -> None:
    """Access (A) (module docstring): refusals logged before anything is read, the "start" line
    before any test feature, the meta binding every written file, the "end" (or "error")
    line."""
    paths = load_paths(args.paths)
    _outside(args.runs_root, args.out, args.test_log)
    target = Path(args.out) / "test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    line = {"access": ACCESS, "command": COMMAND}

    def refuse(reason: str):
        dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "refused",
                                                  "reason": reason, "code": code_version()})
        raise SystemExit(reason)

    try:
        memory = memory_gate(args.memory_need_gb, device)
        if not memory["ok"]:
            raise SystemExit(f"gate (m): {memory} (wait and call again; PLAN.md)")
        order = check_order(dt.read_test_log(args.test_log), args.rescore, args.missing_only,
                            args.runs or ())
        guard_rec = access_device(paths, freeze=args.freeze_commit, lineage=args.lineage,
                                  selection_sha256=args.selection_sha256,
                                  all_ckpts_sha256=args.all_ckpts_sha256,
                                  train_inputs=Path(args.train_inputs))
        blocked = ea.load_blocked(Path(args.blocked))
        selection = _json(gt.FILES["selection"])
        book = _json(gt.FILES["all_ckpts"]) if args.all_ckpts_sha256 is not None else None
        bc.check_n_chunk(args.n_chunk)
        require_numerics_env(device.type == "cuda")
        jobs, absent = jobs_of(selection, book)
        subset = parse_runs(args.runs, jobs) if args.runs else []
        if subset:
            jobs = [j for j in jobs if j.label in subset]
        problems = [x for j in jobs if (x := job_problem(j, args.runs_root, blocked, selection,
                                                         args.n_chunk))]
        if problems:
            raise SystemExit(f"{len(problems)} checkpoint problem(s): {problems[:5]}")
        todo, aside = rescore_files(jobs, args.out, order["mode"], args.rescore,
                                    recorded_files(target))
        k = len(dt.read_test_log(args.test_log)) + 1
        if any(int(f.stem.split("_", 1)[1]) >= k for f in target.glob("meta_*.json")):
            raise SystemExit(f"{target} has a meta of access-log line {k} or later: another "
                             "access log was used")
    except SystemExit as exc:
        refuse(str(exc))
    guard = ea.CheckpointGuard(blocked)
    meta = {"split": "test", "mode": order["mode"], "order": order, "time_utc": now_utc(),
            "code": code_version(), "environment": gpu_environment(), "device": str(device),
            "memory": memory, "rescore": args.rescore, "runs_subset": subset,
            "guard": guard_summary(guard_rec), "absent": absent, "n_chunk": int(args.n_chunk),
            "jobs": [j.label for j in todo], "access_log": os.path.realpath(args.test_log),
            "out": str(target), "numerics": NUMERICS}
    entry = line | {"time_utc": now_utc(), "event": "start", "mode": order["mode"],
                    "rescore": args.rescore, "runs": [j.label for j in todo], "absent": absent,
                    "inputs": sorted({str(f) for d in gt.DATASETS for p in gt.PAIRS
                                      for f in dt.store_folders(paths, d, p)}),
                    "out_dir": str(target), "n_chunk": int(args.n_chunk),
                    "deployed_commit": guard_rec["descent"]["deployed_commit"],
                    "plan_sha256": guard_rec["plan"]["sha256"],
                    "guard": guard_summary(guard_rec)}
    line_no = dt.append_test_log(args.test_log, entry)
    meta |= {"access_log_line": line_no,
             "access_log_entry": json.loads(json.dumps(entry, sort_keys=True, default=str)),
             "written_sha256": {}}
    meta_file = target / f"meta_{line_no}.json"
    write_json(meta_file, meta | {"complete": False})

    def recorded(f: Path) -> None:
        meta["written_sha256"][f.relative_to(target).as_posix()] = sha256_file(f)
        meta["loaded"] = list(guard.loaded)
        write_json(meta_file, meta | {"complete": False})

    try:
        meta["superseded"] = supersede(aside, args.out)
        res = score_jobs(paths, todo, args.runs_root, args.out, device, guard, selection,
                         args.n_chunk, resolver, on_written=recorded)
        after = inputs_check(Path(args.train_inputs), paths)
    except BaseException as exc:
        dt.append_test_log(args.test_log, line | {"time_utc": now_utc(), "event": "error",
                                                  "error": repr(exc), "line": line_no})
        raise
    write_list(target)
    meta |= {"written": [f.relative_to(target).as_posix() for f in res["written"]],
             "nonfinite": res["nonfinite"], "chunk_targets": res["chunk_targets"],
             "loaded": list(guard.loaded), "gate_k": gt.gate_k(guard.loaded, blocked),
             "train_inputs_after": after, "time_utc_end": now_utc(), "complete": True}
    write_json(meta_file, meta)
    dt.append_test_log(args.test_log, line | {
        "time_utc": now_utc(), "event": "end", "line": line_no, "written": len(res["written"]),
        "nonfinite_runs": sorted(k for k, v in res["nonfinite"].items() if any(v.values())),
        "loaded": len(guard.loaded), "gate_k_ok": meta["gate_k"]["ok"],
        "train_inputs_after": after["ok"]})
    print(f"[e023_score] test: {len(res['written'])} files -> {target}")
    if not after["ok"]:
        raise SystemExit(f"training inputs changed during access (A): {after}")


# --- command line -------------------------------------------------------------------------------

def _family_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--family", nargs="+", default=list(gt.MAIN),
                   help="the final main family (PLAN.md D3; replacements mgfn / clip_tsa)")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e023_score")
    ap.add_argument("--paths", help="paths file (default: $WVAD_PATHS or configs/paths.yaml)")
    sub = ap.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inputs", help="(3c) the training-input record (server, no test data)")
    i.add_argument("--out", required=True, type=Path)
    i.add_argument("--e021_runs", required=True, type=Path)
    e = sub.add_parser("env", help="gate (f): a training batch's environment record")
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--train_inputs", default=str(gt.FILES["train_inputs"]))
    v = sub.add_parser("devlr", help="D4 choices from the development chains (validation only)")
    v.add_argument("--runs_root", required=True, type=Path)
    v.add_argument("--out", type=Path, default=gt.FILES["devlr"])
    _family_arg(v)
    q = sub.add_parser("q1d", help="(q1d) sanity floor from the development chains (validation)")
    q.add_argument("--runs_root", required=True, type=Path)
    q.add_argument("--record", type=Path, default=gt.FILES["q1b"])
    _family_arg(q)
    j = sub.add_parser("jobs", help="the chains still to train, one line each")
    j.add_argument("--stage", required=True, choices=("d4", "confirm"))
    j.add_argument("--runs_root", required=True, type=Path)
    j.add_argument("--devlr", type=Path, default=gt.FILES["devlr"])
    j.add_argument("--infra_failed", nargs="*", default=[],
                   help="confirm, single-seed: '<chain>/<run>' whose rerun also failed")
    j.add_argument("--gate_failed", nargs="*", default=[],
                   help="confirm, single-seed: '<chain>/<run>' dropped by a gate before (A)")
    _family_arg(j)
    s = sub.add_parser("score", help="--split val (validation) or test (access (A))")
    s.add_argument("--split", required=True, choices=("val", "test"))
    s.add_argument("--runs_root", required=True, type=Path, help="E023 runs root")
    s.add_argument("--out", required=True, type=Path, help="score root outside the repository")
    s.add_argument("--infra_failed", nargs="*", default=[], help="val: '<chain>/<run>'")
    s.add_argument("--rescore", help="reason: val, move val/ aside; test, a permitted re-read")
    s.add_argument("--missing_only", action="store_true")
    s.add_argument("--runs", nargs="+", metavar="CHAIN/RUN[__RULE]")
    s.add_argument("--test_log")
    s.add_argument("--selection_sha256")
    s.add_argument("--all_ckpts_sha256", help="sha256 of the committed results/e023_all_ckpts.json "
                                              "(D14); omit only if D14 became 'not done'")
    s.add_argument("--memory_need_gb", type=float, help="gate (m): the (3j) maximum (GB)")
    s.add_argument("--n_chunk", type=int, default=bc.N_CHUNK_DEFAULT)
    s.add_argument("--freeze_commit")
    s.add_argument("--lineage", default=str(dt.LINEAGE_FILE))
    s.add_argument("--train_inputs", default=str(gt.FILES["train_inputs"]))
    s.add_argument("--blocked", default=str(gt.FILES["blocked"]))
    _family_arg(s)
    c = sub.add_parser("select", help="checkpoint selection (validation only)")
    c.add_argument("--runs_root", required=True, type=Path)
    c.add_argument("--scores", required=True, type=Path, help="score root (its val/)")
    c.add_argument("--out", default=str(gt.FILES["selection"]))
    c.add_argument("--all_ckpts", default=str(gt.FILES["all_ckpts"]))
    c.add_argument("--infra_failed", nargs="*", default=[])
    c.add_argument("--gate_failed", nargs="*", default=[],
                   help="'<chain>/<run>' dropped by a gate failure before access (A)")
    _family_arg(c)
    args = ap.parse_args(argv)
    if args.command == "score" and args.split == "test":
        missing = [k for k in ("test_log", "selection_sha256", "memory_need_gb")
                   if getattr(args, k) is None]
        if missing:
            ap.error(f"--split test needs {', '.join('--' + m for m in missing)}")
    if args.command == "score" and args.split == "val" and (args.missing_only or args.runs):
        ap.error("--split val scores every run (no --missing_only, --runs)")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    if args.command == "inputs":
        rec = inputs_record(load_paths(args.paths), args.e021_runs)
        _merge(args.out, "train_inputs", rec)
        print(f"[e023_score] (3c) training inputs ok={rec['ok']} -> {args.out}")
    elif args.command == "env":
        rec = batch_env(load_paths(args.paths), args.out, Path(args.train_inputs))
        print(f"{args.out}: environment {rec['env_differences'] or 'as E022'}, inputs ok")
    elif args.command == "devlr":
        rec = devlr_record(args.runs_root, args.family)
        write_json(Path(args.out), rec)
        print(f"[e023_score] D4: {len(rec['choices'])}/{len(rec['settings'])} settings complete "
              f"(no eligible value: {rec['no_eligible']}) -> {args.out}")
    elif args.command == "q1d":
        q1d_command(args)
    elif args.command == "jobs":
        incomplete: list[str] = []
        lines = (d4_jobs(args.runs_root, args.family) if args.stage == "d4" else
                 confirm_jobs(args.runs_root, _json(args.devlr), args.family,
                              infra_failed=args.infra_failed, gate_failed=args.gate_failed,
                              incomplete=incomplete))
        print("\n".join(lines))
        if incomplete:
            print(f"[e023_score] started without a finished record (running or crashed; not "
                  f"listed): {incomplete}", file=sys.stderr)
    elif args.command == "select":
        select_command(args)
    elif args.split == "val":
        score_val(args)
    else:
        score_test(args)


if __name__ == "__main__":
    main()
