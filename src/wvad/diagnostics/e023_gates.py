"""E023 gates (PLAN.md "관문", "코드 규칙" 0, "구현"): the experiment layout shared by the E023
modules, the add-only rule against the E023 base commit, the pre-freeze bookkeeping gates (q2),
(q3), (q5), (q6), (q7) and the (q1d) sanity floor record, the training / selection gates (a),
(b), (d), (f), (g), (p), (s), (w), (w2), (n) and the test gates (t), (m), (k), (v3). Recipes are
compared per dataset (results/e023_recipes.json "datasets": VCV has one official configuration
per dataset). Fixed numeric code from the freeze on (PLAN.md
"코드 규칙" 1). New code (add-only rule); the E022 modules are used through their pure functions
only, and no method module is imported here (the registry, ``wvad.baselines.common.get_method``,
is called by the callers).

Layout (README "Stage F — E023")
    chains          <runs>/<dataset>_<pair>_s<seed>/<run>: a comparison-method chain
                    ``<method>_u<unit>`` (wvad.baselines.train), C.L20 ``hb_l20`` (stage 2 of the
                    frozen wvad.train, with ``hb_l20_stage1``)
    arms            main family: the method's plan name at its official unit (MIL, RTFM, URDMU,
                    VCV; replacements MGFN, CLIP-TSA); X.32 = "<name>.32" for every main-family
                    method whose official unit is not 32 (D11); C.L20 (D9). Per arm three
                    checkpoint rules: sigma (the selection, the arm's own name), logit
                    ("<arm>.logit") and last ("<arm>.last"). E022's reused C, C.last.
    scores          <S>/val/<chain>/<run>.npz (validation re-scoring, C.L20 post-hoc scoring),
                    <S>/test/<chain>/<run>__<rule>.npz (rule sigma, logit, last; D14 "eval")

Code rule 0 (``code_rule``): every line of ``git diff --no-renames --name-status efa10fdc
<commit> -- src/`` is "A"; E022's identity sets (S_train, S_score, S_extract, S_analysis) and its
8 new modules are byte-identical to efa10fdc.

Independent recomputation (w2): window assignment (``e022_gates.window_sets``, written from the
E021 plan text without ``wvad.extract.grid``), W readout and full-validation V-AUC by integer pair
counts (``e022_gates.exact_auc``); this module never uses ``wvad.eval.dense_readout``, the
readouts of ``wvad.baselines.common`` or ``wvad.diagnostics.e023_score`` for it (unit test).
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wvad.baselines.recipes import dataset_recipe
from wvad.data import dense_test as dt
from wvad.diagnostics import e022_gates as g22
from wvad.diagnostics import e023_access as ea
from wvad.diagnostics.e023_stats import SINGLE_SEEDS, collapsed
from wvad.paths import REPO_ROOT

E023_DIR = REPO_ROOT / "experiments" / "E023_localisation_baselines"
RESULTS = E023_DIR / "results"
BASE_COMMIT = "efa10fdcd9511a61aafc4e7917cd1c89044a1b04"       # E022 report commit (PLAN.md 0)
E022_NEW = ("src/wvad/data/dense_test.py", "src/wvad/diagnostics/e022.py",
            "src/wvad/diagnostics/e022_desc.py", "src/wvad/diagnostics/e022_gates.py",
            "src/wvad/diagnostics/e022_score.py", "src/wvad/diagnostics/e022_stats.py",
            "src/wvad/eval/frames.py", "src/wvad/extract/testsplit.py")
DATASETS = ("ucf", "xdv")
PAIRS = ("xcb_xcl", "xcb_siglip")
SEEDS = g22.SEEDS                                   # the 10 confirmation seeds
DEV_SEEDS = (0, 1, 2)
# PLAN.md "단일 seed 실행과 판정" (2026-10-21): per chain the first of SINGLE_SEEDS that did not
# collapse or fail ("multi": the 10 seeds of every chain, restored at the revision)
ATTEMPT_OK, ATTEMPT_COLLAPSED, ATTEMPT_NUMERICAL, ATTEMPT_INFRA, ATTEMPT_GATE = (
    "ok", "collapsed", "numerical", "infrastructure", "gate")
ATTEMPT_STATUSES = (ATTEMPT_OK, ATTEMPT_COLLAPSED, ATTEMPT_NUMERICAL, ATTEMPT_INFRA,
                    ATTEMPT_GATE)
# the "failed" text of a selection entry without checkpoints: an infrastructure failure
# (--infra_failed) or a gate failure before access (A) (select --gate_failed; single-seed: the
# next seed replaces the attempt, PLAN.md "세부 규칙")
INFRA_ENTRY, GATE_ENTRY = "infrastructure (PLAN.md)", "gate failure before access (A) (PLAN.md)"
NO_CHECKPOINT_ENTRIES = (INFRA_ENTRY, GATE_ENTRY)
PLAN_NAMES = {"mil": "MIL", "rtfm": "RTFM", "urdmu": "URDMU", "vadclip_v": "VCV",
              "mgfn": "MGFN", "clip_tsa": "CLIP-TSA"}
MAIN = ("mil", "rtfm", "urdmu", "vadclip_v")
REPLACES = {"clip_tsa": "vadclip_v", "mgfn": "rtfm"}             # D3 (a)
NO_REPLACEMENT = ("mil", "urdmu")
CL20, CL20_RUN, CL20_STAGE1 = "C.L20", "hb_l20", "hb_l20_stage1"
C_ARMS = ("C", "C.last")                            # E022's reused arms
RULES = ("sigma", "logit", "last")
RULE_SUFFIX = {"sigma": "", "logit": ".logit", "last": ".last"}
EVAL_RULE = "eval"                                  # D14 evaluation-point file
EVAL_POINTS = 30
X32 = 32
FILES = {"selection": RESULTS / "e023_selection.json",
         "all_ckpts": RESULTS / "e023_all_ckpts.json",
         "gate": RESULTS / "e023_gate.json", "devlr": RESULTS / "e023_devlr.json",
         "recipes": RESULTS / "e023_recipes.json",
         "train_inputs": RESULTS / "e023_train_inputs.json",
         "blocked": RESULTS / "e023_blocked_ckpts.json",
         "prefreeze": RESULTS / "e023_prefreeze.json", "q1b": RESULTS / "e023_q1b.json",
         "licences": RESULTS / "e023_licences.json",
         "summary": RESULTS / "e023_summary.json", "runs": RESULTS / "e023_runs.csv",
         "access": RESULTS / "e023_test_access.log",
         "scores": RESULTS / "e023_test_scores.sha256"}
TRAIN_ENV_KEYS = ("torch", "cuda", "gpu", "driver")
S_TOL = 1e-5                                        # gate (s): |dz| <= 1e-5 x max(1, |z|)
W2_TOL = 1e-9
MEMORY_FACTOR = 1.2                                 # gate (m)
PERMISSIVE = ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause")
CONFIG_IGNORED = ("seed", "out_dir")                # gate (a), new chains
REPEAT_IGNORED = ("out_dir", "save_eval_ckpts")     # gate (b): the repeat is the same command
Q1D_FLOOR_POINTS = 2.0                              # (q1d): MIL port mean - 2.0 points
Q1D_REFERENCE = "mil"
Q1D_PAIR = "xcb_xcl"
CL20_IGNORED = ("seed", "out_dir", "init", "save_epoch_ckpts")    # gate (a), C.L20


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- layout -----------------------------------------------------------------------------------

def chain_name(dataset: str, pair: str, seed: int) -> str:
    return f"{dataset}_{pair}_s{seed}"


def split_chain(chain: str) -> tuple[str, str, int]:
    """'<dataset>_<pair>_s<seed>' -> (dataset, pair, seed)."""
    d, rest = chain.split("_", 1)
    pair, seed = rest.rsplit("_s", 1)
    return d, pair, int(seed)


def run_name(method: str, unit: int) -> str:
    """The folder of a comparison-method chain inside its chain folder."""
    return f"{method}_u{int(unit)}"


def arm_name(method: str, unit: int, official_unit: int) -> str:
    """MIL, URDMU, ... at the official unit; '<name>.32' for an X.32 arm."""
    if method not in PLAN_NAMES:
        raise KeyError(f"unknown method {method!r}")
    return PLAN_NAMES[method] if int(unit) == int(official_unit) else \
        f"{PLAN_NAMES[method]}.{int(unit)}"


def rule_arm(arm: str, rule: str) -> str:
    """The analysis arm of a checkpoint rule: 'URDMU', 'URDMU.logit', 'URDMU.last'."""
    return arm + RULE_SUFFIX[rule]


def single_seed(status: Mapping[int, str | None], seeds: Sequence[int] | None = None) -> dict:
    """PLAN.md "단일 seed 실행과 판정" for one chain (dataset x pair x arm; validation only, before
    access (A)): ``status`` seed -> the attempt's outcome (ATTEMPT_STATUSES; None or absent: not
    trained / not finished). The seeds are tried in order; the first "ok" one is the chain's seed;
    a collapsed, numerically failed, infrastructure- or gate-failed attempt passes to the next
    seed; at most len(seeds) attempts (default SINGLE_SEEDS). Returns the attempts made, "seed"
    (None until one is ok), "next" (the seed to train next, None when decided) and
    "no_verdict" (every attempt failed: that arm and dataset is "판정 없음")."""
    attempts, chosen, nxt = [], None, None
    for s in SINGLE_SEEDS if seeds is None else seeds:
        st = status.get(s)
        if st is None:
            nxt = s
            break
        if st not in ATTEMPT_STATUSES:
            raise ValueError(f"seed {s}: unknown attempt status {st!r}")
        attempts.append({"seed": int(s), "status": st})
        if st == ATTEMPT_OK:
            chosen = int(s)
            break
    return {"attempts": attempts, "seed": chosen, "next": nxt,
            "no_verdict": chosen is None and nxt is None}


def chain_key(arm: str, dataset: str, pair: str) -> str:
    """The key of a chain in the single-seed record: '<arm>/<dataset>/<pair>'."""
    return f"{arm}/{dataset}/{pair}"


@dataclass(frozen=True)
class Arm:
    name: str
    method: str          # registry name; "hb_l20" for C.L20
    unit: int
    official_unit: int
    role: str            # "main" | "x32" | "mechanism"

    @property
    def run(self) -> str:
        return CL20_RUN if self.role == "mechanism" else run_name(self.method, self.unit)

    @property
    def d14(self) -> bool:
        """The 30 evaluation-point checkpoints are kept and test-scored (D14: main family)."""
        return self.role == "main"

    def to_json(self) -> dict:
        return {"name": self.name, "method": self.method, "unit": self.unit,
                "official_unit": self.official_unit, "role": self.role, "run": self.run,
                "d14": self.d14}


def check_family(methods: Sequence[str]) -> tuple[str, ...]:
    """The final main family (PLAN.md D3): 3 or 4 distinct known methods, MIL and URDMU never
    replaced, a replacement only in place of its own method (VCV -> CLIP-TSA, RTFM -> MGFN)."""
    fam = tuple(methods)
    if len(set(fam)) != len(fam) or not 3 <= len(fam) <= 4:
        raise ValueError(f"family {fam}: 3 or 4 distinct methods (PLAN.md D3, K = 3 or 4)")
    for m in fam:
        if m not in PLAN_NAMES:
            raise ValueError(f"family {fam}: unknown method {m!r}")
        if m in REPLACES and REPLACES[m] in fam:
            raise ValueError(f"family {fam}: {m} replaces {REPLACES[m]}, not both")
    return fam


def arm_table(family: Sequence[str], official_units: Mapping[str, int]) -> dict[str, Arm]:
    """Every new arm of the family: the main family at the official unit, X.32 for every method
    whose official unit is not 32 (D11), and C.L20 (D9)."""
    out = {}
    for m in check_family(family):
        u = int(official_units[m])
        out[arm_name(m, u, u)] = Arm(arm_name(m, u, u), m, u, u, "main")
    for m in family:
        u = int(official_units[m])
        if u != X32:
            out[arm_name(m, X32, u)] = Arm(arm_name(m, X32, u), m, X32, u, "x32")
    out[CL20] = Arm(CL20, CL20_RUN, X32, X32, "mechanism")
    return out


def arms_from_json(rec: Mapping) -> dict[str, Arm]:
    return {k: Arm(v["name"], v["method"], int(v["unit"]), int(v["official_unit"]), v["role"])
            for k, v in rec.items()}


def main_arms(arms: Mapping[str, Arm]) -> list[str]:
    return [k for k, a in arms.items() if a.role == "main"]


def score_file(root: Path, chain: str, run: str, rule: str) -> Path:
    """<root>/test/<chain>/<run>__<rule>.npz (rule sigma, logit, last; D14 eval)."""
    if rule not in (*RULES, EVAL_RULE):
        raise ValueError(f"unknown rule {rule!r}")
    return Path(root) / "test" / chain / f"{run}__{rule}.npz"


def val_file(root: Path, chain: str, run: str) -> Path:
    return Path(root) / "val" / chain / f"{run}.npz"


# --- code rule 0 --------------------------------------------------------------------------------

def code_rule(commit: str | None, *, base: str = BASE_COMMIT, repo: Path = REPO_ROOT) -> dict:
    """PLAN.md "코드 규칙" 0 at ``commit`` (None: the working tree): the add-only rule against
    the E023 base commit, and E022's identity sets and 8 new modules byte-identical to it."""
    rule = g22.add_only(commit, base=base, repo=repo)
    sets = {name: g22.identical(paths, commit, base=base, repo=repo)
            for name, paths in g22.IDENTITY_SETS.items()}
    sets["E022_new"] = g22.identical(E022_NEW, commit, base=base, repo=repo)
    return {"commit": commit or "working tree", "base": base, "add_only": rule, "sets": sets,
            "ok": bool(rule["ok"] and all(v is True for v in sets.values()))}


# --- pre-freeze gates ---------------------------------------------------------------------------

LICENCE_KEY = {"mil": "mil", "rtfm": "rtfm", "urdmu": "urdmu", "vadclip_v": "vcv",
               "mgfn": "mgfn", "clip_tsa": "clip_tsa"}


def gate_q2(licences: Mapping, methods: Sequence[str] = tuple(PLAN_NAMES)) -> dict:
    """(q2): every method and replacement has a licence record (url, commit, type, port); a
    copied port only under a permissive licence."""
    repos = dict(licences.get("repos") or {})
    rows = {}
    for m in methods:
        e = repos.get(LICENCE_KEY[m]) or {}
        complete = all(e.get(k) for k in ("url", "commit", "licence_type", "port"))
        copied = str(e.get("port", "")).startswith("copy")
        permissive = str(e.get("licence_type", "")) in PERMISSIVE
        rows[m] = {"present": bool(e), "complete": complete, "copied": copied,
                   "permissive": permissive, "ok": complete and (permissive or not copied)}
    return {"methods": rows, "ok": bool(rows) and all(r["ok"] for r in rows.values())}


def gate_q3(record: Mapping, recipes: Mapping[str, Mapping[str, Mapping]],
            defaults: Mapping[str, Mapping] | None = None) -> dict:
    """(q3): each port's recipe on each dataset (``recipes``: method -> dataset ->
    Recipe.to_json() of ``common.dataset_method``, obtained through the registry by the caller)
    equals its (3b) record (results/e023_recipes.json "datasets"), and the training command's
    defaults (``defaults``: method -> parsed lr_mult / scale_mult) are the official values
    ("1")."""
    rows = {}
    for m, by_ds in recipes.items():
        entry = (record.get("methods") or {}).get(m) or {}
        equal = {d: (want := dataset_recipe(entry, d)) is not None and dict(rec) == dict(want)
                 for d, rec in by_ds.items()}
        dflt = dict((defaults or {}).get(m) or {"lr_mult": "1", "scale_mult": "1"})
        rows[m] = {"recipe_equal": bool(equal) and all(equal.values()), "datasets": equal,
                   "defaults": dflt,
                   "defaults_ok": dflt.get("lr_mult") == "1" and dflt.get("scale_mult") == "1"}
        rows[m]["ok"] = rows[m]["recipe_equal"] and rows[m]["defaults_ok"]
    return {"methods": rows, "ok": bool(rows) and all(r["ok"] for r in rows.values())}


def _state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]


def same_state(a: Path, b: Path) -> bool:
    sa, sb = _state(a), _state(b)
    return set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def _losses(history: Path) -> list:
    return [h.get("loss") for h in _json(history)["points"]]


def gate_q5(with_eval: Path, without_eval: Path) -> dict:
    """(q5): a chain and its --no_eval twin have the same per-interval losses and bit-equal last
    weights (evaluation draws nothing from the training generators)."""
    a, b = Path(with_eval), Path(without_eval)
    losses = _losses(a / "history.json") == _losses(b / "history.json")
    weights = same_state(a / "ckpt_last.pt", b / "ckpt_last.pt")
    return {"losses_equal": losses, "last_weights_equal": weights, "ok": losses and weights}


def _calls(source: str, name: str) -> int:
    tree = ast.parse(source)
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Name) and n.func.id == name)
        or (isinstance(n.func, ast.Attribute) and n.func.attr == name)))


# short-piece thresholds (q7 RTFM-4, URDMU-2): RTFM's top-k (k = 3) and UR-DMU's context (16)
SHORT_PIECES = {"rtfm_lt3": 3, "urdmu_lt16": 16}


def piece_counts(total_frames: Sequence[int], n_chunk: int,
                 short: Mapping[str, int] = SHORT_PIECES) -> dict:
    """From the frame counts only (no feature, no score): the videos cut by N_chunk and, per
    threshold, the last pieces of the cut videos and the whole (uncut) videos shorter than it,
    in snippets (common.chunk_bounds of n_v = ceil(T / 16))."""
    from wvad.baselines.common import chunk_bounds
    from wvad.extract import grid

    n_v = [grid.n_snippets(int(t)) for t in total_frames]
    cut = [n for n in n_v if n > int(n_chunk)]
    last = [chunk_bounds(n, int(n_chunk))[-1] for n in cut]
    return {"videos": len(n_v), "cut_videos": len(cut),
            "max_snippets": max(n_v) if n_v else None,
            "short": {k: {"snippets_below": int(m),
                          "cut_last_pieces": sum(1 for a, b in last if b - a < m),
                          "whole_videos": sum(1 for n in n_v if n <= int(n_chunk) and n < m)}
                      for k, m in short.items()}}


def gate_q6(n_chunk: int, totals: Mapping[str, Sequence[int]],
            names: Mapping[str, Sequence[str]] | None, recorded: Mapping[str, Sequence[str]],
            sources: Mapping[str, str]) -> dict:
    """(q6): the pieces depend on n_v and N_chunk only (``common.chunk_targets`` of the access
    (0) frame counts equals the recorded target list per dataset: by name with ``names``, else
    by count; access (A)'s own list is compared by name in (v3)) and validation and test call
    the same function (``score_video`` is called in each source of ``sources``: the training
    loop and the test scoring). Each dataset row also records ``pieces`` (``piece_counts``: cut
    videos, short last pieces and short whole videos; information, not a pass condition)."""
    from wvad.baselines.common import chunk_targets

    rows = {}
    for d, t in totals.items():
        idx = chunk_targets(list(t), int(n_chunk))
        rec = list(recorded.get(d) or [])
        if names is None:
            rows[d] = {"targets": len(idx), "recorded": len(rec), "ok": len(idx) == len(rec)}
        else:
            want = [names[d][i] for i in idx]
            rows[d] = {"targets": want, "recorded": rec, "ok": want == rec}
        rows[d]["pieces"] = piece_counts(list(t), int(n_chunk))
    shared = {k: _calls(src, "score_video") > 0 for k, src in sources.items()}
    return {"n_chunk": int(n_chunk), "datasets": rows, "same_function": shared,
            "ok": bool(rows) and all(r["ok"] for r in rows.values()) and bool(shared)
            and all(shared.values())}


def gate_q7(review: Mapping, methods: Sequence[str]) -> dict:
    """(q7): every port has its own code-review record with a Jev judgement (written by
    ``python -m wvad.diagnostics.e023 q7``: results/e023_q1b.json "q7" -> method -> {"jev",
    "review", "review_sha256", "verdict", "findings", "utc"})."""
    rows = {m: {"present": m in review, "jev": (review.get(m) or {}).get("jev")} for m in methods}
    for r in rows.values():
        r["ok"] = r["present"] and r["jev"] is not None
    return {"methods": rows, "ok": bool(rows) and all(r["ok"] for r in rows.values())}


def q1d_record(v_aucs: Mapping[str, Mapping[str, Mapping[int, float | None]]],
               methods: Sequence[str], *, seeds: Sequence[int] = DEV_SEEDS,
               datasets: Sequence[str] = DATASETS, floor_points: float = Q1D_FLOOR_POINTS,
               reference: str = Q1D_REFERENCE) -> dict:
    """(q1d) sanity floor (PLAN.md "관문", "실패 처리"): per dataset (XCB+XCL, development seeds
    0-2, the official recipe at lr x1) each method's mean selected sigma V-AUC (points) against
    the MIL port's mean - ``floor_points``; below the floor the method is kept and flagged
    "적응 하의 약함" (no porting gate). ``v_aucs``: method -> dataset -> seed -> the chain's
    selected sigma V-AUC (0-1; None or a non-finite value: missing). A collapsed chain (PLAN.md
    "붕괴 제외": e023_stats.collapsed, V-AUC < 0.60) is left out of the means and named under
    "collapsed_seeds" (no seed left: mean None, no flag). ``complete`` is False while a value is
    missing (the record is then refused by the callers)."""
    if reference not in methods:
        raise ValueError(f"(q1d) needs the reference port {reference!r} in {list(methods)}")
    def present(v: float | None) -> bool:
        return v is not None and math.isfinite(float(v))
    missing = sorted(f"{m}/{d}/s{s}" for m in methods for d in datasets for s in seeds
                     if not present((v_aucs.get(m, {}).get(d, {}) or {}).get(s)))

    def mean_points(vals: Sequence[float | None]) -> float | None:
        kept = [v for v in vals if not collapsed(v)]
        return 100 * float(np.mean(kept)) if kept and all(present(v) for v in vals) else None
    rows: dict = {m: {"datasets": {}, "flag": False} for m in methods}
    refs = {}
    for d in datasets:
        ref = [(v_aucs.get(reference, {}).get(d, {}) or {}).get(s) for s in seeds]
        ref_mean = mean_points(ref)
        refs[d] = {"reference_mean_points": ref_mean,
                   "floor_points": None if ref_mean is None else ref_mean - floor_points}
        for m in methods:
            vals = [(v_aucs.get(m, {}).get(d, {}) or {}).get(s) for s in seeds]
            mean = mean_points(vals)
            below = None if mean is None or ref_mean is None else bool(
                mean < ref_mean - floor_points)
            rows[m]["datasets"][d] = {"seeds": {str(s): v for s, v in zip(seeds, vals,
                                                                         strict=True)},
                                      "collapsed_seeds": [s for s, v in zip(seeds, vals,
                                                                            strict=True)
                                                          if collapsed(v)],
                                      "mean_points": mean, "below_floor": below}
            rows[m]["flag"] = bool(rows[m]["flag"] or below)
    return {"check": "E023 (q1d) sanity floor (PLAN.md 관문; not a porting gate)",
            "pair": Q1D_PAIR, "seeds": list(seeds), "reference": reference,
            "floor_points": floor_points, "rule": "flag '적응 하의 약함' when the mean selected "
            "sigma V-AUC (points, development seeds) < the MIL port's mean - 2.0 points on a "
            "dataset; the method is kept; collapsed chains (selected sigma V-AUC < 0.60, PLAN.md "
            "붕괴 제외) are left out of the means", "datasets": refs, "methods": rows,
            "missing": missing, "complete": not missing}


# --- training and selection gates ---------------------------------------------------------------

def _without(cfg: Mapping, keys: Sequence[str]) -> dict:
    return {k: v for k, v in cfg.items() if k not in keys}


def gate_a(configs: Mapping[str, Mapping], arms: Mapping[str, Arm], devlr: Mapping,
           recipes: Mapping, n_chunk: int) -> dict:
    """(a) for the comparison-method chains (``configs``: '<chain>/<run>' -> config.json): the
    chains of one arm and setting differ only in {seed, out_dir}; the arm's method and unit, the
    D4 choice of the same method and setting (results/e023_devlr.json "choices"), the (3b)
    recipe of the chain's dataset, the D14 flag, N_chunk, fp32, deterministic mode (README:
    --deterministic with CUBLAS_WORKSPACE_CONFIG=:4096:8, so gate (b) compares two equal runs),
    no float64 / --no_eval / iterations."""
    by_run = {a.run: a for a in arms.values() if a.role != "mechanism"}
    groups: dict[tuple, list] = {}
    rows = {}
    for label, cfg in configs.items():
        chain, run = label.split("/")
        d, p, _ = split_chain(chain)
        arm = by_run.get(run)
        if arm is None:
            rows[label] = {"ok": False, "reason": f"run {run} is no arm of the family"}
            continue
        groups.setdefault((arm.name, d, p), []).append((label, cfg))
        choice = ((devlr.get("choices") or {}).get(f"{arm.method}/{d}/{p}")) or {}
        want = {"method": arm.method, "unit": arm.unit, "dataset": d, "pair": p,
                "lr_mult": choice.get("lr_mult"), "scale_mult": choice.get("scale_mult"),
                "save_eval_ckpts": arm.d14, "n_chunk": int(n_chunk), "deterministic": True,
                "float64": False, "evaluate": True, "eval_points": EVAL_POINTS,
                "iterations": None, "dtype": "torch.float32"}
        bad = sorted(k for k, v in want.items() if cfg.get(k) != v)
        if (cfg.get("deterministic_mode") or {}).get("deterministic") is not True:
            bad.append("deterministic_mode")
        rec = dataset_recipe((recipes.get("methods") or {}).get(arm.method) or {}, d)
        if rec is None or cfg.get("recipe") != rec:
            bad.append("recipe")
        rows[label] = {"different": bad, "ok": not bad}
    for members in groups.values():
        ref = _without(members[0][1], CONFIG_IGNORED)
        for label, cfg in members[1:]:
            if _without(cfg, CONFIG_IGNORED) != ref:
                rows[label]["ok"] = False
                rows[label]["different"] = rows[label]["different"] + ["config_vs_group"]
    return {"chains": rows, "ok": bool(rows) and all(r["ok"] for r in rows.values())}


def gate_a_cl20(configs: Mapping[str, Mapping], references: Mapping[str, Mapping]) -> dict:
    """(a) for C.L20: each stage-2 config equals the E021 HB L20 reference config of its dataset
    and pair (``references``: '<dataset>/<pair>' -> config) except {seed, out_dir, init,
    save_epoch_ckpts}."""
    rows = {}
    for label, cfg in configs.items():
        d, p, _ = split_chain(label.split("/")[0])
        ref = references.get(f"{d}/{p}")
        diff = None if ref is None else sorted(
            k for k in set(_without(cfg, CL20_IGNORED)) | set(_without(ref, CL20_IGNORED))
            if cfg.get(k) != ref.get(k))
        rows[label] = {"different": diff, "ok": diff == []}
    return {"chains": rows, "ok": bool(rows) and all(r["ok"] for r in rows.values())}


def _npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def gate_b(first: Path, second: Path) -> dict:
    """(b): the same chain run twice (the same configuration except {out_dir, save_eval_ckpts};
    both deterministic runs, gate (a)); the record (losses, V-AUCs, validation logits)
    bit-equal, else recorded with its first difference, and the selected points equal in any
    case."""
    a, b = Path(first), Path(second)
    ca, cb = (_without(_json(x / "config.json"), REPEAT_IGNORED) for x in (a, b))
    same_cfg = ca == cb
    det = [bool(c.get("deterministic")) for c in (ca, cb)]
    ha, hb = ([{k: v for k, v in h.items() if k != "seconds"}           # wall time aside
               for h in _json(x / "history.json")["points"]] for x in (a, b))
    za, zb = _npz(a / "val_scores.npz"), _npz(b / "val_scores.npz")
    same_hist = ha == hb if len(ha) == len(hb) else False
    same_z = za["z"].shape == zb["z"].shape and bool(np.array_equal(za["z"], zb["z"]))
    first_diff = next((i + 1 for i, (x, y) in enumerate(zip(ha, hb, strict=False)) if x != y),
                      None)
    sa, sb = _json(a / "selection.json")["selection"], _json(b / "selection.json")["selection"]
    same_sel = {r: (sa.get(r) or {}).get("point") == (sb.get(r) or {}).get("point")
                for r in ("sigma", "logit")}
    return {"record_bit_equal": bool(same_hist and same_z), "first_different_point": first_diff,
            "selection_equal": same_sel, "same_command": same_cfg, "deterministic": det,
            "ok": all(same_sel.values()) and same_cfg}


def gate_d(val_indices: Sequence[int], split_val: Sequence[int]) -> dict:
    """(d): the validation videos recorded by a chain are the split file's, in its order."""
    got, want = [int(i) for i in val_indices], [int(i) for i in split_val]
    return {"n": len(got), "ok": got == want}


def gate_f(commit: str | None, *, inputs: Mapping, env: Mapping[str, Mapping],
           want_env: Mapping, dtypes: Mapping[str, str], tests_now: Mapping,
           tests_before: Mapping, reason: str | None = None, base: str = BASE_COMMIT,
           repo: Path = REPO_ROOT) -> dict:
    """(f): code rule 0 at the training commit, the training inputs (``inputs``: a check with
    "ok"), every batch environment (``env``: name -> torch / cuda / gpu / driver) equal to E022's
    training environment (``want_env``; a different driver alone is recorded), the comparison
    methods' precision fp32 (``dtypes``: chain -> config dtype) and the tests/ record."""
    code = code_rule(commit, base=base, repo=repo)
    envs = {}
    for name, e in env.items():
        diff = [k for k in TRAIN_ENV_KEYS if e.get(k) != want_env.get(k)]
        envs[name] = {"differences": diff, "ok": diff in ([], ["driver"])}
    precision = sorted(k for k, v in dtypes.items() if v != "torch.float32")
    tests = g22.compare_tests(tests_now, tests_before, reason)
    changed = any(r["differences"] == ["driver"] for r in envs.values())
    return {"code": code, "inputs": dict(inputs), "environments": envs,
            "precision_not_fp32": precision, "tests": tests,
            "tag": "(드라이버 변경)" if changed else None,
            "ok": bool(code["ok"] and inputs.get("ok") and envs
                       and all(r["ok"] for r in envs.values()) and not precision
                       and tests["ok"])}


def gate_g(chain_dir: Path, reselect: Callable[[Path], Mapping]) -> dict:
    """(g): the selection recomputed from the chain's val_scores.npz (``reselect``:
    wvad.baselines.train.reselect) equals its selection.json."""
    rec = _json(Path(chain_dir) / "selection.json")
    again = reselect(Path(chain_dir) / "val_scores.npz")
    want = rec.get("selection") or {}
    return {"recomputed": again.get("selection"), "recorded": want,
            "ok": dict(again.get("selection") or {}) == dict(want)}


def gate_p(config: Mapping, arm: Arm, recipe: Mapping) -> dict:
    """(p): the chain's unit and training input are the arm's (main family: the official unit
    and its recipe variant; X.32: 32 on the E021 d16m cache) and its sampler is the recipe's."""
    unit_rec = (recipe.get("units") or {}).get(str(arm.unit)) or {}
    want_variant = "d16m" if arm.role == "x32" else unit_rec.get("variant")
    ok_unit = int(config.get("unit", -1)) == arm.unit and (
        arm.role != "main" or arm.unit == int(recipe.get("official_unit", -1)))
    ok_variant = config.get("variant") == want_variant
    ok_sampler = (config.get("recipe") or {}).get("sampler") == recipe.get("sampler")
    return {"unit": ok_unit, "variant": ok_variant, "sampler": ok_sampler,
            "ok": ok_unit and ok_variant and ok_sampler}


def close(got: np.ndarray, stored: np.ndarray, tol: float = S_TOL) -> tuple[bool, float]:
    """|got - stored| <= tol x max(1, |stored|) elementwise, and the largest relative error."""
    g, s = np.asarray(got, dtype=np.float64), np.asarray(stored, dtype=np.float64)
    if g.shape != s.shape or not (np.isfinite(g).all() and np.isfinite(s).all()):
        return False, math.inf
    rel = np.abs(g - s) / np.maximum(1.0, np.abs(s))
    worst = float(rel.max()) if rel.size else 0.0
    return worst <= tol, worst


def gate_s(rescored: Mapping[str, np.ndarray], stored_z: np.ndarray,
           points: Mapping[str, int], shas: Mapping[str, str],
           eval_shas: Mapping[str, str] | None = None,
           eval_recorded: Mapping[str, str] | None = None) -> dict:
    """(s) of one chain: each kept checkpoint (``rescored``: rule -> its validation logits,
    concatenated; ``points``: rule -> evaluation point; ``shas``: rule -> (recorded, current)
    sha256 pair joined as 'a|b') re-scored equals the stored logits of its point
    (``stored_z`` (30, rows)); the D14 checkpoints by sha256 only."""
    rows = {}
    for rule, z in rescored.items():
        ok, worst = close(z, stored_z[int(points[rule]) - 1])
        rec, cur = str(shas.get(rule, "|")).split("|")
        rows[rule] = {"max_rel_error": worst, "sha256_ok": bool(rec) and rec == cur,
                      "ok": ok and bool(rec) and rec == cur}
    d14 = None
    if eval_recorded is not None:
        bad = sorted(k for k, v in eval_recorded.items() if (eval_shas or {}).get(k) != v)
        d14 = {"n": len(eval_recorded), "different": bad, "ok": not bad}
    return {"rules": rows, "d14": d14,
            "ok": bool(rows) and all(r["ok"] for r in rows.values())
            and (d14 is None or d14["ok"])}


def gate_w(recomputed: Mapping, entry: Mapping) -> dict:
    """(w): the selection recomputed from the stored validation logits equals the committed
    e023_selection.json entry (points of both readouts and every V-AUC)."""
    sel = recomputed.get("selection") or {}
    want = entry.get("selection") or {}
    pts = {r: (sel.get(r) or {}).get("point") == (want.get(r) or {}).get("point")
           for r in ("sigma", "logit")}
    aucs = {r: list(map(float, (recomputed.get("v_auc") or {}).get(r) or []))
            == list(map(float, (entry.get("v_auc") or {}).get(r) or []))
            for r in ("sigma", "logit")}
    return {"points": pts, "v_auc": aucs, "ok": all(pts.values()) and all(aucs.values())}


# --- (w2): independent recomputation ------------------------------------------------------------

def _sigmoid32(x: np.ndarray) -> np.ndarray:
    return torch.sigmoid(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()


def w2_video(z: np.ndarray, sets: Sequence[np.ndarray]) -> tuple[float, float]:
    """(sigma, logit) W-readout video score of one video (no offset): u_k = float32(float64 mean
    of z over J_k); sigma = max fp32 sigma(u_k), logit = max u_k."""
    z = np.asarray(z, dtype=np.float32)
    u = np.array([np.float32(z[j].mean(dtype=np.float64)) for j in sets], dtype=np.float32)
    return float(_sigmoid32(u).max()), float(u.max())


def w2_chain(z_points: np.ndarray, counts: Sequence[int], totals: Sequence[int],
             labels: Sequence[int], entry: Mapping, tol: float = W2_TOL) -> dict:
    """(w2) of one chain from its stored validation logits (``z_points`` (points, rows)) against
    its e023_selection.json ``entry`` ("v_auc" per readout, "selection" per readout)."""
    z = np.asarray(z_points, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    if z.ndim != 2 or z.shape[1] != int(counts.sum()) or len(totals) != len(counts):
        return {"ok": False, "reason": f"shape {z.shape} for {int(counts.sum())} rows"}
    if not np.isfinite(z).all():
        return {"ok": True, "skipped": "non-finite logits (numerical failure)"}
    sets = [g22.window_sets(int(t)) for t in totals]
    scores = {r: np.empty((len(z), len(counts))) for r in ("sigma", "logit")}
    for e in range(len(z)):
        for i, (a, n) in enumerate(zip(first, counts, strict=True)):
            scores["sigma"][e, i], scores["logit"][e, i] = w2_video(z[e, a:a + n], sets[i])
    out = {"ok": True}
    for r in ("sigma", "logit"):
        aucs = [g22.exact_auc(scores[r][e], labels) for e in range(len(z))]
        want = [float(x) for x in (entry.get("v_auc") or {}).get(r) or []]
        point = g22.first_best(aucs)
        diffs = [abs(a - b) for a, b in zip(aucs, want, strict=False)] if len(want) == len(aucs) \
            else [math.inf]
        row = {"point": point, "point_recorded": ((entry.get("selection") or {}).get(r) or {})
               .get("point"), "max_abs_diff": max(diffs) if diffs else math.inf}
        row["ok"] = bool(row["point"] == row["point_recorded"] and row["max_abs_diff"] <= tol)
        out[r] = row
        out["ok"] = out["ok"] and row["ok"]
    return out


def gate_w2(chains: Mapping[str, Mapping]) -> dict:
    """(w2) over the chains (label -> the keyword arguments of ``w2_chain``)."""
    rows = {k: w2_chain(**kw) for k, kw in chains.items()}
    return {"chains": rows, "n": len(rows),
            "ok": bool(rows) and all(r["ok"] for r in rows.values())}


def gate_n(histories: Mapping[str, Sequence[Mapping]], vals: Mapping[str, np.ndarray]) -> dict:
    """(n): non-finite values counted (training losses per chain, stored validation logits)."""
    loss = {k: sum(1 for h in hist for v in (h.get("loss") or {}).values()
                   if not (isinstance(v, (int, float)) and math.isfinite(v))) for k, hist in
            histories.items()}
    z = {k: int(np.size(v) - np.isfinite(np.asarray(v, dtype=np.float64)).sum())
         for k, v in vals.items()}
    return {"loss_nonfinite": loss, "val_nonfinite": z,
            "chains_with_nonfinite": sorted({k for k, v in loss.items() if v}
                                            | {k for k, v in z.items() if v}), "ok": True}


# --- test gates ---------------------------------------------------------------------------------

def gate_t(paths) -> dict:
    """(t), the first step of access (A): the test manifests and annotation files have E020's
    sha256 (bytes, restricted meta read) and the dense test stores E022's (bytes only)."""
    data = dt.manifest_hashes(paths)
    stores = ea.store_check(paths)
    return {"manifests": data, "stores": {k: v for k, v in stores.items() if k != "sha256"},
            "ok": bool(data["ok"] and stores["ok"])}


def gate_m(free_bytes: int | None, need_bytes: int | None, device: str) -> dict:
    """(m), just before access (A): the GPU's free memory >= (3j) maximum x 1.2 (otherwise
    wait); not applicable on a CPU device."""
    if device != "cuda":
        return {"device": device, "skipped": "not a CUDA device", "ok": True}
    if need_bytes is None or free_bytes is None:
        return {"device": device, "ok": False, "reason": "free or needed memory unknown"}
    want = MEMORY_FACTOR * int(need_bytes)
    return {"device": device, "free_bytes": int(free_bytes), "need_bytes": int(need_bytes),
            "factor": MEMORY_FACTOR, "ok": int(free_bytes) >= want}


def gate_k(loaded: Sequence[Mapping], blocked: Sequence[str] | frozenset) -> dict:
    """(k), right after access (A): no loaded checkpoint is on the blocked list."""
    bad = [r for r in loaded if str(r.get("sha256")) in set(blocked)]
    return {"loaded": len(loaded), "blocked_loaded": len(bad), "files": bad, "ok": not bad}


def v3_file(arrays: Mapping[str, np.ndarray], manifest, eval_points: int | None = None) -> dict:
    """(v3) of one score file: names and order as the restricted manifest read, total_frames,
    ceil(T / 16) rows per video, the rows of z (and of the D14 points); non-finite values only
    counted."""
    from wvad.extract import grid

    names = [str(x) for x in np.asarray(arrays.get("names", []))]
    totals = [int(x) for x in np.asarray(arrays.get("total_frames", []))]
    counts = [int(x) for x in np.asarray(arrays.get("counts", []))]
    z = np.asarray(arrays.get("z", np.zeros(0)))
    want_counts = [grid.n_snippets(int(t)) for t in manifest.total_frames]
    rows = int(sum(want_counts))
    shape_ok = (z.shape == (rows,)) if eval_points is None else (z.shape == (eval_points, rows))
    out = {"names_ok": names == list(manifest.names),
           "totals_ok": totals == list(manifest.total_frames),
           "counts_ok": counts == want_counts, "shape_ok": bool(shape_ok),
           "nonfinite": {k: int(np.size(arrays[k]) - np.isfinite(
               np.asarray(arrays[k], dtype=np.float64)).sum())
               for k in ("z", "zc", "sigma") if k in arrays}}
    out["ok"] = all(out[k] for k in ("names_ok", "totals_ok", "counts_ok", "shape_ok"))
    return out


def gate_v3(files: Mapping[str, Mapping], manifests: Mapping, targets: Mapping[str, Sequence],
            access0_targets: Mapping[str, Sequence], bound: Mapping[str, bool]) -> dict:
    """(v3), right after access (A) and before any metric: every expected score file (``files``:
    '<chain>/<run>__<rule>.npz' -> its v3_file row, or None when absent) is present with the
    right structure and bound to the call that wrote it (``bound``), and the videos cut into
    pieces (``targets``, the scoring call's record) are the access (0) list."""
    rows = dict(files)
    missing = sorted(k for k, v in rows.items() if v is None)
    bad = sorted(k for k, v in rows.items() if v is not None and not v["ok"])
    unbound = sorted(k for k in rows if not bound.get(k))
    tg = {d: {"scoring": list(targets.get(d) or []), "access0": list(access0_targets.get(d) or []),
              "ok": list(targets.get(d) or []) == list(access0_targets.get(d) or [])}
          for d in manifests}
    nonfinite = sorted(k for k, v in rows.items() if v is not None and any(v["nonfinite"]
                                                                            .values()))
    return {"files": len(rows), "missing": missing, "bad": bad, "unbound": unbound,
            "targets": tg, "nonfinite_files": nonfinite,
            "ok": bool(rows) and not missing and not bad and not unbound
            and all(t["ok"] for t in tg.values())}
