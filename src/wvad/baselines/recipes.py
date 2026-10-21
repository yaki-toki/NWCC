"""E023 recipe records (PLAN.md "일정·초매개변수 규칙(기본 D4)", "입력 처리 표", (3b), gate (q3)):
the official hyperparameters of each comparison method, its training units and inputs, its
sampler, its input-processing table, its scale constants, its precision and the official source
line of every value, and the D4 validation-only check grid as data. New code (add-only rule).

A method module ``wvad.baselines.<name>`` (``common`` documents the porting contract) exposes a
module-level ``RECIPE = Recipe(...)``; ``recipe_of(name)`` loads it through the registry (a lazy
import of that module), ``recipe_of(name, dataset)`` the recipe of the dataset's official
configuration (``common.dataset_method``: VCV has one per dataset), and ``recipes_record``
writes results/e023_recipes.json with the module RECIPE ("recipe") and both datasets' recipes
("datasets": the configurations the chains run and gates (a), (p), (q3) compare). Nothing here
imports a method module at import time.

Recipe fields (every value from the official code at the commit of results/e023_licences.json;
``sources`` names the file and lines of each)
    method          registry name (mil, rtfm, urdmu, vadclip_v; replacements mgfn, clip_tsa)
    official_unit   T of the official training sequences (32, 200 or 256)
    units           {unit: UnitInput(variant, test_sequence)} for every allowed --unit: the
                    official unit, and 32 (the X.32 flag arm, E021 "d16m", PLAN.md D11) when the
                    official unit is not 32. ``variant`` is the training input: "d16m" (the
                    E021 cache) or a wvad.extract.derive_n variant (d16m200, d16p256, d16m32l2,
                    ...); ``test_sequence`` the validation / test sequence rule: "whole" (the
                    whole video, cut only by the plan's N_chunk rule) or "chunk:<L>" (the
                    method's own fixed pieces of L snippets, e.g. VCV "chunk:256", VCV.32
                    "chunk:32")
    optimizer       OptimizerSpec(kind adam | adamw | sgd | adagrad, lr, weight_decay, betas, eps,
                    momentum)
    schedule        ScheduleSpec(kind constant | multistep, unit step | epoch, milestones, gamma):
                    the lr of step s; multistep multiplies by gamma once per milestone passed, in
                    milestone order (torch MultiStepLR's arithmetic)
    iterations, iteration_unit
                    the official training length: optimizer steps ("step") or epochs ("epoch",
                    one epoch = the sampler's steps_per_epoch())
    sampler, sampler_params
                    the training sampler (wvad.baselines.samplers name) and its parameters
                    (batch_normal, batch_abnormal, ...)
    loss_weights    official loss weights (name -> value)
    scale_constants official feature-norm constants (e.g. RTFM margin 100); the D4 check scales
                    every one of them by --scale_mult
    precision       "fp32" (PLAN.md "입력 처리 표": the official precision)
    input_table     one InputRow per item of INPUT_ITEMS (official, here, keep / adapt, source)
    sources         field -> Source(repo key of e023_licences.json, file, lines, note); the keys of
                    REQUIRED_SOURCES (and "scale_constants" when there are any)
    notes           free text (adaptations, deviations)

D4 (PLAN.md "검증 점검", 재확인 R3-7): learning rate x{1/3, 1, 3} of the official value, widened
once (x1/9 or x9) when an end value wins; then, at the chosen lr, every scale constant
x{1/10, 1, 10} (not widened); development seeds 0-2, the full official schedule, per setting
(dataset x pair). Score of a value: per seed the mean of the 5 highest of the 30 evaluation
points' sigma W-readout validation V-AUCs, then the mean over the 3 seeds (``d4_score``). Choice:
the highest score by strict >; a float64 tie with the official value keeps the official value;
other ties take the value closer to the official one (|log multiplier|), and an equal distance
the smaller multiplier (``d4_pick``). Eligibility (PLAN.md "실패 처리" "붕괴 제외", user
2026-10-20): a value takes part in the choice only when at least ``min_surviving`` of its
development seeds did not collapse (selected sigma V-AUC >= 0.60; the score is over those seeds);
with no eligible value nothing is chosen. Seed mode (PLAN.md "단일 seed 실행과 판정", user
2026-10-21): ``seed_mode`` "single" (the default, ``D4``) scores each value on its first
development seed in the order of ``seeds`` (0, then 1, then 2) that neither collapsed nor scored
non-finite, with ``min_surviving`` 1 (derived and enforced); when no value is eligible the
official value is used and recorded (the caller, PLAN.md "세부 규칙"); "multi" (the 3-seed rule
above, ``min_surviving`` 2, derived and enforced) is restored at the revision.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any

from wvad.paths import REPO_ROOT

E023_DIR = REPO_ROOT / "experiments" / "E023_localisation_baselines"
RESULTS = E023_DIR / "results"
RECIPES_FILE = RESULTS / "e023_recipes.json"
LICENCES_FILE = RESULTS / "e023_licences.json"

UNITS = (32, 200, 256)                                  # --unit values (PLAN.md 구현)
DATASETS = ("ucf", "xdv")
X32 = 32
E021_VARIANT = "d16m"                                   # X.32 training input (E021 cache)
OPTIMIZERS = ("adam", "adamw", "sgd", "adagrad")
SCHEDULES = ("constant", "multistep")
ITERATION_UNITS = ("step", "epoch")
PRECISIONS = ("fp32",)
LICENCE_KEYS = ("mil", "rtfm", "urdmu", "vcv", "mgfn", "clip_tsa")
# the rows of PLAN.md "입력 처리 표" (each recipe fills every row)
INPUT_ITEMS = ("features", "clip_normalisation", "sampler_and_length", "scale_match",
               "input_normalisation", "scale_constants", "text_branch", "top_k",
               "checkpoint_selection", "precision")
INPUT_STATUS = ("keep", "adapt")
REQUIRED_SOURCES = ("optimizer", "lr", "weight_decay", "schedule", "iterations", "batch",
                    "sampler", "loss_weights", "unit", "test_sequence")


class RecipeError(ValueError):
    """A recipe is malformed or inconsistent with the plan's rules."""


def _frozen(mapping: Mapping | None) -> Mapping:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class Source:
    """Where a recipe value comes from in the official repository."""

    repo: str                    # key of results/e023_licences.json "repos"
    file: str                    # path inside the official repository (posix)
    lines: str                   # "L12" or "L12-L30"
    note: str = ""

    def to_json(self) -> dict:
        return {"repo": self.repo, "file": self.file, "lines": self.lines, "note": self.note}


@dataclass(frozen=True)
class OptimizerSpec:
    kind: str
    lr: float
    weight_decay: float = 0.0
    betas: tuple[float, float] | None = None
    eps: float | None = None
    momentum: float | None = None

    def to_json(self) -> dict:
        return {"kind": self.kind, "lr": self.lr, "weight_decay": self.weight_decay,
                "betas": None if self.betas is None else list(self.betas), "eps": self.eps,
                "momentum": self.momentum}


@dataclass(frozen=True)
class ScheduleSpec:
    kind: str = "constant"
    unit: str = "step"
    milestones: tuple[int, ...] = ()
    gamma: float = 1.0

    def to_json(self) -> dict:
        return {"kind": self.kind, "unit": self.unit, "milestones": list(self.milestones),
                "gamma": self.gamma}


@dataclass(frozen=True)
class UnitInput:
    variant: str                 # "d16m" (E021, X.32) or a wvad.extract.derive_n variant
    test_sequence: str = "whole"     # "whole" | "chunk:<L>"

    def chunk(self) -> int | None:
        """L of "chunk:<L>", None for "whole"."""
        return parse_test_sequence(self.test_sequence)

    def to_json(self) -> dict:
        return {"variant": self.variant, "test_sequence": self.test_sequence}


@dataclass(frozen=True)
class InputRow:
    item: str                    # one of INPUT_ITEMS
    official: str
    here: str
    status: str                  # "keep" | "adapt"
    source: str = ""

    def to_json(self) -> dict:
        return {"item": self.item, "official": self.official, "here": self.here,
                "status": self.status, "source": self.source}


@dataclass(frozen=True)
class Recipe:
    method: str
    official_unit: int
    units: Mapping[int, UnitInput]
    optimizer: OptimizerSpec
    schedule: ScheduleSpec
    iterations: int
    iteration_unit: str
    sampler: str
    sampler_params: Mapping[str, Any] = field(default_factory=dict)
    loss_weights: Mapping[str, float] = field(default_factory=dict)
    scale_constants: Mapping[str, float] = field(default_factory=dict)
    precision: str = "fp32"
    input_table: tuple[InputRow, ...] = ()
    sources: Mapping[str, Source] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:          # read-only views (a recipe is a record)
        for name in ("units", "sampler_params", "loss_weights", "scale_constants", "sources"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "input_table", tuple(self.input_table))

    def unit_input(self, unit: int) -> UnitInput:
        if int(unit) not in self.units:
            raise RecipeError(f"{self.method}: unit {unit} is not a registered unit "
                              f"{sorted(self.units)}")
        return self.units[int(unit)]

    def scaled_constants(self, scale_mult: float = 1.0) -> dict[str, float]:
        """The scale constants times the D4 multiplier (refused for a recipe without any)."""
        if not self.scale_constants and float(scale_mult) != 1.0:
            raise RecipeError(f"{self.method} has no scale constant: --scale_mult must be 1")
        return {k: float(v) * float(scale_mult) for k, v in self.scale_constants.items()}

    def to_json(self) -> dict:
        return {"method": self.method, "official_unit": self.official_unit,
                "units": {str(u): ui.to_json() for u, ui in sorted(self.units.items())},
                "optimizer": self.optimizer.to_json(), "schedule": self.schedule.to_json(),
                "iterations": self.iterations, "iteration_unit": self.iteration_unit,
                "sampler": self.sampler, "sampler_params": _jsonable(self.sampler_params),
                "loss_weights": dict(self.loss_weights),
                "scale_constants": dict(self.scale_constants), "precision": self.precision,
                "input_table": [r.to_json() for r in self.input_table],
                "sources": {k: s.to_json() for k, s in self.sources.items()},
                "notes": self.notes}


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_test_sequence(text: str) -> int | None:
    if text == "whole":
        return None
    head, _, tail = str(text).partition(":")
    if head != "chunk" or not tail.isdigit() or int(tail) < 1:
        raise RecipeError(f"test_sequence {text!r}: 'whole' or 'chunk:<L>' (L >= 1)")
    return int(tail)


def _finite(x, positive: bool = False) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and (v > 0 if positive else True)


def _variant_problem(unit: int, variant: str) -> str | None:
    if variant == E021_VARIANT:
        return None if unit == X32 else f"unit {unit}: the E021 '{E021_VARIANT}' cache is 32-unit"
    from wvad.extract.derive_n import VariantError, variant_spec  # light: no data read

    try:
        spec = variant_spec(variant)
    except VariantError as exc:
        return str(exc)
    if spec.unit != unit:
        return f"unit {unit}: variant {variant} is {spec.unit}-unit"
    return None


def check_recipe(recipe: Recipe) -> list[str]:
    """Problems of a recipe against the plan's rules (empty: well-formed). The sampler's
    registration is checked by the registry (common.check_method), after the import."""
    out = []
    if not recipe.method:
        out.append("method name missing")
    if recipe.official_unit not in UNITS:
        out.append(f"official_unit {recipe.official_unit} not in {UNITS}")
    if recipe.official_unit not in recipe.units:
        out.append("units lack the official unit")
    if recipe.official_unit != X32 and X32 not in recipe.units:
        out.append("units lack 32 (the X.32 flag arm, PLAN.md D11)")
    for u, ui in recipe.units.items():
        if u not in UNITS:
            out.append(f"unit {u} not in {UNITS}")
            continue
        p = _variant_problem(u, ui.variant)
        if p:
            out.append(p)
        if u == X32 and recipe.official_unit != X32 and ui.variant != E021_VARIANT:
            out.append(f"X.32 trains on the E021 '{E021_VARIANT}' cache (PLAN.md), not "
                       f"{ui.variant}")
        try:
            parse_test_sequence(ui.test_sequence)
        except RecipeError as exc:
            out.append(str(exc))
    o = recipe.optimizer
    if o.kind not in OPTIMIZERS:
        out.append(f"optimizer {o.kind!r} not in {OPTIMIZERS}")
    if not _finite(o.lr, positive=True):
        out.append(f"lr {o.lr!r} must be finite and > 0")
    if not _finite(o.weight_decay) or float(o.weight_decay) < 0:
        out.append(f"weight_decay {o.weight_decay!r} must be finite and >= 0")
    s = recipe.schedule
    if s.kind not in SCHEDULES or s.unit not in ITERATION_UNITS:
        out.append(f"schedule {s.kind!r} / {s.unit!r} not in {SCHEDULES} / {ITERATION_UNITS}")
    if s.kind == "multistep" and (not s.milestones or list(s.milestones) != sorted(
            set(s.milestones)) or not _finite(s.gamma, positive=True)):
        out.append("multistep needs increasing distinct milestones and a gamma > 0")
    if not isinstance(recipe.iterations, int) or recipe.iterations < 1:
        out.append(f"iterations {recipe.iterations!r} must be a positive integer")
    if recipe.iteration_unit not in ITERATION_UNITS:
        out.append(f"iteration_unit {recipe.iteration_unit!r} not in {ITERATION_UNITS}")
    if recipe.iteration_unit == "step" and isinstance(recipe.iterations, int) \
            and recipe.iterations < 30:
        out.append("fewer than 30 steps: the 30 evaluation points do not fit")
    if not recipe.sampler:
        out.append("sampler missing")
    bad_w = [k for k, v in recipe.loss_weights.items() if not _finite(v)]
    bad_c = [k for k, v in recipe.scale_constants.items() if not _finite(v, positive=True)]
    if bad_w or bad_c:
        out.append(f"non-finite loss weights {bad_w} or non-positive scale constants {bad_c}")
    if recipe.precision not in PRECISIONS:
        out.append(f"precision {recipe.precision!r} not in {PRECISIONS}")
    items = [r.item for r in recipe.input_table]
    if sorted(items) != sorted(INPUT_ITEMS):
        out.append(f"input_table rows {items} != the plan's items {list(INPUT_ITEMS)}")
    if any(r.status not in INPUT_STATUS for r in recipe.input_table):
        out.append(f"input_table status must be one of {INPUT_STATUS}")
    need = list(REQUIRED_SOURCES) + (["scale_constants"] if recipe.scale_constants else [])
    missing = [k for k in need if k not in recipe.sources]
    if missing:
        out.append(f"sources missing for {missing}")
    if any(src.repo not in LICENCE_KEYS for src in recipe.sources.values()):
        out.append(f"source repo keys must be in {LICENCE_KEYS}")
    return out


def require_recipe(recipe: Recipe) -> Recipe:
    problems = check_recipe(recipe)
    if problems:
        raise RecipeError(f"recipe of {recipe.method!r}: {problems}")
    return recipe


def recipe_of(name: str, dataset: str | None = None) -> Recipe:
    """The module-level RECIPE of wvad.baselines.<name> (lazy import, contract-checked), or with
    ``dataset`` the recipe of that dataset's official configuration (``common.method_on``)."""
    from wvad.baselines.common import get_method, method_on  # lazy: no cycle

    return get_method(name).recipe if dataset is None else method_on(name, dataset).recipe


def dataset_recipe(entry: Mapping, dataset: str) -> Mapping | None:
    """The recipe a (3b) method entry records for ``dataset`` ("datasets"; an entry without them
    has one configuration: "recipe")."""
    by_ds = entry.get("datasets") or {}
    return by_ds.get(dataset) if by_ds else entry.get("recipe")


# --- learning rate -----------------------------------------------------------------------------

def lr_at(schedule: ScheduleSpec, base_lr: float, step: int,
          steps_per_epoch: int | None = None) -> float:
    """The lr of optimizer step ``step`` (1-based): the base lr, multiplied by gamma once for
    every milestone already reached (in the schedule's unit; an epoch is ``steps_per_epoch``
    steps), in milestone order as torch MultiStepLR does."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if schedule.kind == "constant":
        return float(base_lr)
    if schedule.kind != "multistep":
        raise RecipeError(f"unknown schedule {schedule.kind!r}")
    if schedule.unit == "epoch":
        if not steps_per_epoch:
            raise RecipeError("an epoch schedule needs the sampler's steps_per_epoch")
        position = (step - 1) // int(steps_per_epoch)
    else:
        position = step - 1
    lr = float(base_lr)
    for m in schedule.milestones:
        if position >= int(m):
            lr = lr * float(schedule.gamma)
    return lr


def total_steps(recipe: Recipe, steps_per_epoch: int | None = None) -> int:
    """S: optimizer steps of the official schedule."""
    if recipe.iteration_unit == "step":
        return int(recipe.iterations)
    if not steps_per_epoch:
        raise RecipeError(f"{recipe.method}: an epoch-based schedule needs the sampler's "
                          "steps_per_epoch")
    return int(recipe.iterations) * int(steps_per_epoch)


# --- D4 grid (validation only) -------------------------------------------------------------

def parse_mult(text: str | float) -> float:
    """A multiplier given as '1/3', '3', '0.1' or a number."""
    if isinstance(text, (int, float)):
        v = float(text)
    else:
        v = float(Fraction(str(text).strip()))
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"multiplier must be finite and > 0, got {text!r}")
    return v


MIN_SURVIVING = {"single": 1, "multi": 2}        # the seed mode fixes the eligibility rule


@dataclass(frozen=True)
class D4Grid:
    lr_mults: tuple[str, ...] = ("1/3", "1", "3")
    widen: Mapping[str, str] = field(default_factory=lambda: {"1/3": "1/9", "3": "9"})
    scale_mults: tuple[str, ...] = ("1/10", "1", "10")
    seeds: tuple[int, ...] = (0, 1, 2)
    eval_points: int = 30
    top: int = 5
    readout: str = "sigma"
    official: str = "1"
    min_surviving: int | None = None             # development seeds left after "붕괴 제외"
    seed_mode: str = "single"                    # PLAN.md "단일 seed 실행과 판정" (2026-10-21)

    def __post_init__(self) -> None:
        object.__setattr__(self, "widen", _frozen(self.widen))
        if self.seed_mode not in MIN_SURVIVING:
            raise ValueError(f"seed mode {self.seed_mode!r} (single, multi)")
        want = MIN_SURVIVING[self.seed_mode]
        if self.min_surviving is None:
            object.__setattr__(self, "min_surviving", want)
        elif self.min_surviving != want:
            raise ValueError(f"the {self.seed_mode}-seed D4 needs min_surviving {want}")

    def to_json(self) -> dict:
        return {"lr_mults": list(self.lr_mults), "widen": dict(self.widen),
                "scale_mults": list(self.scale_mults), "seeds": list(self.seeds),
                "eval_points": self.eval_points, "top": self.top, "readout": self.readout,
                "official": self.official, "min_surviving": self.min_surviving,
                "seed_mode": self.seed_mode,
                "seeds_used": "the first development seed of the list that did not collapse "
                              "(PLAN.md 단일 seed 실행과 판정)" if self.seed_mode == "single"
                              else "every development seed that did not collapse",
                "score": f"per seed the mean of the {self.top} highest of the "
                         f"{self.eval_points} evaluation points' {self.readout} W-readout "
                         "validation V-AUCs, then the mean over the seeds",
                "choice": "strict > on the score; a float64 tie with the official value keeps "
                          "it; other ties: closer to the official value (|log m|), then the "
                          "smaller multiplier",
                "widening": "lr only, once, in the direction of a winning end value; scale "
                            "constants are not widened",
                "eligibility": f"a value takes part in the choice only with at least "
                               f"{self.min_surviving} development seeds that did not collapse "
                               "(selected sigma V-AUC >= 0.60, PLAN.md 붕괴 제외) or fail "
                               "numerically (a non-finite V-AUC); none eligible: "
                               + ("the official value (recorded)" if self.seed_mode == "single"
                                  else "no choice")}


D4 = D4Grid()


def d4_score(v_auc, grid: D4Grid = D4) -> float:
    """(seeds, eval_points) sigma V-AUCs -> per seed the mean of the ``top`` highest, then the
    mean over the seeds (float64). A non-finite value makes the score NaN."""
    import numpy as np

    a = np.asarray(v_auc, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != grid.eval_points or a.shape[0] < 1:
        raise ValueError(f"expected (seeds, {grid.eval_points}) V-AUCs, got {a.shape}")
    if not np.isfinite(a).all():
        return float("nan")
    per_seed = np.sort(a, axis=1)[:, ::-1][:, :grid.top].mean(axis=1)
    return float(per_seed.mean())


def d4_pick(scores: Mapping[str, float], grid: D4Grid = D4,
            surviving: Mapping[str, int | None] | None = None) -> str | None:
    """The chosen multiplier label: highest score by strict >; ties keep the official value,
    else the one closer to it (|log m|), then the smaller multiplier. Non-finite scores never
    win; all non-finite: the official value. ``surviving`` (value -> development seeds that did
    not collapse): a value with fewer than ``grid.min_surviving`` is not eligible and never
    chosen; no eligible value (or only non-finite ones without the official value): None."""
    cand = {k: v for k, v in scores.items()
            if surviving is None or (surviving.get(k) or 0) >= grid.min_surviving}
    if not cand:
        return None
    finite = {k: float(v) for k, v in cand.items() if math.isfinite(float(v))}
    if not finite:
        return grid.official if surviving is None or grid.official in cand else None
    best = max(finite.values())
    tied = [k for k, v in finite.items() if v == best]
    if grid.official in tied:
        return grid.official
    return min(tied, key=lambda k: (abs(math.log(parse_mult(k))), parse_mult(k)))


def d4_widen(pick: str, grid: D4Grid = D4) -> str | None:
    """The one further lr value to try when an end value won (None otherwise)."""
    return grid.widen.get(pick)


# --- the (3b) record ------------------------------------------------------------------------

def licences(path: str | os.PathLike[str] = LICENCES_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recipes_record(names: Sequence[str], licence_file: str | os.PathLike[str] = LICENCES_FILE
                   ) -> dict:
    """results/e023_recipes.json: every named recipe (contract-checked), its module file's
    sha256, the licence record of its sources and the D4 grid."""
    import importlib.util

    from wvad.diagnostics.common import code_version
    from wvad.prep.manifests import sha256_file

    lic = licences(licence_file).get("repos", {})
    methods = {}
    for name in names:
        recipe = recipe_of(name)
        by_ds = {d: recipe_of(name, d) for d in DATASETS}
        spec = importlib.util.find_spec(f"wvad.baselines.{name}")
        origin = Path(spec.origin) if spec is not None and spec.origin else None
        repos = sorted({s.repo for r in (recipe, *by_ds.values()) for s in r.sources.values()})
        methods[name] = {"recipe": recipe.to_json(),
                         "datasets": {d: r.to_json() for d, r in by_ds.items()},
                         "per_dataset_configuration": any(r.to_json() != recipe.to_json()
                                                          for r in by_ds.values()),
                         "module": None if origin is None else
                         origin.relative_to(REPO_ROOT).as_posix(),
                         "module_sha256": None if origin is None else sha256_file(origin),
                         "licences": {r: lic.get(r) for r in repos}}
    return {"check": "E023 (3b) recipes", "code": code_version(), "methods": methods,
            "d4": D4.to_json(), "units": list(UNITS)}


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    from wvad.diagnostics.common import write_json

    p = argparse.ArgumentParser(prog="python -m wvad.baselines.recipes",
                                description="(3b): write results/e023_recipes.json")
    p.add_argument("--methods", nargs="+", default=["mil", "rtfm", "urdmu", "vadclip_v"])
    p.add_argument("--out", type=Path, default=RECIPES_FILE)
    a = p.parse_args(argv)
    rec = recipes_record(a.methods)
    write_json(Path(a.out), rec)
    print(f"[recipes] {len(rec['methods'])} recipes -> {a.out}")


if __name__ == "__main__":
    main()
