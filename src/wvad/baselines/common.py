"""E023 comparison-method interface, registry and shared rules (PLAN.md "설계", "같은 특징 규칙과
학습 단위", "입력 처리 표", "체크포인트 선택", "구현"). New code (add-only rule).

PORTING CONTRACT (one agent per method; each adds exactly ONE file and edits no shared file)
-------------------------------------------------------------------------------------------
File: ``src/wvad/baselines/<name>.py`` with ``<name>`` one of ``mil``, ``rtfm``, ``urdmu``,
``vadclip_v`` (replacements ``mgfn``, ``clip_tsa``, only if D3's replacement rule is triggered).
The registry (``get_method``) imports it lazily by that name; nothing imports it eagerly.

1. Module docstring: the official repository URL and commit and the licence exactly as in
   experiments/E023_localisation_baselines/results/e023_licences.json (key: LICENCE_KEY[name]),
   the porting mode (UR-DMU MIT / VadCLIP Apache-2.0: copied with attribution, the copied
   functions named; MIL, RTFM, MGFN, CLIP-TSA: re-implemented, the official code used only
   outside the repository for the equivalence checks), and a line "Adaptations:" listing every
   change against the official code (input dimension, T, input LayerNorm, removed text branch,
   ...). ``get_method`` (``contract_problems``) refuses a module whose docstring lacks the URL,
   the commit or "Adaptations:".
2. Module-level ``RECIPE = wvad.baselines.recipes.Recipe(...)`` (see that module): official
   optimizer, lr, weight decay, schedule, iterations, batch composition, sampler, loss weights,
   scale constants, fp32, the input-processing table rows and the official source line of every
   value; ``units`` holds the official unit and, when it is not 32, 32 -> UnitInput("d16m")
   (X.32). ``check_recipe`` must return no problem.
3. Module-level ``METHOD = <subclass of BaselineMethod>()`` with ``name = "<name>"`` and
   ``official_unit`` = RECIPE.official_unit, implementing:
     build(feature_dim, *, unit, input_ln, scale) -> torch.nn.Module
         The official network for D = feature_dim (1792 XCB+XCL, 1920 XCB+SigLIP) and training
         length ``unit``, constructed on the CPU right after the loop seeded the generators, in
         the official construction order (so the global torch generator is consumed as by the
         official constructor). With ``input_ln`` (XCB+SigLIP; PLAN.md "입력 처리 표") the model
         applies ``input_layer(feature_dim, True)`` (nn.LayerNorm(D)) to its input first, as a
         submodule (it is in the state_dict); otherwise nothing is added. ``scale`` = the recipe's
         scale constants times --scale_mult (e.g. {"margin": 100.0}).
     loss(model, batch, *, unit, scale) -> LossOutput
         The official training loss of one Batch (normal videos first unless the sampler's order
         says otherwise; ``batch.features`` (B, unit, D) float32 (float64 in the MIL (q1b)
         float64 run), zero rows past each length; ``batch.labels`` (B,) 0 / 1;
         ``batch.lengths`` (B,) valid rows; ``batch.mask`` (B, unit); ``batch.videos``,
         ``batch.n_normal``), the model in train mode (the loop called set_mode(model, True)).
         LossOutput(total=<0-dim tensor>, parts={name: float or 0-dim tensor}).
     sequence_logits(model, x, *, unit) -> torch.Tensor
         x: (n, D) float32 snippet features of ONE video piece in time order (n <= N_chunk), on
         the model's device; the caller has set eval mode (set_mode(model, False)), no_grad and
         autocast off. Returns (n,) float logits whose fp32 sigmoid is the method's official
         snippet score (all four official scores are sigmoids). It implements the method's own
         test-sequence rule for this unit (RECIPE.units[unit].test_sequence): "whole" = the piece
         as one sequence; "chunk:<L>" = the official fixed pieces of L snippets, the last one
         padded and masked as the official test code does (VCV: L = 256; VCV.32: L = 32). The
         plan's N_chunk rule is applied outside, by ``score_video`` (common to validation and
         test): only a video with n_v > N_chunk is cut into non-overlapping N_chunk pieces.
   and optionally overriding (defaults below):
     for_dataset(dataset)            the method with the official configuration of ``dataset``
                                     (default: self, one configuration for both datasets; VCV:
                                     the UCF or the XD option file). Every user of a method on a
                                     dataset - the training chain, the D4 / validation / test
                                     scoring, the (3b) record and the gates - resolves it through
                                     ``dataset_method``
     set_mode(model, train)          model.train(train) (+ e.g. UR-DMU's ``net.flag``)
     make_optimizer(model, lr)       from RECIPE.optimizer (lr = official lr x --lr_mult)
     lr_at(step, base_lr, steps_per_epoch)
                                     the lr of step ``step`` from RECIPE.schedule
                                     (recipes.lr_at: MultiStepLR arithmetic)
     set_lr(optimizer, lr)           every param group's lr (called when the lr changes)
     train_step(model, batch, optimizer, *, unit, scale) -> {name: float}
                                     zero_grad, loss.total.backward(), step (the official order
                                     of RTFM / UR-DMU / VadCLIP); override for clipping etc.
     sequence_multiple(unit)         L of "chunk:<L>" (1 for "whole"): N_chunk must be a
                                     multiple of it
4. A method-specific sampler (e.g. Sultani's per-iteration np.random.permutation draws) is a
   ``samplers.Sampler`` subclass registered at import with ``samplers.register("<name>", cls,
   owner=__name__)``; the generic "window_mean" sampler covers the paired-loader loops. A sampler
   draws only from the global generators the official code uses.
5. Never import ``wvad.baselines.official`` (equivalence harnesses only; an import-graph test
   fails otherwise), never assign into another wvad module, never use the AMP / autocast path
   (training is fp32; the MIL (q1b) check may run the loop in float64).
6. Training inputs and processing are not the method's business: the loop (``train.py``)
   resolves the unit's variant (derived cache or E021 d16m), the scale match, the per-snippet l2
   of an l2 variant and the input LN flag; the same InputPipeline feeds validation and test.

Shared rules defined here
    N_CHUNK_DEFAULT = 4096      the plan's default N_chunk (fixed by measurement (3j); a power of
                                two); chunk_bounds / chunk_targets / score_video
    EVAL_POINTS = 30            evaluation points of a chain (round half up of k S / 30)
    readouts                    W readout of the stored snippet logits on the E021 32-window grid
                                (wvad.eval.dense_readout.readout, no offset: c(v) = 0): "sigma"
                                video score = max fp32 sigma(u_k), "logit" = max u_k; full
                                validation V-AUC by wvad.eval.crossfit.rank_auc; selection =
                                first maximum (strict >, earliest evaluation point on ties)
    input_ln(pair)              LayerNorm on the input iff SigLIP is in the pair (the C rule)
    InputPipeline               per encoder: [l2_rows] -> [scale map] (float64) -> float32,
                                then concatenation in pair order
    isolated_rng(device)        torch.random.fork_rng (CPU and the CUDA device) plus numpy and
                                random state save / restore around every evaluation
    dataset_method(m, dataset)  m.for_dataset(dataset), checked (method_on(name, dataset): the
                                registered method on a dataset)
    file_lock / update_json     read-modify-write of a shared record (results/e023_q1b.json)
                                under an exclusive lock file
"""

from __future__ import annotations

import abc
import contextlib
import importlib
import importlib.util
import json
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from wvad.baselines import samplers
from wvad.baselines.recipes import LICENCES_FILE, Recipe, check_recipe, lr_at, require_recipe
from wvad.data.catalog import ALL_DATASETS, ENCODER_PAIRS
from wvad.data.dense import ScaleMap
from wvad.eval.crossfit import rank_auc
from wvad.eval.dense_readout import readout, sigmoid32
from wvad.extract import grid

MAIN_METHODS = ("mil", "rtfm", "urdmu", "vadclip_v")
REPLACEMENT_METHODS = ("mgfn", "clip_tsa")          # D3: VCV -> CLIP-TSA, RTFM -> MGFN
KNOWN_METHODS = MAIN_METHODS + REPLACEMENT_METHODS
LICENCE_KEY = {"mil": "mil", "rtfm": "rtfm", "urdmu": "urdmu", "vadclip_v": "vcv",
               "mgfn": "mgfn", "clip_tsa": "clip_tsa"}
PACKAGE = "wvad.baselines"
N_CHUNK_DEFAULT = 4096
EVAL_POINTS = 30
READOUTS = ("sigma", "logit")


class MethodError(ValueError):
    """A method module is missing or breaks the porting contract."""


# --- batches and losses -----------------------------------------------------------------------

@dataclass(frozen=True)
class Batch:
    features: torch.Tensor          # (B, T, D) float (zero rows past each length)
    labels: torch.Tensor            # (B,) float 0 / 1
    lengths: torch.Tensor           # (B,) int64 valid rows (T when unpadded)
    videos: tuple[int, ...]         # training-set positions, batch order
    n_normal: int

    @property
    def mask(self) -> torch.Tensor:
        """(B, T) bool: True on valid rows."""
        t = self.features.shape[1]
        return torch.arange(t, device=self.lengths.device)[None, :] < self.lengths[:, None]


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    parts: Mapping[str, float | torch.Tensor] = field(default_factory=dict)


def collate(draw: samplers.StepDraw, video: Callable[[int], torch.Tensor], unit: int, *,
            device: torch.device, dtype: torch.dtype = torch.float32) -> Batch:
    """One step's Batch: each drawn video's training rows (its row selection, if any), zero-padded
    to ``unit`` rows (lengths keep the valid count); a video with more than ``unit`` rows is an
    error (a derived input has at most ``unit`` rows)."""
    xs, lengths = [], []
    for i, v in enumerate(draw.videos):
        x = video(int(v))
        if draw.rows is not None and draw.rows[i] is not None:
            x = x[torch.as_tensor(np.asarray(draw.rows[i], dtype=np.int64))]
        if x.dim() != 2 or x.shape[0] < 1 or x.shape[0] > unit:
            raise ValueError(f"video {v}: {tuple(x.shape)} training rows for unit {unit}")
        lengths.append(int(x.shape[0]))
        if x.shape[0] < unit:
            x = torch.cat([x, x.new_zeros((unit - x.shape[0], x.shape[1]))])
        xs.append(x)
    return Batch(torch.stack(xs).to(device=device, dtype=dtype),
                 torch.tensor(draw.labels, dtype=dtype, device=device),
                 torch.tensor(lengths, dtype=torch.int64, device=device),
                 tuple(int(v) for v in draw.videos), draw.n_normal)


# --- the method interface ---------------------------------------------------------------------

class BaselineMethod(abc.ABC):
    """A comparison method (module docstring, "PORTING CONTRACT")."""

    name: ClassVar[str] = ""
    official_unit: ClassVar[int] = 0

    @property
    def recipe(self) -> Recipe:
        """The module-level RECIPE of the module defining this class."""
        module = importlib.import_module(type(self).__module__)
        rec = getattr(module, "RECIPE", None)
        if not isinstance(rec, Recipe):
            raise MethodError(f"{type(self).__module__} has no module-level RECIPE")
        return rec

    @abc.abstractmethod
    def build(self, feature_dim: int, *, unit: int, input_ln: bool,
              scale: Mapping[str, float]) -> nn.Module:
        """The official network (CPU, official construction order)."""

    @abc.abstractmethod
    def loss(self, model: nn.Module, batch: Batch, *, unit: int,
             scale: Mapping[str, float]) -> LossOutput:
        """The official training loss of one batch (train mode)."""

    @abc.abstractmethod
    def sequence_logits(self, model: nn.Module, x: torch.Tensor, *, unit: int) -> torch.Tensor:
        """(n,) snippet logits of one video piece (n, D) (eval mode, no grad)."""

    def for_dataset(self, dataset: str) -> BaselineMethod:
        """The method with the official configuration of ``dataset`` (default: this one; a
        method whose official code configures the datasets differently overrides it)."""
        return self

    def set_mode(self, model: nn.Module, train: bool) -> None:
        model.train(train)

    def make_optimizer(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        o = self.recipe.optimizer
        kw: dict = {"lr": float(lr), "weight_decay": float(o.weight_decay)}
        if o.eps is not None:
            kw["eps"] = float(o.eps)
        if o.kind in ("adam", "adamw"):
            if o.betas is not None:
                kw["betas"] = tuple(float(b) for b in o.betas)
            cls = torch.optim.Adam if o.kind == "adam" else torch.optim.AdamW
            return cls(model.parameters(), **kw)
        if o.kind == "sgd":
            return torch.optim.SGD(model.parameters(), momentum=float(o.momentum or 0.0), **kw)
        if o.kind == "adagrad":
            return torch.optim.Adagrad(model.parameters(), **kw)
        raise MethodError(f"unknown optimizer {o.kind!r}")

    def set_lr(self, optimizer: torch.optim.Optimizer, lr: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = float(lr)

    def lr_at(self, step: int, base_lr: float, steps_per_epoch: int | None) -> float:
        return lr_at(self.recipe.schedule, base_lr, step, steps_per_epoch)

    def train_step(self, model: nn.Module, batch: Batch, optimizer: torch.optim.Optimizer, *,
                   unit: int, scale: Mapping[str, float]) -> dict[str, float]:
        out = self.loss(model, batch, unit=unit, scale=scale)
        optimizer.zero_grad()
        out.total.backward()
        optimizer.step()
        parts = {k: float(v.detach() if isinstance(v, torch.Tensor) else v)
                 for k, v in out.parts.items()}
        return {"total": float(out.total.detach()), **parts}

    def sequence_multiple(self, unit: int) -> int:
        return self.recipe.unit_input(unit).chunk() or 1


# --- registry -------------------------------------------------------------------------------

def method_module(name: str) -> str:
    if name not in KNOWN_METHODS:
        raise MethodError(f"unknown method {name!r} (known: {KNOWN_METHODS})")
    return f"{PACKAGE}.{name}"


def is_ported(name: str) -> bool:
    """The method's file exists (found without importing it)."""
    return importlib.util.find_spec(method_module(name)) is not None


def available_methods(include_replacements: bool = False) -> tuple[str, ...]:
    names = KNOWN_METHODS if include_replacements else MAIN_METHODS
    return tuple(n for n in names if is_ported(n))


def _licence(name: str) -> dict:
    try:
        rec = json.loads(LICENCES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MethodError(f"licence record {LICENCES_FILE} unreadable ({exc!r})") from exc
    entry = (rec.get("repos") or {}).get(LICENCE_KEY[name])
    if not entry:
        raise MethodError(f"no licence entry {LICENCE_KEY[name]!r} for {name}")
    return entry


def contract_problems(module: ModuleType, name: str) -> list[str]:
    """Problems of a method module against the porting contract (empty: it holds)."""
    out = []
    method = getattr(module, "METHOD", None)
    recipe = getattr(module, "RECIPE", None)
    if not isinstance(method, BaselineMethod):
        out.append("no module-level METHOD (a BaselineMethod instance)")
    if not isinstance(recipe, Recipe):
        out.append("no module-level RECIPE (a recipes.Recipe)")
    if out:
        return out
    if method.name != name or recipe.method != name:
        out.append(f"METHOD.name {method.name!r} / RECIPE.method {recipe.method!r} != {name!r}")
    if method.official_unit != recipe.official_unit:
        out.append(f"METHOD.official_unit {method.official_unit} != RECIPE.official_unit "
                   f"{recipe.official_unit}")
    out += [f"recipe: {p}" for p in check_recipe(recipe)]
    if recipe.sampler not in samplers.registered():
        out.append(f"sampler {recipe.sampler!r} is not registered after the import")
    doc = module.__doc__ or ""
    try:
        lic = _licence(name)
        missing = [k for k in ("url", "commit") if str(lic.get(k) or "") not in doc]
        if missing:
            out.append(f"module docstring lacks the official {missing} of e023_licences.json")
    except MethodError as exc:
        out.append(str(exc))
    if "Adaptations:" not in doc:
        out.append("module docstring lacks the 'Adaptations:' list")
    for u in recipe.units:
        try:
            if N_CHUNK_DEFAULT % method.sequence_multiple(u):
                out.append(f"unit {u}: N_chunk {N_CHUNK_DEFAULT} is not a multiple of the "
                           "method's sequence piece")
        except Exception as exc:  # noqa: BLE001 - reported as a contract problem
            out.append(f"unit {u}: sequence_multiple failed ({exc!r})")
    return out


def get_method(name: str) -> BaselineMethod:
    """The METHOD of wvad.baselines.<name>, imported now (lazily) and contract-checked."""
    mod_name = method_module(name)
    if not is_ported(name):
        raise MethodError(f"method {name!r} is not ported yet ({mod_name} does not exist)")
    module = importlib.import_module(mod_name)
    problems = contract_problems(module, name)
    if problems:
        raise MethodError(f"{mod_name} breaks the porting contract: {problems}")
    return module.METHOD


def main_family() -> dict[str, BaselineMethod]:
    """The ported main-family methods (never the replacements)."""
    return {n: get_method(n) for n in available_methods(False)}


DATASETS = ALL_DATASETS              # E024 stage 2: UBnormal registered beside UCF and XD


def dataset_method(method: BaselineMethod, dataset: str) -> BaselineMethod:
    """``method.for_dataset(dataset)``, checked: a BaselineMethod of the same name and official
    unit whose recipe is well-formed (else recipes.RecipeError) and whose sampler is registered
    (the one place where a method meets a dataset: training chain, D4, validation / test
    scoring, records, gates)."""
    if dataset not in DATASETS:
        raise MethodError(f"unknown dataset {dataset!r} (known: {DATASETS})")
    m = method.for_dataset(dataset)
    if not isinstance(m, BaselineMethod):
        raise MethodError(f"{method.name}.for_dataset({dataset!r}) is no BaselineMethod")
    if m.name != method.name or m.official_unit != method.official_unit:
        raise MethodError(f"{method.name}.for_dataset({dataset!r}) changed the name or the "
                          f"official unit ({m.name!r}, {m.official_unit})")
    require_recipe(m.recipe)                       # RecipeError with the problems
    if m.recipe.sampler not in samplers.registered():
        raise MethodError(f"{method.name} on {dataset}: sampler {m.recipe.sampler!r} is not "
                          "registered")
    return m


def method_on(name: str, dataset: str) -> BaselineMethod:
    """The registered method ``name`` with the official configuration of ``dataset``."""
    return dataset_method(get_method(name), dataset)


# --- records shared by concurrent runs ----------------------------------------------------------

@contextlib.contextmanager
def file_lock(path, *, timeout: float = 600.0, poll: float = 0.2) -> Iterator[None]:
    """An exclusive lock file ``<path>.lock`` (created with O_EXCL, removed on exit), so runs that
    read-modify-write one record (results/e023_q1b.json) never overwrite each other. A lock
    older than the timeout is reported, never broken silently."""
    import os
    import time
    from pathlib import Path

    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):     # Windows: a lock being deleted
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"{lock} is held for more than {timeout} s: another run is "
                                   "writing the record (remove the lock only if no run is "
                                   "alive)") from None
            time.sleep(poll)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def update_json(path, update: Callable[[dict], dict | None], *, default: Mapping | None = None,
                json_default: Callable = str) -> dict:
    """Read ``path`` (``default`` when absent), apply ``update`` (in place or returning the new
    record) and write it atomically, all under ``file_lock``; returns the written record."""
    from pathlib import Path

    p = Path(path)
    with file_lock(p):
        rec = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else \
            json.loads(json.dumps(dict(default or {})))
        new = update(rec)
        rec = rec if new is None else new
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(rec, indent=1, default=json_default) + "\n",
                       encoding="utf-8", newline="\n")
        tmp.replace(p)
    return rec


# --- input processing -------------------------------------------------------------------------

def input_ln(pair: str) -> bool:
    """LayerNorm on the input iff SigLIP is in the pair (C's rule, diagnostic P7)."""
    if pair not in ENCODER_PAIRS:
        raise KeyError(f"unknown encoder pair '{pair}'")
    return "siglip" in ENCODER_PAIRS[pair]


def input_layer(dim: int, enabled: bool) -> nn.Module:
    """nn.LayerNorm(dim) when enabled, else nn.Identity (no random draws either way)."""
    return nn.LayerNorm(int(dim)) if enabled else nn.Identity()


@dataclass(frozen=True)
class InputPipeline:
    """Validation / test processing of concatenated snippet rows: per encoder (``dims``),
    optional l2 (derive_n.l2_rows), optional scale map (float64), float32, concatenated."""

    dims: tuple[int, ...]
    l2: bool = False
    scales: tuple[ScaleMap, ...] | None = None

    def __post_init__(self) -> None:
        if self.scales is not None and len(self.scales) != len(self.dims):
            raise ValueError(f"{len(self.scales)} scale maps for {len(self.dims)} encoders")

    def apply(self, x) -> torch.Tensor:
        from wvad.extract.derive_n import l2_rows

        a = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if a.ndim != 2 or a.shape[1] != sum(self.dims):
            raise ValueError(f"rows of shape {a.shape} for encoder dims {self.dims}")
        parts, start = [], 0
        for i, d in enumerate(self.dims):
            p = np.asarray(a[:, start:start + d], dtype=np.float32)
            start += d
            if self.l2:
                p = l2_rows(p)
            p = self.scales[i].apply(p) if self.scales is not None else p.astype(np.float32)
            parts.append(torch.from_numpy(np.ascontiguousarray(p)))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def to_json(self) -> dict:
        return {"dims": list(self.dims), "l2": self.l2, "scale_match": self.scales is not None}


# --- test-sequence handling (N_chunk) ---------------------------------------------------------

def check_n_chunk(n_chunk: int, multiple: int = 1) -> int:
    n = int(n_chunk)
    if n < 1 or n & (n - 1):
        raise ValueError(f"N_chunk must be a positive power of two, got {n_chunk}")
    if n % int(multiple):
        raise ValueError(f"N_chunk {n} is not a multiple of the method's piece {multiple}")
    return n


def chunk_bounds(n: int, n_chunk: int) -> tuple[tuple[int, int], ...]:
    """[a, b) pieces of a video of n snippets: the whole video when n <= N_chunk, else
    non-overlapping N_chunk pieces (the last shorter). The same function for validation and
    test (gate (q6))."""
    if n < 1:
        raise ValueError(f"a video has at least one snippet, got {n}")
    c = check_n_chunk(n_chunk)
    if n <= c:
        return ((0, int(n)),)
    return tuple((a, min(a + c, int(n))) for a in range(0, int(n), c))


def chunk_targets(total_frames: Sequence[int], n_chunk: int) -> tuple[int, ...]:
    """Positions of the videos cut into pieces: n_v = ceil(T / 16) > N_chunk (from the frame
    counts only, e.g. the access (0) record)."""
    c = check_n_chunk(n_chunk)
    return tuple(i for i, t in enumerate(total_frames) if grid.n_snippets(int(t)) > c)


def score_video(method: BaselineMethod, model: nn.Module, x: torch.Tensor, *, unit: int,
                n_chunk: int, device: torch.device, dtype: torch.dtype = torch.float32
                ) -> np.ndarray:
    """(n,) float32 snippet logits of one processed video (n, D): its pieces (chunk_bounds)
    through method.sequence_logits, no grad, autocast off. The caller sets eval mode."""
    check_n_chunk(n_chunk, method.sequence_multiple(unit))
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for a, b in chunk_bounds(int(x.shape[0]), n_chunk):
            z = method.sequence_logits(model, x[a:b].to(device=device, dtype=dtype), unit=unit)
            z = z.reshape(-1)
            if z.shape[0] != b - a:
                raise MethodError(f"{method.name}: {z.shape[0]} logits for {b - a} snippets")
            out.append(z.detach().float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# --- readouts and selection -------------------------------------------------------------------

def window_logits(z: np.ndarray, total_frames: int) -> np.ndarray:
    """(32,) W-readout window logits u_k (no offset) of one video's snippet logits."""
    return readout(np.asarray(z, dtype=np.float32), int(total_frames), None, "W").logits


def video_scores(z_list: Sequence[np.ndarray], totals: Sequence[int]) -> dict[str, np.ndarray]:
    """(n,) float64 video scores of the two readouts: sigma = max fp32 sigma(u_k), logit = max
    u_k (as E022 C's selection, c(v) = 0)."""
    sig, lg = [], []
    for z, t in zip(z_list, totals, strict=True):
        u = window_logits(z, t)
        sig.append(float(sigmoid32(u).max()))
        lg.append(float(u.max()))
    return {"sigma": np.asarray(sig, dtype=np.float64), "logit": np.asarray(lg, dtype=np.float64)}


def v_aucs(scores: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """(E,) full-validation video AUC of each row of (E, n) video scores (rank formula)."""
    return rank_auc(np.atleast_2d(np.asarray(scores, dtype=np.float64)),
                    np.asarray(labels, dtype=np.int64))


def select_point(aucs: Sequence[float]) -> tuple[int, bool]:
    """0-based index of the first maximum (strict >, earliest on ties) and whether it was tied;
    a non-finite V-AUC makes the selection fail (numerical failure)."""
    a = np.asarray(aucs, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("selection needs finite V-AUCs at every evaluation point")
    best = np.flatnonzero(a == a.max())
    return int(best[0]), len(best) > 1


def saturated(z_list: Sequence[np.ndarray], totals: Sequence[int]) -> int:
    """Videos whose sigma video score is exactly 1.0f (descriptive 8)."""
    return int((video_scores(z_list, totals)["sigma"] == 1.0).sum())


def eval_steps(total: int, points: int = EVAL_POINTS) -> tuple[int, ...]:
    """Steps after which the chain is evaluated: floor(k S / points + 1/2), k = 1..points
    (round half up), distinct and ending at S."""
    if total < points:
        raise ValueError(f"{total} steps cannot hold {points} evaluation points")
    out = tuple((2 * k * int(total) + points) // (2 * points) for k in range(1, points + 1))
    if len(set(out)) != points or out[-1] != total or out[0] < 1:
        raise ValueError(f"evaluation steps {out} are not distinct")
    return out


# --- random state ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    """random, numpy, torch (CPU and every CUDA device), as wvad.train.loop.seed_everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def isolated_rng(device: torch.device) -> Iterator[None]:
    """Generators untouched by the block: torch CPU (and the CUDA device) by fork_rng, numpy's
    global RandomState and Python's random saved and restored."""
    np_state, py_state = np.random.get_state(), random.getstate()
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def deterministic(enabled: bool) -> dict:
    """torch.use_deterministic_algorithms + cudnn flags ((q1b) mode); CUDA needs
    CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment (checked)."""
    import os

    if enabled and torch.cuda.is_available() and \
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic mode on CUDA needs CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(bool(enabled))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(enabled)
    return {"deterministic": bool(enabled),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def finite(x: float) -> bool:
    return math.isfinite(float(x))
