"""E023 training samplers (PLAN.md "같은 특징 규칙과 학습 단위", gate (q1c)): which training videos
form each optimizer step's batch, and which rows of each video it uses. New code (add-only rule).

A sampler draws only from the global generators the official code draws from (torch's default
CPU generator, numpy's global RandomState, Python's ``random``), so the same generator state gives
the same indices (unit-tested), and gate (q1c) compares the official sampler and the port from
one state, bit for bit. The training loop (``wvad.baselines.train``) seeds the generators once,
draws one step at a time in step order, and isolates every evaluation from them
(``common.isolated_rng``), so evaluation never moves a sampler.

    Pools(normal, abnormal)     positions (0-based, into the training-part FeatureSet, its order)
                                of the normal and the anomalous training videos
    StepDraw(videos, labels, rows)
                                one step: the positions in batch order, their 0 / 1 labels and,
                                per video, an int64 row selection into its training input (None:
                                all rows, the fixed per-video input of a derived cache)
    Sampler                     base class: Sampler(pools, params); draw(step) for step 1, 2, ...
                                in order; steps_per_epoch() for epoch-based schedules (or None)
    register(name, factory, owner=...)   the hook of the method modules: a method module that
                                needs its own sampler calls ``samplers.register("<name>",
                                MySampler, owner=__name__)`` at import (never assigns into this
                                module); registering the same factory again is a no-op, another
                                factory under a used name is an error
    create(name, pools, params) a registered sampler; registered() the names

Built-in sampler "window_mean" (PairedLoaderSampler): the generic window-mean sampler of the
official paired-loader loops (RTFM main.py, UR-DMU *_main.py: one shuffled DataLoader per class,
drop_last, re-created when exhausted; VadCLIP *_train.py: both re-created every epoch of
min(len) steps). Each step takes one batch from each class loader; every video's input is its
fixed window-mean rows (the derived cache: rows None). The class loaders are real
``torch.utils.data.DataLoader`` objects over position lists (shuffle=True, drop_last=True,
num_workers=0, generator None), created and advanced in the official order, so the global torch
generator is consumed exactly as by the official loaders (the iterator's base seed when it is
created, RandomSampler's seed at its first batch). Parameters:
    batch_normal, batch_abnormal    videos per class and step (official batch sizes)
    reset       "per_pool" (each loader re-created at (step - 1) % len(loader) == 0, normal first)
                or "per_epoch" (both re-created every min(len) steps, normal first)
    order       "normal_first" (RTFM, UR-DMU, VadCLIP: torch.cat((normal, abnormal))) or
                "abnormal_first"
    eager       False (official num_workers 0). True for an official loader with workers: their
                prefetch takes the first index batch when the iterator is created, so the
                RandomSampler seed is drawn then (the port draws it at creation too)
    crops       1 (default). k > 1 for an official list holding k crop rows per video, each a
                separate item (VadCLIP's 10-crop CLIP lists): each class loader runs over every
                position repeated k times in a row (the official list order: a video's crops
                are consecutive), so one epoch is one pass over the k-crop list - the official
                number of optimizer steps - and a crop row draws its video's single-crop input
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SamplerError(ValueError):
    """A sampler is unknown, misconfigured or driven out of order."""


@dataclass(frozen=True)
class Pools:
    normal: tuple[int, ...]
    abnormal: tuple[int, ...]

    @classmethod
    def from_labels(cls, labels: Sequence[int]) -> Pools:
        lab = [int(y) for y in labels]
        if any(y not in (0, 1) for y in lab):
            raise SamplerError("video labels must be 0 or 1")
        return cls(tuple(i for i, y in enumerate(lab) if y == 0),
                   tuple(i for i, y in enumerate(lab) if y == 1))

    @property
    def size(self) -> int:
        return len(self.normal) + len(self.abnormal)


@dataclass(frozen=True)
class StepDraw:
    videos: tuple[int, ...]
    labels: tuple[int, ...]
    rows: tuple[np.ndarray | None, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.videos) != len(self.labels):
            raise SamplerError("a draw needs one label per video")
        if self.rows is not None and len(self.rows) != len(self.videos):
            raise SamplerError("a draw's row selections must match its videos")

    @property
    def n_normal(self) -> int:
        return sum(1 for y in self.labels if y == 0)

    def key(self) -> tuple:
        """A comparable form (videos, labels, row selections as tuples)."""
        rows = None if self.rows is None else tuple(
            None if r is None else tuple(int(x) for x in np.asarray(r)) for r in self.rows)
        return (self.videos, self.labels, rows)


class Sampler(abc.ABC):
    """A training sampler (module docstring). Subclasses set ``name`` and implement ``draw``."""

    name: ClassVar[str] = ""

    def __init__(self, pools: Pools, params: Mapping[str, Any] | None = None) -> None:
        self.pools = pools
        self.params = dict(params or {})
        self._last_step = 0

    def _advance(self, step: int) -> None:
        if int(step) != self._last_step + 1:
            raise SamplerError(f"{self.name}: step {step} drawn after step {self._last_step} "
                               "(steps are drawn once each, in order)")
        self._last_step = int(step)

    @abc.abstractmethod
    def draw(self, step: int) -> StepDraw:
        """The batch of optimizer step ``step`` (1-based, called once per step, in order)."""

    def steps_per_epoch(self) -> int | None:
        """Steps of one official epoch (epoch-based schedules); None when undefined."""
        return None


_REGISTRY: dict[str, tuple[Callable[..., Sampler], str]] = {}


def register(name: str, factory: Callable[..., Sampler], *, owner: str) -> None:
    """Register ``factory(pools, params) -> Sampler`` under ``name`` (the method modules' hook)."""
    if not name or not isinstance(name, str):
        raise SamplerError("a sampler needs a non-empty name")
    old = _REGISTRY.get(name)
    if old is not None:
        same = (getattr(old[0], "__qualname__", None) == getattr(factory, "__qualname__", None)
                and getattr(old[0], "__module__", None) == getattr(factory, "__module__", None)
                and old[1] == owner)
        if not same:
            raise SamplerError(f"sampler {name!r} is registered by {old[1]} already")
    _REGISTRY[name] = (factory, owner)


def registered() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def owner_of(name: str) -> str:
    if name not in _REGISTRY:
        raise SamplerError(f"unknown sampler {name!r} (registered: {registered()})")
    return _REGISTRY[name][1]


def create(name: str, pools: Pools, params: Mapping[str, Any] | None = None) -> Sampler:
    if name not in _REGISTRY:
        raise SamplerError(f"unknown sampler {name!r} (registered: {registered()}); a method "
                           "module registers its sampler when it is imported")
    sampler = _REGISTRY[name][0](pools, dict(params or {}))
    if not isinstance(sampler, Sampler):
        raise SamplerError(f"sampler factory {name!r} did not return a Sampler")
    return sampler


# --- the generic window-mean sampler ----------------------------------------------------------

class _Positions(Dataset):
    def __init__(self, positions: Sequence[int]) -> None:
        self.positions = tuple(int(p) for p in positions)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> int:
        return self.positions[i]


def _collate(items: list[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in items)


def crop_rows(positions: Sequence[int], crops: int) -> tuple[int, ...]:
    """Every position repeated ``crops`` times in a row (the rows of an official k-crop list,
    a video's crops consecutive)."""
    k = int(crops)
    if k < 1:
        raise SamplerError(f"crops must be >= 1, got {crops}")
    return tuple(int(p) for p in positions for _ in range(k))


class IndexLoader:
    """A shuffled, drop_last DataLoader over pool positions (num_workers 0, global generator):
    the official class loader's index stream. ``eager``: the first batch is taken when the
    iterator is created (an official loader with workers prefetches it then)."""

    def __init__(self, positions: Sequence[int], batch_size: int, eager: bool = False) -> None:
        if batch_size < 1:
            raise SamplerError(f"batch size must be >= 1, got {batch_size}")
        if len(positions) < batch_size:
            raise SamplerError(f"{len(positions)} videos for batches of {batch_size} "
                               "(drop_last leaves none)")
        self.loader = DataLoader(_Positions(positions), batch_size=int(batch_size), shuffle=True,
                                 drop_last=True, num_workers=0, collate_fn=_collate)
        self.eager = bool(eager)

    def __len__(self) -> int:
        return len(self.loader)

    def iterate(self) -> Iterator[tuple[int, ...]]:
        it = iter(self.loader)
        if not self.eager:
            return it
        first = next(it)

        def chained():
            yield first
            yield from it
        return chained()


class PairedLoaderSampler(Sampler):
    """The built-in "window_mean" sampler (module docstring)."""

    name = "window_mean"
    RESETS = ("per_pool", "per_epoch")
    ORDERS = ("normal_first", "abnormal_first")

    def __init__(self, pools: Pools, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(pools, params)
        p = self.params
        unknown = set(p) - {"batch_normal", "batch_abnormal", "reset", "order", "eager", "crops"}
        if unknown:
            raise SamplerError(f"window_mean: unknown parameters {sorted(unknown)}")
        try:
            bn, ba = int(p["batch_normal"]), int(p["batch_abnormal"])
        except KeyError as exc:
            raise SamplerError("window_mean needs batch_normal and batch_abnormal") from exc
        self.reset = p.get("reset", "per_pool")
        self.order = p.get("order", "normal_first")
        if self.reset not in self.RESETS or self.order not in self.ORDERS:
            raise SamplerError(f"window_mean: reset in {self.RESETS}, order in {self.ORDERS}")
        eager = bool(p.get("eager", False))
        self.crops = int(p.get("crops", 1))
        self.normal_loader = IndexLoader(crop_rows(pools.normal, self.crops), bn, eager)
        self.abnormal_loader = IndexLoader(crop_rows(pools.abnormal, self.crops), ba, eager)
        self._n: Iterator | None = None
        self._a: Iterator | None = None

    def steps_per_epoch(self) -> int | None:
        if self.reset == "per_epoch":
            return min(len(self.normal_loader), len(self.abnormal_loader))
        return None

    def draw(self, step: int) -> StepDraw:
        self._advance(step)
        s = int(step) - 1
        if self.reset == "per_pool":
            if s % len(self.normal_loader) == 0:
                self._n = self.normal_loader.iterate()
            if s % len(self.abnormal_loader) == 0:
                self._a = self.abnormal_loader.iterate()
        elif s % self.steps_per_epoch() == 0:
            self._n = self.normal_loader.iterate()
            self._a = self.abnormal_loader.iterate()
        normal, abnormal = next(self._n), next(self._a)
        if self.order == "normal_first":
            return StepDraw(normal + abnormal, (0,) * len(normal) + (1,) * len(abnormal))
        return StepDraw(abnormal + normal, (1,) * len(abnormal) + (0,) * len(normal))


register(PairedLoaderSampler.name, PairedLoaderSampler, owner=__name__)


def draws(sampler: Sampler, steps: int) -> list[StepDraw]:
    """Steps 1..``steps`` of a fresh sampler (tests, gate (q1c) records)."""
    return [sampler.draw(s) for s in range(1, int(steps) + 1)]


def generator_state() -> tuple:
    """The global generators' state (torch CPU, numpy, random) as one comparable tuple."""
    import random

    return (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())


def set_generator_state(state: tuple) -> None:
    import random

    torch.set_rng_state(state[0])
    np.random.set_state(state[1])
    random.setstate(state[2])
