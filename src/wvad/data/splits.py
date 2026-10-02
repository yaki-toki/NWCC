"""Validation splits over the training videos (test data is never involved).

A split file lists disjoint ``train_indices`` and ``val_indices`` into the order of the
training feature cache, plus the video label of every index. A split is identified by
the sha256 of its sorted index lists (``split_sha256``), not by the file bytes, so a
regenerated file with the same indices keeps its identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


class SplitError(ValueError):
    """A split file is inconsistent, or does not match the data it is applied to."""


@dataclass(frozen=True)
class ValSplit:
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    labels: tuple[int, ...]       # video label per training-cache index
    sha256: str                   # identity of the split (index content)
    meta: Mapping[str, object]

    @property
    def n_total(self) -> int:
        return len(self.labels)


def split_sha256(train_indices: Sequence[int], val_indices: Sequence[int]) -> str:
    """sha256 of the compact, key-sorted JSON of the sorted index lists."""
    payload = {"train_indices": sorted(int(i) for i in train_indices),
               "val_indices": sorted(int(i) for i in val_indices)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_split(raw: Mapping[str, object], source: str = "<split>") -> ValSplit:
    """Validate a loaded split mapping and compute its identity."""
    try:
        train = sorted(int(i) for i in raw["train_indices"])  # type: ignore[union-attr]
        val = sorted(int(i) for i in raw["val_indices"])  # type: ignore[union-attr]
        labels = tuple(int(x) for x in raw["labels"])  # type: ignore[union-attr]
        meta = dict(raw.get("meta") or {})  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitError(f"{source}: malformed split ({exc})") from exc
    n_total = int(meta.get("n_total", len(labels)))
    if set(train) & set(val):
        raise SplitError(f"{source}: train and validation indices overlap")
    if sorted(train + val) != list(range(n_total)):
        raise SplitError(f"{source}: train + validation indices do not cover 0..{n_total - 1}")
    if len(labels) != n_total:
        raise SplitError(f"{source}: {len(labels)} labels for {n_total} videos")
    sha = split_sha256(train, val)
    recorded = meta.get("split_sha256")
    if recorded is not None and recorded != sha:
        raise SplitError(f"{source}: recorded split_sha256 {recorded} != computed {sha}")
    return ValSplit(tuple(train), tuple(val), labels, sha, MappingProxyType(meta))


def load_split(path: str | os.PathLike[str]) -> ValSplit:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, Mapping):
        raise SplitError(f"{path}: top level must be a mapping")
    return parse_split(raw, source=str(path))


def check_labels(split: ValSplit, labels: Sequence[int], source: str) -> None:
    """The data the split is applied to must list the same videos with the same labels."""
    got = tuple(int(x) for x in labels)
    if got != split.labels:
        # lengths may differ here; count mismatches over the common prefix
        n_bad = sum(a != b for a, b in zip(got, split.labels, strict=False))
        raise SplitError(f"{source}: video labels do not match the split ({len(got)} vs "
                         f"{split.n_total} videos, {n_bad} mismatched positions)")
