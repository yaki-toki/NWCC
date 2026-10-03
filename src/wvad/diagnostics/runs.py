"""Access to the recorded E003 runs (``python -m wvad.train ... --record``).

Layout: <root>/<dataset>_<pair>_s<seed>/{stage1,base,full}/ with history.json,
selection.json, ckpt_*.pt and the record_*.npz files written by --record.
Only development seeds are accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wvad.diagnostics.common import DEV_SEEDS, check_dev_seed

PAIRS = ("xcb_xcl", "xcb_siglip")
CELLS = ("base", "full")


def run_dir(root: Path, dataset: str, pair: str, seed: int, cell: str) -> Path:
    check_dev_seed(seed)
    return Path(root) / f"{dataset}_{pair}_s{seed}" / cell


def selection(d: Path) -> dict:
    return json.loads((d / "selection.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ValRecord:
    epochs: np.ndarray          # (E,)
    scores: np.ndarray          # (E, n_val, 32)
    val_indices: np.ndarray     # (n_val,) training-cache indices, in score-row order

    def at(self, epoch: int) -> np.ndarray:
        return self.scores[int(np.flatnonzero(self.epochs == epoch)[0])]


def val_record(d: Path) -> ValRecord:
    z = np.load(d / "record_val_scores.npz")
    return ValRecord(z["epochs"], z["scores"], z["val_indices"])


@dataclass(frozen=True)
class PseudoRecord:
    scores: np.ndarray          # (4, n_train, 32): rounds 1-3 and the final model (round 4)
    labels: np.ndarray          # (3, n_train, 32) pseudo-labels of rounds 1-3 (NaN = none)
    train_indices: np.ndarray


def pseudo_record(d: Path) -> PseudoRecord:
    z, f = np.load(d / "record_pseudo.npz"), np.load(d / "record_pseudo_final.npz")
    if not np.array_equal(z["train_indices"], f["train_indices"]):
        raise ValueError(f"{d}: train indices differ between pseudo-label records")
    return PseudoRecord(np.concatenate([z["scores"], f["scores"][None]]), z["labels"],
                        z["train_indices"])


def rows_of(order: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Positions in ``order`` (record row order) of the training-cache ``indices``."""
    pos = {int(i): k for k, i in enumerate(order)}
    return np.array([pos[int(i)] for i in indices])


__all__ = ["CELLS", "DEV_SEEDS", "PAIRS", "PseudoRecord", "ValRecord", "pseudo_record",
           "rows_of", "run_dir", "selection", "val_record"]
