"""Access to the recorded E003 runs (``python -m wvad.train ... --record``).

Layout: <root>/<dataset>_<pair>_s<seed>/{stage1,base,full}/ with history.json,
selection.json, ckpt_*.pt and the record_*.npz files written by --record.
Only development seeds are accepted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wvad.diagnostics.common import DEV_SEEDS, E003_RESULTS, check_dev_seed, write_csv

PAIRS = ("xcb_xcl", "xcb_siglip")
CELLS = ("base", "full")
CELL_WEIGHTS = {"base": {"dense": 0.0, "pseudo": 0.0, "neg_pseudo": 0.0},
                "full": {"dense": 0.05, "pseudo": 1.0, "neg_pseudo": 0.3}}
STAGE2_EPOCHS = 30


def check_run(d: Path, dataset: str, pair: str, seed: int, cell: str) -> None:
    """The run's own configuration (ckpt_last.pt) must match its folder name and cell."""
    cfg = torch.load(d / "ckpt_last.pt", map_location="cpu", weights_only=True)["config"]
    check_dev_seed(cfg["seed"])
    if (cfg["dataset"], cfg["pair"], int(cfg["seed"])) != (dataset, pair, int(seed)):
        raise ValueError(f"{d}: config {cfg['dataset']}/{cfg['pair']}/s{cfg['seed']} does not "
                         f"match the folder")
    w = cfg["weights"]
    if any(w[k] != v for k, v in CELL_WEIGHTS[cell].items()):
        raise ValueError(f"{d}: loss weights {w} are not those of cell '{cell}'")


def run_dir(root: Path, dataset: str, pair: str, seed: int, cell: str) -> Path:
    check_dev_seed(seed)
    d = Path(root) / f"{dataset}_{pair}_s{seed}" / cell
    check_run(d, dataset, pair, seed, cell)
    return d


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
    rec = ValRecord(z["epochs"], z["scores"], z["val_indices"])
    if not np.array_equal(rec.epochs, np.arange(0, STAGE2_EPOCHS + 1)):
        raise ValueError(f"{d}: stage-2 record should hold epochs 0-{STAGE2_EPOCHS}")
    if not np.isfinite(rec.scores).all():
        raise ValueError(f"{d}: non-finite validation scores")
    return rec


@dataclass(frozen=True)
class PseudoRecord:
    scores: np.ndarray          # (4, n_train, 32): rounds 1-3 and the final model (round 4)
    labels: np.ndarray          # (3, n_train, 32) pseudo-labels of rounds 1-3 (NaN = none)
    train_indices: np.ndarray


def pseudo_record(d: Path) -> PseudoRecord:
    z, f = np.load(d / "record_pseudo.npz"), np.load(d / "record_pseudo_final.npz")
    if not np.array_equal(z["train_indices"], f["train_indices"]):
        raise ValueError(f"{d}: train indices differ between pseudo-label records")
    scores = np.concatenate([z["scores"], f["scores"][None]])
    if scores.shape[0] != 4 or z["labels"].shape[0] != 3 or not np.isfinite(scores).all():
        raise ValueError(f"{d}: pseudo-label record must hold rounds 1-3 and a finite round 4")
    return PseudoRecord(scores, z["labels"], z["train_indices"])


def rows_of(order: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Positions in ``order`` (record row order) of the training-cache ``indices``."""
    pos = {int(i): k for k, i in enumerate(order)}
    return np.array([pos[int(i)] for i in indices])


def run_status(root: Path) -> list[dict]:
    """One row per (dataset, pair, seed, cell): completion and selection (runs_status.csv)."""
    rows = []
    for dataset in ("ucf", "xdv"):
        for pair in PAIRS:
            for seed in DEV_SEEDS:
                for cell in CELLS:
                    d = Path(root) / f"{dataset}_{pair}_s{seed}" / cell
                    files = {f: (d / f).is_file() for f in (
                        "selection.json", "record_val_scores.npz", "record_pseudo.npz",
                        "record_pseudo_final.npz")}
                    sel = selection(d) if files["selection.json"] else {}
                    rows.append({"dataset": dataset, "pair": pair, "seed": seed, "cell": cell,
                                 **{k.split(".")[0]: v for k, v in files.items()},
                                 "best_epoch": sel.get("best_epoch"),
                                 "best_val_video_auc": sel.get("best_val_video_auc"),
                                 "complete": all(files.values())})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E003 run status (runs_status.csv)")
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    rows = run_status(args.runs)
    write_csv(E003_RESULTS / "runs_status.csv", rows)
    print(f"{sum(r['complete'] for r in rows)}/{len(rows)} complete")


__all__ = ["CELLS", "DEV_SEEDS", "PAIRS", "PseudoRecord", "ValRecord", "pseudo_record",
           "rows_of", "run_dir", "selection", "val_record"]

if __name__ == "__main__":
    main()
