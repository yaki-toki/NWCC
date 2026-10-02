"""Names of the datasets, encoder pairs and split files used in this project."""

from __future__ import annotations

from pathlib import Path

from wvad.paths import REPO_ROOT, Paths

DATASETS = ("ucf", "xdv")
ENCODER_PAIRS = {"xcb_xcl": ("xcb", "xcl"), "xcb_siglip": ("xcb", "siglip")}
SPLIT_FILES = {"ucf": "data/splits/ucf_val_split_15.json",
               "xdv": "data/splits/xdv_val_split_15.json"}
SEGMENTS_PER_VIDEO = 32


def _check(dataset: str, pair: str | None = None) -> None:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset '{dataset}' (known: {', '.join(DATASETS)})")
    if pair is not None and pair not in ENCODER_PAIRS:
        raise KeyError(f"unknown encoder pair '{pair}' (known: {', '.join(ENCODER_PAIRS)})")


def feature_names(dataset: str, pair: str) -> tuple[str, ...]:
    """Logical feature names (keys of ``paths.yaml``) of an encoder pair, in fusion order."""
    _check(dataset, pair)
    return tuple(f"{dataset}_{enc}" for enc in ENCODER_PAIRS[pair])


def feature_dirs(paths: Paths, dataset: str, pair: str) -> tuple[Path, ...]:
    return tuple(paths.feature_dir(name) for name in feature_names(dataset, pair))


def split_file(dataset: str) -> Path:
    _check(dataset)
    return REPO_ROOT / SPLIT_FILES[dataset]
