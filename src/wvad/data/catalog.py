"""Names of the datasets, encoder pairs, feature variants and split files used in this project.

Feature variants (E021): ``legacy`` is the earlier 32-clip cache of each encoder, found through
``paths.yaml`` (the default everywhere). The other variants are 32-unit caches derived from the
dense 16-frame snippets (``wvad.extract.derive``), in the same ``mil/<split>/data.pt`` format,
inside the encoder's dense folder ``<feature_root>/<dataset>_<encoder>_d16/<variant>/``:

    d16c    per window, the snippet supporting the window centre frame (all encoders)
    d16r    the earlier position policy imitated (UCF: one uniform draw per encoder and window;
            XD: the snippet supporting the window start)
    d16m    per window, the mean of its snippets (scale-matched to d16c)
    d16m2   per window, the mean of m = 2 evenly picked snippets (scale-matched)
    d16r1, d16r2   further UCF draws of d16r, for scoring only (E021 R10)
"""

from __future__ import annotations

from pathlib import Path

from wvad.paths import REPO_ROOT, Paths

DATASETS = ("ucf", "xdv")            # the E020-E023 benchmarks (their gates loop over this pair)
NEW_DATASETS = ("ubn",)              # E024 stage 2: UBnormal (dense caches only, no legacy cache)
ALL_DATASETS = (*DATASETS, *NEW_DATASETS)
ENCODERS = ("xcb", "xcl", "siglip")
ENCODER_PAIRS = {"xcb_xcl": ("xcb", "xcl"), "xcb_siglip": ("xcb", "siglip"),
                 # single encoders (E019 pre-check), used through the same "pair" argument
                 "xcb": ("xcb",), "xcl": ("xcl",), "siglip": ("siglip",)}
SPLIT_FILES = {"ucf": "data/splits/ucf_val_split_15.json",
               "xdv": "data/splits/xdv_val_split_15.json",
               "ubn": "data/splits/ubn_val_official.json"}
SEGMENTS_PER_VIDEO = 32
LEGACY = "legacy"
FEATURE_VARIANTS = (LEGACY, "d16c", "d16r", "d16m", "d16m2")    # what training --features takes
SCORING_VARIANTS = {"d16r1": ("ucf",), "d16r2": ("ucf",)}      # scoring only, UCF only
DENSE_SUFFIX = "d16"


def _check(dataset: str, pair: str | None = None) -> None:
    if dataset not in ALL_DATASETS:
        raise KeyError(f"unknown dataset '{dataset}' (known: {', '.join(ALL_DATASETS)})")
    if pair is not None and pair not in ENCODER_PAIRS:
        raise KeyError(f"unknown encoder pair '{pair}' (known: {', '.join(ENCODER_PAIRS)})")


def check_variant(dataset: str, variant: str) -> None:
    """A training variant, or a scoring-only variant of a dataset that has it."""
    _check(dataset)
    if variant in FEATURE_VARIANTS:
        return
    if variant in SCORING_VARIANTS:
        if dataset not in SCORING_VARIANTS[variant]:
            raise KeyError(f"feature variant '{variant}' exists for "
                           f"{', '.join(SCORING_VARIANTS[variant])} only, not '{dataset}'")
        return
    known = ", ".join((*FEATURE_VARIANTS, *SCORING_VARIANTS))
    raise KeyError(f"unknown feature variant '{variant}' (known: {known})")


def feature_names(dataset: str, pair: str, variant: str = LEGACY) -> tuple[str, ...]:
    """Feature names of an encoder pair, in fusion order. ``legacy``: the logical names (keys of
    ``paths.yaml``), unchanged; a dense variant: ``<dataset>_<encoder>_<variant>``."""
    _check(dataset, pair)
    check_variant(dataset, variant)
    if variant == LEGACY and dataset in NEW_DATASETS:
        raise KeyError(f"'{dataset}' has no legacy 32-clip cache (dense variants only)")
    suffix = "" if variant == LEGACY else f"_{variant}"
    return tuple(f"{dataset}_{enc}{suffix}" for enc in ENCODER_PAIRS[pair])


def dense_name(dataset: str, encoder: str) -> str:
    """Folder name of an encoder's dense snippet cache and derived caches (under feature_root)."""
    _check(dataset)
    if encoder not in ENCODERS:
        raise KeyError(f"unknown encoder '{encoder}' (known: {', '.join(ENCODERS)})")
    return f"{dataset}_{encoder}_{DENSE_SUFFIX}"


def dense_dir(paths: Paths, dataset: str, encoder: str) -> Path:
    return paths.feature_root / dense_name(dataset, encoder)


def feature_dirs(paths: Paths, dataset: str, pair: str,
                 variant: str = LEGACY) -> tuple[Path, ...]:
    """Cache directories of an encoder pair (``legacy``: paths.yaml entries, as before)."""
    if variant == LEGACY:
        return tuple(paths.feature_dir(name) for name in feature_names(dataset, pair))
    feature_names(dataset, pair, variant)                     # validates dataset/pair/variant
    return tuple(dense_dir(paths, dataset, enc) / variant for enc in ENCODER_PAIRS[pair])


def split_file(dataset: str) -> Path:
    _check(dataset)
    return REPO_ROOT / SPLIT_FILES[dataset]
