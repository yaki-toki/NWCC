"""Machine-specific locations of feature caches and annotations.

Large inputs are kept out of the repository. ``configs/paths.yaml`` (git-ignored; start
from ``configs/paths.example.yaml``) maps logical names to directories::

    feature_root: /data/features
    features:
      ucf_xcb: features_xclip_base      # relative -> feature_root / entry
    annotations:
      uca: /data/annotations/uca         # absolute -> used as is

``WVAD_PATHS`` overrides the file location.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"
ENV_VAR = "WVAD_PATHS"


class PathsConfigError(ValueError):
    """The paths file is missing, malformed, or lacks a requested entry."""


@dataclass(frozen=True)
class Paths:
    feature_root: Path
    features: Mapping[str, Path] = field(default_factory=dict)
    annotations: Mapping[str, Path] = field(default_factory=dict)

    def feature_dir(self, name: str) -> Path:
        return _lookup(self.features, name, "features")

    def annotation_dir(self, name: str) -> Path:
        return _lookup(self.annotations, name, "annotations")


def _lookup(table: Mapping[str, Path], name: str, section: str) -> Path:
    if name not in table:
        known = ", ".join(sorted(table)) or "none"
        raise PathsConfigError(f"no '{name}' under '{section}' (known: {known})")
    return table[name]


def _resolve(root: Path, entries: object, section: str) -> dict[str, Path]:
    if entries is None:
        return {}
    if not isinstance(entries, Mapping):
        raise PathsConfigError(f"'{section}' must be a mapping of name -> path")
    out = {}
    for name, value in entries.items():
        if not isinstance(value, str) or not value:
            raise PathsConfigError(f"'{section}.{name}' must be a non-empty string")
        p = Path(value)
        out[str(name)] = p if p.is_absolute() else root / p
    return out


def parse_paths(raw: Mapping[str, object]) -> Paths:
    """Build :class:`Paths` from an already-loaded mapping (no filesystem access)."""
    root_value = raw.get("feature_root")
    if not isinstance(root_value, str) or not root_value:
        raise PathsConfigError("'feature_root' must be a non-empty string")
    root = Path(root_value)
    return Paths(feature_root=root,
                 features=_resolve(root, raw.get("features"), "features"),
                 annotations=_resolve(root, raw.get("annotations"), "annotations"))


def load_paths(path: str | os.PathLike[str] | None = None) -> Paths:
    """Load the paths file: explicit argument, else ``$WVAD_PATHS``, else the default."""
    source = Path(path or os.environ.get(ENV_VAR) or DEFAULT_PATHS_FILE)
    if not source.is_file():
        raise PathsConfigError(
            f"paths file not found: {source} (copy configs/paths.example.yaml to "
            f"configs/paths.yaml or set {ENV_VAR})")
    with source.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise PathsConfigError(f"{source}: top level must be a mapping")
    return parse_paths(raw)
