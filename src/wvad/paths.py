"""Machine-specific locations of feature caches and annotations.

Large inputs are kept out of the repository. ``configs/paths.yaml`` (git-ignored; start
from ``configs/paths.example.yaml``) maps logical names to directories::

    feature_root: ~/data/features        # ~ and $VARS are expanded; a relative root is
                                         # taken relative to the config file's folder
    features:
      ucf_xcb: features_xclip_base       # relative -> feature_root / entry
    annotations:
      uca: /data/annotations/uca         # anchored -> used as is

"Anchored" means absolute or starting at a root: on Windows ``/data`` has no drive and
resolves against the current drive when used. Write Windows paths with forward slashes
or in single quotes (inside double quotes YAML treats backslashes as escapes).

File lookup order: the explicit argument, then ``$WVAD_PATHS``, then
``configs/paths.yaml`` of a source checkout (the default is unavailable when the package
is installed without its repository).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_SOURCE_CHECKOUT = (REPO_ROOT / "pyproject.toml").is_file()
DEFAULT_PATHS_FILE: Path | None = (REPO_ROOT / "configs" / "paths.yaml"
                                   if IN_SOURCE_CHECKOUT else None)
ENV_VAR = "WVAD_PATHS"
SECTIONS = ("feature_root", "features", "annotations")


class PathsConfigError(ValueError):
    """The paths file is missing, malformed, or lacks a requested entry."""


def _empty() -> Mapping[str, Path]:
    return MappingProxyType({})


@dataclass(frozen=True)
class Paths:
    feature_root: Path
    features: Mapping[str, Path] = field(default_factory=_empty)
    annotations: Mapping[str, Path] = field(default_factory=_empty)

    def feature_dir(self, name: str) -> Path:
        return _lookup(self.features, name, "features")

    def annotation_dir(self, name: str) -> Path:
        return _lookup(self.annotations, name, "annotations")


def _lookup(table: Mapping[str, Path], name: str, section: str) -> Path:
    if name not in table:
        known = ", ".join(sorted(table)) or "none"
        raise PathsConfigError(f"no '{name}' under '{section}' (known: {known})")
    return table[name]


def _expand(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _anchored(path: Path) -> bool:
    return path.is_absolute() or bool(path.root)


def _resolve(root: Path, entries: object, section: str) -> Mapping[str, Path]:
    if entries is None:
        return _empty()
    if not isinstance(entries, Mapping):
        raise PathsConfigError(f"'{section}' must be a mapping of name -> path")
    out = {}
    for name, value in entries.items():
        if not isinstance(name, str) or not name:
            raise PathsConfigError(f"'{section}' has an empty or non-string name: {name!r}")
        if not isinstance(value, str) or not value:
            raise PathsConfigError(f"'{section}.{name}' must be a non-empty string")
        p = _expand(value)
        out[name] = p if _anchored(p) else root / p
    return MappingProxyType(out)


def parse_paths(raw: Mapping[str, object], base_dir: Path | None = None) -> Paths:
    """Build :class:`Paths` from a loaded mapping. ``base_dir`` anchors a relative root."""
    unknown = sorted(str(k) for k in raw if k not in SECTIONS)
    if unknown:
        raise PathsConfigError(f"unknown top-level keys {unknown} (allowed: {list(SECTIONS)})")
    root_value = raw.get("feature_root")
    if not isinstance(root_value, str) or not root_value:
        raise PathsConfigError("'feature_root' must be a non-empty string")
    root = _expand(root_value)
    if not _anchored(root):
        if base_dir is None:
            raise PathsConfigError(f"relative feature_root '{root_value}' needs the config "
                                   "file's folder; use an absolute path")
        root = Path(base_dir) / root
    return Paths(feature_root=root,
                 features=_resolve(root, raw.get("features"), "features"),
                 annotations=_resolve(root, raw.get("annotations"), "annotations"))


def _source(path: str | os.PathLike[str] | None) -> tuple[Path, str]:
    if path is not None:
        return Path(path), "the given path"
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env), f"${ENV_VAR}"
    if DEFAULT_PATHS_FILE is None:
        raise PathsConfigError(f"not running from a source checkout: pass a path or set "
                               f"{ENV_VAR}")
    return DEFAULT_PATHS_FILE, "the default location"


def load_paths(path: str | os.PathLike[str] | None = None) -> Paths:
    """Load the paths file: explicit argument, else ``$WVAD_PATHS``, else the default."""
    source, origin = _source(path)
    if not source.is_file():
        hint = ("copy configs/paths.example.yaml to configs/paths.yaml"
                if origin == "the default location" else f"check {origin}")
        raise PathsConfigError(f"paths file not found: {source} ({hint})")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PathsConfigError(f"{source}: not a readable UTF-8 YAML file ({exc})") from exc
    if not isinstance(raw, Mapping):
        raise PathsConfigError(f"{source}: top level must be a mapping")
    return parse_paths(raw, base_dir=source.resolve().parent)
