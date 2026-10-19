"""Official comparison-method code loaded by path for the E023 equivalence checks ((q1), (q1b),
(q1c), (q1e); PLAN.md "관문", (3a), 결정 0002). New code (add-only rule).

The official repositories are cloned OUTSIDE this repository at the commits of
results/e023_licences.json; their root is the environment variable WVAD_OFFICIAL_ROOT with one
folder per repository (REPOS). Nothing is copied: a file is executed from its clone as a module
under a private name (``wvad_official.<method>.<stem>``), with the clone's folders put first on
sys.path for its own sibling imports (``from model import *``, ``import utils``) only while a
``session`` is open; when it closes, sys.path is restored and every module imported from the
clone (and the private names) is dropped from sys.modules, so two repositories' "utils" or
"model" never meet. Modifications of official code stay outside the repository (only their
diff sha256 is recorded, PLAN.md (q1b)).

NEVER imported by the training or scoring code (wvad.baselines.common / train / samplers /
recipes / the method modules, wvad.extract.derive_n, wvad.eval.decompose, the E023 diagnostics):
a unit test checks the import graph and the sources.

    root() / repo_dir(method)       the clone folders (refused inside this repository)
    clone_commit(method)            git HEAD of the clone; check_commit(method) against the
                                    licence record
    with session(method, "src") as s:
        model = s.load("model.py")  a module object of the official file
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType

from wvad.paths import REPO_ROOT

ENV = "WVAD_OFFICIAL_ROOT"
REPOS = {"mil": "AnomalyDetectionCVPR2018", "rtfm": "RTFM", "urdmu": "UR-DMU",
         "vadclip_v": "VadCLIP", "mgfn": "MGFN", "clip_tsa": "CLIP-TSA"}
LICENCE_KEY = {"mil": "mil", "rtfm": "rtfm", "urdmu": "urdmu", "vadclip_v": "vcv",
               "mgfn": "mgfn", "clip_tsa": "clip_tsa"}
PRIVATE = "wvad_official"
LICENCES_FILE = (REPO_ROOT / "experiments" / "E023_localisation_baselines" / "results"
                 / "e023_licences.json")


class OfficialError(RuntimeError):
    """The official clones are not configured or not at the recorded commit."""


def root() -> Path:
    raw = os.environ.get(ENV)
    if not raw:
        raise OfficialError(f"set {ENV} to the folder holding the official clones")
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        raise OfficialError(f"{ENV}={raw} is not a folder")
    if p == REPO_ROOT.resolve() or REPO_ROOT.resolve() in p.parents:
        raise OfficialError(f"{ENV}={raw} is inside the repository: clone outside it")
    return p


def repo_dir(method: str) -> Path:
    if method not in REPOS:
        raise OfficialError(f"unknown method {method!r} (known: {sorted(REPOS)})")
    d = root() / REPOS[method]
    if not d.is_dir():
        raise OfficialError(f"{d} missing: clone {REPOS[method]} under {ENV}")
    return d


def clone_commit(method: str) -> str | None:
    res = subprocess.run(["git", "-c", f"safe.directory={repo_dir(method).as_posix()}",
                          "rev-parse", "HEAD"], cwd=repo_dir(method), capture_output=True,
                         text=True, check=False)
    return res.stdout.strip() or None


def recorded_commit(method: str, licences: str | os.PathLike[str] = LICENCES_FILE) -> str:
    rec = json.loads(Path(licences).read_text(encoding="utf-8"))
    return str(rec["repos"][LICENCE_KEY[method]]["commit"])


def check_commit(method: str, licences: str | os.PathLike[str] = LICENCES_FILE) -> dict:
    """The clone is at the commit of the licence record (else OfficialError)."""
    got, want = clone_commit(method), recorded_commit(method, licences)
    if got != want:
        raise OfficialError(f"{repo_dir(method)} is at {got}, the licence record says {want}")
    return {"method": method, "repo": REPOS[method], "commit": got}


class Session:
    """Loads official files of one clone (``session``)."""

    def __init__(self, method: str, folders: Sequence[Path]) -> None:
        self.method = method
        self.base = repo_dir(method)
        self.folders = tuple(folders)
        self.loaded: dict[str, ModuleType] = {}

    def load(self, relpath: str, name: str | None = None) -> ModuleType:
        """Execute <clone>/<relpath> as a module named wvad_official.<method>.<name or stem>."""
        path = (self.base / relpath).resolve()
        if self.base.resolve() not in path.parents:
            raise OfficialError(f"{relpath} is outside the clone {self.base}")
        mod_name = f"{PRIVATE}.{self.method}.{name or path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise OfficialError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        self.loaded[relpath] = module
        return module


def _from_clone(module: ModuleType, base: Path) -> bool:
    f = getattr(module, "__file__", None)
    if not f:
        return False
    try:
        return base in Path(f).resolve().parents
    except (OSError, ValueError):
        return False


@contextlib.contextmanager
def session(method: str, *subdirs: str) -> Iterator[Session]:
    """The clone's folders (``subdirs`` relative to it; the clone itself when none) first on
    sys.path while open; afterwards sys.path is restored and every module loaded from the clone
    is removed from sys.modules."""
    base = repo_dir(method).resolve()
    folders = [base / s for s in subdirs] if subdirs else [base]
    for f in folders:
        if not f.is_dir():
            raise OfficialError(f"{f} missing in the clone")
    saved_path = list(sys.path)
    before = set(sys.modules)
    sys.path[:0] = [str(f) for f in folders]
    s = Session(method, folders)
    try:
        yield s
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - before:
            mod = sys.modules.get(name)
            if name.startswith(PRIVATE) or (mod is not None and _from_clone(mod, base)):
                sys.modules.pop(name, None)
