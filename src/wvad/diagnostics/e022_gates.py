"""E022 gates (PLAN.md "코드 동일 규칙", "관문", "시험 추출 관문"): checks of the code rule, the
test extraction and the training inputs, plus the independent recomputation (w2). Fixed server
code (PLAN.md "코드 규칙" 1); its module-level imports stay inside S_extract, so the test-split
extraction entry may use it without loading another existing module (unit test).

    code rule   add-only rule (``git diff --no-renames --name-status 7b43cca5 <commit> -- src/``
                lists only "A"), the identity sets S_train (17), S_score, S_extract, S_analysis
                byte-identical to 7b43cca5 (``git diff --quiet``), the import graph of
                ``python -m wvad.train`` (= S_train, no E022 module) and of the test extraction
                entry (existing modules inside S_extract), no run-time reassignment of a frozen
                module attribute (static AST check, module snapshot; review N2)
    tests/      tree sha256 (sha256 of "<path>  <sha256>" lines sorted by path, __pycache__ and
                *.pyc left out) and the run summary of a pytest JUnit XML (review N13)
    (t)         test manifests (restricted read: names, categories, total_frames, meta) and the
                earlier test caches' sha256 against E020's expected values
    (y0)-(y7)   test extraction: code identity, environment, repeat reproduction, headers,
                frame counts, completeness and layout, derived test caches, video access
    (r0)        the E020 m10 chains (sha256 of ckpt_best_val.pt, ckpt_last.pt, selection.json
                against the pre-freeze record; selected epoch and V-AUC against e020_runs.csv)
    inputs      the 42 E021 training-split files and the 6 scale-match arrays against the
                pre-freeze record results/e022_train_inputs.json (written by
                wvad.data.dense_test; review N3): the subsets of (f), (p), (y6) and of the test
                access guard 8 (all), recomputed here
    (f)         per training batch: code identity (S_train, S_score), environment = E021's
                (driver only: recorded, "(드라이버 변경)"), training inputs, the tests/ record
    (w2)        c(v), W and G video scores and the per-epoch full-validation V-AUC recomputed from
                the stored snippet logits; this module imports neither wvad.eval.dense_readout nor
                wvad.diagnostics.e022_score, and the (w2) functions do not use wvad.extract.grid
                either (the window assignment is written again from the E021 PLAN definition);
                the selected epochs equal and every V-AUC within 1e-9

Status codes: "pass_bit", "pass_tolerance", "pass", "driver_only", "fail".
Category labels of a test manifest (restricted read): UCF "Normal" and XD "A" are normal (0),
every other category anomalous (1) (the training manifests' rule; gate (t) checks the counts).
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePath

import numpy as np
import torch

from wvad.data.catalog import DATASETS, ENCODERS, dense_name
from wvad.data.dense import (
    FEATURES_FILE,
    INDEX_FILE,
    META_FILE,
    PHASES,
    PROGRESS_FILE,
    ScaleMap,
    load_scale_map,
    read_index,
    read_json,
    store_dir,
    variant_dir,
)
from wvad.data.features import load_cache
from wvad.diagnostics.common import code_version
from wvad.extract import grid
from wvad.extract.dense import names_sha256, read_progress
from wvad.extract.encoders import ENV_KEYS, same_environment
from wvad.extract.encoders import spec as encoder_spec
from wvad.paths import REPO_ROOT
from wvad.prep.manifests import sha256_file

BASE_COMMIT = "7b43cca5cdf91de0346e216d24d71e49485a3f6b"        # E021 production and training
PAIRS = ("xcb_xcl", "xcb_siglip")
# the confirmation seeds (wvad.diagnostics.e018.CONFIRMATION_SEEDS, outside S_extract; unit test)
SEEDS = (42, 123, 777, 1000, 2025, 7, 13, 99, 555, 31337)
E020_RESULTS = REPO_ROOT / "experiments" / "E020_confirmation_f1" / "results"
E021_RESULTS = REPO_ROOT / "experiments" / "E021_dense_snippet_pilot" / "results"
SEED = 20261012                       # (y6) sample of recomputed d16m videos
PASS_BIT, PASS_TOL, PASS, DRIVER_ONLY, FAIL = ("pass_bit", "pass_tolerance", "pass",
                                               "driver_only", "fail")

# --- identity sets (PLAN.md "동일 집합") ------------------------------------------------------

S_TRAIN = ("src/wvad/__init__.py", "src/wvad/paths.py",
           "src/wvad/train/__init__.py", "src/wvad/train/__main__.py", "src/wvad/train/loop.py",
           "src/wvad/train/pseudo.py", "src/wvad/train/stages.py",
           "src/wvad/models/__init__.py", "src/wvad/models/head.py",
           "src/wvad/losses/__init__.py", "src/wvad/losses/mil.py",
           "src/wvad/data/__init__.py", "src/wvad/data/catalog.py", "src/wvad/data/features.py",
           "src/wvad/data/splits.py", "src/wvad/eval/__init__.py", "src/wvad/eval/metrics.py")
S_SCORE = (*S_TRAIN, "src/wvad/data/dense.py", "src/wvad/data/manifests.py",
           "src/wvad/eval/scoring.py", "src/wvad/eval/dense.py", "src/wvad/extract/__init__.py",
           "src/wvad/extract/grid.py", "src/wvad/prep/__init__.py", "src/wvad/prep/manifests.py",
           "src/wvad/diagnostics/__init__.py", "src/wvad/diagnostics/common.py")
S_EXTRACT = (*(f"src/wvad/extract/{n}.py" for n in ("__init__", "__main__", "video", "preprocess",
                                                    "encoders", "grid", "dense", "derive", "gate",
                                                    "legacy")),
             *(f"src/wvad/data/{n}.py" for n in ("__init__", "catalog", "dense", "features",
                                                 "manifests", "splits")),
             "src/wvad/__init__.py", "src/wvad/paths.py", "src/wvad/prep/__init__.py",
             "src/wvad/prep/manifests.py", "src/wvad/diagnostics/__init__.py",
             "src/wvad/diagnostics/common.py")
S_ANALYSIS = ("src/wvad/eval/metrics.py", "src/wvad/eval/dense_readout.py",
              "src/wvad/eval/crossfit.py", "src/wvad/diagnostics/e020_stats.py")
IDENTITY_SETS = {"S_train": S_TRAIN, "S_score": S_SCORE, "S_extract": S_EXTRACT,
                 "S_analysis": S_ANALYSIS}
TRAIN_ENTRY = "wvad.train.__main__"
TEST_EXTRACT_ENTRY = "wvad.extract.testsplit"


def module_of(path: str) -> str:
    """'src/wvad/a/b.py' -> 'wvad.a.b' ('__init__.py' -> its package)."""
    parts = list(PurePath(path).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


S_TRAIN_MODULES = tuple(sorted(module_of(p) for p in S_TRAIN))
S_EXTRACT_MODULES = tuple(sorted({module_of(p) for p in S_EXTRACT}))
FROZEN_MODULES = tuple(sorted({module_of(p) for p in (*S_EXTRACT, *S_SCORE, *S_ANALYSIS)}))


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --- git: add-only rule and identity sets -----------------------------------------------------

def git(*args: str, repo: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={Path(repo).as_posix()}", *args],
                          cwd=repo, capture_output=True, text=True, check=False)


def _untracked(paths: Sequence[str], repo: Path) -> list[str] | None:
    res = git("ls-files", "--others", "--exclude-standard", "--", *paths, repo=repo)
    return None if res.returncode != 0 else [x for x in res.stdout.splitlines() if x.strip()]


def add_only(commit: str | None, *, base: str = BASE_COMMIT, repo: Path = REPO_ROOT) -> dict:
    """The add-only rule: every line of ``git diff --no-renames --name-status <base> <commit> --
    src/`` is 'A'. ``commit`` None checks the working tree (tracked changes; untracked files
    under src/ are new files, listed as added)."""
    res = git("diff", "--no-renames", "--name-status", base, *([commit] if commit else []),
              "--", "src/", repo=repo)
    if res.returncode != 0:
        return {"base": base, "commit": commit, "ok": False, "error": res.stderr.strip()}
    lines = [ln.split("\t") for ln in res.stdout.splitlines() if ln.strip()]
    rows = [{"status": x[0], "path": x[-1]} for x in lines]
    if commit is None:
        new = _untracked(["src/"], repo)
        if new is None:
            return {"base": base, "commit": None, "ok": False, "error": "git ls-files failed"}
        rows += [{"status": "A", "path": p, "untracked": True} for p in new]
    bad = [r for r in rows if r["status"] != "A"]
    return {"base": base, "commit": commit or "working tree", "lines": rows,
            "added": sorted(r["path"] for r in rows if r["status"] == "A"),
            "not_added": bad, "ok": not bad}


def identical(paths: Sequence[str], commit: str | None, *, base: str = BASE_COMMIT,
              repo: Path = REPO_ROOT) -> bool | None:
    """``git diff --quiet <base> [<commit>] -- <paths>``: True / False; None when git cannot
    tell (an unknown commit)."""
    res = git("diff", "--quiet", base, *([commit] if commit else []), "--", *paths, repo=repo)
    return {0: True, 1: False}.get(res.returncode)


def code_identity(commit: str | None, sets: Sequence[str] = tuple(IDENTITY_SETS), *,
                  base: str = BASE_COMMIT, repo: Path = REPO_ROOT) -> dict:
    """The add-only rule and the named identity sets at ``commit`` (None: working tree)."""
    rule = add_only(commit, base=base, repo=repo)
    same = {s: identical(IDENTITY_SETS[s], commit, base=base, repo=repo) for s in sets}
    return {"commit": commit, "base": base, "add_only": rule, "sets": same,
            "ok": bool(rule["ok"] and all(v is True for v in same.values()))}


def new_modules(commit: str | None = None, *, base: str = BASE_COMMIT,
                repo: Path = REPO_ROOT) -> list[str]:
    """Modules of the Python files added under src/ since ``base`` (E022's new modules)."""
    rule = add_only(commit, base=base, repo=repo)
    return sorted(module_of(p) for p in rule.get("added", []) if p.endswith(".py"))


# --- import graph -----------------------------------------------------------------------------

_LOADED = ("import importlib, json, sys; importlib.import_module({entry!r}); "
           "print(json.dumps(sorted(m for m in sys.modules if m == 'wvad' "
           "or m.startswith('wvad.'))))")


def loaded_modules(entry: str, *, src: Path = REPO_ROOT / "src", python: str | None = None,
                   timeout: float = 600) -> list[str]:
    """The wvad.* modules a fresh interpreter holds after importing ``entry``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src), *filter(None, [env.get("PYTHONPATH")])])
    res = subprocess.run([python or sys.executable, "-c", _LOADED.format(entry=entry)],
                         capture_output=True, text=True, env=env, timeout=timeout, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"importing {entry} failed: {res.stderr[-2000:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def import_graph(entry: str, allowed: Iterable[str], new: Iterable[str], *, exact: bool,
                 forbid_new: bool, loaded: Sequence[str] | None = None, **kw) -> dict:
    """The import graph of ``entry``: its existing (not ``new``) wvad modules inside ``allowed``
    (``exact``: equal to it), and with ``forbid_new`` no new module."""
    mods = list(loaded) if loaded is not None else loaded_modules(entry, **kw)
    new_set, allowed_set = set(new), set(allowed)
    existing = [m for m in mods if m not in new_set]
    out = {"entry": entry, "loaded": mods, "new_loaded": [m for m in mods if m in new_set],
           "outside": sorted(set(existing) - allowed_set),
           "missing": sorted(allowed_set - set(existing)) if exact else []}
    out["ok"] = bool(not out["outside"] and not out["missing"]
                     and not (forbid_new and out["new_loaded"]))
    return out


def train_import_graph(**kw) -> dict:
    """``python -m wvad.train`` loads exactly S_train (17 modules) and no E022 module."""
    return import_graph(TRAIN_ENTRY, S_TRAIN_MODULES, new_modules(), exact=True,
                        forbid_new=True, **kw)


def extract_import_graph(entry: str = TEST_EXTRACT_ENTRY, **kw) -> dict:
    """The test extraction entry loads no existing module outside S_extract."""
    return import_graph(entry, S_EXTRACT_MODULES, new_modules(), exact=False, forbid_new=False,
                        **kw)


# --- no run-time reassignment (review N2) -----------------------------------------------------

MUTATORS = frozenset({"update", "add", "append", "extend", "insert", "pop", "popitem", "clear",
                      "remove", "discard", "setdefault", "sort", "reverse", "__setitem__",
                      "__delitem__", "__setattr__", "__delattr__", "__iadd__", "__ior__"})


def _root(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute | ast.Subscript | ast.Call):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _flat(targets: Iterable[ast.AST]) -> list[ast.AST]:
    out = []
    for t in targets:
        if isinstance(t, ast.Tuple | ast.List):
            out += _flat(t.elts)
        elif isinstance(t, ast.Starred):
            out += _flat([t.value])
        else:
            out.append(t)
    return out


def static_reassignments(source: str, filename: str = "<module>",
                         new: Iterable[str] = ()) -> list[dict]:
    """AST check of one E022 module: an assignment, augmented assignment or ``del`` whose target
    is an attribute or item of a name imported from a frozen wvad module (any wvad module but
    the ``new`` ones; a relative import counts as frozen), ``setattr`` / ``delattr`` on such a
    name, ``importlib.reload``, and in-place mutators (``.update``, ``.add``, ``.append``, ...
    ``__setitem__``) called on such a name. Aliases made by plain assignment are not followed."""
    tree = ast.parse(source, filename)
    skip = set(new)
    frozen, reload_names, importlib_names = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.name == "wvad" or a.name.startswith("wvad.")) and a.name not in skip:
                    frozen.add(a.asname or a.name.split(".")[0])
                if a.name == "importlib":
                    importlib_names.add(a.asname or "importlib")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0 or ((mod == "wvad" or mod.startswith("wvad.")) and mod not in skip):
                frozen |= {a.asname or a.name for a in node.names
                           if f"{mod}.{a.name}" not in skip}
            if mod == "importlib":
                reload_names |= {a.asname or a.name for a in node.names if a.name == "reload"}
    found = []

    def hit(node: ast.AST, kind: str) -> None:
        found.append({"file": filename, "line": node.lineno, "kind": kind,
                      "code": ast.unparse(node)[:120]})
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Delete):
            targets = node.targets if isinstance(node, ast.Assign | ast.Delete) else [node.target]
            for t in _flat(targets):
                if isinstance(t, ast.Attribute | ast.Subscript) and _root(t) in frozen:
                    hit(node, "del" if isinstance(node, ast.Delete) else "assignment")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in ("setattr", "delattr") and node.args \
                    and _root(node.args[0]) in frozen:
                hit(node, f.id)
            elif (isinstance(f, ast.Attribute) and f.attr == "reload"
                  and _root(f.value) in importlib_names) or \
                    (isinstance(f, ast.Name) and f.id in reload_names):
                hit(node, "reload")
            elif isinstance(f, ast.Attribute) and f.attr in MUTATORS and _root(f.value) in frozen:
                hit(node, f"mutator .{f.attr}")
    return found


def static_check(paths: Iterable[Path], new: Iterable[str] = ()) -> dict:
    """``static_reassignments`` over files (E022's new modules; ``new``: their module names)."""
    new = list(new)
    rows = {str(p): static_reassignments(Path(p).read_text(encoding="utf-8"), str(p), new)
            for p in paths}
    return {"files": sorted(rows), "findings": [f for r in rows.values() for f in r],
            "ok": not any(rows.values())}


_CONSTANT = (str, bytes, int, float, complex, bool, type(None), tuple, list, dict, set,
             frozenset, PurePath)


def module_snapshot(modules: Iterable[str] = FROZEN_MODULES) -> dict[str, dict]:
    """Per frozen module: attribute name -> (object id, repr of a constant-like value); every
    module is imported first, and a package's submodule attributes (bound by the import system)
    are left out."""
    mods = {m: importlib.import_module(m) for m in modules}
    out = {}
    for m, mod in mods.items():
        out[m] = {k: (id(v), repr(v) if isinstance(v, _CONSTANT) else None)
                  for k, v in vars(mod).items()
                  if not (k.startswith("__") and k.endswith("__"))
                  and not (isinstance(v, types.ModuleType) and v.__name__ == f"{m}.{k}")}
    return out


def snapshot_changes(before: Mapping[str, Mapping], after: Mapping[str, Mapping]) -> list[str]:
    """Attributes added, removed, rebound or (constants) changed in value."""
    out = []
    for m in sorted(set(before) | set(after)):
        a, b = before.get(m, {}), after.get(m, {})
        out += [f"{m}.{k}: {'added' if k not in a else 'removed' if k not in b else 'changed'}"
                for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return out


# --- tests/ tree and run summary (review N13) -----------------------------------------------

def _skipped_path(rel: PurePath) -> bool:
    return any(p in ("__pycache__", ".pytest_cache") for p in rel.parts) or \
        rel.suffix in (".pyc", ".pyo")


def tests_tree(root: Path = REPO_ROOT / "tests") -> dict:
    """sha256 of the text '<path>  <sha256>\\n' per file, sorted by path (paths relative to the
    repository root, e.g. 'tests/conftest.py'; __pycache__, .pytest_cache and *.pyc left out)."""
    root = Path(root)
    files = {}
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root.parent)
        if f.is_file() and not _skipped_path(rel):
            files[rel.as_posix()] = sha256_file(f)
    text = "".join(f"{p}  {files[p]}\n" for p in sorted(files))
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "n_files": len(files),
            "files": files}


def junit_run(path: Path) -> dict:
    """Counts of a pytest JUnit XML (``--junitxml``): tests, failures, errors, skipped, passed,
    the skipped test ids, its timestamp and sha256."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    counts = {k: sum(int(x.get(k, 0)) for x in suites)
              for k in ("tests", "failures", "errors", "skipped")}
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    skipped = [f"{c.get('classname')}::{c.get('name')}" for x in suites
               for c in x.iter("testcase") if c.find("skipped") is not None]
    return {**counts, "passed": passed, "skipped_tests": skipped,
            "timestamp": suites[0].get("timestamp") if suites else None,
            "junit_sha256": sha256_file(path)}


def tests_record(junit: Path, root: Path = REPO_ROOT / "tests") -> dict:
    """The tests/ tree and the run summary with the commit and time (pre-freeze record, (y0),
    (f))."""
    return {"tree": tests_tree(root), "run": junit_run(junit), "code": code_version(),
            "utc": utc_now()}


def compare_tests(now: Mapping, before: Mapping, reason: str | None = None) -> dict:
    """The run has tests and no failure, error or skip; the tree equals the pre-freeze one, or
    the differences are listed and a reason given (PLAN.md (y0))."""
    run = now["run"]
    run_ok = run["tests"] > 0 and run["failures"] == run["errors"] == run["skipped"] == 0
    a, b = before["tree"].get("files", {}), now["tree"].get("files", {})
    same = now["tree"]["sha256"] == before["tree"]["sha256"]
    diff = {"added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)),
            "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k])}
    return {"tree_sha256": now["tree"]["sha256"], "tree_sha256_before": before["tree"]["sha256"],
            "same_tree": same, "differences": None if same else diff, "reason": reason,
            "run": dict(run), "run_ok": run_ok, "ok": bool(run_ok and (same or reason))}


# --- training inputs (review N3) --------------------------------------------------------------

TRAIN_INPUT_RELS = ("dense16/train/features.npy", "dense16/train/index.npz",
                    "dense16/train/meta.json", "d16m/mil/train/data.pt", "d16m/meta.json",
                    "d16c/mil/train/data.pt", "d16c/meta.json")
SCALE_REL = "d16m/meta.json#scale"          # the scale-match arrays of d16m/meta.json
DENSE_FOLDERS = tuple(dense_name(d, e) for d in DATASETS for e in ENCODERS)
INPUT_SUBSETS = {          # PLAN.md (f), (p), (y6) and the test-access guard 8
    "f": ("d16m/mil/train/data.pt", "d16c/mil/train/data.pt"),
    "p": ("dense16/train/features.npy", "dense16/train/index.npz", "dense16/train/meta.json",
          "d16m/meta.json", SCALE_REL),
    "y6": ("d16m/meta.json", SCALE_REL),
    "guard8": (*TRAIN_INPUT_RELS, SCALE_REL)}


def input_keys(rels: Sequence[str] = (*TRAIN_INPUT_RELS, SCALE_REL),
               folders: Sequence[str] = DENSE_FOLDERS) -> list[str]:
    """'<folder>/<rel>' keys of the record (6 folders x 7 files + 6 scale arrays = 48)."""
    return [f"{f}/{r}" for f in folders for r in rels]


def scale_array_sha256(meta: Mapping | Path) -> str:
    """sha256 of mu_c, sigma_c, mu_v, sigma_v of the d16m scale map (``ScaleMap.from_json`` of
    meta.json "scale"), in this order, as float64 little-endian bytes."""
    m = read_json(meta) if isinstance(meta, str | os.PathLike) else meta
    s = ScaleMap.from_json(m["scale"])
    data = b"".join(np.asarray(getattr(s, k), dtype="<f8").tobytes()
                    for k in ("mu_c", "sigma_c", "mu_v", "sigma_v"))
    return hashlib.sha256(data).hexdigest()


def train_input_hashes(feature_root: Path, rels: Sequence[str] = (*TRAIN_INPUT_RELS, SCALE_REL),
                       folders: Sequence[str] = DENSE_FOLDERS) -> dict[str, str]:
    """Current sha256 of the training inputs under ``feature_root`` (keys of ``input_keys``); an
    unreadable file or scale map gets 'unreadable: <error>' (never equal to a record)."""
    out = {}
    for f in folders:
        for r in rels:
            path = Path(feature_root) / f / r.split("#")[0]
            try:
                out[f"{f}/{r}"] = (scale_array_sha256(path) if r == SCALE_REL
                                   else sha256_file(path))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                out[f"{f}/{r}"] = f"unreadable: {exc!r}"
    return out


def recorded_hashes(record: Mapping) -> dict[str, str]:
    """The flat '<folder>/<rel>' -> sha256 map of the pre-freeze record: results/
    e022_train_inputs.json (its "train_inputs" section) or that section, as
    ``wvad.data.dense_test.train_input_record`` writes it ("files": the 42 files by
    '<folder>/<rel>', "scale_arrays": the six scale-map hashes by folder, here
    '<folder>/d16m/meta.json#scale'), or an already flat map."""
    rec = record.get("train_inputs", record)
    if "files" in rec or "scale_arrays" in rec:
        flat = dict(rec.get("files") or {})
        flat |= {f"{folder}/{SCALE_REL}": sha
                 for folder, sha in (rec.get("scale_arrays") or {}).items()}
    else:
        flat = rec
    return {str(k): str(v) for k, v in flat.items()}


def compare_hashes(recorded: Mapping[str, str], current: Mapping[str, str],
                   keys: Sequence[str]) -> dict:
    """Every key present in both maps with equal sha256."""
    missing = [k for k in keys if k not in recorded or k not in current]
    different = [k for k in keys if k in recorded and k in current and recorded[k] != current[k]]
    return {"checked": len(keys), "missing": missing, "different": different,
            "ok": bool(keys) and not missing and not different}


def inputs_gate(gate: str, recorded: Mapping, current: Mapping[str, str],
                folders: Sequence[str] = DENSE_FOLDERS) -> dict:
    """The training-input sha256 of gate (f), (p), (y6) or the guard 8 against the pre-freeze
    record (live files are never compared with each other)."""
    keys = input_keys(INPUT_SUBSETS[gate], folders)
    return {"gate": gate, **compare_hashes(recorded_hashes(recorded), current, keys)}


# --- (t): the test data -----------------------------------------------------------------------

NORMAL_CATEGORY = {"ucf": "Normal", "xdv": "A"}


def label_of(dataset: str, category: str) -> int:
    """Video label of a test-manifest category (restricted read)."""
    return 0 if str(category) == NORMAL_CATEGORY[dataset] else 1


def manifest_summary(dataset: str, names: Sequence[str], categories: Sequence[str],
                     total_frames: Sequence[int], meta: Mapping) -> dict:
    """Count, anomalous videos, frame sum and the source and annotation sha256 of a restricted
    test-manifest read."""
    if not len(names) == len(categories) == len(total_frames):
        raise ValueError("names, categories and total_frames differ in length")
    return {"n": len(names), "anomalous": sum(label_of(dataset, c) for c in categories),
            "frames": int(sum(int(t) for t in total_frames)),
            "source": dict(meta.get("source") or {}).get("sha256"),
            "annotation": dict(meta.get("annotation") or {}).get("sha256")}


def e020_expected() -> tuple[dict, dict]:
    """E020's expected test data and earlier test-cache sha256 (imported here only)."""
    from wvad.diagnostics.e020_gates import TEST_EXPECTED
    from wvad.diagnostics.e020_score import TEST_CACHE_SHA256

    return dict(TEST_EXPECTED), dict(TEST_CACHE_SHA256)


def gate_t(summaries: Mapping[str, Mapping], cache_sha256: Mapping[str, str], *,
           expected: Mapping[str, Mapping] | None = None,
           caches: Mapping[str, str] | None = None) -> dict:
    """Gate (t): each dataset's manifest summary (``manifest_summary``) and the six earlier test
    caches' sha256 ('<dataset>_<encoder>' -> sha256 of mil/test/data.pt, bytes only) equal
    E020's values."""
    if expected is None or caches is None:
        exp, cac = e020_expected()
        expected, caches = expected or exp, caches or cac
    out = {}
    for d in DATASETS:
        s = dict(summaries.get(d) or {})
        bad = [k for k, v in expected[d].items() if s.get(k) != v]
        keys = [f"{d}_{e}" for e in ENCODERS]
        cache_bad = [k for k in keys if cache_sha256.get(k) != caches[k]]
        out[d] = {"manifest": s, "manifest_mismatch": bad, "cache_mismatch": cache_bad,
                  "ok": not bad and not cache_bad}
    out["ok"] = all(out[d]["ok"] for d in DATASETS)
    return out


# --- (y0), (y1), (y2), (y3), (y4) -------------------------------------------------------------

def gate_y0(commit: str, tests_now: Mapping, tests_before: Mapping, *,
            reason: str | None = None, base: str = BASE_COMMIT, repo: Path = REPO_ROOT) -> dict:
    """Gate (y0): the add-only rule and S_extract identical at the deployed commit, the unit
    tests without failure or skip, and the tests/ tree of the pre-freeze record."""
    ident = (code_identity(commit, ("S_extract",), base=base, repo=repo) if commit
             else {"ok": False, "reason": "no deployed commit"})
    tests = compare_tests(tests_now, tests_before, reason)
    return {"code": ident, "tests": tests, "ok": bool(ident["ok"] and tests["ok"])}


def gate_y1(env: Mapping, want: Mapping | None = None) -> dict:
    """Gate (y1): the server extraction environment = E021 env_extract.json on ENV_KEYS, the pip
    freeze and each encoder's attention implementation; a different driver only is
    "driver_only" (recorded; (y2) decides)."""
    want = want if want is not None else _json(E021_RESULTS / "env_extract.json")
    diff = [k for k in ENV_KEYS if env.get(k) != want.get(k)]
    if "pip_freeze" in want and env.get("pip_freeze") != want.get("pip_freeze"):
        diff.append("pip_freeze")
    for e, w in (want.get("encoders") or {}).items():
        got = (env.get("encoders") or {}).get(e) or {}
        if got.get("attn_implementation") != w.get("attn_implementation"):
            diff.append(f"encoders.{e}.attn_implementation")
    status = PASS if not diff else DRIVER_ONLY if diff == ["driver"] else FAIL
    return {"differences": diff, "driver": [env.get("driver"), want.get("driver")],
            "same_environment": same_environment(dict(env), dict(want)), "status": status,
            "ok": status in (PASS, DRIVER_ONLY)}


Y2_L2, Y2_COS = 1e-3, 0.99999


def compare_rows(new: np.ndarray, stored: np.ndarray, l2: float = Y2_L2,
                 cos: float = Y2_COS) -> dict:
    """Gate (y2) for one (dataset, encoder): rows bit-equal ("pass_bit"), or every snippet with
    relative L2 <= 1e-3 and cosine >= 0.99999 ("pass_tolerance"), else "fail"."""
    a0, b0 = np.asarray(new), np.asarray(stored)
    if a0.shape != b0.shape:
        return {"rows": int(b0.shape[0]) if b0.ndim else 0, "status": FAIL,
                "reason": f"shapes {a0.shape} / {b0.shape}"}
    a, b = a0.astype(np.float64), b0.astype(np.float64)
    nb, na = np.linalg.norm(b, axis=1), np.linalg.norm(a, axis=1)
    err = np.linalg.norm(a - b, axis=1)
    rel = np.where(nb > 0, err / np.where(nb > 0, nb, 1.0), np.where(err > 0, np.inf, 0.0))
    den = na * nb
    cosine = np.where(den > 0, (a * b).sum(axis=1) / np.where(den > 0, den, 1.0),
                      np.where(err > 0, -np.inf, 1.0))
    bit = a0.dtype == b0.dtype and bool(np.array_equal(a0, b0))
    within = bool(np.isfinite(rel).all() and (rel <= l2).all() and (cosine >= cos).all())
    return {"rows": int(len(b)), "bit_equal": bit, "max_rel_l2": float(rel.max(initial=0.0)),
            "median_rel_l2": float(np.median(rel)) if rel.size else 0.0,
            "min_cos": float(cosine.min(initial=1.0)),
            "status": PASS_BIT if bit else PASS_TOL if within else FAIL}


def gate_y2(rows: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Gate (y2) over (dataset, encoder) keys -> (re-extracted rows, stored E021 rows): the
    overall status is the worst one."""
    per = {k: compare_rows(n, s) for k, (n, s) in rows.items()}
    st = [v["status"] for v in per.values()]
    status = FAIL if (not st or FAIL in st) else PASS_TOL if PASS_TOL in st else PASS_BIT
    return {"items": per, "status": status, "ok": status != FAIL}


Y3_DELTA = 16


def gate_y3(rows: Sequence[Mapping]) -> dict:
    """Gate (y3): rows (dataset, index, name, header_frames, manifest_frames) of every test
    video; every delta recorded, |header T - manifest T| <= 16 passes (no decoding)."""
    deltas = [{"dataset": r["dataset"], "index": int(r["index"]), "name": r["name"],
               "delta": int(r["header_frames"]) - int(r["manifest_frames"])} for r in rows]
    bad = [x for x in deltas if abs(x["delta"]) > Y3_DELTA]
    return {"videos": len(deltas), "deltas": deltas,
            "nonzero": sum(x["delta"] != 0 for x in deltas),
            "max_abs_delta": max((abs(x["delta"]) for x in deltas), default=0),
            "violations": bad, "status": FAIL if bad or not deltas else PASS,
            "ok": bool(deltas) and not bad}


Y4_REPLACED = 0.01


def gate_y4(index, names: Sequence[str], totals: Sequence[int],
            y3_passed: Mapping[int, bool]) -> dict:
    """Gate (y4) on the test store's index (``read_index``): T grab attempts per video and a
    replaced share of the used frames <= 1%; the frames in [header T, manifest T) are counted
    separately and, for videos that passed (y3), left out of the 1% rule (PLAN.md (y4))."""
    short, over, tail = [], [], []
    for v, t in enumerate(int(x) for x in totals):
        h = int(index.header_frames[v])
        if int(index.grab_attempts[v]) != t:
            short.append(v)
        used, rep = int(index.used_frames[v]), int(index.replaced[v])
        rep_tail = int(index.replaced_tail[v])
        if rep_tail:
            tail.append({"index": v, "name": names[v], "replaced_tail": rep_tail})
        if y3_passed.get(v, False) and 0 <= h < t:
            used -= int((grid.used_frames(t) >= h).sum())
            rep -= rep_tail
        if used <= 0 or rep / used > Y4_REPLACED:
            over.append({"index": v, "name": names[v], "replaced": rep, "used": used})
    ok = not short and not over and len(totals) == len(names) > 0
    return {"videos": len(totals), "grab_attempts_not_T": short, "over_1pct": over,
            "tail_replaced": tail, "status": PASS if ok else FAIL, "ok": ok}


# --- (y5): completeness and layout of one test store ------------------------------------------

def _finite_rows(path: Path, rows: int, chunk: int = 1 << 15) -> bool:
    arr = np.load(path, mmap_mode="r")
    return all(np.isfinite(np.asarray(arr[a:a + chunk], dtype=np.float32)).all()
               for a in range(0, rows, chunk))


def gate_y5(test_dir: Path, *, dataset: str, encoder: str, names: Sequence[str],
            totals: Sequence[int], labels: Sequence[int], commit: str, environment: Mapping,
            e021_before: Mapping[str, str], e021_after: Mapping[str, str],
            recorded: Mapping) -> dict:
    """Gate (y5) for one (dataset, encoder) test folder ``<feature_root>/<d>_<e>_d16_test``:
    videos and order (names sha256) = the test manifest, n_v = ceil(T / 16), rows = sum n_v,
    every video done, no NaN / infinity, the pinned model revision; the store only under
    dense16/train/; the meta.json runs, progress.jsonl rows and index.npz phases all "sample"
    (code 3), each run's commit mapping with split "test" and the deployed commit, its
    environment = (y1)'s; the E021 records unchanged before and after the extraction."""
    problems = []
    sd = store_dir(test_dir)
    want = {FEATURES_FILE, INDEX_FILE, META_FILE, PROGRESS_FILE}
    present = {f.name for f in sd.iterdir()} if sd.is_dir() else set()
    if not want <= present:
        problems.append(f"missing {sorted(want - present)} in {sd}")
        return {"folder": str(test_dir), "problems": problems, "status": FAIL, "ok": False}
    subdirs = sorted(p.name for p in (Path(test_dir) / "dense16").iterdir())
    if subdirs != ["train"]:
        problems.append(f"dense16/ holds {subdirs}, only train/ expected")
    meta = read_json(sd / META_FILE)
    t = np.asarray(totals, dtype=np.int64)
    n = np.array([-(-int(x) // grid.SNIPPET) for x in t], dtype=np.int64)
    checks = {"dataset": dataset, "encoder": encoder, "videos": len(names),
              "names_sha256": names_sha256(list(names)), "rows": int(n.sum()),
              "revision": encoder_spec(encoder).revision}
    problems += [f"meta {k} {meta.get(k)!r} != {v!r}" for k, v in checks.items()
                 if meta.get(k) != v]
    ix = read_index(test_dir)
    for col, val in (("n", n), ("total_frames", t), ("labels", np.asarray(labels))):
        if not np.array_equal(getattr(ix, col), val):
            problems.append(f"index {col} differs from the manifest")
    if not bool(np.all(ix.done)):
        problems.append(f"{int((~ix.done).sum())} videos not done")
    if not bool(np.all(ix.phase == PHASES["sample"])):
        problems.append("index phases other than sample (3)")
    arr = np.load(sd / FEATURES_FILE, mmap_mode="r")
    if arr.shape[0] != int(n.sum()):
        problems.append(f"features hold {arr.shape[0]} rows, sum n_v = {int(n.sum())}")
    elif not _finite_rows(sd / FEATURES_FILE, arr.shape[0]):
        problems.append("non-finite feature rows")
    runs = meta.get("runs") or []
    if not runs:
        problems.append("meta.json lists no extraction run")
    for r in runs:
        c = r.get("commit") or {}
        if r.get("phase") != "sample":
            problems.append(f"run {r.get('id')}: phase {r.get('phase')}")
        if c.get("split") != "test" or c.get("commit") != commit:
            problems.append(f"run {r.get('id')}: commit mapping {c}")
        if not same_environment(dict(r.get("environment") or {}), dict(environment)):
            problems.append(f"run {r.get('id')}: environment differs from (y1)")
        enc = r.get("encoder") or {}
        if enc.get("loaded_commit") not in (None, checks["revision"]):
            problems.append(f"run {r.get('id')}: loaded revision {enc.get('loaded_commit')}")
    progress = read_progress(test_dir)
    if set(progress) != set(range(len(names))):
        problems.append(f"progress.jsonl lists {len(progress)} of {len(names)} videos")
    if any(row.get("phase") != "sample" or row.get("commit") != commit
           for row in progress.values()):
        problems.append("progress rows with another phase or commit")
    keys = input_keys()
    rec = recorded_hashes(recorded)
    e021 = {"before": compare_hashes(rec, e021_before, keys),
            "after": compare_hashes(rec, e021_after, keys)}
    if not (e021["before"]["ok"] and e021["after"]["ok"]):
        problems.append("E021 training inputs differ from the pre-freeze record")
    return {"folder": str(test_dir), "videos": len(names), "rows": int(n.sum()),
            "runs": [r.get("id") for r in runs], "e021_records": e021, "problems": problems,
            "status": FAIL if problems else PASS, "ok": not problems}


# --- (y6): the derived test caches ------------------------------------------------------------

TEST_DERIVE_RECORD = "e022_test_derive.json"
Y6_TOL = 1e-6


def gate_y6(test_dir: Path, e021_dir: Path, *, dataset: str, names: Sequence[str],
            categories: Sequence[str], totals: Sequence[int], recorded: Mapping,
            n_random: int = 20, seed: int = SEED) -> dict:
    """Gate (y6) for one (dataset, encoder): ``<test_dir>/{d16c,d16m}/mil/test/data.pt`` only
    (no ``<variant>/meta.json``, the derive record present); labels (from the categories),
    total_frames and the order of the restricted manifest read; seg_labels placeholders (all
    -1); d16c rows = the dense rows of the window-centre snippets (values bit for bit); d16m =
    the float64 window means (independent window sets) under E021's d16m map, recomputed for
    ``n_random`` videos (|difference| <= 1e-6); the E021 d16m/meta.json and its scale arrays =
    the pre-freeze record (never re-fitted on test features)."""
    folder = Path(e021_dir).name
    meta_path = variant_dir(e021_dir, "d16m") / META_FILE
    live = {f"{folder}/d16m/meta.json": None, f"{folder}/{SCALE_REL}": None}
    try:
        live = {f"{folder}/d16m/meta.json": sha256_file(meta_path),
                f"{folder}/{SCALE_REL}": scale_array_sha256(meta_path)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        live = {k: f"unreadable: {exc!r}" for k in live}
    record = compare_hashes(recorded_hashes(recorded), live, list(live))
    problems = [] if record["ok"] else \
        ["E021 d16m/meta.json or its scale arrays differ from the record"]
    t = [int(x) for x in totals]
    ix = read_index(test_dir)
    out = {"folder": str(test_dir), "record": record, "sample": [], "d16m_max_abs_diff": None}
    caches = {}
    for v in ("d16c", "d16m"):
        caches[v], found = _test_cache(variant_dir(test_dir, v), v, names, t,
                                       [label_of(dataset, c) for c in categories])
        problems += found
    if len(ix) != len(t) or not np.array_equal(ix.total_frames, t) or problems:
        if len(ix) != len(t) or not np.array_equal(ix.total_frames, t):
            problems.append("the test store's videos or T differ from the manifest")
        return out | {"problems": problems, "status": FAIL, "ok": False}
    store = np.load(store_dir(test_dir) / FEATURES_FILE, mmap_mode="r")
    starts = np.asarray(ix.start, dtype=np.int64)
    for i, f in enumerate(caches["d16c"]["features"]):
        rows = np.asarray(store[starts[i] + centre_snippets(t[i])], dtype=np.float32)
        if not np.array_equal(np.asarray(f, dtype=np.float32), rows):
            problems.append(f"d16c video {i}: rows differ from the window-centre snippets")
            break
    rng = np.random.default_rng(seed)
    sample = sorted(int(i) for i in rng.choice(len(t), min(n_random, len(t)), replace=False))
    scale = load_scale_map(variant_dir(e021_dir, "d16m"))
    worst = 0.0
    for i in sample:
        rows = np.asarray(store[starts[i]:starts[i] + int(ix.n[i])], dtype=np.float64)
        means = np.stack([rows[j].sum(axis=0) / len(j) for j in window_sets(t[i])])
        want = scale.apply(means.astype(np.float32))
        got = np.asarray(caches["d16m"]["features"][i], dtype=np.float64)
        worst = max(worst, float(np.abs(got - want).max()))
    if not (math.isfinite(worst) and worst <= Y6_TOL):
        problems.append(f"d16m means differ by {worst}")
    return out | {"sample": sample, "d16m_max_abs_diff": worst, "problems": problems,
                  "status": FAIL if problems else PASS, "ok": not problems}


def _test_cache(vd: Path, variant: str, names: Sequence[str], totals: Sequence[int],
                labels: Sequence[int]) -> tuple[dict, list[str]]:
    """One derived test cache and its problems: no meta.json, the derive record, videos,
    labels, total_frames and seg_labels (all -1)."""
    problems = []
    if (vd / META_FILE).exists():
        problems.append(f"{vd / META_FILE} exists (scale_maps would read a test folder)")
    if not (vd / TEST_DERIVE_RECORD).is_file():
        problems.append(f"{vd / TEST_DERIVE_RECORD} missing")
    c = load_cache(vd, "test")
    if len(c["features"]) != len(names) or [int(y) for y in c["labels"]] != list(labels):
        problems.append(f"{variant}: videos or labels differ from the manifest")
    if [int(x) for x in c.get("total_frames") or []] != list(totals):
        problems.append(f"{variant}: total_frames differ from the manifest")
    if not all(torch.all(torch.as_tensor(s) == -1) for s in c["seg_labels"]):
        problems.append(f"{variant}: seg_labels are not the placeholder -1")
    return c, problems


# --- (y7): video access -----------------------------------------------------------------------

EXTRACT_COMMAND = "e022-test-dense"


def gate_y7(video_access: Sequence[Mapping], extract_access: Sequence[Mapping],
            test_paths: Mapping[str, set[str]], y2_paths: Mapping[str, set[str]], *,
            local_sha256: Mapping[tuple[str, str], str] | None = None,
            y2_videos: int = 20) -> dict:
    """Gate (y7): every opened file (results/e022_video_access.csv rows dataset, path, sha256,
    utc; the fixed ``run_extraction`` access CSV rows) is a test-manifest path or one of the
    (y2) sample (at most 20 training-split files); each has a sha256 and a time (and the local
    copy's sha256 where one exists); every line of the extraction CSV has command
    'e022-test-dense' and purpose 'sample'."""
    allowed = {d: set(test_paths.get(d, set())) | set(y2_paths.get(d, set()))
               for d in set(test_paths) | set(y2_paths)}
    rows = [dict(r) for r in video_access]
    opened = {(r["dataset"], r["path"]) for r in rows} | \
        {(r["dataset"], r["path"]) for r in extract_access}
    outside = sorted(k for k in opened if k[1] not in allowed.get(k[0], set()))
    train = sorted(k for k in opened if k[1] in y2_paths.get(k[0], set()))
    hashed = {(r["dataset"], r["path"]) for r in rows if r.get("sha256") and r.get("utc")}
    unhashed = sorted(opened - hashed)
    srv = {(r["dataset"], r["path"]): r.get("sha256") for r in rows}
    differ = sorted(k for k, sha in (local_sha256 or {}).items() if k in srv and srv[k] != sha)
    bad_lines = [r for r in extract_access if r.get("command") != EXTRACT_COMMAND
                 or r.get("purpose") != "sample"]
    ok = not outside and len(train) <= y2_videos and not unhashed and not differ \
        and not bad_lines and bool(opened)
    return {"opened": len(opened), "outside_manifest_paths": [list(k) for k in outside],
            "training_split_files": len(train), "without_sha256_or_time": len(unhashed),
            "local_sha256_mismatch": [list(k) for k in differ],
            "extract_csv_lines_not_e022": len(bad_lines), "status": PASS if ok else FAIL,
            "ok": bool(ok)}


# --- (r0): the E020 m10 chains ----------------------------------------------------------------

R0_FILES = ("m10/ckpt_best_val.pt", "m10/ckpt_last.pt", "m10/selection.json")
R0_COLUMNS = ("dataset", "pair", "seed", "cell", "rule", "readout", "s2_epoch", "val_video_auc")


def r0_chains() -> list[str]:
    return [f"{d}_{p}_s{s}" for d in DATASETS for p in PAIRS for s in SEEDS]


def runs_csv_rows(path: Path = E020_RESULTS / "e020_runs.csv") -> list[dict]:
    """The m10 rows of e020_runs.csv, only the columns (r0) uses (no test metric is kept)."""
    import csv

    with open(path, encoding="utf-8") as fh:
        return [{k: r[k] for k in R0_COLUMNS} for r in csv.DictReader(fh) if r["cell"] == "m10"]


def gate_r0(current: Mapping[str, str], recorded: Mapping[str, str],
            selections: Mapping[str, Mapping], runs: Sequence[Mapping],
            histories: Mapping[str, Sequence[Mapping]] | None = None) -> dict:
    """Gate (r0) over the 40 E020 m10 chains: the sha256 of '<chain>/m10/ckpt_best_val.pt',
    ckpt_last.pt and selection.json = the pre-freeze record, the selected epoch and V-AUC of
    selection.json = e020_runs.csv (best rule, s2_epoch and val_video_auc = 100 x
    best_val_video_auc) and, with ``histories``, the last rule's (epoch 30, last V-AUC)."""
    table = {(r["dataset"], r["pair"], int(r["seed"]), r["rule"]): r for r in runs
             if r.get("readout", "logit") == "logit"}
    rows = []
    for chain in r0_chains():
        d, rest = chain.split("_", 1)
        pair, seed = rest.rsplit("_s", 1)
        files = [f"{chain}/{f}" for f in R0_FILES]
        sha = compare_hashes(recorded, current, files)
        row = {"chain": chain, "sha256_ok": sha["ok"], "sha256": sha}
        sel, best = selections.get(chain), table.get((d, pair, int(seed), "best"))
        row["selection_ok"] = bool(
            sel is not None and best is not None
            and int(sel["best_epoch"]) == int(best["s2_epoch"])
            and 100 * float(sel["best_val_video_auc"]) == float(best["val_video_auc"]))
        if histories is not None:
            hist, last = histories.get(chain), table.get((d, pair, int(seed), "last"))
            row["last_ok"] = bool(hist and last is not None and int(last["s2_epoch"]) == 30
                                  and 100 * float(hist[-1]["val_video_auc"])
                                  == float(last["val_video_auc"]))
        row["ok"] = bool(row["sha256_ok"] and row["selection_ok"] and row.get("last_ok", True))
        rows.append(row)
    return {"rows": rows, "n": len(rows), "ok": len(rows) == 40 and all(r["ok"] for r in rows)}


# --- (f): training batches ------------------------------------------------------------------

TRAIN_ENV_KEYS = ("torch", "cuda", "gpu", "driver")


def e021_training_env() -> dict:
    """The E021 training environment: the no-change gate (g1) record (E021 gate (f) target)."""
    env = _json(E021_RESULTS / "e021_identity.json")["g1"]["environment"]
    return {k: env.get(k) for k in TRAIN_ENV_KEYS}


def gate_f(batches: Mapping[str, Mapping], *, recorded: Mapping, tests_now: Mapping,
           tests_before: Mapping, want_env: Mapping | None = None,
           driver_policy: str = "record", reason: str | None = None,
           base: str = BASE_COMMIT, repo: Path = REPO_ROOT) -> dict:
    """Gate (f) per training batch (``batches``: name -> env record with torch, cuda, gpu,
    driver, commit and "train_inputs", the batch's sha256 of the d16m / d16c training caches):
    the add-only rule and S_train, S_score identical at its commit, the environment = E021's
    (only the driver differing: recorded under ``driver_policy`` "record", the verdict tagged
    "(드라이버 변경)"; "fail" fails it), the training inputs = the pre-freeze record; and the
    tests/ record (``compare_tests``)."""
    if driver_policy not in ("record", "fail"):
        raise ValueError(f"unknown driver policy {driver_policy!r}")
    want = dict(want_env if want_env is not None else e021_training_env())
    rows, cache = {}, {}
    for name, env in batches.items():
        commit = env.get("commit")
        if commit not in cache:
            cache[commit] = (code_identity(commit, ("S_train", "S_score"), base=base, repo=repo)
                             if commit else {"ok": False, "reason": "no deployed commit"})
        diff = [k for k in TRAIN_ENV_KEYS if env.get(k) != want.get(k)]
        driver_only = diff == ["driver"]
        env_ok = not diff or (driver_only and driver_policy == "record")
        inputs = inputs_gate("f", recorded, dict(env.get("train_inputs") or {}))
        rows[name] = {"commit": commit, "code_ok": cache[commit]["ok"], "env_differences": diff,
                      "driver_changed": driver_only, "env_ok": env_ok, "inputs": inputs,
                      "ok": bool(cache[commit]["ok"] and env_ok and inputs["ok"])}
    tests = compare_tests(tests_now, tests_before, reason)
    changed = any(r["driver_changed"] for r in rows.values())
    return {"want": want, "batches": rows, "code": {c: v for c, v in cache.items()},
            "tests": tests, "driver_changed": changed,
            "tag": "(드라이버 변경)" if changed else None,
            "ok": bool(rows) and all(r["ok"] for r in rows.values()) and tests["ok"]}


# --- (w2): independent recomputation of the selection -----------------------------------------

WINDOWS, SNIPPET = 32, 16
W2_TOL = 1e-9


def _supports(total_frames: int) -> tuple[np.ndarray, np.ndarray]:
    t = int(total_frames)
    start = np.arange(0, t, SNIPPET, dtype=np.int64)
    return start, np.minimum(start + SNIPPET, t)


def _supporting(frame: int, starts: np.ndarray, ends: np.ndarray) -> int:
    """The snippet whose support [start, end) holds ``frame``."""
    hit = np.flatnonzero((starts <= frame) & (frame < ends))
    if len(hit) != 1:
        raise ValueError(f"frame {frame} lies in {len(hit)} supports")
    return int(hit[0])


def _windows(total_frames: int) -> tuple[np.ndarray, np.ndarray]:
    t = int(total_frames)
    if t < WINDOWS:
        raise ValueError(f"T = {t} < {WINDOWS} windows")
    length = t // WINDOWS
    lo = np.arange(WINDOWS, dtype=np.int64) * length
    hi = lo + length
    hi[-1] = t
    return lo, hi


def window_sets(total_frames: int) -> list[np.ndarray]:
    """J_k of every evaluator window, written again from the E021 PLAN definition ("창·배정
    규칙"): window k = [k L, (k + 1) L) with L = T // 32 (the last to T); a snippet belongs to the
    window holding its centre frame (support start + floor(support length / 2)); an empty window
    takes the snippet supporting its centre frame (window start + floor(window length / 2))."""
    s, e = _supports(total_frames)
    centre = s + (e - s) // 2
    lo, hi = _windows(total_frames)
    sets = []
    for k in range(WINDOWS):
        j = np.flatnonzero((lo[k] <= centre) & (centre < hi[k])).astype(np.int64)
        if j.size == 0:
            j = np.array([_supporting(int(lo[k] + (hi[k] - lo[k]) // 2), s, e)], dtype=np.int64)
        sets.append(j)
    return sets


def centre_snippets(total_frames: int) -> np.ndarray:
    """(32,) the snippet supporting each window centre frame (d16c)."""
    s, e = _supports(total_frames)
    lo, hi = _windows(total_frames)
    return np.array([_supporting(int(a + (b - a) // 2), s, e) for a, b in zip(lo, hi,
                                                                               strict=True)])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """fp32 sigma on the CPU (torch), the scores of the training records."""
    return torch.sigmoid(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()


def video_scores(z: np.ndarray, total_frames: int, gamma: float | None,
                 sets: Sequence[np.ndarray] | None = None) -> dict:
    """W and G video scores (max of fp32 sigma) and c(v) of one video's snippet logits z:
    c(v) = float32(gamma) x float32(float64 mean of all z) (0 without gamma); W: per window the
    float64 mean of z over J_k as float32, plus c(v); G: every z_j + c(v). ``sets``: the
    video's ``window_sets`` when already computed."""
    z = np.asarray(z, dtype=np.float32)
    s, _ = _supports(total_frames)
    if len(z) != len(s):
        raise ValueError(f"{len(z)} logits for {len(s)} snippets (T = {total_frames})")
    c = np.float32(0.0) if gamma is None else \
        np.float32(np.float32(gamma) * np.float32(z.mean(dtype=np.float64)))
    u = np.array([np.float32(z[j].mean(dtype=np.float64))
                  for j in (sets if sets is not None else window_sets(total_frames))],
                 dtype=np.float32)
    w = (u + c).astype(np.float32)
    g = (z + c).astype(np.float32)
    return {"W": float(_sigmoid(w).max()), "G": float(_sigmoid(g).max()), "c": float(c)}


def exact_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Video AUC by counting the (anomalous, normal) pairs (ties one half) in integers; the
    float is the correctly rounded ratio. NaN when a score is not finite."""
    x, y = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if not np.isfinite(x).all():
        return math.nan
    pos, neg = np.sort(x[y == 1]), np.sort(x[y == 0])
    if not len(pos) or not len(neg):
        raise ValueError("both classes are needed")
    below = np.searchsorted(neg, pos, side="left")
    ties = np.searchsorted(neg, pos, side="right") - below
    twice = int(2 * below.sum() + ties.sum())
    return twice / (2 * len(pos) * len(neg))


def first_best(values: Sequence[float]) -> int:
    """1-based index of the highest value, earliest on ties (strict >)."""
    best = 0
    for i, v in enumerate(values):
        if v > values[best]:
            best = i
    return best + 1


def w2_chain(dense_z: np.ndarray, counts: Sequence[int], totals: Sequence[int],
             labels: Sequence[int], gamma: Sequence[float] | None,
             recorded: Mapping[str, Mapping], readouts: Sequence[str] = ("W", "G"),
             tol: float = W2_TOL) -> dict:
    """(w2) of one chain from its stored validation logits (``dense_z`` (epochs, rows): the
    epochs' snippet logits z of the validation videos in order, ``counts`` per video, ``gamma``
    (epochs,) or None): per readout the full-validation V-AUC of every epoch and the selected
    epoch, against ``recorded`` (readout -> {"epoch": 1-based, "v_auc": per-epoch V-AUC} or
    {"epoch", "selected_v_auc"}). Non-finite logits, gamma or V-AUC: numerical failure,
    skipped."""
    z = np.asarray(dense_z, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    g = None if gamma is None else np.asarray(gamma, dtype=np.float32)
    if z.shape[1] != int(counts.sum()) or len(totals) != len(counts):
        return {"ok": False, "reason": f"shape {z.shape} for {int(counts.sum())} rows"}
    if not np.isfinite(z).all() or (g is not None and not np.isfinite(g).all()):
        return {"ok": True, "skipped": "non-finite logits or gamma (numerical failure)"}
    scores = {r: np.empty((len(z), len(counts))) for r in readouts}
    sets = [window_sets(int(t)) for t in totals]               # per video, every epoch
    for e in range(len(z)):
        ge = None if g is None else float(g[e])
        for i, (a, n, t) in enumerate(zip(first, counts, totals, strict=True)):
            v = video_scores(z[e, a:a + n], int(t), ge, sets[i])
            for r in readouts:
                scores[r][e, i] = v[r]
    out = {"ok": True}
    for r in readouts:
        aucs = [exact_auc(scores[r][e], labels) for e in range(len(z))]
        if not all(math.isfinite(a) for a in aucs):
            out[r] = {"skipped": "non-finite V-AUC (numerical failure)", "ok": True}
            continue
        rec = recorded.get(r) or {}
        epoch = first_best(aucs)
        row = {"epoch": epoch, "epoch_recorded": rec.get("epoch"), "v_auc": aucs}
        if "v_auc" in rec:
            want = [float(x) for x in rec["v_auc"]]
            diffs = [abs(a - b) for a, b in zip(aucs, want, strict=False)]
            row["max_abs_diff"] = max(diffs) if len(want) == len(aucs) else math.inf
        else:
            row["max_abs_diff"] = abs(aucs[epoch - 1] - float(rec.get("selected_v_auc",
                                                                        math.nan)))
        row["ok"] = bool(rec.get("epoch") == epoch and math.isfinite(row["max_abs_diff"])
                         and row["max_abs_diff"] <= tol)
        out[r] = row
        out["ok"] = out["ok"] and row["ok"]
    return out


def gate_w2(chains: Mapping[str, Mapping]) -> dict:
    """(w2) over the chains (name -> the keyword arguments of ``w2_chain``)."""
    rows = {name: w2_chain(**kw) for name, kw in chains.items()}
    return {"rows": rows, "n": len(rows), "ok": bool(rows) and all(r["ok"] for r in rows.values())}
