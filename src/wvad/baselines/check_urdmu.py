"""E023 equivalence harness of the UR-DMU port (wvad.baselines.urdmu) against the official code
(PLAN.md "관문" (q1), (q1b), (q1c), (q1e), "실패 처리", "구현"). New code (add-only rule).

An equivalence harness: it loads the official clone (WVAD_OFFICIAL_ROOT/UR-DMU at the commit of
results/e023_licences.json, checked; its files' LF-normalised sha256 pinned in OFFICIAL_SHA256)
by path through the ``official`` loader module (wvad.baselines.official), which every entry point
takes as its first argument (``official``), as check_rtfm and check_vadclip_v do: no source file
under src/ but official.py imports it (import-graph test), so the callers - the tests, the
command line below and the (q1b) official worker's own command line - pass it in. The training
and scoring code never imports this module. Modified official files exist only in a work folder
outside the repository (refused inside it); only their diff sha256 is recorded.

Stand-ins (listed in every result under "stand_ins"; none changes a computed value)
    einops      translayer.py's rearrange: a real einops when importable or found in
                $WVAD_EINOPS_DIR (a folder holding the einops package), else a stand-in doing the
                reshape / permute recipe of einops' torch backend for the two patterns used. The
                stand-in repeats the port's own head split, so the comparison would be circular
                there: every result records "einops" and its "pass" requires the real einops
                (README: pip install einops, or WVAD_EINOPS_DIR); ``allow_einops_stand_in``
                exists for the stand-in's own unit test only
    ipdb        imported (unused) by translayer.py: an empty module when not installed
    visdom      utils.Visualizer's backend: always a recorder (no server); plot_lines are the
                official per-step losses of (q1b)
    np.int      removed in numpy >= 1.24, used by the official UCF loader (dtype=np.int): the
                builtin int (numpy's old alias) while the official code runs
    cuda shim   CPU runs only: Tensor.cuda / Module.cuda return the object, torch.cuda.set_device
                does nothing (the official .cuda() calls only move data); a CUDA run uses none
Test seal: the official test functions load frame_label/<gt>.npy (test annotation); (q1e) answers
that one np.load call with a synthetic array and never opens the file. The (q1b) tree has no test
list; the official test evaluation is off (PLAN.md: "시험 평가는 끄거나 검증으로 바꾼다"), and an
audit hook lists every file the official and port runs open ("blocked" patterns flagged).

(q1)  ``q1(device)``: per (D, T, input LN) - D 1024 (official), 1792, 1920 (+ LN), T 200 / 32 -
      construction (seeded, the official constructor vs METHOD.build: same state_dict bit for
      bit, so the generator is consumed alike), training forward / AD_Loss / gradients under the
      same random stream, one Adam step (ucf_main.py's optimizer vs the recipe's), ``steps`` calls
      of the official train.py ``train`` vs METHOD.train_step (losses as visdom saw them, weights),
      the Test forward (frame and the cls_head logits), and the shape-only rule (only the
      embedding conv's in_channels differs from the official D = 1024 network).
      Pass: bit equality or relative error <= 1e-6 (eval |delta| <= 1e-6); construction bit equal.
(q1c) ``q1c(dataset, labels, lengths, work)``: the official dataset class (dataset_loader.py with
      the split constant set to the training-part anomalous count) in the official DataLoader /
      re-creation loop (ucf_main.py L32-L41, L67-L72; train.py L58-L59) vs the "window_mean"
      sampler from the same seed: the video positions, labels and the generator state afterwards,
      bit for bit; the official window boundaries (UCF get_data, XD random_perturb) equal
      derive_n.linspace_bounds; the official float32 window means against derive_n (reported).
(q1e) ``q1e(device)``: the official test functions (ucf_test.py / xd_test.py ``test``, their own
      crop loop: UCF 10 items, XD batches of 5, one crop here) on the same weights, eval mode,
      fp32, against common.score_video (lengths below and above the unit and N_chunk, the last
      shorter N_chunk piece unpadded, input LN prepended for XCB+SigLIP, both units): logits
      (cls_head input of the sigmoid, hooked) and sigmoid scores, |delta| <= 1e-6. URDMU has no
      BatchNorm and no padded pieces (test "whole"); dropout is off in eval mode (checked).
(q1b) ``q1b(work, dataset, pair, seed)``: the official training pipeline (ucf_main.py /
      xd_main.py, run in a subprocess under official.session with the modified copies first on
      sys.path) and the port (wvad.baselines.train in a subprocess) on the same training-part
      inputs, deterministic mode, then per-step losses, last weights, validation logits and
      V-AUCs compared (PLAN.md "(q1b) 통과 기준"). Official changes, by category: input_dimension
      (config.len_feature), data_path (the list file of the exported training-part videos, the
      dataset split constant), test_off (no test loader, no step-0 test), validation (the
      test block -> the harness's validation at the 30 evaluation points, isolated generators),
      harness_io (the hook import, the final weights). Inputs (``mode``): "snippets" (default,
      the gate of record, PLAN.md 기록 2026-10-19) - the processed snippets (the variant's scale
      map per snippet), the official code makes the windows (float32 means: ULP differences, so
      the numeric rule of R3-6 decides when bits differ; a failure is reported, never answered
      by switching the mode); "port_rows" (extra evidence) - the port's own 200-row training
      inputs as the official files (the official 200-window mean of 200 rows is the identity,
      checked), the bit path. The record (<work>/record.json) has the plan sha256, the commit,
      the official file hashes, the edits and diff sha256, the stand-ins (einops real
      required), the opened-file audit (paths relative to symbolic roots), the comparisons and
      the verdict; ``merge_record`` puts it into results/e023_q1b.json (methods.urdmu.runs
      "<ds>/<pair>/s<seed>/<mode>", verdicts of the snippets runs, evidence of the port_rows
      runs; under a lock).
(q1e) ``q1e_val(official, dataset, pair)``: (q1e) on the processed validation videos (real
      lengths and value ranges: the chain's InputPipeline, the input LN for XCB+SigLIP).

    RUN = python -c "import sys; from wvad.baselines import official, check_urdmu as c;
                     c.main(sys.argv[1:], official=official)"
    RUN q1 --device cpu|cuda --out FILE
    RUN q1c --dataset ucf|xdv --pair xcb_xcl --work DIR --out FILE
    RUN q1e --device cpu|cuda [--n_chunk 4096] --out FILE
    RUN q1e_val --dataset ucf|xdv --pair xcb_xcl|xcb_siglip [--max_videos K] [--device cuda]
                [--paths FILE] --out FILE
    RUN q1b --dataset ucf|xdv --pair xcb_xcl --seed 0|1|2 --work DIR [--mode snippets|port_rows]
            [--device cuda] [--record results/e023_q1b.json]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import difflib
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wvad.baselines import common, samplers
from wvad.baselines import urdmu as port
from wvad.paths import REPO_ROOT

METHOD = "urdmu"
TOL = 1e-6                                   # (q1) relative, (q1e) absolute
Q1B_LOSS_REL = 1e-5                          # numeric criterion: first 50 steps
Q1B_FIRST = 50
Q1B_VAUC_SEED = 1.0                          # V-AUC points
Q1B_VAUC_MEAN = 0.5
E023_DIR = REPO_ROOT / "experiments" / "E023_localisation_baselines"
PLAN_FILE = E023_DIR / "PLAN.md"
Q1B_RESULTS = E023_DIR / "results" / "e023_q1b.json"
HOOK_ENV = "WVAD_E023_Q1B_SPEC"
EINOPS_ENV = "WVAD_EINOPS_DIR"
# official top-level module names (the clone's own imports); none may be loaded from elsewhere
OFFICIAL_TOP = ("model", "memory", "translayer", "utils", "train", "config", "options",
                "dataset_loader", "ucf_test", "xd_test", "ucf_main", "xd_main")
OFFICIAL_SHA256 = {                          # LF-normalised, commit 40cfdf5f
    "ucf_main.py": "51c060b5dd0b9975b72c2e4bd3617837fce3d29236503389b607fe62c8c7fe2a",
    "xd_main.py": "77b9d8ad01d37baade176d51615d64bd57d24ffb91794c146687b20e2df0a787",
    "dataset_loader.py": "ecb15fc00e5789a3c8353f54093224b8e6f0df80054ee65d1f9439011fe30788",
    "ucf_test.py": "b5ae337da9afe3190ac8032bab64ce0cc5ac6467250505714c05d1f4884136a5",
    "xd_test.py": "15ac7996a2e38c36bd5d8140835360d2bf07d634cc891f7c711d77fc007f93fd",
    "train.py": "c5d30ffc7dc401925d5d00c983aa32bf3add77c050b582e3d87b61698d2b9901",
    "model.py": "6cebc33270dd1f28384747a2b45536b26753d18f518509250a8956d9f40d26d5",
    "utils.py": "2df18a9898a53d2269a61706ad0f17b62f9f566e403077f42f9af8fca451a34f",
    "translayer.py": "d2c1596effc7ecd5d0940fcc01845f8a18b2d97a5cc4cdd4726cd08928ace1bf",
    "memory.py": "707f33164bb721bab9767f50b5fe053183341ca634f0e4339837f15a00c6f2de",
    "options.py": "353cd751f5e92e70b88fa366f9d5982c865bf6990cbd2fe6da4718ab1f821d95",
    "config.py": "8357f808ba507a0fc686fd89aeac234392d2429740f694c2c3d492bae779b67b",
}
MAINS = {"ucf": "ucf_main.py", "xdv": "xd_main.py"}
DATASET_CLASS = {"ucf": "UCF_crime", "xdv": "XDVideo"}
SPLIT_CONST = {"ucf": 8100, "xdv": 9525}
TESTS = {"ucf": ("ucf_test.py", "frame_label/gt-ucf.npy", 10),       # file, GT, items per video
         "xdv": ("xd_test.py", "frame_label/xd_gt.npy", 5)}
REAL_CATEGORIES = ("input_dimension", "data_path", "test_off", "validation", "harness_io")
LOSS_KEYS = ("total_loss",) + port.LOSS_KEYS
MODES = ("snippets", "port_rows")
GATE_OF_RECORD = "snippets"                  # PLAN.md 기록 2026-10-19 (review M5)
EINOPS_STAND_IN = "einops: stand-in"
SCALE = {"distance_margin": port.DISTANCE_MARGIN}


class HarnessError(RuntimeError):
    """The harness cannot run as configured (clone, work folder, inputs)."""


# --- comparisons ------------------------------------------------------------------------------

def _t(x) -> torch.Tensor:
    return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))


def compare(a, b) -> dict:
    """Bit equality, max |a - b| and that maximum relative to max |b| (float64 arithmetic)."""
    ta, tb = _t(a), _t(b)
    if ta.shape != tb.shape:
        return {"shape": [list(ta.shape), list(tb.shape)], "bit_equal": False,
                "max_abs": math.inf, "rel": math.inf}
    bit = ta.dtype == tb.dtype and bool(torch.equal(ta, tb))
    diff = (ta.double() - tb.double()).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    scale = float(tb.double().abs().max()) if tb.numel() else 0.0
    rel = max_abs / scale if scale > 0 else (0.0 if max_abs == 0 else math.inf)
    return {"shape": list(ta.shape), "bit_equal": bit, "max_abs": max_abs, "rel": rel}


def ok_rel(c: Mapping, tol: float = TOL) -> bool:
    return bool(c["bit_equal"]) or (math.isfinite(c["rel"]) and c["rel"] <= tol)


def ok_abs(c: Mapping, tol: float = TOL) -> bool:
    return bool(c["bit_equal"]) or (math.isfinite(c["max_abs"]) and c["max_abs"] <= tol)


def _summary(cmps: Mapping[str, Mapping], rel: bool = True) -> dict:
    """Worst of named comparisons (names that fail listed)."""
    test = ok_rel if rel else ok_abs
    vals = list(cmps.values())
    return {"n": len(vals), "bit_equal": all(c["bit_equal"] for c in vals),
            "max_abs": max((c["max_abs"] for c in vals), default=0.0),
            "max_rel": max((c["rel"] for c in vals), default=0.0),
            "failed": sorted(k for k, c in cmps.items() if not test(c))}


def _float_cmp(a: float, b: float) -> dict:
    a, b = float(a), float(b)
    d = abs(a - b)
    return {"bit_equal": a == b, "max_abs": d,
            "rel": d / abs(b) if b != 0 else (0.0 if d == 0 else math.inf)}


def _reseed(seed: int) -> None:
    torch.manual_seed(seed)            # CPU and every CUDA device


@contextlib.contextmanager
def deterministic_mode(enabled: bool = True) -> Iterator[dict]:
    """torch.use_deterministic_algorithms and the cudnn flags for a block, restored after; on a
    machine with CUDA the cuBLAS workspace variable is set for the block when missing."""
    saved = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
             torch.backends.cudnn.deterministic, os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    if enabled and torch.cuda.is_available() and saved[3] is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(bool(enabled))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(enabled) or saved[2]
    try:
        yield {"deterministic": bool(enabled),
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    finally:
        torch.use_deterministic_algorithms(saved[0])
        torch.backends.cudnn.benchmark = saved[1]
        torch.backends.cudnn.deterministic = saved[2]
        if saved[3] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


# --- the official clone and the stand-ins ---------------------------------------------------

def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _lf_text(path: Path) -> str:
    return Path(path).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def official_files(official: types.ModuleType) -> dict[str, str]:
    """LF-normalised sha256 of the official files the harness uses (all must be pinned)."""
    base = official.repo_dir(METHOD)
    got = {f: _lf_sha256(base / f) for f in OFFICIAL_SHA256}
    bad = sorted(f for f, s in got.items() if s != OFFICIAL_SHA256[f])
    if bad:
        raise HarnessError(f"official files {bad} differ from the pinned commit content")
    return got


def _einops_stand_in() -> types.ModuleType:
    mod = types.ModuleType("einops")
    mod.__doc__ = "E023 harness stand-in (einops not installed)"

    def rearrange(tensor, pattern, **axes):
        p = " ".join(str(pattern).split())
        if p == "b n (h d) -> b h n d":
            b, n, hd = tensor.shape
            h = int(axes["h"])
            return tensor.reshape(b, n, h, hd // h).permute(0, 2, 1, 3)
        if p == "b h n d -> b n (h d)":
            b, h, n, d = tensor.shape
            return tensor.permute(0, 2, 1, 3).reshape(b, n, h * d)
        raise NotImplementedError(f"einops stand-in: pattern {pattern!r}")

    mod.rearrange = rearrange
    return mod


def _visdom_recorder() -> types.ModuleType:
    mod = types.ModuleType("visdom")
    mod.__doc__ = "E023 harness stand-in for visdom: records utils.Visualizer calls, no server"
    log: list[tuple] = []

    class Visdom:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def line(self, Y=None, X=None, win=None, **kwargs):  # noqa: N803 - visdom's names
            y = np.asarray(Y, dtype=np.float64).reshape(-1)
            log.append(("line", str(win), None if X is None else np.asarray(X).reshape(-1), y))

        def image(self, *args, **kwargs):
            log.append(("image",))

        def scatter(self, *args, **kwargs):
            log.append(("scatter",))

    mod.Visdom = Visdom
    mod.LOG = log
    return mod


def _tensor_cuda(self, *args, **kwargs):
    return self


def _module_cuda(self, *args, **kwargs):
    return self


def _no_set_device(*args, **kwargs):
    return None


@contextlib.contextmanager
def _stand_ins(device: torch.device) -> Iterator[list[str]]:
    """The stand-ins of the module docstring, removed again on exit."""
    notes: list[str] = []
    saved_modules = {k: sys.modules.get(k) for k in ("einops", "ipdb", "visdom")}
    added_prefixes: list[str] = []

    def put(name: str, module: types.ModuleType, note: str) -> None:
        sys.modules[name] = module
        added_prefixes.append(name)
        notes.append(note)

    if importlib.util.find_spec("einops") is not None:
        notes.append(f"einops: installed ({importlib.import_module('einops').__version__})")
    elif os.environ.get(EINOPS_ENV):
        saved_path = list(sys.path)
        sys.path.insert(0, os.environ[EINOPS_ENV])
        try:
            mod = importlib.import_module("einops")
        finally:
            sys.path[:] = saved_path
        added_prefixes.append("einops")
        notes.append(f"einops: real {mod.__version__} loaded from ${EINOPS_ENV}")
    else:
        put("einops", _einops_stand_in(), f"{EINOPS_STAND_IN} (einops' torch recipe: reshape, "
            "permute[, reshape]) for 'b n (h d) -> b h n d' and 'b h n d -> b n (h d)'")
    if importlib.util.find_spec("ipdb") is None:
        put("ipdb", types.ModuleType("ipdb"), "ipdb: empty module (imported, unused, by "
            "translayer.py)")
    put("visdom", _visdom_recorder(), "visdom: recorder (utils.Visualizer calls recorded, no "
        "server)")
    np_alias = "int" not in vars(np)
    if np_alias:
        np.int = int
        notes.append("np.int: builtin int while the official code runs (numpy >= 1.24 removed "
                     "the alias; dataset_loader.py L54)")
    cpu = device.type != "cuda"
    saved_cuda = (torch.Tensor.__dict__.get("cuda"), nn.Module.cuda, torch.cuda.set_device)
    if cpu:
        torch.Tensor.cuda = _tensor_cuda
        nn.Module.cuda = _module_cuda
        torch.cuda.set_device = _no_set_device
        notes.append("cuda shim: Tensor.cuda / Module.cuda return the object, "
                     "torch.cuda.set_device does nothing (CPU run)")
    try:
        yield notes
    finally:
        if cpu:
            if saved_cuda[0] is None:
                del torch.Tensor.cuda
            else:
                torch.Tensor.cuda = saved_cuda[0]
            nn.Module.cuda = saved_cuda[1]
            torch.cuda.set_device = saved_cuda[2]
        if np_alias:
            del np.int
        for name in list(sys.modules):
            if any(name == p or name.startswith(p + ".") for p in added_prefixes):
                sys.modules.pop(name, None)
        for name, mod in saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod


@dataclass
class Official:
    """An open official session (``official_urdmu``)."""

    session: object
    stand_ins: list[str]
    commit: dict
    files: dict[str, str]

    def load(self, relpath: str) -> types.ModuleType:
        return self.session.load(relpath)

    @staticmethod
    def top(name: str) -> types.ModuleType:
        """A top-level official module as the official scripts import it (``import utils``)."""
        if name not in OFFICIAL_TOP:
            raise HarnessError(f"{name} is not an official module name")
        return importlib.import_module(name)


def einops_check(notes: Sequence[str], allow_stand_in: bool = False) -> dict:
    """Whether the official Attention ran the real einops (review M6): the stand-in repeats
    the port's head split, so a run on it does not pass unless ``allow_stand_in`` (the
    stand-in's own unit test)."""
    stand_in = any(str(n).startswith(EINOPS_STAND_IN) for n in notes)
    return {"real": not stand_in, "allowed_stand_in": bool(allow_stand_in),
            "ok": (not stand_in) or bool(allow_stand_in),
            "fix": None if not stand_in else "pip install einops, or set WVAD_EINOPS_DIR to a "
            "folder holding the einops package (README)"}


@contextlib.contextmanager
def official_urdmu(official: types.ModuleType, device: str | torch.device = "cpu",
                   extra_paths: Sequence[str | os.PathLike[str]] = ()) -> Iterator[Official]:
    """official.session("urdmu") with the stand-ins; ``extra_paths`` (modified copies outside
    the repository) go before the clone on sys.path. Every official top-level module is dropped
    from sys.modules on exit."""
    clash = sorted(m for m in OFFICIAL_TOP if m in sys.modules)
    if clash:
        raise HarnessError(f"modules {clash} are loaded already (they would shadow the "
                           "official ones)")
    for p in extra_paths:
        _outside_repo(Path(p))
    commit = official.check_commit(METHOD)
    files = official_files(official)
    bytecode = sys.dont_write_bytecode
    with _stand_ins(torch.device(device)) as notes, official.session(METHOD) as s:
        saved = list(sys.path)
        sys.path[:0] = [str(Path(p)) for p in extra_paths]
        sys.dont_write_bytecode = True                 # nothing written into the clone
        try:
            yield Official(s, notes, commit, files)
        finally:
            sys.dont_write_bytecode = bytecode
            sys.path[:] = saved
            for name in OFFICIAL_TOP:
                sys.modules.pop(name, None)


def _outside_repo(path: Path) -> Path:
    p = Path(path).resolve()
    if p == REPO_ROOT.resolve() or REPO_ROOT.resolve() in p.parents:
        raise HarnessError(f"{path} is inside the repository: official copies and work files "
                           "stay outside it")
    return p


@contextlib.contextmanager
def _cwd(path: Path) -> Iterator[None]:
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _Wind:
    """The ``wind`` argument of the official train / test functions (a Visualizer's methods)."""

    def __init__(self) -> None:
        self.points: list[tuple[str, float]] = []
        self.arrays: dict[str, np.ndarray] = {}

    def plot_lines(self, name, y, **kwargs) -> None:
        self.points.append((str(name), float(y)))

    def lines(self, name, line, X=None) -> None:  # noqa: N803 - utils.Visualizer's name
        self.arrays[str(name)] = np.asarray(line)

    def disp_image(self, name, img) -> None:
        return None

    def scatter(self, name, data) -> None:
        return None


class _WithInputNorm(nn.Module):
    """The official network with the port's input LayerNorm in front (PLAN.md (q1e): for
    XCB+SigLIP only the input LN is added to the official function)."""

    def __init__(self, norm: nn.Module, net: nn.Module) -> None:
        super().__init__()
        self.input_norm = norm
        self.net = net

    @property
    def flag(self):
        return self.net.flag

    @flag.setter
    def flag(self, value) -> None:
        self.net.flag = value

    @property
    def cls_head(self):
        return self.net.cls_head

    def forward(self, x):
        return self.net(self.input_norm(x))


def _port_name(official_name: str) -> str:
    return official_name[4:] if official_name.startswith("net.") else official_name


def _params_compare(mine: nn.Module, ref: nn.Module, attr: str = "data") -> dict[str, dict]:
    theirs = {_port_name(k): v for k, v in ref.named_parameters()}
    out = {}
    for k, p in mine.named_parameters():
        q = theirs.get(k)
        if q is None:
            out[k] = {"bit_equal": False, "max_abs": math.inf, "rel": math.inf, "missing": True}
            continue
        a, b = getattr(p, attr), getattr(q, attr)
        if a is None or b is None:
            out[k] = {"bit_equal": a is None and b is None, "max_abs": 0.0 if a is b else
                      math.inf, "rel": 0.0 if a is b else math.inf}
        else:
            out[k] = compare(a, b)
    return out


def _state_compare(official_state: Mapping, port_state: Mapping) -> dict:
    """Construction: every official tensor equals the port's bit for bit; the port adds only
    input_norm.*."""
    extra = sorted(k for k in port_state if k not in official_state)
    missing = sorted(k for k in official_state if k not in port_state)
    unequal = sorted(k for k in official_state if k in port_state and not (
        official_state[k].shape == port_state[k].shape
        and torch.equal(official_state[k], port_state[k])))
    ok = not missing and not unequal and all(k.startswith("input_norm.") for k in extra)
    return {"ok": ok, "missing": missing, "extra": extra, "unequal": unequal}


# --- (q1) ----------------------------------------------------------------------------------

Q1_CASES = ((1024, False), (1792, False), (1920, True), (1920, False))


def _q1_case(model_mod, train_mod, dev: torch.device, dim: int, unit: int, ln: bool,
             seed: int, per_class: int, steps: int, ref_shapes: Mapping[str, tuple]) -> dict:
    m = port.METHOD
    torch.manual_seed(seed)
    official_net = model_mod.WSAD(dim, flag="Train", a_nums=60, n_nums=60)  # ucf_main.py L29
    torch.manual_seed(seed)
    mine = m.build(dim, unit=unit, input_ln=ln, scale=SCALE)
    construction = _state_compare(official_net.state_dict(), mine.state_dict())
    shapes = sorted(k for k, v in mine.state_dict().items()
                    if not k.startswith("input_norm.") and tuple(v.shape) != ref_shapes.get(k))
    ref = _WithInputNorm(copy.deepcopy(mine.input_norm), official_net) if ln else official_net
    ref.to(dev)
    mine.to(dev)
    b = 2 * per_class
    g = torch.Generator().manual_seed(seed + 101)
    xs = [(torch.randn(b, unit, dim, generator=g) * 0.8
           + torch.linspace(-0.2, 0.3, dim)).to(dev) for _ in range(steps + 1)]
    y_int = torch.tensor([0] * per_class + [1] * per_class)          # official labels (int64)
    lengths = torch.full((b,), unit, dtype=torch.int64, device=dev)

    def batch(x):
        return common.Batch(x, y_int.float().to(dev), lengths, tuple(range(b)), per_class)

    crit = train_mod.AD_Loss()
    ref.train()
    ref.flag = "Train"
    m.set_mode(mine, True)
    _reseed(seed + 11)
    out_ref = ref(xs[0])
    _reseed(seed + 11)
    out_mine = mine(xs[0])
    forward = {k: compare(out_mine[k], out_ref[k]) for k in out_ref}
    _reseed(seed + 12)
    cost_ref, parts_ref = crit(ref(xs[0]), y_int.to(dev))
    _reseed(seed + 12)
    lo = m.loss(mine, batch(xs[0]), unit=unit, scale=SCALE)
    losses = {"total_loss": compare(lo.total, cost_ref),
              **{k: compare(lo.parts[k], parts_ref[k]) for k in port.LOSS_KEYS}}
    ref.zero_grad(set_to_none=True)
    mine.zero_grad(set_to_none=True)
    cost_ref.backward()
    lo.total.backward()
    grads = _params_compare(mine, ref, "grad")
    opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-4, betas=(0.9, 0.999),
                               weight_decay=0.00005)                  # ucf_main.py L54-L55
    opt_mine = m.make_optimizer(mine, port.RECIPE.optimizer.lr)
    opt_ref.step()
    opt_mine.step()
    step_weights = _params_compare(mine, ref)
    wind, train_steps = _Wind(), []
    for s in range(steps):
        x = xs[s + 1]
        _reseed(seed + 20 + s)
        train_mod.train(ref, iter([(x[:per_class].cpu(), y_int[:per_class])]),
                        iter([(x[per_class:].cpu(), y_int[per_class:])]), opt_ref, crit,
                        wind, s + 1)
        _reseed(seed + 20 + s)
        got = m.train_step(mine, batch(x), opt_mine, unit=unit, scale=SCALE)
        seen = dict(wind.points[-len(LOSS_KEYS):])
        train_steps.append({
            "losses": {k: _float_cmp(got["total" if k == "total_loss" else k], seen[k])
                       for k in LOSS_KEYS},
            "weights": _summary(_params_compare(mine, ref))})
    ref.eval()
    ref.flag = "Test"
    m.set_mode(mine, False)
    captured: list[torch.Tensor] = []
    hook = ref.cls_head.mlp[2].register_forward_hook(
        lambda mod, inp, out: captured.append(out.detach()))
    try:
        with torch.no_grad():
            ev_ref = ref(xs[0])
            ev_mine = mine(xs[0])
            ref(xs[0][per_class][None])                         # one video, as the test feeds
            seq = m.sequence_logits(mine, xs[0][per_class], unit=unit)
    finally:
        hook.remove()
    z_ref = captured[0].reshape(b, unit)
    evaluation = {"frame": compare(ev_mine["frame"], ev_ref["frame"]),
                  "logit": compare(ev_mine["logit"], z_ref),
                  "sequence_logits": compare(seq, captured[1].reshape(unit))}
    shapes_ok = set(shapes) <= {"embedding.conv_1.0.weight"} and (dim != 1024 or not shapes)
    ok = (construction["ok"] and shapes_ok
          and all(ok_rel(c) for c in forward.values())
          and all(ok_rel(c) for c in losses.values())
          and not _summary(grads)["failed"] and not _summary(step_weights)["failed"]
          and all(ok_rel(c) for st in train_steps for c in st["losses"].values())
          and all(not st["weights"]["failed"] for st in train_steps)
          and all(ok_abs(c) for c in evaluation.values()))
    return {"dim": dim, "unit": unit, "input_ln": ln, "batch": [per_class, per_class],
            "construction": construction, "shape_changes_vs_official_1024": shapes,
            "forward": forward, "losses": losses, "gradients": _summary(grads),
            "adam_step_weights": _summary(step_weights), "train_steps": train_steps,
            "evaluation": evaluation, "pass": ok}


def q1(official: types.ModuleType, device: str = "cpu", *,
       cases: Sequence[tuple[int, bool]] = Q1_CASES, units: Sequence[int] = (200, 32),
       seed: int = 0, per_class: int = 2, steps: int = 2, deterministic: bool = True,
       allow_einops_stand_in: bool = False) -> dict:
    """(q1) on toy inputs (module docstring); ``device`` cpu (cuda shim) or cuda; deterministic
    algorithms by default (PLAN.md (q1): "결정적")."""
    dev = torch.device(device)
    with deterministic_mode(deterministic) as det, official_urdmu(official, dev) as off:
        model_mod = off.load("model.py")
        train_mod = off.load("train.py")
        torch.manual_seed(seed)
        ref_shapes = {k: tuple(v.shape) for k, v in model_mod.WSAD(
            1024, flag="Train", a_nums=60, n_nums=60).state_dict().items()}
        rows = [_q1_case(model_mod, train_mod, dev, d, u, ln, seed, per_class, steps,
                         ref_shapes) for d, ln in cases for u in units]
        notes = list(off.stand_ins)
        commit = off.commit
    expected = {"embedding.conv_1.0.weight"}
    shape_rule = all(set(r["shape_changes_vs_official_1024"]) <= expected for r in rows) and \
        all(not r["shape_changes_vs_official_1024"] for r in rows if r["dim"] == 1024)
    ein = einops_check(notes, allow_einops_stand_in)
    return {"check": "E023 (q1) UR-DMU model / loss / gradient / optimizer step",
            "device": str(dev), "deterministic": det, "commit": commit, "stand_ins": notes,
            "einops": ein, "tolerance": TOL,
            "cases": rows, "shape_rule": shape_rule,
            "pass": all(r["pass"] for r in rows) and shape_rule and ein["ok"]}


# --- training trees (the official dataset class on exported files) --------------------------

def _file_name(dataset: str, position: int, label: int) -> str:
    """A name the official label rules read correctly (UCF: "Normal" in the file name; XD:
    "_label_A" in the path for normal videos)."""
    if dataset == "ucf":
        return f"v{position:06d}_{'Normal' if label == 0 else 'Anomaly'}_x264.npy"
    return f"v{position:06d}_label_{'A' if label == 0 else 'B1'}.npy"


def _find_line(lines: Sequence[str], text: str, start: int = 0) -> int:
    hits = [i for i in range(start, len(lines)) if lines[i].strip() == text]
    if not hits:
        raise HarnessError(f"official line {text!r} not found")
    return hits[0]


@dataclass(frozen=True)
class Edit:
    file: str
    category: str
    start: int                      # 0-based first replaced line
    end: int                        # one past the last replaced line (start == end: insertion)
    new: tuple[str, ...]

    def to_json(self, old_lines: Sequence[str]) -> dict:
        return {"file": self.file, "category": self.category,
                "lines": f"L{self.start + 1}-L{self.end}" if self.end > self.start else
                f"after L{self.start}", "old": list(old_lines[self.start:self.end]),
                "new": list(self.new)}


def loader_edits(dataset: str, lines: Sequence[str], n_abnormal: int) -> list[Edit]:
    """dataset_loader.py: the split constant (UCF 8100 / XD 9525 crop items) -> the number of
    anomalous training-part videos (data_path)."""
    k, out = SPLIT_CONST[dataset], []
    for old, new in ((f"self.vid_list = self.vid_list[{k}:]",
                      f"self.vid_list = self.vid_list[{int(n_abnormal)}:]"),
                     (f"self.vid_list = self.vid_list[:{k}]",
                      f"self.vid_list = self.vid_list[:{int(n_abnormal)}]")):
        i = _find_line(lines, old)
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        out.append(Edit("dataset_loader.py", "data_path", i, i + 1, (indent + new,)))
    return out


HOOK_IMPORT = ("from wvad.baselines.check_urdmu import official_hook as wvad_hook  "
               "# E023 (q1b) harness")


def main_edits(dataset: str, lines: Sequence[str], dim: int) -> list[Edit]:
    """<ds>_main.py edits (module docstring, categories)."""
    f = MAINS[dataset]
    i_tqdm = _find_line(lines, "from tqdm import tqdm")
    i_dim = _find_line(lines, "config.len_feature = 1024")
    i_tl = _find_line(lines, "test_loader = data.DataLoader(")
    j_tl = _find_line(lines, "worker_init_fn = worker_init_fn)", i_tl)
    i_t0 = _find_line(lines, "test(net, config, wind, test_loader, test_info, 0)")
    i_blk = _find_line(lines, "if step % 10 == 0 and step > 10:")
    end = len(lines)
    while end > i_blk and not lines[end - 1].strip():
        end -= 1
    return [
        Edit(f, "harness_io", i_tqdm + 1, i_tqdm + 1, (HOOK_IMPORT,)),
        Edit(f, "input_dimension", i_dim, i_dim + 1, (f"    config.len_feature = {int(dim)}",)),
        Edit(f, "test_off", i_tl, j_tl + 1,
             ("    test_loader = None  # E023 (q1b): test evaluation off (no test list read)",)),
        Edit(f, "test_off", i_t0, i_t0 + 1, ("    pass  # E023 (q1b): no test pass at step 0",)),
        Edit(f, "validation", i_blk, end,
             ("        wvad_hook.after_step(net, step)  # E023 (q1b): validation, not test",)),
        Edit(f, "harness_io", end, end, ("    wvad_hook.finish(net)  # E023 (q1b): last weights",)),
    ]


def apply_edits(lines: Sequence[str], edits: Sequence[Edit]) -> list[str]:
    out = list(lines)
    for e in sorted(edits, key=lambda e: (e.start, e.end), reverse=True):
        out[e.start:e.end] = list(e.new)
    return out


def _unified(name: str, old: Sequence[str], new: Sequence[str]) -> str:
    return "".join(difflib.unified_diff([s + "\n" for s in old], [s + "\n" for s in new],
                                        fromfile=f"a/{name}", tofile=f"b/{name}"))


@dataclass(frozen=True)
class Tree:
    """An official working tree outside the repository: list/<DS>_Train.list, features/,
    official_mod/ (the modified copies)."""

    root: Path
    dataset: str
    order: tuple[int, ...]          # list order -> training-part position (abnormal first)
    n_abnormal: int
    edits: tuple[dict, ...]
    diff: str

    @property
    def official_mod(self) -> Path:
        return self.root / "official_mod"


def write_tree(official: types.ModuleType, root: Path, dataset: str,
               rows: Callable[[int], np.ndarray], labels: Sequence[int], *,
               dim: int | None = None, write_main: bool = False) -> Tree:
    """Exported training-part videos (``rows(position)`` float32), the list file (absolute
    posix paths, anomalous videos first, each class in position order, as samplers.Pools) and
    the modified dataset_loader.py (and, with ``write_main``, the modified main)."""
    root = _outside_repo(Path(root))
    if dataset == "xdv" and "_label_A" in root.as_posix():
        raise HarnessError(f"{root}: '_label_A' in the folder would label every XD video normal")
    pools = samplers.Pools.from_labels(labels)
    order = tuple(pools.abnormal) + tuple(pools.normal)
    feats = root / "features"
    feats.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in order:
        path = feats / _file_name(dataset, p, int(labels[p]))
        np.save(path, np.ascontiguousarray(np.asarray(rows(p), dtype=np.float32)))
        entries.append(path.resolve().as_posix())
    (root / "list").mkdir(exist_ok=True)
    list_name = {"ucf": "UCF_Train.list", "xdv": "XD_Train.list"}[dataset]
    (root / "list" / list_name).write_text("".join(e + "\n" for e in entries),
                                           encoding="utf-8", newline="\n")
    clone = official.repo_dir(METHOD)
    mod = root / "official_mod"
    mod.mkdir(exist_ok=True)
    files = {"dataset_loader.py": lambda ls: loader_edits(dataset, ls, len(pools.abnormal))}
    if write_main:
        if dim is None:
            raise HarnessError("the modified main needs the input dimension")
        files[MAINS[dataset]] = lambda ls: main_edits(dataset, ls, int(dim))
    edits, diff = [], []
    for name, make in files.items():
        if _lf_sha256(clone / name) != OFFICIAL_SHA256[name]:
            raise HarnessError(f"{name} differs from the pinned official content")
        old = _lf_text(clone / name).split("\n")
        es = make(old)
        new = apply_edits(old, es)
        (mod / name).write_text("\n".join(new), encoding="utf-8", newline="\n")
        edits += [e.to_json(old) for e in es]
        diff.append(_unified(name, old, new))
    return Tree(root, dataset, order, len(pools.abnormal), tuple(edits), "".join(diff))


def _official_dataset(off: Official, dataset: str, *, is_normal: bool, dim: int,
                      unit: int = port.OFFICIAL_UNIT):
    """The official training dataset (constructed as ucf_main.py L33 / xd_main.py L32; cwd = the
    tree root)."""
    cls = getattr(off.top("dataset_loader"), DATASET_CLASS[dataset])
    return cls(root_dir="outputs/", mode="Train", modal="rgb", num_segments=unit,
               len_feature=dim, is_normal=is_normal)


# --- (q1c) ---------------------------------------------------------------------------------

def _loader(ds, batch: int):
    """ucf_main.py L32-L36: DataLoader(..., batch_size = 64, shuffle = True, num_workers = 0,
    worker_init_fn = None, drop_last = True)."""
    from torch.utils import data

    return data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                           worker_init_fn=None, drop_last=True)


def _window_values(bounds: np.ndarray) -> np.ndarray:
    """Window means of rows j -> j: (r[i] + r[i+1] - 1) / 2, an empty window r[i]."""
    a, b = bounds[:-1].astype(np.float64), bounds[1:].astype(np.float64)
    return np.where(a == b, a, (a + b - 1) / 2).astype(np.float32)


def _official_loop(official: types.ModuleType, tree: Tree, dataset: str, bn: int, ba: int,
                   seed: int, steps: int | None) -> dict:
    """The official loader loop (ucf_main.py L25, L32-L41, L67-L72; train.py L58-L60) on the
    official dataset class: per step the decoded positions (constant rows = position) and
    labels, and the torch generator state afterwards."""
    with official_urdmu(official, "cpu", [tree.official_mod]) as off:
        utils = off.top("utils")
        with _cwd(tree.root):
            nds = _official_dataset(off, dataset, is_normal=True, dim=2)
            ads = _official_dataset(off, dataset, is_normal=False, dim=2)
        nl, al = _loader(nds, bn), _loader(ads, ba)
        s_total = int(steps or 3 * max(len(nl), len(al)) + 1)
        utils.set_seed(seed)
        draws = []
        for step in range(1, s_total + 1):
            if (step - 1) % len(nl) == 0:
                normal_loader_iter = iter(nl)
            if (step - 1) % len(al) == 0:
                abnormal_loader_iter = iter(al)
            ninput, nlabel = next(normal_loader_iter)
            ainput, alabel = next(abnormal_loader_iter)
            data = torch.cat((ninput, ainput), 0)
            draws.append((tuple(int(v) for v in data[:, 0, 0].round()),
                          tuple(int(v) for v in torch.cat((nlabel, alabel), 0))))
        return {"draws": draws, "state": torch.get_rng_state().clone(), "steps": s_total,
                "loader_len": [len(nl), len(al)], "stand_ins": list(off.stand_ins),
                "commit": off.commit}


def _window_check(official: types.ModuleType, tree: Tree, dataset: str, wl: Sequence[int],
                  unit: int) -> dict:
    """Official window boundaries (rows j -> j through get_data; XD random_perturb) against
    derive_n.linspace_bounds."""
    from wvad.extract import derive_n

    with official_urdmu(official, "cpu", [tree.official_mod]) as off:
        utils = off.top("utils")
        with _cwd(tree.root):
            wds = _official_dataset(off, dataset, is_normal=False, dim=1)
        rows = {}
        for i, p in enumerate(tree.order):
            got = wds.get_data(i)[0][:, 0]
            want = _window_values(derive_n.linspace_bounds(int(wl[p]), unit))
            rows[int(wl[p])] = bool(np.array_equal(got, want))
        perturb = {int(n): bool(np.array_equal(
            np.asarray(utils.random_perturb(int(n), unit), dtype=np.int64),
            derive_n.linspace_bounds(int(n), unit))) for n in wl if n <= 65535}
    return {"rows": rows, "perturb": perturb}


def _means_check(official: types.ModuleType, tree: Tree, dataset: str,
                 mrows: Sequence[np.ndarray], unit: int) -> dict:
    """Official float32 window means against derive_n's (float64 means -> float32)."""
    from wvad.extract import derive_n

    spec = derive_n.variant_spec(f"d16m{unit}r")
    out = {}
    with official_urdmu(official, "cpu", [tree.official_mod]) as off:
        with _cwd(tree.root):
            mds = _official_dataset(off, dataset, is_normal=False, dim=mrows[0].shape[1])
        for i, p in enumerate(tree.order):
            want = derive_n.variant_rows(mrows[p], 16 * len(mrows[p]), spec)
            out[str(len(mrows[p]))] = compare(mds.get_data(i)[0], want)
    return out


def q1c(official: types.ModuleType, dataset: str, labels: Sequence[int],
        lengths: Sequence[int], work: Path, *, seed: int = 0, steps: int | None = None,
        window_lengths: Sequence[int] | None = None,
        batch: tuple[int, int] | None = None) -> dict:
    """(q1c) (module docstring): ``labels`` / ``lengths`` (n_v) of the training-part videos in
    position order."""
    params = dict(port.RECIPE.sampler_params)
    if batch is not None:
        params["batch_normal"], params["batch_abnormal"] = int(batch[0]), int(batch[1])
    bn, ba, unit = int(params["batch_normal"]), int(params["batch_abnormal"]), port.OFFICIAL_UNIT
    work = _outside_repo(Path(work))
    tree = write_tree(official, work / "q1c_sampler", dataset,
                      lambda p: np.full((int(lengths[p]), 2), float(p), dtype=np.float32),
                      labels)
    wl = list(window_lengths or sorted({1, 2, 3, 57, 199, 200, 201, 399, 401, 1000, 4097,
                                        *(int(n) for n in lengths)}))
    wtree = write_tree(official, work / "q1c_windows", dataset,
                       lambda p: np.arange(int(wl[p]), dtype=np.float32)[:, None],
                       [1] * len(wl))
    rng = np.random.default_rng(seed)
    mrows = [rng.normal(0.1, 1.3, (n, 6)).astype(np.float32)
             for n in (1, 7, 199, 200, 201, 333, 1999)]
    mtree = write_tree(official, work / "q1c_means", dataset, lambda p: mrows[p],
                       [1] * len(mrows))
    off_loop = _official_loop(official, tree, dataset, bn, ba, seed, steps)
    windows = _window_check(official, wtree, dataset, wl, unit)
    means = _means_check(official, mtree, dataset, mrows, unit)
    pools = samplers.Pools.from_labels(labels)
    common.seed_all(seed)
    sampler = samplers.create(port.RECIPE.sampler, pools, params)
    mine = [(tuple(int(v) for v in d.videos), tuple(int(v) for v in d.labels))
            for d in samplers.draws(sampler, off_loop["steps"])]
    state_equal = bool(torch.equal(off_loop["state"], torch.get_rng_state()))
    first_bad = next((i + 1 for i, (a, b) in enumerate(zip(off_loop["draws"], mine,
                                                            strict=True)) if a != b), None)
    bounds_ok = all(windows["rows"].values())
    perturb_ok = all(windows["perturb"].values()) if windows["perturb"] else None
    ok = first_bad is None and state_equal and bounds_ok and (dataset == "ucf" or perturb_ok)
    return {"check": "E023 (q1c) UR-DMU sampler", "dataset": dataset, "seed": seed,
            "steps": off_loop["steps"], "batch": [bn, ba],
            "pools": [len(pools.normal), len(pools.abnormal)],
            "loader_len": off_loop["loader_len"], "commit": off_loop["commit"],
            "stand_ins": off_loop["stand_ins"], "indices_equal": first_bad is None,
            "first_difference_step": first_bad, "generator_state_equal": state_equal,
            "window_bounds_equal": bounds_ok,
            "window_bounds_failed": sorted(n for n, v in windows["rows"].items() if not v),
            "window_lengths_checked": len(wl), "random_perturb_equal": perturb_ok,
            "window_means_vs_derive_n": means, "edits": list(tree.edits),
            "diff_sha256": hashlib.sha256(tree.diff.encode("utf-8")).hexdigest(),
            "pass": bool(ok)}


# --- (q1e) ---------------------------------------------------------------------------------

class _GtNumpy:
    """numpy for an official test module, except that the one np.load of its GT file returns a
    synthetic array (the test annotation is never opened)."""

    def __init__(self, gt_name: str, n_frames: int) -> None:
        self.gt_name = gt_name
        self.n_frames = int(n_frames)
        self.requests: list[str] = []

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, file, *args, **kwargs):
        if Path(file).as_posix() != self.gt_name:
            raise HarnessError(f"unexpected np.load({file!r}) in the official test function")
        self.requests.append(str(file))
        return (np.arange(self.n_frames) // 16 % 2).astype(np.float32)


class _Loader:
    """An official test_loader: ``dataset`` of the official length, iterated items."""

    def __init__(self, items: Sequence[tuple], length: int) -> None:
        self.items = list(items)
        self.dataset = range(int(length))

    def __iter__(self):
        return iter(self.items)


def official_test_scores(test_mod, dataset: str, net: nn.Module, pieces: Sequence[torch.Tensor]
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """The official test function (``test_mod.test``) on ``pieces`` (one crop each; UCF: each
    piece fed as its 10 items, XD: one batch of the 5): snippet sigmoid scores (frame_predict
    every 16th frame), the hooked cls_head logits and a record."""
    _, gt_name, items = TESTS[dataset]
    n_snip = sum(int(p.shape[0]) for p in pieces)
    label = torch.tensor([0])
    if dataset == "ucf":
        feed = [(p[None].cpu(), label) for p in pieces for _ in range(items)]
    else:
        feed = [(p[None].cpu(), label) for p in pieces]
    loader = _Loader(feed, items * len(pieces))
    fake = _GtNumpy(gt_name, 16 * n_snip)
    captured: list[torch.Tensor] = []
    hook = net.cls_head.mlp[2].register_forward_hook(
        lambda mod, inp, out: captured.append(out.detach().reshape(-1).float().cpu()))
    wind, info = _Wind(), {"step": [], "auc": [], "ap": [], "ac": []}
    saved = test_mod.np
    test_mod.np = fake
    try:
        test_mod.test(net, types.SimpleNamespace(), wind, loader, info, 0)
    finally:
        test_mod.np = saved
        hook.remove()
    frames = wind.arrays["scores"]
    if len(frames) != 16 * n_snip:
        raise HarnessError(f"{len(frames)} official frame scores for {n_snip} snippets")
    scores = np.asarray(frames[::16], dtype=np.float32)
    per = items if dataset == "ucf" else 1
    groups = [captured[i * per:(i + 1) * per] for i in range(len(pieces))]
    same = all(all(torch.equal(g[0], c) for c in g) for g in groups)
    logits = torch.cat([g[0] for g in groups]).numpy()
    return scores, logits, {"gt_requests": fake.requests, "calls": len(captured),
                            "repeated_items_equal": same}


Q1E_LENGTHS = (1, 37, 199, 200, 513, 4100)


def q1e(official: types.ModuleType, device: str = "cpu", *,
        cases: Sequence[tuple[int, bool]] = ((1792, False), (1920, True)),
        lengths: Sequence[int] = Q1E_LENGTHS, n_chunk: int = common.N_CHUNK_DEFAULT,
        units: Sequence[int] = (port.OFFICIAL_UNIT, 32), datasets: Sequence[str] = ("ucf", "xdv"),
        seed: int = 0, deterministic: bool = True, allow_einops_stand_in: bool = False,
        videos: Mapping[str, Sequence[torch.Tensor]] | None = None) -> dict:
    """(q1e) (module docstring): synthetic rows of ``lengths``, or ``videos`` (dataset ->
    processed (n, D) rows, the validation videos of ``q1e_val``; then ``cases`` holds the one
    (D, input LN) of those rows)."""
    from wvad.eval.dense_readout import sigmoid32

    dev = torch.device(device)
    m = port.METHOD
    rows = []
    with deterministic_mode(deterministic) as det, official_urdmu(official, dev) as off:
        model_mod = off.load("model.py")
        tests = {ds: off.load(TESTS[ds][0]) for ds in datasets}
        for dim, ln in cases:
            torch.manual_seed(seed)
            mine = m.build(dim, unit=units[0], input_ln=ln, scale=SCALE)
            g = torch.Generator().manual_seed(seed + 7)
            with torch.no_grad():
                for p in mine.parameters():
                    p.add_(torch.randn(p.shape, generator=g) * 0.02)
            official_net = model_mod.WSAD(dim, flag="Test", a_nums=60, n_nums=60)
            official_net.load_state_dict({k: v for k, v in mine.state_dict().items()
                                          if not k.startswith("input_norm.")})
            mine.to(dev)
            official_net.to(dev)
            m.set_mode(mine, False)
            synth = {ds: [(torch.randn(int(n), dim, generator=g) * 0.9
                           + torch.linspace(-0.5, 0.5, dim)).float() for n in lengths]
                     for ds in (datasets[:1] if videos is None else ())}
            feeds = [(x, datasets if videos is None else (ds,))
                     for ds, xs in (synth.items() if videos is None else videos.items())
                     for x in xs]
            for x, x_datasets in feeds:
                n = int(x.shape[0])
                state = torch.get_rng_state().clone()
                z = {u: common.score_video(m, mine, x, unit=u, n_chunk=n_chunk, device=dev)
                     for u in units}
                again = common.score_video(m, mine, x, unit=units[0], n_chunk=n_chunk,
                                           device=dev)
                rng_untouched = bool(torch.equal(state, torch.get_rng_state()))
                bounds = common.chunk_bounds(int(n), n_chunk)
                with torch.no_grad():
                    xin = mine.input_norm(x.to(dev)).cpu() if ln else x
                pieces = [xin[a:b] for a, b in bounds]
                for ds in x_datasets:
                    scores, logits, rec = official_test_scores(tests[ds], ds, official_net,
                                                               pieces)
                    zc = {str(u): compare(z[u], logits) for u in units}
                    sc = compare(sigmoid32(z[units[0]]), scores)
                    ok = (all(ok_abs(c) for c in zc.values()) and ok_abs(sc)
                          and rng_untouched and np.array_equal(z[units[0]], again)
                          and rec["repeated_items_equal"])
                    rows.append({"dim": dim, "input_ln": ln, "n_v": int(n), "dataset": ds,
                                 "pieces": [b - a for a, b in bounds], "logits": zc,
                                 "scores": sc, "eval_repeat_equal": bool(
                                     np.array_equal(z[units[0]], again)),
                                 "rng_untouched": rng_untouched, "official": rec, "pass": ok})
        notes, commit = list(off.stand_ins), off.commit
    ein = einops_check(notes, allow_einops_stand_in)
    return {"check": "E023 (q1e) UR-DMU inference path", "device": str(dev),
            "deterministic": det, "commit": commit,
            "inputs": "synthetic rows" if videos is None else "processed validation videos",
            "n_chunk": int(n_chunk), "units": list(units), "tolerance": TOL,
            "stand_ins": notes, "einops": ein, "cases": rows,
            "not_applicable": {"last_piece_padding": "URDMU tests the whole video (no padded "
                               "pieces; the last N_chunk piece is shorter, unpadded)",
                               "batchnorm": "no BatchNorm in UR-DMU (LayerNorm, Dropout)"},
            "pass": bool(rows) and all(r["pass"] for r in rows) and ein["ok"]}


def q1e_val(official: types.ModuleType, dataset: str, pair: str, *, paths=None, split=None,
            device: str = "cpu", n_chunk: int = common.N_CHUNK_DEFAULT, seed: int = 0,
            max_videos: int | None = None, allow_einops_stand_in: bool = False) -> dict:
    """(q1e) on the processed validation videos (PLAN.md (q1e): real-length validation videos):
    train.validation_videos (the dense training-split store through the d16m200 InputPipeline)
    of ``dataset`` / ``pair``, input LN for XCB+SigLIP, the same perturbed weights in the
    official WSAD and the port; the official test function of ``dataset`` on every video per
    N_chunk piece, both units."""
    from wvad.baselines import train as tr
    from wvad.data import load_split, split_file

    if paths is None:
        from wvad.paths import load_paths

        paths = load_paths()
    split = split if split is not None else load_split(split_file(dataset))
    variant = port.RECIPE.unit_input(port.OFFICIAL_UNIT).variant
    val = tr.validation_videos(paths, dataset, pair, variant, split, max_videos=max_videos)
    res = q1e(official, device, cases=((int(sum(val.dims)), common.input_ln(pair)),),
              n_chunk=n_chunk, datasets=(dataset,), seed=seed,
              allow_einops_stand_in=allow_einops_stand_in, videos={dataset: list(val.x)})
    for c, v in zip(res["cases"], val.indices, strict=True):
        c["video"] = int(v)
    return res | {"check": "E023 (q1e) UR-DMU on the processed validation videos",
                  "dataset": dataset, "pair": pair, "variant": variant,
                  "pipeline": val.pipeline.to_json(), "videos": len(val.x),
                  "of_validation_videos": len(split.val_indices),
                  "max_snippets": max(int(x.shape[0]) for x in val.x)}


# --- (q1b) official worker hook ------------------------------------------------------------

def _official_val_logits(net, rows: Sequence[torch.Tensor], n_chunk: int,
                         device: torch.device) -> list[np.ndarray]:
    """The official Test forward (ucf_test.py L13-L14, L35: net.eval(); net.flag = "Test";
    net(_data)) on each validation video, whole or in N_chunk pieces; the cls_head logits."""
    captured: list[torch.Tensor] = []
    hook = net.cls_head.mlp[2].register_forward_hook(
        lambda mod, inp, out: captured.append(out.detach()))
    out = []
    try:
        with torch.no_grad():
            net.eval()
            net.flag = "Test"
            for x in rows:
                parts = []
                for a, b in common.chunk_bounds(int(x.shape[0]), n_chunk):
                    captured.clear()
                    net(x[a:b][None].to(device))
                    parts.append(captured[-1].reshape(-1).float().cpu().numpy())
                out.append(np.concatenate(parts).astype(np.float32))
    finally:
        hook.remove()
        net.train()
        net.flag = "Train"
    return out


class _OfficialHook:
    """``wvad_hook`` of the modified official main (configured by $WVAD_E023_Q1B_SPEC)."""

    def __init__(self) -> None:
        self.spec: dict | None = None
        self.points: list[tuple[int, np.ndarray]] = []
        self.rows: list[torch.Tensor] | None = None

    def _load(self) -> dict:
        if self.spec is None:
            path = os.environ.get(HOOK_ENV)
            if not path:
                raise HarnessError(f"{HOOK_ENV} is not set: the hook runs in the q1b worker only")
            self.spec = _read_json(path)
            with np.load(Path(self.spec["work"]) / "val.npz") as f:
                z, counts = f["rows"], f["counts"]
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
            self.rows = [torch.from_numpy(np.ascontiguousarray(z[a:a + n]))
                         for a, n in zip(starts, counts, strict=True)]
        return self.spec

    def after_step(self, net, step: int) -> None:
        spec = self._load()
        if int(step) not in set(spec["eval_steps"]):
            return
        dev = torch.device(spec["device"])
        with common.isolated_rng(dev):
            z = _official_val_logits(net, self.rows, int(spec["n_chunk"]), dev)
        self.points.append((int(step), np.concatenate(z)))

    def finish(self, net) -> None:
        spec = self._load()
        work = Path(spec["work"])
        torch.save({k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                   work / "official_last.pt")
        z = np.stack([p[1] for p in self.points]) if self.points else np.zeros((0, 0))
        np.savez(work / "official_val.npz", z=z,
                 steps=np.asarray([p[0] for p in self.points], np.int64))


official_hook = _OfficialHook()


# --- (q1b) workers ------------------------------------------------------------------------

def _read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n", encoding="utf-8",
                   newline="\n")
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def _audit() -> set[str]:
    """Every path opened from now on in this process (audit event "open")."""
    seen: set[str] = set()

    def hook(event, args):
        if event != "open" or not args:
            return
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            try:
                seen.add(os.path.abspath(os.fsdecode(p)))
            except (TypeError, ValueError, OSError):
                pass

    sys.addaudithook(hook)
    return seen


def _det_flags(enabled: bool) -> dict:
    torch.use_deterministic_algorithms(bool(enabled))
    torch.backends.cudnn.benchmark = False
    if enabled:
        torch.backends.cudnn.deterministic = True
    return {"deterministic": bool(enabled),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _environment(device: torch.device) -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "numpy": np.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def _official_worker(official: types.ModuleType, spec_path: str) -> None:
    """The official pipeline in this process (q1b): official.session, the modified main run as
    __main__ in the tree root, per-step losses from the visdom recorder. Its command line (a
    fresh interpreter, ``_run_worker``) imports the loader and passes it in, like the README's
    harness commands."""
    import runpy

    spec = _read_json(spec_path)
    opened = _audit()
    dev = torch.device(spec["device"])
    work = Path(spec["work"])
    out: dict = {"ok": False, "error": None}
    os.environ[HOOK_ENV] = str(spec_path)
    try:
        out["deterministic"] = _det_flags(bool(spec["deterministic"]))
        with official_urdmu(official, dev, [work / "official_mod"]) as off, _cwd(work):
            argv = [spec["entry"], "--seed", str(spec["seed"]), "--model_path", "models/",
                    "--output_path", "outputs/"]
            if spec.get("iterations"):
                argv += ["--lr", f"[0.0001]*{int(spec['iterations'])}"]
            saved_argv = list(sys.argv)
            sys.argv = argv
            try:
                runpy.run_path(str(work / "official_mod" / spec["entry"]), run_name="__main__")
            finally:
                sys.argv = saved_argv
            log = list(sys.modules["visdom"].LOG)
            out["stand_ins"] = list(off.stand_ins)
            out["argv"] = argv[1:]
        losses: dict[str, list[float]] = {}
        for entry in log:
            if entry[0] == "line":
                losses.setdefault(entry[1], []).extend(float(v) for v in entry[3])
        out |= {"ok": True, "losses": losses}
    except Exception as exc:  # noqa: BLE001 - recorded, the parent decides
        out["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        out["environment"] = _environment(dev)
        out["opened"] = sorted(opened)
        _write_json(work / "official_run.json", out)


class _RecordingURDMU(port.URDMU):
    """The port with every step's losses recorded (q1b; the recipe is urdmu.RECIPE)."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[dict[str, float]] = []

    @property
    def recipe(self):
        return port.RECIPE

    def train_step(self, model, batch, optimizer, *, unit, scale):
        got = super().train_step(model, batch, optimizer, unit=unit, scale=scale)
        self.steps.append(dict(got))
        return got


def _port_worker(spec_path: str) -> None:
    """The port's training chain in this process (q1b)."""
    from wvad.baselines import train as tr
    from wvad.data.splits import parse_split
    from wvad.paths import Paths

    spec = _read_json(spec_path)
    opened = _audit()
    work = Path(spec["work"])
    dev = torch.device(spec["device"])
    method = _RecordingURDMU()
    out: dict = {"ok": False, "error": None}
    try:
        cfg = tr.TrainConfig(method=METHOD, unit=port.OFFICIAL_UNIT, dataset=spec["dataset"],
                             pair=spec["pair"], seed=int(spec["seed"]),
                             out_dir=str(work / "port"), n_chunk=int(spec["n_chunk"]),
                             deterministic=bool(spec["deterministic"]),
                             eval_points=int(spec["eval_points"]),
                             iterations=spec.get("iterations"), device=spec["device"])
        rec = tr.run(cfg, method=method, paths=Paths(feature_root=Path(spec["feature_root"])),
                     split=parse_split(spec["split"]), log=lambda s: None)
        out |= {"ok": True, "failed": rec["failed"], "deterministic_mode":
                rec["config"]["deterministic_mode"]}
    except Exception as exc:  # noqa: BLE001 - recorded, the parent decides
        out["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        out["losses"] = method.steps
        out["environment"] = _environment(dev)
        out["opened"] = sorted(opened)
        _write_json(work / "port_run.json", out)


# --- (q1b) parent -------------------------------------------------------------------------

def _roots(official_root: Path, work: Path, feature_root: Path | None,
           annotations: Sequence[Path] = ()) -> dict:
    roots = {"$WORK": Path(work), "$WVAD_OFFICIAL_ROOT": Path(official_root), "$REPO": REPO_ROOT,
             "$PYTHON_ENV": Path(sys.prefix), "$PYTHON": Path(sys.base_prefix)}
    if feature_root is not None:
        roots["$FEATURE_ROOT"] = Path(feature_root)
    for i, a in enumerate(annotations):
        roots[f"$ANNOTATIONS{i}"] = Path(a)
    roots["~"] = Path.home()
    return {k: v.resolve() for k, v in roots.items()}


def symbolic(path: str | os.PathLike[str], roots: Mapping[str, Path]) -> tuple[str, str]:
    """(root name, posix path relative to it) of the most specific root; ("", path) outside."""
    p = Path(path).resolve()
    best = None
    for name, r in roots.items():
        if p == r or r in p.parents:
            if best is None or len(r.parts) > len(roots[best].parts):
                best = name
    if best is None:
        return "", p.as_posix()
    return best, p.relative_to(roots[best]).as_posix()


_BLOCKED = {"$WVAD_OFFICIAL_ROOT": re.compile(r"(^|/)frame_label/|(^|/)list/[^/]*test", re.I),
            "$FEATURE_ROOT": re.compile(r"test", re.I)}


def audit_summary(opened: Sequence[str], roots: Mapping[str, Path]) -> dict:
    """Opened files grouped by symbolic root (no absolute path), with the blocked hits: test
    lists / frame labels in the clone, any 'test' path under the feature root, anything under an
    annotation root."""
    groups: dict[str, list[str]] = {}
    blocked = []
    for p in opened:
        name, rel = symbolic(p, roots)
        if name.startswith("$ANNOTATIONS") or (name in _BLOCKED and _BLOCKED[name].search(rel)):
            blocked.append(f"{name}/{rel}")
        groups.setdefault(name or "<other>", []).append(rel)
    out: dict = {"blocked": sorted(blocked), "counts": {k: len(v) for k, v in groups.items()}}
    for name in ("$WVAD_OFFICIAL_ROOT", "$FEATURE_ROOT", "$REPO"):
        if name in groups:
            out[name] = sorted(groups[name])
    if "$WORK" in groups:
        out["$WORK"] = sorted(r for r in groups["$WORK"] if not r.startswith("features/"))
        out["$WORK_features"] = sum(r.startswith("features/") for r in groups["$WORK"])
    out["<other>_names"] = sorted({Path(r).name for r in groups.get("<other>", [])})
    out["<other>_count"] = len(groups.get("<other>", []))
    return out


def _plan_hashes() -> dict:
    raw = PLAN_FILE.read_bytes()
    return {"file": PLAN_FILE.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def _repo_head() -> str | None:
    res = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    return res.stdout.strip() or None


def _export(inputs, mode: str, paths, dataset: str, pair: str, split):
    """rows(position) of the official files: the port's own 200-row inputs, or the processed
    snippets of the training-part video."""
    if mode == "port_rows":
        return lambda p: inputs.train.features[p].numpy()
    from wvad.data.catalog import ENCODER_PAIRS, dense_dir
    from wvad.data.dense import DenseSet

    dense = DenseSet([dense_dir(paths, dataset, e) for e in ENCODER_PAIRS[pair]])
    train_idx = tuple(int(i) for i in split.train_indices)
    return lambda p: inputs.pipeline.apply(dense.video(train_idx[p])).numpy()


def _rows_check(official: types.ModuleType, tree: Tree, dataset: str, dim: int,
                want: Callable[[int], np.ndarray]) -> dict:
    """The official dataset's 200-window rows of every exported video against the port's rows."""
    worst, unequal = 0.0, 0
    with official_urdmu(official, "cpu", [tree.official_mod]) as off, _cwd(tree.root):
        ads = _official_dataset(off, dataset, is_normal=False, dim=dim)
        nds = _official_dataset(off, dataset, is_normal=True, dim=dim)
        items = [(ads, i) for i in range(len(ads))] + [(nds, i) for i in range(len(nds))]
        if len(items) != len(tree.order):
            raise HarnessError(f"{len(items)} official items for {len(tree.order)} videos")
        for (ds, i), p in zip(items, tree.order, strict=True):
            got = ds.get_data(i)[0]
            w = np.asarray(want(p), dtype=np.float32)
            if not np.array_equal(got, w):
                unequal += 1
                worst = max(worst, float(np.abs(got.astype(np.float64) - w).max()))
    return {"videos": len(tree.order), "unequal": unequal, "max_abs": worst,
            "bit_equal": unequal == 0}


def _run_worker(role: str, spec_path: Path, log_path: Path, timeout: float | None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"),
                                         *filter(None, [env.get("PYTHONPATH")])])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"         # (q1b) deterministic mode
    env["PYTHONDONTWRITEBYTECODE"] = "1"               # no cache files in the clone
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    loader = "from wvad.baselines import official as _loader; " if role == "official" else ""
    fn = {"official": "_official_worker", "port": "_port_worker"}[role]
    args = f"_loader, {str(spec_path)!r}" if role == "official" else repr(str(spec_path))
    code = f"{loader}from wvad.baselines.check_urdmu import {fn}; {fn}({args})"
    with open(log_path, "w", encoding="utf-8") as log:
        res = subprocess.run([sys.executable, "-c", code], env=env, stdout=log,
                             stderr=subprocess.STDOUT, timeout=timeout, check=False)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return {"returncode": res.returncode, "log_tail": None if res.returncode == 0 else tail,
            "nondeterministic_op": res.returncode != 0 and
            "does not have a deterministic implementation" in tail}


def compare_losses(official: Mapping[str, Sequence[float]], mine: Sequence[Mapping[str, float]]
                   ) -> dict:
    """Per-step losses: bit equality, the first difference, max relative difference of the
    total over the first 50 steps and of every term over all steps."""
    n = len(mine)
    lens = {k: len(official.get(k, [])) for k in LOSS_KEYS}
    if any(v != n for v in lens.values()):
        return {"steps": n, "official_steps": lens, "bit_equal": False,
                "first_difference": None, "max_rel_total_first50": math.inf, "max_rel": {}}
    first, max_rel = None, {}
    for k in LOSS_KEYS:
        pk = "total" if k == "total_loss" else k
        rels = []
        for s in range(n):
            c = _float_cmp(mine[s][pk], official[k][s])
            rels.append(c["rel"])
            if not c["bit_equal"] and (first is None or s + 1 < first["step"]):
                first = {"step": s + 1, "term": k}
        max_rel[k] = max(rels) if rels else 0.0
    first50 = max((_float_cmp(mine[s]["total"], official["total_loss"][s])["rel"]
                   for s in range(min(n, Q1B_FIRST))), default=0.0)
    return {"steps": n, "bit_equal": first is None, "first_difference": first,
            "max_rel_total_first50": first50, "max_rel": max_rel}


def _vaucs(z: np.ndarray, counts, totals, labels) -> dict[str, list[float]]:
    from wvad.baselines.train import point_aucs

    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    out: dict[str, list[float]] = {"sigma": [], "logit": []}
    for e in range(z.shape[0]):
        pa = point_aucs([z[e, a:a + n] for a, n in zip(starts, counts, strict=True)], totals,
                        labels)
        for k in out:
            out[k].append(pa[k])
    return out


def compare_runs(work: Path) -> dict:
    """Losses, last weights, validation logits and V-AUCs of the two runs in ``work``."""
    off, mine = _read_json(work / "official_run.json"), _read_json(work / "port_run.json")
    losses = compare_losses(off.get("losses") or {}, mine.get("losses") or [])
    a = torch.load(work / "official_last.pt", map_location="cpu", weights_only=True)
    b = torch.load(work / "port" / "ckpt_last.pt", map_location="cpu",
                   weights_only=True)["model_state_dict"]
    wc = {k: compare(b[k], a[k]) if k in b else {"bit_equal": False, "max_abs": math.inf,
                                                  "rel": math.inf} for k in a}
    extra = sorted(k for k in b if k not in a)
    with np.load(work / "official_val.npz") as f:
        z_off, steps_off = f["z"], f["steps"]
    with np.load(work / "port" / "val_scores.npz") as f:
        z_mine, counts = f["z"], f["counts"]
        totals, labels, steps = f["total_frames"], f["labels"], f["eval_steps"]
    zc = compare(z_mine, z_off)
    va_off = _vaucs(z_off, counts, totals, labels) if z_off.shape == z_mine.shape else None
    va_mine = _vaucs(z_mine, counts, totals, labels)
    sel = {}
    if va_off is not None:
        for r in ("sigma", "logit"):
            sel[r] = {"official": float(max(va_off[r])), "port": float(max(va_mine[r])),
                      "official_point": int(np.argmax(va_off[r])) + 1,
                      "port_point": int(np.argmax(va_mine[r])) + 1}
            sel[r]["diff_points"] = 100.0 * (sel[r]["port"] - sel[r]["official"])
    return {"losses": losses,
            "weights": {**_summary(wc), "extra_port_keys": extra},
            "validation": {"eval_steps_equal": bool(np.array_equal(steps, steps_off)),
                           "logits": zc, "v_auc_official": va_off, "v_auc_port": va_mine,
                           "v_auc_equal": va_off == va_mine, "selected": sel}}


def seed_verdict(cmp: Mapping) -> dict:
    """PLAN.md (q1b): bit equality of losses, last weights and validation; otherwise the
    numeric rule at seed level (first 50 steps' total loss rel. diff <= 1e-5 and |selected
    sigma V-AUC difference| <= 1.0 point); the 3-seed mean rule is ``dataset_verdict``."""
    losses, val = cmp["losses"], cmp["validation"]
    bit = (losses["bit_equal"] and cmp["weights"]["bit_equal"]
           and not cmp["weights"]["extra_port_keys"] and val["logits"]["bit_equal"]
           and val["v_auc_equal"] and val["eval_steps_equal"])
    sel = val["selected"].get("sigma")
    numeric = (losses["max_rel_total_first50"] <= Q1B_LOSS_REL and sel is not None
               and abs(sel["diff_points"]) <= Q1B_VAUC_SEED)
    reasons = []
    if not bit:
        if losses["max_rel_total_first50"] > Q1B_LOSS_REL:
            reasons.append("first-50 total loss relative difference > 1e-5")
        if sel is None or abs(sel["diff_points"]) > Q1B_VAUC_SEED:
            reasons.append("selected V-AUC difference > 1.0 point")
    return {"criterion": "bit" if bit else "numeric", "pass": bool(bit or numeric),
            "first_difference": losses["first_difference"], "cause": None,
            "reasons": reasons}


def dataset_verdict(records: Sequence[Mapping]) -> dict:
    """The (q1b) verdict of one dataset x pair over its development seeds."""
    if not records:
        return {"pass": False, "reason": "no runs"}
    if any(r.get("selftest") for r in records):
        return {"pass": False, "reason": "self-test records are not (q1b) runs"}
    vs = [r["verdict"] for r in records]
    if all(v["criterion"] == "bit" and v["pass"] for v in vs):
        return {"pass": True, "criterion": "bit", "seeds": [r["seed"] for r in records]}
    diffs = [r["comparison"]["validation"]["selected"].get("sigma", {}).get("diff_points")
             for r in records]
    if any(d is None for d in diffs):
        return {"pass": False, "criterion": "numeric", "reason": "missing V-AUC"}
    mean = float(np.mean(diffs))
    ok = all(v["pass"] for v in vs) and abs(mean) <= Q1B_VAUC_MEAN
    return {"pass": bool(ok), "criterion": "numeric", "seeds": [r["seed"] for r in records],
            "mean_diff_points": mean, "per_seed_diff_points": diffs}


def q1b(official: types.ModuleType, work: str | os.PathLike[str], *, dataset: str,
        pair: str = "xcb_xcl", seed: int = 0, mode: str = GATE_OF_RECORD,
        device: str | None = None, iterations: int | None = None,
        eval_points: int = common.EVAL_POINTS, n_chunk: int = common.N_CHUNK_DEFAULT,
        paths=None, split=None, deterministic: bool = True, selftest: bool = False,
        timeout: float | None = None) -> dict:
    """(q1b) one development run (module docstring); returns the record."""
    from wvad.baselines import train as tr
    from wvad.data import load_split, split_file

    if mode not in MODES:
        raise HarnessError(f"mode {mode!r} not in {MODES}")
    if common.input_ln(pair):
        raise HarnessError(f"(q1b) runs on XCB+XCL (PLAN.md); {pair} has the input LN, which the "
                           "official pipeline does not have")
    work = _outside_repo(Path(work))
    if (work / "record.json").exists():
        raise HarnessError(f"{work} holds a finished run: use a new folder")
    work.mkdir(parents=True, exist_ok=True)
    if paths is None:
        from wvad.paths import load_paths

        paths = load_paths()
    split = split if split is not None else load_split(split_file(dataset))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    commit = official.check_commit(METHOD)
    files = official_files(official)
    inputs = tr.load_inputs(paths, dataset, pair, port.RECIPE.unit_input(port.OFFICIAL_UNIT),
                            split)
    labels = [int(y) for y in inputs.train.labels]
    rows = _export(inputs, mode, paths, dataset, pair, split)
    tree = write_tree(official, work, dataset, rows, labels, dim=inputs.dim, write_main=True)
    rows_check = _rows_check(official, tree, dataset, inputs.dim,
                             lambda p: inputs.train.features[p].numpy())
    allz = [x.numpy() for x in inputs.val]
    np.savez(work / "val.npz", rows=np.concatenate(allz),
             counts=np.asarray([len(x) for x in allz], np.int64))
    steps = int(iterations or port.RECIPE.iterations)
    spec = {"work": str(work), "dataset": dataset, "pair": pair, "seed": int(seed),
            "device": str(dev), "deterministic": bool(deterministic),
            "iterations": iterations, "eval_points": int(eval_points),
            "eval_steps": list(common.eval_steps(steps, int(eval_points))),
            "n_chunk": int(n_chunk), "entry": MAINS[dataset],
            "feature_root": str(Path(paths.feature_root).resolve()),
            "split": {"train_indices": list(split.train_indices),
                      "val_indices": list(split.val_indices), "labels": list(split.labels),
                      "meta": dict(split.meta)}}
    variant = inputs.variant
    inputs = allz = None                     # released before the workers run
    _write_json(work / "spec.json", spec)
    runs = {"official": _run_worker("official", work / "spec.json", work / "official.log",
                                    timeout),
            "port": _run_worker("port", work / "spec.json", work / "port.log", timeout)}
    roots = _roots(official.root(), work, Path(paths.feature_root),
                   tuple(paths.annotations.values()))
    record: dict = {
        "check": "E023 (q1b) official pipeline", "method": METHOD, "dataset": dataset,
        "pair": pair, "seed": int(seed), "mode": mode, "device": str(dev),
        "role": "gate of record" if mode == GATE_OF_RECORD else "extra evidence",
        "selftest": bool(selftest or iterations is not None
                         or int(eval_points) != common.EVAL_POINTS),
        "plan": _plan_hashes(), "repo_head": _repo_head(),
        "official": {"repo_key": METHOD, "root": "$WVAD_OFFICIAL_ROOT",
                     "repo_dir": official.REPOS[METHOD], "commit": commit["commit"],
                     "entry": MAINS[dataset], "files_sha256_lf": files},
        "modifications": {"folder": "$WORK/official_mod", "edits": list(tree.edits),
                          "categories": sorted({e["category"] for e in tree.edits}),
                          "only_allowed_categories": all(e["category"] in REAL_CATEGORIES
                                                         for e in tree.edits),
                          "diff_sha256": hashlib.sha256(tree.diff.encode("utf-8")).hexdigest()},
        "paths": {"work": "$WORK", "features": "$WORK/features",
                  "list": f"$WORK/list/{'UCF' if dataset == 'ucf' else 'XD'}_Train.list",
                  "port_out": "$WORK/port", "feature_root": "$FEATURE_ROOT"},
        "inputs": {"variant": variant, "train_videos": len(labels),
                   "n_normal": labels.count(0), "n_abnormal": labels.count(1),
                   "val_videos": len(split.val_indices), "split_sha256": split.sha256,
                   "official_rows_vs_port_rows": rows_check},
        "schedule": {"steps": steps, "eval_points": int(eval_points),
                     "eval_steps": spec["eval_steps"], "n_chunk": int(n_chunk),
                     "deterministic": bool(deterministic)},
        "workers": runs}
    (work / "official_mod.diff").write_text(tree.diff, encoding="utf-8", newline="\n")
    if runs["official"]["returncode"] or runs["port"]["returncode"]:
        record["verdict"] = {"criterion": None, "pass": False,
                             "reasons": ["a worker failed (workers.*.log_tail)"],
                             "nondeterministic_fallback_needed":
                                 any(r["nondeterministic_op"] for r in runs.values())}
    else:
        off_run, port_run = _read_json(work / "official_run.json"), _read_json(
            work / "port_run.json")
        record["stand_ins"] = off_run.get("stand_ins")
        record["einops"] = einops_check(record["stand_ins"] or [])
        record["environment"] = off_run.get("environment")
        record["opened_files"] = {"official": audit_summary(off_run["opened"], roots),
                                  "port": audit_summary(port_run["opened"], roots)}
        record["comparison"] = compare_runs(work)
        record["verdict"] = seed_verdict(record["comparison"])
        blocked = record["opened_files"]["official"]["blocked"] + \
            record["opened_files"]["port"]["blocked"]
        if blocked or not record["modifications"]["only_allowed_categories"]:
            record["verdict"]["pass"] = False
            record["verdict"]["reasons"].append("blocked file opened or edit outside the "
                                                "allowed categories")
        if not record["einops"]["ok"]:
            record["verdict"]["pass"] = False
            record["verdict"]["reasons"].append("einops stand-in: the head split would be "
                                                "compared with itself (install einops)")
    _write_json(work / "record.json", record)
    return record


def merge_record(record: Mapping, results: Path = Q1B_RESULTS) -> dict:
    """Put a (q1b) record into results/e023_q1b.json (methods.urdmu.runs
    "<ds>/<pair>/s<seed>/<mode>", under a lock) and refresh that dataset x pair's verdict from
    the snippets runs (the gate of record) and its evidence from the port_rows runs."""
    if record.get("selftest") and Path(results).resolve() == Q1B_RESULTS.resolve():
        raise HarnessError("self-test records are not written into results/e023_q1b.json")
    mode = record.get("mode", GATE_OF_RECORD)
    group = f"{record['dataset']}/{record['pair']}"
    key = f"{group}/s{record['seed']}/{mode}"

    def merge(data: dict) -> dict:
        entry = data.setdefault("methods", {}).setdefault(METHOD, {"runs": {}, "verdicts": {}})
        entry.setdefault("evidence", {})
        entry["runs"][key] = json.loads(json.dumps(dict(record), default=_json_default))
        for m, target in ((GATE_OF_RECORD, "verdicts"), ("port_rows", "evidence")):
            runs = [r for k, r in sorted(entry["runs"].items())
                    if k.startswith(group + "/") and k.endswith("/" + m)]
            if runs:
                entry[target][group] = dataset_verdict(runs) | {"mode": m}
        return data

    from wvad.baselines.common import update_json

    return update_json(results, merge, default={
        "check": "E023 (q1b) official pipeline (PLAN.md 관문)", "methods": {}},
        json_default=_json_default)


# --- command line ---------------------------------------------------------------------------

def _training_part(dataset: str, pair: str):
    """Labels and n_v of the training-part videos (split order; training split only)."""
    from wvad.data import load_split, split_file
    from wvad.data.catalog import ENCODER_PAIRS, dense_dir
    from wvad.data.dense import DenseSet
    from wvad.paths import load_paths

    paths = load_paths()
    split = load_split(split_file(dataset))
    dense = DenseSet([dense_dir(paths, dataset, e) for e in ENCODER_PAIRS[pair][:1]])
    from wvad.extract import grid

    idx = list(split.train_indices)
    return ([int(dense.labels[v]) for v in idx],
            [grid.n_snippets(int(dense.total_frames[v])) for v in idx])


def main(argv: Sequence[str] | None = None, *, official: types.ModuleType | None = None
         ) -> None:
    if official is None:
        raise SystemExit("check_urdmu needs the official loader passed in: python -c \"import "
                         "sys; from wvad.baselines import official, check_urdmu as c; "
                         "c.main(sys.argv[1:], official=official)\" <check> ...")
    p = argparse.ArgumentParser(prog="check_urdmu",
                                description="E023 UR-DMU equivalence checks (q1, q1c, q1e, q1b)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("q1")
    a1.add_argument("--device", default="cpu")
    a1.add_argument("--out", type=Path, required=True)
    ac = sub.add_parser("q1c")
    ac.add_argument("--dataset", required=True, choices=sorted(MAINS))
    ac.add_argument("--pair", default="xcb_xcl")
    ac.add_argument("--work", type=Path, required=True)
    ac.add_argument("--seed", type=int, default=0)
    ac.add_argument("--out", type=Path, required=True)
    ae = sub.add_parser("q1e")
    ae.add_argument("--device", default="cpu")
    ae.add_argument("--n_chunk", type=int, default=common.N_CHUNK_DEFAULT)
    ae.add_argument("--out", type=Path, required=True)
    av = sub.add_parser("q1e_val")
    av.add_argument("--dataset", required=True, choices=sorted(MAINS))
    av.add_argument("--pair", required=True, choices=("xcb_xcl", "xcb_siglip"))
    av.add_argument("--device", default="cpu")
    av.add_argument("--n_chunk", type=int, default=common.N_CHUNK_DEFAULT)
    av.add_argument("--max_videos", type=int, help="the first K validation videos and the longest")
    av.add_argument("--paths")
    av.add_argument("--out", type=Path, required=True)
    ab = sub.add_parser("q1b")
    ab.add_argument("--dataset", required=True, choices=sorted(MAINS))
    ab.add_argument("--pair", default="xcb_xcl")
    ab.add_argument("--seed", type=int, required=True)
    ab.add_argument("--work", type=Path, required=True)
    ab.add_argument("--mode", default=GATE_OF_RECORD, choices=MODES,
                    help="snippets: the gate of record; port_rows: extra evidence")
    ab.add_argument("--device", default=None)
    ab.add_argument("--n_chunk", type=int, default=common.N_CHUNK_DEFAULT)
    ab.add_argument("--record", type=Path, default=None,
                    help="merge the record into this file (e.g. results/e023_q1b.json)")
    a = p.parse_args(argv)
    if a.cmd == "q1":
        res = q1(official, a.device)
    elif a.cmd == "q1c":
        labels, lengths = _training_part(a.dataset, a.pair)
        res = q1c(official, a.dataset, labels, lengths, a.work, seed=a.seed)
    elif a.cmd == "q1e":
        res = q1e(official, a.device, n_chunk=a.n_chunk)
    elif a.cmd == "q1e_val":
        from wvad.paths import load_paths

        res = q1e_val(official, a.dataset, a.pair, paths=load_paths(a.paths), device=a.device,
                      n_chunk=a.n_chunk, max_videos=a.max_videos)
    else:
        res = q1b(official, a.work, dataset=a.dataset, pair=a.pair, seed=a.seed, mode=a.mode,
                  device=a.device, n_chunk=a.n_chunk)
        if a.record is not None:
            merge_record(res, a.record)
    if a.cmd != "q1b":
        _write_json(a.out, res)
    verdict = res.get("verdict", {}).get("pass") if a.cmd == "q1b" else res["pass"]
    print(f"[check_urdmu] {a.cmd}: pass={verdict}")


if __name__ == "__main__":
    main()
