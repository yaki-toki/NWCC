"""E023 equivalence harness of the RTFM port (PLAN.md "관문" (q1), (q1b), (q1c), (q1e), "구현").
New code (add-only rule); never imported by the training or scoring code.

The official code (results/e023_licences.json "rtfm", no licence: never copied here) is loaded
from its clone under WVAD_OFFICIAL_ROOT through the ``official`` loader module
(wvad.baselines.official), which every entry point takes as an argument: no source file under
src/ but official.py imports it (import-graph test), so the callers - the tests and the command
line below - pass it in. Necessary modifications are applied in memory to the text read from the
clone (nothing is written into the clone or this repository): edits anchored on a line and a
short token, after the file's sha256 is checked against OFFICIAL_SHA256 (the pinned commit);
their unified diff (the work folder written as "<work_dir>") is hashed into the records.

    model.py    input dimension D (4 tokens: the dilated / 1x1 convolutions' input width, the
                fusing convolution's output width and its BatchNorm, Aggregate(len_feature));
                the crop count of the normal / abnormal feature split (``batch_size*10`` ->
                ``batch_size*ncrops``, single-crop input)
    train.py    on a CPU-only host only: ``label.cuda()`` -> the scores' device
    utils.py    no ``import visdom`` (logging server, not installed); ``np.int`` -> ``int``
                (removed from numpy >= 1.24; the same type)
    dataset.py  data path: the training list file and its abnormal count (official: 810)
    main.py     ((q1b) only) the test import, test loader and test calls removed, validation
                at the port's evaluation steps instead (isolated generators, PLAN.md "시험 평가는
                끄거나 검증으로"), the visdom Visualizer replaced by a recorder of the plotted
                losses, a seed call before the network is built, checkpoint paths in the work
                folder

Gates (each returns a JSON-able dict with "pass")
    check_q1     construction (same seed -> identical state_dict: the official constructor
                 order), train-mode forward, loss and gradients (the official ``train()`` with a
                 gradient probe in place of the optimizer), one Adam step through the official
                 ``train()`` against the port's ``train_step``; same weights, same RNG state;
                 at the official configuration (D 2048, 10 crops, unmodified model.py) and at
                 the adapted one (D, 1 crop, input LN). ``adaptation_shapes``: the adaptation
                 changes only the D-dependent shapes
    check_q1c    the official loader loop (patched official Dataset over position files, the
                 official DataLoader arguments, main.py's re-creation rule, normal then abnormal
                 batch) against the port's sampler from the same seed: positions and labels
    check_q1e    real-length synthetic inputs, same weights (BatchNorm statistics and affine
                 parameters randomised: eval-mode BN and the non-local branch exercised), eval
                 mode, fp32: the official inference (test_10crop.py L11-L19 on the official
                 forward, fc3 captured by a hook) on every N_chunk piece against
                 ``common.score_video``; a piece shorter than k = 3 snippets is compared with
                 the official forward's score path on its modules; input LN in front of the
                 official function for XCB+SigLIP; |delta| <= 1e-6 on logits and scores.
                 ``check_q1e_val``: the same on the processed validation videos of a dataset
                 and pair (real lengths and value ranges: the chain's InputPipeline, the input
                 LayerNorm for XCB+SigLIP)
    run_q1b      the official main.py (patched as above) and the port's ``wvad.baselines.train``
                 chain on the same training inputs, seed and device, deterministic mode:
                 per-iteration losses, final weights, validation logits and V-AUC at every
                 evaluation point; the plan's bit / numeric / non-deterministic criteria
                 (``q1b_verdict``, ``q1b_summary``); the opened-file audit (sys.addaudithook)
                 refuses test lists, ground truth and test stores; ``write_q1b_record`` merges
                 the compact entry into results/e023_q1b.json (key methods.rtfm; under a lock).
                 Inputs: "snippets" (default; the gate of record, PLAN.md 기록 2026-10-19: the
                 official process_feat windows the processed snippets itself, float32 means
                 against derive_n's float64, so the numeric rule of R3-6 decides when bits
                 differ; a failure is reported, the mode is never switched) and "rows" (extra
                 evidence: the official Dataset reads the port's d16m32 rows, its windowing an
                 identity)

Command line (the loader is injected, see above; RUN abbreviates the prefix)::

    RUN = python -c "import sys; from wvad.baselines import official, check_rtfm;
                     check_rtfm.main(sys.argv[1:], official=official)"
    RUN q1b --dataset ucf --pair xcb_xcl --seed 0 --work_dir <outside the repository>
            [--input snippets|rows] [--mode deterministic|nondeterministic]
            [--record results/e023_q1b.json]       (CUDA: CUBLAS_WORKSPACE_CONFIG=:4096:8)
    RUN q1 --out <json>
    RUN q1c --dataset ucf --steps 200 --work_dir <dir> --out <json>
    RUN q1e --out <json>
    RUN q1e_val --dataset ucf --pair xcb_xcl|xcb_siglip [--max_videos K] [--device cuda]
            [--paths FILE] --out <json>
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import difflib
import hashlib
import io
import json
import math
import os
import sys
import types
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wvad.baselines import samplers
from wvad.baselines import train as tr
from wvad.baselines.common import (
    N_CHUNK_DEFAULT,
    Batch,
    chunk_bounds,
    deterministic,
    eval_steps,
    isolated_rng,
    score_video,
    seed_all,
    select_point,
    update_json,
)
from wvad.baselines.common import input_ln as pair_input_ln
from wvad.baselines.recipes import E023_DIR, RESULTS
from wvad.baselines.rtfm import METHOD, RECIPE, RTFM, rtfm_cost
from wvad.eval.dense_readout import sigmoid32
from wvad.paths import REPO_ROOT

KEY = "rtfm"
OFFICIAL_DIM, OFFICIAL_CROPS, OFFICIAL_BATCH = 2048, 10, 32
OFFICIAL_SHA256 = {                          # LF-normalised (CRLF -> LF), commit cf938f4c
    "config.py": "f4b4db1adb8d807d7e1c324bd5f0027bdec15bca978e32d36f2193c352b0bcae",
    "dataset.py": "1f0fba5cec128db185d72d6dd3fd4e18ad0b5cfb3632c7d36f86b9a7d019353f",
    "main.py": "dbfe8ceb561037ea58ed6c5a653811dd21f0d6e121849092da2808be3a8ce086",
    "model.py": "a545c7f9e39cd8f589c64404aba2fdbca6b2714faab6dfc4fa683a85af58bd7b",
    "option.py": "2ac5d30d7d2e1c7880484ae97d02801e5cdf8a020138ecb92b174c85bddbe1ee",
    "train.py": "20bad7116f87b28f84395e39b2b02dff240644fc5f111fe791435ad4e895fdb1",
    "utils.py": "ee2dcf9e3f4466fe7e040f5c0cf01e7a0813515b2d2b1de2d8fa921d2964f62b",
}
BARE = ("model", "train", "utils", "dataset", "option", "config", "test_10crop")
PLAN_FILE = E023_DIR / "PLAN.md"
Q1B_FILE = RESULTS / "e023_q1b.json"
TOL = 1e-6                      # (q1) relative loss / gradient error, (q1e) |delta|
Q1B_REL, Q1B_FIRST = 1e-5, 50   # (q1b) numeric rule: first 50 iterations' relative loss diff
Q1B_SEED_PT, Q1B_MEAN_PT = 1.0, 0.5
NONDET_WINDOW = 500
SCALE = {"margin": float(RECIPE.scale_constants["margin"])}


class HarnessError(RuntimeError):
    """The clone is not the pinned code, an edit does not apply, or a run is refused."""


# --- in-memory edits of the official files ----------------------------------------------------

@dataclass(frozen=True)
class Edit:
    file: str
    first: int                  # 1-based line, inclusive
    last: int
    old: str | None             # token replaced (exactly once in the lines); None: replace lines
    new: str
    reason: str


@dataclass(frozen=True)
class DataPatch:
    list_file: Path             # the training list (abnormal videos first)
    n_abnormal: int


def model_edits(feature_dim: int, ncrops: int) -> list[Edit]:
    out = []
    if int(feature_dim) != OFFICIAL_DIM:
        why = f"input dimension {feature_dim}"
        out += [Edit("model.py", 137, 137, "in_channels=2048", f"in_channels={feature_dim}", why),
                Edit("model.py", 143, 143, "out_channels=2048", f"out_channels={feature_dim}",
                     why),
                Edit("model.py", 146, 146, "BatchNorm1d(2048)", f"BatchNorm1d({feature_dim})",
                     why),
                Edit("model.py", 181, 181, "len_feature=2048", f"len_feature={feature_dim}", why)]
    if int(ncrops) != OFFICIAL_CROPS:
        why = f"{ncrops}-crop input (10 crops hard-coded in the feature split)"
        out += [Edit("model.py", 214, 214, "batch_size*10", "batch_size*ncrops", why),
                Edit("model.py", 217, 217, "batch_size*10", "batch_size*ncrops", why)]
    return out


def train_edits(cpu_labels: bool) -> list[Edit]:
    return [Edit("train.py", 67, 67, "label.cuda()", "label.to(score.device)",
                 "CPU-only host: labels on the scores' device")] if cpu_labels else []


def utils_edits() -> list[Edit]:
    return [Edit("utils.py", 1, 1, None, "", "visdom (logging server) not installed, unused"),
            Edit("utils.py", 35, 35, "dtype=np.int)", "dtype=int)",
                 "np.int (alias of int) removed in numpy >= 1.24")]


def dataset_edits(data: DataPatch) -> list[Edit]:
    why = "data path: the E023 training list"
    return [Edit("dataset.py", 23, 23, "'list/ucf-i3d.list'", repr(Path(data.list_file).as_posix()),
                 why),
            Edit("dataset.py", 47, 47, "[810:]", f"[{int(data.n_abnormal)}:]", why),
            Edit("dataset.py", 51, 51, "[:810]", f"[:{int(data.n_abnormal)}]", why)]


def main_edits() -> list[Edit]:
    off = "test evaluation off (PLAN.md (q1b))"
    return [Edit("main.py", 8, 8, None, "", off),
            Edit("main.py", 14, 14, None, "viz = E023_HOOKS['viz']",
                 "the visdom Visualizer replaced by a recorder of the plotted losses"),
            Edit("main.py", 26, 28, None, "    test_loader = None", off),
            Edit("main.py", 30, 30, "model = Model(", "E023_HOOKS['seed'](); model = Model(",
                 "the (q1b) seed, set before the network is built"),
            Edit("main.py", 37, 37, "'./ckpt'", "E023_HOOKS['ckpt']", "output path: work folder"),
            Edit("main.py", 38, 38, "'./ckpt'", "E023_HOOKS['ckpt']", "output path: work folder"),
            Edit("main.py", 46, 46, None, "", off),
            Edit("main.py", 65, 74, None, "        E023_HOOKS['evaluate'](step, model)",
                 "validation at the port's evaluation steps instead of the test AUC"),
            Edit("main.py", 75, 75, "'./ckpt/' + args.model_name + 'final.pkl'",
                 "E023_HOOKS['final']", "output path: work folder")]


def apply_edits(text: str, edits: Sequence[Edit]) -> str:
    """The text with every edit applied (bottom-up, so line numbers stay those of the file)."""
    lines = text.split("\n")
    for e in sorted(edits, key=lambda e: e.first, reverse=True):
        if not 1 <= e.first <= e.last <= len(lines):
            raise HarnessError(f"{e.file} L{e.first}-L{e.last} outside the file")
        if e.old is None:
            lines[e.first - 1:e.last] = e.new.split("\n")
            continue
        seg = "\n".join(lines[e.first - 1:e.last])
        if seg.count(e.old) != 1:
            raise HarnessError(f"{e.file} L{e.first}: token {e.old!r} found {seg.count(e.old)}x")
        lines[e.first - 1:e.last] = seg.replace(e.old, e.new).split("\n")
    return "\n".join(lines)


def unified_diff(name: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(keepends=True),
                                        after.splitlines(keepends=True), f"a/{name}",
                                        f"b/{name}"))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    """sha256 of the file with CRLF normalised to LF (a Windows checkout with core.autocrlf
    stores CRLF; the pinned values are of the committed LF content)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# --- loading the official code ----------------------------------------------------------------

@dataclass
class OfficialRTFM:
    """The official modules of one session (patched as listed in ``edits``)."""

    modules: dict[str, ModuleType]
    base: Path
    commit: str | None
    edits: tuple[Edit, ...]
    diff: str
    sources: dict[str, str]                     # file -> patched text

    def __getattr__(self, name: str) -> ModuleType:
        mods = self.__dict__.get("modules", {})
        if name in mods:
            return mods[name]
        raise AttributeError(name)

    def diff_sha256(self, work_dir: Path | None = None) -> str:
        text = self.diff if work_dir is None else self.diff.replace(
            Path(work_dir).as_posix(), "<work_dir>")
        return _sha_text(text)

    def describe(self, work_dir: Path | None = None) -> dict:
        return {"commit": self.commit, "diff_sha256": self.diff_sha256(work_dir),
                "edits": [{"file": e.file, "lines": f"L{e.first}-L{e.last}", "reason": e.reason}
                          for e in self.edits]}


@contextlib.contextmanager
def _bare_names(names: Sequence[str]) -> Iterator[None]:
    """Evict the official files' bare module names for the session, restore them after."""
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    try:
        yield
    finally:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


@contextlib.contextmanager
def _no_bytecode() -> Iterator[None]:
    """No __pycache__ written into the clone while its files are loaded."""
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = saved


def _exec_module(private: str, path: Path, text: str, bare: str) -> ModuleType:
    module = types.ModuleType(private)
    module.__file__ = str(path)
    sys.modules[private] = module
    sys.modules[bare] = module
    exec(compile(text, str(path), "exec"), module.__dict__)  # noqa: S102 - pinned official code
    return module


@contextlib.contextmanager
def official_rtfm(official: ModuleType, *, feature_dim: int = OFFICIAL_DIM,
                  ncrops: int = OFFICIAL_CROPS, cpu_labels: bool = False,
                  data: DataPatch | None = None, files: Sequence[str] = ("model", "train"),
                  check_commit: bool = True) -> Iterator[OfficialRTFM]:
    """The official modules ``files`` (stems, loaded in dependency order) of the clone at the
    pinned commit, edited in memory; while open, their bare names resolve to them (so main.py's
    and dataset.py's sibling imports do); afterwards sys.path and sys.modules are restored."""
    commit = official.check_commit(KEY)["commit"] if check_commit else \
        official.clone_commit(KEY)
    edits = {"model": model_edits(feature_dim, ncrops), "train": train_edits(cpu_labels),
             "utils": utils_edits(), "dataset": [] if data is None else dataset_edits(data),
             "option": [], "config": []}
    order = [f for f in ("utils", "dataset", "model", "train", "option", "config") if f in files]
    if "dataset" in order and data is None:
        raise HarnessError("dataset.py needs a DataPatch (the training list)")
    with _bare_names(BARE), official.session(KEY) as s, _no_bytecode():
        mods, applied, diffs, texts = {}, [], [], {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)      # set_default_tensor_type
            for stem in order:
                rel = f"{stem}.py"
                path = s.base / rel
                if _sha_file(path) != OFFICIAL_SHA256[rel]:
                    raise HarnessError(f"{rel} of the clone is not the pinned file")
                before = path.read_text(encoding="utf-8")
                after = apply_edits(before, edits[stem])
                if edits[stem]:
                    mods[stem] = _exec_module(f"{official.PRIVATE}.{KEY}.{stem}", path, after,
                                              stem)
                else:
                    mods[stem] = sys.modules[stem] = s.load(rel)
                applied += edits[stem]
                diffs.append(unified_diff(rel, before, after))
                texts[rel] = after
        yield OfficialRTFM(mods, Path(s.base), commit, tuple(applied), "".join(diffs), texts)


# --- shared helpers ---------------------------------------------------------------------------

def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """bit equality, max |a - b| and max |a - b| / max |b| (shapes must agree)."""
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    if a.shape != b.shape:
        return {"bit_equal": False, "shape": [list(a.shape), list(b.shape)],
                "max_abs": math.inf, "max_rel": math.inf}
    d = float((a - b).abs().max()) if a.numel() else 0.0
    ref = float(b.abs().max()) if b.numel() else 0.0
    return {"bit_equal": bool(torch.equal(a, b)), "max_abs": d,
            "max_rel": d / ref if ref > 0 else (0.0 if d == 0 else math.inf)}


def compare_states(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict:
    if list(a) != list(b):
        return {"bit_equal": False, "keys_equal": False, "max_abs": math.inf,
                "max_rel": math.inf, "differ": sorted(set(a) ^ set(b))}
    rows = {k: compare(a[k], b[k]) for k in a}
    return {"bit_equal": all(r["bit_equal"] for r in rows.values()), "keys_equal": True,
            "max_abs": max((r["max_abs"] for r in rows.values()), default=0.0),
            "max_rel": max((r["max_rel"] for r in rows.values()), default=0.0),
            "differ": [k for k, r in rows.items() if not r["bit_equal"]]}


def close(row: Mapping) -> bool:
    return bool(row["bit_equal"] or row["max_rel"] <= TOL)


def perturb_batchnorm(model: nn.Module, seed: int) -> None:
    """Random affine parameters and running statistics for every BatchNorm (deterministic):
    eval-mode normalisation and the zero-initialised non-local output are then exercised."""
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                n = m.num_features
                m.weight.copy_(1.0 + 0.3 * torch.randn(n, generator=g))
                m.bias.copy_(0.1 * torch.randn(n, generator=g))
                m.running_mean.copy_(0.2 * torch.randn(n, generator=g))
                m.running_var.copy_(0.5 + torch.rand(n, generator=g))


def _rng(device: torch.device):
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _set_rng(state, device: torch.device) -> None:
    torch.set_rng_state(state[0])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state[1], device)


class LossRecorder:
    """Stand-in for the official visdom Visualizer: ``plot_lines`` records the values."""

    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def plot_lines(self, name: str, y, **_kw) -> None:
        self.values.setdefault(str(name), []).append(float(y))

    def lines(self, *_a, **_kw) -> None:
        return None


class GradProbe:
    """Optimizer stand-in: ``zero_grad`` clears, ``step`` records the gradients (no update)."""

    def __init__(self, named: Sequence[tuple[str, nn.Parameter]]) -> None:
        self.named = list(named)
        self.grads: dict[str, torch.Tensor] = {}

    def zero_grad(self) -> None:
        for _, p in self.named:
            p.grad = None

    def step(self) -> None:
        self.grads = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                      for n, p in self.named}


# --- (q1) -------------------------------------------------------------------------------------

def _official_step(off: OfficialRTFM, model, x, batch, optimizer, viz, device) -> None:
    labels = (torch.zeros(batch), torch.ones(batch))
    off.train.train(iter([(x[:batch], labels[0])]), iter([(x[batch:], labels[1])]), model, batch,
                    optimizer, viz, device)


def _port_step(model, x, batch, optimizer, device) -> dict:
    """The port's training step: METHOD.train_step on a single-crop Batch; for multi-crop toy
    inputs the same order (loss, zero_grad, backward, step) on rtfm_cost."""
    labels = torch.cat([torch.zeros(batch), torch.ones(batch)]).to(device)
    if x.shape[1] == 1:
        b = Batch(x[:, 0].to(device), labels,
                  torch.full((x.shape[0],), x.shape[2], dtype=torch.int64, device=device),
                  tuple(range(x.shape[0])), batch)
        return METHOD.train_step(model, b, optimizer, unit=x.shape[2], scale=SCALE)
    out = rtfm_cost(model, x.to(device), labels, batch, margin=SCALE["margin"])
    optimizer.zero_grad()
    out.total.backward()
    optimizer.step()
    return {"total": float(out.total.detach()), "smooth": float(out.smooth.detach()),
            "sparsity": float(out.sparsity.detach())}


def _q1_loss(off: OfficialRTFM, om, pm, x, batch: int, device: torch.device, state) -> dict:
    """Loss, its parts and the gradients from one RNG state: the official train() with a
    gradient probe as its optimizer (it records after backward) against the port's step."""
    viz, probe_o = LossRecorder(), GradProbe(list(om.named_parameters()))
    _official_step(off, om, x, batch, probe_o, viz, device)
    _set_rng(state, device)
    probe_p = GradProbe(list(pm.named_parameters()))
    parts = _port_step(pm, x, batch, probe_p, device)
    names = {"total": "loss", "smooth": "smooth loss", "sparsity": "sparsity loss"}
    rows = {k: compare(torch.tensor(parts[k]), torch.tensor(viz.values[v][0]))
            for k, v in names.items()}
    return {"loss": rows["total"], "parts": {k: rows[k] for k in ("smooth", "sparsity")},
            "gradients": compare_states(probe_o.grads, probe_p.grads)}


def check_q1(off: OfficialRTFM, *, feature_dim: int, ncrops: int, batch: int = 2,
             device: torch.device | str = "cpu", seed: int = 0) -> dict:
    """(q1) on toy inputs (module docstring); ``off`` built with the same feature_dim / ncrops."""
    device, t = torch.device(device), RECIPE.official_unit
    seed_all(seed)
    om = off.model.Model(feature_dim, batch)
    seed_all(seed)
    pm = METHOD.build(feature_dim, unit=t, input_ln=False, scale=SCALE)
    res: dict = {"config": {"feature_dim": feature_dim, "ncrops": ncrops, "batch": batch, "T": t,
                            "device": str(device), "seed": seed},
                 "construction": compare_states(om.state_dict(), pm.state_dict())}
    perturb_batchnorm(om, seed + 11)
    om, pm = om.to(device), pm.to(device)
    sd0 = {k: v.clone() for k, v in om.state_dict().items()}
    g = torch.Generator().manual_seed(seed + 1)
    x = (torch.randn(2 * batch, ncrops, t, feature_dim, generator=g)
         * (0.5 + torch.rand(2 * batch, 1, t, 1, generator=g))).to(device)
    state = _rng(device)

    def fresh():
        om.load_state_dict(sd0)
        pm.load_state_dict(sd0)
        om.train()
        METHOD.set_mode(pm, True)
        _set_rng(state, device)

    fresh()
    oo = om(x)
    _set_rng(state, device)
    po = pm(x, batch)
    names = {"score_abnormal": 0, "score_normal": 1, "feat_abnormal": 2, "feat_normal": 3,
             "scores": 6, "magnitudes": 9}
    res["forward"] = {n: compare(oo[i], getattr(po, n)) for n, i in names.items()}
    fresh()
    res |= _q1_loss(off, om, pm, x, batch, device, state)
    fresh()
    opt_o = torch.optim.Adam(om.parameters(), lr=RECIPE.optimizer.lr,
                             weight_decay=RECIPE.optimizer.weight_decay)
    _official_step(off, om, x, batch, opt_o, LossRecorder(), device)
    _set_rng(state, device)
    _port_step(pm, x, batch, METHOD.make_optimizer(pm, RECIPE.optimizer.lr), device)
    res["step"] = compare_states(om.state_dict(), pm.state_dict())
    res["pass"] = bool(res["construction"]["bit_equal"]
                       and all(r["bit_equal"] for r in res["forward"].values())
                       and close(res["loss"]) and all(close(r) for r in res["parts"].values())
                       and close(res["gradients"]) and close(res["step"]))
    return res


def adaptation_shapes(official_model: nn.Module, port_model: nn.Module, feature_dim: int) -> dict:
    """The adaptation changes shapes only where the official width is 2048 = D: every parameter
    and buffer has the same name, and a differing shape differs only in dimensions 2048 -> D."""
    so, sp = official_model.state_dict(), port_model.state_dict()
    rows, ok = [], list(so) == [k for k in sp if not k.startswith("input_ln.")]
    for k, v in so.items():
        a, b = tuple(v.shape), tuple(sp[k].shape) if k in sp else None
        if a != b:
            same_rank = b is not None and len(a) == len(b)
            dims_ok = same_rank and all(x == y or (x == OFFICIAL_DIM and y == feature_dim)
                                        for x, y in zip(a, b, strict=True))
            ok = ok and dims_ok
            rows.append({"name": k, "official": list(a), "port": None if b is None else list(b)})
    return {"pass": bool(ok), "changed": rows}


# --- (q1c) ------------------------------------------------------------------------------------

def _outside_repo(path: Path) -> Path:
    p = Path(path).resolve()
    if p == REPO_ROOT.resolve() or REPO_ROOT.resolve() in p.parents:
        raise HarnessError(f"{path} is inside the repository: use a folder outside it")
    return p


def position_files(folder: Path, labels: Sequence[int], value: Callable[[int], np.ndarray]
                   ) -> DataPatch:
    """One .npy per training position (``value(position)``, shape (T, 1, D)) and the list
    file, abnormal positions first (ascending), then normal ones: the official list layout."""
    folder = _outside_repo(folder)
    (folder / "feat").mkdir(parents=True, exist_ok=True)
    abn = [i for i, y in enumerate(labels) if int(y) == 1]
    nor = [i for i, y in enumerate(labels) if int(y) == 0]
    lines = []
    for i in abn + nor:
        f = folder / "feat" / f"{i:06d}.npy"
        np.save(f, np.asarray(value(i), dtype=np.float32))
        lines.append(f.as_posix())
    lst = folder / "train.list"
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DataPatch(lst, len(abn))


def _args(off: OfficialRTFM, argv: Sequence[str]):
    return off.option.parser.parse_args(list(argv))


def check_q1c(official: ModuleType, labels: Sequence[int], *, work_dir: Path, steps: int,
              batch: int = OFFICIAL_BATCH, seed: int = 0) -> dict:
    """(q1c): the official loader loop against the port's sampler (module docstring)."""
    data = position_files(Path(work_dir), labels, lambda i: np.full((1, 1, 1), float(i)))
    params = dict(RECIPE.sampler_params) | {"batch_normal": batch, "batch_abnormal": batch}
    with official_rtfm(official, files=("utils", "dataset", "option"), data=data) as off:
        args = _args(off, ["--batch-size", str(batch), "--dataset", "ucf"])
        seed_all(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            loaders = [DataLoader(off.dataset.Dataset(args, test_mode=False, is_normal=normal),
                                  batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=False, drop_last=True) for normal in (True, False)]
        official_draws, its = [], [None, None]
        for step in range(1, int(steps) + 1):
            for j, loader in enumerate(loaders):
                if (step - 1) % len(loader) == 0:
                    its[j] = iter(loader)
            pos, lab = [], []
            for it in its:
                x, y = next(it)
                pos += [int(v) for v in x[:, 0, 0, 0].tolist()]
                lab += [int(v) for v in y.tolist()]
            official_draws.append((tuple(pos), tuple(lab)))
    seed_all(seed)
    s = samplers.create(RECIPE.sampler, samplers.Pools.from_labels(labels), params)
    port = [(d.videos, d.labels) for d in samplers.draws(s, int(steps))]
    first = next((i + 1 for i, (a, b) in enumerate(zip(official_draws, port, strict=True))
                  if a != b), None)
    return {"steps": int(steps), "batch": batch, "seed": seed, "n_videos": len(labels),
            "n_abnormal": data.n_abnormal, "loader_lengths": [len(x) for x in loaders],
            "first_mismatch": first, "pass": first is None,
            "official_diff_sha256": off.diff_sha256(Path(work_dir).resolve())}


# --- (q1e) ------------------------------------------------------------------------------------

def official_piece(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    """(n,) fc3 logits and official scores of one piece (n, D) (eval mode, no grad): the
    official inference (test_10crop.py L11-L19: the loader's (1, n, 1, D) batch permuted, the
    official forward, its scores squeezed and averaged over the batch), fc3 captured by a hook;
    n < k (the forward's top-k would raise): the forward's score path on the official modules."""
    n = int(x.shape[0])
    if n >= int(model.k_abn):
        seen: list[torch.Tensor] = []
        hook = model.fc3.register_forward_hook(lambda _m, _i, o: seen.append(o.detach()))
        try:
            outputs = model(inputs=x[None, :, None, :].permute(0, 2, 1, 3))
        finally:
            hook.remove()
        score = torch.mean(torch.squeeze(outputs[6], 1), 0)
        return seen[-1].reshape(-1), score.reshape(-1), "forward"
    h = model.drop_out(model.Aggregate(x[None]))
    h = model.drop_out(model.relu(model.fc1(h)))
    z = model.fc3(model.drop_out(model.relu(model.fc2(h))))
    return z.reshape(-1), model.sigmoid(z).reshape(-1), "score_path"


def official_scores(model: nn.Module, x: torch.Tensor, n_chunk: int,
                    ln: nn.Module | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The official inference on every N_chunk piece of one video (the plan's piece rule)."""
    zs, ss, paths = [], [], []
    with torch.no_grad():
        xin = x if ln is None else ln(x)
        for a, b in chunk_bounds(int(x.shape[0]), n_chunk):
            z, s, path = official_piece(model, xin[a:b])
            zs.append(z.float().cpu().numpy())
            ss.append(s.float().cpu().numpy())
            paths.append(path)
    return np.concatenate(zs), np.concatenate(ss), paths


def q1e_case(official_model: nn.Module, port_model: nn.Module, x: torch.Tensor, *,
             n_chunk: int, ln: nn.Module | None, device: torch.device) -> dict:
    zo, so, paths = official_scores(official_model, x.to(device), n_chunk, ln)
    zp = score_video(METHOD, port_model, x, unit=RECIPE.official_unit, n_chunk=n_chunk,
                     device=device)
    sp = sigmoid32(zp)
    dz = float(np.abs(zp.astype(np.float64) - zo).max())
    ds = float(np.abs(sp.astype(np.float64) - so).max())
    return {"n": int(x.shape[0]), "pieces": [list(p) for p in chunk_bounds(int(x.shape[0]),
                                                                           n_chunk)],
            "official_paths": paths, "logits_bit_equal": bool(np.array_equal(zp, zo)),
            "scores_bit_equal": bool(np.array_equal(sp, so)), "max_abs_logit": dz,
            "max_abs_score": ds, "pass": bool(dz <= TOL and ds <= TOL)}


def check_q1e(off: OfficialRTFM, *, feature_dim: int, lengths: Sequence[int], input_ln: bool,
              n_chunk: int = N_CHUNK_DEFAULT, device: torch.device | str = "cpu",
              seed: int = 0, videos: Sequence[torch.Tensor] | None = None) -> dict:
    """(q1e) on real-length synthetic videos (module docstring), or on ``videos`` (processed
    (n, D) rows, e.g. the validation videos of ``check_q1e_val``) instead of ``lengths``."""
    device = torch.device(device)
    seed_all(seed)
    om = off.model.Model(feature_dim, OFFICIAL_BATCH)
    perturb_batchnorm(om, seed + 7)
    pm = METHOD.build(feature_dim, unit=RECIPE.official_unit, input_ln=input_ln, scale=SCALE)
    missing = pm.load_state_dict(om.state_dict(), strict=False)
    ln = None
    if input_ln:
        g = torch.Generator().manual_seed(seed + 3)
        with torch.no_grad():
            pm.input_ln.weight.copy_(1.0 + 0.2 * torch.randn(feature_dim, generator=g))
            pm.input_ln.bias.copy_(0.1 * torch.randn(feature_dim, generator=g))
        ln = nn.LayerNorm(feature_dim)
        ln.load_state_dict({k.split(".", 1)[1]: v for k, v in pm.state_dict().items()
                            if k.startswith("input_ln.")})
        ln = ln.to(device).eval()
    om, pm = om.to(device).eval(), pm.to(device)
    METHOD.set_mode(pm, False)
    g = torch.Generator().manual_seed(seed + 5)
    cases = []
    inputs = list(videos) if videos is not None else [
        1.5 * torch.randn(int(n), feature_dim, generator=g) + 0.2 for n in lengths]
    for x in inputs:
        cases.append(q1e_case(om, pm, x, n_chunk=n_chunk, ln=ln, device=device))
    return {"feature_dim": feature_dim, "input_ln": input_ln, "n_chunk": n_chunk,
            "device": str(device), "eval_mode": not (om.training or pm.training),
            "missing_keys": list(missing.missing_keys),
            "unexpected_keys": list(missing.unexpected_keys), "cases": cases,
            "pass": bool(all(c["pass"] for c in cases) and not missing.unexpected_keys
                         and all(k.startswith("input_ln.") for k in missing.missing_keys))}


def check_q1e_val(official: ModuleType, dataset: str, pair: str, *, paths=None, split=None,
                  n_chunk: int = N_CHUNK_DEFAULT, device: torch.device | str = "cpu",
                  seed: int = 0, max_videos: int | None = None) -> dict:
    """(q1e) on the processed validation videos (PLAN.md (q1e): real-length validation videos):
    train.validation_videos (the dense training-split store through the chain's InputPipeline:
    the d16m32 scale map) of ``dataset`` / ``pair``, input LayerNorm for XCB+SigLIP, the same
    randomised weights in the official model and the port; every video per N_chunk piece."""
    from wvad.data import load_split, split_file

    if paths is None:
        from wvad.paths import load_paths

        paths = load_paths(None)
    split = split if split is not None else load_split(split_file(dataset))
    variant = RECIPE.unit_input(RECIPE.official_unit).variant
    val = tr.validation_videos(paths, dataset, pair, variant, split, max_videos=max_videos)
    dim, ln = int(sum(val.dims)), pair_input_ln(pair)
    dev = torch.device(device)
    with official_rtfm(official, feature_dim=dim, ncrops=1, cpu_labels=dev.type == "cpu") as off:
        res = check_q1e(off, feature_dim=dim, lengths=(), input_ln=ln, n_chunk=n_chunk,
                        device=dev, seed=seed, videos=val.x)
    for c, v in zip(res["cases"], val.indices, strict=True):
        c["video"] = int(v)
    return res | {"check": "E023 (q1e) RTFM on the processed validation videos",
                  "dataset": dataset, "pair": pair, "variant": variant,
                  "pipeline": val.pipeline.to_json(), "videos": len(val.x),
                  "of_validation_videos": len(split.val_indices),
                  "max_snippets": max(int(x.shape[0]) for x in val.x),
                  "pass": bool(val.x) and res["pass"]}


# --- (q1b): opened-file audit -----------------------------------------------------------------

class _Audit:
    active: set[str] | None = None
    installed = False

    @classmethod
    def hook(cls, event: str, args) -> None:
        if event == "open" and cls.active is not None and args and \
                isinstance(args[0], (str, bytes, os.PathLike)):
            cls.active.add(os.fsdecode(args[0]))


@contextlib.contextmanager
def opened_files() -> Iterator[set[str]]:
    """Every distinct path opened in this process while the block runs (sys.addaudithook)."""
    if not _Audit.installed:
        sys.addaudithook(_Audit.hook)
        _Audit.installed = True
    seen: set[str] = set()
    _Audit.active = seen
    try:
        yield seen
    finally:
        _Audit.active = None


CODE_SUFFIXES = (".py", ".pyc", ".pyd", ".dll", ".so", ".pth", ".json.gz")


def audit_summary(paths: Sequence[str] | set[str], roots: Mapping[str, Path]) -> dict:
    """Opened data files grouped by root (code files and the Python installation left out) and
    the forbidden ones: anything in the clone's list/ folder (test lists, ground truth), a
    feature-root store of the test split, a manifest or annotation file of the repository."""
    out: dict[str, set[str]] = {k: set() for k in roots}
    out["other"] = set()
    forbidden = set()
    resolved = {k: Path(v).resolve() for k, v in roots.items()}
    prefixes = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    for raw in paths:
        try:
            p = Path(raw).resolve()
        except (OSError, ValueError):
            continue
        if p.suffix.lower() in CODE_SUFFIXES or any(q in p.parents for q in prefixes):
            continue
        key = next((k for k, r in resolved.items() if p == r or r in p.parents), "other")
        rel = p.relative_to(resolved[key]).as_posix() if key != "other" else p.name
        out[key].add(rel)
        parts = rel.split("/")
        if (key == "clone" and parts[0] == "list") or \
                (key == "feature_root" and any(x.endswith("_test") or x == "test"
                                               for x in parts[:-1])) or \
                (key == "repo" and rel.startswith(("data/manifests/", "data/annotations/"))):
            forbidden.add(f"{key}:{rel}")
    files: dict = {k: sorted(v) for k, v in out.items() if k not in ("other", "work")}
    folders: dict[str, int] = {}
    for rel in out.get("work", ()):
        top = "/".join(rel.split("/")[1:3]) if "/" in rel else rel
        folders[top] = folders.get(top, 0) + 1
    files["work"] = {"count": len(out.get("work", ())), "by_folder": dict(sorted(folders.items()))}
    return {"files": files, "other_count": len(out["other"]), "forbidden": sorted(forbidden)}


# --- (q1b): the runner ------------------------------------------------------------------------

@dataclass(frozen=True)
class Q1BConfig:
    dataset: str
    pair: str
    seed: int
    work_dir: str
    input_mode: str = "snippets"        # snippets (gate of record): official process_feat;
    #                                     rows: the port's d16m32 rows (extra evidence)
    mode: str = "deterministic"         # deterministic | nondeterministic (plan fallback)
    iterations: int | None = None       # None: the official 15000 (other values: self-tests)
    batch: int | None = None            # None: the official 32 (other values: self-tests)
    eval_points: int = 30
    n_chunk: int = N_CHUNK_DEFAULT
    paths: str | None = None


class _RecordingRTFM(RTFM):
    """The port with its per-step losses recorded (computation unchanged); ``recipe`` may carry
    another batch size for the synthetic self-test."""

    def __init__(self, recipe) -> None:
        self._recipe = recipe
        self.steps: list[dict] = []

    @property
    def recipe(self):
        return self._recipe

    def train_step(self, model, batch, optimizer, *, unit, scale):
        parts = super().train_step(model, batch, optimizer, unit=unit, scale=scale)
        self.steps.append(parts)
        return parts


def official_validation(model: nn.Module, val: Sequence[torch.Tensor], n_chunk: int,
                        device: torch.device) -> np.ndarray:
    """test_10crop.py L7-L19 on the validation videos (eval mode, no grad; the model stays in
    eval mode as officially, the next train() call sets train mode): concatenated fc3 logits."""
    model.eval()
    return np.concatenate([official_scores(model, x.to(device), n_chunk)[0] for x in val]
                          ).astype(np.float32)


def _official_run(official: ModuleType, cfg: Q1BConfig, *, data: DataPatch, dim: int,
                  steps: int, batch: int, val, device: torch.device, out: Path) -> dict:
    """The patched official main.py once (module docstring); returns losses, logits, weights."""
    out.mkdir(parents=True, exist_ok=True)
    points = set(eval_steps(steps, cfg.eval_points))
    viz, z_points = LossRecorder(), []
    final = out / "final.pkl"

    def evaluate(step: int, model: nn.Module) -> None:
        if step in points:
            with isolated_rng(device):
                z_points.append(official_validation(model, val, cfg.n_chunk, device))

    hooks = {"viz": viz, "evaluate": evaluate, "seed": lambda: seed_all(cfg.seed),
             "ckpt": str(out / "ckpt"), "final": str(final)}
    argv = ["main.py", "--feature-size", str(dim), "--batch-size", str(batch), "--dataset", "ucf"]
    if steps != RECIPE.iterations:
        argv += ["--lr", f"[{RECIPE.optimizer.lr!r}]*{steps}", "--max-epoch", str(steps)]
    with official_rtfm(official, feature_dim=dim, ncrops=1, cpu_labels=device.type == "cpu",
                       data=data, files=("utils", "dataset", "model", "train", "option",
                                         "config")) as off:
        path = off.base / "main.py"
        if _sha_file(path) != OFFICIAL_SHA256["main.py"]:
            raise HarnessError("main.py of the clone is not the pinned file")
        before = path.read_text(encoding="utf-8")
        text = apply_edits(before, main_edits())
        saved_argv = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                exec(compile(text, str(path), "exec"),  # noqa: S102 - pinned official code
                     {"__name__": "__main__", "__file__": str(path), "E023_HOOKS": hooks})
        finally:
            sys.argv = saved_argv
        diff = off.diff + unified_diff("main.py", before, text)
        edits = list(off.edits) + main_edits()
    return {"losses": viz.values.get("loss", []), "smooth": viz.values.get("smooth loss", []),
            "sparsity": viz.values.get("sparsity loss", []), "z": z_points,
            "state": torch.load(final, map_location="cpu", weights_only=True),
            "commit": off.commit, "diff": diff, "edits": edits}


def _first_diff(a: Sequence[float], b: Sequence[float]) -> int | None:
    return next((i + 1 for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                None if len(a) == len(b) else min(len(a), len(b)) + 1)


def _rel(a: Sequence[float], b: Sequence[float], n: int) -> float:
    x, y = np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)
    if x.size == 0:
        return math.inf
    return float((np.abs(x - y) / np.maximum(np.abs(y), 1e-30)).max())


def _selected(aucs: Sequence[float]) -> dict:
    if not all(math.isfinite(v) for v in aucs):
        return {"point": None, "v_auc": math.nan}
    i, tied = select_point(aucs)
    return {"point": i + 1, "v_auc": float(aucs[i]), "tied": tied}


def compare_runs(port: Mapping, off: Mapping, inputs) -> dict:
    """Per-iteration losses, final weights, validation logits and V-AUCs (sigma, logit)."""
    counts = [int(x.shape[0]) for x in inputs.val]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    off_auc: dict[str, list[float]] = {"sigma": [], "logit": []}
    for z in off["z"]:
        pa = tr.point_aucs([z[a:a + n] for a, n in zip(starts, counts, strict=True)],
                           inputs.val_totals, inputs.val_labels)
        for k in off_auc:
            off_auc[k].append(pa[k])
    pz, oz = np.asarray(port["z"]), np.asarray(off["z"])
    res = {"losses": {"n_port": len(port["losses"]), "n_official": len(off["losses"]),
                      "bit_equal": list(port["losses"]) == list(off["losses"]),
                      "first_diff_iteration": _first_diff(port["losses"], off["losses"]),
                      "max_rel_first50": _rel(port["losses"], off["losses"], Q1B_FIRST),
                      "max_abs": float(np.abs(np.subtract(port["losses"], off["losses"])).max())
                      if len(port["losses"]) == len(off["losses"]) else math.inf},
           "parts_bit_equal": {k: list(port[k]) == list(off[k]) for k in ("smooth", "sparsity")},
           "weights": compare_states(port["state"], off["state"]),
           "val_logits": {"bit_equal": bool(pz.shape == oz.shape and np.array_equal(pz, oz)),
                          "max_abs": float(np.abs(pz.astype(np.float64) - oz).max())
                          if pz.shape == oz.shape else math.inf},
           "v_auc": {}}
    for k in ("sigma", "logit"):
        p, o = [float(v) for v in port["v_auc"][k]], off_auc[k]
        res["v_auc"][k] = {"bit_equal": p == o, "port": p, "official": o,
                           "port_selected": _selected(p), "official_selected": _selected(o)}
    return res


def q1b_verdict(cmp: Mapping, mode: str, eps: Mapping | None = None) -> dict:
    """The plan's per-run (q1b) rule: bit (losses, final weights, V-AUC); else numeric (first 50
    iterations' relative loss diff <= 1e-5, selected sigma V-AUC within 1.0 point); the
    non-deterministic fallback with eps_off from two official runs."""
    sig = cmp["v_auc"]["sigma"]
    d_pt = 100.0 * abs(sig["port_selected"]["v_auc"] - sig["official_selected"]["v_auc"])
    bit = bool(cmp["losses"]["bit_equal"] and cmp["weights"]["bit_equal"] and sig["bit_equal"]
               and cmp["v_auc"]["logit"]["bit_equal"])
    out = {"rule": "bit" if bit else mode, "bit_equal": bit, "v_auc_diff_pt": d_pt,
           "first_diff_iteration": cmp["losses"]["first_diff_iteration"]}
    if bit:
        return out | {"pass": True}
    if mode == "deterministic":
        rel = cmp["losses"]["max_rel_first50"]
        return out | {"rule": "numeric", "max_rel_first50": rel,
                      "pass": bool(rel <= Q1B_REL and d_pt <= Q1B_SEED_PT)}
    eps = dict(eps or {})
    bound = max(Q1B_SEED_PT, 2.0 * eps.get("v_auc_pt", math.inf))
    ok = eps.get("port_window_max", math.inf) <= 2.0 * eps.get("eps_off", math.inf) + 1e-6
    return out | {"rule": "nondeterministic", "eps": eps, "v_auc_bound_pt": bound,
                  "pass": bool(ok and d_pt <= bound)}


@contextlib.contextmanager
def _restore_flags(flags: tuple[bool, bool, bool]) -> Iterator[None]:
    """The determinism flags as they were before a (q1b) run (the loop sets them globally)."""
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(flags[0])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[1], flags[2]


def _write_official_inputs(cfg: Q1BConfig, inputs, split, paths, folder: Path) -> DataPatch:
    if cfg.input_mode == "rows":
        return position_files(folder, inputs.train.labels,
                              lambda i: inputs.train.features[i].numpy()[:, None, :])
    if cfg.input_mode != "snippets":
        raise HarnessError(f"--input rows|snippets, got {cfg.input_mode!r}")
    from wvad.data.catalog import ENCODER_PAIRS, dense_dir
    from wvad.data.dense import DenseSet

    dense = DenseSet([dense_dir(paths, cfg.dataset, e) for e in ENCODER_PAIRS[cfg.pair]])
    train_idx = [int(i) for i in split.train_indices]
    return position_files(folder, inputs.train.labels,
                          lambda i: inputs.pipeline.apply(dense.video(train_idx[i])).numpy()
                          [:, None, :])


def _q1b_setup(cfg: Q1BConfig, paths, split) -> tuple:
    """The work folder (outside the repository), the run folder (refused when finished), the
    paths and the validation split (XCB+XCL only, PLAN.md (q1b))."""
    from wvad.data import load_split, split_file

    if cfg.pair != "xcb_xcl":
        raise HarnessError("(q1b) runs on XCB+XCL (PLAN.md; the official code has no input LN)")
    work = _outside_repo(Path(cfg.work_dir))
    run_dir = work / f"rtfm_{cfg.dataset}_{cfg.pair}_s{cfg.seed}_{cfg.input_mode}_{cfg.mode}"
    if (run_dir / "q1b_run.json").exists():
        raise HarnessError(f"{run_dir} holds a finished run")
    if paths is None:
        from wvad.paths import load_paths

        paths = load_paths(cfg.paths)
    return work, run_dir, paths, split if split is not None else load_split(
        split_file(cfg.dataset))


def run_q1b(official: ModuleType, cfg: Q1BConfig, *, paths=None, split=None,
            log: Callable[[str], None] = print) -> dict:
    """(q1b) one development seed (module docstring); returns the run entry (also written, with
    the per-iteration losses, to <work_dir>/<run>/q1b_run.json). Everything from reading the
    inputs to the last official step runs under the opened-file audit."""
    work, run_dir, paths, split = _q1b_setup(cfg, paths, split)
    det = cfg.mode == "deterministic"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # main.py L34
    batch = int(cfg.batch or OFFICIAL_BATCH)
    recipe = dataclasses.replace(RECIPE, sampler_params=dict(RECIPE.sampler_params)
                                 | {"batch_normal": batch, "batch_abnormal": batch})
    steps = int(cfg.iterations or RECIPE.iterations)
    roots = {"work": work, "feature_root": Path(paths.feature_root), "repo": REPO_ROOT,
             "clone": official.repo_dir(KEY)}
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
             torch.backends.cudnn.benchmark)
    with opened_files() as seen, _restore_flags(flags):
        inputs = tr.load_inputs(paths, cfg.dataset, cfg.pair, RECIPE.unit_input(32), split)
        data = _write_official_inputs(cfg, inputs, split, paths, run_dir / "official_inputs")
        method = _RecordingRTFM(recipe)
        prec = tr.run(tr.TrainConfig(method=KEY, unit=32, dataset=cfg.dataset, pair=cfg.pair,
                                     seed=cfg.seed, out_dir=str(run_dir / "port"),
                                     deterministic=det, iterations=steps,
                                     eval_points=cfg.eval_points, n_chunk=cfg.n_chunk,
                                     device=str(device)),
                      method=method, paths=paths, split=split, log=log)
        mode_state = deterministic(det)
        offs = [_official_run(official, cfg, data=data, dim=inputs.dim, steps=steps,
                              batch=batch, val=inputs.val, device=device,
                              out=run_dir / f"official_{r}") for r in range(1 if det else 2)]
    port = _port_result(run_dir / "port", method)
    cmp = compare_runs(port, offs[0], inputs)
    verdict = q1b_verdict(cmp, cfg.mode, None if det else _nondet_eps(port, offs, inputs, cmp,
                                                                      steps))
    audit = audit_summary(seen, roots)
    verdict["pass"] = bool(verdict["pass"] and not audit["forbidden"])
    diff = offs[0]["diff"].replace(work.as_posix(), "<work_dir>")
    entry = _entry(official, cfg, offs[0], diff, device=device, steps=steps, batch=batch) | {
        "deterministic_mode": mode_state, "work_dir": {"name": work.name,
                                                       "outside_repository": True},
        "opened_files": audit, "port": {"selection": prec["selection"], "failed": prec["failed"]},
        "comparison": _compact(cmp), "verdict": verdict}
    if cfg.input_mode == "snippets":
        entry["input_rows"] = _row_check(official, data, inputs)
    (run_dir / "official_patch.diff").write_text(diff, encoding="utf-8", newline="\n")
    full = entry | {"losses": {"port": port["losses"], "official": offs[0]["losses"]},
                    "comparison_full": cmp}
    (run_dir / "q1b_run.json").write_text(json.dumps(_jsonable(full), indent=1) + "\n",
                                          encoding="utf-8", newline="\n")
    log(f"[q1b rtfm] {cfg.dataset}/{cfg.pair} s{cfg.seed} {cfg.input_mode}: "
        f"{verdict['rule']} pass={verdict['pass']}")
    return _jsonable(entry)


def _port_result(port_dir: Path, method: _RecordingRTFM) -> dict:
    """The port chain's per-step losses, validation logits / V-AUCs and last weights."""
    with np.load(port_dir / tr.FILES["val"]) as f:
        z, sig, lg = list(f["z"]), f["v_auc_sigma"].tolist(), f["v_auc_logit"].tolist()
    return {"losses": [s["total"] for s in method.steps],
            "smooth": [s["smooth"] for s in method.steps],
            "sparsity": [s["sparsity"] for s in method.steps], "z": z,
            "v_auc": {"sigma": sig, "logit": lg},
            "state": torch.load(port_dir / tr.FILES["last"], map_location="cpu",
                                weights_only=True)["model_state_dict"]}


def _nondet_eps(port: Mapping, offs: Sequence[Mapping], inputs, cmp: Mapping, steps: int
                ) -> dict:
    """Non-deterministic fallback: eps_off = max |loss official run 1 - run 2| over the first
    min(500, S) iterations, the port's largest deviation there, and the two official runs'
    selected sigma V-AUC difference (points)."""
    w = min(NONDET_WINDOW, steps)
    o1, o2 = np.asarray(offs[0]["losses"][:w]), np.asarray(offs[1]["losses"][:w])
    c2 = compare_runs(port, offs[1], inputs)
    return {"window": w, "eps_off": float(np.abs(o1 - o2).max()),
            "port_window_max": float(np.abs(np.asarray(port["losses"][:w]) - o1).max()),
            "v_auc_pt": 100.0 * abs(cmp["v_auc"]["sigma"]["official_selected"]["v_auc"]
                                    - c2["v_auc"]["sigma"]["official_selected"]["v_auc"])}


def _entry(official: ModuleType, cfg: Q1BConfig, off: Mapping, diff: str, *,
           device: torch.device, steps: int, batch: int) -> dict:
    """The run's identity: configuration, official clone (commit, pinned files, edits, diff
    sha256 with the work folder as "<work_dir>"), plan sha256, code version, environment."""
    from wvad.diagnostics.common import code_version

    return {"method": KEY, "dataset": cfg.dataset, "pair": cfg.pair, "seed": cfg.seed,
            "input_mode": cfg.input_mode, "mode": cfg.mode, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "steps": steps, "batch": batch, "eval_points": cfg.eval_points,
            "n_chunk": cfg.n_chunk,
            "official": {"repo": official.REPOS[KEY],
                         "clone": f"${official.ENV}/{official.REPOS[KEY]}",
                         "commit": off["commit"], "file_sha256": OFFICIAL_SHA256,
                         "diff_sha256": _sha_text(diff),
                         "edits": [{"file": e.file, "lines": f"L{e.first}-L{e.last}",
                                    "reason": e.reason} for e in off["edits"]]},
            "plan_sha256": _sha_file(PLAN_FILE), "code": code_version(),
            "environment": tr._environment(device),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}


def _row_check(official: ModuleType, data: DataPatch, inputs) -> dict:
    """Snippets mode: the official process_feat windows of the stored snippets against the
    port's d16m32 rows (the expected cause of non-bit-equal runs: float32 vs float64 means)."""
    lines = Path(data.list_file).read_text(encoding="utf-8").split()
    order = sorted(range(len(inputs.train.labels)), key=lambda i: (inputs.train.labels[i] == 0, i))
    n_diff, max_abs = 0, 0.0
    with official_rtfm(official, files=("utils",)) as off:
        for i, f in zip(order, lines, strict=True):
            rows = off.utils.process_feat(np.load(f)[:, 0, :], RECIPE.official_unit)
            d = np.abs(rows.astype(np.float64) - inputs.train.features[i].numpy())
            n_diff += int((d > 0).any(axis=1).sum())
            max_abs = max(max_abs, float(d.max()))
    return {"rows_differing": n_diff, "rows_total": RECIPE.official_unit * len(lines),
            "max_abs": max_abs, "note": "numpy float32 window means (official process_feat) vs "
                                        "derive_n float64 -> float32"}


def _compact(cmp: Mapping) -> dict:
    out = {k: v for k, v in cmp.items() if k != "v_auc"}
    out["v_auc"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("port", "official")}
                    | {"max_abs_diff": float(np.abs(np.subtract(v["port"], v["official"])).max())
                       if len(v["port"]) == len(v["official"]) else math.inf}
                    for k, v in cmp["v_auc"].items()}
    return out


def _jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else str(v)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


GATE_OF_RECORD = "snippets"            # PLAN.md 기록 2026-10-19 (review M5); rows: evidence


def q1b_summary(runs: Mapping[str, Mapping]) -> dict:
    """Per dataset and input mode over the recorded seeds: every run passes and, for a numeric
    or non-deterministic rule, the mean selected-V-AUC difference is <= 0.5 point; the
    snippets groups are the gate of record ("gate_of_record"), the rows groups extra
    evidence."""
    out = {}
    for ds, im in sorted({(r["dataset"], r["input_mode"]) for r in runs.values()}):
        rows = [r for r in runs.values() if (r["dataset"], r["input_mode"]) == (ds, im)]
        diffs = []
        for r in rows:
            sig = r["comparison"]["v_auc"]["sigma"]
            diffs.append(100.0 * (float(sig["port_selected"]["v_auc"])
                                  - float(sig["official_selected"]["v_auc"])))
        mean = float(np.mean(diffs)) if diffs else math.nan
        bit = all(r["verdict"]["bit_equal"] for r in rows)
        out[f"{ds}_{im}"] = {"seeds": sorted(int(r["seed"]) for r in rows), "all_bit_equal": bit,
                   "mean_v_auc_diff_pt": mean, "gate_of_record": im == GATE_OF_RECORD,
                   "role": "gate of record" if im == GATE_OF_RECORD else "extra evidence",
                   "pass": bool(rows and all(r["verdict"]["pass"] for r in rows)
                                and (bit or abs(mean) <= Q1B_MEAN_PT))}
    return out


def write_q1b_record(entry: Mapping, path: Path = Q1B_FILE) -> dict:
    """Merge one run entry into the (q1b) record (methods.rtfm.runs[<key>]) and refresh the
    method's summary; other methods' entries are kept."""
    key = f"{entry['dataset']}_{entry['pair']}_s{entry['seed']}_{entry['input_mode']}"

    def merge(rec: dict) -> dict:
        rec.setdefault("check", "E023 (q1b) official pipeline vs port")
        mine = rec.setdefault("methods", {}).setdefault(KEY, {"runs": {}})
        mine["runs"][key] = _jsonable(dict(entry))
        mine["summary"] = q1b_summary(mine["runs"])
        return _jsonable(rec)

    return update_json(Path(path), merge)


# --- command line -----------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_rtfm", description="E023 RTFM equivalence gates")
    sub = p.add_subparsers(dest="gate", required=True)
    q1 = sub.add_parser("q1")
    q1.add_argument("--device", default="cpu")
    q1.add_argument("--out", type=Path)
    q1c = sub.add_parser("q1c")
    q1c.add_argument("--dataset", required=True, choices=tr.DATASETS)
    q1c.add_argument("--steps", type=int, default=200)
    q1c.add_argument("--seed", type=int, default=0)
    q1c.add_argument("--work_dir", required=True, type=Path)
    q1c.add_argument("--out", type=Path)
    q1e = sub.add_parser("q1e")
    q1e.add_argument("--n_chunk", type=int, default=N_CHUNK_DEFAULT)
    q1e.add_argument("--device", default="cpu")
    q1e.add_argument("--out", type=Path)
    qv = sub.add_parser("q1e_val")
    qv.add_argument("--dataset", required=True, choices=tr.DATASETS)
    qv.add_argument("--pair", required=True, choices=tr.PAIRS)
    qv.add_argument("--n_chunk", type=int, default=N_CHUNK_DEFAULT)
    qv.add_argument("--device", default="cpu")
    qv.add_argument("--max_videos", type=int, help="the first K validation videos and the longest")
    qv.add_argument("--paths")
    qv.add_argument("--out", type=Path)
    q1b = sub.add_parser("q1b")
    q1b.add_argument("--dataset", required=True, choices=tr.DATASETS)
    q1b.add_argument("--pair", default="xcb_xcl", choices=("xcb_xcl",))
    q1b.add_argument("--seed", type=int, required=True)
    q1b.add_argument("--work_dir", required=True)
    q1b.add_argument("--input", default=GATE_OF_RECORD, choices=("snippets", "rows"),
                     help="snippets: the gate of record; rows: extra evidence")
    q1b.add_argument("--mode", default="deterministic",
                     choices=("deterministic", "nondeterministic"))
    q1b.add_argument("--n_chunk", type=int, default=N_CHUNK_DEFAULT)
    q1b.add_argument("--paths")
    q1b.add_argument("--record", type=Path, default=Q1B_FILE)
    return p


def run_q1_all(official: ModuleType, device: str = "cpu") -> dict:
    dev = torch.device(device)
    out = {}
    for d, c in ((OFFICIAL_DIM, OFFICIAL_CROPS), (1792, 1), (1920, 1)):
        with official_rtfm(official, feature_dim=d, ncrops=c,
                           cpu_labels=dev.type == "cpu") as off:
            out[f"d{d}_c{c}"] = check_q1(off, feature_dim=d, ncrops=c, device=dev) | {
                "official": off.describe()}
    with official_rtfm(official) as o2048:
        om = o2048.model.Model(OFFICIAL_DIM, 2)
    with official_rtfm(official, feature_dim=1792, ncrops=1) as o1792:
        patched = [tuple(v.shape) for v in o1792.model.Model(1792, 2).state_dict().values()]
    out["adaptation_shapes"] = adaptation_shapes(
        om, METHOD.build(1792, unit=32, input_ln=True, scale=SCALE), 1792)
    out["adaptation_shapes"]["patched_official_equals_port_shapes"] = patched == [
        tuple(v.shape) for v in METHOD.build(1792, unit=32, input_ln=False,
                                             scale=SCALE).state_dict().values()]
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def run_q1e_all(official: ModuleType, n_chunk: int = N_CHUNK_DEFAULT,
                device: str = "cpu") -> dict:
    out = {}
    with official_rtfm(official, feature_dim=1792, ncrops=1) as off:
        out["d1792"] = check_q1e(off, feature_dim=1792, input_ln=False, n_chunk=n_chunk,
                                 lengths=(20, 300, n_chunk + 37, n_chunk + 1), device=device)
    with official_rtfm(official, feature_dim=1920, ncrops=1) as off:
        out["d1920_ln"] = check_q1e(off, feature_dim=1920, input_ln=True, n_chunk=n_chunk,
                                    lengths=(1, 2, 3, 50, n_chunk + 2), device=device)
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def main(argv: Sequence[str] | None = None, *, official: ModuleType | None = None) -> dict:
    if official is None:
        raise SystemExit("check_rtfm needs the official loader passed in: python -c \"import "
                         "sys; from wvad.baselines import official, check_rtfm; check_rtfm.main("
                         "sys.argv[1:], official=official)\" <gate> ...")
    a = _parser().parse_args(argv)
    if a.gate == "q1":
        res = run_q1_all(official, a.device)
    elif a.gate == "q1e":
        res = run_q1e_all(official, a.n_chunk, a.device)
    elif a.gate == "q1e_val":
        from wvad.paths import load_paths

        res = check_q1e_val(official, a.dataset, a.pair, paths=load_paths(a.paths),
                            n_chunk=a.n_chunk, device=a.device, max_videos=a.max_videos)
    elif a.gate == "q1c":
        from wvad.data import load_split, split_file

        sp = load_split(split_file(a.dataset))
        labels = [int(sp.labels[i]) for i in sp.train_indices]
        res = check_q1c(official, labels, work_dir=a.work_dir, steps=a.steps, seed=a.seed)
    else:
        res = run_q1b(official, Q1BConfig(a.dataset, a.pair, a.seed, a.work_dir, a.input, a.mode,
                                          n_chunk=a.n_chunk, paths=a.paths))
        write_q1b_record(res, a.record)
    if getattr(a, "out", None):
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(_jsonable(res), indent=1) + "\n", encoding="utf-8",
                         newline="\n")
    print(f"[check_rtfm] {a.gate}: pass={res.get('pass', res.get('verdict', {}).get('pass'))}")
    return res
