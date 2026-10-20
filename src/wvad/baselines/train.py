"""E023 training loop of the comparison methods (PLAN.md "체크포인트 선택", "같은 특징 규칙과 학습
단위", "검증 채점과 난수", "구현"). New code (add-only rule); fixed numeric code from the freeze on.

    python -m wvad.baselines.train --method <name> --unit 32|200|256 --dataset ucf|xdv
                                   --pair xcb_xcl|xcb_siglip --seed <s> --out_dir <dir>
                                   [--lr_mult 1/3|1|3|...] [--scale_mult 1/10|1|10]
                                   [--save_eval_ckpts] [--n_chunk 4096] [--deterministic]
                                   [--float64 (mil only, gate (q1b))]
                                   [--no_eval (gate (q5): the same loss record, no evaluation)]
                                   [--paths FILE] [--device cuda|cpu]

One chain: the method's official network, loss, optimizer and schedule (its RECIPE on --dataset:
``common.dataset_method`` resolves the dataset's official configuration, e.g. VCV's XD option
file; lr = official lr x --lr_mult, scale constants x --scale_mult: the D4 check), fp32 (no
autocast), the official
sampler over the training-part videos of data/splits/<ds>_val_split_15.json. Training input of
--unit: RECIPE.units[unit].variant, read by the frozen ``load_feature_set(dirs, "train")`` from
the E021 cache (d16m, X.32) or the wvad.extract.derive_n folder. Validation: every validation
video's E021 dense snippets (``DenseSet``, training-split store), processed by the unit's
InputPipeline (l2 of an l2 variant, the variant's scale map: the head's fixed input layer) and
scored whole, or in N_chunk pieces when n_v > N_chunk (``common.score_video``; the method's own
pieces inside), in eval mode, no grad, autocast off.

Order: generators seeded (``common.seed_all``) -> method.build -> .to(device) -> optimizer ->
sampler -> steps 1..S, each: lr (method.lr_at; set only when it changes), sampler.draw, collate,
method.train_step. Evaluation points: after steps floor(k S / 30 + 1/2), k = 1..30
(``common.eval_steps``); each evaluation runs inside ``common.isolated_rng`` (torch.random.fork_rng
on the CPU and the CUDA device, numpy and random saved and restored), so the training path does
not depend on evaluating (gate (q5): --no_eval gives the same losses and last weights).

Selection (C's rule, PLAN.md "선택 판독"): per evaluation point the W readout of the stored
validation snippet logits on the E021 32-window grid (no offset), video score max fp32 sigma
(sigma readout) or max window logit (logit readout), full-validation V-AUC (rank formula); the
first maximum (strict >, earliest point on ties). A non-finite validation logit at any point is
a numerical failure: no selection, only the last checkpoint.

Files in --out_dir (a chain whose selection.json exists is refused: completed chains are not
rerun; a partial chain is rerun from scratch, PLAN.md infrastructure rule)
    config.json         the resolved configuration (written first)
    val_scores.npz      z (30, rows) float32 validation snippet logits (videos in split order,
                        concatenated), counts, total_frames, labels, val_indices, eval_steps,
                        v_auc_sigma, v_auc_logit, saturated (videos with sigma score 1.0f)
    history.json        per evaluation point: step, lr, mean training losses since the previous
                        point, non-finite loss count, both V-AUCs, saturated, seconds
    ckpt_sigma_best.pt, ckpt_logit_best.pt, ckpt_last.pt
                        weights only ({"model_state_dict", "method", "unit", "rule",
                        "eval_point", "step"}; no optimizer state)
    ckpt_eval_XX.pt     with --save_eval_ckpts (D14, temporary until access (A)), XX = 01..30
    selection.json      the record, written last: configuration, recipe, code version,
                        environment, inputs (training cache sha256 from their meta.json, scale
                        arrays, split), steps, selections, checkpoint sha256
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from wvad.baselines import samplers
from wvad.baselines.common import (
    EVAL_POINTS,
    KNOWN_METHODS,
    N_CHUNK_DEFAULT,
    BaselineMethod,
    InputPipeline,
    collate,
    dataset_method,
    deterministic,
    eval_steps,
    get_method,
    input_ln,
    isolated_rng,
    parameter_count,
    score_video,
    seed_all,
    select_point,
    v_aucs,
    video_scores,
)
from wvad.baselines.recipes import UNITS, UnitInput, parse_mult, require_recipe, total_steps
from wvad.data import check_labels, load_feature_set, load_split, split_file
from wvad.data.catalog import ENCODER_PAIRS, FEATURE_VARIANTS, dense_dir
from wvad.data.dense import META_FILE, DenseSet, read_json
from wvad.data.features import FeatureSet

PAIRS = ("xcb_xcl", "xcb_siglip")
DATASETS = ("ucf", "xdv")
FILES = {"config": "config.json", "val": "val_scores.npz", "history": "history.json",
         "record": "selection.json", "sigma": "ckpt_sigma_best.pt",
         "logit": "ckpt_logit_best.pt", "last": "ckpt_last.pt"}
EVAL_CKPT = "ckpt_eval_{:02d}.pt"


@dataclass(frozen=True)
class TrainConfig:
    method: str
    unit: int
    dataset: str
    pair: str
    seed: int
    out_dir: str
    lr_mult: str = "1"
    scale_mult: str = "1"
    save_eval_ckpts: bool = False
    n_chunk: int = N_CHUNK_DEFAULT
    deterministic: bool = False
    float64: bool = False
    evaluate: bool = True
    eval_points: int = EVAL_POINTS
    iterations: int | None = None          # Python API only (tests): overrides S
    paths: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class Inputs:
    train: FeatureSet                       # training-part videos (split order)
    val: tuple[torch.Tensor, ...]           # processed validation snippets, split order
    val_indices: tuple[int, ...]
    val_labels: tuple[int, ...]
    val_totals: tuple[int, ...]
    pipeline: InputPipeline
    variant: str
    train_dirs: tuple[str, ...]
    train_metas: tuple[dict, ...]
    dim: int


def sha256_file(path) -> str:
    from wvad.prep.manifests import sha256_file as _sha

    return _sha(path)


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --- inputs -------------------------------------------------------------------------------------

def training_dirs(paths, dataset: str, pair: str, variant: str) -> tuple:
    """The training cache folders of a variant: the E021 cache (catalog variant, e.g. d16m) or
    the derive_n folder."""
    if variant in FEATURE_VARIANTS:
        from wvad.data.catalog import feature_dirs

        return tuple(feature_dirs(paths, dataset, pair, variant))
    from wvad.extract.derive_n import variant_dirs

    return variant_dirs(paths, dataset, pair, variant)


def input_pipeline(paths, dataset: str, pair: str, variant: str, dims: Sequence[int]
                   ) -> InputPipeline:
    """The validation / test processing of the variant: its scale maps and l2 rule."""
    if variant in FEATURE_VARIANTS:
        from wvad.eval.dense import scale_maps

        maps = scale_maps([dense_dir(paths, dataset, e) for e in ENCODER_PAIRS[pair]], variant)
        return InputPipeline(tuple(int(d) for d in dims), False,
                             None if maps is None else tuple(maps))
    from wvad.extract.derive_n import load_variant_scales, variant_dirs, variant_spec

    maps = load_variant_scales(variant_dirs(paths, dataset, pair, variant))
    return InputPipeline(tuple(int(d) for d in dims), variant_spec(variant).l2,
                         None if maps is None else tuple(maps))


@dataclass(frozen=True)
class ValidationVideos:
    """Processed validation videos of a variant (split order, or the ``positions`` asked for)."""

    x: tuple[torch.Tensor, ...]
    indices: tuple[int, ...]
    labels: tuple[int, ...]
    totals: tuple[int, ...]
    pipeline: InputPipeline
    dims: tuple[int, ...]


def validation_videos(paths, dataset: str, pair: str, variant: str, split,
                      positions: Sequence[int] | None = None,
                      max_videos: int | None = None) -> ValidationVideos:
    """The validation videos (E021 training-split dense store) through the variant's
    InputPipeline - the training chain's and the scoring's validation input; ``positions``:
    indices into the split's validation list (default: all, in split order); ``max_videos``
    (the (q1e) harnesses): the first K validation videos and the longest one."""
    dense = DenseSet([dense_dir(paths, dataset, e) for e in ENCODER_PAIRS[pair]])
    check_labels(split, dense.labels, f"{dataset}/{pair} dense training-split store")
    pipeline = input_pipeline(paths, dataset, pair, variant, dense.dims)
    val_all = tuple(int(i) for i in split.val_indices)
    if positions is None and max_videos is not None and int(max_videos) < len(val_all):
        counts = [int(dense.counts[v]) for v in val_all]
        positions = sorted(set(range(int(max_videos))) | {int(np.argmax(counts))})
    idx = val_all if positions is None else tuple(val_all[int(k)] for k in positions)
    return ValidationVideos(tuple(pipeline.apply(dense.video(v)) for v in idx), idx,
                            tuple(int(dense.labels[v]) for v in idx),
                            tuple(int(dense.total_frames[v]) for v in idx), pipeline,
                            tuple(int(d) for d in dense.dims))


def load_inputs(paths, dataset: str, pair: str, unit_input: UnitInput, split) -> Inputs:
    variant = unit_input.variant
    dirs = training_dirs(paths, dataset, pair, variant)
    fs = load_feature_set(dirs, "train")
    check_labels(split, fs.labels, f"{dataset}/{pair} {variant} training cache")
    val = validation_videos(paths, dataset, pair, variant, split)
    if sum(val.dims) != fs.dim:
        raise ValueError(f"dense dims {val.dims} do not add up to the cache dim {fs.dim}")
    return Inputs(fs.subset(split.train_indices), val.x, val.indices, val.labels, val.totals,
                  val.pipeline, variant, tuple(str(d) for d in dirs),
                  tuple(read_json(Path(d) / META_FILE) for d in dirs), fs.dim)


# --- evaluation and selection ---------------------------------------------------------------

def evaluate(method: BaselineMethod, model, val: Sequence[torch.Tensor], *, unit: int,
             n_chunk: int, device: torch.device, dtype: torch.dtype) -> list[np.ndarray]:
    """Snippet logits of every validation video (eval mode inside, train mode after)."""
    method.set_mode(model, False)
    try:
        return [score_video(method, model, x, unit=unit, n_chunk=n_chunk, device=device,
                            dtype=dtype) for x in val]
    finally:
        method.set_mode(model, True)


def point_aucs(z_list: Sequence[np.ndarray], totals: Sequence[int],
               labels: Sequence[int]) -> dict[str, float]:
    """The two readouts' full-validation V-AUC of one evaluation point (NaN if non-finite)."""
    if not all(np.isfinite(z).all() for z in z_list):
        return {"sigma": math.nan, "logit": math.nan}
    vs = video_scores(z_list, totals)
    return {k: float(v_aucs(vs[k][None, :], labels)[0]) for k in ("sigma", "logit")}


def reselect(val_file: str | Path) -> dict:
    """The selection recomputed from val_scores.npz alone (gates (g), (w); tests)."""
    with np.load(val_file) as f:
        z, counts = f["z"], np.asarray(f["counts"], dtype=np.int64)
        totals, labels = f["total_frames"], f["labels"]
        steps = np.asarray(f["eval_steps"], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    aucs: dict[str, list[float]] = {"sigma": [], "logit": []}
    for e in range(z.shape[0]):
        zl = [z[e, a:a + n] for a, n in zip(starts, counts, strict=True)]
        pa = point_aucs(zl, totals, labels)
        for k in aucs:
            aucs[k].append(pa[k])
    out: dict = {"v_auc": aucs, "failed": None, "selection": {}}
    if not all(np.isfinite(aucs["sigma"])):
        out["failed"] = "non-finite validation logits"
        return out
    for k in ("sigma", "logit"):
        i, tied = select_point(aucs[k])
        out["selection"][k] = {"point": i + 1, "step": int(steps[i]),
                               "v_auc": float(aucs[k][i]), "tied": tied}
    return out


# --- checkpoints ----------------------------------------------------------------------------

def _state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_ckpt(path: Path, state: Mapping, *, method: str, unit: int, rule: str,
              point: int | None, step: int) -> str:
    tmp = path.with_name(path.name + ".part")
    torch.save({"model_state_dict": dict(state), "method": method, "unit": int(unit),
                "rule": rule, "eval_point": point, "step": int(step)}, tmp)
    tmp.replace(path)
    return sha256_file(path)


def _write_json(path: Path, obj) -> None:
    import json

    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8",
                   newline="\n")
    tmp.replace(path)


# --- the loop -------------------------------------------------------------------------------

def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _environment(device: torch.device) -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "numpy": np.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            # recorded, never set (q7 MIL-3): TF32 and the float32 matmul precision
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def _code() -> dict:
    from wvad.diagnostics.common import code_version

    return code_version()


class Chain:
    """One training chain: setup in the constructor, ``train`` runs the steps and the
    evaluation points, ``finish`` writes the checkpoints and records (module docstring)."""

    def __init__(self, cfg: TrainConfig, method: BaselineMethod | None, paths, split,
                 log: Callable[[str], None]) -> None:
        self.t0 = time.time()
        self.cfg, self.log = cfg, log
        self.out = Path(cfg.out_dir)
        if (self.out / FILES["record"]).exists():
            raise SystemExit(f"{self.out / FILES['record']} exists: a completed chain is not "
                             "rerun")
        base = method or get_method(cfg.method)
        if base.name != cfg.method:
            raise SystemExit(f"method object {base.name!r} for --method {cfg.method!r}")
        self.method = dataset_method(base, cfg.dataset)       # the dataset's official config
        self.recipe = require_recipe(self.method.recipe)
        lr_mult, scale_mult = parse_mult(cfg.lr_mult), parse_mult(cfg.scale_mult)
        self.scale = self.recipe.scaled_constants(scale_mult)
        if paths is None:
            from wvad.paths import load_paths

            paths = load_paths(cfg.paths)
        self.split = split if split is not None else load_split(split_file(cfg.dataset))
        self.inputs = load_inputs(paths, cfg.dataset, cfg.pair,
                                  self.recipe.unit_input(cfg.unit), self.split)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = _device(cfg.device)
        self.dtype = torch.float64 if cfg.float64 else torch.float32
        det = deterministic(cfg.deterministic)
        seed_all(cfg.seed)
        model = self.method.build(self.inputs.dim, unit=cfg.unit, input_ln=input_ln(cfg.pair),
                                  scale=self.scale)
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.lr0 = float(self.recipe.optimizer.lr) * lr_mult
        self.optimizer = self.method.make_optimizer(self.model, self.lr0)
        self.sampler = samplers.create(self.recipe.sampler,
                                       samplers.Pools.from_labels(self.inputs.train.labels),
                                       self.recipe.sampler_params)
        self.spe = self.sampler.steps_per_epoch()
        self.steps = int(cfg.iterations) if cfg.iterations else total_steps(self.recipe,
                                                                            self.spe)
        self.points = eval_steps(self.steps, cfg.eval_points)
        self.config = self.resolved(lr_mult, scale_mult, det)
        _write_json(self.out / FILES["config"], self.config)
        self.z_points: list[np.ndarray] = []
        self.aucs: dict[str, list[float]] = {"sigma": [], "logit": []}
        self.hist: list[dict] = []
        self.best: dict[str, tuple[float, int, dict] | None] = {"sigma": None, "logit": None}
        self.failed: str | None = None
        self.eval_ckpts: dict[str, str] = {}
        self.nonfinite_total = 0

    def resolved(self, lr_mult: float, scale_mult: float, det: Mapping) -> dict:
        """The resolved configuration (config.json and the record)."""
        return {**asdict(self.cfg), "lr": self.lr0, "lr_mult_value": lr_mult,
                "scale": self.scale, "scale_mult_value": scale_mult, "steps": self.steps,
                "steps_per_epoch": self.spe, "eval_steps": list(self.points),
                "variant": self.inputs.variant, "pipeline": self.inputs.pipeline.to_json(),
                "input_ln": input_ln(self.cfg.pair), "dtype": str(self.dtype),
                "deterministic_mode": dict(det), "recipe": self.recipe.to_json(),
                "split_sha256": self.split.sha256}

    def train(self) -> None:
        """Steps 1..S; the loss means of every interval are recorded at the evaluation steps
        (with or without evaluating, gate (q5))."""
        cfg, m = self.cfg, self.method
        point_of = {s: k + 1 for k, s in enumerate(self.points)}
        sums: dict[str, float] = {}
        count, nonfinite, lr_now, t_point = 0, 0, self.lr0, time.time()
        m.set_mode(self.model, True)
        for step in range(1, self.steps + 1):
            lr = m.lr_at(step, self.lr0, self.spe)
            if lr != lr_now:
                m.set_lr(self.optimizer, lr)
                lr_now = lr
            batch = collate(self.sampler.draw(step), self.inputs.train.features.__getitem__,
                            cfg.unit, device=self.device, dtype=self.dtype)
            parts = m.train_step(self.model, batch, self.optimizer, unit=cfg.unit,
                                 scale=self.scale)
            nonfinite += not all(math.isfinite(v) for v in parts.values())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            count += 1
            if step in point_of:
                entry = {"point": point_of[step], "step": step, "lr": lr_now,
                         "loss": {n: v / max(count, 1) for n, v in sums.items()},
                         "nonfinite_losses": nonfinite}
                if cfg.evaluate:
                    entry |= self.evaluation_point(point_of[step], step)
                self.hist.append(entry | {"seconds": round(time.time() - t_point, 3)})
                self.nonfinite_total += nonfinite
                sums, count, nonfinite, t_point = {}, 0, 0, time.time()
        self.nonfinite_total += nonfinite

    def evaluation_point(self, k: int, step: int) -> dict:
        """Validation scoring (isolated generators), V-AUCs, running best, D14 checkpoint."""
        cfg, inp = self.cfg, self.inputs
        with isolated_rng(self.device):
            z_list = evaluate(self.method, self.model, inp.val, unit=cfg.unit,
                              n_chunk=cfg.n_chunk, device=self.device, dtype=self.dtype)
        self.z_points.append(np.concatenate(z_list).astype(np.float32))
        pa = point_aucs(z_list, inp.val_totals, inp.val_labels)
        if self.failed is None and not all(math.isfinite(v) for v in pa.values()):
            self.failed = f"non-finite validation logits at evaluation point {k}"
        sat = -1 if self.failed else int((video_scores(z_list, inp.val_totals)["sigma"]
                                          == 1.0).sum())
        for r in ("sigma", "logit"):
            self.aucs[r].append(pa[r])
            b = self.best[r]
            if self.failed is None and (b is None or pa[r] > b[0]):
                self.best[r] = (pa[r], k, _state(self.model))
        if cfg.save_eval_ckpts:
            self.eval_ckpts[f"{k:02d}"] = save_ckpt(
                self.out / EVAL_CKPT.format(k), _state(self.model), method=self.method.name,
                unit=cfg.unit, rule="eval", point=k, step=step)
        self.log(f"[e023 train] {cfg.method} {cfg.dataset}/{cfg.pair} s{cfg.seed} point {k}/"
                 f"{len(self.points)} step {step}: V-AUC sigma {pa['sigma']:.6f} "
                 f"logit {pa['logit']:.6f}")
        return {"v_auc_sigma": pa["sigma"], "v_auc_logit": pa["logit"], "saturated": sat}

    def write_val_scores(self) -> Path:
        inp = self.inputs
        path = self.out / FILES["val"]
        tmp = path.with_name("val_scores.part.npz")
        np.savez(tmp, z=np.stack(self.z_points),
                 counts=np.array([len(z) for z in inp.val], dtype=np.int64),
                 total_frames=np.asarray(inp.val_totals, np.int64),
                 labels=np.asarray(inp.val_labels, np.int64),
                 val_indices=np.asarray(inp.val_indices, np.int64),
                 eval_steps=np.asarray(self.points, np.int64),
                 v_auc_sigma=np.asarray(self.aucs["sigma"], np.float64),
                 v_auc_logit=np.asarray(self.aucs["logit"], np.float64),
                 saturated=np.asarray([h["saturated"] for h in self.hist], np.int64))
        tmp.replace(path)
        return path

    def selected_checkpoints(self, val_path: Path) -> tuple[dict, dict]:
        """The selection recomputed from val_scores.npz (must equal the running choice) and the
        two selected checkpoints written."""
        again, sel, ckpts = reselect(val_path), {}, {}
        for r in ("sigma", "logit"):
            s, b = again["selection"][r], self.best[r]
            if b is None or s["point"] != b[1]:
                raise RuntimeError(f"{r}: the recomputed selection {s} differs from the "
                                   f"running choice {None if b is None else b[1]}")
            sha = save_ckpt(self.out / FILES[r], b[2], method=self.method.name,
                            unit=self.cfg.unit, rule=r, point=s["point"], step=s["step"])
            sel[r] = s
            ckpts[r] = {"file": FILES[r], "sha256": sha, "point": s["point"], "step": s["step"]}
        return sel, ckpts

    def finish(self) -> dict:
        inp, n_points = self.inputs, len(self.points)
        last_sha = save_ckpt(self.out / FILES["last"], _state(self.model),
                             method=self.method.name, unit=self.cfg.unit, rule="last",
                             point=n_points, step=self.steps)
        record: dict = {
            "check": "E023 comparison-method chain (PLAN.md 체크포인트 선택)",
            "config": self.config, "code": _code(), "environment": _environment(self.device),
            "method_module": type(self.method).__module__,
            "parameters": parameter_count(self.model), "failed": self.failed,
            "evaluated": self.cfg.evaluate, "nonfinite_losses": self.nonfinite_total,
            "selection": {}, "eval_ckpts": self.eval_ckpts,
            "checkpoints": {"last": {"file": FILES["last"], "sha256": last_sha,
                                     "point": n_points, "step": self.steps}},
            "inputs": {"variant": inp.variant, "train_dirs": list(inp.train_dirs),
                       "train_data_sha256": [m.get("data_sha256") for m in inp.train_metas],
                       "scale_array_sha256": _scale_hashes(inp.pipeline),
                       "split_sha256": self.split.sha256,
                       "val_indices_sha256": _sha_bytes(
                           np.asarray(inp.val_indices, np.int64).tobytes())}}
        if self.cfg.evaluate:
            val_path = self.write_val_scores()
            record["val_scores_sha256"] = sha256_file(val_path)
            if self.failed is None:
                sel, ckpts = self.selected_checkpoints(val_path)
                record["selection"] = sel
                record["checkpoints"] |= ckpts
        _write_json(self.out / FILES["history"], {"points": self.hist,
                                                  "nonfinite_losses": self.nonfinite_total})
        record["seconds"] = round(time.time() - self.t0, 3)
        _write_json(self.out / FILES["record"], record)
        return record


def run(cfg: TrainConfig, *, method: BaselineMethod | None = None, paths=None, split=None,
        log: Callable[[str], None] = print) -> dict:
    """One chain (module docstring); returns the record written to selection.json."""
    chain = Chain(cfg, method, paths, split, log)
    chain.train()
    return chain.finish()


def _scale_hashes(pipeline: InputPipeline) -> list[str] | None:
    if pipeline.scales is None:
        return None
    from wvad.extract.derive_n import scale_array_sha256

    return [scale_array_sha256(s) for s in pipeline.scales]


# --- command line ---------------------------------------------------------------------------

def parse(argv: Sequence[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(prog="python -m wvad.baselines.train",
                                description="E023 comparison-method training chain")
    p.add_argument("--method", required=True, choices=KNOWN_METHODS)
    p.add_argument("--unit", required=True, type=int, choices=UNITS)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--pair", required=True, choices=PAIRS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lr_mult", default="1", help="D4 lr multiplier, e.g. 1/3, 1, 3, 1/9, 9")
    p.add_argument("--scale_mult", default="1", help="D4 scale-constant multiplier: 1/10, 1, 10")
    p.add_argument("--save_eval_ckpts", action="store_true",
                   help="D14: keep the 30 evaluation-point checkpoints (temporary)")
    p.add_argument("--n_chunk", type=int, default=N_CHUNK_DEFAULT,
                   help="N_chunk (PLAN.md (3j); a power of two)")
    p.add_argument("--deterministic", action="store_true",
                   help="(q1b) mode: deterministic algorithms (CUDA: CUBLAS_WORKSPACE_CONFIG)")
    p.add_argument("--float64", action="store_true", help="gate (q1b) MIL check only")
    p.add_argument("--no_eval", action="store_true", help="gate (q5): train without evaluating")
    p.add_argument("--paths", help="paths file (default configs/paths.yaml)")
    p.add_argument("--device", help="cuda / cpu (default: cuda when available)")
    a = p.parse_args(argv)
    if a.float64 and a.method != "mil":
        p.error("--float64 is the MIL (q1b) check only")
    for m in (a.lr_mult, a.scale_mult):
        try:
            parse_mult(m)
        except (ValueError, ZeroDivisionError) as exc:
            p.error(str(exc))
    return TrainConfig(method=a.method, unit=a.unit, dataset=a.dataset, pair=a.pair,
                       seed=a.seed, out_dir=a.out_dir, lr_mult=a.lr_mult,
                       scale_mult=a.scale_mult, save_eval_ckpts=a.save_eval_ckpts,
                       n_chunk=a.n_chunk, deterministic=a.deterministic, float64=a.float64,
                       evaluate=not a.no_eval, paths=a.paths, device=a.device)


def main(argv: Sequence[str] | None = None) -> None:
    rec = run(parse(argv))
    sel = {k: v["point"] for k, v in rec["selection"].items()}
    print(f"[e023 train] done: failed={rec['failed']} selection={sel} "
          f"-> {rec['config']['out_dir']}")


if __name__ == "__main__":
    main()
