"""Stage 1 (MIL warm-up) and stage 2 (pseudo-label self-training) of the baseline.

Both stages read only the training cache: the validation split is held out and used for
every selection (best checkpoint = highest validation video AUC, strict >, earliest
wins). The test split is never loaded here. Stage 2 starts from a stage-1 checkpoint of
the same split. The order of construction and data access follows the earlier trainers,
so a seed reproduces an earlier run.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.amp import GradScaler

from wvad.data import check_labels, feature_dirs, load_feature_set, load_split, split_file
from wvad.models import SegmentHead
from wvad.paths import load_paths
from wvad.train.loop import (
    LossWeights,
    VideoBags,
    make_loader,
    seed_everything,
    train_one_epoch,
    validation_video_auc,
)
from wvad.train.pseudo import generate_pseudo_labels

INPUT_LAYER_NORM = {"xcb_xcl": False, "xcb_siglip": True,
                    "xcb": False, "xcl": False, "siglip": True}   # LN iff SigLIP is in
CURRICULUM = {1: (3, 0.65), 2: (4, 0.58), 3: (5, 0.50)}     # iteration -> (top_k, confidence)


@dataclass(frozen=True)
class Stage1Config:
    dataset: str
    pair: str
    seed: int
    out_dir: str
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    weights: LossWeights = field(default_factory=LossWeights)
    record: bool = False            # save per-epoch validation scores (does not change training)
    global_mode: str = "none"       # per-video offset on the segment logits (E016; none = off)
    offset_lr_mult: float = 1.0     # learning-rate multiplier of the offset parameters (E017)


def stage2_default_weights() -> LossWeights:
    return LossWeights(dense=0.05, pseudo=1.0, neg_pseudo=0.3)


@dataclass(frozen=True)
class Stage2Config:
    dataset: str
    pair: str
    seed: int
    init: str
    out_dir: str
    iterations: int = 3
    epochs_per_iter: int = 10
    lr: float = 3e-5
    weight_decay: float = 1e-4
    batch_size: int = 32
    curriculum: bool = True
    top_k: int = 4                  # used when curriculum is off
    confidence_min: float = 0.55    # used when curriculum is off
    pseudo_neg_k: int = 4
    top_k_neg: int = 0
    weights: LossWeights = field(default_factory=stage2_default_weights)
    record: bool = False            # save validation and pseudo-label scores (training unchanged)
    global_mode: str = "none"       # must equal the stage-1 checkpoint's (E016)
    offset_lr_mult: float = 1.0     # learning-rate multiplier of the offset parameters (E017)


def _device() -> tuple[torch.device, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, device.type == "cuda"


def _train_val(dataset: str, pair: str):
    split = load_split(split_file(dataset))
    fs = load_feature_set(feature_dirs(load_paths(), dataset, pair), "train")
    check_labels(split, fs.labels, f"{dataset}/{pair} train cache")
    return fs.subset(split.train_indices), fs.subset(split.val_indices), split, fs.dim


def _save(path: str, model, cfg, epoch: int, auc: float, split_sha: str) -> None:
    torch.save({"model_state_dict": {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()},
                "config": asdict(cfg), "epoch": epoch, "val_video_auc": auc,
                "val_split_sha": split_sha}, path)


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def _adam_steps(optimizer: torch.optim.Optimizer) -> int:
    """Optimizer steps taken so far (the GradScaler skips the step on non-finite gradients);
    the largest step count over all parameters, independent of parameter order."""
    return max((int(s["step"]) for s in optimizer.state.values() if "step" in s), default=0)


def _check_offset_lr_mult(cfg) -> None:
    m = cfg.offset_lr_mult
    if not (math.isfinite(m) and m > 0):
        raise ValueError(f"offset_lr_mult must be finite and > 0, got {m}")
    if m != 1.0 and cfg.global_mode == "none":
        raise ValueError(f"offset_lr_mult {m} needs a global_mode with offset parameters")


def _optimizer(model, cfg) -> torch.optim.AdamW:
    """AdamW over the head. Without a global offset (none) this is the earlier optimizer (one
    group, same parameter order). With one, the offset parameters (names offset_*) always form
    a second group with learning rate x offset_lr_mult (E017), also at 1, so the candidates'
    two-group path is the one the no-change gate checks against E016; AdamW's update is
    elementwise, so the split alone changes nothing. Gradients are unchanged."""
    _check_offset_lr_mult(cfg)
    if cfg.global_mode == "none":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    named = list(model.named_parameters())
    rest = [p for n, p in named if not n.startswith("offset_")]
    offset = [p for n, p in named if n.startswith("offset_")]
    return torch.optim.AdamW([{"params": rest},
                              {"params": offset, "lr": cfg.lr * cfg.offset_lr_mult}],
                             lr=cfg.lr, weight_decay=cfg.weight_decay)


def _offset_entry(model, n_batches: int, steps_before: int,
                  optimizer: torch.optim.Optimizer) -> dict:
    """History fields of a global-offset head (never added for global_mode none)."""
    out = {"amp_skipped_steps": n_batches - (_adam_steps(optimizer) - steps_before)}
    if model.global_mode == "mean":
        out["offset_gamma"] = float(model.offset_gamma.detach())
    else:
        out["offset_w_norm"] = float(model.offset_w.detach().norm())
        out["offset_b"] = float(model.offset_b.detach())
    return out


def _save_offset_record(out_dir: str, epochs: np.ndarray, offset_record: list, split) -> None:
    np.savez_compressed(os.path.join(out_dir, "record_val_offset.npz"), epochs=epochs,
                        offsets=np.stack([o for o, _ in offset_record]),
                        local_scores=np.stack([s for _, s in offset_record]),
                        val_indices=np.asarray(split.val_indices))


def _pseudo_array(pseudo: list[torch.Tensor | None], n_segments: int) -> np.ndarray:
    """(n_videos, S) pseudo-labels; NaN where a segment (or a whole video) has none."""
    return np.stack([np.full(n_segments, np.nan, dtype=np.float32) if p is None
                     else p.numpy().astype(np.float32) for p in pseudo])


def run_stage1(cfg: Stage1Config) -> dict:
    _check_offset_lr_mult(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device, use_amp = _device()
    seed_everything(cfg.seed)
    train_fs, val_fs, split, dim = _train_val(cfg.dataset, cfg.pair)
    train_loader = make_loader(VideoBags(train_fs), cfg.batch_size, True, device)
    val_loader = make_loader(VideoBags(val_fs), cfg.batch_size, False, device)
    model = SegmentHead(dim, use_input_ln=INPUT_LAYER_NORM[cfg.pair],
                        global_mode=cfg.global_mode).to(device)
    optimizer = _optimizer(model, cfg)
    scaler = GradScaler() if use_amp else None
    best, best_epoch, history = -math.inf, None, []
    val_record = [] if cfg.record else None
    offset = cfg.global_mode != "none"
    offset_record = [] if cfg.record and offset else None
    for epoch in range(1, cfg.epochs + 1):
        steps_before = _adam_steps(optimizer)
        losses = train_one_epoch(model, train_loader, optimizer, scaler, cfg.weights, device,
                                 use_amp)
        auc = validation_video_auc(model, val_loader, device, record=val_record,
                                   offset_record=offset_record)
        entry = {"epoch": epoch, **losses, "val_video_auc": auc}
        if offset:
            entry |= _offset_entry(model, len(train_loader), steps_before, optimizer)
        history.append(entry)
        _save(os.path.join(cfg.out_dir, "ckpt_last.pt"), model, cfg, epoch, auc, split.sha256)
        if auc > best:
            best, best_epoch = auc, epoch
            _save(os.path.join(cfg.out_dir, "ckpt_best_val.pt"), model, cfg, epoch, auc,
                  split.sha256)
        print(f"[stage1] epoch {epoch}: {losses} val_video_auc={auc:.6f}", flush=True)
    selection = {"best_val_video_auc": best, "best_epoch": best_epoch,
                 "val_split_sha": split.sha256}
    _write_json(os.path.join(cfg.out_dir, "history.json"), history)
    _write_json(os.path.join(cfg.out_dir, "selection.json"), selection)
    if cfg.record:
        np.savez_compressed(os.path.join(cfg.out_dir, "record_val_scores.npz"),
                            epochs=np.arange(1, cfg.epochs + 1), scores=np.stack(val_record),
                            val_indices=np.asarray(split.val_indices))
        if offset_record is not None:
            _save_offset_record(cfg.out_dir, np.arange(1, cfg.epochs + 1), offset_record, split)
    return selection


def _pseudo_schedule(cfg: Stage2Config, iteration: int) -> tuple[int, float]:
    if cfg.curriculum:
        return CURRICULUM.get(iteration, CURRICULUM[max(CURRICULUM)])
    return cfg.top_k, cfg.confidence_min


def run_stage2(cfg: Stage2Config) -> dict:
    _check_offset_lr_mult(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device, use_amp = _device()
    seed_everything(cfg.seed)
    split = load_split(split_file(cfg.dataset))
    init = torch.load(cfg.init, map_location="cpu", weights_only=True)
    if init.get("val_split_sha") != split.sha256:
        raise ValueError(f"{cfg.init} was trained on split {init.get('val_split_sha')}, "
                         f"not {split.sha256}")
    init_mode = init.get("config", {}).get("global_mode", "none")
    if init_mode != cfg.global_mode:
        raise ValueError(f"{cfg.init} has global_mode {init_mode!r}, stage 2 asks for "
                         f"{cfg.global_mode!r}")
    train_fs, val_fs, _, dim = _train_val(cfg.dataset, cfg.pair)
    model = SegmentHead(dim, use_input_ln=INPUT_LAYER_NORM[cfg.pair],
                        global_mode=cfg.global_mode)
    model.load_state_dict(init["model_state_dict"])
    model.to(device)
    bags = VideoBags(train_fs, with_pseudo=True)
    val_loader = make_loader(VideoBags(val_fs), cfg.batch_size, False, device)
    optimizer = _optimizer(model, cfg)
    scaler = GradScaler() if use_amp else None
    val_record = [] if cfg.record else None
    offset = cfg.global_mode != "none"
    offset_record = [] if cfg.record and offset else None
    base = validation_video_auc(model, val_loader, device, record=val_record,
                                offset_record=offset_record)
    best, best_epoch, history = -math.inf, None, []
    pseudo_scores, pseudo_labels, schedule = [], [], []
    for iteration in range(1, cfg.iterations + 1):
        top_k, conf = _pseudo_schedule(cfg, iteration)
        scores_record = [] if cfg.record else None
        pseudo, stats = generate_pseudo_labels(model, train_fs.features, train_fs.labels,
                                               device, top_k, conf, use_amp, cfg.top_k_neg,
                                               cfg.pseudo_neg_k, record=scores_record)
        if cfg.record:
            pseudo_scores.append(torch.stack(scores_record).numpy())
            pseudo_labels.append(_pseudo_array(pseudo, pseudo_scores[-1].shape[1]))
            schedule.append((top_k, conf))
        bags.set_pseudo_labels(pseudo)
        train_loader = make_loader(bags, cfg.batch_size, True, device)
        for ep in range(cfg.epochs_per_iter):
            epoch = (iteration - 1) * cfg.epochs_per_iter + ep + 1
            steps_before = _adam_steps(optimizer)
            losses = train_one_epoch(model, train_loader, optimizer, scaler, cfg.weights,
                                     device, use_amp)
            auc = validation_video_auc(model, val_loader, device, record=val_record,
                                       offset_record=offset_record)
            entry = {"iteration": iteration, "epoch": epoch, "top_k": top_k,
                     "confidence_min": conf, **asdict(stats), **losses, "val_video_auc": auc}
            if offset:
                entry |= _offset_entry(model, len(train_loader), steps_before, optimizer)
            history.append(entry)
            if auc > best:
                best, best_epoch = auc, epoch
                _save(os.path.join(cfg.out_dir, "ckpt_best_val.pt"), model, cfg, epoch, auc,
                      split.sha256)
            print(f"[stage2] iter {iteration} epoch {epoch}: val_video_auc={auc:.6f}",
                  flush=True)
    _save(os.path.join(cfg.out_dir, "ckpt_last.pt"), model, cfg,
          cfg.iterations * cfg.epochs_per_iter, history[-1]["val_video_auc"], split.sha256)
    selection = {"base_val_video_auc": base, "best_val_video_auc": best,
                 "best_epoch": best_epoch, "val_split_sha": split.sha256}
    _write_json(os.path.join(cfg.out_dir, "history.json"), history)
    _write_json(os.path.join(cfg.out_dir, "selection.json"), selection)
    if cfg.record:
        if offset_record is not None:
            _save_offset_record(cfg.out_dir,
                                np.arange(0, cfg.iterations * cfg.epochs_per_iter + 1),
                                offset_record, split)
        _save_stage2_record(cfg, model, train_fs, split, device, use_amp, val_record,
                            pseudo_scores, pseudo_labels, schedule)
    return selection


def _save_stage2_record(cfg: Stage2Config, model, train_fs, split, device, use_amp,
                        val_record, pseudo_scores, pseudo_labels, schedule) -> None:
    """Write the recorded scores, then score the final model (round 4) after everything else
    is saved, so neither the training run nor its outputs depend on this step."""
    n_epochs = cfg.iterations * cfg.epochs_per_iter
    np.savez_compressed(os.path.join(cfg.out_dir, "record_val_scores.npz"),
                        epochs=np.arange(0, n_epochs + 1), scores=np.stack(val_record),
                        val_indices=np.asarray(split.val_indices))
    np.savez_compressed(os.path.join(cfg.out_dir, "record_pseudo.npz"),
                        rounds=np.arange(1, cfg.iterations + 1), scores=np.stack(pseudo_scores),
                        labels=np.stack(pseudo_labels),
                        top_k=np.array([s[0] for s in schedule]),
                        confidence=np.array([s[1] for s in schedule]),
                        train_indices=np.asarray(split.train_indices))
    final_path = os.path.join(cfg.out_dir, "record_pseudo_final.npz")
    if os.path.exists(final_path):
        os.remove(final_path)                       # never leave a stale file from a retry
    try:
        final: list[torch.Tensor] = []
        top_k, conf = schedule[-1]
        generate_pseudo_labels(model, train_fs.features, train_fs.labels, device, top_k, conf,
                               use_amp, cfg.top_k_neg, cfg.pseudo_neg_k, record=final)
        np.savez_compressed(final_path, round=np.array(cfg.iterations + 1),
                            scores=torch.stack(final).numpy(),
                            train_indices=np.asarray(split.train_indices))
    except Exception as exc:                        # the finished run must stay usable
        print(f"[stage2] round-4 scoring failed (run outputs are complete): {exc!r}",
              flush=True)
