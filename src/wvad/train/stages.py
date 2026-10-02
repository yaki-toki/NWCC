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

INPUT_LAYER_NORM = {"xcb_xcl": False, "xcb_siglip": True}
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


def run_stage1(cfg: Stage1Config) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    device, use_amp = _device()
    seed_everything(cfg.seed)
    train_fs, val_fs, split, dim = _train_val(cfg.dataset, cfg.pair)
    train_loader = make_loader(VideoBags(train_fs), cfg.batch_size, True, device)
    val_loader = make_loader(VideoBags(val_fs), cfg.batch_size, False, device)
    model = SegmentHead(dim, use_input_ln=INPUT_LAYER_NORM[cfg.pair]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler() if use_amp else None
    best, best_epoch, history = -math.inf, None, []
    for epoch in range(1, cfg.epochs + 1):
        losses = train_one_epoch(model, train_loader, optimizer, scaler, cfg.weights, device,
                                 use_amp)
        auc = validation_video_auc(model, val_loader, device)
        history.append({"epoch": epoch, **losses, "val_video_auc": auc})
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
    return selection


def _pseudo_schedule(cfg: Stage2Config, iteration: int) -> tuple[int, float]:
    if cfg.curriculum:
        return CURRICULUM.get(iteration, CURRICULUM[max(CURRICULUM)])
    return cfg.top_k, cfg.confidence_min


def run_stage2(cfg: Stage2Config) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    device, use_amp = _device()
    seed_everything(cfg.seed)
    split = load_split(split_file(cfg.dataset))
    init = torch.load(cfg.init, map_location="cpu", weights_only=True)
    if init.get("val_split_sha") != split.sha256:
        raise ValueError(f"{cfg.init} was trained on split {init.get('val_split_sha')}, "
                         f"not {split.sha256}")
    train_fs, val_fs, _, dim = _train_val(cfg.dataset, cfg.pair)
    model = SegmentHead(dim, use_input_ln=INPUT_LAYER_NORM[cfg.pair])
    model.load_state_dict(init["model_state_dict"])
    model.to(device)
    bags = VideoBags(train_fs, with_pseudo=True)
    val_loader = make_loader(VideoBags(val_fs), cfg.batch_size, False, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler() if use_amp else None
    base = validation_video_auc(model, val_loader, device)
    best, best_epoch, history = -math.inf, None, []
    for iteration in range(1, cfg.iterations + 1):
        top_k, conf = _pseudo_schedule(cfg, iteration)
        pseudo, stats = generate_pseudo_labels(model, train_fs.features, train_fs.labels,
                                               device, top_k, conf, use_amp, cfg.top_k_neg,
                                               cfg.pseudo_neg_k)
        bags.set_pseudo_labels(pseudo)
        train_loader = make_loader(bags, cfg.batch_size, True, device)
        for ep in range(cfg.epochs_per_iter):
            epoch = (iteration - 1) * cfg.epochs_per_iter + ep + 1
            losses = train_one_epoch(model, train_loader, optimizer, scaler, cfg.weights,
                                     device, use_amp)
            auc = validation_video_auc(model, val_loader, device)
            history.append({"iteration": iteration, "epoch": epoch, "top_k": top_k,
                            "confidence_min": conf, **asdict(stats), **losses,
                            "val_video_auc": auc})
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
    return selection
