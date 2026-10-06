"""Score a trained head on a feature set the way the training loop scores validation videos
(``wvad.train.loop.validation_video_auc``): the same loader (no shuffle), eval mode, fp32 with
autocast off. Device and batch size are arguments (training: its device and batch 32). Returns
the segment logits (for an offset head z + c), the fp32 probabilities sigma(.) that the
training records store and, for an offset head, its parts from the same batches: the
offset-free logits z (``SegmentHead._parts``, not sigma(z), which saturates at 1.0 in fp32)
and the offsets c(v).

Used in stage F, where a checkpoint selected on validation videos is scored on the test set;
the same function re-scores the validation videos so the path can be checked against the
training records.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.amp import autocast

from wvad.data import FeatureSet
from wvad.models.head import SegmentHead
from wvad.train.loop import VideoBags, make_loader
from wvad.train.stages import INPUT_LAYER_NORM

BATCH_SIZE = 32


def load_head(ckpt: str | os.PathLike[str], dim: int,
              device: torch.device) -> tuple[SegmentHead, dict]:
    """The checkpoint's head (input dimension ``dim``, LayerNorm and offset mode from its
    config), in eval mode on ``device``."""
    obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = obj["config"]
    model = SegmentHead(dim, use_input_ln=INPUT_LAYER_NORM[cfg["pair"]],
                        global_mode=cfg.get("global_mode", "none"))
    model.load_state_dict(obj["model_state_dict"])
    return model.to(device).eval(), obj


def score(model: SegmentHead, features: FeatureSet, device: torch.device,
          batch_size: int = BATCH_SIZE) -> dict[str, np.ndarray]:
    """float32 arrays: 'logits' and 'probs' (n_videos, S); an offset head adds 'local_logits'
    z (n_videos, S) and 'offsets' c (n_videos,)."""
    model.eval()
    offset = model.global_mode != "none"
    out: dict[str, list] = {"logits": [], "probs": []}
    if offset:
        out |= {"local_logits": [], "offsets": []}
    loader = make_loader(VideoBags(features), batch_size, False, device)
    with torch.no_grad():
        for batch in loader:
            feats = batch[0].to(device).float()
            with autocast(device_type=device.type, enabled=False):
                _, seg_logits, seg_scores = model(feats)
                if offset:
                    _, z, c = model._parts(feats)
                    out["local_logits"].extend(z.float().cpu().numpy())
                    out["offsets"].extend(c.float().cpu().numpy())
            out["logits"].extend(seg_logits.float().cpu().numpy())
            out["probs"].extend(seg_scores.float().cpu().numpy())
    return {k: (np.asarray(v, dtype=np.float32) if k == "offsets"
                else np.stack(v).astype(np.float32)) for k, v in out.items()}
