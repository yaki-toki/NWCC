"""Frozen image-text encoders of the dense extraction (E021 PLAN.md "전처리·인코더").

Same pooling as the earlier caches:

    xcb, xcl  XCLIPVisionModel last_hidden_state averaged over the 8 frames and all tokens
              (microsoft/xclip-base-patch16 768-d, microsoft/xclip-large-patch14 1024-d; 224 px,
              CLIP mean / std)
    siglip    SiglipVisionModel pooler_output averaged over the 8 frames
              (google/siglip-so400m-patch14-384, 1152-d; 384 px, mean = std = 0.5)

Revisions are pinned (loaded with ``revision=`` and checked). Precision modes: ``autocast``
(CUDA fp16 autocast, production for both datasets), ``fp32`` (no autocast, TF32 flags left at
their values: PyTorch's default is cuDNN TF32 convolutions on, TF32 matmul off, the earlier XD
setting) and ``fp32_strict`` (TF32 off for convolutions and matmul while it runs). Outputs keep
the encoder's dtype (under autocast XCB / XCL float32, SigLIP float16), as stored.
"""

from __future__ import annotations

import contextlib
import platform
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wvad.extract.preprocess import CLIP_MEAN, CLIP_STD, SIGLIP_MEAN, SIGLIP_STD

PRECISIONS = ("autocast", "fp32", "fp32_strict")
FRAMES_PER_CLIP = 8


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    model_id: str
    revision: str
    family: str                  # "xclip" or "siglip"
    dim: int
    input_size: int
    mean: np.ndarray
    std: np.ndarray
    autocast_dtype: str          # stored dtype of the production (autocast) output


SPECS = {
    "xcb": EncoderSpec("xcb", "microsoft/xclip-base-patch16",
                       "d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6", "xclip", 768, 224,
                       CLIP_MEAN, CLIP_STD, "float32"),
    "xcl": EncoderSpec("xcl", "microsoft/xclip-large-patch14",
                       "a9dd1429a16cf305df2aaea232d5e8dceba1c675", "xclip", 1024, 224,
                       CLIP_MEAN, CLIP_STD, "float32"),
    "siglip": EncoderSpec("siglip", "google/siglip-so400m-patch14-384",
                          "9fdffc58afc957d1a03a25b10dba0329ab15c2a3", "siglip", 1152, 384,
                          SIGLIP_MEAN, SIGLIP_STD, "float16"),
}
LEGACY_BATCH = {"ucf": {"xcb": 8, "xcl": 8, "siglip": 4}, "xdv": {"xcb": 32, "xcl": 32,
                                                                  "siglip": 32}}
LEGACY_PRECISION = {"ucf": "autocast", "xdv": "fp32"}


def spec(name: str) -> EncoderSpec:
    if name not in SPECS:
        raise KeyError(f"unknown encoder '{name}' (known: {', '.join(SPECS)})")
    return SPECS[name]


class Encoder:
    """One frozen encoder in eval mode: clips of 8 frames -> one feature per clip."""

    def __init__(self, encoder_spec: EncoderSpec, model: torch.nn.Module, device: torch.device,
                 snapshot: str | None = None):
        self.spec, self.model, self.device = encoder_spec, model.eval().to(device), device
        self.snapshot = snapshot

    @classmethod
    def load(cls, name: str, device: torch.device) -> Encoder:
        """The pinned revision (offline with HF_HUB_OFFLINE=1 and the files in the cache); the
        cache snapshot holding its config must be that revision."""
        from huggingface_hub import try_to_load_from_cache

        s = spec(name)
        if s.family == "xclip":
            from transformers import XCLIPVisionModel as Model
        else:
            from transformers import SiglipVisionModel as Model
        model = Model.from_pretrained(s.model_id, revision=s.revision)
        loaded = getattr(model.config, "_commit_hash", None)
        cached = try_to_load_from_cache(s.model_id, "config.json", revision=s.revision)
        snapshot = Path(cached).parent.name if isinstance(cached, str) else None
        if (loaded and loaded != s.revision) or (snapshot and snapshot != s.revision):
            raise RuntimeError(f"{s.model_id}: loaded revision {loaded or snapshot} != pinned "
                               f"{s.revision}")
        return cls(s, model, device, snapshot)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B * 8, 3, H, W) float32 pixel values on the device, clip-major -> (B, D) on the CPU."""
        n = pixels.shape[0]
        if n % FRAMES_PER_CLIP:
            raise ValueError(f"{n} frames are not whole clips of {FRAMES_PER_CLIP}")
        b = n // FRAMES_PER_CLIP
        out = self.model(pixel_values=pixels)
        if self.spec.family == "xclip":
            hidden = out.last_hidden_state                       # (B * T, tokens, D)
            _, tokens, dim = hidden.shape
            feats = hidden.reshape(b, FRAMES_PER_CLIP, tokens, dim).mean(dim=[1, 2])
        else:
            feats = out.pooler_output.reshape(b, FRAMES_PER_CLIP, -1).mean(dim=1)
        return feats.cpu()

    def info(self) -> dict:
        cfg = self.model.config
        dtypes = sorted({str(p.dtype) for p in self.model.parameters()})
        return {"name": self.spec.name, "model_id": self.spec.model_id,
                "revision": self.spec.revision,
                "loaded_commit": getattr(cfg, "_commit_hash", None) or self.snapshot,
                "attn_implementation": getattr(cfg, "_attn_implementation", None),
                "parameter_dtypes": dtypes, "input_size": self.spec.input_size,
                "dim": self.spec.dim}


def tf32_settings() -> dict:
    return {"cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn.deterministic": bool(torch.backends.cudnn.deterministic)}


@contextlib.contextmanager
def precision(mode: str, device: torch.device) -> Iterator[None]:
    """Run encoders under one of PRECISIONS (flags restored afterwards)."""
    if mode not in PRECISIONS:
        raise ValueError(f"precision {mode!r} not in {PRECISIONS}")
    if mode == "autocast":
        if device.type != "cuda":
            raise ValueError("autocast (fp16) extraction needs a CUDA device")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
        return
    if mode == "fp32":
        yield
        return
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except ImportError:
        return None


def extraction_environment(pip_freeze: bool = True) -> dict:
    """Versions and flags of this extraction environment (env_extract.json, meta.json runs)."""
    import cv2  # optional dependency (extract)

    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                check=False).stdout.strip() or None
    except OSError:
        driver = None
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "torch": torch.__version__, "torch.version.cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "opencv": cv2.__version__,
           "transformers": _version("transformers"), "numpy": np.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "driver": driver, "tf32": tf32_settings()}
    if pip_freeze:
        env["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                           capture_output=True, text=True,
                                           check=False).stdout.splitlines()
    return env


ENV_KEYS = ("python", "torch", "torch.version.cuda", "cudnn", "opencv", "transformers",
            "numpy", "gpu", "driver", "tf32")


def same_environment(a: dict, b: dict) -> bool:
    """Equal on ENV_KEYS (and pip freeze when both record it)."""
    if any(a.get(k) != b.get(k) for k in ENV_KEYS):
        return False
    return "pip_freeze" not in a or "pip_freeze" not in b or a["pip_freeze"] == b["pip_freeze"]
