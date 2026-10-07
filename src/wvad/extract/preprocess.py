"""Frame preprocessing of the dense extraction (E021 PLAN.md "전처리·인코더").

The decode workers hand over uint8 RGB frames at the encoder's input size: BGR -> RGB
(``cv2.cvtColor``), then ``cv2.resize`` to a square of the input size (default interpolation)
when the frame is not already that size, the order of the earlier readers. The GPU consumer
normalises with one 256-entry lookup table per channel whose values are the earlier numpy
expression ((v / 255) - mean) / std in float32, so the result equals the earlier CPU
normalisation bit for bit; ``normalize_numpy`` is that reference, and ``NumpyNormalizer`` (kind
"numpy" of ``make_normalizer``) the CPU fallback should the gate find a difference.
Dividing by a Python scalar on CUDA becomes a reciprocal multiplication and differs from numpy
(PLAN.md "고정 전 측정"); ``normalization_mismatches`` measures this for the gate record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# torch is imported where it is used, so the decode workers (numpy and OpenCV only) stay light
if TYPE_CHECKING:
    import torch

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def to_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """(size, size, 3) uint8 RGB: cvtColor, then resize when the shape differs."""
    import cv2  # optional dependency (extract)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size))
    return rgb


def black(size: int) -> np.ndarray:
    """The earlier readers' missing frame: zeros of the input size (no conversion changes it)."""
    return np.zeros((size, size, 3), dtype=np.uint8)


def normalize_numpy(rgb: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """The earlier normalisation: (rgb.astype(float32) / 255.0 - mean) / std, float32, (..., 3)."""
    return (np.asarray(rgb).astype(np.float32) / 255.0 - mean) / std


def lookup_table(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(3, 256) float32: entry [c, v] = normalize_numpy of value v in channel c."""
    values = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)     # (256, 3)
    return np.ascontiguousarray(normalize_numpy(values, mean, std).T)


class Normalizer:
    """uint8 frames (N, H, W, 3) on a device -> float32 (N, 3, H, W) by table lookup."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, device: torch.device):
        import torch

        self.table = torch.from_numpy(lookup_table(mean, std)).to(device).reshape(-1)
        self.offsets = (torch.arange(3, device=device) * 256).reshape(1, 1, 1, 3)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        import torch

        if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected uint8 (N, H, W, 3) frames, got {tuple(frames.shape)} "
                             f"{frames.dtype}")
        idx = frames.long() + self.offsets
        return self.table[idx].permute(0, 3, 1, 2).contiguous()


class NumpyNormalizer:
    """The CPU fallback: normalize_numpy on the CPU, then the float32 values to the device."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, device: torch.device):
        self.mean, self.std, self.device = mean, std, device

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        import torch

        x = normalize_numpy(frames.cpu().numpy(), self.mean, self.std)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(self.device)


NORMALIZERS = {"lut": Normalizer, "numpy": NumpyNormalizer}


def make_normalizer(mean: np.ndarray, std: np.ndarray, device: torch.device,
                    kind: str = "lut") -> Normalizer | NumpyNormalizer:
    """``lut`` (production: table lookup on the device) or ``numpy`` (the CPU fallback)."""
    if kind not in NORMALIZERS:
        raise ValueError(f"normalisation {kind!r} not in {tuple(NORMALIZERS)}")
    return NORMALIZERS[kind](mean, std, device)


def clips_to_pixels(clips: torch.Tensor, normalizer: Normalizer | NumpyNormalizer
                    ) -> torch.Tensor:
    """(B, F, H, W, 3) uint8 clips -> (B * F, 3, H, W) float32 pixel values, clip-major (the
    order of the earlier ``clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)``)."""
    b, f = clips.shape[:2]
    return normalizer(clips.reshape(b * f, *clips.shape[2:]))


def normalization_mismatches(device: torch.device) -> dict:
    """Over all 256 values and 3 channels (CLIP mean / std), how many results differ from the
    numpy reference: x / 255 with a Python scalar on ``device`` (256 values), the whole
    normalisation with Python-scalar division (768 pairs), the same with device tensors, and the
    lookup table (768 pairs; must be 0)."""
    import torch

    values = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    ref = normalize_numpy(values, CLIP_MEAN, CLIP_STD)
    ref_div = values[:, 0].astype(np.float32) / 255.0
    x = torch.from_numpy(values).to(device)
    mean_t = torch.from_numpy(CLIP_MEAN).to(device)
    std_t = torch.from_numpy(CLIP_STD).to(device)
    div_scalar = (x[:, 0].float() / 255.0).cpu().numpy()
    full_scalar = ((x.float() / 255.0 - mean_t) / std_t).cpu().numpy()
    full_tensor = ((x.float() / torch.tensor(255.0, device=device) - mean_t) / std_t).cpu().numpy()
    lut = Normalizer(CLIP_MEAN, CLIP_STD, device)(x.reshape(256, 1, 1, 3))
    lut = lut.reshape(256, 3).cpu().numpy()

    def differ(a: np.ndarray, b: np.ndarray) -> int:
        return int((a.view(np.uint32) != b.view(np.uint32)).sum())

    return {"device": str(device), "values": 256, "pairs": 768,
            "div255_python_scalar": differ(div_scalar, ref_div),
            "normalize_python_scalar": differ(full_scalar, ref),
            "normalize_device_tensor": differ(full_tensor, ref),
            "normalize_lookup_table": differ(lut, ref)}
