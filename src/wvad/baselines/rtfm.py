"""RTFM (Tian et al., ICCV 2021, "Weakly-supervised Video Anomaly Detection with Robust Temporal
Feature Magnitude Learning") for E023 (PLAN.md "팔", "입력 처리 표", "구현"). New code (add-only
rule); the porting contract is in wvad.baselines.common.

Official repository: https://github.com/tianyu0207/RTFM at commit
cf938f4c38c908ba19936ecfc60d54de7da0d8b9 (results/e023_licences.json key "rtfm").
Licence: none detected (no LICENSE file in the repository). Porting mode: re-implementation; the
official code is not copied and is used only outside the repository, loaded by path through
wvad.baselines.official, by the equivalence harness wvad.baselines.check_rtfm (gates (q1),
(q1b), (q1c), (q1e)).

What is re-implemented (official file and lines at that commit)
    network     model.py L13-L189: the multi-scale temporal network (three dilated 1-D
                convolutions, dilation 1 / 2 / 4, 512 channels each, a 1x1 convolution to 512
                followed by a 1-D non-local block with a zero-initialised BatchNorm on its output,
                a fusing 3-tap convolution of the 4 x 512 channels back to the input width,
                residual), dropout 0.7, the snippet classifier D -> 512 -> 128 -> 1 with ReLU and
                a sigmoid; Xavier-uniform weights and zero biases for every convolution and
                linear layer, applied after construction. Submodule names and the construction
                order follow the official network, so the global generator is consumed as by its
                constructor and a state_dict loads into either.
    selection   model.py L191-L273: feature magnitudes (l2 over channels, mean over crops); per
                video the k = 32 // 10 = 3 snippets of largest magnitude after a dropout(0.7)
                mask on the magnitudes (abnormal half first, then normal half), their features
                (every crop) and the mean of their scores.
    loss        train.py L10-L22, L50-L79, L94-L105: BCE of the selected mean scores (normal
                0, abnormal 1) + alpha * mean((|m - ||mean f_abn||| + ||mean f_nor||)^2) with
                alpha 1e-4 and the margin m = 100 (a feature-norm scale constant, the recipe's
                ``scale_constants["margin"]``), + 8e-4 * the squared differences of consecutive
                abnormal snippet scores (concatenated over the abnormal videos, last value
                repeated) + 8e-3 * the l2 norm of the abnormal snippet scores; summed in the
                official order.
    training    main.py L20-L25, L40-L41, L48-L63, train.py L82-L112: Adam(lr 1e-3, weight
                decay 5e-3), constant lr, 15000 optimizer steps, each on 32 normal + 32
                abnormal videos from two shuffled drop_last loaders re-created when exhausted
                (the generic "window_mean" sampler, normal first), zero_grad / backward / step.
    scoring     test_10crop.py L11-L19: every snippet of the whole video in one sequence, eval
                mode; the official score is the sigmoid of fc3, ``sequence_logits`` returns the
                fc3 output itself (its fp32 sigmoid is the official score); that per-snippet
                sigmoid is computed off-device in fp32 NumPy (common.score_video), the official
                one on the device: ULP-level differences, within the (q1e) tolerance.

Adaptations: (1) input dimension D = 1792 (XCB+XCL) / 1920 (XCB+SigLIP) instead of the I3D 2048:
the input width of the three dilated convolutions and of the 1x1 convolution, the output width
of the fusing convolution and its BatchNorm (the residual) and the input of fc1; the 512-channel
widths and the fusing convolution's 4 x 512 = 2048 input channels are unchanged. (2) Single crop
instead of 10-crop features: inputs (B, 1, T, D); the official normal/abnormal feature split
hard-codes 10 crops (``features[0:batch_size*10]``), here it uses the input's crop count. (3)
Input LayerNorm (``common.input_layer``) for XCB+SigLIP only (PLAN.md "입력 처리 표", P7). (4) The
normal/abnormal split point is the batch's ``n_normal`` (official: the model's ``batch_size``
argument, the same number); labels stay on the scores' device (official ``label.cuda()``). (5)
Validation / test scoring returns the pre-sigmoid fc3 output; it runs only the network's score
path, so a piece shorter than k = 3 snippets (only the last N_chunk piece or a video under 3
snippets) is scored where the official forward would raise in torch.topk of its magnitude
selection, which does not feed the scores; videos with n_v > N_chunk are cut by the plan's
N_chunk rule (``common.score_video``), otherwise the whole video is one sequence as officially.
(6) Training input: the official per-item ``process_feat`` 32-window mean is precomputed by
wvad.extract.derive_n variant d16m32 (the official float32 numpy windows of the scale-matched
snippets, bit for bit; the scale map fitted on float64 window means; PLAN.md "검증·시험 입력",
기록 2026-10-20). (7) Checkpoint selection on the
validation V-AUC at 30 evaluation points (common, PLAN.md "체크포인트 선택") instead of the test AUC
every 5 steps after step 200; no test call before training; the generators are seeded before
construction (the official code seeds nothing). (8) XD-Violence uses the official settings
above (the repository configures only ShanghaiTech and UCF-Crime; option.py defaults apply to
both).

Paper vs official code (q7 RTFM-1, RTFM-2): the code at cf938f4c is followed wherever it differs
from the paper (PLAN.md rule: the official code is the reference). The paper (arXiv 2101.10030,
Sec. 4.2 "Implementation Details") states Adam with weight decay 0.0005, batch size 64 (32 normal
+ 32 abnormal) for 50 epochs, lr 0.001 for ShanghaiTech and UCF-Crime and lr 0.0001 for
XD-Violence (m = 100, k = 3, T = 32); the code uses weight decay 0.005 (main.py L41), lr 0.001
for every dataset (option.py L11) and 15000 steps (option.py L19). The paper's XD lr is x1/10 of
the code value (closest D4 value: the widened x1/9, base grid x1/3). The paper's feature
magnitude loss (Sec. 3.3, Eq. 6) is the hinge max(0, m - (g_k(X+) - g_k(X-))); the code's is
(|m - ||f_a||| + ||f_n||)^2 (train.py L71-L77); the code also applies dropout to the top-k
selection mask (model.py L231-L232, L253-L254), which the paper does not describe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from wvad.baselines.common import BaselineMethod, Batch, LossOutput, MethodError, input_layer
from wvad.baselines.recipes import (
    InputRow,
    OptimizerSpec,
    Recipe,
    ScheduleSpec,
    Source,
    UnitInput,
)

NAME = "rtfm"
OFFICIAL_UNIT = 32              # model.py L177 num_segments; dataset.py L70 process_feat(., 32)
TOP_K = OFFICIAL_UNIT // 10     # model.py L178-L179
WIDTH = 512                     # model.py L116-L139
HIDDEN = (512, 128)             # model.py L182-L184
DROPOUT = 0.7                   # model.py L186
DILATIONS = (1, 2, 4)           # model.py L115-L135
VARIANT = "d16m32"              # derive_n kind m = utils.py process_feat windows, scale-matched


def _src(file: str, lines: str, note: str = "") -> Source:
    return Source("rtfm", file, lines, note)


RECIPE = Recipe(
    method=NAME,
    official_unit=OFFICIAL_UNIT,
    units={OFFICIAL_UNIT: UnitInput(VARIANT, "whole")},
    optimizer=OptimizerSpec("adam", 1e-3, 5e-3),
    schedule=ScheduleSpec("constant", "step"),
    iterations=15000,
    iteration_unit="step",
    sampler="window_mean",
    sampler_params={"batch_normal": 32, "batch_abnormal": 32, "reset": "per_pool",
                    "order": "normal_first", "eager": False},
    loss_weights={"alpha": 1e-4, "smooth": 8e-4, "sparsity": 8e-3},
    scale_constants={"margin": 100.0},
    precision="fp32",
    input_table=(
        InputRow("features", "10-crop I3D (ResNet-50) snippet features, 2048-d",
                 "frozen VLM dense snippets, single crop, 1792 (XCB+XCL) / 1920 (XCB+SigLIP)",
                 "adapt", "README.md; option.py L5"),
        InputRow("clip_normalisation", "none (raw snippet features)", "none, as officially",
                 "keep", "dataset.py L58-L59"),
        InputRow("sampler_and_length",
                 "32 windows per video by process_feat (np.linspace bounds, window mean, empty "
                 "window: its first snippet); one shuffled drop_last loader per class, 32 + 32 "
                 "videos per step, each re-created when exhausted, normal first",
                 "d16m32 (the same windows, precomputed) and the window_mean sampler "
                 "(per_pool, normal_first)", "keep",
                 "utils.py L32-L41; dataset.py L67-L72; main.py L20-L25, L57-L63; train.py "
                 "L86-L89"),
        InputRow("scale_match", "none",
                 "the d16m32 scale map refitted on the training part (window-mean training "
                 "input), the same map as the validation / test input layer", "adapt",
                 "PLAN.md 검증·시험 입력"),
        InputRow("input_normalisation", "none",
                 "input LayerNorm for XCB+SigLIP only (the C rule, diagnostic P7)", "adapt",
                 "PLAN.md 입력 처리 표"),
        InputRow("scale_constants", "RTFM_loss margin 100 (I3D feature norms)",
                 "official 100; D4 validation check x{1/10, 1, 10}", "adapt",
                 "train.py L71, L102"),
        InputRow("text_branch", "none", "none", "keep", "model.py"),
        InputRow("top_k", "k = num_segments // 10 = 3 snippets of largest feature magnitude "
                 "(dropout 0.7 on the selection mask), abnormal and normal",
                 "the same, T = 32", "keep", "model.py L177-L179, L231-L268"),
        InputRow("checkpoint_selection", "test-set AUC every 5 steps after step 200",
                 "validation V-AUC (sigma W readout) at 30 evaluation points", "adapt",
                 "main.py L65-L74"),
        InputRow("precision", "fp32", "fp32", "keep", "main.py; model.py L4"),
    ),
    sources={
        "optimizer": _src("main.py", "L40-L41", "optim.Adam(model.parameters(), lr=config.lr[0],"
                          " weight_decay=0.005)"),
        "lr": _src("option.py", "L11", "--lr default '[0.001]*15000' (config.py L6 eval)"),
        "weight_decay": _src("main.py", "L41"),
        "schedule": _src("main.py", "L53-L55", "per-step lr list, constant 0.001 (option.py "
                         "L11)"),
        "iterations": _src("option.py", "L19", "--max-epoch 15000; one optimizer step per "
                           "'epoch' (main.py L48-L63)"),
        "batch": _src("option.py", "L12", "--batch-size 32 per class loader (main.py L20-L25); "
                      "normal batch first (train.py L86-L89)"),
        "sampler": _src("main.py", "L20-L25, L57-L61", "shuffle=True, drop_last=True, "
                        "num_workers=0; each loader re-created at (step - 1) % len == 0"),
        "loss_weights": _src("train.py", "L77, L102-L105", "RTFM_loss(alpha=0.0001, margin=100),"
                             " sparsity 8e-3, smooth 8e-4; the BCE term has weight 1 (no "
                             "constant)"),
        "unit": _src("dataset.py", "L70", "process_feat(feature, 32); model.py L177 "
                     "num_segments = 32"),
        "test_sequence": _src("test_10crop.py", "L11-L19", "each test video whole (loader "
                              "batch 1), snippet scores"),
        "scale_constants": _src("train.py", "L71, L102", "margin 100"),
    },
    notes="RTFM official defaults (option.py) for both datasets: the repository has no "
          "XD-Violence configuration. Official unit 32, so there is no X.32 arm. The margin is "
          "the only scale constant (D4 x{1/10, 1, 10}); alpha, smooth and sparsity weights are "
          "loss weights. See the module docstring for the adaptations. Paper vs code (the code "
          "at cf938f4c is followed): the paper (arXiv 2101.10030 Sec. 4.2) states Adam weight "
          "decay 0.0005, lr 0.001 (ShanghaiTech, UCF-Crime) and 0.0001 (XD-Violence), batch 64 "
          "for 50 epochs; the code uses weight decay 0.005, lr 0.001 for every dataset and "
          "15000 steps; the paper's XD lr is x1/10 of the code value (closest D4 value: the "
          "widened x1/9, base grid x1/3). The code's magnitude loss (|m - ||f_a||| + "
          "||f_n||)^2 differs from the paper's hinge (Eq. 6), and the code drops out the top-k "
          "selection mask. Loss parts mag_abnormal / mag_normal: batch means of the top-k "
          "feature magnitudes (diagnostics, not in the cost).",
)


# --- the network ------------------------------------------------------------------------------

def _xavier(module: nn.Module) -> None:
    """Xavier-uniform weight and zero bias for every convolution and linear layer (the official
    initialisation; other modules keep their defaults)."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0)


class NonLocal1d(nn.Module):
    """Embedded-dot-product non-local block over time (no subsampling): affinity of the
    theta / phi projections divided by the length, applied to the g projection, mapped back by a
    1x1 convolution and a BatchNorm whose affine parameters start at zero, plus the input."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_channels = int(channels)
        self.inter_channels = max(1, self.in_channels // 2)
        c, h = self.in_channels, self.inter_channels
        self.g = nn.Conv1d(c, h, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(nn.Conv1d(h, c, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm1d(c))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        self.theta = nn.Conv1d(c, h, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(c, h, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h = x.size(0), self.inter_channels
        value = self.g(x).view(b, h, -1).permute(0, 2, 1)
        query = self.theta(x).view(b, h, -1).permute(0, 2, 1)
        key = self.phi(x).view(b, h, -1)
        affinity = torch.matmul(query, key)
        affinity = affinity / affinity.size(-1)
        mixed = torch.matmul(affinity, value).permute(0, 2, 1).contiguous()
        return self.W(mixed.view(b, h, *x.size()[2:])) + x


def _dilated_branch(dim: int, dilation: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(dim, WIDTH, kernel_size=3, stride=1, dilation=dilation,
                                   padding=dilation),
                         nn.ReLU(), nn.BatchNorm1d(WIDTH))


class Aggregate(nn.Module):
    """The multi-scale temporal network: (B, T, D) -> (B, T, D)."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        d = int(feature_dim)
        self.len_feature = d
        self.conv_1 = _dilated_branch(d, DILATIONS[0])
        self.conv_2 = _dilated_branch(d, DILATIONS[1])
        self.conv_3 = _dilated_branch(d, DILATIONS[2])
        self.conv_4 = nn.Sequential(nn.Conv1d(d, WIDTH, kernel_size=1, stride=1, padding=0,
                                              bias=False), nn.ReLU())
        self.conv_5 = nn.Sequential(nn.Conv1d(4 * WIDTH, d, kernel_size=3, stride=1, padding=1,
                                              bias=False), nn.ReLU(), nn.BatchNorm1d(d))
        self.non_local = NonLocal1d(WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.permute(0, 2, 1)
        local = torch.cat((self.conv_1(channels_first), self.conv_2(channels_first),
                           self.conv_3(channels_first)), dim=1)
        context = self.non_local(self.conv_4(channels_first))
        fused = self.conv_5(torch.cat((local, context), dim=1))
        return (fused + channels_first).permute(0, 2, 1)


class RTFMOutput(NamedTuple):
    score_abnormal: torch.Tensor    # (n_abn, 1) mean sigmoid score of the selected snippets
    score_normal: torch.Tensor      # (n_nor, 1)
    feat_abnormal: torch.Tensor     # (ncrops * n_abn, k, D) selected features, crop-major
    feat_normal: torch.Tensor       # (ncrops * n_nor, k, D)
    scores: torch.Tensor            # (bs, T, 1) sigmoid snippet scores (crop mean)
    magnitudes: torch.Tensor        # (bs, T) feature magnitudes (crop mean)


def _pick(magnitudes: torch.Tensor, keep: torch.Tensor, feats: torch.Tensor,
          scores: torch.Tensor, k: int, ncrops: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The k snippets of largest masked magnitude per video: their features in every crop
    (crop-major) and the mean of their scores."""
    order = torch.topk(magnitudes * keep, k, dim=1)[1]
    n, t, d = magnitudes.shape[0], feats.shape[1], feats.shape[2]
    rows = order.unsqueeze(2).expand([-1, -1, d])
    crops = feats.view(n, ncrops, t, d).permute(1, 0, 2, 3)
    picked = torch.cat([torch.gather(crop, 1, rows) for crop in crops])
    score_rows = order.unsqueeze(2).expand([-1, -1, scores.shape[2]])
    return picked, torch.mean(torch.gather(scores, 1, score_rows), dim=1)


class RTFMNet(nn.Module):
    """The RTFM network with the optional input LayerNorm (module docstring)."""

    def __init__(self, feature_dim: int, input_ln: bool = False) -> None:
        super().__init__()
        self.input_ln = input_layer(feature_dim, input_ln)
        self.num_segments = OFFICIAL_UNIT
        self.k = TOP_K
        self.Aggregate = Aggregate(feature_dim)
        self.fc1 = nn.Linear(feature_dim, HIDDEN[0])
        self.fc2 = nn.Linear(HIDDEN[0], HIDDEN[1])
        self.fc3 = nn.Linear(HIDDEN[1], 1)
        self.drop_out = nn.Dropout(DROPOUT)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(_xavier)

    def _snippets(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, T, D) -> dropped-out temporal features (N, T, D) and fc3 logits (N, T, 1)."""
        feats = self.drop_out(self.Aggregate(x))
        h = self.drop_out(self.relu(self.fc1(feats)))
        h = self.drop_out(self.relu(self.fc2(h)))
        return feats, self.fc3(h)

    def snippet_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """(bs, 1, T, D) -> (bs, T) fc3 logits (the score path only; single crop)."""
        x = self.input_ln(inputs)
        bs, ncrops, t, d = x.size()
        if ncrops != 1:
            raise MethodError(f"rtfm scoring is single-crop, got {ncrops} crops")
        return self._snippets(x.view(-1, t, d))[1].view(bs, t)

    def forward(self, inputs: torch.Tensor, n_normal: int) -> RTFMOutput:
        """inputs (bs, ncrops, T, D), the first ``n_normal`` videos normal."""
        x = self.input_ln(inputs)
        bs, ncrops, t, d = x.size()
        features, logits = self._snippets(x.view(-1, t, d))
        scores = self.sigmoid(logits).view(bs, ncrops, -1).mean(1).unsqueeze(dim=2)
        nb = int(n_normal)
        nor_feats, nor_scores = features[0:nb * ncrops], scores[0:nb]
        abn_feats, abn_scores = features[nb * ncrops:], scores[nb:]
        magnitudes = torch.norm(features, p=2, dim=2).view(bs, ncrops, -1).mean(1)
        nor_mag, abn_mag = magnitudes[0:nb], magnitudes[nb:]
        if nor_mag.shape[0] == 1:          # the official one-normal-video case (its inference)
            abn_mag, abn_scores, abn_feats = nor_mag, nor_scores, nor_feats
        keep = self.drop_out(torch.ones_like(nor_mag))
        feat_abn, score_abn = _pick(abn_mag, keep, abn_feats, abn_scores, self.k, ncrops)
        keep = self.drop_out(torch.ones_like(nor_mag))
        feat_nor, score_nor = _pick(nor_mag, keep, nor_feats, nor_scores, self.k, ncrops)
        return RTFMOutput(score_abn, score_nor, feat_abn, feat_nor, scores, magnitudes)


# --- the loss ---------------------------------------------------------------------------------

class RTFMLoss(NamedTuple):
    total: torch.Tensor
    rtfm: torch.Tensor          # BCE + alpha * magnitude term
    smooth: torch.Tensor
    sparsity: torch.Tensor
    # q7 RTFM-5 diagnostics (detached, never part of the cost): batch means of the top-k feature
    # magnitudes ||mean_k f|| of the abnormal / normal videos, to read against the margin
    mag_abnormal: torch.Tensor | None = None
    mag_normal: torch.Tensor | None = None


def rtfm_cost(model: RTFMNet, inputs: torch.Tensor, labels: torch.Tensor, n_normal: int, *,
              margin: float, weights: Mapping[str, float] | None = None) -> RTFMLoss:
    """The official training objective of one batch: inputs (2n, ncrops, T, D), labels (2n,) 0
    for the first n videos, 1 for the rest (module docstring "loss"; the official order of
    operations, so the autograd graph is the official one)."""
    w = dict(RECIPE.loss_weights if weights is None else weights)
    out = model(inputs, n_normal)
    t = inputs.shape[2]
    flat = out.scores.view(inputs.shape[0] * t, -1).squeeze()
    abn = flat[int(n_normal) * t:]
    sparse = w["sparsity"] * torch.mean(torch.norm(abn, dim=0))
    following = torch.zeros_like(abn)
    following[:-1] = abn[1:]
    following[-1] = abn[-1]
    smooth = w["smooth"] * torch.sum((following - abn) ** 2)
    score = torch.cat((out.score_normal, out.score_abnormal), 0).squeeze()
    cls = F.binary_cross_entropy(score, labels.to(score.device))
    mag_abn = torch.norm(torch.mean(out.feat_abnormal, dim=1), p=2, dim=1)
    far = torch.abs(margin - mag_abn)
    near = torch.norm(torch.mean(out.feat_normal, dim=1), p=2, dim=1)
    rtfm = cls + w["alpha"] * torch.mean((far + near) ** 2)
    return RTFMLoss(rtfm + smooth + sparse, rtfm, smooth, sparse,
                    mag_abn.detach().mean(), near.detach().mean())


# --- the method -------------------------------------------------------------------------------

class RTFM(BaselineMethod):
    """The porting-contract method object (common module docstring)."""

    name = NAME
    official_unit = OFFICIAL_UNIT

    def build(self, feature_dim: int, *, unit: int, input_ln: bool,
              scale: Mapping[str, float]) -> nn.Module:
        self.recipe.unit_input(unit)
        return RTFMNet(int(feature_dim), bool(input_ln))

    def loss(self, model: nn.Module, batch: Batch, *, unit: int,
             scale: Mapping[str, float]) -> LossOutput:
        b, t = int(batch.features.shape[0]), int(batch.features.shape[1])
        n = int(batch.n_normal)
        if t != int(unit) or bool((batch.lengths != t).any()):
            raise MethodError(f"rtfm trains on full {unit}-window inputs, got {t} rows / "
                              f"lengths {batch.lengths.tolist()}")
        expected = torch.cat([torch.zeros(n), torch.ones(b - n)]).to(batch.labels)
        if 2 * n != b or not torch.equal(batch.labels, expected):
            raise MethodError("rtfm batches are n normal videos followed by n abnormal ones")
        if n < 2:
            # q7 RTFM-3: with one normal video the official forward swaps in its inference path
            # (model.py L226-L229: abnormal := normal), a degenerate training batch
            raise MethodError(f"rtfm trains on at least 2 normal + 2 abnormal videos, got "
                              f"n_normal={n} (the official one-video case is its inference)")
        out = rtfm_cost(model, batch.features[:, None], batch.labels, n,
                        margin=float(scale["margin"]), weights=self.recipe.loss_weights)
        return LossOutput(out.total, {"rtfm": out.rtfm.detach(), "smooth": out.smooth.detach(),
                                      "sparsity": out.sparsity.detach(),
                                      "mag_abnormal": float(out.mag_abnormal),
                                      "mag_normal": float(out.mag_normal)})

    def sequence_logits(self, model: nn.Module, x: torch.Tensor, *, unit: int) -> torch.Tensor:
        self.recipe.unit_input(unit)
        return model.snippet_logits(x[None, None]).reshape(-1)


METHOD = RTFM()
