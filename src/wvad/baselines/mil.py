"""E023 comparison method MIL: Sultani, Chen and Shah, "Real-World Anomaly Detection in
Surveillance Videos", CVPR 2018 (PLAN.md "설계" 팔 MIL, "입력 처리 표"). New code (add-only rule).

Official code: https://github.com/WaqasSultani/AnomalyDetectionCVPR2018 at commit
8aa957cdaa7eac821e07115fe63920c3b07da59d (results/e023_licences.json key "mil"); licence: none
detected (no LICENSE file in the repository). Porting mode: re-implementation in PyTorch; the
official Keras 1.1.0 / Theano 1.0.2 code (Python 2) is used only outside the repository, for the
equivalence checks of ``wvad.baselines.check_mil`` (numpy transcription, PLAN.md (q1) / (q1b)).

What is reproduced (official file TrainingAnomalyDetector_public.py, "T" below; Keras 1.1.0
semantics read from its source: initializations.py L35-L53, backend/theano_backend.py
L144-L146 and L995-L1016, layers/core.py L87-L91 and L688-L730, regularizers.py L80-L95,
optimizers.py L251-L278, engine/training.py L317-L344 and L613-L633)
    network     T L29-L35: Dense(512, relu) - Dropout(0.6) - Dense(32, linear) - Dropout(0.6) -
                Dense(1, sigmoid); every Dense weight glorot_normal = np.random.normal(0,
                sqrt(2 / (fan_in + fan_out)), (fan_in, fan_out)) from numpy's global generator
                cast to float32, biases zero; the two Dropout layers each draw their Theano seed
                np.random.randint(1, 10e6) when added (after the preceding Dense weight), and
                ``build`` draws those numbers too, so numpy's stream is consumed in the official
                construction order (W1, seed, W2, seed, W3); no torch draw (skip_init). ReLU is
                Theano 1.0.2's T.nnet.relu = 0.5 (x + |x|) (equal to max(x, 0) in value; its
                gradient at an exact zero is 0.5).
    objective   T L198-L273 custom_objective on the (videos, 32) sigmoid scores, the batch being
                n_exp anomalous then n_exp normal videos: mean over the normal videos of the sum
                over the anomalous videos of max(1 - max_a + max_n, 0) (Theano's maximum: the
                gradient flows at a tie, as clamp_min), + 0.00008 x the score sums and
                + 0.00008 x the squared adjacent differences of the FIRST n_exp videos
                (positional, = the anomalous ones), the per-video max with Theano's gradient
                (every tied element gets the full gradient); + the Keras W_regularizer l2(0.001)
                of each Dense weight, sum(l2 x W^2) with l2 = float32(0.001) (K.cast_to_floatx),
                added to the loss in layer order (it is part of the official batch loss).
    optimizer   T L276-L278: Adagrad(lr=0.01, epsilon=1e-08), Keras 1.1.0 rule
                p - lr g / (sqrt(a + g^2) + eps) = torch.optim.Adagrad (lr_decay 0, initial
                accumulator 0, weight_decay 0: the l2 term is in the loss).
    schedule    T L296-L306: 20,000 iterations, one batch each, constant lr.
    batch       T L89-L103: 60 videos, n_exp = 30 anomalous then 30 normal, drawn per iteration
                as np.random.permutation(N_abnormal)[N_abnormal - 30:] then
                np.random.permutation(N_normal)[N_normal - 30:] (sampler "mil_permutation").
    unit        T L204 n_seg = 32 segments per video (Save_C3DFeatures_32Segments.m L64-L99).
    test        Test_Anomaly_Detector_public.py L117-L121: model.predict_on_batch on one video's
                rows (dropout off); the network is applied row by row, so the test sequence rule
                is "whole".

Adaptations: (1) input dimension 4096 (C3D fc6) -> D = 1792 (XCB+XCL) / 1920 (XCB+SigLIP) frozen
VLM snippet features (the first Dense fan-in and its glorot_normal scale follow D);
(2) training input = the E023 derived cache d16m32l2: every snippet l2-normalised per encoder
(paper Sec. 5.1 clip normalisation, PLAN.md "입력 처리 표"), then the numpy linspace (floor)
32-window mean of wvad.extract.derive_n kind m, then the scale match fitted on the training part;
the official MATLAB script instead uses round(linspace(1, n, 33)) windows that leave out the last
clip and l2-normalises each segment MEAN (Save_C3DFeatures_32Segments.m L65-L88, no per-clip
l2); validation / test snippets get the same l2 and scale map (InputPipeline) and are scored one
by one instead of the official 32 test segments; (3) input LayerNorm for XCB+SigLIP
(common.input_layer, PLAN.md P7), trained by the official optimizer, without the weight
regulariser; (4) the hard-coded pool sizes Num_abnormal = 810 / Num_Normal = 800 -> the training
part's anomalous / normal video counts, indexed in training-part order (official: sorted feature
file names); (5) Keras / Theano -> PyTorch: the dropout masks come from torch's generator
(official: Theano MRG streams seeded by the np.random.randint draws, which are still consumed);
Theano 1.0.2's float32 C sigmoid (exactly 1 above 15 and 0 below -88) is not emulated
(torch.sigmoid differs from it there by < 3.1e-7); the Keras objective wrapper with unit sample
weights (the mean of 1,920 copies of the loss) is the identity and omitted; (6) the loop seeds
random / numpy / torch (official: unseeded); (7) checkpoint = validation V-AUC selection over 30
evaluation points (official: weights saved every 1,000 iterations and at the end); (8) the
network returns logits (the official sigmoid is applied by the loss and the readouts).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from wvad.baselines import samplers
from wvad.baselines.common import BaselineMethod, Batch, LossOutput, input_layer
from wvad.baselines.recipes import (
    InputRow,
    OptimizerSpec,
    Recipe,
    ScheduleSpec,
    Source,
    UnitInput,
)

OFFICIAL_FILE = "TrainingAnomalyDetector_public.py"
TEST_FILE = "Test_Anomaly_Detector_public.py"
SEGMENT_FILE = "Save_C3DFeatures_32Segments.m"
HIDDEN = (512, 32)                       # T L31, L33
DROPOUT = 0.6                            # T L32, L34 (fraction dropped)
SAMPLER = "mil_permutation"


def _src(file: str, lines: str, note: str = "") -> Source:
    return Source("mil", file, lines, note)


RECIPE = Recipe(
    method="mil",
    official_unit=32,
    units={32: UnitInput("d16m32l2", "whole")},
    optimizer=OptimizerSpec("adagrad", lr=0.01, weight_decay=0.0, eps=1e-8),
    schedule=ScheduleSpec("constant", "step"),
    iterations=20000,
    iteration_unit="step",
    sampler=SAMPLER,
    sampler_params={"batch_abnormal": 30, "batch_normal": 30},
    loss_weights={"ranking": 1.0, "sparsity": 0.00008, "smoothness": 0.00008,
                  "weight_l2": 0.001},
    scale_constants={},
    precision="fp32",
    input_table=(
        InputRow("features", "C3D fc6, 4096-d, one row per 16-frame clip",
                 "frozen VLM dense snippets (16 frames), single crop, 1792 (XCB+XCL) / 1920 "
                 "(XCB+SigLIP)", "adapt", f"{SEGMENT_FILE} L29-L36; {OFFICIAL_FILE} L31"),
        InputRow("clip_normalisation",
                 "paper Sec. 5.1: l2 per 16-frame clip, then the segment mean; the official "
                 "MATLAB script instead l2-normalises each segment mean (no per-clip l2)",
                 "the paper's rule as implemented: l2 per snippet and encoder before the window "
                 "mean (derive_n d16m32l2), the same l2 at validation and test (InputPipeline); "
                 "this follows the paper, not the official code, which l2-normalises the "
                 "segment means (disclosed; PLAN.md 기록 2026-10-19)", "adapt",
                 f"{SEGMENT_FILE} L84-L88 (code); paper Sec. 5.1"),
        InputRow("sampler_and_length",
                 "32 segments per video (MATLAB round(linspace(1, n, 33)), last clip left "
                 "out); 30 anomalous + 30 normal videos per iteration by "
                 "np.random.permutation(N)[N - 30:]",
                 "32 windows of derive_n kind m (np.linspace(0, n_v, 33, dtype=int64), empty "
                 "window: snippet r[i]); sampler mil_permutation = the official draw over the "
                 "training-part pools", "adapt",
                 f"{SEGMENT_FILE} L64-L99; {OFFICIAL_FILE} L92-L103"),
        InputRow("scale_match", "none",
                 "scale match of the window-mean input, refitted on the training part "
                 "(derive_n), the same map at validation and test", "adapt", "PLAN.md"),
        InputRow("input_normalisation", "none",
                 "input LayerNorm for XCB+SigLIP only (common.input_ln, diagnostic P7)",
                 "adapt", "PLAN.md"),
        InputRow("scale_constants",
                 "none (the hinge margin 1 acts on sigmoid scores, not on feature norms)",
                 "none", "keep", f"{OFFICIAL_FILE} L267"),
        InputRow("text_branch", "none", "none", "keep", ""),
        InputRow("top_k", "max over the 32 segments of a video (k = 1)", "the same", "keep",
                 f"{OFFICIAL_FILE} L222, L262-L267"),
        InputRow("checkpoint_selection",
                 "none (weights saved every 1,000 iterations and at the end)",
                 "validation V-AUC, sigma W readout, 30 evaluation points", "adapt",
                 f"{OFFICIAL_FILE} L313-L318"),
        InputRow("precision", "float32 (Theano floatX float32 on the GPU)",
                 "fp32 (float64 only in the (q1b) check)", "keep",
                 f"{OFFICIAL_FILE} L26; model.json input_dtype"),
    ),
    sources={
        "optimizer": _src(OFFICIAL_FILE, "L276-L278", "Adagrad(lr=0.01, epsilon=1e-08); "
                          "Keras 1.1.0 optimizers.py L251-L278 (decay 0)"),
        "lr": _src(OFFICIAL_FILE, "L276"),
        "weight_decay": _src(OFFICIAL_FILE, "L31-L35", "W_regularizer=l2(0.001) on every "
                             "Dense weight: a loss term (loss_weights.weight_l2), not "
                             "optimizer decay"),
        "schedule": _src(OFFICIAL_FILE, "L276, L301-L306", "constant lr, one batch per "
                         "iteration"),
        "iterations": _src(OFFICIAL_FILE, "L296", "num_iters = 20000"),
        "batch": _src(OFFICIAL_FILE, "L92-L93, L182-L193", "batchsize=60, n_exp=30; "
                      "anomalous videos first"),
        "sampler": _src(OFFICIAL_FILE, "L100-L103", "np.random.permutation per iteration, "
                        "anomalous draw first"),
        "loss_weights": _src(OFFICIAL_FILE, "L267-L271, L31-L35", "ranking mean, 0.00008 "
                             "sparsity, 0.00008 smoothness; l2(0.001)"),
        "unit": _src(OFFICIAL_FILE, "L204", f"n_seg = 32; {SEGMENT_FILE} L64-L99"),
        "test_sequence": _src(TEST_FILE, "L117-L121", "predict_on_batch on one video's rows "
                              "(row-wise network)"),
    },
    notes="Re-implementation (no licence). Adaptations: see the wvad.baselines.mil docstring. "
          "No scale constant (D4 checks the lr only).",
)

if RECIPE.loss_weights["ranking"] != 1.0:          # T L271: T.mean(z) enters unweighted
    raise RuntimeError("the official ranking term has weight 1")
SPARSITY = float(RECIPE.loss_weights["sparsity"])
SMOOTHNESS = float(RECIPE.loss_weights["smoothness"])
WEIGHT_L2 = float(np.float32(RECIPE.loss_weights["weight_l2"]))     # K.cast_to_floatx(0.001)


# --- Theano gradient rules ----------------------------------------------------------------

class TheanoMax(torch.autograd.Function):
    """Row maximum of a (videos, segments) tensor with Theano 1.0.2's MaxAndArgmax gradient
    (tensor/basic.py L1480: g_x = eq(xmax, x) * g_max, every tied element gets the gradient)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        m = x.amax(dim=1)
        ctx.save_for_backward(x, m)
        return m

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> torch.Tensor:
        x, m = ctx.saved_tensors
        return g[:, None] * (x == m[:, None]).to(g.dtype)


def theano_relu(x: torch.Tensor) -> torch.Tensor:
    """Theano 1.0.2 T.nnet.relu(x, 0) = 0.5 * (x + abs(x)) (Keras 1.1.0 K.relu)."""
    return 0.5 * (x + x.abs())


def keras_glorot_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """Keras 1.1.0 glorot_normal: np.random.normal(0, sqrt(2 / (fan_in + fan_out)),
    (fan_in, fan_out)) from numpy's global generator, as float32 (K.variable)."""
    s = np.sqrt(2. / (fan_in + fan_out))
    return np.asarray(np.random.normal(loc=0.0, scale=s, size=(fan_in, fan_out)),
                      dtype=np.float32)


def keras_dropout_seed() -> int:
    """The numpy draw of Keras 1.1.0 K.dropout (theano_backend.py L1009-L1010) when a Dropout
    layer is added; the seed itself is unused here (torch draws the masks)."""
    return int(np.random.randint(1, 10e6))


# --- the network --------------------------------------------------------------------------

class SultaniNet(nn.Module):
    """input LN (XCB+SigLIP) -> Dense 512 (relu) -> Dropout 0.6 -> Dense 32 -> Dropout 0.6 ->
    Dense 1; forward returns the logits (..., ) of rows (..., D)."""

    def __init__(self, feature_dim: int, input_ln: bool) -> None:
        super().__init__()
        d = int(feature_dim)
        self.input_ln = input_layer(d, input_ln)
        dims = (d, *HIDDEN, 1)
        layers = []
        for i in range(3):              # official order: W1, dropout seed, W2, dropout seed, W3
            fc = nn.utils.skip_init(nn.Linear, dims[i], dims[i + 1])
            w = keras_glorot_normal(dims[i], dims[i + 1])
            with torch.no_grad():
                fc.weight.copy_(torch.from_numpy(np.ascontiguousarray(w.T)))
                fc.bias.zero_()
            layers.append(fc)
            if i < 2:
                keras_dropout_seed()
        self.fc1, self.fc2, self.fc3 = layers
        self.drop1 = nn.Dropout(DROPOUT)
        self.drop2 = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(theano_relu(self.fc1(self.input_ln(x))))
        h = self.drop2(self.fc2(h))
        return self.fc3(h).squeeze(-1)

    def dense_layers(self) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        return self.fc1, self.fc2, self.fc3


def sultani_objective(scores: torch.Tensor, labels: torch.Tensor
                      ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """custom_objective (T L198-L273) of (videos, segments) sigmoid scores whose first half are
    the anomalous videos (labels 1) and second half the normal ones (labels 0)."""
    nvid = int(scores.shape[0])
    n_exp = nvid // 2                                           # T L206 (Python 2 int division)
    vmax = TheanoMax.apply(scores)                              # T L222 sub_max
    normal = torch.nonzero(labels == 0).reshape(-1)             # T L257 F_labels == 32
    abnormal = torch.nonzero(labels == 1).reshape(-1)           # T L258 F_labels == 0
    sub_nor, sub_abn = vmax[normal], vmax[abnormal]             # T L262-L263
    hinge = torch.clamp_min(1 - sub_abn[None, :] + sub_nor[:n_exp, None], 0)   # T L266-L267
    ranking = hinge.sum(dim=1).mean()                           # T L268, L271 T.mean
    sparsity = scores[:n_exp].sum(dim=1).sum()                  # T L223, L251-L252, L271
    smooth = (scores[:n_exp, :-1] - scores[:n_exp, 1:]).square().sum(dim=1).sum()  # T L225-L233
    total = ranking + SPARSITY * sparsity + SMOOTHNESS * smooth          # T L271 (ranking x 1)
    return total, {"ranking": ranking, "sparsity": sparsity, "smoothness": smooth}


def weight_l2(model: SultaniNet) -> list[torch.Tensor]:
    """The Keras W_regularizer terms sum(l2 * W^2) of the three Dense weights, layer order."""
    return [(WEIGHT_L2 * fc.weight.square()).sum() for fc in model.dense_layers()]


# --- the sampler --------------------------------------------------------------------------

class SultaniSampler(samplers.Sampler):
    """load_dataset_Train_batch's draws (T L100-L103): per step np.random.permutation of the
    anomalous pool, its last n_exp entries, then the same for the normal pool; the batch lists
    the anomalous videos first (T L119-L177). Parameters batch_abnormal = batch_normal = n_exp
    (official 30)."""

    name = SAMPLER

    def __init__(self, pools: samplers.Pools, params: Mapping | None = None) -> None:
        super().__init__(pools, params)
        p = self.params
        unknown = set(p) - {"batch_abnormal", "batch_normal"}
        if unknown:
            raise samplers.SamplerError(f"{SAMPLER}: unknown parameters {sorted(unknown)}")
        try:
            na, nn_ = int(p["batch_abnormal"]), int(p["batch_normal"])
        except KeyError as exc:
            raise samplers.SamplerError(f"{SAMPLER} needs batch_abnormal and "
                                        "batch_normal") from exc
        if na != nn_ or na < 1:
            raise samplers.SamplerError(f"{SAMPLER}: the official batch has n_exp videos of "
                                        f"each class (got {na} / {nn_})")
        if len(pools.abnormal) < na or len(pools.normal) < na:
            raise samplers.SamplerError(f"{SAMPLER}: pools of {len(pools.abnormal)} anomalous "
                                        f"/ {len(pools.normal)} normal videos for n_exp {na}")
        self.n_exp = na

    def draw(self, step: int) -> samplers.StepDraw:
        self._advance(step)
        num_abnormal, num_normal, n_exp = len(self.pools.abnormal), len(self.pools.normal), \
            self.n_exp
        abn = np.random.permutation(num_abnormal)[num_abnormal - n_exp:]
        nor = np.random.permutation(num_normal)[num_normal - n_exp:]
        videos = tuple(self.pools.abnormal[int(i)] for i in abn) + \
            tuple(self.pools.normal[int(i)] for i in nor)
        return samplers.StepDraw(videos, (1,) * n_exp + (0,) * n_exp)


samplers.register(SAMPLER, SultaniSampler, owner=__name__)


# --- the method ---------------------------------------------------------------------------

class SultaniMIL(BaselineMethod):
    """The MIL method (module docstring)."""

    name = "mil"
    official_unit = 32

    def build(self, feature_dim: int, *, unit: int, input_ln: bool,
              scale: Mapping[str, float]) -> nn.Module:
        RECIPE.unit_input(unit)
        if scale:
            raise ValueError(f"mil has no scale constant, got {dict(scale)}")
        return SultaniNet(feature_dim, input_ln)

    def loss(self, model: nn.Module, batch: Batch, *, unit: int,
             scale: Mapping[str, float]) -> LossOutput:
        x = batch.features
        nvid, n_seg = int(x.shape[0]), int(x.shape[1])
        if n_seg != int(unit) or bool((batch.lengths != n_seg).any()):
            raise ValueError(f"mil trains on full {unit}-segment bags (official n_seg); got "
                             f"T = {n_seg}, lengths {batch.lengths.tolist()}")
        n_exp = nvid // 2
        lab = batch.labels
        if nvid % 2 or batch.n_normal != n_exp or not bool((lab[:n_exp] == 1).all()):
            raise ValueError("the official batch is n_exp anomalous videos followed by n_exp "
                             f"normal videos; got labels {lab.tolist()}")
        scores = torch.sigmoid(model(x))                        # Dense(1, activation='sigmoid')
        objective, parts = sultani_objective(scores, lab)
        regs = weight_l2(model)
        total = objective
        for r in regs:                                          # Keras: total_loss = r(total)
            total = total + r
        reg = torch.stack([r.detach() for r in regs]).sum()
        return LossOutput(total, {**{k: v.detach() for k, v in parts.items()},
                                  "objective": objective.detach(), "weight_l2": reg})

    def sequence_logits(self, model: nn.Module, x: torch.Tensor, *, unit: int) -> torch.Tensor:
        return model(x).reshape(-1)


METHOD = SultaniMIL()
