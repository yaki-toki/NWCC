"""E023 MIL equivalence harness: gates (q1), (q1c), (q1e) and the (q1b) runner (PLAN.md "관문",
"(q1b) 통과 기준" MIL rule, "실패 처리", "구현"). New code (add-only rule); never imported by the
training or scoring code.

The official MIL code (https://github.com/WaqasSultani/AnomalyDetectionCVPR2018 at the commit of
results/e023_licences.json; Keras 1.1.0 / Theano 1.0.2, Python 2 syntax) is not revived (PLAN.md
(q1b) MIL rule). The reference is a numpy transcription of it (float64 unless stated; "T" =
TrainingAnomalyDetector_public.py, file and lines at each function):

    keras_init(D, ln)               the Keras 1.1.0 construction of T L29-L35 (glorot_normal draws
                                    and the two Dropout seed draws, in order; float32 values)
    ref_forward / ref_backward      Dense(relu) - Dropout - Dense - Dropout - Dense(sigmoid) with
                                    injected 0/1 masks (Keras: x *= mask; x /= retain_prob) and
                                    Theano's gradient rules (relu = 0.5 (x + |x|))
    custom_objective_ref            custom_objective (T L198-L273) line by line
    custom_objective_grad_ref       its gradient under Theano's rules (max: every tied element;
                                    maximum: the first argument at a tie)
    ref_loss_and_grads              + the Keras W_regularizer terms (regularizers.py L90-L95)
    adagrad_ref                     the Keras 1.1.0 Adagrad update (optimizers.py L271-L274)
    official_draw                   T L100-L103 (the pool sizes replace 810 / 800)
    official_predict                Test_Anomaly_Detector_public.py L121 predict_on_batch (dropout
                                    off; the float32 path uses Theano's float32 C sigmoid with its
                                    15 / -88 clamps, the float64 path 19 / -709)

This module never imports wvad.baselines.official (the import-graph test keeps every src module
but official.py away from it): like check_rtfm and check_vadclip_v, the entry points that touch
the clone take the ``official`` loader module as an argument (``run_q1b(..., official=...)``,
``main(argv, official=official)``). It never executes official code: through the loader it
locates the clone (WVAD_OFFICIAL_ROOT), reads git's HEAD and the git blob ids of the transcribed
files (line endings normalised to LF), for the (q1b) record. tests/test_e023_mil*.py
additionally execute the official custom_objective text and loader lines from the clone under
numpy / torch Theano shims and compare them with this transcription.

Precision of the judgement (PLAN.md 기록 2026-10-19, review M3): (q1) and (q1e) of MIL are judged
in float64 (port in float64 against the float64 reference, as the plan's MIL (q1b) rule does);
the fp32 port's errors against the same reference are computed and reported beside the verdict
("fp32_information"; about 3e-6 relative on the gradients, 2.6e-6 at |logit| ~ 6 and 7.6e-5 at
|logit| ~ 156 in (q1e): BLAS summation order of an independent fp32 product, not a path
difference).

Gates
    check_q1(D, input_ln=...)       constructor (bitwise, torch generator untouched), forward,
                                    loss, gradients, two Adagrad steps; eval mode and train mode
                                    with injected masks; float64 (PLAN.md: allowed for MIL), and
                                    the fp32 port against the float64 reference as information
    check_q1c(pools, steps, seed)   the port sampler against official_draw from one numpy state
    check_q1e(D, lengths, ...)      the port's sequence_logits on the common.chunk_bounds pieces
                                    (float64 model) against official_predict (float64) on the
                                    plan's pieces: the gate; common.score_video (fp32, as run)
                                    against the float32 official_predict: reported (an
                                    independent float32 product rounds differently); n_v < 32,
                                    n_v > N_chunk, input LN
    check_q1e_val(dataset, pair)    the same on the processed validation videos (real lengths
                                    and value ranges: the d16m32l2 InputPipeline - per-snippet
                                    l2 and the scale map - and the input LN for XCB+SigLIP)
    run_q1b(cfg)                    the port's own training loop (wvad.baselines.train.Chain,
                                    float64, deterministic, CPU) and reference_training (same seed,
                                    the official draws, the same dropout masks drawn from torch's
                                    CPU generator in the port's order) for 200 iterations; the
                                    per-iteration losses must agree to a relative 1e-5; the record
                                    holds the official clone (relative to WVAD_OFFICIAL_ROOT),
                                    commit, transcribed blob ids, diff sha256 (none: no modified
                                    official file), the opened-file audit (Python audit hook
                                    "open" events, test-like paths flagged), the plan sha256

    RUN = python -c "import sys; from wvad.baselines import official, check_mil;
                     check_mil.main(sys.argv[1:], official=official)"
    RUN q1 [--dims 4096 1792 1920] [--out JSON]
    RUN q1c [--dataset ucf|xdv] [--steps 20000] [--out JSON]
    RUN q1e [--n_chunk 4096] [--out JSON]
    RUN q1e_val --dataset ucf|xdv --pair xcb_xcl|xcb_siglip [--max_videos K] [--paths FILE]
                [--out JSON]
    RUN q1b --dataset ucf --pair xcb_xcl --seed 0 --out_dir DIR [--iterations 200]
            [--paths FILE] [--record FILE]
        --record merges the run into a JSON file (results/e023_q1b.json, under a lock) under
        methods.mil.<dataset>/<pair>/seed<k>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt_time
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn

from wvad.baselines import mil, samplers
from wvad.baselines.common import (
    N_CHUNK_DEFAULT,
    Batch,
    chunk_bounds,
    input_ln,
    score_video,
    update_json,
)
from wvad.baselines.recipes import RESULTS
from wvad.eval.dense_readout import sigmoid32
from wvad.paths import REPO_ROOT

ENV = "WVAD_OFFICIAL_ROOT"
CLONE = "AnomalyDetectionCVPR2018"
JUDGEMENT = ("float64: port in float64 against the float64 reference (PLAN.md 기록 2026-10-19, "
             "review M3); the fp32 port's errors are reported, not judged")
LICENCE_KEY = "mil"
PLAN_FILE = REPO_ROOT / "experiments" / "E023_localisation_baselines" / "PLAN.md"
Q1B_FILE = RESULTS / "e023_q1b.json"
# git blob ids at commit 8aa957c of the files this module transcribes (LF line endings)
OFFICIAL_BLOBS = {"TrainingAnomalyDetector_public.py": "366751765037fdf00a91bee10397249c9eaf4dac",
                  "Test_Anomaly_Detector_public.py": "dfa8bb3e26f38ba382524fbe671e973910a667f9",
                  "model.json": "4cfa453d384b1a982a6793bd02cc6483ab173868",
                  "Save_C3DFeatures_32Segments.m": "642c73c22776a4598b995d04393c7181432ad1f7"}
HIDDEN = (512, 32)                      # T L31, L33
RETAIN = 1. - 0.6                       # Keras 1.1.0 K.dropout: retain_prob = 1. - level
REF_L2 = float(np.float32(0.001))       # WeightRegularizer: K.cast_to_floatx(0.001)
REF_LR, REF_EPS = 0.01, 1e-08           # T L276
LN_EPS = 1e-5                           # torch.nn.LayerNorm default (the adaptation)
Q1_TOL = 1e-6                           # PLAN.md (q1): relative error of loss and gradients
Q1E_TOL = 1e-6                          # PLAN.md (q1e): elementwise |delta|
Q1B_TOL = 1e-5                          # PLAN.md (q1b) MIL: per-iteration relative loss diff
Q1B_ITERATIONS = 200
WEIGHTS = ("W1", "W2", "W3")
# opened-file audit: paths that look like test data (lower-case, posix)
FLAG_PATTERNS = (r"_d16_test", r"/dense16/test", r"/mil/test/", r"data/manifests/[^/]*test",
                 r"data/annotations/", r"testing_normal_videos", r"xd-violence/test",
                 r"temporal_anomaly_annotation", r"samplevideos", r"paper_results",
                 r"(^|/)gt-[^/]*\.npy$", r"_test\.(json|npy|npz|txt|list|csv|pt)$")


# --- the reference: construction --------------------------------------------------------------

def keras_init(feature_dim: int, ln: bool = False) -> dict[str, np.ndarray]:
    """T L29-L35 under Keras 1.1.0 from numpy's global generator: Sequential.add calls each
    layer when added (models.py L257-L308), so W1 = glorot_normal (initializations.py L48-L53:
    np.random.normal(0, sqrt(2 / (fan_in + fan_out)), (fan_in, fan_out)), theano_backend.py
    L144-L146, float32), then Dropout's K.dropout draws np.random.randint(1, 10e6)
    (theano_backend.py L1009-L1010; layers/core.py L87-L91), W2, the second seed, W3; biases
    K.zeros (layers/core.py L694-L698). ``ln``: the adaptation's LayerNorm (1, 0), no draw."""
    out: dict[str, np.ndarray] = {}
    if ln:
        out["ln_w"] = np.ones(int(feature_dim), dtype=np.float32)
        out["ln_b"] = np.zeros(int(feature_dim), dtype=np.float32)
    dims = (int(feature_dim), *HIDDEN, 1)
    for i in range(3):
        fan_in, fan_out = dims[i], dims[i + 1]
        s = np.sqrt(2. / (fan_in + fan_out))
        out[f"W{i + 1}"] = np.asarray(np.random.normal(loc=0.0, scale=s, size=(fan_in, fan_out)),
                                      dtype=np.float32)
        out[f"b{i + 1}"] = np.zeros(fan_out, dtype=np.float32)
        if i < 2:
            np.random.randint(1, 10e6)
    return out


# --- the reference: network ---------------------------------------------------------------

def _sigmoid(a: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-a))


def theano_sigmoid(a: np.ndarray) -> np.ndarray:
    """Theano 1.0.2 ScalarSigmoid C code (tensor/nnet/sigm.py L79-L83): float32 0 below -88 and
    1 above 15, float64 0 below -709 and 1 above 19, else 1 / (1 + exp(-x)) in that dtype."""
    a = np.asarray(a)
    lo, hi = (-88.0, 15.0) if a.dtype == np.float32 else (-709.0, 19.0)
    one = a.dtype.type(1.0)
    with np.errstate(over="ignore"):
        mid = one / (one + np.exp(-a))
    return np.where(a < lo, a.dtype.type(0.0), np.where(a > hi, one, mid)).astype(a.dtype)


def ref_forward(params: Mapping[str, np.ndarray], x: np.ndarray,
                masks: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, dict]:
    """Logits (rows,) of rows (rows, D) and the cache for ref_backward; ``masks`` (rows, 512)
    and (rows, 32) 0 / 1 (train mode) or None (Keras learning phase 0: no dropout)."""
    c: dict = {"masks": masks}
    xin = x
    if "ln_w" in params:
        mu = x.mean(axis=1, keepdims=True)
        var = np.square(x - mu).mean(axis=1, keepdims=True)
        c["xhat"] = (x - mu) / np.sqrt(var + LN_EPS)
        xin = c["xhat"] * params["ln_w"] + params["ln_b"]
    c["xin"] = xin
    c["a1"] = np.dot(xin, params["W1"]) + params["b1"]
    h1 = 0.5 * (c["a1"] + np.abs(c["a1"]))                        # Theano T.nnet.relu
    c["h1d"] = h1 if masks is None else h1 * masks[0] / RETAIN
    a2 = np.dot(c["h1d"], params["W2"]) + params["b2"]
    c["h2d"] = a2 if masks is None else a2 * masks[1] / RETAIN
    logits = (np.dot(c["h2d"], params["W3"]) + params["b3"])[:, 0]
    return logits, c


def ref_backward(params: Mapping[str, np.ndarray], c: Mapping, g_logits: np.ndarray
                 ) -> dict[str, np.ndarray]:
    """Gradients of every parameter given d loss / d logits (Theano's rules: relu'(0) = 0.5)."""
    masks, g = c["masks"], {}
    g_a3 = g_logits[:, None]
    g["W3"], g["b3"] = np.dot(c["h2d"].T, g_a3), g_a3.sum(axis=0)
    g_a2 = np.dot(g_a3, params["W3"].T)
    if masks is not None:
        g_a2 = g_a2 * masks[1] / RETAIN
    g["W2"], g["b2"] = np.dot(c["h1d"].T, g_a2), g_a2.sum(axis=0)
    g_h1 = np.dot(g_a2, params["W2"].T)
    if masks is not None:
        g_h1 = g_h1 * masks[0] / RETAIN
    g_a1 = g_h1 * 0.5 * (1.0 + np.sign(c["a1"]))
    g["W1"], g["b1"] = np.dot(c["xin"].T, g_a1), g_a1.sum(axis=0)
    if "ln_w" in params:
        g_xin = np.dot(g_a1, params["W1"].T)
        g["ln_w"], g["ln_b"] = (g_xin * c["xhat"]).sum(axis=0), g_xin.sum(axis=0)
    return g


# --- the reference: objective ---------------------------------------------------------------

def custom_objective_ref(y_true: np.ndarray, y_pred: np.ndarray, n_seg: int = 32,
                         nvid: int = 60):
    """T L198-L273 line by line (T -> numpy; n_seg / nvid parameters, official 32 / 60; Python 2
    integer division at L206). y_true: 0 on the anomalous videos' rows, 1 on the normal ones'
    (T L182-L192)."""
    y_true = np.asarray(y_true).reshape(-1)                                    # L201
    y_pred = np.asarray(y_pred).reshape(-1)                                    # L202
    n_exp = nvid // 2                                                          # L206
    Num_d = n_seg * nvid                                                       # L207
    sub_max = np.ones_like(y_pred)                                             # L210
    sub_sum_labels = np.ones_like(y_true)                                      # L211
    sub_sum_l1 = np.ones_like(y_true)                                          # L212
    sub_l2 = np.ones_like(y_true)                                              # L213
    for ii in range(0, nvid, 1):                                               # L215
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]                             # L217
        sub_sum_labels = np.concatenate([sub_sum_labels, np.stack([np.sum(mm)])])      # L218
        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]                     # L221
        sub_max = np.concatenate([sub_max, np.stack([np.max(Feat_Score)])])    # L222
        sub_sum_l1 = np.concatenate([sub_sum_l1, np.stack([np.sum(Feat_Score)])])      # L223
        z1 = np.ones_like(Feat_Score)                                          # L225
        z2 = np.concatenate([z1, Feat_Score])                                  # L226
        z3 = np.concatenate([Feat_Score, z1])                                  # L227
        z_22 = z2[n_seg - 1:]                                                  # L228 z2[31:]
        z_44 = z3[:n_seg + 1]                                                  # L229 z3[:33]
        z = z_22 - z_44                                                        # L230
        z = z[1:n_seg]                                                         # L231 z[1:32]
        z = np.sum(np.square(z))                                               # L232
        sub_l2 = np.concatenate([sub_l2, np.stack([z])])                       # L233
    sub_score = sub_max[Num_d:]                                                # L241
    F_labels = sub_sum_labels[Num_d:]                                          # L242
    sub_sum_l1 = sub_sum_l1[Num_d:]                                            # L251
    sub_sum_l1 = sub_sum_l1[:n_exp]                                            # L252
    sub_l2 = sub_l2[Num_d:]                                                    # L253
    sub_l2 = sub_l2[:n_exp]                                                    # L254
    indx_nor = np.equal(F_labels, n_seg).nonzero()[0]                          # L257 (== 32)
    indx_abn = np.equal(F_labels, 0).nonzero()[0]                              # L258
    n_Nor = n_exp                                                              # L260
    Sub_Nor = sub_score[indx_nor]                                              # L262
    Sub_Abn = sub_score[indx_abn]                                              # L263
    z = np.ones_like(y_true)                                                   # L265
    for ii in range(0, n_Nor, 1):                                              # L266
        sub_z = np.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)                       # L267
        z = np.concatenate([z, np.stack([np.sum(sub_z)])])                     # L268
    z = z[Num_d:]                                                              # L270
    z = np.mean(z, axis=-1) + 0.00008 * np.sum(sub_sum_l1) + 0.00008 * np.sum(sub_l2)  # L271
    return z


def custom_objective_grad_ref(y_true: np.ndarray, y_pred: np.ndarray, n_seg: int = 32,
                              nvid: int = 60) -> np.ndarray:
    """d custom_objective / d y_pred (rows,) under Theano's rules: T.max gives every element
    equal to the maximum the gradient (tensor/basic.py L1480), T.maximum(x, 0) passes it to x
    where x >= 0 (scalar/basic.py L1655-L1657), T.sqr -> 2 x, T.mean over the n_Nor terms."""
    y_true = np.asarray(y_true).reshape(-1)
    s = np.asarray(y_pred).reshape(nvid, n_seg)
    n_exp = nvid // 2
    lab = y_true.reshape(nvid, n_seg).sum(axis=1)
    vmax = s.max(axis=1)
    nor, abn = np.flatnonzero(lab == n_seg), np.flatnonzero(lab == 0)
    g_vmax = np.zeros(nvid, dtype=s.dtype)
    for ii in range(n_exp):
        act = ((1 - vmax[abn] + vmax[nor[ii]]) >= 0).astype(s.dtype)
        g_vmax[nor[ii]] += act.sum() / n_exp
        g_vmax[abn] -= act / n_exp
    g = (s == vmax[:, None]).astype(s.dtype) * g_vmax[:, None]
    g[:n_exp] += 0.00008
    d = s[:n_exp, :-1] - s[:n_exp, 1:]
    g[:n_exp, :-1] += 0.00008 * 2.0 * d
    g[:n_exp, 1:] -= 0.00008 * 2.0 * d
    return g.reshape(-1)


def ref_loss_and_grads(params: Mapping[str, np.ndarray], x: np.ndarray, labels: Sequence[int],
                       masks: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """The official batch loss (custom_objective + the three W_regularizer terms, Keras
    training.py L613-L633) of features (nvid, n_seg, D) whose labels (1 anomalous) list the
    anomalous videos first, and its gradients."""
    nvid, n_seg, d = x.shape
    lab = np.asarray(labels, dtype=np.int64)
    y_true = np.repeat(1 - lab, n_seg).astype(x.dtype)                        # T L182-L192
    logits, cache = ref_forward(params, x.reshape(nvid * n_seg, d), masks)
    y = _sigmoid(logits)
    objective = custom_objective_ref(y_true, y, n_seg, nvid)
    total = objective
    for w in WEIGHTS:                                         # regularized_loss += sum(l2 W^2)
        total = total + np.sum(REF_L2 * np.square(params[w]))
    g_y = custom_objective_grad_ref(y_true, y, n_seg, nvid)
    grads = ref_backward(params, cache, g_y * y * (1.0 - y))   # ScalarSigmoid.grad: gz y (1 - y)
    for w in WEIGHTS:
        grads[w] = grads[w] + REF_L2 * 2.0 * params[w]
    return {"total": float(total), "objective": float(objective), "grads": grads,
            "logits": logits}


def adagrad_ref(params: Mapping[str, np.ndarray], grads: Mapping[str, np.ndarray],
                accum: Mapping[str, np.ndarray], lr: float = REF_LR, eps: float = REF_EPS
                ) -> tuple[dict, dict]:
    """Keras 1.1.0 Adagrad (optimizers.py L271-L274): a' = a + g^2, p' = p - lr g / (sqrt(a') +
    eps)."""
    new_p, new_a = {}, {}
    for k, p in params.items():
        a = accum[k] + np.square(grads[k])
        new_a[k] = a
        new_p[k] = p - lr * grads[k] / (np.sqrt(a) + eps)
    return new_p, new_a


def official_draw(num_abnormal: int, num_normal: int, n_exp: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """T L100-L103 (Num_abnormal / Num_Normal = the pool sizes instead of 810 / 800)."""
    Abnor_list_iter = np.random.permutation(num_abnormal)
    Abnor_list_iter = Abnor_list_iter[num_abnormal - n_exp:]
    Norm_list_iter = np.random.permutation(num_normal)
    Norm_list_iter = Norm_list_iter[num_normal - n_exp:]
    return Abnor_list_iter, Norm_list_iter


def official_predict(params: Mapping[str, np.ndarray], x: np.ndarray, dtype=np.float32
                     ) -> tuple[np.ndarray, np.ndarray]:
    """predict_on_batch (Test_Anomaly_Detector_public.py L121) of one video's rows in ``dtype``:
    the pre-sigmoid logits and the Theano sigmoid scores (dropout off)."""
    p = {k: np.asarray(v, dtype=dtype) for k, v in params.items()}
    logits, _ = ref_forward(p, np.asarray(x, dtype=dtype), None)
    logits = logits.astype(dtype)
    return logits, theano_sigmoid(logits)


# --- the port side --------------------------------------------------------------------------

def params_of(model: nn.Module, dtype=np.float64) -> dict[str, np.ndarray]:
    """The port's parameters in the reference layout (Keras W: (in, out))."""
    out: dict[str, np.ndarray] = {}
    if isinstance(model.input_ln, nn.LayerNorm):
        out["ln_w"] = model.input_ln.weight.detach().cpu().numpy().astype(dtype)
        out["ln_b"] = model.input_ln.bias.detach().cpu().numpy().astype(dtype)
    for i, fc in enumerate(model.dense_layers(), start=1):
        out[f"W{i}"] = fc.weight.detach().cpu().numpy().T.astype(dtype)
        out[f"b{i}"] = fc.bias.detach().cpu().numpy().astype(dtype)
    return out


def load_params(model: nn.Module, params: Mapping[str, np.ndarray]) -> None:
    """Copy reference-layout parameters into the port (its dtype and device)."""
    with torch.no_grad():
        if isinstance(model.input_ln, nn.LayerNorm):
            model.input_ln.weight.copy_(torch.as_tensor(params["ln_w"]))
            model.input_ln.bias.copy_(torch.as_tensor(params["ln_b"]))
        for i, fc in enumerate(model.dense_layers(), start=1):
            fc.weight.copy_(torch.as_tensor(np.ascontiguousarray(params[f"W{i}"].T)))
            fc.bias.copy_(torch.as_tensor(params[f"b{i}"]))


class FixedMask(nn.Module):
    """A dropout stand-in applying a given 0 / 1 mask as torch's dropout does (x * (m / (1 -
    p)))."""

    def __init__(self, mask: np.ndarray, p: float = mil.DROPOUT) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.mask.to(dtype=x.dtype, device=x.device).reshape(x.shape) / (1 - self.p)
        return x * noise


@contextlib.contextmanager
def injected_masks(model: nn.Module, masks: tuple[np.ndarray, np.ndarray]) -> Iterator[None]:
    """The port's two Dropout modules replaced by FixedMask while open (harness only)."""
    saved = (model.drop1, model.drop2)
    model.drop1, model.drop2 = FixedMask(masks[0]), FixedMask(masks[1])
    try:
        yield
    finally:
        model.drop1, model.drop2 = saved


def _batch(x: np.ndarray, labels: Sequence[int], dtype: torch.dtype) -> Batch:
    lab = [int(y) for y in labels]
    nvid, n_seg = x.shape[:2]
    return Batch(torch.as_tensor(x, dtype=dtype), torch.tensor(lab, dtype=dtype),
                 torch.full((nvid,), n_seg, dtype=torch.int64), tuple(range(nvid)),
                 sum(1 for y in lab if y == 0))


def _rel(a, b) -> float:
    """max |a - b| / max |b| (0 when both are 0)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = float(np.max(np.abs(b))) if b.size else 0.0
    num = float(np.max(np.abs(a - b))) if a.size else 0.0
    return num / den if den > 0 else num


# --- gate (q1) ------------------------------------------------------------------------------

def _toy(feature_dim: int, seed: int, nvid: int, n_seg: int, x_scale: float):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, x_scale, (nvid, n_seg, feature_dim))
    labels = [1] * (nvid // 2) + [0] * (nvid // 2)
    masks = ((rng.random((nvid * n_seg, HIDDEN[0])) < RETAIN).astype(np.float64),
             (rng.random((nvid * n_seg, HIDDEN[1])) < RETAIN).astype(np.float64))
    return rng, x, labels, masks


def _perturbed(params: Mapping[str, np.ndarray], rng) -> dict[str, np.ndarray]:
    """Non-trivial biases and LN affine on top of the official initial weights."""
    out = {k: np.asarray(v, dtype=np.float64).copy() for k, v in params.items()}
    for k in out:
        if k.startswith("b"):
            out[k] = rng.normal(0.0, 0.1, out[k].shape)
    if "ln_w" in out:
        out["ln_w"] = 1.0 + rng.normal(0.0, 0.1, out["ln_w"].shape)
        out["ln_b"] = rng.normal(0.0, 0.1, out["ln_b"].shape)
    return out


def _port_loss(model, x, labels, masks, dtype) -> tuple[dict, dict, np.ndarray]:
    method = mil.METHOD
    model.zero_grad(set_to_none=True)
    batch = _batch(x, labels, dtype)
    ctx = injected_masks(model, masks) if masks is not None else contextlib.nullcontext()
    method.set_mode(model, masks is not None)
    with ctx:
        out = method.loss(model, batch, unit=x.shape[1], scale={})
        out.total.backward()
        with torch.no_grad():
            logits = model(batch.features).reshape(-1).cpu().numpy()
    grads = {}
    for k, p in (("ln_w", getattr(model.input_ln, "weight", None)),
                 ("ln_b", getattr(model.input_ln, "bias", None))):
        if p is not None:
            grads[k] = p.grad.cpu().numpy().astype(np.float64)
    for i, fc in enumerate(model.dense_layers(), start=1):
        grads[f"W{i}"] = fc.weight.grad.cpu().numpy().T.astype(np.float64)
        grads[f"b{i}"] = fc.bias.grad.cpu().numpy().astype(np.float64)
    vals = {"total": float(out.total.detach()), "objective": float(out.parts["objective"])}
    return vals, grads, logits


def check_q1(feature_dim: int, *, ln: bool = False, seed: int = 0, nvid: int = 60,
             n_seg: int = 32, x_scale: float = 1.0, steps: int = 2, tol: float = Q1_TOL,
             dtype: torch.dtype = torch.float64) -> dict:
    """Gate (q1) on fixed toy inputs: constructor, forward, loss, gradients and ``steps`` Adagrad
    steps of the port (``dtype``) against the float64 reference, in eval mode and in train mode
    with injected masks."""
    method = mil.METHOD
    np.random.seed(seed)
    t_state = torch.get_rng_state()
    model = method.build(feature_dim, unit=n_seg, input_ln=ln, scale={})
    torch_untouched = bool(torch.equal(t_state, torch.get_rng_state()))
    np_after_port = np.random.get_state()
    np.random.seed(seed)
    init = keras_init(feature_dim, ln)
    ctor = {k: bool(np.array_equal(params_of(model, np.float32)[k], v)) for k, v in init.items()}
    ctor_stream = _np_state_equal(np_after_port, np.random.get_state())
    rng, x, labels, masks = _toy(feature_dim, seed + 1, nvid, n_seg, x_scale)
    params = _perturbed(init, rng)
    model = model.to(dtype=dtype)
    modes = {}
    for mode, m in (("eval", None), ("train_injected_masks", masks)):
        load_params(model, params)
        ref = ref_loss_and_grads(params, x, labels, m)
        vals, grads, logits = _port_loss(model, x, labels, m, dtype)
        res = {"logits_rel": _rel(logits, ref["logits"]),
               "loss_rel": _rel(vals["total"], ref["total"]),
               "objective_rel": _rel(vals["objective"], ref["objective"]),
               "grad_rel": {k: _rel(grads[k], ref["grads"][k]) for k in ref["grads"]},
               "loss_port": vals["total"], "loss_reference": ref["total"]}
        # optimizer steps (same batch and masks each step)
        load_params(model, params)
        opt = method.make_optimizer(model, mil.RECIPE.optimizer.lr)
        ref_p, ref_a = dict(params), {k: np.zeros_like(v) for k, v in params.items()}
        step_rel = []
        for _ in range(int(steps)):
            ctx = injected_masks(model, m) if m is not None else contextlib.nullcontext()
            method.set_mode(model, m is not None)
            with ctx:
                method.train_step(model, _batch(x, labels, dtype), opt, unit=n_seg, scale={})
            r = ref_loss_and_grads(ref_p, x, labels, m)
            ref_p, ref_a = adagrad_ref(ref_p, r["grads"], ref_a)
            got = params_of(model)
            step_rel.append({k: _rel(got[k], ref_p[k]) for k in ref_p})
        res["step_rel"] = step_rel
        worst = max([res["logits_rel"], res["loss_rel"], res["objective_rel"],
                     *res["grad_rel"].values(), *[v for s in step_rel for v in s.values()]])
        res["max_rel"], res["ok"] = worst, worst <= tol
        modes[mode] = res
    ok = all(ctor.values()) and ctor_stream and torch_untouched and all(
        r["ok"] for r in modes.values())
    return {"gate": "q1", "feature_dim": int(feature_dim), "input_ln": bool(ln), "seed": seed,
            "nvid": nvid, "n_seg": n_seg, "dtype": str(dtype), "tolerance": tol,
            "constructor_bitwise": ctor, "constructor_numpy_stream_equal": ctor_stream,
            "constructor_torch_generator_untouched": torch_untouched, "modes": modes, "ok": ok}


# --- gate (q1c) -----------------------------------------------------------------------------

def _np_state_equal(a, b) -> bool:
    return a[0] == b[0] and bool(np.array_equal(a[1], b[1])) and tuple(a[2:]) == tuple(b[2:])


def check_q1c(pools: samplers.Pools, *, steps: int, seed: int) -> dict:
    """Gate (q1c): the registered MIL sampler and official_draw from one numpy state, step by
    step (positions, labels, order) and the numpy state after the last step."""
    recipe = mil.RECIPE
    n_exp = int(recipe.sampler_params["batch_abnormal"])
    sampler = samplers.create(recipe.sampler, pools, recipe.sampler_params)
    np.random.seed(seed)
    start = np.random.get_state()
    port = [sampler.draw(s).key() for s in range(1, int(steps) + 1)]
    after_port = np.random.get_state()
    np.random.set_state(start)
    first, labels_ok = None, True
    for s in range(int(steps)):
        a, n = official_draw(len(pools.abnormal), len(pools.normal), n_exp)
        want = tuple(pools.abnormal[int(i)] for i in a) + tuple(pools.normal[int(i)] for i in n)
        labels_ok &= port[s][1] == (1,) * n_exp + (0,) * n_exp
        if first is None and port[s][0] != want:
            first = s + 1
    state_ok = _np_state_equal(after_port, np.random.get_state())
    return {"gate": "q1c", "steps": int(steps), "seed": seed, "pools": [len(pools.abnormal),
                                                                         len(pools.normal)],
            "n_exp": n_exp, "first_mismatch_step": first, "labels_ok": bool(labels_ok),
            "numpy_state_equal_after": state_ok,
            "ok": first is None and bool(labels_ok) and state_ok}


# --- gate (q1e) -----------------------------------------------------------------------------

def official_pieces(n: int, n_chunk: int) -> list[tuple[int, int]]:
    """The plan's N_chunk rule written out independently: the whole video when n <= N_chunk,
    else non-overlapping N_chunk pieces."""
    if n <= n_chunk:
        return [(0, n)]
    return [(a, min(a + n_chunk, n)) for a in range(0, n, n_chunk)]


def check_q1e(feature_dim: int, lengths: Sequence[int], *, n_chunk: int = N_CHUNK_DEFAULT,
              ln: bool = False, seed: int = 0, weight_scale: float = 1.0, x_scale: float = 1.0,
              tol: float = Q1E_TOL, videos: Sequence[np.ndarray] | None = None) -> dict:
    """Gate (q1e) on synthetic rows of real lengths, the same weights (official init, biases /
    LN affine perturbed, weights x ``weight_scale``), eval mode, no grad:
      float64   the port's sequence_logits on each piece of common.chunk_bounds (float64 model)
                against official_predict (float64) on each piece of official_pieces: logits and
                scores within ``tol`` and the same pieces (the gate, as (q1) / (q1b) of MIL run
                in float64)
      fp32      common.score_video (the fp32 path as run) against official_predict (float32):
                reported (|delta| and |delta| / max(1, |logit|)); an independent float32 matrix
                product rounds differently (a few 1e-6 at |logit| ~ 5), so this is
                information, not bit equality.
    ``videos``: processed (n, D) rows (``check_q1e_val``) instead of synthetic rows of
    ``lengths``."""
    method = mil.METHOD
    np.random.seed(seed)
    model = method.build(feature_dim, unit=32, input_ln=ln, scale={})
    rng = np.random.default_rng(seed + 7)
    params = _perturbed(params_of(model), rng)
    for w in WEIGHTS:
        params[w] = params[w] * weight_scale
    model = model.float()
    load_params(model, params)
    params32 = params_of(model, np.float32)              # exactly the port's fp32 weights
    model64 = copy.deepcopy(model).double()
    method.set_mode(model, False)
    method.set_mode(model64, False)
    cpu = torch.device("cpu")
    cases = []
    feeds = [np.asarray(v, dtype=np.float32) for v in videos] if videos is not None else [
        rng.normal(0.0, x_scale, (int(n), int(feature_dim))).astype(np.float32)
        for n in lengths]
    for x in feeds:
        n = int(x.shape[0])
        pieces = official_pieces(int(n), int(n_chunk))
        with torch.no_grad():
            z64 = np.concatenate([method.sequence_logits(
                model64, torch.from_numpy(x[a:b]).double(), unit=32).numpy()
                for a, b in chunk_bounds(int(n), n_chunk)])
        z32 = score_video(method, model, torch.from_numpy(x), unit=32, n_chunk=n_chunk,
                          device=cpu)
        r64 = [official_predict(params32, x[a:b], np.float64) for a, b in pieces]
        r32 = [official_predict(params32, x[a:b], np.float32) for a, b in pieces]
        l64, s64 = np.concatenate([r[0] for r in r64]), np.concatenate([r[1] for r in r64])
        l32, s32 = np.concatenate([r[0] for r in r32]), np.concatenate([r[1] for r in r32])
        with np.errstate(over="ignore"):
            port_s64 = 1.0 / (1.0 + np.exp(-z64))
        case = {"n": int(n), "pieces": len(pieces),
                "same_pieces": list(chunk_bounds(int(n), n_chunk)) == pieces,
                "logit_abs_f64": float(np.max(np.abs(z64 - l64))),
                "score_abs_f64": float(np.max(np.abs(port_s64 - s64))),
                "logit_abs_fp32": float(np.max(np.abs(z32.astype(np.float64) - l32))),
                "score_abs_fp32": float(np.max(np.abs(sigmoid32(z32).astype(np.float64)
                                                      - s32))),
                "logit_rel_fp32": float(np.max(np.abs(z32.astype(np.float64) - l32)
                                               / np.maximum(1.0, np.abs(l64)))),
                "max_abs_logit": float(np.max(np.abs(l64)))}
        case["fp32_within_tol"] = case["logit_abs_fp32"] <= tol and case["score_abs_fp32"] <= tol
        case["ok"] = (case["same_pieces"] and case["logit_abs_f64"] <= tol
                      and case["score_abs_f64"] <= tol)
        cases.append(case)
    return {"gate": "q1e", "feature_dim": int(feature_dim), "input_ln": bool(ln),
            "n_chunk": int(n_chunk), "weight_scale": weight_scale, "x_scale": x_scale,
            "inputs": "synthetic rows" if videos is None else "processed rows",
            "judgement": JUDGEMENT, "tolerance": tol, "cases": cases,
            "fp32_within_tol": all(c["fp32_within_tol"] for c in cases),
            "not_applicable": {
                "last_chunk_padding": "MIL's test sequence is 'whole' (row-wise network): no "
                                      "padded piece; the last N_chunk piece is shorter and "
                                      "unpadded (cases with n > N_chunk)",
                "eval_mode_bn": "no BatchNorm in the official network; eval mode = dropout off"},
            "ok": all(c["ok"] for c in cases)}


def check_q1e_val(dataset: str, pair: str, *, paths=None, split=None,
                  n_chunk: int = N_CHUNK_DEFAULT, seed: int = 0, weight_scale: float = 1.0,
                  max_videos: int | None = None) -> dict:
    """Gate (q1e) on the processed validation videos (PLAN.md (q1e): real-length validation
    videos): train.validation_videos (the dense training-split store through the d16m32l2
    InputPipeline: per-snippet l2, the scale map) of ``dataset`` / ``pair``, the input LN for
    XCB+SigLIP; otherwise ``check_q1e``."""
    from wvad.baselines import train as tr
    from wvad.data import load_split, split_file

    if paths is None:
        from wvad.paths import load_paths

        paths = load_paths(None)
    split = split if split is not None else load_split(split_file(dataset))
    variant = mil.RECIPE.unit_input(32).variant
    val = tr.validation_videos(paths, dataset, pair, variant, split, max_videos=max_videos)
    res = check_q1e(int(sum(val.dims)), (), n_chunk=n_chunk, ln=input_ln(pair), seed=seed,
                    weight_scale=weight_scale, videos=[x.numpy() for x in val.x])
    for c, v in zip(res["cases"], val.indices, strict=True):
        c["video"] = int(v)
    return res | {"check": "E023 (q1e) MIL on the processed validation videos",
                  "dataset": dataset, "pair": pair, "variant": variant,
                  "pipeline": val.pipeline.to_json(), "videos": len(val.x),
                  "of_validation_videos": len(split.val_indices),
                  "max_snippets": max(int(x.shape[0]) for x in val.x),
                  "ok": bool(val.x) and res["ok"]}


# --- the (q1b) reference training -----------------------------------------------------------

def reference_training(features: Sequence, labels: Sequence[int], *, seed: int,
                       iterations: int, ln: bool, n_exp: int = 30) -> dict:
    """The official training loop (T L296-L306) as numpy float64: random / numpy / torch seeded
    with ``seed``; keras_init; per iteration official_draw over the anomalous / normal positions
    (training-part order), the batch anomalous first (T L119-L177), the two dropout masks drawn
    as torch's CPU dropout draws them in the port's forward order (empty((60, 32, 512),
    float64).bernoulli_(1 - p), then (60, 32, 32)), the loss and gradients, adagrad_ref."""
    lab = np.asarray([int(y) for y in labels], dtype=np.int64)
    abn_pool, nor_pool = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    d = int(np.asarray(features[0]).shape[1])
    params = {k: v.astype(np.float64) for k, v in keras_init(d, ln).items()}
    accum = {k: np.zeros_like(v) for k, v in params.items()}
    losses, draws = [], []
    for _ in range(int(iterations)):
        a_idx, n_idx = official_draw(len(abn_pool), len(nor_pool), n_exp)
        vids = [int(abn_pool[i]) for i in a_idx] + [int(nor_pool[i]) for i in n_idx]
        x = np.stack([np.asarray(features[v], dtype=np.float64) for v in vids])
        nvid, n_seg = x.shape[:2]
        m1 = torch.empty((nvid, n_seg, HIDDEN[0]), dtype=torch.float64).bernoulli_(
            1 - mil.DROPOUT).numpy().reshape(nvid * n_seg, HIDDEN[0])
        m2 = torch.empty((nvid, n_seg, HIDDEN[1]), dtype=torch.float64).bernoulli_(
            1 - mil.DROPOUT).numpy().reshape(nvid * n_seg, HIDDEN[1])
        r = ref_loss_and_grads(params, x, [1] * len(a_idx) + [0] * len(n_idx), (m1, m2))
        params, accum = adagrad_ref(params, r["grads"], accum)
        losses.append(r["total"])
        draws.append(tuple(vids))
    return {"losses": np.asarray(losses, dtype=np.float64), "draws": draws, "params": params}


class Recorder(mil.SultaniMIL):
    """The port method recording each step's loss and batch videos (the loop calls train_step
    once per step, in order)."""

    def __init__(self) -> None:
        self.steps: list[tuple[float, tuple[int, ...]]] = []

    @property
    def recipe(self):
        return mil.RECIPE

    def train_step(self, model, batch, optimizer, *, unit, scale):
        out = super().train_step(model, batch, optimizer, unit=unit, scale=scale)
        self.steps.append((float(out["total"]), tuple(batch.videos)))
        return out


# --- opened-file audit ----------------------------------------------------------------------

_AUDIT: dict = {"installed": False, "active": None}


def _audit_hook(event: str, args: tuple) -> None:
    sink = _AUDIT["active"]
    if sink is None or event != "open" or not args:
        return
    try:
        path = args[0]
        if isinstance(path, int):
            return
        sink.append(os.fsdecode(os.fspath(path)))
    except Exception:  # noqa: BLE001 - an audit hook must never raise
        return


@contextlib.contextmanager
def audited_opens() -> Iterator[list[str]]:
    """The paths of every Python-level "open" audit event while open (builtins / io.open,
    os.open, io.open_code; a file opened by C code alone is not seen)."""
    if not _AUDIT["installed"]:
        sys.addaudithook(_audit_hook)
        _AUDIT["installed"] = True
    if _AUDIT["active"] is not None:
        raise RuntimeError("an opened-file audit is already active")
    sink: list[str] = []
    _AUDIT["active"] = sink
    try:
        yield sink
    finally:
        _AUDIT["active"] = None


def _rel_path(p: str, roots: Sequence[tuple[str, Path]]) -> str:
    try:
        path = Path(p).resolve()
    except (OSError, ValueError):
        return f"<unresolved>/{Path(p).name}"
    for label, root in roots:
        try:
            return f"{label}/{path.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return f"<other>/{path.name}"


def audit_summary(opened: Sequence[str], roots: Mapping[str, str | os.PathLike[str] | None]
                  ) -> dict:
    """Opened files relative to the named roots (longest first; anything else <other>/<name>),
    Python library files counted only, test-like paths (FLAG_PATTERNS) listed."""
    lib = {Path(p).resolve() for p in {sys.prefix, sys.base_prefix, sys.exec_prefix}}
    pairs = sorted(((k, Path(v).resolve()) for k, v in roots.items() if v),
                   key=lambda kv: -len(str(kv[1])))
    files, n_lib, flagged = set(), 0, set()
    for p in opened:
        try:
            rp = Path(p).resolve()
        except (OSError, ValueError):
            rp = Path(p)
        if any(root == rp or root in rp.parents for root in lib):
            n_lib += 1
            continue
        rel = _rel_path(p, pairs)
        files.add(rel)
        low = rp.as_posix().lower()
        if any(re.search(pat, low) for pat in FLAG_PATTERNS):
            flagged.add(rel)
    return {"events": len(opened), "python_library_opens": n_lib, "files": sorted(files),
            "flagged_test_like": sorted(flagged), "patterns": list(FLAG_PATTERNS),
            "scope": "Python audit hook 'open' events (C-level file access is not seen)"}


# --- official clone (no code executed) ------------------------------------------------------

def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def official_record(official: ModuleType) -> dict:
    """The clone (relative to WVAD_OFFICIAL_ROOT, through the ``official`` loader), its HEAD
    against the licence record, the git blob ids of the transcribed files (CRLF -> LF) against
    OFFICIAL_BLOBS. Reads, never runs."""
    clone = official.repo_dir("mil")
    head = official.clone_commit("mil")
    want = official.recorded_commit("mil")
    blobs = {}
    for name, blob in OFFICIAL_BLOBS.items():
        got = git_blob_id((clone / name).read_bytes().replace(b"\r\n", b"\n"))
        blobs[name] = {"blob": got, "expected": blob, "ok": got == blob}
    return {"repo": CLONE, "clone": f"<{ENV}>/{CLONE}", "commit": head,
            "commit_recorded": want, "commit_ok": head == want, "transcribed_files": blobs,
            "modified_files": None, "diff_sha256": None,
            "note": "the official Keras 1.1.0 / Theano 1.0.2 pipeline is not revived (PLAN.md "
                    "(q1b) MIL rule); its numpy float64 transcription in "
                    "wvad.baselines.check_mil stands in for it, so no official file is "
                    "modified (no diff)",
            "ok": head == want and all(b["ok"] for b in blobs.values())}


# --- the (q1b) runner -----------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_record(plan: Path = PLAN_FILE) -> dict:
    from wvad.data.dense_test import plan_sha256

    return {"file": plan.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(plan),
            "sha256_before_change_log": plan_sha256(plan)}


def _q1b_result(recorder: Recorder, ref: Mapping, model: nn.Module, iterations: int,
                audit: Mapping) -> dict:
    """Per-iteration losses, batch draws and final weights of the port against the reference."""
    port_losses = np.asarray([s[0] for s in recorder.steps], dtype=np.float64)
    rel = np.abs(port_losses - ref["losses"]) / np.abs(ref["losses"])
    over = np.flatnonzero(rel > Q1B_TOL)
    got = params_of(model)
    weights = {k: {"max_abs": float(np.max(np.abs(got[k] - ref["params"][k]))),
                   "rel": _rel(got[k], ref["params"][k])} for k in ref["params"]}
    draws_equal = [s[1] for s in recorder.steps] == ref["draws"]
    result = {"iterations": iterations, "tolerance": Q1B_TOL,
              "max_rel_loss_diff": float(rel.max()) if rel.size else None,
              "first_iteration_over_tolerance": int(over[0]) + 1 if over.size else None,
              "draws_equal": bool(draws_equal), "final_weights": weights,
              "loss_port": port_losses.tolist(), "loss_reference": ref["losses"].tolist()}
    result["ok"] = (len(port_losses) == iterations and over.size == 0 and bool(draws_equal)
                    and not audit["flagged_test_like"])
    return result


def _q1b_inputs(chain, paths) -> dict:
    root = getattr(paths, "feature_root", None)
    pairs = [("<feature_root>", Path(root).resolve())] if root else []
    labels = [int(y) for y in chain.inputs.train.labels]
    return {"train_dirs": [_rel_path(d, pairs) for d in chain.inputs.train_dirs],
            "train_data_sha256": [m.get("data_sha256") for m in chain.inputs.train_metas],
            "train_videos": len(labels), "abnormal": labels.count(1),
            "normal": labels.count(0), "split_sha256": chain.split.sha256}


def run_q1b(cfg, *, official: ModuleType | None = None, paths=None, split=None, log=print,
            audit_roots: Mapping[str, object] | None = None) -> dict:
    """Gate (q1b), MIL rule: ``cfg`` (wvad.baselines.train.TrainConfig, method mil) is forced to
    float64, deterministic, CPU, ``cfg.iterations or 200`` steps; the port chain runs in
    ``cfg.out_dir`` (the loop's files), then reference_training on the chain's own training
    inputs; returns the record (module docstring). ``official``: the loader module (the clone
    is recorded; None only in self-tests without a clone: "official" is then null and the
    record is no (q1b) run)."""
    from wvad.baselines import train as tr
    from wvad.diagnostics.common import code_version

    if cfg.method != "mil":
        raise ValueError(f"run_q1b is the MIL harness, got --method {cfg.method}")
    iterations = int(cfg.iterations or Q1B_ITERATIONS)
    cfg = replace(cfg, float64=True, deterministic=True, device="cpu", iterations=iterations)
    if paths is None:
        from wvad.paths import load_paths

        paths = load_paths(cfg.paths)
    clone = official_record(official) if official is not None else None
    roots = {"<repo>": REPO_ROOT, "<feature_root>": getattr(paths, "feature_root", None),
             "<out_dir>": Path(cfg.out_dir), "<tmp>": tempfile.gettempdir(), "~": Path.home()}
    if official is not None:
        roots[f"<{ENV}>"] = official.root()
    roots.update(audit_roots or {})
    recorder, was_det = Recorder(), torch.are_deterministic_algorithms_enabled()
    try:
        with audited_opens() as opened:
            chain = tr.Chain(cfg, recorder, paths, split, log)
            chain.train()
            port_record = chain.finish()
            train = chain.inputs.train
            ref = reference_training([x.numpy() for x in train.features], train.labels,
                                     seed=cfg.seed, iterations=iterations,
                                     ln=input_ln(cfg.pair),
                                     n_exp=int(mil.RECIPE.sampler_params["batch_abnormal"]))
    finally:
        torch.use_deterministic_algorithms(was_det)
    audit = audit_summary(opened, roots)
    result = _q1b_result(recorder, ref, chain.model, iterations, audit)
    return {"gate": "q1b", "method": "mil",
            "rule": "PLAN.md (q1b) MIL: numpy float64 reference of the official pipeline "
                    "(Keras 1.1.0 / Theano 1.0.2 not revived), the same batch order, "
                    f"{iterations} iterations, the port in float64; per-iteration relative "
                    f"loss difference <= {Q1B_TOL}",
            "config": {"dataset": cfg.dataset, "pair": cfg.pair, "seed": cfg.seed,
                       "unit": cfg.unit, "iterations": iterations, "dtype": "float64",
                       "device": "cpu", "deterministic": True, "evaluate": cfg.evaluate,
                       "n_chunk": cfg.n_chunk, "variant": chain.inputs.variant,
                       "input_ln": input_ln(cfg.pair),
                       "dropout_masks": "torch CPU generator, the port's draws (the reference "
                                        "draws the same tensors in the same order)"},
            "inputs": _q1b_inputs(chain, paths), "official": clone,
            "harness": {"module": "src/wvad/baselines/check_mil.py",
                        "sha256": _sha256(Path(__file__)),
                        "port_module": "src/wvad/baselines/mil.py",
                        "port_sha256": _sha256(Path(mil.__file__))},
            "plan": plan_record(), "code": code_version(),
            "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                            "numpy": np.__version__},
            "port_chain": {"out_dir": "<out_dir>", "selection_failed": port_record.get("failed"),
                           "evaluated": port_record.get("evaluated")},
            "audit": audit, "result": result,
            "ok": result["ok"] and (clone is None or clone["ok"]),
            "utc": dt_time.datetime.now(dt_time.timezone.utc).isoformat(timespec="seconds")}


def merge_record(path: str | os.PathLike[str], key: str, value: Mapping) -> None:
    """results/e023_q1b.json: ``value`` under methods.mil.<key> (other entries kept; the file is
    read, changed and written under a lock, so concurrent runs never overwrite each other)."""
    entry = json.loads(json.dumps(dict(value), default=str))
    update_json(path, lambda data: data.setdefault("methods", {}).setdefault("mil", {})
                .__setitem__(key, entry),
                default={"check": "E023 (q1b) official-pipeline comparisons (PLAN.md 관문)",
                         "methods": {}})


# --- command line ---------------------------------------------------------------------------

def _write(obj: Mapping, out: str | None) -> None:
    text = json.dumps(obj, indent=1, default=str)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text if not out else f"[check_mil] ok={obj.get('ok')} -> {out}")


def _split_pools(dataset: str) -> samplers.Pools:
    from wvad.data import load_split, split_file

    split = load_split(split_file(dataset))
    return samplers.Pools.from_labels([int(split.labels[i]) for i in split.train_indices])


def main(argv: Sequence[str] | None = None, *, official: ModuleType | None = None) -> None:
    if official is None:
        raise SystemExit("check_mil needs the official loader passed in: python -c \"import sys; "
                         "from wvad.baselines import official, check_mil; check_mil.main("
                         "sys.argv[1:], official=official)\" <gate> ...")
    p = argparse.ArgumentParser(prog="check_mil",
                                description="E023 MIL equivalence gates (q1), (q1c), (q1e), (q1b)")
    sub = p.add_subparsers(dest="gate", required=True)
    a1 = sub.add_parser("q1")
    a1.add_argument("--dims", type=int, nargs="+", default=[4096, 1792, 1920])
    a1.add_argument("--out")
    ac = sub.add_parser("q1c")
    ac.add_argument("--dataset", choices=("ucf", "xdv"),
                    help="the split file's training-part pools (default: 810 / 800)")
    ac.add_argument("--steps", type=int, default=mil.RECIPE.iterations)
    ac.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ac.add_argument("--out")
    ae = sub.add_parser("q1e")
    ae.add_argument("--n_chunk", type=int, default=N_CHUNK_DEFAULT)
    ae.add_argument("--out")
    av = sub.add_parser("q1e_val")
    av.add_argument("--dataset", required=True, choices=("ucf", "xdv"))
    av.add_argument("--pair", required=True, choices=("xcb_xcl", "xcb_siglip"))
    av.add_argument("--n_chunk", type=int, default=N_CHUNK_DEFAULT)
    av.add_argument("--max_videos", type=int, help="the first K validation videos and the longest")
    av.add_argument("--paths")
    av.add_argument("--out")
    ab = sub.add_parser("q1b")
    ab.add_argument("--dataset", required=True, choices=("ucf", "xdv"))
    ab.add_argument("--pair", default="xcb_xcl", choices=("xcb_xcl", "xcb_siglip"))
    ab.add_argument("--seed", required=True, type=int)
    ab.add_argument("--out_dir", required=True)
    ab.add_argument("--iterations", type=int, default=Q1B_ITERATIONS)
    ab.add_argument("--n_chunk", type=int, default=N_CHUNK_DEFAULT)
    ab.add_argument("--no_eval", action="store_true")
    ab.add_argument("--paths")
    ab.add_argument("--record", help=f"merge into this file (e.g. {Q1B_FILE.name})")
    a = p.parse_args(argv)
    if a.gate == "q1":
        rows = [check_q1(d, ln=(d == 1920)) for d in a.dims]
        fp32 = [check_q1(d, ln=(d == 1920), dtype=torch.float32) for d in a.dims]
        info = [{"feature_dim": r["feature_dim"], "input_ln": r["input_ln"],
                 "max_rel": {m: v["max_rel"] for m, v in r["modes"].items()},
                 "grad_max_rel": {m: max(v["grad_rel"].values()) for m, v in r["modes"].items()},
                 "within_1e-6": all(v["ok"] for v in r["modes"].values())} for r in fp32]
        _write({"gate": "q1", "judgement": JUDGEMENT, "runs": rows,
                "fp32_information": info, "ok": all(r["ok"] for r in rows)}, a.out)
    elif a.gate == "q1c":
        pools = _split_pools(a.dataset) if a.dataset else samplers.Pools(
            normal=tuple(range(810, 1610)), abnormal=tuple(range(810)))
        rows = [check_q1c(pools, steps=a.steps, seed=s) for s in a.seeds]
        _write({"gate": "q1c", "dataset": a.dataset, "runs": rows,
                "ok": all(r["ok"] for r in rows)}, a.out)
    elif a.gate == "q1e":
        n = int(a.n_chunk)
        lengths = [1, 5, 31, 32, 33, 517, n, n + 1, 2 * n + 3]
        rows = [check_q1e(d, lengths, n_chunk=n, ln=ln, weight_scale=s)
                for d, ln in ((1792, False), (1920, True)) for s in (1.0, 3.0)]
        _write({"gate": "q1e", "judgement": JUDGEMENT, "runs": rows,
                "fp32_information": [{"feature_dim": r["feature_dim"], "input_ln": r["input_ln"],
                                      "weight_scale": r["weight_scale"],
                                      "max_logit_abs_fp32": max(c["logit_abs_fp32"]
                                                                for c in r["cases"]),
                                      "max_logit_rel_fp32": max(c["logit_rel_fp32"]
                                                                for c in r["cases"]),
                                      "within_1e-6": r["fp32_within_tol"]} for r in rows],
                "ok": all(r["ok"] for r in rows)}, a.out)
    elif a.gate == "q1e_val":
        from wvad.paths import load_paths

        rec = check_q1e_val(a.dataset, a.pair, paths=load_paths(a.paths), n_chunk=a.n_chunk,
                            max_videos=a.max_videos)
        _write(rec | {"judgement": JUDGEMENT}, a.out)
    else:
        from wvad.baselines.train import TrainConfig

        out_dir = Path(a.out_dir)
        cfg = TrainConfig(method="mil", unit=32, dataset=a.dataset, pair=a.pair, seed=a.seed,
                          out_dir=str(out_dir / "port"), n_chunk=a.n_chunk,
                          evaluate=not a.no_eval, iterations=a.iterations, paths=a.paths)
        rec = run_q1b(cfg, official=official)
        _write(rec, str(out_dir / "q1b_mil.json"))
        if a.record:
            merge_record(a.record, f"{a.dataset}/{a.pair}/seed{a.seed}", rec)


if __name__ == "__main__":
    main()
