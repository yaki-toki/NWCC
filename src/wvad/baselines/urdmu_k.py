"""E024 stage 1: UR-DMU + K, the within-video contrast module (experiments/E024_localisation_method/
stage1/PLAN.md "K의 정의", "통제", "코드"). New code (add-only rule). Not a registered E023 method:
a chain runs ``train.run(TrainConfig(method="urdmu_k", ...), method=URDMUK(kcfg, seed))``
(``Chain`` checks only ``base.name == cfg.method``) with UR-DMU's recipe (``RECIPE =
urdmu.RECIPE``: optimizer, lr, sampler, unit 200 on d16m200).

K, per sequence (training: a video's 200 d16m200 window rows; scoring: one N_chunk piece), on the
512-d snippet embedding x_t after ``selfatt`` (before the memory augmentation is concatenated) and
the classifier logit z_t (``cls_head.logits``, before the sigmoid):
    z~_j   = (z_j - mean z) / (std z + 1e-6), the population std (correction 0), no gradient
    runs   r_t = r_{t-1} + 1 where the input row t (processed features, before ``input_norm`` and
           ``embedding``) differs from row t - 1 in any element; every j with r_j = r_t is left
           out of t's context (a d16m200 window copied from its neighbour is no context of it)
    l_tj   = -z~_j / tau - |t - j| / (rho T), -inf where left out; no z~ term when tau is None
           (tau = inf), no distance term when rho is None (global)
    w_t.   = softmax_j(l_t.) in log space (no 0 / 0 underflow), no gradient
    c_t    = sum_j w_tj x_j (gradient through x),  k_t = g(x_t - c_t),  s_t = z_t + k_t
A sequence of one run (T = 1 included) has k = 0. g = Linear(512, 128) -> ReLU -> Linear(128, 1)
(65,793 parameters), drawn from its own ``torch.Generator().manual_seed(1_000_000 + seed)`` with
nn.Linear's default bounds, the last layer zero (k = 0 at the start); the global generators are
never seeded or drawn by g, so the trunk's initialisation, the batch order and the dropout /
reparameterisation draws are those of urdmu.URDMU. The optimizer is the recipe's Adam with the
trunk in ``model.parameters()`` order and g as a second group (same lr and weight decay; Adam
updates per group and element, so the second group leaves the trunk's update unchanged).

The training branch returns ``frame`` = sigma(s) in place of ``cls_head(x)`` (AD_Loss unchanged),
the test branch ``logit`` = s and ``frame`` = sigma(s) (so ``sequence_logits`` returns s); both
add ``xc_norm`` = ||x_t - c_t|| (NaN without context) when the mode has a context. The test
branch adds k to the logit while ``k_in_logit`` holds (default: every mode but trainonly);
``rescore`` sets it False for z0 (k = 0) and True for s (z + k, trainonly included).

Modes (``KConfig.mode``; the effective configuration sets what a mode ignores to None):
    k          the module
    off        gate (k0): k is the constant 0 (no g forward; g frozen, requires_grad False): the
               outputs, the loss and the trunk's updates are urdmu.URDMU's bit for bit
    mlp        C1: k_t = g(x_t) (no context)
    tauinf     C2: tau None (uniform normality weights), rho kept
    mean       C2b: rho and tau None (the own video's mean over the other runs)
    shuffle    C3: the distance term on permuted positions |pi_T(t) - pi_T(j)|, pi_T =
               np.random.default_rng([seed, 1, T]).permutation(T) (the global state untouched)
    trainonly  C6: ``frame`` = sigma(z), the loss AD_Loss - 1/2 BCE_video(sigma(z)) + 1/2
               BCE_video(sigma(z + k)) (AD_Loss's video term: the mean of the top T // 16 + 1),
               which is U's loss while k = 0; the scores are z
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import skip_init

from wvad.baselines import common, urdmu

NAME = "urdmu_k"
RECIPE = urdmu.RECIPE
MODES = ("k", "off", "mlp", "tauinf", "mean", "shuffle", "trainonly")
CONTEXT_MODES = ("k", "tauinf", "mean", "shuffle", "trainonly")
IGNORED = {"off": ("rho", "tau"), "mlp": ("rho", "tau"), "mean": ("rho", "tau"),
           "tauinf": ("tau",)}                  # what a mode ignores: None in its KConfig
EMBED_DIM, HIDDEN = 512, 128
G_PARAMETERS = EMBED_DIM * HIDDEN + HIDDEN + HIDDEN + 1         # 65,793 (PLAN.md 관문 (e))
G_SEED_OFFSET = 1_000_000
STD_EPS = 1e-6


@dataclass(frozen=True)
class KConfig:
    rho: float | None       # positional kernel width as a share of T; None: global (no kernel)
    tau: float | None       # normality temperature; None: tau = inf (uniform normality weights)
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise common.MethodError(f"unknown K mode {self.mode!r} (known: {MODES})")
        for name, v in (("rho", self.rho), ("tau", self.tau)):
            if v is not None and not (math.isfinite(float(v)) and float(v) > 0):
                raise common.MethodError(f"K {name} must be None or finite and > 0, got {v!r}")
        given = [k for k in IGNORED.get(self.mode, ()) if getattr(self, k) is not None]
        if given:
            raise common.MethodError(f"K mode {self.mode} ignores {given}: None in its KConfig")
        if self.mode in ("k", "shuffle", "trainonly") and self.tau is None:
            raise common.MethodError(f"K mode {self.mode} needs a finite tau")
        if self.mode == "shuffle" and self.rho is None:
            raise common.MethodError("K mode shuffle needs a local rho (C3 runs only then)")

    @classmethod
    def of(cls, mode: str, rho: float | None, tau: float | None) -> KConfig:
        """The effective configuration of ``mode`` run with the arguments (rho, tau) (a control
        gets the chosen rho*, tau*): what the mode ignores is None (PLAN.md "폴더 이름")."""
        ignored = IGNORED.get(mode, ())
        rho = None if rho is None or "rho" in ignored else float(rho)
        tau = None if tau is None or "tau" in ignored else float(tau)
        return cls(rho, tau, mode)


# --- the module -----------------------------------------------------------------------------

def make_g(seed: int) -> nn.Sequential:
    """g with its own generator: nn.Linear's default init of the first layer (kaiming_uniform_
    a = sqrt(5), bias U(-1/sqrt(512), 1/sqrt(512))), the last layer zero; skip_init builds the
    layers without drawing from the global generator."""
    gen = torch.Generator().manual_seed(G_SEED_OFFSET + int(seed))
    first = skip_init(nn.Linear, EMBED_DIM, HIDDEN)
    last = skip_init(nn.Linear, HIDDEN, 1)
    nn.init.kaiming_uniform_(first.weight, a=math.sqrt(5), generator=gen)
    bound = 1 / math.sqrt(EMBED_DIM)
    nn.init.uniform_(first.bias, -bound, bound, generator=gen)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(first, nn.ReLU(), last)


def g_parameters() -> int:
    """The parameter count of g (k_config.json, gate (e))."""
    return int(sum(p.numel() for p in make_g(0).parameters()))


def run_ids(x: torch.Tensor) -> torch.Tensor:
    """(B, T) int64 run numbers of (B, T, D) input rows: +1 where row t differs from row t - 1 in
    any element."""
    change = (x[:, 1:] != x[:, :-1]).any(dim=-1)
    return torch.cat([change.new_zeros((x.shape[0], 1)), change], dim=1).long().cumsum(dim=1)


def positions(t: int, kcfg: KConfig, seed: int, like: torch.Tensor) -> torch.Tensor:
    """(T,) positions of the distance term: 0..T-1, or pi_T for shuffle (C3)."""
    if kcfg.mode == "shuffle":
        p = np.random.default_rng([int(seed), 1, int(t)]).permutation(int(t))
        return torch.as_tensor(p, dtype=like.dtype, device=like.device)
    return torch.arange(int(t), dtype=like.dtype, device=like.device)


def context_weights(z: torch.Tensor, runs: torch.Tensor, kcfg: KConfig,
                    pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T, T) weights w_tj (rows sum to 1; 0 on t's own run; all 0 in a one-run sequence) and
    (B, T, 1) bool rows with a context, from (B, T, 1) logits z and (B, T) run numbers; log space,
    no gradient (module docstring)."""
    with torch.no_grad():
        zz = z.reshape(z.shape[0], -1)
        t = zz.shape[1]
        ell = zz.new_zeros((zz.shape[0], t, t))
        if kcfg.tau is not None:
            zt = (zz - zz.mean(dim=-1, keepdim=True)) / (
                zz.std(dim=-1, correction=0, keepdim=True) + STD_EPS)
            ell = ell - (zt / kcfg.tau)[:, None, :]
        if kcfg.rho is not None:
            ell = ell - (pos[:, None] - pos[None, :]).abs() / (kcfg.rho * t)
        same = runs[:, :, None] == runs[:, None, :]
        ok = ~same.all(dim=-1, keepdim=True)
        ell = ell.masked_fill(same, float("-inf")).masked_fill(~ok, 0.0)
        return torch.softmax(ell, dim=-1).masked_fill(~ok, 0.0), ok


class WSAD_K(urdmu.WSAD):  # noqa: N801 - UR-DMU's WSAD + K
    def __init__(self, input_size, flag, a_nums, n_nums, *, kcfg: KConfig, seed: int,
                 input_ln=False, distance_margin=urdmu.DISTANCE_MARGIN):
        super().__init__(input_size, flag, a_nums, n_nums, input_ln=input_ln,
                         distance_margin=distance_margin)
        self.kcfg, self.seed = kcfg, int(seed)
        self.k_in_logit = kcfg.mode != "trainonly"      # the test branch's logit: z + k or z
        self.g = make_g(seed)
        if kcfg.mode == "off":
            self.g.requires_grad_(False)

    def contrast(self, h: torch.Tensor, z: torch.Tensor, runs: torch.Tensor | None
                 ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """k (B, T, 1) and ||x_t - c_t|| (B, T; NaN without context) from the 512-d embedding h
        and the logits z; (None, None) in mode off, no norm for mlp."""
        mode = self.kcfg.mode
        if mode == "off":
            return None, None
        if mode == "mlp":
            return self.g(h), None
        w, ok = context_weights(z, runs, self.kcfg, positions(h.shape[1], self.kcfg, self.seed, z))
        d = h - torch.matmul(w, h)
        norm = torch.linalg.vector_norm(d.detach(), dim=-1).masked_fill(~ok[..., 0], math.nan)
        return self.g(d).masked_fill(~ok, 0.0), norm

    def forward(self, x):
        k_runs = (run_ids(x.reshape(-1, x.shape[-2], x.shape[-1]))
                  if self.kcfg.mode in CONTEXT_MODES else None)
        x = self.input_norm(x)
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
        h = x                                   # x_t of K: 512-d, before the concatenation
        if self.flag == "Train":
            N_x = x[:b * n // 2]                  # normal part
            A_x = x[b * n // 2:]                  # abnormal part
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)

            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            _, A_index = torch.topk(A_att, t // 16 + 1, dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand(
                [-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            _, N_index = torch.topk(N_att, t // 16 + 1, dim=-1)
            anchor_nx = torch.gather(N_x, 1, N_index.unsqueeze(2).expand(
                [-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            _, P_index = torch.topk(N_Aatt, t // 16 + 1, dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand(
                [-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            triplet_margin_loss = self.triplet(urdmu.norm(anchor_nx), urdmu.norm(positivte_nx),
                                               urdmu.norm(negative_ax))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)

            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand(
                [-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            A_aug_new = self.encoder_mu(A_aug)
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand(
                [-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            distance = torch.relu(self.distance_margin
                                  - torch.norm(negative_ax_new, p=2, dim=-1)
                                  + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0))),
                          dim=-1)
            z = self.cls_head.logits(x)
            k, xc = self.contrast(h, z, k_runs)
            s = z if k is None or self.kcfg.mode == "trainonly" else z + k
            pre_att = self.cls_head.mlp[3](s).reshape((b, n, -1)).mean(1)

            out = {
                "frame": pre_att,
                'triplet_margin': triplet_margin_loss,
                'kl_loss': kl_loss,
                'distance': distance,
                'A_att': A_att.reshape((b // 2, n, -1)).mean(1),
                "N_att": N_att.reshape((b // 2, n, -1)).mean(1),
                "A_Natt": A_Natt.reshape((b // 2, n, -1)).mean(1),
                "N_Aatt": N_Aatt.reshape((b // 2, n, -1)).mean(1)
            }
            if self.kcfg.mode == "trainonly":
                out["frame_k"] = self.cls_head.mlp[3](z + k).reshape((b, n, -1)).mean(1)
        else:
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)

            z = self.cls_head.logits(x)
            k, xc = self.contrast(h, z, k_runs)
            s = z if k is None or not self.k_in_logit else z + k
            pre_att = self.cls_head.mlp[3](s).reshape((b, n, -1)).mean(1)
            out = {"frame": pre_att, "logit": s.reshape((b, n, -1)).mean(1)}
        if xc is not None:
            out["xc_norm"] = xc.reshape((b, n, -1)).mean(1)
        return out


def video_bce(frame: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AD_Loss's video term on (B, T) frame scores: BCE of the mean of the top T // 16 + 1."""
    t = frame.size(1)
    return F.binary_cross_entropy(torch.topk(frame, t // 16 + 1, dim=-1)[0].mean(-1),
                                  labels.float())


# --- the E024 method ------------------------------------------------------------------------

class URDMUK(urdmu.URDMU):
    """UR-DMU + K (module docstring); ``sequence_logits`` (inherited) returns the test branch's
    ``logit``: s, or z in modes off and trainonly."""

    name = NAME
    official_unit = urdmu.OFFICIAL_UNIT

    def __init__(self, kcfg: KConfig, seed: int) -> None:
        super().__init__()
        self.kcfg, self.seed = kcfg, int(seed)

    def build(self, feature_dim: int, *, unit: int, input_ln: bool,
              scale: Mapping[str, float]) -> nn.Module:
        self.recipe.unit_input(unit)
        if "distance_margin" not in scale:
            raise common.MethodError("urdmu_k needs the scale constant 'distance_margin'")
        return WSAD_K(int(feature_dim), flag="Train", a_nums=urdmu.A_NUMS, n_nums=urdmu.N_NUMS,
                      kcfg=self.kcfg, seed=self.seed, input_ln=bool(input_ln),
                      distance_margin=float(scale["distance_margin"]))

    def loss(self, model: nn.Module, batch: common.Batch, *, unit: int,
             scale: Mapping[str, float]) -> common.LossOutput:
        """URDMU.loss; trainonly: AD_Loss - 1/2 att_loss + 1/2 BCE_video(sigma(z + k)) (part
        "att_loss_k", the latter term's BCE); a mode with a context reports the step's median
        ||x_t - c_t|| over the positions with a context (part "xc_norm", a detached CPU copy)
        and "xc_norm_steps" = 1. A step without any context position emits neither (no NaN part:
        the loop counts a non-finite part as a non-finite loss). history.json divides each
        part's interval sum by every step of the interval, so xc_norm / xc_norm_steps there is
        the mean over the steps that emitted it."""
        b, t = int(batch.features.shape[0]), int(batch.features.shape[1])
        if t != int(unit):
            raise common.MethodError(f"urdmu_k: {t} training rows for unit {unit}")
        if b % 2 or batch.n_normal * 2 != b:
            raise common.MethodError(f"urdmu_k needs normal-first halves, got {batch.n_normal} "
                                     f"normal of {b}")
        if getattr(model, "flag", None) != "Train" or not model.training:
            raise common.MethodError("urdmu_k: call set_mode(model, True) before the loss")
        result = model(batch.features)
        cost, parts = self.criterion(result, batch.labels)
        out = {k: parts[k] for k in urdmu.LOSS_KEYS}
        out["distance"] = result["distance"]
        if "frame_k" in result:
            att_k = video_bce(result["frame_k"], batch.labels)
            cost = cost - 0.5 * parts["att_loss"] + 0.5 * att_k
            out["att_loss_k"] = att_k
        if "xc_norm" in result:
            v = result["xc_norm"].detach().float().cpu().numpy().ravel()
            v = v[np.isfinite(v)]
            if v.size:
                out["xc_norm"], out["xc_norm_steps"] = float(np.median(v)), 1.0
        return common.LossOutput(cost, out)

    def make_optimizer(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        """The recipe's Adam on the trunk (``model.parameters()`` order without g), g a second
        group with the same lr, betas and weight decay."""
        o = self.recipe.optimizer
        if o.kind != "adam" or o.eps is not None:
            raise common.MethodError(f"urdmu_k expects UR-DMU's Adam, got {o}")
        g_ids = {id(p) for p in model.g.parameters()}
        trunk = [p for p in model.parameters() if id(p) not in g_ids]
        opt = torch.optim.Adam(trunk, lr=float(lr), weight_decay=float(o.weight_decay),
                               betas=tuple(float(b) for b in o.betas))
        opt.add_param_group({"params": list(model.g.parameters())})
        return opt
