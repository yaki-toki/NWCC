"""UR-DMU for E023 (Zhou, Yu and Yang, "Dual Memory Units with Uncertainty Regulation for Weakly
Supervised Video Anomaly Detection", AAAI 2023; PLAN.md "팔" URDMU and X.32 URDMU.32). New code
(add-only rule).

Official repository: https://github.com/henrryzh1/UR-DMU
Commit: 40cfdf5f8bebbc3b59373935f9d6ca00e2f32bbc (2025-01-06)
Licence: MIT (LICENSE sha256 c658ad8c9781a8e1c7dd0e149a0ecf886ca888e7322cf302d1ce8a0c6aceab99;
results/e023_licences.json key "urdmu")
Porting mode: copied with attribution (permissive licence). Copied from the official files, in
their code order and with their names: memory.py ``Memory_Unit`` (``reset_parameters``,
``forward``); translayer.py ``PreNorm``, ``FeedForward``, ``Attention``, ``Transformer``;
model.py ``Temporal``, ``ADCLS_head``, ``WSAD`` (``_reparameterize``, ``latent_loss``,
``forward``); train.py ``norm`` (= utils.py ``norm``) and ``AD_Loss``; the step of train.py
``train`` (forward, loss, zero_grad, backward, step) is the shared loop's default train_step.
Recipe values from ucf_main.py / xd_main.py, options.py, config.py, dataset_loader.py, utils.py,
ucf_test.py / xd_test.py (``RECIPE.sources``).

    MIT License

    Copyright (c) 2023 ZH

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Official recipe (both datasets; ucf_main.py and xd_main.py differ only in the dataset class, the
test batch and the test metric): WSAD(input_size, flag="Train", a_nums=60, n_nums=60);
Adam(lr 1e-4, betas (0.9, 0.999), weight_decay 5e-5); lr list "[0.0001]*3000" = 3000 steps at
a constant lr; per step one batch of 64 normal and one of 64 anomalous training items (two
shuffled drop_last DataLoaders, each re-created when exhausted; torch.cat((normal, abnormal)));
200 training windows per item (UCF ``np.linspace(0, n, 201, dtype=np.int)``, XD
``utils.random_perturb`` = ``np.linspace(0, n, 201, dtype=np.uint16)``: the same deterministic
boundaries for n <= 65535; above it the official uint16 boundaries wrap, the port keeps int64
(check_urdmu (q1c) lists such training lengths and records the largest XD training n_v);
window i = mean of rows r[i]..r[i+1]-1, an empty window takes row r[i]); loss AD_Loss = att
BCE + 0.1 (A_loss + N_Aatt + N_loss + A_Nloss) + 0.1 triplet + 0.001 kl + 0.0001 distance
(distance margin 100, triplet margin 1); net.flag "Train" / "Test" switches the training and
the test forward; the test feeds the whole video and uses the sigmoid output ``frame``. The
training input of the 200 unit is the derived cache ``d16m200``
(wvad.extract.derive_n kind m: these boundaries, scale-matched on the training part), sampled
by the shared "window_mean" sampler (the same DataLoaders, per-pool reset, normal first).

X.32 (URDMU.32, PLAN.md D11): the same network, loss, optimizer and schedule on the E021 ``d16m``
32-window cache (top-k sizes t // 16 + 1 follow T as in the official code); test "whole".

Adaptations: (1) input dimension: input_size = D (1792 XCB+XCL, 1920 XCB+SigLIP) instead of
``config.len_feature = 1024`` (only Temporal's Conv1d in_channels changes); (2) training length
T = 200 (official) or 32 (X.32 flag arm, E021 d16m); (3) input LayerNorm ``input_norm`` =
common.input_layer(D, True) applied first for XCB+SigLIP only (nn.Identity, no state,
otherwise); (4) single crop: the official lists hold every crop (UCF 10, XD 5) as a separate
training item and the official test averages the crops; here one item per video, no crop mean
(WSAD's 4-D crop path is kept); (5) device: the official ``.cuda()`` calls (Attention.forward
tmp_ones / tmp_n, AD_Loss targets) move to the input's device instead (tmp_ones / tmp_n are
still made on the CPU first, then moved); (6) einops.rearrange in Attention.forward is replaced
by the reshape / permute sequence einops' torch backend executes for those two patterns (no
einops dependency); ``import ipdb`` (unused) dropped; (7) the distance margin 100 (model.py L97)
is the constructor argument ``distance_margin`` = the recipe's scale constant (x --scale_mult in
the D4 check; x1 = 100); (8) the Test branch also returns the pre-sigmoid snippet logits
("logit": the input of cls_head's final nn.Sigmoid, ``frame`` unchanged) and sequence_logits
returns them; (9) AD_Loss's terms go to the shared loop (``parts``; plus "distance", which the
official dict omits) instead of visdom; (10) training inputs: the d16m200 cache (training-part
scale match; the official float32 window means of the scale-matched snippets, bit for bit;
official: of raw I3D crops); (11) checkpoint
selection on validation by the shared loop (30 points) instead of the official test-set AUC / AP
every 10 steps after step 10; no test pass at all: every official test pass (step 0 and every
10 steps after step 10) creates a DataLoader iterator that draws from the CPU generator, and the
port and the (q1b) reference have no test passes, so the batch orders equal the official run's
only with the test passes removed; (12) test pieces: the whole video, cut only by the plan's
N_chunk rule (score_video, n_v > N_chunk); the cut limits UR-DMU's global attention (attn1
of the GL-MHSA ``Attention``) and its convolution borders to each piece, so it is not neutral
across methods (per-snippet methods such as MIL are unaffected); the cut videos and the last
pieces under 16 snippets per dataset are counted from the recorded total_frames in gate (q6)
(``e023_gates.piece_counts``); (13) cudnn: the official utils.set_seed sets
cudnn.deterministic = True and benchmark = False; here the shared loop sets cudnn.deterministic
only in --deterministic mode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from wvad.baselines import common
from wvad.baselines.recipes import (
    InputRow,
    OptimizerSpec,
    Recipe,
    ScheduleSpec,
    Source,
    UnitInput,
)

NAME = "urdmu"
OFFICIAL_UNIT = 200
A_NUMS = 60                                   # ucf_main.py L29 / xd_main.py L28
N_NUMS = 60
DISTANCE_MARGIN = 100.0                       # model.py L97
LOSS_KEYS = ("att_loss", "N_Aatt", "A_loss", "N_loss", "A_Nloss", "triplet", "kl_loss")


def _src(file: str, lines: str, note: str = "") -> Source:
    return Source("urdmu", file, lines, note)


RECIPE = Recipe(
    method=NAME,
    official_unit=OFFICIAL_UNIT,
    units={OFFICIAL_UNIT: UnitInput("d16m200", "whole"), 32: UnitInput("d16m", "whole")},
    optimizer=OptimizerSpec("adam", 1e-4, 5e-5, betas=(0.9, 0.999)),
    schedule=ScheduleSpec("constant", "step"),
    iterations=3000,
    iteration_unit="step",
    sampler="window_mean",
    sampler_params={"batch_normal": 64, "batch_abnormal": 64, "reset": "per_pool",
                    "order": "normal_first", "eager": False},
    loss_weights={"att_loss": 1.0, "A_loss": 0.1, "N_Aatt": 0.1, "N_loss": 0.1,
                  "A_Nloss": 0.1, "triplet": 0.1, "kl_loss": 0.001, "distance": 0.0001},
    scale_constants={"distance_margin": DISTANCE_MARGIN},
    precision="fp32",
    input_table=(
        InputRow("features", "I3D 1024-d, UCF 10-crop / XD 5-crop, every crop a separate "
                 "training item; the test averages the crops",
                 "frozen VLM dense snippets, single crop, 1792 / 1920-d; one item per video, "
                 "no crop mean", "adapt", "ucf_main.py L28; list/UCF_Train.list; ucf_test.py "
                 "L37-L40; xd_test.py L32"),
        InputRow("clip_normalisation", "none (I3D features as extracted)", "none", "keep",
                 "dataset_loader.py L46, L113"),
        InputRow("sampler_and_length", "200 window means (UCF np.linspace int, XD "
                 "random_perturb uint16: the same boundaries), paired loaders 64 + 64, "
                 "per-pool re-creation, normal first",
                 "d16m200 (derive_n kind m) through the window_mean sampler; X.32: E021 d16m",
                 "keep", "ucf_main.py L32-L41, L67-L72; dataset_loader.py L53-L60, L115-L124; "
                 "utils.py L42-L44; train.py L58-L61"),
        InputRow("scale_match", "none", "training-part scale match of the window means (and "
                 "the same map on validation / test snippets)", "adapt", ""),
        InputRow("input_normalisation", "none", "input LayerNorm for XCB+SigLIP only (C rule, "
                 "diagnostic P7)", "adapt", ""),
        InputRow("scale_constants", "distance hinge margin 100 on the norms of the "
                 "memory-augmented latent features", "official 100, D4 check x{1/10, 1, 10}",
                 "adapt", "model.py L97"),
        InputRow("text_branch", "none", "none", "keep", ""),
        InputRow("top_k", "k = t // 16 + 1 (13 at T = 200) for the video score, the memory "
                 "and the triplet selections; memory read k = nums // 16 + 1 = 4",
                 "official formula at the training length (X.32: 3)", "keep",
                 "train.py L29-L35; model.py L72-L79; memory.py L24"),
        InputRow("checkpoint_selection", "test-set AUC (UCF) / AP (XD) every 10 steps after "
                 "step 10", "validation V-AUC at 30 evaluation points (sigma W readout)",
                 "adapt", "ucf_main.py L73-L84; xd_main.py L72-L83"),
        InputRow("precision", "fp32", "fp32", "keep", "ucf_main.py (no autocast)"),
    ),
    sources={
        "optimizer": _src("ucf_main.py", "L54-L55", "xd_main.py L53-L54; Adam, betas "
                          "(0.9, 0.999)"),
        "lr": _src("options.py", "L13", "default '[0.0001]*3000'; config.py L6 eval"),
        "weight_decay": _src("ucf_main.py", "L55", "weight_decay = 0.00005; xd_main.py L54"),
        "schedule": _src("ucf_main.py", "L64-L66", "the lr of step s is config.lr[s - 1], "
                         "constant for the default list; xd_main.py L63-L65"),
        "iterations": _src("config.py", "L7", "num_iters = len(lr) = 3000"),
        "batch": _src("ucf_main.py", "L32-L41", "batch_size 64 per class, drop_last; "
                      "train.py L58-L61 torch.cat((normal, abnormal)); xd_main.py L31-L40"),
        "sampler": _src("ucf_main.py", "L67-L72", "each loader re-created at (step - 1) % "
                        "len == 0; UCF windows dataset_loader.py L53-L60; XD windows "
                        "dataset_loader.py L115-L124 + utils.py L42-L44"),
        "loss_weights": _src("train.py", "L41", "cost = anomaly + 0.1 (A + panomaly + N + "
                             "A_N) + 0.1 triplet + 0.001 kl + 0.0001 distance; triplet margin "
                             "1 at model.py L37"),
        "unit": _src("ucf_main.py", "L33, L38", "num_segments = 200; xd_main.py L32, L37"),
        "test_sequence": _src("ucf_test.py", "L35-L47", "the whole video (Test mode keeps every "
                              "snippet, dataset_loader.py L52, L115); xd_test.py L30-L38"),
        "scale_constants": _src("model.py", "L97", "relu(100 - ||negative|| + ||anchor||)"),
    },
    notes=("UR-DMU copied with attribution (MIT); adaptations in the module docstring. The UCF "
           "and XD official scripts share the model, AD_Loss, optimizer, schedule, batch and "
           "unit; they differ in the window code (same boundaries), the test crops and the test "
           "metric."),
)


# --- copied from the official code (MIT, attribution above) ---------------------------------

def norm(data):
    """train.py / utils.py ``norm``."""
    l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
    return torch.div(data, l2)


class Memory_Unit(Module):  # noqa: N801 - official name (memory.py)
    def __init__(self, nums, dim):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.sig = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)
        if self.memory_block is not None:
            self.memory_block.data.uniform_(-stdv, stdv)

    def forward(self, data):  # data B,T,D; memory K,D
        attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block)
                             / (self.dim**0.5))
        temporal_att = torch.topk(attention, self.nums // 16 + 1, dim=-1)[0].mean(-1)
        augment = torch.einsum('btk,kd->btd', attention, self.memory_block)
        return temporal_att, augment


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def _split_heads(t, heads):
    """einops rearrange(t, 'b n (h d) -> b h n d', h=heads): reshape, then permute."""
    return t.reshape(t.shape[0], t.shape[1], heads, t.shape[2] // heads).permute(0, 2, 1, 3)


def _merge_heads(t):
    """einops rearrange(t, 'b h n d -> b n (h d)'): permute, then reshape."""
    b, h, n, d = t.shape
    return t.permute(0, 2, 1, 3).reshape(b, n, h * d)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(2 * inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d = x.size()
        qkvt = self.to_qkv(x).chunk(4, dim=-1)
        q, k, v, t = (_split_heads(p, self.heads) for p in qkvt)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots)

        tmp_ones = torch.ones(n).to(x.device)
        tmp_n = torch.linspace(1, n, n).to(x.device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1, 1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b, self.heads, 1, 1)

        out = torch.cat([torch.matmul(attn1, v), torch.matmul(attn2, t)], dim=-1)
        out = _merge_heads(out)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Temporal(Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class ADCLS_head(Module):  # noqa: N801 - official name (model.py)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim),
                                 nn.Sigmoid())

    def forward(self, x):
        return self.mlp(x)

    def logits(self, x):
        """Adaptation (8): the input of the final nn.Sigmoid (mlp[0..2] in order)."""
        return self.mlp[2](self.mlp[1](self.mlp[0](x)))


class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums, *, input_ln=False,
                 distance_margin=DISTANCE_MARGIN):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums
        self.distance_margin = float(distance_margin)              # adaptation (7)
        self.input_norm = common.input_layer(input_size, input_ln)  # adaptation (3)

        self.embedding = Temporal(input_size, 512)
        self.triplet = nn.TripletMarginLoss(margin=1)
        self.cls_head = ADCLS_head(1024, 1)
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5)
        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))
        return kl_loss

    def forward(self, x):
        x = self.input_norm(x)
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
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

            triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positivte_nx),
                                               norm(negative_ax))

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
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            return {
                "frame": pre_att,
                'triplet_margin': triplet_margin_loss,
                'kl_loss': kl_loss,
                'distance': distance,
                'A_att': A_att.reshape((b // 2, n, -1)).mean(1),
                "N_att": N_att.reshape((b // 2, n, -1)).mean(1),
                "A_Natt": A_Natt.reshape((b // 2, n, -1)).mean(1),
                "N_Aatt": N_Aatt.reshape((b // 2, n, -1)).mean(1)
            }
        else:
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)

            z = self.cls_head.logits(x)                              # adaptation (8)
            pre_att = self.cls_head.mlp[3](z).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att, "logit": z.reshape((b, n, -1)).mean(1)}


class AD_Loss(nn.Module):  # noqa: N801 - official name (train.py)
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        loss = {}

        _label = _label.float()

        triplet = result["triplet_margin"]
        att = result['frame']
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]
        b = _label.size(0) // 2
        t = att.size(1)
        anomaly = torch.topk(att, t // 16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt, t // 16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones(b).to(att.device))

        A_att = torch.topk(A_att, t // 16 + 1, dim=-1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones(b).to(att.device))

        N_loss = self.bce(N_att, torch.ones_like(N_att).to(att.device))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt).to(att.device))

        cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) \
            + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        loss['kl_loss'] = kl_loss
        return cost, loss


# --- the E023 method ------------------------------------------------------------------------

class URDMU(common.BaselineMethod):
    """UR-DMU through the E023 porting contract (module docstring)."""

    name = NAME
    official_unit = OFFICIAL_UNIT

    def __init__(self) -> None:
        self.criterion = AD_Loss()

    def build(self, feature_dim: int, *, unit: int, input_ln: bool,
              scale: Mapping[str, float]) -> nn.Module:
        self.recipe.unit_input(unit)
        if "distance_margin" not in scale:
            raise common.MethodError("urdmu needs the scale constant 'distance_margin'")
        return WSAD(int(feature_dim), flag="Train", a_nums=A_NUMS, n_nums=N_NUMS,
                    input_ln=bool(input_ln), distance_margin=float(scale["distance_margin"]))

    def set_mode(self, model: nn.Module, train: bool) -> None:
        """train.py ``train``: net.train(); net.flag = "Train"; ucf_test.py / xd_test.py
        ``test``: net.eval(); net.flag = "Test"."""
        model.train(train)
        model.flag = "Train" if train else "Test"

    def loss(self, model: nn.Module, batch: common.Batch, *, unit: int,
             scale: Mapping[str, float]) -> common.LossOutput:
        b, t = int(batch.features.shape[0]), int(batch.features.shape[1])
        if t != int(unit):
            raise common.MethodError(f"urdmu: {t} training rows for unit {unit}")
        if b % 2 or batch.n_normal * 2 != b:
            raise common.MethodError(f"urdmu needs normal-first halves, got {batch.n_normal} "
                                     f"normal of {b}")
        if getattr(model, "flag", None) != "Train" or not model.training:
            raise common.MethodError("urdmu: call set_mode(model, True) before the loss")
        result = model(batch.features)
        cost, parts = self.criterion(result, batch.labels)
        out = {k: parts[k] for k in LOSS_KEYS}
        out["distance"] = result["distance"]
        return common.LossOutput(cost, out)

    def sequence_logits(self, model: nn.Module, x: torch.Tensor, *, unit: int) -> torch.Tensor:
        if getattr(model, "flag", None) != "Test" or model.training:
            raise common.MethodError("urdmu: call set_mode(model, False) before scoring")
        return model(x[None])["logit"].reshape(-1)


METHOD = URDMU()
