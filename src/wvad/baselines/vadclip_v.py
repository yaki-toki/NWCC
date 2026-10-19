"""VCV: the visual branch of VadCLIP (Wu et al., "VadCLIP: Adapting Vision-Language Models for
Weakly Supervised Video Anomaly Detection", AAAI 2024) as an E023 comparison method (PLAN.md
"팔" VCV and VCV.32, "비교 기법 선정 기준과 이유", "입력 처리 표"). New code (add-only rule).

Official code: https://github.com/nwpu-zxr/VadCLIP at commit
c41067f07d252efcda18008bea367886070c33b0, licence Apache-2.0 (LICENSE sha256 1eb85fc97224598d...,
results/e023_licences.json key "vcv"). Porting mode: copy with attribution (permissive licence).
Portions of this file are copied from that repository, Copyright the VadCLIP authors (P. Wu,
X. Zhou, G. Pang, L. Zhou, Q. Yan, P. Wang, Y. Zhang), licensed under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0); they were changed as listed under
"Adaptations". Copied (official names kept):
    src/model.py         LayerNorm, QuickGELU, ResidualAttentionBlock, Transformer (L10-L57);
                         CLIPVAD.__init__ (its visual part, L72-L118), build_attention_mask
                         (L124-L135), adj4 (L137-L159), encode_video (L161-L183) and the C-branch
                         line of forward (L203-L204: logits1 = classifier(v + mlp2(v)))
    src/utils/layers.py  GraphConvolution (L61-L109), DistanceAdj (L165-L179)
    src/ucf_train.py     CLAS2 (L27-L39; src/xd_train.py L27-L39 is the same loss)
    src/utils/tools.py   pad, process_split (L258-L289) with the length loop of src/ucf_test.py
                         test() (L23-L51; src/xd_test.py identical): the test-sequence rule
The training-input rule (process_feat, tools.py L265-L273) is wvad.extract.derive_n variant
d16p256; the per-epoch paired loaders (ucf_train.py L62-L71) are samplers "window_mean"; the XD
loader (xd_train.py L131, L67) is the sampler "vadclip_v_xd" registered here.

Official configuration per dataset (ucf_option.py / xd_option.py defaults; ``OFFICIAL``)
    UCF   visual_layers 2, attn_window 8, visual_head 1, visual_length 256, AdamW lr 2e-5 (torch
          defaults: betas (0.9, 0.999), eps 1e-8, weight decay 0.01), MultiStepLR [4, 8] x0.1
          per epoch, 10 epochs, per epoch two shuffled drop_last loaders of 64 normal + 64
          anomalous rows, min(len) steps, normal first
    XD    visual_layers 1, attn_window 64, the same head / length / optimizer form with lr 1e-5,
          MultiStepLR [3, 6, 10] x0.1 per epoch, 10 epochs, one shuffled loader of 96 rows
          (drop_last False), re-iterated every epoch
    both  the official lists (list/ucf_CLIP_rgb.csv, list/xd_CLIP_rgb.csv) hold 10 CLIP crops per
          video, each crop a separate row (UCF 16100 rows, XD 39540), so an official epoch is one
          pass over 10 rows per video: UCF min(floor(10 N_normal / 64), floor(10 N_anomalous /
          64)) optimizer steps, XD ceil(10 N / 96) (``crops`` = 10, adaptation 6)
``RECIPE`` / ``METHOD`` are the UCF configuration (the plan's arm table names window 8);
``RECIPE_XDV`` / ``METHOD_XDV`` the XD one. ``METHOD.for_dataset(dataset)`` (the BaselineMethod
hook, ``common.dataset_method``) returns ``METHODS[dataset]``: the training chain, the D4 /
validation / test scoring, the (3b) record and the gates resolve VCV per dataset, so the XD arm
runs the official XD configuration.

Adaptations: (every change against the official code)
    1. input dimension: visual_width = D (official 512, the CLIP ViT-B/16 image embedding), so
       the temporal transformer, GCN (width int(D / 2)), linear, mlp2, classifier and the frame
       position embeddings are built at D; embed_dim (the removed text branch's width) stays 512
    2. text branch removed (PLAN.md VCV, VadCLIP Table 6 "C-branch only"): no CLIP text encoder,
       no learnable prompts (text_prompt_embeddings), no mlp1 visual-prompt fusion, no logits2,
       no CLASM loss2, no loss3; the training loss is CLAS2 alone (weight 1) and the score is the
       C-branch sigmoid (official prob1)
    3. the removed modules' random draws are replayed in the official construction order (mlp1:
       two nn.Linear; clip.load("ViT-B/16"): the CLIP ViT-B/16 construction and its
       initialize_parameters; text_prompt_embeddings: nn.Embedding(77, 512) and its normal_ of
       std 0.01), so the global torch generator is consumed exactly as by the official
       constructor and every kept parameter equals the official CLIPVAD's from the same seed
       (gate (q1)); the replayed tensors are discarded (no weight file is read)
    4. DistanceAdj on the input's device (official: hard-coded 'cuda' and scipy pdist): the same
       float32 integer distances (numpy) and the same exp(-d / exp(1.)) on that device;
       encode_video's position ids on the input's device (official: self.device)
    5. input LayerNorm (XCB+SigLIP only, PLAN.md "입력 처리 표"): nn.LayerNorm(D) applied first to
       the valid rows (padded rows stay zero, as the official zero padding); identity otherwise
    6. training input: the derived d16p256 cache (the plan's scale match on the snippets, then
       process_feat without random extraction, its float32 windows bit for bit; PLAN.md 기록
       2026-10-20) from single-crop dense snippets (official: 10 CLIP crops per video,
       each a separate list row). The schedule is kept in official optimizer steps: the loaders
       run over every training video repeated ``crops`` = 10 times in a row (the official list
       order), so an epoch is one pass over a 10-row-per-video list (UCF min(floor(10 N_n / 64),
       floor(10 N_a / 64)) steps, XD ceil(10 N / 96); about 10 x the single-crop epoch) and the
       MultiStepLR milestones fall on those epoch boundaries; the 10 rows of a video carry its
       one single-crop input (official: 10 different crops), and a batch may hold one video
       twice, as an official batch may hold two crops of one video
    7. VCV.32 (X.32 flag arm): E021 d16m (32 rows, length 32), visual_length 32 and test pieces
       of 32 snippets; a window longer than the sequence (XD's 64 at 32: the official mask would
       be all -inf, NaN) is cut to the sequence length (one window, full attention)
    8. test sequence: the official process_split pieces (last one zero-padded, plus the official
       all-padding piece when n is a multiple of L) and length loop, applied to each N_chunk piece
       that common.score_video hands over; logits are returned (the official prob1 is their fp32
       sigmoid)
    9. labels: the 0 / 1 video labels (official: 1 - the 'normal' column of get_batch_label's
       one-hot, the same values)
   10. no test-set selection inside training: the official evaluation on the test list and the
       reload of the best test checkpoint at every epoch end (ucf_train.py L98-L120, xd_train.py
       L99-L114) are not ported; selection is on validation after training (PLAN.md 체크포인트
       선택). This changes the training trajectory: the official run restarts every epoch from
       the best test checkpoint so far, here training continues from the current weights (the
       reload needs the test set; disclosed in the (3b) record)
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import FloatTensor, nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset

from wvad.baselines import samplers
from wvad.baselines.common import BaselineMethod, Batch, LossOutput, MethodError, input_layer
from wvad.baselines.recipes import (
    InputRow,
    OptimizerSpec,
    Recipe,
    ScheduleSpec,
    Source,
    UnitInput,
)

NAME = "vadclip_v"
REPO = "vcv"                                  # results/e023_licences.json key
OFFICIAL_UNIT = 256
X32_UNIT = 32
CLASSES = {"ucf": 14, "xdv": 7}               # ucf_option.py L14 / xd_option.py L14


# --- official configuration -------------------------------------------------------------------

@dataclass(frozen=True)
class VCVConfig:
    """One dataset's official VadCLIP configuration (option defaults, training script)."""

    dataset: str                  # "ucf" | "xdv"
    train_script: str             # src/ucf_train.py | src/xd_train.py
    option_file: str              # src/ucf_option.py | src/xd_option.py
    test_script: str              # src/ucf_test.py | src/xd_test.py
    visual_layers: int
    attn_window: int
    lr: float
    milestones: tuple[int, ...]
    batch: int                    # UCF: videos per class and step; XD: videos per step
    sampler: str                  # "window_mean" (UCF) | "vadclip_v_xd" (XD)
    visual_head: int = 1
    visual_length: int = OFFICIAL_UNIT
    embed_dim: int = 512
    gamma: float = 0.1
    epochs: int = 10
    crops: int = 10               # crop rows per video in the official lists (adaptation 6)
    weight_decay: float = 0.01    # torch.optim.AdamW default (the official call passes lr only)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def sampler_params(self) -> dict[str, Any]:
        if self.sampler == "window_mean":
            return {"batch_normal": self.batch, "batch_abnormal": self.batch,
                    "reset": "per_epoch", "order": "normal_first", "crops": self.crops}
        return {"batch": self.batch, "crops": self.crops}

    @property
    def list_file(self) -> str:
        return "list/ucf_CLIP_rgb.csv" if self.dataset == "ucf" else "list/xd_CLIP_rgb.csv"


OFFICIAL = {
    "ucf": VCVConfig("ucf", "src/ucf_train.py", "src/ucf_option.py", "src/ucf_test.py",
                     visual_layers=2, attn_window=8, lr=2e-5, milestones=(4, 8), batch=64,
                     sampler="window_mean"),
    "xdv": VCVConfig("xdv", "src/xd_train.py", "src/xd_option.py", "src/xd_test.py",
                     visual_layers=1, attn_window=64, lr=1e-5, milestones=(3, 6, 10), batch=96,
                     sampler="vadclip_v_xd"),
}


def effective_window(attn_window: int, visual_length: int) -> int:
    """The attention window used (adaptation 7: never longer than the sequence)."""
    return min(int(attn_window), int(visual_length))


# --- copied official building blocks (src/model.py L10-L57) -----------------------------------

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None \
            else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None \
            else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask,
                         attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask)
                                         for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# --- copied official layers (src/utils/layers.py L61-L109, L165-L179) -------------------------

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, residual=True):
        super(GraphConvolution, self).__init__()  # noqa: UP008 - copied verbatim
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not residual:
            self.residual = lambda x: 0
        elif (in_features == out_features):
            self.residual = lambda x: x
        else:
            # self.residual = linear(in_features, out_features)
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                      kernel_size=5, padding=2)

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input, adj):
        # To support batch operations
        support = input.matmul(self.weight)
        output = adj.matmul(support)

        if self.bias is not None:
            output = output + self.bias
        if self.in_features != self.out_features and self.residual:
            input = input.permute(0, 2, 1)
            res = self.residual(input)
            res = res.permute(0, 2, 1)
            output = output + res
        else:
            output = output + self.residual(input)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DistanceAdj(Module):
    """Official DistanceAdj; adaptation 4: computed on ``device`` (official: 'cuda')."""

    def __init__(self):
        super(DistanceAdj, self).__init__()  # noqa: UP008 - copied verbatim
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen, device):
        # official: pdist(arange(max_seqlen).reshape(-1, 1), 'cityblock').astype(np.float32),
        # squareform -> the same float32 integers |i - j|
        arith = np.arange(max_seqlen)
        dist = np.abs(arith[:, None] - arith[None, :]).astype(np.float32)
        dist = torch.from_numpy(dist).to(device)
        dist = torch.exp(-dist / torch.exp(torch.tensor(1.)))
        return torch.unsqueeze(dist, 0).repeat(batch_size, 1, 1).to(device)


# --- random-draw replay of the removed modules (adaptation 3) -------------------------------

def _replay_linear_pair(width: int) -> None:
    """mlp1 (official model.py L99-L103): nn.Linear(w, 4w), nn.Linear(4w, w), discarded."""
    nn.Linear(width, width * 4)
    nn.Linear(width * 4, width)


def _replay_block(width: int, heads: int) -> None:
    """One CLIP ResidualAttentionBlock's construction draws (clip/model.py L172-L183)."""
    nn.MultiheadAttention(width, heads)
    nn.Linear(width, width * 4)
    nn.Linear(width * 4, width)


# CLIP ViT-B/16 as build_model reads it from the official weight file (clip/model.py L402-L431)
CLIP_VIT_B16 = {"embed_dim": 512, "image_resolution": 224, "vision_layers": 12,
                "vision_width": 768, "vision_patch_size": 16, "context_length": 77,
                "vocab_size": 49408, "transformer_width": 512, "transformer_heads": 8,
                "transformer_layers": 12}


def replay_clip_vit_b16(spec: Mapping[str, int] = CLIP_VIT_B16) -> None:
    """The global-generator draws of clip.load("ViT-B/16") (jit False: the archive read and
    load_state_dict draw nothing; build_model -> CLIP.__init__, clip/model.py L243-L327), in
    order: VisionTransformer (conv1, class_embedding, positional_embedding, the blocks, proj),
    the text Transformer's blocks, token_embedding, then initialize_parameters (normal_ of the
    token embedding, the positional embedding, every text block's four weights and the text
    projection). Values are discarded."""
    vw, tw = spec["vision_width"], spec["transformer_width"]
    nn.Conv2d(3, vw, kernel_size=spec["vision_patch_size"], stride=spec["vision_patch_size"],
              bias=False)
    torch.randn(vw)
    torch.randn((spec["image_resolution"] // spec["vision_patch_size"]) ** 2 + 1, vw)
    for _ in range(spec["vision_layers"]):
        _replay_block(vw, vw // 64)
    torch.randn(vw, spec["embed_dim"])
    for _ in range(spec["transformer_layers"]):
        _replay_block(tw, spec["transformer_heads"])
    token = nn.Embedding(spec["vocab_size"], tw)
    nn.init.normal_(token.weight, std=0.02)
    del token
    nn.init.normal_(torch.empty(spec["context_length"], tw), std=0.01)
    for _ in range(spec["transformer_layers"]):
        nn.init.normal_(torch.empty(3 * tw, tw))
        nn.init.normal_(torch.empty(tw, tw))
        nn.init.normal_(torch.empty(4 * tw, tw))
        nn.init.normal_(torch.empty(tw, 4 * tw))
    nn.init.normal_(torch.empty(tw, spec["embed_dim"]))


# --- the network ----------------------------------------------------------------------------

class VadCLIPVisual(nn.Module):
    """CLIPVAD's visual branch (LGT-Adapter + C-branch classifier); module names and
    construction order as the official CLIPVAD, so its state_dict is the official one's visual
    subset (plus ``input_norm`` with the input LayerNorm)."""

    def __init__(self, visual_length: int, visual_width: int, visual_head: int,
                 visual_layers: int, attn_window: int, embed_dim: int = 512,
                 input_ln: bool = False) -> None:
        super().__init__()

        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = effective_window(attn_window, visual_length)
        self.input_norm = input_layer(visual_width, input_ln)          # no random draw

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        _replay_linear_pair(visual_width)                              # mlp1 (text branch)
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        replay_clip_vit_b16()                                          # clip.load("ViT-B/16")

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        text_prompt = nn.Embedding(77, self.embed_dim)                 # text_prompt_embeddings

        # initialize_parameters (official L120-L122)
        nn.init.normal_(text_prompt.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        del text_prompt

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window,
                     i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length,
                     i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=images.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1], x.device)
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def forward(self, visual: torch.Tensor, lengths: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        """logits1 (B, T, 1) of official CLIPVAD.forward L203-L204 (padding_mask is unused by
        the official encode_video). ``valid`` (B, T) bool: rows the input LayerNorm applies to
        (default: rows < lengths); padded rows stay zero (adaptation 5)."""
        if not isinstance(self.input_norm, nn.Identity):
            if valid is None:
                t = torch.arange(visual.shape[1], device=visual.device)
                valid = t[None, :] < torch.as_tensor(lengths, device=visual.device)[:, None]
            visual = torch.where(valid[..., None], self.input_norm(visual), visual)
        visual_features = self.encode_video(visual, None, lengths)
        return self.classifier(visual_features + self.mlp2(visual_features))


# --- copied official loss (src/ucf_train.py L27-L39) ----------------------------------------

def CLAS2(logits, labels, lengths, device):  # noqa: N802 - the official name
    """Official CLAS2 with the binary labels (adaptation 9: the official ``1 - labels[:, 0]`` of
    the one-hot get_batch_label rows, i.e. the same 0 / 1 values)."""
    instance_logits = torch.zeros(0).to(device)
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


# --- copied official test-sequence rule (tools.py L258-L289, ucf_test.py L23-L51) -----------

def pad(feat: torch.Tensor, min_len: int) -> torch.Tensor:
    """tools.pad on a (n, D) tensor: zero rows up to ``min_len`` (unchanged when longer)."""
    clip_length = feat.shape[0]
    if clip_length <= min_len:
        return torch.cat([feat, feat.new_zeros((min_len - clip_length, feat.shape[1]))])
    return feat


def process_split(feat: torch.Tensor, length: int) -> tuple[torch.Tensor, int]:
    """tools.process_split on a (n, D) tensor: (pieces (P, length, D), n). n < length: one
    padded piece; otherwise int(n / length) + 1 pieces, the last one padded (all padding when n
    is a multiple of length, as the official code)."""
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length)[None], clip_length
    split_num = int(clip_length / length) + 1
    pieces = [feat[i * length:i * length + length] for i in range(split_num - 1)]
    pieces.append(pad(feat[(split_num - 1) * length:(split_num - 1) * length + length], length))
    return torch.stack(pieces), clip_length


def piece_lengths(length: int, maxlen: int) -> torch.Tensor:
    """The official test() length loop (ucf_test.py L33-L45), int64 on the CPU."""
    lengths = torch.zeros(int(length / maxlen) + 1)
    for j in range(int(length / maxlen) + 1):
        if j == 0 and length < maxlen:
            lengths[j] = length
        elif j == 0 and length > maxlen:
            lengths[j] = maxlen
            length -= maxlen
        elif length > maxlen:
            lengths[j] = maxlen
            length -= maxlen
        else:
            lengths[j] = length
    return lengths.to(int)


# --- the method -----------------------------------------------------------------------------

def _src(file: str, lines: str, note: str = "") -> Source:
    return Source(REPO, file, lines, note)


def _input_table(cfg: VCVConfig) -> tuple[InputRow, ...]:
    tr = cfg.train_script
    ucf = cfg.dataset == "ucf"
    loader = "L62-L71, L136-L139" if ucf else "L63-L67, L130-L131"
    text = "L15-L25, L78-L93" if ucf else "L15-L25, L74-L89"
    select = "L98-L120" if ucf else "L98-L114"
    return (
        InputRow("features", "CLIP ViT-B/16 frame features (512-d), 10 crops per video, each "
                 "crop a separate training item (list/*_CLIP_rgb.csv)",
                 "frozen VLM dense snippets, single crop, D = 1792 (XCB+XCL) / 1920 "
                 "(XCB+SigLIP); visual_width = D", "adapt",
                 "src/utils/dataset.py L295-L342, src/ucf_option.py L8"),
        InputRow("clip_normalisation", "none (raw CLIP features)", "none", "keep",
                 "src/utils/dataset.py L312-L321"),
        InputRow("sampler_and_length",
                 "UCF: two shuffled drop_last loaders (64 normal + 64 anomalous rows) re-created "
                 "every epoch, min(len) steps; XD: one shuffled loader of 96 rows; each list holds "
                 "10 crop rows per video, so an epoch passes over every video 10 times; "
                 "process_feat to 256 (uniform_extract window mean when longer, zero padding + "
                 "length otherwise)",
                 "the same loaders (samplers window_mean per_epoch / vadclip_v_xd) over every "
                 "training video repeated 10 times in a row (crops 10: the official steps per "
                 "epoch and milestones; each row carries the single-crop input), on the derived "
                 "d16p256 cache (derive_n kind p = process_feat without random extract)",
                 "keep", f"{tr} {loader}, src/utils/tools.py L265-L273, {cfg.list_file}"),
        InputRow("scale_match", "none", "the d16p256 rows scale-matched on the training part "
                 "(derive_n); the same map is the fixed input layer at validation and test",
                 "adapt", "-"),
        InputRow("input_normalisation", "none", "XCB+SigLIP only: nn.LayerNorm(D) on the valid "
                 "rows first (padding stays zero); XCB+XCL: none", "adapt", "-"),
        InputRow("scale_constants", "no feature-norm constant (adj4 cosine threshold 0.7, top-k "
                 "divisor 16, DistanceAdj exp(-|i-j|/e) are norm-free)", "the same", "keep",
                 "src/model.py L137-L159, src/utils/layers.py L165-L179, "
                 f"{tr} L27-L39"),
        InputRow("text_branch", "A branch: CLIP text encoder with learnable prompts, mlp1 "
                 "visual-prompt fusion, logits2 with CLASM (loss2) and loss3",
                 "removed (C branch only: LGT-Adapter + classifier, CLAS2)", "adapt",
                 f"src/model.py L185-L223, {tr} {text}"),
        InputRow("top_k", "CLAS2: mean of the top int(len / 16 + 1) sigmoid scores of the "
                 "valid rows, BCE", "the same at the training length", "keep",
                 f"{tr} L27-L39"),
        InputRow("checkpoint_selection", "the test list during training (UCF AUC every 10 "
                 "iterations, XD AP every epoch) and the best test checkpoint reloaded at "
                 "every epoch end (each epoch restarts from the best test point so far)",
                 "validation V-AUC after training; no reload, so training continues from the "
                 "current weights at every epoch end (a different training trajectory; the "
                 "reload needs the test set)", "adapt", f"{tr} {select}"),
        InputRow("precision", "fp32", "fp32", "keep", "src/model.py L162"),
    )


def make_recipe(cfg: VCVConfig) -> Recipe:
    """The Recipe of one official configuration (``OFFICIAL[dataset]``)."""
    tr, op = cfg.train_script, cfg.option_file
    ucf = cfg.dataset == "ucf"
    sources = {
        "optimizer": _src(tr, "L47" if ucf else "L48", "torch.optim.AdamW(model.parameters(), "
                          "lr=args.lr): betas, eps and weight decay are the torch defaults"),
        "lr": _src(op, "L27"),
        "weight_decay": _src(tr, "L47" if ucf else "L48", "torch.optim.AdamW default 0.01"),
        "schedule": _src(tr, "L48, L113" if ucf else "L49, L98",
                         f"MultiStepLR({list(cfg.milestones)}, {cfg.gamma}) stepped per epoch; "
                         f"{op} L28-L29"),
        "iterations": _src(op, "L16", "max-epoch 10; an epoch is one pass over the 10-crop "
                           "list (crops)"),
        "crops": _src(cfg.list_file, "L2-L16101" if ucf else "L2-L39541",
                      "10 rows per video (<video>__0.npy ... __9.npy), each a separate item"),
        "batch": _src(op, "L20", "UCF: batch-size per class loader, normal first "
                      "(ucf_train.py L73)" if ucf else "XD: one mixed loader (xd_train.py "
                      "L131)"),
        "sampler": _src(tr, "L62-L71, L137-L139" if ucf else "L63-L67, L131",
                        "per-epoch paired loaders" if ucf else "per-epoch single loader, "
                        "shuffle=True, drop_last=False"),
        "loss_weights": _src(tr, "L80-L93" if ucf else "L76-L89",
                             "loss = loss1 + loss2 + loss3; loss2 and loss3 belong to the removed "
                             "text branch"),
        "unit": _src(op, "L7", "visual-length 256; src/utils/tools.py L265-L273 process_feat"),
        "test_sequence": _src(cfg.test_script, "L23-L51",
                              "process_split (src/utils/tools.py L275-L289) pieces of "
                              "visual_length, last one padded, the length loop"),
        "visual_layers": _src(op, "L10"),
        "attn_window": _src(op, "L11"),
        "visual_head": _src(op, "L9"),
        "visual_width": _src(op, "L8", "512 (CLIP); adapted to D"),
        "embed_dim": _src(op, "L6", "the text branch's width (removed)"),
        "model": _src("src/model.py", "L60-L223"),
    }
    return Recipe(
        method=NAME,
        official_unit=OFFICIAL_UNIT,
        units={OFFICIAL_UNIT: UnitInput("d16p256", f"chunk:{OFFICIAL_UNIT}"),
               X32_UNIT: UnitInput("d16m", f"chunk:{X32_UNIT}")},
        optimizer=OptimizerSpec("adamw", cfg.lr, cfg.weight_decay, cfg.betas, cfg.eps),
        schedule=ScheduleSpec("multistep", "epoch", tuple(cfg.milestones), cfg.gamma),
        iterations=cfg.epochs,
        iteration_unit="epoch",
        sampler=cfg.sampler,
        sampler_params=cfg.sampler_params(),
        loss_weights={"clas2": 1.0},
        scale_constants={},
        precision="fp32",
        input_table=_input_table(cfg),
        sources=sources,
        notes=(f"VadCLIP visual branch, {cfg.dataset} configuration: visual_layers "
               f"{cfg.visual_layers}, attn_window {cfg.attn_window}, visual_head "
               f"{cfg.visual_head}, visual_length = unit (256; 32 for VCV.32), visual_width = D, "
               f"embed_dim {cfg.embed_dim} (removed text branch). Schedule in official optimizer "
               f"steps: {cfg.epochs} epochs, an epoch = one pass over the training videos "
               f"repeated crops = {cfg.crops} times (the official {cfg.crops}-crop list rows), "
               f"MultiStepLR {list(cfg.milestones)} x{cfg.gamma} on those epoch boundaries. Text "
               "branch, CLASM loss2 and loss3 removed. The official per-epoch test evaluation "
               "and the reload of the best test checkpoint at every epoch end are removed: the "
               "official trajectory restarts each epoch from the best test point so far, here "
               "training continues from the current weights (selection on validation after "
               "training). See the module docstring 'Adaptations:'."),
    )


class VadCLIPVisualMethod(BaselineMethod):
    """The VCV port (module docstring)."""

    name = NAME
    official_unit = OFFICIAL_UNIT

    def __init__(self, config: VCVConfig = OFFICIAL["ucf"], recipe: Recipe | None = None
                 ) -> None:
        self.config = config
        self._recipe = recipe if recipe is not None else make_recipe(config)

    @property
    def recipe(self) -> Recipe:
        return self._recipe

    def for_dataset(self, dataset: str) -> VadCLIPVisualMethod:
        """The method with the official configuration of ``dataset``: this instance when it is
        already that dataset's (an override or a recording subclass is kept), else
        ``METHODS[dataset]``."""
        if dataset not in METHODS:
            raise MethodError(f"vadclip_v: no official configuration for {dataset!r}")
        return self if self.config.dataset == dataset else METHODS[dataset]

    def with_overrides(self, **changes) -> VadCLIPVisualMethod:
        """A copy with changed configuration fields (tests and the (q1b) self-test only, e.g.
        batch, epochs, crops); the official instances are never changed."""
        return VadCLIPVisualMethod(replace(self.config, **changes))

    def build(self, feature_dim: int, *, unit: int, input_ln: bool,
              scale: Mapping[str, float]) -> nn.Module:
        self.recipe.unit_input(unit)                      # a registered unit only
        if scale:
            raise MethodError(f"vadclip_v has no scale constant, got {dict(scale)}")
        c = self.config
        return VadCLIPVisual(int(unit), int(feature_dim), c.visual_head, c.visual_layers,
                             c.attn_window, c.embed_dim, bool(input_ln))

    def loss(self, model: nn.Module, batch: Batch, *, unit: int,
             scale: Mapping[str, float]) -> LossOutput:
        logits1 = model(batch.features, batch.lengths, batch.mask)
        loss1 = CLAS2(logits1, batch.labels, batch.lengths, batch.features.device)
        return LossOutput(loss1, {"clas2": loss1.detach()})

    def sequence_logits(self, model: nn.Module, x: torch.Tensor, *, unit: int) -> torch.Tensor:
        maxlen = int(model.visual_length)
        if maxlen != int(unit) or self.recipe.unit_input(unit).chunk() != maxlen:
            raise MethodError(f"vadclip_v: unit {unit} for a model of visual_length {maxlen}")
        n = int(x.shape[0])
        visual, len_cur = process_split(x, maxlen)
        lengths = piece_lengths(len_cur, maxlen)
        rows = torch.arange(visual.shape[0] * maxlen, device=x.device).reshape(-1, maxlen)
        logits1 = model(visual, lengths, rows < n)
        logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
        return logits1[0:len_cur].squeeze(-1)


# --- the XD sampler (xd_train.py L131, L67) --------------------------------------------------

class _Positions(Dataset):
    def __init__(self, positions: Sequence[int]) -> None:
        self.positions = tuple(int(p) for p in positions)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> int:
        return self.positions[i]


def _as_tuple(items: list[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in items)


class SingleLoaderSampler(samplers.Sampler):
    """XD's training loader: one ``DataLoader(train_dataset, batch_size, shuffle=True)`` over
    every training-part video in list order, each repeated ``crops`` times in a row (the official
    10-crop list rows; drop_last False, num_workers 0, generator None), iterated once per epoch
    (``for i, item in enumerate(train_loader)``), so the global torch generator is consumed as by
    the official loop (the iterator's base seed and RandomSampler's seed at each epoch's first
    batch). Parameters: ``batch`` (official 96), ``crops`` (default 1; official lists 10)."""

    name = "vadclip_v_xd"

    def __init__(self, pools: samplers.Pools, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(pools, params)
        unknown = set(self.params) - {"batch", "crops"}
        if unknown or "batch" not in self.params:
            raise samplers.SamplerError(f"vadclip_v_xd needs 'batch' (and optionally 'crops'), "
                                        f"got {sorted(self.params)}")
        batch = int(self.params["batch"])
        if batch < 1 or pools.size < 1:
            raise samplers.SamplerError(f"vadclip_v_xd: batch {batch} over {pools.size} videos")
        self.label_of = {**{int(v): 0 for v in pools.normal},
                         **{int(v): 1 for v in pools.abnormal}}
        positions = samplers.crop_rows(sorted(self.label_of), int(self.params.get("crops", 1)))
        self.loader = DataLoader(_Positions(positions), batch_size=batch, shuffle=True,
                                 drop_last=False, num_workers=0, collate_fn=_as_tuple)
        self._it = None

    def steps_per_epoch(self) -> int:
        return len(self.loader)

    def draw(self, step: int) -> samplers.StepDraw:
        self._advance(step)
        if (int(step) - 1) % len(self.loader) == 0:
            self._it = iter(self.loader)
        videos = next(self._it)
        return samplers.StepDraw(videos, tuple(self.label_of[v] for v in videos))


samplers.register(SingleLoaderSampler.name, SingleLoaderSampler, owner=__name__)

RECIPE = make_recipe(OFFICIAL["ucf"])
RECIPE_XDV = make_recipe(OFFICIAL["xdv"])
METHOD = VadCLIPVisualMethod(OFFICIAL["ucf"], RECIPE)
METHOD_XDV = VadCLIPVisualMethod(OFFICIAL["xdv"], RECIPE_XDV)
METHODS = {"ucf": METHOD, "xdv": METHOD_XDV}
