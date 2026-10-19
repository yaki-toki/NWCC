"""Equivalence harness of the VCV port (``wvad.baselines.vadclip_v``) against the official
VadCLIP code (PLAN.md "관문" (q1), (q1b), (q1c), (q1e); "실행 순서" 2; 결정 0002). New code
(add-only rule). Never imported by training or scoring code.

The official clone (https://github.com/nwpu-zxr/VadCLIP, the commit of results/e023_licences.json)
lies outside the repository under $WVAD_OFFICIAL_ROOT and is loaded by path through the
``official`` loader module (wvad.baselines.official), which every entry point takes as its first
argument: no source file under src/ but official.py imports it (import-graph test), so the
callers - the tests and the command line below - pass it in. The official CLIPVAD constructor
(``check_constructor`` also on the CPU) reads the CLIP ViT-B/16 file through ``clip.load``
(~/.cache/clip, downloaded by the official code); its DistanceAdj hard-codes 'cuda', so every
check running the official forward needs a CUDA device: (q1) and (q1e) run on CUDA, not on the
CPU the plan's (q1) procedure names (a recorded deviation, PLAN.md 기록 2026-10-19; each result
carries it under "deviation"); the constructor check and (q1c) run on the CPU.

    (q1)  ``check_q1``: the official constructor and the port from one seed (global generator
          state after construction, every kept parameter bit for bit); on fixed toy inputs the
          training forward (official encode_video + classifier, CLIPVAD.forward L203-L204), CLAS2
          (official function, official get_batch_label labels), the gradients and one AdamW step
          (official call torch.optim.AdamW(model.parameters(), lr=args.lr)); bit equality or
          relative error <= 1e-6. ``check_q1_full_forward``: at the official width 512 the
          unmodified official forward (text branch included) gives the port's logits1
    (q1c) ``check_q1c``: the official datasets and DataLoaders over a synthetic list, driven by
          the official loop structure from one generator state, against the port's sampler:
          index tuples bit for bit
    (q1e) ``check_q1e``: the official test() (ucf_test.py / xd_test.py, its UCFDataset /
          XDDataset test mode and process_split; metrics and dmAP replaced by capturing stubs,
          the model wrapped to its visual branch) on the same weights, per N_chunk piece, against
          common.score_video of the port: |delta| <= 1e-6 on the logits and on the official prob1;
          synthetic rows of chosen lengths, and ``check_q1e_val``: the processed validation videos
          of a dataset and pair (real lengths and value ranges: the chain's InputPipeline - the
          variant's scale map - and, for XCB+SigLIP, the input LayerNorm)
    (q1b) ``run_q1b``: the official training script, modified outside the repository (input
          width and list path by its own command-line options; test list, ground truth and the
          best-test reload removed; text branch cut; per-iteration loss record, deterministic
          mode and an audit hook added), run as a subprocess on the port's training rows (the
          list holds every video ``crops`` = 10 times, as the official 10-crop lists, so both
          runs take the official optimizer steps per epoch), against the port's training chain
          (``train.run``): per-iteration losses and lr, last weights and the last-point
          validation V-AUC bit for bit (PLAN.md "(q1b) 통과 기준"); the record
          (``update_record``, results/e023_q1b.json under a lock) holds the commit, the diff
          sha256, the
          opened-file audit and the plan sha256; paths are written relative to placeholders
          ($WVAD_OFFICIAL_ROOT, <work>, <python>, ~), never as personal absolute paths

    python -c "import sys; from wvad.baselines import official, check_vadclip_v as c;
               c.main(sys.argv[1:], official=official)" <check> ...
        q1  --dataset ucf|xdv [--out FILE]
        q1c --dataset ucf|xdv --work DIR [--out FILE]
        q1e --dataset ucf|xdv --work DIR [--n_chunk N]
        q1e_val --dataset ucf|xdv --pair xcb_xcl|xcb_siglip [--n_chunk N] [--max_videos K]
                [--paths FILE] [--out FILE]
        q1b --dataset ucf|xdv --pair xcb_xcl --seed S --work DIR [--paths FILE] [--record FILE]
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wvad.baselines import common, samplers
from wvad.baselines import vadclip_v as vcv
from wvad.baselines.recipes import E023_DIR, LICENCES_FILE

NAME = vcv.NAME
TOL_REL_Q1 = 1e-6                 # PLAN.md (q1): bit equality or relative error <= 1e-6
TOL_ABS_Q1E = 1e-6                # PLAN.md (q1e): elementwise |delta| <= 1e-6
TOL_REL_Q1B = 1e-5                # PLAN.md (q1b) numeric fallback: first 50 iterations
CUDA_DEVIATION = ("CUDA, not the CPU of PLAN.md (q1): the official DistanceAdj hard-codes "
                  "'cuda' (recorded deviation, PLAN.md 기록 2026-10-19)")
VAUC_POINTS = 1.0                 # (q1b) numeric fallback: per seed |delta V-AUC| <= 1 point
# label maps of the official __main__ blocks (ucf_train.py L134, xd_train.py L128)
LABEL_MAPS = {
    "ucf": {"Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
            "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
            "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
            "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
            "Vandalism": "vandalism"},
    "xdv": {"A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot", "B5": "abuse",
            "B6": "car accident", "G": "explosion"},
}
# list labels written for the synthetic / training lists: only 'normal or not' reaches CLAS2
NORMAL_TEXT = {"ucf": "Normal", "xdv": "A"}
ANOMALY_TEXT = {"ucf": ("Abuse", "Arrest", "Fighting"), "xdv": ("B1-0-0", "B2-B4-0", "G-0-0")}


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("the official VadCLIP code needs CUDA (DistanceAdj hard-codes 'cuda')")
    return torch.device("cuda")


# --- the official modules -------------------------------------------------------------------

@contextlib.contextmanager
def official_vadclip(official: ModuleType, dataset: str) -> Iterator[SimpleNamespace]:
    """The official modules of one dataset's pipeline (loaded by path, dropped afterwards):
    ``train`` (ucf_train / xd_train: CLIPVAD, CLAS2, setup_seed, get_batch_label,
    get_prompt_text, the Dataset class), ``test`` (ucf_test / xd_test), ``tools``, the parsed
    default options ``args`` and the label map."""
    cfg = vcv.OFFICIAL[dataset]
    with official.session(NAME, "src") as s:
        train = s.load(cfg.train_script)
        label_map = dict(LABEL_MAPS[dataset])
        option = sys.modules[Path(cfg.option_file).stem]
        yield SimpleNamespace(
            cfg=cfg, train=train, test=sys.modules[Path(cfg.test_script).stem],
            tools=sys.modules["utils.tools"], option=option, args=option.parser.parse_args([]),
            Dataset=train.UCFDataset if dataset == "ucf" else train.XDDataset,
            label_map=label_map, prompt_text=train.get_prompt_text(label_map))


def build_official(ns: SimpleNamespace, feature_dim: int, unit: int, *, seed: int | None,
                   device: str = "cuda") -> nn.Module:
    """The official CLIPVAD with the official options, visual_width = D, visual_length = unit
    (official main: setup_seed first)."""
    a = ns.args
    if seed is not None:
        ns.train.setup_seed(seed)
    return ns.train.CLIPVAD(a.classes_num, a.embed_dim, int(unit), int(feature_dim),
                            a.visual_head, a.visual_layers, a.attn_window, a.prompt_prefix,
                            a.prompt_postfix, device)


def official_logits1(net: nn.Module, visual: torch.Tensor, lengths) -> torch.Tensor:
    """logits1 of the official forward (model.py L203-L204) without its text branch."""
    v = net.encode_video(visual, None, lengths)
    return net.classifier(v + net.mlp2(v))


def build_port(dataset: str, feature_dim: int, unit: int, *, seed: int, input_ln: bool = False,
               method: vcv.VadCLIPVisualMethod | None = None) -> nn.Module:
    common.seed_all(seed)
    m = method or vcv.METHODS[dataset]
    return m.build(feature_dim, unit=unit, input_ln=input_ln, scale={})


# --- comparisons ----------------------------------------------------------------------------

def compare(a, b) -> dict:
    """Bit equality, max |a - b| and max |a - b| / max |b| of two arrays / tensors."""
    x = torch.as_tensor(a).detach().double().cpu()
    y = torch.as_tensor(b).detach().double().cpu()
    if x.shape != y.shape:
        return {"bit_equal": False, "shape": [list(x.shape), list(y.shape)],
                "max_abs": float("inf"), "max_rel": float("inf")}
    bit = bool(torch.equal(torch.as_tensor(a).detach().cpu(), torch.as_tensor(b).detach().cpu()))
    d = (x - y).abs()
    mabs = float(d.max()) if d.numel() else 0.0
    scale = float(y.abs().max()) if y.numel() else 0.0
    return {"bit_equal": bit, "max_abs": mabs, "max_rel": mabs / scale if scale > 0 else mabs}


def compare_named(port: Mapping[str, torch.Tensor], official: Mapping[str, torch.Tensor]
                  ) -> dict:
    """Every port tensor against the official one of the same name."""
    rows = {}
    for k, v in port.items():
        if k not in official:
            rows[k] = {"bit_equal": False, "missing_in_official": True,
                       "max_abs": float("inf"), "max_rel": float("inf")}
        elif v is None or official[k] is None:
            rows[k] = {"bit_equal": v is None and official[k] is None,
                       "none": [v is None, official[k] is None], "max_abs": 0.0, "max_rel": 0.0}
        else:
            rows[k] = compare(v, official[k])
    return {"n": len(rows), "bit_equal": all(r["bit_equal"] for r in rows.values()),
            "max_rel": max((r["max_rel"] for r in rows.values()), default=0.0),
            "not_bit_equal": sorted(k for k, r in rows.items() if not r["bit_equal"])}


def _passes(c: Mapping, tol: float) -> bool:
    return bool(c["bit_equal"] or c["max_rel"] <= tol)


def _labels01(dataset: str, texts: Sequence[str]) -> list[int]:
    return [0 if t == NORMAL_TEXT[dataset] else 1 for t in texts]


def toy_batch(dataset: str, feature_dim: int, unit: int, lengths: Sequence[int],
              device: torch.device, seed: int = 1234) -> tuple[common.Batch, list[str]]:
    """Fixed toy inputs: normal videos first, zero rows past each length."""
    rs = np.random.RandomState(seed)
    x = rs.standard_normal((len(lengths), unit, feature_dim)).astype(np.float32)
    for i, n in enumerate(lengths):
        x[i, int(n):] = 0.0
    half = len(lengths) // 2
    texts = [NORMAL_TEXT[dataset]] * half + [ANOMALY_TEXT[dataset][i % 3]
                                             for i in range(len(lengths) - half)]
    batch = common.Batch(torch.from_numpy(x).to(device),
                         torch.tensor(_labels01(dataset, texts), dtype=torch.float32,
                                      device=device),
                         torch.tensor(list(lengths), dtype=torch.int64, device=device),
                         tuple(range(len(lengths))), half)
    return batch, texts


# --- (q1) -----------------------------------------------------------------------------------

def check_constructor(official: ModuleType, dataset: str, *, feature_dim: int = 16,
                      unit: int = vcv.OFFICIAL_UNIT, seed: int = 0, device: str = "cpu"
                      ) -> dict:
    """(q1) constructor part (no forward, so no CUDA needed): the official CLIPVAD (its
    clip.load on ``device``) and the port from one seed leave the global generator in the same
    state, and every port parameter equals the official one of the same name, bit for bit."""
    with official_vadclip(official, dataset) as ns:
        off = build_official(ns, feature_dim, unit, seed=seed, device=device)
        rng_off = torch.get_rng_state()
        port = build_port(dataset, feature_dim, unit, seed=seed)
        rng_port = torch.get_rng_state()
        keys = list(off.state_dict())
        osd = {k: v.cpu() for k, v in off.state_dict().items()
               if not k.startswith("clipmodel.")}
    psd = port.state_dict()
    res = compare_named(psd, osd)
    res.update({"dataset": dataset, "feature_dim": feature_dim, "unit": unit, "seed": seed,
                "rng_state_equal": bool(torch.equal(rng_off, rng_port)),
                "shapes_equal": all(tuple(v.shape) == tuple(osd[k].shape)
                                    for k, v in psd.items() if k in osd),
                "official_only": sorted({k.split(".")[0] for k in keys} -
                                        {k.split(".")[0] for k in psd})})
    res["pass"] = bool(res["bit_equal"] and res["rng_state_equal"] and res["shapes_equal"])
    return res


def check_q1(official: ModuleType, dataset: str, *, feature_dim: int = 16,
             unit: int = vcv.OFFICIAL_UNIT, seed: int = 0,
             lengths: Sequence[int] | None = None) -> dict:
    """(q1) constructor, forward, CLAS2, gradients and one optimizer step (module docstring)."""
    dev = require_cuda()
    method = vcv.METHODS[dataset]
    lengths = tuple(lengths or (unit, max(1, unit * 3 // 8), max(1, unit // 7), unit - 1))
    with official_vadclip(official, dataset) as ns:
        off = build_official(ns, feature_dim, unit, seed=seed, device=dev.type)
        rng_off = torch.get_rng_state()
        port = build_port(dataset, feature_dim, unit, seed=seed)
        rng_port = torch.get_rng_state()
        osd = dict(off.state_dict())
        ctor = compare_named(port.state_dict(), osd)
        ctor["rng_state_equal"] = bool(torch.equal(rng_off, rng_port))
        ctor["shapes_equal"] = all(tuple(v.shape) == tuple(osd[k].shape)
                                   for k, v in port.state_dict().items() if k in osd)
        off.to(dev)
        port.to(dev)
        batch, texts = toy_batch(dataset, feature_dim, unit, lengths, dev)
        off.train()
        method.set_mode(port, True)
        lo = official_logits1(off, batch.features, batch.lengths)
        text_labels = ns.train.get_batch_label(texts, ns.prompt_text, ns.label_map).to(dev)
        loss_o = ns.train.CLAS2(lo, text_labels, batch.lengths, dev)
        lp = port(batch.features, batch.lengths, batch.mask)
        out = method.loss(port, batch, unit=unit, scale={})
        loss_o.backward()
        out.total.backward()
        named_o = dict(off.named_parameters())
        grads = compare_named({k: p.grad for k, p in port.named_parameters()},
                              {k: p.grad for k, p in named_o.items()})
        opt_o = torch.optim.AdamW(off.parameters(), lr=ns.args.lr)      # ucf/xd_train.py L47/48
        opt_p = method.make_optimizer(port, method.recipe.optimizer.lr)
        opt_o.step()
        opt_p.step()
        step = compare_named({k: p.detach() for k, p in port.named_parameters()},
                             {k: p.detach() for k, p in named_o.items()})
        labels_ok = bool(torch.equal(1 - text_labels[:, 0].reshape(-1), batch.labels))
    res = {"dataset": dataset, "feature_dim": feature_dim, "unit": unit, "seed": seed,
           "lengths": list(lengths), "device": str(dev), "deviation": CUDA_DEVIATION,
           "constructor": ctor,
           "logits1": compare(lp, lo), "loss": compare(out.total, loss_o),
           "loss_values": [float(out.total.detach()), float(loss_o.detach())],
           "labels_equal": labels_ok, "gradients": grads, "optimizer_step": step,
           "lr": [float(method.recipe.optimizer.lr), float(ns.args.lr)]}
    res["pass"] = bool(ctor["bit_equal"] and ctor["rng_state_equal"] and labels_ok
                       and res["lr"][0] == res["lr"][1]
                       and all(_passes(res[k], TOL_REL_Q1)
                               for k in ("logits1", "loss", "gradients", "optimizer_step")))
    return res


def check_q1_full_forward(official: ModuleType, dataset: str, *, seed: int = 0,
                          lengths: Sequence[int] = (256, 90)) -> dict:
    """At the official width (512 = embed_dim) the unmodified official CLIPVAD.forward (text
    branch included) and the port give the same logits1 (eval mode, same seed)."""
    dev = require_cuda()
    with official_vadclip(official, dataset) as ns:
        off = build_official(ns, 512, vcv.OFFICIAL_UNIT, seed=seed, device=dev.type).to(dev)
        port = build_port(dataset, 512, vcv.OFFICIAL_UNIT, seed=seed).to(dev)
        batch, _ = toy_batch(dataset, 512, vcv.OFFICIAL_UNIT, lengths, dev, seed=77)
        off.eval()
        port.eval()
        with torch.no_grad():
            _, lo, _ = off(batch.features, None, ns.prompt_text, batch.lengths)
            lp = port(batch.features, batch.lengths)
    c = compare(lp, lo)
    return {"dataset": dataset, "device": str(dev), "deviation": CUDA_DEVIATION, "logits1": c,
            "pass": _passes(c, TOL_REL_Q1)}


# --- (q1c) ----------------------------------------------------------------------------------

SYNTH_LABELS = (0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0)


def write_list(folder: Path, dataset: str, videos: Sequence[np.ndarray], labels: Sequence[int],
               repeat: int = 1) -> Path:
    """One .npy per video (float32 rows) and the official list CSV (path,label), in order; each
    video's row ``repeat`` times in a row (the official lists' 10 crop rows per video, here all
    naming the one single-crop file)."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["path,label"]
    for i, (x, y) in enumerate(zip(videos, labels, strict=True)):
        p = folder / f"v{i:05d}.npy"
        np.save(p, np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
        text = NORMAL_TEXT[dataset] if int(y) == 0 else ANOMALY_TEXT[dataset][i % 3]
        lines += [f"{p.as_posix()},{text}"] * int(repeat)
    csv = folder / "list.csv"
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv


def _ids(features: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(round(float(f[0, 0]))) - 1 for f in features)


def official_draws(ns: SimpleNamespace, csv: Path, batch: int, epochs: int) -> list[tuple]:
    """The official loaders (ucf_train.py L136-L139 / xd_train.py L130-L131) driven as the
    official loops do (ucf_train.py L62-L71: both iterators re-created every epoch, min(len)
    steps, normal then anomalous; xd_train.py L63-L67: one loader per epoch)."""
    length = ns.args.visual_length
    out = []
    if ns.cfg.dataset == "ucf":
        nd = ns.Dataset(length, str(csv), False, ns.label_map, True)
        ad = ns.Dataset(length, str(csv), False, ns.label_map, False)
        nl = DataLoader(nd, batch_size=batch, shuffle=True, drop_last=True)
        al = DataLoader(ad, batch_size=batch, shuffle=True, drop_last=True)
        for _ in range(epochs):
            n_it, a_it = iter(nl), iter(al)
            for _ in range(min(len(nl), len(al))):
                nf, nlab, _ = next(n_it)
                af, alab, _ = next(a_it)
                out.append((_ids(nf) + _ids(af), tuple(nlab) + tuple(alab)))
    else:
        d = ns.Dataset(length, str(csv), False, ns.label_map)
        loader = DataLoader(d, batch_size=batch, shuffle=True)
        for _ in range(epochs):
            for _, item in enumerate(loader):
                out.append((_ids(item[0]), tuple(item[1])))
    return out


def check_q1c(official: ModuleType, dataset: str, work: Path, *, seed: int = 0, epochs: int = 3,
              batch: int | None = None, labels: Sequence[int] = SYNTH_LABELS,
              crops: int | None = None) -> dict:
    """(q1c) official loader loop vs the port's sampler from one generator state (the official
    list holds every video ``crops`` times, default the recipe's 10, as the official lists)."""
    batch = int(batch or (3 if dataset == "ucf" else 4))
    crops = int(crops or vcv.METHODS[dataset].config.crops)
    videos = [np.full((3 + i % 5, 4), float(i + 1), dtype=np.float32)
              for i in range(len(labels))]
    with official_vadclip(official, dataset) as ns:
        csv = write_list(Path(work) / f"q1c_{dataset}", dataset, videos, labels, repeat=crops)
        ns.train.setup_seed(seed)
        official = official_draws(ns, csv, batch, epochs)
        state_off = torch.get_rng_state()
    method = vcv.METHODS[dataset].with_overrides(batch=batch, crops=crops)
    common.seed_all(seed)
    s = samplers.create(method.recipe.sampler, samplers.Pools.from_labels(labels),
                        method.recipe.sampler_params)
    port = samplers.draws(s, len(official))
    state_port = torch.get_rng_state()
    idx_equal = [d.videos for d in port] == [o[0] for o in official]
    lab_equal = [d.labels for d in port] == [tuple(_labels01(dataset, o[1])) for o in official]
    return {"dataset": dataset, "seed": seed, "epochs": epochs, "batch": batch, "crops": crops,
            "steps": len(official), "steps_per_epoch": s.steps_per_epoch(),
            "indices_bit_equal": idx_equal, "labels_equal": lab_equal,
            "rng_state_equal_after": bool(torch.equal(state_off, state_port)),
            "pass": bool(idx_equal and lab_equal and torch.equal(state_off, state_port))}


# --- (q1e) ----------------------------------------------------------------------------------

class VisualBranch(nn.Module):
    """The official CLIPVAD seen by the official test(): its forward returns (None, logits1,
    zero logits2) with logits1 of model.py L203-L204 (text branch cut); logits1 kept."""

    def __init__(self, net: nn.Module, n_classes: int) -> None:
        super().__init__()
        self.net = net
        self.n_classes = int(n_classes)
        self.logits1: list[torch.Tensor] = []

    def forward(self, visual, padding_mask, text, lengths):
        v = self.net.encode_video(visual, padding_mask, lengths)
        logits1 = self.net.classifier(v + self.net.mlp2(v))
        self.logits1.append(logits1.detach())
        return None, logits1, logits1.new_zeros(logits1.shape[0], logits1.shape[1],
                                                self.n_classes)


class MemoryTestSet(Dataset):
    """UCFDataset / XDDataset test mode (dataset.py L312-L321, L333-L341) over in-memory rows:
    the official process_split, torch.tensor, label, length."""

    def __init__(self, tools, videos: Sequence[np.ndarray], clip_dim: int, label: str) -> None:
        self.tools, self.videos, self.clip_dim, self.label = tools, videos, clip_dim, label

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index):
        clip_feature, clip_length = self.tools.process_split(self.videos[index], self.clip_dim)
        return torch.tensor(clip_feature), self.label, clip_length


@contextlib.contextmanager
def _metric_stubs(test_mod) -> Iterator[dict]:
    """The official test()'s metric and dmAP calls captured (module attributes of the private
    module object, restored afterwards; the official file is not changed)."""
    names = ("roc_auc_score", "average_precision_score", "dmAP")
    namespace = vars(test_mod)                  # the private official module object's globals
    saved = {n: namespace[n] for n in names}
    captured: dict = {"roc": []}

    def roc(gt, scores):
        captured["roc"].append(np.asarray(scores, dtype=np.float64))
        return 0.5

    namespace.update({"roc_auc_score": roc,
                      "average_precision_score": lambda *a, **k: 0.5,
                      "dmAP": lambda *a, **k: ([0.0] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])})
    try:
        yield captured
    finally:
        namespace.update(saved)


def official_test_logits(ns: SimpleNamespace, net: nn.Module, videos: Sequence[np.ndarray],
                         maxlen: int, device: torch.device, folder: Path | None = None
                         ) -> list[dict]:
    """The official test() over ``videos`` (each a (n, D) float32 piece): through the official
    Dataset class in test mode over written .npy files when ``folder`` is given, else through
    MemoryTestSet. Per video: ``logits`` (the logits1 rows [0:len_cur], ucf_test.py L48) and
    ``prob1`` (the official sigmoid scores, captured from its roc_auc_score call)."""
    if folder is not None:
        csv = write_list(Path(folder), ns.cfg.dataset, videos, [1] * len(videos))
        ds = ns.Dataset(maxlen, str(csv), True, ns.label_map)
    else:
        ds = MemoryTestSet(ns.tools, videos, maxlen, ANOMALY_TEXT[ns.cfg.dataset][0])
    loader = DataLoader(ds, batch_size=1, shuffle=False)                # ucf_test.py L96
    wrapper = VisualBranch(net, len(ns.prompt_text))
    with _metric_stubs(ns.test) as cap, contextlib.redirect_stdout(io.StringIO()):
        ns.test.test(wrapper, loader, maxlen, ns.prompt_text, None, None, None, device.type)
    prob1 = cap["roc"][0][::16]
    out, start = [], 0
    for x, lg in zip(videos, wrapper.logits1, strict=True):
        n = int(np.asarray(x).shape[0])
        logits = lg.reshape(lg.shape[0] * lg.shape[1], lg.shape[2])[0:n].squeeze(-1)
        out.append({"logits": logits.float().cpu().numpy(), "prob1": prob1[start:start + n]})
        start += n
    if start != len(prob1):
        raise RuntimeError(f"official test() scored {len(prob1)} snippets for {start}")
    return out


def perturb(model: nn.Module, seed: int, scale: float = 0.05) -> None:
    """Deterministic non-trivial weights (q1e: trained-like LN / bias values)."""
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))


def load_into_official(off: nn.Module, state: Mapping[str, torch.Tensor]) -> dict:
    """The port's (or a saved) visual-branch weights into an official CLIPVAD; only the text
    branch / CLIP keys may stay unset and the input LayerNorm is not official."""
    kept = {k: v for k, v in state.items() if not k.startswith("input_norm.")}
    res = off.load_state_dict(kept, strict=False)
    allowed = ("clipmodel.", "mlp1.", "text_prompt_embeddings.")
    bad = [k for k in res.missing_keys if not k.startswith(allowed)]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"official CLIPVAD keys: missing {bad}, unexpected "
                           f"{res.unexpected_keys}")
    return {"loaded": len(kept), "missing_text_branch": len(res.missing_keys)}


Q1E_LENGTHS = (5, 255, 256, 257, 300, 512, 700, 4096, 4100)


def check_q1e(official: ModuleType, dataset: str, work: Path, *, feature_dim: int = 16,
              unit: int = vcv.OFFICIAL_UNIT, lengths: Sequence[int] = Q1E_LENGTHS,
              n_chunk: int = common.N_CHUNK_DEFAULT, input_ln: bool = False,
              seed: int = 0) -> dict:
    """(q1e) official test() per N_chunk piece vs common.score_video of the port (same
    weights, eval mode, fp32): n_v < unit, = unit, multiples of unit (the official all-padding
    piece), last-piece padding, n_v > N_chunk, the input LayerNorm (applied to the official
    input rows, PLAN.md (q1e))."""
    dev = require_cuda()
    method = vcv.METHODS[dataset]
    if dataset == "xdv" and vcv.effective_window(method.config.attn_window, unit) != \
            method.config.attn_window:
        raise ValueError("the official XD window exceeds the 32-unit sequence (adaptation 7): "
                         "no official counterpart")
    port = build_port(dataset, feature_dim, unit, seed=seed, input_ln=input_ln)
    perturb(port, seed + 1)
    port.to(dev)
    method.set_mode(port, False)
    rs = np.random.RandomState(seed + 2)
    cases = []
    with official_vadclip(official, dataset) as ns:
        off = build_official(ns, feature_dim, unit, seed=seed, device=dev.type)
        load_into_official(off, port.state_dict())
        off.to(dev)
        for n in lengths:
            x = rs.standard_normal((int(n), feature_dim)).astype(np.float32)
            z = common.score_video(method, port, torch.from_numpy(x), unit=unit,
                                   n_chunk=n_chunk, device=dev)
            pieces = []
            for a, b in common.chunk_bounds(int(n), n_chunk):
                rows = torch.from_numpy(x[a:b])
                if input_ln:
                    with torch.no_grad():
                        rows = port.input_norm(rows.to(dev)).cpu()
                pieces.append(rows.numpy())
            got = official_test_logits(ns, off, pieces, unit, dev,
                                       Path(work) / f"q1e_{dataset}_{unit}_{n}")
            zo = np.concatenate([g["logits"] for g in got])
            po = np.concatenate([g["prob1"] for g in got])
            cl = compare(z, zo)
            cp = compare(torch.sigmoid(torch.from_numpy(z).to(dev)).double(), po)  # as test()
            cases.append({"n": int(n), "pieces": len(pieces), "logits": cl, "prob1": cp,
                          "pass": cl["max_abs"] <= TOL_ABS_Q1E and cp["max_abs"] <= TOL_ABS_Q1E})
    return {"dataset": dataset, "unit": unit, "feature_dim": feature_dim, "n_chunk": n_chunk,
            "input_ln": input_ln, "inputs": "synthetic rows of the listed lengths",
            "device": str(dev), "deviation": CUDA_DEVIATION, "cases": cases,
            "bit_equal": all(c["logits"]["bit_equal"] for c in cases),
            "max_abs": max(c["logits"]["max_abs"] for c in cases),
            "pass": all(c["pass"] for c in cases)}


def check_q1e_val(official: ModuleType, dataset: str, pair: str, *, paths=None, split=None,
                  n_chunk: int = common.N_CHUNK_DEFAULT, seed: int = 0,
                  max_videos: int | None = None, unit: int = vcv.OFFICIAL_UNIT) -> dict:
    """(q1e) on the processed validation videos (PLAN.md (q1e): real-length validation videos):
    the chain's validation input (train.validation_videos: the dense training-split store
    through the variant's InputPipeline), the same perturbed weights in the port and the official
    CLIPVAD (input LayerNorm in front of the official test() for XCB+SigLIP), every video per
    N_chunk piece; |delta| <= 1e-6 on logits and prob1. ``max_videos``: the first K validation
    videos plus the longest one (default: every validation video)."""
    from wvad.baselines import train as tr
    from wvad.data import load_split, split_file

    dev = require_cuda()
    method = vcv.METHODS[dataset]
    if paths is None:
        from wvad.paths import load_paths

        paths = load_paths(None)
    split = split if split is not None else load_split(split_file(dataset))
    variant = method.recipe.unit_input(unit).variant
    n_val = len(split.val_indices)
    val = tr.validation_videos(paths, dataset, pair, variant, split, max_videos=max_videos)
    ln = common.input_ln(pair)
    dim = int(sum(val.dims))
    port = build_port(dataset, dim, unit, seed=seed, input_ln=ln)
    perturb(port, seed + 1)
    port.to(dev)
    method.set_mode(port, False)
    cases = []
    with official_vadclip(official, dataset) as ns:
        off = build_official(ns, dim, unit, seed=seed, device=dev.type)
        load_into_official(off, port.state_dict())
        off.to(dev)
        for v, x in zip(val.indices, val.x, strict=True):
            n = int(x.shape[0])
            z = common.score_video(method, port, x, unit=unit, n_chunk=n_chunk, device=dev)
            pieces = []
            for a, b in common.chunk_bounds(n, n_chunk):
                rows = x[a:b]
                if ln:
                    with torch.no_grad():
                        rows = port.input_norm(rows.to(dev)).cpu()
                pieces.append(rows.numpy())
            got = official_test_logits(ns, off, pieces, unit, dev)
            zo = np.concatenate([g["logits"] for g in got])
            po = np.concatenate([g["prob1"] for g in got])
            cl = compare(z, zo)
            cp = compare(torch.sigmoid(torch.from_numpy(z).to(dev)).double(), po)
            cases.append({"video": int(v), "n": n, "pieces": len(pieces), "logits": cl,
                          "prob1": cp, "pass": cl["max_abs"] <= TOL_ABS_Q1E
                          and cp["max_abs"] <= TOL_ABS_Q1E})
    return {"check": "E023 (q1e) VCV on the processed validation videos", "dataset": dataset,
            "pair": pair, "variant": variant, "pipeline": val.pipeline.to_json(),
            "input_ln": ln, "unit": unit, "n_chunk": n_chunk, "videos": len(cases),
            "of_validation_videos": n_val, "max_snippets": max(c["n"] for c in cases),
            "device": str(dev), "deviation": CUDA_DEVIATION, "cases": cases,
            "bit_equal": all(c["logits"]["bit_equal"] for c in cases),
            "max_abs": max(c["logits"]["max_abs"] for c in cases),
            "pass": bool(cases) and all(c["pass"] for c in cases)}


# --- (q1b) ----------------------------------------------------------------------------------

HEADER = '''# --- E023 (q1b) harness additions (written outside the repository; a recorded diff) ---
import os as _q1b_os
import sys as _q1b_sys
_Q1B = {"opened": [], "losses": [], "lrs": []}


def _q1b_audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = args[0] if isinstance(args[0], str) else args[0].decode()
        _Q1B["opened"].append(_q1b_os.path.abspath(p))


_q1b_sys.addaudithook(_q1b_audit)


def _q1b_deterministic():
    import torch as _t
    _t.use_deterministic_algorithms(True)
    _t.backends.cudnn.benchmark = False
    _t.backends.cudnn.deterministic = True


def _q1b_record(loss, optimizer):
    _Q1B["losses"].append(float(loss.detach()))
    _Q1B["lrs"].append(float(optimizer.param_groups[0]["lr"]))


def _q1b_dump(model):
    import json as _j
    import os as _o
    import torch as _t
    out = _o.environ["E023_Q1B_OUT"]
    _t.save({k: v.detach().cpu() for k, v in model.state_dict().items()
             if not k.startswith("clipmodel.")}, _o.path.join(out, "official_last.pt"))
    with open(_o.path.join(out, "official_run.json"), "w", encoding="utf-8") as f:
        _j.dump({"losses": _Q1B["losses"], "lrs": _Q1B["lrs"],
                 "deterministic": _t.are_deterministic_algorithms_enabled(),
                 "torch": _t.__version__, "opened": _Q1B["opened"]}, f)
# --- end of the additions ---
'''

VISUAL_CALL = ("            _q1b_v = model.encode_video({x}, None, feat_lengths)\n"
               "            logits1 = model.classifier(_q1b_v + model.mlp2(_q1b_v))  "
               "# E023 VCV: CLIPVAD.forward L203-L204 (text branch cut)\n")
LOSS_ONLY = ("            loss = loss1  # E023 VCV: loss2 (CLASM) and loss3 belong to the "
             "removed text branch\n            _q1b_record(loss, optimizer)\n")
END = "        scheduler.step()\n    _q1b_dump(model)\n"


@dataclass(frozen=True)
class Edit:
    """Replace the span from the start of ``start`` to the end of ``end`` (each unique)."""

    why: str
    start: str
    end: str | None
    new: str


def _edits(dataset: str) -> tuple[Edit, ...]:
    ucf = dataset == "ucf"
    x = "visual_features" if ucf else "visual_feat"
    ds = "UCFDataset" if ucf else "XDDataset"
    return (
        Edit("deterministic mode after the official seeding (PLAN.md (q1b))",
             "    setup_seed(args.seed)\n", None,
             "    setup_seed(args.seed)\n    _q1b_deterministic()\n"),
        Edit("no test ground truth is read (PLAN.md 실행 순서 2)",
             "    gt = np.load(args.gt_path)\n",
             "    gtlabels = np.load(args.gt_label_path, allow_pickle=True)\n",
             "    gt = gtsegments = gtlabels = None  # E023 (q1b): no test ground truth\n"),
        Edit("VCV: the visual branch of CLIPVAD.forward (text branch cut)",
             f"            text_features, logits1, logits2 = model({x}, None, prompt_text, "
             "feat_lengths) \n", None, VISUAL_CALL.format(x=x)),
        Edit("VCV: loss = CLAS2 alone; per-iteration loss and lr record",
             "            #loss2\n" if ucf else
             "            loss2 = CLASM(logits2, text_labels, feat_lengths, device)\n",
             "            loss = loss1 + loss2 + loss3\n" if ucf else
             "            loss = loss1 + loss2 + loss3 * 1e-4\n", LOSS_ONLY),
        Edit("no test evaluation / best-test reload during training; final weights dumped",
             "            if step % 1280 == 0 and step != 0:\n" if ucf else
             "            if step % 4800 == 0 and step != 0:\n",
             "    torch.save(checkpoint['model_state_dict'], args.model_path)\n", END),
        Edit("no test list is read",
             f"    test_dataset = {ds}(args.visual_length, args.test_list, True, label_map)\n",
             "    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)\n",
             "    test_loader = None  # E023 (q1b): no test list\n"),
    )


def _unique(text: str, s: str) -> int:
    if text.count(s) != 1:
        raise RuntimeError(f"official anchor found {text.count(s)} times: {s!r}")
    return text.index(s)


def modified_script(dataset: str, official_text: str) -> tuple[str, list[dict]]:
    """The (q1b) training script: HEADER + the official text with ``_edits`` applied."""
    text = official_text.replace("\r\n", "\n")
    applied = []
    for e in _edits(dataset):
        a = _unique(text, e.start)
        b = (a + len(e.start)) if e.end is None else _unique(text, e.end) + len(e.end)
        if b <= a:
            raise RuntimeError(f"official anchors out of order: {e.why}")
        applied.append({"why": e.why, "removed_lines": text[a:b].count("\n"),
                        "added_lines": e.new.count("\n")})
        text = text[:a] + e.new + text[b:]
    return HEADER + text, applied


def script_diff(dataset: str, official_text: str, modified: str) -> str:
    cfg = vcv.OFFICIAL[dataset]
    return "".join(difflib.unified_diff(
        official_text.replace("\r\n", "\n").splitlines(keepends=True),
        modified.splitlines(keepends=True), fromfile=f"a/{cfg.train_script}",
        tofile=f"b/{cfg.train_script} (E023 q1b)"))


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha_floats(values: Sequence[float]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


class Portable:
    """Paths written relative to placeholders (no personal absolute path in a record)."""

    def __init__(self, work: Path, official_root: Path | None = None) -> None:
        roots = [(Path(work), "<work>")]
        if official_root is not None:
            roots.append((Path(official_root), "$WVAD_OFFICIAL_ROOT"))
        roots += [(Path(sys.prefix), "<python>"), (Path(sys.base_prefix), "<python-base>"),
                  (Path.home(), "~")]
        self.roots = [(os.path.abspath(r), t) for r, t in roots]

    def __call__(self, path) -> str:
        p = os.path.abspath(str(path))
        for root, tag in self.roots:
            if os.path.normcase(p) == os.path.normcase(root):
                return tag
            if os.path.normcase(p).startswith(os.path.normcase(root.rstrip(os.sep) + os.sep)):
                return tag + "/" + Path(p[len(root.rstrip(os.sep)) + 1:]).as_posix()
        return "<other>/" + Path(p).name


DATA_SUFFIXES = (".npy", ".npz", ".csv", ".pt", ".pth", ".json", ".txt", ".pkl", ".h5", ".gz",
                 ".zip", ".mp4", ".avi")


def audit(opened: Sequence[str], portable: Portable) -> dict:
    """Opened-file audit: no official list / ground-truth file and no data-like file outside
    <work>, the clone's src, the Python installation and ~/.cache/clip (PLAN.md 실행 순서 2)."""
    names = sorted({portable(p) for p in opened})
    clone_list = [n for n in names if n.lower().startswith("$wvad_official_root/vadclip/list")]
    python = ("<python>/", "<python-base>/")
    allowed = ("<work>/", "$wvad_official_root/vadclip/src/", "~/.cache/clip/") + python
    data = [n for n in names if n.lower().endswith(DATA_SUFFIXES)]
    outside = [n for n in data if not n.lower().startswith(allowed)]
    test_like = [n for n in data if not n.startswith(python)
                 and ("test" in Path(n).name.lower() or Path(n).name.lower().startswith("gt"))]
    return {"n_opened": len(opened), "n_unique": len(names),
            "work_files": sum(n.startswith("<work>/") for n in names),
            "clone_files": [n for n in names if n.startswith("$WVAD_OFFICIAL_ROOT/")],
            "clone_list_files": clone_list, "data_files_outside": outside,
            "test_like": test_like, "unique_sha256": _sha_text("\n".join(names)),
            "ok": not (clone_list or outside or test_like)}


class Recorder(vcv.VadCLIPVisualMethod):
    """The port with a per-iteration record of the training loss and lr (train_step)."""

    def __init__(self, base: vcv.VadCLIPVisualMethod) -> None:
        super().__init__(base.config, base.recipe)
        self.losses: list[float] = []
        self.lrs: list[float] = []

    def train_step(self, model, batch, optimizer, *, unit, scale):
        out = super().train_step(model, batch, optimizer, unit=unit, scale=scale)
        self.losses.append(float(out["total"]))
        self.lrs.append(float(optimizer.param_groups[0]["lr"]))
        return out


@dataclass(frozen=True)
class Q1bRun:
    dataset: str
    pair: str
    seed: int
    work: str
    device: str = "cuda"
    n_chunk: int = common.N_CHUNK_DEFAULT
    batch: int | None = None              # the self-test only (official --batch-size)
    epochs: int | None = None             # the self-test only (official --max-epoch)
    crops: int | None = None              # the self-test only (rows per video in both runs)


def _first_divergence(a: Sequence[float], b: Sequence[float]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0 else abs(a - b)


def official_val_auc(official: ModuleType, dataset: str, feature_dim: int,
                     state: Mapping[str, torch.Tensor], val: Sequence[torch.Tensor],
                     totals: Sequence[int], labels: Sequence[int], n_chunk: int,
                     device: torch.device) -> float:
    """sigma W-readout V-AUC of the official model (saved weights) through the official
    test() on the validation pieces (N_chunk rule, in memory)."""
    with official_vadclip(official, dataset) as ns:
        off = build_official(ns, feature_dim, vcv.OFFICIAL_UNIT, seed=None, device=device.type)
        load_into_official(off, state)
        off.to(device)
        z_list = []
        for x in val:
            a = x.numpy()
            pieces = [a[s:e] for s, e in common.chunk_bounds(a.shape[0], n_chunk)]
            got = official_test_logits(ns, off, pieces, vcv.OFFICIAL_UNIT, device)
            z_list.append(np.concatenate([g["logits"] for g in got]).astype(np.float32))
    vs = common.video_scores(z_list, totals)
    return float(common.v_aucs(vs["sigma"][None, :], labels)[0])


def run_q1b(official: ModuleType, run: Q1bRun, *, paths=None, split=None, log=print
            ) -> dict:
    """(q1b) one (dataset, pair, seed): the official pipeline (subprocess) and the port's chain
    on the same training rows (module docstring). The work folder must be new."""
    from wvad.baselines import train as tr
    from wvad.data import load_split, split_file

    if common.input_ln(run.pair):
        raise ValueError("(q1b) runs on XCB+XCL only (PLAN.md (q1b)); the official code has no "
                         "input LayerNorm")
    of = official
    of.check_commit(NAME)
    dev = torch.device(run.device)
    work = Path(run.work)
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"{work} is not empty: a (q1b) run needs a new folder")
    off_dir, port_dir = work / "official", work / "port"
    off_dir.mkdir(parents=True)
    method = vcv.METHODS[run.dataset]
    over = {k: v for k, v in (("batch", run.batch), ("epochs", run.epochs),
                              ("crops", run.crops)) if v is not None}
    if over:
        method = method.with_overrides(**over)
    if paths is None:
        from wvad.paths import load_paths

        paths = load_paths(None)
    split = split if split is not None else load_split(split_file(run.dataset))
    inputs = tr.load_inputs(paths, run.dataset, run.pair,
                            method.recipe.unit_input(vcv.OFFICIAL_UNIT), split)
    csv = write_list(off_dir / "data", run.dataset,
                     [f.numpy() for f in inputs.train.features], inputs.train.labels,
                     repeat=method.config.crops)
    cfg = vcv.OFFICIAL[run.dataset]
    script_path = of.repo_dir(NAME) / cfg.train_script
    official_text = script_path.read_text(encoding="utf-8")
    script, applied = modified_script(run.dataset, official_text)
    diff = script_diff(run.dataset, official_text, script)
    mod_path = off_dir / f"{Path(cfg.train_script).stem}_q1b.py"
    mod_path.write_text(script, encoding="utf-8", newline="\n")
    (off_dir / "official.diff").write_text(diff, encoding="utf-8", newline="\n")
    cmd = [sys.executable, str(mod_path), "--seed", str(run.seed), "--visual-width",
           str(inputs.dim), "--train-list", str(csv)]
    cmd += ["--batch-size", str(run.batch)] if run.batch is not None else []
    cmd += ["--max-epoch", str(run.epochs)] if run.epochs is not None else []
    env = dict(os.environ, PYTHONPATH=str(of.repo_dir(NAME) / "src"),
               CUBLAS_WORKSPACE_CONFIG=":4096:8", E023_Q1B_OUT=str(off_dir),
               KMP_DUPLICATE_LIB_OK="TRUE")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=off_dir, env=env, capture_output=True, text=True,
                         check=False)
    (off_dir / "stdout.txt").write_text(res.stdout, encoding="utf-8")
    (off_dir / "stderr.txt").write_text(res.stderr, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"official (q1b) run failed ({res.returncode}): {res.stderr[-2000:]}")
    t_off = time.time() - t0
    off_rec = json.loads((off_dir / "official_run.json").read_text(encoding="utf-8"))
    off_state = torch.load(off_dir / "official_last.pt", map_location="cpu", weights_only=True)
    recorder = Recorder(method)
    tcfg = tr.TrainConfig(method=NAME, unit=vcv.OFFICIAL_UNIT, dataset=run.dataset,
                          pair=run.pair, seed=run.seed, out_dir=str(port_dir),
                          deterministic=True, device=str(dev), n_chunk=run.n_chunk)
    t0 = time.time()
    try:
        port_rec = tr.run(tcfg, method=recorder, paths=paths, split=split, log=log)
    finally:
        common.deterministic(False)
    t_port = time.time() - t0
    port_state = torch.load(port_dir / tr.FILES["last"], map_location="cpu",
                            weights_only=True)["model_state_dict"]
    with np.load(port_dir / tr.FILES["val"]) as f:
        auc_port = float(f["v_auc_sigma"][-1])
    auc_off = official_val_auc(of, run.dataset, inputs.dim, off_state, inputs.val,
                               inputs.val_totals, inputs.val_labels, run.n_chunk, dev)
    lo, lp = off_rec["losses"], recorder.losses
    first = _first_divergence(lp, lo)
    weights = compare_named(port_state, off_state)
    portable = Portable(work, of.root())
    opened_file = off_dir / "opened.json"
    opened_file.write_text(json.dumps(sorted({portable(p) for p in off_rec["opened"]}),
                                      indent=0), encoding="utf-8")
    entry = {
        "dataset": run.dataset, "pair": run.pair, "seed": run.seed, "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "torch": torch.__version__, "overrides": over or None,
        "official": {"command": [portable(c) if os.path.isabs(c) else c for c in cmd],
                     "script": cfg.train_script, "script_sha256": _sha_text(official_text),
                     "modified_sha256": _sha_text(script), "diff_sha256": _sha_text(diff),
                     "edits": applied, "deterministic": off_rec["deterministic"],
                     "torch": off_rec["torch"], "seconds": round(t_off, 3),
                     "iterations": len(lo), "losses_sha256": _sha_floats(lo),
                     "audit": audit(off_rec["opened"], portable)},
        "port": {"out_dir": portable(port_dir), "iterations": len(lp),
                 "losses_sha256": _sha_floats(lp), "seconds": round(t_port, 3),
                 "selection_failed": port_rec["failed"],
                 "steps_per_epoch": port_rec["config"]["steps_per_epoch"],
                 "crops": method.config.crops,
                 "schedule": "official optimizer steps: an epoch = one pass over every "
                             "training video repeated crops times (both runs)"},
        "work": portable(work), "training_rows_sha256": _sha_text(json.dumps(
            [hashlib.sha256(f.numpy().tobytes()).hexdigest() for f in inputs.train.features])),
        "losses_bit_equal": lp == lo, "lrs_bit_equal": recorder.lrs == off_rec["lrs"],
        "first_divergence": first,
        "max_rel_first50": max((_rel(a, b) for a, b in zip(lp[:50], lo[:50], strict=False)),
                               default=0.0),
        "weights": {"bit_equal": weights["bit_equal"], "max_rel": weights["max_rel"],
                    "not_bit_equal": weights["not_bit_equal"]},
        "v_auc": {"official": auc_off, "port": auc_port, "abs_diff": abs(auc_off - auc_port)},
        "opened_list_sha256": hashlib.sha256(opened_file.read_bytes()).hexdigest(),
    }
    entry["criterion"], entry["pass"] = verdict(entry)
    return entry


def verdict(entry: Mapping) -> tuple[str, bool]:
    """PLAN.md "(q1b) 통과 기준" for one seed: bit equality of losses, lr, last weights and
    V-AUC; else the numeric rule (first 50 iterations relative difference <= 1e-5, |delta V-AUC|
    <= 1 point; the seed-mean rule is ``seed_mean_ok``). The audit must pass either way."""
    ok_audit = bool(entry["official"]["audit"]["ok"])
    same_len = entry["official"]["iterations"] == entry["port"]["iterations"]
    if entry["losses_bit_equal"] and entry["lrs_bit_equal"] and entry["weights"]["bit_equal"] \
            and entry["v_auc"]["abs_diff"] == 0.0:
        return "bit", ok_audit
    numeric = same_len and entry["lrs_bit_equal"] and entry["max_rel_first50"] <= TOL_REL_Q1B \
        and entry["v_auc"]["abs_diff"] * 100 <= VAUC_POINTS
    return "numeric (R3-6)", bool(numeric and ok_audit)


def seed_mean_ok(entries: Sequence[Mapping]) -> bool:
    """(q1b) numeric rule over the development seeds: mean |delta V-AUC| <= 0.5 point."""
    d = [e["v_auc"]["port"] - e["v_auc"]["official"] for e in entries]
    return bool(d) and abs(float(np.mean(d))) * 100 <= 0.5


def update_record(path: str | os.PathLike[str], entry: Mapping) -> dict:
    """Merge one run into results/e023_q1b.json (other methods' entries untouched)."""
    from wvad.data.dense_test import plan_sha256
    from wvad.diagnostics.common import code_version

    lic = json.loads(LICENCES_FILE.read_text(encoding="utf-8"))["repos"]
    plan = vcv_plan_file()
    run = {**entry, "plan_sha256": plan_sha256(plan) if plan.is_file() else None,
           "code": code_version(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def merge(rec: dict) -> None:
        rec.setdefault("methods", {})
        m = rec["methods"].setdefault(NAME, {"runs": {}})
        m["official"] = {"repo": "VadCLIP", "url": lic[vcv.REPO]["url"],
                         "commit": lic[vcv.REPO]["commit"],
                         "clone": "$WVAD_OFFICIAL_ROOT/VadCLIP (outside the repository)"}
        m["runs"][f"{entry['dataset']}/{entry['pair']}/{entry['seed']}"] = run

    return common.update_json(path, merge, default={
        "check": "E023 (q1b) official-pipeline comparison (PLAN.md 관문)", "methods": {}})


def vcv_plan_file() -> Path:
    return E023_DIR / "PLAN.md"


# --- command line ---------------------------------------------------------------------------

def _write(out: str | None, obj) -> None:
    text = json.dumps(obj, indent=1, default=str)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


def main(argv: Sequence[str] | None = None, *, official: ModuleType | None = None) -> dict:
    import argparse

    if official is None:
        raise SystemExit("check_vadclip_v needs the official loader passed in: python -c \"import "
                         "sys; from wvad.baselines import official, check_vadclip_v as c; "
                         "c.main(sys.argv[1:], official=official)\" <check> ...")
    of = official
    p = argparse.ArgumentParser(prog="check_vadclip_v",
                                description="VCV equivalence checks (q1, q1c, q1e, q1b)")
    p.add_argument("check", choices=("q1", "q1c", "q1e", "q1e_val", "q1b"))
    p.add_argument("--dataset", required=True, choices=sorted(vcv.OFFICIAL))
    p.add_argument("--pair", default="xcb_xcl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work", help="folder outside the repository (q1c, q1e, q1b)")
    p.add_argument("--n_chunk", type=int, default=common.N_CHUNK_DEFAULT)
    p.add_argument("--max_videos", type=int, help="q1e_val: the first K validation videos and "
                                                   "the longest (default: all)")
    p.add_argument("--paths", help="paths file (q1e_val, q1b)")
    p.add_argument("--record", help="results/e023_q1b.json to merge the (q1b) run into")
    p.add_argument("--out", help="write the result JSON here too")
    a = p.parse_args(argv)
    if a.check in ("q1c", "q1e", "q1b") and not a.work:
        p.error("--work is required")
    if a.check == "q1":
        common.deterministic(True)
        try:
            res = {"constructor": check_constructor(of, a.dataset, seed=a.seed),
                   "q1": check_q1(of, a.dataset, seed=a.seed),
                   "q1_32": check_q1(of, a.dataset, unit=vcv.X32_UNIT, seed=a.seed)
                   if a.dataset == "ucf" else None,
                   "full_forward_512": check_q1_full_forward(of, a.dataset, seed=a.seed)}
        finally:
            common.deterministic(False)
    elif a.check == "q1c":
        res = check_q1c(of, a.dataset, Path(a.work), seed=a.seed)
    elif a.check == "q1e":
        common.deterministic(True)
        try:
            res = {"plain": check_q1e(of, a.dataset, Path(a.work), n_chunk=a.n_chunk,
                                      seed=a.seed),
                   "input_ln": check_q1e(of, a.dataset, Path(a.work) / "ln",
                                         n_chunk=a.n_chunk, input_ln=True, seed=a.seed)}
        finally:
            common.deterministic(False)
    elif a.check == "q1e_val":
        from wvad.paths import load_paths

        common.deterministic(True)
        try:
            res = check_q1e_val(of, a.dataset, a.pair, paths=load_paths(a.paths),
                                n_chunk=a.n_chunk, seed=a.seed, max_videos=a.max_videos)
        finally:
            common.deterministic(False)
    else:
        from wvad.paths import load_paths

        res = run_q1b(of, Q1bRun(a.dataset, a.pair, a.seed, a.work, n_chunk=a.n_chunk),
                      paths=load_paths(a.paths))
        if a.record:
            update_record(a.record, res)
    _write(a.out, res)
    return res
