"""E019 (stage D, C05 pre-check): single-encoder heads (XCB, XCL, SigLIP) trained with the M10
pipeline, fused late (mean of their offset-including segment logits) and compared with the
concatenated M10 head (E017 runs) on the E018 threshold delta_d. The outcome is go / stop for the
C05 main experiment (E020) with pre-registered interpretation flags; nothing is adopted
(PLAN.md). Video labels only, no test data, development seeds only.

Logits are reconstructed in float64 from the two fp32 records by the sign of the offset: where
c(v) <= 0 the stored score sigma(z + c) is the probability farther from 1, so z' = logit(stored);
elsewhere z' = logit(sigma(z)) + c. Both arms (fusion and baseline) use this readout.

    python -m wvad.diagnostics.e019 --e017 <e017 copy> --runs <e019 copy>
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import spearmanr

from wvad.data import ENCODER_PAIRS, load_split, split_file
from wvad.diagnostics import e016
from wvad.diagnostics.common import DEV_SEEDS, check_dev_seed, code_version, write_csv, write_json
from wvad.diagnostics.e010 import check_selection, splits_sha256
from wvad.diagnostics.e013 import LAST, same_record_at
from wvad.diagnostics.e014 import GATE_ERRORS
from wvad.diagnostics.e015 import env_check
from wvad.diagnostics.e017 import lr_bound, r_end
from wvad.diagnostics.e018 import E018_RESULTS, e017_reference
from wvad.diagnostics.runs import PAIRS, selection
from wvad.diagnostics.stage_d import DATASETS, MC_LIMIT, R_LARGE, SETTINGS
from wvad.eval.crossfit import (
    METRICS,
    CrossFit,
    crossfit,
    rank_auc,
    select_epoch,
    stratified_half_splits,
    video_auc,
)
from wvad.paths import REPO_ROOT

E019_RESULTS = REPO_ROOT / "experiments" / "E019_encoder_late_fusion" / "results"
ARM = "m10"
ENCODERS = ("xcb", "xcl", "siglip")
DIM = {"xcb": 768, "xcl": 1024, "siglip": 1152}
INPUT_LN = {"xcb": False, "xcl": False, "siglip": True}
FUSED = {p: ENCODER_PAIRS[p] for p in PAIRS}           # setting pair -> (XCB, other encoder)
assert all(len(v) == 2 and v[0] == "xcb" for v in FUSED.values()), FUSED
IGNORED_CONFIG = ("seed", "pair", "out_dir", "init")
START_EPOCH, STAGE2_EPOCHS = 20, 30
OFFSET_LR_MULT = 10.0
P_LO, P_HI = float(np.finfo(np.float32).tiny), 1.0 - 2.0 ** -24
CHECK_LO, CHECK_HI, CHECK_TOL, CHECK_MIN_SHARE = 1e-30, 1 - 1e-3, 1e-3, 0.5   # gate (e)
NEAR_ONE = {"above_1-1e-4": 1 - 1e-4, "above_1-1e-6": 1 - 1e-6}
SEED_PAIRS = ((0, 1), (0, 2), (1, 2))
WEIGHTS = (0.5, 0.25, 0.75)              # candidate order: a tie keeps the fixed weight 1/2
READOUT_CHECK = 0.1                                      # |reconstructed - stored| / delta flag
REF_TOL = 1e-12
CURVE = {"stage1": (5, 10, 15, 20), "stage2": (10, 20, 30)}
V_AUC_ONLY = {"v_auc": video_auc}


# --- logit reconstruction -------------------------------------------------------------------

def logit(p: np.ndarray) -> np.ndarray:
    """float64 logit of fp32 probabilities clipped to [float32 tiny, 1 - 2^-24] (PLAN.md)."""
    q = np.clip(np.asarray(p, dtype=np.float64), P_LO, P_HI)
    return np.log(q) - np.log1p(-q)


def reconstruct(stored: np.ndarray, local: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """z' = z + c (float64): logit(stored) where c(v) <= 0, else logit(sigma(z)) + c."""
    c = np.asarray(offsets, dtype=np.float64)[..., None]
    return np.where(c <= 0, logit(stored), logit(local) + c)


def chosen_probability(stored: np.ndarray, local: np.ndarray,
                       offsets: np.ndarray) -> np.ndarray:
    """The fp32 probability each reconstructed logit comes from (saturation descriptive)."""
    return np.where(np.asarray(offsets)[..., None] <= 0, stored, local)


def record_consistency(stored: np.ndarray, local: np.ndarray, offsets: np.ndarray) -> dict:
    """Gate (e): where both probabilities lie in [1e-30, 1 - 1e-3], |logit(stored) -
    (logit(sigma(z)) + c)| <= 1e-3, on at least half of the entries."""
    s, p = np.asarray(stored), np.asarray(local)
    if s.shape != p.shape or np.shape(offsets) != s.shape[:2]:
        return {"ok": False, "reason": f"shapes {s.shape} / {p.shape} / {np.shape(offsets)}"}
    mask = (s >= CHECK_LO) & (s <= CHECK_HI) & (p >= CHECK_LO) & (p <= CHECK_HI)
    err = np.abs(logit(s) - logit(p) - np.asarray(offsets, dtype=np.float64)[..., None])[mask]
    worst = float(err.max()) if err.size else math.inf
    share = float(mask.mean())
    return {"ok": bool(share >= CHECK_MIN_SHARE and worst <= CHECK_TOL),
            "max_logit_error": worst, "share_checked": share}


# --- runs -----------------------------------------------------------------------------------

@dataclass
class Run:
    stored: np.ndarray            # (30, n, S) stored fp32 scores, stage-2 epochs 1-30
    local: np.ndarray             # (30, n, S) sigma(z), fp32
    offsets: np.ndarray           # (30, n) c(v)
    z: np.ndarray                 # (30, n, S) reconstructed z' (float64)
    history1: list
    history2: list
    s1_scores: np.ndarray         # (20, n, S) stage-1 stored scores (curves)
    selected: float
    epoch: int

    @property
    def free(self) -> np.ndarray:           # offset-free logits z = z' - c
        return self.z - self.offsets.astype(np.float64)[..., None]

    @property
    def gamma2(self) -> list[float]:
        return [h["offset_gamma"] for h in self.history2]


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(history: Sequence[Mapping]) -> bool:
    return all(math.isfinite(v) for h in history for k, v in h.items() if k.startswith("loss_"))


def load_run(d2: Path, labels: np.ndarray, val_idx: np.ndarray,
             split_sha: str) -> Run | str:
    """Checked stage-2 records (epochs 1-30) of one chain, or the reason it failed numerically."""
    d1 = d2.parent / f"{ARM}_stage1"
    z, o, z1 = (np.load(d2 / "record_val_scores.npz"), np.load(d2 / "record_val_offset.npz"),
                np.load(d1 / "record_val_scores.npz"))
    for rec, want in ((z, np.arange(0, STAGE2_EPOCHS + 1)), (o, np.arange(0, STAGE2_EPOCHS + 1)),
                      (z1, np.arange(1, START_EPOCH + 1))):
        if not np.array_equal(rec["epochs"], want):
            raise ValueError(f"{d2}: record epochs differ from {want[0]}-{want[-1]}")
        if not np.array_equal(rec["val_indices"], val_idx):
            raise ValueError(f"{d2}: validation order differs from the split file")
    sel = selection(d2)
    if sel["val_split_sha"] != split_sha:
        raise ValueError(f"{d2}: trained on split {sel['val_split_sha']}, not {split_sha}")
    h1, h2 = _json(d1 / "history.json"), _json(d2 / "history.json")
    keep = z["epochs"] >= 1
    stored, local, off = z["scores"][keep], o["local_scores"][keep], o["offsets"][keep]
    if not (all(np.isfinite(a).all() for a in (z["scores"], o["local_scores"], o["offsets"],
                                                 z1["scores"]))
            and _finite(h1) and _finite(h2)):
        return "non-finite loss or score"
    return Run(stored, local, off, reconstruct(stored, local, off), h1, h2, z1["scores"],
               check_selection(d2, stored, labels, sel), int(sel["best_epoch"]))


@dataclass
class Data:
    labels: np.ndarray
    single: dict                  # (encoder, seed) -> Run | failure reason
    base: dict                    # (pair, seed) -> Run


INFRA_FAILED = "infrastructure failure twice (PLAN.md)"


def load_dataset(e017: Path, runs: Path, dataset: str, infra_failed: Sequence[str] = ()) -> Data:
    """``infra_failed``: chains (e.g. ucf_xcl_s1) whose rerun also failed for infrastructure
    reasons; they count as failed runs like a numerical failure (PLAN.md)."""
    split = load_split(split_file(dataset))
    val_idx = np.asarray(split.val_indices)
    labels = np.array([split.labels[i] for i in val_idx])
    single = {(e, s): INFRA_FAILED if f"{dataset}_{e}_s{s}" in infra_failed else
              load_run(runs / f"{dataset}_{e}_s{s}" / ARM, labels, val_idx, split.sha256)
              for e in ENCODERS for s in DEV_SEEDS}
    base = {}
    for p in PAIRS:
        for s in DEV_SEEDS:
            run = load_run(e017 / f"{dataset}_{p}_s{s}" / ARM, labels, val_idx, split.sha256)
            if isinstance(run, str):
                raise ValueError(f"reused baseline {dataset}_{p}_s{s} failed: {run}")
            base[(p, s)] = run
    return Data(labels, single, base)


def check_infra_failed(runs: Path, names: Sequence[str]) -> dict:
    """--infra_failed (PLAN.md failure handling) accepts only chains of this experiment whose
    stage-2 output is missing; anything else stops, so no run is dropped after the results."""
    valid = {f"{d}_{e}_s{s}" for d in DATASETS for e in ENCODERS for s in DEV_SEEDS}
    out = {}
    for name in names:
        d2 = runs / name / ARM
        state = {f: (d2 / f).is_file() for f in ("selection.json", "record_val_scores.npz")}
        if name not in valid or all(state.values()):
            raise SystemExit(f"--infra_failed {name}: not an E019 chain, or its stage-2 output "
                             f"exists ({state}); PLAN.md failure handling")
        out[name] = {"files_present": state, "reason": INFRA_FAILED}
    return out


def failed_settings(data: Data) -> dict[str, list[str]]:
    """Settings whose encoders have a numerically failed run (PLAN.md failure handling)."""
    return {p: [f"{e}_s{s}" for e in FUSED[p] for s in DEV_SEEDS
                if isinstance(data.single[(e, s)], str)] for p in PAIRS}


# --- gates ----------------------------------------------------------------------------------

def _ckpt(d: Path) -> dict:
    return torch.load(d / LAST, map_location="cpu", weights_only=True)


def config_ok(cfg: Mapping, want: Mapping, encoder: str, seed: int) -> bool:
    """Gate (c), config part: pair = encoder, mean offset, multiplier 10, everything else equal
    to the E017 M10 run (seed 0) of the same dataset except seed / pair / out_dir / init."""
    keys = set(cfg) - set(IGNORED_CONFIG)
    return (cfg.get("pair") == encoder and int(cfg.get("seed", -1)) == seed
            and cfg.get("global_mode") == "mean" and cfg.get("offset_lr_mult") == OFFSET_LR_MULT
            and keys == set(want) - set(IGNORED_CONFIG) and all(cfg[k] == want[k] for k in keys))


def head_ok(state: Mapping[str, torch.Tensor], encoder: str) -> bool:
    """Gate (c), state part: the first linear layer takes the encoder's dimension; an input
    LayerNorm (context.0, 1-D) only with SigLIP."""
    w0, d = state.get("context.0.weight"), DIM[encoder]
    if w0 is None:
        return False
    if INPUT_LN[encoder]:
        w1 = state.get("context.1.weight")
        return tuple(w0.shape) == (d,) and w1 is not None and tuple(w1.shape) == (256, d)
    return tuple(w0.shape) == (256, d) and "context.1.weight" not in state


def order_ok(c: Path, val_idx: np.ndarray) -> bool:
    files = (c / f"{ARM}_stage1" / "record_val_scores.npz", c / ARM / "record_val_scores.npz",
             c / ARM / "record_val_offset.npz")
    return all(np.array_equal(np.load(f)["val_indices"], val_idx) for f in files)


def records_of(d2: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, o = np.load(d2 / "record_val_scores.npz"), np.load(d2 / "record_val_offset.npz")
    if not np.array_equal(z["epochs"], o["epochs"]):
        raise ValueError(f"{d2}: offset and score records cover different epochs")
    return z["scores"], o["local_scores"], o["offsets"]


def consistency_of(d2: Path) -> dict:
    """Gate (e) record check; a numerically failed record is a failed setting, not a gate
    failure (PLAN.md), so it is skipped here."""
    recs = records_of(d2)
    if not all(np.isfinite(a).all() for a in recs):
        return {"ok": True, "skipped": "non-finite record (numerical failure)"}
    return record_consistency(*recs)


def init_is_own(s1: Path, s2: Path) -> bool:
    """Gate (b): the stage-2 init is this chain's stage-1 ckpt_last.pt, compared as the full
    path recorded at training time (E018 code review G3)."""
    out1 = PurePosixPath(_ckpt(s1)["config"]["out_dir"])
    init = PurePosixPath(_ckpt(s2)["config"]["init"])
    return out1.parts[-2:] == (s1.parent.name, s1.name) and init == out1 / LAST


def selection_ok(d2: Path, labels: np.ndarray) -> bool:
    """Gate (g); a numerically failed run is handled as a failed setting, not here."""
    z = np.load(d2 / "record_val_scores.npz")
    sc = z["scores"][z["epochs"] >= 1]
    if not np.isfinite(sc).all():
        return True
    try:
        check_selection(d2, sc, labels, selection(d2))
    except ValueError:
        return False
    return True


def gate_chain(e017: Path, runs: Path, dataset: str, encoder: str, seed: int,
               labels: np.ndarray, val_idx: np.ndarray) -> dict:
    c = runs / f"{dataset}_{encoder}_s{seed}"
    s1, s2 = c / f"{ARM}_stage1", c / ARM
    ref = e017 / f"{dataset}_{PAIRS[0]}_s0"
    rec = consistency_of(s2)
    learned = e016.learned("g1a", e016.offsets_of(s1), e016.offsets_of(s2))
    c_ok = all(config_ok(_ckpt(x)["config"], _ckpt(ref / y)["config"], encoder, seed)
               and head_ok(_ckpt(x)["model_state_dict"], encoder)
               for x, y in ((s1, f"{ARM}_stage1"), (s2, ARM)))
    return {"a": same_record_at(s2, 0, s1, START_EPOCH),
            "b": init_is_own(s1, s2),
            "c": c_ok, "d": order_ok(c, val_idx),
            "e": learned and rec["ok"], "e_learned": learned,
            "g": selection_ok(s2, labels), "record_check": rec}


GATE_KEYS = ("a", "b", "c", "d", "e", "g")


def gates(e017: Path, runs: Path, identity: Path, infra_failed: Sequence[str] = ()) -> dict:
    rows, base_rows = [], []
    for d in DATASETS:
        split = load_split(split_file(d))
        val_idx = np.asarray(split.val_indices)
        labels = np.array([split.labels[i] for i in val_idx])
        for e in ENCODERS:
            for s in DEV_SEEDS:
                if f"{d}_{e}_s{s}" in infra_failed:
                    rows.append({"chain": f"{d}_{e}_s{s}", "excluded": INFRA_FAILED})
                    continue
                try:
                    row = {"chain": f"{d}_{e}_s{s}",
                           **gate_chain(e017, runs, d, e, s, labels, val_idx)}
                except GATE_ERRORS as exc:
                    row = {"chain": f"{d}_{e}_s{s}", "error": repr(exc)}
                rows.append(row)
        for p in PAIRS:
            for s in DEV_SEEDS:
                name = f"{d}_{p}_s{s}"
                try:
                    base_rows.append({"chain": name,
                                      **record_consistency(*records_of(e017 / name / ARM))})
                except GATE_ERRORS as exc:
                    base_rows.append({"chain": name, "ok": False, "error": repr(exc)})
    checked = [r for r in rows if "error" not in r and "excluded" not in r]
    status = {k: all(r.get(k, False) for r in checked) for k in GATE_KEYS}
    status["e"] = status["e"] and all(r["ok"] for r in base_rows)
    env = env_check(runs, identity)
    status["f"] = env["ok"]
    return {"chains": rows, "baseline_records": base_rows, "status": status,
            "environment": env, "errors": [r["chain"] for r in rows if "error" in r]}


# --- readouts and cross-fit -----------------------------------------------------------------

def fuse(za: np.ndarray, zb: np.ndarray, w: float = 0.5) -> np.ndarray:
    return w * za + (1 - w) * zb


def readout(z: np.ndarray) -> np.ndarray:
    return expit(z)


class Fitter:
    """Cross-fits on one dataset's splits (the E018 R, or R = 1000 after escalation)."""

    def __init__(self, labels: np.ndarray, n_splits: int):
        self.labels, self.R = labels, n_splits
        self.splits = stratified_half_splits(labels, n_splits, seed=0)
        self.sha = splits_sha256(self.splits)

    def full(self, scores: np.ndarray) -> CrossFit:
        return crossfit(scores, self.labels, self.splits)

    def v_auc(self, scores: np.ndarray) -> float:
        return crossfit(scores, self.labels, self.splits, V_AUC_ONLY).mean("v_auc")


def ok_seeds(data: Data, pair: str) -> bool:
    return not failed_settings(data)[pair]


def fused_z(data: Data, pair: str, sa: int, sb: int, w: float = 0.5) -> np.ndarray:
    a, o = FUSED[pair]
    return fuse(data.single[(a, sa)].z, data.single[(o, sb)].z, w)


def primary(data: Data, fit: Fitter) -> dict:
    """Cross-fits of the verdict readouts: fused (per setting and seed), baseline in both
    readouts (per pair and seed) and single heads (reconstructed readout)."""
    out = {"fused": {}, "base_logit": {}, "base_stored": {}, "single": {}, "single_stored": {}}
    for p in PAIRS:
        for s in DEV_SEEDS:
            out["base_logit"][(p, s)] = fit.full(readout(data.base[(p, s)].z))
            out["base_stored"][(p, s)] = fit.full(data.base[(p, s)].stored)
            if ok_seeds(data, p):
                out["fused"][(p, s)] = fit.full(readout(fused_z(data, p, s, s)))
    for (e, s), run in data.single.items():
        if not isinstance(run, str):
            out["single"][(e, s)] = fit.full(readout(run.z))
            out["single_stored"][(e, s)] = fit.full(run.stored)
    return out


def seed_mean(fits: Mapping, key: str, metric: str = "v_auc") -> float:
    return float(np.mean([fits[(key, s)].mean(metric) for s in DEV_SEEDS]))


# --- decision -------------------------------------------------------------------------------

def go_rule(delta: Mapping[str, float | None], thr: Mapping[str, float],
            weighted: Mapping[str, float | None] | None = None) -> dict:
    """PLAN.md: go if >= 2 settings have Delta >= delta_d and none <= -delta_d; a failed
    setting (None) counts in neither direction. Stop labels: one setting >= delta_d -> pattern
    mismatch; >= 2 but a setting <= -delta_d -> worse setting, or fixed-weight veto when the
    weight-selected fusion of every worse setting is > -delta_d (``weighted``: its Delta)."""
    live = {k: v for k, v in delta.items() if v is not None}
    hi = [k for k, v in live.items() if v >= thr[k.split("/")[0]]]
    lo = [k for k, v in live.items() if v <= -thr[k.split("/")[0]]]
    if len(hi) >= 2 and not lo:
        decision = "go"
    elif len(hi) == 1:
        decision = "stop_pattern_mismatch"
    elif len(hi) >= 2:
        veto = weighted is not None and all(
            weighted.get(k) is not None and weighted[k] > -thr[k.split("/")[0]] for k in lo)
        decision = "stop_fixed_weight_veto" if veto else "stop_worse_setting"
    else:
        decision = "stop"
    return {"decision": decision, "settings_ge_delta": hi, "settings_le_minus_delta": lo,
            "failed_settings": [k for k, v in delta.items() if v is None]}


def flags(rule: Mapping, fused: Mapping[str, float], controls: Mapping[str, Mapping],
          diversity: Mapping[str, Mapping], thr: Mapping[str, float],
          readout_sensitive: bool) -> dict:
    """Interpretation flags over the settings with Delta >= delta_d (PLAN.md)."""
    hi = rule["settings_ge_delta"]
    explicable = [k for k in hi
                  if fused[k] - controls[k]["R"] < thr[k.split("/")[0]] / 2]
    no_div = [k for k in hi                 # an undetermined value flags (conservative)
              if diversity[k].get("undetermined_rows", 0) > 0
              or not diversity[k]["cross_encoder"] > diversity[k]["same_encoder_reference"]]
    one_dataset = bool(hi) and len({k.split("/")[0] for k in hi}) == 1
    out = {"explicable_by_ensembling_or_single_encoder": explicable,
           "go_explicable": rule["decision"] == "go" and len(hi) - len(explicable) < 2,
           "no_encoder_diversity_evidence": no_div,
           "one_dataset_only": one_dataset,
           "readout_sensitive": readout_sensitive}
    out["user_confirmation_before_e020"] = bool(
        explicable or no_div or one_dataset or readout_sensitive
        or rule["decision"] == "stop_fixed_weight_veto")
    return out


# --- controls and descriptives ----------------------------------------------------------------

def pair_ensemble(fit: Fitter, z_of: Callable[[int], np.ndarray]) -> float:
    """Mean cross-fitted V-AUC of the 2-seed logit-average ensembles over SEED_PAIRS."""
    return float(np.mean([fit.v_auc(readout(fuse(z_of(a), z_of(b)))) for a, b in SEED_PAIRS]))


def controls(data: Data, fit: Fitter, prim: Mapping, pair: str) -> dict:
    """(i) concatenated 2-seed ensembles, (ii) same-encoder 2-seed ensembles, R = max of them,
    (iii) the better single encoder (seed mean, reconstructed readout)."""
    a, o = FUSED[pair]
    out = {"i_concat": pair_ensemble(fit, lambda s: data.base[(pair, s)].z),
           f"ii_{a}": pair_ensemble(fit, lambda s: data.single[(a, s)].z),
           f"ii_{o}": pair_ensemble(fit, lambda s: data.single[(o, s)].z)}
    out["R"] = max(out["i_concat"], out[f"ii_{a}"], out[f"ii_{o}"])
    singles = {e: seed_mean(prim["single"], e) for e in (a, o)}
    best = max(singles, key=singles.get)
    out |= {"iii_better_single": singles[best], "iii_encoder": best}
    return out


def full_epoch(z: np.ndarray, labels: np.ndarray) -> int:
    """Index of the epoch the logit readout selects on all validation videos."""
    return select_epoch(z.max(axis=2), labels, np.arange(len(labels)))[0]


def disagreement(za: np.ndarray, zb: np.ndarray, labels: np.ndarray) -> dict:
    """(n, S) logits of two heads at one epoch: |W_A xor W_B| / |W_A or W_B| over (anomalous,
    normal) pairs (W = pairs a head gets wrong: anomalous max <= normal max) and Spearman
    correlations of the video maxima within anomalous and within normal videos."""
    ma, mb = za.max(axis=1), zb.max(axis=1)
    pos, neg = labels == 1, labels == 0
    wa = ma[pos][:, None] <= ma[neg][None, :]
    wb = mb[pos][:, None] <= mb[neg][None, :]
    union = int((wa | wb).sum())
    return {"disagreement": float((wa ^ wb).sum() / union) if union else math.nan,
            "spearman_anomalous": float(spearmanr(ma[pos], mb[pos])[0]),
            "spearman_normal": float(spearmanr(ma[neg], mb[neg])[0])}


def pair_disagreement(za: np.ndarray, zb: np.ndarray, labels: np.ndarray) -> dict:
    """Both heads at the epoch chosen by their logit average on all validation videos."""
    e = full_epoch(fuse(za, zb), labels)
    return disagreement(za[e], zb[e], labels) | {"epoch": e + 1}


def _mean_rows(rows: Sequence[Mapping]) -> dict:
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "epoch"}


def diversity(data: Data, pair: str) -> dict:
    """Descriptive 3 and the diversity flag's inputs."""
    a, o = FUSED[pair]
    lab = data.labels
    cross = [pair_disagreement(data.single[(a, s)].z, data.single[(o, s)].z, lab)
             for s in DEV_SEEDS]
    same = {e: [pair_disagreement(data.single[(e, x)].z, data.single[(e, y)].z, lab)
                for x, y in SEED_PAIRS] for e in (a, o)}
    concat = [pair_disagreement(data.base[(pair, x)].z, data.base[(pair, y)].z, lab)
              for x, y in SEED_PAIRS]
    same_mean = {e: _mean_rows(v) for e, v in same.items()}
    refs = [v["disagreement"] for v in same_mean.values()]
    undetermined = sum(math.isnan(r["disagreement"]) for r in (*cross, *same[a], *same[o]))
    return {"cross_encoder_rows": cross, "cross_encoder_mean": _mean_rows(cross),
            "same_encoder_mean": same_mean, "concat_mean": _mean_rows(concat),
            "cross_encoder": _mean_rows(cross)["disagreement"],
            "same_encoder_reference": (float(np.nanmax(refs)) if not all(map(math.isnan, refs))
                                       else math.nan),
            "undetermined_rows": int(undetermined)}


def saturation(data: Data, pair: str) -> dict:
    """Per head: shares of the chosen fp32 probability at 1.0f and above 1 - 1e-4 / 1 - 1e-6;
    per seed at the fused readout's full-validation epoch: share of videos whose fused maximum
    segment has a 1.0f probability in either head."""
    a, o = FUSED[pair]
    heads = {}
    for e in (a, o):
        ps = [chosen_probability(r.stored, r.local, r.offsets)
              for r in (data.single[(e, s)] for s in DEV_SEEDS)]
        heads[e] = {"at_1": float(np.mean([np.mean(p == np.float32(1)) for p in ps])),
                    **{k: float(np.mean([np.mean(p > t) for p in ps]))
                       for k, t in NEAR_ONE.items()}}
    videos = []
    for s in DEV_SEEDS:
        ra, ro = data.single[(a, s)], data.single[(o, s)]
        zf = fused_z(data, pair, s, s)
        e = full_epoch(zf, data.labels)
        arg = zf[e].argmax(axis=1)
        rows = np.arange(len(arg))
        sat = [chosen_probability(r.stored[e], r.local[e], r.offsets[e])[rows, arg]
               == np.float32(1) for r in (ra, ro)]
        videos.append(float(np.mean(sat[0] | sat[1])))
    return {"heads": heads, "videos_fused_max_saturated": float(np.mean(videos))}


def decomposition(data: Data, fit: Fitter, pair: str, base: float, thr: float) -> dict:
    """Descriptive 5: (i) offset-free fusion, (ii) offset only, (iii) the verdict readout, and
    the shared-offset emulation (mean of the two gammas x video mean of the offset-free fusion);
    seed means and Delta/delta vs the baseline (reconstructed readout)."""
    a, o = FUSED[pair]
    vals = {"i_offset_free": [], "ii_offset_only": [], "iii_full": [], "shared_offset": []}
    for s in DEV_SEEDS:
        ra, ro = data.single[(a, s)], data.single[(o, s)]
        free = fuse(ra.free, ro.free)
        c = fuse(ra.offsets.astype(np.float64), ro.offsets.astype(np.float64))
        gamma = fuse(np.asarray(ra.gamma2), np.asarray(ro.gamma2))
        shared = free + (gamma[:, None] * free.mean(axis=2))[..., None]
        vals["i_offset_free"].append(fit.v_auc(readout(free)))
        vals["ii_offset_only"].append(fit.v_auc(c[..., None]))
        vals["iii_full"].append(fit.v_auc(readout(fuse(ra.z, ro.z))))
        vals["shared_offset"].append(fit.v_auc(readout(shared)))
    mean = {k: float(np.mean(v)) for k, v in vals.items()}
    return {"v_auc": mean, "delta_over_threshold": {k: (v - base) / thr for k, v in mean.items()}}


def other_fusions(data: Data, fit: Fitter, pair: str) -> dict:
    """Descriptive 6 (probability average; weight chosen with the epoch inside the cross-fit)
    and 7 (all 9 XCB seed x other seed combinations); seed means of cross-fitted V-AUC."""
    a, o = FUSED[pair]
    prob, weighted = [], []
    for s in DEV_SEEDS:
        za, zo = data.single[(a, s)].z, data.single[(o, s)].z
        prob.append(fit.v_auc((readout(za) + readout(zo)) / 2))
        weighted.append(fit.v_auc(np.concatenate([readout(fuse(za, zo, w)) for w in WEIGHTS])))
    combos = {(x, y): fit.v_auc(readout(fused_z(data, pair, x, y)))
              for x in DEV_SEEDS for y in DEV_SEEDS}
    return {"probability_average": float(np.mean(prob)),
            "weight_selected": float(np.mean(weighted)), "weight_selected_per_seed": weighted,
            "all_9_combinations": float(np.mean(list(combos.values()))),
            "matched_3": float(np.mean([combos[(s, s)] for s in DEV_SEEDS]))}


def full_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(rank_auc(scores.max(axis=1)[None], labels)[0])


def head_track(run: Run, dataset: str, labels: np.ndarray) -> dict:
    """Descriptive 9 for one single-head run."""
    g1 = {h["epoch"]: h["offset_gamma"] for h in run.history1}
    return {"gamma_stage1_end": g1[max(g1)], "gamma_stage2_end": run.gamma2[-1],
            "r_end": r_end(run.gamma2, dataset, ARM),
            "curve_stage1": {e: full_auc(run.s1_scores[e - 1], labels) for e in CURVE["stage1"]},
            "curve_stage2": {e: full_auc(run.stored[e - 1], labels) for e in CURVE["stage2"]},
            "amp_skipped_max": max(int(h["amp_skipped_steps"])
                                   for h in (*run.history1, *run.history2))}


def convergence(data: Data, pair: str, encoder: str, thr: float) -> dict:
    """Seed mean of (single - baseline) stage-1 rise from epoch 15 to 20 (full validation), in
    delta; |.| >= 1 -> convergence-speed confound not excluded (E015 rule)."""
    def rise(run: Run) -> float:
        return full_auc(run.s1_scores[19], data.labels) - full_auc(run.s1_scores[14], data.labels)
    v = float(np.mean([rise(data.single[(encoder, s)]) - rise(data.base[(pair, s)])
                       for s in DEV_SEEDS])) / thr
    return {"rise_difference_over_delta": v, "confound_not_excluded": abs(v) >= 1}


# --- analysis -------------------------------------------------------------------------------

def thresholds() -> dict:
    s = _json(E018_RESULTS / "e018_summary.json")
    return {d: {"sd": v["sd"], "delta": v["delta"], "R": int(v["n_splits"]),
                "splits_sha256": v["splits_sha256"]} for d, v in s["datasets"].items()}


def reference_ok(data: Data, fit: Fitter, dataset: str) -> bool:
    """Gate (h): the reused baseline's stored scores reproduce e017_runs.csv (first R)."""
    ref = e017_reference(dataset)
    for p in PAIRS:
        for s in DEV_SEEDS:
            cf = fit.full(data.base[(p, s)].stored)
            if not all(abs(cf.mean(m) - ref[(p, s)][m]) <= REF_TOL for m in METRICS):
                return False
    return True


def fit_primary(data: Data, cal: Mapping) -> tuple[Fitter, dict, dict]:
    """Cross-fit at the E018 R; MC rule: escalate every readout of the dataset to R = 1000."""
    fit = Fitter(data.labels, cal["R"])
    if fit.sha != cal["splits_sha256"]:
        raise ValueError(f"splits {fit.sha} differ from E018 {cal['splits_sha256']}")
    prim = primary(data, fit)
    sd = cal["sd"]["v_auc"]
    worst = max((cf.mc_se("v_auc") for cf in prim["fused"].values()), default=0.0) / sd
    mc = {"first_R": cal["R"], "first_mc_se_over_sd": worst, "escalated": False}
    if worst > MC_LIMIT and cal["R"] < R_LARGE:
        big = Fitter(data.labels, R_LARGE)
        if splits_sha256(big.splits[:cal["R"]]) != cal["splits_sha256"]:
            raise ValueError("first splits of the larger set differ from E018's")
        fit, prim = big, primary(data, big)
        mc |= {"escalated": True, "R": R_LARGE, "splits_sha256": fit.sha,
               "mc_se_over_sd": max(cf.mc_se("v_auc") for cf in prim["fused"].values()) / sd}
        mc["exceeds_at_max_R"] = mc["mc_se_over_sd"] > MC_LIMIT      # recorded only
    elif worst > MC_LIMIT:
        mc["exceeds_at_max_R"] = True                                 # recorded only
    return fit, prim, mc


def setting_values(data: Data, prim: Mapping, pair: str) -> dict:
    vals = {"base_logit": {m: seed_mean(prim["base_logit"], pair, m) for m in METRICS},
            "base_stored": {m: seed_mean(prim["base_stored"], pair, m) for m in METRICS}}
    if ok_seeds(data, pair):
        vals["fused"] = {m: seed_mean(prim["fused"], pair, m) for m in METRICS}
        vals["fused_per_seed"] = [prim["fused"][(pair, s)].mean("v_auc") for s in DEV_SEEDS]
    return vals


def setting_descriptives(data: Data, fit: Fitter, prim: Mapping, pair: str, v: Mapping,
                         cal: Mapping) -> dict:
    thr = cal["delta"]["v_auc"]
    base, fused = v["base_logit"]["v_auc"], v["fused"]["v_auc"]
    ctrl = controls(data, fit, prim, pair)
    return {
        "delta": fused - base,
        "delta_stored_baseline": fused - v["base_stored"]["v_auc"],
        "aux_over_threshold": {m: (v["fused"][m] - v["base_logit"][m]) / cal["delta"][m]
                               for m in METRICS},
        "controls": ctrl,
        "fused_minus_controls_over_delta": {k: (fused - x) / thr for k, x in ctrl.items()
                                            if isinstance(x, float)},
        "diversity": diversity(data, pair),
        "saturation": saturation(data, pair),
        "decomposition": decomposition(data, fit, pair, base, thr),
        "other_fusions": other_fusions(data, fit, pair),
        "fused_seed_sd_over_sd_d": float(np.std(v["fused_per_seed"], ddof=1))
        / cal["sd"]["v_auc"],
    }


def single_entry(prim: Mapping, encoder: str, v: Mapping, thr: float) -> dict:
    """Descriptive 1: a single head's seed-mean V-AUC in both readouts, each against the
    baseline in the same readout."""
    logit_v = seed_mean(prim["single"], encoder)
    stored_v = seed_mean(prim["single_stored"], encoder)
    return {"v_auc": logit_v, "delta_over_threshold": (logit_v - v["base_logit"]["v_auc"]) / thr,
            "v_auc_stored": stored_v,
            "delta_over_threshold_stored": (stored_v - v["base_stored"]["v_auc"]) / thr}


def analyze_dataset(dataset: str, data: Data, cal: Mapping) -> dict:
    fit, prim, mc = fit_primary(data, cal)
    thr = cal["delta"]["v_auc"]
    out = {"R": fit.R, "splits_sha256": fit.sha, "mc": mc, "settings": {}, "prim": prim}
    for p in PAIRS:
        v = setting_values(data, prim, p)
        check = (v["base_logit"]["v_auc"] - v["base_stored"]["v_auc"]) / thr
        entry = {"values": v, "readout_check_over_delta": check,
                 "readout_check_flag": abs(check) > READOUT_CHECK,
                 "single": {e: single_entry(prim, e, v, thr) for e in FUSED[p]
                            if all((e, s) in prim["single"] for s in DEV_SEEDS)}}
        if "fused" in v:
            entry |= setting_descriptives(data, fit, prim, p, v, cal)
        out["settings"][f"{dataset}/{p}"] = entry
    return out


def decide(settings: Mapping[str, dict], cal: Mapping) -> dict:
    thr = {d: cal[d]["delta"]["v_auc"] for d in DATASETS}
    keys = [f"{d}/{p}" for d, p in SETTINGS]
    delta = {k: settings[k].get("delta") for k in keys}
    stored = {k: settings[k].get("delta_stored_baseline") for k in keys}
    def weighted_vs(base: str) -> dict:
        return {k: (settings[k]["other_fusions"]["weight_selected"]
                    - settings[k]["values"][base]["v_auc"])
                if "other_fusions" in settings[k] else None for k in keys}
    weighted = weighted_vs("base_logit")
    rule = go_rule(delta, thr, weighted)
    rule_stored = go_rule(stored, thr, weighted_vs("base_stored"))
    live = [k for k in keys if delta[k] is not None]
    fl = flags(rule, {k: settings[k]["values"]["fused"]["v_auc"] for k in live},
               {k: settings[k]["controls"] for k in live},
               {k: settings[k]["diversity"] for k in live}, thr,
               (rule["decision"] == "go") != (rule_stored["decision"] == "go"))
    fl["stored_label_differs"] = rule["decision"] != rule_stored["decision"]   # recorded only

    def over(x: Mapping[str, float | None]) -> dict:
        return {k: (v / thr[k.split("/")[0]] if v is not None else None) for k, v in x.items()}
    return {"rule": rule, "rule_stored_baseline": rule_stored, "flags": fl,
            "deltas": delta, "deltas_over_threshold": over(delta),
            "deltas_stored_baseline_over_threshold": over(stored),
            "weight_selected_delta_over_threshold": over(weighted),
            "readout_check_flags": [k for k in keys if settings[k]["readout_check_flag"]]}


def head_descriptives(data: Mapping[str, Data], cal: Mapping) -> dict:
    out = {}
    for d in DATASETS:
        thr = cal[d]["delta"]["v_auc"]
        for e in ENCODERS:
            runs = {s: data[d].single[(e, s)] for s in DEV_SEEDS}
            if any(isinstance(r, str) for r in runs.values()):
                out[f"{d}/{e}"] = {"failed": [s for s, r in runs.items() if isinstance(r, str)]}
                continue
            tracks = {s: head_track(r, d, data[d].labels) for s, r in runs.items()}
            out[f"{d}/{e}"] = {
                "per_seed": tracks, "lr_bound": lr_bound(list(tracks.values())),
                "convergence_vs_baseline": {p: convergence(data[d], p, e, thr)
                                            for p in PAIRS if e in FUSED[p]}}
    return out


def run_rows(dataset: str, data: Data, prim: Mapping) -> list[dict]:
    rows = []

    def row(kind: str, name: str, seed: int, cf: CrossFit, extra: Mapping | None = None):
        rows.append({"dataset": dataset, "kind": kind, "name": name, "seed": seed,
                     "n_splits": len(cf.epochs), **{m: cf.mean(m) for m in METRICS},
                     "mc_se_v_auc": cf.mc_se("v_auc"), "tie_rate": cf.tie_rate, **(extra or {})})
    for (p, s), cf in prim["base_stored"].items():
        row("baseline_stored", p, s, cf)
    for (p, s), cf in prim["base_logit"].items():
        row("baseline_reconstructed", p, s, cf)
    for (e, s), cf in prim["single"].items():
        r = data.single[(e, s)]
        row("single", e, s, cf, {"selected_epoch": r.epoch, "selected_v_auc": r.selected,
                                 "gamma_stage2_end": r.gamma2[-1]})
    for (e, s), cf in prim["single_stored"].items():
        row("single_stored", e, s, cf)
    for (p, s), cf in prim["fused"].items():
        row("fused", p, s, cf)
    for (e, s), reason in data.single.items():
        if isinstance(reason, str):
            rows.append({"dataset": dataset, "kind": "single", "name": e, "seed": s,
                         "failed": reason})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="E019 C05 pre-check: single heads and late fusion")
    ap.add_argument("--e017", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--infra_failed", nargs="*", default=[],
                    help="chains whose rerun also failed for infrastructure reasons, e.g. "
                         "ucf_xcl_s1 (PLAN.md failure handling: counted as failed runs)")
    args = ap.parse_args(argv)
    for s in DEV_SEEDS:
        check_dev_seed(s)
    infra = check_infra_failed(args.runs, args.infra_failed)
    gate = gates(args.e017, args.runs, E019_RESULTS / "e019_identity.json", args.infra_failed)
    gate["infra_failed"] = infra
    write_json(E019_RESULTS / "e019_gate.json", {"code": code_version(), **gate})
    if gate["errors"]:
        raise SystemExit(f"runs missing or unreadable for {gate['errors']}; see e019_gate.json")
    if not all(gate["status"].values()):
        raise SystemExit("gate failed (PLAN.md failure handling); see results/e019_gate.json")
    cal = thresholds()
    data = {d: load_dataset(args.e017, args.runs, d, args.infra_failed) for d in DATASETS}
    gate["status"]["h"] = all(reference_ok(data[d], Fitter(data[d].labels, cal[d]["R"]), d)
                              for d in DATASETS)                 # first R, before the analysis
    write_json(E019_RESULTS / "e019_gate.json", {"code": code_version(), **gate})
    if not gate["status"]["h"]:
        raise SystemExit("gate (h) failed: the reused M10 runs differ from e017_runs.csv")
    analyses = {d: analyze_dataset(d, data[d], cal[d]) for d in DATASETS}
    settings = {k: v for a in analyses.values() for k, v in a["settings"].items()}
    decision = decide(settings, cal)
    write_json(E019_RESULTS / "e019_summary.json", {
        "check": "E019 C05 pre-check: single-encoder heads and late fusion (PLAN.md)",
        "code": code_version(), "gate": gate["status"],
        "thresholds": {d: {"delta": cal[d]["delta"], "sd": cal[d]["sd"]} for d in DATASETS},
        "failed_settings": {d: failed_settings(data[d]) for d in DATASETS},
        "infra_failed": list(args.infra_failed),
        "cross_fit": {d: {"R": a["R"], "splits_sha256": a["splits_sha256"], "mc": a["mc"]}
                      for d, a in analyses.items()},
        **decision, "settings": settings, "heads": head_descriptives(data, cal)})
    write_csv(E019_RESULTS / "e019_runs.csv",
              [r for d in DATASETS for r in run_rows(d, data[d], analyses[d]["prim"])])
    print(decision["rule"]["decision"],
          {k: (round(v, 2) if v is not None else None)
           for k, v in decision["deltas_over_threshold"].items()}, decision["flags"])


if __name__ == "__main__":
    main()
