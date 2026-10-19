"""E023 pre-freeze measurements (PLAN.md "고정 전 측정 항목" (3f), (3g), (3j); D5, D6, D7
re-question). Validation and synthetic data only: this module never reads a test file (no test
manifest, store, score or annotation; it imports none of the modules that do, unit-tested in a
fresh interpreter). New code (add-only rule); the comparison methods only through the registry.

    (3f) spread     UCF validation anomalous videos with a UCA proxy interval: per video the frame
                    AUC (UCA proxy labels) of each development chain's selected checkpoint (the
                    C type: E021 HB M10 seeds 0-2, its W-readout epoch with gamma; a port: seeds
                    0-2 XCB+XCL, its sigma-best point), the per-video difference port - C averaged
                    over the seeds, set to mean 0 per method; output only its SD and the 99%
                    bootstrap half-width of the mean of 140 videos (R 5000, default_rng(20261019)),
                    and the port's seed SD of the proxy (mean, sign and level never output)
    (3g) power      the true advantage over the strongest comparison method needed for a joint
                    probability 0.8 under D5 (a), (b), (c) (Welch, Holm over the 8 contrasts /
                    over B*'s 2, alpha 0.01, n 10 vs 10), C's SD measured (UCF 0.090 / XD 0.170)
                    and with zero pair correlation (0.153 / 0.136), the comparison SDs (UCF proxy
                    x {0.5, 1, 2} and the grid {0.3, 1, 2, 3, 5}; XD the grid only), without and
                    with the 99% bootstrap lower-bound requirement (test-sample deviation
                    N(0, (h / 2.576)^2), lower bound D - h; UCF h from (3f), XD h in {1, 2, 3});
                    scenario: the strongest method distinct, the other three 1, 2, 3 points
                    further behind with SDs 0.2, 0.5, 1.0. The designated computation of the D5
                    recommendation (확인 N4-1, E5-4): per dataset the single-dataset required
                    advantage (the other dataset's contrasts rejected first in Holm's order; (b):
                    B* of this dataset alone at alpha), with the bootstrap requirement, UCF SD =
                    the median of the four proxy SDs and h = the median of the four (3f) half-
                    widths, XD SD 2 and h 2, C's measured SD; "(b)" is recommended when (a) needs
                    >= 0.5 L points more than (b) in any dataset. Output: SDs, half-widths and
                    required advantages only
    (3j) memory     per method and unit the peak forward memory (torch.cuda.max_memory_allocated)
                    of a synthetic sequence of the access (0) maximum n_v and of the longest
                    validation video, scored by common.score_video at N_chunk 4096, lowered by
                    powers of two until it fits; N_chunk = the largest value fitting every method,
                    and the maximum bytes (gate (m))

    python -m wvad.diagnostics.e023_prefreeze spread --e021_scores DIR --chains mil=D0,D1,D2 ...
    python -m wvad.diagnostics.e023_prefreeze power [--n_sim 3000]
    python -m wvad.diagnostics.e023_prefreeze memory --access0 FILE [--methods ...]
each merged into results/e023_prefreeze.json (keys "3f", "3g", "3j").
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from wvad.baselines import common as bc
from wvad.eval import frames as fr
from wvad.eval.crossfit import rank_auc
from wvad.eval.dense_readout import readout
from wvad.extract import grid
from wvad.paths import REPO_ROOT

RESULTS = REPO_ROOT / "experiments" / "E023_localisation_baselines" / "results"
OUT = RESULTS / "e023_prefreeze.json"
PROXY_FILE = REPO_ROOT / "data" / "annotations" / "ucf_train_uca_proxy.json"
DEV_SEEDS = (0, 1, 2)
TEST_VIDEOS_UCF = 140
BOOT_R, BOOT_SEED = 5000, 20261019
ALPHA = 0.01
N_SEEDS = 10
TARGET = 0.8
Z99 = 2.576
C_SD = {"measured": {"ucf": 0.090, "xdv": 0.170}, "corr0": {"ucf": 0.153, "xdv": 0.136}}
SD_GRID = (0.3, 1.0, 2.0, 3.0, 5.0)
PROXY_SCALE = (0.5, 1.0, 2.0)
XD_HALF = (1.0, 2.0, 3.0)
OTHERS = ((1.0, 0.2), (2.0, 0.5), (3.0, 1.0))      # (further behind, SD) of the other three
MARGIN_L = {"ucf": 0.5, "xdv": 2.0}                # D6 (a): for option (c)
DESIGNATED_XD = {"sd": 2.0, "half_width": 2.0}
RECOMMEND_GAP = 0.5
N_CHUNK_CANDIDATES = (4096, 2048, 1024, 512, 256, 128, 64, 32)
DELTA_GRID = tuple(round(0.1 * i, 10) for i in range(0, 151))


def _merge(path: Path, key: str, value) -> None:
    p = Path(path)
    rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rec[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")


# --- (3f) spread ------------------------------------------------------------------------------

def split_rows(flat: np.ndarray, counts: Sequence[int]) -> list[np.ndarray]:
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return [np.asarray(flat[a:a + n]) for a, n in zip(starts, counts, strict=True)]


def c_type_epoch(dense_z: np.ndarray, counts: Sequence[int], totals: Sequence[int],
                 labels: Sequence[int], gamma: Sequence[float] | None) -> int:
    """0-based epoch of E022 C's rule on an E021 HB M10 validation file: the highest
    full-validation V-AUC of the W readout with c(v) (video score max fp32 sigma), earliest on
    ties."""
    scores = np.empty((len(dense_z), len(counts)))
    for e in range(len(dense_z)):
        g = None if gamma is None else float(gamma[e])
        for i, z in enumerate(split_rows(dense_z[e], counts)):
            scores[e, i] = readout(z, int(totals[i]), g, "W").video_score
    i, _ = bc.select_point(rank_auc(scores, np.asarray(labels, dtype=np.int64)))
    return i


def video_aucs(z_list: Sequence[np.ndarray], totals: Sequence[int],
               frame_labels: Mapping[int, np.ndarray]) -> dict[int, float]:
    """Frame AUC (points) of each video with proxy labels holding both classes: the snippet
    logits repeated over their supports (per-video ranks: an offset c(v) changes nothing)."""
    out = {}
    for i, lab in frame_labels.items():
        lab = np.asarray(lab)
        if lab.min() == lab.max():
            continue
        out[i] = fr.video_metric(fr.snippet_frames(z_list[i], int(totals[i])), lab.astype(int),
                                 "auc")
    return out


def proxy_labels(names: Sequence[str], labels: Sequence[int], totals: Sequence[int], proxy,
                 variant: str) -> dict[int, np.ndarray]:
    """Validation position -> UCA proxy frame labels of the anomalous videos with an interval."""
    return {i: proxy.frame_labels(names[i], variant, int(totals[i])) for i in range(len(names))
            if int(labels[i]) == 1 and proxy.has(names[i], variant)}


def centred_spread(diffs: Sequence[float], n_test: int = TEST_VIDEOS_UCF,
                   n_resamples: int = BOOT_R, seed: int = BOOT_SEED) -> dict:
    """SD of the mean-zeroed per-video differences and the 99% percentile half-width of the
    bootstrap mean of ``n_test`` videos drawn from them (only spreads are returned)."""
    x = np.asarray(diffs, dtype=np.float64)
    if len(x) < 2:
        return {"n_videos": len(x), "sd": math.nan, "half_width_99": math.nan}
    c = x - x.mean()
    rng = np.random.default_rng(seed)
    means = c[rng.integers(0, len(c), size=(n_resamples, n_test))].mean(axis=1)
    lo, hi = np.percentile(means, [0.5, 99.5])
    return {"n_videos": len(x), "sd": float(c.std(ddof=1)), "n_test": n_test,
            "half_width_99": float((hi - lo) / 2), "R": n_resamples, "seed": seed}


def spread(c_runs: Sequence[Mapping], ports: Mapping[str, Sequence[Mapping]],
           frame_labels: Mapping[int, np.ndarray], totals: Sequence[int]) -> dict:
    """(3f) from validation logits: ``c_runs`` per development seed {"z": [per-video snippet
    logits]} of the C type, ``ports`` method -> the same per seed; returns per method the
    spread of the seed-mean differences (mean 0) and the port's seed SD of the proxy L."""
    c_auc = [video_aucs(r["z"], totals, frame_labels) for r in c_runs]
    out = {}
    for m, runs in ports.items():
        p_auc = [video_aucs(r["z"], totals, frame_labels) for r in runs]
        common = sorted(set.intersection(*(set(a) for a in (*c_auc, *p_auc))))
        diffs = [float(np.mean([p[v] - c[v] for p, c in zip(p_auc, c_auc, strict=True)]))
                 for v in common]
        seed_l = [float(np.mean([p[v] for v in common])) for p in p_auc] if common else []
        out[m] = {**centred_spread(diffs),
                  "proxy_seed_sd": float(np.std(seed_l, ddof=1)) if len(seed_l) > 1
                  else math.nan, "seeds": len(runs)}
    return out


def e021_c_runs(e021_scores: Path, dataset: str = "ucf", pair: str = "xcb_xcl",
                seeds: Sequence[int] = DEV_SEEDS) -> tuple[list[dict], dict]:
    """The C type's development runs from the E021 stage-2 score files (hb_m10)."""
    runs, meta = [], {}
    for s in seeds:
        f = Path(e021_scores) / "stage2" / f"{dataset}_{pair}_s{s}" / "hb_m10.npz"
        with np.load(f) as sf:
            counts, totals = sf["dense_counts"], sf["total_frames"]
            e = c_type_epoch(sf["dense_z"], counts, totals, sf["labels"], sf["gamma"])
            runs.append({"z": split_rows(sf["dense_z"][e], counts), "epoch": e + 1})
            meta = {"val_indices": [int(x) for x in sf["val_indices"]],
                    "totals": [int(x) for x in totals], "labels": [int(x) for x in sf["labels"]]}
    return runs, meta


def port_runs(dirs: Sequence[Path]) -> list[dict]:
    """A port's development chains: the sigma-best point's validation logits (val_scores.npz,
    selection.json of wvad.baselines.train)."""
    out = []
    for d in dirs:
        sel = json.loads((Path(d) / "selection.json").read_text(encoding="utf-8"))
        point = int(sel["selection"]["sigma"]["point"])
        with np.load(Path(d) / "val_scores.npz") as z:
            out.append({"z": split_rows(z["z"][point - 1], z["counts"]), "point": point,
                        "val_indices": [int(x) for x in z["val_indices"]]})
    return out


# --- (3g) power ---------------------------------------------------------------------------------

def _welch(a: np.ndarray, b: np.ndarray, shift: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised two-sided Welch over the last axis: (mean difference [+ shift], p)."""
    n_a, n_b = a.shape[-1], b.shape[-1]
    va, vb = a.var(axis=-1, ddof=1) / n_a, b.var(axis=-1, ddof=1) / n_b
    d = a.mean(axis=-1) - b.mean(axis=-1)
    if shift is not None:
        d = d + shift
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1))
    p = 2 * stats.t.sf(np.abs(d / se), df)
    return d, p


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p over the last axis (vectorised e020_stats.holm)."""
    m = p.shape[-1]
    order = np.argsort(p, axis=-1, kind="stable")
    ps = np.take_along_axis(p, order, axis=-1)
    adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * ps), axis=-1)
    out = np.empty_like(adj)
    np.put_along_axis(out, order, adj, axis=-1)
    return out


class Draws:
    """Common random numbers of one simulation: standard normal seed values (sims, datasets,
    5 arms, n) and test-sample deviations (sims, datasets, 4 contrasts)."""

    def __init__(self, n_sim: int, seed: int = BOOT_SEED, datasets: int = 2):
        rng = np.random.default_rng(seed)
        self.seeds = rng.standard_normal((n_sim, datasets, 5, N_SEEDS))
        self.test = rng.standard_normal((n_sim, datasets, 4))


def met_probabilities(delta: Sequence[float], c_sd: Sequence[float], b_sd: Sequence[float],
                      half: Sequence[float] | None, draws: Draws, *,
                      fixed: Sequence[bool] | None = None, alpha: float = ALPHA,
                      margin: Sequence[float] = (0.5, 2.0)) -> dict[str, np.ndarray]:
    """Per dataset (axis 1) whether D5 (a), (b), (c) is met in every simulation: C (SD
    ``c_sd``), the strongest method ``delta`` behind (SD ``b_sd``), the others 1, 2, 3 points
    further (SD 0.2, 0.5, 1.0); Welch, Holm over the 8 contrasts (a, c; and S_d != 3 of b) and
    over B*'s 2 (b); with ``half`` the test-sample deviation and the lower-bound requirement.
    ``fixed[d]``: dataset d is taken as met, its p-values 0 (rejected first in Holm's order)."""
    nd = draws.seeds.shape[1]
    gaps = np.array([[delta[d], *(delta[d] + g for g, _ in OTHERS)] for d in range(nd)])
    sds = np.array([[b_sd[d], *(s for _, s in OTHERS)] for d in range(nd)])
    c = draws.seeds[:, :, 0, :] * np.asarray(c_sd)[None, :, None]
    b = -gaps[None, :, :, None] + draws.seeds[:, :, 1:, :] * sds[None, :, :, None]
    shift = None if half is None else draws.test * (np.asarray(half) / Z99)[None, :, None]
    d, p = _welch(c[:, :, None, :], b, shift)                  # (sims, datasets, 4)
    fx = np.asarray(fixed if fixed is not None else [False] * nd, dtype=bool)
    p = np.where(fx[None, :, None], 0.0, p)
    ph = _holm(p.reshape(p.shape[0], -1)).reshape(p.shape)
    sig = ph < alpha
    lower_ok = np.ones_like(sig) if half is None else (d - np.asarray(half)[None, :, None]) > 0
    pos = sig & (d > 0)
    neg = sig & (d < 0)
    sub = pos & (d >= np.asarray(margin)[None, :, None])
    a_met = pos.all(axis=-1) & lower_ok.all(axis=-1)
    c_met = sub.all(axis=-1) & lower_ok.all(axis=-1)
    bstar = np.argmax(b.mean(axis=-1), axis=-1)               # the strongest by its seed mean
    take = bstar[..., None]
    db = np.take_along_axis(d, take, axis=-1)[..., 0]
    pb = np.take_along_axis(np.where(fx[None, :, None], 1.0, p), take, axis=-1)[..., 0]
    pb = np.where(fx[None, :], 0.0, pb)
    pbh = _holm(pb)
    lb = np.ones_like(db, dtype=bool) if half is None else \
        (db - np.asarray(half)[None, :]) > 0
    b_met = (pbh < alpha) & (db > 0) & lb & ~neg.any(axis=-1)
    met = {"a": a_met, "b": b_met, "c": c_met}
    return {k: np.where(fx[None, :], True, v) for k, v in met.items()}


def required(prob: Callable[[float], Mapping[str, float]], grid_: Sequence[float] = DELTA_GRID,
             target: float = TARGET) -> dict[str, float]:
    """The smallest grid advantage whose probability reaches ``target``, per option (NaN when
    none does)."""
    out: dict[str, float] = {}
    for x in grid_:
        pr = prob(float(x))
        for k, v in pr.items():
            if k not in out and v >= target:
                out[k] = float(x)
        if len(out) == len(pr):
            break
    return {k: out.get(k, math.nan) for k in ("a", "b", "c")}


def joint_required(c_sd: Mapping[str, float], b_sd: Mapping[str, float],
                   half: Mapping[str, float] | None, draws: Draws,
                   grid_: Sequence[float] = DELTA_GRID) -> dict[str, float]:
    """Both datasets met (D5 (i)) with the same true advantage in both."""
    ds = ("ucf", "xdv")

    def prob(x: float) -> dict[str, float]:
        met = met_probabilities([x, x], [c_sd[d] for d in ds], [b_sd[d] for d in ds],
                                None if half is None else [half[d] for d in ds], draws,
                                margin=[MARGIN_L[d] for d in ds])
        return {k: float(v.all(axis=1).mean()) for k, v in met.items()}
    return required(prob, grid_)


def single_required(dataset: str, c_sd: float, b_sd: float, half: float | None, draws: Draws,
                    grid_: Sequence[float] = DELTA_GRID) -> dict[str, float]:
    """One dataset met, the other taken as met (its contrasts rejected first)."""
    i = ("ucf", "xdv").index(dataset)

    def prob(x: float) -> dict[str, float]:
        delta, sds, cs, hs = [x, x], [b_sd, b_sd], [c_sd, c_sd], [half or 0.0] * 2
        fixed = [j != i for j in range(2)]
        met = met_probabilities(delta, cs, sds, None if half is None else hs, draws,
                                fixed=fixed, margin=[MARGIN_L["ucf"], MARGIN_L["xdv"]])
        return {k: float(v[:, i].mean()) for k, v in met.items()}
    return required(prob, grid_)


def power(spread_rec: Mapping[str, Mapping], *, n_sim: int = 3000,
          grid_: Sequence[float] = DELTA_GRID, seed: int = BOOT_SEED) -> dict:
    """(3g) (module docstring) from the (3f) record (method -> sd, half_width_99,
    proxy_seed_sd)."""
    draws = Draws(n_sim, seed)
    proxy = {m: float(v["proxy_seed_sd"]) for m, v in spread_rec.items()
             if math.isfinite(float(v.get("proxy_seed_sd", math.nan)))}
    halves = {m: float(v["half_width_99"]) for m, v in spread_rec.items()
              if math.isfinite(float(v.get("half_width_99", math.nan)))}
    med_sd = float(np.median(list(proxy.values()))) if proxy else math.nan
    med_h = float(np.median(list(halves.values()))) if halves else math.nan
    ucf_sds = sorted({*(round(med_sd * k, 6) for k in PROXY_SCALE if math.isfinite(med_sd)),
                      *SD_GRID})
    rows = []
    for c_name, c_sd in C_SD.items():
        for su in ucf_sds:
            for sx in SD_GRID:
                b_sd = {"ucf": su, "xdv": sx}
                rows.append({"c_sd": c_name, "sd": b_sd, "half_width": None, "boot": False,
                             "required": joint_required(c_sd, b_sd, None, draws, grid_)})
                if math.isfinite(med_h):
                    for hx in XD_HALF:
                        h = {"ucf": med_h, "xdv": hx}
                        rows.append({"c_sd": c_name, "sd": b_sd, "half_width": h, "boot": True,
                                     "required": joint_required(c_sd, b_sd, h, draws, grid_)})
    designated = {}
    for d, sd, h in (("ucf", med_sd, med_h),
                     ("xdv", DESIGNATED_XD["sd"], DESIGNATED_XD["half_width"])):
        if not (math.isfinite(sd) and math.isfinite(h)):
            designated[d] = {"sd": sd, "half_width": h, "required": None}
            continue
        r = single_required(d, C_SD["measured"][d], sd, h, draws, grid_)
        designated[d] = {"sd": sd, "half_width": h, "c_sd": C_SD["measured"][d], "required": r}
    gaps = [v["required"]["a"] - v["required"]["b"] for v in designated.values()
            if v["required"] and all(math.isfinite(v["required"][k]) for k in ("a", "b"))]
    rec = "b" if any(g >= RECOMMEND_GAP for g in gaps) else ("a" if gaps else None)
    return {"n_sim": n_sim, "seed": seed, "target": TARGET, "alpha": ALPHA, "n": N_SEEDS,
            "c_sd": C_SD, "sd": {"ucf_proxy": proxy, "ucf_values": ucf_sds, "grid": SD_GRID,
                                 "proxy_scale": PROXY_SCALE},
            "half_widths": {"ucf": halves, "ucf_median": med_h, "xdv": XD_HALF},
            "scenario": {"others": OTHERS}, "joint": rows,
            "designated": {**designated, "recommend": rec, "rule": "(b) when (a) needs >= 0.5 "
                           "L points more than (b) in any dataset"}}


# --- (3j) memory --------------------------------------------------------------------------------

def peak_forward(method: bc.BaselineMethod, model, x: torch.Tensor, *, unit: int, n_chunk: int,
                 device: torch.device) -> int | None:
    """Peak allocated bytes of one scored sequence (None off CUDA); the forward must run."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    bc.score_video(method, model, x, unit=unit, n_chunk=n_chunk, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    return None


def _oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def memory(methods: Mapping[str, bc.BaselineMethod], units: Mapping[str, Sequence[int]],
           dims: Sequence[tuple[int, bool]], max_snippets: int, longest_val: int,
           device: torch.device, candidates: Sequence[int] = N_CHUNK_CANDIDATES,
           seed: int = BOOT_SEED) -> dict:
    """(3j): per method, unit and input (dimension, input LayerNorm) the largest candidate
    N_chunk whose forward of the two sequences runs, with its peak bytes; the common N_chunk
    and the maximum bytes."""
    g = torch.Generator().manual_seed(seed)
    rows, fits = {}, []
    for name, m in methods.items():
        for u in units[name]:
            for dim, ln in dims:
                model = m.build(int(dim), unit=int(u), input_ln=bool(ln),
                                scale=m.recipe.scaled_constants(1.0)).to(device=device)
                m.set_mode(model, False)
                row = {"n_chunk": None, "peak_bytes": None, "tried": []}
                for n in candidates:
                    if n % m.sequence_multiple(u):
                        continue
                    try:
                        peaks = [peak_forward(m, model, torch.randn(int(t), int(dim), generator=g),
                                              unit=int(u), n_chunk=n, device=device)
                                 for t in (max_snippets, longest_val)]
                    except RuntimeError as exc:
                        if not _oom(exc):
                            raise
                        row["tried"].append({"n_chunk": n, "oom": True})
                        continue
                    row["tried"].append({"n_chunk": n, "oom": False})
                    known = [p for p in peaks if p is not None]
                    row |= {"n_chunk": n, "peak_bytes": max(known) if known else None}
                    break
                rows[f"{name}/u{u}/d{dim}"] = row
                fits.append(row["n_chunk"])
                del model
    common = min(fits) if fits and all(f is not None for f in fits) else None
    peaks = [r["peak_bytes"] for r in rows.values() if r["peak_bytes"] is not None]
    return {"device": str(device), "max_snippets": int(max_snippets),
            "longest_validation_snippets": int(longest_val), "rows": rows, "n_chunk": common,
            "max_peak_bytes": max(peaks) if peaks else None,
            "gate_m_need_gb": (max(peaks) / 2 ** 30) if peaks else None}


def dataset_configurations(names: Sequence[str]) -> dict[str, bc.BaselineMethod]:
    """Every method's official configuration per dataset (common.method_on): one entry when
    both datasets share it, "<name>/<dataset>" entries when they differ (VCV: 2 layers and
    window 8 on UCF, 1 layer and window 64 on XD), so (3j) measures every network trained."""
    out: dict[str, bc.BaselineMethod] = {}
    for n in names:
        by_ds = {d: bc.method_on(n, d) for d in bc.DATASETS}
        if all(m is by_ds["ucf"] for m in by_ds.values()):
            out[n] = by_ds["ucf"]
        else:
            out |= {f"{n}/{d}": m for d, m in by_ds.items()}
    return out


def max_snippets_of(access0: Mapping) -> int:
    """The access (0) record's maximum n_v (its "max_snippets", else from max_total_frames)."""
    if access0.get("max_snippets") is not None:
        return int(access0["max_snippets"])
    return grid.n_snippets(int(access0["max_total_frames"]))


# --- command line -----------------------------------------------------------------------------

def _chains(items: Sequence[str]) -> dict[str, list[Path]]:
    out = {}
    for it in items:
        name, sep, dirs = it.partition("=")
        if not sep or not dirs:
            raise SystemExit(f"--chains takes METHOD=DIR,DIR,DIR, got {it!r}")
        out[name] = [Path(x) for x in dirs.split(",")]
    return out


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e023_prefreeze")
    p.add_argument("--out", type=Path, default=OUT)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spread", help="(3f) validation spread (UCF, UCA proxy)")
    s.add_argument("--e021_scores", required=True, type=Path)
    s.add_argument("--chains", nargs="+", required=True, help="METHOD=DIR_s0,DIR_s1,DIR_s2")
    s.add_argument("--proxy", type=Path, default=PROXY_FILE)
    s.add_argument("--variant", default="M2A")
    w = sub.add_parser("power", help="(3g) power simulation from the (3f) record")
    w.add_argument("--n_sim", type=int, default=3000)
    m = sub.add_parser("memory", help="(3j) forward memory per method")
    m.add_argument("--access0", required=True, type=Path)
    m.add_argument("--methods", nargs="+", default=list(bc.MAIN_METHODS))
    m.add_argument("--dataset", default="xdv", help="the validation set of the longest video")
    a = p.parse_args(argv)
    if a.cmd == "spread":
        from wvad.data.manifests import load_manifest, manifest_file
        from wvad.diagnostics.uca import load_proxy

        c_runs, meta = e021_c_runs(a.e021_scores)
        names_all = load_manifest(manifest_file("ucf")).names
        names = [names_all[i] for i in meta["val_indices"]]
        labels = proxy_labels(names, meta["labels"], meta["totals"], load_proxy(a.proxy),
                              a.variant)
        ports = {k: port_runs(v) for k, v in _chains(a.chains).items()}
        for k, runs in ports.items():
            if any(r["val_indices"] != meta["val_indices"] for r in runs):
                raise SystemExit(f"{k}: validation videos differ from the C type's")
        rec = {"variant": a.variant, "seeds": list(DEV_SEEDS), "pair": "xcb_xcl",
               "methods": spread(c_runs, ports, labels, meta["totals"]),
               "videos_with_proxy": len(labels),
               "xd": "not measured (no validation frame labels); E022 pooled half-width 0.96 "
                     "as a rough reference"}
        _merge(a.out, "3f", rec)
        print(f"[e023_prefreeze] (3f) -> {a.out}")
    elif a.cmd == "power":
        rec3f = json.loads(Path(a.out).read_text(encoding="utf-8"))["3f"]["methods"]
        _merge(a.out, "3g", power(rec3f, n_sim=a.n_sim))
        print(f"[e023_prefreeze] (3g) -> {a.out}")
    else:
        from wvad.data import load_split, split_file
        from wvad.data.catalog import ENCODER_PAIRS, dense_dir
        from wvad.data.dense import DenseSet
        from wvad.paths import load_paths

        access0 = json.loads(Path(a.access0).read_text(encoding="utf-8"))
        paths = load_paths()
        split = load_split(split_file(a.dataset))
        dense = DenseSet([dense_dir(paths, a.dataset, e) for e in ENCODER_PAIRS["xcb_xcl"]])
        longest = int(max(dense.counts[int(i)] for i in split.val_indices))
        methods = dataset_configurations(a.methods)
        units = {n: sorted(mm.recipe.units) for n, mm in methods.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dims = [(sum(dense.dims), bc.input_ln("xcb_xcl")),
                (sum(DenseSet([dense_dir(paths, a.dataset, e) for e in ENCODER_PAIRS[
                    "xcb_siglip"]]).dims), bc.input_ln("xcb_siglip"))]
        rec = memory(methods, units, dims, max_snippets_of(access0), longest, device)
        _merge(a.out, "3j", rec)
        print(f"[e023_prefreeze] (3j) N_chunk {rec['n_chunk']} -> {a.out}")


if __name__ == "__main__":
    main()
