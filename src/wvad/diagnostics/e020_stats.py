"""Statistics of the E020 verdict and its pre-registered analyses (PLAN.md): two-sided
seed-paired t tests with Holm over the datasets (or the four settings), the dataset grades G_d
with the substantive margins m_d (point estimate) and equivalence by the 90% seed interval, the
overall verdict, the Wilcoxon signed-rank test (zero_method 'pratt', method 'auto'), the flags,
the treatment-failure sensitivity, the 2x2 factorial effects and the video bootstrap of the
frame metrics (anomalous and normal videos resampled separately, the same resamples for every
run; resample 0 is the original sample).

Values are maps (pair, seed, cell) -> metric in points for one dataset. Failed runs are absent:
every test uses the complete cases (seeds with every needed run in every pair it averages).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from wvad.diagnostics.e020_score import CELLS, SEEDS
from wvad.diagnostics.runs import PAIRS
from wvad.eval.metrics import DEFAULT_MIN_FRAMES

ALPHA = 0.05
MARGIN = {"ucf": 0.5, "xdv": 2.0}              # m_d: UCF frame AUC / XD frame AP points
CONTRASTS = {"m10_minus_l20": ("m10", "l20"), "base_minus_l20": ("base", "l20"),
             "m10b_minus_m10": ("m10b", "m10"), "m10b_minus_base": ("m10b", "base")}
EFFECTS = ("A_best_minus_last", "B_offset", "AxB")
POSITIVE = ("substantive_positive", "statistical_positive")
BOOT_R, BOOT_SEED, BOOT_CHUNK = 2000, 20261006, 100   # chunk: resamples per pass (memory)

Values = Mapping[tuple[str, int, str], float]


# --- tests ----------------------------------------------------------------------------------

def paired(diffs: Sequence[float]) -> dict:
    """Two-sided one-sample t test of seed-paired differences against 0: n, mean, seed SD, t, p,
    d_z and the 90% / 95% t intervals (NaN with fewer than two seeds; zero spread: p = 0 for a
    non-zero mean, 1 for a zero mean)."""
    x = np.asarray(diffs, dtype=float)
    n = len(x)
    mean = float(x.mean()) if n else math.nan
    nan2 = [math.nan, math.nan]
    if n < 2:
        return {"n": n, "mean": mean, "sd": math.nan, "t": math.nan, "p": math.nan,
                "d_z": math.nan, "ci90": nan2, "ci95": nan2}
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    if se > 0:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), n - 1))
    else:
        t = 0.0 if mean == 0 else math.copysign(math.inf, mean)
        p = 1.0 if mean == 0 else 0.0
    ci = {name: [mean - q * se, mean + q * se]
          for name, q in (("ci90", float(stats.t.ppf(0.95, n - 1))),
                          ("ci95", float(stats.t.ppf(0.975, n - 1))))}
    return {"n": n, "mean": mean, "sd": sd, "t": float(t), "p": p,
            "d_z": mean / sd if sd > 0 else math.nan, **ci}


def _wilcoxon(x: np.ndarray, method) -> float:
    return float(stats.wilcoxon(x, zero_method="pratt", method=method).pvalue)


def wilcoxon_p(diffs: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p (scipy, zero_method 'pratt', method 'auto'); 1 when
    every difference is 0, NaN with fewer than two seeds."""
    x = np.asarray(diffs, dtype=float)
    if len(x) < 2:
        return math.nan
    if not np.any(x):
        return 1.0
    return _wilcoxon(x, "auto")


def wilcoxon_method(diffs: Sequence[float]) -> str:
    """The p-value method SciPy's method='auto' used for these differences: SciPy's rule (n > 50
    asymptotic; else exact without ties or zeros, a permutation test with them for n <= 13,
    asymptotic beyond), confirmed by recomputing p with that method; 'unknown' when it differs
    (another SciPy rule), 'none' where wilcoxon_p does not call SciPy."""
    x = np.asarray(diffs, dtype=float)
    if len(x) < 2 or not np.any(x):
        return "none"
    a = np.abs(x)
    ties = bool(np.any(a == 0) or len(np.unique(a)) < len(a))
    name = ("asymptotic" if len(x) > 50 or (ties and len(x) > 13)
            else "permutation" if ties else "exact")
    try:
        method = stats.PermutationMethod() if name == "permutation" else name
        same = _wilcoxon(x, method) == _wilcoxon(x, "auto")
    except (AttributeError, ValueError):            # an older SciPy: other names or no class
        same = False
    return name if same else "unknown"


def holm(p: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values over the family ``p``; a NaN p counts as 1."""
    keys = list(p)
    vals = [float(p[k]) if math.isfinite(p[k]) else 1.0 for k in keys]
    order = sorted(range(len(keys)), key=lambda i: vals[i])
    adj, running = [1.0] * len(keys), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(keys) - rank) * vals[i]))
        adj[i] = running
    return dict(zip(keys, adj, strict=True))


def conclusion(p_holm: float, mean: float) -> str:
    """A dataset's conclusion: 'positive' / 'negative' when significant after Holm, else 'none'
    (the readout and distribution flags compare these)."""
    if p_holm < ALPHA and mean > 0:
        return "positive"
    if p_holm < ALPHA and mean < 0:
        return "negative"
    return "none"


def seed_diffs(vals: Values, a: str, b: str, pairs: Sequence[str] = PAIRS,
               seeds: Sequence[int] = SEEDS) -> tuple[list[int], list[float]]:
    """Complete seeds (cells a and b in every pair) and their pair-averaged a - b."""
    kept = [s for s in seeds if all((p, s, c) in vals for p in pairs for c in (a, b))]
    return kept, [float(np.mean([vals[(p, s, a)] - vals[(p, s, b)] for p in pairs]))
                  for s in kept]


def _with_holm(res: dict) -> dict:
    adj = holm({k: r["p"] for k, r in res.items()})
    for k, r in res.items():
        r["p_holm"] = adj[k]
        r["conclusion"] = conclusion(adj[k], r["mean"])
    return res


def contrast(vals_by_d: Mapping[str, Values], a: str, b: str, kind: str = "t") -> dict:
    """Per dataset: test of the pair-averaged a - b over the complete seeds (paired t, or
    Wilcoxon with kind 'wilcoxon', with the method SciPy used); Holm over the datasets."""
    out = {}
    for d, vals in vals_by_d.items():
        kept, diffs = seed_diffs(vals, a, b)
        row = {"seeds": kept, "seeds_dropped": len(SEEDS) - len(kept), "diffs": diffs}
        if kind == "t":
            out[d] = row | paired(diffs)
        else:
            out[d] = row | {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else math.nan,
                            "p": wilcoxon_p(diffs), "method": wilcoxon_method(diffs)}
    return _with_holm(out)


def setting_contrast(vals_by_d: Mapping[str, Values], a: str, b: str) -> dict:
    """Per setting (dataset/pair): paired t of a - b over its complete seeds; Holm over the
    four settings."""
    out = {}
    for d, vals in vals_by_d.items():
        for p in PAIRS:
            kept, diffs = seed_diffs(vals, a, b, pairs=(p,))
            out[f"{d}/{p}"] = {"seeds": kept, "seeds_dropped": len(SEEDS) - len(kept),
                               "diffs": diffs, **paired(diffs)}
    return _with_holm(out)


# --- verdict --------------------------------------------------------------------------------

def within(ci: Sequence[float], margin: float) -> bool:
    """The interval lies inside the open band (-margin, +margin); NaN -> False."""
    return bool(-margin < ci[0] and ci[1] < margin)


def grade(p_holm: float, mean: float, ci90: Sequence[float], margin: float) -> str:
    """Dataset grade G_d (PLAN.md, first match; mean = point estimate D_d in points):
    1. substantive_positive: Holm p < 0.05 and mean >= m_d
    2. statistical_positive: Holm p < 0.05 and 0 < mean < m_d
    3. negative:             Holm p < 0.05 and mean < 0
    4. equivalent:           90% seed t interval inside (-m_d, +m_d)
    5. inconclusive"""
    significant = p_holm < ALPHA
    if significant and mean >= margin:
        return "substantive_positive"
    if significant and 0 < mean < margin:
        return "statistical_positive"
    if significant and mean < 0:
        return "negative"
    return "equivalent" if within(ci90, margin) else "inconclusive"


def verdict(grades: Mapping[str, str]) -> str:
    """Overall verdict (PLAN.md, first match):
    1. every dataset substantive positive -> better_substantive (the only way to condition (2))
    2. every dataset positive (substantive or statistical) -> better_statistical_only
    3. one positive, one negative -> dataset_dependent
    4. any negative -> worse
    5. exactly one positive -> one_dataset_significant
    6. otherwise -> not_met (reported with the grades)"""
    g = list(grades.values())
    positive = [x for x in g if x in POSITIVE]
    negative = [x for x in g if x == "negative"]
    if g and all(x == "substantive_positive" for x in g):
        return "better_substantive"
    if g and len(positive) == len(g):
        return "better_statistical_only"
    if positive and negative:
        return "dataset_dependent"
    if negative:
        return "worse"
    if len(positive) == 1:
        return "one_dataset_significant"
    return "not_met"


def main_test(vals_by_d: Mapping[str, Values], a: str = "m10", b: str = "l20") -> dict:
    """Paired t per dataset with Holm over the datasets, the grades G_d and the verdict."""
    res = contrast(vals_by_d, a, b)
    grades = {d: grade(r["p_holm"], r["mean"], r["ci90"], MARGIN[d]) for d, r in res.items()}
    return {"contrast": f"{a}_minus_{b}", "datasets": res, "grades": grades,
            "verdict": verdict(grades)}


def flags(*, main: Mapping, sigma: Mapping, wilcoxon: Mapping, vs_base: Mapping,
          localisation: Mapping, settings: Mapping, boot: Mapping[str, Mapping]) -> dict:
    """Pre-registered flags of one checkpoint rule (PLAN.md); lists of datasets / settings
    (keyword arguments: the inputs are alike and must not be swapped).
    readout: G_d changes under the sigma readout; distribution: Wilcoxon significance (Holm)
    differs from the t significance; video sample: G_d positive or negative while the bootstrap
    95% interval contains 0."""
    res, grades = main["datasets"], main["grades"]

    def negative(r: Mapping) -> list[str]:
        return [k for k, v in r.items() if v["conclusion"] == "negative"]
    return {"not_better_than_base": negative(vs_base),
            "setting_dependent": negative(settings),
            "localisation_harm": negative(localisation),
            "readout_sensitive": [d for d in grades if grades[d] != sigma["grades"][d]],
            "distribution_sensitive": [d for d in res if (res[d]["p_holm"] < ALPHA)
                                       != (wilcoxon[d]["p_holm"] < ALPHA)],
            "sensitive_to_test_video_sample": [
                d for d in grades if (grades[d] in POSITIVE or grades[d] == "negative")
                and boot[d]["contains_0"]]}


def impute_minimum(vals: Values, failed: Sequence[tuple[str, int, str]]) -> dict:
    """Treatment-failure sensitivity (PLAN.md): each failed (pair, seed, cell) takes the lowest
    value of that cell over the other seeds of the same pair."""
    out = dict(vals)
    for p, s, c in failed:
        pool = [v for (pp, _, cc), v in vals.items() if pp == p and cc == c]
        if pool:
            out[(p, s, c)] = min(pool)
    return out


# --- factorial ------------------------------------------------------------------------------

def effects(v: Mapping[str, float]) -> dict[str, float]:
    """One seed's factorial effects from its four cells (PLAN.md)."""
    return {"A_best_minus_last": 0.5 * ((v["base"] - v["l20"]) + (v["m10b"] - v["m10"])),
            "B_offset": 0.5 * ((v["m10"] - v["l20"]) + (v["m10b"] - v["base"])),
            "AxB": (v["m10"] - v["l20"]) - (v["m10b"] - v["base"])}


def effect_table(vals: Values, pairs: Sequence[str]) -> dict:
    """Mean and 95% seed t interval of each effect over the seeds with all four cells in every
    pair (cells averaged over the pairs first; the effects are linear)."""
    kept = [s for s in SEEDS if all((p, s, c) in vals for p in pairs for c in CELLS)]
    per = [effects({c: float(np.mean([vals[(p, s, c)] for p in pairs])) for c in CELLS})
           for s in kept]
    out = {"seeds": kept, "seeds_dropped": len(SEEDS) - len(kept)}
    for k in EFFECTS:
        r = paired([e[k] for e in per])
        out[k] = {"mean": r["mean"], "ci95": r["ci95"], "sd": r["sd"], "n": r["n"],
                  "per_seed": [e[k] for e in per]}
    return out


def factorial(vals_by_d: Mapping[str, Values]) -> dict:
    """Effects per setting and pair-averaged, the four per-setting contrasts (Holm over the
    four settings per contrast) and the non-inferiority statement for L20 (base - L20, 90% seed
    interval upper bound < m_d, pair-averaged)."""
    eff = {}
    for d, vals in vals_by_d.items():
        for p in PAIRS:
            eff[f"{d}/{p}"] = effect_table(vals, (p,))
        eff[f"{d}/pair_mean"] = effect_table(vals, PAIRS)
    base = contrast(vals_by_d, "base", "l20")
    return {"effects": eff,
            "contrasts": {k: setting_contrast(vals_by_d, a, b) for k, (a, b) in CONTRASTS.items()},
            "l20_non_inferiority": {d: {"margin": MARGIN[d], "mean": r["mean"], "ci90": r["ci90"],
                                        "l20_not_harmful": bool(r["ci90"][1] < MARGIN[d])}
                                    for d, r in base.items()}}


# --- video bootstrap ------------------------------------------------------------------------

def bootstrap_weights(labels: Sequence[int], n_resamples: int = BOOT_R,
                      seed: int = BOOT_SEED) -> np.ndarray:
    """(R + 1, n) draw counts. Row 0 is the original sample (every video once); in rows 1..R
    the anomalous videos, then the normal videos, are drawn with replacement to their own
    count, from default_rng(seed)."""
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    w = np.zeros((n_resamples + 1, len(y)), dtype=np.int64)
    w[0] = 1
    rows = np.arange(1, n_resamples + 1)[:, None]
    for cls in (1, 0):
        idx = np.flatnonzero(y == cls)
        draws = idx[rng.integers(0, len(idx), size=(n_resamples, len(idx)))]
        np.add.at(w, (np.broadcast_to(rows, draws.shape), draws), 1)
    return w


@dataclass(frozen=True)
class Atoms:
    """Video segments as frame-metric atoms sorted by score: the segment's score, its positive
    and negative frame counts, its video; ``starts`` = first atom of every distinct score."""
    score: np.ndarray
    pos: np.ndarray
    neg: np.ndarray
    video: np.ndarray
    starts: np.ndarray


def segment_atoms(scores: np.ndarray, gts: Sequence[np.ndarray],
                  min_frames: int = DEFAULT_MIN_FRAMES) -> Atoms:
    """The evaluator's frame expansion (wvad.eval.metrics: T // S frames per segment, the last
    segment also takes T % S; videos under ``min_frames`` frames skipped) as one atom per
    segment."""
    parts = []
    for i, gt in enumerate(gts):
        t, s = len(gt), scores.shape[1]
        if t < min_frames:
            continue
        if t < s:
            raise ValueError(f"video {i}: {t} frames cannot hold {s} segments")
        counts = np.full(s, t // s)
        counts[-1] += t % s
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = np.add.reduceat(np.asarray(gt, dtype=np.float64), first)
        parts.append((np.asarray(scores[i], dtype=np.float64), pos, counts - pos,
                      np.full(s, i)))
    score, pos, neg, video = (np.concatenate(x) for x in zip(*parts, strict=True))
    order = np.argsort(score, kind="stable")
    score = score[order]
    starts = np.flatnonzero(np.r_[True, score[1:] != score[:-1]])
    return Atoms(score, pos[order], neg[order], video[order], starts)


def weighted_auc_ap(at: Atoms, weights: np.ndarray,
                    chunk: int = BOOT_CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """Frame AUC and AP (points) for every row of ``weights`` (R, n_videos), each video's frames
    counted weight times: AUC by the rank formula (ties one half), AP over the distinct
    thresholds as sklearn's average_precision_score."""
    w = np.atleast_2d(weights)
    aucs, aps = [], []
    for lo in range(0, len(w), chunk):
        ww = w[lo:lo + chunk][:, at.video].astype(np.float64)
        pg = np.add.reduceat(ww * at.pos, at.starts, axis=1)       # ascending distinct scores
        ng = np.add.reduceat(ww * at.neg, at.starts, axis=1)
        ptot, ntot = pg.sum(axis=1), ng.sum(axis=1)
        below = np.cumsum(ng, axis=1) - ng
        aucs.append((pg * (below + 0.5 * ng)).sum(axis=1) / (ptot * ntot))
        tp, fp = np.cumsum(pg[:, ::-1], axis=1), np.cumsum(ng[:, ::-1], axis=1)
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        aps.append((pg[:, ::-1] * prec).sum(axis=1) / ptot)
    return 100 * np.concatenate(aucs), 100 * np.concatenate(aps)


def bootstrap_diff(scores: Mapping[tuple[str, int, str], np.ndarray], gts: Sequence[np.ndarray],
                   labels: Sequence[int], metric: str, seeds: Sequence[int],
                   a: str = "m10", b: str = "l20",
                   point: Mapping[tuple[str, int, str], float] | None = None) -> dict:
    """95% percentile interval (resamples 1..R) of the seed mean of the pair-averaged a - b
    under the video bootstrap; ``scores`` (pair, seed, cell) -> (n, S) for the complete
    ``seeds``. Resample 0 is the original sample: with ``point`` (the evaluator's values) it
    must reproduce them (largest |difference| reported; infinite when either is not finite).
    Non-finite resampled differences raise ValueError."""
    out = {"R": BOOT_R, "seed": BOOT_SEED, "metric": metric, "seeds": list(seeds)}
    if not seeds:
        return out | {"resample0": math.nan, "mean": math.nan, "ci95": [math.nan, math.nan],
                      "contains_0": False, "point_check_max_abs_diff": math.nan}
    w = bootstrap_weights(labels)
    pick = 0 if metric == "frame_auc" else 1
    per, check = {}, 0.0
    for k, sc in scores.items():
        per[k] = weighted_auc_ap(segment_atoms(sc, gts), w)[pick]
        if point is not None:
            d = abs(float(per[k][0]) - point[k])
            check = max(check, d) if math.isfinite(d) else math.inf
    diff = np.mean([np.mean([per[(p, s, a)] - per[(p, s, b)] for p in PAIRS], axis=0)
                    for s in seeds], axis=0)
    if not np.isfinite(diff).all():
        raise ValueError(f"{metric}: {int(np.sum(~np.isfinite(diff)))} non-finite bootstrap "
                         "differences")
    lo, hi = (float(x) for x in np.percentile(diff[1:], [2.5, 97.5]))
    return out | {"R": int(len(w) - 1), "resample0": float(diff[0]),
                  "mean": float(diff[1:].mean()), "ci95": [lo, hi],
                  "contains_0": bool(lo <= 0 <= hi), "point_check_max_abs_diff": check}
