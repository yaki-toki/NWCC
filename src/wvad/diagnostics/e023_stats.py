"""Statistics of the E023 verdict and its pre-registered checks (PLAN.md "판정 (사전 등록)",
"조건 (2)", "조건 (3)", "기전 가족", "표지", "결과의 쓰임"). Judgement-analysis code (PLAN.md "코드
규칙" 3); pure functions of per-run metric values and of the video bootstraps. New code (add-only
rule); wvad.diagnostics.e022_stats and e020_stats are reused unchanged.

Values are maps (pair, seed, arm) -> metric in points for one dataset; failed runs are absent. A
seed's value is the mean over the two feature pairs (pair mean); each arm uses its own complete
seeds (both pairs present). Arms: E022's C and C.last; the comparison methods by plan name (MIL,
RTFM, URDMU, VCV, ...), their checkpoint rules "<arm>.logit" / "<arm>.last", the X.32 arms
"<arm>.32" and C.L20 / C.L20.last.

    tests       Welch t (Welch-Satterthwaite df; conditions (2), (3)), seed-paired t
                (e022_stats.paired; mechanism, distribution flag), Mann-Whitney U (two-sided,
                SciPy method 'auto'; distribution flag), Holm (e020_stats.holm) at alpha 0.01
    grades      G_kd (first match): substantive positive (Holm p < alpha, D >= m_L), statistical
                positive, negative, equivalent (98% Welch interval inside (-m_L, +m_L)),
                inconclusive (e022_stats.grade, rule 'point'); without margins (D6 (c)) neither
                substantive nor equivalent; a contrast with fewer than 8 complete seeds in an
                arm (numerical failures and collapsed chains left out): "판정 없음(비교 기법 수치
                불안정·붕괴)", counted as inconclusive (p = NaN, Holm 1)
    collapse    PLAN.md "실패 처리" "붕괴 제외" (user, 2026-10-20): a chain whose selected sigma
                validation V-AUC is below 0.60 leaves every rule (``collapsed``; fixed in the
                selection record before access (A), no rerun, no replacement seed); the flag
                "비교 기법 퇴화" keeps only the test-constancy criterion (``degenerate``)
    B*_d        the main-family arm of the highest seed-mean L (ties: smaller seed SD, then name);
                C - B*_d by Welch, Holm over the two datasets
    S_d         1 all substantive positive, 2 all positive, 3 a negative grade, 4 C - B*_d positive
                and no negative, 5 otherwise
    condition 2 options D5 (a) / (b) / (c) and the dataset requirement (i) / (ii) as parameters
                (D5, D6 undecided: every combination reported); common requirements: D2
                approved, gates (h) / (h2) passed, no "판정 없음" contrast ("사용자 결정 대기")
    condition 3 Welch over the 8 contrasts of the pooled main metric; substantive inferiority
                (Holm p < alpha and D <= -m_d); the flag "합산 지표 열세" (significantly negative,
                or D <= -m_d without significance); the per-contrast wording rule ("합산 지표에서도
                낫다" / "같다" / "실질 열세" / "실질 열세 검출 안 됨"); the 98% lower bounds
    bootstrap   L: the included anomalous test videos resampled with replacement, R 5000,
                default_rng(20261019), the same resamples for every arm and run, percentile
                intervals 99% / 95%, resample 0 the original sample; pooled: anomalous and normal
                videos separately (e020_stats.bootstrap_weights, the same seed), frame atoms of
                the snippet supports (e022_stats.snippet_atoms). With equal seed lists a contrast
                is the seed mean of the pair-averaged differences (= e022_stats.bootstrap_contrast
                bit for bit), else the difference of the two arms' seed means (Welch form)
    claims      secondary_improvements_allowed = False; abstract-level pooled improvement never
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from wvad.diagnostics import e020_stats as st20
from wvad.diagnostics import e022_stats as st22

DATASETS = ("ucf", "xdv")
PAIRS = ("xcb_xcl", "xcb_siglip")
SEEDS = st22.SEEDS
NAMES = {"ucf": "UCF", "xdv": "XD"}
ALPHA = 0.01
T_LEVEL = 0.98                                   # equivalence / descriptive Welch interval
BOOT_LEVEL = 0.99
BOOT_LEVELS = (0.99, 0.95)
BOOT_R, BOOT_SEED = 5000, 20261019
MIN_COMPLETE = 8
COLLAPSE_V_AUC = 0.60                            # "붕괴 제외": selected sigma V-AUC (fraction)
COLLAPSED = "collapsed"                          # the failure kind (e022_stats.Failure.kind)
MARGIN_POOLED = dict(st20.MARGIN)                # m_d: UCF frame AUC 0.5, XD frame AP 2.0
MARGIN_L = {"a": {"ucf": 0.5, "xdv": 2.0}, "b": {"ucf": 1.0, "xdv": 2.0}, "c": None}   # D6
D5_OPTIONS, REQUIREMENTS, D6_OPTIONS = ("a", "b", "c"), ("i", "ii"), ("a", "b", "c")

SUB, STAT, NEG, EQ, INC = st22.SUB, st22.STAT, st22.NEG, st22.EQ, st22.INC
POSITIVE = (SUB, STAT)
NO_VERDICT = "판정 없음(비교 기법 수치 불안정·붕괴)"
GRADE_NAMES = {SUB: "실질 양", STAT: "통계 양", NEG: "음", EQ: "동등", INC: "결론 없음"}
CATEGORIES = {1: "모든 비교 기법보다 낫다(실질)", 2: "모든 비교 기법보다 낫다(통계적으로만)",
              3: "비교 기법에 뒤짐(목록)", 4: "가장 강한 비교 기법보다 낫다", 5: "미충족(등급표)"}
MET, NOT_MET, PENDING, GATE_FAILED, NOT_COUNTED, UNDECIDED, NA = (
    "met", "not_met", "user_decision_pending", "gate_failure_no_verdict",
    "not_counted_d2", "d5_undecided", "not_applicable")
STATUS_NAMES = {MET: "충족", NOT_MET: "미충족", PENDING: "사용자 결정 대기",
                GATE_FAILED: "관문 실패(판정 없음)", NOT_COUNTED: "D2 미승인(조건 (2)에 쓰지 않음)",
                UNDECIDED: "D5·D6 미정(모든 선택지의 결과)", NA: "해당 없음"}
WORDING = {"better": "합산 지표에서도 낫다", "same": "합산 지표에서 같다",
           "substantive_inferiority": "실질 열세(기법·데이터셋)",
           "no_substantive_inferiority_detected": "실질 열세 검출 안 됨"}
C3_PASS = "통과: 실질 열세 검출 안 됨"
C3_FAIL = "미통과({items})"
NOT_WORSE_TEXT = "같은 특징 비교 기법보다 합산 지표에서 실질적으로 나쁘지 않음"
FLAG_NAMES = {"setting_dependent": "설정 의존", "readout_sensitive": "판독 민감",
              "selection_readout_sensitive": "선택 판독 민감",
              "distribution_sensitive": "분포 민감",
              "checkpoint_rule_dependent": "체크포인트 규칙 의존",
              "sensitive_to_test_video_sample": "시험 영상 표본에 민감",
              "training_unit_sensitive": "학습 단위 민감",
              "l_metric_kind_sensitive": "L 지표 종류 민감",
              "pooled_inferiority": "합산 지표 열세",
              "baseline_degenerate": "비교 기법 퇴화",
              "numerically_unstable": "수치 불안정·붕괴",
              "weak_under_adaptation": "적응 하의 약함"}
MECH_RAISED, MECH_LOWERED, MECH_NONE = "offset_raised", "offset_lowered", "not_established"

Values = Mapping[tuple[str, int, str], float]
ByDataset = Mapping[str, Values]


# --- user decisions -------------------------------------------------------------------------

@dataclass(frozen=True)
class Decisions:
    """PLAN.md "사용자 결정 기록": D1, D2, D3, D9, D13 (a) and the default bundle accepted on
    2026-10-19; D5 (wording, dataset requirement), D6 (m_L) and the D7 re-question answered on
    2026-10-21 (RECORDED). None (an undecided instance): every option is reported, condition
    (2) "D5·D6 미정"."""

    d2: str = "a"
    d5: str | None = None
    d5_requirement: str | None = None
    d6: str | None = None
    d9: str = "a"
    d13: str = "a"

    def __post_init__(self):
        if self.d2 not in ("a", "b") or self.d9 not in ("a", "b") or \
                self.d13 not in ("a", "b", "c"):
            raise ValueError("D2 / D9 in (a, b), D13 in (a, b, c)")
        if self.d5 not in (None, *D5_OPTIONS) or self.d5_requirement not in (None,
                                                                              *REQUIREMENTS):
            raise ValueError(f"D5 {self.d5!r} / requirement {self.d5_requirement!r}")
        if self.d6 not in (None, *D6_OPTIONS):
            raise ValueError(f"D6 {self.d6!r}")
        if (self.d5 is None) != (self.d5_requirement is None) or \
                (self.d5 is None) != (self.d6 is None):
            raise ValueError("D5, its dataset requirement and D6 are decided together")
        if self.d5 == "c" and self.d6 == "c":
            raise ValueError("D5 (c) needs D6 (a) or (b) (PLAN.md 의존)")
        if self.d13 in ("a", "b") and self.d9 != "a":
            raise ValueError("D13 (a) / (b) needs D9 (a) (review F3c)")

    @property
    def decided(self) -> bool:
        return self.d5 is not None

    def margin_options(self) -> dict[str, dict | None]:
        """D6 option -> m_L (all options while undecided)."""
        keys = D6_OPTIONS if self.d6 is None else (self.d6,)
        return {k: MARGIN_L[k] for k in keys}


RECORDED = Decisions(d5="b", d5_requirement="i", d6="a")    # PLAN.md 사용자 결정 기록, 2026-10-21


# --- seed values and tests ------------------------------------------------------------------

def pair_means(vals: Values, arm: str, pairs: Sequence[str] = PAIRS,
               seeds: Sequence[int] = SEEDS) -> dict[int, float]:
    """Seed -> mean over the pairs, for the seeds holding the arm in every pair."""
    return {s: float(np.mean([vals[(p, s, arm)] for p in pairs])) for s in seeds
            if all((p, s, arm) in vals for p in pairs)}


def welch(x: Sequence[float], y: Sequence[float], level: float = T_LEVEL) -> dict:
    """Two-sided Welch t of mean(x) - mean(y): Welch-Satterthwaite df, the interval at
    ``level``; NaN with fewer than two values in either group; zero spread in both: p 0 for a
    non-zero difference, 1 for none (as e020_stats.paired)."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    out = {"n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()) if len(a) else math.nan,
           "mean_b": float(b.mean()) if len(b) else math.nan, "level": level}
    out["mean"] = out["mean_a"] - out["mean_b"]
    nan2 = [math.nan, math.nan]
    if len(a) < 2 or len(b) < 2:
        return out | {"sd_a": math.nan, "sd_b": math.nan, "se": math.nan, "df": math.nan,
                      "t": math.nan, "p": math.nan, "ci": nan2}
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    qa, qb = va / len(a), vb / len(b)
    se = math.sqrt(qa + qb)
    out |= {"sd_a": math.sqrt(va), "sd_b": math.sqrt(vb), "se": se}
    if se == 0:
        t = 0.0 if out["mean"] == 0 else math.copysign(math.inf, out["mean"])
        return out | {"df": math.nan, "t": t, "p": 1.0 if out["mean"] == 0 else 0.0,
                      "ci": [out["mean"], out["mean"]]}
    df = (qa + qb) ** 2 / ((qa ** 2) / (len(a) - 1) + (qb ** 2) / (len(b) - 1))
    t = out["mean"] / se
    q = float(stats.t.ppf(st22._probability(level), df))
    return out | {"df": float(df), "t": float(t), "p": float(2 * stats.t.sf(abs(t), df)),
                  "ci": [out["mean"] - q * se, out["mean"] + q * se]}


def mann_whitney(x: Sequence[float], y: Sequence[float]) -> dict:
    """Two-sided Mann-Whitney U (SciPy, method 'auto'); NaN with an empty group."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not len(a) or not len(b):
        return {"u": math.nan, "p": math.nan, "n_a": len(a), "n_b": len(b)}
    r = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return {"u": float(r.statistic), "p": float(r.pvalue), "n_a": len(a), "n_b": len(b)}


def seed_paired(xa: Mapping[int, float], xb: Mapping[int, float], level: float = T_LEVEL
                ) -> dict:
    """Seed-number paired t (e022_stats.paired) over the seeds both arms hold."""
    common = [s for s in xa if s in xb]
    return {"seeds": common, **st22.paired([xa[s] - xb[s] for s in common], level)}


def holm(p: Mapping[str, float]) -> dict[str, float]:
    return st20.holm(p)


def conclusion(p_holm: float, mean: float, alpha: float = ALPHA) -> str:
    return st22.conclusion(p_holm, mean, alpha)


def grade(p_holm: float, mean: float, ci: Sequence[float], margin: float | None, *,
          alpha: float = ALPHA) -> str:
    """G_kd (module docstring); ``margin`` None (D6 (c)): no substantive, no equivalent."""
    if margin is None:
        sig = p_holm < alpha
        return STAT if sig and mean > 0 else NEG if sig and mean < 0 else INC
    return st22.grade(p_holm, mean, ci, margin, alpha=alpha, rule="point")


# --- contrasts ------------------------------------------------------------------------------

def key(arm: str, dataset: str) -> str:
    return f"{arm}/{dataset}"


def contrast_row(vals: Values, a: str, b: str, *, kind: str = "welch", level: float = T_LEVEL,
                 min_complete: int = MIN_COMPLETE, pairs: Sequence[str] = PAIRS,
                 seeds: Sequence[int] = SEEDS) -> dict:
    """One contrast a - b of one dataset: Welch (each arm's complete seeds), seed-paired t
    ('paired') or Mann-Whitney U ('mwu'); "no_verdict" when an arm has fewer than
    ``min_complete`` complete seeds (its p is then NaN)."""
    xa, xb = pair_means(vals, a, pairs, seeds), pair_means(vals, b, pairs, seeds)
    row = {"a": a, "b": b, "seeds_a": list(xa), "seeds_b": list(xb),
           "values_a": list(xa.values()), "values_b": list(xb.values())}
    short = [x for x, n in ((a, len(xa)), (b, len(xb))) if n < min_complete]
    if kind == "welch":
        row |= welch(row["values_a"], row["values_b"], level)
    elif kind == "paired":
        row |= seed_paired(xa, xb, level)
    elif kind == "mwu":
        row |= mann_whitney(row["values_a"], row["values_b"])
        row["mean"] = (float(np.mean(row["values_a"])) - float(np.mean(row["values_b"]))
                       if xa and xb else math.nan)
    else:
        raise ValueError(f"unknown test {kind!r} (welch, paired, mwu)")
    row["no_verdict"] = f"{NO_VERDICT}: {', '.join(short)}" if short else None
    if short:
        row["p"] = math.nan
    return row


def family(vals_by_d: ByDataset, arms: Sequence[str], c: str = "C", *, kind: str = "welch",
           alpha: float = ALPHA, level: float = T_LEVEL, margins: Mapping[str, float] | None,
           min_complete: int = MIN_COMPLETE, pairs: Sequence[str] = PAIRS,
           seeds: Sequence[int] = SEEDS) -> dict:
    """C - B for every arm and dataset ('<arm>/<dataset>'), Holm over the whole family at
    ``alpha`` (a "no_verdict" contrast stays in it with p = NaN), conclusions and grades."""
    rows = {key(b, d): contrast_row(vals, c, b, kind=kind, level=level,
                                    min_complete=min_complete, pairs=pairs, seeds=seeds)
            for d, vals in vals_by_d.items() for b in arms}
    adj = holm({k: r["p"] for k, r in rows.items()})
    for k, r in rows.items():
        d = k.rsplit("/", 1)[1]
        r["dataset"] = d
        r["p_holm"] = adj[k]
        r["conclusion"] = conclusion(adj[k], r["mean"], alpha) if not r["no_verdict"] else "none"
        if kind == "welch":
            r["grade"] = INC if r["no_verdict"] else grade(
                adj[k], r["mean"], r["ci"], None if margins is None else margins[d], alpha=alpha)
    return rows


def strongest(vals_by_d: ByDataset, arms: Sequence[str], *, min_complete: int = MIN_COMPLETE,
              pairs: Sequence[str] = PAIRS, seeds: Sequence[int] = SEEDS) -> dict:
    """B*_d per dataset: the arm (>= ``min_complete`` complete seeds) with the highest seed mean
    of L; ties: the smaller seed SD, then the name."""
    out = {}
    for d, vals in vals_by_d.items():
        cand = {}
        for b in arms:
            x = list(pair_means(vals, b, pairs, seeds).values())
            if len(x) >= min_complete:
                cand[b] = {"n": len(x), "mean": float(np.mean(x)),
                           "sd": float(np.std(x, ddof=1)) if len(x) > 1 else math.nan}
        best = min(cand, key=lambda b: (-cand[b]["mean"], cand[b]["sd"], b)) if cand else None
        out[d] = {"arm": best, "candidates": cand}
    return out


def bstar_test(vals_by_d: ByDataset, bstar: Mapping[str, Mapping], c: str = "C", *,
               alpha: float = ALPHA, level: float = T_LEVEL,
               min_complete: int = MIN_COMPLETE, pairs: Sequence[str] = PAIRS,
               seeds: Sequence[int] = SEEDS) -> dict:
    """C - B*_d by Welch per dataset, Holm over the datasets."""
    rows = {}
    for d, vals in vals_by_d.items():
        b = bstar[d]["arm"]
        rows[d] = ({"b": None, "p": math.nan, "mean": math.nan, "no_verdict": NO_VERDICT}
                   if b is None else contrast_row(vals, c, b, level=level,
                                                  min_complete=min_complete, pairs=pairs,
                                                  seeds=seeds))
    adj = holm({d: r["p"] for d, r in rows.items()})
    for d, r in rows.items():
        r["p_holm"] = adj[d]
        r["conclusion"] = conclusion(adj[d], r["mean"], alpha) if not r.get("no_verdict") \
            else "none"
    return rows


def category(grades: Mapping[str, str], no_verdict: Sequence[str], bstar_row: Mapping) -> dict:
    """S_d of one dataset (``grades``: arm -> G_kd; "no_verdict" arms counted inconclusive,
    named after the category)."""
    g = list(grades.values())
    neg = [a for a, x in grades.items() if x == NEG]
    if g and all(x == SUB for x in g):
        code = 1
    elif g and all(x in POSITIVE for x in g):
        code = 2
    elif neg:
        code = 3
    elif bstar_row.get("conclusion") == "positive":
        code = 4
    else:
        code = 5
    text = CATEGORIES[code]
    if code == 3:
        text = f"{text}: {', '.join(neg)}"
    if code == 5:
        text = f"{text}: {', '.join(f'{a} {GRADE_NAMES[x]}' for a, x in grades.items())}"
    if no_verdict:
        text = f"{text} [{NO_VERDICT}: {', '.join(no_verdict)}]"
    return {"code": code, "plan": text, "negative": neg, "no_verdict": list(no_verdict)}


def main_test(l_by_d: ByDataset, arms: Sequence[str], c: str = "C", *,
              margins: Mapping[str, float] | None, alpha: float = ALPHA,
              level: float = T_LEVEL, min_complete: int = MIN_COMPLETE) -> dict:
    """The 8 contrasts C - B_k of L (Welch, Holm), B*_d, C - B*_d and S_d."""
    fam = family(l_by_d, arms, c, alpha=alpha, level=level, margins=margins,
                 min_complete=min_complete)
    bs = strongest(l_by_d, arms, min_complete=min_complete)
    bt = bstar_test(l_by_d, bs, c, alpha=alpha, level=level, min_complete=min_complete)
    cats = {}
    for d in l_by_d:
        grades = {fam[key(b, d)]["b"]: fam[key(b, d)]["grade"] for b in arms}
        nv = [fam[key(b, d)]["b"] for b in arms if fam[key(b, d)]["no_verdict"]]
        cats[d] = category(grades, nv, bt[d])
    return {"contrasts": fam, "strongest": bs, "bstar": bt, "categories": cats,
            "no_verdict": sorted(k for k, r in fam.items() if r["no_verdict"])}


# --- condition (2) --------------------------------------------------------------------------

def lower(boot: Mapping | None, level: float = BOOT_LEVEL) -> float:
    return st22.interval(boot, level)[0]


def dataset_met(option: str, cat: Mapping, contrasts: Sequence[Mapping],
                bstar_row: Mapping, boot_contrasts: Mapping[str, Mapping],
                boot_bstar: Mapping | None, level: float = BOOT_LEVEL) -> bool:
    """One dataset under D5 ``option`` (PLAN.md "조건 (2)")."""
    lowers_ok = all(math.isfinite(lower(boot_contrasts.get(r["b"]), level))
                    and lower(boot_contrasts.get(r["b"]), level) > 0 for r in contrasts)
    if option == "a":
        return cat["code"] in (1, 2) and lowers_ok
    if option == "b":
        lo = lower(boot_bstar, level)
        return bstar_row.get("conclusion") == "positive" and math.isfinite(lo) and lo > 0 \
            and cat["code"] != 3
    if option == "c":
        return cat["code"] == 1 and lowers_ok
    raise ValueError(f"unknown D5 option {option!r}")


def condition_2_one(main: Mapping, boot: Mapping[str, Mapping[str, Mapping]],
                    boot_bstar: Mapping[str, Mapping], option: str, requirement: str, *,
                    d2: str, gates_ok: bool, level: float = BOOT_LEVEL) -> dict:
    """Condition (2) under one D5 option and dataset requirement (``boot``: dataset -> arm ->
    the L bootstrap of C - arm; ``boot_bstar``: dataset -> that of C - B*_d)."""
    datasets = list(main["categories"])
    met = {}
    for d in datasets:
        rows = [r for k, r in main["contrasts"].items() if r["dataset"] == d]
        met[d] = dataset_met(option, main["categories"][d], rows, main["bstar"][d],
                             boot.get(d) or {}, boot_bstar.get(d), level)
    if requirement == "i":
        ok = all(met.values())
    elif requirement == "ii":
        ok = any(met.values()) and all(met[d] or main["categories"][d]["code"] != 3
                                       for d in datasets)
    else:
        raise ValueError(f"unknown dataset requirement {requirement!r}")
    if d2 != "a":
        status = NOT_COUNTED
    elif not gates_ok:
        status = GATE_FAILED
    elif main["no_verdict"]:
        status = PENDING
    else:
        status = MET if ok else NOT_MET
    return {"option": option, "requirement": requirement, "met_by_dataset": met,
            "met_datasets": [d for d in datasets if met[d]], "status": status,
            "plan": STATUS_NAMES[status]}


def condition_2(mains: Mapping[str, Mapping], boot: Mapping, boot_bstar: Mapping, *,
                decisions: Decisions = RECORDED, gates_ok: bool,
                level: float = BOOT_LEVEL) -> dict:
    """Condition (2), the only gate: with D5 / D6 decided the one result, else every valid
    combination (``mains``: D6 option -> main_test under its margins)."""
    combos = {}
    d6s = list(mains)
    for d6 in d6s:
        for opt in ((decisions.d5,) if decisions.decided else D5_OPTIONS):
            for req in ((decisions.d5_requirement,) if decisions.decided else REQUIREMENTS):
                name = f"D5({opt})/{req}/D6({d6})"
                if opt == "c" and MARGIN_L[d6] is None:
                    combos[name] = {"status": NA, "plan": "D5 (c)는 D6 (a)·(b)를 전제로 함"}
                    continue
                combos[name] = condition_2_one(mains[d6], boot, boot_bstar, opt, req,
                                               d2=decisions.d2, gates_ok=gates_ok, level=level)
    if decisions.decided:
        res = next(iter(combos.values()))
        return {"decided": True, **res, "combinations": combos}
    return {"decided": False, "status": UNDECIDED, "plan": STATUS_NAMES[UNDECIDED],
            "combinations": combos}


# --- condition (3) --------------------------------------------------------------------------

def pooled_wording(row: Mapping, boot_lower: float, margin: float, flagged: bool,
                   alpha: float = ALPHA) -> str:
    """PLAN.md "합산 지표 문구 규칙" of one contrast (codes of WORDING)."""
    sig = row["p_holm"] < alpha
    if not flagged and sig and row["mean"] > 0 and math.isfinite(boot_lower) and boot_lower > 0:
        return "better"
    if not flagged and st20.within(row["ci"], margin):
        return "same"
    if sig and row["mean"] <= -margin:
        return "substantive_inferiority"
    return "no_substantive_inferiority_detected"


def condition_3(pooled_by_d: ByDataset, arms: Sequence[str], c: str = "C", *,
                boot: Mapping[str, Mapping[str, Mapping]], alpha: float = ALPHA,
                level: float = T_LEVEL, margins: Mapping[str, float] = MARGIN_POOLED,
                min_complete: int = MIN_COMPLETE) -> dict:
    """Condition (3), a check (not a gate): Welch over the 8 contrasts of the pooled main metric
    (Holm, alpha), substantive inferiority, the flag "합산 지표 열세", the wording of each
    contrast (``boot``: dataset -> arm -> pooled bootstrap of C - arm) and the 98% lower
    bounds."""
    fam = family(pooled_by_d, arms, c, alpha=alpha, level=level, margins=margins,
                 min_complete=min_complete)
    inferior, flagged, wording, lowers98 = [], [], {}, {}
    for k, r in fam.items():
        m = margins[r["dataset"]]
        sig = r["p_holm"] < alpha
        if sig and r["mean"] <= -m:
            inferior.append(k)
        if (sig and r["mean"] < 0) or (not sig and r["mean"] <= -m):
            flagged.append(k)
        b = lower((boot.get(r["dataset"]) or {}).get(r["b"]))
        wording[k] = {"code": pooled_wording(r, b, m, k in flagged, alpha), "boot_lower_99": b}
        wording[k]["text"] = WORDING[wording[k]["code"]]
        lowers98[k] = r["ci"][0]
    not_worse = bool(fam) and all(math.isfinite(lowers98[k]) and
                                  lowers98[k] > -margins[fam[k]["dataset"]] for k in fam)
    status = "passed" if not inferior else "not_passed"
    plan = C3_PASS if not inferior else C3_FAIL.format(items=", ".join(inferior))
    return {"contrasts": fam, "substantive_inferiority": inferior, "status": status,
            "plan": plan, "pooled_inferiority_flag": flagged, "wording": wording,
            "lower_98": lowers98, "not_substantively_worse_descriptive": not_worse,
            "not_substantively_worse_text": NOT_WORSE_TEXT if not_worse else None,
            "narrowed_claim": None if not inferior else
            "위치 찾기 우위는 (" + ", ".join(inferior) + ") 대비 합산 지표 열세와 함께 온다",
            "abstract_pooled_improvement_allowed": False}


# --- mechanism (question 3) -----------------------------------------------------------------

MECHANISM = {"M1": ("C", "C.L20"), "M1.last": ("C.last", "C.L20.last")}


def mechanism(l_by_d: ByDataset, *, alpha: float = ALPHA, level: float = T_LEVEL,
              pairs: Sequence[str] = PAIRS, seeds: Sequence[int] = SEEDS) -> dict:
    """M1 = C - C.L20 and M1.last = C.last - C.L20.last of L, seed-paired t on the pair means,
    Holm over the 4 contrasts; the wording per dataset (both positive: raised; M1 negative:
    lowered; else not established)."""
    rows = {}
    for name, (a, b) in MECHANISM.items():
        for d, vals in l_by_d.items():
            kept, diffs = st22.seed_combination(vals, {a: 1, b: -1}, pairs, seeds)
            rows[f"{name}/{d}"] = {"a": a, "b": b, "dataset": d, "seeds": kept,
                                   "diffs": diffs, **st22.paired(diffs, level)}
    adj = holm({k: r["p"] for k, r in rows.items()})
    for k, r in rows.items():
        r["p_holm"] = adj[k]
        r["conclusion"] = conclusion(adj[k], r["mean"], alpha)
    wording = {}
    for d in l_by_d:
        m1, last = rows[f"M1/{d}"]["conclusion"], rows[f"M1.last/{d}"]["conclusion"]
        wording[d] = MECH_RAISED if m1 == last == "positive" else \
            MECH_LOWERED if m1 == "negative" else MECH_NONE
    return {"contrasts": rows, "wording": wording,
            "other_heads": "not tested (D10: no +O arm; a limitation)"}


# --- flags ----------------------------------------------------------------------------------

def _changed(main: Mapping, other: Mapping) -> list[str]:
    return [k for k, r in main.items() if r.get("grade") != (other.get(k) or {}).get("grade")]


def _rename(rows: Mapping[str, Mapping], suffix: str) -> dict:
    """'<arm><suffix>/<d>' rows keyed '<arm>/<d>' (grade comparisons across rules)."""
    out = {}
    for k, r in rows.items():
        arm, d = k.rsplit("/", 1)
        out[key(arm[:-len(suffix)] if suffix and arm.endswith(suffix) else arm, d)] = r
    return out


def setting_family(vals_by_d: ByDataset, arms: Sequence[str], c: str = "C", *,
                   alpha: float = ALPHA, level: float = T_LEVEL,
                   min_complete: int = MIN_COMPLETE) -> dict:
    """C - B per setting (dataset / pair; each pair's own seed values), Welch, Holm over all
    (4 settings x 4 methods = 16)."""
    rows = {}
    for d, vals in vals_by_d.items():
        for p in PAIRS:
            for b in arms:
                rows[f"{b}/{d}/{p}"] = contrast_row(vals, c, b, level=level,
                                                    min_complete=min_complete, pairs=(p,))
    adj = holm({k: r["p"] for k, r in rows.items()})
    for k, r in rows.items():
        r["p_holm"] = adj[k]
        r["conclusion"] = conclusion(adj[k], r["mean"], alpha) if not r["no_verdict"] else "none"
    return rows


def collapsed(v_auc: float | None, threshold: float = COLLAPSE_V_AUC) -> bool:
    """PLAN.md "실패 처리" "붕괴 제외" (user, 2026-10-20; validation only): a chain whose selected
    validation V-AUC of the sigma readout (fraction) is below ``threshold`` (0.60: below it
    collapses, 0.60 itself does not) leaves every rule of the chain, like a numerical failure; no
    rerun, no replacement seed. None or a non-finite value (no selection: a numerical failure,
    handled as such) is not a collapse."""
    return v_auc is not None and math.isfinite(float(v_auc)) and float(v_auc) < threshold


def degenerate(constant_share: Mapping[tuple[str, str, int, str], float],
               arms: Sequence[str]) -> dict:
    """"비교 기법 퇴화" per arm and dataset (PLAN.md "실패 처리" "퇴화 표지"; a flag, the run is
    kept): a run whose test scores are constant inside more than half of the videos
    (``constant_share``: (dataset, pair, seed, arm) -> share; the caller leaves out only the
    collapsed runs, so a numerically failed run with finite test scores can still raise it). The
    former validation criterion (>= 3 seeds below V-AUC 0.60) is now the exclusion rule
    ``collapsed``."""
    out = {}
    for d in DATASETS:
        for a in arms:
            const = sorted(f"{p}/s{s}" for (dd, p, s, aa), v in constant_share.items()
                           if dd == d and aa == a and v > 0.5)
            if const:
                out[key(a, d)] = {"constant_runs": const}
    return out


def flags(*, main: Mapping, arms: Sequence[str], l_sigma_main: Mapping,
          l_logit_rule_main: Mapping, mwu: Mapping, paired_t: Mapping, settings: Mapping,
          last_rule_main: Mapping, boot: Mapping[str, Mapping[str, Mapping]],
          used_in_condition_2: Mapping[str, Sequence[str]] | None, x32_main: Mapping | None,
          x32_reference: Mapping | None, opposite_main: Mapping, condition3: Mapping,
          degenerate_rows: Mapping, weak: Sequence[str], alpha: float = ALPHA,
          level: float = BOOT_LEVEL) -> dict:
    """PLAN.md "표지" (each a list of '<arm>/<dataset>' or '<arm>/<dataset>/<pair>'):

    setting_dependent           a significantly negative C - B per setting (Holm over 16)
    readout_sensitive           G_kd changes with the comparison method's sigma-fp32 readout L
    selection_readout_sensitive G_kd changes with its logit-best checkpoint
    distribution_sensitive      Mann-Whitney U or the seed-number paired t (each Holm over 8)
                                significant where Welch is not, or the reverse
    checkpoint_rule_dependent   C.last - B.last grades differ
    sensitive_to_test_video_sample   a positive or negative grade whose 99% L bootstrap holds 0,
                                among the contrasts not used by condition (2) (per D5 option
                                while undecided: ``used_in_condition_2``)
    training_unit_sensitive     C - X.32 grades (Holm over the X.32 family) differ from C - X
    l_metric_kind_sensitive     grades change with the opposite within-video metric
    pooled_inferiority          condition (3)'s flag
    baseline_degenerate         ``degenerate`` (test-constant runs, kept)
    numerically_unstable        the "판정 없음" contrasts (numerical failures or collapses)
    weak_under_adaptation       (q1d)"""
    fam = main["contrasts"]
    sig = {k: r["p_holm"] < alpha for k, r in fam.items()}
    out = {"setting_dependent": sorted(k for k, r in settings.items()
                                       if r["conclusion"] == "negative"),
           "readout_sensitive": _changed(fam, l_sigma_main),
           "selection_readout_sensitive": _changed(fam, _rename(l_logit_rule_main, ".logit")),
           "distribution_sensitive": sorted(
               k for k in fam if not fam[k]["no_verdict"] and (
                   (mwu[k]["p_holm"] < alpha) != sig[k]
                   or (paired_t[k]["p_holm"] < alpha) != sig[k])),
           "checkpoint_rule_dependent": _changed(fam, _rename(last_rule_main, ".last")),
           "l_metric_kind_sensitive": _changed(fam, opposite_main),
           "pooled_inferiority": list(condition3["pooled_inferiority_flag"]),
           "baseline_degenerate": dict(degenerate_rows),
           "numerically_unstable": list(main["no_verdict"]),
           "weak_under_adaptation": list(weak)}
    signed = [k for k, r in fam.items() if r["grade"] in (*POSITIVE, NEG)]
    holds0 = [k for k in signed if st22._contains_0(st22.interval(
        (boot.get(fam[k]["dataset"]) or {}).get(fam[k]["b"]), level))]
    if used_in_condition_2 is None:
        out["sensitive_to_test_video_sample"] = holds0
    else:
        out["sensitive_to_test_video_sample"] = {
            opt: sorted(k for k in holds0 if k not in set(used)) for opt, used in
            used_in_condition_2.items()}
    if x32_main is None or x32_reference is None:
        out["training_unit_sensitive"] = None
    else:                         # x32_main: keyed by the main-family arm of each X.32 arm
        out["training_unit_sensitive"] = sorted(
            k for k, r in x32_main.items()
            if r.get("grade") != (x32_reference.get(k) or {}).get("grade"))
    return out


def raised(flag) -> bool:
    if isinstance(flag, Mapping):
        return any(bool(v) for v in flag.values())
    return bool(flag)


def used_contrasts(main: Mapping, option: str | None) -> dict[str, list[str]] | None:
    """The contrasts whose bootstrap condition (2) uses: (a) / (c) all eight, (b) C - B*_d."""
    def of(opt):
        if opt in ("a", "c"):
            return list(main["contrasts"])
        return [key(b["b"], d) for d, b in main["bstar"].items() if b.get("b")]
    if option is None:
        return {o: of(o) for o in D5_OPTIONS}
    return {option: of(option)}


# --- next step, claims ------------------------------------------------------------------------

def next_step(cond2: Mapping, mech: Mapping, cond3: Mapping, *,
              decisions: Decisions = RECORDED) -> dict:
    """PLAN.md "결과의 쓰임" (D13 (a) table): the row from condition (2) under the chosen D5 and
    the mechanism wording of the datasets that met it."""
    status = cond2.get("status")
    common = {"condition_3": cond3["plan"], "narrowed_claim": cond3.get("narrowed_claim"),
              "statistical_only_wording": None}
    if status == UNDECIDED:
        return {"row": "d5_undecided", "paper": None, **common,
                "user_decides": ["D5", "D6", "D7 재질문"]}
    if status == MET:
        ds = cond2["met_datasets"]
        raised_all = bool(ds) and all(mech["wording"].get(d) == MECH_RAISED for d in ds)
        row = "method_paper" if raised_all else "method_paper_limited_wording"
        paper = ("방법 논문(TCSVT): 위치 찾기 우위와 학습 중 오프셋 기전 문구" if raised_all else
                 "방법 논문(TCSVT), 문구 제한: C 파이프라인의 위치 찾기 우위만, 오프셋 덕분이라고 "
                 "쓰지 않음")
        return {"row": row, "paper": paper, "adopt_c": decisions.d2 == "a", **common,
                "user_decides": []}
    if status == NOT_MET:
        return {"row": "not_met", "paper": "방법 논문 경로를 닫음", **common,
                "user_decides": ["감사·평가 논문(D14 자료) 또는 중단"]}
    return {"row": "user_decision", "paper": None, **common,
            "user_decides": [f"조건 (2) {STATUS_NAMES.get(status, status)}"]}


def claims(main_by_d6: Mapping, cond2: Mapping, cond3: Mapping) -> dict:
    """PLAN.md "지표 고정과 부가 주장": no secondary improvement is claimed (abstract, claim 1, the
    gate), never an abstract-level pooled improvement."""
    return {"secondary_improvements_allowed": False,
            "abstract_pooled_improvement_allowed": False,
            "localisation_claim_datasets": list(cond2.get("met_datasets") or []),
            "statistical_only": {d6: [d for d, c in m["categories"].items() if c["code"] == 2]
                                 for d6, m in main_by_d6.items()},
            "pooled_wording": {k: v["text"] for k, v in cond3["wording"].items()}}


# --- the whole judgement ----------------------------------------------------------------------

def _flags_of(main: Mapping, fam: Sequence[str], x32_arms: Mapping[str, str], l_logit: ByDataset,
              l_sigma: ByDataset, l_opposite: ByDataset, margins: Mapping[str, float] | None,
              shared: Mapping, boot_l: Mapping, cond3: Mapping, decisions: Decisions,
              degenerate_rows: Mapping, weak: Sequence[str], alpha: float) -> dict:
    """The flags under one D6 option (its margins grade every comparison)."""
    def graded(vals, arms, c="C"):
        return family(vals, arms, c, alpha=alpha, margins=margins)
    x32_main = None
    if x32_arms:
        rows = graded(l_logit, list(x32_arms))
        x32_main = {key(x32_arms[r["b"]], r["dataset"]): r for r in rows.values()}
    return flags(main=main, arms=fam, l_sigma_main=graded(l_sigma, fam),
                 l_logit_rule_main=graded(l_logit, [f"{a}.logit" for a in fam]),
                 mwu=shared["mwu"], paired_t=shared["paired_t"], settings=shared["settings"],
                 last_rule_main=graded(l_logit, [f"{a}.last" for a in fam], "C.last"),
                 boot=boot_l, used_in_condition_2=used_contrasts(main, decisions.d5),
                 x32_main=x32_main, x32_reference=main["contrasts"] if x32_arms else None,
                 opposite_main=graded(l_opposite, fam), condition3=cond3,
                 degenerate_rows=degenerate_rows, weak=weak, alpha=alpha)


def evaluate(*, l_logit: ByDataset, l_sigma: ByDataset, l_opposite: ByDataset,
             pooled: ByDataset, family_arms: Sequence[str], x32_arms: Mapping[str, str],
             boot_l: Mapping[str, Mapping[str, Mapping]],
             boot_l_bstar: Mapping[str, Mapping[str, Mapping]] | None = None,
             boot_pooled: Mapping[str, Mapping[str, Mapping]],
             degenerate_rows: Mapping | None = None, weak: Sequence[str] = (),
             gates_ok: bool, decisions: Decisions = RECORDED,
             alpha: float = ALPHA) -> dict:
    """Condition (2), condition (3), the mechanism, the flags, the next step and the claims.

    l_logit     L (logit readout) of every arm and rule (C, C.last, the family and its .logit /
                .last rules, X.32, C.L20, C.L20.last)
    l_sigma     L of the sigma-fp32 readout (the comparison methods, readout flag)
    l_opposite  the opposite within-video metric (logit readout)
    pooled      the pooled main metric (UCF frame AUC, XD frame AP), logit readout
    x32_arms    X.32 arm -> its main-family arm (URDMU.32 -> URDMU)
    boot_l      dataset -> arm -> L bootstrap of C - arm (main family)
    boot_l_bstar   dataset -> {"bstar": the L bootstrap of C - B*_d} (defaults to boot_l)
    boot_pooled dataset -> arm -> pooled bootstrap of C - arm"""
    fam = list(family_arms)
    mains = {d6: main_test(l_logit, fam, margins=MARGIN_L[d6], alpha=alpha)
             for d6 in decisions.margin_options()}
    first = next(iter(mains.values()))
    bstar_boot = {d: ((boot_l_bstar or {}).get(d) or {}).get("bstar")
                  or (boot_l.get(d) or {}).get(first["strongest"][d]["arm"])
                  for d in l_logit}
    cond2 = condition_2(mains, boot_l, bstar_boot, decisions=decisions, gates_ok=gates_ok)
    cond3 = condition_3(pooled, fam, boot=boot_pooled, alpha=alpha)
    mech = mechanism(l_logit, alpha=alpha)
    # the readout flag changes the comparison methods' readout only: C keeps its logit L
    sigma = {d: {**dict(l_sigma.get(d) or {}), **{k: v for k, v in l_logit[d].items()
                                                  if k[2] == "C"}} for d in l_logit}
    shared = {"mwu": family(l_logit, fam, kind="mwu", alpha=alpha, margins=None),
              "paired_t": family(l_logit, fam, kind="paired", alpha=alpha, margins=None),
              "settings": setting_family(l_logit, fam, alpha=alpha)}
    flag_sets = {d6: _flags_of(main, fam, x32_arms, l_logit, sigma, l_opposite, MARGIN_L[d6],
                               shared, boot_l, cond3, decisions, degenerate_rows or {}, weak,
                               alpha)
                 for d6, main in mains.items()}
    step = next_step(cond2, mech, cond3, decisions=decisions)
    return {"decisions": {"d2": decisions.d2, "d5": decisions.d5,
                          "d5_requirement": decisions.d5_requirement, "d6": decisions.d6,
                          "d9": decisions.d9, "d13": decisions.d13},
            "alpha": alpha, "t_level": T_LEVEL, "boot_level": BOOT_LEVEL,
            "margins_pooled": MARGIN_POOLED, "margins_l": decisions.margin_options(),
            "main": mains, "condition_2": cond2, "condition_3": cond3, "mechanism": mech,
            "flags": flag_sets,
            "flags_raised_plan": {d6: [FLAG_NAMES[k] for k, v in f.items() if raised(v)]
                                  for d6, f in flag_sets.items()},
            "next_step": step, "claims": claims(mains, cond2, cond3)}


# --- video bootstraps -------------------------------------------------------------------------

def l_weights(n_videos: int, n_resamples: int = BOOT_R, seed: int = BOOT_SEED) -> np.ndarray:
    """(R + 1, n) draw counts of the included anomalous videos: row 0 every video once, rows
    1..R n draws with replacement from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_resamples + 1, int(n_videos)), dtype=np.int64)
    w[0] = 1
    draws = rng.integers(0, int(n_videos), size=(n_resamples, int(n_videos)))
    rows = np.broadcast_to(np.arange(1, n_resamples + 1)[:, None], draws.shape)
    np.add.at(w, (rows, draws), 1)
    return w


def l_resampled(per_video: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(R + 1,) L of one run on every resample: the weighted mean of its per-video values."""
    v = np.asarray(per_video, dtype=np.float64)
    if v.ndim != 1 or v.shape[0] != weights.shape[1]:
        raise ValueError(f"{v.shape} per-video values for {weights.shape[1]} videos")
    return (weights @ v) / weights.shape[1]


def _combine(per: Mapping[tuple[str, int, str], np.ndarray], a: str, b: str,
             seeds_a: Sequence[int], seeds_b: Sequence[int],
             pairs: Sequence[str]) -> np.ndarray:
    if list(seeds_a) == list(seeds_b):           # e022_stats.bootstrap_contrast's expression
        return np.mean([np.mean([per[(p, s, a)] - per[(p, s, b)] for p in pairs], axis=0)
                        for s in seeds_a], axis=0)
    ma = np.mean([np.mean([per[(p, s, a)] for p in pairs], axis=0) for s in seeds_a], axis=0)
    mb = np.mean([np.mean([per[(p, s, b)] for p in pairs], axis=0) for s in seeds_b], axis=0)
    return ma - mb


def _summary(diff: np.ndarray, levels: Sequence[float], meta: dict) -> dict:
    if not np.isfinite(diff).all():
        raise ValueError(f"{int(np.sum(~np.isfinite(diff)))} non-finite bootstrap differences")
    keys = {st22.level_key(x): x for x in levels}
    intervals = {k: st22.percentile_interval(diff[1:], x) for k, x in keys.items()}
    return meta | {"R": int(len(diff) - 1), "resample0": float(diff[0]),
                   "mean": float(diff[1:].mean()), "intervals": intervals,
                   "contains_0": {k: bool(v[0] <= 0 <= v[1]) for k, v in intervals.items()}}


def _empty(levels: Sequence[float], meta: dict) -> dict:
    keys = [st22.level_key(x) for x in levels]
    return meta | {"resample0": math.nan, "mean": math.nan,
                   "intervals": dict.fromkeys(keys, [math.nan, math.nan]),
                   "contains_0": dict.fromkeys(keys, False), "point_check_max_abs_diff": math.nan}


def l_bootstrap(per_video: Mapping[tuple[str, int, str], np.ndarray], weights: np.ndarray,
                a: str, b: str, seeds_a: Sequence[int], seeds_b: Sequence[int], *,
                levels: Sequence[float] = BOOT_LEVELS,
                point: Mapping[tuple[str, int, str], float] | None = None,
                pairs: Sequence[str] = PAIRS, cache: dict | None = None) -> dict:
    """Video bootstrap of the L contrast a - b (module docstring); ``per_video``: (pair, seed,
    arm) -> the run's per-video L in the included-video order; resample 0 must reproduce
    ``point`` (largest |difference| recorded); ``cache`` keeps the per-run resampled L."""
    meta = {"metric": "L", "contrast": f"{a}_minus_{b}", "seeds_a": list(seeds_a),
            "seeds_b": list(seeds_b), "levels": [float(x) for x in levels],
            "seed": BOOT_SEED}
    if not seeds_a or not seeds_b:
        return _empty(levels, meta)
    cache = {} if cache is None else cache
    per, check = {}, 0.0
    for arm, seeds in ((a, seeds_a), (b, seeds_b)):
        for p in pairs:
            for s in seeds:
                k = (p, s, arm)
                if k not in cache:
                    cache[k] = l_resampled(per_video[k], weights)
                per[k] = cache[k]
                if point is not None:
                    d = abs(float(per[k][0]) - float(point[k]))
                    check = max(check, d) if math.isfinite(d) else math.inf
    return _summary(_combine(per, a, b, seeds_a, seeds_b, pairs), levels, meta) | {
        "point_check_max_abs_diff": check}


def pooled_bootstrap(atoms: Mapping[tuple[str, int, str], st20.Atoms | Callable[[], st20.Atoms]],
                     weights: np.ndarray, metric: str, a: str, b: str,
                     seeds_a: Sequence[int], seeds_b: Sequence[int], *,
                     levels: Sequence[float] = BOOT_LEVELS,
                     point: Mapping[tuple[str, int, str], float] | None = None,
                     pairs: Sequence[str] = PAIRS, cache: dict | None = None,
                     chunk: int = st20.BOOT_CHUNK) -> dict:
    """Video bootstrap of the pooled contrast a - b (``weights``: e020_stats.bootstrap_weights of
    the test labels; each run's metric by e020_stats.weighted_auc_ap of its atoms, cached per
    run)."""
    if metric not in ("frame_auc", "frame_ap"):
        raise ValueError(f"unknown metric {metric!r}")
    meta = {"metric": metric, "contrast": f"{a}_minus_{b}", "seeds_a": list(seeds_a),
            "seeds_b": list(seeds_b), "levels": [float(x) for x in levels],
            "seed": BOOT_SEED}
    if not seeds_a or not seeds_b:
        return _empty(levels, meta)
    pick = 0 if metric == "frame_auc" else 1
    cache = {} if cache is None else cache
    per, check = {}, 0.0
    for arm, seeds in ((a, seeds_a), (b, seeds_b)):
        for p in pairs:
            for s in seeds:
                k = (p, s, arm)
                if k not in cache:
                    at = atoms[k]() if callable(atoms[k]) else atoms[k]
                    cache[k] = st20.weighted_auc_ap(at, weights, chunk)[pick]
                per[k] = cache[k]
                if point is not None:
                    d = abs(float(per[k][0]) - float(point[k]))
                    check = max(check, d) if math.isfinite(d) else math.inf
    return _summary(_combine(per, a, b, seeds_a, seeds_b, pairs), levels, meta) | {
        "point_check_max_abs_diff": check}
