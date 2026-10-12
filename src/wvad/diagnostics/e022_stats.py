"""Statistics of the E022 verdict and its pre-registered analyses (PLAN.md "판정 (사전 등록)",
"표지", "분해 대비", "결과의 쓰임"; review F17). Local analysis code, fixed at the freeze commit
(PLAN.md "코드 규칙" 3); pure functions of per-run metric values and of the video bootstrap.

    tests       seed-paired two-sided t (df n - 1) and Wilcoxon signed-rank (zero_method 'pratt',
                method 'auto') per dataset, Holm over the datasets (or the four settings); the
                family level alpha and the seed t interval level are arguments (D5 (a): alpha
                0.01, interval 1 - 2 alpha = 98%; D5 (b): 0.005, 99%)
    grades      G_d, first match: substantive positive (Holm p < alpha, D_d >= m_d; D4 (b): the
                video-bootstrap lower bound at the D11 level >= m_d instead), statistical positive
                (Holm p < alpha, D_d > 0, not 1: under D4 (b) also D_d >= m_d with a lower bound
                < m_d), negative, equivalent (seed t interval inside (-m_d, +m_d)), inconclusive
    verdict     E020's categories and table (first match), named "미충족(동등)" when both datasets
                are equivalent;
                no verdict when a dataset has fewer than 8 complete seeds: "관문 실패(판정 없음)"
                with >= 3 seeds of a dataset dropped for reasons other than a numerical failure of
                the HB chain (treatment), otherwise "처치 수치 불안정(사용자 결정)", reported with
                the imputed result (each failed C run takes the lowest C value of its dataset and
                pair, ``e020_stats.impute_minimum``)
    condition 2 (1) verdict "낫다(실질)", (2) D12 approved and D6 counted ((a) or (c)), (3) with D10
                (a) / (c) the bootstrap interval at the D11 level excludes 0 in both datasets
                (lower > 0), with D10 (c) also no blocking flag (re-extraction driven, also when it
                cannot be computed; within-video localisation harm (substantive); length / offset
                driven), (4) no treatment-related numerical failure (else: the user decides)
    flags       setting dependent, within-video localisation harm (and its substantive form,
                L mean <= -m_d), length / offset driven, readout, distribution, checkpoint rule,
                extreme readout, test-video sample, re-extraction driven, no snippet-resolution
                share, dense numerical instability, E020 baselines (D7 (a))
    P9          "나쁨": "나쁨(실질)" when a negative dataset has D_d <= -m_d (E3 stops), else "작은
                음(문턱 미만)" (E3: the user decides)
    bootstrap   anomalous and normal videos resampled separately, R = 5000 (argument),
                default_rng(20261012), the same resamples for every run, percentile intervals at
                the given levels (D11 (a): 99%, and 95%); resample 0 is the original sample.
                ``unit_atoms`` builds the frame-metric atoms of constant-score units whose number
                differs between videos (32 segments, snippet supports, evaluator windows)

wvad.diagnostics.e020_stats is reused unchanged; at its defaults (alpha 0.05, 90% equivalence
interval, R 2000, seed 20261006, 95% percentile interval) the functions here return its values
(unit test).

Values are maps (pair, seed, arm) -> metric in points for one dataset; failed runs are absent and
every test uses the complete cases (seeds with every needed run in both pairs). Arms: R0, C, R1,
B, A1.S, C.W, C.G, C.last, C.o32, R0.last (and the E020 cells L20, base under D7 (a)).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import stats

from wvad.diagnostics import e020_stats as st
from wvad.diagnostics.e018 import CONFIRMATION_SEEDS
from wvad.diagnostics.runs import PAIRS
from wvad.eval.metrics import DEFAULT_MIN_FRAMES
from wvad.extract import grid

DATASETS = ("ucf", "xdv")
SEEDS = CONFIRMATION_SEEDS
MARGIN = dict(st.MARGIN)                  # m_d: UCF frame AUC 0.5, XD frame AP 2.0 points
NAMES = {"ucf": "UCF", "xdv": "XD"}
ALPHA = 0.01                              # D5 (a)
MIN_COMPLETE = 8                          # complete seeds per dataset for a verdict
BOOT_R, BOOT_SEED = 5000, 20261012
BOOT_LEVELS = (0.99, 0.95)                # D11 (a) at D5 (a), and the 95% interval
MAIN = ("C", "R0")
SECONDARY = ("C.last", "R0.last")         # the secondary checkpoint rule (epoch 30)
EXTREME = ("C.G", "R0")
RESOLUTION = ("C.W", "R0")
# the HB chains' arms whose numerical failures are treatment-related (PLAN.md "처치 관련 수치
# 실패": C, C.G, C.last, B; the variants C.W and C.o32 are descriptive and not counted)
TREATMENT_ARMS = ("C", "C.G", "C.last", "B")
NUMERICAL, INFRASTRUCTURE, GATE = "numerical", "infrastructure", "gate"
FAILURE_KINDS = (NUMERICAL, INFRASTRUCTURE, GATE)

SUB, STAT, NEG, EQ, INC = ("substantive_positive", "statistical_positive", "negative",
                           "equivalent", "inconclusive")
POSITIVE = (SUB, STAT)
BETTER_SUB, BETTER_STAT = "better_substantive", "better_statistical_only"
DEPENDENT, WORSE, ONE = "dataset_dependent", "worse", "one_dataset_significant"
NOT_MET = "not_met"
GATE_FAILED, TREATMENT_UNSTABLE = "gate_failure_no_verdict", "treatment_numerically_unstable"
PLAN_NAMES = {SUB: "실질 양", STAT: "통계 양", NEG: "음", EQ: "동등", INC: "결론 없음",
              BETTER_SUB: "낫다(실질)", BETTER_STAT: "낫다(통계적으로만)",
              DEPENDENT: "데이터셋 의존", WORSE: "나쁨", ONE: "한 데이터셋만 유의",
              NOT_MET: "미충족", GATE_FAILED: "관문 실패(판정 없음)",
              TREATMENT_UNSTABLE: "처치 수치 불안정(사용자 결정)"}
# condition (2): status codes and the reasons of "미충족(사유)"
MET, DEFERRED, NOT_MET_REASONS = "met", "deferred_to_user", "not_met_with_reasons"
REASONS = {"principle_2_exception_not_approved": "원칙 2 예외 미승인",
           "f1_format_not_counted": "F1 형식(D6 (b))", "video_sample": "영상 표본",
           "blocking_flags": "막는 표지(D10 (c))",
           "treatment_numerical_failure": "처치 관련 수치 실패"}
FLAG_NAMES = {"setting_dependent": "설정 의존", "localisation_harm": "영상 안 국소화 손해",
              "localisation_harm_substantive": "영상 안 국소화 손해(실질)",
              "length_offset_driven": "길이·오프셋 주도 가능", "readout_sensitive": "판독 민감",
              "distribution_sensitive": "분포 민감",
              "checkpoint_rule_dependent": "체크포인트 규칙 의존",
              "extreme_readout_sensitive": "극값 판독 민감",
              "sensitive_to_test_video_sample": "시험 영상 표본에 민감",
              "reextraction_driven": "재추출 주도 가능",
              "no_resolution_share": "스니펫 해상도에서 오는 몫 없음",
              "dense_numerically_unstable": "조밀 수치 불안정",
              "not_better_than_e020_baselines": "E020 기준(L20·base) 대비 개선 아님"}
P9_WORSE = ("조밀 표현(평균 학습·스니펫 채점): 시험 프레임 지표에서 나쁨(실질)(E022, M10, "
            "{datasets})")
P9_SMALL = ("조밀 표현(평균 학습·스니펫 채점): 시험 프레임 지표에서 작은 음(문턱 미만; E022, M10, "
            "{datasets})")
P9_UNSTABLE = "조밀 표현: 시험에서 수치 불안정(E022, M10, {datasets}; 대치 결과 {imputed})"
# decomposition contrasts (PLAN.md "분해 대비"): name -> arm weights, the decisions they need
DECOMPOSITION = {"R1-R0": ({"R1": 1, "R0": -1}, "r1"), "C-R1": ({"C": 1, "R1": -1}, "r1"),
                 "B-R1": ({"B": 1, "R1": -1}, "r1"), "C-B": ({"C": 1, "B": -1}, None),
                 "C-C.o32": ({"C": 1, "C.o32": -1}, None), "C-C.W": ({"C": 1, "C.W": -1}, None),
                 "A1.S-R1": ({"A1.S": 1, "R1": -1}, "a1s"), "C-A1.S": ({"C": 1, "A1.S": -1}, "a1s"),
                 "interaction_2x2": ({"C": 1, "B": -1, "A1.S": -1, "R1": 1}, "a1s")}
IDENTITIES = {"C-R0 = (C-R1) + (R1-R0)": ("C-R0", ("C-R1", "R1-R0"), "r1"),
              "C-R1 = (C-B) + (B-R1)": ("C-R1", ("C-B", "B-R1"), "r1"),
              "C-R1 = (C-A1.S) + (A1.S-R1)": ("C-R1", ("C-A1.S", "A1.S-R1"), "a1s")}

Values = Mapping[tuple[str, int, str], float]
ByDataset = Mapping[str, Values]


# --- user decisions -------------------------------------------------------------------------

_CHOICES = {"d1": "ab", "d2": "ab", "d3": "ab", "d4": "ab", "d5": "ab", "d6": "abc", "d7": "ab",
            "d8": "ab", "d9": "ab", "d10": "abc", "d11": "ab", "d12": "ab"}


@dataclass(frozen=True)
class Decisions:
    """The user decisions D1-D12 (PLAN.md "사용자 결정 기록", 2026-10-12): D1 a, D2 a, D3 a,
    D4 a, D5 a, D6 c, D7 a, D8 a, D9 a, D10 a, D11 a, D12 a by default."""

    d1: str = "a"
    d2: str = "a"
    d3: str = "a"
    d4: str = "a"
    d5: str = "a"
    d6: str = "c"
    d7: str = "a"
    d8: str = "a"
    d9: str = "a"
    d10: str = "a"
    d11: str = "a"
    d12: str = "a"

    def __post_init__(self):
        for k, allowed in _CHOICES.items():
            if getattr(self, k) not in tuple(allowed):
                raise ValueError(f"{k.upper()} must be one of {tuple(allowed)}, got "
                                 f"{getattr(self, k)!r}")
        if self.d1 != "a":
            raise ValueError("D1 (b): E022 stops; nothing is analysed")
        if self.d6 == "c" and self.d3 != "a":
            raise ValueError("D6 (c) (A1.S) needs D3 (a) (the H1 chains)")

    @property
    def alpha(self) -> float:
        """Family level of every test (D5)."""
        return 0.01 if self.d5 == "a" else 0.005

    @property
    def t_level(self) -> float:
        """Seed t interval of the equivalence and decomposition (1 - 2 alpha: 98% / 99%)."""
        return 0.98 if self.d5 == "a" else 0.99

    @property
    def boot_level(self) -> float:
        """Video-bootstrap interval of the flags and condition (2) (D11; (a) follows D5)."""
        if self.d11 == "b":
            return 0.95
        return 0.99 if self.d5 == "a" else 0.995

    @property
    def substantive_rule(self) -> str:
        return "point" if self.d4 == "a" else "bootstrap_lower"

    @property
    def d6_effective(self) -> str:
        """D6 as recorded: (b) when D12 is not approved (PLAN.md D12 (b))."""
        return self.d6 if self.d12 == "a" else "b"

    @property
    def r1(self) -> bool:
        return self.d3 == "a"

    @property
    def a1s(self) -> bool:
        return self.d6_effective == "c"


RECORDED = Decisions()


# --- tests ----------------------------------------------------------------------------------

def _probability(level: float) -> float:
    """Upper quantile probability of a two-sided interval (0.90 -> 0.95 exactly as a literal)."""
    if not 0 < level < 1:
        raise ValueError(f"interval level must lie in (0, 1), got {level}")
    return round(0.5 + level / 2, 12)


def t_interval(mean: float, sd: float, n: int, level: float) -> list[float]:
    """Two-sided seed t interval mean +- t_{(1 + level) / 2, n - 1} sd / sqrt(n) (the
    e020_stats.paired formula); NaN with fewer than two seeds."""
    q = _probability(level)
    if n < 2:
        return [math.nan, math.nan]
    se = sd / math.sqrt(n)
    t = float(stats.t.ppf(q, n - 1))
    return [mean - t * se, mean + t * se]


def paired(diffs: Sequence[float], level: float = 0.98) -> dict:
    """``e020_stats.paired`` (n, mean, seed SD, t, p, d_z, ci90, ci95) and the interval at
    ``level`` ("ci", "level")."""
    r = st.paired(diffs)
    return r | {"ci": t_interval(r["mean"], r["sd"], r["n"], level), "level": level}


def conclusion(p_holm: float, mean: float, alpha: float = ALPHA) -> str:
    """'positive' / 'negative' when significant after Holm (p < alpha), else 'none'."""
    if p_holm < alpha and mean > 0:
        return "positive"
    if p_holm < alpha and mean < 0:
        return "negative"
    return "none"


def _with_holm(res: dict, alpha: float) -> dict:
    adj = st.holm({k: r["p"] for k, r in res.items()})
    for k, r in res.items():
        r["p_holm"] = adj[k]
        r["conclusion"] = conclusion(adj[k], r["mean"], alpha)
    return res


def seed_combination(vals: Values, weights: Mapping[str, float], pairs: Sequence[str] = PAIRS,
                     seeds: Sequence[int] = SEEDS) -> tuple[list[int], list[float]]:
    """Complete seeds (every weighted arm in every pair) and their pair-averaged sum of
    weight x value; a difference a - b is ``e020_stats.seed_diffs`` (bit for bit)."""
    w = dict(weights)
    if len(w) == 2 and sorted(w.values()) == [-1, 1]:
        a, b = (k for k in sorted(w, key=lambda k: -w[k]))
        return st.seed_diffs(vals, a, b, pairs, seeds)
    kept = [s for s in seeds if all((p, s, c) in vals for p in pairs for c in w)]
    return kept, [float(np.mean([sum(c * vals[(p, s, a)] for a, c in w.items()) for p in pairs]))
                  for s in kept]


def _row(kept: Sequence[int], diffs: Sequence[float], kind: str, level: float,
         seeds: Sequence[int]) -> dict:
    row = {"seeds": list(kept), "seeds_dropped": len(seeds) - len(kept), "diffs": list(diffs)}
    if kind == "t":
        return row | paired(diffs, level)
    if kind == "wilcoxon":
        return row | {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else math.nan,
                      "p": st.wilcoxon_p(diffs), "method": st.wilcoxon_method(diffs)}
    raise ValueError(f"unknown test {kind!r} (t, wilcoxon)")


def combination_contrast(vals_by_d: Mapping[str, Values], weights: Mapping[str, float], *,
                         kind: str = "t", alpha: float = ALPHA, level: float = 0.98,
                         pairs: Sequence[str] = PAIRS, seeds: Sequence[int] = SEEDS) -> dict:
    """Per dataset: test of the pair-averaged weighted arm sum over its complete seeds (paired t
    with the interval at ``level``, or Wilcoxon); Holm over the datasets at ``alpha``."""
    out = {}
    for d, vals in vals_by_d.items():
        kept, diffs = seed_combination(vals, weights, pairs, seeds)
        out[d] = _row(kept, diffs, kind, level, seeds)
    return _with_holm(out, alpha)


def contrast(vals_by_d: Mapping[str, Values], a: str, b: str, *, kind: str = "t",
             alpha: float = ALPHA, level: float = 0.98, pairs: Sequence[str] = PAIRS,
             seeds: Sequence[int] = SEEDS) -> dict:
    """``e020_stats.contrast`` of a - b with the level alpha and the interval level as
    arguments."""
    return combination_contrast(vals_by_d, {a: 1, b: -1}, kind=kind, alpha=alpha, level=level,
                                pairs=pairs, seeds=seeds)


def setting_contrast(vals_by_d: Mapping[str, Values], a: str, b: str, *, alpha: float = ALPHA,
                     level: float = 0.98, pairs: Sequence[str] = PAIRS,
                     seeds: Sequence[int] = SEEDS) -> dict:
    """Per setting (dataset/pair): paired t of a - b over its complete seeds; Holm over the
    settings (four)."""
    out = {}
    for d, vals in vals_by_d.items():
        for p in pairs:
            kept, diffs = st.seed_diffs(vals, a, b, (p,), seeds)
            out[f"{d}/{p}"] = _row(kept, diffs, "t", level, seeds)
    return _with_holm(out, alpha)


# --- grades and verdict ---------------------------------------------------------------------

def grade(p_holm: float, mean: float, ci: Sequence[float], margin: float, *,
          alpha: float = ALPHA, rule: str = "point", boot_lower: float | None = None) -> str:
    """Dataset grade G_d (first match): substantive positive (Holm p < alpha, D_d > 0 and
    "substantive": D_d >= m_d under D4 (a) ``rule`` 'point', the bootstrap lower bound >= m_d
    under D4 (b) 'bootstrap_lower'), statistical positive (significant, D_d > 0, not 1),
    negative (significant, D_d < 0), equivalent (``ci`` inside the open band (-m_d, +m_d)),
    inconclusive. At alpha 0.05 and rule 'point' this is ``e020_stats.grade``."""
    if rule == "point":
        substantive = mean >= margin
    elif rule == "bootstrap_lower":
        substantive = boot_lower is not None and math.isfinite(boot_lower) and \
            boot_lower >= margin
    else:
        raise ValueError(f"unknown substantive rule {rule!r} (point, bootstrap_lower)")
    significant = p_holm < alpha
    if significant and mean > 0 and substantive:
        return SUB
    if significant and mean > 0:
        return STAT
    if significant and mean < 0:
        return NEG
    return EQ if st.within(ci, margin) else INC


def verdict(grades: Mapping[str, str]) -> str:
    """Overall verdict: the E020 categories and table (``e020_stats.verdict``, first match)."""
    return st.verdict(grades)


def plan_name(verdict_code: str, grades: Mapping[str, str] | None = None) -> str:
    """The plan's name of a verdict: "미충족(동등)" when every dataset is equivalent."""
    if verdict_code == NOT_MET and grades and all(g == EQ for g in grades.values()):
        return "미충족(동등)"
    return PLAN_NAMES[verdict_code]


def interval(boot: Mapping | None, level: float) -> list[float]:
    """The bootstrap percentile interval at ``level`` ([nan, nan] when absent)."""
    if not boot:
        return [math.nan, math.nan]
    return list((boot.get("intervals") or {}).get(level_key(level), [math.nan, math.nan]))


def main_test(vals_by_d: Mapping[str, Values], a: str = "C", b: str = "R0", *,
              alpha: float = ALPHA, level: float = 0.98, margins: Mapping[str, float] = MARGIN,
              rule: str = "point", boot: Mapping[str, Mapping] | None = None,
              boot_level: float = 0.99, pairs: Sequence[str] = PAIRS,
              seeds: Sequence[int] = SEEDS) -> dict:
    """Paired t per dataset with Holm over the datasets, the grades G_d (rule 'bootstrap_lower':
    D4 (b), with ``boot`` the per-dataset bootstrap of a - b) and the verdict."""
    if rule == "bootstrap_lower" and boot is None:
        raise ValueError("D4 (b) grades need the video bootstrap of the contrast")
    res = contrast(vals_by_d, a, b, alpha=alpha, level=level, pairs=pairs, seeds=seeds)
    lower = {d: interval((boot or {}).get(d), boot_level)[0] for d in res}
    grades = {d: grade(r["p_holm"], r["mean"], r["ci"], margins[d], alpha=alpha, rule=rule,
                       boot_lower=lower[d]) for d, r in res.items()}
    v = verdict(grades)
    return {"contrast": f"{a}_minus_{b}", "alpha": alpha, "level": level, "rule": rule,
            "datasets": res, "grades": grades, "verdict": v, "verdict_plan": plan_name(v, grades)}


# --- complete cases, failures and the no-verdict categories ---------------------------------

class Failure(NamedTuple):
    """A run left out of its rule: kind 'numerical' (non-finite loss, V-AUC or scores),
    'infrastructure' (failed twice) or 'gate' (a gate failure of that run)."""

    dataset: str
    pair: str
    seed: int
    arm: str
    kind: str

    @property
    def treatment(self) -> bool:
        """A numerical failure of the HB chains' counted arms (C, C.G, C.last, B; PLAN.md)."""
        return self.kind == NUMERICAL and self.arm in TREATMENT_ARMS


def treatment_failures(failures: Sequence[Failure]) -> list[Failure]:
    for f in failures:
        if f.kind not in FAILURE_KINDS:
            raise ValueError(f"unknown failure kind {f.kind!r} ({FAILURE_KINDS})")
    return sorted(f for f in failures if f.treatment)


def completeness(vals: Values, failures: Sequence[Failure], a: str = "C", b: str = "R0", *,
                 pairs: Sequence[str] = PAIRS, seeds: Sequence[int] = SEEDS) -> dict:
    """Complete seeds of one dataset for a - b and, per dropped seed, its missing runs and their
    reasons; a seed is dropped for the treatment only when every missing run is a recorded
    numerical failure of an HB arm (an unrecorded absence counts as a gate reason)."""
    complete, dropped = [], {}
    for s in seeds:
        missing = [(p, x) for p in pairs for x in (a, b) if (p, s, x) not in vals]
        if not missing:
            complete.append(s)
            continue
        reasons = []
        for p, x in missing:
            kinds = sorted({f.kind for f in failures if (f.pair, f.seed, f.arm) == (p, s, x)})
            reasons.append({"pair": p, "arm": x, "kinds": kinds or ["unrecorded"],
                            "treatment": bool(kinds) and all(
                                Failure("", p, s, x, k).treatment for k in kinds)})
        dropped[s] = {"missing": reasons, "treatment_only": all(r["treatment"] for r in reasons)}
    return {"seeds": complete, "n": len(complete), "dropped": dropped,
            "dropped_other": [s for s, v in dropped.items() if not v["treatment_only"]],
            "dropped_treatment": [s for s, v in dropped.items() if v["treatment_only"]]}


def no_verdict(comp: Mapping[str, Mapping], min_complete: int = MIN_COMPLETE) -> str | None:
    """PLAN.md "주 검정" (review N9): None when every dataset has >= 8 complete seeds; else
    "관문 실패(판정 없음)" when a dataset lost >= 3 seeds to reasons other than a treatment
    numerical failure, otherwise "처치 수치 불안정(사용자 결정)"."""
    if all(c["n"] >= min_complete for c in comp.values()):
        return None
    if any(len(c["dropped_other"]) >= 3 for c in comp.values()):
        return GATE_FAILED
    return TREATMENT_UNSTABLE


def impute_treatment(vals_by_d: Mapping[str, Values], failures: Sequence[Failure],
                     arm: str = "C") -> tuple[dict, list[Failure]]:
    """Each treatment-failed ``arm`` run that is absent takes the lowest value of that arm over
    the other seeds of its dataset and pair (``e020_stats.impute_minimum``)."""
    treat = [f for f in treatment_failures(failures) if f.arm == arm]
    out, used = {}, []
    for d, vals in vals_by_d.items():
        keys = [(f.pair, f.seed, arm) for f in treat if f.dataset == d
                and (f.pair, f.seed, arm) not in vals]
        out[d] = st.impute_minimum(vals, keys)
        used += [f for f in treat if f.dataset == d and (f.pair, f.seed, arm) in keys]
    return out, used


def decide(primary: Mapping[str, Values], *, failures: Sequence[Failure] = (),
           boot: Mapping[str, Mapping] | None = None, decisions: Decisions = RECORDED,
           margins: Mapping[str, float] = MARGIN) -> dict:
    """The verdict of C - R0 (logit readout, primary metric, primary checkpoint rules): the
    complete-case rule and its categories, the main test, and with treatment-related numerical
    failures the imputed main test (sensitivity; the result of "처치 수치 불안정")."""
    kw = {"alpha": decisions.alpha, "level": decisions.t_level, "margins": margins,
          "rule": decisions.substantive_rule, "boot": boot, "boot_level": decisions.boot_level}
    comp = {d: completeness(v, [f for f in failures if f.dataset == d], *MAIN)
            for d, v in primary.items()}
    main = main_test(primary, *MAIN, **kw)
    treat = treatment_failures(failures)
    out = {"completeness": comp, "main": main, "treatment_failures": [list(f) for f in treat]}
    if treat:
        imputed_vals, used = impute_treatment(primary, failures)
        out["imputed"] = {"runs": [list(f) for f in used], **main_test(imputed_vals, *MAIN, **kw)}
    category = no_verdict(comp)
    out["verdict"] = category or main["verdict"]
    out["verdict_plan"] = plan_name(out["verdict"], main["grades"] if category is None else None)
    out["grades"] = main["grades"] if category is None else None
    return out


# --- flags ----------------------------------------------------------------------------------

def _negative(res: Mapping[str, Mapping]) -> list[str]:
    return [k for k, v in res.items() if v["conclusion"] == "negative"]


def _positive(res: Mapping[str, Mapping]) -> list[str]:
    return [k for k, v in res.items() if v["conclusion"] == "positive"]


def _grade_changed(main: Mapping, other: Mapping | None) -> list[str] | None:
    if other is None:
        return None
    return [d for d in main["grades"] if main["grades"][d] != other["grades"].get(d)]


def flags(*, main: Mapping, sigma: Mapping, wilcoxon: Mapping, settings: Mapping,
          within: Mapping, z_only: Mapping, boot: Mapping[str, Mapping], last_rule: Mapping,
          extreme: Mapping, c_minus_r1: Mapping | None, r1_minus_r0: Mapping | None,
          resolution: Mapping | None, c_minus_cw: Mapping | None,
          treatment: Sequence[Failure], e020: Mapping[str, Mapping] | None,
          alpha: float, margins: Mapping[str, float], boot_level: float) -> dict:
    """The pre-registered flags (PLAN.md "표지"; keyword arguments: the inputs are alike). Lists
    of datasets (settings for the setting flag); None when a flag's inputs are absent.

    setting_dependent           C - R0 per setting significantly negative (Holm over 4)
    localisation_harm           L of C - R0 significantly negative (Holm over the datasets);
      _substantive              of those, the seed mean <= -m_d (review N7)
    localisation_improvement_claimable   L of C - R0 significantly positive
    length_offset_driven        G_d positive while neither the z-only primary metric nor L of
                                C - R0 is significantly positive
    readout_sensitive / checkpoint_rule_dependent / extreme_readout_sensitive
                                G_d differs under the sigma readout / the secondary checkpoint
                                rule (C.last - R0.last) / C.G - R0
    distribution_sensitive      Wilcoxon significance (Holm) differs from the t's
    sensitive_to_test_video_sample   G_d positive or negative while the bootstrap interval at
                                ``boot_level`` contains 0
    reextraction_driven         G_d positive and C - R1 not significantly positive, or the mean
                                R1 - R0 >= half the mean C - R0; "computable" False without R1
                                (it then counts as raised under D10 (c))
    no_resolution_share         C.W - R0 has the grade of C - R0 and C - C.W is not
                                significantly positive
    dense_numerically_unstable  >= 1 treatment-related numerical failure
    not_better_than_e020_baselines   C - L20 or C - base significantly negative (D7 (a))"""
    res, grades = main["datasets"], main["grades"]
    positive = [d for d in grades if grades[d] in POSITIVE]
    harm = _negative(within)
    out = {"setting_dependent": _negative(settings),
           "localisation_harm": harm,
           "localisation_harm_substantive": [d for d in harm
                                             if within[d]["mean"] <= -margins[d]],
           "localisation_improvement_claimable": _positive(within),
           "length_offset_driven": [d for d in positive
                                    if z_only[d]["conclusion"] != "positive"
                                    and within[d]["conclusion"] != "positive"],
           "readout_sensitive": _grade_changed(main, sigma),
           "distribution_sensitive": [d for d in res if (res[d]["p_holm"] < alpha)
                                      != (wilcoxon[d]["p_holm"] < alpha)],
           "checkpoint_rule_dependent": _grade_changed(main, last_rule),
           "extreme_readout_sensitive": _grade_changed(main, extreme),
           "sensitive_to_test_video_sample": [
               d for d in grades if (grades[d] in POSITIVE or grades[d] == NEG)
               and _contains_0(interval(boot.get(d), boot_level))]}
    if c_minus_r1 is None or r1_minus_r0 is None:
        out["reextraction_driven"] = {"computable": False, "datasets": None}
    else:
        out["reextraction_driven"] = {"computable": True, "datasets": [
            d for d in positive if c_minus_r1[d]["conclusion"] != "positive"
            or r1_minus_r0[d]["mean"] >= 0.5 * res[d]["mean"]]}
    if resolution is None or c_minus_cw is None:
        out["no_resolution_share"] = None
    else:
        out["no_resolution_share"] = [d for d in grades
                                      if resolution["grades"].get(d) == grades[d]
                                      and c_minus_cw[d]["conclusion"] != "positive"]
    out["dense_numerically_unstable"] = {"raised": bool(treatment), "n": len(treatment),
                                         "runs": [list(f) for f in treatment]}
    if e020 is None:
        out["not_better_than_e020_baselines"] = None
    else:
        out["not_better_than_e020_baselines"] = [f"{k}/{d}" for k, r in e020.items()
                                                 for d in _negative(r)]
    return out


def _contains_0(ci: Sequence[float]) -> bool:
    """A finite interval containing 0; an undefined interval counts as containing 0."""
    lo, hi = ci
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return True
    return bool(lo <= 0 <= hi)


def raised(flag) -> bool:
    """Whether a flag stands (lists non-empty; re-extraction also when not computable)."""
    if isinstance(flag, Mapping) and "computable" in flag:
        return not flag["computable"] or bool(flag["datasets"])
    if isinstance(flag, Mapping) and "raised" in flag:
        return bool(flag["raised"])
    return bool(flag)


# --- condition (2), P9 and the next step ----------------------------------------------------

BLOCKING = ("reextraction_driven", "localisation_harm_substantive", "length_offset_driven")


def condition_2(verdict_code: str, *, decisions: Decisions = RECORDED,
                boot: Mapping[str, Mapping] | None, flag_values: Mapping,
                treatment: Sequence[Failure], datasets: Sequence[str] = DATASETS) -> dict:
    """Condition (2) of the paper start (PLAN.md "논문 착수 조건의 평가", review N8): parts 1-4,
    the status ("met", "deferred_to_user" when only 4 fails, "not_met_with_reasons" when 1
    holds and 2 or 3 fails, else "not_met") and its plan text."""
    parts: dict[str, bool | None] = {
        "1_better_substantive": verdict_code == BETTER_SUB,
        "2_counted": decisions.d12 == "a" and decisions.d6_effective in ("a", "c")}
    lowers = {d: interval((boot or {}).get(d), decisions.boot_level)[0] for d in datasets}
    if decisions.d10 in ("a", "c"):
        parts["3_video_sample"] = all(math.isfinite(x) and x > 0 for x in lowers.values())
    else:
        parts["3_video_sample"] = None                   # D10 (b): not part of condition (2)
    blocking = {k: raised(flag_values.get(k)) for k in BLOCKING}
    parts["3_blocking_flags"] = (not any(blocking.values())) if decisions.d10 == "c" else None
    parts["4_no_treatment_failure"] = not treatment
    reasons = []
    if not parts["2_counted"]:
        reasons.append("principle_2_exception_not_approved" if decisions.d12 != "a"
                       else "f1_format_not_counted")
    if parts["3_video_sample"] is False:
        reasons.append("video_sample")
    if parts["3_blocking_flags"] is False:
        reasons.append("blocking_flags")
    first3 = parts["2_counted"] and parts["3_video_sample"] is not False and \
        parts["3_blocking_flags"] is not False
    if not parts["1_better_substantive"]:
        status, plan = NOT_MET, "미충족"
        reasons = []
    elif first3 and parts["4_no_treatment_failure"]:
        status, plan = MET, "충족"
    elif first3:
        status, plan = DEFERRED, "낫다(실질), 조건 (2) 사용자 결정 대기(처치 관련 수치 실패)"
        reasons = ["treatment_numerical_failure"]
    else:
        if not parts["4_no_treatment_failure"]:
            reasons.append("treatment_numerical_failure")
        status = NOT_MET_REASONS
        plan = f"낫다(실질), 조건 (2) 미충족({', '.join(REASONS[r] for r in reasons)})"
    return {"status": status, "plan": plan, "parts": parts, "reasons": reasons,
            "bootstrap_lower": lowers, "boot_level": decisions.boot_level,
            "blocking": blocking if decisions.d10 == "c" else None}


def _names(ds: Sequence[str]) -> str:
    return ", ".join(NAMES.get(d, d) for d in ds)


def next_step(verdict_code: str, cond: Mapping, main: Mapping, *,
              decisions: Decisions = RECORDED, margins: Mapping[str, float] = MARGIN,
              completeness_by_d: Mapping[str, Mapping] | None = None,
              imputed: Mapping | None = None) -> dict:
    """PLAN.md "결과에 따른 다음 단계": the P9 entry (by size for "나쁨"), E3 and the paper start;
    the user decides wherever the plan says so."""
    res, grades = main["datasets"], main["grades"]
    out: dict = {"verdict": verdict_code, "p9": None, "p9_by_user": False, "e3": None,
                 "paper_start": "hold", "adopt_c": False, "user_decides": []}
    options = "(a) E3 중단, (b) 조밀 특징은 경보 평가 규약에만, (c) 다른 방향"
    if verdict_code == GATE_FAILED:
        out["user_decides"] = ["다음 단계(관문 실패, P9 기재 없음)"]
    elif verdict_code == TREATMENT_UNSTABLE:
        short = [d for d, c in (completeness_by_d or {}).items() if c["n"] < MIN_COMPLETE]
        imp = plan_name(imputed["verdict"], imputed.get("grades")) if imputed else "없음"
        out["p9"] = P9_UNSTABLE.format(datasets=_names(short), imputed=imp)
        out["p9_by_user"] = True
        out["e3"] = "user"
        out["user_decides"] = ["P9 기재 여부", "E3 계속 여부"]
    elif verdict_code == WORSE:
        neg = [d for d in grades if grades[d] == NEG]
        big = [d for d in neg if res[d]["mean"] <= -margins[d]]
        if big:
            out["p9"], out["e3"] = P9_WORSE.format(datasets=_names(big)), "stop"
        else:
            out["p9"], out["e3"] = P9_SMALL.format(datasets=_names(neg)), "user"
            out["user_decides"] = ["E3 계속 여부"]
    elif verdict_code == BETTER_SUB and cond["status"] == MET:
        out["paper_start"] = "condition_2_met"
        out["adopt_c"] = decisions.d12 == "a" and decisions.d8 == "a"
        out["next"] = "F2 계획"
        if decisions.d8 == "b":
            out["user_decides"] = ["채택(C를 논문 파이프라인으로)"]
    elif verdict_code == BETTER_SUB and cond["status"] == DEFERRED:
        out["user_decides"] = ["조건 (2) 충족 여부(처치 관련 수치 실패)"]
    else:
        out["user_decides"] = [f"다음 단계: {options}"]
    return out


# --- decomposition --------------------------------------------------------------------------

def decomposition(vals_by_d: Mapping[str, Values], *, decisions: Decisions = RECORDED,
                  pairs: Sequence[str] = PAIRS, seeds: Sequence[int] = SEEDS) -> dict:
    """PLAN.md "분해 대비" (descriptive, not a verdict): per contrast the pair-averaged seed
    differences with paired t (Holm over the datasets at alpha) and the seed t interval
    (1 - 2 alpha), the per-setting values, and the seed-wise identities."""
    have = {"r1": decisions.r1, "a1s": decisions.a1s, None: True}
    kw = {"alpha": decisions.alpha, "level": decisions.t_level, "pairs": pairs, "seeds": seeds}
    out = {}
    for name, (w, need) in DECOMPOSITION.items():
        if not have[need]:
            continue
        sett = {}
        for d, vals in vals_by_d.items():
            for p in pairs:
                kept, diffs = seed_combination(vals, w, (p,), seeds)
                sett[f"{d}/{p}"] = _row(kept, diffs, "t", decisions.t_level, seeds)
        out[name] = {"weights": dict(w), "datasets": combination_contrast(vals_by_d, w, **kw),
                     "settings": sett}
    return {"contrasts": out, "identities": identities(vals_by_d, decisions=decisions,
                                                         pairs=pairs, seeds=seeds)}


def identities(vals_by_d: Mapping[str, Values], *, decisions: Decisions = RECORDED,
               pairs: Sequence[str] = PAIRS, seeds: Sequence[int] = SEEDS) -> dict:
    """Seed-wise checks of the decomposition identities over the seeds holding every arm:
    the largest |left - sum of the right parts| per dataset (floating-point size)."""
    have = {"r1": decisions.r1, "a1s": decisions.a1s}
    named = {**{k: w for k, (w, _) in DECOMPOSITION.items()}, "C-R0": {"C": 1, "R0": -1}}
    out = {}
    for label, (left, right, need) in IDENTITIES.items():
        if not have[need]:
            continue
        arms = set(named[left]) | {a for r in right for a in named[r]}
        rows = {}
        for d, vals in vals_by_d.items():
            kept = [s for s in seeds if all((p, s, a) in vals for p in pairs for a in arms)]
            worst = 0.0
            for s in kept:
                lv = seed_combination(vals, named[left], pairs, (s,))[1][0]
                rv = sum(seed_combination(vals, named[r], pairs, (s,))[1][0] for r in right)
                worst = max(worst, abs(lv - rv))
            rows[d] = {"seeds": len(kept), "max_abs_residual": worst}
        out[label] = rows
    return out


# --- everything of one analysis -------------------------------------------------------------

def evaluate(primary: Mapping[str, Values], *, sigma: Mapping[str, Values],
             z_only: Mapping[str, Values], within: Mapping[str, Values],
             boot: Mapping[str, Mapping], failures: Sequence[Failure] = (),
             e020: Mapping[str, Values] | None = None, decisions: Decisions = RECORDED,
             margins: Mapping[str, float] = MARGIN,
             boot_by_test: Mapping[str, Mapping[str, Mapping]] | None = None) -> dict:
    """The verdict, condition (2), the flags, the next step and the decomposition.

    primary   logit readout, primary metric (UCF frame AUC, XD frame AP), every arm (R0, C, R1,
              B, A1.S, C.W, C.G, C.last, C.o32, R0.last)
    sigma     sigma readout, primary metric (C, R0)
    z_only    z-only readout (offset removed), primary metric (C, R0)
    within    within-video localisation L, logit readout (C, R0)
    boot      per dataset ``bootstrap_contrast`` of C - R0 (logit readout, primary metric)
    e020      the E020 cells L20 and base (logit, best rule; D7 (a)), merged with C
    boot_by_test   D4 (b) only: bootstraps of the other grade comparisons ("sigma", "last",
              "extreme", "resolution")"""
    a, lv, bl = decisions.alpha, decisions.t_level, decisions.boot_level
    rule = decisions.substantive_rule
    kw = {"alpha": a, "level": lv, "margins": margins, "rule": rule, "boot_level": bl}

    def graded(name: str, vals: Mapping[str, Values], x: str, y: str) -> dict:
        b = (boot_by_test or {}).get(name)
        if rule == "bootstrap_lower" and b is None:
            raise ValueError(f"D4 (b): the {name} grades need their video bootstrap")
        return main_test(vals, x, y, boot=b, **kw)

    dec = decide(primary, failures=failures, boot=boot, decisions=decisions, margins=margins)
    main = dec["main"]
    treat = treatment_failures(failures)
    tests = {"sigma_readout": graded("sigma", sigma, *MAIN),
             "wilcoxon": contrast(primary, *MAIN, kind="wilcoxon", alpha=a, level=lv),
             "settings": setting_contrast(primary, *MAIN, alpha=a, level=lv),
             "within": contrast(within, *MAIN, alpha=a, level=lv),
             "z_only": contrast(z_only, *MAIN, alpha=a, level=lv),
             "last_rule": graded("last", primary, *SECONDARY),
             "extreme": graded("extreme", primary, *EXTREME),
             "resolution": graded("resolution", primary, *RESOLUTION),
             "c_minus_cw": contrast(primary, "C", "C.W", alpha=a, level=lv)}
    if decisions.r1:
        tests["c_minus_r1"] = contrast(primary, "C", "R1", alpha=a, level=lv)
        tests["r1_minus_r0"] = contrast(primary, "R1", "R0", alpha=a, level=lv)
    e020_tests = None
    if decisions.d7 == "a" and e020 is not None:
        merged = {d: {**e020[d], **{k: v for k, v in primary[d].items() if k[2] == "C"}}
                  for d in primary}
        e020_tests = {f"C-{x}": contrast(merged, "C", x, alpha=a, level=lv)
                      for x in ("L20", "base")}
        tests["e020_baselines"] = e020_tests
    fl = flags(main=main, sigma=tests["sigma_readout"], wilcoxon=tests["wilcoxon"],
               settings=tests["settings"], within=tests["within"], z_only=tests["z_only"],
               boot=boot, last_rule=tests["last_rule"], extreme=tests["extreme"],
               c_minus_r1=tests.get("c_minus_r1"), r1_minus_r0=tests.get("r1_minus_r0"),
               resolution=tests["resolution"], c_minus_cw=tests["c_minus_cw"], treatment=treat,
               e020=e020_tests, alpha=a, margins=margins, boot_level=bl)
    cond = condition_2(dec["verdict"], decisions=decisions, boot=boot, flag_values=fl,
                       treatment=treat, datasets=tuple(primary))
    step = next_step(dec["verdict"], cond, main, decisions=decisions, margins=margins,
                     completeness_by_d=dec["completeness"], imputed=dec.get("imputed"))
    raised_flags = [FLAG_NAMES[k] for k, v in fl.items() if k in FLAG_NAMES and raised(v)]
    # PLAN.md "결과의 쓰임": without "낫다(실질)" no secondary, per-setting or decomposition
    # result is claimed as an improvement; a localisation improvement only where L is positive
    claims = {"secondary_improvements_allowed": dec["verdict"] == BETTER_SUB,
              "localisation_improvement": list(fl["localisation_improvement_claimable"])}
    return {"decisions": {k: getattr(decisions, k) for k in _CHOICES},
            "alpha": a, "t_level": lv, "boot_level": bl, "substantive_rule": rule,
            **dec, "tests": tests, "bootstrap": dict(boot), "flags": fl,
            "flags_raised_plan": raised_flags, "condition_2": cond, "next_step": step,
            "claims": claims, "decomposition": decomposition(primary, decisions=decisions)}


# --- video bootstrap ------------------------------------------------------------------------

def window_frames(total_frames: int, units: int = grid.WINDOWS) -> np.ndarray:
    """Frames of the evaluator windows / 32-unit segments (T // S each, the last also T % S;
    ``wvad.eval.metrics.expand_segments``)."""
    t = int(total_frames)
    if t < units:
        raise ValueError(f"{t} frames cannot hold {units} units")
    counts = np.full(units, t // units, dtype=np.int64)
    counts[-1] += t % units
    return counts


def snippet_frames(total_frames: int) -> np.ndarray:
    """Frames of the snippet supports [16 j, min(16 j + 16, T)) (``grid.snippet_support``)."""
    sup = grid.snippet_support(total_frames)
    return (sup[:, 1] - sup[:, 0]).astype(np.int64)


def unit_atoms(scores: Sequence[np.ndarray], frames: Sequence[np.ndarray],
               gts: Sequence[np.ndarray], min_frames: int = DEFAULT_MIN_FRAMES) -> st.Atoms:
    """Frame-metric atoms of constant-score units: video i has units with scores ``scores[i]``
    covering ``frames[i]`` consecutive frames each (the counts sum to the video's T = len(gt);
    the number of units may differ between videos), one atom per unit with its positive and
    negative frame counts; videos under ``min_frames`` frames are skipped as by the evaluator.
    With 32 units of ``window_frames`` this is ``e020_stats.segment_atoms`` (bit for bit)."""
    if not len(scores) == len(frames) == len(gts):
        raise ValueError(f"{len(scores)} score rows, {len(frames)} frame rows, {len(gts)} videos")
    parts = []
    for i, (s, f, gt) in enumerate(zip(scores, frames, gts, strict=True)):
        t = len(gt)
        if t < min_frames:
            continue
        s, f = np.asarray(s, dtype=np.float64), np.asarray(f, dtype=np.int64)
        if s.ndim != 1 or s.shape != f.shape or not s.size:
            raise ValueError(f"video {i}: {s.shape} scores for {f.shape} units")
        if (f < 1).any() or int(f.sum()) != t:
            raise ValueError(f"video {i}: unit frames {int(f.sum())} (each >= 1) for T = {t}")
        if not np.isfinite(s).all():
            raise ValueError(f"video {i}: non-finite unit scores")
        first = np.concatenate([[0], np.cumsum(f)[:-1]])
        pos = np.add.reduceat(np.asarray(gt, dtype=np.float64), first)
        parts.append((s, pos, f - pos, np.full(len(s), i)))
    if not parts:
        raise ValueError(f"no video has at least {min_frames} frames")
    score, pos, neg, video = (np.concatenate(x) for x in zip(*parts, strict=True))
    order = np.argsort(score, kind="stable")
    score = score[order]
    starts = np.flatnonzero(np.r_[True, score[1:] != score[:-1]])
    return st.Atoms(score, pos[order], neg[order], video[order], starts)


def window_atoms(values: np.ndarray, gts: Sequence[np.ndarray],
                 min_frames: int = DEFAULT_MIN_FRAMES) -> st.Atoms:
    """Atoms of (n, S) unit values over the evaluator windows: the 32-unit arms (R0, R1, B) and
    the window-resolution C.W."""
    v = np.asarray(values)
    return unit_atoms(list(v), [window_frames(len(g), v.shape[1]) for g in gts], gts, min_frames)


def snippet_atoms(values: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                  min_frames: int = DEFAULT_MIN_FRAMES) -> st.Atoms:
    """Atoms of per-video snippet values over their supports (C, C.G, C.last, C.o32, A1.S)."""
    return unit_atoms(values, [snippet_frames(len(g)) for g in gts], gts, min_frames)


def level_key(level: float) -> str:
    """'99', '95', '99.5' (the keys of the bootstrap intervals)."""
    return f"{round(100 * level, 6):g}"


def percentile_interval(x: np.ndarray, level: float) -> list[float]:
    """Percentile interval [100 (1 - level) / 2, 100 (1 + level) / 2] (95%: 2.5, 97.5 exactly)."""
    _probability(level)
    q = [round(50 * (1 - level), 10), round(50 * (1 + level), 10)]
    lo, hi = (float(v) for v in np.percentile(x, q))
    return [lo, hi]


def bootstrap_contrast(atoms: Mapping[tuple[str, int, str], st.Atoms | Callable[[], st.Atoms]],
                       labels: Sequence[int], metric: str, seeds: Sequence[int],
                       a: str = "C", b: str = "R0", *, n_resamples: int = BOOT_R,
                       seed: int = BOOT_SEED, levels: Sequence[float] = BOOT_LEVELS,
                       point: Mapping[tuple[str, int, str], float] | None = None,
                       chunk: int = st.BOOT_CHUNK, pairs: Sequence[str] = PAIRS) -> dict:
    """Video bootstrap of the seed mean of the pair-averaged a - b (``metric`` 'frame_auc' or
    'frame_ap', points) over the complete ``seeds``: anomalous and normal videos resampled
    separately (``e020_stats.bootstrap_weights``, the same resamples for every run), each run's
    metric on every resample from its atoms (``atoms``: (pair, seed, arm) -> Atoms, or a
    function returning them, called once), percentile intervals at ``levels`` over resamples
    1..R. Resample 0 is the original sample: with ``point`` (the evaluator's values) it must
    reproduce them (largest |difference|; infinite when either is not finite). Non-finite
    resampled differences raise ValueError. At R 2000, seed 20261006 and level 0.95 this is
    ``e020_stats.bootstrap_diff``."""
    if metric not in ("frame_auc", "frame_ap"):
        raise ValueError(f"unknown metric {metric!r}")
    keys = {level_key(x): x for x in levels}
    out = {"R": int(n_resamples), "seed": int(seed), "metric": metric, "seeds": list(seeds),
           "contrast": f"{a}_minus_{b}", "levels": [float(x) for x in levels]}
    if not seeds:
        nan2 = [math.nan, math.nan]
        return out | {"resample0": math.nan, "mean": math.nan,
                      "intervals": dict.fromkeys(keys, nan2),
                      "contains_0": dict.fromkeys(keys, False),
                      "point_check_max_abs_diff": math.nan}
    w = st.bootstrap_weights(labels, n_resamples, seed)
    pick = 0 if metric == "frame_auc" else 1
    per, check = {}, 0.0
    for p in pairs:
        for s in seeds:
            for c in (a, b):
                k = (p, s, c)
                at = atoms[k]() if callable(atoms[k]) else atoms[k]
                per[k] = st.weighted_auc_ap(at, w, chunk)[pick]
                if point is not None:
                    d = abs(float(per[k][0]) - point[k])
                    check = max(check, d) if math.isfinite(d) else math.inf
    diff = np.mean([np.mean([per[(p, s, a)] - per[(p, s, b)] for p in pairs], axis=0)
                    for s in seeds], axis=0)
    if not np.isfinite(diff).all():
        raise ValueError(f"{metric}: {int(np.sum(~np.isfinite(diff)))} non-finite bootstrap "
                         "differences")
    intervals = {k: percentile_interval(diff[1:], x) for k, x in keys.items()}
    return out | {"R": int(len(w) - 1), "resample0": float(diff[0]),
                  "mean": float(diff[1:].mean()), "intervals": intervals,
                  "contains_0": {k: bool(v[0] <= 0 <= v[1]) for k, v in intervals.items()},
                  "point_check_max_abs_diff": check}
