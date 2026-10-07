"""E021 verdict rules (PLAN.md "문턱", "판정 (사전 등록)", "해석 표지", "추출 관문"): pure
functions of the cross-fitted seed means, plus the reading of the lineage thresholds (the E010
and E018 summaries) and of the extraction gate record. Local analysis code; no annotation file
and neither the glance nor the UCA module is imported (wvad.diagnostics.e021 stage1 / verdict).

Notation (PLAN.md): a setting s = (dataset, pair), four per pipeline P (l20: E013 lineage, delta
from E010; m10: E017 lineage, delta from E018). Delta_P(X - Y; s) = development-seed mean of arm
X's cross-fitted metric - arm Y's; z = Delta / delta_{P,d} (V-AUC), z^LS = Delta^LS / delta^LS
(LS-AUC). A setting with a numerical failure has z = None: it counts neither as z <= -1 nor as
z > -1, and its dataset is not non-inferior.

    bad_P(X - Y)    >= 3 of the 4 settings with z <= -1 (bad^LS: z^LS)
    NI_P(X - Y)     in each dataset: mean z of the two pairs > -1 and every z > -2

Stage 1 (A0): NO-GO when, for both readouts (primary and logit), some pipeline has
bad(A0 - R1v) and bad^LS(A0 - R1v); otherwise "continue" with sub-flags. Final (C): the first of
NO-GO (some P: bad and bad^LS of C - R1), hold (re-extraction loss) (some P: bad(C - R0), not
bad(C - R1)), GO (gates passed, no numerical failure, both P: NI and NI^LS of C - R1 and C - R0),
hold; recomputed with the logit readout (hold and hold (re-extraction loss) count as one coarse
"hold"): a different coarse category makes the final category hold (readout sensitive). The
extreme-readout flag recomputes the table with the verdict arm read as G (A0.G, C.G).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from wvad.diagnostics.e010 import DELTA_FACTOR, SD_DF
from wvad.diagnostics.e021_score import DATASETS, PAIRS, PIPELINES
from wvad.paths import REPO_ROOT

Setting = tuple[str, str]
SETTINGS: tuple[Setting, ...] = tuple((d, p) for d in DATASETS for p in PAIRS)
PIPELINE_NAMES = tuple(PIPELINES)
PRIMARY, LOGIT = "primary", "logit"
READOUTS = (PRIMARY, LOGIT)
VERDICT_METRICS = ("v_auc", "ls_auc")
LINEAGE = {"l20": REPO_ROOT / "experiments" / "E010_dev_metric_calibration" / "results"
                  / "e010_summary.json",
           "m10": REPO_ROOT / "experiments" / "E018_delta_recalibration" / "results"
                  / "e018_summary.json"}
SD_RATIO_LIMIT = 1.5
STAGE_LIMIT_HOURS = 24.0

# categories: English codes in the outputs, the plan's names beside them
CONTINUE, STAGE1_NOGO = "continue", "stage1_NO-GO"
GO, HOLD, HOLD_LOSS, HOLD_READOUT, NOGO = ("GO", "hold", "hold_reextraction_loss",
                                           "hold_readout_sensitive", "NO-GO")
NOT_RUN, GATE_FAILED = "hold_stage2_not_run", "gate_failure_no_verdict"
FINAL_CATEGORIES = (GO, HOLD, HOLD_LOSS, HOLD_READOUT, NOGO, STAGE1_NOGO, NOT_RUN, GATE_FAILED)
NOT_RUN_REASONS = {"time": "시간", "videos": "영상 불일치", "defect": "결함"}
PLAN_NAMES = {GO: "GO", HOLD: "보류", HOLD_LOSS: "보류(재추출 손해)",
              HOLD_READOUT: "보류(판독 민감)", NOGO: "NO-GO", STAGE1_NOGO: "1단계 NO-GO",
              NOT_RUN: "보류(2단계 미실행: {reason})", GATE_FAILED: "관문 실패(판정 없음)",
              CONTINUE: "계속"}
COARSE = {GO: GO, HOLD: HOLD, HOLD_LOSS: HOLD, NOGO: NOGO}

Z = Mapping[tuple[str, str, str, str], Mapping[Setting, float | None]]   # (readout, P, contrast,
                                                                         # metric) -> z per setting


def plan_name(category: str, reason: str | None = None) -> str:
    if category == NOT_RUN:
        if reason not in NOT_RUN_REASONS:
            raise ValueError(f"stage 2 not run needs a reason in {sorted(NOT_RUN_REASONS)}")
        return PLAN_NAMES[NOT_RUN].format(reason=NOT_RUN_REASONS[reason])
    return PLAN_NAMES[category]


# --- thresholds -----------------------------------------------------------------------------

def lineage_thresholds(paths: Mapping[str, Path] | None = None) -> dict:
    """delta_{P,d}, SD and its df, R and the splits sha256 per pipeline and dataset, read from the
    E010 (l20) and E018 (m10) summaries; both lineages use the same R and splits per dataset."""
    out = {}
    for p, f in (paths or LINEAGE).items():
        s = json.loads(Path(f).read_text(encoding="utf-8"))
        out[p] = {d: {"delta": dict(v["delta"]), "sd": dict(v["sd"]), "R": int(v["n_splits"]),
                      "splits_sha256": v["splits_sha256"], "sd_df": int(v.get("df", SD_DF)),
                      "source": Path(f).name} for d, v in s["datasets"].items()}
    for d in DATASETS:
        got = {(v[d]["R"], v[d]["splits_sha256"]) for v in out.values()}
        if len(got) != 1:
            raise ValueError(f"{d}: the lineages use different splits {got}")
    return out


def pooled_sd(groups: Sequence[Sequence[float]]) -> tuple[float, int]:
    """df-weighted pooled SD (the E018 pooling): sum of squared deviations from each group's mean
    over sum (n_g - 1); groups with fewer than two values are dropped; (nan, 0) without any."""
    gs = [np.asarray(g, dtype=np.float64) for g in groups if len(g) >= 2]
    df = sum(len(g) - 1 for g in gs)
    if df == 0:
        return math.nan, 0
    return math.sqrt(sum(float(((g - g.mean()) ** 2).sum()) for g in gs) / df), df


def ls_threshold(r0_values: Mapping[str, Sequence[float]]) -> dict:
    """delta^LS = 2 sqrt(2/3) SD^LS, SD^LS pooled over the pairs from R0's LS-AUC of its seeds
    (0-2 and 100-102; a numerically failed run is left out: df-weighted). None if undefined."""
    sd, df = pooled_sd([[x for x in v if x is not None and math.isfinite(x)]
                        for v in r0_values.values()])
    ok = df > 0 and math.isfinite(sd) and sd > 0
    return {"sd": sd, "df": df, "delta": DELTA_FACTOR * sd if ok else None}


def z_of(delta: float | None, threshold: float | None) -> float | None:
    """Delta / threshold; None for a failed setting or an undefined threshold."""
    if delta is None or threshold is None or not math.isfinite(delta):
        return None
    if not (math.isfinite(threshold) and threshold > 0):
        return None
    return delta / threshold


# --- the rules ------------------------------------------------------------------------------

def bad_type(z: Mapping[Setting, float | None]) -> bool:
    """At least 3 of the 4 settings with z <= -1 (a failed setting counts neither way)."""
    return sum(v is not None and v <= -1 for v in z.values()) >= 3


def non_inferior(z: Mapping[Setting, float | None]) -> bool:
    """In each dataset: both pairs' z known, their mean > -1 and each > -2 (the per-setting
    floor, R2); a dataset with a failed setting is not non-inferior."""
    for d in DATASETS:
        vals = [z.get((d, p)) for p in PAIRS]
        if any(v is None for v in vals):
            return False
        if not (vals[0] + vals[1]) / 2 > -1 or not all(v > -2 for v in vals):
            return False
    return True


def four_type(z: Mapping[Setting, float | None]) -> str:
    """Stage-D four types on z (threshold delta_{P,d}; descriptive): adopt (>= 3 settings with
    z >= 1 and none <= -1), worse (>= 3 with z <= -1), not detected (all four |z| < 1), hold."""
    live = [v for v in z.values() if v is not None]
    hi, lo = sum(v >= 1 for v in live), sum(v <= -1 for v in live)
    if hi >= 3 and lo == 0:
        return "adopt"
    if lo >= 3:
        return "worse"
    if len(live) == len(z) and all(abs(v) < 1 for v in live):
        return "not_detected"
    return "hold"


def _bad_both(z: Z, readout: str, pipeline: str, contrast: str) -> bool:
    return (bad_type(z[(readout, pipeline, contrast, "v_auc")])
            and bad_type(z[(readout, pipeline, contrast, "ls_auc")]))


def stage1_category(z: Z, arm: str = "A0",
                    pipelines: Sequence[str] = PIPELINE_NAMES) -> tuple[str, dict]:
    """1단계 NO-GO when the NO-GO condition (some P: bad and bad^LS of arm - R1v) holds in both
    readouts; otherwise "continue". Returns the category and the condition per readout."""
    main = f"{arm}-R1v"
    cond = {r: {p: _bad_both(z, r, p, main) for p in pipelines} for r in READOUTS}
    holds = {r: any(v.values()) for r, v in cond.items()}
    return (STAGE1_NOGO if all(holds.values()) else CONTINUE), {"by_pipeline": cond,
                                                                "holds": holds}


def stage1_verdict(z: Z, pipelines: Sequence[str] = PIPELINE_NAMES) -> dict:
    """Stage-1 category and its sub-flags (PLAN.md 1단계 판정): harmless (both P: NI and NI^LS of
    A0 - R1v and A0 - R0, primary readout), re-extraction loss (some P: bad(A0 - R0) and not
    bad(A0 - R1v)), readout sensitive (the NO-GO condition in one readout only), extreme readout
    sensitive (A0 read as G gives another stage-1 category), other (none of them)."""
    category, cond = stage1_category(z, "A0", pipelines)
    g_category, g_cond = stage1_category(z, "A0.G", pipelines)
    flags = {
        "harmless": all(non_inferior(z[(PRIMARY, p, c, m)]) for p in pipelines
                        for c in ("A0-R1v", "A0-R0") for m in VERDICT_METRICS),
        "reextraction_loss": any(bad_type(z[(PRIMARY, p, "A0-R0", "v_auc")])
                                 and not bad_type(z[(PRIMARY, p, "A0-R1v", "v_auc")])
                                 for p in pipelines),
        "readout_sensitive": cond["holds"][PRIMARY] != cond["holds"][LOGIT],
        "extreme_readout_sensitive": g_category != category,
    }
    flags["other"] = not any(flags.values())
    return {"category": category, "category_plan": plan_name(category),
            "nogo_condition": cond, "flags": flags,
            "g_readout": {"category": g_category, "nogo_condition": g_cond}}


def final_table(z: Z, readout: str, arm: str = "C", pipelines: Sequence[str] = PIPELINE_NAMES,
                gates_ok: bool = True, numerical_ok: bool = True) -> dict:
    """The final table under one readout (first matching row): NO-GO, hold (re-extraction
    loss), GO, hold."""
    c1, c0 = f"{arm}-R1", f"{arm}-R0"
    nogo = {p: _bad_both(z, readout, p, c1) for p in pipelines}
    loss = {p: bad_type(z[(readout, p, c0, "v_auc")]) and not bad_type(z[(readout, p, c1, "v_auc")])
            for p in pipelines}
    ni = {p: {c: {m: non_inferior(z[(readout, p, c, m)]) for m in VERDICT_METRICS}
              for c in (c1, c0)} for p in pipelines}
    if any(nogo.values()):
        category = NOGO
    elif any(loss.values()):
        category = HOLD_LOSS
    elif gates_ok and numerical_ok and all(v for p in ni.values() for c in p.values()
                                            for v in c.values()):
        category = GO
    else:
        category = HOLD
    return {"category": category, "nogo": nogo, "reextraction_loss": loss, "non_inferior": ni}


def final_verdict(z: Z, gates_ok: bool, numerical_ok: bool,
                  pipelines: Sequence[str] = PIPELINE_NAMES) -> dict:
    """Final category from the primary and logit tables (readout sensitivity), the extreme-
    readout flag (C read as G, primary readout, coarse categories) and the per-pipeline tables
    (pipeline dependence flag)."""
    tables = {r: final_table(z, r, "C", pipelines, gates_ok, numerical_ok) for r in READOUTS}
    p, lg = tables[PRIMARY]["category"], tables[LOGIT]["category"]
    category = HOLD_READOUT if COARSE[p] != COARSE[lg] else p
    g = final_table(z, PRIMARY, "C.G", pipelines, gates_ok, numerical_ok)
    per_pipeline = {q: final_table(z, PRIMARY, "C", (q,), gates_ok, numerical_ok)["category"]
                    for q in pipelines}
    return {"category": category, "category_plan": plan_name(category),
            "primary": p, "logit": lg,
            "logit_name": lg if category == p and lg != p else None,
            "tables": tables, "g_readout": g,
            "extreme_readout_sensitive": COARSE[g["category"]] != COARSE[p],
            "per_pipeline": per_pipeline,
            "pipeline_dependent": len(set(per_pipeline.values())) > 1}


# --- stage-2 start conditions ---------------------------------------------------------------

def start_conditions(stage1: str | None, gates_ok: bool, projected_hours: float | None = None,
                     time_approved: bool | None = None, sha_recheck_ok: bool | None = None,
                     code_same: bool | None = None) -> dict:
    """PLAN.md stage-2 start conditions: (1) stage-1 verdict "continue", (2) every stage-1 gate
    passed, (3) projected stage-2 extraction <= 24 h, or the user approved more (False: the user
    stopped), (4) the training-part sha256 recheck equal to (x6), (5) extraction, training and
    scoring code unchanged since stage 1. None = not known yet. The category if stage 2 is not
    run: the first failing condition in the order (1), (2), (3), (4), (5)."""
    if projected_hours is not None and projected_hours <= STAGE_LIMIT_HOURS:
        c3 = True
    else:
        c3 = time_approved
    cond = {"1": stage1 == CONTINUE, "2": bool(gates_ok), "3": c3, "4": sha_recheck_ok,
            "5": code_same}
    failed = [k for k, v in cond.items() if v is False]
    category = reason = None
    if stage1 == STAGE1_NOGO:
        category = STAGE1_NOGO
    elif not cond["1"] or "2" in failed:
        category = GATE_FAILED                  # no stage-1 verdict, or a stage-1 gate failed
    elif failed:
        reason = {"3": "time", "4": "videos", "5": "defect"}[failed[0]]
        category = NOT_RUN
    return {"conditions": cond, "projected_stage2_hours": projected_hours,
            "limit_hours": STAGE_LIMIT_HOURS, "failed": failed,
            "pending": [k for k, v in cond.items() if v is None],
            "ok": all(v is True for v in cond.values()),
            "category_if_stopped": category, "reason": reason,
            "category_if_stopped_plan": plan_name(category, reason) if category else None}


# --- extraction gates and user approvals ------------------------------------------------------

PASS_BIT, PASS_TOL, PASS = "pass_bit", "pass_tolerance", "pass"
APPROX, APPROVED = "approximate_equivalence_user_approved", "pass_user_approved_exception"
FAILED, MISSING = "fail", "missing"
PASSING = (PASS_BIT, PASS_TOL, PASS, APPROX, APPROVED)
STATUS_PLAN = {PASS_BIT: "통과(비트)", PASS_TOL: "통과(허용 오차)", PASS: "통과(허용 오차)",
               APPROX: "근사 동등(사용자 승인)", APPROVED: "통과(사용자 승인 예외: 영상 목록)",
               FAILED: "실패", MISSING: "기록 없음"}
ENCODERS = ("xcb", "xcl", "siglip")


def _approved_x1(approvals: Mapping, dataset: str, encoder: str) -> bool:
    return any(a.get("dataset") == dataset and a.get("encoder") == encoder
               for a in approvals.get("x1", []))


def _approved_videos(approvals: Mapping, gate: str, dataset: str | None = None,
                     phase: str | None = None) -> set[str]:
    out: set[str] = set()
    for a in approvals.get(gate, []):
        if (dataset is None or a.get("dataset") == dataset) and \
                (phase is None or a.get("phase") == phase):
            out |= {str(v) for v in a.get("videos", [])}
    return out


def _status(raw: str | None) -> str:
    return {PASS_BIT: PASS_BIT, PASS_TOL: PASS_TOL, PASS: PASS, FAILED: FAILED}.get(raw, FAILED) \
        if raw is not None else MISSING


def extraction_gates(record: Mapping, approvals: Mapping, stage: int,
                     names: Mapping[str, Sequence[str]]) -> dict:
    """Status of every extraction gate item the stage needs (results/e021_extract_gate.json,
    written with ``python -m wvad.extract ... --record``), with the user approvals of
    results/e021_approvals.json: (x1) fail -> approximate equivalence (user approved); (x3a) and
    (x3) fail -> pass (user approved exception: video list) when every listed video is approved.
    ``names``: manifest names per dataset (the (x3) index of a short-grab video). Returns the
    items, whether all pass, and the verdict tags (approximate equivalence per (dataset,
    encoder), decoding exceptions per dataset with their video count)."""
    items: dict[str, str] = {}
    approx: list[tuple[str, str]] = []
    exceptions: dict[str, set[str]] = {d: set() for d in DATASETS}
    norm = record.get("x0_normalization")
    if norm is None:
        items["x0"] = MISSING
    else:
        counts = norm.get("x0_normalization", norm)
        items["x0"] = PASS_BIT if counts.get("normalize_lookup_table") == 0 else FAILED
    for d in DATASETS:
        for kind in ("x1", "x2"):
            encs = (record.get(f"{kind}_{d}") or {}).get("encoders", {})
            for e in ENCODERS:
                st = _status((encs.get(e) or {}).get("summary", {}).get("status"))
                if kind == "x1" and st == FAILED and _approved_x1(approvals, d, e):
                    st = APPROX
                    approx.append((d, e))
                items[f"{kind}_{d}_{e}"] = st
    x3a = record.get("x3a")
    st = _status(None if x3a is None else x3a.get("status"))
    if st == FAILED and x3a is not None:
        bad = {(str(v.get("dataset")), str(v.get("name"))) for v in x3a.get("violations", [])}
        ok = {(d, n) for d in DATASETS for n in _approved_videos(approvals, "x3a", d)}
        if bad and bad <= ok:
            st = APPROVED
            for d, n in bad:
                exceptions[d].add(n)
    items["x3a"] = st
    x6 = record.get("x6") or {}
    items["x6"] = _status(x6.get("status") if x6 else None)
    if stage == 2:
        rc = x6.get("recheck")
        items["x6_recheck"] = MISSING if rc is None else (PASS if not rc.get("changed") else FAILED)
    phases = ("val",) if stage == 1 else ("val", "trainpart")
    for d in DATASETS:
        for ph in phases:
            for kind in ("x3", "x4"):
                entry = record.get(f"{kind}_{d}_{ph}") or {}
                for e in ENCODERS:
                    r = entry.get(e)
                    st = _status(None if r is None else r.get("status"))
                    if kind == "x3" and st == FAILED and r is not None:
                        bad = {str(v.get("name")) for v in r.get("over_1pct", [])}
                        bad |= {str(names[d][int(i)]) for i in r.get("grab_attempts_not_T", [])}
                        if bad and bad <= _approved_videos(approvals, "x3", d, ph):
                            st = APPROVED
                            exceptions[d] |= bad
                    items[f"{kind}_{d}_{ph}_{e}"] = st
        for k in ((1,) if stage == 1 else (1, 2)):
            entry = record.get(f"x5_{d}_stage{k}")
            if entry is None:
                items[f"x5_{d}_stage{k}"] = MISSING
                continue
            for e in ENCODERS:
                for v, r in (entry.get(e) or {"": None}).items():
                    items[f"x5_{d}_stage{k}_{e}{'_' + v if v else ''}"] = _status(
                        None if r is None else r.get("status"))
    tags = [{"kind": "approximate_equivalence", "dataset": d, "encoder": e,
             "plan": f"근사 동등: {d}, {e}"} for d, e in approx]
    tags += [{"kind": "decoding_exception", "dataset": d, "videos": len(v),
              "plan": f"해독 예외: {d}, {len(v)}"} for d, v in exceptions.items() if v]
    return {"items": items, "plan_status": {k: STATUS_PLAN[v] for k, v in items.items()},
            "ok": all(v in PASSING for v in items.values()),
            "failed": sorted(k for k, v in items.items() if v not in PASSING),
            "tags": tags}


def label(category: str, reason: str | None, tags: Sequence[Mapping]) -> str:
    """The verdict as written in the plan, with its approximate-equivalence and decoding-
    exception tags, e.g. 'GO (근사 동등: ucf, siglip) (해독 예외: xdv, 2)'."""
    return " ".join([plan_name(category, reason), *(f"({t['plan']})" for t in tags)])


# --- interpretation flags (descriptive, pre-registered) --------------------------------------

def any_abs_ge1(z: Mapping[Setting, float | None]) -> bool:
    return any(v is not None and abs(v) >= 1 for v in z.values())


def one_dataset_only(z: Mapping[Setting, float | None]) -> bool:
    """|z| >= 1 settings exist in exactly one dataset."""
    return len({k[0] for k, v in z.items() if v is not None and abs(v) >= 1}) == 1


def sign_mismatches(delta_v: Mapping[Setting, float | None],
                    delta_ls: Mapping[Setting, float | None]) -> list[Setting]:
    """Settings whose V-AUC Delta and LS-AUC Delta have different signs (np.sign; 0 is its own
    sign)."""
    return [k for k in delta_v if delta_v[k] is not None and delta_ls.get(k) is not None
            and np.sign(delta_v[k]) != np.sign(delta_ls[k])]


def sd_ratio_flag(ratio: float | None) -> bool:
    """delta validity (E013 rule as a ratio): arm SD / lineage SD > 1.5 or < 1/1.5."""
    return ratio is not None and math.isfinite(ratio) and (
        ratio > SD_RATIO_LIMIT or ratio < 1 / SD_RATIO_LIMIT)
