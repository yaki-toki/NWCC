"""E020 (stage F1) seal of the test set (PLAN.md, test-scoring guards), the parts that need no
run layout; used by wvad.diagnostics.e020_score (server) and wvad.diagnostics.e020 (local):

    plan seal       PLAN.sha256 = sha256 of PLAN.md before its change log (freeze once)
    committed code  a known commit, no uncommitted tracked change
    gate files      copies of results/e020_gate.json that unlock base (gate (s) passed on this
                    server's validation scores) and the later cells (gate (v) passed on base
                    scores of this commit and plan)
    validation run  the test scoring runs in the environment and on the commit of the server's
                    validation scoring (gate (s))
    access log      one JSON line per test-scoring call (<runs>/test_access.log, append only);
                    line k names <out>/test/meta_<k>.json, and every call uses the same log
    path (ii)       P000's thread set-up: OMP_NUM_THREADS = MKL_NUM_THREADS = 6 exported before
                    Python starts and torch.get_num_threads() 6 (checked, never set)
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from wvad.diagnostics.common import code_version
from wvad.diagnostics.e015 import ENV_KEYS
from wvad.paths import REPO_ROOT
from wvad.prep.manifests import sha256_file

E020_DIR = REPO_ROOT / "experiments" / "E020_confirmation_f1"
CHANGELOG = "## 변경 기록"                 # PLAN.sha256 covers PLAN.md before this line
# gate statuses of `e020 gates --check v` that must all hold before the later cells are scored
UNLOCK = ("a", "b", "c", "d", "e", "f", "t", "s", "test_scores", "v")
# gate statuses of `e020 gates --scores` that must all hold before base is scored (order 3)
GATE_S = ("a", "b", "c", "d", "e", "f", "t", "s")
PATH_II_THREADS = 6
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")         # P000: both 6 in the environment


# --- plan seal ------------------------------------------------------------------------------

def plan_frozen_part(data: bytes) -> bytes:
    """PLAN.md up to its first line that starts with '## 변경 기록' (that line excluded), so
    entries appended to the change log keep the frozen hash."""
    marker = CHANGELOG.encode("utf-8")
    if data.startswith(marker):
        return b""
    pos = data.find(b"\n" + marker)
    if pos < 0:
        raise ValueError(f"no line starting with {CHANGELOG!r}")
    return data[:pos + 1]


def plan_sha256(plan: Path) -> str:
    return hashlib.sha256(plan_frozen_part(Path(plan).read_bytes())).hexdigest()


def write_plan_seal(plan_dir: Path | None = None) -> str:
    """Freeze: write PLAN.sha256 as '<sha256>  PLAN.md (before ## 변경 기록)'; once only."""
    d = Path(plan_dir or E020_DIR)
    if (d / "PLAN.sha256").exists():
        raise SystemExit(f"{d / 'PLAN.sha256'} exists: the plan is frozen once")
    h = plan_sha256(d / "PLAN.md")
    (d / "PLAN.sha256").write_text(f"{h}  PLAN.md (before {CHANGELOG})\n", encoding="utf-8",
                                   newline="\n")
    return h


def plan_seal(plan_dir: Path | None = None) -> dict:
    """PLAN.md before its change log must still have the sha256 of PLAN.sha256 (first token)."""
    d = Path(plan_dir or E020_DIR)
    try:
        want = (d / "PLAN.sha256").read_text(encoding="utf-8").split()[0]
        got = plan_sha256(d / "PLAN.md")
    except (OSError, ValueError, IndexError) as exc:
        raise SystemExit(f"plan seal unreadable in {d} ({exc!r}): freeze the plan first") from exc
    if got != want:
        raise SystemExit(f"PLAN.md before {CHANGELOG!r} has sha256 {got}, PLAN.sha256 {want}: "
                         "the plan changed after the freeze")
    return {"file": "PLAN.md", "sha256": got}


# --- code, gate files and the validation scoring --------------------------------------------

def code_guard() -> dict:
    """Committed code: known commit, no uncommitted tracked change (a deployed copy reports
    "dirty" as unknown)."""
    code = code_version()
    if code["dirty"] is True or not code["commit"]:
        raise SystemExit(f"test scoring needs committed code (clean tracked tree, known commit); "
                         f"code version {code}")
    return code


def gate_v_unlock(path: Path | None, commit: str, plan: str) -> dict:
    """The later cells are scored only after gate (v) passed (PLAN.md order 4-5): the local
    gate file (results/e020_gate.json of `e020 gates --check v`, copied here) must show the
    gates before test scoring, the base score files and (v) passed, (v) on base scores made
    with this deployed commit and plan hash."""
    if path is None:
        raise SystemExit("--cells l20 m10b m10 needs --gate_v <copy of results/e020_gate.json>")
    try:
        gate = json.loads(Path(path).read_text(encoding="utf-8"))
        failed = [g for g in UNLOCK if gate["status"].get(g) is not True]
        if gate["errors"] or failed or gate["v"]["ok"] is not True:
            raise SystemExit(f"{path}: gates {failed} not passed or runs unreadable "
                             f"{gate['errors']}: the later cells are not scored (PLAN.md)")
        scored = gate["v"]["scored_with"]
        same = scored["commit"] == commit and scored["plan_sha256"] == plan
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"{path}: no gate (v) record ({exc!r})") from exc
    if not same:
        raise SystemExit(f"{path}: gate (v) was checked on scores of {scored}, not of commit "
                         f"{commit} and plan {plan}")
    return {"file": str(path), "sha256": sha256_file(path), "scored_with": scored,
            "gate_code": gate.get("code")}


def gate_s_unlock(path: Path | None, out: Path) -> dict:
    """Base (and p000_base) is scored only after (a)-(f), (t) and (s) passed (PLAN.md order 3):
    the local gate file (results/e020_gate.json of `e020 gates --scores`, copied here) shows
    them passed with no unreadable run, was written from a clean working tree and checked (s)
    on this server's validation scores (sha256 of <out>/val/scores.sha256)."""
    if path is None:
        raise SystemExit("--cells base / p000_base need --gate_s <copy of results/e020_gate.json>")
    listed = Path(out) / "val" / "scores.sha256"
    try:
        gate = json.loads(Path(path).read_text(encoding="utf-8"))
        failed = [g for g in GATE_S if gate["status"].get(g) is not True]
        bad = failed or gate["errors"] or gate["code"]["dirty"] is not False
        recorded, here = gate["s"]["val_list_sha256"], sha256_file(listed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"{path}: no gate (s) record, or {listed} unreadable ({exc!r})") from exc
    if bad:
        raise SystemExit(f"{path}: gates {failed} not passed, runs unreadable {gate['errors']} "
                         f"or written from a dirty tree ({gate['code']}): base is not scored")
    if recorded != here:
        raise SystemExit(f"{path}: gate (s) checked validation scores listed with sha256 "
                         f"{recorded}, {listed} has {here}")
    return {"file": str(path), "sha256": sha256_file(path), "val_list_sha256": here,
            "gate_code": gate.get("code")}


def same_scoring_env(out: Path, env: Mapping, commit: str) -> dict:
    """Test scoring runs where and on what gate (s) re-scored the validation videos:
    <out>/val/meta.json must record the same torch, CUDA, GPU and driver, and this commit
    (not from a dirty tree)."""
    path = Path(out) / "val" / "meta.json"
    try:
        val = json.loads(path.read_text(encoding="utf-8"))
        val_env, val_code = val["environment"], val.get("code") or {}
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"{path} unreadable ({exc!r}): score --split val first (gate (s))") \
            from exc
    if any(val_env.get(k) != env.get(k) for k in ENV_KEYS):
        raise SystemExit(f"environment {env} differs from the validation scoring's {val_env}")
    if val_code.get("commit") != commit or val_code.get("dirty") is True:
        raise SystemExit(f"the validation scoring ran on code {val_code}, this call on commit "
                         f"{commit}: score --split val again with this commit (--rescore or a "
                         "new --out) and pass gate (s) first")
    return {k: val_env.get(k) for k in ENV_KEYS}


def outside_repo(*paths: Path) -> None:
    """Score files and the access log stay outside the (deployed) repository (PLAN.md)."""
    for p in paths:
        if Path(p).expanduser().resolve().is_relative_to(REPO_ROOT.resolve()):
            raise SystemExit(f"{p} is inside the repository {REPO_ROOT}: use the runs folder")


# --- access log and path (ii) threads -------------------------------------------------------

def read_log(path: Path) -> list[dict]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append_log(path: Path, entry: Mapping) -> int:
    """Append one JSON line; returns its line number."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return len(read_log(path))


def full_reads(log: Sequence[Mapping], cell: str) -> int:
    """Access-log calls that read a cell's test features in full (not --missing_only)."""
    return sum(cell in (e.get("cells") or []) and not e.get("missing_only") for e in log)


def check_log_index(target: Path, log_path: Path) -> str:
    """meta_<k>.json names the call's access-log line k: refused when <out>/test already has
    meta_<k>.json or a meta of a later line, or an earlier call used another access log
    (realpath, returned)."""
    real = os.path.realpath(log_path)
    log = read_log(log_path)
    k = len(log) + 1
    metas = [json.loads(f.read_text(encoding="utf-8")) for f in Path(target).glob("meta_*.json")]
    if (Path(target) / f"meta_{k}.json").exists() or any(
            int(m.get("access_log_line") or 0) >= k for m in metas):
        raise SystemExit(f"{target} has a meta of access-log line {k} or later, {log_path} has "
                         f"{k - 1} lines: another log or score root was used")
    other = {x.get("access_log") for x in (*log, *metas)} - {real}
    if other:
        raise SystemExit(f"earlier calls used the access log {sorted(map(str, other))}, not "
                         f"{real}")
    return real


def path_ii_threads(needed: bool) -> dict:
    """Path (ii) runs under P000's thread set-up: OMP_NUM_THREADS and MKL_NUM_THREADS 6 in the
    environment and torch's thread count 6 (checked, not set; ``needed``: the call has path
    (ii) jobs). torch's parallel info and MKL version are recorded."""
    info = torch.__config__.parallel_info()
    out = {"threads_env": {k: os.environ.get(k) for k in THREAD_VARS},
           "torch_num_threads": torch.get_num_threads(), "torch_parallel_info": info,
           "mkl": next((ln.strip() for ln in info.splitlines() if "Math Kernel Library" in ln),
                       None)}
    want = str(PATH_II_THREADS)
    if needed and (any(v != want for v in out["threads_env"].values())
                   or out["torch_num_threads"] != PATH_II_THREADS):
        raise SystemExit(f"path (ii) needs {' = '.join(THREAD_VARS)} = {want} exported before "
                         f"Python starts and {want} torch threads (P000): "
                         f"{out['threads_env']}, {out['torch_num_threads']} threads")
    return out
