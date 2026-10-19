"""E023 test access device, access (0) and the blocked-checkpoint guard (PLAN.md "실행 순서와
시험셋 봉인" (3c), (3d), "시험 접근 장치" item 7, gates (t), (k), (h)). New code (add-only rule);
the E022 guard pieces (wvad.data.dense_test) are reused unchanged.

    python -m wvad.diagnostics.e023_access check --test_log LOG --e022_scores S [--out JSON]
                                                 [--paths FILE]
        test access (0), before the freeze, after the D1 approval: refused (a logged "refused"
        line, nothing read) unless the E023 PLAN.md records D1 as (a) 승인; then a "start" line
        before any test file is touched, and an "end" line (or "error") with the record:
          - both test manifests through the restricted reader (dense_test.read_manifest: names,
            categories, total_frames; the intervals are skipped unread) with their counts
            against E020's record, and per dataset the maximum total_frames and snippet count
            and the videos cut into pieces for each candidate N_chunk (from total_frames only;
            PLAN.md (3j), 조각 규칙)
          - the manifests' bytes against E020's git blobs, their recorded source / annotation
            sha256 and the annotation files' sha256 against E020's record
            (dense_test.manifest_hashes; files hashed, never parsed)
          - the dense test stores and derived test caches (48 files) against
            results/e022_extract_gate.json test_cache_sha256_{ucf,xdv} (dense_test.store_hashes;
            bytes only)
          - the 80 E022 C / C.last score files (<S>/test/<chain>/{C,C.last}.npz) against
            results/e022_test_scores.sha256 (bytes only, never deserialised)
          - results/e022_selection.json against the selection sha256 of E022's access-device
            records (and committed unchanged, in a git checkout)
        ``ok`` is the conjunction; the command exits 1 when it is false (after the end line).
    python -m wvad.diagnostics.e023_access log PART [PART ...]
        results/e023_test_access.log: the access logs merged in time order (identical lines once)
    python -m wvad.diagnostics.e023_access blocked --runs e021=DIR --runs e022=DIR [--out JSON]
        results/e023_blocked_ckpts.json: every checkpoint file (*.pt) of the E021 and E022 HB and
        H1 runs (<DIR>/<chain>/<run>/..., run "hb_*" or "h1_*": not hbm, h1r, h0) with sha256
        and size, keyed "<label>/<chain>/<run>/<file>" (no absolute path)

The test access log (--test_log) stays outside the repository (append-only, one JSON line per
event); its committed copy is results/e023_test_access.log (``log``).

Loader guard (access device item 7, gate (k)): ``CheckpointGuard(load_blocked())`` hashes every
checkpoint before it is deserialised and refuses (BlockedCheckpointError) any file whose sha256
is in the blocked list, wherever it lies; a missing or empty list refuses every load (fail
closed). It records the files it loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt_time
import json
import os
import re
import subprocess
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from wvad.data import dense_test as dt
from wvad.paths import REPO_ROOT

E023_DIR = REPO_ROOT / "experiments" / "E023_localisation_baselines"
RESULTS = E023_DIR / "results"
LOG_COPY = RESULTS / "e023_test_access.log"
BLOCKED_FILE = RESULTS / "e023_blocked_ckpts.json"
E022_EXTRACT_GATE = dt.RESULTS / "e022_extract_gate.json"
E022_TEST_SCORES = dt.RESULTS / "e022_test_scores.sha256"
E022_SELECTION = dt.SELECTION_FILE
DATASETS = ("ucf", "xdv")
PAIRS = ("xcb_xcl", "xcb_siglip")
SEEDS = (42, 123, 777, 1000, 2025, 7, 13, 99, 555, 31337)    # = e022_gates.SEEDS (tested)
REUSED_ARMS = ("C", "C.last")
STORE_KEYS = ("test_cache_sha256_ucf", "test_cache_sha256_xdv")
N_CHUNK_CANDIDATES = (4096, 2048, 1024, 512, 256)
BLOCKED_HEADS = ("hb", "h1")
CHANGELOG = dt.CHANGELOG
D1_LINE = re.compile(r'^[ \t]*- D1: "(?P<answer>[^"\n]*)" → \((?P<choice>[a-z])\) '
                     r'(?P<label>[^\n]*?)[ \t]*$', re.M)
APPROVED = "승인"
PURPOSE = "pre-freeze list and hash check (E023 PLAN.md 3d), no test value opened"


class BlockedCheckpointError(RuntimeError):
    """A checkpoint of the blocked list (E021 / E022 HB, H1) was about to be loaded."""


def utc_now() -> str:
    return dt_time.datetime.now(dt_time.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | os.PathLike[str]) -> str:
    from wvad.prep.manifests import sha256_file as _sha

    return _sha(path)


def _json(path: str | os.PathLike[str]) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | os.PathLike[str], obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8",
                   newline="\n")
    os.replace(tmp, p)


# --- D1 approval (E023 PLAN.md) ---------------------------------------------------------------

def d1_approval(plan_dir: str | os.PathLike[str] = E023_DIR) -> dict:
    """The part of PLAN.md before its change log records exactly one D1 decision answered
    "승인..." with choice (a) 승인 (the "결정 답" line); SystemExit otherwise."""
    try:
        text = dt.plan_frozen_part((Path(plan_dir) / "PLAN.md").read_bytes()).decode("utf-8")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"PLAN.md unreadable in {plan_dir} ({exc!r})") from exc
    found = list(D1_LINE.finditer(text))
    ok = (len(found) == 1 and found[0]["choice"] == "a" and found[0]["label"] == APPROVED
          and found[0]["answer"].startswith(APPROVED))
    if not ok:
        raise SystemExit(f"E023 PLAN.md records the D1 decisions "
                         f"{[(m['answer'], m['choice'], m['label']) for m in found]}, not "
                         f"exactly one (a) {APPROVED}: no test access (PLAN.md D1)")
    m = found[0]
    import hashlib

    return {"decision": "D1", "answer": m["answer"], "choice": m["choice"],
            "label": m["label"],
            "line_sha256": hashlib.sha256(m.group(0).strip().encode("utf-8")).hexdigest()}


# --- the checks of access (0) -----------------------------------------------------------------

def chains() -> list[str]:
    return [f"{d}_{p}_s{s}" for d in DATASETS for p in PAIRS for s in SEEDS]


def read_sha256_list(path: str | os.PathLike[str]) -> dict[str, str]:
    """sha256sum format: '<sha256>  <relative path>' per line."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, rel = line.split(maxsplit=1)
            out[rel.strip()] = sha
    return out


def reused_score_hashes(scores_root: str | os.PathLike[str],
                        listed_file: str | os.PathLike[str] = E022_TEST_SCORES) -> dict:
    """The 80 E022 C / C.last score files (bytes only) against e022_test_scores.sha256."""
    listed = read_sha256_list(listed_file)
    files = {}
    for c in chains():
        for arm in REUSED_ARMS:
            rel = f"{c}/{arm}.npz"
            f = Path(scores_root) / "test" / rel
            got = sha256_file(f) if f.is_file() else None
            files[rel] = {"sha256": got, "e022": listed.get(rel),
                          "ok": got is not None and got == listed.get(rel)}
    bad = sorted(k for k, v in files.items() if not v["ok"])
    return {"root": str(scores_root), "list": Path(listed_file).name,
            "list_sha256": sha256_file(listed_file), "files": files, "count": len(files),
            "mismatched": bad, "ok": not bad and len(files) == 80}


def store_check(paths, gate_file: str | os.PathLike[str] = E022_EXTRACT_GATE) -> dict:
    """The dense test stores and derived test caches (bytes) against E022's extraction record."""
    gate = _json(gate_file)
    recorded = {k: v for key in STORE_KEYS for k, v in dict(gate.get(key) or {}).items()}
    got = dt.store_hashes(paths)
    keys = sorted(set(recorded) | set(got))
    bad = [k for k in keys if recorded.get(k) is None or got.get(k) != recorded.get(k)]
    return {"record": Path(gate_file).name, "files": len(recorded),
            "sha256": {k: got.get(k) for k in keys}, "mismatched": bad,
            "ok": not bad and len(recorded) == 48}


def _git_committed(file: Path, root: Path) -> dict | None:
    """In a git checkout: the file is tracked and unchanged against HEAD (None elsewhere)."""
    from wvad.diagnostics.common import code_version

    if code_version().get("source") != "git":
        return None
    rel = file.resolve().relative_to(root.resolve()).as_posix()

    def git(*a):
        return subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *a], cwd=root,
                              capture_output=True, check=False)
    tracked = git("ls-files", "--error-unmatch", "--", rel).returncode == 0
    clean = git("diff", "--quiet", "HEAD", "--", rel).returncode == 0
    return {"file": rel, "tracked": tracked, "unchanged": clean, "ok": tracked and clean}


def selection_check(file: str | os.PathLike[str] = E022_SELECTION,
                    gate_file: str | os.PathLike[str] = E022_EXTRACT_GATE,
                    root: Path = REPO_ROOT) -> dict:
    """results/e022_selection.json has the sha256 E022's access device bound (every guard record
    of the extraction gate file names one value) and is committed unchanged (git checkout)."""
    gate = _json(gate_file)
    bound = sorted({str(v["selection"]["sha256"]) for k, v in gate.items()
                    if k.startswith("guard_") and isinstance(v, Mapping)
                    and isinstance(v.get("selection"), Mapping)})
    got = sha256_file(file) if Path(file).is_file() else None
    git = _git_committed(Path(file), root) if got else None
    ok = len(bound) == 1 and got == bound[0] and (git is None or git["ok"])
    return {"file": Path(file).name, "sha256": got, "e022_bound": bound, "git": git, "ok": ok}


def manifest_record(dataset: str, candidates: Sequence[int] = N_CHUNK_CANDIDATES) -> dict:
    """The restricted read of one test manifest: counts against E020, total_frames (manifest
    order), the maximum T and snippet count, and the videos cut for each candidate N_chunk."""
    from wvad.baselines.common import chunk_targets
    from wvad.extract import grid

    m = dt.read_manifest(dataset)
    want = dt.E020_TEST[dataset]
    totals = list(m.total_frames)
    n_max = max(grid.n_snippets(t) for t in totals)
    return {"manifest": m.summary(),
            "counts_ok": (len(m), m.anomalous, m.frames) == (want["n"], want["anomalous"],
                                                             want["frames"]),
            "total_frames": totals, "max_total_frames": max(totals), "max_snippets": n_max,
            "chunk_targets": {str(c): [m.names[i] for i in chunk_targets(totals, c)]
                              for c in candidates}}


def _zero_body(paths, scores_root, extract_gate, test_scores, selection, root) -> dict:
    out: dict = {d: manifest_record(d) for d in DATASETS}
    hashes = dt.manifest_hashes(paths)
    for d in DATASETS:
        out[d]["hashes"] = hashes[d]
    out["max_total_frames"] = max(out[d]["max_total_frames"] for d in DATASETS)
    out["max_snippets"] = max(out[d]["max_snippets"] for d in DATASETS)
    out["stores"] = store_check(paths, extract_gate)
    out["e022_scores"] = reused_score_hashes(scores_root, test_scores)
    out["e022_selection"] = selection_check(selection, extract_gate, root)
    checks = {f"{d}_counts": out[d]["counts_ok"] for d in DATASETS}
    checks |= {f"{d}_{k}": out[d]["hashes"][k]["ok"] for d in DATASETS
               for k in ("manifest", "provenance", "annotation")}
    checks |= {"stores": out["stores"]["ok"], "e022_scores": out["e022_scores"]["ok"],
               "e022_selection": out["e022_selection"]["ok"]}
    out["checks"] = checks
    out["ok"] = all(checks.values())
    return out


def access_zero(paths, test_log: str | os.PathLike[str], scores_root: str | os.PathLike[str],
                *, plan_dir: str | os.PathLike[str] = E023_DIR,
                extract_gate: str | os.PathLike[str] = E022_EXTRACT_GATE,
                test_scores: str | os.PathLike[str] = E022_TEST_SCORES,
                selection: str | os.PathLike[str] = E022_SELECTION,
                root: Path = REPO_ROOT) -> dict:
    """Test access (0) (module docstring): refused before anything is read unless D1 is
    approved; "start" line, the checks, "end" (or "error") line with the record."""
    from wvad.diagnostics.common import code_version

    dt.outside_repo(test_log, root=root)
    base = {"access": "0", "command": "check"}
    try:
        approval = d1_approval(plan_dir)
    except SystemExit as exc:
        dt.append_test_log(test_log, {**base, "time_utc": utc_now(), "event": "refused",
                                      "reason": str(exc)})
        raise
    plan_sha = dt.plan_sha256(Path(plan_dir) / "PLAN.md")
    dt.append_test_log(test_log, {
        **base, "time_utc": utc_now(), "event": "start", "code": code_version(),
        "plan_sha256_now": plan_sha, "approval": approval, "purpose": PURPOSE,
        "files": {"manifests": list(dt.MANIFEST_FILES.values()),
                  "manifest_fields_read": list(dt.READ_FIELDS),
                  "annotations": list(dt.ANNOTATION_FILES.values()),
                  "dense_test_stores": "<feature_root>/<ds>_<enc>_d16_test (sha256 only)",
                  "e022_scores": f"{scores_root}/test/<chain>/{{C,C.last}}.npz (sha256 only)",
                  "e022_selection": Path(selection).name}})
    try:
        out = _zero_body(paths, scores_root, extract_gate, test_scores, selection, root)
    except BaseException as exc:
        dt.append_test_log(test_log, {**base, "time_utc": utc_now(), "event": "error",
                                      "error": repr(exc)})
        raise
    out |= {"plan_sha256_now": plan_sha, "approval": approval}
    dt.append_test_log(test_log, {**base, "time_utc": utc_now(), "event": "end",
                                  "result": out})
    return out


# --- access-log copy --------------------------------------------------------------------------

def merged_lines(texts: Sequence[str]) -> list[str]:
    """Log lines of several copies: every line once, ordered by its time (stable)."""
    rows, seen = [], set()
    for i, text in enumerate(texts):
        for j, line in enumerate(text.splitlines()):
            if not line.strip() or line in seen:
                continue
            seen.add(line)
            try:
                t = str(json.loads(line).get("time_utc") or "")
            except ValueError:
                t = ""
            rows.append((t, i, j, line))
    return [r[3] for r in sorted(rows)]


def merge_logs(parts: Sequence[str | os.PathLike[str]],
               out: str | os.PathLike[str] = LOG_COPY) -> int:
    lines = merged_lines([Path(p).read_text(encoding="utf-8") for p in parts])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    return len(lines)


# --- blocked checkpoints (3c) and the loader guard (item 7, gate (k)) -------------------------

def blocked_head(run: str) -> bool:
    """A run folder of an HB or H1 head (hb_l20, h1_m10_stage1, ...; not hbm, h1r, h0)."""
    return run.split("_", 1)[0] in BLOCKED_HEADS


def blocked_files(root: str | os.PathLike[str]) -> dict[str, Path]:
    """<chain>/<run>/<relative file> -> path of every *.pt under the HB / H1 run folders."""
    base = Path(root)
    if not base.is_dir():
        raise SystemExit(f"runs root {root} is not a folder")
    out = {}
    for chain in sorted(p for p in base.iterdir() if p.is_dir()):
        for run in sorted(p for p in chain.iterdir() if p.is_dir() and blocked_head(p.name)):
            for f in sorted(run.rglob("*.pt")):
                out[f"{chain.name}/{run.name}/{f.relative_to(run).as_posix()}"] = f
    return out


def blocked_record(roots: Mapping[str, str | os.PathLike[str]]) -> dict:
    """The (3c) list: sha256 and size of every HB / H1 checkpoint under each labelled root."""
    from wvad.diagnostics.common import code_version

    files = {}
    for label, root in roots.items():
        for key, f in blocked_files(root).items():
            files[f"{label}/{key}"] = {"sha256": sha256_file(f), "size": f.stat().st_size}
    shas = sorted({v["sha256"] for v in files.values()})
    return {"check": "E023 (3c) blocked checkpoints (E021 / E022 HB, H1)", "utc": utc_now(),
            "code": code_version(), "roots": sorted(roots), "heads": list(BLOCKED_HEADS),
            "files": files, "count": len(files), "sha256_set": shas}


def load_blocked(path: str | os.PathLike[str] = BLOCKED_FILE) -> frozenset[str]:
    """The blocked sha256 set (SystemExit when the list is missing, unreadable or empty)."""
    try:
        rec = _json(path)
        shas = frozenset(str(v["sha256"]) for v in rec["files"].values())
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SystemExit(f"blocked-checkpoint list {path} unreadable ({exc!r}): no checkpoint "
                         "is loaded") from exc
    if not shas:
        raise SystemExit(f"blocked-checkpoint list {path} is empty: no checkpoint is loaded")
    return shas


class CheckpointGuard:
    """Refuses to deserialise a blocked checkpoint (module docstring)."""

    def __init__(self, blocked: Collection[str]) -> None:
        self.blocked = frozenset(str(s) for s in blocked)
        if not self.blocked:
            raise SystemExit("an empty blocked list guards nothing: refused")
        self.loaded: list[dict] = []

    def check(self, path: str | os.PathLike[str]) -> str:
        sha = sha256_file(path)
        if sha in self.blocked:
            raise BlockedCheckpointError(f"{path} (sha256 {sha}) is an E021 / E022 HB or H1 "
                                         "checkpoint: C is never scored again (PLAN.md item 7)")
        return sha

    def load(self, path: str | os.PathLike[str], map_location="cpu"):
        import torch

        sha = self.check(path)
        obj = torch.load(path, map_location=map_location, weights_only=True)
        self.loaded.append({"file": str(path), "sha256": sha, "time_utc": utc_now()})
        return obj


# --- command line ---------------------------------------------------------------------------

def _roots(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for it in items:
        label, sep, path = it.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--runs takes LABEL=DIR, got {it!r}")
        out[label] = path
    return out


def main(argv: Sequence[str] | None = None) -> None:
    from wvad.paths import load_paths

    p = argparse.ArgumentParser(prog="python -m wvad.diagnostics.e023_access")
    sub = p.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check", help="test access (0)")
    ck.add_argument("--test_log", required=True, help="test access log (outside the repository)")
    ck.add_argument("--e022_scores", required=True,
                    help="E022 score root holding test/<chain>/{C,C.last}.npz")
    ck.add_argument("--out", type=Path, help="the record (JSON)")
    ck.add_argument("--paths", help="paths file (default configs/paths.yaml)")
    lg = sub.add_parser("log", help="merge access logs into results/e023_test_access.log")
    lg.add_argument("parts", nargs="+", type=Path)
    bl = sub.add_parser("blocked", help="(3c) results/e023_blocked_ckpts.json")
    bl.add_argument("--runs", action="append", required=True, help="LABEL=DIR (e021, e022)")
    bl.add_argument("--out", type=Path, default=BLOCKED_FILE)
    a = p.parse_args(argv)
    if a.cmd == "check":
        out = access_zero(load_paths(a.paths), a.test_log, a.e022_scores)
        if a.out is not None:
            _write_json(a.out, out)
        print(f"[e023_access] access (0): ok={out['ok']} max_total_frames="
              f"{out['max_total_frames']} max_snippets={out['max_snippets']}"
              f" checks={ {k: v for k, v in out['checks'].items() if not v} or 'all pass'}")
        if not out["ok"]:
            raise SystemExit(1)
    elif a.cmd == "log":
        print(f"[e023_access] {merge_logs(a.parts)} lines -> {LOG_COPY}")
    else:
        rec = blocked_record(_roots(a.runs))
        _write_json(a.out, rec)
        print(f"[e023_access] {rec['count']} blocked checkpoints -> {a.out}")


if __name__ == "__main__":
    main()
