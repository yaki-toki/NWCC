"""Small helpers shared by the diagnostics: code version stamp and result writers."""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from wvad.paths import REPO_ROOT

E002_RESULTS = REPO_ROOT / "experiments" / "E002_diagnostics" / "results"
E003_RESULTS = REPO_ROOT / "experiments" / "E003_training_diagnostics" / "results"
DEV_SEEDS = (0, 1, 2)


def check_dev_seed(seed: int) -> None:
    """Diagnostics use development seeds only (confirmation seeds stay unused)."""
    if int(seed) not in DEV_SEEDS:
        raise ValueError(f"seed {seed} is not a development seed {DEV_SEEDS}")


def code_version() -> dict:
    """Commit of the analysis code and whether tracked files had uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
                              cwd=REPO_ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def write_json(path: Path, obj: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
