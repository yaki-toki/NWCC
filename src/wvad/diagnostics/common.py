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
    """Commit of the analysis code and whether tracked files had uncommitted changes.

    A deployed copy without git (server, ``git archive``) reports the commit written to
    ``DEPLOYED_COMMIT`` at deployment (README, server conventions); "dirty" is then unknown.
    """
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}",
                                   *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=False).stdout.strip()
        except OSError:
            return ""
    commit = git("rev-parse", "HEAD")
    if commit:
        return {"commit": commit, "source": "git",
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    deployed = REPO_ROOT / "DEPLOYED_COMMIT"
    if deployed.is_file():
        return {"commit": deployed.read_text(encoding="utf-8").strip(), "source": "deployed",
                "dirty": None}
    return {"commit": None, "source": None, "dirty": None}


def gpu_environment() -> dict:
    """torch, CUDA, GPU name and NVIDIA driver of the machine running the code (server gates)."""
    import torch
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        driver = None
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": driver}


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
