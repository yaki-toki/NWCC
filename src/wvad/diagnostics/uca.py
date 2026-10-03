"""UCA proxy intervals for anomalous UCF-Crime TRAINING videos (diagnostics only).

UCA (Yuan et al., CVPR 2024; Apache-2.0, academic and research use only) gives sentence
annotations with [start, end] seconds for UCF-Crime videos. For each anomalous training
video, sentences that contain a stem of the video's own category lexicon are selected:

    M1  substring match (the previous project's v1 rule; its onsets were audited)
    M2  word-start match (regex \\b + stem); avoids e.g. "hit" inside "white"

and turned into frame intervals (frame f is positive when start <= f / fps < end):

    S   one interval per selected sentence (descriptive)
    A   union of the sentence intervals (overlapping or touching intervals merged)
    B   hull: earliest start to latest end

Videos are looked up only by the names in the training manifest; every other UCA entry
(including test videos) is ignored and counted. Proxies are never used as training
signals or selection criteria.

    python -m wvad.diagnostics.uca --out data/annotations/ucf_train_uca_proxy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wvad.data.manifests import TrainManifest, load_manifest, manifest_file
from wvad.paths import load_paths

UCA_FILES = ("UCFCrime_Train.json", "UCFCrime_Val.json", "UCFCrime_Test.json")
MATCHINGS = ("M1", "M2")
SHAPES = ("S", "A", "B")
VARIANTS = ("M1A", "M1B", "M2A", "M2B")

# Per-category stem lexicon, copied verbatim from the previous project
# (scripts/rebuttal_phase2/uca_audit.py, LEX; sha256 of that file in E002 config.yaml).
LEXICON: Mapping[str, tuple[str, ...]] = {
    "Abuse": ("abus", "beat", "hit", "kick", "slap", "drag", "attack", "assault", "violen",
              "crush"),
    "Arrest": ("arrest", "handcuff", "apprehend", "detain", "polic"),
    "Arson": ("arson", "fire", "ignit", "burn", "flame", "gasolin", "lighter", "torch"),
    "Assault": ("assault", "attack", "beat", "punch", "kick", "hit", "fight", "strik"),
    "Burglary": ("burglar", "break", "steal", "sneak", "intrud", "climb", "pry", "theft",
                 "thief", "loot"),
    "Explosion": ("explos", "explod", "blast", "blow", "detonat", "fireball"),
    "Fighting": ("fight", "punch", "kick", "brawl", "wrestl", "hit", "beat", "scuffl"),
    "RoadAccidents": ("accident", "crash", "collid", "collis", "hit", "knock", "run over",
                      "ran over", "overturn", "rollover"),
    "Robbery": ("rob", "gun", "threat", "snatch", "steal", "loot", "hold up", "holdup",
                "weapon", "knife"),
    "Shooting": ("shoot", "shot", "gun", "fire at", "fired at", "gunman", "open fire"),
    "Shoplifting": ("shoplift", "steal", "conceal", "pocket", "hide", "slip", "stuff", "theft",
                    "take"),
    "Stealing": ("steal", "theft", "thief", "snatch", "take away", "took away", "grab",
                 "make off", "made off"),
    "Vandalism": ("vandal", "smash", "break", "damag", "destroy", "spray", "kick", "throw"),
}


def sentence_matches(sentence: str, stems: Sequence[str], matching: str) -> bool:
    low = sentence.lower()
    if matching == "M1":
        return any(s in low for s in stems)
    if matching == "M2":
        return any(re.search(r"\b" + re.escape(s), low) for s in stems)
    raise ValueError(f"unknown matching '{matching}'")


def seconds_to_frames(start: float, end: float, fps: float,
                      total_frames: int) -> tuple[int, int] | None:
    """Half-open frame interval [f0, f1) of frames f with start <= f / fps < end."""
    f0 = max(0, math.ceil(round(start * fps, 6)))
    f1 = min(total_frames, math.ceil(round(end * fps, 6)))
    return (f0, f1) if f1 > f0 else None


def union(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def hull(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    return [(min(a for a, _ in intervals), max(b for _, b in intervals))]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_uca(directory: Path) -> tuple[dict[str, dict], dict[str, str]]:
    merged: dict[str, dict] = {}
    hashes = {}
    for name in UCA_FILES:
        path = directory / name
        merged.update(json.loads(path.read_text(encoding="utf-8")))
        hashes[name] = sha256_file(path)
    return merged, hashes


def video_proxy(entry: Mapping | None, category: str, fps: float, total_frames: int) -> dict:
    out: dict = {"uca": bool(entry and entry.get("sentences"))}
    for matching in MATCHINGS:
        idx, frames = [], []
        if out["uca"]:
            for i, (ts, sent) in enumerate(zip(entry["timestamps"], entry["sentences"],
                                               strict=True)):
                if sentence_matches(sent, LEXICON[category], matching):
                    iv = seconds_to_frames(float(ts[0]), float(ts[1]), fps, total_frames)
                    if iv is not None:
                        idx.append(i)
                        frames.append(iv)
        out[matching] = {"sentences": idx, "S": [list(iv) for iv in frames],
                         "A": [list(iv) for iv in union(frames)],
                         "B": [list(iv) for iv in hull(frames)]}
    return out


def build_proxy(manifest: TrainManifest, uca: Mapping[str, dict],
                uca_hashes: Mapping[str, str]) -> dict:
    names = set(manifest.names)
    videos = {}
    for i, (name, cat, label) in enumerate(zip(manifest.names, manifest.categories,
                                                manifest.labels, strict=True)):
        if label != 1:
            continue
        fps = manifest.fps[i]
        if fps is None or fps <= 0:
            raise ValueError(f"{name}: manifest has no fps")
        entry = uca.get(name)
        videos[name] = {"index": i, "category": cat, "fps": fps,
                        "total_frames": manifest.total_frames[i],
                        "uca_duration": None if entry is None else entry.get("duration"),
                        **video_proxy(entry, cat, fps, manifest.total_frames[i])}
    meta = {"source": "UCA (Yuan et al., CVPR 2024), Apache-2.0, academic and research use only",
            "uca_files": dict(uca_hashes),
            "rule": "category stem lexicon; M1 substring, M2 word-start; frames start<=f/fps<end",
            "n_uca_entries": len(uca),
            "n_uca_in_training_manifest": sum(1 for n in uca if n in names),
            "n_uca_ignored": sum(1 for n in uca if n not in names),
            "n_anomalous_training_videos": len(videos)}
    return {"meta": meta, "videos": videos}


@dataclass(frozen=True)
class Proxy:
    videos: Mapping[str, Mapping]
    meta: Mapping[str, object]

    def has(self, name: str, variant: str) -> bool:
        return bool(self.intervals(name, variant))

    def intervals(self, name: str, variant: str) -> list[tuple[int, int]]:
        """Frame intervals of a variant ('M1A', 'M2B', 'M1S', ...); [] when there is none."""
        matching, shape = variant[:2], variant[2:]
        if matching not in MATCHINGS or shape not in SHAPES:
            raise ValueError(f"unknown proxy variant '{variant}'")
        v = self.videos.get(name)
        return [] if v is None else [tuple(iv) for iv in v[matching][shape]]

    def frame_labels(self, name: str, variant: str, total_frames: int) -> np.ndarray:
        labels = np.zeros(total_frames, dtype=bool)
        for a, b in self.intervals(name, variant):
            labels[a:b] = True
        return labels


def load_proxy(path: str | Path) -> Proxy:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Proxy(raw["videos"], raw["meta"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build UCA proxy intervals for UCF training videos.")
    ap.add_argument("--uca_dir", type=Path, help="default: paths.yaml annotations.uca")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    uca_dir = args.uca_dir or load_paths().annotation_dir("uca")
    mpath = manifest_file("ucf")
    manifest = load_manifest(mpath)
    uca, hashes = load_uca(uca_dir)
    proxy = build_proxy(manifest, uca, hashes)
    proxy["meta"]["manifest_sha256"] = sha256_file(mpath)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proxy, indent=1) + "\n", encoding="utf-8", newline="\n")
    m = proxy["meta"]
    print(f"{m['n_anomalous_training_videos']} anomalous training videos; UCA entries "
          f"{m['n_uca_entries']} (in training manifest {m['n_uca_in_training_manifest']}, "
          f"ignored {m['n_uca_ignored']}) -> {args.out}")


if __name__ == "__main__":
    main()
