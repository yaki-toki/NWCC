"""E002/C2: how often one clip per segment misses a UCA proxy event (no model, no training).

Clip rule of the previous extractor (S = 32 segments, segment length L = T // 32): a clip
starting at s observes frames s, s+2, ..., s+14 (clamped to the last frame); it may spill
into the next segment when L < 16.

* random policy (training caches): s uniform over the integers
  [seg_start, max(seg_start, min(seg_end - 16, T - 16))], independently per segment;
* centre policy (the previous test caches, applied here to training videos to compare the
  policies): s = seg_start + (L - 16) // 2, clipped to [0, T - 16].

An event (a proxy interval [a, b) in frames) is missed when no observed frame of any of the
32 clips falls inside it; under the random policy this probability is the product over
segments of the per-segment miss probabilities. Also reported: the expected fraction of
event frames that are observed frames, and XD / UCF clip-span geometry.

    python -m wvad.diagnostics.c2
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from wvad.data.manifests import TrainManifest, load_manifest, manifest_file
from wvad.diagnostics.c0 import PROXY_FILE
from wvad.diagnostics.common import E002_RESULTS, code_version, write_csv, write_json
from wvad.diagnostics.uca import Proxy, load_proxy

N_SEGMENTS, CLIP_FRAMES, STRIDE, PLACEMENT_SPAN = 32, 8, 2, 16
MIN_FRAMES = 256
CONFIRM_MISS_RATE = 0.10
VERDICT_VARIANTS = ("M1A", "M2A")
ALL_VARIANTS = ("M1A", "M2A", "M1B", "M2B", "M1S", "M2S")
LENGTH_BINS = (0.5, 1.0, 2.0)          # event length in segment lengths
POLICIES = ("random", "centre")


def start_range(k: int, total: int) -> tuple[int, int]:
    """Inclusive range of random-policy clip starts of segment k."""
    seg_len = total // N_SEGMENTS
    seg_start, seg_end = k * seg_len, (k + 1) * seg_len
    max_start = max(seg_start, min(seg_end - PLACEMENT_SPAN, total - PLACEMENT_SPAN))
    return seg_start, max(seg_start, max_start)


def centre_start(k: int, total: int) -> int:
    seg_len = total // N_SEGMENTS
    s = k * seg_len + (seg_len - PLACEMENT_SPAN) // 2
    return max(0, min(s, total - PLACEMENT_SPAN))


def observed(starts: np.ndarray, total: int) -> np.ndarray:
    """(n, 8) observed frames of clips starting at ``starts``."""
    f = starts[:, None] + STRIDE * np.arange(CLIP_FRAMES)[None, :]
    return np.clip(f, 0, total - 1)


def distinct_inside(obs: np.ndarray, a: int, b: int, total: int) -> np.ndarray:
    """Number of distinct observed frames inside [a, b) per clip (rows of ``obs``).

    Frames of one clip are distinct except repeats of the last frame after clamping.
    """
    n = ((obs >= a) & (obs < b)).sum(axis=1)
    if a <= total - 1 < b:
        n = n - np.maximum((obs == total - 1).sum(axis=1) - 1, 0)
    return n


def event_stats(a: int, b: int, total: int) -> dict:
    """Miss probability and expected observed-frame fraction of event [a, b), both policies."""
    p_miss, exp_frames, centre_frames = 1.0, 0.0, 0
    for k in range(N_SEGMENTS):
        lo, hi = start_range(k, total)
        if hi + STRIDE * (CLIP_FRAMES - 1) >= a and lo < b:
            n = distinct_inside(observed(np.arange(lo, hi + 1), total), a, b, total)
            p_miss *= 1.0 - float((n > 0).mean())
            exp_frames += float(n.mean())
        centre_frames += int(distinct_inside(
            observed(np.array([centre_start(k, total)]), total), a, b, total)[0])
    length = b - a
    return {"random_miss": p_miss, "centre_miss": float(centre_frames == 0),
            "random_observed_fraction": exp_frames / length,
            "centre_observed_fraction": centre_frames / length}


def length_bin(length: int, total: int) -> str:
    x = length / (total // N_SEGMENTS)
    edges = (0.0, *LENGTH_BINS, np.inf)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if lo <= x < hi:
            return f"[{lo:g},{hi:g})L"
    raise AssertionError(x)


def collect_events(manifest: TrainManifest, proxy: Proxy, variant: str) -> list[dict]:
    events = []
    for name, v in proxy.videos.items():
        total = manifest.total_frames[v["index"]]
        if total < MIN_FRAMES:
            continue
        for a, b in proxy.intervals(name, variant):
            events.append({"video": name, "category": v["category"], "a": a, "b": b,
                           "length": b - a, "bin": length_bin(b - a, total),
                           **event_stats(a, b, total)})
    return events


def summarize(events: Sequence[dict], policy: str) -> dict:
    miss = np.array([e[f"{policy}_miss"] for e in events])
    length = np.array([e["length"] for e in events], dtype=float)
    by_video: dict[str, list[float]] = {}
    for e, m in zip(events, miss, strict=True):
        by_video.setdefault(e["video"], []).append(m)
    return {"n_events": len(events), "n_videos": len(by_video),
            "miss_rate": float(miss.mean()),
            "miss_rate_video_mean": float(np.mean([np.mean(v) for v in by_video.values()])),
            "miss_rate_length_weighted": float((miss * length).sum() / length.sum()),
            "observed_frame_fraction": float(np.mean([e[f"{policy}_observed_fraction"]
                                                       for e in events]))}


def clip_geometry(dataset: str, manifest: TrainManifest) -> dict:
    seg_len = np.array([t // N_SEGMENTS for t in manifest.total_frames if t >= N_SEGMENTS],
                       dtype=float)
    cover = np.minimum(1.0, (STRIDE * (CLIP_FRAMES - 1) + 1) / seg_len)
    q = np.percentile(cover, [10, 25, 50, 75, 90])
    return {"dataset": dataset, "n_videos": len(seg_len),
            "segment_frames_median": float(np.median(seg_len)),
            **{f"span_fraction_q{p}": float(v) for p, v in zip((10, 25, 50, 75, 90), q,
                                                              strict=True)}}


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="E002 C2 clip coverage").parse_args(argv)
    ucf = load_manifest(manifest_file("ucf"))
    proxy = load_proxy(PROXY_FILE)
    rows, verdict_rates = [], {}
    for variant in ALL_VARIANTS:
        events = collect_events(ucf, proxy, variant)
        groups = {"all": events}
        if variant in VERDICT_VARIANTS:
            for e in events:
                groups.setdefault(f"category={e['category']}", []).append(e)
                groups.setdefault(f"length={e['bin']}", []).append(e)
        for scope, evs in groups.items():
            for policy in POLICIES:
                r = summarize(evs, policy)
                rows.append({"variant": variant, "policy": policy, "scope": scope, **r})
                if scope == "all" and variant in VERDICT_VARIANTS:
                    verdict_rates[(variant, policy)] = r["miss_rate"]
    confirmed = [p for p in POLICIES
                 if all(verdict_rates[(v, p)] >= CONFIRM_MISS_RATE for v in VERDICT_VARIANTS)]
    geometry = [clip_geometry("ucf", ucf),
                clip_geometry("xdv", load_manifest(manifest_file("xdv")))]
    summary = {"check": "E002/C2 clip coverage (PLAN.md)", "code": code_version(),
               "min_frames": MIN_FRAMES,
               "verdict_miss_rates": {f"{v}/{p}": round(r, 4)
                                      for (v, p), r in verdict_rates.items()},
               "verdict": "confirm" if confirmed else "hold",
               "confirming_policies": confirmed, "threshold": CONFIRM_MISS_RATE}
    write_csv(E002_RESULTS / "c2_coverage.csv", rows)
    write_csv(E002_RESULTS / "c2_xd_geometry.csv", geometry)
    write_json(E002_RESULTS / "c2_summary.json", summary)
    for r in rows:
        if r["scope"] == "all":
            print(f"{r['variant']} {r['policy']:6s} events={r['n_events']} "
                  f"miss={r['miss_rate']:.4f} observed_frac={r['observed_frame_fraction']:.4f}")
    print("verdict:", summary["verdict"], summary["verdict_miss_rates"])


if __name__ == "__main__":
    main()
