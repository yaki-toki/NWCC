"""Build the test-set manifests in data/manifests/ (stage F) from the previous project's lists,
cross-checked against the official temporal annotations. No model score is read.

UCF-Crime: rows, frame counts (video headers, cv2) and intervals come from the previous
``ucf_test_manifest.json`` (test feature-cache order: anomalous videos in annotation-file
order, then normal videos in directory-listing order); every row is checked against
``Temporal_Anomaly_Annotation_for_Testing_Videos.txt`` (category, up to two intervals).
XD-Violence: rows, labels, intervals and frame counts come from the previous
``xdv_test_manifest.json`` (sorted listing; frame counts equal the cache's); every anomalous
row is checked against ``annotations/test_annotation.txt`` (exact name, or the extractor's
prefix rule), and every normal row must have no annotation.

    python -m wvad.prep.eval_manifests ucf --previous <ucf_test_manifest.json> \
        --annotation <Temporal_Anomaly_Annotation_for_Testing_Videos.txt> \
        --out data/manifests/ucf_test.json
    python -m wvad.prep.eval_manifests xdv --previous <xdv_test_manifest.json> \
        --annotation <XD-Violence/annotations/test_annotation.txt> \
        --out data/manifests/xdv_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from wvad.data.eval_manifests import EvalManifest, eval_manifest_to_json, parse_eval_manifest
from wvad.prep.manifests import sha256_file

EXPECTED = {"ucf": (290, 140), "xdv": (800, 500)}       # (videos, anomalous)


def _source(path: Path) -> dict:
    return {"file": path.name, "sha256": sha256_file(path)}


def _rows(previous: Path) -> list[dict]:
    samples = json.loads(previous.read_text(encoding="utf-8"))["samples"]
    if [s["index"] for s in samples] != list(range(len(samples))):
        raise ValueError(f"{previous}: rows are not in index order 0..{len(samples) - 1}")
    return samples


def parse_ucf_annotation(path: Path) -> dict[str, tuple[str, list[tuple[int, int]]]]:
    """'Abuse028_x264.mp4  Abuse  165  240  -1  -1' -> {stem: (category, intervals)}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if not tok:
            continue
        nums = [int(x) for x in tok[2:]]
        pairs = zip(nums[::2], nums[1::2], strict=True)
        out[PurePosixPath(tok[0]).stem] = (tok[1], [(s, e) for s, e in pairs if s != -1])
    return out


def build_ucf(previous: Path, annotation: Path) -> EvalManifest:
    rows, ann = _rows(previous), parse_ucf_annotation(annotation)
    names = [PurePosixPath(r["filename"]).stem for r in rows]
    for r, name in zip(rows, names, strict=True):
        cat, iv = ann[name]
        if cat != r["category"] or [tuple(x) for x in r["event_ranges"]] != iv:
            raise ValueError(f"{name}: previous manifest differs from the annotation file")
        if bool(r["is_anomaly"]) != (cat != "Normal"):
            raise ValueError(f"{name}: label does not follow the category")
    if set(names) != set(ann):
        raise ValueError("test rows and annotation-file videos differ")
    raw = {"meta": {"dataset": "ucf", "split": "test", "n": len(rows),
                    "source": _source(previous), "annotation": _source(annotation),
                    "order": "test feature-cache order (previous UCFCrimeMILDataset, split=test)",
                    "total_frames": "video header, cv2 CAP_PROP_FRAME_COUNT (previous build)"},
           "names": names, "categories": [r["category"] for r in rows],
           "labels": [int(bool(r["is_anomaly"])) for r in rows],
           "intervals": [r["event_ranges"] for r in rows],
           "total_frames": [r["total_frames"] for r in rows], "fps": [r["fps"] for r in rows]}
    return _checked("ucf", raw)


def parse_xdv_annotation(path: Path) -> dict[str, tuple[str, list[tuple[int, int]]]]:
    """'<name> <code> s e [s e ...]' -> {name: (code, intervals)}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if not tok:
            continue
        nums = [int(x) for x in tok[2:]]
        out[tok[0]] = (tok[1], list(zip(nums[::2], nums[1::2], strict=True)))
    return out


def match_annotation(name: str, keys) -> tuple[str | None, str | None]:
    """The extractor's rule: exact name, else a unique prefix match in either direction."""
    if name in keys:
        return name, "exact"
    hits = [k for k in keys if name.startswith(k) or k.startswith(name)]
    if len(hits) > 1:
        raise ValueError(f"{name}: ambiguous annotation prefix match {hits}")
    return (hits[0], "prefix") if hits else (None, None)


def xdv_code(name: str) -> str:
    """'..._label_B2-0-0' -> 'B2' (the training manifest's rule)."""
    return name.rsplit("_label_", 1)[1].split("-")[0]


def build_xdv(previous: Path, annotation: Path) -> EvalManifest:
    rows, ann = _rows(previous), parse_xdv_annotation(annotation)
    used, prefix = set(), 0
    for r in rows:
        key, how = match_annotation(r["name"], ann)
        if bool(r["label"]) != (key is not None):
            raise ValueError(f"{r['name']}: label {r['label']} but annotation match {key}")
        if key is not None:
            if [tuple(x) for x in r["intervals"]] != ann[key][1] or key in used:
                raise ValueError(f"{r['name']}: intervals differ from the annotation or reused")
            used.add(key)
            prefix += how == "prefix"
    if used != set(ann):
        raise ValueError(f"annotation entries without a test row: {sorted(set(ann) - used)}")
    raw = {"meta": {"dataset": "xdv", "split": "test", "n": len(rows),
                    "source": _source(previous), "annotation": _source(annotation),
                    "order": "test feature-cache order (sorted listing, none unreadable)",
                    "annotation_prefix_matches": prefix,
                    "total_frames": "the cache's total_frames (previous build)"},
           "names": [r["name"] for r in rows], "categories": [xdv_code(r["name"]) for r in rows],
           "labels": [int(r["label"]) for r in rows], "intervals": [r["intervals"] for r in rows],
           "total_frames": [r["total_frames"] for r in rows], "fps": [r["fps"] for r in rows]}
    return _checked("xdv", raw)


def _checked(dataset: str, raw: dict) -> EvalManifest:
    m = parse_eval_manifest(raw, source=f"{dataset} test manifest")
    n, n_anom = EXPECTED[dataset]
    if (len(m), sum(m.labels)) != (n, n_anom):
        raise ValueError(f"{dataset}: {len(m)} videos / {sum(m.labels)} anomalous, expected "
                         f"{n} / {n_anom}")
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build a test-set manifest (stage F).")
    ap.add_argument("dataset", choices=["ucf", "xdv"])
    ap.add_argument("--previous", required=True, type=Path)
    ap.add_argument("--annotation", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    build = build_ucf if args.dataset == "ucf" else build_xdv
    m = build(args.previous, args.annotation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(eval_manifest_to_json(m), indent=0) + "\n", encoding="utf-8",
                        newline="\n")
    print(f"{args.dataset}: {len(m)} videos ({sum(m.labels)} anomalous), "
          f"{sum(m.total_frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
