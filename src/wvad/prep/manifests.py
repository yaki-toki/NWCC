"""Build the training manifests in data/manifests/ from the previous project's lists.

UCF-Crime: names and order come from the previous ``ucf_train_manifest.json`` (paths in the
order of the training feature cache); frame count and fps are read from each video header
with OpenCV, the same call the feature extractor used. Only the listed training videos are
opened. XD-Violence: names, label codes, labels and frame counts are copied from the
previous ``xdv_train_manifest.json``.

    python -m wvad.prep.manifests ucf --previous <ucf_train_manifest.json> \
        --dataset_root <UCF-Crime root> --out data/manifests/ucf_train.json
    python -m wvad.prep.manifests xdv --previous <xdv_train_manifest.json> \
        --out data/manifests/xdv_train.json

Needs ``opencv-python`` for UCF (``pip install -e ".[prep]"``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from wvad.data.manifests import TrainManifest, manifest_to_json, parse_manifest

UCF_ROOT_NAME = "Anomaly-Detection-Dataset"
NORMAL_PREFIX = "Training-Normal-Videos"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ucf_rel_path(previous_path: str) -> str:
    """'./Anomaly-Detection-Dataset/<part>/.../<file>.mp4' -> '<part>/.../<file>.mp4'."""
    parts = PurePosixPath(previous_path.replace("\\", "/")).parts
    if UCF_ROOT_NAME not in parts:
        raise ValueError(f"unexpected UCF path (no '{UCF_ROOT_NAME}'): {previous_path}")
    return str(PurePosixPath(*parts[parts.index(UCF_ROOT_NAME) + 1:]))


def ucf_category(rel_path: str) -> str:
    parts = PurePosixPath(rel_path).parts
    return "Normal" if parts[0].startswith(NORMAL_PREFIX) else parts[-2]


def read_header(path: Path) -> tuple[int, float]:
    import cv2  # optional dependency, only needed here

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video: {path}")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def build_ucf(previous: Path, dataset_root: Path) -> TrainManifest:
    import cv2

    paths = json.loads(previous.read_text(encoding="utf-8"))["paths"]
    rel = [ucf_rel_path(p) for p in paths]
    headers = [read_header(dataset_root / r) for r in rel]
    cats = [ucf_category(r) for r in rel]
    raw = {"meta": {"dataset": "ucf", "split": "train", "n": len(rel),
                    "source": {"file": previous.name, "sha256": sha256_file(previous)},
                    "headers": f"cv2 {cv2.__version__} CAP_PROP_FRAME_COUNT, CAP_PROP_FPS",
                    "order": "training feature cache order (previous UCFCrimeMILDataset)"},
           "names": [PurePosixPath(r).stem for r in rel], "categories": cats,
           "labels": [0 if c == "Normal" else 1 for c in cats],
           "total_frames": [h[0] for h in headers], "fps": [h[1] for h in headers],
           "rel_paths": rel}
    return parse_manifest(raw, source="ucf manifest")


def build_xdv(previous: Path) -> TrainManifest:
    prev = json.loads(previous.read_text(encoding="utf-8"))
    n = len(prev["names"])
    raw = {"meta": {"dataset": "xdv", "split": "train", "n": n,
                    "source": {"file": previous.name, "sha256": sha256_file(previous)},
                    "skipped_unreadable": prev.get("meta", {}).get("skipped_unreadable", []),
                    "order": "training feature cache order (sorted listing minus unreadable)"},
           "names": prev["names"], "categories": prev["label_codes"], "labels": prev["labels"],
           "total_frames": prev["total_frames"], "fps": [None] * n, "rel_paths": [None] * n}
    return parse_manifest(raw, source="xdv manifest")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build a training manifest.")
    ap.add_argument("dataset", choices=["ucf", "xdv"])
    ap.add_argument("--previous", required=True, type=Path)
    ap.add_argument("--dataset_root", type=Path, help="UCF-Crime root (ucf only)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.dataset == "ucf":
        if args.dataset_root is None:
            ap.error("ucf needs --dataset_root")
        m = build_ucf(args.previous, args.dataset_root)
    else:
        m = build_xdv(args.previous)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest_to_json(m), indent=0) + "\n", encoding="utf-8",
                        newline="\n")
    print(f"{args.dataset}: {len(m)} videos -> {args.out}")


if __name__ == "__main__":
    main()
