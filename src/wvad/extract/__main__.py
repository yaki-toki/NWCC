"""Command line of the E021 dense extraction (commands are listed in the README).

    python -m wvad.extract gate env    --out env_extract.json
    python -m wvad.extract gate tests  --junit <pytest JUnit XML> --record <gate json>
    python -m wvad.extract gate norm   --record <gate json>
    python -m wvad.extract gate x1     --dataset D --access_log CSV --out JSON [--record ...]
    python -m wvad.extract gate x2     --dataset D --x1 JSON --access_log CSV --out JSON [...]
    python -m wvad.extract bench       --out_root DIR --access_log CSV --out JSON [--record ...]
    python -m wvad.extract header      --dataset D --access_log CSV --out CSV [--subset trainpart]
                                       [--sha_only]
    python -m wvad.extract dense       --dataset D --phase val|trainpart --access_log CSV
    python -m wvad.extract derive      --dataset D --stage 1|2
    python -m wvad.extract check x3a|x3|x4|x5|x6 ... --out JSON [--record ...]

Heavy modules are imported inside the commands, so the spawned decode workers (which import
this module again) load only numpy and OpenCV. ``--record`` merges a command's summary into the
gate record (results/e021_extract_gate.json) under its own key. Every video file is opened only
through the training manifest and appended to the access log first.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ENCODERS = ("xcb", "xcl", "siglip")
PRODUCTION_BATCH = "xcb=8,xcl=8,siglip=4"
STAGE_LIMIT_HOURS = 24          # per-stage stop threshold of the projected extraction time
NORMALIZE = ("lut", "numpy")


def _batch(text: str) -> dict[str, int]:
    """Batch sizes as xcb=8,xcl=8,siglip=4; an encoder not named keeps the production batch."""
    out = {"xcb": 8, "xcl": 8, "siglip": 4}
    for item in text.split(","):
        name, _, value = item.partition("=")
        if name not in ENCODERS or not value.isdigit() or int(value) < 1:
            raise argparse.ArgumentTypeError(f"batch must look like {PRODUCTION_BATCH}: {text}")
        out[name] = int(value)
    return out


def _write_json(path: str | os.PathLike[str], obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")


def merge_record(path: str | os.PathLike[str] | None, key: str, value) -> None:
    """Set ``key`` of the JSON gate record (created when missing)."""
    if path is None:
        return
    p = Path(path)
    record = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    record[key] = value
    _write_json(p, record)


def _summaries(result: dict) -> dict:
    """A gate result without its per-clip rows and frame hashes (for the gate record)."""
    out = {k: v for k, v in result.items() if k not in ("encoders", "frame_hashes")}
    out["encoders"] = {e: {k: v for k, v in r.items() if k != "rows"}
                       for e, r in result.get("encoders", {}).items()}
    return out


def _data(dataset: str):
    from wvad.data.catalog import split_file
    from wvad.data.manifests import load_manifest, manifest_file
    from wvad.data.splits import check_labels, load_split

    manifest, split = load_manifest(manifest_file(dataset)), load_split(split_file(dataset))
    check_labels(split, manifest.labels, f"{dataset} training manifest")
    return manifest, split


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encoders(names, device):
    from wvad.extract.encoders import Encoder

    return {e: Encoder.load(e, device) for e in names}


def _provenance() -> tuple[dict, dict]:
    from wvad.diagnostics.common import code_version
    from wvad.extract.encoders import extraction_environment

    return code_version(), extraction_environment()


# --- commands -------------------------------------------------------------------------------

def cmd_gate(args) -> None:
    from wvad.extract import gate
    from wvad.extract.dense import log_access, rel_video_path, utc_now, video_path
    from wvad.paths import load_paths

    device = _device()
    if args.check == "env":
        import torch

        commit, env = _provenance()         # models are loaded on the CPU (attention only)
        infos = {e: enc.info() for e, enc in _encoders(args.encoders, torch.device("cpu")).items()}
        _write_json(args.out, {**env, "encoders": infos, "commit": commit, "utc": utc_now()})
        return
    if args.check == "tests":
        from wvad.diagnostics.common import code_version

        result = gate.junit_summary(args.junit) | {
            "commit": code_version(), "legacy_repo": os.environ.get("WVAD_LEGACY_REPO"),
            "utc": utc_now()}
        merge_record(args.record, "x0_tests", result)
        print(json.dumps(result))
        return
    if args.check == "norm":
        from wvad.extract.preprocess import normalization_mismatches

        result = {"x0_normalization": normalization_mismatches(device), "utc": utc_now()}
        merge_record(args.record, "x0_normalization", result)
        print(json.dumps(result))
        return
    paths, (manifest, split) = load_paths(), _data(args.dataset)

    def open_video(index: int) -> str:
        log_access(args.access_log, [{"utc": utc_now(), "command": f"gate {args.check}",
                                      "dataset": args.dataset, "index": index,
                                      "name": manifest.names[index],
                                      "path": rel_video_path(args.dataset, manifest, index),
                                      "purpose": args.check}])
        return os.fspath(video_path(paths, args.dataset, manifest, index))

    encoders = _encoders(args.encoders, device)
    commit, env = _provenance()
    if args.check == "x1":
        result = gate.run_x1(paths, args.dataset, manifest, split, encoders, device, open_video)
    else:
        x1 = json.loads(Path(args.x1).read_text(encoding="utf-8"))
        result = gate.run_x2(paths, args.dataset, manifest, x1, encoders, device, open_video,
                             args.batch, normalize=args.normalize)
    result |= {"commit": commit, "environment": env, "utc": utc_now()}
    _write_json(args.out, result)
    merge_record(args.record, f"{args.check}_{args.dataset}", _summaries(result))
    print(json.dumps({e: r["summary"] for e, r in result["encoders"].items()}))


def cmd_bench(args) -> None:
    from wvad.data.catalog import DATASETS
    from wvad.extract import dense
    from wvad.extract.encoders import tf32_settings
    from wvad.paths import load_paths

    paths, device = load_paths(), _device()
    if Path(args.out_root).resolve() == Path(paths.feature_root).resolve():
        raise SystemExit("--out_root must be a scratch folder, not the feature_root")
    encoders = _encoders(args.encoders, device)
    commit, env = _provenance()
    runs, stage = [], {"stage1_hours": 0, "stage2_hours": 0}
    for ds in DATASETS:
        manifest, split = _data(ds)
        layout = dense.Layout.from_manifest(ds, manifest)
        stage["stage1_hours"] += int(layout.counts[sorted(split.val_indices)].sum())
        stage["stage2_hours"] += int(layout.counts[sorted(split.train_indices)].sum())
        sample = dense.bench_sample(split)
        dirs = {e: Path(args.out_root) / f"{ds}_{e}_d16" for e in encoders}
        runs.append({"dataset": ds, "videos": sample, **dense.run_extraction(
            dirs, layout.subset(sample), dense.tasks_for(paths, ds, manifest, sample, True),
            encoders, device, "autocast", args.batch, "sample", args.access_log, "bench",
            args.workers, args.chunk, environment=env, commit=commit,
            normalize=args.normalize)})
    total = {"snippets": sum(r["snippets"] for r in runs),
             "seconds": sum(r["seconds"] for r in runs)}
    projection = dense.projected_hours(total, stage)
    projection |= {f"{k}_over_{STAGE_LIMIT_HOURS}h": (v is None or v > STAGE_LIMIT_HOURS)
                   for k, v in list(projection.items()) if k.endswith("_hours")}
    result = {"runs": runs, "total": total, "projection": projection, "stage_snippets": stage,
              "limit_hours": STAGE_LIMIT_HOURS, "batch": args.batch, "workers": args.workers,
              "chunk": args.chunk, "tf32": tf32_settings()}
    _write_json(args.out, result)
    merge_record(args.record, "throughput", {k: v for k, v in result.items() if k != "runs"})
    print(json.dumps(result["projection"]))


HEADER_FIELDS = ("dataset", "index", "name", "path", "size", "sha256", "header_frames",
                 "manifest_frames", "delta", "fps", "utc")


def cmd_header(args) -> None:
    from wvad.extract.dense import log_access, rel_video_path, utc_now, video_path
    from wvad.extract.video import read_header
    from wvad.paths import load_paths
    from wvad.prep.manifests import sha256_file

    paths, (manifest, split) = load_paths(), _data(args.dataset)
    indices = (sorted(split.train_indices) if args.subset == "trainpart"
               else list(range(len(manifest))))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER_FIELDS, lineterminator="\n")
        w.writeheader()
        for i in indices:
            rel = rel_video_path(args.dataset, manifest, i)
            log_access(args.access_log, [{"utc": utc_now(), "command": "header",
                                          "dataset": args.dataset, "index": i,
                                          "name": manifest.names[i], "path": rel,
                                          "purpose": "sha256" if args.sha_only else "header"}])
            path = video_path(paths, args.dataset, manifest, i)
            row = {"dataset": args.dataset, "index": i, "name": manifest.names[i],
                   "path": rel, "size": path.stat().st_size, "sha256": sha256_file(path),
                   "manifest_frames": manifest.total_frames[i], "utc": utc_now()}
            if not args.sha_only:
                frames, fps = read_header(path)
                row |= {"header_frames": frames, "delta": frames - manifest.total_frames[i],
                        "fps": fps}
            w.writerow(row)
    print(f"{len(indices)} videos -> {out}")


def cmd_dense(args) -> None:
    from wvad.extract import dense
    from wvad.paths import load_paths

    paths, (manifest, split), device = load_paths(), _data(args.dataset), _device()
    layout = dense.Layout.from_manifest(args.dataset, manifest)
    indices = dense.phase_videos(split, args.phase)
    encoders = _encoders(args.encoders, device)
    commit, env = _provenance()
    result = dense.run_extraction(
        dense.store_dirs(paths, args.dataset, args.encoders), layout,
        dense.tasks_for(paths, args.dataset, manifest, indices), encoders, device, "autocast",
        args.batch, args.phase, args.access_log, "dense", args.workers, args.chunk,
        environment=env, commit=commit, normalize=args.normalize)
    print(json.dumps({k: result[k] for k in ("phase", "videos_requested", "videos_pending",
                                             "videos_finished", "snippets", "seconds",
                                             "errors", "worker_failures")}))
    if result["errors"] or result["worker_failures"]:
        raise SystemExit(f"{len(result['errors'])} videos failed, worker failures "
                         f"{result['worker_failures']} (left undone; see meta.json)")


def cmd_derive(args) -> None:
    from wvad.data.catalog import dense_dir
    from wvad.diagnostics.common import code_version
    from wvad.extract import dense, derive
    from wvad.paths import load_paths

    paths, (manifest, split) = load_paths(), _data(args.dataset)
    layout = dense.Layout.from_manifest(args.dataset, manifest)
    commit = code_version()
    for e in args.encoders:
        metas = derive.derive_stage(dense_dir(paths, args.dataset, e),
                                    paths.feature_dir(f"{args.dataset}_{e}"), layout, split,
                                    args.dataset, e, args.stage, commit)
        print(e, {v: m["filled"] for v, m in metas.items()})


def _csv_rows(files) -> list[dict]:
    rows = []
    for f in files or []:
        with open(f, newline="", encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def cmd_check(args) -> None:
    from wvad.data.catalog import dense_dir
    from wvad.extract import dense, derive, gate
    from wvad.paths import load_paths

    if args.check == "x3a":
        result = gate.check_x3a(_csv_rows(args.headers))
    elif args.check == "x6":
        allowed = {}
        for ds in ("ucf", "xdv"):
            allowed[ds] = dense.manifest_video_paths(ds, _data(ds)[0])
        server = _csv_rows(args.server)
        result = gate.check_x6(server, _csv_rows(args.local), _csv_rows(args.access_log),
                               allowed)
        if args.recheck:
            now = {(r["dataset"], r["path"]): r["sha256"] for r in _csv_rows(args.recheck)}
            was = {(r["dataset"], r["path"]): r["sha256"] for r in server}
            changed = [list(k) for k, v in now.items() if was.get(k) != v]
            result["recheck"] = {"files": len(now), "changed": changed}
            if changed:
                result["status"] = gate.FAIL
    else:
        paths, (manifest, split) = load_paths(), _data(args.dataset)
        layout = dense.Layout.from_manifest(args.dataset, manifest)
        result = {}
        for e in args.encoders:
            folder = dense_dir(paths, args.dataset, e)
            if args.check == "x3":
                ok = {int(r["index"]): abs(int(r["delta"])) <= gate.X3A_DELTA
                      for r in _csv_rows(args.headers) if r["dataset"] == args.dataset}
                result[e] = gate.check_x3(folder, manifest,
                                          dense.phase_videos(split, args.phase), ok)
            elif args.check == "x4":
                env = json.loads(Path(args.env).read_text(encoding="utf-8"))
                result[e] = gate.check_x4(folder, manifest, dense.phase_videos(split, args.phase),
                                          e, args.commit, env)
            else:
                variants = list(derive.STAGE_VARIANTS[args.stage][args.dataset])
                if args.stage == 2 and args.dataset == "ucf":
                    variants += ["d16r1", "d16r2"]
                result[e] = gate.check_x5(folder, layout, split, args.dataset, e, variants,
                                          args.stage,
                                          legacy_dir=paths.feature_dir(f"{args.dataset}_{e}"))
    key = args.check + ("" if args.check in ("x3a", "x6") else
                        f"_{args.dataset}_{args.phase or f'stage{args.stage}'}")
    _write_json(args.out, result)
    merge_record(args.record, key, result)
    print(json.dumps(result)[:2000])


# --- parser ---------------------------------------------------------------------------------

def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m wvad.extract")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, dataset=True):
        if dataset:
            p.add_argument("--dataset", required=True, choices=("ucf", "xdv"))
        p.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))

    g = sub.add_parser("gate", help="(x0) unit-test and normalisation records, (x1), (x2), "
                                    "environment")
    g.add_argument("check", choices=("env", "tests", "norm", "x1", "x2"))
    g.add_argument("--junit", help="tests: pytest --junitxml output of the (x0) unit tests")
    g.add_argument("--dataset", choices=("ucf", "xdv"))
    g.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    g.add_argument("--access_log")
    g.add_argument("--x1", help="(x2): the (x1) result of the same dataset")
    g.add_argument("--batch", type=_batch, default=_batch(PRODUCTION_BATCH))
    g.add_argument("--normalize", choices=NORMALIZE, default="lut",
                   help="(x2): lut (production) or numpy (the CPU fallback)")
    g.add_argument("--out")
    g.add_argument("--record")
    b = sub.add_parser("bench", help="throughput on the fixed validation sample")
    common(b, dataset=False)
    b.add_argument("--out_root", required=True)
    b.add_argument("--access_log", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--record", help="gate record JSON to merge the summary into")
    b.add_argument("--workers", type=int, default=32)
    b.add_argument("--chunk", type=int, default=32)
    b.add_argument("--batch", type=_batch, default=_batch(PRODUCTION_BATCH))
    b.add_argument("--normalize", choices=NORMALIZE, default="lut")
    h = sub.add_parser("header", help="(x3a) header frame counts and (x6) sha256")
    h.add_argument("--dataset", required=True, choices=("ucf", "xdv"))
    h.add_argument("--subset", choices=("all", "trainpart"), default="all")
    h.add_argument("--sha_only", action="store_true")
    h.add_argument("--access_log", required=True)
    h.add_argument("--out", required=True)
    d = sub.add_parser("dense", help="production extraction of a phase")
    common(d)
    d.add_argument("--phase", required=True, choices=("val", "trainpart"))
    d.add_argument("--access_log", required=True)
    d.add_argument("--workers", type=int, default=32)
    d.add_argument("--chunk", type=int, default=32)
    d.add_argument("--batch", type=_batch, default=_batch(PRODUCTION_BATCH))
    d.add_argument("--normalize", choices=NORMALIZE, default="lut",
                   help="lut: table lookup on the GPU (production); numpy: the CPU fallback")
    r = sub.add_parser("derive", help="32-unit caches of a stage")
    common(r)
    r.add_argument("--stage", type=int, required=True, choices=(1, 2))
    c = sub.add_parser("check", help="(x3a), (x3), (x4), (x5), (x6)")
    c.add_argument("check", choices=("x3a", "x3", "x4", "x5", "x6"))
    c.add_argument("--dataset", choices=("ucf", "xdv"))
    c.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    c.add_argument("--phase", choices=("val", "trainpart"))
    c.add_argument("--stage", type=int, choices=(1, 2))
    c.add_argument("--headers", nargs="+", help="header CSVs (x3a, x3)")
    c.add_argument("--server", nargs="+", help="(x6) server header CSVs")
    c.add_argument("--local", nargs="+", help="(x6) local sha256 CSVs")
    c.add_argument("--access_log", nargs="+", help="(x6) access logs")
    c.add_argument("--recheck", nargs="+", help="(x6) stage-2 server sha256 CSVs")
    c.add_argument("--env", help="(x4) env_extract.json")
    c.add_argument("--commit", help="(x4) the frozen extraction commit")
    c.add_argument("--out", required=True)
    c.add_argument("--record")
    args = ap.parse_args(argv)
    _validate(ap, args)
    return args


def _validate(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    need = {("gate", "x1"): ("dataset", "access_log", "out"),
            ("gate", "x2"): ("dataset", "access_log", "out", "x1"),
            ("gate", "env"): ("out",), ("gate", "tests"): ("junit", "record"),
            ("check", "x3a"): ("headers",), ("check", "x6"): ("server", "local", "access_log"),
            ("check", "x3"): ("dataset", "phase", "headers"),
            ("check", "x4"): ("dataset", "phase", "env", "commit"),
            ("check", "x5"): ("dataset", "stage")}
    for name in need.get((args.command, getattr(args, "check", None)), ()):
        if getattr(args, name, None) in (None, []):
            ap.error(f"{args.command} {args.check} needs --{name}")


def main(argv=None) -> None:
    args = parse(argv)
    {"gate": cmd_gate, "bench": cmd_bench, "header": cmd_header, "dense": cmd_dense,
     "derive": cmd_derive, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    main()
