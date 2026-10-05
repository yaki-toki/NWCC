"""Command line: ``python -m wvad.train stage1 ...`` / ``python -m wvad.train stage2 ...``.

Defaults are the baseline settings; every run is listed in the README with its flags.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from wvad.data import DATASETS, ENCODER_PAIRS
from wvad.train.loop import LossWeights
from wvad.train.stages import (
    Stage1Config,
    Stage2Config,
    run_stage1,
    run_stage2,
    stage2_default_weights,
)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--pair", required=True, choices=list(ENCODER_PAIRS))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--record", action="store_true",
                   help="save validation (and stage-2 pseudo-label) scores; training unchanged")
    p.add_argument("--lambda_normal_seg", type=float,
                   help="BCE weight on every normal-video segment (stage D1; default 0 = off)")


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m wvad.train")
    sub = ap.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("stage1", help="MIL warm-up")
    _common(s1)
    s1.add_argument("--epochs", type=int)
    s2 = sub.add_parser("stage2", help="pseudo-label self-training from a stage-1 checkpoint")
    _common(s2)
    s2.add_argument("--init", required=True, help="stage-1 ckpt_best_val.pt")
    s2.add_argument("--iterations", type=int)
    s2.add_argument("--epochs_per_iter", type=int)
    s2.add_argument("--no_curriculum", action="store_true")
    s2.add_argument("--lambda_dense", type=float, help="dense ranking weight (default 0.05)")
    s2.add_argument("--lambda_pseudo", type=float, help="pseudo-positive weight (default 1.0)")
    s2.add_argument("--lambda_neg_pseudo", type=float,
                    help="pseudo-negative weight (default 0.3)")
    return ap.parse_args(argv)


def _given(args: argparse.Namespace, *names: str) -> dict:
    return {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}


def main(argv=None) -> None:
    args = parse(argv)
    base = {"dataset": args.dataset, "pair": args.pair, "seed": args.seed,
            "out_dir": args.out_dir, "record": args.record, **_given(args, "lr", "batch_size")}
    normal = {} if args.lambda_normal_seg is None else {"normal_seg": args.lambda_normal_seg}
    if args.stage == "stage1":
        weights = replace(LossWeights(), **normal)
        result = run_stage1(Stage1Config(**base, weights=weights, **_given(args, "epochs")))
    else:
        overrides = {k: v for k, v in (("dense", args.lambda_dense),
                                        ("pseudo", args.lambda_pseudo),
                                        ("neg_pseudo", args.lambda_neg_pseudo)) if v is not None}
        weights = replace(stage2_default_weights(), **overrides, **normal)
        cfg = Stage2Config(**base, init=args.init, curriculum=not args.no_curriculum,
                           weights=weights, **_given(args, "iterations", "epochs_per_iter"))
        result = run_stage2(cfg)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
