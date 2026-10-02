# frozen-vlm-wvad

Weakly supervised video anomaly detection (WVAD) on frozen vision-language encoders.

**Status: redesign in progress.** This repository restarts the method from first principles after a pre-registered, validation-only ablation of the previous method returned **NO-GO**. In that ablation:
- the dense ranking term had no measurable effect;
- pseudo-label self-training helped only on UCF-Crime;
- the auxiliary risk head did not improve detection.

See [`experiments/P000_prior_validation_ablation`](experiments/P000_prior_validation_ablation). No method claims are made until the confirmation experiment of the roadmap (stage F) passes.

## Principles
- **Sealed test sets.** During development, model, checkpoint, threshold and hyperparameter choices use only held-out *training* videos (validation split). Each test set is read once, in a pre-registered confirmation experiment.
- **One change at a time.** Every design change is compared with the current baseline under identical features, splits, budget and seeds.
- **Everything recorded.** Each experiment keeps its plan, configuration, small result tables (CSV/JSON) and verdict under `experiments/`. Negative results stay.

## Layout
```
configs/        paths.example.yaml (copy to paths.yaml, git-ignored), experiment configs
data/           validation splits and manifests (small, tracked); features live outside
docs/           ROADMAP.md, DIAGNOSIS.md, decisions/ (design records)
experiments/    one folder per experiment: PLAN.md, config, results/, REPORT.md
paper/          LaTeX source (written after the confirmation experiment)
scripts/        command-line entry points
src/wvad/       library code
tests/          pytest suite (CPU only)
```

## Setup
```bash
pip install torch            # CUDA build of your choice
pip install -e ".[dev]"
cp configs/paths.example.yaml configs/paths.yaml   # then point it at your feature caches
pytest && ruff check .
```

## Data
Features are pre-extracted per video (32 segments × encoder dimension) for UCF-Crime and XD-Violence with X-CLIP (base, large) and SigLIP-so400m encoders. They are not distributed with this repository; `configs/paths.yaml` maps logical names to local directories. See [`data/README.md`](data/README.md).

## Roadmap
[`docs/ROADMAP.md`](docs/ROADMAP.md) (stages A–F with gates). Planning documents are written in Korean; code, comments and the paper are in English.

## License
MIT (see [LICENSE](LICENSE)). Third-party annotations keep their own licenses.
