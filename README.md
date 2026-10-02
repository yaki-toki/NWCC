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
- **Commands live in this README.** There are no wrapper scripts; every run is a command listed below, updated as stages progress.

## Layout
```
configs/        paths.example.yaml (copy to paths.yaml, git-ignored), experiment configs
data/           validation splits (small, tracked); features live outside the repository
docs/           ROADMAP.md, DIAGNOSIS.md, decisions/ (design records)
experiments/    one folder per experiment: PLAN.md, config, results/, REPORT.md
paper/          LaTeX source (written after the confirmation experiment)
src/wvad/       library code and module entry points (python -m wvad....)
tests/          local only, git-ignored: unit tests and equivalence checks
```

## Setup
```bash
pip install torch                                  # CUDA build of your choice
pip install -e ".[dev]"
cp configs/paths.example.yaml configs/paths.yaml   # point it at the feature caches
```

## Run procedure
Commands are run from the repository root. Each stage adds its commands here.

### Server conventions
```bash
# deploy the committed code (non-destructive: configs/paths.yaml and outputs on the
# server are kept), then the local-only tests/ folder
git archive --format=tar HEAD | ssh <server> 'mkdir -p ~/frozen-vlm-wvad && tar -x -C ~/frozen-vlm-wvad'
tar -cf - tests | ssh <server> 'tar -x -C ~/frozen-vlm-wvad'

# on the server, in ~/frozen-vlm-wvad (once: create configs/paths.yaml)
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE
# long runs: setsid nohup <command> > <log outside the repo> 2>&1 < /dev/null &
```

### Stage B — E001 baseline port ([plan](experiments/E001_baseline_port/PLAN.md))
**B1 data layer.** On the server (reads training caches only):
```bash
python tests/equivalence/check_b1_data.py --legacy_repo <previous project root>
```
This writes `experiments/E001_baseline_port/results/b1_data_equivalence.json`. Copy that file back and commit it:
```bash
scp <server>:frozen-vlm-wvad/experiments/E001_baseline_port/results/b1_data_equivalence.json \
    experiments/E001_baseline_port/results/
```

## Local checks (not tracked)
```bash
pytest            # unit tests in tests/
ruff check .
```

## Data
Features are pre-extracted per video (32 segments × encoder dimension) for UCF-Crime and XD-Violence with X-CLIP (base, large) and SigLIP-so400m encoders. They are not distributed with this repository; `configs/paths.yaml` maps logical names to local directories. See [`data/README.md`](data/README.md).

## Roadmap
[`docs/ROADMAP.md`](docs/ROADMAP.md) (stages A–F with gates). Planning documents are written in Korean; code, comments, this README and the paper are in English.

## License
MIT (see [LICENSE](LICENSE)). Third-party annotations keep their own licenses.
