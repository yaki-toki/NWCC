# frozen-vlm-wvad

Weakly supervised video anomaly detection (WVAD) on frozen vision-language encoders. The method is named **SEAR-D**; its components, and the expansion of the acronym, are fixed only after the confirmation experiment. SEAR-D is a redesign of the earlier SEAR method; results of the earlier method are not results of SEAR-D.

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
experiments/    one folder per experiment: PLAN.md, config, results/, REPORT.md
paper/          LaTeX source (written after the confirmation experiment)
src/wvad/       library code and module entry points (python -m wvad....)
docs/, tests/   local only, git-ignored: planning notes, unit tests, equivalence checks
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

# on the server, in ~/frozen-vlm-wvad (once: create configs/paths.yaml), with the
# Python environment active (non-interactive shells: put its bin/ first on PATH)
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

**B2 evaluator.** Locally, with the previous project available (no dataset is read):
```bash
WVAD_LEGACY_REPO=<previous project root> pytest tests/test_eval_metrics.py tests/equivalence/test_b2_legacy_eval.py
```

**B3 model and losses.** Locally (CPU, synthetic inputs), then on the server with earlier checkpoints (validation videos only):
```bash
WVAD_LEGACY_REPO=<previous project root> pytest tests/equivalence/test_b3_legacy_model.py tests/equivalence/test_b4_legacy_loop.py
python tests/equivalence/check_b3_checkpoints.py --legacy_repo <previous project root> \
    --p000_runs <previous project root>/outputs        # writes results/b3_checkpoint_equivalence.json
```
Copy result files back with `scp` as in B1, here and in B4.

**B4 trainer, reproduction gate.** On the server; runs are written outside the repository. Each (dataset, pair, seed) runs stage 1, then the baseline cell (`base`: no dense ranking, no pseudo-labels) and the earlier method's cell (`full`: defaults) of stage 2:
```bash
RUNS=~/wvad_runs/e001 && mkdir -p $RUNS
for spec in "ucf xcb_xcl" "xdv xcb_siglip"; do
  set -- $spec
  for seed in 42 123; do
    d=$RUNS/${1}_${2}_s${seed}
    setsid nohup bash -c "
      python -m wvad.train stage1 --dataset $1 --pair $2 --seed $seed --out_dir $d/stage1 &&
      { python -m wvad.train stage2 --dataset $1 --pair $2 --seed $seed --init $d/stage1/ckpt_best_val.pt \
          --out_dir $d/base --lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 &
        python -m wvad.train stage2 --dataset $1 --pair $2 --seed $seed --init $d/stage1/ckpt_best_val.pt \
          --out_dir $d/full &
        wait; }" > $d.log 2>&1 < /dev/null &
  done
done
# after all chains finish (each chain ends with two selection.json files):
ls $RUNS/*/{base,full}/selection.json | wc -l                   # 8
python tests/equivalence/check_b4_reproduction.py --runs $RUNS \
    --p000_runs <previous project root>/outputs   # writes results/b4_reproduction.csv, b4_checkpoint_identity.json
```

### Stage C — E002 data and geometry diagnostics ([plan](experiments/E002_diagnostics/PLAN.md))
No training; runs locally on CPU (raw UCF-Crime videos are needed only for the manifest and the fingerprint check). Results go to `experiments/E002_diagnostics/results/`.

**C0 inputs.** Training manifests and UCA proxy intervals (already committed under `data/`; rebuild only if the sources change):
```bash
pip install -e ".[prep]"                                   # OpenCV, for video headers
python -m wvad.prep.manifests ucf --previous <previous project>/scripts/rebuttal_phase2/ucf_train_manifest.json \
    --dataset_root <UCF-Crime root> --out data/manifests/ucf_train.json
python -m wvad.prep.manifests xdv --previous <previous project>/scripts/rebuttal_phase2/xdv_train_manifest.json \
    --out data/manifests/xdv_train.json
python -m wvad.diagnostics.uca --out data/annotations/ucf_train_uca_proxy.json   # reads paths.yaml annotations.uca
```

**C0 gates.** The fingerprint check (G4, G5; local only) re-encodes 30 training videos on CPU, then the summary adds G1–G3:
```bash
ssh <server> sha256sum <server feature_root>/features_xclip_base/mil/train/data.pt    # for G5
python tests/checks/c0_fingerprint.py --legacy_repo <previous project root> --server_sha256 <hash>
python -m wvad.diagnostics.c0 --ucf_train_list <UCF-Crime root>/Anomaly_Train.txt   # c0_alignment.json
```

**C1, C2** (only after all C0 gates pass):
```bash
python -m wvad.diagnostics.c1      # c1_resolution.csv, c1_summary.json
python -m wvad.diagnostics.c2      # c2_coverage.csv, c2_xd_geometry.csv, c2_summary.json
```

## Local checks (not tracked)
```bash
pytest            # unit tests in tests/
ruff check .
```

## Data
Features are pre-extracted per video (32 segments × encoder dimension) for UCF-Crime and XD-Violence with X-CLIP (base, large) and SigLIP-so400m encoders. They are not distributed with this repository; `configs/paths.yaml` maps logical names to local directories. See [`data/README.md`](data/README.md).

## Roadmap
Development proceeds in stages: A repository, B baseline port (E001), C diagnostics, D training redesign, E architecture redesign, F pre-registered confirmation experiment and paper. Each stage has a gate that must pass before the next starts.

## License
MIT (see [LICENSE](LICENSE)). Third-party annotations keep their own licenses.
