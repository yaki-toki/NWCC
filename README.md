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
git rev-parse HEAD | ssh <server> 'cat > ~/frozen-vlm-wvad/DEPLOYED_COMMIT'   # commit stamp for results made there
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

### Stage C — E003 training-based diagnostics ([plan](experiments/E003_training_diagnostics/PLAN.md))
Dev-seed training with recorded validation and pseudo-label scores (server GPU), then analysis of the recorded scores. Commands are added as each step (C0b cross-encoder alignment, C-hook, runs, C1 final, C3–C5) is implemented.

**C0b** (local CPU, training caches only; must pass before anything else):
```bash
python -m wvad.diagnostics.c0b     # results/c0b_cross_encoder.json
# fallback when a cell fails (plan change log): per-video fingerprint by local re-extraction
python tests/checks/e003_xd_fingerprint.py --encoder xcl --legacy_repo <previous project root>
python tests/checks/e003_xd_fingerprint.py --encoder siglip --legacy_repo <previous project root>
```

**C-hook** (server). `--record` must leave training unchanged: the seed-42 E001 chains are re-run with it into a separate folder and compared with E001, then the recorded scores are checked for consistency:
```bash
CH=~/wvad_runs/e003_chook && mkdir -p $CH
for spec in "ucf xcb_xcl" "xdv xcb_siglip"; do
  set -- $spec; d=$CH/${1}_${2}_s42
  setsid nohup bash -c "
    python -m wvad.train stage1 --dataset $1 --pair $2 --seed 42 --out_dir $d/stage1 --record &&
    { python -m wvad.train stage2 --dataset $1 --pair $2 --seed 42 --init $d/stage1/ckpt_best_val.pt \
        --out_dir $d/base --lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record &
      python -m wvad.train stage2 --dataset $1 --pair $2 --seed 42 --init $d/stage1/ckpt_best_val.pt \
        --out_dir $d/full --record &
      wait; }" > $d.log 2>&1 < /dev/null &
done
ls $CH/*/{base,full}/record_pseudo_final.npz | wc -l          # 4 when done
python tests/equivalence/check_chook.py --e001_runs ~/wvad_runs/e001 --chook_runs $CH   # chook_identity.json
```

**Dev-seed runs** (server, after C-hook passes): the same chain for seeds 0, 1, 2 × both datasets × both pairs, with `--record`, into `RUNS=~/wvad_runs/e003`:
```bash
RUNS=~/wvad_runs/e003 && mkdir -p $RUNS
for spec in "ucf xcb_xcl" "ucf xcb_siglip" "xdv xcb_xcl" "xdv xcb_siglip"; do
  set -- $spec
  for seed in 0 1 2; do
    d=$RUNS/${1}_${2}_s${seed}
    setsid nohup bash -c "
      python -m wvad.train stage1 --dataset $1 --pair $2 --seed $seed --out_dir $d/stage1 --record &&
      { python -m wvad.train stage2 --dataset $1 --pair $2 --seed $seed --init $d/stage1/ckpt_best_val.pt \
          --out_dir $d/base --lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record &
        python -m wvad.train stage2 --dataset $1 --pair $2 --seed $seed --init $d/stage1/ckpt_best_val.pt \
          --out_dir $d/full --record &
        wait; }" > $d.log 2>&1 < /dev/null &
  done
done
ls $RUNS/*/{base,full}/record_pseudo_final.npz | wc -l        # 24 when done
```

**Analysis.** C3 needs one more server pass (autocast re-scoring of the selected checkpoints); then copy the runs back and analyse locally (CPU; C5 takes about 35 minutes):
```bash
python -m wvad.diagnostics.c3 rescore --runs ~/wvad_runs/e003              # server GPU
ssh <server> 'tar -cf - -C ~/wvad_runs e003' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.runs --runs <local runs root>/e003      # runs_status.csv
python -m wvad.diagnostics.c1_final --runs <local runs root>/e003  # c1_final.json
python -m wvad.diagnostics.c3 analyze --runs <local runs root>/e003
python -m wvad.diagnostics.c4 --runs <local runs root>/e003
python -m wvad.diagnostics.c5 --runs <local runs root>/e003
```

### Stage C — E004 temporal-context linear probe ([plan](experiments/E004_temporal_context_probe/PLAN.md))
Local CPU only (UCF training split; UCA proxy labels for diagnosis). About 360-460 logistic-regression fits, 4 in parallel x 8 BLAS threads (3-6 h); every fit is saved under the output folder, so an interrupted run resumes with the same command (refused if the commit or config changed). Commit before running.
```bash
python -m wvad.diagnostics.c6 --out <local runs root>/e004     # results/c6_probe.csv, c6_cv.csv, c6_summary.json
```

### Stage D — E010 development-metric calibration ([plan](experiments/E010_dev_metric_calibration/PLAN.md))
No training: cross-fitted validation video AUC on the recorded E003 runs fixes the stage-D noise level and adoption rule. Runs on CPU from a local copy of `~/wvad_runs/e003` (only `record_val_scores.npz`, `selection.json` and `ckpt_last.pt` of each `base`/`full` run are read); writes `results/e010_runs.csv` and `results/e010_summary.json`.
```bash
python -m wvad.diagnostics.e010 --runs <local copy of ~/wvad_runs/e003>
```

### Stage D — E011 D1: BCE on every normal-video segment ([plan](experiments/E011_dense_normal_negatives/PLAN.md))
Candidates `d1a` (`--lambda_normal_seg 0.1`) and `d1b` (`1.0`) in both stages, against the E003 `base` runs (reused after the no-change gate). Deploy the committed code first (server conventions).

**No-change gate** (server): the default weight must reproduce two E003 base chains bit for bit; writes `results/e011_identity.json`.
```bash
G=~/wvad_runs/e011_gate && mkdir -p $G
for spec in "ucf xcb_xcl" "xdv xcb_siglip"; do
  set -- $spec; d=$G/${1}_${2}_s0
  setsid nohup bash -c "
    python -m wvad.train stage1 --dataset $1 --pair $2 --seed 0 --out_dir $d/stage1 --record &&
    python -m wvad.train stage2 --dataset $1 --pair $2 --seed 0 --init $d/stage1/ckpt_best_val.pt \
        --out_dir $d/base --lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record" \
    > $d.log 2>&1 < /dev/null &
done
ls $G/*/base/selection.json | wc -l                              # 2 when done, and then
pgrep -fc "[w]vad.train"                                        # 0 (records are written after selection.json)
# after both chains finish:
python tests/equivalence/check_e011_gate.py --e003_runs ~/wvad_runs/e003 --gate_runs $G
# if it fails, the plan's cause check re-runs the same two chains with the E003 deploy
# commit c059790 (deployed to a separate folder) into another gate folder and compares again
```

**Candidate runs** (server, only after the gate passes): 24 chains, all at once.
```bash
RUNS=~/wvad_runs/e011 && mkdir -p $RUNS
for spec in "ucf xcb_xcl" "ucf xcb_siglip" "xdv xcb_xcl" "xdv xcb_siglip"; do
  set -- $spec
  for seed in 0 1 2; do
    for arm in "d1a 0.1" "d1b 1.0"; do
      set -- $spec $arm; d=$RUNS/${1}_${2}_s${seed}
      mkdir -p $d
      setsid nohup bash -c "
        python -m wvad.train stage1 --dataset $1 --pair $2 --seed $seed --out_dir $d/${3}_stage1 \
            --lambda_normal_seg $4 --record &&
        python -m wvad.train stage2 --dataset $1 --pair $2 --seed $seed \
            --init $d/${3}_stage1/ckpt_best_val.pt --out_dir $d/$3 --lambda_dense 0 \
            --lambda_pseudo 0 --lambda_neg_pseudo 0 --lambda_normal_seg $4 --record" \
        > $d.$3.log 2>&1 < /dev/null &
    done
  done
done
ls $RUNS/*/d1?/selection.json | wc -l                            # 24 when done, and then
pgrep -fc "[w]vad.train"                                        # 0 before copying the runs
# fewer: check the chain logs (a stage 1 with val_video_auc=nan never starts stage 2;
# that is a numerical failure, not an infrastructure one, and is not re-run)
```

**Analysis** (local CPU): copy the runs and the gate result back, then
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e011' | tar -xf - -C <local runs root>
scp <server>:frozen-vlm-wvad/experiments/E011_dense_normal_negatives/results/e011_identity.json \
    experiments/E011_dense_normal_negatives/results/
python -m wvad.diagnostics.e011 --base_runs <local runs root>/e003 --runs <local runs root>/e011
```
The analysis refuses to reuse the E003 baseline unless `e011_identity.json` reports `all_ok`. If the gate failed for environment reasons only, re-run the 12 baseline chains with the candidates' commit (the E003 dev-seed commands without the `full` cell: folders `stage1` and `base` only) and pass them with `--fresh_baseline <root>` instead of `--base_runs`.

### Stage D — E012 training budget of the base pipeline ([plan](experiments/E012_training_budget/PLAN.md))
2 x 2 nested budgets against the E003 `base` runs (stage 1 20 / stage 2 30): `s1` (40, 30), `s2` (20, 60, started from the E003 stage-1 checkpoint) and `s12` (40, 60); `s1` and `s12` share one 40-epoch stage 1. The training code is unchanged (`--epochs`, `--epochs_per_iter`); deploy the committed code first (server conventions) and check GPU and host memory (up to 36 processes).
```bash
RUNS=~/wvad_runs/e012 E3=~/wvad_runs/e003 && mkdir -p $RUNS
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
for spec in "ucf xcb_xcl" "ucf xcb_siglip" "xdv xcb_xcl" "xdv xcb_siglip"; do
  set -- $spec
  for seed in 0 1 2; do
    c=${1}_${2}_s${seed}; d=$RUNS/$c; mkdir -p $d
    T="--dataset $1 --pair $2 --seed $seed"
    setsid nohup bash -c "
      python -m wvad.train stage2 $T --init $E3/$c/stage1/ckpt_best_val.pt --out_dir $d/s2 \
          --epochs_per_iter 20 $Z > $d.s2.log 2>&1 &
      python -m wvad.train stage1 $T --epochs 40 --out_dir $d/s40_stage1 --record > $d.s40.log 2>&1 &&
      { python -m wvad.train stage2 $T --init $d/s40_stage1/ckpt_best_val.pt --out_dir $d/s1 \
            $Z > $d.s1.log 2>&1 &
        python -m wvad.train stage2 $T --init $d/s40_stage1/ckpt_best_val.pt --out_dir $d/s12 \
            --epochs_per_iter 20 $Z > $d.s12.log 2>&1 & wait; }
      wait" < /dev/null > /dev/null 2>&1 &
  done
done
ls $RUNS/*/{s1,s2,s12}/selection.json | wc -l                    # 36 when done, and then
pgrep -fc "[w]vad.train"                                        # 0 before copying
```
Before launching, `ls ~/wvad_runs/e003/*/stage1/ckpt_best_val.pt | wc -l` must print 12. Fewer than 36 at the end: check the chain logs. A numerical failure (non-finite losses or scores) is not re-run. An infrastructure failure is re-run once; re-running `s40_stage1` also means re-running `s1` and `s12`.
**Analysis** (local CPU): the determinism gates (a)-(e) are checked first and written to `results/e012_gate.json`; the analysis stops if a gate fails (PLAN.md fallbacks).
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e012' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e012 --base_runs <local runs root>/e003 --runs <local runs root>/e012
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
