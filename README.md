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

### Stage D — E013 stage 2 from the stage-1 last checkpoint ([plan](experiments/E013_stage1_last_checkpoint/PLAN.md))
Stage 2 (30 epochs, base cell) started from `ckpt_last.pt` of the existing stage-1 runs: `l20` from the E003 20-epoch stage 1, `l40` from the E012 40-epoch stage 1; the per-pair arm Lmix (XCB+XCL from `l40`, XCB+SigLIP from `l20`) is composed in the analysis. The training code is unchanged; `--init` accepts any stage-1 checkpoint (its help text names `ckpt_best_val.pt`). Deploy the committed code first.
```bash
RUNS=~/wvad_runs/e013 E3=~/wvad_runs/e003 E12=~/wvad_runs/e012 && mkdir -p $RUNS
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
for spec in "ucf xcb_xcl" "ucf xcb_siglip" "xdv xcb_xcl" "xdv xcb_siglip"; do
  set -- $spec
  for seed in 0 1 2; do
    c=${1}_${2}_s${seed}; d=$RUNS/$c; mkdir -p $d
    T="--dataset $1 --pair $2 --seed $seed"
    setsid nohup python -m wvad.train stage2 $T --init $E3/$c/stage1/ckpt_last.pt \
        --out_dir $d/l20 $Z > $d.l20.log 2>&1 < /dev/null &
    setsid nohup python -m wvad.train stage2 $T --init $E12/$c/s40_stage1/ckpt_last.pt \
        --out_dir $d/l40 $Z > $d.l40.log 2>&1 < /dev/null &
  done
done
ls $RUNS/*/l?0/selection.json | wc -l                            # 24 when done, and then
pgrep -fc "[w]vad.train"                                        # 0 before copying
```
Before launching, `ls ~/wvad_runs/e003/*/stage1/ckpt_last.pt ~/wvad_runs/e012/*/s40_stage1/ckpt_last.pt | wc -l` must print 24. Fewer than 24 at the end: check the logs. A numerical failure is not re-run, and Lmix is failed if it contains a failed run. An infrastructure failure is re-run once.

**Analysis** (local CPU): gates (a)-(c) first (`results/e013_gate.json`, including whether each init path is right); stops if a gate fails.
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e013' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e013 --e003 <local runs root>/e003 --e012 <local runs root>/e012 \
    --runs <local runs root>/e013
```

### Stage D — E014 D3: self-training at two strengths ([plan](experiments/E014_self_training/PLAN.md))
Stage 2 with pseudo-label self-training (3 rounds x 10 epochs, curriculum) from the E003 stage-1 `ckpt_last.pt`, the same start as the E013 `l20` baseline: `st3` (pseudo weights 1.0 / 0.3) and `stw` (0.1 / 0.03). The baseline is the existing E013 `l20` runs (reused; the analysis cross-checks them against `e013_runs.csv`). The training code is unchanged: before deploying, the diff below must be empty. Deploy the committed code, check GPU and host memory, record the server environment, and launch one dataset per batch (12 processes; both arms of a chain in the same batch; PLAN.md change log).
```bash
git diff 8b46731b -- src/wvad/train src/wvad/losses src/wvad/models src/wvad/data src/wvad/eval \
    src/wvad/paths.py                                            # local, before deploying: empty
nvidia-smi --query-gpu=memory.used,memory.total --format=csv && free -g      # on the server
RUNS=~/wvad_runs/e014 E3=~/wvad_runs/e003 && mkdir -p $RUNS
python -c "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, \
    'cudnn': torch.backends.cudnn.version(), 'device': torch.cuda.get_device_name(0)}))" > $RUNS/env.json
ds=ucf                                   # first batch; when it is done, run the loop again with ds=xdv
for pair in xcb_xcl xcb_siglip; do
  for seed in 0 1 2; do
    c=${ds}_${pair}_s${seed}; d=$RUNS/$c; mkdir -p $d
    T="--dataset $ds --pair $pair --seed $seed --init $E3/$c/stage1/ckpt_last.pt --lambda_dense 0 --record"
    setsid nohup python -m wvad.train stage2 $T --lambda_pseudo 1.0 --lambda_neg_pseudo 0.3 \
        --out_dir $d/st3 > $d.st3.log 2>&1 < /dev/null &
    setsid nohup python -m wvad.train stage2 $T --lambda_pseudo 0.1 --lambda_neg_pseudo 0.03 \
        --out_dir $d/stw > $d.stw.log 2>&1 < /dev/null &
  done
done
ls $RUNS/*/st?/selection.json | wc -l                            # 12 per batch, 24 when done
pgrep -fc "[w]vad.train"                                        # 0 before the next batch / copying
```
Before launching, `ls ~/wvad_runs/e003/*/stage1/ckpt_last.pt | wc -l` must print 12 and `~/wvad_runs/e013/*/l20/record_pseudo.npz` must exist for all 12 chains; `env.json` must hold the torch and CUDA versions (an empty file means the command failed). A numerical failure is not re-run (that arm is failed); an infrastructure failure is re-run once. A run that stops before writing `ckpt_last.pt`: read its log; non-finite losses or scores make it a numerical failure (PLAN.md change log), anything else an infrastructure failure.

**Analysis** (local CPU): gates (a)-(c) first (`results/e014_gate.json`, with the server environment, whether each init path and MIL weight is right, and which round-1 arrays differ); stops if a gate fails or `env.json` is missing or empty. `results/e014_summary.json` is written in three steps: verdict and choice, then the descriptives, then the UCF UCA proxy AP (descriptive only; `--no_proxy` skips it).
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e014' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e014 --e003 <local runs root>/e003 --e013 <local runs root>/e013 \
    --runs <local runs root>/e014
```

### Stage D — E015 D2: video BCE on segment scores ([plan](experiments/E015_topk_mil/PLAN.md))
The video BCE uses the top segment probability (`kb1`, `--video_topk 1`) or the mean of the top-3 segment probabilities (`kb3`, `--video_topk 3`) instead of the attention-pooled logit, in both stages; each arm is retrained from stage 1 (20 epochs) -> `ckpt_last.pt` -> stage 2 (30 epochs, no pseudo labels). The baseline is the existing E013 `l20` runs. `--video_topk` defaults to 0 (attention pooling, the earlier code path).

**No-change gate** (server, before any candidate): two chains with the new code at the default; the XD chain gives `--video_topk 0` explicitly in both stages. Deploy the committed code first.
```bash
git diff 8b46731b -- src/wvad/models src/wvad/data src/wvad/eval src/wvad/paths.py   # local: empty
GATE=~/wvad_runs/e015_gate_$(head -c 8 DEPLOYED_COMMIT)    # one folder per deployed commit
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
mkdir "$GATE" && for spec in "ucf xcb_xcl" "xdv xcb_siglip --video_topk 0"; do   # launches only into a new folder
  set -- $spec; c=${1}_${2}_s0; T="--dataset $1 --pair $2 --seed 0 $3 $4"
  setsid nohup bash -c "
    python -m wvad.train stage1 $T --out_dir $GATE/$c/stage1 --record > $GATE/$c.s1.log 2>&1 &&
    python -m wvad.train stage2 $T --init $GATE/$c/stage1/ckpt_last.pt --out_dir $GATE/$c/l20 \
        $Z > $GATE/$c.l20.log 2>&1" < /dev/null > /dev/null 2>&1 &
done
ls $GATE/*/{stage1,l20}/selection.json | wc -l                  # 4 when done, and
pgrep -fc "wvad[.]train"                                        # 0 (records are written after selection.json), then
python tests/equivalence/check_e015_gate.py --e003_runs ~/wvad_runs/e003 \
    --e013_runs ~/wvad_runs/e013 --gate_runs $GATE              # must print all_ok=True
```
The checker also requires that the gate runs come from this deployment (new config key, files newer than `DEPLOYED_COMMIT`). Copy `experiments/E015_topk_mil/results/e015_identity.json` back from the server and commit it.
If the gate fails, check the cause before running candidates (plan): deploy commit `bb671a2` to a separate folder and run the same two chains there into another gate folder; its CLI has no `--video_topk`, so the XD chain runs without it. If those runs also differ from E003 / E013 the cause is the environment, otherwise the code.

**Candidates** (server, same deployed commit as the gate): check GPU and host memory first and use what is free (shared H100: memory only). With room for 24 processes (about 1 GB GPU and 2.5 GB host each, plus 2 GB / 8 GB spare) launch both datasets at once (`DS="ucf xdv"`); otherwise one dataset per batch (`DS=ucf`, then `DS=xdv`). Each dataset first records its environment and commit and checks them against the passed `e015_identity.json` (gate (f)); it is launched only if that check passes.
```bash
RUNS=~/wvad_runs/e015 && mkdir -p $RUNS
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv && free -g
DS="ucf xdv"                             # or DS=ucf, then DS=xdv when that batch is done
for ds in $DS; do
  python -c "from wvad.diagnostics.e015 import record_env; record_env('$RUNS/env_$ds.json')" || continue
  for pair in xcb_xcl xcb_siglip; do
    for seed in 0 1 2; do
      c=${ds}_${pair}_s${seed}; d=$RUNS/$c; mkdir -p $d
      for arm in kb1 kb3; do
        T="--dataset $ds --pair $pair --seed $seed --video_topk ${arm#kb}"
        setsid nohup bash -c "
          python -m wvad.train stage1 $T --out_dir $d/${arm}_stage1 --record > $d.${arm}_s1.log 2>&1 &&
          python -m wvad.train stage2 $T --init $d/${arm}_stage1/ckpt_last.pt --out_dir $d/$arm \
              $Z > $d.$arm.log 2>&1" < /dev/null > /dev/null 2>&1 &
      done
    done
  done
done
ls $RUNS/*/kb?/selection.json | wc -l                            # 12 per dataset, 24 when done, then
pgrep -fc "wvad[.]train"                                        # 0 before the next batch / copying
```
A numerical failure is not re-run (that arm is failed); an infrastructure failure is re-run once (stage 1 again means stage 2 again). An exception in the new code path is a code defect: fix it, pass the no-change gate again and re-run all 24 chains with that commit.

**Analysis** (local CPU): gates (a)-(f) first (`results/e015_gate.json`: epoch-0 scores, init paths, `video_topk` in the configs, the option is active and k is used, the attention branch is never updated, commit and environment as in the no-change gate); stops if a gate fails. The verdict uses Delta alone (`outcome_delta`); `results/e015_summary.json` is written in three steps: verdict and choice, then the descriptives, then the UCF UCA proxy AP (descriptive only; `--no_proxy` skips it).
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e015' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e015 --e003 <local runs root>/e003 --e013 <local runs root>/e013 \
    --runs <local runs root>/e015
```

### Stage D — E016 global offset on segment logits ([plan](experiments/E016_global_offset/PLAN.md))
The head adds a per-video offset c(v) to every segment logit, in both stages: `g1a` (`--global_mode mean`, c = gamma x mean segment logit, gamma from 0) or `g1b` (`--global_mode stats`, c linear in the mean/std of the pre-dropout context features, from 0). The video BCE keeps the attention-pooled offset-free logit. Each arm is retrained from stage 1 (20 epochs) -> `ckpt_last.pt` -> stage 2 (30 epochs, no pseudo labels). The baseline is the existing E013 `l20` runs. `--global_mode` defaults to none (the earlier head, no new parameters); stage 2 stops if its mode differs from the stage-1 checkpoint's. With `--record`, candidate runs also write `record_val_offset.npz` (c(v) and the offset-free scores of the same validation pass).

**No-change gate** (server, before any candidate): two chains with the new code at the default; the XD chain gives `--global_mode none` explicitly in both stages. Deploy the committed code first.
```bash
git diff 18cb944 -- src/wvad/data src/wvad/eval src/wvad/losses src/wvad/paths.py   # local: empty
GATE=~/wvad_runs/e016_gate_$(head -c 8 DEPLOYED_COMMIT)    # one folder per deployed commit
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
mkdir "$GATE" && for spec in "ucf xcb_xcl" "xdv xcb_siglip --global_mode none"; do   # launches only into a new folder
  set -- $spec; c=${1}_${2}_s0; T="--dataset $1 --pair $2 --seed 0 $3 $4"
  setsid nohup bash -c "
    python -m wvad.train stage1 $T --out_dir $GATE/$c/stage1 --record > $GATE/$c.s1.log 2>&1 &&
    python -m wvad.train stage2 $T --init $GATE/$c/stage1/ckpt_last.pt --out_dir $GATE/$c/l20 \
        $Z > $GATE/$c.l20.log 2>&1" < /dev/null > /dev/null 2>&1 &
done
ls $GATE/*/{stage1,l20}/selection.json | wc -l                  # 4 when done, and
pgrep -fc "wvad[.]train"                                        # 0 (records are written after selection.json), then
python tests/equivalence/check_e016_gate.py --e003_runs ~/wvad_runs/e003 \
    --e013_runs ~/wvad_runs/e013 --gate_runs $GATE              # must print all_ok=True
```
The checker also requires that the gate runs come from this deployment (new config key, files newer than `DEPLOYED_COMMIT`) and that the default writes no `record_val_offset.npz`. Copy `experiments/E016_global_offset/results/e016_identity.json` back from the server and commit it.
If the gate fails, check the cause before running candidates (plan): deploy commit `b9d4b95` to a separate folder and run the same two chains there into another gate folder; its CLI has no `--global_mode`, so the XD chain runs without it. Compare them from that folder with `tests/equivalence/check_e015_gate.py` (deployed with the tests as usual; same arguments; its new-code marker `video_topk` exists at `b9d4b95`), not with `check_e016_gate.py`: that checker requires the `global_mode` key and, run from the main tree, would overwrite `e016_identity.json`. If those runs also differ from E003 / E013 the cause is the environment, otherwise the code.

**Candidates** (server, same deployed commit as the gate): check GPU and host memory first and use what is free (shared H100: memory only). With room for 24 processes (about 1 GB GPU and 2.5 GB host each, plus 2 GB / 8 GB spare) launch both datasets at once (`DS="ucf xdv"`); otherwise one dataset per batch (`DS=ucf`, then `DS=xdv`). Each dataset first records its environment and commit and checks them against the passed `e016_identity.json` (gate (f)); it is launched only if that check passes.
```bash
RUNS=~/wvad_runs/e016 && mkdir -p $RUNS
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv && free -g
DS="ucf xdv"                             # or DS=ucf, then DS=xdv when that batch is done
for ds in $DS; do
  python -c "from wvad.diagnostics.e016 import record_env; record_env('$RUNS/env_$ds.json')" || continue
  for pair in xcb_xcl xcb_siglip; do
    for seed in 0 1 2; do
      c=${ds}_${pair}_s${seed}; d=$RUNS/$c; mkdir -p $d
      for arm in g1a:mean g1b:stats; do
        a=${arm%:*}; T="--dataset $ds --pair $pair --seed $seed --global_mode ${arm#*:}"
        setsid nohup bash -c "
          python -m wvad.train stage1 $T --out_dir $d/${a}_stage1 --record > $d.${a}_s1.log 2>&1 &&
          python -m wvad.train stage2 $T --init $d/${a}_stage1/ckpt_last.pt --out_dir $d/$a \
              $Z > $d.$a.log 2>&1" < /dev/null > /dev/null 2>&1 &
      done
    done
  done
done
ls $RUNS/*/g1?/selection.json | wc -l                            # 12 per dataset, 24 when done, then
pgrep -fc "wvad[.]train"                                        # 0 before the next batch / copying
```
A numerical failure is not re-run (that arm is failed); an infrastructure failure is re-run once (stage 1 again means stage 2 again). An exception in the new code path is a code defect: fix it, pass the no-change gate again and re-run all 24 chains with that commit.

**Analysis** (local CPU): gates (a)-(f) first (`results/e016_gate.json`: epoch-0 scores, init paths, `global_mode` in the candidate and baseline configs, the option is active and the arms differ, the offset parameters were learned in both stages, commit and environment as in the no-change gate); stops if a gate fails or a run or its offset record is unreadable. The verdict uses Delta alone (`outcome_delta`); `results/e016_summary.json` is written in three steps: verdict and choice, then the descriptives (readout decomposition, G1a budget rule, G0 labelled selection-contaminated), then the UCF UCA proxy AP (descriptive only; `--no_proxy` skips it).
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e016' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e016 --e003 <local runs root>/e003 --e013 <local runs root>/e013 \
    --runs <local runs root>/e016
```

### Stage D — E017 learning-rate multiplier on the G1a gamma ([plan](experiments/E017_offset_scale/PLAN.md))
The E016 G1a head (`--global_mode mean`) with the learning rate of the offset parameters multiplied: `m3` (`--offset_lr_mult 3`) and `m10` (`--offset_lr_mult 10`), in both stages; each arm is retrained from stage 1 (20 epochs) -> `ckpt_last.pt` -> stage 2 (30 epochs, no pseudo labels). The baseline is the existing E013 `l20` runs; the E016 `g1a` runs (multiplier 1) are a descriptive reference arm. With a global offset the offset parameters always form their own AdamW group (learning rate x multiplier, x 1 by default); without one (`none`) the optimizer is the earlier one, and a multiplier other than 1 is rejected.

**No-change gate** (server, before any candidate): three chains with the new code: two at the default (the XD chain gives `--global_mode none --offset_lr_mult 1` explicitly in both stages) compared with E003 / E013, and one G1a chain (`--global_mode mean`, multiplier 1, two-group path) compared with E016 `g1a_stage1` / `g1a`. Deploy the committed code first.
```bash
GATE=~/wvad_runs/e017_gate_$(head -c 8 DEPLOYED_COMMIT)    # one folder per deployed commit
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
mkdir "$GATE" && for spec in "ucf xcb_xcl stage1 l20" "xdv xcb_siglip stage1 l20 --global_mode none --offset_lr_mult 1" \
                             "ucf xcb_xcl g1a_stage1 g1a --global_mode mean"; do   # launches only into a new folder
  set -- $spec; c=${1}_${2}_s0; T="--dataset $1 --pair $2 --seed 0 $5 $6 $7 $8"
  setsid nohup bash -c "
    python -m wvad.train stage1 $T --out_dir $GATE/$c/$3 --record > $GATE/$c.$3.log 2>&1 &&
    python -m wvad.train stage2 $T --init $GATE/$c/$3/ckpt_last.pt --out_dir $GATE/$c/$4 \
        $Z > $GATE/$c.$4.log 2>&1" < /dev/null > /dev/null 2>&1 &
done
ls $GATE/*/{l20,g1a}/selection.json | wc -l                    # 3 when done, and
pgrep -fc "wvad[.]train"                                        # 0 (records are written after selection.json), then
python tests/equivalence/check_e017_gate.py --e003_runs ~/wvad_runs/e003 --e013_runs ~/wvad_runs/e013 \
    --e016_runs ~/wvad_runs/e016 --gate_runs $GATE              # must print all_ok=True
```
Copy `experiments/E017_offset_scale/results/e017_identity.json` back from the server and commit it. If the gate fails, check the cause before running candidates (plan): deploy commit `77e2222` to a separate folder and run the same chains there without `--offset_lr_mult`; compare the first two with that folder's `check_e016_gate.py` and the third directly with the E016 G1a runs (the comparisons of `check_e017_gate.py`, without its new-code marker).

**Candidates** (server, same deployed commit as the gate): check GPU and host memory first and use what is free (shared H100: memory only). With room for 24 processes (about 1 GB GPU and 2.5 GB host each, plus 2 GB / 8 GB spare) launch both datasets at once (`DS="ucf xdv"`); otherwise one dataset per batch. Each dataset first records its environment and commit and checks them against the passed `e017_identity.json` (gate (f)).
```bash
RUNS=~/wvad_runs/e017 && mkdir -p $RUNS
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv && free -g
DS="ucf xdv"                             # or DS=ucf, then DS=xdv when that batch is done
for ds in $DS; do
  python -c "from wvad.diagnostics.e017 import record_env; record_env('$RUNS/env_$ds.json')" || continue
  for pair in xcb_xcl xcb_siglip; do
    for seed in 0 1 2; do
      c=${ds}_${pair}_s${seed}; d=$RUNS/$c; mkdir -p $d
      for arm in m3 m10; do
        T="--dataset $ds --pair $pair --seed $seed --global_mode mean --offset_lr_mult ${arm#m}"
        setsid nohup bash -c "
          python -m wvad.train stage1 $T --out_dir $d/${arm}_stage1 --record > $d.${arm}_s1.log 2>&1 &&
          python -m wvad.train stage2 $T --init $d/${arm}_stage1/ckpt_last.pt --out_dir $d/$arm \
              $Z > $d.$arm.log 2>&1" < /dev/null > /dev/null 2>&1 &
      done
    done
  done
done
ls $RUNS/*/m{3,10}/selection.json | wc -l                        # 12 per dataset, 24 when done, then
pgrep -fc "wvad[.]train"                                        # 0 before the next batch / copying
```
A numerical failure is not re-run (that arm is failed); an infrastructure failure is re-run once (stage 1 again means stage 2 again). An exception in the new code path is a code defect: fix it, pass the no-change gate again and re-run all 24 chains with that commit.

**Analysis** (local CPU): gates (a)-(f) first (`results/e017_gate.json`); stops if a gate fails or a run or its offset record is unreadable. The E016 G1a runs must reproduce `e016_runs.csv`. The verdict uses Delta alone (`outcome_delta`); `results/e017_summary.json` is written in three steps: verdict and choice, then the descriptives (gamma trajectories and the learning-rate-bound rule, dose-response with the m = 1 reference, readout decomposition, saturation), then the UCF UCA proxy AP on the stored scores and on the logit readout (descriptive only; `--no_proxy` skips it).
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e017' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e017 --e003 <local runs root>/e003 --e013 <local runs root>/e013 \
    --e016 <local runs root>/e016 --runs <local runs root>/e017
```

### Stage D — E018 delta recalibration on the M10 pipeline ([plan](experiments/E018_delta_recalibration/PLAN.md))
The stage-D threshold delta is re-estimated on the adopted pipeline (E017 M10) from 6 seeds per (dataset, pair): the E017 dev-seed runs (0-2) and three calibration-only seeds (100-102, never used in comparisons or stage-F confirmation). Same metric, splits and MC rule as E010; no new code path: the calibration chains run the E017 M10 command on the E017 gate commit (`2b0cfdf`).
```bash
RUNS=~/wvad_runs/e018 && mkdir -p $RUNS
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
for ds in ucf xdv; do
  python -c "from wvad.diagnostics.e017 import record_env; record_env('$RUNS/env_$ds.json')" || continue
  for pair in xcb_xcl xcb_siglip; do
    for seed in 100 101 102; do
      c=${ds}_${pair}_s${seed}; d=$RUNS/$c; mkdir -p $d
      T="--dataset $ds --pair $pair --seed $seed --global_mode mean --offset_lr_mult 10"
      setsid nohup bash -c "
        python -m wvad.train stage1 $T --out_dir $d/m10_stage1 --record > $d.m10_s1.log 2>&1 &&
        python -m wvad.train stage2 $T --init $d/m10_stage1/ckpt_last.pt --out_dir $d/m10 \
            $Z > $d.m10.log 2>&1" < /dev/null > /dev/null 2>&1 &
    done
  done
done
```
**Analysis** (local CPU): gates (config equal to E017 M10 except seed, init chain, environment, record/selection, dev-seed runs equal `e017_runs.csv`), then SD_d and delta_d per dataset (`results/e018_summary.json`, read by later stage-D analyses).
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e018' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e018 --e017 <local runs root>/e017 --runs <local runs root>/e018
```

### Stage D — E019 C05 pre-check: single-encoder heads and late fusion ([plan](experiments/E019_encoder_late_fusion/PLAN.md))
Single-encoder "pairs" `xcb`, `xcl`, `siglip` (input LayerNorm only with SigLIP) trained with the M10 pipeline; their offset-including logits (reconstructed in float64 from the two fp32 records by the sign of the offset) are averaged per seed and epoch (late fusion) and compared with the concatenated M10 baseline in the same readout. Go to the C05 main experiment (E020) if at least two settings gain delta (E018) and none loses delta; interpretation flags (explicable by 2-model ensembling or one encoder, no encoder-diversity evidence, one dataset only, readout-sensitive) require user confirmation before E020.

**No-change gate** (server): the catalog change must leave the two original pairs unchanged.
```bash
GATE=~/wvad_runs/e019_gate_$(head -c 8 DEPLOYED_COMMIT)
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
mkdir "$GATE" && for spec in "ucf xcb_xcl stage1 l20" \
                             "xdv xcb_siglip m10_stage1 m10 --global_mode mean --offset_lr_mult 10"; do
  set -- $spec; c=${1}_${2}_s0; T="--dataset $1 --pair $2 --seed 0 $5 $6 $7 $8"
  setsid nohup bash -c "
    python -m wvad.train stage1 $T --out_dir $GATE/$c/$3 --record > $GATE/$c.$3.log 2>&1 &&
    python -m wvad.train stage2 $T --init $GATE/$c/$3/ckpt_last.pt --out_dir $GATE/$c/$4 \
        $Z > $GATE/$c.$4.log 2>&1" < /dev/null > /dev/null 2>&1 &
done
pgrep -fc "wvad[.]train"                                        # 0, then
python tests/equivalence/check_e019_gate.py --e003_runs ~/wvad_runs/e003 --e013_runs ~/wvad_runs/e013 \
    --e017_runs ~/wvad_runs/e017 --gate_runs $GATE              # must print all_ok=True
```
A third gate chain (UCF XCB+XCL seed 0 with the M10 options, compared with E017 `m10_stage1` / `m10`; plan review F7) was launched together with the candidates; the same check covers it (6 comparisons).

**Candidates** (server, 18 chains = 2 datasets x 3 encoders x seeds 0-2, plus the F7 gate chain), as launched on 2026-10-06 12:06 before the plan freeze (training commands fixed by the first draft `PLAN_draft1.md`):
```bash
RUNS=~/wvad_runs/e019 && mkdir -p $RUNS
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
ID=experiments/E019_encoder_late_fusion/results/e019_identity.json
for ds in ucf xdv; do
  python -c "from wvad.diagnostics.e017 import record_env; record_env(\"$RUNS/env_$ds.json\", \"$ID\")" || continue
  for enc in xcb xcl siglip; do
    for seed in 0 1 2; do
      c=${ds}_${enc}_s${seed}; d=$RUNS/$c; mkdir -p $d
      T="--dataset $ds --pair $enc --seed $seed --global_mode mean --offset_lr_mult 10"
      setsid nohup bash -c "
        python -m wvad.train stage1 $T --out_dir $d/m10_stage1 --record > $d.m10_s1.log 2>&1 &&
        python -m wvad.train stage2 $T --init $d/m10_stage1/ckpt_last.pt --out_dir $d/m10 \
            $Z > $d.m10.log 2>&1" < /dev/null > /dev/null 2>&1 &
    done
  done
done
G=~/wvad_runs/e019_gate_70d41384; c=ucf_xcb_xcl_s0
T="--dataset ucf --pair xcb_xcl --seed 0 --global_mode mean --offset_lr_mult 10"
setsid nohup bash -c "
  python -m wvad.train stage1 $T --out_dir $G/$c/m10_stage1 --record > $G/$c.m10_stage1.log 2>&1 &&
  python -m wvad.train stage2 $T --init $G/$c/m10_stage1/ckpt_last.pt --out_dir $G/$c/m10 \
      $Z > $G/$c.m10.log 2>&1" < /dev/null > /dev/null 2>&1 &
```
**Analysis** (local CPU, after the plan freeze): gates (a)-(h), go/stop, flags and descriptives; delta, R and splits from `e018_summary.json`. A chain whose rerun also failed for infrastructure reasons is passed with `--infra_failed <chain>` (counted as a failed run, PLAN.md).
```bash
ssh <server> 'tar -cf - -C ~/wvad_runs e019' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e019 --e017 <local runs root>/e017 --runs <local runs root>/e019
```

### Stage F — E020 F1 confirmation experiment ([plan](experiments/E020_confirmation_f1/PLAN.md))
2x2 factorial on the test set, used once after the plan freeze: stage-2 start (A: stage-1 best-validation vs last checkpoint) x offset (B: off vs M10), 2 datasets x 2 feature pairs x the 10 confirmation seeds. Test manifests are copied from the previous project's lists and checked against the official annotations (no model score is read):
```bash
python -m wvad.prep.eval_manifests ucf --previous <ucf_test_manifest.json> \
    --annotation <Temporal_Anomaly_Annotation_for_Testing_Videos.txt> --out data/manifests/ucf_test.json
python -m wvad.prep.eval_manifests xdv --previous <xdv_test_manifest.json> \
    --annotation <XD-Violence/annotations/test_annotation.txt> --out data/manifests/xdv_test.json
```
**Training** (server, deployed commit `70d4138`, before the plan freeze; no test data): L20 and M10 chains (stage 1 -> `ckpt_last.pt` -> stage 2), 24 at a time, as launched on 2026-10-06 13:14:
```bash
export RUNS=~/wvad_runs/e020 Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
mkdir -p $RUNS && ID=experiments/E019_encoder_late_fusion/results/e019_identity.json
for ds in ucf xdv; do
  python -c "from wvad.diagnostics.e017 import record_env; record_env('$RUNS/env_$ds.json', '$ID')" || exit 1
done
for seed in 42 123 777 1000 2025 7 13 99 555 31337; do
  for ds in ucf xdv; do for pair in xcb_xcl xcb_siglip; do for arm in l20 m10; do
    echo $arm $ds $pair $seed
  done; done; done
done > $RUNS/jobs.txt
setsid nohup xargs -P 24 -L 1 bash -c '
  arm=$0; ds=$1; pair=$2; seed=$3; c=${ds}_${pair}_s${seed}; d=$RUNS/$c; mkdir -p $d
  T="--dataset $ds --pair $pair --seed $seed"
  if [ $arm = m10 ]; then T="$T --global_mode mean --offset_lr_mult 10"; fi
  python -m wvad.train stage1 $T --out_dir $d/${arm}_stage1 --record > $d.${arm}_s1.log 2>&1 &&
  python -m wvad.train stage2 $T --init $d/${arm}_stage1/ckpt_last.pt --out_dir $d/$arm $Z > $d.$arm.log 2>&1
' < $RUNS/jobs.txt > $RUNS/queue.log 2>&1 &
```
The base and m10b cells (stage 2 only, from each chain's stage-1 `ckpt_best_val.pt`; m10b with the M10 options) were queued at 13:48 in the same shell (same deploy, `RUNS` and `Z`); the queue waited for the first `xargs`, then ran 24 at a time. This batch wrote no environment record (gate (e) uses the deploy evidence below):
```bash
for seed in 42 123 777 1000 2025 7 13 99 555 31337; do
  for ds in ucf xdv; do for pair in xcb_xcl xcb_siglip; do for cell in base m10b; do
    echo $cell $ds $pair $seed
  done; done; done
done > $RUNS/jobs_best.txt
FIRST=$(pgrep -x xargs | head -1); echo "first queue xargs pid: $FIRST"   # the 13:14 queue
setsid nohup bash -c "
  while kill -0 $FIRST 2>/dev/null; do sleep 20; done
  date > $RUNS/queue_best.start
  xargs -P 24 -L 1 bash -c '
    cell=\$0; ds=\$1; pair=\$2; seed=\$3; c=\${ds}_\${pair}_s\${seed}; d=$RUNS/\$c
    T=\"--dataset \$ds --pair \$pair --seed \$seed\"; s1=l20_stage1
    if [ \$cell = m10b ]; then T=\"\$T --global_mode mean --offset_lr_mult 10\"; s1=m10_stage1; fi
    python -m wvad.train stage2 \$T --init \$d/\$s1/ckpt_best_val.pt --out_dir \$d/\$cell $Z > \$d.\$cell.log 2>&1
  ' < $RUNS/jobs_best.txt > $RUNS/queue_best.log 2>&1
" < /dev/null > /dev/null 2>&1 &
```
**Plan freeze** (local, after the code review; no e020 run output opened before): `PLAN.sha256` holds the sha256 of `PLAN.md` before its `## 변경 기록` line (change-log entries keep the hash), written as `<sha256>  PLAN.md (before ## 변경 기록)`. Commit the plan, the reviewed code, the test manifests and this README first: `freeze` refuses unless `PLAN.md` is committed unmodified at HEAD and the tracked tree is clean. Then commit `PLAN.sha256` alone: that commit is the freeze commit (its tree holds the plan committed before). Every test-scoring call must run on it or a descendant (checked by the analysis). It is the only commit that may touch `PLAN.sha256`: `gates --check v` and `analyze` stop unless the file was added once, by the freeze commit, with the hash of the current plan and of `PLAN.md` in that commit (a later "re-freeze" is refused). Check the seal history right after the freeze commit, before deploying:
```bash
git add experiments/E020_confirmation_f1 src data/manifests README.md && git commit   # plan, reviewed code, manifests
python -m wvad.diagnostics.e020 freeze
git add experiments/E020_confirmation_f1/PLAN.sha256 && git commit    # the freeze commit: PLAN.sha256 alone
python -c "from wvad.diagnostics.e020 import seal_history; print(seal_history())"
```
**Scoring deploy** (server): the freeze commit goes to a separate folder, so the training deploy keeps `DEPLOYED_COMMIT` `70d4138`. Server commands below run in `~/frozen-vlm-wvad-e020` with `SCORES`, the score root (`~/wvad_runs/e020_scores`; for a new root after a forward-path fix (step 4), change it and its local copy `S` of step 1); path (ii) (CPU, batch 256) runs under P000's thread set-up: `OMP_NUM_THREADS` and `MKL_NUM_THREADS` exported as 6 before Python starts (the scoring checks them and `torch.get_num_threads()` = 6, never sets threads itself, and records torch's parallel info and MKL version).
```bash
git archive --format=tar HEAD | ssh <server> 'mkdir -p ~/frozen-vlm-wvad-e020 && tar -x -C ~/frozen-vlm-wvad-e020'
git rev-parse HEAD | ssh <server> 'cat > ~/frozen-vlm-wvad-e020/DEPLOYED_COMMIT'
ssh <server> 'cp ~/frozen-vlm-wvad/configs/paths.yaml ~/frozen-vlm-wvad-e020/configs/'
# on the server, in ~/frozen-vlm-wvad-e020, with the Python environment active:
export PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
SCORES=~/wvad_runs/e020_scores        # the score root; a new one after a forward-path fix (step 4)
```
**1. Gates before scoring** (local CPU): (a)-(f), (t) and the selection part of (v0). Gate (e) checks the 13:14 batch by `env_{ucf,xdv}.json` and the 13:48 batch by the deploy evidence (file times on the server, read before copying anything; the commit must be the E019 gate's, the deploy must precede the queue and the first base / m10b log, the first log must not be later than the last, and all 80 base / m10b logs must be there; a run of `--infra_failed` may lack its log). The evidence is written by `wvad.diagnostics.e020_gates.write_deploy_evidence`, run from the scoring deploy. Every `gates` and `analyze` call writes `results/e020_gate.json`, also when a gate fails: commit it either way (a failed gate file is a record, and the clean-tree commands refuse while it is uncommitted):
```bash
D=~/frozen-vlm-wvad; R=~/wvad_runs/e020         # on the server, in ~/frozen-vlm-wvad-e020; D: the training deploy
python -c "from wvad.diagnostics.e020_gates import write_deploy_evidence; write_deploy_evidence('$D', '$R', '$R/deploy_evidence.json')"
# local
ssh <server> 'tar -cf - -C ~/wvad_runs e020 e019_gate_70d41384' | tar -xf - -C <local runs root>
mkdir -p experiments/E020_confirmation_f1/results
cp <local runs root>/e020/deploy_evidence.json experiments/E020_confirmation_f1/results/e020_deploy_evidence.json
G="--runs <local runs root>/e020 --e019_gate <local runs root>/e019_gate_70d41384 \
   --deploy_evidence experiments/E020_confirmation_f1/results/e020_deploy_evidence.json"
S=<local runs root>/e020_scores       # local copy of $SCORES (same folder name; a new root changes both)
python -m wvad.diagnostics.e020 gates $G         # results/e020_gate.json; commit it and the evidence
```
**2. Validation scoring** (server): path (i) for the 160 runs x 2 rules (gate (s)), path (ii) for the 40 base runs (pre-check against P000's `val_video_auc`, 6 decimals) and the base weights against P000's checkpoints (gate (v0), recorded in `val/meta.json` with the commit and environment, which every test call must share). Then locally (s), the pre-check (`s_precheck`, not blocking) and (v0) (not blocking; only passing base runs enter (v)); the gate file records the sha256 of the `val/scores.sha256` it checked and is copied to the server for step 3:
```bash
python -m wvad.diagnostics.e020_score --runs ~/wvad_runs/e020 --out $SCORES \
    --split val --p000_root ~/sear_phase1/outputs
ssh <server> "tar -cf - -C ~/wvad_runs $(basename $S)/val" | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e020 gates $G --scores $S     # clean tree; commit the gate file
scp experiments/E020_confirmation_f1/results/e020_gate.json <server>:wvad_runs/e020_gate_s.json
```
Validation scores are written once. After a rerun the plan allows before test scoring ((b), (c) or (e) failure, infrastructure), copy the runs again (the `ssh <server> 'tar ... e020 e019_gate_70d41384'` line of step 1), score all of them again with `--rescore '<reason>'` (the old `val/` moves to `val/superseded/<UTC time>/`), then rerun `gates`, commit and copy the gate file. A code change after test scoring began (step 4) needs a new `--out` root instead (refused under the old one once an access-log line names its `test/`, even of a call cut short).

**3. Base test scoring** (server, once, after (a)-(f), (t), (s) passed): the 40 base runs x 2 rules by both paths. The command refuses unless the copied gate file shows (a)-(f), (t) and (s) passed with no unreadable run, was written from a clean tree and checked this server's `val/scores.sha256` (same sha256); it also checks the frozen plan hash, the commit and environment of the validation scoring, that every checkpoint has the sha256 its validation file recorded and names its own run (config, `init`, `out_dir`), and path (ii)'s threads. Right before any test feature is deserialised (after the six test caches' sha256 matched the plan's table, a read of bytes only), it appends one line to `~/wvad_runs/e020/test_access.log` (its realpath recorded; refused if the score root already has a meta of that line or later, or an earlier call used another log) and writes the call's `test/meta_<k>.json` with `"complete": false`, rewritten with `"complete": true` at the end. Score files are written under a temporary name (`*.npz.part`) and renamed. The scoring runs detached (an ssh drop does not stop it); wait for its end (`tail $SCORES.base.log`: the last line counts the files written). If it stopped early for infrastructure reasons (the log ends in an error, the meta keeps `"complete": false`), only the missing files are scored, before any metric (`--rescore '<reason>' --missing_only`: a file that is not a readable zip counts as missing and is moved aside unread; not a full read, and gate (v) binds both calls). Only if (v0) failed for all 40 runs (and was evaluated without error), P000's own `abl_base` checkpoints are scored by path (ii) for (v) (each first loaded as the scoring loads it):
```bash
setsid nohup python -m wvad.diagnostics.e020_score --runs ~/wvad_runs/e020 --out $SCORES --split test \
    --cells base --gate_s ~/wvad_runs/e020_gate_s.json > $SCORES.base.log 2>&1 < /dev/null &
# only after an infrastructure failure of that call (the missing files):
# setsid nohup python -m wvad.diagnostics.e020_score --runs ~/wvad_runs/e020 --out $SCORES --split test --cells base \
#     --rescore '<reason>' --missing_only --gate_s ~/wvad_runs/e020_gate_s.json > $SCORES.base_missing.log 2>&1 < /dev/null &
# only if (v0) failed for every base run (detached in the same way):
# python -m wvad.diagnostics.e020_score ... --split test --cells p000_base --gate_s ~/wvad_runs/e020_gate_s.json --p000_root ~/sear_phase1/outputs
```
**4. Gate (v)** (local, clean working tree, before any other test scoring): path (ii) sigma readout vs P000 `abl_base` (best / final rows) within 1e-4 points, on the files of the last full base read (and its `--missing_only` calls; a call cut short counts), all of one commit and plan hash and run with P000's threads; (i) - (ii) differences are recorded only. If (v) fails, the later cells are not scored (PLAN.md failure rule); commit the failed gate file as a record. The cause is looked for in the base scores only:
- metric or ground-truth computation: a local fix only, no redeploy (the later cells must be scored by the deployed commit that wrote the base and validation scores; a redeploy is refused). Commit the fix, run `gates --check v` again on the saved scores (no test feature is read again), then `analyze` with `--allow_code_change '<reason>'` and the reason in the plan's change log.
- forward path: the fix is committed and deployed (as above), then scored under a new `--out` root (set `SCORES` and `S` to it, e.g. `~/wvad_runs/e020_scores_2`): `--split val`, local `gates $G --scores $S` (commit and copy the gate file), and base once more with `--rescore '<reason>' --gate_s ...`. Base is read in full at most twice and never after a later cell (the command refuses).
```bash
ssh <server> "tar -cf - -C ~/wvad_runs $(basename $S)/test" | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e020 gates $G --scores $S --check v
git add experiments/E020_confirmation_f1/results/e020_gate.json && git commit    # passed or failed
```
**5. Later cells** (server): l20, m10b, m10 by path (i), refused unless the copied gate file shows (v) passed on base scores of this deployed commit and plan hash. Detached and waited for as in step 3 (`tail $SCORES.later.log`); after an infrastructure failure, only the missing files are scored, before any metric (the same command with `--rescore '<reason>' --missing_only`; a file that is not a readable zip counts as missing and is moved aside unread).
```bash
scp experiments/E020_confirmation_f1/results/e020_gate.json <server>:wvad_runs/e020_gate_v.json
setsid nohup python -m wvad.diagnostics.e020_score --runs ~/wvad_runs/e020 --out $SCORES --split test \
    --cells l20 m10b m10 --gate_v ~/wvad_runs/e020_gate_v.json > $SCORES.later.log 2>&1 < /dev/null &
```
**6. Analysis** (local, clean working tree): copy the access log and the score sha256 list into `results/` and commit them (`*.log` is git-ignored: add it with `-f`), then the seal history, the gates, (v), the test files (each with the checkpoint sha256 of the local run copy and of its validation file, the selected epoch (last rule: 30), the recorded `val_video_auc` and its path's conditions) and the access log (every call on the freeze commit or a descendant, frozen plan hash, base first and read in full at most twice, the later cells on the commit and plan of (v), one access log, every call of this score root with its meta; every call on the validation scoring's commit and the later cells unlocked by this (v)) are checked again before the verdict, flags, factorial, bootstrap and descriptives (`results/e020_summary.json` with the library versions and the Wilcoxon method SciPy used, `results/e020_runs.csv` in points). Files under `src/` or `data/manifests/` changed since the freeze commit stop it unless `--allow_code_change '<reason>'` is given (the reason also goes to the plan's change log).
```bash
ssh <server> "tar -cf - -C ~/wvad_runs $(basename $S)/test" | tar -xf - -C <local runs root>
scp <server>:wvad_runs/e020/test_access.log experiments/E020_confirmation_f1/results/e020_test_access.log
scp <server>:wvad_runs/$(basename $S)/test/scores.sha256 experiments/E020_confirmation_f1/results/e020_test_scores.sha256
git add -f experiments/E020_confirmation_f1/results/e020_test_access.log
git add experiments/E020_confirmation_f1/results/e020_test_scores.sha256 && git commit
python -m wvad.diagnostics.e020 analyze $G --scores $S     # after a metric fix: --allow_code_change '<reason>'
```

### Stage E — E021 dense-snippet pilot ([plan](experiments/E021_dense_snippet_pilot/PLAN.md))
Validation only, GO/NO-GO, two stages pre-registered in one plan. The training split is re-extracted as gap-free 16-frame snippets (`python -m wvad.extract`, one cache per dataset and encoder under `<feature_root>/<dataset>_<encoder>_d16/`): stage 1 the validation videos, stage 2 the training part. 32-unit caches derived from the snippets (`d16c`, `d16r`, UCF `d16r1` / `d16r2`, `d16m`, `d16m2`) train the heads `h1` (d16c), `h1r` (d16r), `hb` (d16m) and `hbm` (d16m2) beside `h0` (the earlier caches, `--features legacy`), in both pipelines (`l20`: E013; `m10`: E017 `--global_mode mean --offset_lr_mult 10`); seeds 0-2 (development) and 100-102 (calibration, SD only; h0, h1, hb). Every stage-2 epoch checkpoint is scored on the validation videos (`wvad.diagnostics.e021_score`), and the local analysis (`wvad.diagnostics.e021`) gives the stage-1 verdict (A0), then the final verdict (C, one of eight categories). The extraction, training and scoring code (`src/wvad/{extract,train,models,data,eval}`, `e021_score.py`) is frozen from the stage-1 production to the end of stage 2 (gate (f)): the server keeps the deploy of the no-change gate commit for both stages (no redeploy in between). Results go to `experiments/E021_dense_snippet_pilot/results/` (`E` below); caches, checkpoints, scores and logs stay outside the repository.
```bash
E=experiments/E021_dense_snippet_pilot/results      # both machines (repository / deploy root)
L=<local runs root>/e021_extract                    # local gate outputs
X=~/wvad_runs/e021_extract; RUNS=~/wvad_runs/e021; S=~/wvad_runs/e021_scores     # server
```
`configs/paths.yaml` of both machines needs `raw_video_root` (the folder holding `Anomaly-Detection-Dataset/` and `XD-Violence/train/`); `feature_root` receives the dense caches. Every video is opened only through the training manifest and appended to an access log first.

**1. Implementation gate (x0)** (local; the code review follows): unit tests, then the CUDA normalisation record (the lookup table must equal numpy on all 768 value-channel pairs; the Python-scalar division mismatches are recorded).
```bash
pytest tests/test_e021_*.py
python -m wvad.extract gate norm --record $E/e021_extract_gate.json
```
**2. Freeze** (after the pre-freeze probes are recorded in `results/e021_prefreeze_probes.json`): `PLAN.sha256` = sha256 of `PLAN.md` before `## 변경 기록` (as E020). Commit the plan, the reviewed code and this README first; `freeze` refuses unless `PLAN.md` is committed unmodified and the tracked tree is clean. Then commit `PLAN.sha256` alone (the freeze commit; the extraction, training and analysis commits are it or descendants). The analysis commands stop when the plan changed after the freeze.
```bash
git add experiments/E021_dense_snippet_pilot src data README.md && git commit
python -m wvad.diagnostics.e021 freeze
git add experiments/E021_dense_snippet_pilot/PLAN.sha256 && git commit
```
**3. Code equivalence (x1)** (local RTX 4080, `riskdet`, earlier precision and batches; XD 12 training-part videos, UCF 10 validation videos with the per-encoder position recovery):
```bash
mkdir -p $L && for ds in xdv ucf; do
  python -m wvad.extract gate x1 --dataset $ds --access_log $L/access_local.csv --out $L/x1_$ds.json \
      --record $E/e021_extract_gate.json
done
```
If (x1) fails: PLAN.md "추출 관문" (find the cause, fix and record the fix in the change log; otherwise stop and report; with the user's approval that (dataset, encoder) continues as approximate equivalence). User approvals are written to `$E/e021_approvals.json` before the analysis reads them: `{"x1": [{"dataset": .., "encoder": .., "decision": "<the user's words>", "date": ..}], "x3a": [{"dataset": .., "videos": [<names>], ..}], "x3": [{"dataset": .., "phase": "val"|"trainpart", "videos": [<names>], ..}]}`.

**4. Server extraction environment** (once; a separate conda environment, the training environment is not touched), the pinned model snapshots, then the environment record, the sha256 of all 5,560 training-split videos with their header frame counts, and on the local machine the same sha256:
```bash
# local: deploy the freeze commit (server conventions) and copy the three pinned model snapshots
tar -cf - -C ~/.cache/huggingface/hub models--microsoft--xclip-base-patch16 \
    models--microsoft--xclip-large-patch14 models--google--siglip-so400m-patch14-384 \
    | ssh <server> 'mkdir -p ~/.cache/huggingface/hub && tar -x -C ~/.cache/huggingface/hub'
scp $L/x1_xdv.json $L/x1_ucf.json <server>:wvad_runs/e021_extract/
# server, once
conda create -y -n wvad-extract python=3.10 && conda activate wvad-extract
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121    # cu124 only if the driver needs it (record it; (x2) decides)
pip install transformers==5.0.0 opencv-python==4.13.0.90 numpy==2.2.6 pyyaml
# server, in ~/frozen-vlm-wvad, for every `python -m wvad.extract` command below
conda activate wvad-extract && export PYTHONPATH=src HF_HUB_OFFLINE=1 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p $X && python -m wvad.extract gate env --out $E/env_extract.json   # pip freeze, CUDA, cuDNN, driver, TF32, attention
for ds in ucf xdv; do
  python -m wvad.extract header --dataset $ds --access_log $X/access_server.csv --out $X/header_$ds.csv
done
# local (the same files, sha256 only)
for ds in ucf xdv; do
  python -m wvad.extract header --dataset $ds --sha_only --access_log $L/access_local.csv --out $L/sha_$ds.csv
done
scp <server>:wvad_runs/e021_extract/{header_ucf.csv,header_xdv.csv,access_server.csv} $L/
python -m wvad.extract check x6 --server $L/header_ucf.csv $L/header_xdv.csv --local $L/sha_ucf.csv \
    $L/sha_xdv.csv --access_log $L/access_server.csv --out $L/x6.json --record $E/e021_extract_gate.json
python -m wvad.extract check x3a --headers $L/header_ucf.csv $L/header_xdv.csv --out $L/x3a.json \
    --record $E/e021_extract_gate.json
```
(x6) or (x3a) failing (a sha256 mismatch, a path outside the training manifest, |header T - manifest T| > 16): stop and report (PLAN.md). Then, on the server, the environment gate (x2) on the (x1) videos and positions (production decoder, autocast, batches 8 / 8 / 4; frame sha256 against the local record) and the throughput on the fixed 20 validation videos with both stages' projected hours (a stage over 24 h on the H100 is reported first and not started without the user's decision). Server gate summaries go to a server-side record that is merged into the local one:
```bash
for ds in xdv ucf; do
  python -m wvad.extract gate x2 --dataset $ds --x1 $X/x1_$ds.json --access_log $X/access_server.csv \
      --out $X/x2_$ds.json --record $X/gate_server.json
done
python -m wvad.extract bench --out_root $X/bench --access_log $X/access_server.csv --out $X/bench.json \
    --record $X/gate_server.json
# local: merge the server record (its keys are server-only) and keep the environment record
scp <server>:wvad_runs/e021_extract/gate_server.json $L/ && scp <server>:frozen-vlm-wvad/$E/env_extract.json $E/
python -c "import json, sys; from wvad.extract.__main__ import merge_record; \
[merge_record(sys.argv[1], k, v) for k, v in json.load(open(sys.argv[2], encoding='utf-8')).items()]" \
    $E/e021_extract_gate.json $L/gate_server.json
```
If (x2) fails and cannot be fixed, the user decides on local production; then (x2) is run locally with the production decoder, precision and batches (`gate x2` as above on the local machine, `--record $E/e021_extract_gate.json`), its environment becomes `env_extract.json`, and (x6) / (x3a) use the local files (PLAN.md R12).

**5. Stage-1 production** (server, extraction environment): the 833 validation videos (completion flags allow resuming with the same command), then (x3) (grabs = T, replaced share <= 1%, the frames past the header count excluded for videos that passed (x3a)) and (x4) (order, labels, n_v, rows, completion, finite, model revision, commit and environment = the frozen commit and `env_extract.json`); a video over 1% is listed and stops the stage after the extraction (PLAN.md).
```bash
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 60 > $X/gpu_memory_val.csv &   # descriptive 5
setsid nohup bash -c "python -m wvad.extract dense --dataset ucf --phase val --access_log $X/access_server.csv &&
  python -m wvad.extract dense --dataset xdv --phase val --access_log $X/access_server.csv" \
  > $X/dense_val.log 2>&1 < /dev/null &
pgrep -fc "wvad[.]extract dense"                                # 0 when done (the log ends with a summary), then
pkill -f "nvidia-smi --query-gpu=timestamp"
for ds in ucf xdv; do
  python -m wvad.extract check x3 --dataset $ds --phase val --headers $X/header_ucf.csv $X/header_xdv.csv \
      --out $X/x3_${ds}_val.json --record $X/gate_server.json
  python -m wvad.extract check x4 --dataset $ds --phase val --env $E/env_extract.json \
      --commit $(cat DEPLOYED_COMMIT) --out $X/x4_${ds}_val.json --record $X/gate_server.json
done
```
**6. Stage-1 derived caches** (server): `d16c`, `d16r` (and UCF `d16r1`, `d16r2`) as full-size files with the training-part rows NaN, then (x5); merge the record as in step 4.
```bash
for ds in ucf xdv; do
  python -m wvad.extract derive --dataset $ds --stage 1
  python -m wvad.extract check x5 --dataset $ds --stage 1 --out $X/x5_${ds}_1.json --record $X/gate_server.json
done
```
**7. No-change gate (g1), h0 chains, (g2)** (server, the training environment; check GPU and host memory first: 24 processes need about 1 GB GPU and 2.5 GB host each). (g1): UCF XCB+XCL seed 0 without the new options against E003 `stage1` / E013 `l20`; it writes `$E/e021_identity.json`, which every batch's environment record is checked against (`record_env`: the environment and commit of (g1), which must have passed). Then the 48 h0 chains (`--features legacy --save_epoch_ckpts` in both stages; E013 `l20` and E017 `m10` commands; seeds 0-2 and 100-102), one dataset per batch of 24, and (g2) on them (development chains against E003 / E013 and E017, M10 calibration chains against E018; 30 epoch checkpoints per stage 2).
```bash
GATE=~/wvad_runs/e021_gate_$(head -c 8 DEPLOYED_COMMIT)
Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record"
mkdir "$GATE" && c=ucf_xcb_xcl_s0 && T="--dataset ucf --pair xcb_xcl --seed 0" && setsid nohup bash -c "
  python -m wvad.train stage1 $T --out_dir $GATE/$c/stage1 --record > $GATE/$c.stage1.log 2>&1 &&
  python -m wvad.train stage2 $T --init $GATE/$c/stage1/ckpt_last.pt --out_dir $GATE/$c/l20 \
      $Z > $GATE/$c.l20.log 2>&1" < /dev/null > /dev/null 2>&1 &
pgrep -fc "wvad[.]train"                                        # 0 when done, then
python tests/equivalence/check_e021_gate.py g1 --e003_runs ~/wvad_runs/e003 --e013_runs ~/wvad_runs/e013 \
    --gate_runs $GATE                                           # must print g1 all_ok=True
export RUNS Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record --save_epoch_ckpts"
mkdir -p $RUNS && nvidia-smi --query-gpu=memory.used,memory.total --format=csv && free -g
ds=ucf                                   # first batch; when it is done, the same lines with ds=xdv
python -c "from wvad.diagnostics.e021_gates import record_env; record_env('$RUNS/env_$ds.json')" || exit 1
for pair in xcb_xcl xcb_siglip; do for seed in 0 1 2 100 101 102; do for p in l20 m10; do
  echo $p $ds $pair $seed; done; done; done > $RUNS/jobs_h0_$ds.txt
setsid nohup xargs -P 24 -L 1 bash -c '
  p=$0; ds=$1; pair=$2; seed=$3; c=${ds}_${pair}_s${seed}; d=$RUNS/$c; mkdir -p $d
  T="--dataset $ds --pair $pair --seed $seed --features legacy"
  if [ $p = m10 ]; then T="$T --global_mode mean --offset_lr_mult 10"; fi
  python -m wvad.train stage1 $T --out_dir $d/h0_${p}_stage1 --record --save_epoch_ckpts > $d.h0_${p}_s1.log 2>&1 &&
  python -m wvad.train stage2 $T --init $d/h0_${p}_stage1/ckpt_last.pt --out_dir $d/h0_$p $Z > $d.h0_$p.log 2>&1
' < $RUNS/jobs_h0_$ds.txt > $RUNS/queue_h0_$ds.log 2>&1 &
ls $RUNS/*/h0_{l20,m10}/selection.json | wc -l                  # 24 per batch, 48 when done, then
pgrep -fc "wvad[.]train"                                        # 0 before the next batch / (g2)
python tests/equivalence/check_e021_gate.py g2 --e003_runs ~/wvad_runs/e003 --e013_runs ~/wvad_runs/e013 \
    --e017_runs ~/wvad_runs/e017 --e018_runs ~/wvad_runs/e018 --runs $RUNS     # g2 all_ok=True
```
Copy `$E/e021_identity.json` back and commit it. A numerical failure is not re-run; an infrastructure failure is re-run once with the same command (stage 1 again means stage 2 again).

**8. Stage-1 scoring** (server, training environment): the 48 h0 runs x 30 epoch checkpoints on the earlier validation features, `d16c`, `d16r` (UCF `d16r1`, `d16r2`) and every dense snippet; the inputs of gates (s), (s2), (p). A rerun after an infrastructure failure is the same command (finished files are kept).
```bash
setsid nohup python -m wvad.diagnostics.e021_score --runs $RUNS --out $S --stage 1 > $S.stage1.log 2>&1 < /dev/null &
```
**9. Stage-1 analysis** (local): gates (a)-(h), (s), (s2), (p), (f), the extraction gates of stage 1 with the approvals and (g1), (g2), then the stage-1 verdict, flags, descriptives and the start conditions ((3) from the stage-1 throughput in the dense caches' `meta.json`). A failed gate stops before any verdict (`results/e021_stage1_gate.json`). The epoch checkpoints stay on the server (the analysis reads the score files).
```bash
ssh <server> 'tar -cf - --exclude="ckpt_epoch_*" -C ~/wvad_runs e021 e021_scores' | tar -xf - -C <local runs root>
scp <server>:wvad_runs/e021_extract/gate_server.json $L/      # (x2), throughput, (x3), (x4), (x5)
python -c "import json, sys; from wvad.extract.__main__ import merge_record; \
[merge_record(sys.argv[1], k, v) for k, v in json.load(open(sys.argv[2], encoding='utf-8')).items()]" \
    $E/e021_extract_gate.json $L/gate_server.json
for ds in ucf xdv; do scp <server>:<server feature_root>/${ds}_xcb_d16/dense16/train/meta.json $L/meta_$ds.json; done
python -m wvad.diagnostics.e021 stage1 --runs <local runs root>/e021 --scores <local runs root>/e021_scores \
    --extract_meta $L/meta_ucf.json $L/meta_xdv.json
git add $E && git commit                                        # e021_stage1_summary.json and its gate / runs files
```
**10. Start conditions** (`start_conditions` of the summary): (1) "continue", (2) every stage-1 gate passed, (3) projected stage-2 extraction <= 24 h (otherwise the user approves or stops), (4) the training-part sha256 recheck (step 11), (5) the frozen code unchanged. Stage 2 not run, or a gate that failed without being fixed, gives the final category directly (then step 17):
```bash
python -m wvad.diagnostics.e021 verdict --not_run gate --reason '<the failed gate>'      # 관문 실패(판정 없음)
python -m wvad.diagnostics.e021 verdict --not_run time|videos|defect --reason '<..>'    # 보류(2단계 미실행: ..)
python -m wvad.diagnostics.e021 verdict                         # after a stage-1 NO-GO: 1단계 NO-GO
```
**11. Before stage 2** (server, extraction environment, the same deploy): the training-part sha256 recheck against (x6) (checked locally: a changed file fails, 보류(2단계 미실행: 영상 불일치)) and the environment against `env_extract.json`.
```bash
for ds in ucf xdv; do
  python -m wvad.extract header --dataset $ds --subset trainpart --sha_only --access_log $X/access_server.csv \
      --out $X/recheck_$ds.csv
done
python -m wvad.extract gate env --out $X/env_extract_stage2.json
python -c "import json, sys; from wvad.extract.encoders import same_environment as s; \
assert s(json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))), 'environment changed'" \
    $E/env_extract.json $X/env_extract_stage2.json
# local
scp <server>:wvad_runs/e021_extract/{recheck_ucf.csv,recheck_xdv.csv,access_server.csv} $L/
python -m wvad.extract check x6 --server $L/header_ucf.csv $L/header_xdv.csv --local $L/sha_ucf.csv \
    $L/sha_xdv.csv --access_log $L/access_server.csv --recheck $L/recheck_ucf.csv $L/recheck_xdv.csv \
    --out $L/x6_recheck.json --record $E/e021_extract_gate.json
```
**12-13. Stage-2 extraction** (server, extraction environment, same commit, environment and files): the 4,727 training-part videos, (x3) / (x4) over the whole training split, the stage-2 derived caches (`d16c`, `d16r`, `d16m`, `d16m2` with the scale match fitted on the training part; the stage-1 files kept in `stage1_val/`), (x5), the record merge (as in step 4) and the local copies of the caches.
```bash
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 60 > $X/gpu_memory_trainpart.csv &
setsid nohup bash -c "python -m wvad.extract dense --dataset ucf --phase trainpart --access_log $X/access_server.csv &&
  python -m wvad.extract dense --dataset xdv --phase trainpart --access_log $X/access_server.csv" \
  > $X/dense_trainpart.log 2>&1 < /dev/null &
pgrep -fc "wvad[.]extract dense"                                # 0 when done, then
pkill -f "nvidia-smi --query-gpu=timestamp"
for ds in ucf xdv; do
  for ph in val trainpart; do
    python -m wvad.extract check x3 --dataset $ds --phase $ph --headers $X/header_ucf.csv $X/header_xdv.csv \
        --out $X/x3_${ds}_$ph.json --record $X/gate_server.json
    python -m wvad.extract check x4 --dataset $ds --phase $ph --env $E/env_extract.json \
        --commit $(cat DEPLOYED_COMMIT) --out $X/x4_${ds}_$ph.json --record $X/gate_server.json
  done
  python -m wvad.extract derive --dataset $ds --stage 2
  python -m wvad.extract check x5 --dataset $ds --stage 2 --out $X/x5_${ds}_2.json --record $X/gate_server.json
done
# local: the record merge and the cache copies
scp <server>:wvad_runs/e021_extract/gate_server.json $L/
python -c "import json, sys; from wvad.extract.__main__ import merge_record; \
[merge_record(sys.argv[1], k, v) for k, v in json.load(open(sys.argv[2], encoding='utf-8')).items()]" \
    $E/e021_extract_gate.json $L/gate_server.json
ssh <server> 'tar -cf - -C <server feature_root> ucf_xcb_d16 ucf_xcl_d16 ucf_siglip_d16 xdv_xcb_d16 xdv_xcl_d16 \
    xdv_siglip_d16' | tar -xf - -C <local dense copy root>
```
**14. Stage-2 training** (server, training environment): 144 chains (h1 and hb seeds 0-2 and 100-102, h1r and hbm seeds 0-2; both pipelines; `--features` the head's variant and `--save_epoch_ckpts` in both stages) in 6 batches of 24 (b00-b02 UCF, b03-b05 XD), each after the memory check and its environment record (the (g1) environment and commit, (g1) and (g2) passed; gate (f)).
```bash
export RUNS Z="--lambda_dense 0 --lambda_pseudo 0 --lambda_neg_pseudo 0 --record --save_epoch_ckpts"
for ds in ucf xdv; do for pair in xcb_xcl xcb_siglip; do for p in l20 m10; do
  for h in h1 hb; do for seed in 0 1 2 100 101 102; do echo $h $p $ds $pair $seed; done; done
  for h in h1r hbm; do for seed in 0 1 2; do echo $h $p $ds $pair $seed; done; done
done; done; done > $RUNS/jobs_s2.txt                            # 144 lines, UCF first
split -l 24 -d $RUNS/jobs_s2.txt $RUNS/jobs_s2_b                # jobs_s2_b00 .. jobs_s2_b05
b=00; ds=ucf                    # then b=01 ucf, b=02 ucf, b=03 xdv, b=04 xdv, b=05 xdv, one after the other
nvidia-smi --query-gpu=memory.used,memory.total --format=csv && free -g
python -c "from wvad.diagnostics.e021_gates import record_env; \
record_env('$RUNS/env_s2_${ds}_b$b.json', ('g1', 'g2'))" || exit 1
setsid nohup xargs -P 24 -L 1 bash -c '
  h=$0; p=$1; ds=$2; pair=$3; seed=$4; c=${ds}_${pair}_s${seed}; d=$RUNS/$c; mkdir -p $d
  case $h in h1) f=d16c;; h1r) f=d16r;; hb) f=d16m;; hbm) f=d16m2;; esac
  T="--dataset $ds --pair $pair --seed $seed --features $f"
  if [ $p = m10 ]; then T="$T --global_mode mean --offset_lr_mult 10"; fi
  python -m wvad.train stage1 $T --out_dir $d/${h}_${p}_stage1 --record --save_epoch_ckpts > $d.${h}_${p}_s1.log 2>&1 &&
  python -m wvad.train stage2 $T --init $d/${h}_${p}_stage1/ckpt_last.pt --out_dir $d/${h}_$p $Z > $d.${h}_$p.log 2>&1
' < $RUNS/jobs_s2_b$b > $RUNS/queue_s2_b$b.log 2>&1 &
pgrep -fc "wvad[.]train"                                        # 0 before the next batch
ls $RUNS/*/{h1,h1r,hb,hbm}_{l20,m10}/selection.json | wc -l      # 144 after the last batch
```
**15. Stage-2 scoring** (server): the 144 runs on their own 32-unit features and (h1, hb, hbm) every dense snippet (hb, hbm through their scale-match maps, with the (s3) input).
```bash
setsid nohup python -m wvad.diagnostics.e021_score --runs $RUNS --out $S --stage 2 > $S.stage2.log 2>&1 < /dev/null &
```
**16. Final analysis** (local): the gates of both stages (and (s3)), then the final verdict (C; readout sensitivity, extreme-readout flag, tags of approximate equivalence and approved decoding exceptions), flags and descriptives: `results/e021_gate.json`, `e021_runs.csv`, `e021_verdict.json`.
```bash
ssh <server> 'tar -cf - --exclude="ckpt_epoch_*" -C ~/wvad_runs e021 e021_scores' | tar -xf - -C <local runs root>
python -m wvad.diagnostics.e021 verdict --runs <local runs root>/e021 --scores <local runs root>/e021_scores
```
**17. Verdict file, then the temporal-annotation diagnostics once** (`describe` runs only when `results/e021_verdict.json` is committed unmodified; it records the file's sha256 at its start and end). The access log, the header sha256 lists and the batch environment records are kept with the results. After a hold or no-verdict category, the user's option and whether it came before or after `describe` are appended to the verdict file (`choice`).
```bash
scp <server>:wvad_runs/e021_extract/access_server.csv $E/e021_video_access.csv
scp <server>:wvad_runs/e021_extract/gpu_memory_{val,trainpart}.csv $E/      # extraction memory (descriptive 5)
scp <server>:wvad_runs/e021/env_*.json $E/ && cp $L/header_ucf.csv $L/header_xdv.csv $E/
git add $E && git commit                                        # e021_verdict.json first
python -m wvad.diagnostics.e021 describe --runs <local runs root>/e021 --scores <local runs root>/e021_scores
git add $E/e021_describe.json && git commit
python -m wvad.diagnostics.e021 choice --option i|ii|iii|iv --quote '<the user's words>'   # hold / no verdict only
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
