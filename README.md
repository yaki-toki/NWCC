# frozen-vlm-wvad

NWCC (normality-weighted context contrast) for weakly supervised video anomaly localisation on frozen vision-language features. NWCC is added to a published base anomaly detector, UR-DMU, without changing its trunk, objective or recipe: each snippet logit is corrected by a learned residual of the difference between the snippet embedding and a normality-weighted context drawn from the same video. Features come from frozen X-CLIP (base, large) and SigLIP-so400m encoders; no encoder is fine-tuned.

The method was evaluated under a pre-registered, sealed protocol: the configuration was selected on validation videos only, the base detector was chosen by a same-feature comparison of four published detectors, and UBnormal was tested once. The pre-registered criterion was **not met** (both run-inclusive intervals of the within-video localisation contrast include zero). A post-hoc analysis of the completed models, kept apart from that verdict, is reported with it. Every number in the manuscript is generated from the record files under `experiments/`; nothing is typed by hand.

The manuscript and its figure and number generators are kept out of the repository until submission ([`paper/README.md`](paper/README.md)). The repository is private at the time of writing; source videos and derived UBnormal features are not redistributed (UBnormal is CC BY-NC-ND 4.0).

## What is in the repository
| Folder | Contents |
|---|---|
| `src/wvad/` | library code and module entry points (`python -m wvad....`): dense feature extraction, data manifests, the UR-DMU port and NWCC (`baselines/urdmu_k.py`), the three other ported detectors, the experiment modules (`diagnostics/e02*.py`) and the evaluation code |
| `data/` | manifests, splits and small annotation tables (sha256 in `data/SHA256SUMS`); features and videos live outside the repository ([`data/README.md`](data/README.md)) |
| `experiments/E021_dense_snippet_pilot/` | dense 16-frame snippet extraction of the training split: gates, environment, video access log |
| `experiments/E022_dense_confirmation/` | extraction of the UCF-Crime and XD-Violence test splits: gates, access logs, score sha256 |
| `experiments/E023_localisation_baselines/` | the base anomaly detector choice: four ported detectors on the same features, equivalence checks against their official code, selection, gates, verdict |
| `experiments/E024_localisation_method/` | NWCC: stage 0 (UR-DMU reference chains), stage 1 (candidate selection and controls, validation only), stage 2 (UBnormal preparation), stage 3 (sealed confirmation, `results/e024_stage3.json`), two post-hoc rounds (`posthoc_20261026/`, `posthoc_20261027/`) and the data-figure cases (`figures_20261027/`) |
| `configs/` | `paths.example.yaml` (copy to `paths.yaml`, git-ignored) |
| `docs/`, `tests/` | local only, git-ignored: planning notes, unit tests, equivalence checks |

Each experiment folder holds its plan (`PLAN.md`, sealed by the commit that adds `PLAN.sha256` alone), its small result files and its report ([`experiments/README.md`](experiments/README.md)). The development history that preceded these four experiments (E001–E020 and the prior ablation P000) was removed from the tree on 2026-10-28 and remains in the commit history; the manuscript claims nothing from it.

## Principles
- **Sealed test sets.** Model, checkpoint, threshold and hyperparameter choices use only held-out training videos (validation split). UBnormal was read once, in the sealed stage-3 scoring. The UCF-Crime and XD-Violence test splits were read three times across E022, E023 and the E024 auxiliary reading (non-independent; recorded in the plans).
- **Everything recorded.** Each experiment keeps its plan, small result tables (CSV/JSON), gate and environment records and its verdict under `experiments/`. Negative results stay.
- **Commands live in this README.** There are no wrapper scripts. The blocks below are the commands as run; the full gate-by-gate procedure of each experiment is in its `PLAN.md`.

## Setup
```bash
pip install torch                                  # CUDA build of your choice
pip install -e ".[dev]"
cp configs/paths.example.yaml configs/paths.yaml   # feature caches, raw videos, annotations
```
`configs/paths.yaml` needs `feature_root` (the dense caches and derived training inputs), `raw_video_root` (the folder holding `Anomaly-Detection-Dataset/`, `XD-Violence/` and `UBnormal/`) and the annotation paths. The dense extraction runs in its own pinned environment (`pip install -e ".[extract]"` after the CUDA `torch`): `transformers==5.0.0`, `opencv-python==4.13.0.90`, `numpy==2.2.6`.

### Server conventions
```bash
# deploy the committed code (non-destructive: configs/paths.yaml and outputs on the server are kept)
git archive --format=tar HEAD | ssh <server> 'mkdir -p ~/frozen-vlm-wvad && tar -x -C ~/frozen-vlm-wvad'
git rev-parse HEAD | ssh <server> 'cat > ~/frozen-vlm-wvad/DEPLOYED_COMMIT'   # commit stamp for results made there
# on the server, in ~/frozen-vlm-wvad, with the Python environment active
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE
# extraction commands (video decoding, encoders) run in the extraction environment instead
conda activate wvad-extract && export PYTHONPATH=src HF_HUB_OFFLINE=1 KMP_DUPLICATE_LIB_OK=TRUE
# long runs: setsid nohup <command> > <log outside the repo> 2>&1 < /dev/null &
```
Every analysis command checks that the tree is clean and that the deployed commit is the experiment's freeze commit or a descendant that changed none of its fixed code; the sealed commits are recorded in the plans.

## Pipeline

### 1. Dense snippet features (E021, [plan](experiments/E021_dense_snippet_pilot/PLAN.md))
The training split of UCF-Crime and XD-Violence is extracted as gap-free 16-frame snippets, one cache per dataset and encoder under `<feature_root>/<dataset>_<encoder>_d16/` (phases `val`, the validation videos, and `trainpart`, the training part). Every video is opened only through the training manifest and appended to an access log first. Headers are hashed on both machines and compared.
```bash
E=experiments/E021_dense_snippet_pilot/results
X=~/wvad_runs/e021_extract                                      # server records outside the repository
python -m wvad.extract gate env --out $E/env_extract.json       # pip freeze, CUDA, cuDNN, driver, TF32, attention
for ds in ucf xdv; do
  python -m wvad.extract header --dataset $ds --access_log $X/access_server.csv --out $X/header_$ds.csv
  python -m wvad.extract dense --dataset $ds --phase val --access_log $X/access_server.csv
  python -m wvad.extract dense --dataset $ds --phase trainpart --access_log $X/access_server.csv
  python -m wvad.extract check x3 --dataset $ds --phase trainpart --headers $X/header_ucf.csv $X/header_xdv.csv \
      --out $X/x3_${ds}_trainpart.json --record $X/gate_server.json
  python -m wvad.extract check x4 --dataset $ds --phase trainpart --env $E/env_extract.json \
      --commit $(cat DEPLOYED_COMMIT) --out $X/x4_${ds}_trainpart.json --record $X/gate_server.json
  python -m wvad.extract derive --dataset $ds --stage 2           # the 32-unit caches derived from the snippets
done
```
The gate records (`e021_extract_gate.json`), the header hashes, the access log and the extraction environment are committed under `$E`.

### 2. Test-split features (E022, [plan](experiments/E022_dense_confirmation/PLAN.md))
The 1,090 UCF-Crime and XD-Violence test videos are extracted into `<feature_root>/<dataset>_<encoder>_d16_test/` by an access device that checks the freeze commit, the plan hash and the committed selection before any video is opened, and logs every read.
```bash
LOG=~/wvad_runs/e022/test_access.log; X=~/wvad_runs/e022_extract
A="--test_log $LOG --record $X/gate_server.json --selection_sha256 $SEL --freeze_commit $FREEZE"
python -m wvad.extract.testsplit header $A --video_log $X/video_access.csv --out $X/header_test.csv
python -m wvad.extract.testsplit dense $A --video_log $X/video_access.csv --access_log $X/extract_access.csv
python -m wvad.extract.testsplit derive $A
```
`e022_extract_gate.json`, the access CSVs, `e022_test_access.log` and `e022_test_scores.sha256` under `experiments/E022_dense_confirmation/results/` are the record of this read; E023 and E024 check the test caches against them before every later read.

### 3. Derived training inputs of the ported detectors (E023)
Each ported recipe trains on its official unit, derived from the snippets on the training split only (`<feature_root>/<dataset>_<encoder>_d16_e023/<variant>/`): `d16m32l2` for MIL, `d16m32` for RTFM, `d16m200` for UR-DMU and NWCC, `d16p256` for the VadCLIP visual branch. The UBnormal inputs are derived in stage 2 of E024 (below).
```bash
for ds in ucf xdv; do
  python -m wvad.extract.derive_n --dataset $ds --variants d16m32l2 d16m32 d16m200 d16p256 --record $RUNS/e023_train_inputs.json
done
```

### 4. Base anomaly detector choice (E023, [plan](experiments/E023_localisation_baselines/PLAN.md))
Four published detectors (MIL, RTFM, UR-DMU, the VadCLIP visual branch) are ported with their official network, loss, optimiser and schedule (`src/wvad/baselines/`, one file per method, porting contract in `common.py`) and compared on the same frozen features with validation-only selection. The ports were checked against the official code before the freeze (harnesses `wvad.baselines.check_<method>`, loader `wvad.baselines.official`; the official repositories are cloned outside the repository under `WVAD_OFFICIAL_ROOT` at the commits of `results/e023_licences.json`, and nothing from them is copied). UR-DMU had the highest within-video localisation on both datasets and became the base anomaly detector.
```bash
E=experiments/E023_localisation_baselines/results; RUNS=~/wvad_runs/e023; S=~/wvad_runs/e023_scores
export CUBLAS_WORKSPACE_CONFIG=:4096:8                           # deterministic chains
python -m wvad.baselines.recipes --out $E/e023_recipes.json       # the official recipes, as ported
# one chain: method, unit, dataset, encoder pair, seed (lr and scale multipliers from the validation check D4)
python -m wvad.baselines.train --method urdmu --unit 200 --dataset ucf --pair xcb_xcl --seed 42 \
    --lr_mult 1 --scale_mult 1 --out_dir $RUNS/ucf_xcb_xcl_s42/urdmu_u200 --n_chunk 4096 --deterministic
python -m wvad.diagnostics.e023_score jobs --stage confirm --runs_root $RUNS --family "mil rtfm urdmu vadclip_v"   # the chain list
# validation scoring and selection, the training gates, then the one sealed test read
python -m wvad.diagnostics.e023_score score --split val --runs_root $RUNS --out $S --family "mil rtfm urdmu vadclip_v"
python -m wvad.diagnostics.e023_score select --runs_root <local runs root>/e023 --scores <local runs root>/e023_scores --family "mil rtfm urdmu vadclip_v"
python -m wvad.diagnostics.e023 gates --stage train --runs <local runs root>/e023 --scores <local runs root>/e023_scores \
    --e021_runs <local runs root>/e021 --junit <pytest junit xml> --n_chunk 4096
python -m wvad.diagnostics.e023_score score --split test --runs_root $RUNS --out $S --test_log $RUNS/test_access.log \
    --selection_sha256 $SEL --all_ckpts_sha256 $ALL --memory_need_gb <gate_m_need_gb> --n_chunk 4096 --freeze_commit $FREEZE
python -m wvad.diagnostics.e023 gates --stage v3 --scores <local runs root>/e023_scores --test_log <local test access log> \
    --access0 <access0.json> --n_chunk 4096
python -m wvad.diagnostics.e023 analyze --runs <local runs root>/e023 --scores <local runs root>/e023_scores \
    --e022_scores <local runs root>/e022_scores --test_log <local test access log>
```
Outputs: `e023_selection.json`, `e023_gate.json`, `e023_summary.json`, `e023_runs.csv` (the per-run L and pooled metrics behind the base detector tables) and `e023_test_access.log` under `$E`.

### 5. NWCC (E024, [plan](experiments/E024_localisation_method/PLAN.md))
NWCC lives in `src/wvad/baselines/urdmu_k.py`: it adds `k_t = g(x_t − c_t)` to the UR-DMU snippet logit, where `c_t` is the normality-weighted context of the other snippets of the same sequence (duplicate `d16m200` rows masked) and `g` a 65,793-parameter MLP drawn from its own generator, so the trunk, batch order and dropout draws equal those of the UR-DMU run with the same seed. E024 ran in four stages: stage 0 (reference chains), stage 1 (candidate selection and controls), stage 2 (UBnormal preparation) and stage 3 (the sealed confirmation). All stages run in deterministic mode (`--deterministic`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) and refuse test paths until the sealed stage-3 read.

**Stage 0 — reference chains** ([plan](experiments/E024_localisation_method/stage0/PLAN.md)): UR-DMU at its official recipe, development seeds 0-2, four settings.
```bash
E=experiments/E024_localisation_method/results; U0=~/wvad_runs/e024/stage0
python -m wvad.diagnostics.e024_stage0 pins --e021_scores <local runs root>/e021_scores      # inputs record, then the freeze
python -m wvad.diagnostics.e024_stage0 freeze
python -m wvad.baselines.train --method urdmu --unit 200 --dataset ucf --pair xcb_xcl --seed 0 \
    --lr_mult 1 --scale_mult 1 --n_chunk 4096 --deterministic --out_dir $U0/ucf_xcb_xcl_s0/urdmu_u200   # x 12 (ucf, xdv x 2 pairs x 3 seeds)
python -m wvad.diagnostics.e024_stage0 gate-b --a $U0/ucf_xcb_xcl_s0/urdmu_u200 --b <repeat chain> --out $U0/gate_b.json
python -m wvad.diagnostics.e024_stage0 gate-h --runs $U0 --e023_runs ~/wvad_runs/e023 --out $U0/gate_h.json
python -m wvad.diagnostics.e024_stage0 analyze --e021_scores <local runs root>/e021_scores --runs <local runs root>/e024/stage0
```

**Stage 1 — candidate selection and controls, validation only** ([plan](experiments/E024_localisation_method/stage1/PLAN.md)): six NWCC configurations (ρ ∈ {g, 0.1, 0.02} × τ ∈ {1, 0.25}) on the four settings and three seeds (72 chains), the selection rule fixed in the plan, then the required controls (C1 `mlp`, C6 `trainonly`) and the no-training smoothing control; `rescore` checks every chain's stored logits bit for bit and writes the k = 0 re-scoring.
```bash
RUNS1=~/wvad_runs/e024/stage1
python -m wvad.diagnostics.e024_stage1 pins --u_runs <local runs root>/e024/stage0 && python -m wvad.diagnostics.e024_stage1 freeze
python -m wvad.diagnostics.e024_stage1 train --dataset ucf --pair xcb_xcl --seed 0 --rho g --tau 1 --mode off --out_dir $RUNS1/ucf_xcb_xcl_s0/gate_k0
python -m wvad.diagnostics.e024_stage1 gate-k0 --runs $RUNS1 --u_runs $U0 --out $RUNS1/gate_k0.json      # must pass before any NWCC chain
python -m wvad.diagnostics.e024_stage1 train --dataset <ds> --pair <pair> --seed <s> --rho <rho> --tau <tau> --mode k \
    --out_dir $RUNS1/<ds>_<pair>_s<s>/k_r<rho>_t<tau>                                                  # x 72
python -m wvad.diagnostics.e024_stage1 gate-b --runs $RUNS1 --out $RUNS1/gate_b.json
python -m wvad.diagnostics.e024_stage1 gate-h --runs $RUNS1 --u_runs $U0 --out $RUNS1/gate_h.json
python -m wvad.diagnostics.e024_stage1 gate-e --runs $RUNS1 --u_runs $U0 --out $RUNS1/gate_e.json
for d in $RUNS1/*/k_r*/ $RUNS1/*/ctl_*/; do python -m wvad.diagnostics.e024_stage1 rescore --chain ${d%/}; done
python -m wvad.diagnostics.e024_stage1 analyze --u_runs <local runs root>/e024/stage0 --runs <local runs root>/e024/stage1          # stage A
python -m wvad.diagnostics.e024_stage1 train --dataset <ds> --pair <pair> --seed <s> --rho <rho*> --tau <tau*> --mode mlp|trainonly \
    --out_dir $RUNS1/<ds>_<pair>_s<s>/ctl_<mode>_r<rho*>_t<tau*>                                        # x 24, the controls
python -m wvad.diagnostics.e024_stage1 smooth --u_runs <local runs root>/e024/stage0 --runs <local runs root>/e024/stage1
python -m wvad.diagnostics.e024_stage1 analyze --u_runs <local runs root>/e024/stage0 --runs <local runs root>/e024/stage1 \
    --stage_a $E/e024_stage1.json                                                                       # stage B
```
The selected configuration was `k_rg_t0.25` and the final control `ctl_trainonly_rg_t0.25` (`results/e024_stage1.json`).

**Stage 2 — UBnormal preparation** ([plan](experiments/E024_localisation_method/stage3/PLAN.md), the stage-2 part): UBnormal is registered as the dataset `ubn`. The archive is checked by sha256 and unpacked under `raw_video_root/UBnormal/Scene<s>/`; the six official split lists (sha256 pinned in `wvad.data.ubnormal`) and the video headers give `data/manifests/ubn_train.json` (official training then validation videos, 332), `data/splits/ubn_val_official.json` (64) and `data/manifests/ubn_test.json` (211 videos, no intervals: the tracks files are read once, at the sealed scoring). The dense stores and the `d16m200` inputs are extracted as for UCF-Crime and XD-Violence, and the UR-DMU development chains (seeds 0-2, learning-rate multiplier 0.5, 1 or 2) fix the collapse threshold and the multiplier of stage 3 before its freeze. No test value is read.
```bash
S2=~/wvad_runs/e024/stage2
sha256sum -c <raw>/ubnormal/UBnormal.zip.sha256 && unzip -q <raw>/ubnormal/UBnormal.zip -d <raw>/UBnormal
for f in normal_training abnormal_training normal_validation abnormal_validation normal_test abnormal_test; do
  curl -sS -o <raw>/ubnormal_meta/${f}_video_names.txt https://raw.githubusercontent.com/lilygeorgescu/UBnormal/main/scripts/${f}_video_names.txt; done
python -m wvad.prep.ubnormal manifests --lists <raw>/ubnormal_meta --access_log $S2/access_server.csv
python -m wvad.prep.ubnormal tracks-check --lists <raw>/ubnormal_meta --access_log $S2/access_server.csv --out $S2/tracks_check.json
(cd data && sha256sum manifests/ubn_train.json manifests/ubn_test.json splits/ubn_val_official.json >> SHA256SUMS && sha256sum -c SHA256SUMS)
python -m wvad.extract header --dataset ubn --access_log $S2/access_server.csv --out $S2/header_ubn.csv
python -m wvad.extract dense --dataset ubn --phase val --access_log $S2/access_server.csv
python -m wvad.extract dense --dataset ubn --phase trainpart --access_log $S2/access_server.csv
python -m wvad.prep.ubnormal extract-test --access_log $S2/access_server.csv          # ubn_<encoder>_d16_test, pixels only
python -m wvad.extract.derive_n --dataset ubn --variants d16m200 --record $E/e024_stage3_train_inputs.json
for m in 0.5 1 2; do for pair in xcb_xcl xcb_siglip; do for seed in 0 1 2; do
  python -m wvad.baselines.train --method urdmu --unit 200 --dataset ubn --pair $pair --seed $seed --out_dir $S2/${pair}_s${seed}/urdmu_u200_lr$m \
      --lr_mult $m --scale_mult 1 --n_chunk 4096 --deterministic
done; done; done
```

**Stage 3 — sealed confirmation** ([plan](experiments/E024_localisation_method/stage3/PLAN.md)): arms P (`k_rg_t0.25`), U (`urdmu_u200`) and C6 (`ctl_trainonly_rg_t0.25`) on UBnormal (primary) and UCF-Crime and XD-Violence (auxiliary, non-independent), two encoder pairs, new seeds 3-5 (54 chains). The learning-rate multiplier of the UBnormal chains comes from the stage-2 development chains; `score-test` is the one sealed test read and refuses a second run.
```bash
S3=~/wvad_runs/e024/stage3; SC=~/wvad_runs/e024/stage3_scores
python -m wvad.diagnostics.e024_stage3 pins --u_dev <local runs root>/e024/stage2 --lists <local runs root>/e024/stage2/ubnormal_meta \
    --tracks_check <local runs root>/e024/stage2/tracks_check.json --zip_sha256 <sha256 of UBnormal.zip>
python -m wvad.diagnostics.e024_stage3 freeze                    # commit stage3/PLAN.sha256 alone, deploy exactly that commit
M=$(python -c "import json; print(json.load(open('$E/e024_stage3_inputs.json'))['multiplier']['chosen_lr_mult'])")
# per cell <ds> <pair> <seed> (m = $M for ubn, 1 otherwise): U first, then P and C6
python -m wvad.baselines.train --method urdmu --unit 200 --dataset <ds> --pair <pair> --seed <s> --out_dir $S3/<cell>/urdmu_u200 \
    --lr_mult <m> --scale_mult 1 --n_chunk 4096 --deterministic
python -m wvad.diagnostics.e024_stage3 train --dataset <ds> --pair <pair> --seed <s> --arm p --lr_mult <m> --scale_mult 1 --out_dir $S3/<cell>/k_rg_t0.25
python -m wvad.diagnostics.e024_stage3 train --dataset <ds> --pair <pair> --seed <s> --arm c6 --lr_mult <m> --scale_mult 1 --out_dir $S3/<cell>/ctl_trainonly_rg_t0.25
python -m wvad.diagnostics.e024_stage3 train --dataset ubn --pair xcb_xcl --seed 3 --arm k0 --lr_mult $M --scale_mult 1 --out_dir $S3/ubn_xcb_xcl_s3/gate_k0
python -m wvad.diagnostics.e024_stage3 gate-k0 --runs $S3
python -m wvad.diagnostics.e024_stage3 gates --runs $S3
python -m wvad.diagnostics.e024_stage3 score-test --runs $S3 --out $SC --access_log $S3/test_access.csv --freeze_commit $FREEZE
python -m wvad.diagnostics.e024_stage3 analyze --runs <local runs root>/e024/stage3 --scores <local runs root>/e024/stage3_scores \
    --access_log <local runs root>/e024/stage3/test_access.csv
```
Outputs: `results/e024_stage3.json` (outcome row `not_met`; the two contrasts' classes and intervals; the auxiliary readings) and `results/e024_stage3_runs.csv`. No arm, seed, statistic, criterion or wording rule changed after the test was read.

### 6. Post-hoc analysis (exploratory, after the verdict)
The first round (`posthoc_20261026/`, scripts beside its results) compares the arms on the review-budget metrics of the stored test scores. The second round re-scores the completed P checkpoints with uniform context weights in a separate deployed copy, after reproducing the sealed logits bit for bit, and compares every arm with four untrained temporal rules (`posthoc_20261027/`). Neither touches the sealed scores, the access log or the verdict.
```bash
python -m wvad.diagnostics.e024_posthoc score-uniform --runs ~/wvad_runs/e024/stage3 --scores ~/wvad_runs/e024/stage3_scores \
    --out ~/wvad_runs/e024/posthoc_uniform --access_log ~/wvad_runs/e024/posthoc_access.csv       # server, separate deployed copy
python -m wvad.diagnostics.e024_posthoc analyze                                                   # local -> posthoc_20261027/results.json, per_video.csv
```

## Paper
The manuscript, its figure and number generators and their commands are local until submission ([`paper/README.md`](paper/README.md)). Every value in it is a macro generated from the record files above.

## Local checks (not tracked)
```bash
pytest            # unit tests in tests/
ruff check .
```
The UR-DMU equivalence tests run the official attention, which imports the real `einops`: install it, or set `WVAD_EINOPS_DIR` to a folder holding a copy of it outside the repository; without either they are skipped (with that reason).

## License
MIT (see [LICENSE](LICENSE)). Third-party annotations and datasets keep their own licenses.
