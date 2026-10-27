# Final method: training scope, parameters and performance (record, 2026-10-27)

Generated from `experiments/E023_localisation_baselines/results/e023_summary.json` (descriptives 1_cells, 7_efficiency) and `experiments/E024_localisation_method/results/e024_stage3.json` (runs: parameters, L). The manuscript takes the same numbers through `wvad.paper.numbers` (macros `\EthreeParams*`, `\EthreeTrainMin*`, `\EtenParams*`).

## 1. Training scope of the final proposed detector

| Item | Final proposed detector (Proposed = UR-DMU + context-contrast module) |
|---|---|
| Frozen | the two VLM encoders of a pair (X-CLIP-B/16 + X-CLIP-L/14, or X-CLIP-B/16 + SigLIP-So400m); features extracted once and cached; no fine-tuning (a condition shared by every compared method) |
| Trained | the whole UR-DMU detector (embedding, temporal convolution, global-local attention, memories, classifier) together with the module g (two-layer perceptron 512 -> 128 -> 1), from the UR-DMU initialisation and recipe |
| Trainable parameters | 7,738,370 (X-CLIP pair) / 7,938,818 (SigLIP pair) |
| UR-DMU alone | 7,672,577 / 7,873,025 |
| Added by the module | 65,793 (0.86% of UR-DMU) / 65,793 (0.84%) |
| Training needed on a new dataset | full UR-DMU training with the module (official UR-DMU recipe); the module is not attached to a trained UR-DMU checkpoint without training (Correction-off removes it from a trained detector; nothing adds it) |
| Inference | offline per video or piece (the context needs the piece being scored), as UR-DMU's self-attention already does; the extra cost of the module beyond UR-DMU was not measured |

Supported by the record: frozen features and no encoder fine-tuning; the module adds 65,793 trainable parameters (under 1% of UR-DMU). Not supported: a training-free method; a small overall trainable model (UR-DMU itself is 7.67-7.87 M); a shorter training time or a lower end-to-end video cost (not measured under controlled conditions).

## 2. Published detectors on the same frozen features (E023; one run per setting; validation-only selection; sealed single test read)

| Method | Trainable params, X-CLIP / SigLIP pair | Train wall time, min (mean +- sd over the 4 settings; shared GPU) | UCF-Crime L (AUC) | UCF-Crime frame AUC | XD-Violence L (AP) | XD-Violence frame AP |
|---|---|---|---|---|---|---|
| MIL | 0.93 M / 1.00 M | 15 +- 0 | 70.79 | 85.96 | 72.37 | 77.90 |
| RTFM | 21.70 M / 23.22 M | 177 +- 7 | 67.86 | 84.67 | 63.70 | 63.24 |
| UR-DMU | 7.67 M / 7.87 M | 18 +- 2 | 70.93 | 86.26 | 73.19 | 79.24 |
| VadCLIP visual | 88.81-146.17 M (depends on dataset and pair) | 103 +- 46 | 58.47 | 71.28 | 68.53 | 73.22 |

Per-setting values (X-CLIP pair / SigLIP pair):

| Method | UCF L | UCF AUC | XD L | XD AP |
|---|---|---|---|---|
| MIL | 69.72 / 71.87 | 85.29 / 86.63 | 70.53 / 74.21 | 75.83 / 79.97 |
| RTFM | 63.52 / 72.21 | 82.92 / 86.42 | 70.56 / 56.84 | 72.54 / 53.95 |
| UR-DMU | 70.60 / 71.27 | 84.48 / 88.04 | 70.47 / 75.90 | 75.64 / 82.84 |
| VadCLIP visual | 68.72 / 48.21 | 84.99 / 57.57 | 62.42 / 74.63 | 76.78 / 69.67 |

UR-DMU has the highest L on both datasets and in all four settings (the pre-registered rule B*). Pairwise intervals between these four detectors were not part of the pre-registered analysis (its contrasts were relative to the development head; supplement Table S11); the levels are one run per setting. Published numbers of these methods come from other features (I3D, CLIP) and test-set selection and are not combined with this table.

## 3. The proposed detector and UR-DMU under the confirmation protocol (E024 stage 3; seeds 3-5, three runs per setting; a different protocol, hence a separate table)

| Dataset | Arm | Params, X-CLIP / SigLIP | L, X-CLIP pair (3-run mean) | L, SigLIP pair | L, both pairs |
|---|---|---|---|---|---|
| ubn | UR-DMU | 7.67 M / 7.87 M | 58.38 | 65.42 | 61.90 |
| ubn | Proposed | 7.74 M / 7.94 M | 63.56 | 68.37 | 65.96 |
| ucf | UR-DMU | 7.67 M / 7.87 M | 70.73 | 71.42 | 71.07 |
| ucf | Proposed | 7.74 M / 7.94 M | 71.42 | 72.02 | 71.72 |
| xdv | UR-DMU | 7.67 M / 7.87 M | 71.38 | 73.53 | 72.45 |
| xdv | Proposed | 7.74 M / 7.94 M | 71.34 | 73.02 | 72.18 |

The pre-registered L contrasts (Proposed - UR-DMU; video x run intervals) are in `results/e024_stage3.json`: inconclusive on all three datasets (verdict not_met). Confirmation-run wall times: supplement Table S4 (`reporting/cost_sources.json`).

## 4. Cost records: what exists and what does not

- Exists: trainable parameter counts of every arm (E023 and stage-3 chain records); training wall time per E023 run (shared GPU, one run per setting) and per confirmation run (Table S4); the extraction throughput sample (`reporting/cost_sources.json`).
- Does not exist: a controlled training-time comparison (the GPU was shared); GPU memory; inference latency of the module beyond UR-DMU; end-to-end cost from raw video (feature extraction dominates and is common to every method).
- Not applicable: timings of earlier development heads (different input processing); they are not transferred to the final method.

## 5. Figure

`paper/figs/fig_base.pdf` (`python -m wvad.paper.result_figs --only base`): L against trainable parameters for the four published detectors per setting (log axis). The confirmation runs are not drawn in the same panel (different protocol); their values are in Section 3 above and in the manuscript's Tables III, V and VI.

