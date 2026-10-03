# data

Small, tracked inputs. Feature caches, videos and checkpoints are **not** stored here; see `configs/paths.example.yaml`.

| File | Content | Origin |
|---|---|---|
| `splits/ucf_val_split_15.json` | UCF-Crime validation split: 15% of the 1,610 training videos, stratified by category (241 videos: 121 anomalous, 120 normal), seed 20260930 | previous project, `split_sha256` 18944283… |
| `splits/xdv_val_split_15.json` | XD-Violence validation split: 15% of the 3,950 training videos present in the feature caches, stratified by label code (592 videos: 285 violent, 307 normal), seed 20260930 | previous project, `split_sha256` 1a11521a… |
| `manifests/ucf_train.json` | The 1,610 UCF-Crime training videos in feature-cache order: name, category, label, frame count and fps (read from the video headers with OpenCV, as at extraction), path inside the dataset root | previous project's training list; `python -m wvad.prep.manifests ucf` |
| `manifests/xdv_train.json` | The 3,950 XD-Violence training videos in feature-cache order: name, label code, label, frame count | previous project's training list; `python -m wvad.prep.manifests xdv` |
| `annotations/ucf_train_uca_proxy.json` | UCA proxy intervals (frames) for the anomalous UCF-Crime **training** videos: sentences matching the category lexicon (M1 substring, M2 word-start), one interval per sentence (S), their union (A) and hull (B). Diagnostics only; never used for training or model selection | derived from UCA; `python -m wvad.diagnostics.uca` |

Each split stores the indices of the validation and remaining training videos in the order of the training feature cache, plus the per-video labels used to check that a cache matches the split.

- `split_sha256` (inside each JSON, quoted above) hashes the split's index content; it identifies the split. It differs from the file hashes in `SHA256SUMS`, which only guard the files against corruption.
- XD-Violence officially has 3,954 training videos. The previous project's caches hold 3,950. The two videos that could not be decoded during extraction are listed in `manifests/xdv_train.json` (`meta.skipped_unreadable`); the other two are not identified in the previous records.
- Manifests list training videos only. Test manifests and test temporal annotations are not stored here before the confirmation experiment (stage F); the test set stays sealed.

## UCA annotations
The proxy intervals are derived from UCA (UCF-Crime Annotation; Yuan et al., "Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges", CVPR 2024, arXiv:2309.13925), released under Apache-2.0 for academic and research purposes only (https://github.com/Xuange923/Surveillance-Video-Understanding). The raw UCA files are not redistributed here (`configs/paths.yaml`, `annotations.uca`). The UCF-Crime videos keep their own terms.

Known limits of the proxy: only the onset of the M1 rule was compared with test annotations in the previous project (median absolute error 0.053 of the video length); interval ends and lengths are unverified, and sentences can be longer than the events they describe. Coverage differs strongly by category (see `experiments/E002_diagnostics/results/c0_proxy_by_category.csv`).

`SHA256SUMS` lists the file hashes; run `sha256sum -c SHA256SUMS` from this folder to check them.
