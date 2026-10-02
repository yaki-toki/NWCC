# data

Small, tracked inputs. Feature caches, videos and checkpoints are **not** stored here; see `configs/paths.example.yaml`.

| File | Content | Origin |
|---|---|---|
| `splits/ucf_val_split_15.json` | UCF-Crime validation split: 15% of the 1,610 training videos, stratified by category (241 videos: 121 anomalous, 120 normal), seed 20260930 | previous project, `split_sha256` 18944283… |
| `splits/xdv_val_split_15.json` | XD-Violence validation split: 15% of the 3,950 training videos present in the feature caches, stratified by label code (592 videos: 285 violent, 307 normal), seed 20260930 | previous project, `split_sha256` 1a11521a… |

Each split stores the indices of the validation and remaining training videos in the order of the training feature cache, plus the per-video labels used to check that a cache matches the split.

- `split_sha256` (inside each JSON, quoted above) hashes the split's index content; it identifies the split. It differs from the file hashes in `SHA256SUMS`, which only guard the files against corruption.
- XD-Violence officially has 3,954 training videos. The previous project's caches hold 3,950. Its manifest records two videos that could not be decoded during extraction; the reason for the other two is not recorded. Stage B1 will document the exact list.

Test and training manifests (frame counts, temporal annotations) will be added in stage B1 with machine-specific paths removed.

`SHA256SUMS` lists the file hashes; run `sha256sum -c SHA256SUMS` from this folder to check them.
