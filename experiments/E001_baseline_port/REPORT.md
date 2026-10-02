# E001 — 진행 보고 (로드맵 단계 B)

계획: [`PLAN.md`](PLAN.md) (고정본 sha256 `0b39aa33…`, 변경 없음)

| 하위 단계 | 상태 | 결과 |
|---|---|---|
| B1 데이터 계층 | **통과** (2026-10-02) | 아래 |
| B2 평가기 | **통과** (2026-10-02) | 아래 |
| B3 모델·손실 | 미실시 | — |
| B4 학습기 + 재현 관문 | 미실시 | — |

## B1 — 데이터 계층
- **실행**: 서버, 커밋 `37ed252`
  - 실행 당시 배포한 커밋 이름은 `0b180dc`였다.
  - 이후 이력 정리로 `tests/`·`docs/`가 빠지면서 커밋 이름이 바뀌었다.
  - 실행에 관여하는 내용은 같다. `src/`, `data/`, `configs/`, `pyproject.toml`의 git 트리 해시가 두 커밋에서 일치한다.
  - 명령: `README.md` "Stage B — B1"
  - 검증 코드: `tests/equivalence/check_b1_data.py`(로컬 전용)
  - 비교 대상: 이전 프로젝트 학습기의 `MILFeatureDataset`과 `sear.dataset.val_split`
- **읽은 데이터**: 학습(`train`) 분할만. 시험 데이터는 읽지 않았다.
- **결과**: `results/b1_data_equivalence.json`

| 데이터셋 | 특징 쌍 | 영상 수(새/이전) | 차원 | 특징 불일치 | 구간 레이블 불일치 | 레이블 불일치 | 검증 부분 집합 불일치 | 분할 해시 |
|---|---|---|---|---|---|---|---|---|
| UCF | XCB+XCL | 1610/1610 | 1792 | 0 | 0 | 0 | 0/241 | 같음 (18944283…) |
| UCF | XCB+SigLIP | 1610/1610 | 1920 | 0 | 0 | 0 | 0/241 | 같음 |
| XD | XCB+XCL | 3950/3950 | 1792 | 0 | 0 | 0 | 0/592 | 같음 (1a11521a…) |
| XD | XCB+SigLIP | 3950/3950 | 1920 | 0 | 0 | 0 | 0/592 | 같음 |

- **판정**: 새 데이터 계층은 이전 로더와 비트 단위로 같다(`torch.equal`, 구간 레이블은 NaN 위치까지 비교).
- **새 코드가 더 엄격해진 점**: 인코더 캐시 사이에서 영상 목록·레이블·구간 수가 같은지 확인한다. 이전 로더는 확인하지 않았다.
- **실패·재실행**: 없음.

## B2 — 평가기
- 코드: `src/wvad/eval/metrics.py`. 단정밀도 이상(float32 점수, sklearn 계산)으로만 평가하며 반정밀도 경로는 두지 않는다.
  - `frame_labels`: 끝 프레임 포함, 영상 길이 밖은 잘라 냄
  - `expand_segments`: 남는 프레임은 마지막 구간
  - `frame_auc_ap`: 32프레임 미만 영상 제외
  - `max_video_auc`: 검증 선택용, 구간 최댓값
- 확인 방법: 로컬 시험(`tests/`, git 제외)
  - 단위 시험 6개: 규칙별 합성 입력, sklearn 기준값, 오류 처리
  - 이전 평가기 대조 4개(`tests/equivalence/test_b2_legacy_eval.py`)
    - 대조 대상(이전 프로젝트 코드): `eval_multidataset.frame_labels("xdv")`, `expand_scores`, `frame_metrics`, `eval_dcrc.ucf_frame_labels`(= `evaluate_metrics` 규칙), `val_split.max_video_auc`
    - 입력: 무작위 합성 영상 300개 × seed 3개(점수 동점 약 30% 포함)
- 결과: 프레임 레이블, 프레임 복제, 프레임 AUC·AP, 영상 AUC가 모두 이전 함수와 **정확히 같았다**. 전체 시험 43개 통과.
- 읽은 데이터: 없음. 시험 특징·주석을 읽지 않았다.
- 이전 프로젝트의 UCF 수치(원고)를 만든 반정밀도 평가 경로(`evaluate_metrics.py`, GPU autocast)는 옮기지 않았다. 그 경로와는 셀에 따라 최대 1.5 AP 차이가 난다(P000 보고서 3절).
