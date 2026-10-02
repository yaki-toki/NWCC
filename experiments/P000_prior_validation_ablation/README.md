# P000 — 이전 방법의 검증 프로토콜 절제 (근거 자료, 읽기 전용)

이 저장소를 만들기 직전(2026-10-02)에 이전 프로젝트에서 수행한 사전등록 절제다. 이 결과(**NO-GO**)가 재설계의 출발점이다. 결과 파일은 바꾸지 않고 그대로 보존한다.

## 요약
- **질문**: 이전 방법의 두 구성요소가 같은 헤드·같은 학습 일정의 기본 모델보다 성능을 일관되게 올리는가.
  - 조밀 순위 손실 "TC-MIL"
  - 의사 레이블 자기학습 "CPL"
  - 보조 위험 헤드가 탐지에 필요한가도 함께 확인했다.
- **설계**
  - 2×2 + 커리큘럼 제거 1셀 + 보조 변형 4종
  - 규모: UCF-Crime·XD-Violence × XCB+XCL·XCB+SigLIP × seed 10개 = 360회
  - 선택: 검증 분할로만 선택
  - 시험셋: 학습이 모두 끝난 뒤 1회만 사용
- **판정**: NO-GO.
  - 개선은 의사 레이블에서만 나왔다. UCF에서 +1.2/+2.4 AP로 유의했고, XD에서는 보정 후 유의하지 않았다.
  - 조밀 순위 손실과 커리큘럼은 효과가 없었다.
  - 보조 헤드는 탐지를 개선하지 않았다.
- 상세 내용은 `REPORT.md`, 사전등록 규칙은 `PLAN.md`(고정본 해시 `PLAN_frozen.sha256`)에 있다.

## 파일
| 파일 | 내용 |
|---|---|
| `REPORT.md` | 결과 보고서(독립 검증 반영본) |
| `PLAN.md`, `PLAN_frozen.sha256` | 사전등록 문서와 고정본 해시. 변경 기록은 운영 사실 3건이며, 판정 규칙은 바뀌지 않았다. |
| `results/dcrc_{ucf,xdv}_runs.csv` | seed별 원자료. 실행 × 체크포인트 규칙(best/final) × 점수 신호마다 한 행이다. 열: `dataset,pair,config,feat,seed,role,signal,head,frame_auc,frame_ap,val_video_auc` |
| `results/per_run/{ucf,xdv}/*.json` | 실행별 평가 기록(360개) |
| `results/phase4_ucf_runs.csv`, `results/phase6_xdv_runs.csv` | 재현성 점검용 이전 실행 재채점 |
| `results/cells.csv`, `contrasts.csv`, `per_seed_diffs.csv`, `dcrc_stats.json`, `tables.md` | 집계, 짝 대비, Holm 보정, 판정 |
| `results/cost.json` | 파라미터 수, 추론 시간, 메모리 |
| `configs/dcrc_config.json`, `configs/commands_{ucf,xdv}.txt` | 셀 설정과 실제 실행 명령 720줄 |

## 주의
- 문서 안의 경로(`scripts/dcrc/…`, `outputs/…`, `results/dcrc_ablation/…`, `submission/…`)는 **이전 프로젝트**의 구성을 가리킨다. 이 저장소의 경로가 아니다.
- 원본 로그(약 380MB)와 체크포인트는 이전 프로젝트와 서버에 남아 있다.
- 구성 이름 대응:

| 이 실험의 이름 | 뜻 | 이전 원고의 이름 |
|---|---|---|
| `abl_base` | B, 기본 | schedule-only |
| `abl_full` | F | 보조 없는 기존 방법 |
| `aux_k3_evid` | 3헤드 증거 보조 | 원고의 기존 방법 |

- UCF 수치는 단정밀도 평가 기준이다. 원고의 반정밀도 수치와는 셀에 따라 최대 1.5 AP 다르다(`REPORT.md` 3절).
