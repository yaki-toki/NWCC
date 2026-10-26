from analyze import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data=json.loads((ROOT/'results.json').read_text());follow=json.loads((ROOT/'followup.json').read_text())
rows=data['rows']; ubrows=[r for r in rows if r['dataset']=='ubn' and r['population']=='overall']
def row(metric,contrast,d='ubn'):
 return next(r for r in rows if r['dataset']==d and r['population']=='overall' and r['metric']==metric and r['contrast']==contrast)
def policy(metric,contrast,d='ubn'):
 return next(r for r in follow['datasets'][d]['policy_results'] if r['metric']==metric and r['contrast']==contrast)
def vals(getter,metric):
 return [getter(metric,'P-U')['B_mean'],getter(metric,'P-C6')['B_mean'],getter(metric,'P-I2')['B_mean'],getter(metric,'P-U')['P_mean']]
def ci(x):return f'[{x[0]:+.2f}, {x[1]:+.2f}]'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12.2,4.9),gridspec_kw={'width_ratios':[1.1,1]})
ax=axes[0];x=np.arange(4);w=.35
for dx,v,label,color in [(-w/2,vals(row,'precision10'),'All six fixed models','#2878a0'),(w/2,vals(policy,'precision10'),'Validation-selected models','#d87736')]:
 bars=ax.bar(x+dx,v,w,label=label,color=color)
 ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
ax.axhline(follow['datasets']['ubn']['random_precision10'],color='#888888',ls=':',lw=1.3,label='Uniform review')
ax.set_xticks(x,['UR-DMU','C6','P with K off\n(same checkpoint)','P'])
ax.set_ylabel('Anomaly precision in top 10% of video time (%)')
ax.set_ylim(0,95);ax.set_title('UBnormal: quality of the review shortlist',loc='left',fontweight='bold');ax.legend(loc='lower left',fontsize=8,frameon=True,facecolor='white',framealpha=.95)
ax=axes[1]
for c,color,marker in [('P-U','#2878a0','o'),('P-C6','#8e6c9e','s'),('P-I2','#d87736','D')]:
 y=row('precision10',c)['by_seed'];ax.plot(SEEDS,y,label=c,color=color,marker=marker,lw=2)
ax.axhline(0,color='#888888',lw=1);ax.set_xticks(SEEDS);ax.set_xlabel('Training seed (encoder pairs averaged)')
ax.set_ylabel('Precision@10% gain (percentage points)');ax.set_title('Gains vary substantially across training seeds',loc='left',fontweight='bold');ax.legend(frameon=False)
fig.suptitle('Completed-model exploration: better prioritization, uncertain training robustness',fontsize=13,fontweight='bold',y=.98)
fig.text(.06,.025,'Post hoc | 112 mixed-label anomalous videos | Equal video weights | No retraining | Original confirmation remains not_met',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.08,1,.91));fig.savefig(ROOT/'capability_summary.png',dpi=180);fig.savefig(ROOT/'capability_summary.pdf');plt.close(fig)

def table(headers,lines):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in line)+'</tr>' for line in lines)+'</tbody></table>'

intro='''E024 완료 모델 사후 분석 — 2026-10-26

결론
UBnormal의 현재 학습 완료 모델에서는 높은 점수의 후보를 먼저 검토할 때 실제 이상 프레임의 비율이 개선된다. 같은 P 체크포인트에서 K만 끈 I2보다 P가 우수한 경우가 있어, 직접 추론 보정 효과를 일괄적으로 없다고 해석해서는 안 된다. 다만 학습 seed가 달라도 개선된다는 확정 근거는 부족하다. 원래 사전 등록 판정 not_met는 유지한다.

분석 범위와 검증
54개 학습 사슬의 저장 점수, P의 18개 동일 체크포인트 I2 점수를 재사용했다. 60개 봉인 점수/라벨/요약 파일의 해시를 검증했고, 원래 L 값 54개 및 I2 값 18개를 1e-9 오차 이내에서 재현했다. 재학습·재추론·원고·봉인 코드 수정은 없다. 정확한 원래 프레임 정답을 사용했고, 스니펫별 이상 프레임 수 및 UBnormal tracks 해시를 대조했다.
UBnormal의 원래 L 평가 대상은 정상·이상 프레임이 함께 있는 이상 영상 112개이다. 전체 시험 211개나 이상 영상 158개 전체에 대한 end-to-end 검출 성능이 아니다. 각 영상 및 두 특징 쌍·세 seed에 같은 가중치를 주며, 여섯 모델의 점수를 합친 앙상블은 아니다.

핵심 평가 정의
Precision@10%: 각 영상에서 모델 점수가 높은 프레임을 총 영상 길이의 10%만큼 선택한 뒤, 선택 프레임 중 실제 이상 프레임의 비율. 동점은 균등 분할하며 정답을 보고 동점을 고르지 않는다. 하나의 연속 구간을 요구하지 않고, 이동/전환 비용도 모델링하지 않는다.
Recall@5%: 같은 방식으로 상위 5%를 검토할 때 영상 안 전체 이상 프레임 중 회수한 비율.
평균 TPR(FPR 0–5%): ROC의 해당 구간 면적을 0.05로 나눈 값. sklearn의 McClish 표준화 pAUC와 다르며 이를 역변환해 구현을 검증했다. 영상마다 정답으로 ROC를 구성한 지표로, 배포 시 공통 임계값으로 5% 오탐률이 보장된다는 뜻은 아니다.
이 지표들을 수학적으로 새 지표라고 주장하지 않는다. 방법의 실제 사용 목적을 평가하는 표준적 순위/검토 예산 지표로 제시한다.

사후 탐색과 불확실성
첫 분석은 기존 결과를 본 뒤, 새 수치 계산 전에 18종 지표·3대비·3데이터셋·이상 비율 구간을 정했다. 총 594개 대비를 모두 저장했다. 평균값이 유리한 결과만 제거/선택해 저장하지 않았다. 5,000회 paired bootstrap으로 (1) 현재 모델들을 고정한 영상 변동, (2) 영상과 seed를 교차 재표본한 변동을 구분했다. 지표/대비 내 max-standardized-deviation, 모집단·데이터셋 간 Bonferroni를 사용한 탐색적 동시 구간도 제공한다. 이는 독립 확증을 대체하지 않으며 3개 seed로 학습 분포를 추정하는 데 한계가 있다. 다중비교를 고려한 영상×seed 구간의 하한이 양인 결과는 없다.
UBnormal의 29개 scene을 군집으로 재표본하는 민감도 분석도 수행했다. 현재 모델 고정 시 Precision@10%의 P-U와 P-I2 향상은 이 분석에서도 양의 구간이지만 P-C6는 불확정이다. 따라서 모든 통제에 대해 보편적으로 이겼다고 쓰지 않는다.

Claude 보고의 검토
1. 확정 not_met와 제시된 L 수치는 맞다. '확정 입증 없음'을 '완료 모델에서 유용한 개선을 보여줄 수도 없음'으로 확대해서는 안 된다.
2. 검증 V-AUC 평균과 붕괴 0으로 검출 비저하를 입증할 수 없다. 실제 시험 영상 AUC 평균은 UBnormal P 62.04/U 61.67, UCF 96.41/96.37, XD 99.23/99.28이다. 이 평균들 역시 비열등 검정을 대체하지 않는다. 원래 UCF 합산 AP 열세도 유지해 보고해야 한다.
3. 효과 불확정의 원인을 단순히 seed 3개 탓으로 확정할 수 없다. 큰 학습 변동이 실제로 관찰되었고 seed를 더 추가하면 결론이 좋아진다는 보장은 없다.
4. 선택된 rho 전역에서 문맥 가중치는 질의별로 학습되는 attention이 아니다. 기본 정상성 가중치를 공유하고 자기 연속 복제군을 제외한 뒤 재정규화한다. 또 실제 채점 문맥은 N_chunk 단위다. '스니펫마다 독립적으로 학습한 정상 기준'이나 긴 영상 전체를 언제나 사용하는 표현은 피한다. 구현 자체를 바꿀 근거는 아니다.
5. C6는 손실 및 체크포인트 선택이 다른 대안이다. 추론 시 K의 직접 기여는 같은 체크포인트 P-I2로 판단한다. 그 비교가 양이어도 정상 가중·복제 제외 각각의 독립적 필요성까지 입증한 것은 아니다.

검증 기반 선택의 추가 분석
첫 분석 후에 제안한 별도 탐색이다. 각 데이터셋·특징 쌍·방법에서 seed 3/4/5 중 이미 기록된 검증 sigma V-AUC가 최대인 것을 선택했고, 동률이면 작은 seed로 정했다. 시험 점수를 선택 함수에 사용하지 않았지만 이 정책을 검토하기 시작한 시점은 시험 결과를 본 뒤이므로 사후 분석이다. I2는 P와 반드시 같은 체크포인트를 쓴다. 모든 데이터셋에 같은 규칙을 적용했다.
UBnormal 선택: P=(4,4), U=(5,4), C6=(4,5), 쌍 순서는 XCB+XCL, XCB+SigLIP. 선택된 고정 모델의 Precision@10%는 P 81.07/U 71.81/C6 66.37/I2 72.39이다. 세 대비는 이 분석 범위의 다중비교를 고려한 조건부 영상 구간에서도 양이다. 그러나 현재 선택된 모델의 성능이지, 다시 3개를 학습하고 선택했을 때의 성능 보장은 아니다. 기존 all-seed 주분석을 이것으로 교체하지 않는다.
UCF·XD에서는 같은 선택 규칙을 쓰더라도 직접 K 보정의 일반적 이득은 뚜렷하지 않다. 이 결과들도 followup.json에 전부 포함했다.

단순 시간 사전분포 대조
시간에 따라 증가/감소하거나 가운데가 높은 학습 없는 점수와 상수 점수도 확인했다. UBnormal에서는 시간 증가 점수만으로 평균 TPR(FPR 0–5%)가 43.65로 전체 모델 평균 P 36.69보다 높다. 따라서 이 pAUC 하나를 실효성의 핵심 증거로 삼지 않는다. Precision@10%는 무작위 60.02, 가운데 점수 65.45, 전체 P 68.67, 검증 선택 P 81.07이다. 특정 데이터셋의 라벨 시간 분포와 높은 이상 프레임 비율을 함께 공개한다.

원고에서 가능한 핵심 설명
'UBnormal의 완료 모델을 대상으로 한 사후 분석에서 제안 보정은 제한된 검토 예산에서 이상 프레임의 우선순위를 개선했다. 직접 보정의 기여는 동일 체크포인트 비교로 관찰되었다. 다만 학습 seed에 따른 변동이 커 사전 등록한 학습 절차의 확정 우월성 기준은 충족하지 못했다.'
신규성은 지표 이름이 아니라 정상성 가중 영상 내 문맥과 잔차 점수 보정의 설계에서 설명한다. '일반적 향상 입증', '모든 통제 우위', '새 지표 최초 제안', '검출 비저하 입증'은 현재 자료로 확대하지 않는다.

Claude에게 전달할 지시
1. 원래 e024_stage3.json과 not_met 판정을 유지한다. 추가 학습·모델 수정·새 관문은 만들지 않는다.
2. 이 폴더의 PLAN.txt, results.json, followup.json과 분석 스크립트를 읽고 기존 저장 점수만으로 사후 분석 표를 원고에 반영한다. 단순 시간 점수 대조와 UCF·XD의 불확정/음의 결과도 남긴다.
3. 원고 중심 질문을 '이상 영상 안에서 한정된 검토 시간에 더 정확한 후보를 우선 제시하는가'로 구체화한다. Precision@10%를 사용 목적이 분명한 사후 평가로 설명하고 5%·20% 결과를 함께 제시한다. 지표를 새롭다고 포장하지 않는다.
4. 전체 seed 평균과 검증 기반 선택 모델은 별도 표로 둔다. 선택 정책이 시험 결과 확인 후 제안됐음을 표시하고, 원래 주분석/성공 기준을 바꾸지 않는다.
5. P-I2로 직접 추론 효과를 설명하고, seed·특징 쌍별 결과와 조건부/영상×seed 불확실성을 구분한다. C6를 pure inference ablation으로 부르지 않는다.
6. '실효성 근거가 전혀 없음'으로 원고를 종결하지 말고 관찰된 후보 순위 개선과 학습 안정성의 한계를 함께 기술한다. 제출 가능성을 통계 판정 하나로 단정하지 않는다.

실행
C:/Users/danla/anaconda3/envs/riskdet/python.exe -X utf8 F:/Vision/wvad_runs/e024_posthoc_20261026/analyze.py
C:/Users/danla/anaconda3/envs/riskdet/python.exe -X utf8 F:/Vision/wvad_runs/e024_posthoc_20261026/followup.py
C:/Users/danla/anaconda3/envs/riskdet/python.exe -X utf8 F:/Vision/wvad_runs/e024_posthoc_20261026/make_report.py
'''

table_text='\nUBnormal 전체 완료 모델 평균 (모든 값 %)\n지표 | UR-DMU | C6 | I2 | P\n'
for metric in ['auc','ap','precision10','recall05','recall10','recall20','pauc05']:
 table_text+=metric+' | '+' | '.join(f'{x:.4f}' for x in vals(row,metric))+'\n'
table_text+='\nPrecision@10% 전체 모델 평균 차이 (pp)\n대비 | 차이 | 조건부 전역 동시 구간 | 영상 x seed 명목 구간\n'
for c in ['P-U','P-C6','P-I2']:
 r=row('precision10',c);table_text+=f"{c} | {r['improvement_pp']:+.4f} | {ci(r['global_family_conditional_ci'])} | {ci(r['crossed_ci'])}\n"
(ROOT/'REPORT.txt').write_text(intro+table_text,encoding='utf-8')

sections='<h1>E024: 완료 모델의 개선 능력 분석</h1><p class="lead">관찰된 장점은 UBnormal에서의 후보 우선순위 개선입니다. 학습 안정성까지 포함한 확정 우월성은 아직 불확정입니다.</p><img src="capability_summary.png" alt="Fixed models and validation-selected models compared; seed variability"/>'
sections+='<h2>전체 완료 모델 평균 — UBnormal</h2>'+table(['지표 (%)','UR-DMU','C6','동일 P, K 꺼짐','P'],[[m,*[f'{x:.2f}' for x in vals(row,m)]] for m in ['auc','ap','precision10','recall05','recall10','recall20','pauc05']])
sections+='<h2>탐색 범위를 고려한 조건부 구간과 seed 불확실성</h2>'+table(['Precision@10% 대비','차이 (pp)','모델 고정·전역 동시 구간','영상×seed 명목 구간'],[[c,f"{row('precision10',c)['improvement_pp']:+.2f}",ci(row('precision10',c)['global_family_conditional_ci']),ci(row('precision10',c)['crossed_ci'])] for c in ['P-U','P-C6','P-I2']])
sections+='<h2>검증으로 seed 선택 — 추가 사후 분석</h2>'+table(['지표 (%)','UR-DMU','C6','동일 P, K 꺼짐','P'],[[m,*[f'{x:.2f}' for x in vals(policy,m)]] for m in ['auc','ap','precision10','recall05','pauc05']])
sections+='<h2>판단·방법·Claude 지시</h2><pre>'+html.escape(intro)+'</pre>'
sections+='<h2>모든 지표 — 모집단 전체</h2>'
for d in ['ubn','ucf','xdv']:
 lines=[]
 for m in METRICS:
  rs=[row(m,c,d) for c in ['P-U','P-C6','P-I2']]
  lines.append([m,f"{rs[0]['P_mean']:.3f}",*[f"{r['B_mean']:.3f}" for r in rs],*[f"{r['improvement_pp']:+.3f} {ci(r['global_family_crossed_ci'])}" for r in rs]])
 sections+='<details><summary>'+d.upper()+'</summary>'+table(['지표','P','U','C6','I2','P-U 및 전역 동시 영상×seed 구간','P-C6','P-I2'],lines)+'</details>'
sections+='<h2>선행 자료</h2><p>통상적인 AUROC만으로 실용적인 이상 포착 성능을 설명하기 어렵다는 문제는 <a href="https://openaccess.thecvf.com/content/CVPR2026/html/Acharya_The_Road_Less_Seen_Segment_Exploration_for_Weakly_Supervised_Video_CVPR_2026_paper.html">The Road Less Seen (CVPR 2026)</a>에서도 다룹니다. 이를 새 문제·새 평가 지표로 최초 주장하지 않습니다.</p><h2>재현 파일</h2><p><a href="REPORT.txt">텍스트 보고 및 Claude 지시</a> · <a href="results.csv">594개 대비 CSV</a> · <a href="results.json">주 사후 분석 JSON</a> · <a href="followup.json">선택 정책·단순 기준·scene 민감도 JSON</a> · <a href="analyze.py">분석 코드</a> · <a href="followup.py">추가 분석 코드</a></p>'
page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>E024 completed-model analysis</title><style>body{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.65 "Malgun Gothic",sans-serif;color:#183044;background:#fbfcfd}h1{font-size:30px}h2{margin-top:36px;color:#23536b}.lead{font-size:19px;background:#e8f2f7;padding:18px;border-left:5px solid #2878a0}img{width:100%;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}td,th{padding:9px;border-bottom:1px solid #dce3e8;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf0f4}pre{white-space:pre-wrap;font:15px/1.8 "Malgun Gothic",sans-serif}details{margin:16px 0;overflow:auto}summary{font-weight:bold;cursor:pointer}a{color:#17658a}</style>'+sections+'</html>'
(ROOT/'report.html').write_text(page,encoding='utf-8')
print('Report and figure saved:',ROOT)
