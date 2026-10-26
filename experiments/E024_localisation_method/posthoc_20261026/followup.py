"""Explicit second-round sensitivity analyses; never alter original stage-3 verdict."""
from analyze import *
import re
from wvad.baselines.common import video_scores

RUNS=Path('F:/Vision/wvad_runs/e024/stage3')
FOLDERS={'p':'k_rg_t0.25','u':'urdmu_u200','c6':'ctl_trainonly_rg_t0.25'}

def intervals(delta, weights, alpha=.05, seed_weights=None):
 # delta: contrast, video, seed, metric; weights: B,video (may be cluster weights).
 nv=delta.shape[1]; ns=delta.shape[2]; nm=delta.shape[3]
 est=delta.mean(axis=(1,2)); reps=[]
 den=weights.sum(axis=1)
 for x in delta:
  v=(weights@x.reshape(nv,-1)).reshape(len(weights),ns,nm)/den[:,None,None]
  reps.append(v.mean(axis=1) if seed_weights is None else (v*seed_weights[:,:,None]).sum(axis=1)/ns)
 reps=np.asarray(reps); se=reps.std(axis=1,ddof=1)
 maximum=(np.abs(reps-est[:,None,:])/np.maximum(se[:,None,:],1e-15)).max(axis=(0,2))
 critical=float(np.quantile(maximum,1-alpha)); rows=[]
 for ci,c in enumerate(['P-U','P-C6','P-I2']):
  for mi,m in enumerate(METRICS):
   rows.append({'contrast':c,'metric':m,'improvement_pp':float(est[ci,mi]*100),
    'nominal_ci':(np.quantile(reps[ci,:,mi],[.025,.975])*100).tolist(),
    'family_ci':((est[ci,mi]+np.array([-1,1])*critical*se[ci,mi])*100).tolist()})
 return rows

def main():
 out={'status':'Second-round post-hoc exploration; original all-seed not_met unchanged',
  'selection_rule':'For each dataset/pair/arm, maximum existing validation sigma V-AUC; ties smaller seed. I2 retains P seed. Rule proposed after viewing test results.',
  'metric_validation':'pAUC checked independently against inverse of sklearn standardized partial AUC.',
  'script_sha256':sha(__file__),'datasets':{}}
 for d in ['ubn','ucf','xdv']:
  cached=npz(ROOT/f'{d}_per_video.npz'); values=cached['values']; nv=values.shape[3]
  selected=np.empty((4,2,nv,len(METRICS))); choices=[]
  for ai,a in enumerate(ARMS[:3]):
   for pi,p in enumerate(PAIRS):
    va=[]
    for s in SEEDS:
     info=json.loads((RUNS/f'{d}_{p}_s{s}'/FOLDERS[a]/'selection.json').read_text())
     assert info['failed'] is None
     assert info['code']['commit']=='a5f310b07004a32cd199c632f8fc92b1491c916c'
     va.append(info['selection']['sigma']['v_auc'])
    si=int(np.argmax(va)); selected[ai,pi]=values[ai,pi,si]
    if a=='p': selected[3,pi]=values[3,pi,si]
    choices.append({'arm':a,'pair':p,'seed':SEEDS[si],'validation_auc':va[si],'all_validation_aucs':va})
  rng=np.random.default_rng(20261027)
  w=rng.multinomial(nv,np.full(nv,1/nv),size=BOOT).astype(float)
  mean=selected.mean(axis=1)
  delta=np.stack([mean[0]-mean[b] for b in [1,2,3]])[:,:,None,:]*DIRECTION
  policy=intervals(delta,w,alpha=.05/3)
  for r in policy:
   mi=METRICS.index(r['metric']);ai=['P-U','P-C6','P-I2'].index(r['contrast'])+1
   r.update(P_mean=float(mean[0,:,mi].mean()*100),B_mean=float(mean[ai,:,mi].mean()*100))
  lab=npz(SCORES/d/'labels.npz');m,gt=annotations(d,lab)
  naive={}
  for name in ['constant','rising','falling','center']:
   vals=[]
   for i in cached['indices']:
    n=int(lab['counts'][i]);t=np.linspace(0,1,n)
    z={'constant':np.zeros(n),'rising':t,'falling':-t,'center':-np.abs(t-.5)}[name]
    vals.append(metrics(z,gt[i]))
   naive[name]=dict(zip(METRICS,(np.mean(vals,axis=0)*100).tolist()))
  off=np.r_[0,np.cumsum(lab['counts'])]; test_video=[]
  for p in PAIRS:
   for s in SEEDS:
    for a in ARMS:
     z=npz(SCORES/d/f'{p}_s{s}'/f'{"p" if a=="i2" else a}.npz')['z0' if a=='i2' else 'z']
     readouts=video_scores([z[off[i]:off[i+1]] for i in range(len(gt))],lab['total_frames'])
     test_video.append({'pair':p,'seed':s,'arm':a,
      'sigma_auc':float(roc_auc_score(lab['labels'],readouts['sigma'])*100),
      'logit_auc':float(roc_auc_score(lab['labels'],readouts['logit'])*100)})
  detail={'validation_choices':choices,'policy_results':policy,'untrained_baselines':naive,
   'test_video_auc':test_video,'random_precision10':float(cached['prevalence'].mean()*100)}
  if d=='ubn':
   scenes=[re.search(r'scene_(\d+)',name).group(1) for name in cached['names']]
   unique=sorted(set(scenes));assign=np.array([[s==c for s in scenes] for c in unique],float)
   cw=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=BOOT)@assign
   avg=values.mean(axis=1);dd=np.stack([avg[0]-avg[b] for b in [1,2,3]]).transpose(0,2,1,3)*DIRECTION
   sw=rng.multinomial(3,[1/3]*3,size=BOOT)
   detail['scene_count']=len(unique)
   detail['scene_conditional']=intervals(dd,cw)
   detail['scene_crossed']=intervals(dd,cw,seed_weights=sw)
  out['datasets'][d]=detail
  print(d,'validation-selected models:',[(x['arm'],x['pair'],x['seed']) for x in choices],flush=True)
  for r in policy:
   if r['metric'] in ['auc','ap','pauc05','precision10','recall05']:
    print('policy',r['contrast'],r['metric'],round(r['improvement_pp'],3),np.round(r['nominal_ci'],3).tolist(),flush=True)
  print('naive:',{k:{m:round(v[m],2) for m in ['auc','pauc05','precision10','recall05']} for k,v in naive.items()},flush=True)
  print('test_video_auc',{a:round(np.mean([x['sigma_auc'] for x in test_video if x['arm']==a]),3) for a in ARMS},flush=True)
 dump(ROOT/'followup.json',out)

if __name__=='__main__':main()
