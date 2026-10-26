from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys, json, hashlib, zipfile, csv, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

REPO=Path('F:/Vision/frozen-vlm-wvad')
ROOT=Path(__file__).resolve().parent
SCORES=Path('F:/Vision/wvad_runs/e024/stage3_scores')
sys.path.insert(0,str(REPO/'src'))
from wvad.data import ubnormal as ub
from wvad.data.eval_manifests import load_eval_manifest
PAIRS=['xcb_xcl','xcb_siglip']; SEEDS=[3,4,5]; ARMS=['p','u','c6','i2']
QS=[.05,.1,.2]
METRICS=['auc','ap','pauc05','pauc10','tpr05','tpr10',
 'recall05','recall10','recall20','precision10','review80','peak_hit',
 'window_recall10','window_oracle10','window_recall20','window_oracle20',
 'event_recall10','event_recall20']
DIRECTION=np.array([-1 if m=='review80' else 1 for m in METRICS])
BOOT=5000

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def npz(p):
 with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}

def annotations(d,lab):
 manifest=REPO/f'data/manifests/{d}_test.json'
 m=load_eval_manifest(manifest)
 if d=='ubn':
  pinned=json.loads((REPO/'experiments/E024_localisation_method/results/e024_stage3_inputs.json').read_text())['tracks']['test']
  gt=[]
  with zipfile.ZipFile(REPO/'UBnormal.zip') as archive:
   for name,y,t in zip(m.names,m.labels,m.total_frames):
    if y:
     rel=ub.tracks_rel_path(name).removeprefix('UBnormal/')
     raw=archive.read(rel)
     assert hashlib.sha256(raw).hexdigest()==pinned[name],name
     gt.append(ub.frame_labels(t,ub.parse_tracks(raw.decode('utf-8-sig'),rel)))
    else: gt.append(np.zeros(t,np.uint8))
 else: gt=m.frame_gts()
 pos=np.concatenate([np.add.reduceat(np.asarray(g,np.int64),np.arange(0,len(g),16)) for g in gt])
 assert np.array_equal(pos,lab['pos']),d+' temporal annotation mismatch'
 assert np.array_equal(m.total_frames,lab['total_frames'])
 return m,gt

def group_curve(z,pos,length):
 order=np.argsort(-z,kind='stable')
 zo=z[order]; p=pos[order]; n=length[order]
 starts=np.r_[0,np.flatnonzero(np.diff(zo)!=0)+1]
 pp=np.add.reduceat(p,starts).astype(float)
 nn=np.add.reduceat(n,starts).astype(float)
 return zo[starts],pp,nn

def rank_mask(z,curve,budget):
 zz,pp,nn=curve
 counts=np.r_[0,np.cumsum(nn)]
 i=min(np.searchsorted(counts,budget,side='right')-1,len(zz)-1)
 share=np.clip((budget-counts[i])/nn[i],0,1)
 return np.where(z>zz[i],1.,np.where(z==zz[i],share,0.))

def metrics(z,g):
 z=np.asarray(z,float); g=np.asarray(g,np.uint8); t=len(g); p=float(g.sum()); n=t-p
 assert 0<p<t
 lens=np.minimum(16,t-np.arange(len(z))*16)
 pos=np.add.reduceat(g.astype(np.int64),np.arange(0,t,16))
 curve=group_curve(z,pos,lens); zz,pp,nn=curve
 tp=np.r_[0,np.cumsum(pp)]/p; fp=np.r_[0,np.cumsum(nn-pp)]/n
 auc=float(np.trapezoid(tp,fp))
 ap=float(np.sum((pp/p)*np.cumsum(pp)/np.cumsum(nn)))
 vals=[auc,ap]
 for alpha in [.05,.1]:
  keep=fp<alpha
  vals.append(float(np.trapezoid(np.r_[tp[keep],np.interp(alpha,fp,tp)],np.r_[fp[keep],alpha])/alpha))
 vals += [float(np.interp(alpha,fp,tp)) for alpha in [.05,.1]]
 cum_n=np.r_[0,np.cumsum(nn)]; cum_p=np.r_[0,np.cumsum(pp)]
 rec=[float(np.interp(q*t,cum_n,cum_p)/p) for q in QS]
 vals += rec+[rec[1]*p/(.1*t)]
 target=.8*p; j=int(np.searchsorted(cum_p,target,side='left'))
 reviewed=cum_n[j-1]+(target-cum_p[j-1])/pp[j-1]*nn[j-1]
 vals += [float(reviewed/t),float(pp[0]/nn[0])]
 frames=np.repeat(z,lens); centered=frames-frames.mean()
 cs=np.r_[0,np.cumsum(centered)]; cg=np.r_[0,np.cumsum(g,dtype=np.int64)]
 for q in [.1,.2]:
  w=max(1,int(np.ceil(q*t))); score=(cs[w:]-cs[:-w])/w; mass=cg[w:]-cg[:-w]
  best=np.isclose(score,score.max(),rtol=0,atol=1e-10)
  selected=float(mass[best].mean())
  vals += [selected/p,selected/float(mass.max())]
 starts=np.flatnonzero(np.diff(np.r_[0,g,0].astype(int))==1)
 ends=np.flatnonzero(np.diff(np.r_[0,g,0].astype(int))==-1)
 for q in [.1,.2]:
  chosen=np.repeat(rank_mask(z,curve,q*t),lens)
  csel=np.r_[0,np.cumsum(chosen)]
  vals.append(float(np.mean((csel[ends]-csel[starts])/(ends-starts))))
 return np.asarray(vals)

def self_test():
 rng=np.random.default_rng(1)
 for n in [2,5,13]:
  z=rng.integers(0,4,n).astype(float); g=rng.integers(0,2,n*16-7,dtype=np.uint8)
  v=metrics(z,g); f=np.repeat(z,16)[:len(g)]
  assert np.allclose(v[:2],[roc_auc_score(g,f),average_precision_score(g,f)])
  for j,alpha in enumerate([.05,.1]):
   standardized=roc_auc_score(g,f,max_fpr=alpha)
   raw=((2*standardized-1)*(alpha-.5*alpha**2)+.5*alpha**2)/alpha
   assert np.isclose(v[2+j],raw),(v[2+j],raw)
  assert np.allclose(v,metrics(3*z+7,g))
 g=np.r_[np.ones(32,np.uint8),np.zeros(32,np.uint8)]
 good=metrics(np.array([2,2,0,0]),g); bad=metrics(np.array([0,0,2,2]),g)
 assert good[0]==1 and bad[0]==0
 assert np.isclose(good[7],.2) and np.isclose(bad[7],0)
 assert np.isclose(good[9],1) and np.isclose(bad[9],0)
 assert np.isclose(good[10],.4) and np.isclose(bad[10],.9)
 const=metrics(np.ones(4),g)
 assert np.allclose(const[[0,1,9,11]],[.5,.5,.5,.5])
 assert np.allclose(const[[6,7,8]],[.05,.1,.2])
 assert np.isclose(const[10],.8)
 print('metric self-tests passed',flush=True)

def bootstrap_population(values,idx,label):
 # values: arm,pair,seed,video,metric. Averaging pairs preserves paired contrasts.
 selected=values[:,:,:,idx,:]; mean=selected.mean(axis=1)
 delta=np.stack([mean[0]-mean[b] for b in [1,2,3]])*DIRECTION
 delta=delta.transpose(0,2,1,3) # contrast,video,seed,metric
 nv=len(idx); rng=np.random.default_rng(20261026)
 w=rng.multinomial(nv,np.full(nv,1/nv),size=BOOT).astype(float)
 sd=rng.multinomial(3,[1/3]*3,size=BOOT).astype(float)
 reps=[]; vids=[]; estimates=[]
 for dd in delta:
  weighted=(w@dd.reshape(nv,-1)).reshape(BOOT,3,len(METRICS))/nv
  reps.append((weighted*sd[:,:,None]).sum(axis=1)/3)
  vids.append(weighted.mean(axis=1)); estimates.append(dd.mean(axis=(0,1)))
 reps=np.asarray(reps); vids=np.asarray(vids); estimates=np.asarray(estimates)
 def simultaneous(samples):
  se=samples.std(axis=1,ddof=1)
  standardized=np.abs(samples-estimates[:,None,:])/np.maximum(se[:,None,:],1e-15)
  maxima=standardized.max(axis=(0,2)); critical=float(np.quantile(maxima,.95))
  return se,critical,maxima
 se,critical,maxima=simultaneous(reps); vse,vcrit,vmax=simultaneous(vids)
 out=[]
 for ci,b in enumerate([1,2,3]):
  for mi,name in enumerate(METRICS):
   est=estimates[ci,mi]; a=selected[0,:,:,:,mi]; bb=selected[b,:,:,:,mi]
   bypair=((a-bb)*DIRECTION[mi]).mean(axis=(1,2))*100
   out.append({'population':label,'n_videos':nv,'metric':name,'higher_is_better':bool(DIRECTION[mi]>0),
    'contrast':'P-'+ARMS[b].upper(),'P_mean':float(a.mean()*100),'B_mean':float(bb.mean()*100),
    'improvement_pp':float(est*100),'crossed_ci':(np.quantile(reps[ci,:,mi],[.025,.975])*100).tolist(),
    'conditional_ci':(np.quantile(vids[ci,:,mi],[.025,.975])*100).tolist(),
    'simultaneous_crossed_ci':((est+np.array([-1,1])*critical*se[ci,mi])*100).tolist(),
    'simultaneous_conditional_ci':((est+np.array([-1,1])*vcrit*vse[ci,mi])*100).tolist(),
    'by_seed':(delta[ci,:,:,mi].mean(axis=0)*100).tolist(),'by_pair':bypair.tolist(),
    '_se':float(se[ci,mi]),'_vse':float(vse[ci,mi])})
 return out,maxima,vmax,delta

def main():
 self_test()
 done=json.loads((SCORES/'done.json').read_text())
 bad=[k for k,h in done['files'].items() if sha(SCORES/k)!=h]
 assert not bad,bad
 provenance={'status':'post-hoc exploratory; original not_met unchanged','verified_score_files':len(done['files']),
  'source_done_sha256':sha(SCORES/'done.json'),'script_sha256':sha(__file__),'boot':BOOT,'metric_family':METRICS,
  'contrasts':['P-U','P-C6','P-I2'],'datasets':{}}
 all_rows=[]
 global_populations=0
 for d in ['ubn','ucf','xdv']:
  lab=npz(SCORES/d/'labels.npz'); off=np.r_[0,np.cumsum(lab['counts'])]
  positive=np.add.reduceat(lab['pos'],off[:-1])
  ids=json.loads((SCORES/d/'summary.json').read_text())['l_videos']
  pr=positive[ids]/lab['total_frames'][ids]
  global_populations+=sum(n>=10 for n in [len(ids),(pr<=.25).sum(),((pr>.25)&(pr<=.5)).sum(),(pr>.5).sum()])
 provenance['family_correction']='Bootstrap max-standardized-deviation within 18 metrics x 3 contrasts; Bonferroni across populations/datasets. Approximate exploratory intervals.'
 provenance['global_populations']=int(global_populations)
 original=json.loads((REPO/'experiments/E024_localisation_method/results/e024_stage3.json').read_text())
 for d in ['ubn','ucf','xdv']:
  start=time.time(); lab=npz(SCORES/d/'labels.npz'); m,gt=annotations(d,lab)
  mixed=[i for i,g in enumerate(gt) if lab['valid'][i] and 0<int(np.sum(g))<len(g) and len(g)>=32 and lab['labels'][i]==1]
  summary=json.loads((SCORES/d/'summary.json').read_text()); assert mixed==summary['l_videos']
  prevalence=np.array([gt[i].mean() for i in mixed]); offsets=np.r_[0,np.cumsum(lab['counts'])]
  values=np.empty((4,2,3,len(mixed),len(METRICS)))
  for pi,p in enumerate(PAIRS):
   for si,s in enumerate(SEEDS):
    for ai,a in enumerate(ARMS):
     z=npz(SCORES/d/f'{p}_s{s}'/f'{"p" if a=="i2" else a}.npz')
     zs=z['z0' if a=='i2' else 'z']
     for vi,i in enumerate(mixed): values[ai,pi,si,vi]=metrics(zs[offsets[i]:offsets[i+1]],gt[i])
     key=str(z['chain'].item()); reference=summary['chains'][key]['i2' if a=='i2' else 'metrics']['per_video']
     lindex=1 if d=='xdv' else 0
     assert np.allclose(values[ai,pi,si,:,lindex]*100,reference,rtol=0,atol=1e-9),(d,p,s,a)
  np.savez_compressed(ROOT/f'{d}_per_video.npz',values=values,names=np.array([m.names[i] for i in mixed]),
   metrics=np.array(METRICS),prevalence=prevalence,indices=np.array(mixed),
   total_frames=np.array([len(gt[i]) for i in mixed]))
  groups={'overall':np.arange(len(mixed)),'prevalence_le25':np.flatnonzero(prevalence<=.25),
    'prevalence_25to50':np.flatnonzero((prevalence>.25)&(prevalence<=.5)),
    'prevalence_gt50':np.flatnonzero(prevalence>.5)}
  results=[]; maxima=[]; vmax=[]
  for label,idx in groups.items():
   if len(idx)<10: continue
   rows,mx,vmx,delta=bootstrap_population(values,idx,label)
   results+=rows;maxima.append(mx);vmax.append(vmx)
  for gi,(mx,vmx) in enumerate(zip(maxima,vmax)):
   critical=float(np.quantile(mx,1-.05/len(maxima)));vcritical=float(np.quantile(vmx,1-.05/len(vmax)))
   gc=float(np.quantile(mx,1-.05/global_populations));gvc=float(np.quantile(vmx,1-.05/global_populations))
   for row in results[gi*54:(gi+1)*54]:
    est=row['improvement_pp']; se=row.pop('_se');vse=row.pop('_vse')
    row.update(dataset=d,dataset_family_crossed_ci=(est+np.array([-1,1])*critical*se*100).tolist(),
     dataset_family_conditional_ci=(est+np.array([-1,1])*vcritical*vse*100).tolist(),
     global_family_crossed_ci=(est+np.array([-1,1])*gc*se*100).tolist(),
     global_family_conditional_ci=(est+np.array([-1,1])*gvc*vse*100).tolist())
  all_rows+=results
  provenance['datasets'][d]={'videos':len(gt),'mixed_videos':len(mixed),
   'populations':{k:len(v) for k,v in groups.items()},'all_original_L_reproduced':True,
   'annotation_source':'existing sealed manifest/UBnormal archive tracks; counts verified exactly',
   'family_population_count':len(maxima)}
  dump(ROOT/f'{d}_results.json',results)
  print(d,'done',len(mixed),'videos',round(time.time()-start,1),'seconds',flush=True)
 dump(ROOT/'results.json',{'provenance':provenance,'rows':all_rows})
 with open(ROOT/'results.csv','w',newline='',encoding='utf-8-sig') as f:
  fields=list(all_rows[0]); w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
 print('POSITIVE dataset-wide corrected crossed intervals:',flush=True)
 for r in all_rows:
  if r['dataset_family_crossed_ci'][0]>0: print(r['dataset'],r['population'],r['contrast'],r['metric'],round(r['improvement_pp'],3),r['dataset_family_crossed_ci'],r['by_seed'],flush=True)
 print('COMPLETE',str(ROOT),flush=True)

if __name__=='__main__':main()
