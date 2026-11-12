"""Offline audit of training traces and existing TRAIN-distribution diagnostics; no DEV reads."""
import argparse,collections,hashlib,json,math,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'runtime/diagnostics/training_audit_20261112'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(name,x):
 OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def training():
 rt=HERE/'runtime';model=ROOT/'finetuning/model/sft'/HERE.name
 source=read(rt/'source_manifest.json');verified={}
 for rel,h in source['files'].items():
  assert sha(ROOT/rel)==h,rel
  verified[rel]=h
 steps={a:rows(rt/(a+'_steps.jsonl')) for a in ['baseline','kd']}
 assert all(len(x)==1250 for x in steps.values())
 assert all(b['order']==k['order'] for b,k in zip(steps['baseline'],steps['kd']))
 result={'scope':'Training logs, code and metadata only. No DEV/TEST samples or prediction pairs read.',
 'source_files_verified':verified,'stages':{},'gradient_norm_logged_every_5_steps':{},'coverage':{},'curves':steps}
 for a,ss in steps.items():
  assert [s['step'] for s in ss]==list(range(1,1251))
  assert all(s['microbatches']==32 and all(math.isfinite(s[k]) for k in ['ce','weighted_kd']) for s in ss)
  counts=collections.Counter(tuple(x) for s in ss for x in s['order'])
  assert len(counts)==20000 and set(counts.values())=={2}
  result['coverage'][a]={'unique_language_views':len(counts),'total_exposures':sum(counts.values()),'each_view_seen':2}
 for label,l,h in [('first100',0,100),('epoch1',0,625),('epoch2',625,1250),('last100',1150,1250),('all',0,1250)]:
  b=steps['baseline'][l:h];k=steps['kd'][l:h];nbo=sum(sum(lang=='bo' for uid,lang in s['order']) for s in k)
  result['stages'][label]={'baseline_ce':statistics.mean(s['ce'] for s in b),'kd_ce':statistics.mean(s['ce'] for s in k),
   'ce_delta':statistics.mean(y['ce']-x['ce'] for x,y in zip(b,k)),
   'weighted_kl_mean':statistics.mean(s['weighted_kd'] for s in k),
   'bo_kl_response_mean':sum(s['weighted_kd']*s['microbatches'] for s in k)/(.2*nbo),
   'bo_exposures':nbo,'total_exposures':32*len(k)}
 for a in steps:
  logs=read(model/a/'trainer_state.json')['log_history'];g=[s['grad_norm'] for s in logs if 'grad_norm'in s]
  assert all(math.isfinite(x) for x in g)
  result['gradient_norm_logged_every_5_steps'][a]={'n':len(g),'mean':statistics.mean(g),'min':min(g),'max':max(g),'above_1':sum(x>1 for x in g)}
  total=statistics.mean(s['ce']+s['weighted_kd'] for s in steps[a]);reported=read(rt/(a+'_complete.json'))['metrics']['train_loss']
  assert abs(total-reported)<1e-6
  result.setdefault('loss_log_consistency',{})[a]={'reconstructed':total,'trainer_reported':reported}
 sys.path.insert(0,str(HERE.parents[1]/'src'))
 import torch
 result['saved_training_arguments']={}
 for a in steps:
  args=torch.load(model/a/'training_args.bin',map_location='cpu',weights_only=False)
  result['saved_training_arguments'][a]={k:getattr(args,k) for k in ['max_grad_norm','gradient_accumulation_steps','learning_rate','num_train_epochs','warmup_ratio','bf16','fp16','weight_decay']}
 for a in steps:
  for s in steps[a]:s.pop('order',None)
 result['missing_diagnostics']=['Per-language CE','Baseline KL to the same teacher','Separate CE/KD gradient norms and cosine','KL by response token position','Fixed held-out training-distribution validation loss']
 result['limits']='Loss magnitudes are not gradient contributions. Training losses do not establish a cause of held-out accuracy changes.'
 save('training_metrics.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['curves','source_files_verified']},ensure_ascii=True))
def teacher(path):
 import numpy as np
 data=rows(path/'score_rows.jsonl');buckets=collections.defaultdict(list);checks={}
 for row in data:
  file=path/'tokens'/(row['uid']+'.npz');z=np.load(file);checks[row['uid']]=sha(file)
  # Archived diagnostic: last token is EOS; body metrics exclude EOS, exactly as original scorer.
  idx=np.arange(len(z['token_ids']));body=idx<len(idx)-1
  for label,mask in [('all_body',body),('first1',body&(idx<1)),('first8',body&(idx<8)),('after8',body&(idx>=8))]:
   if not mask.any():continue
   pg=np.exp(z['logp_plus'][mask]);delta=z['delta'][mask]
   buckets[label].append({'tokens':int(mask.sum()),'nll_with_reference':float(-z['logp_plus'][mask].mean()),'nll_without_reference':float(-z['logp_base'][mask].mean()),
    'teacher_probability_on_given_token':float(pg.mean()),'top1_matches_given_token':float((z['argmax_plus'][mask]==z['token_ids'][mask]).mean()),
    'given_token_probability_ge_099':float((pg>=.99).mean()),'reference_gain_negative_fraction':float((delta<0).mean()),
    'effective_mixture_non_given_token_mass':float(((1-pg)/6).mean())})
 result={'scope':'Reanalysis of archived 200 TRAIN-sample teacher-forcing diagnostic; no new model execution. Sample means, not pooled-token means.',
  'samples':len(data),'files_sha256':checks,'statistics':{k:{field:float(np.mean([x[field] for x in v])) for field in v[0]} for k,v in buckets.items()},
  'soft_target_identity':'CE(y,p)+0.2 KL(q||p) = 1.2 CE((onehot(y)+0.2q)/1.2,p)-0.2 H(q)',
  'limits':'Token disagreement is not semantic wrongness. These 200 training samples are not representative guarantees for the entire 10k. Diagnostic excludes EOS; training includes EOS plus newline.',
  'diagnostic_protocol_sha256':sha(path/'protocol.json')}
 save('teacher_diagnostic_reanalysis.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'}))
def plot():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 d=read(OUT/'training_metrics.json');fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained');w=50
 for a,color,label in [('baseline','#2467a6','Bilingual SFT'),('kd','#d26b29','CE + KD')]:
  ss=d['curves'][a];x=np.arange(w,1251);y=np.convolve([r['ce'] for r in ss],np.ones(w)/w,mode='valid');axes[0].plot(x,y,color=color,label=label)
 ss=d['curves']['kd'];axes[1].plot(np.arange(w,1251),np.convolve([r['weighted_kd'] for r in ss],np.ones(w)/w,mode='valid'),color='#d26b29')
 axes[0].set_title('Hard-target CE (both languages)');axes[0].legend();axes[0].set_ylabel('Nats / response token')
 axes[1].set_title('Weighted KL (averaged over all views)');axes[1].set_ylabel('Contribution to training loss')
 for ax in axes:ax.set_xlabel('Optimizer step');ax.axvline(625,color='gray',ls='--',lw=.8);ax.grid(alpha=.2)
 fig.suptitle('Training diagnostics only; 50-step moving mean');fig.savefig(OUT/'training_curves.png',dpi=180);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['training','teacher','plot']);p.add_argument('--teacher-results',type=Path);a=p.parse_args()
 if a.mode=='training':training()
 elif a.mode=='teacher':teacher(a.teacher_results)
 else:plot()
