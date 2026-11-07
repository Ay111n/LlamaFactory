"""Repair validation and the ONLY authorized Full-G2 run; no CE baseline run."""
import argparse,copy,hashlib,importlib.metadata,importlib.util,inspect,json,math,os,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch,yaml
from datasets import load_from_disk
from transformers import set_seed,Trainer
from experiment_paths import *
from full_g2_trainer import FullG2Trainer,FullBoundaryCollator,add_vector,gram64,norm64
from llamafactory.hparams import get_train_args,FinetuningArguments
from llamafactory.data import SFTDataCollatorWith4DAttentionMask,get_template_and_fix_tokenizer
from llamafactory.model import load_model,load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

MODEL=model_root();DATA=data_root();ANALYSIS=diagnostics_root()
COEF=.0012;STEPS=9339

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def configuration(stage):
 cfg=yaml.safe_load(require_path('original_yaml').read_text())
 cfg.update(output_dir=str(MODEL/('validation' if stage!='train' else 'runs/Full-G2')),
            termination_dataset_path=str(require_path('prepared_dataset')),termination_loss_mode='dynamic_gold_margin',termination_loss_coef=COEF,
            data_seed=3407,optim='adamw_torch',packing=False,train_on_prompt=False,remove_unused_columns=False)
 return cfg

def build(stage):
 set_seed(3407);cfg=configuration(stage)
 with patch('torch.cuda.device_count',return_value=1):ma,da,ta,fa,ga=get_train_args(cfg)
 ta._n_gpu=1
 set_seed(3407)
 assert ma.use_reentrant_gc is True and ta.gradient_checkpointing_kwargs is None
 tokenmod=load_tokenizer(ma);token=tokenmod['tokenizer'];template=get_template_and_fix_tokenizer(token,da)
 model=load_model(token,ma,fa,True)
 collator=SFTDataCollatorWith4DAttentionMask(template=template,model=model,pad_to_multiple_of=8,label_pad_token_id=-100,
   block_diag_attn=False,neat_packing=False,attn_implementation=getattr(model.config,'_attn_implementation',None),compute_dtype=ma.compute_dtype,**tokenmod)
 trainer=FullG2Trainer(model=model,args=ta,finetuning_args=fa,data_collator=FullBoundaryCollator(collator),train_dataset=load_from_disk(str(require_path('prepared_dataset'))),**tokenmod)
 assert sum(p.numel() for p in trainer.term_params)==835190784
 layout=[(n,list(p.shape)) for n,p in model.named_parameters() if p.requires_grad]
 reference=require_path('gold_experiment_root')/'diagnostics/gradient_attribution_20261025/sources/final_G2.json'
 assert sorted(layout)==sorted((n,list(s)) for n,s in json.loads(reference.read_text())['parameter_layout'])
 out=diagnostic_path(Path(cfg['output_dir'])/'full_g2_steps.jsonl').parent;out.mkdir(parents=True,exist_ok=True)
 trainer.log_root=out
 (out/'resolved_config.json').write_text(json.dumps(cfg,indent=2))
 (out/'trainable_parameters.json').write_text(json.dumps({'count':sum(p.numel() for p in trainer.term_params),'layout':layout,'same_names_and_shapes_as_small_G2':True},indent=2))
 return trainer,cfg

def freeze_windows():
 assert not (DATA/'probe_windows.pt').exists()
 manifest=json.loads(require_path('prepared_manifest').read_text());assert manifest['gate_pass'] and manifest['rows']==49800
 rows=[json.loads(l) for l in require_path('sample_audit').open()]
 rng=np.random.default_rng(3407)
 specs=[('mixed_8',16,8,None,None),('all_no_think',16,0,None,None),('one_think',16,1,None,None),
        ('mostly_think',16,15,None,None),('all_think',16,16,None,None),
        ('math_mixed',16,8,'Math',None),('long_mixed',16,8,None,2048),('tail_8',8,None,None,None)]
 selections=[]
 for name,n,nt,domain,min_length in specs:
  if name=='tail_8':
   from accelerate.data_loader import SeedableRandomSampler
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(3407);order=list(SeedableRandomSampler(range(49800),data_seed=3407))
   indices=order[-8:]
  else:
   choices=[]
   for flag,count in [(True,nt),(False,n-nt)]:
    pool=[r['source_index'] for r in rows if r['think']==flag and (domain is None or r['domain']==domain) and (min_length is None or r['length']>=min_length)]
    if len(pool)<count:raise ValueError(f'Insufficient samples for prespecified {name}: {flag}')
    if min_length is not None and count:
     longest=max(pool,key=lambda i:rows[i]['length']);choices.append(longest)
     choices.extend(rng.choice([i for i in pool if i!=longest],count-1,replace=False).tolist())
    else:choices.extend(rng.choice(pool,count,replace=False).tolist())
   indices=sorted(choices)
  selections.append({'name':name,'indices':indices,'think':sum(rows[i]['think'] for i in indices),
   'length_min':min(rows[i]['length'] for i in indices),'length_max':max(rows[i]['length'] for i in indices),
   'domains':{d:sum(rows[i]['domain']==d for i in indices) for d in sorted({rows[i]['domain'] for i in indices})}})
 (DATA/'probe_selection.json').write_text(json.dumps(selections,indent=2))
 dataset=load_from_disk(str(require_path('prepared_dataset')))
 # CPU-only tokenizer/collation, preserving raw record indices and native padding.
 from transformers import AutoTokenizer,DataCollatorForSeq2Seq
 token=AutoTokenizer.from_pretrained(require_path('base_model'),local_files_only=True,padding_side='right')
 collator=FullBoundaryCollator(DataCollatorForSeq2Seq(token,pad_to_multiple_of=8,label_pad_token_id=-100))
 windows=[[collator([dataset[i]]) for i in spec['indices']] for spec in selections]
 torch.save(windows,DATA/'probe_windows.pt')
 samples=[]
 for flag,needed in [(True,2),(False,1)]:
  for r in rows:
   if r['think']!=flag:continue
   record=dataset[r['source_index']]
   samples.append({'source_index':r['source_index'],'think':flag,'input_ids':record['input_ids'][:512]})
   needed-=1
   if not needed:break
 (DATA/'export_inputs.json').write_text(json.dumps({'samples':samples},indent=2))
 (ROOT/'configuration.json').write_text(json.dumps(configuration('train'),indent=2))
 (ROOT/'full_g2_49800.yaml').write_text('# Authorized only after gates.json passed. Use run_full_g2.py train, not the historical Trainer.\n'+yaml.safe_dump(configuration('train'),sort_keys=False))
 current={'python':__import__('sys').version,'executable':__import__('sys').executable,'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate','peft','datasets','spacy','flash-attn']}}
 evidence={'current':current,'historical_runtime_fully_confirmed':False,
  'historical_evidence':'Saved comparison CSV explicitly marks original runtime/source and gradient-checkpoint kwargs unverified; original output training-state path is not present.',
  'references':[str(require_path('legacy_experiment_root')/'historical_config_comparison.csv'),str(require_path('original_yaml'))],
  'claim_boundary':'Strictly matched to current recoverable stock LF/HF/Accelerate native CE path; no claim of byte-identical historical runtime',
  'native_backward_source':inspect.getsource(Trainer.training_step),'accelerator_backward_source':inspect.getsource(__import__('accelerate').Accelerator.backward)}
 (ROOT/'environment_evidence.json').write_text(json.dumps(evidence,indent=2))
 print('FROZEN_WINDOWS',json.dumps(selections),flush=True)

def real_ce_equivalence(trainer,windows):
 batches=windows[0];model=trainer.model
 args=copy.copy(trainer.args);args.output_dir=str(MODEL/'validation/native_ce_reference');args._n_gpu=1
 fin=copy.deepcopy(trainer.finetuning_args);fin.termination_loss_mode='none';fin.termination_loss_coef=0.;fin.termination_dataset_path=None
 native=CustomSeq2SeqTrainer(model=model,args=args,finetuning_args=fin,processor=None,tokenizer=trainer.processing_class,
   data_collator=trainer.data_collator,train_dataset=trainer.train_dataset)
 versions=[p._version for p in trainer.term_params];vectors=[];losses=[];RNG=[]
 saved_coef=trainer.finetuning_args.termination_loss_coef;trainer.finetuning_args.termination_loss_coef=0.
 try:
  for t in [native,trainer]:
   for p in trainer.term_params:p.grad=None
   t.current_gradient_accumulation_steps=len(batches);t.state.max_steps=STEPS
   count=Trainer._get_num_items_in_batch(t,batches,t.args.device)
   if t is trainer:t.get_batch_samples(iter(batches),len(batches),t.args.device)
   total=0.
   with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
    for b in batches:
     clean=dict(b)
     if t is native:
      for key in ['source_index','term_positions','term_targets']:clean.pop(key,None)
     total+=float(t.training_step(model,clean,num_items_in_batch=count))
    RNG.append(torch.cuda.get_rng_state(0).cpu().tolist())
   vectors.append(add_vector(None,[p.grad for p in trainer.term_params],trainer.buffer_device));losses.append(total)
  maximum=0.
  for a,b in zip(*vectors):
   if a is None or b is None:
    if a is not None or b is not None:raise AssertionError('Gradient presence differs')
   else:
    # Chunked comparison avoids an extra full lm_head diagnostic allocation.
    a,b=a.reshape(-1),b.reshape(-1)
    for start in range(0,a.numel(),1048576):
     x,y=a[start:start+1048576],b[start:start+1048576]
     torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-6)
     maximum=max(maximum,float((x-y).abs().max()))
  assert math.isclose(*losses,rel_tol=2e-4,abs_tol=2e-6) and RNG[0]==RNG[1]
  assert versions==[p._version for p in trainer.term_params]
  result={'passed':True,'actual_8B':True,'window_microbatches':len(batches),'native_loss':losses[0],'full_g2_lambda0_loss':losses[1],
   'max_parameter_gradient_error':maximum,'rtol':2e-4,'atol':2e-6,'same_rng_after_forward_backward':True,
   'same_supervised_tokens':True,'native_accelerator_ga':native.accelerator.gradient_accumulation_steps,'full_g2_accelerator_ga':trainer.accelerator.gradient_accumulation_steps,
   'optimizer_updates':0,'parameters_unchanged':True}
  (ROOT/'real_ce_equivalence.json').write_text(json.dumps(result,indent=2));print('REAL_CE_EQUIVALENCE',json.dumps(result),flush=True)
 finally:
  trainer.finetuning_args.termination_loss_coef=saved_coef
  for p in trainer.term_params:p.grad=None
  del vectors,native

def real_validation():
 assert json.loads((ROOT/'unit_gate.json').read_text())['passed']
 assert (DATA/'probe_windows.pt').exists()
 trainer,cfg=build('validate')
 trainer.model=trainer.accelerator.prepare_model(trainer.model)
 trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs)
 windows=torch.load(DATA/'probe_windows.pt',map_location='cpu',weights_only=False)
 selections=json.loads((DATA/'probe_selection.json').read_text())
 for selected,window in zip(selections,windows):
  for source_index,frozen in zip(selected['indices'],window):
   actual=trainer.data_collator([trainer.train_dataset[source_index]])
   assert set(actual)==set(frozen),'Frozen collator keys differ from original SFT collator'
   for key in actual:
    if torch.is_tensor(actual[key]):assert torch.equal(actual[key],frozen[key]),key
    else:assert actual[key]==frozen[key],key
 # Record actual runtime GC kwargs through the LF wrapper closure.
 functions=[]
 for module in trainer.model.modules():
  f=getattr(module,'_gradient_checkpointing_func',None)
  if f is None:continue
  inner=inspect.getclosurevars(f).nonlocals.get('gradient_checkpointing_func',f) if inspect.isfunction(f) else f
  functions.append(getattr(inner,'keywords',{}))
 assert functions and all(f.get('use_reentrant') is True for f in functions)
 (ROOT/'real_checkpointing.json').write_text(json.dumps({'gradient_checkpointing':True,'kwargs':functions,'configured_gradient_checkpointing_kwargs':None},indent=2,default=str))
 real_ce_equivalence(trainer,windows)
 rows=[trainer.probe_window(trainer.model,w,0,i,'initial') for i,w in enumerate(windows)]
 vals=[r['R'] for r in rows]
 gate=all(math.isfinite(x) for x in vals) and .001<=float(np.quantile(vals,.9))<=1
 diagnostic={'gate_pass':gate,'rows':rows,'R_distribution':{k:float(np.quantile(vals,q)) for k,q in [('min',0),('p10',.1),('p50',.5),('p90',.9),('p95',.95),('max',1)]},
  'scope':'8 prespecified representative windows, not population/random full-training distribution','parameters_updated':False}
 (ROOT/'initial_gradients.json').write_text(json.dumps(diagnostic,indent=2))
 if not gate:raise RuntimeError('Initial gradient strength gate failed; no retuning')
 # One genuine update at the first positive LR of the UNCHANGED 9339-step warmup schedule.
 trainer.create_optimizer_and_scheduler(num_training_steps=STEPS)
 initial_lr=trainer.optimizer.param_groups[0]['lr'];trainer.lr_scheduler.step()
 scheduled_lr=trainer.optimizer.param_groups[0]['lr'];assert scheduled_lr>0
 batch=windows[0];trainer.get_batch_samples(iter(batch),len(batch),trainer.args.device);trainer.current_gradient_accumulation_steps=len(batch);trainer.state.max_steps=STEPS
 sample_params=[(n,p) for n,p in trainer.model.named_parameters() if p.requires_grad and 'lora_B' in n][:3]
 before=[p.detach().reshape(-1)[:256].clone() for n,p in sample_params]
 trainer.optimizer.zero_grad(set_to_none=True)
 for b in batch:trainer.training_step(trainer.model,b,num_items_in_batch=trainer._window_tokens)
 preclip=norm64([p.grad for p in trainer.term_params]);trainer.accelerator.clip_grad_norm_(trainer.term_params,trainer.args.max_grad_norm)
 trainer.optimizer.step();trainer.optimizer.zero_grad(set_to_none=True)
 changes=[float((p.detach().reshape(-1)[:256]-old).abs().max()) for (n,p),old in zip(sample_params,before)]
 assert max(changes)>0 and all(math.isfinite(v) for v in changes)
 smoke={'passed':True,'optimizer_updates':1,'native_initial_schedule_lr':initial_lr,'validation_schedule_position':1,'scheduled_positive_lr':scheduled_lr,
        'hyperparameter_retuning':False,'reason':'original warmup starts at LR0; validate one update at its first positive scheduled rate, then discard model/optimizer',
        'sampled_lora_B_max_changes':changes,'grad_norm_preclip':preclip,'formal_reinitialize_base':True}
 (ROOT/'real_update.json').write_text(json.dumps(smoke,indent=2));print('REAL_UPDATE_PASSED',json.dumps(smoke),flush=True)

def train():
 assert all(json.loads((ROOT/'gates.json').read_text())[k] for k in ['unit','real_ce','real_gradient_strength','real_update','input_fingerprints'])
 output=MODEL/'runs/Full-G2'
 if output.exists() and any(output.iterdir()):raise FileExistsError('Do not overwrite or silently resume existing Full-G2')
 frozen=json.loads((ROOT/'frozen_inputs.json').read_text())
 for p,h in frozen['files'].items():assert sha(p)==h,f'Frozen source/input changed: {p}'
 trainer,cfg=build('train');trainer.probe_windows=torch.load(DATA/'probe_windows.pt',weights_only=False,map_location='cpu')
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
  loader=trainer.get_train_dataloader();values=trainer.set_initial_training_values(trainer.args,loader,16)
  assert values[-1]==STEPS and values[-2]==49800 and len(trainer.train_dataset)==49800
  del loader
 start=time.time();result=trainer.train()
 assert trainer.state.global_step==STEPS
 trainer.save_model();trainer.save_state();trainer.save_metrics('train',result.metrics)
 for step in [1500,3000,4500,6000,7500,9000]:assert (output/f'checkpoint-{step}'/'adapter_model.safetensors').exists()
 # All native optimizer/checkpoint states have been saved. Release only in-memory
 # optimizer state before final read-only vector diagnostics; no more updates occur.
 trainer.optimizer.zero_grad(set_to_none=True)
 trainer.optimizer.state.clear()
 trainer.gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs
 for i,w in enumerate(trainer.probe_windows):trainer.probe_window(trainer.model,w,STEPS,i,'post_final_update')
 report={'arm':'Full-G2','steps':STEPS,'seconds':time.time()-start,'metrics':result.metrics,'checkpoint_steps':[1500,3000,4500,6000,7500,9000],
         'native_backward':True,'standard_CE_retrained':False,'TEST_used':False}
 diagnostic_path(output/'completed.json').write_text(json.dumps(report,indent=2));print('FULL_TRAINING_COMPLETED',json.dumps(report),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','validate','train']);a=p.parse_args()
 {'prepare':freeze_windows,'validate':real_validation,'train':train}[a.stage]()
