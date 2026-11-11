"""Matched bilingual CE vs reference-privileged KD, single GPU 5 pilot."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
FRAMEWORK=HERE.parents[1]
ROOT=FRAMEWORK.parent
RUN=HERE.name
RUNTIME=HERE/'runtime'
DATA=ROOT/'finetuning/sft/mixed_sft/final_data/crossref_adaptation_pilot_2000_20261110'
CACHE=ROOT/'finetuning/sft/mixed_sft/processed_data'/RUN
MODELS=ROOT/'finetuning/model/sft'/RUN
BASE=Path('/home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2')
TEACHER=ROOT/'finetuning/model/sft/crossref_teacher_9960_20261111/merged/crossref'
SEED=3407

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
def emit(x):print(json.dumps(x,ensure_ascii=True),flush=True)

def config(arm):
 d=read(HERE.parent/'crossref_adaptation_pilot_2000_20261110/training_recipe.json')
 # One rolling recoverable checkpoint per arm, at most two across this experiment.
 d.update(model_name_or_path=str(BASE),dataset='baseline',dataset_dir=str(DATA),
          output_dir=str(MODELS/arm),save_total_limit=1,save_strategy='steps',save_steps=250,
          dataloader_num_workers=0,plot_loss=False,remove_unused_columns=False)
 return d

def setup_args(arm,cpu=False):
 from llamafactory.hparams import get_train_args
 d=config(arm)
 if cpu:d.update(use_cpu=True,bf16=False,flash_attn='disabled',preprocessing_num_workers=1)
 return get_train_args(d)

def prepare():
 import torch,transformers,peft,flash_attn
 from llamafactory.model import load_tokenizer
 from llamafactory.data import get_template_and_fix_tokenizer
 from transformers import AutoTokenizer
 assert Path(sys.executable).resolve()==Path('/opt/anaconda3/envs/training/bin/python').resolve()
 RUNTIME.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 assert not (CACHE/'examples.jsonl').exists()
 raw=rows(DATA/'baseline.jsonl');pairs=rows(DATA/'train_pairs.jsonl');pairmap={r['uid']:r for r in pairs}
 teacherdata=ROOT/'finetuning/sft/mixed_sft/final_data/crossref_teacher_9960_20261111'
 teacheruids={r['uid'] for r in rows(teacherdata/'train_pairs.jsonl')}
 assert len(raw)==4000 and len(pairs)==len(pairmap)==2000 and set(pairmap)<=teacheruids
 assert 'Math_no_think_0010880' not in pairmap
 assert not set(pairmap)&{r['uid'] for r in rows(teacherdata/'diagnostic_pairs.jsonl')}
 ma,da,ta,fa,ga=setup_args('baseline',True)
 tok=load_tokenizer(ma)['tokenizer'];template=get_template_and_fix_tokenizer(tok,da)
 tt=AutoTokenizer.from_pretrained(TEACHER)
 assert tok.get_vocab()==tt.get_vocab() and tok.chat_template==tt.chat_template
 wrapper=(ROOT/'benchmark/eval/analysis/teacher_probe_prompt_v2_20261110/teacher_prompt.txt').read_text(encoding='utf-8').strip()
 result=[];longest=0;teachermax=0
 for i,r in enumerate(raw):
  uid=r['source_uid'];lang=r['language'];assert lang in ['en','bo']
  response=r['conversations'][1]['value'];assert '<think>' not in response and '</think>' not in response
  src,target=template.encode_oneturn(tok,[{'role':'user','content':r['conversations'][0]['value']},{'role':'assistant','content':response}],r['system'])
  assert src and target and len(src)+len(target)<=8192
  meta={'uid':uid,'language':lang,'index':i,'prompt_length':len(src),'target':target}
  if lang=='bo':
   pair=pairmap[uid];content=wrapper
   for lng,key in [('english','question'),('english','answer'),('tibetan','question')]:content=content.replace('{'+lng+'.'+key+'}',pair[lng][key])
   prompt=tt.apply_chat_template([{'role':'system','content':'Respond in Tibetan.'},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   tp=tt.encode(prompt,add_special_tokens=False)
   body=tt.encode(pair['tibetan']['answer'],add_special_tokens=False)
   assert target==body+tt.encode('<|im_end|>\n',add_special_tokens=False)
   assert tt.encode(prompt+pair['tibetan']['answer'],add_special_tokens=False)==tp+body
   meta.update(teacher_input_ids=tp+target[:-1],teacher_prompt_length=len(tp))
   teachermax=max(teachermax,len(tp)+len(target)-1);assert teachermax<=32768
  result.append({'input_ids':src+target,'labels':[-100]*len(src)+target,'meta':meta})
  longest=max(longest,len(src)+len(target))
 output=CACHE/'examples.jsonl';output.write_text(''.join(json.dumps(r)+'\n' for r in result))
 for arm in ['baseline','kd']:save(RUNTIME/(arm+'_config.json'),config(arm))
 save(RUNTIME/'preflight.json',{'status':'passed','source_pairs':2000,'views':4000,'bo_views':sum(r['meta']['language']=='bo' for r in result),
      'max_student_tokens':longest,'max_teacher_tokens':teachermax,'examples_sha256':sha(output),
      'source_sha256':{n:sha(DATA/n) for n in ['train_pairs.jsonl','baseline.jsonl','manifest.json']},
      'teacher_merge_manifest_sha256':sha(TEACHER/'merge_manifest.json'),'base':str(BASE),'teacher':str(TEACHER),
      'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'flash_attn':flash_attn.__version__},
      'objective':'per-response CE averaged across 4000 en/bo views; bo-only 0.2 KL so dataset objective is 0.5 CE_en+0.5 CE_bo+0.1 KD_bo; native EOS and trailing newline included',
      'old_baseline_reused':False,'normalization_change':'fresh matched arms use response means; historical native token-weighted baseline is not primary control',
      'planned_steps':250,'checkpoint_limit_per_arm':1,'known_malformed_uid_present':False,'student_training_started':False})
 emit(read(RUNTIME/'preflight.json'))

def verify():
 p=read(RUNTIME/'preflight.json');assert sha(CACHE/'examples.jsonl')==p['examples_sha256']
 for n,h in p['source_sha256'].items():assert sha(DATA/n)==h
 assert sha(TEACHER/'merge_manifest.json')==p['teacher_merge_manifest_sha256']
 return p

def cache_teacher():
 import torch
 from transformers import AutoModelForCausalLM
 assert Path(sys.executable).resolve()==Path('/opt/anaconda3/envs/inference/bin/python').resolve()
 verify();torch.set_num_threads(4)
 out=CACHE/'teacher_hidden';out.mkdir(exist_ok=True)
 model=AutoModelForCausalLM.from_pretrained(TEACHER,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True)
 model.eval().requires_grad_(False)
 torch.save(model.lm_head.weight.detach().cpu(),CACHE/'teacher_head.pt')
 selected=[r for r in rows(CACHE/'examples.jsonl') if r['meta']['language']=='bo']
 started=time.time();checks=[];files={}
 with torch.inference_mode():
  for i,r in enumerate(selected):
   m=r['meta'];x=torch.tensor([m['teacher_input_ids']],device='cuda')
   h=model.model(input_ids=x,use_cache=False).last_hidden_state[0,m['teacher_prompt_length']-1:].contiguous()
   assert len(h)==len(m['target'])
   assert torch.isfinite(h).all()
   path=out/(m['uid']+'.pt');torch.save(h.cpu(),path);files[m['uid']]={'file':str(path),'sha256':sha(path),'shape':list(h.shape)}
   if i==0:
    reloaded=torch.load(path,weights_only=True).cuda();assert torch.equal(reloaded,h)
    target=torch.tensor(m['target'],device='cuda');v=model.lm_head(h[:64]).float().log_softmax(-1)
    v2=torch.nn.functional.linear(reloaded[:64],torch.load(CACHE/'teacher_head.pt',weights_only=True).cuda()).float().log_softmax(-1)
    assert torch.equal(v,v2)
    checks.append({'hidden_cache_exact':True,'head_logprobs_exact':True,'uid':m['uid']})
   if (i+1)%50==0:emit({'stage':'teacher_cache','completed':i+1,'total':2000,'seconds':time.time()-started})
 save(CACHE/'teacher_cache_manifest.json',{'status':'completed','samples':2000,'files':files,'head_sha256':sha(CACHE/'teacher_head.pt'),
      'examples_sha256':sha(CACHE/'examples.jsonl'),'teacher_merge_manifest_sha256':sha(TEACHER/'merge_manifest.json'),
      'checks':checks,'seconds':time.time()-started,'dtype':'bfloat16','no_logit_topk_approximation':True})
 emit({'stage':'teacher_cache_complete','seconds':time.time()-started})


def train(arm,smoke=False):
 import torch
 from transformers import TrainerCallback,set_seed
 from transformers.modeling_outputs import CausalLMOutput
 from llamafactory.model import load_model,load_tokenizer
 from llamafactory.data import get_template_and_fix_tokenizer
 from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
 from losses import response_losses
 assert Path(sys.executable).resolve()==Path('/opt/anaconda3/envs/training/bin/python').resolve()
 verify();torch.set_num_threads(4);set_seed(SEED)
 tag='smoke' if smoke else arm
 ma,da,ta,fa,ga=setup_args(arm)
 tok=load_tokenizer(ma)['tokenizer'];get_template_and_fix_tokenizer(tok,da)
 model=load_model(tok,ma,fa,is_trainable=True)
 assert not model.get_base_model().lm_head.weight.requires_grad
 allrows=rows(CACHE/'examples.jsonl')
 tw=None;cachemeta=None
 if arm=='kd':
  cachemeta=read(CACHE/'teacher_cache_manifest.json');assert cachemeta['status']=='completed'
  assert cachemeta['examples_sha256']==sha(CACHE/'examples.jsonl')
  assert cachemeta['head_sha256']==sha(CACHE/'teacher_head.pt')
  tw=torch.load(CACHE/'teacher_head.pt',weights_only=True).to('cuda')
 initial=hashlib.sha256()
 for n,p in model.named_parameters():
  if p.requires_grad:initial.update(n.encode());initial.update(p.detach().cpu().float().numpy().tobytes())
 initial_hash=initial.hexdigest()
 save(RUNTIME/(tag+'_initialization.json'),{'trainable_sha256':initial_hash,'seed':SEED,
       'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'smoke':smoke})
 def collate(rs):
  assert len(rs)==1
  r=rs[0];n=len(r['input_ids']);pad=(-n)%8
  return {'input_ids':torch.tensor([r['input_ids']+[tok.pad_token_id]*pad]),
          'attention_mask':torch.tensor([[1]*n+[0]*pad]),'labels':torch.tensor([r['labels']+[-100]*pad]),'meta':r['meta']}
 class PilotTrainer(CustomSeq2SeqTrainer):
  def __init__(self,**kw):
   super().__init__(**kw);self.model_accepts_loss_kwargs=False;self.window=[]
  def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None,**kwargs):
   m=inputs['meta'];body=model.get_base_model()
   h=body.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],use_cache=False).last_hidden_state
   start=m['prompt_length']-1;n=len(m['target']);h=h[0,start:start+n]
   target=torch.tensor(m['target'],device=h.device)
   th=None
   if arm=='kd' and m['language']=='bo':
    entry=cachemeta['files'][m['uid']];path=Path(entry['file'])
    assert sha(path)==entry['sha256']
    th=torch.load(path,weights_only=True).to(h.device)
    assert th.shape==h.shape
   head=body.lm_head.weight.detach().to(h.dtype)
   ce,kd=response_losses(h,head,target,th,tw if th is not None else None)
   weight=.2 if arm=='kd' and m['language']=='bo' else 0.
   loss=ce+weight*kd
   assert torch.isfinite(loss)
   self.window.append({'uid':m['uid'],'language':m['language'],'ce':float(ce.detach()),'kd':float(kd.detach()),'weighted_kd':float((weight*kd).detach())})
   return (loss,CausalLMOutput(loss=loss)) if return_outputs else loss
 class Progress(TrainerCallback):
  def on_step_end(self,args,state,control,**kw):
   rs=trainer.window;trainer.window=[]
   row={'step':state.global_step,'epoch':state.epoch,'time':time.time(),
        'ce':sum(r['ce'] for r in rs)/len(rs),'weighted_kd':sum(r['weighted_kd'] for r in rs)/len(rs),
        'microbatches':len(rs),'order':[(r['uid'],r['language']) for r in rs],
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
   with (RUNTIME/(tag+'_steps.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
   save(RUNTIME/(tag+'_progress.json'),row)
   if state.global_step%5==0:emit({k:v for k,v in row.items() if k!='order'})
   if smoke:control.should_training_stop=True;control.should_save=False
   return control
 if smoke:
  # Two short examples and both longest-language examples test backward/memory.
  short=min(allrows,key=lambda r:len(r['input_ids']))
  longest=max(allrows,key=lambda r:len(r['input_ids']))
  longestbo=max((r for r in allrows if r['meta']['language']=='bo'),key=lambda r:len(r['input_ids']))
  bo=min((r for r in allrows if r['meta']['language']=='bo'),key=lambda r:len(r['input_ids']))
  trainrows=[short,bo,longest,longestbo];ta.max_steps=1;ta.gradient_accumulation_steps=4;ta.save_strategy='no';ta.warmup_ratio=0.;ta.warmup_steps=0
  ta.output_dir=str(CACHE/'smoke_no_checkpoint');ta.report_to=[]
 else:
  trainrows=allrows
  assert not (MODELS/arm).exists(), 'No overwrite or implicit resume'
 trainer=PilotTrainer(model=model,args=ta,finetuning_args=fa,processor=None,tokenizer=tok,model_args=ma,
       data_collator=collate,train_dataset=trainrows,callbacks=[Progress()])
 if smoke:
  # Native model loss equality before any update, with dropout disabled.
  model.eval();sample=min(allrows,key=lambda r:len(r['input_ids']));batch=collate([sample]);batch={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in batch.items()}
  with torch.no_grad():
   native=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],labels=batch['labels'],use_cache=False).loss
   h=model.get_base_model().model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).last_hidden_state
   start=sample['meta']['prompt_length']-1;target=torch.tensor(sample['meta']['target'],device='cuda')
   ce=response_losses(h[0,start:start+len(target)],model.get_base_model().lm_head.weight.detach().to(h.dtype),target)[0]
   error=float(abs(native-ce));assert error<.02,(native,ce)
  model.train()
 started=time.time();out=trainer.train()
 if smoke:
  delta=any(p.grad is not None for p in model.parameters() if p.requires_grad)
  final=hashlib.sha256()
  for n,p in model.named_parameters():
   if p.requires_grad:final.update(n.encode());final.update(p.detach().cpu().float().numpy().tobytes())
  assert final.hexdigest()!=initial_hash
  save(RUNTIME/'smoke.json',{'status':'passed','native_ce_error':error,'steps':trainer.state.global_step,
       'finite_logged_losses':all(math.isfinite(x['loss']) for x in trainer.state.log_history if 'loss' in x),
       'longest_student_tokens':len(longest['input_ids']),'longest_bo_tokens':len(longestbo['input_ids']),
       'parameters_updated_only_in_discarded_smoke_model':True,'no_checkpoint_saved':True,
       'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-started})
 else:
  trainer.save_model();trainer.save_state();trainer.save_metrics('train',out.metrics)
  assert trainer.state.global_step==250
  assert len(list((MODELS/arm).glob('checkpoint-*')))<=1
  save(RUNTIME/(arm+'_complete.json'),{'status':'completed','steps':trainer.state.global_step,'seconds':time.time()-started,
       'adapter_sha256':sha(MODELS/arm/'adapter_model.safetensors'),'initialization_sha256':initial_hash,'metrics':out.metrics})
 emit({'arm':arm,'smoke':smoke,'seconds':time.time()-started,'stage':'completed'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','cache','train','smoke']);p.add_argument('--arm',choices=['baseline','kd'],default='kd');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='cache':cache_teacher()
 else:train(a.arm,a.command=='smoke')
