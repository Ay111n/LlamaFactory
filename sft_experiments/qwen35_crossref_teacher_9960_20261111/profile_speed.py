"""Isolated paired performance probe; never changes either production training run."""
import argparse,json,os,time,sys,statistics,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
FRAMEWORK=HERE.parents[1];PROJECT=FRAMEWORK.parent
R=HERE/'runtime/speed_diagnosis'

def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def main(arm):
 import torch
 from datasets import load_from_disk
 from transformers import TrainerCallback
 from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
 from llamafactory.train.tuner import run_exp
 qrun=HERE.name;srun='crossref_teacher_9960_20261111';run=qrun if arm=='qwen35' else srun
 cfg=json.loads((FRAMEWORK/'sft_experiments'/run/'runtime/crossref.json').read_text())
 source=load_from_disk(cfg['tokenized_path']);source=source['train'] if hasattr(source,'keys') and 'train' in source else source
 choice=R/'selection.json'
 if not choice.exists():
  assert arm=='qwen35'
  raw=[json.loads(x) for x in (PROJECT/'finetuning/sft/mixed_sft/final_data'/srun/'crossref.jsonl').read_text().splitlines()]
  lengths=[len(x) for x in source['input_ids']];groups=collections.defaultdict(list)
  for i,row in enumerate(raw):groups[(row['language'],row['view'])].append(i)
  indices=[];stats={}
  for key,ids in groups.items():
   mean=statistics.mean(lengths[i] for i in ids);n=2 if key[0]=='en' else 1
   indices.extend(sorted(ids,key=lambda i:(abs(lengths[i]-mean),i))[:n]);stats[str(key)]=dict(mean_tokens=mean,count=len(ids))
  assert len(indices)==4
  write(choice,dict(indices=indices,rows=[{k:raw[i][k] for k in ['uid','source_uid','language','view']}|dict(qwen_tokens=lengths[i]) for i in indices],group_statistics=stats))
 selection=json.loads(choice.read_text());indices=selection['indices'];subset=source.select(indices)
 cache=PROJECT/'finetuning/sft/mixed_sft/processed_data'/qrun/'speed_diagnosis'/arm
 assert not cache.exists();subset.save_to_disk(str(cache))
 output=R/(arm+'_trainer')
 cfg.update(tokenized_path=str(cache),output_dir=str(output),gradient_accumulation_steps=4,max_steps=3,logging_steps=1,save_strategy='no',save_only_model=True,plot_loss=False,report_to='none',resume_from_checkpoint=None)
 cfg.pop('save_steps',None)
 write(R/(arm+'_config.json'),cfg)
 if arm=='qwen35':
  from memory_ce import install
  install()
 class Measure(TrainerCallback):
  def __init__(self):self.timings=[];self.prof=None;self.hooks=[]
  def on_train_begin(self,args,state,control,model=None,**kwargs):
   for name,module in model.named_modules():
    category=None
    if name.endswith('.lm_head') or name=='lm_head':category='lm_head_forward'
    elif name.endswith('.linear_attn'):category='linear_attention_forward'
    elif name.endswith('.self_attn'):category='full_attention_forward'
    elif name.endswith('.mlp'):category='mlp_forward'
    if category:
     stack=[]
     def pre(m,a,category=category,stack=stack):
      ctx=torch.profiler.record_function('component:'+category);ctx.__enter__();stack.append(ctx)
     def post(m,a,o,stack=stack):stack.pop().__exit__(None,None,None)
     self.hooks.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
  def on_step_begin(self,args,state,control,**kwargs):
   torch.cuda.synchronize();self.started=time.perf_counter()
   if state.global_step==1:
    self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_stack=False)
    self.prof.__enter__()
  def on_step_end(self,args,state,control,**kwargs):
   torch.cuda.synchronize();self.timings.append(dict(step=state.global_step,seconds=time.perf_counter()-self.started,profiled=state.global_step==2))
   if state.global_step==2:
    self.prof.__exit__(None,None,None)
    kernels=collections.defaultdict(lambda:[0,0.])
    for e in self.prof.events():
     if str(e.device_type).endswith('CUDA'):
      kernels[e.name][0]+=1;kernels[e.name][1]+=e.time_range.elapsed_us()
    ops=[]
    for e in self.prof.key_averages():
     ops.append(dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,self_device_us=e.self_device_time_total,total_device_us=e.device_time_total))
    write(R/(arm+'_profile.json'),dict(kernels=sorted([dict(name=k,count=v[0],gpu_us=v[1]) for k,v in kernels.items()],key=lambda x:-x['gpu_us']),operations=sorted(ops,key=lambda x:-x['self_device_us']),cpu_operations=sorted(ops,key=lambda x:-x['self_cpu_us'])[:80]))
    self.prof=None
   write(R/(arm+'_timings.json'),dict(timings=self.timings,source_indices=indices,input_lengths=[len(x) for x in subset['input_ids']],microbatches_per_update=4,production_training_unchanged=True))
  def on_train_end(self,args,state,control,**kwargs):
   for hook in self.hooks:hook.remove()
 callback=Measure()
 # Probe artifacts are timing/state JSON only; never export a probe model or checkpoint.
 CustomSeq2SeqTrainer.save_model=lambda *a,**k:None
 run_exp(cfg,callbacks=[callback])
 assert len(callback.timings)==3
 write(R/(arm+'_completion.json'),dict(status='passed',steps=3,model_exported=False,production_training_unchanged=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['qwen35','sft_new2'],required=True);a=p.parse_args();main(a.arm)
