import sys,json,time,hashlib,math,statistics
from itertools import islice
from pathlib import Path
import torch
from datasets import DatasetDict
from torch.utils.data import SequentialSampler
from transformers import TrainerCallback
from llamafactory.data import get_dataset,get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.tuner import run_exp

here=Path(__file__).resolve().parent
runtime=here/'runtime'
cfg=json.loads((here/'recipe.json').read_text())
source=Path(cfg['dataset_dir'])/'en_cot_10000.jsonl'
manifest=json.loads((source.parent/'manifest.json').read_text())
with source.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['outputs'][source.name]['sha256']
m,d,t,*_=get_train_args(cfg)
module=load_tokenizer(m)
tok=module['tokenizer']
template=get_template_and_fix_tokenizer(tok,d)
data=get_dataset(template,m,d,t,stage='sft',**module)['train_dataset']
assert len(data)==10000
lengths=[]; uids=[]; empty=0; supervised=0
with source.open() as f:
    for i in range(0,10000,128):
        raw=[json.loads(l) for l in islice(f,128)]
        q=[f"<|im_start|>user\n{r['messages'][0]['content']}<|im_end|>\n<|im_start|>assistant\n" for r in raw]
        a=[r['messages'][1]['content']+'<|im_end|>\n' for r in raw]
        x=tok(q,add_special_tokens=False,verbose=False).input_ids
        y=tok(a,add_special_tokens=False,verbose=False).input_ids
        cached=data[i:i+len(raw)]
        for r,p,b,ids,labels in zip(raw,x,y,cached['input_ids'],cached['labels'],strict=True):
            assert ids==p+b and labels==[-100]*len(p)+b and len(ids)<=8192,r['uid']
            assert r['messages'][1]['content'].startswith('<think>\n') and '</think>\n\n' in r['messages'][1]['content']
            assert tok.convert_tokens_to_ids('<|im_end|>') in b
            empty+=r['messages'][1]['content'].startswith('<think>\n\n</think>')
            lengths.append(len(ids)); uids.append(r['uid']); supervised+=len(b)
    assert not f.read(1)
assert len(lengths)==len(set(uids))==10000 and empty==7283
assert sum(lengths)==manifest['outputs'][source.name]['total_tokens']
def save(name,obj):
    p=runtime/name
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(p)
save('preflight.json',dict(rows=10000,empty_think=empty,nonempty_think=10000-empty,max_tokens=max(lengths),
     total_tokens=sum(lengths),supervised_tokens=supervised,mask_exact=True,zero_truncated=True,
     uid_order_sha256=hashlib.sha256(('\n'.join(uids)+'\n').encode()).hexdigest()))
assert hashlib.sha256(('\n'.join(uids)+'\n').encode()).hexdigest()==manifest['uid_order_sha256']
Full=sorted(range(10000),key=lambda i:(lengths[i],uids[i]))
n=16
selected=[]
for k in range(n):
    b=Full[k*10000//n:(k+1)*10000//n]
    mu=statistics.mean(lengths[i] for i in b)
    selected.append(min(b,key=lambda i:(abs(lengths[i]-mu),uids[i])))
steps=4
per_step=sum(lengths[i] for i in selected)
save('selection.json',{'uids':[uids[i] for i in selected],'lengths':[lengths[i] for i in selected],'tokens_per_step':per_step})
cache=runtime/'encoded_probe'
DatasetDict(train=data.select(selected*steps)).save_to_disk(str(cache))
cfg.update(tokenized_path=str(cache),output_dir=str(runtime/'trainer'),max_steps=steps,
           save_strategy='no',save_only_model=True,disable_tqdm=True,plot_loss=False,learning_rate=0.)
cfg.pop('save_steps',None)
save('probe_config.json',cfg)
sys.path.insert(0,str(here.parent/'qwen35_crossref_teacher_9960_20261111'))
from memory_ce import install
install(warmup=False)
CustomSeq2SeqTrainer.save_model=lambda *a,**k:None
CustomSeq2SeqTrainer._get_train_sampler=lambda self,*a,**k:SequentialSampler(self.train_dataset)
orig=CustomSeq2SeqTrainer.training_step
done=[]
def track(self,model,inputs,*args,**kwargs):
    i=selected[len(done)%n]
    assert int(inputs['attention_mask'].sum().item())==lengths[i]
    assert torch.all(inputs['labels'][inputs['attention_mask']==0]==-100)
    v=orig(self,model,inputs,*args,**kwargs)
    done.append(v.detach())
    return v
CustomSeq2SeqTrainer.training_step=track
records=[]
class Time(TrainerCallback):
    def on_step_begin(self,args,state,control,**kw):
        torch.manual_seed(3407)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start=time.perf_counter()
    def on_step_end(self,args,state,control,**kw):
        torch.cuda.synchronize()
        sec=time.perf_counter()-self.start
        losses=done[-n:]
        assert len(losses)==n and all(math.isfinite(float(v)) for v in losses)
        row=dict(step=state.global_step,seconds=sec,tokens=per_step,loss=sum(map(float,losses)),
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        records.append(row)
        save('steps.json',records)
        print('PROBE_STEP',json.dumps(row),flush=True)
run_exp(cfg,callbacks=[Time()])
assert len(records)==steps and len(done)==steps*n
rates=[per_step/r['seconds'] for r in records[1:]]
result=dict(steady_tokens_per_second=rates,median_rate=statistics.median(rates),
            projected_hours=sum(lengths)/statistics.median(rates)/3600,
            observed_hours_range=[sum(lengths)/max(rates)/3600,sum(lengths)/min(rates)/3600],
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in records),
            peak_reserved_gib=max(r['peak_reserved_gib'] for r in records),
            official_training_started=False)
save('result.json',result)
print('RESULT',json.dumps(result),flush=True)
