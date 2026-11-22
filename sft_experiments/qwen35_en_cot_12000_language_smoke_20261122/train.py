import json, math, time, sys
from pathlib import Path
from itertools import islice
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TrainerCallback
import torch

here=Path(__file__).resolve().parent
root=here.parents[2]
runtime=here/'runtime'
runtime.mkdir(exist_ok=True)
previous=here.parent/'qwen35_crossref_teacher_9960_20261111'
cfg=json.loads((previous/'runtime/crossref.json').read_text())
data=root/'finetuning/sft/mixed_sft/final_data/bo_cot_sft_12000_three_variants_20261122'
file=data/'en_cot_12000.jsonl'
out=root/'finetuning/model/sft'/here.name
assert not out.exists()
tok=AutoTokenizer.from_pretrained(cfg['model_name_or_path'],local_files_only=True)
cache=runtime/'encoded'
def encode():
    with file.open() as stream:
        while batch:=list(islice(stream,128)):
            rows=list(map(json.loads,batch))
            q=[f"<|im_start|>user\n{x['messages'][0]['content']}<|im_end|>\n<|im_start|>assistant\n" for x in rows]
            a=[x['messages'][1]['content']+'<|im_end|>\n' for x in rows]
            for p,r in zip(tok(q,add_special_tokens=False,verbose=False).input_ids,
                           tok(a,add_special_tokens=False,verbose=False).input_ids,strict=True):
                assert len(p)+len(r)<=10240
                yield dict(input_ids=p+r,labels=[-100]*len(p)+r,attention_mask=[1]*(len(p)+len(r)))
if not cache.exists():
    dataset=Dataset.from_generator(encode,cache_dir=str(runtime/'build_cache'))
    assert len(dataset)==12000
    DatasetDict(train=dataset).save_to_disk(str(cache))
cfg.update(tokenized_path=str(cache),output_dir=str(out),cutoff_len=10240,enable_thinking=True,
           gradient_accumulation_steps=4,max_steps=8,num_train_epochs=1,save_strategy='steps',
           save_steps=8,save_total_limit=1,save_only_model=True,dataloader_num_workers=0)
(runtime/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
sys.path.insert(0,str(previous))
from memory_ce import install
from llamafactory.train.tuner import run_exp
class Trace(TrainerCallback):
    def on_step_begin(self,*args,**kwargs):
        torch.cuda.synchronize()
        self.start=time.monotonic()
    def on_step_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        with (runtime/'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=state.global_step,seconds=time.monotonic()-self.start,
                                          peak_gib=torch.cuda.max_memory_allocated()/2**30))+'\n')
install()
run_exp(cfg,callbacks=[Trace()])
ckpt=out/'checkpoint-8'
assert (ckpt/'adapter_model.safetensors').is_file()
(runtime/'checkpoint.json').write_text(json.dumps(dict(checkpoint=str(ckpt),updates=8),indent=2)+'\n')
