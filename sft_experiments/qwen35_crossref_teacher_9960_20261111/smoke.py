"""Exercise a real longest-example backward/update before the full Qwen3.5 run."""
import json,os,math,time
from pathlib import Path
import teacher
from datasets import load_from_disk

teacher.verify_preflight()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
source=load_from_disk(str(teacher.CACHE/'crossref'))
if hasattr(source,'keys') and 'train' in source:source=source['train']
lengths=[len(x) for x in source['input_ids']]
longest=max(range(len(lengths)),key=lengths.__getitem__)
selected=source.select([0,longest])
cache=teacher.CACHE/'longest_smoke'
if cache.exists():assert load_from_disk(str(cache)).to_dict()==selected.to_dict()
else:selected.save_to_disk(str(cache))
cfg=teacher.read(teacher.RUNTIME/'crossref.json')
cfg.update(tokenized_path=str(cache),output_dir=str(teacher.MODELS/('longest_smoke_attempt'+os.environ.get('SMOKE_ATTEMPT','1'))),gradient_accumulation_steps=1,
           max_steps=2,logging_steps=1,save_strategy='no',save_only_model=True,plot_loss=False)
teacher.write(teacher.RUNTIME/'smoke_config.json',cfg)
import torch
from memory_ce import install
install()
from llamafactory.train.tuner import run_exp
started=time.time();run_exp(cfg)
s=teacher.read(Path(cfg['output_dir'])/'trainer_state.json')
assert s['global_step']==2
assert all(math.isfinite(r['loss']) for r in s['log_history'] if 'loss' in r)
teacher.write(teacher.RUNTIME/'smoke_passed.json',dict(status='passed',global_step=2,longest_row_index=longest,
    selected_lengths=[lengths[0],lengths[longest]],max_allocated_bytes=torch.cuda.max_memory_allocated(),
    elapsed_seconds=time.time()-started,log_history=s['log_history'],full_training_restarts_from_original=True,memory_ce_sha256=teacher.sha(teacher.HERE/'memory_ce.py')))
