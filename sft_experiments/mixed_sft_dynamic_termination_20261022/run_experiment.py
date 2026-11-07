"""Gated calibration and single-GPU runs; no full-data training entrypoint."""
from experiment_paths import diagnostic_path
from experiment_paths import experiment_root,benchmark_root,data_root,model_root,paths,require_path,script_path,stage_log,evaluation_analysis_root

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from datasets import load_from_disk
from transformers import set_seed
from llamafactory.data import SFTDataCollatorWith4DAttentionMask,get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model,load_tokenizer
from llamafactory.train.sft.termination_trainer import BoundaryCollator,TerminationTrainer

ROOT=Path(__file__).resolve().parent
BASE=paths()['base_model']
MODES={'S0':'none','S1':'margin','S2':'dynamic','S1-M':'matched_margin'}


def config(arm,coef,smoke=False):
    return dict(model_name_or_path=str(require_path('base_model')),stage='sft',do_train=True,finetuning_type='lora',
        lora_target='all',lora_rank=64,lora_alpha=128,lora_dropout=.05,additional_target='lm_head',
        dataset='mixed_sft_final_49800',dataset_dir=str(require_path('source_dataset')),
        template='qwen3',cutoff_len=4096,packing=False,train_on_prompt=False,
        termination_dataset_path=str(data_root()/'prepared/dataset'),termination_loss_mode=MODES[arm],
        termination_loss_coef=coef,output_dir=str(model_root()/('smoke' if smoke else 'runs')/arm),
        logging_steps=10,save_steps=375,save_total_limit=1,save_only_model=False,report_to='none',
        overwrite_output_dir=False,per_device_train_batch_size=1,gradient_accumulation_steps=16,
        learning_rate=1e-4,num_train_epochs=3.,max_steps=1 if smoke else -1,lr_scheduler_type='cosine',
        warmup_ratio=.05,bf16=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},flash_attn='fa2',seed=3407,
        data_seed=3407,optim='adamw_torch',dataloader_num_workers=0,
        remove_unused_columns=False,save_strategy='no' if smoke else 'steps',
        include_num_input_tokens_seen=True)


def build(cfg):
    set_seed(3407)
    ma,da,ta,fa,ga=get_train_args(cfg)
    tokenmod=load_tokenizer(ma);token=tokenmod['tokenizer']
    template=get_template_and_fix_tokenizer(token,da)
    model=load_model(token,ma,fa,True)
    collator=SFTDataCollatorWith4DAttentionMask(template=template,model=model,pad_to_multiple_of=8,
        label_pad_token_id=-100,block_diag_attn=False,neat_packing=False,
        attn_implementation=getattr(model.config,'_attn_implementation',None),compute_dtype=ma.compute_dtype,**tokenmod)
    trainer=TerminationTrainer(model=model,args=ta,finetuning_args=fa,
        data_collator=BoundaryCollator(collator),train_dataset=load_from_disk(fa.termination_dataset_path),
        **tokenmod)
    log_dir = Path(os.environ['SFT_DIAGNOSTICS_DIR']) if os.environ.get('SFT_DIAGNOSTICS_DIR') else diagnostic_path(Path(cfg['output_dir'])/'termination_steps.jsonl').parent
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer.term_log = log_dir / 'termination_steps.jsonl'
    return trainer


def init_hash(trainer):
    h=hashlib.sha256()
    for name,p in trainer.model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            flat=p.detach().reshape(-1)
            for start in range(0,flat.numel(),1048576):
                h.update(flat[start:start+1048576].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def calibrate():
    trainer=build(config('S1-M',0.,True))
    trainer.model=trainer.accelerator.prepare_model(trainer.model)
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    loader=iter(trainer.get_train_dataloader())
    rows=[]
    before=init_hash(trainer)
    for i in range(8):
        batches=[next(loader) for _ in range(16)]
        rows.append(trainer.gradient_probe(trainer.model,batches,-i-1))
        print('CALIBRATION',i,rows[-1],flush=True)
    assert before==init_hash(trainer),'Calibration changed parameters'
    r=max(float(np.quantile([x['ordinary_ratio'] for x in rows],.9)),
          float(np.quantile([x['dynamic_ratio'] for x in rows],.9)))
    if not math.isfinite(r) or r<=0:raise ValueError('Invalid gradient calibration')
    coef=float(format(.1/r,'.2g'))
    result={'coefficient':coef,'ratio_p90':r,'rows':rows,'initial_trainable_sha256':before,
            'purpose':'initial stability calibration, not trajectory matching'}
    (ROOT/'calibration.json').write_text(json.dumps(result,indent=2))
    print('CALIBRATION_COMPLETE',coef,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','smoke','train'])
    p.add_argument('--arm',choices=list(MODES),default='S1-M');a=p.parse_args()
    if a.stage=='calibrate':return calibrate()
    cal=json.loads((ROOT/'calibration.json').read_text())
    if a.stage=='train':
        gates=json.loads((ROOT/'validation_gates.json').read_text())
        assert gates['unit_tests'] and gates['human_review_32'] and gates['real_smoke'] and gates['integration_tests']
    cfg=config(a.arm,cal['coefficient'],a.stage=='smoke')
    directory=diagnostic_path(Path(cfg['output_dir'])/'completed.json').parent;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'completed.json').exists():raise FileExistsError('Run is already completed')
    (directory/'resolved_config.json').write_text(json.dumps(cfg,indent=2))
    trainer=build(cfg)
    assert init_hash(trainer)==cal['initial_trainable_sha256'],'Initial trainable parameters differ'
    start=time.time()
    result=trainer.train()
    if a.stage=='train':trainer.save_model();trainer.save_state();trainer.save_metrics('train',result.metrics)
    torch.cuda.synchronize()
    report={'stage':a.stage,'arm':a.arm,'global_step':trainer.state.global_step,
            'seconds':time.time()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
            'metrics':result.metrics}
    assert report['global_step']==(1 if a.stage=='smoke' else 1125)
    (directory/'completed.json').write_text(json.dumps(report,indent=2))
    print('COMPLETED',report,flush=True)


if __name__=='__main__':main()
