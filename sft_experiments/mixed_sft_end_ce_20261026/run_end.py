"""Gated end-only experiment runner, sharing the original model/config builder."""
import argparse, importlib.util, json, os, time, hashlib
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
from experiment_paths import *
from llamafactory.train.sft.end_ce_trainer import EndCETrainer

ARMS={'E1':'end_ce','E2':'dynamic_end_ce','E1-M':'matched_end_ce'}
spec=importlib.util.spec_from_file_location('end_legacy',require_path('legacy_experiment_root')/'run_experiment.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
legacy.TerminationTrainer=EndCETrainer
_get_args=legacy.get_train_args
def single_model_args(*args,**kwargs):
    # The second visible GPU is a storage device, not a training replica.
    with patch('torch.cuda.device_count',return_value=1):
        values=_get_args(*args,**kwargs)
    # Visible device 1 is a gradient-buffer pool, NOT a DataParallel replica.
    values[2]._n_gpu=1
    return values
legacy.get_train_args=single_model_args

def config(arm,smoke=False):
    old=legacy.MODES
    legacy.MODES=ARMS
    try: cfg=legacy.config(arm,.0012,smoke)
    finally: legacy.MODES=old
    cfg['termination_dataset_path']=str(require_path('prepared_dataset'))
    if smoke: cfg.update(learning_rate=1e-6,warmup_ratio=0.)
    return cfg

def load_windows():
    windows=torch.load(require_path('fixed_probe_cache'),map_location='cpu',weights_only=False)
    manifest=json.loads(require_path('fixed_probe_manifest').read_text())
    sig=[]
    for w in windows:
        h=hashlib.sha256()
        for b in w:
            for k in ['input_ids','labels','attention_mask']:h.update(b[k].contiguous().numpy().tobytes())
        sig.append(h.hexdigest())
    assert sig==manifest['signatures'] and len(windows)==8
    return windows,sig

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','smoke','train']);p.add_argument('--arm',choices=list(ARMS),default='E1');a=p.parse_args()
    cfg=config(a.arm,a.stage!='train')
    if a.stage=='calibrate': cfg['output_dir']=str(model_root()/'calibration')
    out=diagnostic_path(Path(cfg['output_dir'])/'completed.json').parent
    if (out/'completed.json').exists() or (out/'termination_steps.jsonl').exists():raise FileExistsError(out)
    if a.stage!='calibrate': assert json.loads((ROOT/'initial_gradient_gate.json').read_text())['gate_pass']
    if a.stage=='train': assert all(json.loads((ROOT/'validation_gates.json').read_text()).values())
    out.mkdir(parents=True,exist_ok=True)
    (out/'resolved_config.json').write_text(json.dumps(cfg,indent=2))
    os.environ['SFT_DIAGNOSTICS_DIR'] = str(out)
    trainer=legacy.build(cfg)
    assert trainer.args.n_gpu==1
    windows,sig=load_windows();trainer.end_probe_windows=windows
    (out/'fixed_probe_windows.json').write_text(json.dumps(sig,indent=2))
    initial={'seed':3407,'data_seed':3407,'base':str(require_path('base_model')),
      'trainable_count':sum(p.numel() for p in trainer.term_params),
      'trainable_dtypes':sorted({str(p.dtype) for p in trainer.term_params}),
      'sampled_initial_values':{n:p.detach().reshape(-1)[:8].float().cpu().tolist() for n,p in trainer.model.named_parameters() if p.requires_grad and ('layers.0.' in n or 'lm_head.modules_to_save' in n)}}
    (out/'initialization.json').write_text(json.dumps(initial,indent=2))
    assert initial['trainable_count']==835190784
    if a.stage=='calibrate':
        trainer.model=trainer.accelerator.prepare_model(trainer.model)
        trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        rows=[trainer._probe_window(trainer.model,w,-1,i) for i,w in enumerate(windows)]
        summary={}
        for key in ['end','dynamic_end','matched_end']:
            vals=[r[key+'_R'] for r in rows]
            summary[key]={k:float(np.quantile(vals,q)) for k,q in [('min',0),('p10',.1),('p50',.5),('p90',.9),('p95',.95),('max',1)]}
            summary[key]['gate_pass']=all(np.isfinite(vals)) and .001<=summary[key]['p90']<=1.
        result={'gate_pass':all(x['gate_pass'] for x in summary.values()),'coefficient':.0012,'rows':rows,'summary':summary}
        (ROOT/'initial_gradient_gate.json').write_text(json.dumps(result,indent=2))
        (diagnostics_root()/'initial_gradient_gate.json').write_text(json.dumps(result,indent=2))
        if not result['gate_pass']: raise RuntimeError('Initial strength gate failed; no lambda retuning')
        return
    start=time.time()
    if a.stage=='smoke': trainer.gradient_probe=lambda *a,**k:None
    versions=[p._version for p in trainer.term_params]
    result=trainer.train()
    assert trainer.state.global_step==(1 if a.stage=='smoke' else 1125)
    assert any(p._version!=v for p,v in zip(trainer.term_params,versions)),'No update occurred'
    if a.stage=='train':
        trainer.save_model();trainer.save_state();trainer.save_metrics('train',result.metrics)
        trainer.probe_after_update=True
        trainer.gradient_probe(trainer.model,windows[0],1125)
    done={'arm':a.arm,'stage':a.stage,'global_step':trainer.state.global_step,'seconds':time.time()-start,
          'metrics':result.metrics,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
          'buffer_device':os.environ.get('END_BUFFER_DEVICE','cpu')}
    (out/'completed.json').write_text(json.dumps(done,indent=2));print('COMPLETED',json.dumps(done),flush=True)
if __name__=='__main__':main()
