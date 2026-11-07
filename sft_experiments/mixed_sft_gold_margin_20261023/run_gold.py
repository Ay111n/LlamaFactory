"""Reuse the validated LF model/collator/config path, adding Gold objectives."""
import argparse, importlib.util, json, math, time, hashlib
from pathlib import Path
import numpy as np
import torch
from experiment_paths import *
from llamafactory.train.sft.gold_termination_trainer import GoldTerminationTrainer

ROOT=experiment_root()
MODES={'G1':'gold_margin','G2':'dynamic_gold_margin','G3':'dynamic_boundary_ce',
       'G1-M':'matched_gold_margin','G3-M':'matched_dynamic_boundary_ce'}
spec=importlib.util.spec_from_file_location('legacy_gold_build',require_path('legacy_experiment_root')/'run_experiment.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
legacy.ROOT=ROOT;legacy.MODES=MODES;legacy.TerminationTrainer=GoldTerminationTrainer
def config(arm,smoke=False):
    cfg=legacy.config(arm,.0012,smoke)
    cfg['termination_dataset_path']=str(require_path('prepared_dataset'))
    if smoke:
        # A one-step warmup would make the first LR zero and test no real update.
        cfg.update(learning_rate=1e-6,warmup_ratio=0.)
    return cfg
def build(arm,smoke=False):return legacy.build(config(arm,smoke))
def calibrate():
    trainer=build('G1',True)
    trainer.model=trainer.accelerator.prepare_model(trainer.model)
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    before=legacy.init_hash(trainer)
    expected=json.loads((require_path('legacy_experiment_root')/'calibration.json').read_text())['initial_trainable_sha256']
    assert before==expected
    loader=iter(trainer.get_train_dataloader());rows=[]
    for i in range(8):
        batches=[next(loader) for _ in range(16)]
        rows.append(trainer.gradient_probe(trainer.model,batches,-i-1))
        print('GOLD_PROBE',i,json.dumps(rows[-1]),flush=True)
    assert before==legacy.init_hash(trainer),'Diagnostic modified parameters'
    summary={}
    for k in ('gold','dynamic_gold','dynamic_ce'):
        vals=[r[k+'_R'] for r in rows]
        if not all(math.isfinite(v) for v in vals):raise FloatingPointError(k)
        summary[k]={n:float(np.quantile(vals,p)) for n,p in [('min',0),('p10',.1),('p50',.5),('p90',.9),('p95',.95),('max',1)]}
        summary[k]['gate_pass']=.001<=summary[k]['p90']<=1
    passed=all(x['gate_pass'] for x in summary.values())
    result={'coefficient':.0012,'initial_trainable_sha256':before,'rows':rows,'summary':summary,
            'gate_pass':passed,'scope':'8 fixed effective batches; initial configuration diagnostic only'}
    (ROOT/'gold_calibration.json').write_text(json.dumps(result,indent=2))
    print('CALIBRATION_GATE',passed,json.dumps(summary),flush=True)
    if not passed:raise RuntimeError('Diagnostic gate failed; no training and no lambda retuning')
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','smoke','train']);p.add_argument('--arm',choices=list(MODES),default='G1');a=p.parse_args()
    if a.stage=='calibrate':return calibrate()
    calibration=json.loads((ROOT/'gold_calibration.json').read_text());assert calibration['gate_pass']
    if a.stage=='train':assert all(json.loads((ROOT/'validation_gates.json').read_text()).values())
    cfg=config(a.arm,a.stage=='smoke');out=diagnostic_path(Path(cfg['output_dir'])/'completed.json').parent
    if (out/'termination_steps.jsonl').exists() or (out/'completed.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);(out/'resolved_config.json').write_text(json.dumps(cfg,indent=2))
    os.environ['SFT_DIAGNOSTICS_DIR'] = str(out)
    trainer=legacy.build(cfg)
    initial=legacy.init_hash(trainer)
    assert initial==calibration['initial_trainable_sha256']
    if a.stage=='train':
        # Replay all eight calibration windows, without advancing training RNG.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            loader=iter(trainer.get_train_dataloader())
            windows=[]
            for i in range(8):
                batch=[next(loader) for _ in range(16)]
                cpu=[{k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in b.items()} for b in batch]
                count=sum(int(b['labels'][:,1:].ne(-100).sum()) for b in cpu)
                thinks=sum(bool(p) for b in cpu for p in b['term_positions'])
                assert count==calibration['rows'][i]['window_ce_tokens'] and thinks==calibration['rows'][i]['window_think']
                windows.append(cpu)
            trainer.gold_probe_windows=windows
            del loader,batch
        signature=[]
        for window in windows:
            h=hashlib.sha256()
            for b in window:
                for key in ['input_ids','labels','attention_mask']:h.update(b[key].contiguous().numpy().tobytes())
            signature.append(h.hexdigest())
        (out/'fixed_probe_windows.json').write_text(json.dumps(signature,indent=2))
    start=time.time();result=trainer.train()
    if a.stage=='train':trainer.save_model();trainer.save_state();trainer.save_metrics('train',result.metrics)
    torch.cuda.synchronize()
    if a.stage=='smoke':assert legacy.init_hash(trainer)!=initial,'Smoke optimizer did not change parameters'
    report={'arm':a.arm,'stage':a.stage,'global_step':trainer.state.global_step,'seconds':time.time()-start,
            'peak_memory_bytes':torch.cuda.max_memory_allocated(),'metrics':result.metrics}
    assert report['global_step']==(1 if a.stage=='smoke' else 1125)
    (out/'completed.json').write_text(json.dumps(report,indent=2));print('COMPLETED',json.dumps(report),flush=True)
if __name__=='__main__':main()
