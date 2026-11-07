"""Resume only S1-M from an explicit full checkpoint; never dispatch evaluation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from experiment_paths import experiment_root, model_root, require_path, diagnostic_path
from run_experiment import build, config, init_hash

ROOT=experiment_root()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args()
    checkpoint=Path(args.checkpoint).resolve()
    directory=(model_root()/'runs/S1-M').resolve()
    if checkpoint.parent!=directory:raise ValueError('Checkpoint must belong to the existing S1-M run')
    for name in ['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin']:
        if not (checkpoint/name).is_file():raise FileNotFoundError(checkpoint/name)
    saved=json.loads((checkpoint/'trainer_state.json').read_text())
    if saved['max_steps']!=1125 or saved['global_step']>=1125:raise ValueError('Unexpected checkpoint budget')
    if diagnostic_path(directory/'completed.json').exists():raise FileExistsError('S1-M already completed')
    gates=json.loads((ROOT/'validation_gates.json').read_text())
    assert all(gates[k] for k in ['unit_tests','human_review_32','real_smoke','integration_tests','calibration'])
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    attempt=ROOT/'resume_attempts'/stamp
    attempt.mkdir(parents=True,exist_ok=False)
    def status(stage,**extra):
        data=dict(stage=stage,arm='S1-M',checkpoint=str(checkpoint),resume_step=saved['global_step'],
                  target_step=1125,attempt=str(attempt),automatic_evaluation=False,full_training_started=False,
                  updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        (ROOT/'resume_status.json').write_text(json.dumps(data,indent=2))
        print(json.dumps(data),flush=True)
    try:
        status('validating_checkpoint')
        cal=json.loads((ROOT/'calibration.json').read_text())
        cfg=config('S1-M',cal['coefficient'])
        cfg['resume_from_checkpoint']=str(checkpoint)
        # Keep the source checkpoint and both subsequent checkpoints.
        cfg['save_total_limit']=3
        old=json.loads(diagnostic_path(directory/'resolved_config.json').read_text())
        changes={k:{'previous':old.get(k),'resumed':v} for k,v in cfg.items() if old.get(k)!=v}
        allowed={'model_name_or_path','dataset_dir','termination_dataset_path','output_dir','resume_from_checkpoint','save_total_limit'}
        if set(changes)-allowed:raise ValueError(f'Unexpected hyperparameter changes: {set(changes)-allowed}')
        hashes={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
        stale={}
        for name in ['termination_steps.jsonl','gradient_probes.jsonl','resolved_config.json']:
            p=diagnostic_path(directory/name)
            if p.exists():
                shutil.copy2(p,attempt/name)
                if p.suffix=='.jsonl':
                    rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
                    stale[name]={'rows':len(rows),'maximum_logged_step':max(x['step'] for x in rows),
                                 'uncheckpointed_rows':sum(x['step']>saved['global_step'] for x in rows)}
        manifest=dict(checkpoint_hashes=hashes,config_changes=changes,previous_logs=stale,
                      note='Existing full logs preserved. Active logs retain only checkpoint-consistent steps.')
        (attempt/'resume_manifest.json').write_text(json.dumps(manifest,indent=2))
        (attempt/'resolved_resume_config.json').write_text(json.dumps(cfg,indent=2))
        status('loading_model')
        trainer=build(cfg)
        assert init_hash(trainer)==cal['initial_trainable_sha256'],'Base initialization identity mismatch'
        # Reconstruct the original epoch-zero fixed probe before HF restores saved training RNG.
        with torch.random.fork_rng(devices=[]):
            loader=trainer.get_train_dataloader()
            if hasattr(loader,'set_epoch'):loader.set_epoch(0)
            iterator=iter(loader)
            trainer.probe_batches=[]
            for _ in range(16):
                batch=next(iterator)
                trainer.probe_batches.append({k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in batch.items()})
            del iterator,loader
        probe_hashes=[]
        for batch in trainer.probe_batches:
            for ids,mask in zip(batch['input_ids'],batch['attention_mask']):
                values=ids[mask.bool()].tolist()
                probe_hashes.append(hashlib.sha256(json.dumps(values).encode()).hexdigest())
        (attempt/'fixed_probe_token_hashes.json').write_text(json.dumps(probe_hashes,indent=2))
        for name in ['termination_steps.jsonl','gradient_probes.jsonl']:
            archived=attempt/name
            if archived.exists():
                lines=[line for line in archived.read_text().splitlines() if line.strip() and json.loads(line)['step']<=saved['global_step']]
                diagnostic_path(directory/name).write_text('\n'.join(lines)+'\n')
        status('restoring_and_training',optimizer_scheduler_rng_restore_requested=True)
        started=time.time()
        result=trainer.train(resume_from_checkpoint=str(checkpoint))
        assert trainer.state.global_step==1125
        trainer.save_model();trainer.save_state();trainer.save_metrics('train_resume',result.metrics)
        torch.cuda.synchronize()
        report=dict(stage='train',arm='S1-M',global_step=1125,resumed_from=str(checkpoint),
                    resumed_from_step=saved['global_step'],resume_seconds=time.time()-started,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(),metrics=result.metrics,
                    automatic_evaluation=False,full_training_started=False)
        diagnostic_path(directory/'completed.json').write_text(json.dumps(report,indent=2))
        status('training_completed',global_step=1125)
    except Exception as exc:
        status('failed',reason=repr(exc))
        raise

if __name__=='__main__':main()