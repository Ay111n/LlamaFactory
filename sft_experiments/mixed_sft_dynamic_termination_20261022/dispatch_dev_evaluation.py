"""DEV-only dispatch, separate from resumed training and historical stopped jobs."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

from experiment_paths import diagnostic_path
from experiment_paths import (experiment_root,benchmark_root,model_root,require_path,paths,
                              script_path,evaluation_results_root,evaluation_analysis_root)
ROOT=experiment_root();RESULTS=evaluation_results_root();ANALYSIS=evaluation_analysis_root()
TRAIN=str(require_path('training_python'));INFER=str(require_path('inference_python'))
ENV=dict(os.environ,SFT_EXPERIMENT_ROOT=str(ROOT),PYTHONPATH=str(paths()['framework_root']/'src'),
         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
LOCK=threading.RLock();GPU_LOCKS={i:threading.Lock() for i in [1,6,7]}
STATE={'status':'running','full_training_started':False,'scope':'666 DEV only','arms':{}}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def update(arm,**values):
    with LOCK:
        STATE['arms'][arm].update(values)
        STATE['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp=ANALYSIS/'dev_dispatch_status.tmp'
        temp.write_text(json.dumps(STATE,indent=2));temp.replace(ANALYSIS/'dev_dispatch_status.json')
        print(json.dumps({'arm':arm,**values}),flush=True)

def wait_gpu(gpu,arm):
    while True:
        memory=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                    '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory<=64:return
        update(arm,stage='waiting_for_idle_gpu',gpu=gpu)
        time.sleep(20)

def execute(arm,gpu,stage,python,name,*args):
    wait_gpu(gpu,arm)
    folder=ANALYSIS/'logs' if stage in ['evaluate','preflight'] else ROOT
    folder.mkdir(parents=True,exist_ok=True)
    log=folder/f'dev_{stage}_{arm}.log'
    with log.open('a') as f:
        child=subprocess.Popen([python,str(script_path(name)),*args],
            env=dict(ENV,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT)
        update(arm,stage=stage,pid=child.pid,gpu=gpu,log=str(log))
        code=child.wait()
    if code:raise RuntimeError(f'{arm} {stage} failed (exit {code}); see {log}')

def verify_existing_export(arm):
    manifest=json.loads((ROOT/f'export_{arm}.json').read_text())
    assert manifest['verification_version']=='v2-preserve-nonpersistent-FP32-rotary-buffers'
    assert manifest['reload_max_logit_error']<=1e-3
    assert len(manifest['all_saved_tensors_bitwise_equal'])>300
    target=model_root()/'merged'/arm
    for name,expected in manifest['files'].items():
        assert sha(target/name)==expected,f'Export file changed: {arm}/{name}'
    record={'arm':arm,'files_verified':len(manifest['files']),
            'all_saved_tensors_bitwise_equal':len(manifest['all_saved_tensors_bitwise_equal']),
            'reload_max_logit_error':manifest['reload_max_logit_error'],
            'export_manifest_sha256':sha(ROOT/f'export_{arm}.json')}
    (ANALYSIS/f'export_integrity_{arm}.json').write_text(json.dumps(record,indent=2))

def arm_job(arm,gpu):
    try:
        completed=diagnostic_path(model_root()/'runs'/arm/'completed.json')
        while not completed.exists():
            if arm!='S1-M':raise FileNotFoundError(completed)
            resume=ROOT/'resume_status.json'
            if resume.exists() and json.loads(resume.read_text()).get('stage')=='failed':
                raise RuntimeError('S1-M training failed; evaluation not started')
            update(arm,stage='waiting_for_training',gpu=gpu)
            time.sleep(60)
        assert json.loads(completed.read_text())['global_step']==1125
        with GPU_LOCKS[gpu]:
            if not (ROOT/f'export_{arm}.json').exists():
                if not (model_root()/'merged'/arm).exists():
                    execute(arm,gpu,'export_v2',TRAIN,'export_arm_v2.py','--arm',arm)
                execute(arm,gpu,'verify_v2',TRAIN,'verify_export_v2.py','--arm',arm)
            update(arm,stage='checking_export_integrity')
            verify_existing_export(arm)
            assert not (RESULTS/f'evaluation_{arm}.json').exists(),'Evaluation already completed; refuse duplicate'
            execute(arm,gpu,'evaluate',INFER,'evaluate_arm.py','--arm',arm)
        index=json.loads((RESULTS/f'evaluation_{arm}.json').read_text())
        rows=[json.loads(l) for l in (Path(index['run_dir'])/'predictions.jsonl').open()]
        assert len(rows)==len({r['uid'] for r in rows})==666
        update(arm,stage='completed',status='completed',pid=None,run_dir=index['run_dir'],sample_count=666)
    except Exception as exc:
        update(arm,stage='failed',status='failed',reason=str(exc),pid=None)
        (ANALYSIS/f'dev_error_{arm}.txt').write_text(traceback.format_exc())
        raise

def main():
    RESULTS.mkdir(parents=True,exist_ok=True);ANALYSIS.mkdir(parents=True,exist_ok=True)
    for arm,gpu in [('S0',1),('S1',6),('S2',7),('S1-M',1)]:
        STATE['arms'][arm]={'gpu':gpu,'stage':'queued','status':'running'}
    bench=benchmark_root()
    files=[bench/'eval/scripts'/n for n in ['eval_multilingual_mcqa_vllm.py',
          'mcqa_answer_extraction_multilingual.py','multilingual_mcqa_data_registry.json']]
    files += [bench/'eval/tibetan_formateval_validation/trilingual_aligned/tibetan/mcqa_4way.jsonl',
              script_path('evaluate_arm.py'),RESULTS/'historical_b0/run_manifest.json',RESULTS/'historical_b0/predictions.jsonl']
    campaign={'authorized_by':'User requested DEV evaluation of completed models with identical historical settings on 2026-10-23',
              'historical_reference':'B0 checkpoint-6000 (different training budget)',
              'frozen_comparison_files':{str(p):sha(p) for p in files},
              'arms':list(STATE['arms']),'scope':'666 Tibetan DEV MCQA4, English instructions, Thinking ON',
              'automatic_full_training':False,'dispatcher_sha256':sha(Path(__file__))}
    (RESULTS/'evaluation_campaign.json').write_text(json.dumps(campaign,indent=2))
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(arm_job,arm,gpu) for arm,gpu in [('S0',1),('S1',6),('S2',7),('S1-M',1)]]
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as exc:errors.append(str(exc))
    if errors:raise RuntimeError(str(errors))
    with (ANALYSIS/'analysis_after_resume.log').open('a') as log:
        subprocess.run([TRAIN,str(script_path('analyze_results.py'))],env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    STATE.update(status='completed',stage='all_four_dev_complete')
    (ANALYSIS/'dev_dispatch_status.json').write_text(json.dumps(STATE,indent=2))

if __name__=='__main__':main()