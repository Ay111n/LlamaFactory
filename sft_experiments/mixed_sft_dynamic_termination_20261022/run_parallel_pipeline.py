"""Resource-aware parallel execution of independent single-GPU experiments.

Adopts the existing smoke process; never changes per-arm batch or DDP settings.
Only completely idle L20 GPUs are claimed. Model exports are serialized to
bound host RAM usage. No full-data training entrypoint exists.
"""
from experiment_paths import diagnostic_path
from experiment_paths import experiment_root,benchmark_root,data_root,model_root,paths,require_path,script_path,stage_log,evaluation_analysis_root

from experiment_paths import evaluation_analysis_root
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parent
TRAIN=str(require_path('training_python'))
INFER=str(require_path('inference_python'))
BASE_ENV=dict(os.environ,PYTHONPATH=os.pathsep.join([str(paths()['framework_root']/'src'),str(ROOT)]),HF_HUB_OFFLINE='1',
              TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
LOCK=threading.RLock()
EXPORT_LOCK=threading.Lock()
STATE={'stage':'waiting_for_existing_smoke','status':'running','arms':{},'full_training_started':False,
       'scheduling':'independent single-GPU arms; unchanged batch/accumulation/steps'}


def write_status():
    with LOCK:
        STATE['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp=ROOT/'pipeline_status.tmp'
        tmp.write_text(json.dumps(STATE,indent=2))
        tmp.replace(ROOT/'pipeline_status.json')


def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                  '--format=csv,noheader'],text=True)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    active={line.split(',')[0].strip() for line in busy.splitlines() if ',' in line}
    uuid_by_index={int(line.split(',')[0]):line.split(',')[1].strip() for line in uuids.splitlines()}
    result=[]
    for line in raw.splitlines():
        index,name,memory,util=[x.strip() for x in line.split(',')]
        index=int(index)
        result.append({'index':index,'name':name,'memory_mib':int(memory),'utilization':int(util),
                       'idle':name=='NVIDIA L20' and int(memory)<=64 and int(util)<=5
                              and uuid_by_index[index] not in active})
    return result


def execute(arm,gpu,stage,python,script,*args):
    with LOCK:
        STATE['arms'][arm].update(stage=stage,status='running')
        write_status()
    env=dict(BASE_ENV,CUDA_VISIBLE_DEVICES=str(gpu))
    with stage_log(stage,arm).open('a') as log:
        process=subprocess.Popen([python,str(script_path(script)),*args],env=env,stdout=log,stderr=subprocess.STDOUT)
        with LOCK:
            STATE['arms'][arm]['pid']=process.pid
            write_status()
        result=process.wait()
    if result:raise RuntimeError(f'{arm} {stage} failed ({result}), see {stage}_{arm}.log')


def run_arm(arm,gpu,adopt_pid=None):
    try:
        if adopt_pid is None:
            execute(arm,gpu,'train',TRAIN,'run_experiment.py','train','--arm',arm)
        else:
            while alive(adopt_pid):time.sleep(15)
            completion=json.loads((diagnostic_path(model_root()/'runs'/arm/'completed.json')).read_text())
            assert completion['global_step']==1125
        with LOCK:
            STATE['arms'][arm].update(stage='waiting_for_export',status='running')
            write_status()
        with EXPORT_LOCK:
            # CPU export may temporarily hold several weight copies.
            while True:
                values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith('MemAvailable:')}
                if values['MemAvailable']>=40*1024*1024:break
                time.sleep(15)
            execute(arm,gpu,'export',TRAIN,'export_arm.py','--arm',arm)
        execute(arm,gpu,'evaluate',INFER,'evaluate_arm.py','--arm',arm)
        with LOCK:
            STATE['arms'][arm].update(stage='completed',status='completed',pid=None)
            write_status()
    except Exception as exc:
        with LOCK:
            STATE['arms'][arm].update(status='failed',reason=str(exc),pid=None)
            write_status()
        (ROOT/f'parallel_error_{arm}.txt').write_text(traceback.format_exc())
        raise


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    try:return p.read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--adopt-smoke-pid',type=int,required=True)
    p.add_argument('--adopt-arm',choices=['S0','S1','S2','S1-M'])
    p.add_argument('--adopt-pid',type=int);p.add_argument('--adopt-gpu',type=int)
    args=p.parse_args()
    if args.adopt_arm:
        assert args.adopt_pid is not None and args.adopt_gpu is not None
        cmd=Path(f'/proc/{args.adopt_pid}/cmdline').read_bytes().split(b'\0')
        assert cmd[1].endswith(b'/run_experiment.py') and b'train' in cmd and args.adopt_arm.encode() in cmd
    write_status()
    while alive(args.adopt_smoke_pid):time.sleep(5)
    smoke=json.loads((diagnostic_path(model_root()/'smoke/S1-M/completed.json')).read_text())
    assert smoke['global_step']==1
    steps=[json.loads(x) for x in (diagnostic_path(model_root()/'smoke/S1-M/termination_steps.jsonl')).read_text().splitlines()]
    assert math.isfinite(steps[-1]['grad_norm_preclip']) and steps[-1]['match_relative_error']<2e-5
    assert 'INTEGRATION_PASS' in (ROOT/'integration_tests.log').read_text()
    assert '\nOK\n' in (ROOT/'unit_tests_training.log').read_text()
    assert json.loads((data_root()/'manual_review_results.json').read_text())['count']==32
    (ROOT/'validation_gates.json').write_text(json.dumps(dict(unit_tests=True,human_review_32=True,
        integration_tests=True,real_smoke=True,calibration=True),indent=2))
    manifest={'authorized_change':'Use idle GPUs for concurrent independent single-GPU arms',
              'original_input_fingerprint':'input_code_environment_fingerprints.json',
              'scheduler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'initial_inventory':gpu_inventory(),'same_gpu_model':'NVIDIA L20',
              'per_arm_world_size':1,'per_device_batch_size':1,'gradient_accumulation_steps':16,
              'optimizer_steps':1125,'priority':['S1-M','S0','S1','S2'],
              'adopted_arm':args.adopt_arm,'adopted_pid':args.adopt_pid,'adopted_gpu':args.adopt_gpu}
    (ROOT/'parallel_execution_manifest.json').write_text(json.dumps(manifest,indent=2))
    pending=['S1-M','S0','S1','S2'];running={};failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        if args.adopt_arm:
            pending.remove(args.adopt_arm)
            STATE['arms'][args.adopt_arm]={'gpu':args.adopt_gpu,'stage':'train','status':'running',
                'pid':args.adopt_pid,'adopted_existing_process':True}
            write_status()
            future=pool.submit(run_arm,args.adopt_arm,args.adopt_gpu,args.adopt_pid)
            running[future]=(args.adopt_arm,args.adopt_gpu)
        while pending or running:
            for future,(arm,gpu) in list(running.items()):
                if future.done():
                    try:future.result()
                    except Exception as exc:failed.append({'arm':arm,'error':str(exc)})
                    del running[future]
            if failed:
                # Keep independent in-flight work, but don't launch further jobs after a failure.
                pending=[]
            reserved={gpu for arm,gpu in running.values()}
            inventory=gpu_inventory()
            available=[x['index'] for x in inventory if x['idle'] and x['index'] not in reserved]
            while available and pending:
                gpu=available.pop(0);arm=pending.pop(0)
                with LOCK:
                    STATE['arms'][arm]={'gpu':gpu,'stage':'starting','status':'running'}
                    write_status()
                future=pool.submit(run_arm,arm,gpu)
                running[future]=(arm,gpu)
            with LOCK:
                STATE.update(stage='parallel_small_scale',pending=list(pending),gpu_inventory=inventory)
                write_status()
            if pending or running:time.sleep(15)
    if failed:raise RuntimeError(f'Parallel stage failures: {failed}')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==manifest['scheduler_sha256']
    evaluation_analysis_root().mkdir(parents=True,exist_ok=True)
    with (evaluation_analysis_root()/'analysis.log').open('a') as log:
        subprocess.run([TRAIN,str(script_path('analyze_results.py'))],env=BASE_ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    STATE.update(stage='small_scale_complete',status='completed')
    write_status()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        STATE.update(stage='halted',status='failed',reason=str(exc))
        write_status()
        (ROOT/'parallel_pipeline_error.txt').write_text(traceback.format_exc())
        raise
