"""Continue DEV evaluation after strict export-check recovery; retain training."""
from experiment_paths import diagnostic_path
from experiment_paths import experiment_root,benchmark_root,data_root,model_root,paths,require_path,script_path,stage_log,evaluation_analysis_root

from experiment_paths import evaluation_analysis_root
import argparse
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

ROOT=Path(__file__).resolve().parent
TRAIN=str(require_path('training_python'));INFER=str(require_path('inference_python'))
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join([str(paths()['framework_root']/'src'),str(ROOT)]),HF_HUB_OFFLINE='1',
         TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
LOCK=threading.RLock();EXPORT=threading.Lock()
STATE={'stage':'export_verification_recovery','status':'running','full_training_started':False,'arms':{}}

def write():
    with LOCK:
        STATE['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp=ROOT/'recovery_status.tmp';tmp.write_text(json.dumps(STATE,indent=2));tmp.replace(ROOT/'pipeline_status.json')
        (ROOT/'recovery_status.json').write_text(json.dumps(STATE,indent=2))

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def stage(arm,gpu,name,python,script,*args):
    while True:
        memory=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory<=64:break
        with LOCK:STATE['arms'][arm].update(stage='waiting_for_idle_gpu',status='running');write()
        time.sleep(15)
    with LOCK:STATE['arms'][arm].update(stage=name,status='running');write()
    with stage_log(name,arm).open('a') as log:
        process=subprocess.Popen([python,str(script_path(script)),*args],env=dict(ENV,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        with LOCK:STATE['arms'][arm]['pid']=process.pid;write()
        result=process.wait()
    if result:raise RuntimeError(f'{arm} {name} failed: {result}')

def arm_job(arm,gpu,adopt_verifier=None,training_pid=None):
    try:
        if training_pid:
            with LOCK:STATE['arms'][arm].update(stage='train',pid=training_pid);write()
            while alive(training_pid):time.sleep(15)
        assert json.loads((diagnostic_path(model_root()/'runs'/arm/'completed.json')).read_text())['global_step']==1125
        if adopt_verifier:
            # Main thread acquires this lock before workers start.
            try:
                while alive(adopt_verifier):time.sleep(5)
                assert (ROOT/f'export_{arm}.json').exists(),'Strict verification did not pass'
            finally:EXPORT.release()
        else:
            with LOCK:STATE['arms'][arm].update(stage='waiting_for_verification',pid=None);write()
            with EXPORT:
                if not (model_root()/'merged'/arm).exists():
                    stage(arm,gpu,'export_v2',TRAIN,'export_arm_v2.py','--arm',arm)
                stage(arm,gpu,'verify_v2',TRAIN,'verify_export_v2.py','--arm',arm)
        stage(arm,gpu,'evaluate',INFER,'evaluate_arm.py','--arm',arm)
        with LOCK:STATE['arms'][arm].update(stage='completed',status='completed',pid=None);write()
    except Exception as exc:
        with LOCK:STATE['arms'][arm].update(status='failed',reason=str(exc));write()
        (ROOT/f'recovery_error_{arm}.txt').write_text(traceback.format_exc())
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--s0-verifier-pid',required=True,type=int)
    p.add_argument('--s1m-training-pid',required=True,type=int);a=p.parse_args()
    for arm,gpu in [('S0',0),('S1',1),('S2',7),('S1-M',6)]:
        STATE['arms'][arm]={'gpu':gpu,'stage':'waiting','status':'running'}
    STATE['arms']['S0'].update(stage='verify_v2',pid=a.s0_verifier_pid)
    write()
    files=['verify_export_v2.py','export_arm_v2.py','recover_eval_pipeline.py']
    (ROOT/'export_recovery_manifest.json').write_text(json.dumps({'reason':'Preserve nonpersistent FP32 rotary buffers; original frozen files unchanged',
        'code_sha256':{name:hashlib.sha256(script_path(name).read_bytes()).hexdigest() for name in files}},indent=2))
    EXPORT.acquire()
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(arm_job,'S0',0,a.s0_verifier_pid),pool.submit(arm_job,'S1',1),
               pool.submit(arm_job,'S2',7),pool.submit(arm_job,'S1-M',6,None,a.s1m_training_pid)]
        for future in concurrent.futures.as_completed(tasks):
            try:future.result()
            except Exception as exc:errors.append(str(exc))
    if errors:raise RuntimeError(str(errors))
    evaluation_analysis_root().mkdir(parents=True,exist_ok=True)
    with (evaluation_analysis_root()/'analysis.log').open('a') as log:
        subprocess.run([TRAIN,str(script_path('analyze_results.py'))],env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    STATE.update(stage='small_scale_complete',status='completed');write()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        STATE.update(stage='halted',status='failed',reason=str(exc));write()
        raise
