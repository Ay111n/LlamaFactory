"""Adopt live GPU 5 baseline; run KD on GPU 2, then exclusive GPU 5 DEV."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import experiment as e
import pipeline as serial


def alive(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return False
    return (p/'stat').read_text().split()[2]!='Z'


def main():
    handoff=e.read(e.RUNTIME/'parallel_handoff.json')
    baseline_pid=handoff['baseline_pid']
    assert not alive(handoff['previous_supervisor_pid'])
    assert alive(baseline_pid)
    assert 'experiment.py' in (Path('/proc')/str(baseline_pid)/'cmdline').read_bytes().decode()
    assert e.read(e.RUNTIME/'gpu2_probe/smoke.json')['status']=='passed'
    assert not (e.MODELS/'kd').exists()
    e.verify()
    locks=[]
    for name in ['.crossref_teacher_gpu_5.lock','.student_crossref_gpu_2.lock']:
        f=(e.ROOT/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    raw=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert int(raw.strip())>=24576,('GPU2 no longer has validated launch headroom',raw)
    serial.STATE.update(status='starting_parallel',gpu=None,student_gpus={'baseline':5,'kd':2},
        steps=handoff['previous_state']['steps'],adopted_baseline_pid=baseline_pid,
        resource_policy='GPU2 KD after longest-sequence probe; allocator capped at 23 GiB; GPU5 DEV stays exclusive')
    cmd=[serial.TRAIN,str(e.HERE/'limited_kd.py')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',TMPDIR=str(e.ROOT/'tmp_q35'),PYTHONUNBUFFERED='1',
        PYTHONPATH=str(e.FRAMEWORK/'src'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (e.RUNTIME/'train_kd.log').open('w') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=e.FRAMEWORK)
        serial.state('parallel_training',student_pids={'baseline':baseline_pid,'kd':child.pid})
        init_checked=False
        while True:
            complete={a:(e.RUNTIME/(a+'_complete.json')).exists() for a in ['baseline','kd']}
            if (e.RUNTIME/'kd_initialization.json').exists() and not init_checked:
                lhs=e.read(e.RUNTIME/'baseline_initialization.json')['trainable_sha256']
                rhs=e.read(e.RUNTIME/'kd_initialization.json')['trainable_sha256']
                assert lhs==rhs,'Initial LoRA weights differ'
                init_checked=True
            for arm,pid in [('baseline',baseline_pid),('kd',child.pid)]:
                if complete[arm]:
                    done=e.read(e.RUNTIME/(arm+'_complete.json'));assert done['status']=='completed' and done['steps']==250
                elif not alive(pid):raise RuntimeError((arm,'exited without completion',pid))
            serial.state('parallel_training',completed_arms=[a for a in complete if complete[a]],same_initialization_verified=init_checked)
            if all(complete.values()):break
            time.sleep(15)
        code=child.wait();assert code==0
    assert init_checked
    serial.STATE['steps'].append({'stage':'parallel_training','completed':True,'student_gpus':{'baseline':5,'kd':2}})
    e.save(e.RUNTIME/'training_complete.json',{'status':'completed','same_initialization':True,'student_gpus':{'baseline':5,'kd':2},
        'arms':{a:e.read(e.RUNTIME/(a+'_complete.json')) for a in ['baseline','kd']}})
    # GPU 2 is no longer used; inference remains on the original exclusive GPU 5 protocol.
    fcntl.flock(locks[1],fcntl.LOCK_UN);locks[1].close()
    for arm in ['baseline','kd']:
        serial.execute('merge_'+arm,[serial.INFER,str(serial.MERGE),'--base-model',str(e.BASE),'--adapter',str(e.MODELS/arm),
            '--output',str(e.MODELS/'merged'/arm),'--label',e.RUN+'_'+arm])
        serial.execute('dev_'+arm,[serial.INFER,str(serial.EVAL),'evaluate','--arm',arm],True)
    serial.execute('compare',[serial.INFER,str(serial.EVAL),'compare'])
    serial.state('completed');e.save(e.RUNTIME/'pipeline_complete.json',serial.STATE)


if __name__=='__main__':
    try:main()
    except BaseException:
        serial.state('failed',traceback=traceback.format_exc());raise
