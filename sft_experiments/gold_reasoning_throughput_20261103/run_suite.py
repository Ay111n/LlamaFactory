"""Run bounded benchmarks sequentially on the two authorized GPUs only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
PYTHON='/opt/anaconda3/envs/training/bin/python'
VISIBLE='GPU-5fc7820d-0470-f42a-a5c7-3f3cbfb2df88,GPU-c98046c2-90b5-a959-9e88-f09a146ee29e'

def write(name,obj):
    tmp=ROOT/(name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(ROOT/name)

def supervise():
    raw=subprocess.check_output(['nvidia-smi','--id=1,2','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    inventory=[line.split(',') for line in raw.splitlines()]
    assert {row[0].strip() for row in inventory}==set(VISIBLE.split(','))
    assert all(int(row[1])<64 for row in inventory), inventory
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=VISIBLE,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_DATASETS_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_PROGRESS_BARS='1',FLASH_ATTENTION_DETERMINISTIC='0')
    jobs=[('equivalence',['equivalence','--run-id','equivalence'])]
    jobs += [(v,['train','--variant',v,'--run-id',v,'--steps','12']) for v in ['baseline','cached','pipeline']]
    for name,arguments in jobs:
        with (ROOT/f'{name}.log').open('x') as log:
            p=subprocess.Popen([PYTHON,'-u',str(ROOT/'benchmark.py')]+arguments,cwd=ROOT.parent.parent,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state={'state':'running','stage':name,'worker_pid':p.pid,'supervisor_pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),'full_training':False}
            write('suite_status.json',state)
            rc=p.wait()
        state.update(returncode=rc,finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(f'{name}_process.json',state)
        if rc:
            state['state']='failed'
            write('suite_status.json',state)
            return rc
    write('suite_status.json',{'state':'completed','full_training':False,'finished_at_utc':datetime.now(timezone.utc).isoformat()})
    return 0

if __name__=='__main__':
    if len(sys.argv)>1:
        sys.exit(supervise())
    assert not (ROOT/'suite_supervisor.log').exists()
    with (ROOT/'suite_supervisor.log').open('x') as log:
        p=subprocess.Popen([PYTHON,'-u',str(ROOT/'run_suite.py'),'supervise'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write('suite_launch.json',{'pid':p.pid,'full_training':False})
    print(json.dumps({'supervisor_pid':p.pid,'bounded_benchmark':True}))
