"""Sequential GPU 5 cache, verified student training, merge and matched DEV."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import experiment as e

TRAIN='/opt/anaconda3/envs/training/bin/python'
INFER='/opt/anaconda3/envs/inference/bin/python'
EVAL=e.ROOT/'benchmark/eval/scripts'/e.RUN/'evaluate.py'
MERGE=e.ROOT/'finetuning/scripts/model/ce_g2_thinking_dev_32768_20261031/merge_lora_adapter.py'
STATE={'status':'starting','pid':os.getpid(),'gpu':5,'started':time.time(),'steps':[]}

def state(stage,**kw):
 STATE.update(status=stage,updated=time.time(),**kw);e.save(e.RUNTIME/'pipeline_status.json',STATE)

def idle():
 raw=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 mem,util=map(int,raw.strip().split(','));return mem<100 and util==0

def wait_idle(stage):
 state('waiting_for_gpu_5',next_stage=stage)
 while not idle():time.sleep(15)

def execute(stage,cmd,gpu=False):
 if gpu:wait_idle(stage)
 state(stage)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='5' if gpu else '',TMPDIR=str(e.ROOT/'tmp_q35'),
          PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(e.FRAMEWORK/'src'),
          OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 logfile=e.RUNTIME/(stage+'.log')
 with logfile.open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,cwd=e.FRAMEWORK)
  state(stage,child_pid=p.pid,command=cmd)
  code=p.wait()
 STATE['steps'].append({'stage':stage,'returncode':code,'log':str(logfile),'finished':time.time()})
 state(stage,child_pid=None)
 assert code==0,(stage,code)

def main():
 e.verify()
 lock=(e.ROOT/'.crossref_teacher_gpu_5.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (e.RUNTIME/'pipeline_complete.json').exists()
 assert all(not (e.MODELS/a).exists() for a in ['baseline','kd'])
 execute('unit_checks',[TRAIN,str(e.HERE/'test_losses.py')])
 execute('teacher_cache',[INFER,str(e.HERE/'experiment.py'),'cache'],True)
 execute('smoke',[TRAIN,str(e.HERE/'experiment.py'),'smoke','--arm','kd'],True)
 assert e.read(e.RUNTIME/'smoke.json')['status']=='passed'
 for arm in ['baseline','kd']:
  execute('train_'+arm,[TRAIN,str(e.HERE/'experiment.py'),'train','--arm',arm],True)
  assert e.read(e.RUNTIME/(arm+'_complete.json'))['status']=='completed'
 init=[e.read(e.RUNTIME/(a+'_initialization.json'))['trainable_sha256'] for a in ['baseline','kd']]
 assert init[0]==init[1]
 e.save(e.RUNTIME/'training_complete.json',{'status':'completed','same_initialization':True,'arms':{a:e.read(e.RUNTIME/(a+'_complete.json')) for a in ['baseline','kd']}})
 for arm in ['baseline','kd']:
  execute('merge_'+arm,[INFER,str(MERGE),'--base-model',str(e.BASE),'--adapter',str(e.MODELS/arm),
       '--output',str(e.MODELS/'merged'/arm),'--label',e.RUN+'_'+arm])
  execute('dev_'+arm,[INFER,str(EVAL),'evaluate','--arm',arm],True)
 execute('compare',[INFER,str(EVAL),'compare'])
 state('completed');e.save(e.RUNTIME/'pipeline_complete.json',STATE)

if __name__=='__main__':
 try:main()
 except BaseException:
  state('failed',traceback=traceback.format_exc());raise
