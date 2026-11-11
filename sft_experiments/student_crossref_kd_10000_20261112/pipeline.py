"""Prepare, longest-sequence smoke, and launch both bounded full-data students."""
import os,sys,time,json,subprocess,fcntl,traceback
from pathlib import Path
from full import e,STEPS,PILOT,HERE
PY='/opt/anaconda3/envs/training/bin/python'
GPUS={'baseline':0,'kd':3}
STATE={'pid':os.getpid(),'started':time.time(),'student_gpus':GPUS,'steps':[]}
def state(stage,**kw):
 STATE.update(status=stage,updated=time.time(),**kw);e.save(e.RUNTIME/'pipeline_status.json',STATE)
def env(gpu):
 return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',TMPDIR=str(e.ROOT/'tmp_q35'),PYTHONUNBUFFERED='1',PYTHONPATH=str(e.FRAMEWORK/'src'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def wait_room(gpu,stage):
 while True:
  free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
  if free>=28672:return
  state('waiting_for_memory',next_stage=stage,gpu=gpu,free_mib=free);time.sleep(20)
def execute(stage,command,gpu=None):
 if gpu is not None:wait_room(gpu,stage)
 with (e.RUNTIME/(stage+'.log')).open('w') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env(gpu),cwd=e.FRAMEWORK)
  state(stage,child_pid=p.pid);code=p.wait()
 assert code==0,(stage,code)
 STATE['steps'].append({'stage':stage,'returncode':code,'finished':time.time()})
def main():
 e.RUNTIME.mkdir(exist_ok=True)
 lock=(e.RUNTIME/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not any((e.MODELS/a).exists() for a in GPUS)
 if not (e.RUNTIME/'preflight.json').exists():execute('prepare',[PY,str(HERE/'full.py'),'prepare'])
 e.verify()
 execute('smoke',[PY,str(HERE/'full.py'),'smoke','--arm','kd'],3)
 assert e.read(e.RUNTIME/'smoke.json')['status']=='passed'
 children={};logs={};locks=[]
 for arm,gpu in GPUS.items():
  f=(e.ROOT/('.student_full_gpu_'+str(gpu)+'.lock')).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  wait_room(gpu,'train_'+arm)
  logs[arm]=(e.RUNTIME/('train_'+arm+'.log')).open('w')
  children[arm]=subprocess.Popen([PY,str(HERE/'full.py'),'train','--arm',arm],stdout=logs[arm],stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env(gpu),cwd=e.FRAMEWORK)
 state('parallel_training',student_pids={a:p.pid for a,p in children.items()},completed_arms=[])
 verified=False
 while True:
  if not verified and all((e.RUNTIME/(a+'_initialization.json')).exists() for a in children):
   hashes=[e.read(e.RUNTIME/(a+'_initialization.json'))['trainable_sha256'] for a in children];assert len(set(hashes))==1
   e.save(e.RUNTIME/'same_initialization.json',{'passed':True,'trainable_sha256':hashes[0]});verified=True
  done=[]
  for arm,p in children.items():
   complete=e.RUNTIME/(arm+'_complete.json')
   if complete.exists():
    m=e.read(complete);assert m['status']=='completed' and m['steps']==STEPS;done.append(arm)
   elif p.poll() is not None:raise RuntimeError((arm,'exited without completion',p.returncode))
  state('parallel_training',same_initialization_verified=verified,completed_arms=done)
  if len(done)==2:break
  time.sleep(20)
 for p in children.values():assert p.wait()==0
 assert verified
 lhs=e.rows(e.RUNTIME/'baseline_steps.jsonl');rhs=e.rows(e.RUNTIME/'kd_steps.jsonl')
 assert len(lhs)==len(rhs)==STEPS and all(a['order']==b['order'] for a,b in zip(lhs,rhs))
 assert sum(len(list((e.MODELS/a).glob('checkpoint-*'))) for a in children)<=2
 state('training_completed',same_training_order_verified=True)
 e.save(e.RUNTIME/'training_complete.json',{'status':'completed','same_initialization':True,'same_training_order':True,'steps':STEPS,'arms':{a:e.read(e.RUNTIME/(a+'_complete.json')) for a in children}})
if __name__=='__main__':
 try:main()
 except BaseException:
  state('failed',traceback=traceback.format_exc());raise
