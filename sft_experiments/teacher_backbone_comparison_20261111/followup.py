"""Wait for authorized teacher arms, merge, and generate the matched 40 diagnostics."""
import argparse,fcntl,json,os,subprocess,time,traceback,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
RUNTIME=HERE/'runtime'
PYTHON='/opt/anaconda3/envs/inference/bin/python'
MERGE=PROJECT/'finetuning/scripts/model/teacher_backbone_comparison_20261111/merge_teacher.py'
EVAL=PROJECT/'benchmark/eval/scripts/teacher_backbone_comparison_20261111/diagnose.py'
ARMS={'sft_new2':('crossref_teacher_9960_20261111',5),'qwen35':('qwen35_crossref_teacher_9960_20261111',0)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def state(arm,stage,**kw):write(RUNTIME/(arm+'_status.json'),dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
def call(cmd,log,env):
 with log.open('wb') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=PROJECT,check=True)
def worker(arm):
 run,gpu=ARMS[arm];train=PROJECT/'LLaMA-Factory/sft_experiments'/run/'runtime'
 state(arm,'waiting_for_training');deadline=time.time()+72*3600
 while not (train/'completion.json').exists():
  if time.time()>deadline:raise TimeoutError(run)
  time.sleep(30)
 complete=read(train/'completion.json');assert complete['status']=='completed',complete
 for path,h in read(RUNTIME/'launch.json')['asset_hashes'].items():assert sha(Path(path))==h,path
 adapter=PROJECT/'finetuning/model/sft'/run/'crossref';merged=PROJECT/'finetuning/model/sft'/run/'merged/crossref'
 base=read(train/'preflight.json')['model']
 lock=(RUNTIME/'cpu_merge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 state(arm,'merging')
 call([PYTHON,str(MERGE),'--base-model',base,'--adapter',str(adapter),'--output',str(merged),'--label',arm],
      RUNTIME/(arm+'_merge.log'),dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1'))
 lock.close();state(arm,'waiting_for_gpu',gpu=gpu)
 while True:
  raw=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  mem,util=map(int,raw.strip().split(','))
  if mem<100 and util==0:break
  if time.time()>deadline:raise TimeoutError('GPU unavailable')
  time.sleep(30)
 state(arm,'diagnosing',gpu=gpu)
 call([PYTHON,str(EVAL),'generate','--arm',arm,'--model',str(merged)],RUNTIME/(arm+'_diagnose.log'),
      dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1'))
 state(arm,'generation_completed',semantic_review_pending=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS));a=p.parse_args();RUNTIME.mkdir(exist_ok=True)
 if a.arm:
  try:worker(a.arm)
  except BaseException:state(a.arm,'failed',traceback=traceback.format_exc());raise
  return
 assert not (RUNTIME/'launch.json').exists()
 children=[]
 for arm in ARMS:
  log=(RUNTIME/(arm+'_worker.log')).open('wb')
  child=subprocess.Popen([PYTHON,str(Path(__file__)),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  children.append((arm,child,log))
 write(RUNTIME/'launch.json',dict(pid=os.getpid(),workers={a:p.pid for a,p,_ in children},asset_hashes={str(p):sha(p) for p in [MERGE,EVAL,Path(__file__)]}))
 result={a:p.wait() for a,p,_ in children}
 for _,_,log in children:log.close()
 if all(rc==0 for rc in result.values()):
  call([PYTHON,str(EVAL),'compare'],RUNTIME/'comparison.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  write(RUNTIME/'completion.json',dict(status='generation_complete_review_pending',returncodes=result))
 else:write(RUNTIME/'completion.json',dict(status='failed',returncodes=result))
if __name__=='__main__':main()
