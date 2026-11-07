"""Use all seven idle L20s: retain initial job, five model workers, GPU buffer pool."""
import pathlib,os,subprocess,json,datetime,time,traceback,fcntl,hashlib
HERE=pathlib.Path(__file__).resolve().parent;EXP=HERE.parents[1]
PY='/opt/anaconda3/envs/training/bin/python'
ENV=dict(os.environ,SFT_EXPERIMENT_ROOT=str(EXP),SFT_EXPERIMENT_PATHS=str(EXP/'paths.json'),PYTHONPATH='/home/nlp/data/lyl/LLaMA-Factory/src',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
ANALYSIS=pathlib.Path('/home/nlp/data/lyl/benchmark/eval/analysis/mixed_sft_gold_margin_gradient_attribution_20261025')
STATE={'status':'running','optimizer_updates':0,'generation_calls':0,'DEV_or_TEST_used':False,'parallelism':'existing initial worker + five final model workers + shared GPU FP32 buffer pool','jobs':{}}

def status(phase,**extra):
 STATE.update(phase=phase,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
 p=HERE/'status.parallel.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(HERE/'status.json')
def alive(pid):
 p=pathlib.Path('/proc')/str(pid)/'stat'
 if not p.exists():return False
 try:return p.read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def main():
 lock=(HERE/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 processes={}
 try:
  initial_pid=int((HERE/'retained_initial.pid').read_text())
  assert alive(initial_pid) or (HERE/'completion/initial_shared.json').exists()
  assert '\nOK\n' in (HERE/'unit_parallel.log').read_text()
  assert json.loads((HERE/'gpu_buffer_test.json').read_text())['exact_FP32_equal']
  original=json.loads((HERE/'protocol_manifest.json').read_text())
  original['retained_initial_probe_source_sha256']=original['probe_source_sha256']
  original.setdefault('previous_final_probe_source_sha256',[]).append(original.get('final_probe_source_sha256'))
  original['final_probe_source_sha256']=hashlib.sha256((HERE/'component_probe.py').read_bytes()).hexdigest()
  original['execution_revision']='Direct checkpoint loading, no tensor-by-tensor checks; exact FP32 GPU buffer accumulation; existing initial CPU worker retained.'
  (HERE/'protocol_manifest.json').write_text(json.dumps(original,indent=2))
  shared=pathlib.Path('/dev/shm/gold_margin_gradient_attribution_20261025')
  if shared.exists():assert not list(shared.iterdir())
  else:shared.mkdir()
  STATE['jobs']['initial']={'pid':initial_pid,'status':'running','gpus':[0]}
  cards={'G1':1,'G2':2,'G3':3,'G1-M':5,'G3-M':6}
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
  available={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
  assert all(available[g]<=64 for g in [1,2,3,5,6,7]),available
  for arm,gpu in cards.items():
   env=dict(ENV,CUDA_VISIBLE_DEVICES=f'{gpu},7',ATTR_GPU_BUFFERS='1',ATTR_CROSS_DIR=str(shared),ATTR_CPU_KEYS='dynamic_gold_a,plain_stop' if arm=='G3-M' else '')
   with (HERE/(arm+'.log')).open('a') as log:
    p=subprocess.Popen([PY,'-u',str(HERE/'component_probe.py'),arm],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   processes[arm]=p;STATE['jobs'][arm]={'pid':p.pid,'status':'running','model_gpu':gpu,'buffer_gpu':7}
   status('parallel_read_only_vjps')
   # Avoid five simultaneous CPU model/adapter loading peaks.
   time.sleep(12)
  while True:
   initial_done=(HERE/'completion/initial_shared.json').exists()
   if initial_done:STATE['jobs']['initial']['status']='completed'
   elif not alive(initial_pid):raise RuntimeError('Retained initial job exited without completion')
   for arm,p in processes.items():
    code=p.poll()
    if code is not None:
     if code:raise RuntimeError(f'{arm} exited {code}; see {arm}.log')
     assert (HERE/'completion'/('final_'+arm+'.json')).exists()
     STATE['jobs'][arm]['status']='completed'
   status('parallel_read_only_vjps')
   if all(j['status']=='completed' for j in STATE['jobs'].values()):break
   time.sleep(20)
  assert not list(shared.iterdir());shared.rmdir()
  assert hashlib.sha256((HERE/'component_probe.py').read_bytes()).hexdigest()==original['final_probe_source_sha256']
  status('summarize_and_plot')
  with (HERE/'summary.log').open('w') as log:
   subprocess.run([PY,str(ANALYSIS/'summarize_attribution.py'),'--diagnostic-root',str(HERE),'--output',str(ANALYSIS)],env=dict(ENV,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
  status('completed',status='completed')
 except Exception as exc:
  status('halted',status='failed',reason=str(exc));(HERE/'parallel_error.txt').write_text(traceback.format_exc())
  # Stop only this controller's own read-only children; preserve outputs and the initial worker.
  for p in processes.values():
   if p.poll() is None:p.terminate()
  raise

if __name__=='__main__':main()
