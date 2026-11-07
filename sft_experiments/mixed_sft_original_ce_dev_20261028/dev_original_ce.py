"""Upload-ready original CE adapters: merge, six DEV evaluations and DEV selection. No training."""
import concurrent.futures,datetime,hashlib,json,os,shutil,subprocess,threading,time,traceback
from pathlib import Path
from experiment_paths import *
STEPS=[1500,3000,4500,6000,7500,9000]
STATE={'status':'running','stage':'waiting_for_upload_and_low_occupancy_GPUs','training':False,'TEST_used':False,'checkpoints':{str(x):{'stage':'waiting_for_transfer'} for x in STEPS}}
LOCK=threading.RLock()
ENV=dict(os.environ,SFT_EXPERIMENT_ROOT=str(ROOT),SFT_EXPERIMENT_PATHS=str(ROOT/'paths.json'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
EXPORTER=ROOT.parent.parent.parent/'finetuning/scripts/model/mixed_sft_original_ce_dev_20261028/export_original_ce.py'
assert EXPORTER.is_file(),EXPORTER
TRAIN=str(require_path('training_python'));INFER=str(require_path('inference_python'))
def save():
 with LOCK:
  STATE['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  p=ROOT/'status.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'status.json')
_previous_candidates=set()
def idle():
 global _previous_candidates
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 candidates=[]
 for line in raw.splitlines():
  i,u,n,total,used,util=[x.strip() for x in line.split(',')]
  if n=='NVIDIA L20' and int(used)<=4096 and int(total)-int(used)>=40960:
   candidates.append((int(used),int(util),int(i)))
 current={g for _,_,g in candidates}
 stable=current & _previous_candidates
 _previous_candidates=current
 return [g for _,_,g in sorted(candidates) if g in stable]

def execute(stage,script,args,gpu,python):
 env=dict(ENV,CUDA_VISIBLE_DEVICES=str(gpu))
 folder=(evaluation_analysis_root()/'logs') if 'evaluate' in stage or stage=='analysis' else ROOT/'logs';folder.mkdir(parents=True,exist_ok=True)
 with (folder/(stage+'.log')).open('a') as f:subprocess.run(['nice','-n','10',python,'-u',str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def job(step,gpu):
 arm=f'checkpoint-{step}'
 with LOCK:STATE['checkpoints'][str(step)].update(stage='merge',gpu=gpu)
 save()
 try:
  if shutil.disk_usage(model_root()).free<25*1024**3:raise RuntimeError('Insufficient free disk; preserve history')
  execute('export_'+arm,EXPORTER,['--arm',arm],gpu,TRAIN)
  manifest=ROOT/f'export_{arm}.json';gate=json.loads(manifest.read_text());assert gate['bf16_save_reload']['passed']
  with LOCK:STATE['checkpoints'][str(step)]['stage']='DEV'
  save();execute('evaluate_'+arm,evaluation_analysis_root()/'evaluate_arm.py',['--arm',arm],gpu,INFER)
  meta=evaluation_results_root()/f'evaluation_{arm}.json';x=json.loads(meta.read_text());x.update(verified_export_manifest=str(manifest),verified_export_manifest_sha256=sha(manifest),local_adapter_manifest=str(model_root()/'adapters'/arm/'transfer_verified.json'))
  meta.write_text(json.dumps(x,indent=2))
  with LOCK:STATE['checkpoints'][str(step)]['stage']='completed'
  save()
 except Exception as e:
  with LOCK:STATE['checkpoints'][str(step)].update(stage='failed',error=str(e));STATE['status']='failed'
  save();raise
def main():
 save();pending=STEPS.copy();running={};failed=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  while pending or running:
   for future,(step,gpu) in list(running.items()):
    if future.done():
     try:future.result()
     except Exception:failed.append(traceback.format_exc())
     del running[future]
   if failed:pending=[]
   ready=[s for s in pending if (model_root()/'adapters'/f'checkpoint-{s}'/'transfer_verified.json').is_file()]
   available=[g for g in idle() if g not in {g for _,g in running.values()}]
   for step in ready:
    with LOCK:STATE['checkpoints'][str(step)]['stage']='waiting_for_low_occupancy_GPU'
   while ready and available and len(running)<6:
    step=ready.pop(0);pending.remove(step);gpu=available.pop(0);running[pool.submit(job,step,gpu)]=(step,gpu)
   save()
   if pending or running:time.sleep(20)
 if failed:
  STATE['status']='failed';save();(ROOT/'error.txt').write_text('\n'.join(failed));return
 STATE['stage']='analysis_and_selection';save();execute('analysis',evaluation_analysis_root()/'analyze_original_ce.py',[], '', TRAIN)
 STATE.update(stage='completed',status='completed',selection=json.loads((evaluation_analysis_root()/'best_checkpoint.json').read_text()));save()
if __name__=='__main__':main()
