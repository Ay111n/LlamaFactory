"""Single Full-G2 run, gated validation, six checkpoint DEV runs, manual selection."""
import concurrent.futures,datetime,hashlib,importlib.metadata,json,os,shutil,subprocess,threading,time,traceback
from pathlib import Path
from experiment_paths import *

TRAIN=str(require_path('training_python'));INFER=str(require_path('inference_python'))
ENV=dict(os.environ,SFT_EXPERIMENT_ROOT=str(ROOT),SFT_EXPERIMENT_PATHS=str(paths_file()),
 PYTHONPATH=str(require_path('framework_root')/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
STEPS=[1500,3000,4500,6000,7500,9000]
STATE={'status':'running','full_G2_started':False,'standard_CE_retrained':False,'other_full_experiments_started':False,'TEST_used':False,'manual_checkpoint_selection':True,'checkpoints':{}}
LOCK=threading.RLock()

def save(stage,**kw):
 with LOCK:
  STATE.update(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
  p=ROOT/'pipeline_status.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'pipeline_status.json')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def wait_pid(pid):
 while (Path('/proc')/str(pid)).exists():
  try:
   if (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]=='Z':return
  except FileNotFoundError:return
  time.sleep(10)

def idle():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 occupied={s.split(',')[0].strip() for s in apps.splitlines() if ',' in s}
 return [int(i) for line in raw.splitlines() for i,u,n,m,v in [[s.strip() for s in line.split(',')]] if n=='NVIDIA L20' and int(m)<64 and int(v)<=5 and u not in occupied]

def execute(stage,script,args=(),devices='',python=TRAIN,buffer=False):
 env=dict(ENV,CUDA_VISIBLE_DEVICES=str(devices))
 if buffer:env['FULL_G2_BUFFER_DEVICE']='cuda:1'
 else:env.pop('FULL_G2_BUFFER_DEVICE',None)
 log=stage_log(stage)
 with log.open('a') as f:subprocess.run([python,'-u',str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def prepare_exporter():
 original=require_path('framework_root').parent/'finetuning/scripts/model/mixed_sft_end_ce_20261026/export_end.py'
 target=require_path('framework_root').parent/'finetuning/scripts/model'/ROOT.name/'export_full_g2.py'
 s=original.read_text()
 needle="run=model_root()/'runs'/a.arm;checkpoint=run/'checkpoint-1125'"
 assert needle in s
 s=s.replace(needle,"run=model_root()/'runs/Full-G2';checkpoint=run/a.arm\n    assert a.arm in [f'checkpoint-{s}' for s in [1500,3000,4500,6000,7500,9000]]")
 s=s.replace("assert json.loads((run/'completed.json').read_text())['global_step']==1125","assert json.loads((run/'completed.json').read_text())['steps']==9339")
 s=s.replace('end-ce-v1-functional-two-stage','full-g2-v1-functional-two-stage')
 if target.exists():raise FileExistsError(target)
 target.write_text(s)
 return target

def freeze_inputs(exporter):
 # Reuse prior cache identity checks; no model-weight rescans or history mutation.
 before=json.loads((require_path('preflight_root')/'input_fingerprints.json').read_text())
 critical=[require_path('original_yaml'),require_path('base_model')/'config.json',require_path('base_model')/'tokenizer.json']
 critical+=list(require_path('prepared_dataset').glob('*'))
 critical+=[require_path('framework_root')/'src/llamafactory/train/sft'/n for n in ['trainer.py','termination_trainer.py','gold_termination_trainer.py','termination_gold.py']]
 for p in critical:
  if p.is_file():assert sha(p)==before['files'][str(p)],f'Original source/cache changed: {p}'
 files=[p for p in critical if p.is_file()]+list(ROOT.glob('*.py'))+[paths_file(),ROOT/'protocol.json',ROOT/'full_g2_49800.yaml',exporter]
 files+=list(data_root().glob('*.json'))+[data_root()/'probe_windows.pt']
 files += [script_path('evaluate_arm.py'),evaluation_analysis_root()/'analyze_full_g2.py']
 import inspect
 from transformers import Trainer
 from accelerate import Accelerator
 files += [Path(inspect.getfile(Trainer)),Path(inspect.getfile(Accelerator))]
 m={'files':{str(p):sha(p) for p in files},'old_trainers_and_yaml_unchanged':True,'base_weight_rescan':False,
    'native_gradient_accumulation_scale':16,'auxiliary_coefficient':.0012,'authorization':'one Full-G2 only, no CE retraining, six DEV checkpoints, manual selection'}
 (ROOT/'frozen_inputs.json').write_text(json.dumps(m,indent=2))
 source=evaluation_results_root()/'historical_b0/run_manifest.json'
 historical=json.loads(source.read_text())
 scriptroot=benchmark_root()/'eval/scripts'
 for name,key in [('eval_multilingual_mcqa_vllm.py','evaluator_sha256'),('mcqa_answer_extraction_multilingual.py','answer_extraction_module_sha256'),('multilingual_mcqa_data_registry.json','data_registry_sha256')]:
  assert sha(scriptroot/name)==historical['source_code'][key]
 data=benchmark_root()/'eval/tibetan_formateval_validation/trilingual_aligned/tibetan/mcqa_4way.jsonl'
 assert sha(data)==historical['protocol']['data_sha256']
 (ROOT/'dev_protocol_source_gate.json').write_text(json.dumps({'passed':True,'count':666,'DEV_sha256':sha(data),'no_TEST':True,'greedy':True,'max_tokens':4096,'manual_selection':True},indent=2))

def checkpoint_job(step,gpu,exporter):
 arm=f'checkpoint-{step}'
 with LOCK:STATE['checkpoints'][arm]={'stage':'export','gpu':gpu}
 save('checkpoint_exports_and_DEV')
 try:
  execute('export_'+arm,exporter,('--arm',arm),str(gpu))
  gate=json.loads((ROOT/f'export_{arm}.json').read_text());assert gate['fp32_merge']['passed'] and gate['bf16_save_reload']['passed']
  with LOCK:STATE['checkpoints'][arm]['stage']='DEV'
  save('checkpoint_exports_and_DEV')
  execute('evaluate_'+arm,script_path('evaluate_arm.py'),('--arm',arm),str(gpu),INFER)
  meta=evaluation_results_root()/f'evaluation_{arm}.json';x=json.loads(meta.read_text())
  x['verified_export_manifest']=str(ROOT/f'export_{arm}.json');x['verified_export_manifest_sha256']=sha(ROOT/f'export_{arm}.json');meta.write_text(json.dumps(x,indent=2))
  with LOCK:STATE['checkpoints'][arm]['stage']='completed'
  save('checkpoint_exports_and_DEV')
 except Exception as e:
  with LOCK:STATE['checkpoints'][arm]['stage']='failed';STATE['checkpoints'][arm]['error']=str(e)
  save('checkpoint_failure',status='failed');raise

def main():
 try:
  save('waiting_for_real_validation')
  wait_pid(int((ROOT/'validation.pid').read_text()))
  assert json.loads((ROOT/'unit_gate.json').read_text())['passed']
  assert json.loads((ROOT/'real_ce_equivalence.json').read_text())['passed']
  assert json.loads((ROOT/'initial_gradients.json').read_text())['gate_pass']
  assert json.loads((ROOT/'real_update.json').read_text())['passed']
  checkpoints=json.loads((ROOT/'real_checkpointing.json').read_text())
  assert checkpoints['gradient_checkpointing'] and all(k['use_reentrant'] is True for k in checkpoints['kwargs'])
  exporter=prepare_exporter();freeze_inputs(exporter)
  gates={'unit':True,'real_ce':True,'real_gradient_strength':True,'real_update':True,'input_fingerprints':True,'native_reentrant':True,'DEV_source_protocol':True}
  (ROOT/'gates.json').write_text(json.dumps(gates,indent=2))
  save('waiting_for_training_GPU')
  while True:
   available=idle()
   if len(available)>=2:break
   time.sleep(15)
  if shutil.disk_usage(model_root()).free<190*1024**3:raise RuntimeError('Insufficient free disk for retained native checkpoints plus six exports; no cleanup of history')
  save('full_G2_training',full_G2_started=True,training_GPU=available[0],buffer_GPU=available[1])
  execute('train_Full-G2',ROOT/'run_full_g2.py',('train',),f'{available[0]},{available[1]}',buffer=True)
  save('training_completed_prepare_DEV')
  pending=STEPS.copy();running={}
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
   while pending or running:
    for future,(step,gpu) in list(running.items()):
     if future.done():future.result();del running[future]
    reserved={g for step,g in running.values()};available=[g for g in idle() if g not in reserved]
    while pending and available and len(running)<6:
     if shutil.disk_usage(model_root()).free<25*1024**3:raise RuntimeError('Disk limit reached before export; preserve history')
     step=pending.pop(0);gpu=available.pop(0);running[pool.submit(checkpoint_job,step,gpu,exporter)]=(step,gpu)
    if pending or running:time.sleep(10)
  save('analysis');execute('analysis',evaluation_analysis_root()/'analyze_full_g2.py')
  save('completed',status='completed');(ROOT/'completed.json').write_text(json.dumps(STATE,indent=2))
 except Exception as e:
  save('halted',status='failed',reason=str(e));(ROOT/'pipeline_error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
