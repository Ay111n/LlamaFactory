"""Explicitly authorized six checkpoint DEV; never runs training."""
import concurrent.futures,datetime,json,os,shutil,time,traceback,threading
from pathlib import Path
import pipeline_full_g2 as base
from experiment_paths import *
STEPS=[1500,3000,4500,6000,7500,9000]
STATE={'status':'running','training_completed_steps':9339,'merge_logits_gate':'waived_by_user','TEST_used':False,'manual_selection':True,'checkpoints':{}}
LOCK=threading.RLock()
EXPORTER=Path('/home/nlp/data/lyl/finetuning/scripts/model/mixed_sft_g2_full_49800_20261026/export_full_g2_authorized_dev.py')
def save():
 with LOCK:
  STATE['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  p=ROOT/'dev_resume_status.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'dev_resume_status.json')
def job(step,gpu):
 arm=f'checkpoint-{step}'
 with LOCK:STATE['checkpoints'][arm]={'stage':'export','gpu':gpu}
 save()
 try:
  destination=model_root()/'merged'/arm
  if destination.exists():
   # Preserve earlier incomplete exports; never touch training checkpoints.
   assert destination.resolve().parent==(model_root()/'merged').resolve()
   archive=model_root()/'incomplete_exports_20261028'/arm
   archive.parent.mkdir(parents=True,exist_ok=True)
   if archive.exists():raise FileExistsError(archive)
   destination.rename(archive)
  if shutil.disk_usage(model_root()).free<25*1024**3:raise RuntimeError('Insufficient disk; preserve files and stop')
  base.execute('export_authorized_'+arm,EXPORTER,('--arm',arm),str(gpu))
  manifest=ROOT/f'export_{arm}.json';gate=json.loads(manifest.read_text())
  assert gate['merge_tolerance_waived_by_user'] and gate['bf16_save_reload']['passed']
  with LOCK:STATE['checkpoints'][arm]['stage']='DEV'
  save()
  base.execute('evaluate_authorized_'+arm,evaluation_analysis_root()/'evaluate_arm.py',('--arm',arm),str(gpu),base.INFER)
  meta=evaluation_results_root()/f'evaluation_{arm}.json';x=json.loads(meta.read_text())
  x.update(verified_export_manifest=str(manifest),verified_export_manifest_sha256=base.sha(manifest),merge_logits_gate='waived_by_user')
  meta.write_text(json.dumps(x,indent=2))
  with LOCK:STATE['checkpoints'][arm]['stage']='completed'
  save()
 except Exception as e:
  with LOCK:STATE['checkpoints'][arm].update(stage='failed',error=str(e));STATE['status']='failed'
  save();raise
def main():
 save();pending=STEPS.copy();running={};failures=[]
 assert json.loads((model_root()/'runs/Full-G2/completed.json').read_text())['steps']==9339
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  while pending or running:
   for future,(step,gpu) in list(running.items()):
    if future.done():
     try:future.result()
     except Exception:failures.append(traceback.format_exc())
     del running[future]
   if failures:pending=[]
   available=[g for g in base.idle() if g not in {g for _,g in running.values()}]
   while pending and available and len(running)<2:
    step=pending.pop(0);gpu=available.pop(0)
    running[pool.submit(job,step,gpu)]=(step,gpu)
   if pending or running:time.sleep(10)
 if failures:
  (ROOT/'dev_resume_error.txt').write_text('\n'.join(failures));return
 base.execute('analysis_authorized',evaluation_analysis_root()/'analyze_full_g2.py')
 report=evaluation_analysis_root()/'report.md'
 with report.open('a') as f:f.write('\n\n导出说明：按2026-10-28用户明确授权，六组均不以挂载adapter与合并后logits容差作为门禁；保留adapter/lm_head加载核验、BF16保存重载验证及原DEV协议。不能将本次评测表述为合并前后logits一致性已通过。\n')
 STATE['status']='completed';save()
if __name__=='__main__':main()
