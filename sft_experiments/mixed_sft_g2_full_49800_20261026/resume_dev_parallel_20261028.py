"""Adopt two existing DEV jobs and expand to six independent checkpoint jobs."""
import concurrent.futures,json,time,traceback
from pathlib import Path
import resume_dev_authorized_20261028 as work
from experiment_paths import *
work.STATE.update(json.loads((ROOT/'dev_resume_status.json').read_text()))
work.STATE['scheduler']='six_parallel_checkpoint_jobs'
ADOPTED={1500:(2668904,0),3000:(2668905,2)}
def adopt(step,pid,gpu):
 arm=f'checkpoint-{step}'
 work.base.wait_pid(pid)
 meta=evaluation_results_root()/f'evaluation_{arm}.json'
 if not meta.is_file():raise RuntimeError(f'Adopted DEV process {pid} exited without result metadata for {arm}')
 x=json.loads(meta.read_text());assert (Path(x['run_dir'])/'predictions.jsonl').is_file()
 manifest=ROOT/f'export_{arm}.json'
 x.update(verified_export_manifest=str(manifest),verified_export_manifest_sha256=work.base.sha(manifest),merge_logits_gate='waived_by_user')
 meta.write_text(json.dumps(x,indent=2))
 with work.LOCK:work.STATE['checkpoints'][arm].update(stage='completed',adopted_process=pid)
 work.save()
def main():
 work.save();pending=[4500,6000,7500,9000];running={};failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for step,(pid,gpu) in ADOPTED.items():running[pool.submit(adopt,step,pid,gpu)]=(step,gpu)
  while pending or running:
   for future,(step,gpu) in list(running.items()):
    if future.done():
     try:future.result()
     except Exception:
      failures.append(traceback.format_exc())
      with work.LOCK:work.STATE['checkpoints'][f'checkpoint-{step}'].update(stage='failed',error=failures[-1])
     del running[future]
   if failures:pending=[]
   available=[g for g in work.base.idle() if g not in {g for _,g in running.values()}]
   while pending and available and len(running)<6:
    step=pending.pop(0);gpu=available.pop(0)
    running[pool.submit(work.job,step,gpu)]=(step,gpu)
   work.save()
   if pending or running:time.sleep(10)
 if failures:
  work.STATE['status']='failed';work.save();(ROOT/'dev_parallel_error.txt').write_text('\n'.join(failures));return
 work.base.execute('analysis_authorized_parallel',evaluation_analysis_root()/'analyze_full_g2.py')
 report=evaluation_analysis_root()/'report.md'
 with report.open('a') as f:f.write('\n\n按2026-10-28用户授权，六组不以挂载adapter与合并后logits容差作为门禁；该验证已豁免，不能表述为一致性验证通过。保留BF16保存重载检查与原DEV协议。\n')
 work.STATE['status']='completed';work.save()
if __name__=='__main__':main()
