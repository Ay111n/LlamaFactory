"""Fixed gated five-arm schedule, with no full-data training entrypoint."""
import concurrent.futures,datetime,json,os,shutil,subprocess,threading,time,traceback
from pathlib import Path
from experiment_paths import *
TRAIN=str(require_path('training_python'));INFER=str(require_path('inference_python'))
ENV=dict(os.environ,SFT_EXPERIMENT_ROOT=str(ROOT),SFT_EXPERIMENT_PATHS=str(paths_file()),
 PYTHONPATH=str(require_path('framework_root')/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
LOCK=threading.RLock();EXPORT_LOCK=threading.Lock();STATE={'full_training_started':False,'arms':{},'status':'running'}
def status(stage,**extra):
 with LOCK:
  STATE.update(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
  p=ROOT/'status.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'pipeline_status.json')
def idle_gpu():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 occupied={x.split(',')[0].strip() for x in active.splitlines() if ',' in x}
 return [int(i) for line in raw.splitlines() for i,u,n,m,v in [[s.strip() for s in line.split(',')]] if n=='NVIDIA L20' and int(m)<=64 and int(v)<=5 and u not in occupied]
def execute(stage,script,args=(),gpu=0,python=TRAIN):
 status(stage)
 env=dict(ENV,CUDA_VISIBLE_DEVICES=str(gpu))
 log=stage_log(stage)
 with log.open('a') as f:
  subprocess.run([python,'-u',str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def smoke(arm,gpu):
 execute('smoke_'+arm,ROOT/'run_gold.py',('smoke','--arm',arm),gpu)
 assert json.loads((diagnostic_path(model_root()/'smoke'/arm/'completed.json')).read_text())['global_step']==1
def run_arm(arm,gpu):
 with LOCK:STATE['arms'][arm]={'gpu':gpu,'state':'training'}
 execute('train_'+arm,ROOT/'run_gold.py',('train','--arm',arm),gpu)
 with EXPORT_LOCK:
  # Preserve the original v2 parameter-only casting and strict verifier.
  for name in ('export_arm_v2.py','verify_export_v2.py'):
   execute(name[:-3]+'_'+arm,script_path(name),('--arm',arm),gpu)
 export=json.loads((ROOT/f'export_{arm}.json').read_text())
 assert export['all_saved_tensors_bitwise_equal'] and export['rtol']==export['atol']==1e-3
 execute('evaluate_'+arm,script_path('evaluate_arm.py'),('--arm',arm),gpu,INFER)
 with LOCK:STATE['arms'][arm]['state']='completed'
def wave(arms,worker):
 pending=list(arms);running={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  while pending or running:
   for f,(arm,gpu) in list(running.items()):
    if f.done():f.result();del running[f]
   reserved={g for a,g in running.values()}
   available=[g for g in idle_gpu() if g not in reserved]
   while pending and available and len(running)<3:
    arm=pending.pop(0);gpu=available.pop(0)
    # Bound physical disk use before allocating new checkpoints/exports.
    if shutil.disk_usage(model_root().parent).free < 25*1024**3:raise RuntimeError('Insufficient free disk for next stage; no historical cleanup')
    running[pool.submit(worker,arm,gpu)]=(arm,gpu)
   if pending or running:time.sleep(10)
def main():
 try:
  status('waiting_for_calibration')
  pid=int((ROOT/'calibration.pid').read_text())
  while (Path('/proc')/str(pid)).exists():
   try:
    if (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]=='Z':break
   except FileNotFoundError:break
   time.sleep(5)
  cal=json.loads((ROOT/'gold_calibration.json').read_text())
  out=diagnostics_root();out.mkdir(parents=True,exist_ok=True)
  (out/'initial_gradient_distribution.json').write_text(json.dumps(cal,indent=2))
  report=['# Gold 损失初始梯度门禁','', '固定 λ=0.0012；8 个固定有效训练批次，基座参数未更新。R 为裁剪前 λ‖g_aux‖/‖g_CE‖。','',
   '|目标|min|P10|P50|P90|P95|max|通过|','|---|---:|---:|---:|---:|---:|---:|---|']
  for objective,summary in cal['summary'].items():
   report.append('|'+objective+'|'+'|'.join(f'{summary[k]:.6g}' for k in ['min','p10','p50','p90','p95','max'])+'|'+str(summary['gate_pass'])+'|')
  report+=['',f"门禁总结果：{cal['gate_pass']}。P90 必须处于 [0.001,1]；这只是配置诊断，不是方法优劣判据。",'不通过时不启动正式训练、不修改 λ，也不把弱辅助信号归因为方法无效。']
  (out/'initial_gradient_report.md').write_text('\n'.join(report),encoding='utf-8')
  assert cal['gate_pass']
  assert '\nOK\n' in (ROOT/'unit_integration.log').read_text()
  assert json.loads((data_root()/'gold_audit_summary.json').read_text())['continue_boundaries']==219572
  model_root().mkdir(parents=True,exist_ok=True);evaluation_results_root().mkdir(parents=True,exist_ok=True)
  old=require_path('legacy_results_root')/'historical_b0'
  dest=evaluation_results_root()/'historical_b0'
  if not dest.exists():shutil.copytree(old,dest)
  execute('freeze_inputs',ROOT/'freeze_gold.py')
  status('real_update_validation')
  wave(['G1','G2','G3','G1-M','G3-M'],smoke)
  (ROOT/'validation_gates.json').write_text(json.dumps({'unit_integration':True,'gold_audit':True,'initial_gradient_gate':True,'real_update_all_five':True},indent=2))
  status('main_three')
  wave(['G1','G2','G3'],run_arm)
  status('matched_controls')
  wave(['G1-M','G3-M'],run_arm)
  execute('training_diagnostics',ROOT/'diagnostics/analyze_trajectory.py')
  execute('analysis',script_path('analyze_results.py'))
  status('completed',status='completed')
 except Exception as exc:
  status('halted',status='failed',reason=str(exc))
  (ROOT/'pipeline_error.txt').write_text(traceback.format_exc())
  # Preserve completed and independent in-flight work; never change lambda or boundary sets.
  raise
if __name__=='__main__':main()
