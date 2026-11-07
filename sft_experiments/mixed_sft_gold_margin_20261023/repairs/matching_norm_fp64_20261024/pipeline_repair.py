"""Validate accurate norm reduction, preserve failed attempts, rerun both controls."""
import sys,os,json,datetime,subprocess,threading,concurrent.futures,hashlib,shutil,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiment_paths import *
import pipeline_gold as previous
TRAIN=str(require_path('training_python'));INFER=str(require_path('inference_python'))
ENV=dict(previous.ENV)
LOCK=threading.RLock();EXPORT_LOCK=threading.Lock();FAILED=threading.Event()
STATE=json.loads((ROOT/'pipeline_status.json').read_text())
STATE['repair']='matching_norm_fp64_20261024'

def status(stage,**extra):
    with LOCK:
        STATE.update(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        p=ROOT/'repair_status.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'pipeline_status.json')
        (HERE/'recovery_status.json').write_text(json.dumps(STATE,indent=2))

def execute(stage,script,args=(),gpu=0,python=TRAIN):
    if FAILED.is_set():raise RuntimeError('Another recovery stage failed')
    status(stage,status='running')
    env=dict(ENV,CUDA_VISIBLE_DEVICES=str(gpu))
    log=HERE/(stage+'.log')
    try:
        with log.open('a') as f:subprocess.run([python,'-u',str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    except Exception as exc:
        FAILED.set();status(stage,status='failed',reason=str(exc));raise

def wave(worker):
    pending=['G1-M','G3-M'];running={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        while pending or running:
            for future,(arm,gpu) in list(running.items()):
                if future.done():future.result();del running[future]
            available=[g for g in previous.idle_gpu() if g not in {g for a,g in running.values()}]
            while pending and available and not FAILED.is_set():
                arm=pending.pop(0);gpu=available.pop(0)
                if shutil.disk_usage(model_root()).free<25*1024**3:raise RuntimeError('Insufficient disk; no historical cleanup')
                running[pool.submit(worker,arm,gpu)]=(arm,gpu)
            if pending or running:time.sleep(10)

def smoke(arm,gpu):
    execute('smoke_norm64_'+arm,HERE/'run_matched.py',('smoke','--arm',arm),gpu)
    d=model_root()/'smoke_norm_fp64'/arm
    assert json.loads((d/'completed.json').read_text())['global_step']==1
    row=json.loads((d/'termination_steps.jsonl').read_text().strip())
    assert row['match_relative_error']<2e-5

def train(arm,gpu):
    with LOCK:STATE['arms'][arm]={'gpu':gpu,'state':'training','norm_reduction':'FP64'}
    execute('train_norm64_'+arm,HERE/'run_matched.py',('train','--arm',arm),gpu)
    if FAILED.is_set():return
    with EXPORT_LOCK:
        for name in ['export_arm_v2.py','verify_export_v2.py']:
            execute(name[:-3]+'_norm64_'+arm,script_path(name),('--arm',arm),gpu)
    verified=json.loads((ROOT/('export_'+arm+'.json')).read_text())
    assert verified['all_saved_tensors_bitwise_equal'] and verified['rtol']==verified['atol']==1e-3
    execute('evaluate_norm64_'+arm,script_path('evaluate_arm.py'),('--arm',arm),gpu,INFER)
    with LOCK:STATE['arms'][arm]['state']='completed'

def archive_attempts():
    evidence={}
    for arm in ['G1-M','G3-M']:
        src=(model_root()/'runs'/arm).resolve()
        dest=(model_root()/'failed_attempts'/'norm_fp32_20261024'/arm).resolve()
        assert src.parent==(model_root()/'runs').resolve() and dest.parent.parent.parent==model_root().resolve()
        assert not dest.exists() and not (src/'completed.json').exists()
        rows=[json.loads(x) for x in (src/'termination_steps.jsonl').read_text().splitlines()]
        evidence[arm]={'source':str(src),'archive':str(dest),'recorded_windows':len(rows),'last_recorded_step':rows[-1]['step'],'reason':'Original FP32 norm reduction failed or was stopped for a consistent repair of both matched controls'}
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dest))
    (HERE/'archived_attempts.json').write_text(json.dumps(evidence,indent=2))

def main():
    try:
        status('matching_norm_validation',status='repairing')
        execute('unit_norm64',HERE/'test_norm_repair.py')
        assert json.loads((HERE/'numerics_validation.json').read_text())['passed']
        wave(smoke)
        (HERE/'validation_gates_repair.json').write_text(json.dumps({'unit_integration':True,'stable_large_vector_norm':True,'real_update_G1_M':True,'real_update_G3_M':True},indent=2))
        archive_attempts()
        files=[p for p in HERE.glob('*.py')]
        manifest={'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'original_frozen_manifest_sha256':hashlib.sha256((ROOT/'frozen_inputs.json').read_bytes()).hexdigest(),'scope':'Norm accumulation only. Full FP32 gradients, objectives, coefficient, tolerance, optimizer, seeds, data order and budget unchanged. Both controls restart from the same base.','old_failures':json.loads((HERE/'archived_attempts.json').read_text())}
        (HERE/'repair_manifest.json').write_text(json.dumps(manifest,indent=2))
        wave(train)
        for p,h in manifest['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        execute('analysis_after_norm64',script_path('analyze_results.py'))
        report=evaluation_analysis_root()/'report.md'
        with report.open('a',encoding='utf-8') as f:f.write('\n\n## 匹配范数的数值修复\n\n原 FP32 范数归约触发误差门禁，两个未完成匹配运行均已归档。G1-M、G3-M 在保持完整 FP32 梯度向量的前提下，采用 FP64 范数归约，从相同基座重跑；未放宽 2e-5 阈值、调整 λ 或改变预算。修复的验证、源码指纹及归档记录位于训练工程 repairs/matching_norm_fp64_20261024/。\n')
        status('completed',status='completed')
    except Exception as exc:
        FAILED.set();status('halted_norm_repair',status='failed',reason=str(exc))
        (HERE/'recovery_error.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
