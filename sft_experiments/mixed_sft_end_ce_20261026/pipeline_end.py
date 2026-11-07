"""Fixed three-arm schedule; all gates precede formal training."""
import concurrent.futures,datetime,hashlib,json,os,shutil,subprocess,time,traceback,threading
from pathlib import Path
from experiment_paths import *
TRAIN=str(require_path('training_python'));INFER=str(require_path('inference_python'))
ID='mixed_sft_end_ce_20261026';ARMS=['E1','E2','E1-M']
ENV=dict(os.environ,SFT_EXPERIMENT_ROOT=str(ROOT),SFT_EXPERIMENT_PATHS=str(paths_file()),
 PYTHONPATH=str(require_path('framework_root')/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
STATE={'status':'running','arms':{},'full_training_started':False,'TEST_used':False};LOCK=threading.RLock()

def save(stage,**kw):
    with LOCK:
        STATE.update(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
        tmp=ROOT/'pipeline_status.tmp';tmp.write_text(json.dumps(STATE,indent=2));tmp.replace(ROOT/'pipeline_status.json')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def execute(stage,script,args=(),devices='0',buffer=False,python=TRAIN):
    env=dict(ENV,CUDA_VISIBLE_DEVICES=str(devices))
    if buffer:env['END_BUFFER_DEVICE']='cuda:1'
    else:env.pop('END_BUFFER_DEVICE',None)
    log=stage_log(stage)
    with log.open('a') as f:subprocess.run([python,'-u',str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def wait_pid(pid):
    while (Path('/proc')/str(pid)).exists():
        try:
            if (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]=='Z':break
        except FileNotFoundError:break
        time.sleep(10)

def idle():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    occupied={s.split(',')[0].strip() for s in apps.splitlines() if ',' in s}
    return [int(i) for line in raw.splitlines() for i,u,n,m,v in [[s.strip() for s in line.split(',')]] if n=='NVIDIA L20' and int(m)<64 and int(v)<=5 and u not in occupied]

def freeze():
    import torch
    import run_end
    windows=torch.load(require_path('fixed_probe_cache'),weights_only=False,map_location='cpu')
    chosen=[];counts={True:0,False:0}
    for wi,w in enumerate(windows):
        for bi,b in enumerate(w):
            think=bool(b['term_positions'][0])
            if counts[think] >= (2 if think else 1):continue
            valid=int(b['attention_mask'][0].sum())
            chosen.append({'window':wi,'microbatch':bi,'think':think,'input_ids':b['input_ids'][0,:min(valid,512)].tolist()})
            counts[think]+=1
        if len(chosen)==3:break
    assert counts=={True:2,False:1}
    (data_root()/'export_inputs.json').write_text(json.dumps({'samples':chosen,'selection':'first two Think and first No-think in frozen probe order; first <=512 nonpadding tokens'},indent=2))
    gold=require_path('gold_results_root');refs={}
    old=json.loads((gold/'reused_baselines.json').read_text())['S0'];refs['S0']=old
    for arm in ['G1','G2','G3-M']:
        r=Path(json.loads((gold/f'evaluation_{arm}.json').read_text())['run_dir'])
        refs[arm]={'run_dir':str(r),'predictions_sha256':sha(r/'predictions.jsonl'),'manifest_sha256':sha(r/'run_manifest.json')}
    for ref in refs.values():
        assert sha(Path(ref['run_dir'])/'predictions.jsonl')==ref['predictions_sha256']
    (evaluation_results_root()/'reused_baselines.json').write_text(json.dumps(refs,indent=2))
    shutil.copytree(gold/'historical_b0',evaluation_results_root()/'historical_b0')
    historical_model_root=Path(json.loads((require_path('gold_experiment_root')/'paths.json').read_text())['model_root'])
    original_cfg=json.loads((historical_model_root/'runs/G1/resolved_config.json').read_text())
    diffs=[]
    for arm in ARMS:
        current=run_end.config(arm)
        for key in sorted(set(current)|set(original_cfg)):
            changed=current.get(key)!=original_cfg.get(key)
            if changed and key not in ['output_dir','termination_loss_mode']:
                raise ValueError(f'Unexpected historical configuration difference: {arm} {key}')
            diffs.append({'arm':arm,'key':key,'historical':original_cfg.get(key),'new':current.get(key),'changed':changed})
    (diagnostics_root()/'configuration_comparison.json').write_text(json.dumps(diffs,indent=2))
    files=list(ROOT.glob('*.py'))+[paths_file(),ROOT/'protocol.json',data_root()/'export_inputs.json',require_path('fixed_probe_cache'),require_path('fixed_probe_manifest')]
    files+=list(require_path('prepared_dataset').rglob('*'))
    files+=[require_path('framework_root')/'src/llamafactory/train/sft'/n for n in ['termination_end_ce.py','end_ce_trainer.py','termination_trainer.py','termination_loss.py']]
    files+=[require_path('framework_root')/'src/llamafactory/hparams/finetuning_args.py',require_path('framework_root')/'src/llamafactory/train/sft/workflow.py']
    files+=[require_path('framework_root').parent/'finetuning/scripts/model'/ID/'export_end.py']
    files+=[evaluation_analysis_root()/'analyze_end.py',script_path('evaluate_arm.py')]
    files+=[require_path('benchmark_root')/'eval/scripts'/n for n in ['eval_multilingual_mcqa_vllm.py','mcqa_answer_extraction_multilingual.py','multilingual_mcqa_data_registry.json']]
    manifest={'files':{str(p):sha(p) for p in files if p.is_file()},'base_model':str(require_path('base_model')),'base_weight_rescan':False,'baseline_results':refs}
    (ROOT/'frozen_inputs.json').write_text(json.dumps(manifest,indent=2))

def run_arm(arm,model_gpu,buffer_gpu,smoke=False):
    with LOCK:STATE['arms'][arm]={'state':'smoke' if smoke else 'training','model_gpu':model_gpu,'buffer_gpu':buffer_gpu}
    save('smoke' if smoke else 'training')
    execute(('smoke_' if smoke else 'train_')+arm,ROOT/'run_end.py',('smoke' if smoke else 'train','--arm',arm),f'{model_gpu},{buffer_gpu}',True)
    if smoke:return
    with LOCK:STATE['arms'][arm]['state']='exporting'
    save('training_export_evaluation')
    export=require_path('framework_root').parent/'finetuning/scripts/model'/ID/'export_end.py'
    execute('export_'+arm,export,('--arm',arm),str(model_gpu))
    gate=json.loads((ROOT/f'export_{arm}.json').read_text())
    assert gate['fp32_merge']['passed'] and gate['bf16_save_reload']['passed']
    with LOCK:STATE['arms'][arm]['state']='evaluating'
    save('training_export_evaluation')
    execute('evaluate_'+arm,script_path('evaluate_arm.py'),('--arm',arm),str(model_gpu),python=INFER)
    meta=evaluation_results_root()/f'evaluation_{arm}.json'
    info=json.loads(meta.read_text());info['verified_export_manifest']=str(ROOT/f'export_{arm}.json');info['verified_export_manifest_sha256']=sha(ROOT/f'export_{arm}.json');meta.write_text(json.dumps(info,indent=2))
    with LOCK:STATE['arms'][arm]['state']='completed'
    save('training_export_evaluation')

def wave(smoke=False):
    pending=ARMS.copy();running={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while pending or running:
            for f,assignment in list(running.items()):
                if f.done():f.result();del running[f]
            reserved={g for arm,a,b in running.values() for g in [a,b]}
            available=[g for g in idle() if g not in reserved]
            while pending and len(available)>=2 and len(running)<3:
                if shutil.disk_usage(model_root()).free<25*1024**3:raise RuntimeError('Insufficient disk; no deletion of historical data')
                arm=pending.pop(0);a,b=available.pop(0),available.pop(0)
                running[pool.submit(run_arm,arm,a,b,smoke)]=(arm,a,b)
            if pending or running:time.sleep(10)

def main():
    try:
        save('waiting_for_initial_gates')
        for name in ['integration','calibration']:wait_pid(int((ROOT/f'{name}.pid').read_text()))
        assert '\nOK\n' in (ROOT/'unit_integration.log').read_text(),'Integration failed'
        assert json.loads((ROOT/'formula_validation.json').read_text())['passed']
        calibration=json.loads((ROOT/'initial_gradient_gate.json').read_text())
        assert calibration['gate_pass']
        report=['# 初始梯度强度门禁','','固定 λ=0.0012；相同8个有效窗口；无参数更新。','',
                '|目标|min|P10|P50|P90|P95|max|通过|','|---|---:|---:|---:|---:|---:|---:|---|']
        for key,value in calibration['summary'].items():
            report.append('|'+key+'|'+'|'.join(f'{value[n]:.6g}' for n in ['min','p10','p50','p90','p95','max'])+'|'+str(value['gate_pass'])+'|')
        report+=['','R=λ‖g_aux‖/‖g_CE‖，覆盖全部可训练参数；匹配组为自身当前参数上的反事实动态结束梯度范数匹配。',
                 '门禁通过仅表示配置有诊断价值，不是方法有效性结论。逐批分母、损失、范数、余弦及匹配比例见initial_gradient_gate.json。']
        (diagnostics_root()/'initial_gradient_report.md').write_text('\n'.join(report),encoding='utf-8')
        freeze();save('real_update_validation');wave(True)
        initializations=[json.loads((diagnostic_path(model_root()/'smoke'/a/'initialization.json')).read_text()) for a in ARMS]
        reference=json.loads((diagnostic_path(model_root()/'calibration/initialization.json')).read_text())
        assert all(x==reference for x in initializations),'Seed/base/trainable-layout or sampled initialization differs'
        (ROOT/'initialization_comparison.json').write_text(json.dumps({'all_three_match_calibration':True,'basis':'same base/config/seed, trainable counts/dtypes and sampled initial values; no full weight scan'},indent=2))
        (ROOT/'validation_gates.json').write_text(json.dumps({'formulas':True,'integration':True,'initial_gradient_gate':True,'all_three_real_updates':True,'inputs_frozen':True},indent=2))
        save('formal_training');wave(False)
        execute('training_diagnostics',ROOT/'diagnostics/analyze_trajectory.py',devices='')
        save('analysis');execute('analysis',evaluation_analysis_root()/'analyze_end.py',devices='')
        save('completed',status='completed')
        (ROOT/'completed.json').write_text(json.dumps(STATE,indent=2))
    except Exception as e:
        save('halted',status='failed',reason=str(e));(ROOT/'pipeline_error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
