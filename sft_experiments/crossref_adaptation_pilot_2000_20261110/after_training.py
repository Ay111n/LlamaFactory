"""Wait for this pilot, merge final adapters, then run matched diagnostics and DEV."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
RUN=HERE.name
TRAIN_RUNTIME=HERE/'runtime_v2'
RUNTIME=HERE/'runtime_followup'
BENCH=PROJECT/'benchmark'
EVAL=BENCH/'eval/scripts'/RUN/'evaluate.py'
PYTHON='/opt/anaconda3/envs/inference/bin/python'
MERGE=PROJECT/'finetuning/scripts/model/ce_g2_thinking_dev_32768_20261031/merge_lora_adapter.py'
MODELS=PROJECT/'finetuning/model/sft'/RUN
CONDITIONS=['baseline','crossref']


def read(p):return json.loads(p.read_text())


def save(obj):
    RUNTIME.mkdir(parents=True,exist_ok=True)
    p=RUNTIME/'execution.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(p)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def gpus():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(a):[int(b),int(c)] for a,b,c in (x.split(',') for x in s.splitlines())}


def worker(condition,gpu):
    assert read(TRAIN_RUNTIME/'completion.json')['status']=='completed'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
    out=BENCH/'eval/analysis'/RUN;out.mkdir(parents=True,exist_ok=True)
    for mode in ['diagnostic','dev']:
        cmd=[PYTHON,str(EVAL),mode,'--condition',condition,'--model',str(MODELS/'merged'/condition)]
        log=out/(condition+'_'+mode+'.log')
        assert not log.exists()
        with log.open('w') as fh:
            result=subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT,env=env,cwd=BENCH)
        if result.returncode:raise RuntimeError((condition,mode,result.returncode))


def run():
    assert not (RUNTIME/'execution.json').exists()
    assert EVAL.is_file() and MERGE.is_file()
    check=read(BENCH/'eval/analysis'/RUN/'preflight.json')
    assert sha(EVAL)==check['script_sha256']
    merge_hash=sha(MERGE)
    state={'status':'waiting_for_training','started_unix':time.time(),'training_runtime':str(TRAIN_RUNTIME),
           'evaluation_script_sha256':sha(EVAL),'merge_script_sha256':merge_hash,
           'steps':[],'semantic_review_pending':True}
    save(state)
    try:
        while not (TRAIN_RUNTIME/'completion.json').exists():
            if time.time()-state['started_unix']>86400:raise TimeoutError('Training did not finish within 24 hours')
            time.sleep(30)
        trained=read(TRAIN_RUNTIME/'completion.json')
        assert trained['status']=='completed',trained
        state['training_completion']=trained;state['status']='merging';save(state)
        base=read(TRAIN_RUNTIME/'preflight.json')['model']
        for c in CONDITIONS:
            assert sha(MERGE)==merge_hash
            adapter=MODELS/c;assert sha(adapter/'adapter_model.safetensors')==trained['conditions'][c]['adapter_sha256']
            cmd=[PYTHON,str(MERGE),'--base-model',base,'--adapter',str(adapter),
                 '--output',str(MODELS/'merged'/c),'--label',RUN+'_'+c]
            log=RUNTIME/(c+'_merge.log')
            with log.open('w') as fh:
                result=subprocess.run(cmd,stdout=fh,stderr=subprocess.STDOUT,
                     env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),cwd=PROJECT)
            state['steps'].append({'kind':'merge','condition':c,'exit_code':result.returncode,'log':str(log)})
            save(state)
            assert result.returncode==0,(c,'merge failed')
            manifest=read(MODELS/'merged'/c/'merge_manifest.json')
            assert manifest['adapter_files_sha256']['adapter_model.safetensors']==trained['conditions'][c]['adapter_sha256']
        state['status']='evaluating';save(state)
        pending=list(CONDITIONS);active={};finished=set();failed=set();waiting_since=time.time()
        while pending or active:
            occupied={x[2] for x in active.values()}
            free=[g for g,(memory,utilization) in gpus().items() if memory<100 and utilization==0 and g not in occupied]
            free.sort(key=lambda g:(g not in [6,3],g))
            while pending and free:
                c=pending.pop(0);gpu=free.pop(0)
                log=RUNTIME/(c+'_evaluation_worker.log');fh=log.open('w')
                proc=subprocess.Popen([PYTHON,str(Path(__file__)),'worker','--condition',c,'--gpu',str(gpu)],
                                      stdout=fh,stderr=subprocess.STDOUT,cwd=PROJECT)
                active[c]=(proc,fh,gpu)
                state['steps'].append({'kind':'evaluate','condition':c,'gpu':gpu,'pid':proc.pid,'status':'running'})
                save(state)
            for c,(proc,fh,gpu) in list(active.items()):
                if proc.poll() is not None:
                    fh.close();del active[c]
                    for item in state['steps']:
                        if item['kind']=='evaluate' and item['condition']==c:
                            item.update(status='completed' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
                    save(state)
                    if proc.returncode:failed.add(c)
                    else:finished.add(c)
            if pending and not active and time.time()-waiting_since>86400:raise TimeoutError('No free evaluation GPU')
            if pending or active:time.sleep(15)
        assert not failed,('Evaluation failed',sorted(failed))
        assert finished==set(CONDITIONS)
        result=subprocess.run([PYTHON,str(EVAL),'compare'],capture_output=True,text=True,cwd=PROJECT)
        (RUNTIME/'comparison.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr[-2000:]
        state.update(status='completed',finished_unix=time.time(),result_root=str(BENCH/'eval/results'/RUN),
                     analysis_root=str(BENCH/'eval/analysis'/RUN),semantic_review_pending=True)
        save(state)
    except Exception as e:
        state.update(status='failed',error=repr(e),finished_unix=time.time());save(state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker'])
    p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.command=='run':run()
    else:worker(a.condition,a.gpu)
