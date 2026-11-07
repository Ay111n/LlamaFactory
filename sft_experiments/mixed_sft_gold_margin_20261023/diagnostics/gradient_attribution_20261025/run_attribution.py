"""Durable read-only diagnostic schedule; never invokes a training entrypoint."""
import concurrent.futures,datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parents[1]
sys.path.insert(0,str(EXPERIMENT))
from experiment_paths import require_path,data_root

PY=str(require_path('training_python'))
ANALYSIS=Path('/home/nlp/data/lyl/benchmark/eval/analysis/mixed_sft_gold_margin_gradient_attribution_20261025')
ENV=dict(os.environ,SFT_EXPERIMENT_ROOT=str(EXPERIMENT),SFT_EXPERIMENT_PATHS=str(EXPERIMENT/'paths.json'),PYTHONPATH=str(require_path('framework_root')/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
STATE={'status':'running','optimizer_updates':0,'generation_calls':0,'DEV_or_TEST_used':False,'jobs':{}}

def status(phase,**extra):
    STATE.update(phase=phase,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    p=HERE/'status.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(HERE/'status.json')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def mem_gib():
    row=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return int(row.split()[1])/1024**2

def idle():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    occupied={s.split(',')[0].strip() for s in used.splitlines() if ',' in s}
    return [int(i) for line in raw.splitlines() for i,u,n,m,v in [[x.strip() for x in line.split(',')]] if n=='NVIDIA L20' and int(m)<=64 and int(v)<=5 and u not in occupied]

def run(job,gpus):
    STATE['jobs'][job]={'status':'running','gpus':gpus,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    env=dict(ENV,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)))
    with (HERE/(job+'.log')).open('w') as log:
        p=subprocess.Popen([PY,'-u',str(HERE/'component_probe.py'),job],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        STATE['jobs'][job]['pid']=p.pid
        code=p.wait()
    if code:
        STATE['jobs'][job]['status']='failed'
        status('worker_failed',status='failed',reason=f'{job} exit {code}; see {job}.log')
        raise RuntimeError(f'{job} failed with exit {code}; see {job}.log')
    completion='initial_shared' if job=='initial' else 'final_pair' if job=='pair' else 'final_'+job
    assert json.loads((HERE/'completion'/f'{completion}.json').read_text())['complete']
    STATE['jobs'][job]['status']='completed'

def main():
    lock=(HERE/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        status('waiting_for_fixed_windows')
        pid=int((HERE/'prepare.pid').read_text())
        while (Path('/proc')/str(pid)).exists():
            try:
                if (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]=='Z':break
            except FileNotFoundError:break
            time.sleep(5)
        manifest=data_root()/'gradient_attribution_20261025/manifest.json';assert manifest.exists()
        assert '\nOK\n' in (HERE/'unit_final.log').read_text()
        source={'probe_source_sha256':sha(HERE/'component_probe.py'),'test_source_sha256':sha(HERE/'test_component_probe.py'),'cache_manifest':json.loads(manifest.read_text()),'stage_rng':'Original initial CUDA seed state; saved final checkpoint RNG states, identical across all five arms; forked/restored per window.','normalization':'CE sum/window shifted-token count; auxiliary sum/window Think count; original category balance and detached weights','optimizer_updates':0}
        import torch
        rng_hashes={}
        for arm in ['G1','G2','G3','G1-M','G3-M']:
            p=__import__('experiment_paths').model_root()/'runs'/arm/'checkpoint-1125/rng_state.pth'
            saved=torch.load(p,map_location='cpu',weights_only=False)
            cuda=saved['cuda'][0] if isinstance(saved['cuda'],list) else saved['cuda']
            rng_hashes[arm]={'cuda':hashlib.sha256(cuda.numpy().tobytes()).hexdigest(),'cpu':hashlib.sha256(saved['cpu'].numpy().tobytes()).hexdigest()}
        assert len({v['cuda'] for v in rng_hashes.values()})==1
        assert len({v['cpu'] for v in rng_hashes.values()})==1
        source['native_final_rng_hashes']=rng_hashes
        (HERE/'protocol_manifest.json').write_text(json.dumps(source,indent=2))
        status('verify_original_frozen_sources')
        original=json.loads((EXPERIMENT/'frozen_inputs.json').read_text())
        for name,h in original['files'].items():
            assert sha(Path(name))==h,f'Original frozen input changed: {name}'
            print('VERIFIED',name,flush=True)
        (HERE/'original_sources_verified.json').write_text(json.dumps({'verified_files':len(original['files']),'frozen_manifest_sha256':sha(EXPERIMENT/'frozen_inputs.json')},indent=2))
        pending=['initial','pair','G1','G3','G1-M'];running={};budget={'initial':35,'pair':34,'G1':17,'G3':20,'G1-M':21}
        cap=max(0,mem_gib()-18)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            while pending or running:
                for future,(job,gpus) in list(running.items()):
                    if future.done():future.result();del running[future]
                reserved={g for job,gpus in running.values() for g in gpus}
                free=[g for g in idle() if g not in reserved]
                while pending and len(running)<2:
                    job=pending[0];need=2 if job=='pair' else 1
                    claimed=sum(budget[job] for job,_ in running.values())
                    if len(free)<need or claimed+budget[job]>cap or mem_gib()<budget[job]+16:break
                    pending.pop(0);cards=free[:need];free=free[need:]
                    running[pool.submit(run,job,cards)]=(job,cards)
                status('read_only_component_vjps',pending=pending,memory_available_gib=mem_gib(),reserved_budget_cap_gib=cap)
                if pending or running:time.sleep(20)
        assert sha(HERE/'component_probe.py')==source['probe_source_sha256']
        assert sha(HERE/'test_component_probe.py')==source['test_source_sha256']
        status('summarize_and_plot')
        script=ANALYSIS/'summarize_attribution.py';assert script.exists()
        (HERE/'analysis_source_sha256.txt').write_text(sha(script))
        with (HERE/'summary.log').open('w') as log:
            subprocess.run([PY,str(script),'--diagnostic-root',str(HERE),'--output',str(ANALYSIS)],env=dict(ENV,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
        status('completed',status='completed')
    except Exception as exc:
        status('halted',status='failed',reason=str(exc));(HERE/'error.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
