"""Matched native SFT pilot. Run on L20 using the existing training environment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
FRAMEWORK=HERE.parents[1]
PROJECT=FRAMEWORK.parent
RUN=HERE.name
DATA=PROJECT/'finetuning/sft/mixed_sft/final_data'/RUN
CACHE=PROJECT/'finetuning/sft/mixed_sft/processed_data'/RUN
MODELS=PROJECT/'finetuning/model/sft'/RUN
RUNTIME=HERE/'runtime'
PYTHON='/opt/anaconda3/envs/training/bin/python'
CONDITIONS=['baseline','crossref']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def data_check():
    m=read(DATA/'manifest.json')
    for name,entry in m['outputs'].items():
        assert sha(DATA/name)==entry['sha256'],name
    return sha(DATA/'manifest.json')


def configs(model):
    recipe=read(HERE/'training_recipe.json')
    return {c:{**recipe,'model_name_or_path':str(model),'dataset':c,'dataset_dir':str(DATA),
               'tokenized_path':str(CACHE/c),'output_dir':str(MODELS/c)} for c in CONDITIONS}


def preflight(model):
    assert Path(sys.executable).resolve()==Path(PYTHON).resolve()
    assert not (RUNTIME/'preflight.json').exists()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['HF_DATASETS_CACHE']=str(CACHE/'hf_cache')
    import torch,transformers,peft,flash_attn
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_dataset,get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    model=model.resolve()
    assert (model/'config.json').is_file()
    data_manifest_sha=data_check()
    RUNTIME.mkdir(parents=True,exist_ok=True)
    recipe=configs(model)
    details={}; previous_targets=None
    for c,cfg in recipe.items():
        assert not (CACHE/c).exists()
        write(RUNTIME/(c+'.json'),cfg)
        cpu_cfg={**cfg,'bf16':False,'flash_attn':'disabled','use_cpu':True,'preprocessing_num_workers':1}
        model_args,data_args,training_args,_,_=get_train_args(cpu_cfg)
        tokenizer=load_tokenizer(model_args)['tokenizer']
        template=get_template_and_fix_tokenizer(tokenizer,data_args)
        raw=rows(DATA/(c+'.jsonl'))
        assert len(raw)==4000
        expected=[];lengths=[];prompts=[]
        for r in raw:
            response=r['conversations'][1]['value']
            assert '<think>' not in response and '</think>' not in response
            messages=[{'role':'user','content':r['conversations'][0]['value']},
                      {'role':'assistant','content':response}]
            source,target=template.encode_oneturn(tokenizer,messages,r['system'])
            assert source and target
            expected.append(target)
            lengths.append(len(source)+len(target))
            prompts.append(len(source))
        assert max(lengths)<=cfg['cutoff_len'],(c,max(lengths))
        native=get_dataset(template,model_args,data_args,training_args,'sft',tokenizer)['train_dataset']
        assert len(native)==len(raw)
        targets=[]
        for i,row in enumerate(native):
            target=[x for x in row['labels'] if x!=-100]
            assert target==expected[i],(c,i,'native target differs')
            assert len(row['input_ids'])==lengths[i],(c,i,'input truncated or changed')
            assert row['labels'][:prompts[i]]==[-100]*prompts[i]
            targets.append(target)
        if previous_targets is not None: assert targets==previous_targets
        previous_targets=targets
        packed=json.dumps(targets,separators=(',',':')).encode()
        details[c]={'records':len(native),'max_sequence_tokens':max(lengths),
                    'total_input_tokens':sum(lengths),'supervised_tokens_per_epoch':sum(map(len,targets)),
                    'target_sequence_sha256':hashlib.sha256(packed).hexdigest(),
                    'prompt_mask_and_untruncated_native_targets_verified':True}
    cache_hashes={p.relative_to(CACHE).as_posix():sha(p) for p in CACHE.rglob('*') if p.is_file()}
    config_hashes={c:sha(RUNTIME/(c+'.json')) for c in CONDITIONS}
    config_meta={p.name:sha(p) for p in model.iterdir() if p.is_file() and p.suffix in ['.json','.jinja']}
    weight_meta={p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in model.glob('*.safetensors')}
    result={'status':'passed','recorded_utc':datetime.now(timezone.utc).isoformat(),
      'python':sys.executable,'versions':{'torch':torch.__version__,'transformers':transformers.__version__,
       'peft':peft.__version__,'flash_attn':flash_attn.__version__},'model':str(model),
      'model_config_hashes':config_meta,'model_weight_metadata':weight_meta,
      'data_manifest_sha256':data_manifest_sha,'conditions':details,
      'cache_files_sha256':cache_hashes,'configs_sha256':config_hashes,
      'recipe_sha256':sha(HERE/'training_recipe.json'),'script_sha256':sha(Path(__file__)),
      'planned_optimizer_steps_per_arm':250,'training_started':False}
    write(RUNTIME/'preflight.json',result)
    print(json.dumps({'status':'passed','conditions':details},ensure_ascii=False),flush=True)


def verify_preflight():
    p=read(RUNTIME/'preflight.json')
    assert p['status']=='passed'
    assert data_check()==p['data_manifest_sha256']
    assert sha(HERE/'training_recipe.json')==p['recipe_sha256']
    assert sha(Path(__file__))==p['script_sha256']
    for c,h in p['configs_sha256'].items():assert sha(RUNTIME/(c+'.json'))==h
    for rel,h in p['cache_files_sha256'].items():assert sha(CACHE/rel)==h
    model=Path(p['model'])
    for name,h in p['model_config_hashes'].items():assert sha(model/name)==h
    for name,s in p['model_weight_metadata'].items():
        actual=(model/name).stat()
        assert actual.st_size==s['bytes'] and actual.st_mtime_ns==s['mtime_ns']
    return p


def launch(gpus):
    assert len(gpus)==2 and len(set(gpus))==2
    verify_preflight()
    assert not (RUNTIME/'launch.json').exists()
    for c in CONDITIONS:assert not (MODELS/c).exists()
    report=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    available={int(s.split(',')[0]):[int(x.strip()) for x in s.split(',')[1:]] for s in report.splitlines()}
    assert all(available[g][0]<100 and available[g][1]==0 for g in gpus),available
    assert shutil.disk_usage(PROJECT).free>70*1024**3
    with (RUNTIME/'supervisor.log').open('ab') as log:
        p=subprocess.Popen([PYTHON,str(Path(__file__)),'run','--gpus',*map(str,gpus)],
                           stdout=log,stderr=subprocess.STDOUT,cwd=FRAMEWORK,start_new_session=True)
    write(RUNTIME/'launch.json',{'supervisor_pid':p.pid,'gpus':gpus,'at_unix':time.time(),
          'gpu_inventory':available,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=FRAMEWORK,text=True).strip()})
    print(json.dumps({'supervisor_pid':p.pid,'gpus':gpus}),flush=True)


def run(gpus):
    verify_preflight()
    started=time.time();children={};logs={}
    for c,gpu in zip(CONDITIONS,gpus):
        env=os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false',
                    'PYTHONPATH':str(FRAMEWORK/'src'),'PYTHONUNBUFFERED':'1',
                    'HF_DATASETS_CACHE':str(CACHE/'hf_cache')})
        logs[c]=(RUNTIME/(c+'.log')).open('ab')
        p=subprocess.Popen([PYTHON,str(FRAMEWORK/'src/train.py'),str(RUNTIME/(c+'.json'))],
                           stdout=logs[c],stderr=subprocess.STDOUT,env=env,cwd=FRAMEWORK)
        children[c]=p
    write(RUNTIME/'children.json',{c:{'pid':p.pid,'gpu':gpus[i]} for i,(c,p) in enumerate(children.items())})
    finished={}
    while len(finished)<2:
        for c,p in children.items():
            if c not in finished and p.poll() is not None:
                logs[c].close()
                item={'returncode':p.returncode,'elapsed_seconds':time.time()-started}
                state=MODELS/c/'trainer_state.json'
                if state.is_file():
                    s=read(state);item.update({'global_step':s['global_step'],'epoch':s['epoch'],
                           'finite_logged_losses':all(math.isfinite(x['loss']) for x in s['log_history'] if 'loss' in x)})
                adapter=MODELS/c/'adapter_model.safetensors'
                if adapter.is_file():item['adapter_sha256']=sha(adapter)
                item['completed']=p.returncode==0 and item.get('global_step')==250 and item.get('finite_logged_losses',False) and adapter.is_file()
                finished[c]=item;write(RUNTIME/('exit_'+c+'.json'),item)
        if len(finished)<2:time.sleep(5)
    verify_preflight()
    write(RUNTIME/'completion.json',{'status':'completed' if all(x['completed'] for x in finished.values()) else 'failed',
          'training_only':True,'evaluation_pending':True,'conditions':finished,'elapsed_seconds':time.time()-started})


def status():
    result={}
    for n in ['launch.json','children.json','completion.json']:
        if (RUNTIME/n).exists():result[n]=read(RUNTIME/n)
    for c in CONDITIONS:
        logs=MODELS/c/'trainer_log.jsonl'
        if logs.exists():
            lines=logs.read_text(errors='replace').splitlines()
            valid=[]
            for line in lines:
                try:valid.append(json.loads(line))
                except json.JSONDecodeError:pass
            if valid:result[c]=valid[-1]
        log=RUNTIME/(c+'.log')
        if c not in result and log.exists():result[c]={'log_tail':log.read_text(errors='replace').splitlines()[-3:]}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','launch','run','status'])
    p.add_argument('--model',type=Path);p.add_argument('--gpus',type=int,nargs=2)
    a=p.parse_args()
    if a.command=='preflight':preflight(a.model)
    elif a.command=='launch':launch(a.gpus)
    elif a.command=='run':run(a.gpus)
    else:status()
