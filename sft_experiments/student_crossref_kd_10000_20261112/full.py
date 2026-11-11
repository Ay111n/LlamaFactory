"""Full 10k matched students, sharing the validated pilot trainer and losses."""
import sys,os,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PILOT=HERE.parent/'student_crossref_kd_2000_20261111'
sys.path.insert(0,str(PILOT))
import experiment as e
ORIGINAL_CONFIG=e.config
e.HERE=HERE;e.RUN=HERE.name;e.RUNTIME=HERE/'runtime'
e.DATA=e.ROOT/'finetuning/sft/mixed_sft/final_data/student_crossref_10000_20261112'
e.CACHE=e.ROOT/'finetuning/sft/mixed_sft/processed_data'/e.RUN
e.MODELS=e.ROOT/'finetuning/model/sft'/e.RUN
FULLCACHE=e.ROOT/'finetuning/sft/mixed_sft/processed_data/teacher_cache_10000_20261112'
STEPS=1250

def config(arm):
 d=ORIGINAL_CONFIG(arm);d.update(save_steps=STEPS,save_total_limit=1)
 return d
e.config=config

def verify():
 p=e.read(e.RUNTIME/'preflight.json')
 assert e.sha(e.CACHE/'examples.jsonl')==p['examples_sha256']
 for n,h in p['source_sha256'].items():assert e.sha(e.DATA/n)==h
 assert e.sha(e.TEACHER/'merge_manifest.json')==p['teacher_merge_manifest_sha256']
 assert e.sha(FULLCACHE/'teacher_cache_manifest.json')==p['full_teacher_cache_manifest_sha256']
 return p
e.verify=verify

def prepare():
 import torch,transformers,peft,flash_attn
 from llamafactory.model import load_tokenizer
 from llamafactory.data import get_template_and_fix_tokenizer
 assert sys.executable=='/opt/anaconda3/envs/training/bin/python'
 e.RUNTIME.mkdir(exist_ok=True);e.CACHE.mkdir(parents=True,exist_ok=True)
 assert not (e.CACHE/'examples.jsonl').exists()
 manifest=e.read(e.DATA/'manifest.json')
 for n,m in manifest['outputs'].items():assert e.sha(e.DATA/n)==m['sha256']
 pairs=e.rows(e.DATA/'train_pairs.jsonl');raw=e.rows(e.DATA/'baseline.jsonl')
 assert len(pairs)==len({r['uid'] for r in pairs})==10000 and len(raw)==20000
 cache=e.read(FULLCACHE/'teacher_cache_manifest.json');assert cache['status']=='completed' and cache['samples']==10000
 assert e.sha(FULLCACHE/'teacher_examples.jsonl')==cache['examples_sha256']
 teacher={r['uid']:r for r in e.rows(FULLCACHE/'teacher_examples.jsonl')}
 assert set(teacher)==set(cache['files'])=={r['uid'] for r in pairs}
 ma,da,ta,fa,ga=e.setup_args('baseline',True)
 tok=load_tokenizer(ma)['tokenizer'];template=get_template_and_fix_tokenizer(tok,da)
 from transformers import AutoTokenizer
 tt=AutoTokenizer.from_pretrained(e.TEACHER,local_files_only=True)
 assert tok.get_vocab()==tt.get_vocab() and tok.chat_template==tt.chat_template
 result=[]
 for i,r in enumerate(raw):
  response=r['conversations'][1]['value'];assert '<think>' not in response and '</think>' not in response
  src,target=template.encode_oneturn(tok,[{'role':'user','content':r['conversations'][0]['value']},{'role':'assistant','content':response}],r['system'])
  assert src and target and len(src)+len(target)<=8192
  m={'uid':r['source_uid'],'language':r['language'],'index':i,'prompt_length':len(src),'target':target}
  if m['language']=='bo':assert target==teacher[m['uid']]['target']
  result.append({'input_ids':src+target,'labels':[-100]*len(src)+target,'meta':m})
  if (i+1)%2000==0:print(json.dumps({'stage':'prepare','views':i+1}),flush=True)
 for uid,entry in cache['files'].items():
  assert Path(entry['file']).is_file() and Path(entry['file']).stat().st_size>0
  assert entry['shape']==[len(teacher[uid]['target']),4096]
 assert e.sha(Path(cache['head_file']))==cache['head_sha256']
 (e.CACHE/'teacher_head.pt').symlink_to(cache['head_file'])
 output=e.CACHE/'examples.jsonl';output.write_text(''.join(json.dumps(r)+'\n' for r in result))
 e.save(e.CACHE/'teacher_cache_manifest.json',{**cache,'examples_sha256':e.sha(output),'teacher_examples_sha256':cache['examples_sha256'],'full_cache_manifest_sha256':e.sha(FULLCACHE/'teacher_cache_manifest.json')})
 for arm in ['baseline','kd']:e.save(e.RUNTIME/(arm+'_config.json'),config(arm))
 p={'status':'passed','source_pairs':10000,'views':20000,'bo_views':10000,'planned_steps':STEPS,
 'examples_sha256':e.sha(output),'source_sha256':{n:e.sha(e.DATA/n) for n in ['train_pairs.jsonl','baseline.jsonl','manifest.json']},
 'teacher_merge_manifest_sha256':e.sha(e.TEACHER/'merge_manifest.json'),'full_teacher_cache_manifest_sha256':e.sha(FULLCACHE/'teacher_cache_manifest.json'),
 'max_student_tokens':max(len(r['input_ids']) for r in result),'max_bo_tokens':max(len(r['input_ids']) for r in result if r['meta']['language']=='bo'),
 'base':str(e.BASE),'teacher':str(e.TEACHER),'versions':{'python':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'flash_attn':flash_attn.__version__},
 'objective':'0.5 CE_en + 0.5 CE_bo + 0.1 KL_bo; per-response means; frozen reference-privileged teacher; native OFF target terminator/newline',
 'cache_integrity_policy':'completed full-cache hash audit reused; manifest hash verified; every accessed tensor SHA-256 is verified by the unchanged trainer','diagnostic_policy':manifest['diagnostic_policy'],'known_malformed_uid_retained':manifest['known_malformed_uid_retained'],'checkpoint_limit_per_arm':1}
 e.save(e.RUNTIME/'preflight.json',p);print(json.dumps(p),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','train']);p.add_argument('--arm',choices=['baseline','kd'],default='kd');a=p.parse_args()
 if a.command=='prepare':prepare()
 else:
  import torch
  torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory,0)
  e.train(a.arm,a.command=='smoke',expected_steps=STEPS)
