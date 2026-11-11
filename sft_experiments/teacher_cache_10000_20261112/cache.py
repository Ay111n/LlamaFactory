"""Complete the original 10k teacher-forcing cache, reusing verified pilot entries."""
import os,sys,json,time,hashlib,traceback,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUNTIME=HERE/'runtime'
CACHE=ROOT/'finetuning/sft/mixed_sft/processed_data'/HERE.name
OLD=ROOT/'finetuning/sft/mixed_sft/processed_data/student_crossref_kd_2000_20261111'
TEACHER=ROOT/'finetuning/model/sft/crossref_teacher_9960_20261111/merged/crossref'
SOURCE=ROOT/'finetuning/sft/mixed_sft/final_data/bilingual_qa_10000_20261110/bilingual_qa_10000.jsonl'
DIAG=ROOT/'finetuning/sft/mixed_sft/final_data/crossref_teacher_9960_20261111/diagnostic_pairs.jsonl'
PROMPT=ROOT/'benchmark/eval/analysis/teacher_probe_prompt_v2_20261110/teacher_prompt.txt'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def status(stage,**kw):
 x={'stage':stage,'pid':os.getpid(),'updated':time.time(),**kw};save(RUNTIME/'status.json',x);print(json.dumps(x),flush=True)
def main():
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 assert sys.executable=='/opt/anaconda3/envs/inference/bin/python'
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
 RUNTIME.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 lock=(RUNTIME/'cache.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory,0)
 old=read(OLD/'teacher_cache_manifest.json');assert old['status']=='completed' and old['samples']==2000
 assert old['teacher_merge_manifest_sha256']==sha(TEACHER/'merge_manifest.json')
 assert sha(OLD/'teacher_head.pt')==old['head_sha256']
 assert sha(OLD/'examples.jsonl')==old['examples_sha256']
 pairs=rows(SOURCE);assert len(pairs)==len({r['uid'] for r in pairs})==10000
 diag={r['uid'] for r in rows(DIAG)};assert len(diag)==40
 tok=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True)
 wrapper=PROMPT.read_text().strip();examples=[]
 for pair in pairs:
  content=wrapper
  for lng,key in [('english','question'),('english','answer'),('tibetan','question')]:content=content.replace('{'+lng+'.'+key+'}',pair[lng][key])
  prompt=tok.apply_chat_template([{'role':'system','content':'Respond in Tibetan.'},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  tp=tok.encode(prompt,add_special_tokens=False);body=tok.encode(pair['tibetan']['answer'],add_special_tokens=False)
  target=body+tok.encode('<|im_end|>\n',add_special_tokens=False)
  assert tok.encode(prompt+pair['tibetan']['answer'],add_special_tokens=False)==tp+body
  ex={'uid':pair['uid'],'teacher_input_ids':tp+target[:-1],'teacher_prompt_length':len(tp),'target':target,'diagnostic_holdout':pair['uid'] in diag}
  assert len(ex['teacher_input_ids'])<=32768
  examples.append(ex)
 byid={e['uid']:e for e in examples};files={}
 for r in rows(OLD/'examples.jsonl'):
  m=r['meta']
  if m['language']!='bo':continue
  e=byid[m['uid']]
  for k in ['teacher_input_ids','teacher_prompt_length','target']:assert e[k]==m[k],(m['uid'],k)
  entry=old['files'][m['uid']];assert sha(Path(entry['file']))==entry['sha256'];files[m['uid']]=entry
 assert len(files)==2000
 ep=CACHE/'teacher_examples.jsonl'
 payload=''.join(json.dumps(e)+'\n' for e in examples)
 if ep.exists():assert ep.read_text()==payload
 else:ep.write_text(payload)
 contract={'source_sha256':sha(SOURCE),'prompt_sha256':sha(PROMPT),'examples_sha256':sha(ep),'teacher_merge_manifest_sha256':sha(TEACHER/'merge_manifest.json'),'head_sha256':old['head_sha256'],'head_file':str(OLD/'teacher_head.pt'),'samples':10000,'diagnostic_holdout_uids':sorted(diag),'known_malformed_uids':['Math_no_think_0010880'],'source_unchanged':True,'torch':torch.__version__,'transformers':transformers.__version__,'attention':'sdpa','dtype':'bfloat16','gpu':0,'allocator_limit_gib':24,'reused_pilot_samples':2000,'no_logit_topk_approximation':True}
 if (CACHE/'contract.json').exists():assert read(CACHE/'contract.json')==contract
 else:save(CACHE/'contract.json',contract)
 journal=CACHE/'new_entries.jsonl'
 if journal.exists():
  for row in rows(journal):
   assert row['uid'] not in files and row['uid'] in byid
   assert sha(Path(row['entry']['file']))==row['entry']['sha256'];files[row['uid']]=row['entry']
 pending=sorted((e for e in examples if e['uid'] not in files),key=lambda e:len(e['teacher_input_ids']),reverse=True)
 status('loading_teacher',completed=len(files),total=10000,pending=len(pending),max_teacher_tokens=max(len(e['teacher_input_ids']) for e in examples),target_positions=sum(len(e['target']) for e in examples))
 model=AutoModelForCausalLM.from_pretrained(TEACHER,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True)
 model.eval().requires_grad_(False)
 assert torch.equal(model.lm_head.weight.detach().cpu(),torch.load(OLD/'teacher_head.pt',weights_only=True))
 out=CACHE/'teacher_hidden';out.mkdir(exist_ok=True)
 started=time.time();initial=len(files)
 with torch.inference_mode():
  # Validate the shared teacher and cache coordinates on an existing pilot item.
  uid=next(iter(old['files']));e=byid[uid]
  h=model.model(input_ids=torch.tensor([e['teacher_input_ids']],device='cuda'),use_cache=False).last_hidden_state[0,e['teacher_prompt_length']-1:].contiguous()
  previous=torch.load(old['files'][uid]['file'],weights_only=True).cuda()
  assert torch.equal(h,previous),'Teacher cache reproduction differs'
  save(RUNTIME/'reuse_check.json',{'uid':uid,'hidden_exact':True,'head_exact':True,'all_2000_input_target_and_hash_checks':True})
  del h,previous
  for i,e in enumerate(pending):
   h=model.model(input_ids=torch.tensor([e['teacher_input_ids']],device='cuda'),use_cache=False).last_hidden_state[0,e['teacher_prompt_length']-1:].contiguous()
   assert h.shape[0]==len(e['target']) and torch.isfinite(h).all()
   path=out/(e['uid']+'.pt');tmp=path.with_suffix('.pt.tmp');torch.save(h.cpu(),tmp);tmp.replace(path)
   entry={'file':str(path),'sha256':sha(path),'shape':list(h.shape)};files[e['uid']]=entry
   with journal.open('a') as f:f.write(json.dumps({'uid':e['uid'],'entry':entry})+'\n');f.flush();os.fsync(f.fileno())
   if i==0:save(RUNTIME/'longest_check.json',{'passed':True,'uid':e['uid'],'teacher_tokens':len(e['teacher_input_ids']),'target_tokens':len(e['target']),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
   if i==0 or (i+1)%50==0:status('caching',completed=len(files),total=10000,new_completed=i+1,new_total=len(pending),seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
 assert set(files)==set(byid)
 for uid,entry in files.items():assert sha(Path(entry['file']))==entry['sha256'] and entry['shape'][0]==len(byid[uid]['target'])
 save(CACHE/'teacher_cache_manifest.json',{**contract,'status':'completed','files':files,'new_samples':8000,'total_tensor_bytes':sum(Path(x['file']).stat().st_size for x in files.values()),'seconds_this_run':time.time()-started,'holdout_note':'40 diagnostic items cached but marked; caching does not authorize training on holdout items'})
 status('completed',completed=10000,total=10000,new_samples=8000,reused_samples=2000,manifest_sha256=sha(CACHE/'teacher_cache_manifest.json'),seconds=time.time()-started)
if __name__=='__main__':
 try:main()
 except Exception:
  status('failed',error=traceback.format_exc());raise
