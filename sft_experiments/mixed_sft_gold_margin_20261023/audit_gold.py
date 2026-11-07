"""Read frozen cache, audit Gold IDs/types and baseline identity without rewriting it."""
import csv,hashlib,json,unicodedata
from collections import Counter
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from experiment_paths import *

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def classify(s,special=False):
 if special:return 'special'
 if '\ufffd' in s:return 'byte_fragment'
 if s and s.isspace():return 'newline_only' if '\n' in s or '\r' in s else 'whitespace_only'
 stripped=s.strip()
 if stripped and all(unicodedata.category(c).startswith('P') for c in stripped):return 'punctuation'
 if stripped and all(c.isnumeric() for c in stripped):return 'numbers'
 if any(c.isalpha() for c in stripped):return 'contains_letters'
 return 'symbols_or_other'
def main():
 out=data_root();out.mkdir(parents=True,exist_ok=True)
 if (out/'gold_audit_summary.json').exists():raise FileExistsError('Audit already exists')
 ds=load_from_disk(str(require_path('prepared_dataset')));assert len(ds)==6000
 token=AutoTokenizer.from_pretrained(require_path('base_model'),local_files_only=True)
 totals=Counter();token_counts=Counter();sample_rows=[];contexts=[];context_counts=Counter();selected=[];fp=[];thinks=0;ends=0
 special=set(token.all_special_ids)
 for row in ds:
  selected.append(row['source_index']);counts=Counter();pos=row['term_positions'];ys=row['term_targets']
  if pos:thinks+=1;assert ys.count(1)==1
  assert len(pos)==len(set(pos))==len(ys)
  for p,y in zip(pos,ys):
   assert 0<=p<len(row['labels'])-1
   gold=row['labels'][p+1]
   assert gold>=0 and gold==row['input_ids'][p+1] and (gold==151668)==bool(y)
   if y:ends+=1;continue
   s=token.decode([gold],skip_special_tokens=False,clean_up_tokenization_spaces=False)
   category=classify(s,gold in special);totals[category]+=1;counts[category]+=1;token_counts[gold]+=1
   totals['leading_whitespace_attached']+=int(bool(s) and s[0].isspace() and not s.isspace())
   if context_counts[category]<8:
    context_counts[category]+=1
    contexts.append(dict(source_index=row['source_index'],position=p,gold_token=gold,text=s,category=category,
     context=token.decode(row['input_ids'][max(0,p-15):p+15],skip_special_tokens=False,clean_up_tokenization_spaces=False)))
  sample_rows.append(dict(source_index=row['source_index'],counts=dict(counts),continue_count=sum(counts.values())))
  digest=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
  fp.append((digest(row['input_ids']),digest(row['labels'])))
 assert thinks==ends==3183 and sum(token_counts.values())==219572
 legacy_manifest=json.loads((require_path('prepared_dataset').parent/'preparation_manifest.json').read_text())
 actual=hashlib.sha256(json.dumps(fp,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
 assert actual==legacy_manifest['dataset_fingerprint']
 total=sum(token_counts.values());exclusive={k:totals[k] for k in ('special','byte_fragment','newline_only','whitespace_only','punctuation','numbers','contains_letters','symbols_or_other')}
 result={'rows':6000,'think':thinks,'no_think':6000-thinks,'continue_boundaries':total,'ends':ends,
   'counts':exclusive,'fractions':{k:v/total for k,v in exclusive.items()},
   'leading_whitespace_attached':totals['leading_whitespace_attached'],'cache_fingerprint':actual,
   'selected_order_sha256':digest(selected),'interpretation':'format tokens are not semantic content; no boundary changes'}
 (out/'gold_audit_summary.json').write_text(json.dumps(result,indent=2))
 for name,rows in [('gold_per_sample',sample_rows),('gold_contexts',contexts),('gold_common_tokens',
  [dict(id=i,count=n,text=token.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)) for i,n in token_counts.most_common()])]:
  with (out/(name+'.jsonl')).open('w') as f:
   for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 baseline={}
 refs=require_path('legacy_results_root')
 for arm in ['S0','S1','S2']:
  pointer=refs/f'evaluation_{arm}.json';run=Path(json.loads(pointer.read_text())['run_dir'])
  rows=[json.loads(l) for l in (run/'predictions.jsonl').open()]
  assert len(rows)==len({r['uid'] for r in rows})==666
  manifest=json.loads((run/'run_manifest.json').read_text())
  baseline[arm]={'run_dir':str(run),'predictions_sha256':sha(run/'predictions.jsonl'),
    'manifest_sha256':sha(run/'run_manifest.json'),'sample_order_sha256':manifest['protocol']['sample_order_sha256'],
    'prompt_sequence_sha256':manifest['protocol']['prompt_sequence_sha256'],'correct':sum(r['answer_ability_is_correct'] for r in rows)}
 results=evaluation_results_root();results.mkdir(parents=True,exist_ok=True)
 (results/'reused_baselines.json').write_text(json.dumps(baseline,indent=2))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
