"""Offline delivery audit only: no model loading, inference or training."""
import csv,hashlib,json,math,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BENCHMARK=ROOT.parents[2]
RESULTS=BENCHMARK/'eval/results/mixed_sft_gold_margin_20261023'
TRAINING=BENCHMARK.parent/'LlamaFactory/sft_experiments/mixed_sft_gold_margin_20261023'
RECORDS=TRAINING/'delivery_20261025'
ARMS=['S0','G1','G2','G3','G1-M','G3-M','S1','S2']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return [json.loads(line) for line in path.open(encoding='utf-8')]
def exact_mcnemar(gain,loss):
 n=gain+loss
 return min(1.,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.

def main():
 data={};evidence={'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'offline artifact audit; no inference or training','arms':{},'file_hashes':{}}
 metrics={r['arm']:r for r in csv.DictReader((ROOT/'metrics.csv').open(encoding='utf-8-sig'))}
 sources=json.loads((RESULTS/'reused_baseline_snapshots/sources.json').read_text())
 funcs={'closed':lambda r:bool(r['thinking_closed']),'thinking_truncated':lambda r:bool(r['is_truncated'] and not r['thinking_closed']),'correct':lambda r:bool(r['answer_ability_is_correct']),'extractable':lambda r:bool(r['answer_ability'])}
 for arm in ARMS:
  if arm in sources:
   run=RESULTS/'reused_baseline_snapshots'/arm
   for name,h in sources[arm]['files'].items():assert sha(run/name)==h
  else:
   remote=json.loads((RESULTS/f'evaluation_{arm}.json').read_text())['run_dir']
   relative=remote.split('/mixed_sft_gold_margin_20261023/',1)[1]
   run=RESULTS/relative
  rows=read(run/'predictions.jsonl');data[arm]=rows
  assert len(rows)==len({r['uid'] for r in rows})==666
  signature=[(r['uid'],r['prompt_sha256']) for r in rows]
  if arm=='S0':reference=signature
  assert signature==reference
  manifest=json.loads((run/'run_manifest.json').read_text())['protocol']
  assert manifest['data_sha256']=='1604b77b1ea245f7ab44b144f0589007242500bc6e87d5f805a7c6a651270b2d'
  assert manifest['prompt_sequence_sha256']=='923302a5b84a2dd6bb1c41b58e3e6fec06d2ac427adcd24efda14c016166c0cc'
  assert manifest['generation']['temperature']==0 and manifest['generation']['max_tokens']==4096 and manifest['generation']['seed']==42
  assert '</think>' not in manifest['generation']['stop']
  for key,fn in funcs.items():assert sum(fn(r) for r in rows)==int(metrics[arm][key])
  assert all(bool(r['is_truncated'])==(r['finish_reason']=='length') for r in rows)
  for r in rows:
   ids=r['generated_token_ids']
   assert r['thinking_tokens_raw']==(ids.index(151668) if 151668 in ids else len(ids))
  evidence['arms'][arm]={'run':str(run),'prediction_sha256':sha(run/'predictions.jsonl'),'records':666,'metrics_verified':True,'protocol_verified':True}
  if arm not in sources:
   pre=json.loads((ROOT/f'protocol_preflight_{arm}.json').read_text())
   assert pre['all_prompt_hashes_match'] and pre['all_prompt_token_ids_match'] and pre['no_external_thinking_stop']
   d=RECORDS/'run_records'/arm
   assert json.loads((d/'completed.json').read_text())['global_step']==1125
   steps=read(d/'termination_steps.jsonl');probes=read(d/'gold_gradient_probes.jsonl')
   assert [r['step'] for r in steps]==list(range(1,1126))
   assert len(probes)==96
   if arm.endswith('-M'):
    error=max(r['match_relative_error'] for r in steps);assert error<2e-5
    evidence['arms'][arm]['maximum_match_relative_error']=error
   evidence['arms'][arm]['steps']=1125
 for pair in json.loads((ROOT/'paired_statistics.json').read_text()):
  x,y=data[pair['baseline']],data[pair['candidate']]
  for key,fn in funcs.items():
   gain=sum(not fn(a) and fn(b) for a,b in zip(x,y));loss=sum(fn(a) and not fn(b) for a,b in zip(x,y))
   assert pair[key]['gain']==gain and pair[key]['loss']==loss and pair[key]['delta_count']==gain-loss
   assert math.isclose(pair[key]['mcnemar_exact_p'],exact_mcnemar(gain,loss),abs_tol=1e-12)
 manifest=json.loads((RECORDS/'delivery_manifest_20261025.json').read_text())
 for name,h in manifest['files'].items():assert sha(TRAINING/name)==h
 evidence['training_record_files_verified']=len(manifest['files'])
 evidence['candidate_count_gate']={arm:((int(metrics['S0']['thinking_truncated'])-int(metrics[arm]['thinking_truncated'])>=10) and (int(metrics[arm]['correct'])-int(metrics['S0']['correct'])>=-3)) for arm in ARMS if arm not in sources}
 evidence['primary_holm_min']=min(p[k]['holm_p'] for p in json.loads((ROOT/'paired_statistics.json').read_text()) if p['primary'] for k in funcs)
 for path in ROOT.iterdir():
  if path.is_file() and path.name in ['report.md','metrics.csv','paired_statistics.json','paired_transitions.csv','S0_fixed_cohorts.csv','gradient_trajectory_summary.json','transition_cases.jsonl']:evidence['file_hashes'][path.name]=sha(path)
 (ROOT/'delivery_audit_20261025.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'verified_arms':len(data),'records_per_arm':666,'training_record_files':evidence['training_record_files_verified'],'primary_holm_min':evidence['primary_holm_min'],'candidate_count_gate':evidence['candidate_count_gate']},ensure_ascii=False))

if __name__=='__main__':main()
