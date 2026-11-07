import hashlib,json,importlib.metadata,sys
from pathlib import Path
from experiment_paths import *
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 cal=json.loads((ROOT/'gold_calibration.json').read_text())
 report=diagnostics_root()/'initial_gradient_report.md'
 lines=['','## 继续/结束分项损失','',
  '|目标|继续项均值（含类别权重）|结束项均值（含类别权重）|','|---|---:|---:|']
 for key in ['gold','dynamic_gold','dynamic_ce']:
  cont=sum(r['parts'][key+'_continue'] for r in cal['rows'])/len(cal['rows'])
  end=sum(r['parts'][key+'_end'] for r in cal['rows'])/len(cal['rows'])
  lines.append(f'|{key}|{cont:.8g}|{end:.8g}|')
  if end>0 and cont/end<1e-4:
   lines.append(f'\n{key} 初始继续项相对结束项非常小；总 R 合格不代表继续监督足够强。后续单独报告其轨迹，不能将零增益直接归因于方法无效。\n')
 lines+=['','上述分项为损失统计，不冒充各类别梯度范数；参数梯度 R 是完整平衡辅助目标的整批结果。']
 report.write_text(report.read_text(encoding='utf-8')+'\n'.join(lines),encoding='utf-8')
 base=require_path('base_model');lf=require_path('framework_root')
 files=list(base.glob('*.safetensors'))+list(base.glob('*.json'))+list(base.glob('*.jinja'))
 files+=list(require_path('prepared_dataset').rglob('*'))
 files=[p for p in files if p.is_file()]
 files+=list(ROOT.glob('*.py'))+[paths_file(),ROOT/'protocol.json']
 files+=[require_path('legacy_experiment_root')/'run_experiment.py',require_path('legacy_analysis_root')/'analyze_results.py']
 files+=list((lf/'src/llamafactory/train/sft').glob('*termination*.py'))
 files += [lf/'src/llamafactory/hparams/finetuning_args.py',lf/'src/llamafactory/train/sft/workflow.py']
 files += [script_path(n) for n in ['evaluate_arm.py','export_arm_v2.py','verify_export_v2.py','analyze_results.py']]
 files += [benchmark_root()/'eval/scripts'/n for n in ['eval_multilingual_mcqa_vllm.py','mcqa_answer_extraction_multilingual.py','multilingual_mcqa_data_registry.json']]
 files += [benchmark_root()/'eval/tibetan_formateval_validation/trilingual_aligned/tibetan/mcqa_4way.jsonl']
 evidence={'files':{str(p):sha(p) for p in files},'python':sys.version,'executable':sys.executable,
  'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate','peft','spacy','datasets','flash-attn']}}
 if (ROOT/'frozen_inputs.json').exists():raise FileExistsError('Frozen manifest already exists')
 (ROOT/'frozen_inputs.json').write_text(json.dumps(evidence,indent=2))
 print('FROZEN',len(files),flush=True)
if __name__=='__main__':main()
