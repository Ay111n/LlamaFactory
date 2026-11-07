import json,shutil
from pathlib import Path
B=Path('/home/nlp/data/lyl');ID='mixed_sft_g2_full_preflight_20261026'
R=B/'LLaMA-Factory/sft_experiments'/ID
if R.exists():raise FileExistsError(R)
R.mkdir(parents=True)
p={'framework':str(B/'LLaMA-Factory'),'engineering':str(R),
 'data':str(B/'finetuning/sft/processed_data'/ID),'data_scripts':str(B/'finetuning/sft/scripts'/ID),
 'analysis':str(B/'benchmark/eval/analysis'/ID),'model_output':str(B/'finetuning/model/sft_experiments/mixed_sft_g2_full_49800_20261026'),
 'base':'/home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2',
 'yaml':str(B/'LLaMA-Factory/examples/train_lora/qwen3_mixed_sft_49800_lora_v1.yaml'),
 'train':str(B/'LLaMA-Factory/data/mixed_sft_final_49800_sharegpt.jsonl'),
 'provenance':str(B/'finetuning/sft/processed_data/mixed_sft_dynamic_termination_20261022/source_provenance.jsonl'),
 'small_cache':str(B/'finetuning/sft/processed_data/mixed_sft_dynamic_termination_20261022/prepared/dataset'),
 'dev':str(B/'benchmark/eval/tibetan_formateval_validation/trilingual_aligned/tibetan/mcqa_4way.jsonl'),
 'test':str(B/'benchmark/eval/tibetan_formateval_test/mcqa_4way.jsonl'),
 'training_python':'/opt/anaconda3/envs/training/bin/python',
 'historical_checkpoint_audit':False,'checkpoint_selection':'manual after listing retained checkpoints on 666 DEV',
 'formal_training_authorized':False}
for k in ['data','data_scripts','analysis']:
 Path(p[k]).mkdir(parents=True,exist_ok=False)
(R/'paths.json').write_text(json.dumps(p,indent=2))
S=Path(__file__).parent
for name in ['test_ce_path.py','probe_full.py','prepare_config.py']:
 if (S/name).exists():shutil.copy2(S/name,R/name)
if (S/'audit_data.py').exists():shutil.copy2(S/'audit_data.py',Path(p['data_scripts'])/'audit_data.py')
print(R)
