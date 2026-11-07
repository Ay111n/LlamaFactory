import json,shutil
from pathlib import Path
B=Path('/home/nlp/data/lyl');ID='mixed_sft_g2_full_49800_20261026';R=B/'LLaMA-Factory/sft_experiments'/ID
if R.exists():raise FileExistsError(R)
R.mkdir(parents=True);S=Path(__file__).parent
p={'framework_root':str(B/'LLaMA-Factory'),'benchmark_root':str(B/'benchmark'),
 'data_root':str(B/'finetuning/sft/processed_data'/ID),'model_root':str(B/'finetuning/model/sft_experiments'/ID),
 'results_root':str(B/'benchmark/eval/results'/ID),'analysis_root':str(B/'benchmark/eval/analysis'/ID),
 'legacy_experiment_root':str(B/'LLaMA-Factory/sft_experiments/mixed_sft_dynamic_termination_20261022'),
 'legacy_model_scripts':str(B/'finetuning/scripts/model/mixed_sft_dynamic_termination_20261022'),
 'legacy_analysis_root':str(B/'benchmark/eval/analysis/mixed_sft_dynamic_termination_20261022'),
 'gold_experiment_root':str(B/'LLaMA-Factory/sft_experiments/mixed_sft_gold_margin_20261023'),
 'prepared_dataset':str(B/'finetuning/sft/processed_data/mixed_sft_g2_full_preflight_20261026/dataset'),
 'prepared_manifest':str(B/'finetuning/sft/processed_data/mixed_sft_g2_full_preflight_20261026/preparation_manifest.json'),
 'sample_audit':str(B/'finetuning/sft/processed_data/mixed_sft_g2_full_preflight_20261026/sample_audit.jsonl'),
 'preflight_root':str(B/'LLaMA-Factory/sft_experiments/mixed_sft_g2_full_preflight_20261026'),
 'original_yaml':str(B/'LLaMA-Factory/examples/train_lora/qwen3_mixed_sft_49800_lora_v1.yaml'),
 'base_model':'/home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2',
 'historical_model':str(B/'model/qwen3-8bbo-sft-new2-mixed-checkpoint6000-merged'),
 'training_python':'/opt/anaconda3/envs/training/bin/python','inference_python':'/opt/anaconda3/envs/inference/bin/python'}
for k in ['data_root','model_root','results_root','analysis_root']:Path(p[k]).mkdir(parents=True,exist_ok=False)
(R/'paths.json').write_text(json.dumps(p,indent=2))
shutil.copy2(Path(p['gold_experiment_root'])/'experiment_paths.py',R/'experiment_paths.py')
for name in ['full_g2_trainer.py','test_full_g2.py','run_full_g2.py','pipeline_full_g2.py']:
 if (S/name).exists():shutil.copy2(S/name,R/name)
for name in ['analyze_full_g2.py']:
 if (S/name).exists():shutil.copy2(S/name,Path(p['analysis_root'])/name)
model_scripts=B/'finetuning/scripts/model'/ID;model_scripts.mkdir(parents=True,exist_ok=False)
if (S/'export_full_g2.py').exists():shutil.copy2(S/'export_full_g2.py',model_scripts/'export_full_g2.py')
# Evaluation is shared through experiment_paths.script_path('evaluate_arm.py'); no executable copy.
shutil.copytree(B/'benchmark/eval/results/mixed_sft_gold_margin_20261023/historical_b0',Path(p['results_root'])/'historical_b0')
protocol={'id':ID,'arm':'Full-G2','lambda':.0012,'seed':3407,'data_seed':3407,'dataset_count':49800,'epochs':3,'steps':9339,
 'required_checkpoints':[1500,3000,4500,6000,7500,9000],'manual_selection':True,'automatic_best_selection':False,'TEST_used':False,
 'retrain_standard_CE':False,'other_full_experiments':False,'diagnostic_steps':[1500,3000,4500,6000,7500,9000,9339],
 'native_backward':'CE and original G2 auxiliary jointly handled by stock LF/HF training_step + Accelerator.backward',
 'historical_runtime_uncertainty':'strict equality to the recoverable CURRENT native path; exact historical environment not proved',
 'real_CE_test_rtol':2e-4,'real_CE_test_atol':2e-6,'initial_R_p90_gate':[.001,1.0]}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2))
print(R)
