"""Record immutable inputs and code/environment identity before training."""
from experiment_paths import experiment_root,benchmark_root,data_root,model_root,paths,require_path,script_path,stage_log,evaluation_analysis_root,paths_file

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
from experiment_paths import benchmark_root,evaluation_results_root
BENCHMARK=benchmark_root()
LF=require_path('framework_root')
BASE=require_path('base_model')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

files=list(BASE.glob('*.safetensors'))+list(BASE.glob('*.json'))+list(BASE.glob('*.jinja'))
files += [require_path('source_dataset')/'mixed_sft_final_49800_sharegpt.jsonl',paths_file()]
files += list(ROOT.glob('*.py'))
files += [script_path(name) for name in ('prepare_dataset.py','prepare_provenance.py','export_arm.py','export_arm_v2.py','verify_export_v2.py','evaluate_arm.py','analyze_results.py','partial_dev_review.py')]
files += list((evaluation_results_root()/'historical_b0').glob('*.json*'))
files += [BENCHMARK/'eval'/'tibetan_formateval_validation/trilingual_aligned/tibetan/mcqa_4way.jsonl',
          BENCHMARK/'eval'/'scripts/eval_multilingual_mcqa_vllm.py',
          BENCHMARK/'eval'/'scripts/mcqa_answer_extraction_multilingual.py',
          BENCHMARK/'eval'/'scripts/multilingual_mcqa_data_registry.json']
files += [LF/'src/llamafactory/hparams/finetuning_args.py',LF/'src/llamafactory/train/sft/workflow.py']
files += list((LF/'src/llamafactory/train/sft').glob('termination_*.py'))
manifest={'python':sys.version,'executable':sys.executable,
    'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate','peft','spacy','datasets','flash-attn']},
    'git_head':subprocess.check_output(['git','-C',str(LF),'rev-parse','HEAD'],text=True).strip(),
    'git_branch':subprocess.check_output(['git','-C',str(LF),'branch','--show-current'],text=True).strip(),
    'files':{str(p):sha(p) for p in files}}
(ROOT/'input_code_environment_fingerprints.json').write_text(json.dumps(manifest,indent=2))
print('Frozen input/code/environment files:',len(files))
