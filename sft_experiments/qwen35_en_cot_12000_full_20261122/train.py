"""Launch one full English-CoT SFT epoch from the original Qwen3.5-9B."""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import TrainerCallback

here = Path(__file__).resolve().parent
root = here.parents[2]
runtime = here / 'runtime'
smoke = here.parent / 'qwen35_en_cot_12000_language_smoke_20261122' / 'runtime'
model_dir = root / 'finetuning/model/sft' / here.name
data = root / 'finetuning/sft/mixed_sft/final_data/bo_cot_sft_12000_three_variants_20261122'
source = data / 'en_cot_12000.jsonl'
with source.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
manifest = json.loads((data / 'manifest.json').read_text())
assert digest == manifest['outputs'][source.name]['sha256']
cfg = json.loads((smoke / 'config.json').read_text())
cache = Path(cfg['tokenized_path'])
assert len(load_from_disk(str(cache))['train']) == 12000
assert not model_dir.exists()
cfg.pop('max_steps', None)
cfg['dataset'] = 'en_cot_12000'
cfg.pop('dataset_dir', None)
cfg.update(output_dir=str(model_dir), num_train_epochs=1, gradient_accumulation_steps=16,
           cutoff_len=10240, save_steps=25, save_total_limit=2, save_only_model=False,
           enable_thinking=True, resume_from_checkpoint=None)
(runtime / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
(runtime / 'input_manifest.json').write_text(json.dumps({
    'dataset': str(source), 'dataset_sha256': digest, 'rows': 12000,
    'uid_order_sha256': manifest['uid_order_sha256'], 'source_checkpoint': cfg['model_name_or_path'],
    'encoded_cache': str(cache), 'expected_steps': 750,
}, indent=2) + '\n')
sys.path.insert(0, str(here.parent / 'qwen35_crossref_teacher_9960_20261111'))
from memory_ce import install
from llamafactory.train.tuner import run_exp


class Progress(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        assert state.max_steps == 750
        print('FULL_TRAIN_READY', state.max_steps, flush=True)

    def on_step_begin(self, *args, **kwargs):
        torch.cuda.synchronize()
        self.start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        with (runtime / 'steps.jsonl').open('a') as f:
            f.write(json.dumps({'step': state.global_step, 'seconds': time.monotonic() - self.start,
                                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}) + '\n')


install()
run_exp(cfg, callbacks=[Progress()])
