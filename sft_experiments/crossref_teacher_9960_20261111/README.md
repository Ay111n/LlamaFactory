# Expanded candidate teacher

Train only crossref from the original sft-new2 model, not the pilot adapter.
9960 unique paired examples; 19920 views: English ordinary 9960, Tibetan ordinary 4980, Tibetan reference 4980.
The 40 reviewed diagnostic UIDs and normalized duplicate questions are excluded. The other 160 old probe UIDs return to training.
The exact pilot training_recipe.json is reused: single GPU LoRA, batch 1, accumulation 32, 2 epochs, lr 5e-5, no thinking, response-only CE.
Expected optimizer steps: ceil(19920/32)*2 = 1246. No distillation, merge, evaluation, or baseline run is scheduled.

Preflight uses the training interpreter, verifies hashes, native tokenization, response masks and no truncation. Queue waits for a completely idle GPU; it never stops other jobs. Queue state and failures are saved under runtime/.

```sh
PYTHONPATH=src /opt/anaconda3/envs/training/bin/python sft_experiments/crossref_teacher_9960_20261111/teacher.py preflight --model /home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2
PYTHONPATH=src /opt/anaconda3/envs/training/bin/python sft_experiments/crossref_teacher_9960_20261111/teacher.py queue
```
