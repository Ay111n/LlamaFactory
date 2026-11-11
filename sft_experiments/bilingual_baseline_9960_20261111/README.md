# Full-data bilingual SFT control

Authorized 2026-11-11 on GPU 0. Uses the exact 9,960 training UIDs and 19,920 response targets/order of crossref_teacher_9960_20261111; excludes the same 40 diagnostic UIDs. All prompts are ordinary same-language QA: English 9,960 and Tibetan 9,960. No answer/option extraction or new responses.

Starts fresh rank-32 LoRA from sft-new2 with the identical teacher recipe: seed/data_seed 3407, effective batch 32, learning rate 5e-5, two epochs, 1,246 optimizer steps. Native response token sequences must exactly match the teacher preflight hash. Only the prompts differ; reference input prefill costs are not equal FLOPs.

Run baseline.py preflight --model /home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2 under the training Python with framework src on PYTHONPATH, then baseline.py queue. The supervisor accepts only idle GPU 0, rechecks input/cache/model hashes, refuses existing output directories and saves exit/completion status. It does not merge, evaluate, or distill automatically. Subsequent main evaluation uses the registered Tibetan OFF 666DEV protocol.

Runtime evidence is ignored under runtime/; model artifacts remain in finetuning/model/sft/bilingual_baseline_9960_20261111/baseline.

Stopped by explicit user request on 2026-11-11 after latest logged step 220/1246, before the first epoch checkpoint. Runtime user_stop.json records the reason; raw supervisor completion reports the interruption, while queue_status.json records stopped_by_user. Superseded by teacher_backbone_comparison_20261111. No completed model or evaluation is claimed.
