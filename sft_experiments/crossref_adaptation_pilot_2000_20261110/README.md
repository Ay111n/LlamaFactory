# Matched bilingual and cross-reference SFT pilot

This pilot uses the frozen 2,000-pair subset prepared under finetuning, excluding the prior 200 diagnostic UIDs. Each arm has 4,000 training records with exactly the same order and response targets. The reference arm replaces 1,000 Tibetan prompts with the existing v2 English-reference wrapper. It adds no generated response or answer-label field.

The native LlamaFactory Qwen3 template runs with enable_thinking=false. Only the existing response and native termination tokens receive loss; the empty thought prefix is prompt-masked. All native target sequences must be identical between arms and untruncated before launch. Both arms start from the same supplied sft-new2 path, use rank-32 LoRA, learning rate 5e-5, effective batch 32, two epochs and seed 3407. Reference inputs cost additional prefill computation; this is a matched-supervision comparison, not equal FLOPs.

Run `pilot.py preflight --model /path/to/sft-new2` with the training environment. Then run `pilot.py launch --gpus 6 7`. Launch rechecks the data/cache/source hashes, empty output directories and idle selected GPUs. The supervisor records PIDs, exit codes, native trainer state and adapter hashes. `pilot.py status` reads progress without starting work.

Training artifacts and resolved machine paths stay in ignored runtime/, weights/checkpoints stay under finetuning/model/sft/. A completed training run is not an effectiveness result. Subsequent checks must use the held-out training-side diagnostic and Tibetan-only 666DEV; no TEST tuning or reference response supplied to the deployed student.
