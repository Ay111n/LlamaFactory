# Answer-only paired SFT control

Native CE on the original 49,800 questions and Tibetan answers. Only the English reasoning body is removed. Empty think markers remain supervised. The experiment is defined by the approved paired-data contract.

`prepare.py` validates and tokenizes both views with the native processor. `train.py --phase preflight` runs 100 technical-validation steps. `train.py --phase main` starts a fresh three-epoch student. `pipeline.py` manages exclusive idle-GPU use, validates checkpoints and delegates independent DEV evaluation to benchmark. Runtime records under `artifacts/` and model/data outputs are excluded from Git.

Run with the training Conda interpreter. Preserve seed 3407, effective batch 16, qwen3, cutoff 4096, LoRA rank 64 and the native CE Trainer. Evaluation compares seven ON checkpoints to matched CE steps and evaluates the selected ON-best checkpoint once in OFF mode. DEV selection is exploratory.
