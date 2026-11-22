# Qwen3.5-9B English CoT, Tibetan answer smoke run

The training input is the frozen 12,000-row English-CoT view in
finetuning/sft/mixed_sft/final_data/bo_cot_sft_12000_three_variants_20261122.
Training reads the encoded data in this experiment's runtime directory. The
inherited crossref dataset name in the framework configuration is unused when
tokenized_path is set.

The run uses a single L20, BF16 LoRA (rank 32), cutoff 10,240, microbatch 1,
four gradient-accumulation microbatches, and eight updates. User-prompt tokens
are masked from loss. It saves an adapter-only checkpoint.

After training exits, infer.py loads checkpoint-8. Tibetan-only prompts are
decoded greedily with thinking enabled. It saves reasoning and final answer
separately and records whether the closing thinking tag was generated.
The --extended run checks one open response and one DEV question at a larger
generation budget. This is a language smoke test.
