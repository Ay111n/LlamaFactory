Full 10,000-pair matched bilingual CE vs CE+KD students.
Both start fresh from original sft-new2, same LoRA seed/data order and pilot objective.
Two epochs / 20,000 en-bo views / batch 1 * accumulation 32 = 1,250 updates.
All raw pairs retained per current user instruction, including 40 previous diagnostic items: no independent student diagnostic claims on these items.
Known malformed Math_no_think_0010880 is flagged in data manifest, not silently removed.
Reuse complete 10k frozen teacher cache; no teacher generation or online teacher training.
Native OFF, response mean CE, BO-only 0.2 full-vocab KL; same validated pilot trainer and loss implementation.
GPU3 KD and GPU0 baseline share GPUs after >=28GiB free check; allocator capped at24GiB.
Longest-example KD smoke must pass before both formal jobs start.
Only one final recoverable checkpoint per arm, two total in this run.
Existing pilot GPU5 exclusive evaluation remains untouched.
Cache integrity: reuse the completed 10k hash audit, verify its manifest, and verify each accessed tensor hash in the trainer; avoid an unnecessary extra 33GiB read before launch.
