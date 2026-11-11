Complete teacher supervision cache for the original 10,000 English-Tibetan pairs.
Reuse 2,000 verified pilot tensors without editing them; generate 8,000 missing items on GPU 0.
Use the same frozen full crossref teacher, prompt v2, OFF, BF16 SDPA, full response and native terminator/newline.
Cache hidden states plus frozen head, equivalent to full vocabulary supervision without top-k approximation.
All source responses are unchanged. Preserve diagnostic holdout labels and the previously flagged malformed UID.
This command only prepares the cache; it does not launch full student training.
Run with inference Python, CUDA_VISIBLE_DEVICES=0, TMPDIR=/home/nlp/data/lyl/tmp_q35.
Runtime and tensors are excluded from Git. Cache has resumable per-item hash journal and final full hash validation.
