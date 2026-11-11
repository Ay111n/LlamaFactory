Matched 2000-pair student pilot: bilingual CE vs cross-language reference KD.

Both students start from the original sft-new2, with fresh rank-32 alpha-64
all-linear LoRA, dropout .05, seed/data_seed 3407, 2 epochs, 4000 views,
batch 1 x accumulation 32, lr 5e-5, cosine, 5% warmup, BF16, FA2 and gradient
checkpointing. Student inputs are ordinary English or ordinary Tibetan QA.
Teacher is the frozen merged full crossref model. It receives English question
and reference response plus Tibetan question, using native OFF prompt v2.

Reuse exact prior pilot 2000 train_pairs and baseline.jsonl, with source hashes.
The known malformed Math_no_think_0010880 is absent. No answers/options are
extracted. Original complete responses are preserved. Empty think tags remain
in masked native OFF prompts. Supervised targets include native im_end/newline.

Loss: per-response CE averaged over all 4000 views; Tibetan views additionally
receive .2 forward KL, giving dataset objective .5 CE_en + .5 CE_bo + .1 KD_bo.
This response normalization is shared by both freshly trained arms. Historical
native token-weighted pilot baseline is not the primary matched control.
No confidence gates, probability projection, top-k truncation or extra traces.
Teacher hidden states and frozen output head are cached in BF16 exactly; full
vocabulary FP32 log-softmax/KL is reconstructed in chunks. No teacher backprop.

Same GPU 5 sequentially: CPU preflight -> frozen teacher cache (inference env)
-> discarded real update smoke (training env) -> baseline -> KD -> CPU merges
(inference env) -> matched OFF 666DEV (inference env) -> paired comparison.
GPU busy checks and lock prevent starting over occupied GPU. Other users' jobs
are not terminated. No TEST or preference optimization. No GPU 0 changes.
A final recoverable checkpoint per arm at step 250: at most two total, plus
final adapter exports. No intermediate checkpoints or automatic resume.

Numerical tests compare dense CE/KL values and hidden gradients; real smoke
checks native CE agreement, longest student inputs, finite training and actual
LoRA update. Both real arms are reset from identical seeds after smoke.
Runtime artifacts are excluded from Git; results belong under benchmark.

Gradient accumulation: explicit training_step applies exactly one division by the actual window size via Accelerator.backward. CPU Trainer integration checks cover GAS 1, 4, 32 and a partial window. Attempt 01 stopped and archived after detecting duplicate native scaling for this custom-loss path; both final arms restart from the original seed. Cached frozen teacher targets are unchanged.
