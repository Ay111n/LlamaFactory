# Offline target-probability weighting probe

`probe_target_weights.py` reads existing per-token probability arrays, source offsets and a matching tokenizer. It never imports the trainer, loads model weights, performs inference, creates an optimizer or changes parameters. It verifies the input SHA-256 manifest before and after analysis.

Example: `python probe_target_weights.py --input-dir <fragment-diagnostic> --output-dir <new-output-directory> --tokenizer <tokenizer.json>`

The proposed detached weight is `a=max(q-p,0)/(1-p)`, `w=(1+lambda*a)/mean(1+lambda*a)`. The primary descriptive scale is lambda=1; lambda=.2 and 2 are algebraic sensitivity summaries, not training runs or selected hyperparameters. The fixed diagnostic batch uses body-token normalization; it does not reproduce a future minibatch trajectory. The probability records correspond to the completed ordinary bilingual SFT endpoint, not the original student initialization.

Numerics use float64 calculations from cached log probabilities and expm1 to avoid cancellation near probability one. Cache saturation cannot be recovered. The script verifies the scalar identity `a*(1-p)=max(q-p,0)` and exact tokenizer target IDs and offsets.

High weights are distinct from high added supervision. The per-position CE logit-gradient L1 norm is `2*(1-p)`. Its positive change after normalization is `2*max(w-1,0)*(1-p)`. This is an analytic logit-space proxy, not a parameter-gradient norm or optimizer update estimate. Positive lambda changes its scale but not its ranking under the same batch normalization.

The review packet includes each response's maximum positive logit boost and a fixed-hash sample of 24 other positions from the highest global decile. Per-response maximum weights are retained separately to expose saturation. Semantic judgments require separate review of the actual weighted token, surrounding text, English source and cached alternatives. Historical window categories are only auxiliary and must not be treated as new token labels. Review labels and probability records remain runtime artifacts outside Git.
