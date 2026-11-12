# Fixed-batch gradient diagnostic

`diagnose_gradients.py` inspects the two completed adapters without creating an optimizer, calling Trainer, updating parameters, or saving model weights. It reuses the original tokenized training examples, frozen teacher cache, full-vocabulary forward KL and response-mean losses.

Select 32 Tibetan training views before evaluating either model: eight deterministic hash-selected records from each input-length quartile. The UID list and input hashes are private runtime artifacts. This is an endpoint diagnostic on a small training batch, not held-out accuracy evaluation or a reconstruction of historical training gradients.

Run in the verified `training` environment on an available GPU:

```sh
python diagnose_gradients.py --arm prepare
python diagnose_gradients.py --arm both
```

`both` launches the baseline and KD measurements sequentially in separate processes. It refuses to overwrite a completed arm summary. Model parameters are loaded from the original base plus each final adapter, preserving the LoRA parameterization used in training. Dropout is disabled to remove stochastic differences; gradient checkpointing remains active. The frozen teacher uses the same archived gold response prefix and native OFF tokenization, including EOS and newline.

For each response, compute CE and KL and backpropagate them separately. Report raw KL gradients and the actual weighted contribution `0.2 * grad(KL)`. Aggregate gradients by averaging response gradients, not by averaging their norms. Positive cosine indicates locally compatible descent directions; negative cosine indicates a first-order conflict. Cosine and projection refer to raw LoRA gradients before clipping or Adam preconditioning. The batch contains Tibetan views only; it does not measure interactions with English CE in mixed training batches.

Assertions check selected tensor hashes, finite losses/gradients, trainable parameter count, unchanged parameter version counters and trainable hashes, and unchanged on-disk adapter hashes. Runtime outputs include the frozen protocol, per-response training diagnostics, aggregate endpoint summaries and execution log. They are not committed to Git.
