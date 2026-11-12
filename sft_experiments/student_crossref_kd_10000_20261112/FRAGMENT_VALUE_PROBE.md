# Fragment supervision-value probe

Read-only teacher-forced inference on the same 32 training views frozen for the gradient diagnostic. No generated answers, optimizer, backward, model updates, separate answer extraction, or DEV/TEST use.

Use the verified `inference` environment. The current environment has SDPA but no flash-attn; no dependency is installed. Baseline loads the original base plus the completed bilingual adapter, frozen. The teacher is the existing merged reference-adapted model. Reference-conditioned distributions reuse the verified hidden-state cache. A live recomputation of the first item checks numerical compatibility.

Compare reference-conditioned teacher (`plus`), identical instruction wrapper with both English fields empty (`empty`), the teacher with the student's ordinary Tibetan-only prompt (`natural`), and the ordinary bilingual SFT student (`baseline`). The natural condition checks whether an empty-reference instruction inflates the apparent reference advantage. The English reference includes both the question and full response, not an extracted answer.

Split original response body into consecutive 32-token windows; include the last partial window and exclude native terminator/newline. For each window compute R = mean(log q_plus(y) - log q_empty(y)), G = mean(log q_plus(y) - log p_baseline(y)). The predeclared primary gate is R > 0.1 and G > 0.1 nats/token. Report thresholds 0, 0.05, 0.1, 0.2 without optimizing on an evaluation score. Also require gain against `natural` > threshold for a robustness count.

For exploratory semantic review, hash-sample 24 selected and 12 nonselected windows, with six highest-scoring windows kept separately as illustrations. Review full paired responses and context; do not infer semantic error from token disagreement alone. Distinguish content in a window from the actual token positions receiving probability gains.

Review categories: substantive knowledge/reasoning content; wording/organization/format; mixed; uncertain. A substantive fragment contains an entity, quantity, formula or relation whose change would alter the claim. Classifying a fragment as substantive does not prove the teacher correct, the baseline wrong, or that the student cannot freely generate the answer. Describe whether high-gain positions concern such content or common lexical/grammatical choices. These are assistant judgments, not independent Tibetan expert gold labels.

Commands:

```sh
python probe_fragment_value.py prepare
python probe_fragment_value.py both
```

All private artifacts are under runtime/diagnostics/fragment_value_20261112 and excluded from Git.
