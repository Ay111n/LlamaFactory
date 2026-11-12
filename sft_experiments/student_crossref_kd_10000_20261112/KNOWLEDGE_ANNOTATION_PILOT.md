# Knowledge-phrase annotation pilot

This pilot uses the existing merged reference-adapted teacher on the same 32 paired training responses used in the earlier gradient/fragment diagnostics. It does not train, generate alternative answers, extract option-based supervision, or change model/data artifacts.

The teacher receives question context, the complete original English response and complete original Tibetan response. It may extract up to three short knowledge-bearing Tibetan passages with verbatim English evidence. Types cover entity relations, causes/comparisons, quantities/formulas and conditions/negation. The prompt requests preservation of meaning-changing qualifiers and exclusion of organization/formatting-only fragments. An uncertain alignment label is available.

Run `annotate_knowledge_spans.py both` in the verified inference environment. Generation uses native OFF, temperature 0, seed 3407, max 1536 new tokens, batches of four, BF16 and the installed vLLM. JSON-schema constrained decoding assists format only; it does NOT restrict output strings to substrings of the inputs. No automatic repair, retries or fuzzy alignment are used.

Structural validation distinguishes parseability, exact Tibetan extraction, exact English evidence, and unique localization. Exact quotes are mapped to character/token positions without modifying the text. Semantic review is separate: exact quotes can still be mismatched in meaning, incomplete, stylistic or overbroad. A reference-derived checklist with one core proposition per input and additional critical qualifier checks is written before inspecting annotations and is not provided to the teacher. Review judgments are exploratory assistant assessments, not independent Tibetan-expert gold labels or verification of factual claims in the source answers.

Runtime artifacts include source pairs, protocol and hashes, raw predictions/finish reasons, structural validation, the separate review checklist and semantic audit. They remain outside Git. The source pairs and trained teacher are unchanged.
