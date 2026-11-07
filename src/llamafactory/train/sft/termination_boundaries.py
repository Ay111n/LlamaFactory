"""Rule-detected English sentence/newline boundaries with token-offset audit."""

import re


def find_boundaries(text, nlp):
    """Return character offsets after punctuation/newlines outside fenced code."""
    code = [(m.start(), m.end()) for m in re.finditer(r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)", text)]
    def allowed(p):
        return not any(a < p <= b for a, b in code)
    candidates = {}
    for sent in nlp(text).sents:
        part = text[sent.start_char:sent.end_char].rstrip()
        if re.search(r'''[.!?]["'”’\)\]]*$''', part):
            p = sent.start_char + len(part)
            if allowed(p):
                candidates.setdefault(p, []).append("sentence_end")
    for m in re.finditer(r"\r?\n+", text):
        if text[:m.start()].strip() and allowed(m.end()):
            candidates.setdefault(m.end(), []).append("newline")
    return sorted(candidates.items())


def map_boundaries(rendered, ids, offsets, think_start, think_end, nlp, end_id):
    """Only align exact token ends, optionally consuming adjacent whitespace."""
    if len(ids) != len(offsets):
        raise ValueError("Token offset length mismatch")
    end_tokens = [i for i, (a, b) in enumerate(offsets) if a == think_end and ids[i] == end_id]
    if len(end_tokens) != 1:
        raise ValueError("True end must map exactly to a single token")
    true_pos = end_tokens[0] - 1
    positions, labels, audit = [], [], []
    seen = {true_pos}
    for offset, kinds in find_boundaries(rendered[think_start:think_end], nlp):
        char = think_start + offset
        limit = char
        while limit < think_end and rendered[limit].isspace():
            limit += 1
        eligible = [i for i, (a, b) in enumerate(offsets)
                    if a < b and char <= b <= limit and b <= think_end]
        pos = max(eligible) if eligible else None
        reason = ""
        if pos is None:
            reason = "not_exactly_token_alignable"
        elif pos in seen:
            reason = "true_end_collision" if pos == true_pos else "duplicate_token_position"
        else:
            seen.add(pos)
            positions.append(pos)
            labels.append(0)
        audit.append({"char_offset": char, "logit_position": pos, "types": kinds, "excluded": reason})
    positions.append(true_pos)
    labels.append(1)
    audit.append({"char_offset": think_end, "logit_position": true_pos, "types": ["true_end"], "excluded": ""})
    return positions, labels, audit
