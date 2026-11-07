"""Verify all original CE input IDs/labels; append aligned runtime views.

Frozen training/diagnostic sources are read-only. Derived inputs go to finetuning.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
LF = ROOT.parent.parent
BASE = "/home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2"
SOURCE = LF / "data/mixed_sft_final_49800_sharegpt.jsonl"
PREVIOUS = Path("/home/nlp/data/lyl/finetuning/sft/processed_data/mixed_sft_g2_full_preflight_20261026")
DATA = Path("/home/nlp/data/lyl/finetuning/sft/processed_data/gold_reasoning_preflight_20261103")
EXPECTED_SOURCE = "02d330dfa0b69ad63888d5a50d4ed25b3a135ed980f3498a61bb70c7d55f5b33"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA / "dataset").exists():
        raise FileExistsError("Do not replace prepared inputs")
    assert sha(SOURCE) == EXPECTED_SOURCE
    prior = load_from_disk(str(PREVIOUS / "dataset"))
    metadata = [json.loads(l) for l in (PREVIOUS / "source_metadata.jsonl").open()]
    raw = [json.loads(l) for l in SOURCE.open()]
    exclusions = json.loads((ROOT / "training_exclusion.json").read_text())
    assert exclusions["allow_gradient_updates"] is False and len(exclusions["records"]) == 1000
    heldout = {r["uid"] for r in exclusions["records"]}
    heldout_questions = {r["question_sha256"] for r in exclusions["records"]}
    assert len(prior) == len(metadata) == len(raw) == 49800
    assert len({r["uid"] for r in metadata}) == 49800
    assert not heldout.intersection(r["uid"] for r in metadata)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    counts, lengths, answer_lengths, zero_lengths, audits = Counter(), [], [], [], []
    start = time.time()

    def rows():
        for i, (original, meta) in enumerate(zip(raw, metadata)):
            old = prior[i]
            messages = original["messages"]
            assert len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant"
            question, content = messages[0]["content"], messages[1]["content"]
            assert hashlib.sha256(question.encode()).hexdigest() == meta["user_hash"]
            assert hashlib.sha256(question.encode()).hexdigest() not in heldout_questions
            assert content.startswith("<think>\n") and content.count("</think>") == 1
            split = content.index("</think>")
            think = bool(content[len("<think>"):split].strip())
            assert think == meta["nonempty_reasoning"]
            body_start = split + len("</think>\n\n")
            assert content[split:body_start] == "</think>\n\n"
            body = content[body_start:]
            assert body
            target = content + "<|im_end|>\n"
            encoded = tok(target, add_special_tokens=False, return_offsets_mapping=True)
            assistant_start = next(j for j, x in enumerate(old["labels"]) if x != -100)
            prompt_ids = old["input_ids"][:assistant_start]
            expected_prompt = tok.encode("<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
            assert prompt_ids == expected_prompt, (i, "prompt")
            assert prompt_ids + encoded.input_ids == old["input_ids"], (i, "input_ids")
            assert [-100] * assistant_start + encoded.input_ids == old["labels"], (i, "labels")
            assert len(old["input_ids"]) <= 4096
            positions = [assistant_start + j for j, (a, b) in enumerate(encoded.offset_mapping) if b > body_start and a < len(content)]
            gold = [old["input_ids"][j] for j in positions]
            assert positions and positions == list(range(positions[0], positions[-1] + 1))
            zero_content = "<think>\n\n</think>\n\n" + body
            zero_encoded = tok(zero_content + "<|im_end|>\n", add_special_tokens=False, return_offsets_mapping=True)
            zero_start = len("<think>\n\n</think>\n\n")
            zero_positions = [assistant_start + j for j, (a, b) in enumerate(zero_encoded.offset_mapping) if b > zero_start and a < len(zero_content)]
            zero_ids = prompt_ids + zero_encoded.input_ids
            assert [zero_ids[j] for j in zero_positions] == gold, (i, "answer_alignment")
            counts["think" if think else "no_think"] += 1
            counts["ce_tokens"] += len(encoded.input_ids)
            counts["answer_tokens_think"] += len(gold) if think else 0
            counts["zero_tokens_think"] += len(zero_ids) if think else 0
            lengths.append(len(old["input_ids"]))
            answer_lengths.append(len(gold) if think else 0)
            zero_lengths.append(len(zero_ids) if think else 0)
            audits.append({"index": i, "uid": meta["uid"], "think": think, "length": lengths[-1], "answer_tokens": len(gold), "zero_length": len(zero_ids), "domain": meta["domain"]})
            yield {"input_ids": old["input_ids"], "attention_mask": old["attention_mask"], "labels": old["labels"],
                   "meta": {"index": i, "uid": meta["uid"], "think": think, "length": len(old["input_ids"]),
                            "positions": positions, "zero_positions": zero_positions, "zero_ids": zero_ids, "gold": gold}}
            if (i + 1) % 5000 == 0:
                print("PREPARE", i + 1, round(time.time() - start, 1), flush=True)

    dataset = Dataset.from_generator(rows, cache_dir=str(DATA / "build_cache"))
    dataset.save_to_disk(str(DATA / "dataset"))
    assert counts["think"] == 26419 and counts["no_think"] == 23381 and counts["ce_tokens"] == 57260173
    with (DATA / "audit.jsonl").open("w") as f:
        for r in audits:
            f.write(json.dumps(r) + "\n")
    manifest = {"rows": len(dataset), "counts": dict(counts), "source_sha256": sha(SOURCE), "metadata_sha256": sha(PREVIOUS / "source_metadata.jsonl"),
                "heldout_exclusion_sha256": sha(ROOT / "training_exclusion.json"), "input_ids_and_labels_exactly_match_original": True,
                "heldout_uid_and_exact_question_overlap": 0, "max_length": max(lengths), "mean_length": sum(lengths) / len(lengths),
                "sum_input_tokens": sum(lengths), "sum_think_answer_tokens": sum(answer_lengths), "sum_think_zero_tokens": sum(zero_lengths),
                "truncated": 0, "answer_alignment_passed": len(dataset), "dataset_fingerprint": dataset._fingerprint,
                "elapsed_seconds": time.time() - start}
    write(DATA / "manifest.json", manifest)
    write(ROOT / "data_manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
