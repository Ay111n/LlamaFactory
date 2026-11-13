"""Read-only verification of the exact teacher/base weight files and tokenizer."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from prepare_data import sha, write, BASE, SOURCE, EXPECTED_SOURCE

ROOT = Path(__file__).resolve().parent
REFERENCE = Path("/home/nlp/data/lyl/finetuning/model/sft/en_cot_49800/merged/checkpoint-9000")


def main():
    manifest = json.loads((REFERENCE / "merge_manifest.json").read_text())
    records = []
    for root, expected in ((Path(BASE), manifest["base_model_files_sha256"]), (REFERENCE, manifest["merged_output_files_sha256"])):
        for name, wanted in expected.items():
            actual = sha(root / name)
            records.append({"path": str(root / name), "sha256": actual, "expected": wanted, "passed": actual == wanted})
            assert actual == wanted, str(root / name)
    student = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(REFERENCE, local_files_only=True)
    assert student.get_vocab() == teacher.get_vocab()
    assert student.backend_tokenizer.to_str() == teacher.backend_tokenizer.to_str()
    assert sha(SOURCE) == EXPECTED_SOURCE
    write(ROOT / "provenance.json", {"passed": True, "records": records, "vocabulary": len(student),
        "tokenizer_behavior_identical": True, "source_sha256": EXPECTED_SOURCE, "merge_manifest_sha256": sha(REFERENCE / "merge_manifest.json")})
    print("PROVENANCE_PASSED", len(records), flush=True)


if __name__ == "__main__":
    main()
