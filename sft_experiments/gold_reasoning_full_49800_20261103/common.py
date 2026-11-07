"""Paths and small IO helpers for the authorized full training run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRAMEWORK = ROOT.parent.parent
PREFLIGHT = ROOT.parent / "gold_reasoning_preflight_20261103"
DATA = Path("/home/nlp/data/lyl/finetuning/sft/processed_data/gold_reasoning_preflight_20261103")
SOURCE = FRAMEWORK / "data/mixed_sft_final_49800_sharegpt.jsonl"
BASE = "/home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2"
REFERENCE = "/home/nlp/data/lyl/model/CE/merged/checkpoint-9000"
MODEL_ROOT = Path("/home/nlp/data/lyl/finetuning/model/sft_experiments/gold_reasoning_full_49800_20261103/main")
PYTHON = "/opt/anaconda3/envs/training/bin/python"
EXPECTED_STEPS = 9339
SAVE_STEPS = 1500
EXPECTED_CHECKPOINTS = [1500, 3000, 4500, 6000, 7500, 9000, 9339]
GPU_ROLES = {
    "student": {"index": 1, "uuid": "GPU-5fc7820d-0470-f42a-a5c7-3f3cbfb2df88"},
    "teacher": {"index": 2, "uuid": "GPU-c98046c2-90b5-a959-9e88-f09a146ee29e"},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
