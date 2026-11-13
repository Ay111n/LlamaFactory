"""Fixed paths and atomic experiment records."""

# Resolve historical manifest paths in memory; preserve recorded IDs and hashes.
import sys as _cot_sys
from pathlib import Path as _CotPath
for _cot_parent in _CotPath(__file__).resolve().parents:
    if (_cot_parent / 'finetuning/scripts/cot_paths.py').is_file():
        _cot_sys.path.insert(0, str(_cot_parent / 'finetuning/scripts'))
        break
from cot_paths import resolve_record as _resolve_cot_record, recorded_hash_matches, recorded_hash_matches
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

NAME = "answer_only_49800_20261107"
THESIS = Path("/home/nlp/data/lyl")
FRAMEWORK = THESIS / "LLaMA-Factory"
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
DATA = THESIS / "finetuning/sft/mixed_sft/final_data" / NAME
SOURCE = DATA.parent / "en_cot_49800_20260919"
PROCESSED = THESIS / "finetuning/sft/mixed_sft/processed_data" / NAME
MODELS = THESIS / "finetuning/model/sft" / NAME
BASE = "/home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2"
BENCHMARK = THESIS / "benchmark"
ANALYSIS = BENCHMARK / "eval/analysis/answer_only_vs_en_cot_49800_20261107"
PYTHON = "/opt/anaconda3/envs/training/bin/python"
INFERENCE = "/opt/anaconda3/envs/inference/bin/python"
STEPS = [1500, 3000, 4500, 6000, 7500, 9000, 9339]
CHECKPOINT_FILES = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                    "scheduler.pt", "trainer_state.json", "rng_state.pth"]

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return _resolve_cot_record(json.loads(Path(path).read_text(encoding="utf-8")))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

def checkpoint(path, expected):
    path = Path(path)
    missing = [n for n in CHECKPOINT_FILES if not (path/n).is_file() or (path/n).stat().st_size == 0]
    if missing or read(path/"trainer_state.json")["global_step"] != expected:
        raise ValueError(f"Incomplete checkpoint {path}: {missing}")
    return {"step": expected, "path": str(path), "files": {n:(path/n).stat().st_size for n in CHECKPOINT_FILES},
            "adapter_sha256":sha(path/"adapter_model.safetensors"), "time":now()}
