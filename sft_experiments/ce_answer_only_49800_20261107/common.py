"""Fixed paths and atomic experiment records."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

NAME = "ce_answer_only_49800_20261107"
THESIS = Path("/home/nlp/data/lyl")
FRAMEWORK = THESIS / "LLaMA-Factory"
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
DATA = THESIS / "finetuning/sft/mixed_sft/final_data" / NAME
SOURCE = DATA.parent / "mixed_sft_sampling_20260919_base_tokenizer"
PROCESSED = THESIS / "finetuning/sft/mixed_sft/processed_data" / NAME
MODELS = THESIS / "finetuning/model/sft" / NAME
BASE = "/home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2"
BENCHMARK = THESIS / "benchmark"
ANALYSIS = BENCHMARK / "eval/analysis" / NAME
PYTHON = "/opt/anaconda3/envs/training/bin/python"
INFERENCE = "/opt/anaconda3/envs/inference/bin/python"
STEPS = [1500, 3000, 4500, 6000, 7500, 9000, 9339]
CHECKPOINT_FILES = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                    "scheduler.pt", "trainer_state.json", "rng_state.pth"]

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

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
