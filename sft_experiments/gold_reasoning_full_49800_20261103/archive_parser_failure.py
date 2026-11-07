"""Archive only this run's zero-update parser failure, preserving all evidence."""
import json
import os
import shutil
from common import *

failure = json.loads((ROOT / "failure.json").read_text())
assert "save_safetensors" in failure["error"]
assert not (ROOT / "steps.jsonl").exists() and not (ROOT / "initialization.json").exists()
assert not MODEL_ROOT.exists() or not any(MODEL_ROOT.iterdir())
process = json.loads((ROOT / "process.json").read_text())
try:
    os.kill(process["worker_pid"], 0)
except ProcessLookupError:
    pass
else:
    raise RuntimeError("Previous worker is still alive; refusing archive/relaunch")
archive = ROOT / "startup_attempts/parser_failure_before_model_load"
archive.mkdir(parents=True, exist_ok=False)
(archive / "source_snapshot").mkdir()
for path in ROOT.glob("*.py"):
    shutil.copy2(path, archive / "source_snapshot" / path.name)
for name in ("prepared.json", "input_manifest.json", "source_manifest.json", "protocol.json", "config_difference_from_preflight.json", "full_training.yaml",
             "preparation.log", "supervisor.log", "supervisor.json", "process.json", "process_exit.json", "failure.json", "launch_inventory.json", "training.log"):
    path = ROOT / name
    if path.exists():
        path.rename(archive / name)
write(archive / "archive_record.json", {"optimizer_updates": 0, "model_loaded": False,
    "reason": "Transformers 5.2 parser no longer accepts explicit save_safetensors; remove this redundant field.",
    "archived_at_utc": utc_now()})
print("ARCHIVED_ZERO_UPDATE_PARSER_FAILURE", archive)
