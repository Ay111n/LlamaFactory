"""Preserve the zero-update startup failure caused by a runtime fingerprint check."""
import os
import shutil
from common import *

failure = json.loads((ROOT / "failure.json").read_text())
assert "dataset._fingerprint" in failure["traceback"]
assert not (ROOT / "steps.jsonl").exists() and not (ROOT / "training_started.json").exists()
assert not MODEL_ROOT.exists() or not any(MODEL_ROOT.iterdir())
process = json.loads((ROOT / "process.json").read_text())
try:
    os.kill(process["worker_pid"], 0)
except ProcessLookupError:
    pass
else:
    raise RuntimeError("Previous worker is still alive; refusing archive/relaunch")
archive = ROOT / "startup_attempts/fingerprint_check_before_training"
archive.mkdir(parents=True, exist_ok=False)
(archive / "source_snapshot").mkdir()
for path in ROOT.glob("*.py"):
    shutil.copy2(path, archive / "source_snapshot" / path.name)
for name in ("prepared.json", "input_manifest.json", "source_manifest.json", "protocol.json", "config_difference_from_preflight.json", "full_training.yaml",
             "preparation.log", "supervisor.log", "supervisor.json", "process.json", "process_exit.json", "failure.json", "launch_inventory.json", "training.log",
             "initialization.json", "configuration_check.json"):
    path = ROOT / name
    if path.exists():
        path.rename(archive / name)
write(archive / "archive_record.json", {"optimizer_updates": 0, "student_loaded": True, "training_started": False,
    "reason": "datasets.load_from_disk calls with_format and changes runtime fingerprint; verify persisted state plus Arrow file SHA-256 instead.",
    "archived_at_utc": utc_now()})
print("ARCHIVED_ZERO_UPDATE_FINGERPRINT_FAILURE", archive)
