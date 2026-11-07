"""Read progress without importing CUDA or modifying the training process."""
import argparse
import json
from pathlib import Path
import statistics
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", default="attempt02")
args = parser.parse_args()
root = Path(__file__).resolve().parent
folder = root / args.run_id
path = folder / "steps.jsonl"
rows = [json.loads(l) for l in path.open()] if path.exists() else []
if rows:
    steady = rows[3:] or rows
    mean_seconds = statistics.mean(r["seconds"] for r in steady)
    result = {"completed_steps": len(rows), "target_steps": 50,
        "measured_step_seconds_mean": mean_seconds,
        "estimated_remaining_training_minutes": max(0, (50-len(rows))*mean_seconds-(time.time()-rows[-1]["timestamp"])) / 60,
        "seconds_since_last_complete_step": time.time()-rows[-1]["timestamp"],
        "ce_mean": statistics.mean(r["ce"] for r in rows),
        "extra_to_ce_ratio_of_sums": sum(r["weighted_kd"]+r["weighted_anchor"] for r in rows)/sum(r["ce"] for r in rows),
        "student_peak_allocated_gib": max(r["student_peak_allocated_gib"] for r in rows),
        "teacher_peak_allocated_gib": max(r["teacher_peak_allocated_gib"] for r in rows),
        "all_logged_losses_finite": all(__import__('math').isfinite(r[k]) for r in rows for k in ("ce","weighted_kd","weighted_anchor")),
        "last_step": {k: rows[-1][k] for k in ("step", "seconds", "ce", "weighted_kd", "weighted_anchor", "max_length")},
        "completed_artifact_present": (folder / "completion.json").exists()}
else:
    result = {"completed_steps": 0}
for stage in ("validate", "train"):
    failure = folder / (stage+"_failure.json")
    if failure.exists():
        result[stage+"_failure"] = json.loads(failure.read_text())
print(json.dumps(result, indent=2))
