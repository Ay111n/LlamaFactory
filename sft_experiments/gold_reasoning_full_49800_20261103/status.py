"""Read-only status, measured progress and ETA; does not import CUDA."""
import json
import os
import statistics
import time
from common import *


def main():
    result = {"experiment": str(ROOT), "checked_at_utc": utc_now()}
    for name in ("process", "training_started", "process_exit", "failure", "launch_failure", "latest_checkpoint", "completion", "initialization"):
        path = ROOT / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())
    if "process" in result:
        try:
            os.kill(result["process"]["worker_pid"], 0)
            result["worker_alive"] = True
        except ProcessLookupError:
            result["worker_alive"] = False
    path = ROOT / "steps.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.open() if line.endswith("\n")]
        if rows:
            steady = rows[3:] or rows
            recent = steady[-50:]
            seconds = statistics.mean(r["seconds"] for r in recent)
            progress = json.loads((ROOT / "progress.json").read_text())
            result.update(completed_steps=len(rows), total_steps=EXPECTED_STEPS,
                recent_mean_step_seconds=seconds, seconds_since_last_step=time.time()-rows[-1]["timestamp"],
                remaining_training_hours=max(0, (EXPECTED_STEPS-len(rows))*seconds-(time.time()-rows[-1]["timestamp"]))/3600,
                current={k: progress[k] for k in ("step", "ce", "weighted_kd", "weighted_anchor", "extra_to_ce", "lr", "teacher_unchanged", "student_peak_allocated_gib", "teacher_peak_allocated_gib", "grad_norm") if k in progress})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
