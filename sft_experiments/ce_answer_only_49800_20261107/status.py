"""Read current pipeline, training and evaluation progress without changing state."""
from collections import deque
import json
import os
from pathlib import Path
from common import ARTIFACTS, ANALYSIS, NAME, STEPS, read

def main():
    result = {"experiment": NAME}
    pipeline = ARTIFACTS / "pipeline.json"
    if pipeline.exists():
        result["pipeline"] = read(pipeline)
        pid = result["pipeline"].get("pid")
        cmd = Path(f"/proc/{pid}/cmdline")
        result["supervisor_alive"] = bool(cmd.exists() and NAME in cmd.read_bytes().decode(errors="replace"))
    for phase in ("preflight", "main"):
        directory = ARTIFACTS / phase
        for key in ("progress", "completion"):
            path = directory / (key + ".json")
            if path.exists():
                result[phase + "_" + key] = read(path)
        steps = directory / "steps.jsonl"
        if steps.exists():
            with steps.open() as stream:
                tail = [json.loads(line) for line in deque(stream, maxlen=100)]
            if tail:
                recent = tail[5:] if len(tail) > 10 else tail
                mean = sum(row["seconds"] for row in recent) / len(recent)
                result[phase + "_recent_step_seconds"] = mean
                result[phase + "_remaining_training_hours"] = (tail[-1]["max_steps"] - tail[-1]["step"]) * mean / 3600
    result["evaluated_on_steps"] = [step for step in STEPS if (ANALYSIS / f"on-{step}.json").exists()]
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
