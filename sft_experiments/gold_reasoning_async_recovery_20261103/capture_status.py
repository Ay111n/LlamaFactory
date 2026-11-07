"""Read-only recovery snapshot; checks process identity and matched sample order."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
FULL = ROOT.parent / "gold_reasoning_async_full_repaired_49800_20261103"
FAILED = ROOT.parent / "gold_reasoning_async_full_49800_20261103"


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def rows(path):
    return [json.loads(line) for line in path.open() if line.endswith("\n")] if path.is_file() else []


def main():
    process = load(FULL / "process.json")
    step_rows = rows(FULL / "steps.jsonl")
    logs = rows(FULL / "trainer_logs.jsonl")
    old = rows(ROOT.parent / "gold_reasoning_preflight_20261103/attempt02/steps.jsonl")
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "process": process,
        "failed_run": load(FAILED / "process.json"), "failed_run_steps": len(rows(FAILED / "steps.jsonl")),
        "equivalence_passed": bool((load(ROOT / "equivalence/completion.json") or {}).get("passed")),
        "copy_lifetime_original": load(ROOT / "copy_lifetime_original.json"),
        "copy_lifetime_safe": load(ROOT / "copy_lifetime_safe.json"),
        "training_steps": len(step_rows)}
    if process:
        command = Path(f"/proc/{process['worker_pid']}/cmdline")
        report["worker_alive_with_matching_entry"] = command.exists() and str(FULL / "run_full.py").encode() in command.read_bytes()
    if step_rows:
        recent = step_rows[3:][-50:] or step_rows
        report.update(last_progress=load(FULL / "progress.json"), recent_mean_step_seconds=statistics.mean(r["seconds"] for r in recent),
            matched_source_order_steps=min(len(old), len(step_rows)),
            source_order_matches_preflight=all(a["source_indices"] == b["source_indices"] for a,b in zip(old,step_rows)),
            lr_matches_preflight=all(a["lr"] == b["lr"] for a,b in zip(old,step_rows)),
            teacher_frozen_every_step=all(r["teacher_unchanged"] for r in step_rows),
            prefetch_drained_every_step=all(r["prefetch_drained"] for r in step_rows),
            peak_student_gib=max(r["student_peak_allocated_gib"] for r in step_rows),
            peak_teacher_gib=max(r["teacher_peak_allocated_gib"] for r in step_rows),
            maximum_grad_norm=max(r["grad_norm"] for r in logs if "grad_norm" in r))
        report["remaining_hours_estimate"] = (9339-len(step_rows))*report["recent_mean_step_seconds"]/3600 if report.get("worker_alive_with_matching_entry") else None
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu", "--format=csv"], text=True)
    report["gpu_inventory"] = gpu
    journal = subprocess.check_output(["journalctl", "-k", "--since", "2026-11-03 07:20:00", "--no-pager"], text=True)
    report["xid_lines"] = [line for line in journal.splitlines() if "Xid" in line]
    if process:
        since = process["started_at_utc"].split(".")[0].replace("T", " ")
        recent_journal = subprocess.check_output(["journalctl", "-k", "--since", since, "--no-pager"], text=True)
        report["xid_lines_since_restart"] = [line for line in recent_journal.splitlines() if "Xid" in line]
    (ROOT / "recovery_snapshot.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["copy_lifetime_original", "copy_lifetime_safe", "gpu_inventory", "last_progress", "xid_lines"]}, indent=2))
    if step_rows:
        print("LAST", json.dumps({k:step_rows[-1][k] for k in ["step", "seconds", "ce", "weighted_kd", "weighted_anchor", "student_allocation_retries", "teacher_allocation_retries", "student_oom_count", "teacher_oom_count"]}))


if __name__ == "__main__":
    main()
