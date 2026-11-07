"""Detached supervisor for this explicitly authorized full training run."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from common import *


def supervise():
    lock = (ROOT / "launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((ROOT / "prepared.json").read_text())["passed"]
    assert not (ROOT / "training.log").exists(), "Refuse a duplicate or implicit restart"
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    inventory = {int(r[0]): {"uuid": r[1].strip(), "used_mib": int(r[2]), "utilization": int(r[3])} for r in (line.split(",") for line in raw.splitlines())}
    for role, requested in GPU_ROLES.items():
        current = inventory[requested["index"]]
        assert current["uuid"] == requested["uuid"]
        assert current["used_mib"] <= 64 and current["utilization"] <= 5, (role, current, "Requested GPU is occupied; no other process will be stopped")
    write(ROOT / "launch_inventory.json", inventory)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(GPU_ROLES[role]["uuid"] for role in ("student", "teacher")),
        PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_DATASETS_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_PROGRESS_BARS="1", FLASH_ATTENTION_DETERMINISTIC="0")
    with (ROOT / "training.log").open("x") as log:
        worker = subprocess.Popen([PYTHON, "-u", str(ROOT / "run_full.py")], cwd=FRAMEWORK,
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        process_info = {"state": "running", "supervisor_pid": os.getpid(), "worker_pid": worker.pid,
              "started_at_utc": utc_now(), "gpu_roles": GPU_ROLES, "command": [PYTHON, str(ROOT / "run_full.py")]}
        write(ROOT / "process.json", process_info)
        returncode = worker.wait()
    process_info.update(state="completed" if returncode == 0 else "failed", returncode=returncode, exited_at_utc=utc_now())
    write(ROOT / "process.json", process_info)
    write(ROOT / "process_exit.json", {"returncode": returncode, "state": "completed" if returncode == 0 else "failed",
        "exited_at_utc": utc_now(), "completion_present": (ROOT / "completion.json").exists()})
    return returncode


def start():
    assert sys.executable == PYTHON
    assert (ROOT / "prepared.json").is_file()
    assert not (ROOT / "supervisor.log").exists(), "Do not overwrite a prior launch"
    with (ROOT / "supervisor.log").open("x") as log:
        process = subprocess.Popen([PYTHON, "-u", str(ROOT / "launch_full.py"), "supervise"], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT / "supervisor.json", {"pid": process.pid, "launched_at_utc": utc_now(), "detached": True})
    print(json.dumps({"supervisor_pid": process.pid, "detached": True, "experiment": str(ROOT)}), flush=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "supervise":
            sys.exit(supervise())
        elif len(sys.argv) == 1:
            start()
        else:
            raise ValueError("Usage: launch_full.py [supervise]")
    except Exception as error:
        write(ROOT / "launch_failure.json", {"error": repr(error), "traceback": traceback.format_exc(), "timestamp_utc": utc_now()})
        raise
