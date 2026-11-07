"""Run the bounded preflight only, holding a cooperative per-experiment lock."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/anaconda3/envs/training/bin/python"
GPU_UUIDS = ["GPU-5fc7820d-0470-f42a-a5c7-3f3cbfb2df88", "GPU-0997660c-d531-a4b1-9fe2-213bb2f5527e"]


def main(run_id, steps):
    assert str(Path(sys.executable)) == PYTHON
    lock = (ROOT / "preflight.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    inventory = {r[0].strip(): (int(r[1]), int(r[2])) for r in (line.split(",") for line in raw.splitlines())}
    for uuid in GPU_UUIDS:
        assert inventory[uuid][0] <= 64 and inventory[uuid][1] <= 5, (uuid, inventory[uuid], "GPU is not idle; no process will be killed")
    (ROOT / "launch_inventory.json").write_text(json.dumps({"gpus": GPU_UUIDS, "inventory": inventory}, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(GPU_UUIDS), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_DATASETS_DISABLE_PROGRESS_BARS="1")
    for stage in ("validate", "train"):
        with (ROOT / f"{stage}_{run_id}.log").open("x") as log:
            subprocess.run([PYTHON, "-u", str(ROOT / "run_preflight.py"), stage, "--steps", str(steps), "--run-id", run_id], cwd=ROOT.parent.parent, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (ROOT / "pipeline_complete.json").write_text(json.dumps({"preflight_only": True, "steps": steps, "run_id": run_id}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="attempt02")
    parser.add_argument("--steps", type=int, choices=(50,100), default=50)
    args = parser.parse_args()
    main(args.run_id, args.steps)
