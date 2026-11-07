import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-5fc7820d-0470-f42a-a5c7-3f3cbfb2df88,GPU-c98046c2-90b5-a959-9e88-f09a146ee29e", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_DATASETS_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
with (root / "validation.log").open("x") as log:
    worker = subprocess.Popen(["/opt/anaconda3/envs/training/bin/python", "-u", str(root / "validate_repair.py")], cwd=root.parent.parent,
        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root / "validation_process.json").write_text(json.dumps({"pid": worker.pid, "optimizer_updates": 0})+"\n")
print(worker.pid)
