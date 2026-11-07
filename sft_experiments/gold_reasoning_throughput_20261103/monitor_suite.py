"""Read-only, one-second GPU samples while bounded benchmarks run."""
import csv
import json
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parent
while True:
    status=json.loads((root/'suite_status.json').read_text())
    if status['state']!='running':break
    stage=status['stage']
    progress=root/stage/'progress.json'
    try:step=json.loads(progress.read_text())['step'] if progress.exists() else 0
    except json.JSONDecodeError:step=None
    raw=subprocess.check_output(['nvidia-smi','--id=1,2','--query-gpu=index,utilization.gpu,utilization.memory,power.draw,memory.used','--format=csv,noheader,nounits'],text=True)
    devices={int(row[0]):dict(zip(['gpu_util','memory_util','power_w','memory_mib'],map(float,row[1:]))) for row in csv.reader(raw.splitlines())}
    with (root/'gpu_samples.jsonl').open('a') as stream:
        stream.write(json.dumps({'time':time.time(),'stage':stage,'last_completed_step':step,'devices':devices})+'\n')
    time.sleep(1)
