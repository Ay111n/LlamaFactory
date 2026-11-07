"""Run a bounded component profile only after the comparison releases both GPUs."""
import json
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parent
deadline=time.monotonic()+3600
while True:
    state=json.loads((root/'suite_status.json').read_text())
    if state['state']=='failed':raise RuntimeError('Comparison failed; do not run another job')
    if state['state']=='completed':break
    assert time.monotonic()<deadline
    time.sleep(2)
import os
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-5fc7820d-0470-f42a-a5c7-3f3cbfb2df88,GPU-c98046c2-90b5-a959-9e88-f09a146ee29e',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_DATASETS_DISABLE_PROGRESS_BARS='1',FLASH_ATTENTION_DETERMINISTIC='0')
with (root/'component_profile.log').open('x') as log:
    p=subprocess.Popen(['/opt/anaconda3/envs/training/bin/python','-u',str(root/'profile_components.py')],cwd=root.parent.parent,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'profile_process.json').write_text(json.dumps({'state':'running','pid':p.pid,'full_training':False})+'\n')
    rc=p.wait()
(root/'profile_process.json').write_text(json.dumps({'state':'completed' if rc==0 else 'failed','pid':p.pid,'returncode':rc,'full_training':False})+'\n')
