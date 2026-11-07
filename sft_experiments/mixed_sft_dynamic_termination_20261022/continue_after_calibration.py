"""Wait for the current calibration process, then run the gated small experiment."""
from experiment_paths import experiment_root,benchmark_root,data_root,model_root,paths,require_path,script_path,stage_log,evaluation_analysis_root

import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
pid=int(sys.argv[1])
(root/'pipeline_status.json').write_text(json.dumps({'stage':'calibration','status':'running',
    'full_training_started':False,'note':'Uses existing /opt/anaconda3/envs/training; no new environment'},indent=2))
while Path(f'/proc/{pid}').exists():time.sleep(5)
if not (root/'calibration.json').exists():
    (root/'pipeline_status.json').write_text(json.dumps({'stage':'calibration','status':'failed',
        'full_training_started':False,'reason':'Calibration did not complete; see calibration.log'},indent=2))
    raise SystemExit(1)
raise SystemExit(subprocess.call([str(require_path('training_python')),str(root/'run_pipeline.py')]))
