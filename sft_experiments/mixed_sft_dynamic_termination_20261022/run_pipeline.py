"""Authorized small-scale pipeline. Deliberately has no full-training stage."""
from experiment_paths import diagnostic_path
from experiment_paths import experiment_root,benchmark_root,data_root,model_root,paths,require_path,script_path,stage_log,evaluation_analysis_root

from experiment_paths import evaluation_analysis_root
import datetime
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT=Path(__file__).resolve().parent
TRAIN=str(require_path('training_python'))
INFER=str(require_path('inference_python'))
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join([str(paths()['framework_root']/'src'),str(ROOT)]),CUDA_VISIBLE_DEVICES='0',
         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')

def status(stage,state='running',**extra):
    value=dict(stage=stage,status=state,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               full_training_started=False,**extra)
    (ROOT/'pipeline_status.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)

def execute(stage,python,script,*args):
    status(stage)
    with stage_log(stage).open('a') as log:
        result=subprocess.run([python,str(script_path(script)),*args],env=ENV,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{stage} failed, see {stage}.log')

def main():
    try:
        check=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(check.strip())>500:raise RuntimeError('GPU 0 is occupied; refusing to start')
        assert (ROOT/'calibration.json').exists()
        assert json.loads((data_root()/'manual_review_results.json').read_text())['count']==32
        assert 'INTEGRATION_PASS' in (ROOT/'integration_tests.log').read_text()
        assert '\nOK\n' in (ROOT/'unit_tests_training.log').read_text()
        execute('freeze_inputs',TRAIN,'freeze_inputs.py')
        execute('smoke',TRAIN,'run_experiment.py','smoke','--arm','S1-M')
        smoke=json.loads((diagnostic_path(model_root()/'smoke/S1-M/completed.json')).read_text())
        assert smoke['global_step']==1
        (ROOT/'validation_gates.json').write_text(json.dumps(dict(unit_tests=True,human_review_32=True,
            integration_tests=True,real_smoke=True,calibration=True),indent=2))
        for arm in ['S0','S1','S2','S1-M']:
            execute('train_'+arm,TRAIN,'run_experiment.py','train','--arm',arm)
            execute('export_'+arm,TRAIN,'export_arm.py','--arm',arm)
            execute('evaluate_'+arm,INFER,'evaluate_arm.py','--arm',arm)
        execute('analysis',TRAIN,'analyze_results.py')
        status('small_scale_complete','completed')
    except Exception as exc:
        status('halted','failed',reason=str(exc))
        (ROOT/'pipeline_error.txt').write_text(traceback.format_exc())
        raise

if __name__=='__main__':main()
