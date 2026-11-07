"""The original frozen runner, with a process-local accurate norm reducer."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from stable_norm import install
install()
import run_gold as original
from experiment_paths import model_root

base_config=original.config
def config(arm,smoke=False):
    assert arm in ['G1-M','G3-M']
    cfg=base_config(arm,smoke)
    if smoke:cfg['output_dir']=str(model_root()/'smoke_norm_fp64'/arm)
    else:assert all(json.loads((HERE/'validation_gates_repair.json').read_text()).values())
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    (output/'matching_numerics.json').write_text(json.dumps({'norm_reduction':'FP64 over full FP32 gradient vectors','coefficient':.0012,'match_relative_tolerance':2e-5,'stable_norm_sha256':hashlib.sha256((HERE/'stable_norm.py').read_bytes()).hexdigest()},indent=2))
    return cfg
original.config=config
if __name__=='__main__':original.main()
