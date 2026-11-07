"""Explicit Gold experiment routes; legacy files are read-only references."""
import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def paths():
    config=Path(os.environ.get('SFT_EXPERIMENT_PATHS',ROOT/'paths.json'))
    raw=json.loads(config.read_text(encoding='utf-8'))
    return {k:None if v is None else (Path(v).resolve() if Path(v).is_absolute() else (ROOT/v).resolve())
            for k,v in raw.items() if k!='schema'}
def require_path(key):
    p=paths().get(key)
    if p is None or not p.exists():raise FileNotFoundError(f'Configure existing {key}: {p}')
    return p
def experiment_root():return ROOT
def benchmark_root():return paths()['benchmark_root']
def data_root():return paths()['data_root']
def model_root():return paths()['model_root']
def evaluation_results_root():return paths()['results_root']
def evaluation_analysis_root():return paths()['analysis_root']
def script_path(name):
    if name in ('export_arm_v2.py','verify_export_v2.py','test_export_rotary_buffers.py'):
        return paths()['legacy_model_scripts']/name
    if name=='evaluate_arm.py':return paths()['legacy_analysis_root']/name
    if name=='analyze_results.py':return evaluation_analysis_root()/name
    return ROOT/name
def stage_log(stage,arm=None):
    folder=evaluation_analysis_root()/'logs' if 'evaluat' in stage or stage=='analysis' else ROOT/'logs'
    folder.mkdir(parents=True,exist_ok=True)
    return folder/(stage+('_'+arm if arm else '')+'.log')
sys.path.insert(0,str(require_path('framework_root')/'src'))
