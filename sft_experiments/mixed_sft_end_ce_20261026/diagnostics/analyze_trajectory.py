"""Summarize saved custom training diagnostics; no DEV/TEST inputs."""
import json,sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_paths import model_root, diagnostics_root, diagnostic_path
class Records:
    @staticmethod
    def read(path):
        return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
old = Records()
ARMS = ["G1","G2","G3","G1-M","G3-M"]

def main():
    out = diagnostics_root()
    out.mkdir(parents=True, exist_ok=True)
    trajectory={}
    for arm in ['E1','E2','E1-M']:
        run=model_root()/'runs'/arm;steps=old.read(diagnostic_path(run/'termination_steps.jsonl'));assert len(steps)==1125
        probes=old.read(diagnostic_path(run/'end_gradient_probes.jsonl'));groups={}
        for r in probes:groups.setdefault(f"{r['step']}_{r['phase']}",[]).append(r)
        assert all(len(v)==8 for v in groups.values())
        assert '1125_post_update_final' in groups
        trajectory[arm]={'steps':1125,'groups':{},'maximum_match_relative_error':max(r.get('match_relative_error',0) for r in steps)}
        if arm=='E1-M':assert trajectory[arm]['maximum_match_relative_error']<2e-5
        for step,rows in groups.items():
            trajectory[arm]['groups'][step]={k:{n:float(np.quantile([r[k] for r in rows],q)) for n,q in [('min',0),('p10',.1),('p50',.5),('p90',.9),('p95',.95),('max',1)]} for k in ['end_R','dynamic_end_R','matched_end_R','match_scale']}
    (out/'gradient_trajectory_summary.json').write_text(json.dumps(trajectory,indent=2))

if __name__ == '__main__': main()
