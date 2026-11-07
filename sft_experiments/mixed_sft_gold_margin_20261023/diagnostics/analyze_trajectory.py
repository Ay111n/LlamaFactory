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
    signatures=[json.loads(diagnostic_path(model_root()/'runs'/arm/'fixed_probe_windows.json').read_text()) for arm in ARMS]
    assert all(x==signatures[0] for x in signatures),'Fixed probe samples differ across arms'
    for arm in ARMS:
     probes=old.read(diagnostic_path(model_root()/'runs'/arm/'gold_gradient_probes.jsonl'))
     trajectory[arm]={'probe_count':len(probes),'steps':sorted({r['step'] for r in probes}),'by_step':{}}
     for step in trajectory[arm]['steps']:
      batch=[r for r in probes if r['step']==step]
      assert len(batch)==8
      trajectory[arm]['by_step'][str(step)]={key:{name:float(np.quantile([r[key+'_R'] for r in batch],q)) for name,q in [('min',0),('p10',.1),('p50',.5),('p90',.9),('p95',.95),('max',1)]} for key in ['gold','dynamic_gold','dynamic_ce']}
     for objective in ['gold','dynamic_gold','dynamic_ce']:
      v=[r[objective+'_R'] for r in probes]
      trajectory[arm][objective+'_R']={n:float(np.quantile(v,q)) for n,q in [('min',0),('p50',.5),('p90',.9),('max',1)]}
      trajectory[arm][objective+'_parts']={k:float(np.mean([r['parts'][objective+'_'+k] for r in probes])) for k in ['continue','end']}
     if arm.endswith('-M'):
      steps=old.read(diagnostic_path(model_root()/'runs'/arm/'termination_steps.jsonl'))
      assert len(steps)==1125 and max(r.get('match_relative_error',0) for r in steps)<2e-5
      trajectory[arm]['maximum_match_relative_error']=max(r.get('match_relative_error',0) for r in steps)
    (out/'gradient_trajectory_summary.json').write_text(json.dumps(trajectory,indent=2))

if __name__ == '__main__': main()
