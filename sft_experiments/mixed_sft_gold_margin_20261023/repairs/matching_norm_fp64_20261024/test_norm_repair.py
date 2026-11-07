import sys,json,math,unittest,importlib.util,hashlib
from pathlib import Path
import torch
from stable_norm import grad_norm,match_scale,install

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
torch.set_num_threads(8)
install()
rows=[]
# Large dynamic range reproduces loss of precision in the former norm reduction.
x=torch.full((64_000_000,),1e-4,dtype=torch.float32);x[0]=1.
truth=math.sqrt(1.+(x.numel()-1)*float(x[1])**2)
assert math.isclose(grad_norm([x]),truth,rel_tol=1e-10)
for scale in [.21381539225401902,.1,.37,2.]:
    y=x*scale
    c,n1,n2=match_scale([x],[y])
    actual=grad_norm([x*c]);error=abs(actual-n2)/n2
    old1=float(torch.linalg.vector_norm(x));old2=float(torch.linalg.vector_norm(y))
    old_error=abs(old2-scale*old1)/(scale*old1)
    assert error<2e-5
    rows.append(dict(scale=scale,old_fp32_scaling_error=old_error,stable_match_relative_error=error))
assert rows[0]['old_fp32_scaling_error']>2e-5
assert match_scale([torch.zeros(3)],[torch.zeros(3)])[0]==1.
for candidate,reference in [([torch.zeros(3)],[torch.ones(3)]),([torch.tensor([float('nan')])],[torch.ones(1)])]:
    try:match_scale(candidate,reference)
    except FloatingPointError:pass
    else:raise AssertionError('Invalid match accepted')
del x,y
spec=importlib.util.spec_from_file_location('gold_existing_tests',ROOT/'test_gold.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=HERE # Preserve the original validation artifacts.
suite=unittest.defaultTestLoader.loadTestsFromModule(module)
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
integration=json.loads((HERE/'integration_errors.json').read_text())
report={'passed':True,'gradient_vectors':'FP32 unchanged','norm_accumulation':'FP64','coefficient':.0012,'matching_tolerance':2e-5,'large_vector_checks':rows,'integration_cases':len(integration),'maximum_integration_abs_error':max(r['max_error'] for r in integration)}
(HERE/'numerics_validation.json').write_text(json.dumps(report,indent=2))
print('NUMERICS_VALIDATION',json.dumps(report),flush=True)
