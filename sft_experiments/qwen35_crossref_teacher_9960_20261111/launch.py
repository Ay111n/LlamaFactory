"""Launch full training only after CPU and longest-example checks pass."""
import os
from pathlib import Path
import teacher
teacher.verify_preflight()
smoke=teacher.read(teacher.RUNTIME/'smoke_passed.json')
assert smoke['status']=='passed'
assert smoke.get('optimizer_state_and_accumulated_gradients_exercised') is True
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
assert smoke['memory_ce_sha256']==teacher.sha(teacher.HERE/'memory_ce.py')
short=teacher.PROJECT/'tmp_q35'
assert short.is_symlink() and short.resolve()==(teacher.RUNTIME/'temp').resolve()
for key,path in {'TMPDIR':short,'TRITON_CACHE_DIR':teacher.RUNTIME/'triton_cache',
                 'TORCHINDUCTOR_CACHE_DIR':teacher.RUNTIME/'torch_cache','CUDA_CACHE_PATH':teacher.RUNTIME/'cuda_cache'}.items():
 path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ['PYTHONPATH']=str(teacher.FRAMEWORK/'src')
os.execv(teacher.PYTHON,[teacher.PYTHON,'-u',str(Path(__file__).with_name('teacher.py')),'queue'])
