"""Remove CUDA user-annotation ranges from kernel totals without rerunning GPU work.

All three record_function ranges appear in the saved top-20 aggregate, so the
exact duplicated times can be subtracted. Preserve the raw summary for audit.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
path=root/'component_profile/completion.json'
raw=json.loads(path.read_text())
labels={'cuda:1 teacher.hidden','cuda:1 teacher.target_projected','cuda:1 teacher.target_anchor'}
annotations={name:value for name,value in raw['top_kernels_ms'] if name in labels}
assert set(annotations)==labels
backup=path.with_name('completion_raw_including_cuda_ranges.json')
assert not backup.exists()
backup.write_bytes(path.read_bytes())
fixed=dict(raw)
fixed['device_activity_ms_raw_including_annotations']=raw['device_activity_ms']
fixed['device_activity_ms']=dict(raw['device_activity_ms'])
fixed['device_activity_ms']['cuda:1/kernel']-=sum(annotations.values())
assert fixed['device_activity_ms']['cuda:1/kernel']>0
fixed['removed_user_annotation_ranges_ms']=annotations
fixed['top_kernels_ms']=[(name,value) for name,value in raw['top_kernels_ms'] if name not in labels]
fixed['aggregation_correction']='Exclude the three teacher.* CUDA user ranges, which overlap their child kernels; exact saved aggregates were subtracted. No GPU work rerun.'
path.write_text(json.dumps(fixed,indent=2)+'\n')
comparison=json.loads((root/'comparison.json').read_text())
comparison['component_profile']=fixed
(root/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (root/'throughput_report.txt').open('a',encoding='utf-8') as stream:
    stream.write('\nCUDA汇总已排除与底层kernel重复计数的三个teacher.*自定义范围；原汇总保存在component_profile/completion_raw_including_cuda_ranges.json。\n')
    stream.write(f"本次4条样本中，学生kernel合计{fixed['device_activity_ms']['cuda:0/kernel']/1000:.3f}秒，教师kernel合计{fixed['device_activity_ms']['cuda:1/kernel']/1000:.3f}秒；复制和memset合计{(fixed['device_activity_ms']['cuda:0/copy_or_memset']+fixed['device_activity_ms']['cuda:1/copy_or_memset'])/1000:.3f}秒。这些是CUDA活动时间，不能与含分析器开销的墙钟时间直接相加比较。\n")
source=(root/'profile_components.py').read_text()
source=source.replace("if 'CUDA' not in str(event.device_type):continue", "if 'CUDA' not in str(event.device_type) or event.name.startswith('teacher.'):continue")
(root/'profile_components_corrected.py').write_text(source)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
(root/'delivery_manifest.json').write_text(json.dumps({'files':files,'created_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
output=Path('/tmp/gold_reasoning_throughput_20261103_delivery.tar.gz')
output.rename(output.with_name('gold_reasoning_throughput_20261103_delivery_before_profile_fix.tar.gz'))
with tarfile.open(output,'w:gz') as archive:
    for relative in list(files)+['delivery_manifest.json']:archive.add(root/relative,arcname=relative,recursive=False)
print(json.dumps({'corrected_device_activity_ms':fixed['device_activity_ms'],'files':len(files),'bundle_sha256':sha(output)},indent=2))
