"""Freeze a verified result bundle after all bounded GPU work has exited."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert json.loads((ROOT/'suite_status.json').read_text())['state']=='completed'
    assert json.loads((ROOT/'profile_process.json').read_text())['state']=='completed'
    for path,expected in json.loads((ROOT/'source_manifest.json').read_text())['files'].items():
        assert sha(Path(path))==expected,path
    full=ROOT.parent/'gold_reasoning_full_49800_20261103'
    assert json.loads((full/'process.json').read_text())['state']=='stopped_by_user'
    assert not Path('/proc/873499').exists()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    pids=[json.loads((ROOT/f'{v}_process.json').read_text())['worker_pid'] for v in ['equivalence','baseline','cached','pipeline']]
    pids.append(json.loads((ROOT/'profile_process.json').read_text())['pid'])
    assert not any(str(pid) in [part.strip() for part in row.split(',')][1:2] for pid in pids for row in processes.splitlines())
    summary=json.loads((ROOT/'comparison.json').read_text())
    profile=json.loads((ROOT/'component_profile/completion.json').read_text())
    summary['component_profile']=profile
    summary['all_benchmark_gpu_processes_exited']=True
    summary['additional_gpu_plan']['four_gpu_requirements'].append('synchronize the combined gradients of both backward branches once per accumulation window, despite different Think/no_think layouts across ranks')
    write(ROOT/'comparison.json',summary)
    with (ROOT/'throughput_report.txt').open('a',encoding='utf-8') as stream:
        stream.write('\nGPU监控补充（各版第4至12步期间采样）：\n')
        for name,item in summary['comparison'].items():
            stream.write(f"{name}: 学生GPU平均利用率{item['mean_gpu_util']['1']:.2f}%，教师{item['mean_gpu_util']['2']:.2f}%，采样点{item['gpu_sampling_points']}。\n")
        stream.write('CUDA分析为独立的4样本诊断，包含分析器开销，不能替代上面的12步吞吐对照。详见component_profile/completion.json。\n')
        stream.write('所有本次GPU诊断进程均已退出，原全量训练保持停止。\n')
    stopped=ROOT/'stopped_full_run'
    stopped.mkdir(exist_ok=False)
    for name in ['user_stop.json','process.json','process_exit.json','failure.json','progress.json','steps.jsonl','trainer_logs.jsonl','training.log']:
        path=full/name
        if path.exists():shutil.copy2(path,stopped/name)
    write(ROOT/'final_verification.json',{'passed':True,'full_training_restarted':False,'all_benchmark_workers_exited':True,
        'source_hashes_unchanged':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'gpu_processes':processes})
    files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
    write(ROOT/'delivery_manifest.json',{'files':files,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    output=Path('/tmp/gold_reasoning_throughput_20261103_delivery.tar.gz')
    assert not output.exists()
    with tarfile.open(output,'w:gz') as archive:
        for relative in list(files)+['delivery_manifest.json']:
            archive.add(ROOT/relative,arcname=relative,recursive=False)
    print(json.dumps({'bundle':str(output),'files':len(files),'bundle_sha256':sha(output),'bytes':output.stat().st_size},indent=2))

if __name__=='__main__':main()
