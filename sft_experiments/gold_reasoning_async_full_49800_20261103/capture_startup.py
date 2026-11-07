"""Capture stable startup evidence without interrupting ongoing training."""
from datetime import datetime, timedelta, timezone
import math
import os
import shutil
import statistics
import subprocess
import tarfile
from common import *

def read(name):return json.loads((ROOT/name).read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]

def main():
    assert not (ROOT/'startup_verified').exists()
    process=read('process.json')
    assert process['state']=='running'
    os.kill(process['worker_pid'],0)
    os.kill(process['supervisor_pid'],0)
    assert os.getsid(process['worker_pid'])==process['worker_pid']
    assert os.getsid(process['supervisor_pid'])==process['supervisor_pid']
    assert not any((ROOT/name).exists() for name in ['failure.json','launch_failure.json','process_exit.json'])
    measured=rows(ROOT/'steps.jsonl')
    logs={x['step']:x for x in rows(ROOT/'trainer_logs.jsonl') if 'grad_norm' in x}
    measured=[x for x in measured if x['step'] in logs]
    assert len(measured)>=4
    assert [x['step'] for x in measured]==list(range(1,len(measured)+1))
    prepared=0
    for row in measured:
        prepared+=row['think']
        assert row['teacher_unchanged'] and row['prefetch_drained']
        assert row['teacher_prepared_samples']==prepared
        assert all(math.isfinite(row[k]) for k in ['ce','weighted_kd','weighted_anchor','seconds'])
        assert math.isfinite(logs[row['step']]['grad_norm']) and logs[row['step']]['grad_norm']>0
        assert abs(logs[row['step']]['loss']-row['ce']-row['weighted_kd']-row['weighted_anchor'])<.001
    init=read('initialization.json')
    assert init['fresh_from']==BASE and init['lora_B_all_zero']
    assert init['adapter_loaded'] is None and init['resume'] is None and not init['preflight_adapter_loaded']
    env=read('environment.json')
    assert env['cache_and_async_prefetch'] and env['trainer_class']=='PipelineTrainer' and env['reference_class']=='PipelineTargets'
    assert env['trainer_source']==str(OPTIMIZATION/'optimized.py')
    assert 'PreflightCallback' not in env['callback_types'] and 'FullTrainingCallback' in env['callback_types']
    assert read('dataset_loaded.json')['rows']==49800 and read('dataset_loaded.json')['arrow_sha256_verified']
    cfg=read('configuration_check.json')
    assert cfg['passed'] and cfg['save_steps']==1500 and cfg['epochs']==3
    assert read('training_started.json')['target_steps']==9339
    for path,expected in read('source_manifest.json')['files'].items():assert sha(path)==expected,path
    prior=rows(OPTIMIZATION/'pipeline/steps.jsonl')
    assert all(a['source_indices']==b['source_indices'] for a,b in zip(measured,prior))
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    selected=[]
    for line in gpu.splitlines():
        uuid,pid,memory=(x.strip() for x in line.split(','))
        if int(pid)==process['worker_pid']:selected.append({'uuid':uuid,'used_mib':int(memory)})
    assert {x['uuid'] for x in selected}=={x['uuid'] for x in GPU_ROLES.values()}
    report={'passed':True,'scope':'startup only; full training continues','checked_at_utc':utc_now(),
        'worker_pid':process['worker_pid'],'supervisor_pid':process['supervisor_pid'],'detached':True,
        'completed_optimizer_steps':len(measured),'total_steps':9339,'epochs':3,'training_rows':49800,
        'fresh_sft_new2_verified':True,'cache_and_async_prefetch_verified':True,'same_sample_order_as_pipeline_benchmark':True,
        'losses_and_gradients_finite':True,'teacher_frozen':True,'prefetch_drained_after_each_step':True,
        'first_steps_seconds':[row['seconds'] for row in measured],
        'mean_seconds_after_first3':statistics.mean(row['seconds'] for row in measured[3:]),
        'grad_norms':[logs[row['step']]['grad_norm'] for row in measured],
        'student_peak_allocated_gib':max(row['student_peak_allocated_gib'] for row in measured),
        'teacher_peak_allocated_gib':max(row['teacher_peak_allocated_gib'] for row in measured),
        'student_peak_reserved_gib':max(row['student_peak_reserved_gib'] for row in measured),
        'teacher_peak_reserved_gib':max(row['teacher_peak_reserved_gib'] for row in measured),
        'checkpoint_steps_configured':EXPECTED_CHECKPOINTS,'checkpoint_saved_at_capture':(ROOT/'latest_checkpoint.json').exists(),
        'estimated_full_training_hours_from_validated_benchmark':read('protocol.json')['full_training_estimate_hours_from_preflight'],
        'gpu_process_occupancy':selected}
    write(ROOT/'startup_verification.json',report)
    local_time=datetime.fromisoformat(report['checked_at_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')
    content=f'''缓存＋异步预取全量训练：启动检查通过，任务继续在后台运行。
检查时间：{local_time}
GPU 1：sft-new2全新初始化的学生；GPU 2：冻结FP32 CE-9000教师。
实现：直接导入已验证的 {OPTIMIZATION/'optimized.py'}。
原49,800条数据；3 epoch；有效批量16；总计9339步。
已核验完整步骤：{len(measured)}。训练PID：{process['worker_pid']}；监督进程PID：{process['supervisor_pid']}。
进程使用独立会话，退出SSH不会停止任务。

验证：原输入Arrow文件哈希一致，学生252个LoRA B矩阵从零开始，未载入短测adapter或旧检查点。
使用PipelineTrainer和PipelineTargets；无12/50步预检停止条件。
损失和梯度均正常，教师参数保持冻结，每个更新步结束时预取队列已清空。
样本顺序与吞吐短测一致。首批步骤耗时（秒）：{', '.join(f'{x:.2f}' for x in report['first_steps_seconds'])}。
峰值已分配显存：学生{report['student_peak_allocated_gib']:.3f}GiB，教师{report['teacher_peak_allocated_gib']:.3f}GiB。

保存1500、3000、4500、6000、7500、9000和最终9339步；包含adapter、optimizer、scheduler、RNG、trainer_state。
当前尚未到首个保存步数。模型目录：{MODEL_ROOT}
按此前通过验证的短测，纯训练约66小时（2.75天），另计检查点保存时间；正式运行的速度仍需随进度观察。
本次只验证运行状态，尚无新模型准确率结论。

实时查询：ssh L20 '{PYTHON} {ROOT}/status.py'
startup_verified保留本次固定快照；后续进度以服务器实时日志为准。
'''
    (ROOT/'startup_report.txt').write_text(content,encoding='utf-8')
    snapshot=ROOT/'startup_verified'
    snapshot.mkdir()
    for path in ROOT.iterdir():
        if path.is_file() and path.name!='launch.lock' and not path.name.endswith('.tmp'):
            shutil.copy2(path,snapshot/path.name)
    files={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()}
    write(snapshot/'snapshot_manifest.json',{'captured_at_utc':report['checked_at_utc'],'files':files,'live_training_continues':True})
    archive=Path('/tmp/gold_reasoning_async_full_49800_20261103_startup.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for path in snapshot.iterdir():tar.add(path,arcname=path.name,recursive=False)
    print(json.dumps({'startup_verification':report,'archive':str(archive),'archive_sha256':sha(archive),'snapshot_files':len(files)},indent=2))

if __name__=='__main__':main()
