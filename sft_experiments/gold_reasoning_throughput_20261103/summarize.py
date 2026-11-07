"""Summarize paired finite benchmarks, preserving measured vs projected results."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())
def records(path):return [json.loads(x) for x in path.read_text().splitlines()]

def main():
    names=['baseline','cached','pipeline']
    completions={name:read(ROOT/name/'completion.json') for name in names}
    assert all(x['passed'] for x in completions.values())
    rows={name:records(ROOT/name/'steps.jsonl') for name in names}
    assert len({len(x) for x in rows.values()})==1
    for name in names:
        assert all(a['source_indices']==b['source_indices'] and a['ce_tokens']==b['ce_tokens'] and a['input_tokens']==b['input_tokens']
                   for a,b in zip(rows['baseline'],rows[name]))
    eq=read(ROOT/'equivalence/completion.json')
    assert eq['passed']
    gpu=records(ROOT/'gpu_samples.jsonl')
    compared={}
    base=completions['baseline']['steady_mean_seconds']
    for name in names:
        x=completions[name]
        samples=[r for r in gpu if r['stage']==name and r['last_completed_step'] is not None and 3<=r['last_completed_step']<12]
        compared[name]={**x,'speedup_vs_baseline':base/x['steady_mean_seconds'],
            'step_time_reduction_fraction':1-x['steady_mean_seconds']/base,
            'estimated_9339_steps_hours':9339*x['steady_mean_seconds']/3600,
            'estimated_9339_steps_days':9339*x['steady_mean_seconds']/86400,
            'memory_peak_including_stress_student_gib':max(x['student_peak_gib'],x['long_sequence_probe']['student_peak_gib']),
            'memory_peak_including_stress_teacher_gib':max(x['teacher_peak_gib'],x['long_sequence_probe']['teacher_peak_gib']),
            'gpu_sampling_points':len(samples),
            'mean_gpu_util':{device:statistics.mean(row['devices'][device]['gpu_util'] for row in samples) for device in ('1','2')} if samples else None,
            'max_abs_logged_ce_difference':max(abs(a['ce']-b['ce']) for a,b in zip(rows['baseline'],rows[name]))}
    winner=min(names,key=lambda name:compared[name]['steady_mean_seconds'])
    stop=read(ROOT.parent/'gold_reasoning_full_49800_20261103'/'user_stop.json')
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'full_training_stopped':stop,
        'all_samples_and_order_match':True,'full_training_restarted':False,
        'equivalence_all_trainable_parameters':eq['all_trainable_parameters'],
        'equivalence_max_abs_gradient_error':max(row['gradients']['maximum_absolute_error'] for row in eq['windows'] if row['gradients'] is not None),
        'benchmark_count_each':12,'warmup_excluded':3,'measurement_steps_each':9,
        'comparison':compared,'fastest_measured':winner,
        'measurement_scope':'12 optimizer updates per variant from fresh sft-new2; first 3 excluded for timing; final extrapolation excludes saves and evaluation',
        'additional_gpu_plan':{'two_gpus':'Use the fastest validated implementation; one student + one frozen teacher',
            'four_gpus':'Two independent student-teacher pairs; DDP only on students, global effective batch 16 = 2 students x microbatch 1 x accumulation 8',
            'four_gpu_measured':False,'four_gpu_ideal_additional_speedup_ceiling':2,
            'four_gpu_gradient_bytes_per_student':835190784*4,
            'four_gpu_requirements':['global supervised-token CE normalization','global auxiliary sample normalization','gradient accumulation with no_sync','replicated initialization and new equivalence check'],
            'topology':'Prefer pair (1,2) and pair (5,6); each pair stays within its NUMA node. DDP between students crosses SYS; no NVLink.',
            'three_gpus':'Two students sharing one teacher requires a service queue and may move the bottleneck to teacher; not measured'},
        'offline_full_cache_costs':{'FP32_hidden_states_gib':12128157*4096*4*2/2**30,'FP32_Q_and_anchor_tib':12128157*161280*4*2/2**40,
            'observed_host_available_gib_approx':180,'observed_disk_available_gib_approx':295}}
    (ROOT/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['吞吐优化实测报告','',f'原全量训练已于 {stop["stopped_at_utc"]} 停止，最后完整步骤 {stop["last_completed_step"]}；没有保存到首个checkpoint。',
        '本轮只做有步数上限的独立诊断，不恢复全量训练。','',
        '实测协议：GPU 1 学生、GPU 2 冻结教师；真实8B；每版全新初始化，完全相同的12个累积批次；每个批次16条。',
        '损失、数据、精度、随机种子、优化器及9339步学习率日程相同；排除前3步，统计后9步。',
        f'等价性：比较全部 {eq["all_trainable_parameters"]:,} 个可训练参数，确定性FA2下最大梯度误差 {summary["equivalence_max_abs_gradient_error"]}。',
        '覆盖混合样本、最长序列、最长答案及纯no_think窗口；教师和学生参数在等价性检查中不更新。','',
        '方案 | 秒/步 | 相对吞吐 | 原输入token/秒 | 学生/教师峰值GiB（含有优化器状态的长序列检查） | 9339步外推小时']
    for name in names:
        x=compared[name]
        lines.append(f'{name} | {x["steady_mean_seconds"]:.3f} | {x["speedup_vs_baseline"]:.3f}x | {x["steady_input_tokens_per_second"]:.1f} | {x["memory_peak_including_stress_student_gib"]:.3f}/{x["memory_peak_including_stress_teacher_gib"]:.3f} | {x["estimated_9339_steps_hours"]:.2f}')
    lines.extend(['',f'本次最快通过检查的方案：{winner}。',
        'cached：逐样本缓存Q与无推理教师分布，复用同一结果；保持全词表FP32，降低输出层和Q构造的重复计算。',
        'pipeline：在上述缓存上增加一个后台线程与独立CUDA流，预先计算下一条Think样本；当前样本与最多一个预取样本共存。',
        '吞吐结果是这组短程样本的实测，完整训练时长为外推，并未完成3个epoch或评测准确率。','',
        '更多GPU：4张卡可组成两组学生/教师，学生之间用DDP同步；保持全局有效批量16，每学生累积8条。',
        '建议同NUMA配对(1,2)与(5,6)，只在学生间跨NUMA同步梯度；当前机器没有NVLink。',
        '4卡尚未实测，理想额外上限2倍，实际受梯度通信和样本长度不均衡影响；不能写成已经实现的加速。',
        '每学生可训练参数的FP32梯度约3.11 GiB，必须避免每个微批次都同步，并核对跨rank的CE token及辅助项样本归一化。',
        '3卡共享教师方案需要额外队列设计，教师可能成为瓶颈，不作为优先实现。','',
        '离线缓存整个数据集不是本次推荐：两路FP32隐藏状态约370GiB，完整两份目标约14.23TiB；当前可用主存约180GiB、磁盘约295GiB。',
        '当前代码与结果仅用于吞吐优化和后续选择；全量训练保持停止。'])
    (ROOT/'throughput_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'fastest':winner,'comparison':{name:{k:compared[name][k] for k in ['steady_mean_seconds','speedup_vs_baseline','estimated_9339_steps_hours','memory_peak_including_stress_student_gib','memory_peak_including_stress_teacher_gib']} for name in names}},indent=2))

if __name__=='__main__':main()
