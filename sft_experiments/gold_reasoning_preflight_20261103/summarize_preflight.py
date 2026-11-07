"""Summarize actual optimizer updates and transparent wall-clock estimates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main(run_id):
    folder = ROOT / run_id
    rows = [json.loads(l) for l in (folder / "steps.jsonl").open()]
    completion = json.loads((folder / "completion.json").read_text())
    data = json.loads((ROOT / "data_manifest.json").read_text())
    equivalent = json.loads((folder / "ce_equivalence.json").read_text())
    stress = json.loads((folder / "long_sequence_memory.json").read_text())
    assert len(rows) == completion["optimizer_updates"] and len(rows) in (50, 100)
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    assert completion["passed"] and equivalent["passed"]
    assert all(r["microbatches"] == 16 for r in rows)
    steady = rows[3:]
    seconds = np.array([r["seconds"] for r in steady])
    ratios = np.array([r["extra_to_ce"] for r in rows])
    n = sum(r["microbatches"] for r in rows)
    n_think = sum(r["think"] for r in rows)
    ce = np.mean([r["ce"] for r in rows])
    kd = np.mean([r["weighted_kd"] for r in rows])
    anchor = np.mean([r["weighted_anchor"] for r in rows])
    full_steps = completion["configured_scheduler_steps"]
    rng = np.random.default_rng(3407)
    bootstrap = np.mean(rng.choice(seconds, (10000, len(seconds)), replace=True), axis=1)
    result = {
        "status": "preflight_complete_not_full_training", "optimizer_updates": len(rows), "training_examples_seen": n,
        "unique_training_examples_seen": len({i for r in rows for i in r["source_indices"]}),
        "think_seen": n_think, "no_think_seen": n-n_think, "full_dataset_think_fraction": data["counts"]["think"] / data["rows"],
        "sample_think_fraction": n_think/n, "sample_mean_input_tokens": sum(r["input_tokens"] for r in rows)/n,
        "full_mean_input_tokens": data["mean_length"], "sample_max_input_tokens": max(r["max_length"] for r in rows),
        "ce_equivalence_passed": True, "max_ce_gradient_absolute_error": max(r["gradients"]["maximum_absolute_error"] for r in equivalent["windows"]),
        "loss": {"ce_mean": float(ce), "weighted_projected_kl_mean": float(kd), "weighted_anchor_mean": float(anchor),
                 "extra_over_ce_ratio_of_means": float((kd+anchor)/ce), "per_step_extra_over_ce_p05_p50_p95": np.quantile(ratios, [.05,.5,.95]).tolist(),
                 "unweighted_kd_think_mean": sum(r["kd_sum"] for r in rows)/n_think,
                 "unweighted_anchor_think_mean": sum(r["anchor_sum"] for r in rows)/n_think},
        "memory_gib": {k: max(r[k] for r in rows) for k in ("student_peak_allocated_gib", "student_peak_reserved_gib", "teacher_peak_allocated_gib", "teacher_peak_reserved_gib")},
        "extreme_length_memory_gib": {k: max(r[k] for r in stress["samples"]) for k in ("student_peak_allocated_gib", "student_peak_reserved_gib", "teacher_peak_allocated_gib", "teacher_peak_reserved_gib")},
        "extreme_length_backward_probes": stress,
        "timing": {"all_training_seconds": completion["training_seconds"], "save_seconds": completion["save_seconds"],
                   "actual_optimizer_steps_per_second": len(rows)/completion["training_seconds"],
                   "actual_examples_per_second": n/completion["training_seconds"],
                   "native_trainer_speed_metrics_ignored": "Early-stop Trainer metrics divide the planned full-run samples/steps by short-run elapsed time; use actual steps.jsonl counts instead.",
                   "steady_steps": len(steady), "excluded_startup_steps": 3, "steady_seconds_mean": float(seconds.mean()),
                   "steady_seconds_p10_p50_p90_p95": np.quantile(seconds, [.1,.5,.9,.95]).tolist(),
                   "mean_teacher_hidden_seconds": float(np.mean([r["teacher_seconds"] for r in steady])),
                   "mean_student_plus_and_target_seconds": float(np.mean([r["plus_seconds"] for r in steady])),
                   "mean_student_zero_anchor_seconds": float(np.mean([r["zero_seconds"] for r in steady]))},
        "full_3epoch_estimate": {"steps": full_steps, "hours_at_observed_mean": float(seconds.mean()*full_steps/3600),
                                "hours_at_observed_step_p10_p90": (np.quantile(seconds,[.1,.9])*full_steps/3600).tolist(),
                                "bootstrap_mean_95pct_hours": (np.quantile(bootstrap,[.025,.975])*full_steps/3600).tolist(),
                                "scope": "Observed current implementation and shared-server conditions; excludes DEV evaluation, checkpoint saving, export, startup and future contention. Step-percentile range is sensitivity, not a confidence interval for full runtime."},
        "teacher_unchanged": completion["teacher_unchanged"], "student_updated": completion["student_parameters_changed"],
        "heldout_diagnostic_gradient_updates": 0, "dev_test_generation": False, "full_training_started": False}
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    memory = result["memory_gib"]
    estimate = result["full_3epoch_estimate"]
    text = f"""修正 Q 的双 L20 短程训练预检

已完成 {len(rows)} 个优化器更新步骤，未启动全量训练。
学生：sft-new2，沿用 LoRA r64 / alpha128 / dropout0.05，并训练完整 lm_head。
教师：CE-9000，独立 L20，冻结 FP32。学生保持原 BF16 训练。
数据：原 49,800 条；固定 1,000 条 Think 诊断集不参与梯度更新。
日程：保留 3 epoch / 9,339 步跨度，在第 {len(rows)} 步硬停止。

CE 和梯度一致性
实际 8B 模型、全部 835,190,784 个可训练参数检查通过。
最大梯度绝对差：{result['max_ce_gradient_absolute_error']:.9g}。
检查同时包含原生 CE 重复运行对照；一致性检查开启确定性 FlashAttention。
正式短程预检恢复原训练默认设置，不为计时开启该确定性开关。
比较范围是当前可恢复的原生 LLaMA-Factory 训练栈，不声称历史软件逐字节一致。

损失量级
实际使用 {n} 条样本：Think {n_think} 条、no_think {n-n_think} 条。
平均 CE：{ce:.6f}；加权 Q* KL：{kd:.6f}；加权无推理锚定：{anchor:.6f}。
两项额外损失之和 / CE（均值之比）：{(kd+anchor)/ce:.2%}。
额外损失先对每条 Think 的回答 token 求均值，再对每个16条累积窗口全部样本求均值，
no_think贡献零；因此不能直接与此前纯Think诊断的15.70%做同口径比较。
标量损失比例不代表梯度强度，也不代表答题准确率提升。

显存和耗时
50步训练中的学生峰值 allocated / reserved：{memory['student_peak_allocated_gib']:.2f} / {memory['student_peak_reserved_gib']:.2f} GiB。
50步训练中的教师峰值 allocated / reserved：{memory['teacher_peak_allocated_gib']:.2f} / {memory['teacher_peak_reserved_gib']:.2f} GiB。
保留真实优化器状态的极端长度探针峰值另见 summary.json / long_sequence_memory.json。
去掉前3步启动影响后，平均每个优化器步骤 {seconds.mean():.2f} 秒。
完整3 epoch纯训练点估计：{estimate['hours_at_observed_mean']:.2f} 小时。
以本次单步P10-P90外推的敏感性范围：{estimate['hours_at_observed_step_p10_p90'][0]:.2f}—{estimate['hours_at_observed_step_p10_p90'][1]:.2f} 小时。
该范围不是完整训练时长的置信区间。预检实际训练耗时 {completion['training_seconds']/60:.2f} 分钟。
估计不含DEV评测、全部检查点写盘和导出时间，也不能保证未来服务器争用相同。
原生Trainer在回调提前停止后仍用计划的全程样本/步数计算吞吐，日志中的
train_samples_per_second和train_steps_per_second不适用于本次预检；本报告使用实际50步计时。

逐步记录见 steps.jsonl。此次没有运行DEV/TEST生成；预检模型不能作为主实验效果结果。
主实验应从sft-new2重新初始化，不能把这50步预检直接计作已验证的主实验。
"""
    (folder / "report.txt").write_text(text, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="attempt01")
    main(parser.parse_args().run_id)
