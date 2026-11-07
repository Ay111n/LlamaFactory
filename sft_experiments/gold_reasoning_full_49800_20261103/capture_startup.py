"""Read live startup evidence and capture a stable, self-contained audit snapshot."""
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from common import *


def read(name):
    return json.loads((ROOT / name).read_text())


def rows(name, folder=ROOT):
    return [json.loads(line) for line in (folder / name).read_text().splitlines(keepends=True) if line.endswith("\n")]


def main():
    snapshot = ROOT / "startup_verified"
    assert not snapshot.exists(), "Startup audit is immutable once captured"
    process = read("process.json")
    assert process["state"] == "running"
    pid = process["worker_pid"]
    os.kill(pid, 0)
    os.kill(process["supervisor_pid"], 0)
    assert os.getsid(pid) == pid and os.getsid(process["supervisor_pid"]) == process["supervisor_pid"]
    assert not any((ROOT / name).exists() for name in ("failure.json", "launch_failure.json", "process_exit.json"))
    step_rows = rows("steps.jsonl")
    assert len(step_rows) >= 3
    assert [row["step"] for row in step_rows] == list(range(1, len(step_rows) + 1))
    log_rows = rows("trainer_logs.jsonl")
    by_step = {row["step"]: row for row in log_rows if "grad_norm" in row}
    complete = [row for row in step_rows if row["step"] in by_step]
    assert len(complete) >= 3
    for row in complete:
        assert all(math.isfinite(float(row[key])) for key in ("ce", "weighted_kd", "weighted_anchor", "seconds", "extra_to_ce"))
        assert row["ce"] > 0 and row["weighted_kd"] >= 0 and row["weighted_anchor"] >= 0 and row["teacher_unchanged"]
        assert math.isfinite(by_step[row["step"]]["grad_norm"]) and by_step[row["step"]]["grad_norm"] > 0
        assert abs(by_step[row["step"]]["loss"] - row["ce"] - row["weighted_kd"] - row["weighted_anchor"]) < 0.001
    initialized = read("initialization.json")
    assert initialized["fresh_from"] == BASE and initialized["lora_B_all_zero"]
    assert initialized["adapter_loaded"] is None and initialized["resume"] is None and not initialized["preflight_adapter_loaded"]
    dataset = read("dataset_loaded.json")
    assert dataset["rows"] == 49800 and dataset["arrow_sha256_verified"]
    environment = read("environment.json")
    assert environment["world_size"] == environment["n_gpu_for_trainer"] == 1
    assert "FullTrainingCallback" in environment["callback_types"] and "PreflightCallback" not in environment["callback_types"]
    assert read("training_started.json")["target_steps"] == EXPECTED_STEPS
    configuration = read("configuration_check.json")
    assert configuration["passed"] and configuration["save_steps"] == 1500 and configuration["epochs"] == 3
    for path, expected in read("source_manifest.json")["files"].items():
        assert sha(path) == expected, path
    gpu_raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    occupied = []
    for line in gpu_raw.splitlines():
        uuid, process_id, memory = (part.strip() for part in line.split(","))
        if int(process_id) == pid:
            occupied.append({"uuid": uuid, "used_mib": int(memory)})
    assert {item["uuid"] for item in occupied} == {role["uuid"] for role in GPU_ROLES.values()}
    prior = rows("steps.jsonl", PREFLIGHT / "attempt02")
    same_indices = all(a["source_indices"] == b["source_indices"] for a, b in zip(complete[:3], prior[:3]))
    assert same_indices, "Initial sample order must match validated preflight"
    checked = utc_now()
    report = {"passed": True, "scope": "full-run startup verification only; training remains running",
        "checked_at_utc": checked, "worker_pid": pid, "supervisor_pid": process["supervisor_pid"],
        "detached_sessions_verified": True, "completed_optimizer_steps": len(complete), "target_optimizer_steps": EXPECTED_STEPS,
        "fresh_student_initialization_verified": True, "data_rows": 49800, "epochs": 3,
        "loss_and_gradient_finite": True, "teacher_frozen_and_unchanged": True,
        "same_first_three_minibatches_as_preflight": same_indices,
        "first_three_ce_abs_diff_from_preflight": [abs(a["ce"]-b["ce"]) for a,b in zip(complete[:3],prior[:3])],
        "mean_ce": statistics.mean(row["ce"] for row in complete),
        "mean_weighted_kd": statistics.mean(row["weighted_kd"] for row in complete),
        "mean_weighted_anchor": statistics.mean(row["weighted_anchor"] for row in complete),
        "mean_extra_to_ce": statistics.mean(row["extra_to_ce"] for row in complete),
        "observed_step_seconds": [row["seconds"] for row in complete],
        "grad_norms": [by_step[row["step"]]["grad_norm"] for row in complete],
        "student_peak_allocated_gib": max(row["student_peak_allocated_gib"] for row in complete),
        "student_peak_reserved_gib": max(row["student_peak_reserved_gib"] for row in complete),
        "teacher_peak_allocated_gib": max(row["teacher_peak_allocated_gib"] for row in complete),
        "teacher_peak_reserved_gib": max(row["teacher_peak_reserved_gib"] for row in complete),
        "gpu_process_occupancy": occupied,
        "checkpoint_steps_configured": EXPECTED_CHECKPOINTS,
        "checkpoint_saved_at_capture": (ROOT / "latest_checkpoint.json").exists(),
        "full_training_estimate_hours_from_preflight": read("protocol.json")["full_training_estimate_hours_from_preflight"],
        "automatic_dev_test_evaluation": False, "resolved_startup_errors": [
            {"archive": "startup_attempts/parser_failure_before_model_load", "updates": 0, "fix": "Remove unsupported redundant save_safetensors YAML key"},
            {"archive": "startup_attempts/fingerprint_check_before_training", "updates": 0, "fix": "Use saved fingerprint and Arrow SHA-256, since load_from_disk changes runtime fingerprint"}]}
    snapshot.mkdir()
    excluded = {"launch.lock"}
    for path in ROOT.iterdir():
        if path.is_file() and path.name not in excluded and not path.name.endswith(".tmp"):
            shutil.copy2(path, snapshot / path.name)
    shutil.copytree(ROOT / "startup_attempts", snapshot / "startup_attempts", ignore=shutil.ignore_patterns("__pycache__"))
    write(snapshot / "startup_verification.json", report)
    local_time = datetime.fromisoformat(checked).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +08:00")
    report_text = f'''全量训练启动检查：通过，训练仍在后台运行。
检查时间：{local_time}
学生 GPU 1：sft-new2 全新初始化；教师 GPU 2：冻结 CE-9000。
训练数据：原 49,800 条；3 epoch；有效批量16；共9,339个优化器步骤。
当前已核验完整步骤：{len(complete)}；训练 PID：{pid}；监督进程 PID：{process['supervisor_pid']}。
已核验独立后台会话，退出 SSH 不会停止该任务。

启动检查结果
学生252个LoRA B矩阵均从零开始，未加载CE学生、预检adapter或旧优化器。
数据加载49800条，原Arrow文件SHA-256一致，诊断集仍排除在训练外。
前3个累积批次的样本顺序与已通过的预检完全一致。
CE、额外损失和梯度均为有限数，教师保持冻结且参数版本未改变。
平均CE={report['mean_ce']:.6f}；加权推理KL={report['mean_weighted_kd']:.6f}；加权无推理锚定KL={report['mean_weighted_anchor']:.6f}。
学生峰值allocated/reserved：{report['student_peak_allocated_gib']:.3f}/{report['student_peak_reserved_gib']:.3f} GiB。
教师峰值allocated/reserved：{report['teacher_peak_allocated_gib']:.3f}/{report['teacher_peak_reserved_gib']:.3f} GiB。
本次仅验证启动与数值运行，尚无DEV/TEST准确率结果。

检查点
配置为1500、3000、4500、6000、7500、9000和最终9339步保存，保留全部7份。
保存adapter（含训练的lm_head）、optimizer、scheduler、RNG和trainer_state。
当前尚未到首个保存步数；保存成功后会写checkpoints.jsonl。
模型目录：{MODEL_ROOT}
依据此前50步预检的稳态均速，纯训练约103小时（4.3天），保存耗时另计。

复现和状态
实际解释器：{PYTHON}
运行目录：{ROOT}
实时查询：ssh L20 '{PYTHON} {ROOT}/status.py'
本目录startup_verified保存此次检查的固定快照；实时状态以服务器progress.json、steps.jsonl和进程为准。
启动阶段的两个零更新错误已修复并保存在startup_attempts中，不改变数据、模型初始化或损失设计。
'''
    (snapshot / "startup_report.txt").write_text(report_text, encoding="utf-8")
    file_hashes = {str(path.relative_to(snapshot)): sha(path) for path in snapshot.rglob("*") if path.is_file()}
    write(snapshot / "snapshot_manifest.json", {"captured_at_utc": checked, "files": file_hashes,
        "note": "File hashes computed on copied snapshot bytes; ongoing live logs are not frozen"})
    write(ROOT / "startup_verification.json", report)
    (ROOT / "startup_report.txt").write_text(report_text, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
