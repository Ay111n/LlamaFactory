"""Capture recovery evidence without touching model artifacts or prior logs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "capture_status.py")], check=True)
snapshot = json.loads((ROOT / "recovery_snapshot.json").read_text())
hardware = subprocess.check_output(["nvidia-smi", "-i", "1,2", "-q", "-d", "ECC,ROW_REMAPPER"], text=True)
(ROOT / "hardware_ecc.txt").write_text(hardware, encoding="utf-8")
equivalence = json.loads((ROOT / "equivalence/completion.json").read_text())
full_name = "gold_reasoning_async_full_repaired_49800_20261103"
failed_name = "gold_reasoning_async_full_49800_20261103"
step = snapshot["training_steps"]
mean = snapshot.get("recent_mean_step_seconds", 0)
hours = snapshot.get("remaining_hours_estimate", 0)
errors = [r["gradients"]["maximum_absolute_error"] for r in equivalence["windows"] if r["gradients"]]
lines = [
    f"检查时间（UTC）：{snapshot['checked_at_utc']}",
    "原全量训练于 2026-11-03 07:27:03 UTC（北京时间 15:27:03）报错，完成 19/9339 步，在第 20 步反向传播期间退出；尚未保存任何检查点。",
    "原错误：CUDA error: unspecified launch failure。内核记录为教师 GPU 2、原训练 PID 2287813 的 Xid 31，CE3 拷贝引擎 FAULT_PDE ACCESS_TYPE_VIRT_WRITE。两卡检查时 ECC 错误及重映射错误计数均为 0。",
    "在相同两卡、PyTorch 2.6.0+cu124 环境中，独立复现了跨卡拷贝目标张量提前释放、随后回收显存导致的相同 CUDA 719 和相同类型 Xid 31。复现 PID 2483241；这是有意进行的诊断，不是修复后训练再次失败。",
    "修复：使用独立的非默认拷贝流，并将源、目标张量都登记到实际执行拷贝的流，延迟显存回收直至拷贝完成。保留 FP32 全词表缓存、单样本异步预取和原 KL 实现，未修改训练目标、数据、优化器和精度。",
    "证据边界：已证明原传输路径存在可复现的显存生命周期缺陷，故障特征与训练一致。原进程没有保存崩溃瞬间的分配器快照，不能声称已经逐条复原其唯一触发时序。",
    "修复后的显存回收压力测试连续 10 次通过，FP32 传输逐元素完全相同。",
    f"实际 8B 验证：最长样本相邻混合批次、原第 20 步全部样本、纯 no_think 批次，损失相同；全部 {equivalence['all_trainable_parameters']} 个可训练参数梯度最大绝对差均为 {max(errors)}；随机数状态一致；验证无优化器更新。",
    f"新全量运行：{full_name}；当前完成 {step}/9339 步；进程身份核验={snapshot.get('worker_alive_with_matching_entry')}；原失败运行保留在 {failed_name}。",
    "仍使用 GPU 1（学生）和 GPU 2（冻结教师 CE-9000）。学生从 sft-new2 重新初始化，原 49800 条数据、3 epochs、有效 batch 16；每 1500 步及结束保存，未载入诊断 adapter 或旧失败运行状态。",
    f"已比对前 {snapshot.get('matched_source_order_steps',0)} 步的样本索引和学习率：样本顺序一致={snapshot.get('source_order_matches_preflight')}，学习率一致={snapshot.get('lr_matches_preflight')}。教师全程冻结={snapshot.get('teacher_frozen_every_step')}；每步预取队列正常排空={snapshot.get('prefetch_drained_every_step')}。",
    f"最近实测均值 {mean:.2f} 秒/步；剩余训练粗估 {hours:.1f} 小时，尚未计入检查点写盘，长样本会引起波动。",
    f"本次已观察阶段峰值 allocated：学生 {snapshot.get('peak_student_gib',0):.2f} GiB，教师 {snapshot.get('peak_teacher_gib',0):.2f} GiB。重启后内核新增 Xid：{snapshot.get('xid_lines_since_restart')}。",
    "复现脚本、错误原文、梯度对比、源代码哈希与运行快照随本报告保留；未启动 DEV/TEST 测评。",
    "参考：PyTorch v2.6.0 Copy.cu https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/Copy.cu",
    "同类上游报告（不同软硬件版本，不能单独作为本机根因证明）：https://github.com/pytorch/pytorch/issues/196257",
]
(ROOT / "recovery_report.txt").write_text("\n\n".join(lines)+"\n", encoding="utf-8")
for name, reason in [(failed_name, "第 20 步期间 CUDA 报错，最后完成 19 步，无检查点"),
                     ("gold_reasoning_full_49800_20261103", "已按用户要求停止，以优化吞吐")]:
    directory = ROOT.parent / name
    if directory.is_dir():
        (directory / "CURRENT_STATUS.txt").write_text(
            f"本目录为历史运行：{reason}。\n当前运行及实时状态入口：../{full_name}/status.py\n"
            f"故障诊断与恢复报告：../{ROOT.name}/recovery_report.txt\n"
            f"导航更新时间 UTC：{snapshot['checked_at_utc']}\n", encoding="utf-8")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
bundle = Path(tempfile.mkdtemp(prefix=f"gold_reasoning_recovery_{stamp}_"))
names = [failed_name, ROOT.name, full_name]
manifest = {}
for name in names:
    source = ROOT.parent / name
    for path in source.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.name.endswith((".tmp", ".lock")):
            continue
        relative = Path(name) / path.relative_to(source)
        dest = bundle / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        manifest[str(relative)] = hashlib.sha256(dest.read_bytes()).hexdigest()
navigation = ROOT.parent / "gold_reasoning_full_49800_20261103/CURRENT_STATUS.txt"
relative = Path("gold_reasoning_full_49800_20261103/CURRENT_STATUS.txt")
(bundle / relative).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(navigation, bundle / relative)
manifest[str(relative)] = hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
(bundle / "export_manifest.json").write_text(json.dumps({"created_at_utc": stamp,"files": manifest}, indent=2)+"\n")
archive = Path(str(bundle)+".tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    for child in bundle.iterdir():
        tar.add(child, arcname=child.name)
print("EXPORT", json.dumps({"archive": str(archive), "snapshot_directory":str(bundle), "files": len(manifest), "sha256":hashlib.sha256(archive.read_bytes()).hexdigest()}), flush=True)
