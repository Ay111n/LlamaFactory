"""Full-run telemetry and checkpoint auditing; no preflight stop callback."""
import math
import os
from pathlib import Path
import time
import torch
from transformers import TrainerCallback
from common import *

REQUIRED_CHECKPOINT_FILES = ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth")


class FullTrainingCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None
        self.reference_versions = None
        self.last_row = None
        self.saved_steps = []

    def check_reference(self):
        model = self.trainer.reference.model
        assert self.reference_versions == [p._version for p in model.parameters()]
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())

    def check_prefetch_drained(self):
        assert self.trainer.reference.future is None and not self.trainer.reference.jobs
        assert self.trainer.active_data is None

    def on_train_begin(self, args, state, control, **kwargs):
        assert state.global_step == 0 and state.max_steps == EXPECTED_STEPS and state.num_train_epochs == 3
        assert args.save_steps == SAVE_STEPS and not args.save_only_model
        self.reference_versions = [p._version for p in self.trainer.reference.model.parameters()]
        self.check_reference()
        self.check_prefetch_drained()
        write(ROOT / "training_started.json", {"pid": os.getpid(), "started_at_utc": utc_now(),
             "global_step": 0, "target_steps": state.max_steps, "fresh_from_sft_new2": True,
             "gpu_roles": GPU_ROLES, "checkpoint_steps": EXPECTED_CHECKPOINTS,
             "cache_and_async_prefetch": True, "trainer_class": type(self.trainer).__name__})

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        self.check_reference()
        self.check_prefetch_drained()
        row = dict(trainer.window_metrics)
        row.update(step=state.global_step, total_steps=state.max_steps, epoch=state.epoch,
            seconds=time.perf_counter()-trainer.window_started, lr=trainer.optimizer.param_groups[0]["lr"],
            student_peak_allocated_gib=torch.cuda.max_memory_allocated(0)/1024**3,
            student_peak_reserved_gib=torch.cuda.max_memory_reserved(0)/1024**3,
            teacher_peak_allocated_gib=torch.cuda.max_memory_allocated(1)/1024**3,
            teacher_peak_reserved_gib=torch.cuda.max_memory_reserved(1)/1024**3,
            teacher_unchanged=True, prefetch_drained=True,
            teacher_prepared_samples=trainer.reference.prepared_samples,
            timestamp=time.time(), updated_at_utc=utc_now())
        row["extra_to_ce"] = (row["weighted_kd"]+row["weighted_anchor"])/row["ce"]
        assert all(math.isfinite(row[key]) for key in ("ce", "weighted_kd", "weighted_anchor", "seconds", "extra_to_ce"))
        self.last_row = row
        append(ROOT / "steps.jsonl", row)
        write(ROOT / "progress.json", row)
        print("FULL_TRAIN_STEP", json.dumps({k: row[k] for k in ("step", "seconds", "ce", "weighted_kd", "weighted_anchor", "extra_to_ce", "student_peak_allocated_gib", "teacher_peak_allocated_gib")}), flush=True)
        # Retain the native periodic policy and explicitly guarantee final save.
        if state.global_step % SAVE_STEPS == 0 or state.global_step == state.max_steps:
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            values = {key: float(value) if torch.is_tensor(value) else value for key, value in logs.items()}
            for key in ("loss", "grad_norm", "learning_rate"):
                if key in values:
                    assert math.isfinite(float(values[key])), (key, values[key])
            append(ROOT / "trainer_logs.jsonl", dict(step=state.global_step, timestamp=time.time(), **values))
            if self.last_row is not None and "grad_norm" in values:
                self.last_row["grad_norm"] = float(values["grad_norm"])
                write(ROOT / "progress.json", self.last_row)

    def on_save(self, args, state, control, **kwargs):
        self.check_prefetch_drained()
        assert state.global_step in EXPECTED_CHECKPOINTS
        folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        for name in REQUIRED_CHECKPOINT_FILES:
            assert (folder / name).is_file(), (folder, name)
        entry = {"step": state.global_step, "path": str(folder), "completed_at_utc": utc_now(),
                 "files": {name: (folder/name).stat().st_size for name in REQUIRED_CHECKPOINT_FILES}}
        append(ROOT / "checkpoints.jsonl", entry)
        self.saved_steps.append(state.global_step)
        write(ROOT / "latest_checkpoint.json", entry)
        print("CHECKPOINT_COMPLETE", json.dumps(entry), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        assert state.global_step == EXPECTED_STEPS
        assert self.saved_steps == EXPECTED_CHECKPOINTS, self.saved_steps
        self.check_reference()
        self.check_prefetch_drained()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.check_prefetch_drained()
