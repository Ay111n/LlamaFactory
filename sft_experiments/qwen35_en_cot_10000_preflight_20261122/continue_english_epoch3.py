"""Resume a matched CoT LoRA variant for two more epochs."""
import json
import math
import sys
import zipfile
from pathlib import Path

from transformers import TrainerCallback

HERE = Path(__file__).resolve().parent
VARIANTS = ("empty", "bo", "en")
START_STEP = 625
END_STEP = 1875
WARMUP = 32


class ContinuationSchedule(TrainerCallback):
    def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
        from torch.optim.lr_scheduler import LambdaLR

        assert state.global_step == START_STEP and state.max_steps == END_STEP
        assert state.epoch == 1.0 and optimizer.state
        scheduler = getattr(lr_scheduler, "scheduler", lr_scheduler)
        assert isinstance(scheduler, LambdaLR) and scheduler.last_epoch == START_STEP
        assert all(math.isclose(x, args.learning_rate) for x in scheduler.base_lrs)
        state.save_steps = int(args.save_steps)

        def factor(step):
            elapsed = max(0, step - START_STEP)
            if elapsed < WARMUP:
                return elapsed / WARMUP
            progress = min(1.0, (elapsed - WARMUP) / (END_STEP - START_STEP - WARMUP))
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler.lr_lambdas = [factor] * len(scheduler.base_lrs)
        start_lrs = [base * factor(START_STEP) for base in scheduler.base_lrs]
        for group, lr in zip(scheduler.optimizer.param_groups, start_lrs, strict=True):
            group["lr"] = lr
        scheduler._last_lr = start_lrs
        print(f"RESUMED optimizer_state_entries={len(optimizer.state)} global_step={state.global_step} "
              f"max_steps={state.max_steps} save_steps={state.save_steps} "
              f"continuation_warmup={WARMUP} peak_lr={args.learning_rate}", flush=True)


def main(variant):
    run = HERE / "runtime_three" / variant
    config = json.loads((run / "fresh_config.json").read_text())
    output = Path(config["output_dir"])
    checkpoint = output / "checkpoint-625"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert state["global_step"] == state["max_steps"] == START_STEP and state["epoch"] == 1.0
    assert config["dataset"] == f"{variant}_cot_10000" and config["cutoff_len"] == 8192
    assert config["per_device_train_batch_size"] == 1 and config["gradient_accumulation_steps"] == 16
    assert config["save_steps"] == 200 and config["save_total_limit"] == 1 and not config["save_only_model"]
    assert Path(config["tokenized_path"]).is_dir()
    assert (checkpoint / "adapter_model.safetensors").stat().st_size > 0
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        with zipfile.ZipFile(checkpoint / filename) as archive:
            assert archive.infolist()
    destination = output.with_name(f"qwen35_{variant}_cot_10000_8192_epoch3")
    assert not destination.exists(), destination
    config.update(output_dir=str(destination), resume_from_checkpoint=str(checkpoint), num_train_epochs=3)
    (run / "epoch3_config.json").write_text(json.dumps(config, indent=2) + "\n")
    sys.path.insert(0, str(HERE.parent / "qwen35_crossref_teacher_9960_20261111"))
    from memory_ce import install
    from llamafactory.train.tuner import run_exp
    install(warmup=False)
    run_exp(config, callbacks=[ContinuationSchedule()])
    final = destination / "checkpoint-1875"
    end = json.loads((final / "trainer_state.json").read_text())
    assert end["epoch"] == 3.0 and end["global_step"] == end["max_steps"] == END_STEP
    for filename in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        assert (final / filename).stat().st_size > 0
    (run / "epoch3_completion.json").write_text(json.dumps(
        dict(checkpoint=str(final), epoch=end["epoch"], global_step=end["global_step"]), indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] not in VARIANTS):
        raise SystemExit("Expected one of: empty, bo, en")
    main(sys.argv[1] if len(sys.argv) == 2 else "en")
