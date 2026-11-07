"""Actual-model, all-parameter gradient equivalence for the copy lifetime fix."""
import gc
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
for name in ("gold_reasoning_preflight_20261103", "gold_reasoning_throughput_20261103"):
    sys.path.insert(0, str(ROOT.parent / name))
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
import torch
from transformers import AutoModelForCausalLM
import run_preflight as original
from reasoning_trainer import ReasoningTrainer
from losses import ReferenceTargets
from safe_pipeline import SafePipelineTargets, PipelineTrainer


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    assert json.loads((ROOT / "copy_lifetime_safe.json").read_text())["passed"]
    original.ROOT = ROOT
    original.MODEL_ROOT = Path("/home/nlp/data/lyl/finetuning/model/sft_experiments/gold_reasoning_async_recovery_20261103")
    trainer, _, _, directory = original.build("validate", "equivalence", 1)
    trainer.model = trainer.accelerator.prepare_model(trainer.model)
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs)
    teacher = AutoModelForCausalLM.from_pretrained(original.REF, local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa", device_map={"": "cuda:1"})
    teacher.eval().requires_grad_(False)
    torch.cuda.synchronize(1)
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    versions = [p._version for p in params]
    tv = [p._version for p in teacher.parameters()]
    audit = [json.loads(line) for line in (original.DATA / "audit.jsonl").open()]
    think = [r["index"] for r in audit if r["think"]]
    no = [r["index"] for r in audit if not r["think"]]
    old20 = next(json.loads(line) for line in (ROOT.parent / "gold_reasoning_preflight_20261103/attempt02/steps.jsonl").open() if json.loads(line)["step"] == 20)
    windows = [("adjacent_extremes", [25190, 31985]+think[:2]+no[:2]),
               ("original_step20", old20["source_indices"]), ("no_think", no[2:4])]
    reports = []
    for name, indices in windows:
        batches = [trainer.data_collator([trainer.train_dataset[i]]) for i in indices]
        count = trainer._get_num_items_in_batch(batches, trainer.args.device)
        losses = []
        for variant, cls in [("baseline", ReferenceTargets), ("safe_pipeline", SafePipelineTargets)]:
            trainer.__class__ = ReasoningTrainer if variant == "baseline" else PipelineTrainer
            trainer.reference = cls(teacher, beta=.5, cap=2., chunk=64)
            trainer.model.zero_grad(set_to_none=True)
            trainer.current_gradient_accumulation_steps = len(batches)
            trainer.start_window(batches, count)
            with torch.random.fork_rng(devices=[0, 1]):
                if variant == "safe_pipeline":
                    trainer.reference.stage(batches)
                loss = sum(float(trainer.training_step(trainer.model, b, num_items_in_batch=count)) for b in batches)
                rng = torch.cuda.get_rng_state(0).cpu()
            losses.append(loss)
            if variant == "baseline":
                copied = original.cpu_gradient_copy(params)
                expected_rng = rng
                comparison = None
            else:
                comparison = original.compare_gradients(copied, params)
                assert math.isclose(loss, losses[0], rel_tol=2e-4, abs_tol=2e-6)
                assert torch.equal(rng, expected_rng)
                assert trainer.reference.future is None and not trainer.reference.jobs
            row = {"window": name, "indices": indices, "variant": variant, "loss": loss,
                "gradients": comparison, "same_rng": True, "passed": True,
                "metrics": dict(trainer.window_metrics),
                "student_peak_gib": torch.cuda.max_memory_allocated(0)/2**30,
                "teacher_peak_gib": torch.cuda.max_memory_allocated(1)/2**30}
            reports.append(row)
            write(directory / "partial.json", reports)
            print("REPAIR_EQUIVALENCE", json.dumps({k: row[k] for k in ["window", "variant", "loss", "gradients", "passed"]}), flush=True)
            if hasattr(trainer.reference, "close"):
                trainer.reference.close()
            trainer.model.zero_grad(set_to_none=True)
            gc.collect()
        del copied
    assert versions == [p._version for p in params] and tv == [p._version for p in teacher.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    write(directory / "completion.json", {"passed": True, "actual_8B": True, "all_trainable_parameters": sum(p.numel() for p in params),
        "optimizer_updates": 0, "student_teacher_unchanged": True, "windows": reports})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(ROOT / "validation_failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise
