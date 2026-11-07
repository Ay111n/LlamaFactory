"""Freeze configuration and verify inputs without training or modifying sources."""
import json
from pathlib import Path
import shutil
import yaml
from common import *


def main():
    if (ROOT / "prepared.json").exists():
        raise FileExistsError("Preparation already frozen; do not silently replace it")
    latest = json.loads((PREFLIGHT / "LATEST.json").read_text())
    assert latest["passed"] and latest["optimizer_updates"] == 50
    for name in ("losses.py", "reasoning_trainer.py"):
        assert sha(PREFLIGHT / name) == latest["core_code_sha256"][name]
    assert json.loads((PREFLIGHT / "attempt02/ce_equivalence.json").read_text())["passed"]
    assert json.loads((PREFLIGHT / "attempt02/long_sequence_memory.json").read_text())["passed"]
    optimization_manifest = json.loads((OPTIMIZATION / "delivery_manifest.json").read_text())["files"]
    for name in ("optimized.py", "equivalence/completion.json", "pipeline/completion.json", "comparison.json"):
        assert sha(OPTIMIZATION / name) == optimization_manifest[name], name
    equivalence = json.loads((OPTIMIZATION / "equivalence/completion.json").read_text())
    pipeline = json.loads((OPTIMIZATION / "pipeline/completion.json").read_text())
    assert equivalence["passed"] and equivalence["all_trainable_parameters"] == 835190784
    assert pipeline["passed"] and pipeline["optimizer_updates"] == 12
    assert pipeline["long_sequence_probe"]["passed"] and pipeline["teacher_unchanged"]
    assert all(row["gradients"] is None or row["gradients"]["maximum_absolute_error"] == 0 for row in equivalence["windows"])
    repair = json.loads((RECOVERY / "equivalence/completion.json").read_text())
    assert repair["passed"] and repair["optimizer_updates"] == 0 and repair["all_trainable_parameters"] == 835190784
    assert all(row["gradients"] is None or row["gradients"]["maximum_absolute_error"] == 0 for row in repair["windows"])
    assert json.loads((RECOVERY / "copy_lifetime_safe.json").read_text())["passed"]
    assert not json.loads((RECOVERY / "copy_lifetime_original.json").read_text())["passed"]
    baseline = yaml.safe_load((FRAMEWORK / "examples/train_lora/qwen3_mixed_sft_49800_lora_v1.yaml").read_text())
    cfg = dict(baseline)
    cfg.update(output_dir=str(MODEL_ROOT), remove_unused_columns=False, data_seed=3407,
        optim="adamw_torch", packing=False, train_on_prompt=False, logging_steps=1,
        disable_tqdm=True, report_to="none", save_strategy="steps", save_steps=SAVE_STEPS,
        save_total_limit=10, save_only_model=False,
        num_train_epochs=3.0, max_steps=-1, resume_from_checkpoint=None, overwrite_output_dir=False)
    assert cfg["model_name_or_path"] == BASE and cfg.get("adapter_name_or_path") is None
    assert cfg["per_device_train_batch_size"] == 1 and cfg["gradient_accumulation_steps"] == 16
    assert cfg["lora_rank"] == 64 and cfg["lora_alpha"] == 128 and cfg["additional_target"] == "lm_head"
    assert cfg["learning_rate"] == 1e-4 and cfg["warmup_ratio"] == .05 and cfg["cutoff_len"] == 4096
    old_cfg = json.loads((PREFLIGHT / "attempt02/train_config.json").read_text())
    allowed = {"output_dir", "save_strategy", "save_steps", "save_total_limit", "save_only_model", "max_steps"}
    changes = {key: {"preflight": old_cfg.get(key), "full": cfg.get(key)} for key in set(old_cfg) | set(cfg) if old_cfg.get(key) != cfg.get(key)}
    assert set(changes).issubset(allowed), changes
    dataset_manifest = json.loads((DATA / "manifest.json").read_text())
    state = json.loads((DATA / "dataset/state.json").read_text())
    assert dataset_manifest["rows"] == 49800 and dataset_manifest["input_ids_and_labels_exactly_match_original"]
    assert dataset_manifest["heldout_uid_and_exact_question_overlap"] == 0
    assert state["_fingerprint"] == dataset_manifest["dataset_fingerprint"]
    assert sha(SOURCE) == dataset_manifest["source_sha256"]
    inputs = {str(SOURCE): sha(SOURCE), str(DATA / "manifest.json"): sha(DATA / "manifest.json")}
    for item in state["_data_files"]:
        path = DATA / "dataset" / item["filename"]
        inputs[str(path)] = sha(path)
    inputs[str(DATA / "dataset/state.json")] = sha(DATA / "dataset/state.json")
    inputs[str(PREFLIGHT / "training_exclusion.json")] = sha(PREFLIGHT / "training_exclusion.json")
    assert inputs[str(PREFLIGHT / "training_exclusion.json")] == dataset_manifest["heldout_exclusion_sha256"]
    prior_provenance = json.loads((PREFLIGHT / "provenance.json").read_text())
    for record in prior_provenance["records"]:
        assert sha(record["path"]) == record["sha256"], record["path"]
        inputs[record["path"]] = record["sha256"]
        print("VERIFIED", Path(record["path"]).name, flush=True)
    assert shutil.disk_usage(MODEL_ROOT.parent.parent).free > 90 * 1024**3
    (ROOT / "full_training.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write(ROOT / "config_difference_from_preflight.json", changes)
    write(ROOT / "input_manifest.json", {"files": inputs, "rows": 49800, "teacher": REFERENCE, "student": BASE,
          "tokenizer_behavior_identical_in_preflight": prior_provenance["tokenizer_behavior_identical"], "verified_at_utc": utc_now()})
    sources = {str(p): sha(p) for p in ROOT.glob("*.py")}
    sources.update({str(PREFLIGHT / name): sha(PREFLIGHT / name) for name in ("losses.py", "reasoning_trainer.py")})
    sources[str(OPTIMIZATION / "optimized.py")] = sha(OPTIMIZATION / "optimized.py")
    for name in ("safe_pipeline.py", "validate_repair.py", "copy_lifetime_probe.py", "equivalence/completion.json", "copy_lifetime_safe.json", "copy_lifetime_original.json"):
        sources[str(RECOVERY / name)] = sha(RECOVERY / name)
    sources[str(ROOT / "full_training.yaml")] = sha(ROOT / "full_training.yaml")
    write(ROOT / "source_manifest.json", {"files": sources, "shared_loss_implementation": str(PREFLIGHT), "shared_optimized_implementation": str(OPTIMIZATION / "optimized.py")})
    write(ROOT / "protocol.json", {"authorized": "User explicitly requested 49,800-row full training using two L20 GPUs with caching and asynchronous prefetch",
        "student": BASE, "reference": REFERENCE, "fresh_initialization": True, "resume": None,
        "gpu_roles": GPU_ROLES, "epochs": 3, "expected_optimizer_steps": EXPECTED_STEPS,
        "checkpoint_steps": EXPECTED_CHECKPOINTS, "checkpoint_contains_optimizer_scheduler_rng": True,
        "lambda": .1, "mu": .1, "beta": .5, "cap": 2., "reference_dtype": "float32",
        "student_training": "original BF16 LoRA plus lm_head", "auxiliary_normalization": "per-answer mean, averaged over all samples per accumulation window; no_think contributes zero",
        "diagnostic_set_updates": False, "automatic_dev_test_evaluation": False,
        "cache_and_async_prefetch": True, "max_prefetched_samples": 1,
        "teacher_seconds_metric": "CPU wait for a prepared target, not total teacher GPU compute",
        "full_training_estimate_hours_from_preflight": pipeline["steady_mean_seconds"] * EXPECTED_STEPS / 3600,
        "optimization_validation": str(OPTIMIZATION), "optimized_code_sha256": sha(OPTIMIZATION / "optimized.py"),
        "previous_full_run_continued": False,
        "recovery_validation": str(RECOVERY),
        "repair_scope": "Explicit non-default peer-copy writer stream recorded on source and destination storage; unchanged FP32 targets and loss/gradient arithmetic",
        "failed_run_retained": str(ROOT.parent / "gold_reasoning_async_full_49800_20261103"),
        "baseline_config": str(FRAMEWORK / "examples/train_lora/qwen3_mixed_sft_49800_lora_v1.yaml")})
    write(ROOT / "prepared.json", {"passed": True, "created_at_utc": utc_now(), "source_files": len(sources), "input_files": len(inputs)})
    print("FULL_RUN_PREPARED", json.dumps(changes), flush=True)


if __name__ == "__main__":
    main()
