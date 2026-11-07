"""Fresh sft-new2 student, frozen CE-9000, original 49,800 rows, three epochs."""
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, Trainer, set_seed
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from common import *
from full_callback import FullTrainingCallback, REQUIRED_CHECKPOINT_FILES

# Reuse the exact validated implementation; do not duplicate or edit its losses.
sys.path.insert(0, str(PREFLIGHT))
sys.path.insert(0, str(OPTIMIZATION))
sys.path.insert(0, str(RECOVERY))
from reasoning_trainer import ReasoningCollator
from safe_pipeline import SafePipelineTargets as PipelineTargets, PipelineTrainer


def main():
    assert sys.executable == PYTHON
    assert json.loads((ROOT / "prepared.json").read_text())["passed"]
    for path, expected in json.loads((ROOT / "source_manifest.json").read_text())["files"].items():
        assert sha(path) == expected, path
    if MODEL_ROOT.exists() and any(MODEL_ROOT.iterdir()):
        raise FileExistsError("Full model output must be empty; no implicit resume or overwrite")
    cfg = yaml.safe_load((ROOT / "full_training.yaml").read_text())
    assert cfg["model_name_or_path"] == BASE and cfg.get("adapter_name_or_path") is None
    assert cfg["resume_from_checkpoint"] is None and cfg["num_train_epochs"] == 3
    dataset = load_from_disk(str(DATA / "dataset"))
    manifest = json.loads((DATA / "manifest.json").read_text())
    saved_state = json.loads((DATA / "dataset/state.json").read_text())
    input_hashes = json.loads((ROOT / "input_manifest.json").read_text())["files"]
    # datasets.load_from_disk applies with_format(), which changes its runtime
    # fingerprint. Verify the persisted state and actual Arrow bytes instead.
    assert len(dataset) == manifest["rows"] == 49800
    assert saved_state["_fingerprint"] == manifest["dataset_fingerprint"]
    expected_arrow = {str(DATA / "dataset" / item["filename"]) for item in saved_state["_data_files"]}
    assert {item["filename"] for item in dataset.cache_files} == expected_arrow
    for path in expected_arrow | {str(DATA / "dataset/state.json"), str(DATA / "manifest.json")}:
        assert sha(path) == input_hashes[path], path
    write(ROOT / "dataset_loaded.json", {"rows": len(dataset), "saved_fingerprint": saved_state["_fingerprint"],
        "loaded_fingerprint": dataset._fingerprint, "arrow_sha256_verified": True,
        "explanation": "load_from_disk applies with_format and recomputes the runtime fingerprint",
        "checked_at_utc": utc_now()})
    assert torch.cuda.device_count() == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ",".join(GPU_ROLES[role]["uuid"] for role in ("student", "teacher"))
    assert os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "0"
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    with patch("torch.cuda.device_count", return_value=1):
        ma, da, ta, fa, ga = get_train_args(cfg)
    ta._n_gpu = 1
    assert ta.world_size == 1 and ma.adapter_name_or_path is None
    set_seed(3407)
    tokenmod = load_tokenizer(ma)
    token = tokenmod["tokenizer"]
    template = get_template_and_fix_tokenizer(token, da)
    student = load_model(token, ma, fa, True)
    assert {str(p.device) for p in student.parameters()} == {"cuda:0"}
    trainable = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _, p in trainable) == 835190784
    low_rank_b = [(n, p) for n, p in trainable if "lora_B" in n]
    assert low_rank_b and all(int(torch.count_nonzero(p)) == 0 for _, p in low_rank_b)
    sample_parameters = [(n, p) for n, p in trainable if "lora_B" in n or "lm_head.modules_to_save" in n]
    initial_values = {n: p.detach().reshape(-1)[:1024].float().cpu().clone() for n, p in sample_parameters}
    write(ROOT / "initialization.json", {"fresh_from": BASE, "adapter_loaded": None, "resume": None,
        "lora_B_all_zero": True, "lora_B_matrices": len(low_rank_b), "trainable_parameters": sum(p.numel() for _, p in trainable),
        "preflight_adapter_loaded": False, "seed": 3407, "timestamp_utc": utc_now()})
    native_collator = SFTDataCollatorWith4DAttentionMask(template=template, model=student,
        pad_to_multiple_of=8, label_pad_token_id=-100, block_diag_attn=False, neat_packing=False,
        attn_implementation=getattr(student.config, "_attn_implementation", None), compute_dtype=ma.compute_dtype, **tokenmod)
    callback = FullTrainingCallback()
    trainer = PipelineTrainer(model=student, args=ta, finetuning_args=fa,
        data_collator=ReasoningCollator(native_collator), train_dataset=dataset,
        callbacks=[callback], kd_coef=.1, anchor_coef=.1, chunk=64, **tokenmod)
    callback.trainer = trainer
    assert all(type(cb).__name__ != "PreflightCallback" for cb in trainer.callback_handler.callbacks)
    reference = AutoModelForCausalLM.from_pretrained(REFERENCE, local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa", device_map={"": "cuda:1"})
    reference.eval().requires_grad_(False)
    assert {str(p.device) for p in reference.parameters()} == {"cuda:1"}
    # Establish model-load completion before the separate teacher work stream.
    torch.cuda.synchronize(1)
    trainer.reference = PipelineTargets(reference, beta=.5, cap=2., chunk=64)
    write(ROOT / "environment.json", {"python": sys.executable,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate", "datasets", "flash-attn")},
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_roles": GPU_ROLES,
        "world_size": ta.world_size, "n_gpu_for_trainer": ta.n_gpu, "tf32": False,
        "callback_types": [type(cb).__name__ for cb in trainer.callback_handler.callbacks],
        "trainer_source": inspect.getfile(PipelineTrainer), "reference_source": inspect.getfile(PipelineTargets),
        "trainer_class": type(trainer).__name__, "reference_class": type(trainer.reference).__name__,
        "cache_and_async_prefetch": True, "max_prefetched_samples": 1,
        "native_save_checkpoint_source": inspect.getsource(Trainer._save_checkpoint)})
    torch.cuda.empty_cache()
    started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=None)
    finally:
        trainer.reference.close()
    assert trainer.state.global_step == EXPECTED_STEPS
    callback.check_reference()
    for step in EXPECTED_CHECKPOINTS:
        for filename in REQUIRED_CHECKPOINT_FILES:
            assert (MODEL_ROOT / f"checkpoint-{step}" / filename).is_file()
    changes = [float((p.detach().reshape(-1)[:1024].float().cpu() - initial_values[n]).abs().max()) for n, p in sample_parameters]
    assert max(changes) > 0
    trainer.save_state()
    write(ROOT / "completion.json", {"passed": True, "optimizer_updates": trainer.state.global_step,
        "epochs": trainer.state.epoch, "training_seconds": time.perf_counter()-started,
        "checkpoint_steps": callback.saved_steps, "final_checkpoint": str(MODEL_ROOT / f"checkpoint-{EXPECTED_STEPS}"),
        "teacher_unchanged": True, "student_parameters_changed": True, "sampled_parameter_max_change": max(changes),
        "metrics": result.metrics, "completed_at_utc": utc_now()})
    print("FULL_TRAINING_COMPLETED", EXPECTED_STEPS, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(ROOT / "failure.json", {"type": type(error).__name__, "error": str(error),
              "traceback": traceback.format_exc(), "timestamp_utc": utc_now()})
        raise
