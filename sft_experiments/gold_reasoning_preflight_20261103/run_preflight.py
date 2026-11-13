"""Actual native-CE equivalence and 50/100-step two-L20 training preflight.

The scheduler has the original 3-epoch horizon. A hard-stop callback ends the
run at 50/100 optimizer updates; there is no full-training launcher here.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, Trainer, set_seed
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from losses import ReferenceTargets
from reasoning_trainer import ReasoningTrainer, ReasoningCollator, PreflightCallback
from prepare_data import DATA, SOURCE, EXPECTED_SOURCE, sha, write

ROOT = Path(__file__).resolve().parent
LF = ROOT.parent.parent
REF = "/home/nlp/data/lyl/finetuning/model/sft/en_cot_49800/merged/checkpoint-9000"
MODEL_ROOT = Path("/home/nlp/data/lyl/finetuning/model/sft_experiments/gold_reasoning_preflight_20261103")
ORIGINAL = LF / "examples/train_lora/qwen3_mixed_sft_49800_lora_v1.yaml"


def build(stage, run_id, stop_steps):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert torch.cuda.device_count() == 2, "Exactly one student and one teacher GPU must be visible"
    cfg = yaml.safe_load(ORIGINAL.read_text())
    cfg.update(output_dir=str(MODEL_ROOT / run_id / stage), remove_unused_columns=False,
               data_seed=3407, optim="adamw_torch", packing=False, train_on_prompt=False,
               save_strategy="no", logging_steps=1, disable_tqdm=True, report_to="none")
    # Prevent native Trainer's single-process DataParallel auto-detection.
    # Both GPUs remain physically visible for explicit teacher device placement.
    with patch("torch.cuda.device_count", return_value=1):
        ma, da, ta, fa, ga = get_train_args(cfg)
    ta._n_gpu = 1
    set_seed(3407)
    tokenmod = load_tokenizer(ma)
    token = tokenmod["tokenizer"]
    template = get_template_and_fix_tokenizer(token, da)
    model = load_model(token, ma, fa, True)
    assert {str(p.device) for p in model.parameters()} == {"cuda:0"}
    native_collator = SFTDataCollatorWith4DAttentionMask(template=template, model=model,
        pad_to_multiple_of=8, label_pad_token_id=-100, block_diag_attn=False, neat_packing=False,
        attn_implementation=getattr(model.config, "_attn_implementation", None), compute_dtype=ma.compute_dtype, **tokenmod)
    directory = ROOT / run_id
    directory.mkdir(parents=True, exist_ok=True)
    callback = PreflightCallback(stop_steps, directory)
    trainer = ReasoningTrainer(model=model, args=ta, finetuning_args=fa,
        data_collator=ReasoningCollator(native_collator), train_dataset=load_from_disk(str(DATA / "dataset")),
        callbacks=[callback], **tokenmod)
    callback.trainer = trainer
    layout = [(n, list(p.shape), str(p.dtype)) for n, p in model.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 835190784, trainable
    assert len(trainer.train_dataset) == 49800 and ta.gradient_accumulation_steps == 16
    write(directory / f"{stage}_config.json", cfg)
    write(directory / "trainable_parameters.json", {"count": trainable, "layout": layout})
    write(directory / "environment.json", {
        "python": sys.executable, "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate", "datasets", "flash-attn")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "student_logical_device": 0, "teacher_logical_device": 1,
        "world_size": ta.world_size, "n_gpu_for_trainer": ta.n_gpu, "tf32": False,
        "native_training_step": inspect.getsource(Trainer.training_step),
        "native_compute_loss": inspect.getsource(Trainer.compute_loss),
        "accelerator_backward": inspect.getsource(type(trainer.accelerator).backward),
        "historical_runtime_identical_claimed": False,
        "comparison_scope": "Current recoverable native LF CE implementation with original YAML and actual 8B model"})
    return trainer, tokenmod, native_collator, directory


def cpu_gradient_copy(params):
    return [None if p.grad is None else p.grad.detach().float().cpu().clone() for p in params]


def compare_gradients(expected, params):
    max_error = sq_error = sq_reference = dot = sq_new = 0.0
    for original, param in zip(expected, params):
        current = param.grad
        assert (original is None) == (current is None)
        if original is None:
            continue
        current = current.detach().float().cpu().reshape(-1)
        original = original.reshape(-1)
        for start in range(0, original.numel(), 1048576):
            x, y = original[start:start+1048576], current[start:start+1048576]
            torch.testing.assert_close(y, x, rtol=2e-4, atol=2e-6)
            xd, yd = x.double(), y.double()
            max_error = max(max_error, float((x-y).abs().max()))
            sq_error += float((xd-yd).square().sum())
            sq_reference += float(xd.square().sum())
            sq_new += float(yd.square().sum())
            dot += float((xd*yd).sum())
    return {"maximum_absolute_error": max_error, "relative_l2_error": math.sqrt(sq_error / max(sq_reference, 1e-30)),
            "norm_ratio": math.sqrt(sq_new / max(sq_reference, 1e-30)),
            "cosine": dot / max(math.sqrt(sq_new*sq_reference), 1e-30)}


def validation(run_id):
    # FA2's default backward uses nondeterministic reductions. The equivalence
    # gate compares both native and new implementations with deterministic FA2;
    # the separate training process retains the original default setting.
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    trainer, tokenmod, native_collator, directory = build("validate", run_id, 50)
    trainer.kd_coef = trainer.anchor_coef = 0.0
    trainer.model = trainer.accelerator.prepare_model(trainer.model)
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs)
    native = CustomSeq2SeqTrainer(model=trainer.model, args=copy.deepcopy(trainer.args),
        finetuning_args=copy.deepcopy(trainer.finetuning_args), data_collator=native_collator,
        train_dataset=trainer.train_dataset, **tokenmod)
    native.current_gradient_accumulation_steps = trainer.current_gradient_accumulation_steps = 16
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    versions = [p._version for p in params]
    audit = [json.loads(l) for l in (DATA / "audit.jsonl").open()]
    think = [r["index"] for r in audit if r["think"]]
    no_think = [r["index"] for r in audit if not r["think"]]
    longest = max((r for r in audit if r["think"]), key=lambda r: r["length"])["index"]
    most_answer = max((r for r in audit if r["think"]), key=lambda r: r["answer_tokens"])["index"]
    selected_think = list(dict.fromkeys([longest, most_answer] + think))[:8]
    assert all(audit[i]["think"] for i in selected_think)
    windows = [("mixed_16_including_longest", selected_think + no_think[:8]), ("all_no_think_tail_8", no_think[8:16])]
    reports = []
    for name, indices in windows:
        batches = [trainer.data_collator([trainer.train_dataset[i]]) for i in indices]
        count = native._get_num_items_in_batch(batches, native.args.device)
        losses, rngs, copied = [], [], None
        for repeat_index, current in enumerate((native, native, trainer)):
            trainer.model.zero_grad(set_to_none=True)
            current.current_gradient_accumulation_steps = len(batches)
            if current is trainer:
                trainer.start_window(batches, count)
            total = 0.0
            with torch.random.fork_rng(devices=[0, 1]):
                for batch in batches:
                    item = dict(batch)
                    if current is native:
                        item.pop("meta")
                    total += float(current.training_step(trainer.model, item, num_items_in_batch=count))
                rngs.append(torch.cuda.get_rng_state(0).cpu())
            losses.append(total)
            if repeat_index == 0:
                copied = cpu_gradient_copy(params)
            elif repeat_index == 1:
                native_repeat = compare_gradients(copied, params)
            else:
                gradients = compare_gradients(copied, params)
        assert all(math.isclose(losses[0], v, rel_tol=2e-4, abs_tol=2e-6) for v in losses[1:])
        assert all(torch.equal(rngs[0], v) for v in rngs[1:]) and versions == [p._version for p in params]
        row = {"name": name, "indices": indices, "microbatches": len(batches), "native_loss": losses[0],
               "native_repeat_loss": losses[1], "new_loss_aux_disabled": losses[2], "ce_tokens": int(count), "max_length": max(audit[i]["length"] for i in indices),
               "same_rng": True, "native_repeat_gradients": native_repeat, "gradients": gradients, "passed": True}
        reports.append(row)
        write(directory / "ce_equivalence_partial.json", reports)
        print("CE_EQUIVALENCE", json.dumps(row), flush=True)
        del copied, batches
        trainer.model.zero_grad(set_to_none=True)
        gc.collect()
    result = {"passed": True, "actual_8B": True, "optimizer_updates": 0, "parameters_unchanged": True,
              "teacher_forward_calls": 0, "all_trainable_parameters_compared": sum(p.numel() for p in params),
              "rtol": 2e-4, "atol": 2e-6, "flash_attention_deterministic_for_equivalence_only": True, "windows": reports}
    write(directory / "ce_equivalence.json", result)


def train(run_id, steps):
    directory = ROOT / run_id
    assert json.loads((directory / "ce_equivalence.json").read_text())["passed"]
    assert json.loads((ROOT / "unit_checks.json").read_text())["passed"]
    assert not (directory / "steps.jsonl").exists(), "Use a new run-id; never silently resume preflight"
    trainer, tokenmod, _, directory = build("train", run_id, steps)
    # FP32 teacher preserves the diagnostic precision contract; not autocast BF16.
    reference = AutoModelForCausalLM.from_pretrained(REF, local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa", device_map={"": "cuda:1"})
    reference.eval().requires_grad_(False)
    trainer.reference = ReferenceTargets(reference, beta=0.5, cap=2.0, chunk=64)
    ref_versions = [p._version for p in reference.parameters()]
    sample_names = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad and ("lora_B" in n or "lm_head.modules_to_save" in n)]
    before = {n: p.detach().reshape(-1)[:1024].float().cpu().clone() for n, p in sample_names}
    write(directory / "protocol.json", {
        "authorized_optimizer_updates": steps, "full_training_authorized": False, "student_initialization": "fresh sft-new2 plus original LF LoRA initialization; no CE-9000 student initialization",
        "student_base": str(trainer.model.config._name_or_path), "reference": REF, "reference_dtype": "float32",
        "student_precision": "original LF BF16 compute, FP32 trainable adapters/head", "beta": 0.5, "cap": 2.0,
        "lambda": 0.1, "mu": 0.1, "auxiliary_normalization": "mean answer tokens per Think sample; sum over Think / all samples in accumulation window",
        "ce_normalization": "unchanged native LF window supervised-token count and Accelerator backward scaling",
        "full_scheduler_steps_expected": 9339, "epochs": 3, "effective_batch": 16,
        "heldout_1000_used_for_training": False, "dev_or_test_generation": False,
        "teacher_student_communication": "full-vocabulary FP32 target chunks over explicit cuda:1 to cuda:0 transfers",
        "source_sha256": sha(SOURCE), "code_sha256": {p.name: sha(p) for p in ROOT.glob("*.py")}})
    assert sha(SOURCE) == EXPECTED_SOURCE
    torch.cuda.empty_cache()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=None)
    elapsed = time.perf_counter() - started
    assert trainer.state.global_step == steps and trainer.state.max_steps == 9339
    # Cover extreme sequence/answer lengths with the real, already allocated
    # optimizer state. These are backward-only probes: no optimizer updates.
    audit = [json.loads(l) for l in (DATA / "audit.jsonl").open()]
    stress_indices = list(dict.fromkeys([
        max((r for r in audit if r["think"]), key=lambda r: r["length"])["index"],
        max((r for r in audit if r["think"]), key=lambda r: r["answer_tokens"])["index"],
        max((r for r in audit if not r["think"]), key=lambda r: r["length"])["index"]]))
    stress_reports = []
    versions_before_stress = [p._version for p in trainer.model.parameters()]
    with torch.random.fork_rng(devices=[0, 1]):
        for index in stress_indices:
            trainer.model.zero_grad(set_to_none=True)
            for device in (0, 1):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            batch = trainer.data_collator([trainer.train_dataset[index]])
            count = trainer._get_num_items_in_batch([batch], trainer.args.device)
            trainer.start_window([batch], count)
            loss = trainer.training_step(trainer.model_wrapped, batch, num_items_in_batch=count)
            stress_reports.append(dict(audit[index], loss=float(loss),
                student_peak_allocated_gib=torch.cuda.max_memory_allocated(0)/1024**3,
                student_peak_reserved_gib=torch.cuda.max_memory_reserved(0)/1024**3,
                teacher_peak_allocated_gib=torch.cuda.max_memory_allocated(1)/1024**3,
                teacher_peak_reserved_gib=torch.cuda.max_memory_reserved(1)/1024**3))
            trainer.model.zero_grad(set_to_none=True)
    assert versions_before_stress == [p._version for p in trainer.model.parameters()]
    write(directory / "long_sequence_memory.json", {"passed": True, "optimizer_state_present": True,
        "optimizer_updates": 0, "parameters_unchanged": True, "samples": stress_reports})
    assert ref_versions == [p._version for p in reference.parameters()]
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
    changes = {n: float((p.detach().reshape(-1)[:1024].float().cpu() - before[n]).abs().max()) for n, p in sample_names}
    assert max(changes.values()) > 0
    trainer.save_state()
    # A preflight adapter is saved separately; it is not a main-experiment checkpoint.
    save_started = time.perf_counter()
    trainer.save_model()
    save_seconds = time.perf_counter() - save_started
    write(directory / "completion.json", {
        "passed": True, "optimizer_updates": trainer.state.global_step, "configured_scheduler_steps": trainer.state.max_steps,
        "training_seconds": elapsed, "save_seconds": save_seconds, "teacher_unchanged": True,
        "extreme_length_backward_probes_passed": True,
        "teacher_gradients_absent": True, "student_parameters_changed": True,
        "sampled_parameter_max_change": max(changes.values()), "model_output": trainer.args.output_dir,
        "full_training_started": False, "teacher_forward_calls": trainer.reference.forward_calls,
        "metrics": result.metrics, "completed_at_epoch_seconds": time.time()})
    print("PREFLIGHT_COMPLETE", json.dumps({"updates": steps, "training_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "train"))
    parser.add_argument("--run-id", default="attempt01")
    parser.add_argument("--steps", type=int, choices=(50, 100), default=50)
    args = parser.parse_args()
    try:
        validation(args.run_id) if args.stage == "validate" else train(args.run_id, args.steps)
    except Exception as error:
        (ROOT / args.run_id).mkdir(exist_ok=True)
        write(ROOT / args.run_id / f"{args.stage}_failure.json", {"error": repr(error), "time": time.time()})
        raise
