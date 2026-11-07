"""Single student GPU + dedicated frozen teacher GPU; native LF CE path.

CE uses native token normalization over the accumulation window. Auxiliary
terms are per-answer means, averaged over ALL examples in the same window
(no_think contributes zero). All branches use native Accelerator.backward.
"""
import json
import math
from pathlib import Path
import time

import torch
from transformers import TrainerCallback
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from losses import answer_kl


class ReasoningCollator:
    def __init__(self, native):
        self.native = native

    def __call__(self, rows):
        assert len(rows) == 1, "Preflight uses original microbatch size 1"
        result = self.native([{k: v for k, v in row.items() if k != "meta"} for row in rows])
        result["meta"] = [row["meta"] for row in rows]
        return result


class PreflightCallback(TrainerCallback):
    def __init__(self, stop_steps, root):
        self.stop_steps, self.root = stop_steps, Path(root)
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        t = self.trainer
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        elapsed = time.perf_counter() - t.window_started
        row = dict(t.window_metrics)
        row.update(step=state.global_step, seconds=elapsed, lr=t.optimizer.param_groups[0]["lr"],
                   student_peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 1024**3,
                   student_peak_reserved_gib=torch.cuda.max_memory_reserved(0) / 1024**3,
                   teacher_peak_allocated_gib=torch.cuda.max_memory_allocated(1) / 1024**3,
                   teacher_peak_reserved_gib=torch.cuda.max_memory_reserved(1) / 1024**3,
                   epoch=state.epoch, timestamp=time.time())
        row["extra_to_ce"] = (row["weighted_kd"] + row["weighted_anchor"]) / row["ce"]
        if not all(math.isfinite(row[k]) for k in ("ce", "weighted_kd", "weighted_anchor", "seconds", "extra_to_ce")):
            raise FloatingPointError("Nonfinite preflight metrics")
        with (self.root / "steps.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        (self.root / "progress.json").write_text(json.dumps(row, indent=2))
        print("PREFLIGHT_STEP", json.dumps({k: row[k] for k in ("step", "seconds", "ce", "weighted_kd", "weighted_anchor", "extra_to_ce", "student_peak_allocated_gib", "teacher_peak_allocated_gib")}), flush=True)
        if state.global_step >= self.stop_steps:
            control.should_training_stop = True
            control.should_save = False
        return control


class ReasoningTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, reference=None, kd_coef=0.1, anchor_coef=0.1, chunk=64, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.args.world_size == 1 and self.args.n_gpu == 1
        assert not self.is_deepspeed_enabled and not self.is_fsdp_enabled
        assert self.compute_loss_func is None and self.label_smoother is None and self.model_accepts_loss_kwargs
        self.reference = reference
        self.kd_coef, self.anchor_coef, self.chunk = kd_coef, anchor_coef, chunk
        self.window_size = self.args.gradient_accumulation_steps
        self.window_metrics = None
        self.active_data = None

    def get_batch_samples(self, *args, **kwargs):
        batches, count = super().get_batch_samples(*args, **kwargs)
        if batches:
            self.start_window(batches, count)
        return batches, count

    def start_window(self, batches, count):
        self.window_size = len(batches)
        assert int(count) == sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in batches)
        self.window_started = time.perf_counter()
        for device in (0, 1):
            torch.cuda.reset_peak_memory_stats(device)
        metas = [b["meta"][0] for b in batches]
        self.window_metrics = {"ce": 0.0, "kd_sum": 0.0, "anchor_sum": 0.0, "weighted_kd": 0.0,
                               "weighted_anchor": 0.0, "think": sum(m["think"] for m in metas), "microbatches": len(batches),
                               "ce_tokens": int(count), "input_tokens": sum(m["length"] for m in metas),
                               "think_input_tokens": sum(m["length"] for m in metas if m["think"]),
                               "answer_tokens_think": sum(len(m["gold"]) for m in metas if m["think"]),
                               "zero_tokens_think": sum(len(m["zero_ids"]) for m in metas if m["think"]),
                               "max_length": max(m["length"] for m in metas), "source_indices": [m["index"] for m in metas],
                               "teacher_seconds": 0.0, "plus_seconds": 0.0, "zero_seconds": 0.0}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        clean = dict(inputs)
        meta = clean.pop("meta")[0]
        ce, outputs = super().compute_loss(model, clean, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs)
        total = ce
        if meta["think"] and self.kd_coef:
            assert self.active_data is not None
            positions = torch.tensor(meta["positions"], device=outputs.logits.device) - 1
            selected = outputs.logits[0].index_select(0, positions)
            kd = answer_kl(selected, self.reference, self.active_data, "projected", self.chunk)
            total = total + self.kd_coef * kd / self.window_size
            if self.window_metrics is not None:
                self.window_metrics["kd_sum"] += float(kd.detach())
                self.window_metrics["weighted_kd"] += self.kd_coef * float(kd.detach()) / self.window_size
        if self.window_metrics is not None:
            self.window_metrics["ce"] += float(ce.detach())
        return (total, outputs) if return_outputs else total

    def training_step(self, model, inputs, num_items_in_batch=None):
        meta = inputs["meta"][0]
        enabled = meta["think"] and (self.kd_coef or self.anchor_coef)
        self.active_data = None
        if enabled:
            assert self.reference is not None
            start = time.perf_counter()
            ids = inputs["input_ids"][0, :meta["length"]].tolist()
            self.active_data = self.reference.prepare(ids, meta["zero_ids"], meta["positions"], meta["zero_positions"], meta["gold"])
            torch.cuda.synchronize(1)
            if self.window_metrics is not None:
                self.window_metrics["teacher_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        result = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        torch.cuda.synchronize(0)
        if self.window_metrics is not None:
            self.window_metrics["plus_seconds"] += time.perf_counter() - start
        if enabled and self.anchor_coef:
            start = time.perf_counter()
            ids = torch.tensor([meta["zero_ids"]], device=self.args.device, dtype=torch.long)
            with self.compute_loss_context_manager():
                outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                positions = torch.tensor(meta["zero_positions"], device=outputs.logits.device) - 1
                selected = outputs.logits[0].index_select(0, positions)
                anchor = answer_kl(selected, self.reference, self.active_data, "anchor", self.chunk)
                loss = self.anchor_coef * anchor / self.window_size
            # Same native backward scaling as the CE + projected KL branch.
            self.accelerator.backward(loss)
            result = result + loss.detach()
            if self.window_metrics is not None:
                self.window_metrics["anchor_sum"] += float(anchor.detach())
                self.window_metrics["weighted_anchor"] += float(loss.detach())
            del outputs, selected, anchor, loss
            torch.cuda.synchronize(0)
            if self.window_metrics is not None:
                self.window_metrics["zero_seconds"] += time.perf_counter() - start
        self.active_data = None
        return result
