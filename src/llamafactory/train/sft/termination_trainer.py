"""Single-GPU, BF16 termination experiment integration (tested on Transformers 5.2).

The normal Trainer is untouched unless termination_dataset_path is selected.
Auxiliary gradients are summed over the optimizer window before matching.
"""

import json
import math
from pathlib import Path

import torch

from .trainer import CustomSeq2SeqTrainer
from .termination_loss import boundary_losses, grad_norm, match_scale


class BoundaryCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, examples):
        positions = [list(x["term_positions"]) for x in examples]
        targets = [list(x["term_targets"]) for x in examples]
        clean = [{k: v for k, v in x.items() if k not in ("term_positions", "term_targets", "source_index")}
                 for x in examples]
        result = self.collator(clean)
        if self.collator.tokenizer.padding_side != "right":
            raise ValueError("Termination training requires right padding")
        result["term_positions"], result["term_targets"] = positions, targets
        return result


class TerminationTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1 or self.args.fp16 or self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("This audited experiment supports single-GPU BF16/FP32 only")
        self.args.remove_unused_columns = False
        self.end_id = self.processing_class.convert_tokens_to_ids("</think>")
        if self.processing_class.encode("</think>", add_special_tokens=False) != [self.end_id]:
            raise ValueError("The end marker must be a single token")
        self.term_params = [p for p in self.model.parameters() if p.requires_grad]
        self.term_log = Path(self.args.output_dir) / "termination_steps.jsonl"
        self.term_log.parent.mkdir(parents=True, exist_ok=True)
        self.probe_batches = None
        self._window_size = 0

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches, count = super().get_batch_samples(epoch_iterator, num_batches, device)
        self._window_size = len(batches)
        self._window_think = sum(bool(p) for b in batches for p in b["term_positions"])
        self._window_tokens = sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in batches)
        self._window_seen = 0
        self._g1 = self._g2 = None
        self._metrics = {"ce": 0., "ordinary": 0., "dynamic": 0., "think": self._window_think,
                         "continue": 0, "end_only": 0}
        if self.probe_batches is None:
            self.probe_batches = [{k: v.clone().cpu() if torch.is_tensor(v) else v for k, v in b.items()}
                                  for b in batches]
        return batches, count

    @staticmethod
    def _add_gradients(buffers, grads):
        if buffers is None:
            return [g.detach().to(device="cpu", dtype=torch.float32).clone() if g is not None else None for g in grads]
        for i, g in enumerate(grads):
            if g is not None:
                if buffers[i] is None:
                    buffers[i] = g.detach().to(device="cpu", dtype=torch.float32).clone()
                else:
                    buffers[i].add_(g.detach().to(device="cpu", dtype=torch.float32))
        return buffers

    def _forward_losses(self, model, batch, token_count, think_count):
        batch = dict(batch)
        pos, ys = batch.pop("term_positions"), batch.pop("term_targets")
        batch.pop("source_index", None)
        # Qwen3 ForCausalLMLoss uses shifted CE sum / num_items_in_batch.
        outputs = model(**batch, num_items_in_batch=token_count, use_cache=False)
        ordinary, dynamic, stats = boundary_losses(outputs.logits, pos, ys, self.end_id)
        den = max(think_count, 1)
        return outputs.loss, ordinary / den, dynamic / den, stats

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        mode = self.finetuning_args.termination_loss_mode
        coef = self.finetuning_args.termination_loss_coef
        with self.compute_loss_context_manager():
            ce, one, two, stats = self._forward_losses(model, inputs, self._window_tokens, self._window_think)
        if mode == "matched_margin" and stats["think"]:
            g1 = torch.autograd.grad(one, self.term_params, retain_graph=True, allow_unused=True)
            self._g1 = self._add_gradients(self._g1, g1)
            del g1
            g2 = torch.autograd.grad(two, self.term_params, retain_graph=True, allow_unused=True)
            self._g2 = self._add_gradients(self._g2, g2)
            del g2
        aux = two if mode == "dynamic" else one
        loss = ce + (coef * aux if mode != "none" else 0)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        # Explicit normalization above; avoid Accelerate's additional GA division.
        loss.backward()
        self._window_seen += 1
        for k, v in (("ce", ce), ("ordinary", one), ("dynamic", two)):
            self._metrics[k] += float(v.detach())
        for k in ("continue", "end_only"):
            self._metrics[k] += stats[k]
        reporting_correction = 0.
        if self._window_seen == self._window_size:
            c = 1.
            if mode == "matched_margin" and self._g1 is not None:
                c, n1, n2 = match_scale(self._g1, self._g2)
                measured_match = grad_norm([g * c if g is not None else None for g in self._g1])
                relative_error = abs(measured_match - n2) / n2 if n2 else 0.
                if relative_error > 2e-5:
                    raise FloatingPointError("Gradient strength matching tolerance exceeded")
                for p, g in zip(self.term_params, self._g1):
                    if g is not None:
                        p.grad.add_(g.to(p.device), alpha=coef * (c - 1))
                self._metrics.update(match_scale=c, ordinary_grad_norm=n1, dynamic_grad_norm=n2,
                                     matched_grad_norm=measured_match, match_error=abs(measured_match-n2),
                                     match_relative_error=relative_error)
                reporting_correction = coef * (c - 1) * self._metrics["ordinary"]
            self._metrics.update(step=self.state.global_step + 1, mode=mode, coefficient=coef,
                                 total=self._metrics["ce"] + (coef*c*self._metrics["ordinary"] if mode in
                                 ("margin", "matched_margin") else coef*self._metrics["dynamic"] if mode == "dynamic" else 0),
                                 grad_norm_preclip=grad_norm([p.grad for p in self.term_params]),
                                 peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
            if not math.isfinite(self._metrics["grad_norm_preclip"]):
                raise FloatingPointError("Non-finite gradients")
            with self.term_log.open("a") as f:
                f.write(json.dumps(self._metrics) + "\n")
            self._g1 = self._g2 = None
            if (self.state.global_step + 1) % 100 == 0 or self.state.global_step + 1 == self.state.max_steps:
                self.gradient_probe(model, self.probe_batches, self.state.global_step + 1)
        return loss.detach() + reporting_correction

    def gradient_probe(self, model, batches, step):
        """No parameter/.grad/RNG updates; common dropout masks for all objectives."""
        cuda_devices = list(range(torch.cuda.device_count()))
        buffers = [None, None, None]
        tokens = sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in batches)
        thinks = sum(bool(p) for b in batches for p in b["term_positions"])
        was_training = model.training
        with torch.random.fork_rng(devices=cuda_devices):
            model.train()
            for b in batches:
                with self.compute_loss_context_manager():
                    losses = self._forward_losses(model, self._prepare_inputs(b), tokens, thinks)[:3]
                for i, loss in enumerate(losses):
                    g = torch.autograd.grad(loss, self.term_params, retain_graph=i < 2, allow_unused=True)
                    buffers[i] = self._add_gradients(buffers[i], g)
                    del g
        model.train(was_training)
        norms = [grad_norm(g or []) for g in buffers]
        row = {"step": step, "ce_norm": norms[0], "ordinary_norm": norms[1], "dynamic_norm": norms[2]}
        row["coefficient"] = self.finetuning_args.termination_loss_coef
        for i, name in ((1, "ordinary"), (2, "dynamic")):
            row[name + "_ratio"] = norms[i] / norms[0] if norms[0] else None
            row[name + "_weighted_ratio"] = self.finetuning_args.termination_loss_coef * norms[i] / norms[0] if norms[0] else None
            dot = sum(float((a.float()*b.float()).sum()) for a, b in zip(buffers[0], buffers[i])
                      if a is not None and b is not None)
            row[name + "_cosine_ce"] = dot / (norms[0]*norms[i]) if norms[0]*norms[i] else None
        with (self.term_log.parent / "gradient_probes.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row
