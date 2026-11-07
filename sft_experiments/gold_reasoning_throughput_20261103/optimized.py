"""Exact FP32 target reuse and bounded one-sample teacher prefetch.

Imports the frozen loss/projection and CE implementation. No loss coefficients,
token/sample reductions, precision, or sample-order changes.
"""
from concurrent.futures import ThreadPoolExecutor
import time
import torch
import torch.nn.functional as F
from losses import ReferenceTargets, projected_log_probs, answer_kl
from reasoning_trainer import ReasoningTrainer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


class CachedTargets(ReferenceTargets):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_calls = self.projection_calls = self.cache_hits = 0

    @torch.no_grad()
    def target(self, data, start, end, mode):
        cache = data.setdefault("_targets", {})
        key = (start, end)
        if key not in cache:
            with torch.autocast("cuda", enabled=False):
                r0 = F.log_softmax(self.model.lm_head(data["zero"][start:end]).float(), -1)
                rp = F.log_softmax(self.model.lm_head(data["plus"][start:end]).float(), -1)
                q = projected_log_probs(rp, r0, data["gold"][start:end], self.beta, self.cap)
            cache[key] = {"projected": q, "anchor": r0}
            self.head_calls += 2
            self.projection_calls += 1
        else:
            self.cache_hits += 1
        self.target_calls += 1
        return cache[key][mode]


class PipelineTargets(CachedTargets):
    """At most one ready/in-flight sample in addition to the current sample."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frozen_teacher")
        self.stream = torch.cuda.Stream(device=self.device)
        self.jobs = []
        self.future = None
        self.prepared_samples = 0

    def stage(self, batches):
        assert not self.jobs and self.future is None
        self.jobs = [(batch["input_ids"][0, :batch["meta"][0]["length"]].tolist(), batch["meta"][0])
                     for batch in batches if batch["meta"][0]["think"]]
        self._launch_next()

    def _launch_next(self):
        assert self.future is None
        if self.jobs:
            ids, meta = self.jobs.pop(0)
            self.future = self.executor.submit(self._produce, ids, meta)

    @torch.no_grad()
    def _produce(self, ids, meta):
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            data = ReferenceTargets.prepare(self, ids, meta["zero_ids"], meta["positions"], meta["zero_positions"], meta["gold"])
            for start in range(0, len(meta["gold"]), self.chunk):
                CachedTargets.target(self, data, start, min(len(meta["gold"]), start+self.chunk), "projected")
            data["_signature"] = (tuple(ids), tuple(meta["zero_ids"]), tuple(meta["positions"]), tuple(meta["zero_positions"]), tuple(meta["gold"]))
            # Only this worker waits for its stream. The student can run meanwhile.
            self.stream.synchronize()
        self.prepared_samples += 1
        return data

    def prepare(self, plus_ids, zero_ids, plus_positions, zero_positions, gold):
        assert self.future is not None, "stage the accumulation window before use"
        data = self.future.result()
        self.future = None
        assert data["_signature"] == tuple(tuple(x) for x in (plus_ids, zero_ids, plus_positions, zero_positions, gold))
        self._launch_next()
        return data

    @torch.no_grad()
    def target(self, data, start, end, mode):
        # All tensors were completed before publication and stay alive in data.
        return data["_targets"][(start, end)][mode]

    def close(self):
        self.executor.shutdown(wait=True)
        self.jobs.clear()
        self.future = None


class PipelineTrainer(ReasoningTrainer):
    def get_batch_samples(self, *args, **kwargs):
        batches, count = super().get_batch_samples(*args, **kwargs)
        if batches and (self.kd_coef or self.anchor_coef):
            self.reference.stage(batches)
        return batches, count

    def training_step(self, model, inputs, num_items_in_batch=None):
        meta = inputs["meta"][0]
        enabled = meta["think"] and (self.kd_coef or self.anchor_coef)
        self.active_data = None
        if enabled:
            start = time.perf_counter()
            ids = inputs["input_ids"][0, :meta["length"]].tolist()
            self.active_data = self.reference.prepare(ids, meta["zero_ids"], meta["positions"], meta["zero_positions"], meta["gold"])
            self.window_metrics["teacher_seconds"] += time.perf_counter()-start
        start = time.perf_counter()
        # Reuse native backward and inherited compute_loss verbatim. Avoid the
        # original synchronous teacher preparation inside ReasoningTrainer.
        result = CustomSeq2SeqTrainer.training_step(self, model, inputs, num_items_in_batch=num_items_in_batch)
        torch.cuda.synchronize(0)
        self.window_metrics["plus_seconds"] += time.perf_counter()-start
        if enabled and self.anchor_coef:
            start = time.perf_counter()
            ids = torch.tensor([meta["zero_ids"]], device=self.args.device, dtype=torch.long)
            with self.compute_loss_context_manager():
                outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                positions = torch.tensor(meta["zero_positions"], device=outputs.logits.device)-1
                selected = outputs.logits[0].index_select(0, positions)
                anchor = answer_kl(selected, self.reference, self.active_data, "anchor", self.chunk)
                loss = self.anchor_coef*anchor/self.window_size
            self.accelerator.backward(loss)
            result = result+loss.detach()
            self.window_metrics["anchor_sum"] += float(anchor.detach())
            self.window_metrics["weighted_anchor"] += float(loss.detach())
            del outputs, selected, anchor, loss
            torch.cuda.synchronize(0)
            self.window_metrics["zero_seconds"] += time.perf_counter()-start
        self.active_data = None
        return result
