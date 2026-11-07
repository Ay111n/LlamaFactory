"""Gold objectives on the audited legacy optimizer-window implementation."""
import json
from pathlib import Path
import torch
from .termination_trainer import TerminationTrainer
from .termination_loss import grad_norm
from .termination_gold import gold_boundary_losses, GOLD_MODES


class GoldTerminationTrainer(TerminationTrainer):
    def __init__(self, *args, **kwargs):
        self.gold_objective = kwargs['finetuning_args'].termination_loss_mode
        if self.gold_objective not in GOLD_MODES:
            raise ValueError(self.gold_objective)
        super().__init__(*args, **kwargs)

    def get_batch_samples(self, *args, **kwargs):
        batches, count = super().get_batch_samples(*args, **kwargs)
        if self._window_tokens <= 0:
            raise ValueError("No supervised shifted CE tokens in optimizer window")
        self._metrics['gold_objective'] = self.gold_objective
        self._metrics['window_ce_tokens'] = self._window_tokens
        self._metrics['window_microbatches'] = self._window_size
        self._metrics['auxiliary_aliases'] = {'ordinary': GOLD_MODES[self.gold_objective][0], 'dynamic': 'dynamic_gold'}
        return batches, count

    def _all_losses(self, model, batch, token_count, think_count):
        batch = dict(batch)
        pos, kinds = batch.pop('term_positions'), batch.pop('term_targets')
        batch.pop('source_index', None)
        if token_count <= 0:
            raise ValueError("CE denominator must be positive")
        output = model(**batch, num_items_in_batch=token_count, use_cache=False)
        losses, stats, parts = gold_boundary_losses(output.logits, batch['labels'], pos, kinds,
                                                    self.end_id, batch['input_ids'])
        den = max(think_count, 1)
        return [output.loss]+[losses[k]/den for k in ('gold','dynamic_gold','dynamic_ce')], stats, parts

    def _forward_losses(self, model, batch, token_count, think_count):
        losses, stats, _ = self._all_losses(model, batch, token_count, think_count)
        candidate = 3 if GOLD_MODES[self.gold_objective][0] == 'dynamic_ce' else 1
        return losses[0], losses[candidate], losses[2], stats

    def training_step(self, *args, **kwargs):
        # The old window/reduction/matching/backward code remains the single implementation.
        original = self.finetuning_args.termination_loss_mode
        self.finetuning_args.termination_loss_mode = GOLD_MODES[self.gold_objective][1]
        try:
            return super().training_step(*args, **kwargs)
        finally:
            self.finetuning_args.termination_loss_mode = original

    def gradient_probe(self, model, batches, step):
        windows = getattr(self, 'gold_probe_windows', None)
        if step > 0 and windows:
            rows=[]
            for index,window in enumerate(windows):
                self._gold_probe_index=index
                rows.append(self._gradient_probe_one(model,window,step))
            return {'step':step,'windows':rows}
        self._gold_probe_index=None
        return self._gradient_probe_one(model,batches,step)

    def _gradient_probe_one(self, model, batches, step):
        buffers = [None]*4
        tokens = sum(int(b['labels'][:, 1:].ne(-100).sum()) for b in batches)
        thinks = sum(bool(p) for b in batches for p in b['term_positions'])
        training = model.training
        components = {}
        counts = {'think': 0, 'continue': 0, 'end_only': 0}
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            try:
                model.train()
                for b in batches:
                    with self.compute_loss_context_manager():
                        losses, stats, parts = self._all_losses(model, self._prepare_inputs(b), tokens, thinks)
                    if not all(bool(torch.isfinite(x)) for x in losses):
                        raise FloatingPointError("Non-finite probe loss")
                    for i, loss in enumerate(losses):
                        gradients = torch.autograd.grad(loss, self.term_params, retain_graph=i<3, allow_unused=True)
                        buffers[i] = self._add_gradients(buffers[i], gradients)
                        del gradients
                    for k,v in parts.items(): components[k] = components.get(k,0.)+v/max(thinks,1)
                    for k in counts: counts[k] += stats[k]
            finally:
                model.train(training)
        norms = [grad_norm(g or []) for g in buffers]
        if norms[0] <= 0:
            raise FloatingPointError("Zero CE gradient cannot define diagnostic ratio")
        row = {'step':step, 'window_index':self._gold_probe_index, 'ce_norm':norms[0], 'window_ce_tokens':tokens,
               'window_think':thinks, 'counts':counts, 'parts':components,
               'coefficient':self.finetuning_args.termination_loss_coef}
        for i,k in enumerate(('gold','dynamic_gold','dynamic_ce'),1):
            row[k+'_norm'] = norms[i]
            row[k+'_R'] = self.finetuning_args.termination_loss_coef*norms[i]/norms[0]
            dot=sum(float((a*b).sum()) for a,b in zip(buffers[0],buffers[i]) if a is not None and b is not None)
            row[k+'_cosine_ce'] = dot/(norms[0]*norms[i]) if norms[i] else None
        with (self.term_log.parent/'gold_gradient_probes.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        return row
