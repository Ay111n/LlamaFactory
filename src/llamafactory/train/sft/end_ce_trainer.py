"""Reuse the audited window/matching/backward implementation for end-only CE."""
import json, math, os
import torch
from . import termination_trainer as engine
from .termination_end_ce import END_MODES, end_boundary_losses


def norm64(grads):
    # FP32 vectors; chunked FP64 reduction avoids a full FP64 lm_head allocation.
    total = None
    for g in grads:
        if g is None: continue
        flat = g.detach().reshape(-1)
        for start in range(0, flat.numel(), 1048576):
            x = flat[start:start+1048576].double()
            part = x.square().sum()
            if total is None: total = part
            else: total.add_(part.to(total.device))
    return math.sqrt(float(total)) if total is not None else 0.


def match64(g1, g2):
    n1, n2 = norm64(g1), norm64(g2)
    if not math.isfinite(n1+n2) or (n1 == 0 and n2 != 0):
        raise FloatingPointError(f'Unmatchable end gradients: {n1}, {n2}')
    return (n2/n1 if n1 else 1.), n1, n2


def dot64(xs, ys):
    total = None
    for a,b in zip(xs,ys):
        if a is None or b is None: continue
        a,b = a.reshape(-1), b.reshape(-1)
        for start in range(0,a.numel(),1048576):
            part=(a[start:start+1048576].double()*b[start:start+1048576].double()).sum()
            if total is None: total=part
            else: total.add_(part.to(total.device))
    return float(total) if total is not None else 0.


class EndCETrainer(engine.TerminationTrainer):
    def __init__(self,*args,**kwargs):
        self.end_objective=kwargs['finetuning_args'].termination_loss_mode
        if self.end_objective not in END_MODES: raise ValueError(self.end_objective)
        super().__init__(*args,**kwargs)
        self.end_probe_windows=None

    def get_batch_samples(self,*args,**kwargs):
        batches,count=super().get_batch_samples(*args,**kwargs)
        if self._window_tokens<=0:raise ValueError('No valid shifted CE tokens')
        self._metrics.update(end_objective=self.end_objective,window_ce_tokens=self._window_tokens,
                             window_microbatches=self._window_size,
                             auxiliary_aliases={'ordinary':'end_ce','dynamic':'dynamic_end_ce'})
        return batches,count

    @staticmethod
    def _add_gradients(buffers, grads):
        # Optional otherwise-idle GPU used ONLY for FP32 buffers, never for DDP.
        device=os.environ.get('END_BUFFER_DEVICE','cpu')
        if buffers is None:
            return [None if g is None else g.detach().to(device=device,dtype=torch.float32,copy=True) for g in grads]
        for i,g in enumerate(grads):
            if g is None: continue
            if buffers[i] is None: buffers[i]=g.detach().to(device=device,dtype=torch.float32,copy=True)
            else: buffers[i].add_(g.detach().to(buffers[i].device,dtype=torch.float32))
        return buffers

    def _forward_losses(self,model,batch,token_count,think_count):
        batch=dict(batch)
        pos,kinds=batch.pop('term_positions'),batch.pop('term_targets')
        batch.pop('source_index',None)
        if token_count<=0: raise ValueError('No valid shifted CE tokens')
        output=model(**batch,num_items_in_batch=token_count,use_cache=False)
        one,two,stats=end_boundary_losses(output.logits,batch['labels'],pos,kinds,self.end_id,batch['input_ids'])
        return output.loss,one/max(think_count,1),two/max(think_count,1),stats

    def training_step(self,*args,**kwargs):
        original=self.finetuning_args.termination_loss_mode
        # Process-local function substitution: old source and old runs are unchanged.
        old_norm,old_match=engine.grad_norm,engine.match_scale
        engine.grad_norm,engine.match_scale=norm64,match64
        self.finetuning_args.termination_loss_mode=END_MODES[self.end_objective]
        try: return super().training_step(*args,**kwargs)
        finally:
            engine.grad_norm,engine.match_scale=old_norm,old_match
            self.finetuning_args.termination_loss_mode=original

    def gradient_probe(self,model,batches,step):
        windows=self.end_probe_windows if step>0 and self.end_probe_windows else [batches]
        rows=[]
        for index,window in enumerate(windows):
            rows.append(self._probe_window(model,window,step,index))
        return {'step':step,'windows':rows} if len(rows)>1 else rows[0]

    def _probe_window(self,model,batches,step,index):
        tokens=sum(int(b['labels'][:,1:].ne(-100).sum()) for b in batches)
        thinks=sum(bool(p) for b in batches for p in b['term_positions'])
        buffers=[None]*3; values=[0.]*3
        training=model.training
        versions=[p._version for p in self.term_params]
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            try:
                model.train()
                for batch in batches:
                    with self.compute_loss_context_manager():
                        losses=self._forward_losses(model,self._prepare_inputs(batch),tokens,thinks)[:3]
                    if not all(bool(torch.isfinite(x)) for x in losses): raise FloatingPointError('Nonfinite probe loss')
                    for i,loss in enumerate(losses):
                        grads=torch.autograd.grad(loss,self.term_params,retain_graph=i<2,allow_unused=True)
                        buffers[i]=self._add_gradients(buffers[i],grads);values[i]+=float(loss.detach())
                        del grads
            finally: model.train(training)
        assert versions==[p._version for p in self.term_params],'Probe changed parameters'
        norms=[norm64(v or []) for v in buffers]
        if norms[0]<=0 or not all(math.isfinite(v) for v in norms): raise FloatingPointError('Invalid CE/aux norm')
        c,n1,n2=match64(buffers[1],buffers[2])
        measured=norm64([None if g is None else g*c for g in buffers[1]])
        error=abs(measured-n2)/n2 if n2 else 0.
        if error>=2e-5: raise FloatingPointError('Probe matching failed')
        coef=self.finetuning_args.termination_loss_coef
        row={'step':step,'window_index':index,'phase':'post_update_final' if step==1125 and getattr(self,'probe_after_update',False) else 'pre_update',
             'window_ce_tokens':tokens,'window_think':thinks,'window_microbatches':len(batches),
             'ce_norm':norms[0],'ce_loss':values[0],'end_loss':values[1],'dynamic_end_loss':values[2],
             'end_norm':norms[1],'dynamic_end_norm':norms[2],
             'end_R':coef*norms[1]/norms[0],'dynamic_end_R':coef*norms[2]/norms[0],
             'matched_end_R':coef*measured/norms[0], 'match_scale':c,'match_relative_error':error,
             'end_cosine_ce':dot64(buffers[0],buffers[1])/(norms[0]*norms[1]) if norms[1] else None,
             'dynamic_end_cosine_ce':dot64(buffers[0],buffers[2])/(norms[0]*norms[2]) if norms[2] else None,
             'end_dynamic_cosine':dot64(buffers[1],buffers[2])/(norms[1]*norms[2]) if norms[1]*norms[2] else None}
        with (self.term_log.parent/'end_gradient_probes.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print('END_PROBE',json.dumps(row),flush=True)
        return row
