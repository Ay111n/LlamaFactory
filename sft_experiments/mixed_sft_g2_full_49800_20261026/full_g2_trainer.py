"""Full-G2 only: native LF CE/Trainer backward; unchanged Gold auxiliary formula."""
import contextlib,json,math,os,time
from pathlib import Path
import torch
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.sft.termination_trainer import BoundaryCollator
from llamafactory.train.sft.termination_gold import gold_boundary_losses


class FullBoundaryCollator(BoundaryCollator):
    def __call__(self,rows):
        result=super().__call__(rows)
        result['source_index']=[r.get('source_index',-1) for r in rows]
        return result


def add_vector(buffers,grads,device):
    if buffers is None:return [None if g is None else g.detach().to(device=device,dtype=torch.float32,copy=True) for g in grads]
    for i,g in enumerate(grads):
        if g is None:continue
        if buffers[i] is None:buffers[i]=g.detach().to(device=device,dtype=torch.float32,copy=True)
        else:buffers[i].add_(g.detach().to(device=buffers[i].device,dtype=torch.float32))
    return buffers


def gram64(vectors,params):
    device=next((g.device for v in vectors for g in (v or []) if g is not None),torch.device('cpu'))
    out=torch.zeros((len(vectors),len(vectors)),dtype=torch.float64,device=device)
    for i,p in enumerate(params):
        values=[v[i] if v else None for v in vectors]
        if all(g is None for g in values):continue
        for start in range(0,p.numel(),1048576):
            length=min(1048576,p.numel()-start)
            x=torch.zeros((len(values),length),dtype=torch.float64,device=device)
            for j,g in enumerate(values):
                if g is not None:x[j].copy_(g.reshape(-1)[start:start+length])
            out.add_(x@x.T)
    if not bool(torch.isfinite(out).all()):raise FloatingPointError('Nonfinite gradient Gram matrix')
    return out.cpu()


def norm64(grads):
    device=next((g.device for g in grads if g is not None),torch.device('cpu'))
    total=torch.zeros((),dtype=torch.float64,device=device)
    for g in grads:
        if g is None:continue
        v=g.detach().reshape(-1)
        for start in range(0,v.numel(),1048576):total.add_(v[start:start+1048576].double().square().sum().to(device))
    result=math.sqrt(float(total))
    if not math.isfinite(result):raise FloatingPointError('Nonfinite gradient norm')
    return result


class FullG2Trainer(CustomSeq2SeqTrainer):
    """All backward/scaling/clipping/optimizer behavior is inherited from native HF/LF."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.args.world_size!=1 or self.args.n_gpu>1 or self.args.fp16 or self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError('Full-G2 requires single model GPU and BF16/FP32, without distributed engines')
        if self.finetuning_args.termination_loss_mode!='dynamic_gold_margin':raise ValueError('Only original G2 is supported')
        if self.compute_loss_func is not None or self.label_smoother is not None or not self.model_accepts_loss_kwargs:
            raise ValueError('Native token-normalized CE path is required')
        self.end_id=self.processing_class.convert_tokens_to_ids('</think>')
        if self.processing_class.encode('</think>',add_special_tokens=False)!=[self.end_id]:raise ValueError('Stop must be one token')
        self.term_params=[p for p in self.model.parameters() if p.requires_grad]
        self.buffer_device=os.environ.get('FULL_G2_BUFFER_DEVICE','cpu')
        self.probe_windows=None
        # Full vector diagnostics run before training and after the final saved state.
        # Per-update losses/gradient norms are logged without changing the training loop.
        self.probe_steps=set()
        self.log_root=Path(self.args.output_dir);self.log_root.mkdir(parents=True,exist_ok=True)
        self._window_tokens=0;self._window_think=0;self._window_size=0;self._seen=0

    def get_batch_samples(self,*args,**kwargs):
        batches,count=super().get_batch_samples(*args,**kwargs)
        if not batches:return batches,count
        self._window_tokens=sum(int(b['labels'][:,1:].ne(-100).sum()) for b in batches)
        self._window_think=sum(bool(p) for b in batches for p in b['term_positions'])
        self._window_size=len(batches);self._seen=0
        if self._window_tokens<=0 or count is None or int(count)!=self._window_tokens:
            raise ValueError('Native label denominator differs from shifted CE denominator')
        self._metrics={'ce':0.,'dynamic_gold':0.,'continue_loss':0.,'stop_loss':0.,'total':0.,
                       'window_think':self._window_think,'window_ce_tokens':self._window_tokens,
                       'microbatches':len(batches),'continue_boundaries':0,'end_only':0,
                       'source_indices':[i for b in batches for i in b.get('source_index',[])]}
        return batches,count

    def loss_bundle(self,model,batch,token_count,think_count,force_aux=True):
        clean=dict(batch);pos=clean.pop('term_positions');kinds=clean.pop('term_targets');clean.pop('source_index',None)
        # Native LF compute_loss is the ONLY standard CE implementation.
        ce,outputs=super().compute_loss(model,clean,return_outputs=True,num_items_in_batch=token_count)
        if force_aux:
            losses,stats,parts=gold_boundary_losses(outputs.logits,clean['labels'],pos,kinds,self.end_id,clean['input_ids'])
            aux=losses['dynamic_gold']/max(think_count,1)
            cont=parts['dynamic_gold_continue']/max(think_count,1)
            stop=parts['dynamic_gold_end']/max(think_count,1)
        else:
            aux=ce.detach()*0;cont=stop=0.
            stats={'think':sum(bool(x) for x in pos),'continue':sum(y.count(0) for y in kinds),'end_only':sum(y==[1] for y in kinds)}
        return ce,aux,outputs,stats,cont,stop

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None,**kwargs):
        coefficient=self.finetuning_args.termination_loss_coef
        ce,aux,outputs,stats,cont,stop=self.loss_bundle(model,inputs,num_items_in_batch,self._window_think,force_aux=coefficient!=0)
        total=ce+coefficient*aux if coefficient else ce
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite Full-G2 loss')
        if hasattr(self,'_metrics'):
            for key,value in [('ce',ce),('dynamic_gold',aux),('total',total)]:self._metrics[key]+=float(value.detach())
            self._metrics['continue_loss']+=cont;self._metrics['stop_loss']+=stop
            self._metrics['continue_boundaries']+=stats['continue'];self._metrics['end_only']+=stats['end_only']
        return (total,outputs) if return_outputs else total

    def training_step(self,*args,**kwargs):
        # DO NOT call raw loss.backward or introduce any GA divisions here.
        result=super().training_step(*args,**kwargs)
        self._seen+=1
        if self._seen==self._window_size:
            row=dict(self._metrics,step=self.state.global_step+1,coefficient=self.finetuning_args.termination_loss_coef,
                     accelerator_ga=self.accelerator.gradient_accumulation_steps,
                     grad_norm_preclip=norm64([p.grad for p in self.term_params]),
                     lr=self.optimizer.param_groups[0]['lr'] if self.optimizer else None,
                     elapsed_utc=time.time(),backward='inherited native Accelerator.backward')
            with (self.log_root/'full_g2_steps.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if self.probe_windows and row['step'] in self.probe_steps:
                for i,window in enumerate(self.probe_windows):self.probe_window(self.model,window,row['step'],i,'pre_optimizer')
        return result

    def probe_window(self,model,batches,step,index,phase='initial'):
        """Reentrant-compatible VJPs; preserve every existing .grad reference and RNG."""
        prior=[p.grad for p in self.term_params]
        versions=[p._version for p in self.term_params]
        modes=[(m,m.training) for m in model.modules()]
        tokens=sum(int(b['labels'][:,1:].ne(-100).sum()) for b in batches)
        thinks=sum(bool(x) for b in batches for x in b['term_positions'])
        native_count=int(self._get_num_items_in_batch(batches,self.args.device))
        if native_count!=tokens or tokens<=0:raise ValueError('Invalid native probe denominator')
        buffers=[None]*4;values=[0.,0.,0.,0.];continuing=0;end_only=0
        try:
            for p in self.term_params:p.grad=None
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                model.train()
                for raw in batches:
                    b=self._prepare_inputs(raw)
                    with self.compute_loss_context_manager():ce,aux,outputs,stats,c,s=self.loss_bundle(model,b,tokens,thinks,True)
                    if not bool(torch.isfinite(ce+aux)):raise FloatingPointError('Nonfinite probe objective')
                    values[0]+=float(ce.detach());values[1]+=c;values[2]+=s;values[3]+=float(aux.detach())
                    continuing+=stats['continue'];end_only+=stats['end_only']
                    masks=[None,outputs.logits.new_zeros((*outputs.logits.shape[:2],1)),outputs.logits.new_zeros((*outputs.logits.shape[:2],1)),None]
                    for bi,(pos,ys) in enumerate(zip(b['term_positions'],b['term_targets'])):
                        for p,y in zip(pos,ys):masks[2 if y else 1][bi,p,0]=1
                    objectives=[ce,aux,aux,aux]
                    for gi,loss in enumerate(objectives):
                        if gi and not stats['think']:continue
                        handle=outputs.logits.register_hook(lambda g,mask=masks[gi]:g*mask) if masks[gi] is not None else None
                        try:
                            self.accelerator.backward(loss,retain_graph=gi<3 and bool(stats['think']))
                            buffers[gi]=add_vector(buffers[gi],[p.grad for p in self.term_params],self.buffer_device)
                        finally:
                            if handle is not None:handle.remove()
                            for p in self.term_params:p.grad=None
                    del outputs,objectives,ce,aux,loss,masks
        finally:
            for p,g in zip(self.term_params,prior):p.grad=g
            for module,training in modes:module.training=training
        assert versions==[p._version for p in self.term_params],'Probe changed parameters'
        assert all(p.grad is g for p,g in zip(self.term_params,prior)),'Probe did not restore gradient objects'
        gram=gram64(buffers,self.term_params);norms=gram.diag().clamp_min(0).sqrt().tolist()
        if norms[0]<=0:raise FloatingPointError('Zero CE norm')
        def cosine(a,b):return float(gram[a,b])/(norms[a]*norms[b]) if norms[a]*norms[b] else None
        # BF16 independently masked VJPs can have rounding residuals; retain it explicitly.
        residual=max(0.,float(gram[1,1]+gram[2,2]+gram[3,3]+2*gram[1,2]-2*gram[1,3]-2*gram[2,3]))
        relative=math.sqrt(residual)/norms[3] if norms[3] else 0.
        coef=self.finetuning_args.termination_loss_coef
        row={'step':step,'window_index':index,'phase':phase,'window_ce_tokens':tokens,'window_think':thinks,
             'microbatches':len(batches),'continue_boundaries':continuing,'end_only':end_only,
             'ce_loss':values[0],'continue_loss':values[1],'stop_loss':values[2],'auxiliary_loss':values[3],
             'ce_norm':norms[0],'continue_norm':norms[1],'stop_norm':norms[2],'auxiliary_norm':norms[3],
             'R':coef*norms[3]/norms[0],'continue_R':coef*norms[1]/norms[0],'stop_R':coef*norms[2]/norms[0],
             'continue_stop_cosine':cosine(1,2),'continue_ce_cosine':cosine(1,0),'stop_ce_cosine':cosine(2,0),
             'auxiliary_ce_cosine':cosine(3,0),'component_sum_relative_residual':relative,
             'continue_projection':float(gram[1,3]/gram[3,3]) if norms[3] else None,
             'stop_projection':float(gram[2,3]/gram[3,3]) if norms[3] else None,
             'norm_convention':'all trainable parameters; native Accelerator.backward scaling already included',
             'accelerator_ga':self.accelerator.gradient_accumulation_steps,'parameters_unchanged':True,'gradient_objects_restored':True}
        with (self.log_root/'full_g2_probes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print('FULL_G2_PROBE',json.dumps(row),flush=True)
        return row
