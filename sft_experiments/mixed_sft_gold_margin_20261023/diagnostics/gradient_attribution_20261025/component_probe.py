"""Read-only full-parameter gradient attribution using the frozen loss implementation.

Components are VJPs of the ORIGINAL scalar loss with disjoint logits-row masks.
No duplicate Gold objective and no optimizer/training/generation call is used.
"""
import argparse,contextlib,gc,hashlib,json,math,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parents[1]
sys.path.insert(0,str(EXPERIMENT))
import run_gold as original
from experiment_paths import paths,require_path,model_root,data_root
from llamafactory.train.sft.termination_gold import gold_boundary_losses

ARMS={'G1':'gold','G2':'dynamic_gold','G3':'dynamic_ce','G1-M':'gold','G3-M':'dynamic_ce'}
COEF=.0012
CACHE=data_root()/'gradient_attribution_20261025'
LOCAL_BUFFER_BYTES=0

def allocate_buffer(g,name):
    global LOCAL_BUFFER_BYTES
    target='cpu'
    cpu_keys=set(os.environ.get('ATTR_CPU_KEYS','').split(','))
    if os.environ.get('ATTR_GPU_BUFFERS')=='1' and name!='ce' and name not in cpu_keys:
        size=g.numel()*4
        if LOCAL_BUFFER_BYTES+size<=5_000_000_000:
            target='cuda:0';LOCAL_BUFFER_BYTES+=size
        else:target='cuda:1'
    return g.detach().to(device=target,dtype=torch.float32,copy=True)

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def gram(vectors,sizes,chunk=1048576):
    """FP64 Gram reduction, while the accumulated parameter vectors remain FP32."""
    keys=list(vectors);matrix=torch.zeros((len(keys),len(keys)),dtype=torch.float64)
    for index,size in enumerate(sizes):
        values=[vectors[key][index] for key in keys]
        if all(v is None for v in values):continue
        flat=[None if v is None else v.reshape(-1) for v in values]
        for start in range(0,size,chunk):
            length=min(chunk,size-start);x=torch.zeros((len(keys),length),dtype=torch.float64)
            for i,value in enumerate(flat):
                if value is not None:x[i].copy_(value[start:start+length])
            matrix.add_(x@x.T)
    if not torch.isfinite(matrix).all():raise FloatingPointError('Non-finite gradient Gram matrix')
    return keys,matrix

def component_grad(loss,logits,params,mask=None):
    handle=logits.register_hook(lambda g:g*mask) if mask is not None else None
    try:return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    finally:
        if handle is not None:handle.remove()

def add(buffers,gradients,name='ce'):
    if buffers is None:return [None if g is None else allocate_buffer(g,name) for g in gradients]
    for i,g in enumerate(gradients):
        if g is not None:
            if buffers[i] is None:buffers[i]=allocate_buffer(g,name)
            elif buffers[i].device==g.device:buffers[i].add_(g.detach())
            else:
                a=buffers[i].view(-1);b=g.detach().view(-1)
                for start in range(0,b.numel(),8_388_608):a[start:start+8_388_608].add_(b[start:start+8_388_608].to(a.device,non_blocking=a.is_cuda))
    return buffers

def masks(logits,positions,kinds):
    c=logits.new_zeros((*logits.shape[:2],1));s=torch.zeros_like(c)
    for b,(pos,ys) in enumerate(zip(positions,kinds)):
        for p,y in zip(pos,ys):(s if y else c)[b,p,0]=1
    return c,s

@contextlib.contextmanager
def rng_scope(seed,device,native=None):
    py=random.getstate();numpy=np.random.get_state()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            torch.manual_seed(seed)
            if devices:torch.cuda.manual_seed_all(seed)
            if native is not None:
                torch.set_rng_state(native['cpu'])
                if device.type=='cuda':torch.cuda.set_rng_state(native['cuda'],device=device)
                if 'python' in native:random.setstate(native['python'])
                if 'numpy' in native:np.random.set_state(native['numpy'])
            yield
        finally:
            random.setstate(py);np.random.set_state(numpy)

def native_rng(stage,arm,device):
    if stage=='final':
        path=model_root()/'runs'/arm/'checkpoint-1125/rng_state.pth'
        state=torch.load(path,map_location='cpu',weights_only=False)
        if isinstance(state['cuda'],list):state['cuda']=state['cuda'][0]
        state['source']=str(path)
    else:
        state={'cpu':torch.get_rng_state().clone(),'cuda':torch.cuda.get_rng_state(device).clone(),'python':random.getstate(),'numpy':np.random.get_state(),'source':'Original seed-3407 initial LF model loading, before any probe forward'}
    state['cuda_sha256']=hashlib.sha256(state['cuda'].numpy().tobytes()).hexdigest()
    return state

def collect_window(model,batches,params,objectives,device,seed,need_plain_stop,end_id=151668,extra_totals=(),native=None):
    global LOCAL_BUFFER_BYTES
    LOCAL_BUFFER_BYTES=0
    tokens=sum(int(b['labels'][:,1:].ne(-100).sum()) for b in batches)
    thinks=sum(bool(p) for b in batches for p in b['term_positions'])
    if tokens<=0:raise ValueError('Invalid window CE denominator')
    den=max(thinks,1);buffers={};numbers={'ce':0.};counts={'think':0,'continue':0,'end_only':0}
    with rng_scope(seed,device,native):
        for micro,b in enumerate(batches):
            clean={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items() if k not in ['term_positions','term_targets','source_index']}
            context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
            with context:
                output=model(**clean,num_items_in_batch=tokens,use_cache=False)
                # Accelerator's training wrapper casts model outputs to FP32 too.
                z=output.logits.float()
                losses,stats,parts=gold_boundary_losses(z,clean['labels'],b['term_positions'],b['term_targets'],end_id,clean['input_ids'])
            if not torch.isfinite(output.loss):raise FloatingPointError('Non-finite CE loss')
            c,s=masks(z,b['term_positions'],b['term_targets'])
            requests=[('ce',output.loss,None)]
            for key in objectives:
                requests += [(key+'_c',losses[key]/den,c),(key+'_s',losses[key]/den,s),(key+'_a',losses[key]/den,None)]
                numbers[key+'_c']=numbers.get(key+'_c',0.)+parts[key+'_continue']/den
                numbers[key+'_s']=numbers.get(key+'_s',0.)+parts[key+'_end']/den
                numbers[key+'_a']=numbers.get(key+'_a',0.)+float(losses[key].detach())/den
            for key in extra_totals:
                assert key not in objectives
                requests.append((key+'_a',losses[key]/den,None))
                numbers[key+'_a']=numbers.get(key+'_a',0.)+float(losses[key].detach())/den
            if need_plain_stop and 'gold' not in objectives:
                requests.append(('plain_stop',losses['gold']/den,s))
            for name,loss,mask in requests:
                if not torch.isfinite(loss):raise FloatingPointError(name)
                gradients=component_grad(loss,z,params,mask)
                buffers[name]=add(buffers.get(name),gradients,name)
                del gradients
            numbers['ce']+=float(output.loss.detach())
            for key in counts:counts[key]+=stats[key]
            del requests,output,losses,z,c,s,clean,loss,mask
            print('MICRO',micro+1,'/',len(batches),flush=True)
    return buffers,{'window_ce_tokens':tokens,'window_think':thinks,'microbatches':len(batches),'counts':counts,'losses':numbers}

def statistics(keys,matrix,objective,plain_stop_key,reference_key,matched,coefficient=COEF):
    ix={k:i for i,k in enumerate(keys)}
    def dot(a,b):return float(matrix[ix[a],ix[b]])
    def norm(a):return math.sqrt(max(0.,dot(a,a)))
    def cosine(a,b):
        den=norm(a)*norm(b);return dot(a,b)/den if den else None
    c,s,a=objective+'_c',objective+'_s',objective+'_a'
    nc,ns,na,nce=norm(c),norm(s),norm(a),norm('ce')
    if nce<=0:raise FloatingPointError('Zero CE gradient')
    target=norm(reference_key) if matched else na
    if matched and na==0 and target!=0:raise FloatingPointError('Unmatchable zero denominator')
    scale=target/na if matched and na else 1.
    sum2=dot(c,c)+dot(s,s)+2*dot(c,s)
    # Independent unmasked VJP is compared with the sum of the masked VJPs.
    error2=max(0.,sum2+na*na-2*dot(c,a)-2*dot(s,a))
    actual_c=dot(c,a)/(na*na) if na else None
    actual_s=dot(s,a)/(na*na) if na else None
    ns_sum=math.sqrt(max(0.,sum2))
    sum_c=(dot(c,c)+dot(c,s))/sum2 if sum2>0 else None
    sum_s=(dot(s,s)+dot(c,s))/sum2 if sum2>0 else None
    nplain=norm(plain_stop_key)
    cos_plain=cosine(a,plain_stop_key)
    residual_to_plain=math.sqrt(max(0.,1-cos_plain*cos_plain)) if cos_plain is not None else None
    return {'continue_grad_norm':nc,'stop_grad_norm':ns,'aux_actual_grad_norm':na,'aux_components_sum_norm':ns_sum,
      'ce_grad_norm':nce,'continue_stop_norm_ratio':nc/ns if ns else None,
      'cos_continue_stop':cosine(c,s),'cos_continue_ce':cosine(c,'ce'),'cos_stop_ce':cosine(s,'ce'),'cos_aux_ce':cosine(a,'ce'),
      'cos_sum_aux_actual':(dot(c,a)+dot(s,a))/(ns_sum*na) if ns_sum*na else None,
      'sum_vs_original_relative_l2_error':math.sqrt(error2)/na if na else 0.,
      'continue_projection_share_sum':sum_c,'stop_projection_share_sum':sum_s,
      'continue_projection_share_actual':actual_c,'stop_projection_share_actual':actual_s,
      'rounding_residual_projection_share_actual':1-actual_c-actual_s if na else None,
      'continue_projection_length_actual':dot(c,a)/na if na else None,'stop_projection_length_actual':dot(s,a)/na if na else None,
      'cancellation_factor':ns_sum/(nc+ns) if nc+ns else None,
      'plain_balanced_stop_ce_grad_norm':nplain,'cos_aux_plain_stop_ce':cos_plain,
      'relative_residual_after_best_fit_stop_ce':residual_to_plain,
      'best_fit_stop_ce_multiplier':dot(a,plain_stop_key)/(nplain*nplain) if nplain else None,
      'lambda':coefficient,'matched':matched,'probe_match_coefficient':scale,'counterfactual_dynamic_gold_norm':target if matched else None,
      'effective_scale':coefficient*scale,'effective_continue_norm':coefficient*scale*nc,'effective_stop_norm':coefficient*scale*ns,
      'effective_aux_norm':coefficient*scale*na,'effective_continue_R':coefficient*scale*nc/nce,
      'effective_stop_R':coefficient*scale*ns/nce,'effective_aux_R':coefficient*scale*na/nce,
      'cos_aux_plain_stop_ce_note':'Plain stop CE uses the original category coefficients and window Think denominator; Gold stop CE identity is unit-verified.'}

def vector_hash(model):return original.legacy.init_hash(SimpleNamespace(model=model))

def make_model(arm,stage,device):
    cfg=original.config(arm,False);cfg['output_dir']=str(HERE/'load_only'/stage/arm)
    checkpoint=None
    if stage=='final':
        checkpoint=model_root()/'runs'/arm/'checkpoint-1125'
        assert json.loads((checkpoint/'trainer_state.json').read_text())['global_step']==1125
        cfg['adapter_name_or_path']=str(checkpoint)
        cfg['create_new_adapter']=False
    # Reuse exactly the LF model-loading APIs of the frozen runner, without a Trainer/optimizer.
    original.legacy.__dict__['set_seed'](3407)
    # Parse as a CPU loading context so LF does not demand distributed TRAINING
    # merely because a read-only pair comparison exposes two CUDA devices.
    # BF16/model/adapter arguments are unchanged; actual placement is explicit.
    parsing_cfg=dict(cfg,use_cpu=True)
    ma,da,ta,fa,_=original.legacy.get_train_args(parsing_cfg)
    tokenmod=original.legacy.load_tokenizer(ma)
    template=original.legacy.get_template_and_fix_tokenizer(tokenmod['tokenizer'],da)
    model=original.legacy.load_model(tokenmod['tokenizer'],ma,fa,True).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==835190784
    if checkpoint is not None:
        # Load native FP32 adapter/lm_head values directly, after LF's BF16 construction.
        # No per-tensor comparison or repeated weight hash scan is needed for attribution.
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        state=load_file(checkpoint/'adapter_model.safetensors',device='cpu')
        set_peft_model_state_dict(model,state,adapter_name='default')
        del state
    evidence={'arm':arm,'stage':stage,'config':cfg,'checkpoint':str(checkpoint) if checkpoint else None,
      'trainable_parameters':sum(p.numel() for p in params),'trainable_before_sha256':vector_hash(model) if stage=='initial' else None,
      'parameter_layout':[(n,list(p.shape)) for n,p in model.named_parameters() if p.requires_grad],
      'parameter_versions_before':[(n,p._version) for n,p in model.named_parameters()]}
    if stage=='initial':assert evidence['trainable_before_sha256']==json.loads((EXPERIMENT/'gold_calibration.json').read_text())['initial_trainable_sha256']
    if checkpoint is not None:evidence['weight_comparison_skipped_per_user']=True
    dump(HERE/'sources'/f'{stage}_{arm}.json',evidence)
    return model,params,evidence

def prepare_windows():
    assert not (CACHE/'fixed_windows.pt').exists()
    cfg=original.config('G1',False);cfg['output_dir']=str(HERE/'prepare_only')
    trainer=original.legacy.build(cfg)
    assert trainer.optimizer is None
    assert vector_hash(trainer.model)==json.loads((EXPERIMENT/'gold_calibration.json').read_text())['initial_trainable_sha256']
    calibration=json.loads((EXPERIMENT/'gold_calibration.json').read_text())['rows']
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        loader=iter(trainer.get_train_dataloader());windows=[];signatures=[]
        for i in range(8):
            batch=[next(loader) for _ in range(16)]
            cpu=[{k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in b.items()} for b in batch]
            assert sum(int(b['labels'][:,1:].ne(-100).sum()) for b in cpu)==calibration[i]['window_ce_tokens']
            assert sum(bool(p) for b in cpu for p in b['term_positions'])==calibration[i]['window_think']
            h=hashlib.sha256()
            for b in cpu:
                for key in ['input_ids','labels','attention_mask']:h.update(b[key].contiguous().numpy().tobytes())
            signatures.append(h.hexdigest());windows.append(cpu)
        del loader,batch
    for arm in ARMS:assert signatures==json.loads((model_root()/'runs'/arm/'fixed_probe_windows.json').read_text())
    CACHE.mkdir(parents=True,exist_ok=True);torch.save(windows,CACHE/'fixed_windows.pt')
    dump(CACHE/'manifest.json',{'signatures':signatures,'cache_sha256':sha(CACHE/'fixed_windows.pt'),'origin':'Exact replay of the original fixed eight optimizer windows; all five saved signatures verified','windows':8,'microbatches_per_window':16})
    del trainer;gc.collect();torch.cuda.empty_cache()
    print('FIXED_WINDOWS_VERIFIED',flush=True)

def finish_model(model,evidence):
    assert all(p.grad is None for p in model.parameters())
    assert [(n,p._version) for n,p in model.named_parameters()]==evidence['parameter_versions_before']
    after=vector_hash(model) if evidence['trainable_before_sha256'] is not None else None
    assert after==evidence['trainable_before_sha256']
    evidence.update(trainable_after_sha256=after,parameters_unchanged=True,no_optimizer_created=True,no_training_call=True)
    dump(HERE/'sources'/f"{evidence['stage']}_{evidence['arm']}.json",evidence)

def case_rows(buffers,meta,arms,stage,index,params):
    keys,matrix=gram(buffers,[p.numel() for p in params]);rows=[]
    for arm in arms:
        objective=ARMS[arm];plain='gold_s' if 'gold_s' in buffers else 'plain_stop'
        stats=statistics(keys,matrix,objective,plain,'dynamic_gold_a',arm.endswith('-M'))
        numbers=meta['losses']
        row={'arm':arm,'stage':stage,'window_index':index,'analysis_seed':3407,**meta,**stats,
          'continue_loss':numbers[objective+'_c'],'stop_loss':numbers[objective+'_s'],'aux_loss':numbers[objective+'_a'],
          'effective_continue_loss':stats['effective_scale']*numbers[objective+'_c'],'effective_stop_loss':stats['effective_scale']*numbers[objective+'_s'],
          'precision_note':'Original BF16 forward/backward. Independent component VJPs can have a rounding residual; both sum-axis and actual-original-axis projections are reported.'}
        rows.append(row)
    if stage=='initial':
        ix={k:j for j,k in enumerate(keys)}
        def same_model_cos(x,y):
            den=math.sqrt(max(0.,float(matrix[ix[x],ix[x]]*matrix[ix[y],ix[y]])))
            return float(matrix[ix[x],ix[y]])/den if den else None
        append_rows(HERE/'raw/initial_pair_cross.jsonl',[{'stage':'initial','window_index':index,
          'cos_cross_model_aux':same_model_cos('dynamic_gold_a','dynamic_ce_a'),
          'cos_cross_model_continue':same_model_cos('dynamic_gold_c','dynamic_ce_c'),
          'cos_cross_model_stop':same_model_cos('dynamic_gold_s','dynamic_ce_s'),
          'coordinate_system':'Same initial parameters and forward pass; the positive matching scalar does not change direction.'}])
    return rows

def append_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def run_single(arm,stage,initial=False):
    output=HERE/'raw'/('initial_shared.jsonl' if initial else 'final_'+arm+'.jsonl')
    if initial:assert not output.exists()
    saved=[json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
    assert all(r['arm']==arm and r['stage']==stage for r in saved)
    done={r['window_index'] for r in saved};assert len(done)==len(saved)
    cross_path=HERE/'raw/final_pair_cross.jsonl'
    cross_done={json.loads(x)['window_index'] for x in cross_path.read_text().splitlines()} if cross_path.exists() else set()
    device=torch.device('cuda:0');model,params,source=make_model(arm,stage,device)
    rng=native_rng(stage,arm,device);source.update(rng_source=rng['source'],rng_cuda_sha256=rng['cuda_sha256'])
    windows=torch.load(CACHE/'fixed_windows.pt',weights_only=True)
    for i,batches in enumerate(windows):
        if i in done and (arm not in ['G2','G3-M'] or i in cross_done):continue
        objectives=['gold','dynamic_gold','dynamic_ce'] if initial else [ARMS[arm]]
        extra=['dynamic_gold'] if arm.endswith('-M') and 'dynamic_gold' not in objectives else []
        buffers,meta=collect_window(model,batches,params,objectives,device,3407,True,extra_totals=extra,native=rng)
        meta.update(rng_source=rng['source'],rng_cuda_sha256=rng['cuda_sha256'])
        rows=case_rows(buffers,meta,list(ARMS) if initial else [arm],stage,i,params)
        if i not in done:append_rows(output,rows)
        if stage=='final' and arm=='G2':
            folder=Path(os.environ['ATTR_CROSS_DIR']);path=folder/f'window_{i}.pt'
            while any(folder.glob('window_*.pt')):time.sleep(2)
            tmp=path.with_suffix('.tmp')
            torch.save({k:buffers['dynamic_gold_'+k] for k in ['c','s','a']},tmp);tmp.replace(path)
        if stage=='final' and arm=='G3-M':
            path=Path(os.environ['ATTR_CROSS_DIR'])/f'window_{i}.pt'
            while not path.exists():time.sleep(2)
            payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            aligned={**{'G2_'+k:payload[k] for k in ['c','s','a']},**{'G3M_'+k:buffers['dynamic_ce_'+k] for k in ['c','s','a']}}
            keys,g=gram(aligned,[p.numel() for p in params]);ix={k:j for j,k in enumerate(keys)}
            def cos(x,y):
                den=math.sqrt(max(0.,float(g[ix[x],ix[x]]*g[ix[y],ix[y]])))
                return float(g[ix[x],ix[y]])/den if den else None
            append_rows(HERE/'raw/final_pair_cross.jsonl',[{'stage':'final','window_index':i,'cos_cross_model_aux':cos('G2_a','G3M_a'),'cos_cross_model_continue':cos('G2_c','G3M_c'),'cos_cross_model_stop':cos('G2_s','G3M_s'),'coordinate_system':'Identical parameter names and shapes, different trained parameter values.'}])
            del aligned,payload;gc.collect();path.unlink()
        del buffers,rows
        gc.collect()
        for cuda_device in range(torch.cuda.device_count()):
            with torch.cuda.device(cuda_device):torch.cuda.empty_cache()
        print('WINDOW_DONE',stage,arm,i,flush=True)
    finish_model(model,source)
    dump(HERE/'completion'/('initial_shared.json' if initial else 'final_'+arm+'.json'),{'complete':True,'windows':8,'arms':list(ARMS) if initial else [arm]})

def run_pair():
    assert torch.cuda.device_count()==2
    path=HERE/'raw/final_pair.jsonl';cross=HERE/'raw/final_pair_cross.jsonl'
    assert not path.exists() and not cross.exists()
    model1,p1,e1=make_model('G2','final',torch.device('cuda:0'))
    model2,p2,e2=make_model('G3-M','final',torch.device('cuda:1'))
    assert e1['parameter_layout']==e2['parameter_layout']
    rng1=native_rng('final','G2',torch.device('cuda:0'));rng2=native_rng('final','G3-M',torch.device('cuda:1'))
    assert rng1['cuda_sha256']==rng2['cuda_sha256']
    for e,rng in [(e1,rng1),(e2,rng2)]:e.update(rng_source=rng['source'],rng_cuda_sha256=rng['cuda_sha256'])
    windows=torch.load(CACHE/'fixed_windows.pt',weights_only=True)
    for i,b in enumerate(windows):
        first,meta=collect_window(model1,b,p1,['dynamic_gold'],torch.device('cuda:0'),3407,True,native=rng1)
        meta.update(rng_source=rng1['source'],rng_cuda_sha256=rng1['cuda_sha256'])
        r1=case_rows(first,meta,['G2'],'final',i,p1)[0];append_rows(path,[r1])
        keep={k:first[k] for k in ['dynamic_gold_c','dynamic_gold_s','dynamic_gold_a']};del first
        second,meta=collect_window(model2,b,p2,['dynamic_ce'],torch.device('cuda:1'),3407,True,extra_totals=['dynamic_gold'],native=rng2)
        meta.update(rng_source=rng2['source'],rng_cuda_sha256=rng2['cuda_sha256'])
        r2=case_rows(second,meta,['G3-M'],'final',i,p2)[0];append_rows(path,[r2])
        aligned={**{'G2_'+k:keep['dynamic_gold_'+k] for k in ['c','s','a']},**{'G3M_'+k:second['dynamic_ce_'+k] for k in ['c','s','a']}}
        keys,g=gram(aligned,[p.numel() for p in p1]);ix={k:j for j,k in enumerate(keys)}
        def cos(x,y):
            den=math.sqrt(max(0.,float(g[ix[x],ix[x]]*g[ix[y],ix[y]])))
            return float(g[ix[x],ix[y]])/den if den else None
        append_rows(cross,[{'stage':'final','window_index':i,'cos_cross_model_aux':cos('G2_a','G3M_a'),'cos_cross_model_continue':cos('G2_c','G3M_c'),'cos_cross_model_stop':cos('G2_s','G3M_s'),'G2_effective_aux_norm':r1['effective_aux_norm'],'G3M_effective_aux_norm':r2['effective_aux_norm'],'coordinate_system':'Identical ordered LoRA/lm_head parameter names and shapes; parameter values differ; this is not functional equivalence.'}])
        del keep,second,aligned
        print('PAIR_WINDOW_DONE',i,flush=True)
    finish_model(model1,e1);finish_model(model2,e2)
    dump(HERE/'completion/final_pair.json',{'complete':True,'windows':8,'arms':['G2','G3-M']})

def main():
    torch.set_num_threads(8)
    parser=argparse.ArgumentParser();parser.add_argument('job',choices=['prepare','initial','G1','G2','G3','G1-M','G3-M','pair']);args=parser.parse_args()
    if args.job=='prepare':prepare_windows()
    elif args.job=='initial':run_single('G1','initial',True)
    elif args.job=='pair':run_pair()
    else:run_single(args.job,'final')

if __name__=='__main__':main()
