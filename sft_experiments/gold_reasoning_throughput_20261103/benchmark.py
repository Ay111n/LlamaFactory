"""Isolated correctness/throughput probes; never starts full training."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PREFLIGHT = ROOT.parent / "gold_reasoning_preflight_20261103"
sys.path.insert(0, str(PREFLIGHT))
import torch
from transformers import AutoModelForCausalLM
import run_preflight as original
from losses import ReferenceTargets
from reasoning_trainer import ReasoningTrainer
from optimized import CachedTargets, PipelineTargets, PipelineTrainer


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def setup(variant, run_id, steps):
    original.ROOT = ROOT
    original.MODEL_ROOT = Path('/home/nlp/data/lyl/finetuning/model/sft_experiments/gold_reasoning_throughput_20261103')
    original.ReasoningTrainer = PipelineTrainer if variant == "pipeline" else ReasoningTrainer
    trainer, _, _, directory = original.build("benchmark", run_id, steps)
    teacher = AutoModelForCausalLM.from_pretrained(original.REF, local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa", device_map={"":"cuda:1"})
    teacher.eval().requires_grad_(False)
    provider = {"baseline":ReferenceTargets, "cached":CachedTargets, "pipeline":PipelineTargets}[variant]
    trainer.reference = provider(teacher, beta=.5, cap=2., chunk=64)
    write(directory/'protocol.json', {"variant":variant,"optimizer_step_cap":steps,
        "full_training":False,"student_fresh_sft_new2":True,"teacher_FP32":True,"TF32":False,
        "lambda":.1,"mu":.1,"beta":.5,"cap":2.,"effective_batch":16,"scheduler_horizon":9339,
        "precision_loss_data_order_unchanged":True,"gpu_visible":os.environ['CUDA_VISIBLE_DEVICES']})
    return trainer, directory


def equivalence(run_id):
    os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
    trainer, directory = setup('baseline',run_id,1)
    trainer.model = trainer.accelerator.prepare_model(trainer.model)
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs)
    params=[p for p in trainer.model.parameters() if p.requires_grad]
    versions=[p._version for p in params]
    teacher=trainer.reference.model
    teacher_versions=[p._version for p in teacher.parameters()]
    audit=[json.loads(line) for line in (original.DATA/'audit.jsonl').open()]
    think=[r['index'] for r in audit if r['think']]
    no_think=[r['index'] for r in audit if not r['think']]
    longest=max((r for r in audit if r['think']),key=lambda r:r['length'])['index']
    answer=max((r for r in audit if r['think']),key=lambda r:r['answer_tokens'])['index']
    # Adjacent maximum cases also stress two simultaneously retained targets.
    windows=[('mixed_with_longest',[longest,answer]+think[:2]+no_think[:2]),('no_think_only',no_think[2:4])]
    reports=[]
    for name,indices in windows:
        batches=[trainer.data_collator([trainer.train_dataset[i]]) for i in indices]
        count=trainer._get_num_items_in_batch(batches,trainer.args.device)
        losses=[]
        baseline_grad=None
        rng_end=None
        for variant,cls in [('baseline',ReferenceTargets),('cached',CachedTargets),('pipeline',PipelineTargets)]:
            trainer.__class__=PipelineTrainer if variant=='pipeline' else ReasoningTrainer
            trainer.reference=cls(teacher,beta=.5,cap=2.,chunk=64)
            trainer.model.zero_grad(set_to_none=True)
            trainer.current_gradient_accumulation_steps=len(batches)
            trainer.start_window(batches,count)
            with torch.random.fork_rng(devices=[0,1]):
                if variant=='pipeline':
                    trainer.reference.stage(batches)
                loss=sum(float(trainer.training_step(trainer.model,b,num_items_in_batch=count)) for b in batches)
                current_rng=torch.cuda.get_rng_state(0).cpu()
            losses.append(loss)
            if variant=='baseline':
                baseline_grad=original.cpu_gradient_copy(params)
                rng_end=current_rng
                compare=None
            else:
                compare=original.compare_gradients(baseline_grad,params)
                assert math.isclose(loss,losses[0],rel_tol=2e-4,abs_tol=2e-6)
                assert torch.equal(rng_end,current_rng)
            row={'window':name,'variant':variant,'indices':indices,'loss':loss,'gradients':compare,
                'same_rng':True,'metrics':dict(trainer.window_metrics),'passed':True,
                'student_peak_gib':torch.cuda.max_memory_allocated(0)/2**30,
                'teacher_peak_gib':torch.cuda.max_memory_allocated(1)/2**30}
            reports.append(row)
            write(directory/'equivalence_partial.json',reports)
            print('EQUIVALENCE',json.dumps({k:row[k] for k in ['window','variant','loss','gradients','passed','student_peak_gib','teacher_peak_gib']}),flush=True)
            if hasattr(trainer.reference,'close'):trainer.reference.close()
            trainer.model.zero_grad(set_to_none=True)
            gc.collect()
        del baseline_grad
    assert versions==[p._version for p in params]
    assert teacher_versions==[p._version for p in teacher.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    write(directory/'completion.json',{'passed':True,'actual_8B':True,'all_trainable_parameters':sum(p.numel() for p in params),
        'optimizer_updates':0,'student_teacher_unchanged':True,'rtol':2e-4,'atol':2e-6,'windows':reports})


def train(variant,run_id,steps):
    assert 4<=steps<=20
    assert json.loads((ROOT/'equivalence/completion.json').read_text())['passed']
    directory=ROOT/run_id
    assert not (directory/'steps.jsonl').exists()
    trainer,directory=setup(variant,run_id,steps)
    reference=trainer.reference.model
    ref_versions=[p._version for p in reference.parameters()]
    torch.cuda.empty_cache()
    start=time.perf_counter()
    trainer.train(resume_from_checkpoint=None)
    elapsed=time.perf_counter()-start
    assert trainer.state.global_step==steps and trainer.state.max_steps==9339
    assert ref_versions==[p._version for p in reference.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
    measured=[json.loads(line) for line in (directory/'steps.jsonl').open()]
    steady=measured[3:]
    report={'passed':True,'variant':variant,'optimizer_updates':steps,'total_seconds':elapsed,
        'steady_steps':len(steady),'steady_mean_seconds':statistics.mean(r['seconds'] for r in steady),
        'steady_input_tokens_per_second':sum(r['input_tokens'] for r in steady)/sum(r['seconds'] for r in steady),
        'teacher_unchanged':True,'student_peak_gib':max(r['student_peak_allocated_gib'] for r in measured),
        'teacher_peak_gib':max(r['teacher_peak_allocated_gib'] for r in measured),
        'teacher_forward_calls':trainer.reference.forward_calls,'target_calls':trainer.reference.target_calls,
        'head_calls':getattr(trainer.reference,'head_calls',None),'projection_calls':getattr(trainer.reference,'projection_calls',None),
        'full_training_started':False,'model_saved':False}
    # Test the longest samples with optimizer states present, without updating.
    audit=[json.loads(line) for line in (original.DATA/'audit.jsonl').open()]
    extremes=list(dict.fromkeys([max((r for r in audit if r['think']),key=lambda r:r['length'])['index'],
        max((r for r in audit if r['think']),key=lambda r:r['answer_tokens'])['index'],
        max((r for r in audit if not r['think']),key=lambda r:r['length'])['index']]))
    trainer.model.zero_grad(set_to_none=True)
    batches=[trainer.data_collator([trainer.train_dataset[i]]) for i in extremes]
    count=trainer._get_num_items_in_batch(batches,trainer.args.device)
    trainer.current_gradient_accumulation_steps=len(batches)
    trainer.start_window(batches,count)
    if variant=='pipeline':trainer.reference.stage(batches)
    loss=sum(float(trainer.training_step(trainer.model_wrapped,b,num_items_in_batch=count)) for b in batches)
    assert math.isfinite(loss)
    report['long_sequence_probe']={'passed':True,'optimizer_states_present':bool(trainer.optimizer.state),
        'optimizer_updates':0,'indices':extremes,'student_peak_gib':torch.cuda.max_memory_allocated(0)/2**30,
        'teacher_peak_gib':torch.cuda.max_memory_allocated(1)/2**30,'loss':loss}
    trainer.model.zero_grad(set_to_none=True)
    if hasattr(trainer.reference,'close'):trainer.reference.close()
    write(directory/'completion.json',report)
    print('BENCHMARK_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['equivalence','train'])
    parser.add_argument('--variant',choices=['baseline','cached','pipeline'],default='baseline')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--steps',type=int,default=12)
    args=parser.parse_args()
    (ROOT/args.run_id).mkdir(exist_ok=True,parents=True)
    try:
        equivalence(args.run_id) if args.mode=='equivalence' else train(args.variant,args.run_id,args.steps)
    except BaseException as error:
        write(ROOT/args.run_id/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'time':time.time()})
        raise
