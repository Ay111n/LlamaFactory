"""Run native CE with telemetry; no custom loss or custom Trainer."""
import argparse
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import sys
import time
import torch
import yaml
from safetensors import safe_open
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp
from common import *

class AuditCallback(TrainerCallback):
    def __init__(self, phase, resumed):
        self.phase, self.resumed = phase, resumed
        self.directory = ARTIFACTS/phase
        self.directory.mkdir(parents=True,exist_ok=True)
        self.started = None
        self.last = None
        self.durations = []
        self.last_grad_norm = None
        self.microbatches = 0

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        assert torch.cuda.device_count()==1
        assert args.world_size==1 and args.gradient_accumulation_steps==16
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert any("lm_head" in n for n,p in trainable)
        lora_b=[p for n,p in trainable if "lora_B" in n]
        fresh=all(not bool(torch.count_nonzero(p)) for p in lora_b)
        if not self.resumed: assert fresh and state.global_step==0
        write(self.directory/"initialization.json",{
            "base":BASE,"fresh_lora_B_zero":fresh,"resume":self.resumed,
            "trainable_parameters":sum(p.numel() for _,p in trainable),
            "trainer":"native CustomSeq2SeqTrainer CE","seed":args.seed,
            "total_steps":state.max_steps,"max_input_length":4096,"time":now(),
            "versions":{p:importlib.metadata.version(p) for p in ("torch","transformers","peft","accelerate","datasets","flash-attn")},
            "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES")})
        assert state.max_steps==(100 if self.phase=="preflight" else 9339)
        torch.cuda.reset_peak_memory_stats()
        self.started=self.last=time.monotonic()

    def on_step_begin(self,args,state,control,**kwargs):
        self.last=time.monotonic()

    def on_substep_end(self,args,state,control,**kwargs):
        self.microbatches += 1

    def on_step_end(self,args,state,control,**kwargs):
        self.microbatches += 1
        torch.cuda.synchronize()
        duration=time.monotonic()-self.last
        self.durations.append(duration)
        value={"step":state.global_step,"max_steps":state.max_steps,"epoch":state.epoch,
            "seconds":duration,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"time":now()}
        append(self.directory/"steps.jsonl",value)
        write(self.directory/"progress.json",value)
        if state.global_step==state.max_steps:control.should_save=True
        return control

    def on_log(self,args,state,control,logs=None,**kwargs):
        if logs:
            for key in ("loss","grad_norm"):
                if key in logs and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"Nonfinite {key}: {logs}")
            if "grad_norm" in logs:self.last_grad_norm=float(logs["grad_norm"])
            append(self.directory/"trainer_logs.jsonl",{"step":state.global_step,**logs,"time":now()})

    def on_save(self,args,state,control,**kwargs):
        p=Path(args.output_dir)/f"checkpoint-{state.global_step}"
        record=checkpoint(p,state.global_step)
        with safe_open(p/"adapter_model.safetensors",framework="pt",device="cpu") as f:
            assert any("lm_head" in n for n in f.keys())
        record["phase"]=self.phase
        write(self.directory/f"checkpoint-{state.global_step}.json",record)

    def on_train_end(self,args,state,control,**kwargs):
        assert state.global_step==state.max_steps
        if self.phase=="main" and not self.resumed:
            assert self.microbatches==149400 and abs(state.epoch-3.0)<1e-8
        assert self.last_grad_norm is not None and math.isfinite(self.last_grad_norm)
        times=self.durations[5:] or self.durations
        write(self.directory/"completion.json",{
            "completed":True,"step":state.global_step,"epoch":state.epoch,
            "observed_example_presentations_this_process":self.microbatches,
            "duration_seconds":time.monotonic()-self.started,
            "steady_mean_step_seconds":sum(times)/len(times),"last_gradient_norm":self.last_grad_norm,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"time":now()})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["preflight","main"],required=True)
    parser.add_argument("--resume")
    args=parser.parse_args()
    assert sys.executable==PYTHON
    if args.phase=="main":
        for filename,expected in read(ARTIFACTS/"source_manifest.json")["files"].items():
            assert sha(filename)==expected,filename
    prepared=read(ARTIFACTS/"prepared.json")
    assert prepared["passed"]
    for relative, expected in prepared["cache_sha256"].items():
        assert sha(PROCESSED/"dataset"/relative)==expected
    cfg=yaml.safe_load((ARTIFACTS/(args.phase+".yaml")).read_text())
    assert cfg["model_name_or_path"]==BASE and not cfg.get("adapter_name_or_path")
    output=Path(cfg["output_dir"])
    if args.resume:
        resume=Path(args.resume).resolve()
        assert resume.parent==output.resolve() and resume.name.startswith("checkpoint-")
        checkpoint(resume,int(resume.name.split("-")[-1]))
        cfg["resume_from_checkpoint"]=str(resume)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing implicit resume/overwrite: {output}")
    assert not cfg.get("termination_dataset_path") and not cfg.get("use_asft_loss",False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    run_exp(cfg,callbacks=[AuditCallback(args.phase,args.resume)])

if __name__=="__main__":
    main()
