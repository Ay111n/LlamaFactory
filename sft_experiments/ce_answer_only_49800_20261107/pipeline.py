"""One exclusive job: technical preflight, fresh full training, then seven DEV checkpoints."""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import *

@contextmanager
def gpu_lease():
    while True:
        rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"],text=True)
        apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True)
        occupied={r.split(",")[0].strip() for r in apps.splitlines()}
        lease=None
        for line in rows.splitlines():
            index,uuid,name,memory,util=[v.strip() for v in line.split(",")]
            if "L20" not in name or uuid in occupied or int(memory)>64 or int(util)>5:continue
            handle=Path(f"/tmp/thesis_ce_answer_only_gpu{index}.lock").open("a")
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            # Close the race between the inventory and the lock.
            current=subprocess.check_output(["nvidia-smi","-i",index,"--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"],text=True).strip().split(",")
            current_apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits"],text=True)
            if int(current[0])<=64 and int(current[1])<=5 and not any(r.split(",")[0].strip()==uuid for r in current_apps.splitlines()):
                lease=(int(index),uuid,handle)
                break
            handle.close()
        if lease:
            break
        write(ARTIFACTS/"resource_wait.json",{"waiting_for_idle_L20":True,"time":now()})
        time.sleep(30)
    try:
        yield lease[0],lease[1]
    finally:
        lease[2].close()

def run(command,log,env):
    with Path(log).open("x") as stream:
        p=subprocess.Popen(command,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        write(ARTIFACTS/"child.json",{"pid":p.pid,"command":command,"log":str(log),"time":now()})
        samples=[]
        while p.poll() is None:
            if env.get("CUDA_VISIBLE_DEVICES"):
                value=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"],text=True)
                samples.append({"time":now(),"gpu_usage":value.strip()})
            time.sleep(5)
        write(Path(log).with_suffix(".resource.json"),{"samples":samples,"returncode":p.returncode})
        if p.returncode:
            raise subprocess.CalledProcessError(p.returncode,command)

def stage(value,**extra):
    write(ARTIFACTS/"pipeline.json",{"state":"running","stage":value,"pid":os.getpid(),"time":now(),**extra})

def main():
    assert sys.executable==PYTHON
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    lock=(ARTIFACTS/"pipeline.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="8",CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONPATH=str(FRAMEWORK/"src"))
    try:
        assert read(ARTIFACTS/"prepared.json")["passed"]
        for phase in ("preflight","main"):
            done=ARTIFACTS/phase/"completion.json"
            expected=100 if phase=="preflight" else 9339
            if done.exists():
                assert read(done)["completed"]
                checkpoint(MODELS/phase/f"checkpoint-{expected}",expected)
            else:
                with gpu_lease() as (gpu,uuid):
                    stage("training_"+phase,gpu=gpu,gpu_uuid=uuid)
                    gpuenv=dict(env,CUDA_VISIBLE_DEVICES=uuid)
                    command=[PYTHON,"-u",str(HERE/"train.py"),"--phase",phase]
                    # Resume only this run's last audited checkpoint, never a preflight adapter.
                    saved=sorted((ARTIFACTS/phase).glob("checkpoint-*.json"),
                        key=lambda p:int(p.stem.split("-")[-1])) if (ARTIFACTS/phase).exists() else []
                    if saved:
                        resume=read(saved[-1])
                        checkpoint(resume["path"],resume["step"])
                        command.extend(["--resume",resume["path"]])
                    attempt=len(list(ARTIFACTS.glob(phase+"-attempt*.log")))+1
                    run(command,ARTIFACTS/f"{phase}-attempt{attempt}.log",gpuenv)
                assert read(done)["completed"]
            if phase=="preflight":
                assert read(done)["last_gradient_norm"]>=0
                if not (ARTIFACTS/"preflight_reload.json").exists():
                    stage("preflight_cpu_reload")
                    run([INFERENCE,str(THESIS/"finetuning/scripts/model"/NAME/"verify_adapter.py"),
                        "--base",BASE,"--adapter",str(MODELS/"preflight/checkpoint-100"),
                        "--output",str(ARTIFACTS/"preflight_reload.json")],
                        ARTIFACTS/"preflight_reload.log",dict(env,CUDA_VISIBLE_DEVICES=""))
                assert read(ARTIFACTS/"preflight_reload.json")["passed"]
            if args.preflight_only:
                write(ARTIFACTS/"pipeline.json",{"state":"preflight_passed","pid":os.getpid(),"time":now()})
                return
        for step in STEPS:
            checkpoint(MODELS/"main"/f"checkpoint-{step}",step)
        evaluator=BENCHMARK/"eval/scripts"/NAME/"workflow.py"
        for step in STEPS:
            if (ANALYSIS/f"on-{step}.json").exists():continue
            with gpu_lease() as (gpu,uuid):
                stage("dev_on",step=step,gpu=gpu,gpu_uuid=uuid)
                run([INFERENCE,str(evaluator),"--step",str(step),"--mode","on","--gpu",str(gpu)],
                    ARTIFACTS/f"dev-on-{step}.log",dict(env,CUDA_VISIBLE_DEVICES=""))
        comparison=read(ANALYSIS/"comparison.json")
        assert comparison["complete_on"]
        best=comparison["best_on"]["a_step"]
        if not (ANALYSIS/f"off-{best}.json").exists():
            with gpu_lease() as (gpu,uuid):
                stage("dev_off",step=best,gpu=gpu,gpu_uuid=uuid)
                run([INFERENCE,str(evaluator),"--step",str(best),"--mode","off","--gpu",str(gpu)],
                    ARTIFACTS/f"dev-off-{best}.log",dict(env,CUDA_VISIBLE_DEVICES=""))
        comparison=read(ANALYSIS/"comparison.json")
        assert "best_off" in comparison
        write(ARTIFACTS/"pipeline.json",{"state":"complete","pid":os.getpid(),"time":now(),
            "trained_steps":9339,"evaluated_steps":STEPS,"off_step":best,"analysis":str(ANALYSIS/"report.md")})
    except BaseException as error:
        write(ARTIFACTS/"pipeline.json",{"state":"failed","pid":os.getpid(),"time":now(),
            "error":repr(error),"traceback":traceback.format_exc()})
        raise
    finally:
        lock.close()

if __name__=="__main__":
    main()
