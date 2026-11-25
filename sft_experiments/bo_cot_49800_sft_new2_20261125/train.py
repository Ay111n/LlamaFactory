import hashlib
import json
import math
import sys
from itertools import islice
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.tuner import run_exp
from transformers import TrainerCallback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
R = HERE / "runtime"
DATA = PROJECT / "finetuning/sft/mixed_sft/final_data"
SOURCE = DATA / "bo_cot_bilingual_49800_20261121/tibetan_reasoning_49800_sharegpt.jsonl"
OLD = DATA / "en_cot_49800_20260919/en_cot_49800_sharegpt.jsonl"
OUTPUT = PROJECT / "finetuning/model/sft/bo_cot_49800_sft_new2_20261125"
REGISTRY = {"bo_cot_49800": {"file_name": str(SOURCE), "formatting": "sharegpt",
    "columns": {"messages": "messages"}, "tags": {"role_tag": "role", "content_tag": "content",
    "user_tag": "user", "assistant_tag": "assistant"}}}
CFG = dict(model_name_or_path="/home/nlp/data/model/LLM/merged/qwen3-8bbo-sft-new2",
    stage="sft", do_train=True, finetuning_type="lora", lora_target="all", lora_rank=64, lora_alpha=128,
    lora_dropout=0.05, additional_target="lm_head", dataset="bo_cot_49800", dataset_dir=str(R),
    template="qwen3", cutoff_len=6144, packing=False, train_on_prompt=False,
    preprocessing_num_workers=16, dataloader_num_workers=4, output_dir=str(OUTPUT),
    per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=1e-4,
    num_train_epochs=3, warmup_ratio=0.05, lr_scheduler_type="cosine", bf16=True,
    gradient_checkpointing=True, flash_attn="fa2", seed=3407, data_seed=3407,
    save_strategy="steps", save_steps=1500, save_total_limit=2, save_only_model=False,
    logging_steps=10, report_to="none", plot_loss=True)


def save(name, payload):
    p=R/name
    t=p.with_suffix(".tmp")
    t.write_text(json.dumps(payload,indent=2)+"\n")
    t.replace(p)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def prepare():
    assert not OUTPUT.exists()
    meta=json.loads(SOURCE.with_name("manifest.json").read_text())["outputs"][SOURCE.name]
    assert sha(SOURCE)==meta["sha256"]
    assert sha(OLD)=="02d330dfa0b69ad63888d5a50d4ed25b3a135ed980f3498a61bb70c7d55f5b33"
    save("dataset_info.json",REGISTRY)
    m,d,t,*_=get_train_args(CFG)
    module=load_tokenizer(m)
    tok=module["tokenizer"]
    template=get_template_and_fix_tokenizer(tok,d)
    dataset=get_dataset(template,m,d,t,stage="sft",**module)["train_dataset"]
    assert len(dataset)==49800
    seen=set()
    order=hashlib.sha256()
    lengths=[]
    counts=dict(think=0,supervised=0,over4096=0,english_supervised=0)
    with SOURCE.open(encoding="utf8") as source, OLD.open(encoding="utf8") as old:
        for start in range(0,len(dataset),128):
            a=[json.loads(x) for x in islice(source,128)]
            b=[json.loads(x) for x in islice(old,128)]
            data=dataset[start:start+len(a)]
            prompts=[]
            targets=[]
            english=[]
            for x,y in zip(a,b,strict=True):
                u,s=x["messages"]
                ou,os=y["messages"]
                uid=x["uid"]
                assert uid not in seen and u==ou and u["role"]=="user" and s["role"]==os["role"]=="assistant"
                seen.add(uid)
                order.update((uid+"\n").encode())
                trace,sep,answer=s["content"].partition("</think>")
                otr,osep,oans=os["content"].partition("</think>")
                assert sep==osep=="</think>" and answer==oans and answer.strip()
                assert trace.startswith("<think>\n") and otr.startswith("<think>\n")
                is_think=bool(trace.removeprefix("<think>").strip())
                assert is_think==bool(otr.removeprefix("<think>").strip())
                counts["think"]+=is_think
                prompts.append(f"<|im_start|>user\n{u['content']}<|im_end|>\n<|im_start|>assistant\n")
                targets.append(s["content"]+"<|im_end|>\n")
                english.append(os["content"]+"<|im_end|>\n")
            enc=lambda x:tok(x,add_special_tokens=False,verbose=False).input_ids
            counts["english_supervised"]+=sum(map(len,enc(english)))
            for row,p,z,ids,labels in zip(a,enc(prompts),enc(targets),data["input_ids"],data["labels"],strict=True):
                assert ids==p+z and labels==[-100]*len(p)+z,row["uid"]
                assert len(ids)<=6144
                lengths.append(len(ids))
                counts["supervised"]+=len(z)
                counts["over4096"]+=len(ids)>4096
        assert not source.read(1) and not old.read(1)
    assert counts["think"]==26419 and order.hexdigest()==meta["id_order_sha256"]
    longest=max(range(len(lengths)),key=lengths.__getitem__)
    DatasetDict(train=dataset).save_to_disk(str(R/"encoded"))
    DatasetDict(train=dataset.select([longest]*16)).save_to_disk(str(R/"probe_encoded"))
    save("config.json",CFG | {"tokenized_path":str(R/"encoded")})
    save("preflight.json",dict(rows=len(dataset),max_tokens=lengths[longest],zero_truncation=True,
        exact_loss_masks=True,uid_order_sha256=order.hexdigest(),source_sha256=meta["sha256"],**counts))


class Verify(TrainerCallback):
    def __init__(self, total):
        self.total=total
    def on_train_begin(self,args,state,control,**kwargs):
        assert state.max_steps==self.total,(state.max_steps,self.total)
        print("TRAINING_VERIFIED_STEPS",state.max_steps,flush=True)
    def on_log(self,args,state,control,logs=None,**kwargs):
        if logs and "loss" in logs:
            assert math.isfinite(logs["loss"])
            if "grad_norm" in logs:
                assert math.isfinite(logs["grad_norm"])


def smoke():
    import torch
    c=json.loads((R/"config.json").read_text())
    assert not OUTPUT.exists()
    c.update(tokenized_path=str(R/"probe_encoded"),output_dir=str(R/"probe"),max_steps=1,
             save_strategy="no",logging_steps=1,plot_loss=False)
    CustomSeq2SeqTrainer.save_model=lambda *a,**k: None
    run_exp(c,callbacks=[Verify(1)])
    torch.cuda.synchronize()
    save("smoke.json",dict(steps=1, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


def train():
    assert (R/"smoke.json").is_file() and not OUTPUT.exists()
    run_exp(json.loads((R/"config.json").read_text()),callbacks=[Verify(9339)])
    for step in (9000,9339):
        assert json.loads((OUTPUT/f"checkpoint-{step}/trainer_state.json").read_text())["global_step"]==step
    save("completion.json",{"steps":9339,"epoch":3,"output":str(OUTPUT)})


if __name__=="__main__":
    {"prepare":prepare,"smoke":smoke,"train":train}[sys.argv[1]]()
