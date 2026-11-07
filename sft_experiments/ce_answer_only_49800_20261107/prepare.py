"""Verify paired data and tokenize with the native LlamaFactory SFT processor."""
import csv
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import numpy as np
import yaml
from datasets import Dataset, DatasetDict
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.data.processor.processor_utils import infer_seqlen
from llamafactory.hparams import ModelArguments, DataArguments
from llamafactory.model import load_tokenizer
from common import *

def describe(values):
    v = np.asarray(values, dtype=np.int64)
    return {"total":int(v.sum()),"min":int(v.min()),"median":float(np.median(v)),
            "p95":float(np.quantile(v,.95)),"max":int(v.max()),"mean":float(v.mean())}

def main():
    assert sys.executable == PYTHON
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if (ARTIFACTS/"prepared.json").exists():
        record = read(ARTIFACTS/"prepared.json")
        for name, expected in record["cache_sha256"].items():
            assert sha(PROCESSED/"dataset"/name) == expected
        print("Verified existing prepared dataset.", flush=True)
        return
    manifest = read(DATA/"manifest.json")
    for name, info in manifest["outputs"].items():
        assert sha(DATA/name) == info["sha256"], name
    for relative, expected in manifest["sources"].items():
        assert sha(THESIS/relative) == expected
    tokenizer_module = load_tokenizer(ModelArguments(model_name_or_path=BASE))
    tokenizer = tokenizer_module["tokenizer"]
    da = DataArguments(template="qwen3", dataset=NAME, cutoff_len=4096, enable_thinking=True,
                       packing=False, train_on_prompt=False, preprocessing_num_workers=16)
    template = get_template_and_fix_tokenizer(tokenizer, da)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"), data_args=da)
    metrics = defaultdict(list)
    flags = defaultdict(int)
    ids = []
    empty = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    output_csv = PROCESSED/"paired_lengths.csv"
    def generate():
        with (SOURCE/"mixed_sft_final_49800_sharegpt.jsonl").open(encoding="utf-8") as old, \
             (DATA/"answer_only_49800_sharegpt.jsonl").open(encoding="utf-8") as new, \
             (DATA/"answer_only_49800.jsonl").open(encoding="utf-8") as meta, \
             output_csv.open("w",encoding="utf-8",newline="") as stats:
            writer = csv.writer(stats)
            writer.writerow(["uid","source_think_type","old_input","new_input","old_supervised","new_supervised",
                             "old_prompt_cut","new_prompt_cut","old_target_cut","new_target_cut"])
            for index, (lold,lnew,lmeta) in enumerate(zip(old,new,meta,strict=True)):
                before, after, original = map(json.loads,(lold,lnew,lmeta))
                ids.append(after["uid"])
                assert after["uid"] == original["uid"]
                assert before["messages"][0] == after["messages"][0]
                assert after["messages"][-1]["content"] == "<think>\n\n</think>\n\n"+original["messages"][-1]["content"]
                results, cuts = [], []
                for label, record in (("old",before),("new",after)):
                    messages = record["messages"]
                    source, target = template.encode_multiturn(tokenizer,messages)[0]
                    if label == "new":
                        assert target[:len(empty)] == empty, index
                    source_n,target_n = infer_seqlen(len(source),len(target),4096-int(template.efficient_eos))
                    encoded, labels = processor._encode_data_example(
                        messages[:-1], messages[-1:], "", "", [], [], [])
                    assert len(encoded) == len(labels) <= 4096
                    assert all(t == -100 for t in labels[:source_n])
                    assert labels[source_n:source_n+target_n] == target[:target_n]
                    assert target_n >= len(empty) if label == "new" else True
                    metrics[label+"_input"].append(len(encoded))
                    metrics[label+"_supervised"].append(sum(t!=-100 for t in labels))
                    flags[label+"_prompt_truncated"] += int(source_n<len(source))
                    flags[label+"_target_truncated"] += int(target_n<len(target))
                    results.append((len(encoded),sum(t!=-100 for t in labels),encoded,labels,source))
                    cuts.extend([int(source_n<len(source)),int(target_n<len(target))])
                assert results[0][4] == results[1][4], index
                if original["think_type"] == "no_think":
                    assert results[0][2:4] == results[1][2:4], index
                writer.writerow([after["uid"],original["think_type"],
                    results[0][0],results[1][0],results[0][1],results[1][1],
                    cuts[0],cuts[2],cuts[1],cuts[3]])
                yield {"input_ids":results[1][2],"attention_mask":[1]*len(results[1][2]),"labels":results[1][3]}
                if (index+1)%2500 == 0:
                    print(f"Native paired tokenization {index+1}/49800",flush=True)
    cache = PROCESSED/("build_cache_"+str(os.getpid()))
    ds = Dataset.from_generator(generate, cache_dir=str(cache))
    assert len(ds) == len(ids) == len(set(ids)) == 49800
    assert __import__("hashlib").sha256(("\n".join(ids)+"\n").encode()).hexdigest()==manifest["id_order_sha256"]
    target = PROCESSED/"dataset"
    if target.exists():
        raise FileExistsError(target)
    DatasetDict({"train":ds}).save_to_disk(str(target))
    registered = HERE/"data"
    registered.mkdir(exist_ok=True)
    write(registered/"dataset_info.json", {NAME:{"file_name":str(DATA/"answer_only_49800_sharegpt.jsonl"),
          "formatting":"sharegpt","columns":{"messages":"messages"},
          "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}})
    cfg = yaml.safe_load((FRAMEWORK/"examples/train_lora/qwen3_mixed_sft_49800_lora_v1.yaml").read_text())
    cfg.update(dataset=NAME,dataset_dir=str(registered),tokenized_path=str(target),
        output_dir=str(MODELS/"main"),enable_thinking=True,packing=False,train_on_prompt=False,
        seed=3407,data_seed=3407,report_to="none",remove_unused_columns=True,
        resume_from_checkpoint=None)
    assert cfg["model_name_or_path"]==BASE
    assert cfg["num_train_epochs"]==3 and cfg["gradient_accumulation_steps"]==16
    for phase in ("main","preflight"):
        phase_cfg=dict(cfg,output_dir=str(MODELS/phase))
        if phase=="preflight":
            phase_cfg.update(max_steps=100,save_steps=100,save_total_limit=2)
        (ARTIFACTS/(phase+".yaml")).write_text(yaml.safe_dump(phase_cfg,sort_keys=False),encoding="utf-8")
    cache_hashes = {p.relative_to(target).as_posix():sha(p) for p in target.rglob("*") if p.is_file()}
    result={"passed":True,"created_at":now(),"rows":49800,"id_order_sha256":manifest["id_order_sha256"],
        "empty_think_tags_supervised":True,"prompt_tokens_equal_before_truncation":True,
        "no_think_encodings_unchanged":True,"statistics":{k:describe(v) for k,v in metrics.items()},
        "truncation_counts":dict(flags),"paired_lengths_sha256":sha(output_csv),
        "cache_sha256":cache_hashes,"tokenizer_sha256":sha(Path(BASE)/"tokenizer.json"),
        "template":type(template).__name__,"enable_thinking":template.enable_thinking,
        "versions":{p:importlib.metadata.version(p) for p in ("transformers","datasets","tokenizers","torch")}}
    write(ARTIFACTS/"prepared.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()
