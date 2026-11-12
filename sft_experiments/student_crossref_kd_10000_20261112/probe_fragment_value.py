"""Read-only fragment supervision probe; existing train responses, no generation/training."""
import argparse,hashlib,json,os,sys,time,subprocess,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import full
from full import e
OUT=HERE/"runtime/diagnostics/fragment_value_20261112"
PROMPT=e.ROOT/"benchmark/eval/analysis/teacher_probe_prompt_v2_20261110/teacher_prompt.txt"

def prepare():
 from transformers import AutoTokenizer
 frozen=e.read(OUT/"fixed_batch_protocol.json");uids=[r["uid"] for r in frozen["samples"]]
 assert e.sha(e.CACHE/"examples.jsonl")==frozen["examples_sha256"]
 examples={r["meta"]["uid"]:r for r in e.rows(e.CACHE/"examples.jsonl") if r["meta"]["language"]=="bo"}
 pairs={r["uid"]:r for r in e.rows(e.DATA/"train_pairs.jsonl") if r["uid"] in uids}
 tok=AutoTokenizer.from_pretrained(e.TEACHER,local_files_only=True)
 cached={r["uid"]:r for r in e.rows(full.FULLCACHE/"teacher_examples.jsonl") if r["uid"] in uids}
 wrapper=PROMPT.read_text().strip();selected=[]
 for uid in uids:
  r=examples[uid];pair=pairs[uid];c=cached[uid]
  assert r["meta"]["target"]==c["target"]
  body=tok(pair["tibetan"]["answer"],add_special_tokens=False,return_offsets_mapping=True)
  assert body["input_ids"]+tok.encode("<|im_end|>\n",add_special_tokens=False)==c["target"]
  empty=wrapper.replace("{english.question}","").replace("{english.answer}","").replace("{tibetan.question}",pair["tibetan"]["question"])
  prompt=tok.apply_chat_template([{"role":"system","content":"Respond in Tibetan."},{"role":"user","content":empty}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
  # Natural no-reference control is exactly the student's archived ordinary prompt.
  selected.append({"uid":uid,"pair":pair,"student":r,"teacher":c,"empty_reference_prompt":prompt,"body_tokens":len(body["input_ids"]),"offsets":body["offset_mapping"]})
 e.save(OUT/"selected.json",selected)
 protocol={"samples":32,"window_tokens":32,"primary_threshold":0.1,"sensitivity_thresholds":[0.,.05,.1,.2],"selection":"Reused frozen 32-view gradient diagnostic; no outcome-driven selection","conditions":["teacher_cached_reference","teacher_same_wrapper_empty_reference","teacher_natural_ordinary_prompt","ordinary_bilingual_SFT"],"gate":"Mean log q_plus(y)-log q_empty(y)>threshold AND mean log q_plus(y)-log p_B(y)>threshold; strict >","robust_gate":"Primary gate AND gain over natural no-reference teacher >threshold","body_only":True,"tail_windows":"Include final partial window; report count and token coverage separately","scope":"Teacher-forced training-response diagnostic; no answer extraction, generation, DEV/TEST use, optimizer or backward","sources":{"fixed_batch_protocol":e.sha(OUT/"fixed_batch_protocol.json"),"examples":e.sha(e.CACHE/"examples.jsonl"),"pairs":e.sha(e.DATA/"train_pairs.jsonl"),"teacher_cache_manifest":e.sha(e.CACHE/"teacher_cache_manifest.json"),"prompt":e.sha(PROMPT)},"script_sha256":e.sha(Path(__file__))}
 e.save(OUT/"protocol.json",protocol)
 print("prepared",len(selected),flush=True)

def run(arm):
 import torch,numpy as np,transformers,peft
 from transformers import AutoModelForCausalLM,AutoTokenizer
 assert sys.executable=="/opt/anaconda3/envs/inference/bin/python"
 torch.set_num_threads(4)
 selected=e.read(OUT/"selected.json");cache=e.read(e.CACHE/"teacher_cache_manifest.json")
 tok=AutoTokenizer.from_pretrained(e.TEACHER,local_files_only=True)
 modelpath=e.BASE if arm=="baseline" else e.TEACHER
 model=AutoModelForCausalLM.from_pretrained(modelpath,dtype=torch.bfloat16,attn_implementation="sdpa",device_map={"":"cuda:0"},local_files_only=True)
 adapter=None
 if arm=="baseline":
  adapter=e.MODELS/"baseline"/"adapter_model.safetensors";adapter_hash=e.sha(adapter)
  model=peft.PeftModel.from_pretrained(model,str(e.MODELS/"baseline"),is_trainable=False)
 model.eval().requires_grad_(False)
 body=model.get_base_model() if arm=="baseline" else model
 versions={n:p._version for n,p in model.named_parameters()}
 def score_hidden(h,target):
  values=[];ids=[];probs=[]
  for start in range(0,len(target),64):
   logp=torch.log_softmax(torch.nn.functional.linear(h[start:start+64],body.lm_head.weight.to(h.dtype)).float(),-1)
   t=torch.tensor(target[start:start+64],device="cuda")
   values.append(logp.gather(-1,t[:,None]).squeeze(-1).cpu().numpy())
   v,j=logp.topk(3,dim=-1);ids.append(j.cpu().numpy());probs.append(v.exp().cpu().numpy())
  return {"logp":np.concatenate(values),"top_ids":np.concatenate(ids),"top_probs":np.concatenate(probs)}
 def score_input(ids,start,target):
  with torch.autocast("cuda",dtype=torch.bfloat16):
   h=body.model(input_ids=torch.tensor([ids],device="cuda"),use_cache=False).last_hidden_state[0,start:start+len(target)]
   return score_hidden(h,target)
 (OUT/"tokens").mkdir(exist_ok=True)
 begin=time.time();validation={}
 with torch.inference_mode():
  if arm=="teacher":
   head=torch.load(e.CACHE/"teacher_head.pt",map_location="cpu",weights_only=True)
   assert torch.equal(body.lm_head.weight.detach().cpu(),head);del head
  for i,r in enumerate(selected):
   uid=r["uid"];s=r["student"];m=s["meta"];target=m["target"];output={}
   if arm=="baseline":output={"baseline_"+k:v for k,v in score_input(s["input_ids"],m["prompt_length"]-1,target).items()}
   else:
    entry=cache["files"][uid];assert e.sha(Path(entry["file"]))==entry["sha256"]
    h=torch.load(entry["file"],weights_only=True,map_location="cpu").cuda()
    plus=score_hidden(h,target);del h
    output.update({"plus_"+k:v for k,v in plus.items()})
    if i==0:
     c=r["teacher"];live=score_input(c["teacher_input_ids"],c["teacher_prompt_length"]-1,target)
     error=np.abs(live["logp"]-plus["logp"])
     validation={"cached_vs_live_mean_abs_logp_error":float(error.mean()),"cached_vs_live_max_abs_logp_error":float(error.max())}
     assert error.mean()<.01 and error.max()<.1,validation
    for name,prompt in [("empty",r["empty_reference_prompt"]),("natural",s["input_ids"][:m["prompt_length"]])]:
     output.update({name+"_"+k:v for k,v in score_input(prompt+target[:-1],len(prompt)-1,target).items()})
   assert all(np.isfinite(v).all() for v in output.values())
   np.savez_compressed(OUT/"tokens"/(uid+"_"+arm+".npz"),**output)
   print(json.dumps({"arm":arm,"completed":i+1,"seconds":time.time()-begin}),flush=True)
 assert all(p._version==versions[n] for n,p in model.named_parameters())
 if adapter:assert e.sha(adapter)==adapter_hash
 e.save(OUT/(arm+"_complete.json"),{"status":"completed","samples":32,"seconds":time.time()-begin,"all_parameters_frozen":True,"parameter_versions_unchanged":True,"adapter_sha256":adapter_hash if adapter else None,"cache_validation":validation,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"versions":{"python":sys.executable,"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__},"gpu":os.environ.get("CUDA_VISIBLE_DEVICES")})

def analyze():
 import numpy as np
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(e.TEACHER,local_files_only=True)
 allwindows=[];sample=[]
 for r in e.read(OUT/"selected.json"):
  uid=r["uid"];n=r["body_tokens"];y=np.array(r["student"]["meta"]["target"])
  b=np.load(OUT/"tokens"/(uid+"_baseline.npz"));q=np.load(OUT/"tokens"/(uid+"_teacher.npz"))
  R=q["plus_logp"]-q["empty_logp"];N=q["plus_logp"]-q["natural_logp"];G=q["plus_logp"]-b["baseline_logp"]
  sample.append({"uid":uid,"body_tokens":n,"R":float(R[:n].mean()),"G":float(G[:n].mean()),"R_natural":float(N[:n].mean()),"baseline_ce_body":float(-b["baseline_logp"][:n].mean())})
  for start in range(0,n,32):
   end=min(start+32,n);offsets=r["offsets"][start:end];lo=min(x[0] for x in offsets);hi=max(x[1] for x in offsets);answer=r["pair"]["tibetan"]["answer"]
   window={"id":uid+":"+str(start),"uid":uid,"start":start,"end":end,"tokens":end-start,"R":float(R[start:end].mean()),"G":float(G[start:end].mean()),"R_natural":float(N[start:end].mean()),"text":answer[lo:hi],"context":answer[max(0,lo-120):min(len(answer),hi+120)],"plus_top1_gold_fraction":float((q["plus_top_ids"][start:end,0]==y[start:end]).mean()),"baseline_top1_gold_fraction":float((b["baseline_top_ids"][start:end,0]==y[start:end]).mean())}
   top=sorted(range(start,end),key=lambda t:float(min(R[t],G[t])),reverse=True)[:4]
   window["high_gain_positions"]=[{"t":t,"gold":tok.decode([int(y[t])]),"prefix":tok.decode(y[max(0,t-8):t]),"R":float(R[t]),"G":float(G[t]),"teacher_top3":[tok.decode([int(j)]) for j in q["plus_top_ids"][t]],"baseline_top3":[tok.decode([int(j)]) for j in b["baseline_top_ids"][t]],"teacher_top3_probs":q["plus_top_probs"][t].tolist(),"baseline_top3_probs":b["baseline_top_probs"][t].tolist()} for t in top]
   allwindows.append(window)
 def stats(ws,t):
  selected=[w for w in ws if w["R"]>t and w["G"]>t];robust=[w for w in selected if w["R_natural"]>t]
  return {"windows":len(ws),"tokens":sum(w["tokens"] for w in ws),"reference_gain_windows":sum(w["R"]>t for w in ws),"student_gap_windows":sum(w["G"]>t for w in ws),"selected_windows":len(selected),"selected_tokens":sum(w["tokens"] for w in selected),"selected_samples":len({w["uid"] for w in selected}),"robust_windows":len(robust),"robust_tokens":sum(w["tokens"] for w in robust)}
 summary={str(t):stats(allwindows,t) for t in [0,.05,.1,.2]}
 selected=[w for w in allwindows if w["R"]>.1 and w["G"]>.1]
 selected.sort(key=lambda w:hashlib.sha256(("review-3407:"+w["id"]).encode()).hexdigest())
 rejected=[w for w in allwindows if not(w["R"]>.1 and w["G"]>.1)]
 rejected.sort(key=lambda w:hashlib.sha256(("review-3407:"+w["id"]).encode()).hexdigest())
 review=[{**w,"review_group":"selected_hash_sample"} for w in selected[:24]]+[{**w,"review_group":"not_selected_hash_sample"} for w in rejected[:12]]
 top=sorted(selected,key=lambda w:min(w["R"],w["G"]),reverse=True)[:6]
 for name,data in [("windows",allwindows),("review",review),("top6",top),("samples",sample)]:
  (OUT/(name+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in data),encoding="utf-8")
 e.save(OUT/"summary.json",{"thresholds":summary,"by_source_at_0.1":{group:stats([w for w in allwindows if w["uid"].startswith(group)],.1) for group in ["Knowledge","Math"]},"review_selection":"24 selected + 12 nonselected windows by fixed hash, plus separate top6 illustrations; semantic review is exploratory, not human gold labels"})
 print(json.dumps(summary),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","baseline","teacher","analyze","both"]);a=p.parse_args()
 if a.mode=="prepare":prepare()
 elif a.mode=="analyze":analyze()
 elif a.mode=="both":
  for mode in ["baseline","teacher","analyze"]:subprocess.run([sys.executable,str(Path(__file__).resolve()),mode],check=True)
 else:run(a.mode)
