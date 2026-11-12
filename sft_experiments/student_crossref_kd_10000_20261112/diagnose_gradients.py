"""Fixed training-batch diagnostics. No optimizer, Trainer, update or model save."""
import argparse, gc, hashlib, json, math, os, statistics, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"src"))
import full
from full import e
OUT=HERE/"runtime/diagnostics/fixed_batch_gradients_20261112"

def prepare():
 e.verify()
 rows=[r for r in e.rows(e.CACHE/"examples.jsonl") if r["meta"]["language"]=="bo"]
 rows.sort(key=lambda r:(len(r["input_ids"]),r["meta"]["uid"]))
 chosen=[]
 for b in range(4):
  bucket=rows[b*2500:(b+1)*2500]
  bucket.sort(key=lambda r:hashlib.sha256(("fixed-grad-3407:"+r["meta"]["uid"]).encode()).hexdigest())
  chosen.extend(bucket[:8])
 meta=e.read(e.CACHE/"teacher_cache_manifest.json")
 protocol={"selection":"32 Tibetan training views; 8 deterministic hash-selected from each input-length quartile, before model diagnostics", "seed_label":"fixed-grad-3407", "examples_sha256":e.sha(e.CACHE/"examples.jsonl"),"teacher_manifest_sha256":e.sha(e.CACHE/"teacher_cache_manifest.json"),"teacher_head_sha256":e.sha(e.CACHE/"teacher_head.pt"),"script_sha256":e.sha(Path(__file__)),"weight":0.2,"temperature":1,"aggregation":"Mean response-token loss per sample; batch gradient is arithmetic mean over 32 responses", "gradient_space":"All trainable LoRA parameters, FP32; no clipping or optimizer preconditioning", "dropout":"Disabled for deterministic endpoint diagnosis; differs from stochastic training dropout", "teacher_forcing":"Identical archived target prefix and OFF protocol; EOS/newline included", "optimizer_created":False,"parameter_updates":0,"samples":[{"uid":r["meta"]["uid"],"input_tokens":len(r["input_ids"]),"response_tokens":len(r["meta"]["target"]),"teacher_sha256":meta["files"][r["meta"]["uid"]]["sha256"]} for r in chosen]}
 OUT.mkdir(parents=True,exist_ok=True)
 path=OUT/"protocol.json"
 if path.exists():assert e.read(path)==protocol
 else:e.save(path,protocol)
 return chosen,meta,protocol

def metrics(a,b):
 import torch
 aa=float(torch.dot(a,a));bb=float(torch.dot(b,b));ab=float(torch.dot(a,b))
 assert all(math.isfinite(x) for x in [aa,bb,ab]) and aa>0 and bb>0
 return {"ce_grad_norm":math.sqrt(aa),"kl_grad_norm":math.sqrt(bb),"weighted_kl_grad_norm":.2*math.sqrt(bb),"weighted_kl_to_ce_norm_ratio":.2*math.sqrt(bb/aa),"cosine":ab/math.sqrt(aa*bb),"ce_dot_kl":ab,"angle_degrees":math.degrees(math.acos(max(-1,min(1,ab/math.sqrt(aa*bb))))),"combined_grad_norm":math.sqrt(max(0,aa+.4*ab+.04*bb)),"weighted_kl_projection_relative_ce":.2*ab/aa}

def run(arm):
 import torch, transformers, peft, flash_attn
 from llamafactory.model import load_model,load_tokenizer
 from llamafactory.data import get_template_and_fix_tokenizer
 from losses import response_losses
 assert sys.executable=="/opt/anaconda3/envs/training/bin/python"
 torch.set_num_threads(4);transformers.set_seed(3407)
 selected,meta,protocol=prepare()
 assert not (OUT/(arm+"_summary.json")).exists(),"Do not overwrite completed diagnostic"
 adapter=e.MODELS/arm/"adapter_model.safetensors"
 disk_before=e.sha(adapter)
 ma,da,ta,fa,ga=e.setup_args(arm)
 ma.adapter_name_or_path=[str(e.MODELS/arm)]
 fa.create_new_adapter=False
 tok=load_tokenizer(ma)["tokenizer"];get_template_and_fix_tokenizer(tok,da)
 model=load_model(tok,ma,fa,is_trainable=True).cuda()
 # Keep checkpointing active but remove stochastic dropout for matched measurements.
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
 model.enable_input_require_grads()
 model.train()
 for module in model.modules():
  if isinstance(module,torch.nn.Dropout):module.eval()
 body=model.get_base_model()
 params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 assert sum(p.numel() for _,p in params)==87293952
 assert all("lora_" in n for n,p in params)
 versions={n:p._version for n,p in model.named_parameters()}
 def digest():
  h=hashlib.sha256()
  for n,p in params:h.update(n.encode());h.update(p.detach().float().cpu().numpy().tobytes())
  return h.hexdigest()
 before=digest()
 def gradients():
  assert all(p.grad is not None for _,p in params)
  v=torch.cat([p.grad.detach().float().cpu().reshape(-1) for _,p in params])
  assert torch.isfinite(v).all()
  return v
 head=body.lm_head.weight.detach()
 assert e.sha(e.CACHE/"teacher_head.pt")==protocol["teacher_head_sha256"]
 tw=torch.load(e.CACHE/"teacher_head.pt",weights_only=True,map_location="cpu").cuda()
 sums=[torch.zeros(sum(p.numel() for _,p in params),dtype=torch.float32) for _ in range(2)]
 output=[];started=time.time()
 for i,r in enumerate(selected):
  m=r["meta"];entry=meta["files"][m["uid"]];assert e.sha(Path(entry["file"]))==entry["sha256"]
  th=torch.load(entry["file"],weights_only=True,map_location="cpu").cuda()
  n=len(r["input_ids"]);pad=(-n)%8
  x=torch.tensor([r["input_ids"]+[tok.pad_token_id]*pad],device="cuda")
  mask=torch.tensor([[1]*n+[0]*pad],device="cuda")
  model.zero_grad(set_to_none=True)
  with torch.autocast("cuda",dtype=torch.bfloat16):
   h=body.model(input_ids=x,attention_mask=mask,use_cache=False).last_hidden_state[0]
   h=h[m["prompt_length"]-1:m["prompt_length"]-1+len(m["target"])]
   assert h.shape==th.shape
   target=torch.tensor(m["target"],device="cuda")
   losses=response_losses(h,head.to(h.dtype),target,th,tw)
  assert torch.isfinite(losses).all()
  ce,kl=[float(v) for v in losses.detach()]
  losses[0].backward(retain_graph=True);gce=gradients()
  model.zero_grad(set_to_none=True)
  losses[1].backward();gkl=gradients()
  row={"index":i,"uid":m["uid"],"ce":ce,"kl":kl,**metrics(gce,gkl),"elapsed_seconds":time.time()-started}
  sums[0].add_(gce/len(selected));sums[1].add_(gkl/len(selected))
  output.append(row)
  with (OUT/(arm+"_samples.jsonl")).open("a") as f:f.write(json.dumps(row)+"\n")
  print(json.dumps({"arm":arm,"completed":i+1,"total":len(selected),"ce":ce,"kl":kl,"cosine":row["cosine"],"seconds":row["elapsed_seconds"]}),flush=True)
  model.zero_grad(set_to_none=True)
  del gce,gkl,h,losses,th,x,mask,target
 assert all(p._version==versions[n] for n,p in model.named_parameters()),"Parameter in-place mutation"
 assert digest()==before and e.sha(adapter)==disk_before,"Parameters changed"
 result={"arm":arm,"status":"completed","samples":len(output),"ce":statistics.mean(r["ce"] for r in output),"kl":statistics.mean(r["kl"] for r in output),"batch_gradient":metrics(*sums),"sample_gradient_summary":{"mean_cosine":statistics.mean(r["cosine"] for r in output),"negative_cosine_count":sum(r["cosine"]<0 for r in output),"median_weighted_ratio":statistics.median(r["weighted_kl_to_ce_norm_ratio"] for r in output)},"parameters_unchanged":True,"parameter_version_counters_unchanged":True,"trainable_sha256":before,"adapter_sha256":disk_before,"trainable_parameters":sum(p.numel() for _,p in params),"seconds":time.time()-started,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"versions":{"python":sys.executable,"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,"flash_attn":flash_attn.__version__},"device":torch.cuda.get_device_name(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"script_sha256":e.sha(Path(__file__))}
 e.save(OUT/(arm+"_summary.json"),result);print(json.dumps(result),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--arm",choices=["prepare","baseline","kd","both"],required=True);a=p.parse_args()
 if a.arm=="prepare":prepare();print("protocol prepared")
 elif a.arm=="both":
  for arm in ["baseline","kd"]:subprocess.run([sys.executable,str(Path(__file__).resolve()),"--arm",arm],check=True)
 else:run(a.arm)
