"""Bounded isolated metadata-cache feasibility check, never installed in production."""
import json,time,os,statistics
from pathlib import Path
import torch
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.utils.index import prepare_chunk_indices
R=Path(__file__).resolve().parent/'runtime/speed_diagnosis'

def main():
 torch.manual_seed(3407)
 pos=torch.arange(3038,device='cuda')[None,:]
 # Same original helper result, retained for this immutable position tensor only.
 cached=prepare_fa_kwargs_from_position_ids(pos)
 fresh=prepare_fa_kwargs_from_position_ids(pos)
 assert torch.equal(cached[0][0],fresh[0][0]) and cached[1]==fresh[1]
 q,k,v=[torch.randn(1,3038,32,128,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
 g=torch.full((1,3038,32),-.1,device='cuda',dtype=torch.float32,requires_grad=True)
 beta=torch.full((1,3038,32),.5,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 params=[q,k,v,g,beta]
 def run(reuse):
  for x in params:x.grad=None
  cu=(cached if reuse else prepare_fa_kwargs_from_position_ids(pos))[0][0]
  out,_=chunk_gated_delta_rule(q,k,v,g=g,beta=beta,cu_seqlens=cu,use_qk_l2norm_in_kernel=True)
  out.float().square().mean().backward()
  return out
 a=run(False).detach().clone();ag=[x.grad.clone() for x in params]
 b=run(True).detach().clone();bg=[x.grad.clone() for x in params]
 equality=[torch.equal(x,y) for x,y in zip([a]+ag,[b]+bg)]
 torch.testing.assert_close(a,b,rtol=0,atol=0)
 for x,y in zip(ag,bg):torch.testing.assert_close(x,y,rtol=0,atol=0)
 for _ in range(2):run(False);run(True)
 timings=[]
 for reuse in [False,True,True,False,True,False,False,True]:
  torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(8):run(reuse)
  torch.cuda.synchronize();timings.append(dict(cached=reuse,seconds_per_call=(time.perf_counter()-start)/8))
 # Metadata-only repeated across the 24 linear-attention layers and recomputation.
 mt=[]
 for reuse in [False,True,True,False]:
  torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(480):
   cu=(cached if reuse else prepare_fa_kwargs_from_position_ids(pos))[0][0]
   prepare_chunk_indices(cu,64)
  torch.cuda.synchronize();mt.append(dict(cached=reuse,seconds_per_call=(time.perf_counter()-start)/480))
 med={str(x):statistics.median(t['seconds_per_call'] for t in timings if t['cached']==x) for x in [False,True]}
 out=dict(status='passed',exact_output_and_gradient_equality=equality,shape=[1,3038,32,128],timings=timings,median_seconds=med,speedup=med['False']/med['True'],metadata_only_timings=mt,scope='isolated delta-rule block, not full-model training',production_applied=False)
 (R/'metadata_cache_check.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
