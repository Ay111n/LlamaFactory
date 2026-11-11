"""Check production-vocabulary CE values and gradients against native Transformers."""
import json,torch
from pathlib import Path
from transformers.loss.loss_utils import ForCausalLMLoss
from memory_ce import memory_ce
results=[]
for dtype in [torch.float32,torch.bfloat16]:
 for supplied in [False,True]:
  torch.manual_seed(3407)
  x=torch.randn(2,17,248320,device='cuda',dtype=dtype).requires_grad_()
  y=x.detach().clone().requires_grad_()
  labels=torch.randint(0,248320,(2,17),device='cuda');labels[:,:7]=-100
  kwargs=dict(num_items_in_batch=torch.tensor(31,device='cuda')) if supplied else {}
  a=ForCausalLMLoss(x,labels,248320,**kwargs);b=memory_ce(y,labels,248320,**kwargs)
  a.backward();b.backward()
  torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
  torch.testing.assert_close(x.grad,y.grad,rtol=.02 if dtype==torch.bfloat16 else 2e-5,atol=1e-6 if dtype==torch.bfloat16 else 1e-8)
  results.append(dict(dtype=str(dtype),supplied_denominator=supplied,native_loss=a.item(),kernel_loss=b.item(),
                      loss_abs_error=abs(a.item()-b.item()),gradient_max_abs_error=(x.grad-y.grad).abs().max().item()))
  del x,y,a,b
p=Path(__file__).parent/'runtime/memory_ce_equivalence.json'
p.write_text(json.dumps(dict(status='passed',vocabulary=248320,cases=results),indent=2)+'\n')
print(p.read_text())
