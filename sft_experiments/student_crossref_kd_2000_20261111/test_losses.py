import json
import torch
import torch.nn.functional as F
from losses import response_losses

torch.manual_seed(3407)
checks=[]
for n in [1,7,65]:
 for use_teacher in [False,True]:
  h=torch.randn(n,13,requires_grad=True)
  w=torch.randn(31,13)
  th=torch.randn_like(h).detach();tw=torch.randn_like(w)
  g=torch.randint(31,(n,))
  ls=F.log_softmax(h@w.T,-1)
  ce=F.nll_loss(ls,g)
  q=F.softmax(th@tw.T,-1)
  kl=(q*(q.log()-ls)).sum(-1).mean() if use_teacher else ce*0
  ref=ce+.2*kl
  grad=torch.autograd.grad(ref,h)[0]
  got=response_losses(h,w,g,th if use_teacher else None,tw if use_teacher else None,chunk=5)
  gotgrad=torch.autograd.grad(got[0]+.2*got[1],h)[0]
  assert torch.allclose(got,torch.stack((ce,kl)),atol=3e-6,rtol=2e-6)
  assert torch.allclose(grad,gotgrad,atol=2e-6,rtol=2e-5)
  checks.append({'length':n,'teacher':use_teacher,'loss_error':float(abs(got[0]-ce)),'gradient_max_error':float((grad-gotgrad).abs().max())})
print(json.dumps({'status':'passed','checks':checks},indent=2))
