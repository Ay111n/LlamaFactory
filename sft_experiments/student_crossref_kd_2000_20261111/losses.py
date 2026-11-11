"""Exact full-vocabulary CE and forward KL with chunked frozen output heads."""
import torch
import torch.nn.functional as F

class HiddenLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, teacher_hidden, teacher_weight, chunk):
        assert not weight.requires_grad
        enabled = teacher_hidden.numel() > 0
        assert not teacher_hidden.requires_grad and not teacher_weight.requires_grad
        ctx.save_for_backward(hidden, weight, target, teacher_hidden, teacher_weight)
        ctx.chunk, ctx.enabled = chunk, enabled
        ce = hidden.new_zeros((), dtype=torch.float32)
        kl = ce.clone()
        for start in range(0, len(target), chunk):
            end = min(start + chunk, len(target))
            logp = F.log_softmax(F.linear(hidden[start:end], weight).float(), -1)
            ce -= logp.gather(1, target[start:end,None]).sum()
            if enabled:
                logq = F.log_softmax(F.linear(teacher_hidden[start:end], teacher_weight).float(), -1)
                kl += (logq.exp() * (logq-logp)).sum()
        return torch.stack((ce, kl)) / len(target)

    @staticmethod
    def backward(ctx, grad):
        hidden, weight, target, th, tw = ctx.saved_tensors
        result = torch.empty_like(hidden)
        n = len(target)
        for start in range(0, n, ctx.chunk):
            end = min(start+ctx.chunk,n)
            prob = F.softmax(F.linear(hidden[start:end], weight).float(),-1)
            d = prob.clone()
            d.scatter_add_(1,target[start:end,None], -torch.ones((end-start,1),device=d.device,dtype=d.dtype))
            d *= grad[0] / n
            if ctx.enabled:
                q = F.softmax(F.linear(th[start:end], tw).float(),-1)
                d += (prob*q.sum(-1,keepdim=True)-q)*(grad[1]/n)
            result[start:end] = d.to(hidden.dtype).matmul(weight)
        return result,None,None,None,None,None


def response_losses(hidden, weight, target, teacher_hidden=None, teacher_weight=None, chunk=64):
    if teacher_hidden is None:
        teacher_hidden = hidden.new_empty(0)
        teacher_weight = hidden.new_empty(0)
    return HiddenLoss.apply(hidden,weight,target,teacher_hidden,teacher_weight,chunk)
