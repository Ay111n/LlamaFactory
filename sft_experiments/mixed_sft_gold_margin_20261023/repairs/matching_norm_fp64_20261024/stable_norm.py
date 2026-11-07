"""Accurate norm reduction of the unchanged full FP32 gradient vectors."""
import math
import torch

def grad_norm(grads):
    squares=[]
    for grad in grads:
        if grad is not None:
            value=float(torch.linalg.vector_norm(grad.detach(),dtype=torch.float64))
            squares.append(value*value)
    result=math.sqrt(math.fsum(squares))
    if not math.isfinite(result):
        raise FloatingPointError('Non-finite full gradient norm')
    return result

def match_scale(candidate,reference):
    n1,n2=grad_norm(candidate),grad_norm(reference)
    if n1==0 and n2!=0:
        raise FloatingPointError(f'Zero candidate norm with nonzero reference: {n2}')
    return (n2/n1 if n1 else 1.),n1,n2

def install():
    from llamafactory.train.sft import termination_trainer,gold_termination_trainer
    termination_trainer.grad_norm=grad_norm
    termination_trainer.match_scale=match_scale
    gold_termination_trainer.grad_norm=grad_norm
