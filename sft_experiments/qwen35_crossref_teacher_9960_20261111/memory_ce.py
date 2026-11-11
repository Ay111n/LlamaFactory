"""Same causal response CE using the existing FlashAttention Triton CE kernel."""
import torch
from torch.nn import functional as F
from flash_attn.losses.cross_entropy import CrossEntropyLoss

def memory_ce(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100, shift_labels=None, **kwargs):
    if shift_labels is None:
        shift_labels=F.pad(labels,(0,1),value=ignore_index)[...,1:].contiguous()
    targets=shift_labels.reshape(-1).to(logits.device)
    flat=logits.reshape(-1,vocab_size)
    reduction='sum' if num_items_in_batch is not None else 'mean'
    loss=CrossEntropyLoss(ignore_index=ignore_index,reduction=reduction,inplace_backward=True)(flat,targets)
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):num_items_in_batch=num_items_in_batch.to(loss.device)
        loss=loss/num_items_in_batch
    return loss

def warmup_linear_attention():
    # Compile and autotune backward kernels before model weights/logits occupy VRAM.
    # Preserve RNG state; these synthetic tensors never update model parameters.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import json, time
    from pathlib import Path
    started=time.time()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(3407)
        shape=(1,14472,32,128)
        q=torch.randn(shape,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        v=torch.randn_like(q,requires_grad=True)
        g=torch.full(shape[:-1],-.1,device='cuda',dtype=torch.float32,requires_grad=True)
        beta=torch.full(shape[:-1],.5,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        cu=torch.tensor([0,shape[1]],device='cuda',dtype=torch.int32)
        out,_=chunk_gated_delta_rule(q,k,v,g=g,beta=beta,use_qk_l2norm_in_kernel=True,cu_seqlens=cu)
        out.float().square().mean().backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all().item() for x in [q,k,v,g,beta])
        torch.cuda.synchronize()
        del q,k,v,g,beta,cu,out
    torch.cuda.empty_cache()
    report=dict(status='passed',shape=list(shape),varlen=True,elapsed_seconds=time.time()-started,rng_preserved=True)
    (Path(__file__).parent/'runtime/fla_warmup.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FLA forward/backward warmup passed before model loading.',flush=True)


def install_loss_only_outputs():
    # The SFT trainer consumes only loss; prevent Accelerate from casting unused
    # full-vocabulary logits to FP32. Loss keeps the original autograd graph.
    from functools import wraps
    from transformers import Qwen3_5ForConditionalGeneration
    cls=Qwen3_5ForConditionalGeneration
    original=cls.forward
    if getattr(original, '_loss_only_training_outputs', False):
        return
    @wraps(original)
    def forward(self, *args, **kwargs):
        outputs=original(self, *args, **kwargs)
        if self.training and kwargs.get('labels') is not None and getattr(outputs, 'loss', None) is not None:
            return type(outputs)(**{k:v for k,v in outputs.items() if k != 'logits'})
        return outputs
    forward._loss_only_training_outputs=True
    cls.forward=forward


def install(warmup=True):
    from transformers.loss.loss_utils import LOSS_MAPPING
    LOSS_MAPPING['ForCausalLM']=memory_ce
    LOSS_MAPPING['ForConditionalGeneration']=memory_ce
    install_loss_only_outputs()
    print('Response CE: FlashAttention Triton kernel; unchanged shift/mask/reduction.',flush=True)
    if warmup:
        warmup_linear_attention()
