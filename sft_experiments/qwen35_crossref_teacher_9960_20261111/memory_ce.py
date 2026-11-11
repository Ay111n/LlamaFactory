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
    loss=CrossEntropyLoss(ignore_index=ignore_index,reduction=reduction,inplace_backward=False)(flat,targets)
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):num_items_in_batch=num_items_in_batch.to(loss.device)
        loss=loss/num_items_in_batch
    return loss

def install():
    from transformers.loss.loss_utils import LOSS_MAPPING
    LOSS_MAPPING['ForCausalLM']=memory_ce
    LOSS_MAPPING['ForConditionalGeneration']=memory_ce
    print('Response CE: FlashAttention Triton kernel; unchanged shift/mask/reduction.',flush=True)
