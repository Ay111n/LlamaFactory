"""Check real tiny Qwen3.5 loss/gradients and inference outputs through FP32 conversion."""
import json,torch
from pathlib import Path
from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
from accelerate.utils.operations import convert_to_fp32
from memory_ce import install
install(warmup=False)
d=json.loads(Path('/home/nlp/data/model/LLM/raw/qwen3.5-9b/config.json').read_text())
t=d['text_config'];t.update(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16,layer_types=['full_attention'],use_cache=False)
t['rope_parameters']=dict(rope_type='default',rope_theta=10000,partial_rotary_factor=1.,mrope_section=[2,3,3],mrope_interleaved=True)
v=d['vision_config'];v.update(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=64,num_position_embeddings=16,patch_size=2,spatial_merge_size=2)
config=Qwen3_5Config.from_dict(d);config._attn_implementation='sdpa'
torch.manual_seed(3407)
model=Qwen3_5ForConditionalGeneration(config).to(device='cuda',dtype=torch.bfloat16).train()
ids=torch.randint(2,250,(1,24),device='cuda');labels=ids.clone();labels[:,:8]=-100
kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False)
original=type(model).forward.__wrapped__
baseline=convert_to_fp32(original(model,**kwargs));assert baseline.logits.dtype==torch.float32
baseline.loss.backward();grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None};value=baseline.loss.detach().clone()
model.zero_grad(set_to_none=True)
updated=convert_to_fp32(model(**kwargs));assert updated.logits is None and 'logits' not in updated
updated.loss.backward();torch.testing.assert_close(value,updated.loss,rtol=0,atol=0)
maxdiff=0.
for n,p in model.named_parameters():
 if n in grads:
  torch.testing.assert_close(grads[n],p.grad,rtol=0,atol=0);maxdiff=max(maxdiff,(grads[n]-p.grad).abs().max().item())
with torch.no_grad():
 assert model(input_ids=ids,use_cache=False).logits is not None
 model.eval();assert model(**kwargs).logits is not None
result=dict(status='passed',native_model_class=type(model).__name__,loss=float(value),loss_abs_error=abs(float(value-updated.loss)),gradient_max_abs_error=maxdiff,training_logits_omitted=True,inference_logits_preserved=True,eval_with_labels_logits_preserved=True)
p=Path(__file__).parent/'runtime/loss_only_output_check.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
