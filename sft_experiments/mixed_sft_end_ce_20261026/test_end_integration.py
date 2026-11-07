"""Analytic logits tests plus real HF whole-window accumulation integration."""
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast,DataCollatorForSeq2Seq,TrainerCallback
from experiment_paths import ROOT
from llamafactory.hparams import TrainingArguments,FinetuningArguments
from llamafactory.train.sft.termination_end_ce import end_boundary_losses,END_MODES as GOLD_MODES
from llamafactory.train.sft.termination_trainer import BoundaryCollator,TerminationTrainer
from llamafactory.train.sft.end_ce_trainer import EndCETrainer
from llamafactory.train.sft.termination_loss import match_scale


def gold_boundary_losses(*args,**kwargs):
 one,two,stats=end_boundary_losses(*args,**kwargs)
 return dict(gold=one,dynamic_gold=two,dynamic_ce=two),stats,{}


class EndIntegrationTests(unittest.TestCase):
 def test_optimizer_window_integration(self):
  torch.manual_seed(3407);torch.set_num_threads(2)
  tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'<pad>':0,'<unk>':1,'</think>':2,'a':3,'b':4,'c':5,'d':6,'e':7},unk_token='<unk>')),pad_token='<pad>',unk_token='<unk>')
  cfg=Qwen3Config(vocab_size=8,hidden_size=8,intermediate_size=12,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=8,attention_dropout=0.,tie_word_embeddings=False)
  net=Qwen3ForCausalLM(cfg);initial=copy.deepcopy(net.state_dict())
  collator=BoundaryCollator(DataCollatorForSeq2Seq(tok,pad_to_multiple_of=8))
  reports=[]
  with tempfile.TemporaryDirectory() as temp:
   for n,nthink in [(16,0),(16,1),(16,8),(16,15),(16,16),(7,0),(7,1),(7,3),(7,7)]:
    records=[]
    for i in range(n):
     flag=i<nthink;ids=[3,4,5,2,6,7]+([4,5] if i%2 else [])
     records.append(dict(input_ids=ids,labels=[-100,-100]+ids[2:],attention_mask=[1]*len(ids),term_positions=([1,2] if i%2 else [2]) if flag else [],term_targets=([0,1] if i%2 else [1]) if flag else []))
    batch=collator(records);clean={k:v for k,v in batch.items() if not k.startswith('term_')}
    net.load_state_dict(initial);params=list(net.parameters());tokens=int(clean['labels'][:,1:].ne(-100).sum())
    output=net(**clean,num_items_in_batch=tokens,use_cache=False)
    explicit=F.cross_entropy(output.logits[:,:-1].reshape(-1,8),clean['labels'][:,1:].reshape(-1),ignore_index=-100,reduction='sum')/tokens
    torch.testing.assert_close(output.loss,explicit)
    losses,_,_=gold_boundary_losses(output.logits,clean['labels'],batch['term_positions'],batch['term_targets'],2,clean['input_ids'])
    losses={k:v/max(nthink,1) for k,v in losses.items()}
    for mode in ['none']+list(GOLD_MODES):
     for coef in ([.0012,0.] if mode!='none' else [0.]):
      candidate='gold' if 'boundary_ce' not in mode else 'dynamic_ce'
      if mode=='dynamic_end_ce':candidate='dynamic_gold'
      scale=1.
      if mode.startswith('matched_'):
       g1=torch.autograd.grad(losses[candidate],params,retain_graph=True,allow_unused=True)
       g2=torch.autograd.grad(losses['dynamic_gold'],params,retain_graph=True,allow_unused=True)
       scale=match_scale(g1,g2)[0]
      target=output.loss+(coef*scale*losses[candidate] if mode!='none' else 0)
      expected=torch.autograd.grad(target,params,retain_graph=True,allow_unused=True)
      actual=[]
      class Capture(TrainerCallback):
       def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):actual.extend(p.grad.detach().clone() if p.grad is not None else None for p in model.parameters())
      model=Qwen3ForCausalLM(cfg);model.load_state_dict(initial)
      args=TrainingArguments(output_dir=f'{temp}/{n}-{nthink}-{mode}-{coef}',per_device_train_batch_size=1,gradient_accumulation_steps=16,max_steps=1,learning_rate=0,report_to='none',use_cpu=True,max_grad_norm=0,save_strategy='no',remove_unused_columns=False,seed=3407,disable_tqdm=True)
      if n==16 and nthink==8 and mode.startswith('matched_') and coef:
       args.max_grad_norm=.1
       norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in expected if g is not None]))
       factor=(.1/(norm+1e-6)).clamp(max=1)
       expected=[g*factor if g is not None else None for g in expected]
      fa=FinetuningArguments(termination_loss_mode=mode,termination_loss_coef=coef,termination_dataset_path='synthetic')
      cls=TerminationTrainer if mode=='none' else EndCETrainer
      trainer=cls(model=model,args=args,finetuning_args=fa,processor=None,tokenizer=tok,data_collator=collator,train_dataset=Dataset.from_list(records),callbacks=[Capture()])
      # Skip scheduled final probe here; probe invariance is exercised separately.
      trainer.gradient_probe=lambda *a,**k:None
      trainer.train()
      errors=[]
      for a,e in zip(actual,expected):
       if e is None:self.assertTrue(a is None or not a.any())
       else:torch.testing.assert_close(a,e,rtol=2e-4,atol=2e-6);errors.append(float((a-e).abs().max()))
      reports.append(dict(n=n,think=nthink,mode=mode,coefficient=coef,max_error=max(errors)))
   (ROOT/'integration_errors.json').write_text(json.dumps(reports,indent=2))
 def test_probe_invariance(self):
  tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'<pad>':0,'<unk>':1,'</think>':2,'a':3,'b':4,'c':5},unk_token='<unk>')),pad_token='<pad>',unk_token='<unk>')
  model=Qwen3ForCausalLM(Qwen3Config(vocab_size=6,hidden_size=8,intermediate_size=12,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=8,attention_dropout=.1))
  collator=BoundaryCollator(DataCollatorForSeq2Seq(tok))
  rows=[dict(input_ids=[3,4,5,2],labels=[-100,-100,5,2],attention_mask=[1]*4,term_positions=[1,2],term_targets=[0,1])]
  with tempfile.TemporaryDirectory() as tmp:
   args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to='none',remove_unused_columns=False)
   fa=FinetuningArguments(termination_loss_mode='end_ce',termination_loss_coef=.0012,termination_dataset_path='synthetic')
   trainer=EndCETrainer(model=model,args=args,finetuning_args=fa,processor=None,tokenizer=tok,data_collator=collator,train_dataset=Dataset.from_list(rows))
   params=list(model.parameters())
   for p in params:p.grad=torch.ones_like(p)
   weights=[p.detach().clone() for p in params];grads=[p.grad.clone() for p in params];rng=torch.get_rng_state().clone()
   trainer.end_probe_windows=[[collator(rows)]]*8
   result=trainer.gradient_probe(model,[collator(rows)],10)
   self.assertEqual(len(result['windows']),8)
   self.assertTrue(torch.equal(rng,torch.get_rng_state()))
   for p,w,g in zip(params,weights,grads):
    torch.testing.assert_close(p,w,rtol=0,atol=0);torch.testing.assert_close(p.grad,g,rtol=0,atol=0)

if __name__=='__main__':unittest.main(verbosity=2)
