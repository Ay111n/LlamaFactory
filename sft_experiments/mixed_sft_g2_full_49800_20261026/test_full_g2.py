"""FP32 actual native training_step equivalence, including reentrant checkpointing."""
import copy,importlib.util,json,math,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast,DataCollatorForSeq2Seq,Trainer
from peft import LoraConfig,get_peft_model
from experiment_paths import ROOT,require_path
from llamafactory.hparams import TrainingArguments,FinetuningArguments
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.sft.termination_gold import gold_boundary_losses
from full_g2_trainer import FullG2Trainer,FullBoundaryCollator


def model(state=None,reentrant=True):
 m=Qwen3ForCausalLM(Qwen3Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=8,attention_dropout=0.,tie_word_embeddings=False,use_cache=False))
 m=get_peft_model(m,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],modules_to_save=['lm_head']))
 if state is not None:m.load_state_dict(state)
 m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':reentrant});m.enable_input_require_grads()
 return m

def fixtures(n,think):
 rows=[]
 for i in range(n):
  ids=[3,4,5,2,6,7]+[3+j%12 for j in range((7*i)%41)]
  rows.append(dict(input_ids=ids,labels=[-100,-100]+ids[2:],attention_mask=[1]*len(ids),source_index=i,
                   term_positions=([1,2] if i%2 else [2]) if i<think else [],term_targets=([0,1] if i%2 else [1]) if i<think else []))
 return rows

class FullG2Tests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(3407)
  vocab={str(i):i for i in range(16) if i!=2};vocab['</think>']=2
  self.tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab,unk_token='1')),pad_token='0',unk_token='1')
  self.collate=FullBoundaryCollator(DataCollatorForSeq2Seq(self.tok,pad_to_multiple_of=8))
  self.initial=copy.deepcopy(model().state_dict())

 def build(self,path,rows,coef=0.,native=False,reentrant=True):
  m=model(self.initial,reentrant)
  args=TrainingArguments(output_dir=str(path),use_cpu=True,report_to='none',gradient_accumulation_steps=16,per_device_train_batch_size=1,
   remove_unused_columns=False,save_strategy='no',seed=3407,data_seed=3407,max_grad_norm=1.,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':reentrant})
  fin=FinetuningArguments(termination_loss_mode='none' if native else 'dynamic_gold_margin',termination_loss_coef=coef,termination_dataset_path=None if native else 'synthetic')
  cls=CustomSeq2SeqTrainer if native else FullG2Trainer
  t=cls(model=m,args=args,finetuning_args=fin,processor=None,tokenizer=self.tok,data_collator=self.collate,train_dataset=Dataset.from_list(rows))
  return t

 def test_native_and_auxiliary_gradients(self):
  reports=[]
  with tempfile.TemporaryDirectory() as tmp:
   for n,think in [(16,0),(16,1),(16,8),(16,15),(16,16),(8,0),(8,1),(8,4),(8,8),(3,1)]:
    rows=fixtures(n,think);batches=[self.collate([r]) for r in rows]
    for reentrant in [True,False]:
     for coef in [0.,.0012]:
      reference=self.build(Path(tmp)/f'ref-{n}-{think}-{reentrant}-{coef}',rows,native=True,reentrant=reentrant)
      candidate=self.build(Path(tmp)/f'new-{n}-{think}-{reentrant}-{coef}',rows,coef=coef,reentrant=reentrant)
      snapshots=[copy.deepcopy(t.model.state_dict()) for t in [reference,candidate]]
      count=Trainer._get_num_items_in_batch(reference,batches,torch.device('cpu'))
      candidate.get_batch_samples(iter(batches),n,torch.device('cpu'))
      self.assertEqual(int(count),candidate._window_tokens)
      result=[];values=[]
      for is_new,t in [(False,reference),(True,candidate)]:
       t.current_gradient_accumulation_steps=n;t.state.max_steps=9339;t.model.train();torch.manual_seed(271828)
       total=0.
       for batch in batches:
        if is_new:total+=float(t.training_step(t.model,dict(batch),num_items_in_batch=count))
        elif not coef:
         clean={k:v for k,v in batch.items() if k not in ['term_positions','term_targets','source_index']}
         total+=float(t.training_step(t.model,clean,num_items_in_batch=count))
        else:
         clean={k:v for k,v in batch.items() if k not in ['term_positions','term_targets','source_index']}
         ce,out=t.compute_loss(t.model,clean,return_outputs=True,num_items_in_batch=count)
         auxiliary,_,_=gold_boundary_losses(out.logits,clean['labels'],batch['term_positions'],batch['term_targets'],2,clean['input_ids'])
         loss=ce+coef*auxiliary['dynamic_gold']/max(think,1)
         t.accelerator.backward(loss);total+=float(loss.detach())
       result.append([None if p.grad is None else p.grad.detach().clone() for p in t.model.parameters() if p.requires_grad]);values.append(total)
       self.assertIsNone(t.optimizer)
      errors=[]
      for a,b in zip(*result):
       if a is None or b is None:self.assertTrue(a is None and b is None)
       else:torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-6);errors.append(float((a-b).abs().max()))
      self.assertTrue(math.isclose(*values,rel_tol=2e-4,abs_tol=2e-6))
      for trainer,snapshot in zip([reference,candidate],snapshots):
       for key,value in trainer.model.state_dict().items():torch.testing.assert_close(value,snapshot[key],rtol=0,atol=0)
      # Joint clipping acts on precisely the same accumulated gradient vector.
      for trainer in [reference,candidate]:trainer.accelerator.clip_grad_norm_([p for p in trainer.model.parameters() if p.requires_grad],.01)
      for a,b in zip([p for p in reference.model.parameters() if p.requires_grad],candidate.term_params):
       if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=2e-6)
      reports.append({'n':n,'think':think,'reentrant':reentrant,'coefficient':coef,'native_or_manual_joint_loss':values[0],'new_loss':values[1],
       'max_parameter_gradient_error':max(errors,default=0.),'ce_tokens':int(count),'accelerator_ga':candidate.accelerator.gradient_accumulation_steps,'optimizer_updates':0})
  (ROOT/'unit_gradient_equivalence.json').write_text(json.dumps(reports,indent=2))

 def test_reentrant_probe_invariance_and_component_sum(self):
  rows=fixtures(8,4);batches=[self.collate([r]) for r in rows]
  with tempfile.TemporaryDirectory() as tmp:
   trainer=self.build(Path(tmp),rows,coef=.0012,reentrant=True)
   params=trainer.term_params
   for p in params:p.grad=torch.ones_like(p)
   original=[p.grad for p in params];grad_values=[g.clone() for g in original]
   weights=[p.detach().clone() for p in params]
   torch.manual_seed(17);rng=torch.get_rng_state().clone();random_state=random.getstate();np_state=np.random.get_state()
   trainer.model.eval();mode_flags=[m.training for m in trainer.model.modules()]
   row=trainer.probe_window(trainer.model,batches,0,0,'unit')
   self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(random.getstate(),random_state)
   self.assertTrue(np.array_equal(np.random.get_state()[1],np_state[1]));self.assertEqual(mode_flags,[m.training for m in trainer.model.modules()])
   for p,g,gv,w in zip(params,original,grad_values,weights):
    self.assertIs(p.grad,g);torch.testing.assert_close(p.grad,gv,rtol=0,atol=0);torch.testing.assert_close(p,w,rtol=0,atol=0)
   self.assertLess(row['component_sum_relative_residual'],2e-4)
   self.assertTrue(math.isclose(row['auxiliary_loss'],row['continue_loss']+row['stop_loss'],rel_tol=2e-5,abs_tol=2e-6))
   (ROOT/'unit_probe_invariance.json').write_text(json.dumps(row,indent=2))

if __name__=='__main__':
 # Reuse the original Gold formula/edge tests without running historical integration writers.
 spec=importlib.util.spec_from_file_location('old_gold_tests',require_path('gold_experiment_root')/'test_gold.py')
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(FullG2Tests),
  old.GoldTests('test_formula_direction_detach'),old.GoldTests('test_extreme_empty_invalid')])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 (ROOT/'unit_gate.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}))
 raise SystemExit(0 if result.wasSuccessful() else 1)
