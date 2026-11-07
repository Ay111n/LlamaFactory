"""Native LF vs G2(lambda=0): loss/backward only, never optimizer updates."""
import copy,hashlib,inspect,json,math,os,tempfile
from pathlib import Path
import torch,yaml
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast,DataCollatorForSeq2Seq,Trainer,set_seed
from peft import LoraConfig,get_peft_model
from llamafactory.hparams import TrainingArguments,FinetuningArguments,get_train_args
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.sft.gold_termination_trainer import GoldTerminationTrainer
from llamafactory.train.sft.termination_trainer import BoundaryCollator

R=Path(os.environ['G2_PREFLIGHT_ROOT']);paths=json.loads((R/'paths.json').read_text())

def net(state=None):
 m=Qwen3ForCausalLM(Qwen3Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=8,attention_dropout=0.,tie_word_embeddings=False))
 m=get_peft_model(m,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],modules_to_save=['lm_head']))
 if state is not None:m.load_state_dict(state)
 return m

def records(n,think):
 rows=[]
 for i in range(n):
  ids=[3,4,5,2,6,7]+[3+j%12 for j in range((i*7)%41)]
  ids=[3,4,5,2]+[x if x!=2 else 3 for x in ids[4:]]
  flag=i<think
  rows.append(dict(input_ids=ids,labels=[-100,-100]+ids[2:],attention_mask=[1]*len(ids),term_positions=([1,2] if i%2 else [2]) if flag else [],term_targets=([0,1] if i%2 else [1]) if flag else []))
 return rows

def main():
 torch.set_num_threads(2);set_seed(3407)
 raw=yaml.safe_load(Path(paths['yaml']).read_text())
 ma,da,ta,fa,ga=get_train_args(dict(raw,use_cpu=True))
 effective={'seed':ta.seed,'data_seed':ta.data_seed,'optim':str(ta.optim),'lr_scheduler_type':str(ta.lr_scheduler_type),
  'gradient_checkpointing':ta.gradient_checkpointing,'gradient_checkpointing_kwargs':ta.gradient_checkpointing_kwargs,
  'use_reentrant_gc':ma.use_reentrant_gc,'train_on_prompt':da.train_on_prompt,'packing':da.packing,
  'cutoff_len':da.cutoff_len,'label_smoothing_factor':ta.label_smoothing_factor,
  'average_tokens_across_devices':ta.average_tokens_across_devices,'disable_shuffling':fa.disable_shuffling,
  'dataloader_num_workers':ta.dataloader_num_workers,'dataloader_drop_last':ta.dataloader_drop_last}
 (R/'original_yaml_effective.json').write_text(json.dumps(effective,indent=2))
 tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i):i for i in range(16)}|{'</think>':2},unk_token='1')),pad_token='0',unk_token='1')
 # Avoid a second token spelling for ID2 in a WordLevel map.
 vocab={str(i):i for i in range(16) if i!=2};vocab['</think>']=2
 tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab,unk_token='1')),pad_token='0',unk_token='1')
 collator=BoundaryCollator(DataCollatorForSeq2Seq(tok,pad_to_multiple_of=8))
 initial=copy.deepcopy(net().state_dict());reports=[]
 with tempfile.TemporaryDirectory() as tmp:
  cases=[(16,8),(8,4)] if os.environ.get('CONFIRM_ONLY')=='1' else [(16,0),(16,1),(16,8),(16,15),(16,16),(8,0),(8,1),(8,4),(8,8)]
  for n,t in cases:
   rows=records(n,t);batches=[collator([r]) for r in rows]
   den=sum(int(b['labels'][:,1:].ne(-100).sum()) for b in batches)
   paths_grads=[];losses=[];flags=[]
   for kind in ['native','g2_lambda0']:
    model=net(initial);before=copy.deepcopy(model.state_dict())
    args=TrainingArguments(output_dir=f'{tmp}/{n}-{t}-{kind}',use_cpu=True,report_to='none',gradient_accumulation_steps=16,per_device_train_batch_size=1,remove_unused_columns=False,save_strategy='no',seed=3407,data_seed=3407)
    fin=FinetuningArguments(termination_loss_mode='dynamic_gold_margin' if kind!='native' else 'none',termination_loss_coef=0.,termination_dataset_path='synthetic' if kind!='native' else None)
    cls=CustomSeq2SeqTrainer if kind=='native' else GoldTerminationTrainer
    trainer=cls(model=model,args=args,finetuning_args=fin,processor=None,tokenizer=tok,data_collator=collator,train_dataset=Dataset.from_list(rows))
    trainer.current_gradient_accumulation_steps=n;trainer.state.max_steps=100000
    native_den=Trainer._get_num_items_in_batch(trainer,batches,torch.device('cpu'))
    assert int(native_den)==den
    if kind!='native':trainer.get_batch_samples(iter(batches),n,torch.device('cpu'))
    trainer.gradient_probe=lambda *a,**kw:None
    torch.manual_seed(271828);loss=0.
    for b in batches:
     inp=dict(b)
     if kind=='native':inp.pop('term_positions');inp.pop('term_targets')
     loss+=float(trainer.training_step(model,inp,num_items_in_batch=native_den))
    paths_grads.append([None if p.grad is None else p.grad.detach().clone() for p in model.parameters() if p.requires_grad]);losses.append(loss)
    flags.append({'model_accepts_loss_kwargs':trainer.model_accepts_loss_kwargs,'accelerator_gradient_accumulation_steps':trainer.accelerator.gradient_accumulation_steps})
    assert trainer.optimizer is None
    for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
   maxerr=0.;diffnorm=0.;refnorm=0.;othernorm=0.;dot=0.;scaled_maxerr=0.;scaled_pass=True;passed=True
   for a,b in zip(*paths_grads):
    if a is None or b is None:
     passed &= a is None and b is None;continue
    maxerr=max(maxerr,float((a-b).abs().max()));diffnorm+=float((a-b).double().square().sum());refnorm+=float(a.double().square().sum())
    othernorm+=float(b.double().square().sum());dot+=float((a.double()*b.double()).sum())
    scaled_maxerr=max(scaled_maxerr,float((a-b/16).abs().max()))
    try:torch.testing.assert_close(a,b/16,rtol=2e-4,atol=2e-6)
    except AssertionError:scaled_pass=False
    try:torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-6)
    except AssertionError:passed=False
   passed &= math.isclose(losses[0],losses[1],rel_tol=2e-4,abs_tol=2e-6)
   # Deterministic logits quantify per-sample vs global-token weighting independently.
   model=net(initial).eval();means=[];sums=[];counts=[]
   with torch.no_grad():
    for b in batches:
     clean={k:v for k,v in b.items() if not k.startswith('term_')}
     c=int(clean['labels'][:,1:].ne(-100).sum());v=float(model(**clean,use_cache=False).loss)
     means.append(v);counts.append(c);sums.append(v*c)
   reports.append({'n':n,'think':t,'window_ce_tokens':den,'native_loss':losses[0],'g2_lambda0_loss':losses[1],
    'passed':bool(passed),'maximum_absolute_gradient_error':maxerr,'relative_l2_gradient_error':math.sqrt(diffnorm/refnorm) if refnorm else 0.,
    'g2_to_native_gradient_norm_ratio':math.sqrt(othernorm/refnorm),'gradient_cosine':dot/math.sqrt(othernorm*refnorm),
    'native_equals_g2_divided_16':scaled_pass,'maximum_error_after_dividing_g2_by_16':scaled_maxerr,
    'per_sample_mean':sum(means)/n,'window_token_mean':sum(sums)/sum(counts),'min_ce_tokens':min(counts),'max_ce_tokens':max(counts),'implementation_flags':flags})
  # Verify actual sampler behavior for full N, omitted data_seed vs explicit seed.
  orders=[]
  from accelerate.data_loader import SeedableRandomSampler
  for seed in [None,3407]:
   torch.manual_seed(3407)
   sampler=SeedableRandomSampler(range(49800),data_seed=seed)
   order=list(iter(sampler));orders.append(order)
  order_match=orders[0]==orders[1]
  sampler_report={'N':49800,'omitted_vs_explicit_3407_equal':order_match,
   'order_sha256':hashlib.sha256(json.dumps(orders[1]).encode()).hexdigest(),
   'scope':'current Accelerate SeedableRandomSampler; does not establish historical runtime identity'}
 result={'gate_pass':all(r['passed'] for r in reports),'optimizer_updates':0,'actual_stock_training_step_executed':True,
  'reports':reports,'original_yaml_effective':effective,'sampler':sampler_report,
  'current_runtime_only':'Historical runtime/source identity is not established by current-code equivalence',
  'maximum_gradient_error':max(r['maximum_absolute_gradient_error'] for r in reports)}
 name='ce_scale_confirmation.json' if os.environ.get('CONFIRM_ONLY')=='1' else 'ce_equivalence.json'
 (R/name).write_text(json.dumps(result,indent=2));(Path(paths['analysis'])/name).write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 if not result['gate_pass']:raise RuntimeError('Native CE vs G2 lambda=0 mismatch; stop training preparation')
if __name__=='__main__':main()
