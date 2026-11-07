"""Exercise real HF accumulation, CE reduction, and matched-margin correction."""
from experiment_paths import experiment_root,benchmark_root,data_root,model_root,paths,require_path,script_path,stage_log,evaluation_analysis_root

import copy
import json
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast,
                          Seq2SeqTrainingArguments,DataCollatorForSeq2Seq,TrainerCallback)
from llamafactory.hparams import FinetuningArguments, TrainingArguments
from llamafactory.train.sft.termination_trainer import BoundaryCollator,TerminationTrainer
from llamafactory.train.sft.termination_loss import boundary_losses,match_scale


def main():
    torch.manual_seed(3407)
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'<pad>':0,'<unk>':1,'</think>':2,'a':3,'b':4,'c':5,'d':6,'e':7},unk_token='<unk>')),
        pad_token='<pad>',unk_token='<unk>')
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=8,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        attention_dropout=0.,tie_word_embeddings=False))
    initial=copy.deepcopy(model.state_dict())
    records=[{'input_ids':[3,4,5,2,6,7], 'labels':[-100,-100,5,2,6,7],
              'attention_mask':[1]*6,'term_positions':[1,2],'term_targets':[0,1]},
             {'input_ids':[3,4,6,5], 'labels':[-100,-100,6,5],
              'attention_mask':[1]*4,'term_positions':[],'term_targets':[]}]
    collator=BoundaryCollator(DataCollatorForSeq2Seq(tokenizer,pad_to_multiple_of=8))
    batch=collator(records)
    clean={k:v for k,v in batch.items() if k not in ('term_positions','term_targets')}
    params=list(model.parameters())
    output=model(**clean,num_items_in_batch=6,use_cache=False)
    explicit=torch.nn.functional.cross_entropy(output.logits[:,:-1].reshape(-1,8),
        clean['labels'][:,1:].reshape(-1),ignore_index=-100,reduction='sum')/6
    torch.testing.assert_close(output.loss,explicit)
    one,two,_=boundary_losses(output.logits,batch['term_positions'],batch['term_targets'],2)
    g1=torch.autograd.grad(one,params,retain_graph=True,allow_unused=True)
    g2=torch.autograd.grad(two,params,retain_graph=True,allow_unused=True)
    c,_,_=match_scale(g1,g2)
    gradients={}
    for mode,aux in [('none',one*0),('margin',one),('dynamic',two),('matched_margin',c*one)]:
        gradients[mode]=torch.autograd.grad(output.loss+.2*aux,params,retain_graph=True,allow_unused=True)
    reports=[]
    with tempfile.TemporaryDirectory() as temp:
        for mode in gradients:
            net=Qwen3ForCausalLM(model.config);net.load_state_dict(initial)
            captured=[]
            class Capture(TrainerCallback):
                def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
                    captured.extend(p.grad.detach().cpu().clone() if p.grad is not None else None for p in model.parameters())
            args=TrainingArguments(output_dir=str(Path(temp)/mode),per_device_train_batch_size=1,
                gradient_accumulation_steps=2,max_steps=1,learning_rate=0,report_to='none',use_cpu=True,
                max_grad_norm=0,save_strategy='no',remove_unused_columns=False,seed=3407)
            fa=FinetuningArguments(termination_loss_mode=mode,termination_loss_coef=.2,termination_dataset_path='synthetic')
            trainer=TerminationTrainer(model=net,args=args,finetuning_args=fa,processor=None,
                tokenizer=tokenizer,data_collator=collator,train_dataset=Dataset.from_list(records),callbacks=[Capture()])
            # Diagnostic probes must preserve training RNG, weights and existing gradients.
            for parameter in net.parameters():parameter.grad=torch.ones_like(parameter)
            saved_weights=[x.detach().clone() for x in net.parameters()]
            saved_grads=[x.grad.clone() for x in net.parameters()]
            rng=torch.get_rng_state().clone()
            trainer.gradient_probe(net,[batch],0)
            assert torch.equal(rng,torch.get_rng_state())
            for parameter,weight,gradient in zip(net.parameters(),saved_weights,saved_grads):
                torch.testing.assert_close(parameter,weight,rtol=0,atol=0)
                torch.testing.assert_close(parameter.grad,gradient,rtol=0,atol=0)
            net.zero_grad()
            trainer.train()
            errors=[]
            for actual,expected in zip(captured,gradients[mode]):
                if expected is None:
                    assert actual is None or not actual.any()
                else:
                    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
                    errors.append(float((actual-expected).abs().max()))
            reports.append({'mode':mode,'maximum_gradient_error':max(errors)})
        print('INTEGRATION_PASS',json.dumps(reports))


if __name__=='__main__':main()
