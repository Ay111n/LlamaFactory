"""CPU integration check of actual Trainer+Accelerator accumulation path."""
import json
from pathlib import Path
import torch
from transformers import Trainer,TrainingArguments,TrainerCallback
from training_step import mean_training_step

checks=[]
class MeanTrainer(Trainer):
    training_step=mean_training_step
    def compute_loss(self,model,inputs,**kwargs):
        return model.weight.sum()*inputs['scale'].float().mean()
class Capture(TrainerCallback):
    def on_pre_optimizer_step(self,args,state,control,**kw):
        seen.append(float(kw['model'].weight.grad))
for gas,n in [(1,1),(4,4),(4,3),(32,32)]:
    seen=[]
    args=TrainingArguments(output_dir='/home/nlp/data/lyl/tmp_q35/student_accumulation_check',use_cpu=True,
        num_train_epochs=1,per_device_train_batch_size=1,gradient_accumulation_steps=gas,
        learning_rate=0.,max_grad_norm=0.,save_strategy='no',report_to=[],remove_unused_columns=False,disable_tqdm=True)
    model=torch.nn.Linear(1,1,bias=False)
    trainer=MeanTrainer(model=model,args=args,train_dataset=[{'scale':float(i+1)} for i in range(n)],callbacks=[Capture()])
    trainer.model_accepts_loss_kwargs=False
    trainer.train()
    expected=(n+1)/2
    assert len(seen)==1 and abs(seen[0]-expected)<1e-5,(gas,n,seen,expected)
    checks.append({'GAS':gas,'examples':n,'actual_gradient':seen[0],'expected_gradient':expected})
result={'status':'passed','checks':checks}
Path(__file__).with_name('runtime').mkdir(exist_ok=True)
Path(__file__).with_name('runtime').joinpath('accumulation_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
