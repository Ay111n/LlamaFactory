"""Single-L20 convolution A/B on a fixed selection from the English-CoT SFT data."""
import hashlib
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import pyarrow.compute as pc
import torch
from datasets import DatasetDict
from torch.nn import functional as F
from torch.utils.data import SequentialSampler
from transformers import AutoConfig, TrainerCallback
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from fla.modules.convolution import causal_conv1d

HERE = Path(__file__).resolve().parent
WORK = Path(sys.argv[1]).resolve()
WORK.mkdir(parents=True, exist_ok=False)
FIXED = Path(sys.argv[2]).resolve()
previous = HERE.parent / 'qwen35_crossref_teacher_9960_20261111'
sys.path.insert(0, str(previous))
from memory_ce import install
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.tuner import run_exp

ARMS = ('torch', 'fla', 'torch', 'fla', 'fla', 'torch')
MICROS = 16


def save(name, payload):
    path = WORK / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def triton_conv(x, weight, bias, activation, seq_idx=None):
    assert seq_idx is None
    output, _ = causal_conv1d(x.transpose(1, 2), weight=weight, bias=bias,
                             activation=activation, backend='triton')
    return output.transpose(1, 2)


def torch_conv(x, weight, bias):
    conv = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1]-1, groups=x.shape[1])
    return F.silu(conv[..., :x.shape[-1]])


def operator(model_config):
    config = model_config.text_config
    channels = (2 * config.linear_num_key_heads * config.linear_key_head_dim
                + config.linear_num_value_heads * config.linear_value_head_dim)
    width = config.linear_conv_kernel_dim
    out = []
    gen = torch.Generator(device='cuda').manual_seed(3407)
    for n in (1267, 2048, 4096, 4568):
        x = torch.randn((1, n, channels), generator=gen, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
        w = torch.randn((channels, width), generator=gen, device='cuda', dtype=torch.bfloat16) * 0.1
        dy = torch.randn((1, n, channels), generator=gen, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
        arms = {'torch': lambda t: torch_conv(t, w, None), 'fla': lambda t: triton_conv(t, w, None, 'silu')}
        measured = {name:[] for name in arms}
        numerics = {}
        for name in ('torch','fla','fla','torch') * 6:
            a = x.detach().requires_grad_(True)
            start, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            start.record()
            y = arms[name](a)
            middle.record()
            dx, = torch.autograd.grad(y, a, dy)
            end.record()
            end.synchronize()
            if name not in numerics:
                numerics[name] = (y.detach(), dx.detach())
            measured[name].append([start.elapsed_time(middle), middle.elapsed_time(end), start.elapsed_time(end)])
        relative = [
            float((a.float()-b.float()).norm()/a.float().norm())
            for a,b in zip(numerics['torch'],numerics['fla'],strict=True)
        ]
        assert all(v < 0.025 for v in relative), relative
        values = {k:dict(forward_ms=st.median(r[0] for r in v[2:]),
                         backward_ms=st.median(r[1] for r in v[2:]),
                         full_ms=st.median(r[2] for r in v[2:]),
                         raw_samples=v[2:]) for k,v in measured.items()}
        out.append(dict(length=n,channels=channels,kernel_width=width,timings=values,
                        relative_l2_output=relative[0],relative_l2_input_grad=relative[1]))
        save('operator.json', out)
        print('OPERATOR',n, {k:round(v['full_ms'],3) for k,v in values.items()},flush=True)
    torch.cuda.empty_cache()


from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer

cfg = json.loads((HERE/'recipe.json').read_text())
fixed = json.loads(FIXED.read_text())
dataset_file = Path(cfg['dataset_dir'])/'en_cot_10000.jsonl'
manifest = json.loads((dataset_file.parent/'manifest.json').read_text())
sha = manifest['outputs'][dataset_file.name]['sha256']
with dataset_file.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest() == sha
model_args, data_args, training_args, *_ = get_train_args(cfg)
tokenizer_module = load_tokenizer(model_args)
template = get_template_and_fix_tokenizer(tokenizer_module['tokenizer'],data_args)
all_data = get_dataset(template,model_args,data_args,training_args,stage='sft',**tokenizer_module)['train_dataset']
lengths = pc.list_value_length(all_data.data.column('input_ids')).to_pylist()
assert len(lengths)==10000 and max(lengths)<=cfg['cutoff_len']
with dataset_file.open(encoding='utf-8') as f:
    by_uid = {json.loads(line)['uid']:i for i,line in enumerate(f)}
positions = [by_uid[uid] for uid in fixed['uids']]
chosen = [{'index':i,'uid':uid,'tokens':lengths[i]} for i,uid in zip(positions,fixed['uids'],strict=True)]
assert [row['tokens'] for row in chosen]==fixed['lengths']
tokens_per_step = sum(r['tokens'] for r in chosen)
assert tokens_per_step==fixed['tokens_per_step']
assert sum(lengths)==manifest['outputs'][dataset_file.name]['total_tokens']
expected_labels=[all_data[i]['labels'] for i in positions]
supervised=sum(sum(label!=-100 for label in labels) for labels in expected_labels)
assert supervised==29886
save('selection.json',{'samples':chosen,'arms':ARMS,'source_sha256':sha,
                       'total_dataset_tokens':sum(lengths),'probe_tokens_per_update':tokens_per_step,
                       'supervised_per_update':supervised})
print('SELECTED',tokens_per_step,[r['tokens'] for r in chosen],flush=True)
assert cfg['per_device_train_batch_size']==1 and cfg['gradient_accumulation_steps']==MICROS
cfg.update(tokenized_path=str(WORK/'encoded'),output_dir=str(WORK/'trainer'),max_steps=len(ARMS),
           learning_rate=0.0,save_strategy='no',save_only_model=True,plot_loss=False,disable_tqdm=True)
DatasetDict(train=all_data.select(positions*len(ARMS))).save_to_disk(cfg['tokenized_path'])
save('config.json',cfg)
operator(AutoConfig.from_pretrained(cfg['model_name_or_path'],local_files_only=True))
install(warmup=False)
CustomSeq2SeqTrainer.save_model = lambda *a,**k: None
CustomSeq2SeqTrainer._get_train_sampler = lambda self,*a,**k: SequentialSampler(self.train_dataset)
orig_step = CustomSeq2SeqTrainer.training_step
samples = []
updates = []


class Timing(TrainerCallback):
    def on_train_begin(self,args,state,control,model=None,**kwargs):
        self.layers = [m for m in model.modules() if isinstance(m,Qwen3_5GatedDeltaNet)]
        assert self.layers and all(m.causal_conv1d_fn is None for m in self.layers)
        assert all(m.chunk_gated_delta_rule.__module__.startswith('fla.') for m in self.layers)
        save('paths.json',{'gated_delta_layers':len(self.layers),
                           'delta_kernel':self.layers[0].chunk_gated_delta_rule.__module__,
                           'baseline_conv':'native torch fallback from Qwen3.5',
                           'comparison_conv':'FLA Triton causal_conv1d, loaded only for this process'})

    def on_step_begin(self,args,state,control,**kwargs):
        arm = ARMS[state.global_step]
        for m in self.layers:
            m.causal_conv1d_fn = None if arm=='torch' else triton_conv
        torch.manual_seed(1234)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def on_step_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        wall = time.perf_counter()-self.start
        i=state.global_step-1
        selected=samples[i*MICROS:(i+1)*MICROS]
        assert len(selected)==MICROS
        assert all(math.isfinite(float(x)) for x in selected)
        row=dict(step=state.global_step,arm=ARMS[i],warmup=i<2,wall_seconds=wall,
                 tokens=tokens_per_step,tokens_per_second=tokens_per_step/wall,
                 loss_sum=sum(float(x) for x in selected),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                 peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        updates.append(row)
        save('updates.json',updates)
        print('MEASURED',json.dumps(row),flush=True)


def checked_step(self,model,inputs,*args,**kwargs):
    item=chosen[len(samples)%MICROS]
    actual=int(inputs['attention_mask'].sum().item())
    assert actual==item['tokens'],(actual,item)
    labels=inputs['labels'][0].detach().cpu()
    active=inputs['attention_mask'][0].detach().cpu().bool()
    assert labels[active].tolist()==expected_labels[len(samples)%MICROS]
    assert torch.all(labels[~active]==-100)
    loss=orig_step(self,model,inputs,*args,**kwargs)
    samples.append(loss.detach())
    return loss


CustomSeq2SeqTrainer.training_step=checked_step
run_exp(cfg,callbacks=[Timing()])
assert len(updates)==len(ARMS) and len(samples)==len(ARMS)*MICROS
measured={k:[x for x in updates if x['arm']==k and not x['warmup']] for k in ('torch','fla')}
result={}
for k,v in measured.items():
    speeds=[x['tokens_per_second'] for x in v]
    result[k]=dict(median_tokens_per_second=st.median(speeds),
                   hours_per_epoch=sum(lengths)/st.median(speeds)/3600,
                   observed_epoch_hours=[sum(lengths)/s/3600 for s in speeds],
                   seconds_per_update=[x['wall_seconds'] for x in v],
                   max_allocated_gib=max(x['peak_allocated_gib'] for x in v))
result['gain_percent']=(result['fla']['median_tokens_per_second']/result['torch']['median_tokens_per_second']-1)*100
result['median_loss_delta']=st.median(x['loss_sum'] for x in measured['fla'])-st.median(x['loss_sum'] for x in measured['torch'])
result['official_training_started']=False
save('result.json',result)
print('RESULT',json.dumps(result),flush=True)
