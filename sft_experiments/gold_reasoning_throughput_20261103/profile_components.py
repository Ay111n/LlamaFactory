"""One bounded CUDA profile; timing is diagnostic, not the throughput benchmark."""
import json
import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from benchmark import setup, ROOT, write
from losses import ReferenceTargets
import losses


class TimedTargets(ReferenceTargets):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.events=[]

    def timed(self,name,fn,*args):
        with torch.cuda.device(self.device):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            with record_function('teacher.'+name):
                value=fn(*args)
            end.record()
        self.events.append((name,start,end))
        return value

    def hidden(self,*args):
        return self.timed('hidden',super().hidden,*args)

    def target(self,*args):
        return self.timed('target_'+args[-1],super().target,*args)


def main():
    trainer,directory=setup('baseline','component_profile',1)
    trainer.model=trainer.accelerator.prepare_model(trainer.model)
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs)
    teacher=trainer.reference.model
    provider=TimedTargets(teacher,beta=.5,cap=2.,chunk=64)
    trainer.reference=provider
    previous=json.loads((ROOT.parent/'gold_reasoning_preflight_20261103'/'attempt02'/'steps.jsonl').read_text().splitlines()[1])
    indices=previous['source_indices'][:4]
    batches=[trainer.data_collator([trainer.train_dataset[i]]) for i in indices]
    count=trainer._get_num_items_in_batch(batches,trainer.args.device)
    trainer.current_gradient_accumulation_steps=len(batches)
    trainer.start_window(batches,count)
    # Warm up the exact samples once, without parameter updates.
    for batch in batches:trainer.training_step(trainer.model,batch,num_items_in_batch=count)
    trainer.model.zero_grad(set_to_none=True)
    provider.events.clear()
    trainer.start_window(batches,count)
    begin=time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
        for batch in batches:trainer.training_step(trainer.model,batch,num_items_in_batch=count)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
    elapsed=time.perf_counter()-begin
    timers={}
    for name,start,end in provider.events:
        timers.setdefault(name,{'calls':0,'cuda_timeline_ms':0.})
        timers[name]['calls']+=1
        timers[name]['cuda_timeline_ms']+=start.elapsed_time(end)
    device_summary={}
    kernels={}
    for event in prof.events():
        if 'CUDA' not in str(event.device_type):continue
        device=int(event.device_index)
        duration=float(event.device_time_total)
        category='copy_or_memset' if ('memcpy' in event.name.lower() or 'memset' in event.name.lower()) else 'kernel'
        key=f'cuda:{device}/{category}'
        device_summary[key]=device_summary.get(key,0.)+duration/1000
        key=f'cuda:{device} {event.name}'
        kernels[key]=kernels.get(key,0.)+duration/1000
    result={'scope':'four actual samples, forward/backward, zero optimizer updates; profiler overhead included',
        'indices':indices,'wall_seconds':elapsed,'teacher_cuda_event_ranges':timers,
        'device_activity_ms':device_summary,'top_kernels_ms':sorted(kernels.items(),key=lambda x:x[1],reverse=True)[:20],
        'phase_metrics':trainer.window_metrics,'teacher_FP32':True,'TF32':False,
        'peer_access_0_to_1':torch.cuda.can_device_access_peer(0,1)}
    write(directory/'completion.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
