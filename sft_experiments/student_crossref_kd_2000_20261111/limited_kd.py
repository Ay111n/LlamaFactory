"""Run the unchanged KD arm on GPU 2 with a bounded PyTorch allocator."""
import os
import torch
import experiment as e

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
limit = 23 * 1024**3
torch.cuda.set_per_process_memory_fraction(limit / torch.cuda.get_device_properties(0).total_memory, 0)
e.save(e.RUNTIME/'kd_resource_limit.json', {'gpu':2,'allocator_limit_gib':23,'pid':os.getpid(),
       'note':'CUDA context and external allocations are additional; no data, loss, or training settings changed.'})
e.train('kd', smoke=False)
