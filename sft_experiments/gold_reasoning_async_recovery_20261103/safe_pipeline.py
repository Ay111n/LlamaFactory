"""Protect cross-device target-copy storage without changing loss arithmetic.

PyTorch 2.6 Copy.cu queues GPU-to-GPU writes on the source device. Explicitly
record that non-default writer stream on BOTH source and destination storage,
so allocator cache release cannot unmap a destination before a peer writes it.
The original KL sees an already-local FP32 tensor; its extra .to() is a no-op.
"""
import torch
from optimized import PipelineTargets, PipelineTrainer


def copy_target(source, destination, writer_stream):
    assert source.device == writer_stream.device
    assert writer_stream.cuda_stream != 0
    with torch.cuda.stream(writer_stream):
        moved = source.to(destination)
        source.record_stream(writer_stream)
        moved.record_stream(writer_stream)
    return moved


class SafePipelineTargets(PipelineTargets):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_calls = 0

    @torch.no_grad()
    def target(self, data, start, end, mode):
        source = super().target(data, start, end, mode)
        moved = copy_target(source, torch.device("cuda:0"), self.copy_stream)
        self.copy_calls += 1
        return moved

    def close(self):
        super().close()
        self.copy_stream.synchronize()
