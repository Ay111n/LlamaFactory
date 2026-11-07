"""Bounded reproduction of premature peer-copy destination cache release."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name in ("gold_reasoning_preflight_20261103", "gold_reasoning_throughput_20261103"):
    sys.path.insert(0, str(ROOT.parent / name))
import torch
from safe_pipeline import copy_target

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["original", "safe"])
args = p.parse_args()
torch.set_num_threads(8)
src, dst = torch.device("cuda:1"), torch.device("cuda:0")
report = {"mode": args.mode, "torch": torch.__version__, "pid": __import__("os").getpid(), "passed": False}
try:
    torch.cuda.set_device(dst)
    source = torch.full((64, 161280), 0.125, device=src, dtype=torch.float32)
    writer = torch.cuda.Stream(device=src)
    warm = source.to(dst)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    del warm
    torch.cuda.empty_cache()
    start = time.perf_counter()
    for repeat in range(10 if args.mode == "safe" else 1):
        if args.mode == "safe":
            with torch.cuda.stream(writer):
                torch.cuda._sleep(200_000_000)
            moved = copy_target(source, dst, writer)
        else:
            with torch.cuda.device(src):
                torch.cuda._sleep(200_000_000)
            moved = source.to(dst)
        del moved
        torch.cuda.empty_cache()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
    exact = copy_target(source, dst, writer)
    torch.testing.assert_close(exact.cpu(), source.cpu(), rtol=0, atol=0)
    report.update(passed=True, repetitions=repeat+1, seconds=time.perf_counter()-start, exact_fp32=True)
except BaseException as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    (ROOT / f"copy_lifetime_{args.mode}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
