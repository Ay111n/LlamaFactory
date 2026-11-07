"""Finalize evidence only after the bounded training and stress checks finish."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main(run_id):
    folder = ROOT / run_id
    completion = json.loads((folder / "completion.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    assert completion["passed"] and completion["optimizer_updates"] in (50, 100)
    assert completion["teacher_unchanged"] and completion["teacher_gradients_absent"]
    assert completion["student_parameters_changed"] and not completion["full_training_started"]
    assert json.loads((folder / "ce_equivalence.json").read_text())["passed"]
    assert json.loads((ROOT / "unit_checks.json").read_text())["passed"]
    assert json.loads((ROOT / "provenance.json").read_text())["passed"]
    assert json.loads((folder / "long_sequence_memory.json").read_text())["passed"]
    core = ("losses.py", "reasoning_trainer.py", "run_preflight.py", "prepare_data.py", "test_losses.py")
    hashes = {}
    for name in core:
        hashes[name] = sha(ROOT / name)
        assert hashes[name] == protocol["code_sha256"][name] == sha(folder / "source_snapshot" / name)
    output = Path(completion["model_output"])
    adapter = json.loads((output / "adapter_config.json").read_text())
    assert adapter["r"] == 64 and adapter["lora_alpha"] == 128 and adapter["modules_to_save"] == ["lm_head"]
    assert (output / "adapter_model.safetensors").stat().st_size > 3_000_000_000
    subprocess.run([sys.executable, str(ROOT / "summarize_preflight.py"), "--run-id", run_id], check=True, stdout=subprocess.DEVNULL)
    result = {"passed": True, "run_id": run_id, "training_code_unchanged_during_run": True,
              "core_code_sha256": hashes, "optimizer_updates": completion["optimizer_updates"],
              "adapter_path": str(output), "adapter_bytes": (output / "adapter_model.safetensors").stat().st_size,
              "adapter_sha256": sha(output / "adapter_model.safetensors"),
              "full_training_started": False, "report": str(folder / "report.txt")}
    (ROOT / "LATEST.json").write_text(json.dumps(result, indent=2)+"\n")
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != "delivery_manifest.json":
            files[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    (ROOT / "delivery_manifest.json").write_text(json.dumps({"files": files, "count": len(files)}, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="attempt02")
    main(parser.parse_args().run_id)
