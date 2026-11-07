"""Remove only this run's redundant dataset-construction cache after completion."""
import json
from pathlib import Path
import shutil
from prepare_data import DATA, ROOT, write

completion = json.loads((ROOT / "attempt02/completion.json").read_text())
assert completion["passed"] and completion["optimizer_updates"] == 50
saved = (DATA / "dataset").resolve()
state = json.loads((saved / "state.json").read_text())
assert state["_data_files"]
for item in state["_data_files"]:
    assert (saved / item["filename"]).is_file()
    assert (saved / item["filename"]).resolve().parent == saved
cache = (DATA / "build_cache").resolve()
assert cache.parent == DATA.resolve() and cache.name == "build_cache"
assert not (DATA / "build_cache").is_symlink()
before = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()) if cache.exists() else 0
if cache.exists():
    shutil.rmtree(cache)
write(ROOT / "cache_cleanup.json", {"removed_only": str(cache), "bytes_freed": before,
      "saved_dataset_preserved": str(saved), "original_inputs_and_models_untouched": True})
print("REMOVED_REDUNDANT_BUILD_CACHE", before)
