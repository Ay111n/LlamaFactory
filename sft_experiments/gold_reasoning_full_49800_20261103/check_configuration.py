"""Parse the real full-run YAML using the actual training environment; no model."""
from unittest.mock import patch
import yaml
from llamafactory.hparams import get_train_args
from common import *

cfg = yaml.safe_load((ROOT / "full_training.yaml").read_text())
with patch("torch.cuda.device_count", return_value=1):
    ma, da, ta, fa, ga = get_train_args(cfg)
ta._n_gpu = 1
assert ma.model_name_or_path == BASE and ma.adapter_name_or_path is None
assert ta.num_train_epochs == 3 and ta.max_steps == -1 and ta.save_steps == 1500
assert ta.save_strategy.value == "steps" and not ta.save_only_model and ta.save_total_limit == 10
assert ta.gradient_accumulation_steps == 16 and ta.per_device_train_batch_size == 1 and ta.world_size == 1
assert fa.lora_rank == 64 and fa.additional_target == ["lm_head"]
write(ROOT / "configuration_check.json", {"passed": True, "model_loaded": False, "optimizer_updates": 0,
    "epochs": ta.num_train_epochs, "save_steps": ta.save_steps, "save_strategy": ta.save_strategy.value,
    "save_only_model": ta.save_only_model, "fresh_base": ma.model_name_or_path, "checked_at_utc": utc_now()})
print("FULL_CONFIGURATION_PARSED", flush=True)
