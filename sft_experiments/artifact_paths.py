"""Separate custom training records from model packages; no writes on import."""
from pathlib import Path
import json

def diagnostic_path(experiment_root, model_root, model_file):
    experiment_root, model_root, model_file = map(Path, (experiment_root, model_root, model_file))
    suffix = model_file.relative_to(model_root).as_posix()
    current = experiment_root / 'diagnostics/run_records' / suffix
    locations = experiment_root / 'diagnostics/record_locations.json'
    if current.exists() or not locations.exists():
        return current
    archive = json.loads(locations.read_text(encoding='utf-8')).get(suffix)
    return experiment_root / archive if archive is not None else current
