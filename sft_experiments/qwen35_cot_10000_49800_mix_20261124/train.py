"""Prepare and run the three matched 10k Qwen3.5 SFT variants."""
import hashlib
import math
import json
import sys
from itertools import islice
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
VARIANTS = ("empty", "bo", "en")
RECIPE = json.loads((HERE / "recipe.json").read_text())
DATA = Path(RECIPE["dataset_dir"])
MANIFEST = json.loads((DATA / "manifest.json").read_text())


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare():
    from datasets import DatasetDict
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    RUNTIME.mkdir(exist_ok=False)
    reports = {}
    trace_count = sum(cell["selected"] for cell in MANIFEST["cells"] if cell["think"] == "think")
    reference = None
    for variant in VARIANTS:
        alias = f"{variant}_cot_10000"
        source = DATA / f"{alias}.jsonl"
        output = Path(RECIPE["output_dir"]).with_name(f"qwen35_{alias}_49800_mix_20261124")
        assert not output.exists(), output
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        meta = MANIFEST["outputs"][source.name]
        assert digest == meta["sha256"], source
        config = dict(RECIPE, dataset=alias, output_dir=str(output))
        model_args, data_args, train_args, *_ = get_train_args(config)
        module = load_tokenizer(model_args)
        tok = module["tokenizer"]
        template = get_template_and_fix_tokenizer(tok, data_args)
        dataset = get_dataset(template, model_args, data_args, train_args, stage="sft", **module)["train_dataset"]
        assert len(dataset) == 10000
        pairs, lengths, supervised = [], [], 0
        nonempty = 0
        with source.open(encoding="utf-8") as stream:
            for start in range(0, len(dataset), 128):
                rows = [json.loads(line) for line in islice(stream, 128)]
                prompts = [row["messages"][0]["content"] for row in rows]
                responses = [row["messages"][1]["content"] for row in rows]
                prefices = tok([f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for q in prompts],
                              add_special_tokens=False, verbose=False).input_ids
                targets = tok([a + "<|im_end|>\n" for a in responses],
                              add_special_tokens=False, verbose=False).input_ids
                actual = dataset[start:start + len(rows)]
                for row, q, a, prefix, target, ids, labels in zip(
                        rows, prompts, responses, prefices, targets, actual["input_ids"], actual["labels"], strict=True):
                    assert a.startswith("<think>\n"), row["uid"]
                    trace, answer = a.split("</think>\n\n", 1)
                    nonempty += bool(trace.removeprefix("<think>").strip())
                    pairs.append((row["uid"], q, answer))
                    assert ids == prefix + target and labels == [-100] * len(prefix) + target, row["uid"]
                    assert len(ids) < config["cutoff_len"], row["uid"]
                    lengths.append(len(ids))
                    supervised += len(target)
            assert not stream.read(1)
        order = hashlib.sha256(("".join(uid + "\n" for uid, _, _ in pairs)).encode()).hexdigest()
        assert order == MANIFEST["uid_order_sha256"]
        assert len(set(uid for uid, _, _ in pairs)) == len(pairs) == len(lengths) == 10000
        assert nonempty == (0 if variant == "empty" else trace_count)
        assert sum(lengths) == meta["total_tokens"] and max(lengths) == meta["max_tokens"]
        assert supervised == meta["supervised_tokens"]
        if reference is None:
            reference = pairs
        else:
            assert pairs == reference, alias
        run_dir = RUNTIME / variant
        run_dir.mkdir()
        config["tokenized_path"] = str(run_dir / "encoded")
        DatasetDict(train=dataset).save_to_disk(config["tokenized_path"])
        write_json(run_dir / "config.json", config)
        reports[variant] = dict(rows=len(dataset), nonempty_think=nonempty, total_tokens=sum(lengths),
                                max_tokens=max(lengths), supervised_tokens=supervised, sha256=digest,
                                output_dir=str(output))
        print("VERIFIED", variant, reports[variant], flush=True)
    write_json(RUNTIME / "preflight.json", dict(uid_order_sha256=order, tibetan_questions_and_answers_match=True,
                                               exact_loss_masks=True, variants=reports))


def train(variant):
    from transformers import TrainerCallback

    report = json.loads((RUNTIME / "preflight.json").read_text())
    assert set(report["variants"]) == set(VARIANTS)
    run_dir = RUNTIME / variant
    config = json.loads((run_dir / "config.json").read_text())
    destination = Path(config["output_dir"])
    assert str(destination) == report["variants"][variant]["output_dir"] and not destination.exists()

    class MatchedSchedule(TrainerCallback):
        def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            from torch.optim.lr_scheduler import LambdaLR

            schedule = getattr(lr_scheduler, "scheduler", lr_scheduler)
            assert isinstance(schedule, LambdaLR)
            assert state.global_step == 0 and state.max_steps == 1875 and schedule.last_epoch == 0
            assert args.warmup_steps == 32

            def lr_scale(step):
                phase_end = 625 if step < 625 else 1250
                position = step if step < 625 else step - 625
                if position < 32:
                    return position / 32
                progress = min(1.0, (position - 32) / (phase_end - 32))
                return 0.5 * (1 + math.cos(math.pi * progress))

            schedule.lr_lambdas = [lr_scale] * len(schedule.base_lrs)
            print(f"MATCHED_SCHEDULE variant={variant} phases=625+1250 warmup=32 peak_lr={args.learning_rate}",
                  flush=True)

    sys.path.insert(0, str(HERE.parent / "qwen35_crossref_teacher_9960_20261111"))
    from memory_ce import install
    from llamafactory.train.tuner import run_exp

    install(warmup=False)
    run_exp(config, callbacks=[MatchedSchedule()])
    final = destination / "checkpoint-1875"
    end = json.loads((final / "trainer_state.json").read_text())
    assert end["epoch"] == 3.0 and end["global_step"] == end["max_steps"] == 1875
    for filename in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        assert (final / filename).stat().st_size > 0
    write_json(run_dir / "completion.json", dict(checkpoint=str(final), epoch=3, step=1875))


if __name__ == "__main__":
    command = sys.argv[1:]
    if command == ["prepare"]:
        prepare()
    elif len(command) == 2 and command[0] == "train" and command[1] in VARIANTS:
        train(command[1])
    else:
        raise SystemExit("Usage: train.py prepare | train {empty,bo,en}")
