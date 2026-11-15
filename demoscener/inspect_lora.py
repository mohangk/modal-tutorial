"""Inspect downloaded adapters; optionally verify that they change predictions.

Run `uv run python run.py --download-only` first. This script uses cached files,
does not train, and does not contact Modal.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from run import ADAPTER_ID, ADAPTER_REVISION, BASE_ID, BASE_REVISION, DEFAULT_PREFIX


def inspect_adapters(adapter_path: Path) -> list[dict]:
    from safetensors import safe_open

    results = []
    for stage in ("cpt", "curriculum", "grpo"):
        directory = adapter_path / stage
        config = json.loads((directory / "adapter_config.json").read_text())
        weights = directory / "adapter_model.safetensors"
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with safe_open(weights, framework="pt", device="cpu") as tensors:
            names = list(tensors.keys())
            count = sum(math.prod(tensors.get_slice(k).get_shape()) for k in names)
            stem = "base_model.model.model.layers.0.self_attn.q_proj"
            examples = {
                side: dict(shape=tensors.get_slice(f"{stem}.lora_{side}.weight").get_shape(),
                           dtype=str(tensors.get_tensor(f"{stem}.lora_{side}.weight").dtype))
                for side in ("A", "B")
            }
        results.append(dict(
            stage=stage, rank=config["r"], alpha=config["lora_alpha"],
            scaling=config["lora_alpha"] / config["r"],
            target_modules=sorted(config["target_modules"]),
            tensor_count=len(names), parameter_count=count,
            file_bytes=weights.stat().st_size, sha256=digest,
            first_attention_query=examples,
        ))
    return results


def check_predictions(base_path: str, adapter_path: Path, prefix: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        base_logits = model(**inputs).logits[0, -1].float()
    base_probs = base_logits.softmax(-1)
    results = []
    for stage in ("cpt", "curriculum", "grpo"):
        if stage == "cpt":
            model = PeftModel.from_pretrained(
                model, str(adapter_path / stage), adapter_name=stage,
                is_trainable=False, autocast_adapter_dtype=False,
            )
        else:
            model.load_adapter(
                str(adapter_path / stage), adapter_name=stage,
                is_trainable=False, autocast_adapter_dtype=False,
            )
        model.set_adapter(stage)
        model.eval()
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
        probs = logits.softmax(-1)
        values, indices = probs.topk(5)
        results.append(dict(
            stage=stage, active_adapters=model.active_adapters,
            max_absolute_logit_change_from_base=float((logits - base_logits).abs().max()),
            probability_l1_change_from_base=float((probs - base_probs).abs().sum()),
            top_next_tokens=[dict(token=tokenizer.decode([i]), probability=p)
                             for i, p in zip(indices.tolist(), values.tolist())],
        ))
    return dict(prefix=prefix, comparisons=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-logits", action="store_true",
                        help="Also load the 7B base on CPU and compare next-token scores")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    adapter_path = Path(snapshot_download(
        ADAPTER_ID, revision=ADAPTER_REVISION, local_files_only=True,
        allow_patterns=["README.md", "*/adapter_config.json", "*/adapter_model.safetensors"],
    ))
    report = dict(adapter_revision=ADAPTER_REVISION,
                  adapters=inspect_adapters(adapter_path))
    if args.check_logits:
        base_path = snapshot_download(
            BASE_ID, revision=BASE_REVISION, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE"],
        )
        report["base_revision"] = BASE_REVISION
        report["prediction_check"] = check_predictions(base_path, adapter_path, args.prefix)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
