"""Download Demoscener and compare frozen training stages on a local CPU."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
# Keep large downloads inside this experiment, away from the Modal project.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_ID = "Qwen/Qwen2.5-Coder-7B"
BASE_REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"
ADAPTER_ID = "datavorous/Demoscener-7B"
ADAPTER_REVISION = "315703a76fb81f7771225a092cd1f2af77ee12ba"
STAGES = ("base", "cpt", "curriculum", "grpo")
DEFAULT_PREFIX = "c.width=1920;for(i=0;i<80;i++)x.fillRect("


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0,
                        help="0 = greedy comparison; positive = sample continuations")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 1 or args.temperature < 0:
        parser.error("tokens/threads must be positive; temperature must be nonnegative")

    from huggingface_hub import snapshot_download

    print(f"Downloading/caching {BASE_ID} at {BASE_REVISION}", flush=True)
    base_path = snapshot_download(
        BASE_ID, revision=BASE_REVISION,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE"],
        max_workers=4,
    )
    print(f"Downloading/caching {ADAPTER_ID} at {ADAPTER_REVISION}", flush=True)
    adapter_path = Path(snapshot_download(
        ADAPTER_ID, revision=ADAPTER_REVISION,
        allow_patterns=["README.md", "*/adapter_config.json", "*/adapter_model.safetensors"],
        max_workers=3,
    ))
    if args.download_only:
        print(f"Base: {base_path}\nAdapters: {adapter_path}", flush=True)
        return

    import torch
    from importlib.metadata import version
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    torch.set_num_threads(args.threads)
    print(f"Loading base on CPU in BF16 ({args.threads} threads)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="cpu",
        local_files_only=True, attn_implementation="sdpa",
    ).eval()
    stages_needed = [s for s in args.stages if s != "base"]
    if stages_needed:
        first, *others = stages_needed
        model = PeftModel.from_pretrained(
            model, str(adapter_path / first), adapter_name=first,
            is_trainable=False, autocast_adapter_dtype=False,
        )
        for stage in others:
            model.load_adapter(
                str(adapter_path / stage), adapter_name=stage,
                is_trainable=False, autocast_adapter_dtype=False,
            )
        model.eval()

    inputs = tokenizer(args.prefix, return_tensors="pt", add_special_tokens=False)
    results = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = ROOT / "outputs" / stamp
    output_dir.mkdir(parents=True)

    for stage in args.stages:
        set_seed(args.seed)
        if stage != "base":
            model.set_adapter(stage)
        print(f"\nGenerating {stage} …", flush=True)
        started = perf_counter()
        generation = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        if args.temperature > 0:
            generation.update(temperature=args.temperature, top_p=0.95)
        # Disabling adapters restores the shared base; it does not train anything.
        from contextlib import nullcontext
        context = model.disable_adapter() if stage == "base" and stages_needed else nullcontext()
        with context, torch.inference_mode():
            output = model.generate(**inputs, **generation)
        tokens = output[0, inputs.input_ids.shape[-1]:]
        continuation = tokenizer.decode(tokens, skip_special_tokens=True)
        result = dict(stage=stage, prefix=args.prefix, continuation=continuation,
                      code=args.prefix + continuation, tokens=len(tokens),
                      seconds=round(perf_counter() - started, 2))
        results.append(result)
        (output_dir / f"{stage}.js").write_text(result["code"] + "\n")
        print(f"{result['tokens']} tokens in {result['seconds']}s\n{result['code']}", flush=True)
        # Write after every stage so a partial comparison is still usable.
        report = dict(
            base=BASE_ID, base_revision=BASE_REVISION,
            adapters=ADAPTER_ID, adapter_revision=ADAPTER_REVISION,
            device="cpu", dtype="bfloat16", seed=args.seed,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            versions={p: version(p) for p in ("torch", "transformers", "peft")},
            results=results,
        )
        (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        from preview import write_preview
        write_preview(output_dir, results)

    print(f"\nSaved comparison to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
