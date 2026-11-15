# Demoscener: local animation-code inference

Run the author's [Demoscener-7B](https://huggingface.co/datavorous/Demoscener-7B)
LoRA adapters on `Qwen/Qwen2.5-Coder-7B` Base. This is local inference with existing
weights, not new training or a Modal GPU launch.

## Setup and run

```bash
cd /home/mohan/Workspace/modal-tutorial/demoscener
uv sync --locked
uv run python -u run.py
```

This independent environment uses CPU PyTorch, Transformers, and PEFT. The first
run downloads about 15.2 GB of base weights and 485 MB of adapters. Downloads live
in `.cache/huggingface/` unless you override `HF_HOME`. Caches, environments,
logs, and outputs are ignored by Git.

The default compares the base and three independently loaded adapters: `cpt`,
`curriculum`, and `grpo`. Use one adapter at a time, not all three stacked together.
Generation uses BF16 weights on the CPU, eight threads, greedy decoding, and
96 new tokens by default. The base and adapter repository revisions are pinned.

## Explore generation

```bash
uv run python run.py --download-only
uv run python run.py --stages grpo --prefix 'c.width=1920;' --temperature 0.8 --seed 1 --max-new-tokens 512
```

Positive temperature enables sampling; zero selects greedy decoding. Changing
seeds explores different sampling choices. Increase the token budget if output
is cut off, but generated code is not guaranteed to be valid JavaScript.

Results are saved in `outputs/<timestamp>/`: `results.json` records settings and
continuations, `.js` files contain complete prefixes plus continuations, and
`viewer.html` provides click-to-run sandboxed animation previews.

## Inspect the LoRA adapters

```bash
uv run python inspect_lora.py
uv run python inspect_lora.py --check-logits --output outputs/lora-inspection.json
```

The inspector reads cached files. Its optional prediction check loads the base
on the CPU and verifies that active adapters change next-token scores.

See [the author's article](https://datavorous.com/writing/demoscener/) for the
training story. Return to [the learning index](../README.md).
