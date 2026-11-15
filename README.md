# Learning LLM training and reinforcement learning

A learning notebook with runnable experiments. Shared concepts live at the root;
each tutorial owns its scripts and Python environment in a separate folder.

## Start here

- [Reinforcement learning basics](reinforcement-learning-basics.md): rewards,
  rollouts, GRPO, batching, and open questions.
- [Modal Dojo + GSM8K](modal-gsm8k/README.md): the original Qwen3.5-4B training
  quickstart, now in its own folder.

- [Demoscener](demoscener/README.md): local CPU inference with animation-code LoRA
  adapters, saved previews, and adapter inspection.

## Run an experiment

The root is no longer a Python project. From here, select the experiment:

```bash
uv sync --project modal-gsm8k --locked
uv run --project modal-gsm8k python -c "import modal, modal_dojo; print(modal.__version__)"
```

Alternatively, enter `modal-gsm8k/` and use its README's commands. Its dependency
pins and training settings are preserved. Paid GPU runs require confirmation.

[AGENTS.md](AGENTS.md) contains shared operating agreements; the Modal folder
preserves its own setup details, run history, and troubleshooting notes.
