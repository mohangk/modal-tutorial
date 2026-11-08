# modal-tutorial

Learning project for LLM post-training on [Modal](https://modal.com) with
[Modal Dojo](https://dojo.modal.dev).

The main script, `training_quickstart.py`, follows the
[Modal training quickstart](https://modal.com/docs/guide/training-quickstart):
it trains Qwen3.5-4B on GSM8K with reinforcement learning and a custom,
verifiable reward function.

## Setup

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run modal token info
```

Versions are pinned for compatibility:

- `modal==1.6.1` — required exactly by `modal-dojo==0.1.1`.
- Python 3.12 — Modal 1.6.1 does not support Python 3.15+.

## Run the training quickstart

```bash
nohup uv run python -u training_quickstart.py > training.log 2>&1 &
disown
tail -f training.log
```

Wait for `Launched training <run-id>: app=ap-...` before closing the terminal.
The script has no Modal entrypoint, so use `uv run python`, not `modal run`.

Monitor runs:

```bash
uv run modal-dojo run list
uv run modal-dojo run logs <run-id> --follow
uv run modal-dojo open
```

## Documentation

- [`AGENTS.md`](AGENTS.md) — environment, current state, gotchas, and next steps for
  coding agents.
- [`reinforcement-learning-basics.md`](reinforcement-learning-basics.md) — RL/GRPO
  concepts, batching arithmetic, dataset examples, and open questions.

## Notes

The recipe defaults are a smoke test: one rollout step, two prompts, two responses
per prompt, and one optimizer update. See `reinforcement-learning-basics.md` before
scaling the run.
