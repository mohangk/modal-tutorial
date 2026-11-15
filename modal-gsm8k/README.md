# Modal Dojo: reinforcement learning on GSM8K

This is the original Python project, moved out of the repository root so the root
can index shared learning material. Its scripts, dependency lockfile, Python
version, and training configuration are preserved.

The [training script](training_quickstart.py) runs **Qwen3.5-4B** on math questions
using **GRPO** through Modal Dojo/Slime. A Python reward function checks the final
numeric answer against the dataset label: correct = 1, incorrect = 0. This is
reinforcement learning with verifiable rewards (RLVR).

Read [the shared RL basics](../reinforcement-learning-basics.md) for the training
loop and batching arithmetic. Compare with [Demoscener](../demoscener/README.md),
where the author used human aesthetic preferences as feedback.

## Setup

```bash
cd /home/mohan/Workspace/modal-tutorial/modal-gsm8k
uv sync --locked
uv run python -c "import modal, modal_dojo; print(modal.__version__)"
uv run modal token info
```

Python is selected as 3.12. Dependencies stay pinned to `modal==1.6.1` and
`modal-dojo==0.1.1`: Dojo requires that exact Modal version. Modal 1.6.1 does not
support Python 3.15+. Use this project's `uv run modal`, rather than a global CLI.

## Scripts

| File | What it does |
| --- | --- |
| [get_started.py](get_started.py) | Basic remote function: square 42 on a Modal worker |
| [training_quickstart.py](training_quickstart.py) | Configure a model, GSM8K dataset, answer checker, and detached training launch |
| [AGENTS.md](AGENTS.md) | Detailed setup, recorded run state, troubleshooting, and resume semantics |

The training script uses `train[:120]` as its prompt pool, but defaults generate
only four responses: two prompts × two responses, with one optimizer update.
This is a pipeline smoke test, not meaningful training yet. We have not added
LoRA to this script; [LoRA](../lora-basics.md) is demonstrated by the separate
Demoscener adapters.

## Run the training quickstart

**Paid GPU training requires explicit confirmation.** Once authorized, run from
this folder:

```bash
nohup uv run python -u training_quickstart.py > training.log 2>&1 &
disown
tail -f training.log
```

This script has no Modal entrypoint. Run it using `uv run python`, not
`uv run modal run`. Wait for `Launched training <run-id>: app=ap-...` before closing
the terminal. Interrupting setup before that message can stop the remote app;
after launch, training is detached.

Observe existing runs:

```bash
uv run modal-dojo run list
uv run modal-dojo run logs <run-id> --follow
uv run modal-dojo open
```

The first recorded attempt stopped before model download and produced no
checkpoint. See [the full history](AGENTS.md#run-history--current-state); it is a
record of past checks, not a live assertion about cloud state.

Back to [the learning index](../README.md).
