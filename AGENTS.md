# Agent notes: modal-tutorial

## What this repo is

A learning/scratch project for running LLM post-training on Modal using
[Modal Dojo](https://dojo.modal.dev). The main script is `training_quickstart.py`,
copied from the [Modal training quickstart](https://modal.com/docs/guide/training-quickstart):
Qwen3.5-4B + GSM8K + a custom verifiable reward, trained with RL (GRPO via Slime).

`reinforcement-learning-basics.md` contains the concepts and questions covered so far.
Read it before changing training hyperparameters.

## Environment — do not casually change

- Python is pinned to **3.12** in `.python-version` (uv).
- `requires-python = ">=3.12,<3.15"` in `pyproject.toml`.
- Dependencies are pinned: `modal==1.6.1`, `modal-dojo==0.1.1`.
- **Why modal 1.6.1:** `modal-dojo==0.1.1` hard-pins `modal==1.6.1`. It is the only
  version that resolves.
- **Why Python 3.12:** Modal 1.6.1 refuses to import on Python 3.15+ with
  `RuntimeError: This version of Modal does not support Python 3.15+`.
  Python 3.15 support arrived in Modal 1.6.2, which dojo does not allow.
- Always use `uv run ...` in this project. The global `modal` CLI may be a different
  version (1.6.3) installed via `uv tool`; `uv run modal` uses the project's 1.6.1.

Setup/verification:

```bash
uv sync
uv run python -c "import modal, modal_dojo; print(modal.__version__)"
uv run modal token info
```

## Authentication

- Token already configured in `~/.modal.toml`, workspace `mohangk`.
- Verify with `uv run modal token info`.
- Only run `uv run modal setup` if the token is invalid; it opens a browser.

## Files

| File | Purpose |
|---|---|
| `training_quickstart.py` | The Modal Dojo RL quickstart: model, dataset, reward, `config.launch()`. |
| `get_started.py` | Basic Modal hello-world (`modal run get_started.py`). |
| `pyproject.toml` / `uv.lock` | Pinned Python dependencies. |
| `reinforcement-learning-basics.md` | RL/Dojo concepts, math, and open questions. |

## How to run the training quickstart

`training_quickstart.py` has **no Modal entrypoint or `@app.function`**. Do not run
`modal run training_quickstart.py`; it will fail. Run it as a normal Python script:

```bash
cd /home/mohan/Workspace/modal-tutorial
nohup uv run python -u training_quickstart.py > training.log 2>&1 &
disown
tail -f training.log
```

Wait until it prints:

```text
Launched training <run-id>: app=ap-xxxxx, function_call=fc-xxxxx
```

At that point the Modal app is detached; you can close the terminal. Before that
line, the script is still doing image build / weight download / checkpoint
conversion and **Ctrl-C will stop the remote app** — `launch()` intentionally calls
`stop_app()` on interruption during setup.

To block locally until completion instead, use `config.train()` rather than
`config.launch()`.

## Observe, stop, resume

```bash
uv run modal-dojo run list
uv run modal-dojo run get <run-id>
uv run modal-dojo run logs <run-id> --follow
uv run modal-dojo open

uv run modal app list
uv run modal app stop <app-id>
```

Resume only works if a checkpoint exists:

```python
from modal_dojo import TrainConfig, TrainingRun

run = TrainingRun.from_id("<run-id>")
checkpoint = run.latest_checkpoint()

config = TrainConfig(
    model=model,
    dataset=dataset,
    resume_from_checkpoint=checkpoint,  # weights only; fresh optimizer/LR
    recipe=Qwen3_5_4B_Recipe(...),
)
config.launch()
```

- `resume_from_checkpoint=checkpoint` — start from weights, fresh optimizer and LR schedule.
- `recipe.load="<checkpoint dir>"` — continue in place with optimizer state and iteration count.
- `start_rollout_id` — set the starting iteration when continuing manually.

## Where the remote execution lives

The local script only builds a config. `config.launch()` is the trigger:

1. `modal_dojo/common/train.py` → `TrainConfig.launch()` (around line 634).
2. `_build_app()` dispatches to `build_slime_app()` in
   `modal_dojo/frameworks/slime/launcher.py` (around line 352).
3. `build_slime_app()` creates a `modal.App` with remote functions:
   `download`, `prepare_dataset`, `convert_checkpoint`, and `train`.
4. `launch()` does `app.run(detach=True)`, calls `app.download.remote(...)`,
   converts the checkpoint if needed, then `app.train.spawn(...)`.
5. The remote `train` function runs Slime + SGLang on a Modal GPU (`H100` by default).

Useful paths inside `.venv` if debugging:

```text
.venv/lib/python3.12/site-packages/modal_dojo/common/train.py
.venv/lib/python3.12/site-packages/modal_dojo/train_recipes/slime_recipe/recipe.py
.venv/lib/python3.12/site-packages/modal_dojo/frameworks/slime/launcher.py
```

## Current recipe defaults and what they mean

`training_quickstart.py` only sets `custom_rm_function`, so all other values are
`SlimeRecipe` defaults:

```text
gpu_type                 H100 (1 node, 1 GPU, colocated)
num_rollout              1
rollout_batch_size       2 prompts
n_samples_per_prompt     2 responses per prompt
global_batch_size        4 samples per optimizer step
rollout_max_response_len 4096
rollout_temperature      1.0
advantage_estimator      grpo
lr                       5e-7
save_interval            defaults to num_rollout (1 checkpoint at end)
```

Arithmetic:

```text
samples per rollout = 2 prompts × 2 responses = 4
optimizer updates   = 1 rollout × 4 samples / global_batch_size 4 = 1
```

This is a smoke test: 2 prompts out of the 120-row pool, 4 generations, 1 update.
It validates the pipeline; it does not meaningfully train the model.

## Dataset

- `skrishna/gsm8k_only_answer`, split `train[:120]` (7,473 rows total in train).
- Columns: `text` = GSM8K question, `label` = final numeric answer only.
- No reasoning traces. The model must generate its own reasoning.
- Reward `gsm8k_rm` extracts `\boxed{...}` or the last number and returns binary
  1.0/0.0 against `sample.label`.

## Run history / current state

- First attempt: run `coplanar-subfloor-9e69b7d34ee6`, app
  `ap-qeMyZUS4XPXiLJW4xTbAbF`.
- It was interrupted during OCI image unpack, before model download.
  The app is stopped; the run record is stuck at `pending` / `Initializing`;
  there is no checkpoint and it cannot be resumed.
- Volumes exist but were empty when checked:
  `huggingface-cache`, `slime-qwen3_5_4b_recipe-checkpoints`,
  `slime-qwen3_5_4b_recipe-data`, `modal-dojo-metadata`,
  `modal-dojo-dashboard-overlay`.
- The `dojo-dashboard` app is deployed and running.
- Next run should reuse cached image layers, but must still download weights and
  prepare the dataset once. After that, the HF cache volume is committed and
  subsequent runs reuse the weights.
- `modal-dojo cleanup` only removes failed/cancelled runs older than 7 days, so it
  will not remove this `pending` record.

## Gotchas

- `modal-dojo` pins `modal==1.6.1` exactly. Do not upgrade `modal` alone.
- Python 3.15 breaks Modal 1.6.1 at import time.
- Ctrl-C before `Launched training` stops the remote app.
- Ctrl-C after `Launched training` only disconnects; the app keeps running.
- Modal containers are ephemeral. Reuse comes from image caches and Volumes, not
  from reattaching to an old container.
- The `huggingface-cache` volume is only populated after a successful download and
  commit; an interrupted download may leave nothing reusable.

## Next steps

1. Finish one successful smoke run and confirm a checkpoint and reward logs.
2. Move to a real training config (tutorial-style):
   `num_rollout=10`, `rollout_batch_size=16`, `n_samples_per_prompt=8`,
   `global_batch_size=16`, `save_interval=10`.
3. Add a held-out `eval_dataset` and evaluate before/after checkpoints.
4. Explore reward shaping, group size, temperature, and KL settings (see the
   open questions in `reinforcement-learning-basics.md`).
5. Serve the trained checkpoint with `Endpoint` when there is something worth serving.

## Working agreements for agents

- Do not launch paid GPU runs without explicit confirmation.
- Never print or commit the token secret in `~/.modal.toml`.
- Prefer `uv run` over global `modal` for this repo.
- Keep `modal==1.6.1` and `modal-dojo==0.1.1` pinned unless the user asks to change them.
- Update this file and `reinforcement-learning-basics.md` when the setup or the
  training configuration changes.
