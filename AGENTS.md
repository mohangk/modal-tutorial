# Agent notes: LLM learning repository

The root holds shared learning notes and an index of independent experiments.
The original Modal Dojo project now lives in `modal-gsm8k/`. Read its `AGENTS.md`
before working on its scripts, setup, or training configuration.

There is no root Python project. Enter an experiment directory or use
`uv run --project modal-gsm8k ...`. Always use `uv run` for this project.

Preserve Python 3.12, `modal==1.6.1`, and `modal-dojo==0.1.1` unless the user asks
for changes. Read `reinforcement-learning-basics.md` before tuning training.
Do not launch paid GPU runs without explicit confirmation. Never print or commit
credentials. Preserve existing uncommitted work and update affected documentation.

`demoscener/` owns a separate CPU PyTorch/PEFT environment, model cache, and saved
outputs. Keep those dependencies out of the Modal project. Load only one author
stage adapter at a time. Its inference scripts do not train or contact Modal.
