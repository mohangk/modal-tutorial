# Agent notes: LLM learning repository

## Layout and learning goals

The root indexes shared concepts and independent tutorials. Read `README.md` for
the learning map, `training-basics.md` for CPT/SFT/RL/curriculum/LoRA distinctions,
`reinforcement-learning-basics.md` for RL, and `lora-basics.md` for parameter-efficient
adaptation. Keep general teaching material at the root
and concrete commands, outputs, and case studies with their experiments.

- `modal-gsm8k/`: original Modal Dojo project, moved from the root without changing
  dependency pins or training hyperparameters. Read its `AGENTS.md` before working
  there; it preserves setup details, run history, and launch/stop/resume gotchas.
- `demoscener/`: local CPU inference, separate environment, existing model cache
  and saved outputs. `README.md` contains exercises; `lora-and-training.md` explains
  the author's customization and what a reproduction would require.

## Environments

- There is no root Python project. Use `uv run --project modal-gsm8k ...` or
  `uv run --project demoscener ...`, or `cd` into the appropriate folder.
- Always use `uv run` for Python and Modal commands in an experiment.
- Preserve Modal's Python 3.12 selection, `modal==1.6.1`, and
  `modal-dojo==0.1.1` unless the user asks to change them. Dojo hard-pins Modal.
- Keep Demoscener's PyTorch/PEFT dependencies out of the Modal project.
- Do not move, delete, or commit large model caches or saved local outputs.

## Working agreements

- Do not launch paid GPU runs without explicit confirmation.
- Never print or commit token secrets in `~/.modal.toml` or other credentials.
- Read shared RL notes before changing training hyperparameters. Update the
  relevant README, agent notes, and shared learning notes when setup changes.
- Distinguish training from inference; objective/reward from trainable parameters;
  author-reported facts from our measurements and proposed reproduction choices.
- LoRA is a parameter-efficient training method, not only an adapter file format.
  Curriculum schedules example/task selection and can accompany CPT, SFT, or RL;
  these terms are not mutually exclusive methods.
- Demoscener adapters are independently loaded onto the same Qwen base, not stacked.
  Its local scripts do not train or contact Modal.
- Preserve existing uncommitted work while making requested improvements.
