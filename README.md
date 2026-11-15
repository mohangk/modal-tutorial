# Learning LLM training and reinforcement learning

A personal learning notebook with small, runnable experiments. The aim is to
understand what changes a model's behavior: its training examples, training
objective, trainable parameters, and the way we sample its outputs.

## Learning map

| Topic | Start here | Concrete example |
| --- | --- | --- |
| How CPT, SFT, RL, curriculum, and LoRA fit together | [Training basics](training-basics.md) | Multiple choices that can apply to the same training run |
| Supervised learning vs reinforcement learning | [RL basics](reinforcement-learning-basics.md#sft-vs-rl) | Numeric targets vs rewards in GSM8K |
| Rollouts, rewards, groups, and advantages | [RL training loop](reinforcement-learning-basics.md#the-training-loop) | [Modal + GSM8K](modal-gsm8k/README.md) |
| LoRA: adapting a model without updating all its weights | [LoRA basics](lora-basics.md) | [Demoscener's actual adapters](demoscener/lora-and-training.md) |
| Prefixes, greedy decoding, temperature, and seeds | [Demoscener learning exercise](demoscener/README.md#learning-exercise) | Generate different code without changing model weights |
| Continued pretraining, curriculum, and human preference RL | [Demoscener case study](demoscener/lora-and-training.md#how-the-author-customized-the-model) | Compare the three author-published adapters |
| Evaluation and reproduction | [Demoscener reproduction path](demoscener/lora-and-training.md#what-we-can-reproduce) | Separate working code, personal taste, and evidence of improvement |

## Tutorials and experiments

| Folder | What it teaches | Current state |
| --- | --- | --- |
| [modal-gsm8k/](modal-gsm8k/README.md) | Run Qwen3.5-4B RL with an automatic math-answer reward on Modal Dojo | Smoke-test configuration; the recorded first attempt stopped before training |
| [demoscener/](demoscener/README.md) | Run Qwen2.5-Coder-7B with LoRA adapters trained on animation code and human preferences | Downloaded; local CPU inference and adapter comparisons verified |

Each folder owns its Python dependencies, lockfile, environment, and experiment
instructions. **The repository root is an index, not a Python project.** There is
no root `uv sync`; enter an experiment folder or use `uv run --project <folder>`.
The existing Demoscener model cache and saved results stay in `demoscener/`.

## Suggested learning practice

1. **Identify the feedback.** Start with [training basics](training-basics.md),
   then read the supervised/RL comparison. For each
   experiment, write down its input, output, and what counts as a good result.
2. **Separate training from inference.** Start with the local
   [Demoscener exercise](demoscener/README.md#learning-exercise). Change prefixes,
   temperature, and seeds. Observe that outputs can change while weights stay fixed.
3. **Understand what an adapter changes.** Read [LoRA basics](lora-basics.md),
   inspect a real A/B matrix pair, and compare base/CPT/curriculum/GRPO predictions.
4. **Calculate a reward group.** Pick a favorite among several generated results,
   assign illustrative 1/0 rewards, and subtract the group mean. Compare this human
   preference signal with [the math checker](modal-gsm8k/training_quickstart.py).
5. **Ask what would prove improvement.** Keep a few prefixes or math questions
   out of training, compare multiple examples, and judge validity separately from
   preference. A single good animation or high training reward is not enough.
6. **Study training before scaling it.** Follow the
   [reproduction outline](demoscener/lora-and-training.md#what-we-can-reproduce),
   then consider a small new LoRA training experiment. Paid GPU runs need explicit
   confirmation; nothing in the local inference exercises starts one.

## Commands at a glance

Local inference, no Modal job:

```bash
cd demoscener
uv sync --locked
uv run python -u run.py --stages grpo --prefix 'c.width=1920;' --temperature 0.8 --seed 1
```

Original Modal project's environment check, from the repository root:

```bash
uv sync --project modal-gsm8k --locked
uv run --project modal-gsm8k python -c "import modal, modal_dojo; print(modal.__version__)"
```

The Modal project still pins Python 3.12, `modal==1.6.1`, and `modal-dojo==0.1.1`.
Read its [run instructions](modal-gsm8k/README.md) before launching training.

## Repository notes

[AGENTS.md](AGENTS.md) describes shared operating agreements. Experiment-specific
Modal setup details and recorded run history live in
[modal-gsm8k/AGENTS.md](modal-gsm8k/AGENTS.md).
