# Reinforcement Learning Basics

Notes from setting up the Modal Dojo training quickstart. The goal is to understand
what the training run is doing, what the numbers in the recipe mean, and which
questions are worth exploring next.

## The task we are training

`training_quickstart.py` trains **Qwen3.5-4B** on a slice of **GSM8K**, a dataset of
grade-school math word problems. The model reads a question, generates an answer,
and a Python function checks whether the final number matches the ground truth.

This is **reinforcement learning with verifiable rewards (RLVR)**. "Verifiable"
means the reward comes from a deterministic checker, not a learned reward model or a
human labeler.

## SFT vs RL

| | Supervised fine-tuning (SFT) | Reinforcement learning (RLVR) |
|---|---|---|
| Training data | prompt **+ target response** | prompt **+ way to verify an answer** |
| What the model imitates | the target text | nothing directly |
| Learning signal | token-level cross-entropy against targets | scalar reward on generated responses |
| Our dataset | would need worked solutions | only questions and final numeric labels |
| Mode in Dojo | `loss_type="sft_loss"` | `loss_type="policy_loss"` (default) |

The `gsm8k_only_answer` dataset deliberately has **no reasoning traces**. The model
has to discover a reasoning path that produces the right number. That is why this is
RL and not SFT.

## The training loop

```text
             ┌─────────────────────────────────────────────┐
             │                                             │
             ▼                                             │
   sample prompts from dataset                             │
             │                                             │
             ▼                                             │
   generate N responses per prompt  ← "rollout"            │
             │                                             │
             ▼                                             │
   score each response with reward function                │
             │                                             │
             ▼                                             │
   compare responses within each prompt's group            │
   → advantage: better or worse than the group average     │
             │                                             │
             ▼                                             │
   optimizer update(s) on the generated samples            │
             │                                             │
             ▼                                             │
   save checkpoint every save_interval steps ──────────────┘
```

One pass through this loop is a **rollout step**. `num_rollout` controls how many
times it repeats.

## Core vocabulary

- **Prompt** — one input from the dataset. Here, one GSM8K question.
- **Response / completion / sample** — one model generation for a prompt. In the
  reward function this is `sample.response`.
- **Trajectory** — the full interaction. For single-turn math, it is just
  prompt + response. For a multi-turn agent, it would be the sequence of messages
  and tool calls.
- **Rollout** — the act of running the current model to generate responses and
  scoring them. RL calls it a rollout because the policy "rolls out" its behavior to
  collect experience.
- **Rollout batch** — the set of prompts used in one rollout step.
- **Group** — all responses generated for the *same* prompt. GRPO compares responses
  within a group.
- **Reward** — a scalar score for one response. Here it is binary: `1.0` correct,
  `0.0` incorrect.
- **Advantage** — a response's reward relative to its group's average. Positive
  advantage means "increase the probability of this response"; negative means
  "decrease it."
- **Policy** — the model being trained. Initially the base Qwen3.5-4B.
- **Reference model** — a frozen copy of the starting model, used for a KL penalty
  to discourage drifting too far. Disabled by default here (`use_kl_loss=False`).
- **Critic** — a separate model that predicts value in PPO. GRPO does not need one
  (`use_critic=False`).
- **GRPO** — Group Relative Policy Optimization. Estimates advantages by comparing
  responses within a group, avoiding a critic. This recipe's default
  (`advantage_estimator="grpo"`).
- **Checkpoint** — saved model weights, written to a Modal Volume.
- **Epoch** — one pass over the dataset. `num_epoch` can replace `num_rollout`.

## GRPO and advantage, concretely

For a prompt with responses scoring `r1, r2, ..., rn`, GRPO computes roughly:

```text
advantage_i = (r_i - mean(r)) / std(r)
```

Example with two responses:

```text
rewards:      [1.0, 0.0]
mean:          0.5
std:           0.5
advantages:   [+1.0, -1.0]
```

The correct response gets pushed up, the wrong one pushed down.

Example with two wrong responses:

```text
rewards:      [0.0, 0.0]
mean:          0.0
std:           0.0
advantages:   [0.0, 0.0]
```

No variance means no signal. This is the central practical issue with small groups.

## Batching arithmetic

The recipe exposes four numbers that are easy to confuse:

| Knob | Meaning | Quickstart default |
|---|---|---:|
| `num_rollout` | number of rollout/training steps | 1 |
| `rollout_batch_size` | **prompts** sampled per rollout step | 2 |
| `n_samples_per_prompt` | **responses** generated per prompt (group size) | 2 |
| `global_batch_size` | **generated samples** per optimizer update | 4 |

```text
samples generated per rollout = rollout_batch_size × n_samples_per_prompt
                              = 2 × 2 = 4

optimizer updates per rollout = samples per rollout / global_batch_size
                              = 4 / 4 = 1

total samples for the run     = num_rollout × rollout_batch_size × n_samples_per_prompt
                              = 1 × 2 × 2 = 4
```

With the defaults, the entire run generates **four responses** and performs **one
optimizer update**. The 120-row dataset is only a pool; only 2 prompts are ever used.

### What `global_batch_size` influences

- **How often weights change.** Smaller global batch → more optimizer steps from the
  same generated data, but each step is noisier. Larger → fewer, smoother updates.
- **Gradient stability.** More samples per update average out noise.
- **LR/weight-decay schedule shape.** The launcher scales scheduler steps by
  `global_batch_size`.
- **Not** how many samples are generated (that's the rollout knobs), and **not** the
  reward or advantage calculation.

Rule of thumb:

```text
global_batch_size ≤ rollout_batch_size × n_samples_per_prompt
```

and ideally it divides that number evenly. If it is larger than the samples produced
per rollout, there may be no full optimizer step at all.

## The dataset, concretely

`skrishna/gsm8k_only_answer` has 7,473 training rows. The quickstart uses the first
120. Each row has:

- `text` — the question.
- `label` — the final numeric answer only.

Examples:

| Question | Label |
|---|---:|
| Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? | 72 |
| Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? | 10 |
| Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? | 5 |
| Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read? | 42 |
| James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year? | 624 |

The original GSM8K version of the first row includes the worked solution:

```text
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72
```

`gsm8k_only_answer` keeps the question and `72`, and removes the reasoning. The model
has to discover the reasoning through RL.

## What the reward function sees

`gsm8k_rm(args, sample, **kwargs)` receives one generated `sample`:

- `sample.response` — the model's text.
- `sample.label` — the ground-truth answer.
- `args` — training/rollout arguments from the framework; unused here.
- `**kwargs` — extra context; keeps the signature forward-compatible.

It extracts the final answer:

1. Prefer the last `\boxed{...}` value.
2. Otherwise take the last number, allowing negatives, commas, and decimals.
3. Return `1.0` if it equals the label, else `0.0`.

Because the reward only checks the final number, any reasoning path that reaches the
right answer is reinforced. There is no penalty for a strange but correct derivation.

## Quickstart vs tutorial values

| Setting | Quickstart file | RL tutorial |
|---|---:|---:|
| `num_rollout` | 1 | 10 |
| `rollout_batch_size` | 2 | 16 |
| `n_samples_per_prompt` | 2 | 8 |
| `global_batch_size` | 4 | 16 |
| Samples per rollout | 4 | 128 |
| Total samples | 4 | 1,280 |
| Optimizer updates | 1 | ~80 |
| `save_interval` | end of run | every 10 rollouts |

The quickstart is a plumbing test: the smallest run that exercises every stage once.
The tutorial is a small real run: it gives each prompt an 8-response group and uses
16 prompts per rollout. Even the tutorial values are modest; its own text notes the
model is still weak after 10 rollouts and continues for 20 more.

## Why group size matters

`n_samples_per_prompt` controls how much evidence GRPO has per prompt.

- With `2`, a group is often all-correct or all-wrong → advantage zero.
- With `8`, there is more likely to be a mix, so there is a gradient signal.
- More samples also cost more generation time, linearly.

There is a tradeoff between **group size** (quality of the advantage estimate) and
**number of prompts** (diversity of the update). Both multiply into the rollout cost.

## Scaling and cost

Generation cost scales roughly with:

```text
num_rollout × rollout_batch_size × n_samples_per_prompt × response length
```

The quickstart generates 4 responses. The tutorial generates 1,280 — about 320× more.
Increasing any of the three factors increases GPU time roughly linearly.

Other cost notes:

- The first run pays for image build, weight download, and checkpoint conversion.
  Those are cached or committed to Volumes and reused later.
- A stopped app's containers are gone. Reuse comes from image caches and Volumes.
- With `colocate=True`, training and rollout generation share the same GPU(s).

## Open questions to explore

1. **Reward shaping.** Binary 0/1 is simple but sparse. Would partial credit (e.g.
   penalize a syntactically valid but wrong number less) help? What are the risks of
   reward hacking?
2. **Group size.** How does `n_samples_per_prompt` affect learning? Compare 2, 4, 8,
   16 on the same prompt budget.
3. **Prompt diversity vs repetitions.** Is it better to use more prompts with fewer
   samples each, or fewer prompts with larger groups?
4. **Temperature and top-p.** How much exploration is needed for math? What happens
   at temperature 0.6 vs 1.0?
5. **KL penalty.** Why is `use_kl_loss=False` the default here? When would enabling
   it prevent drift or collapse?
6. **Advantage estimator.** How does GRPO compare with PPO or DAPO in this setup?
7. **Dynamic sampling / DAPO.** How do `over_sampling_batch_size` and
   `dynamic_sampling_filter_path` filter out uninformative groups?
8. **Evaluation.** Add a held-out GSM8K split and compare base vs checkpoint accuracy.
   Training reward alone can be misleading.
9. **Saturation.** Qwen3.5-4B may already solve many of the first 120 questions. How
   does baseline accuracy affect the amount of learning signal?
10. **Overfitting.** With only 120 prompts, how quickly does the model memorize?
11. **Checkpointing and serving.** How do `resume_from_checkpoint` and
    `recipe.load` differ? How do you deploy a checkpoint with `Endpoint`?
12. **Multi-GPU and multi-node.** How do `actor_num_gpus_per_node`,
    `tensor_model_parallel_size`, and `@modal.clustered` change the picture?
13. **Cost.** What is the actual dollar cost of a smoke test vs a tutorial-sized run?
14. **SFT vs RL.** What would the same task look like as SFT if we had worked
    solutions? Which is more sample-efficient?

## Glossary quick reference

```text
advantage          reward relative to the group mean
checkpoint         saved weights (and optionally optimizer state)
completion         one generated response
critic             value model used by PPO; unused by GRPO
GRPO               group-relative policy optimization
group              responses to the same prompt
policy             the model being trained
prompt             one dataset input
RLVR               RL with verifiable rewards
rollout            generate + score responses from the current policy
sample             one generated response (prompt + response)
SFT                supervised fine-tuning on target responses
trajectory         full interaction; prompt + response here
```

## References

- Modal training quickstart: https://modal.com/docs/guide/training-quickstart
- Modal Dojo docs: https://dojo.modal.dev
- RL tutorial: https://dojo.modal.dev/tutorials/rl_basics
- Dataset: https://huggingface.co/datasets/skrishna/gsm8k_only_answer
- Original GSM8K: https://huggingface.co/datasets/openai/gsm8k
