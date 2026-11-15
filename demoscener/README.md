# Demoscener: seeing what post-training changes

This experiment runs the model from [Post training a model to explore
Demoscene](https://datavorous.com/writing/demoscener/). It sits alongside the
GSM8K/Modal experiment so we can compare two uses of reinforcement learning.

The author's [model card](https://huggingface.co/datavorous/Demoscener-7B)
provides three independently loadable LoRA adapters. An adapter is a small set of
learned changes applied to a base model. These adapters need
`Qwen/Qwen2.5-Coder-7B` **Base**, rather than the Instruct variant. Load each
adapter onto the base independently; do not stack all three together.

## Run locally

```bash
cd /home/mohan/Workspace/modal-tutorial/demoscener
uv sync
uv run python -u run.py
```

The default compares `base`, `cpt`, `curriculum`, and `grpo` using the same
unfinished JavaScript program. It uses greedy decoding, CPU inference, BF16
weights, and eight CPU threads. A first run downloads about 15.2 GB of base
weights plus about 485 MB for the three adapters. Allow extra space for Python
packages and download caches. CPU generation can be slow.

Everything specific to this experiment stays here:

- `.venv/`: separate Python environment with CPU PyTorch, Transformers, and PEFT.
- `.cache/huggingface/`: model downloads (unless you explicitly override `HF_HOME`).
- `outputs/<timestamp>/`: completions, generation settings, and package versions.
- `uv.lock`: resolved versions for this experiment. The sibling `../modal-gsm8k/` project has its own preserved
  lockfile and Modal dependency pins.

Download without loading or running the model:

```bash
uv run python -u run.py --download-only
```

Run just the author's preference-trained adapter:

```bash
uv run python -u run.py --stages grpo
```

Explore alternatives after making a deterministic comparison:

```bash
uv run python -u run.py --stages grpo --temperature 0.8 --seed 1
uv run python -u run.py --stages grpo --temperature 0.8 --seed 2
```

Supply a different unfinished program with `--prefix '...'`. Increase
`--max-new-tokens` if a completion is cut off. The 96-token default follows the
author's example; it does not enforce Dwitter's 140-character convention or
guarantee runnable code. This is a code-completion model: provide a program prefix,
rather than a chat instruction such as "make a beautiful animation."

The script pins both Hugging Face repository revisions. It does not train,
contact Modal, or launch a paid GPU. Read results in `results.json` and the
individual `.js` files. A browser preview is written alongside them.

## Connect the stages to the ML/RL basics

All four stages generate code in the same way: predict a next token, append it,
and repeat. What differs is how their parameters were trained beforehand.

| Stage | Training signal used by the author | What to look for now |
| --- | --- | --- |
| `base` | Broad next-token pretraining | What a general code model already knows |
| `cpt` | Next tokens from existing Dweets | More familiarity with tiny animation idioms |
| `curriculum` | Next tokens, with popular Dweets sampled increasingly often | Whether changing the example distribution changes the style |
| `grpo` | Human preferences among generated completions | Whether you like the generated results more |

**CPT and curriculum are not RL here.** They learn from existing code. In
self-supervised next-token training, that code supplies its own target tokens.
The curriculum changes which examples are sampled, rather than rewarding freshly
generated actions.

**CPT means continued pretraining**, not an entirely different loss from SFT.
Both commonly predict target tokens; CPT learns from a code corpus, whereas
typical instruction SFT learns desired responses to inputs. **Curriculum** changes
example sampling and could be used with CPT, SFT, or RL. Here it accompanies CPT.
**LoRA is also a training method:** it limits learning to small adjustment
matrices, which are saved as adapters afterward. These concepts can apply to the
same run. Read [training basics](../training-basics.md) for the full distinction.

**The GRPO stage is RL with human feedback.** In the article, the author generated
eight continuations for a prefix, rendered them, picked a favorite or skipped the
group, updated the policy, and repeated with fresh generations. The policy is the
code-generating language model. Its actions are generated tokens; its output is
a complete proposed program. The article does not specify the exact numeric
reward encoding or every detail of its GRPO loss.

**Our script performs inference only.** It loads weights the author already
trained. Changing the seed changes sampling; selecting your favorite changes
nothing in the model unless a later training step uses that feedback. No
gradients, optimizer updates, or new RL training happen here.

## Compare with the math experiment

| Concept | GSM8K experiment | Demoscener article |
| --- | --- | --- |
| Input | Math question | Unfinished JavaScript program |
| Policy output | Proposed answer/solution | Code continuation |
| Reward source | Numeric answer checker | Human aesthetic choice |
| Feedback type | Verifiable correctness (RLVR) | Human preference (RLHF) |
| GRPO group | Two responses per prompt by default | Eight continuations per prefix |
| What we run here | GSM8K script would launch training | This folder only runs frozen checkpoints |

Both connect to the group-relative advantages explained in
[the shared RL notes](../reinforcement-learning-basics.md#grpo-and-advantage-concretely).
Different rewards can encourage different behavior even when the training
algorithm belongs to the same family.

## Learning exercise

Work through this in small steps. Change one setting at a time and keep the
previous result for comparison. All commands below run locally from this folder:

```bash
cd /home/mohan/Workspace/modal-tutorial/demoscener
```

Every run prints its output directory. Open that directory's `viewer.html` in a
browser and click **Run animation**. `results.json` records the settings and raw
completions; the `.js` files contain the prefix plus generated continuation.
These exercises perform inference only: none updates weights or starts Modal.

### 1. Understand what the prefix has already decided

A prefix is the beginning of the program, not a description of what to draw. The
model continues from exactly where it ends; it does not rewrite it.

```javascript
c.width=1920;for(i=0;i<80;i++)x.fillRect(
```

This already commits every **training stage** to the same structure:

- `c.width=1920;` sets the canvas width and clears/resets the canvas each frame.
- `for(i=0;i<80;i++)` repeats the following statement 80 times.
- `x.fillRect(` starts a rectangle call. The next tokens must supply its arguments
  before a valid program can do something else.

The model can choose coordinates, dimensions, expressions, and more code after
that call, but rectangles and a loop are already in the input. Similar-looking
results are unsurprising. Compare it with `c.width=1920;`, which leaves drawing
operations open, or `c.width=1920;for(i=0;i<80;i++)`, which retains the loop while
allowing a different operation.

Try these two runs with the **same stage and greedy decoding**:

```bash
uv run python run.py --stages grpo --temperature 0 --max-new-tokens 512 \
  --prefix 'c.width=1920;for(i=0;i<80;i++)x.fillRect('
uv run python run.py --stages grpo --temperature 0 --max-new-tokens 512 \
  --prefix 'c.width=1920;'
```

**Observe:** which code was supplied, which code was generated, and whether the
shorter prefix changes the type of drawing. More freedom can also produce less
reliable code; it is not automatically better. Neither command changes weights.

### 2. Understand greedy decoding versus sampling

At each step, the model predicts scores/probabilities for the next token.
**Greedy decoding** chooses the most probable token, appends it, and repeats.
Our `--temperature 0` setting selects this path (`do_sample=False`).

Suppose candidate tokens have probabilities `4: 60%`, `9: 30%`, `0: 10%`. Greedy
chooses `4`. **Sampling** can choose any candidate retained by its sampling rules,
with more likely tokens favored. A positive temperature enables sampling:

```bash
uv run python run.py --stages grpo --prefix 'c.width=1920;' \
  --temperature 0.8 --seed 1 --max-new-tokens 512
```

Our sampling path also uses `top_p=0.95`; other generation settings can come from
model defaults. This is not unrestricted sampling over every possible token.
See [Transformers' generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies).

**Observe:** compare this with the short-prefix greedy result. You have changed
how the same model's distribution is explored, not taught it a new skill.

### 3. Understand what a seed does

A seed initializes the pseudorandom number generator used for sampling. It gives
us a repeatable sequence of random choices, not a quality level or a new model.
Run the sampling command above twice with seed 1, then change only `--seed 2`.

| Comparison | What to expect on the same setup |
| --- | --- |
| Same prefix, stage, temperature, seed, and budget | Usually the same generated code |
| Different seed with sampling enabled | A different random path; a different output is possible, not guaranteed |
| Different seed with greedy decoding | No sampling-driven change |
| Same seed but different adapter or temperature | Output can change because token probabilities/selection rules changed |

Exact reproduction also depends on hardware, precision, and library versions.
The seed controls our **code generation**. If the generated JavaScript itself
uses `Math.random()`, its browser randomness is separate. Animation time `t`
also changes the image even for fixed code.

**Practice:** explain why seed 100 is not "more creative" than seed 1, and why
using the same seed across stages does not force the same answer.

### 4. Explore temperature without changing the model

Keep the prefix, stage, seed, and token budget fixed. Repeat the sampling command
with temperatures `0.3`, `0.8`, and `1.2`.

Lower positive temperatures concentrate probability toward likely tokens; higher
values flatten the distribution. Higher temperature can increase exploration and
also syntax errors or incoherent code. It does not guarantee uniqueness or quality.

**Observe:** count valid programs and compare visual variety. One output at each
setting is just an initial observation. For a stronger comparison, repeat each
setting over the same collection of seeds.

### 5. Give the model enough room, then diagnose invalid JavaScript

The default is **96 new tokens**, excluding the prefix. A token is a piece of
text, not necessarily one character or one complete JavaScript expression.

Use `--max-new-tokens 512` for more room. This is an upper bound: generation can
stop sooner if the model emits an end-of-sequence token. Omitting the argument
restores the script's 96-token default; it does not mean unlimited generation.
There is no unlimited flag. Model context and available memory also impose limits,
so choose a finite budget instead of letting repetitive output run indefinitely.

Open `results.json` and compare each result's `tokens` with `max_new_tokens`:

- **At the limit:** the result hit its budget. Missing a closing parenthesis,
  quote, or brace may be truncation. Retry the same prefix, stage, temperature,
  and seed with a larger budget to isolate that explanation. An EOS token exactly
  at the limit can make this count ambiguous; inspect the actual ending too.
- **Below the limit:** this run stopped before exhausting its budget. More room
  alone will not repair its syntax. Try a different sample or prefix and compare.

In our saved example, a sampled GRPO program cut off at 96 tokens; the same
seed/settings with a 512-token budget finished in 103 tokens. The limit was an
upper bound, not a target length. This is a concrete example of fixing truncation,
not evidence that a larger budget prevents every syntax error.

If Node.js is installed, check syntax without executing generated code:

```bash
node --check outputs/<timestamp>/grpo.js
```

Use a real output directory in place of `<timestamp>`. Passing this check does
not guarantee runtime success, a visible image, termination, or an attractive
animation. The preview shows caught syntax/runtime errors; code with an infinite
loop can still stall its tab. Generated code is kept as-is rather than repaired
silently, so the comparison reflects what the model actually produced.

### 6. Change the adapter while keeping the input and decoding fixed

```bash
uv run python run.py --stages base cpt curriculum grpo \
  --prefix 'c.width=1920;' --temperature 0 --max-new-tokens 512
```

Now you are changing the learned adjustment **in use**, rather than just input or
sampling. Base uses no adapter. Each other stage applies one independent adapter
to the same base; we do not stack CPT + curriculum + GRPO.

Read [LoRA basics](../lora-basics.md), then inspect the real configuration:

```bash
uv run python inspect_lora.py
```

**Observe:** rank 16, alpha 32, two matrices per target projection, and roughly
40.37 million adapter parameters. Explain how adding a small trainable branch can
change outputs while original base weights stay frozen during LoRA training.
See [the concrete matrix shapes](lora-and-training.md#what-is-inside-the-downloaded-adapters).

For a deeper check, `inspect_lora.py --check-logits` compares next-token scores.
Similar top choices do not imply identical probabilities or inactive adapters.
The [case study](lora-and-training.md#why-the-first-outputs-looked-so-similar)
records our measured differences.

**Connect the terminology:** describe the `curriculum` checkpoint using three
choices: next-token learning from code, increasing preference for popular
training examples, and LoRA parameter updates. Then design an example of SFT
using both a curriculum and LoRA. Explain why switching from CPT to a reward-based
GRPO objective would not, by itself, require abandoning LoRA.

### 7. Turn your choices into an illustrative RL signal

Generate several GRPO outputs using the same prefix/settings and different seeds.
Without looking at stage names or seed numbers, choose a favorite. If none are
useful, record a skip rather than inventing a preference.

For a four-output group, suppose your illustrative rewards are:

```text
rewards:                 [0,     0,     1,     0]
group mean:              0.25
reward minus the mean:   [-0.25, -0.25, +0.75, -0.25]
```

Calculate this yourself, then try all zeros and all ones. With no reward variation,
there is no relative preference signal. The author used groups of eight; these
four values are a learning exercise, not their published reward implementation.

**Connect to RL:** generating candidates is a rollout; your selection supplies
feedback. A training algorithm would then compute its full loss, backpropagate,
update trainable parameters, and sample again. Our script stops before those
training steps. Selecting a favorite in your notes does not change the adapter.
Compare this with [GRPO advantages](../reinforcement-learning-basics.md#grpo-and-advantage-concretely)
and [the automatic math reward](../modal-gsm8k/training_quickstart.py).

### 8. Keep a small experiment log and evaluate fairly

Record the following for each run:

| Prefix | Stage | Temperature | Seed | Budget / generated tokens | Syntax/runtime success | Preference and reason |
| --- | --- | --- | --- | --- | --- | --- |
| Your supplied code | Base/CPT/curriculum/GRPO | e.g. 0.8 | e.g. 1 | e.g. 512 / 103 | Record separately | Motion, composition, variety, etc. |

Keep the actual source and output directory alongside these notes. Use multiple
prefixes and seeds, and judge visual quality separately from syntax/runtime
success. Training stage, temperature, and token budget are different variables;
changing all three at once makes a result harder to explain.

**Check your understanding:** which of these changes trains the model: a new
prefix, a new seed, a new adapter loaded for inference, selecting a favorite, or
an optimizer update using preference feedback? Only the last one updates learned
parameters here. Loading an adapter selects existing learned values.

When ready, read [how the author customized Demoscener and what reproduction
would require](lora-and-training.md#what-we-can-reproduce). Start with a small CPT
experiment before attempting a human-feedback training loop.

The model card cautions that later stages were trained on quantized bases. Our
shared BF16 base follows its comparison example, but may produce different
probabilities from the original training setup. Comparing these checkpoints is
an educational observation, not a controlled proof of the effect of RL alone.

## First local comparison (2026-11-15)

All four stages generated successfully on the local CPU with the default prefix.
The base completion reached the 96-token limit and ended inside a string, so it
was not valid JavaScript. CPT completed a row of moving bars; curriculum completed
a ring of squares; GRPO completed several rows of moving bars. These descriptions
come from inspecting the code, rather than a blinded visual evaluation.

Generation took approximately 9–31 seconds per stage, excluding model loading.
The three adapter completions passed JavaScript syntax checks. The base result
needs a larger token budget before drawing conclusions about its capabilities.
This one comparison demonstrates that the checkpoints produce different outputs;
it does not establish that the GRPO output is more beautiful or generally better.

Back to [the learning index](../README.md) and
[the shared learning notes](../reinforcement-learning-basics.md).
